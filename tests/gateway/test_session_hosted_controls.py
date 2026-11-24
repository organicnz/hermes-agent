"""Canonical room controls retain the original attempt fence."""
import time
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, begin_runtime_epoch


def test_unknown_discard_is_exact_and_never_requeues(tmp_path, monkeypatch):
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway import hosted_room_driver as tasks
    from gateway.hosted_rooms import create_room, local_authority_gateway_id
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    class Service(CanonicalHostedRoomService):
        pass
    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(db=db, profile_id=str(tmp_path), epoch=begin_runtime_epoch(db, instance_id='test'))
        service = Service(authority, None)
        service.authorize_room('alice', 'room', create=True)
        gateway = local_authority_gateway_id()
        create_room(db.db_path, room_id='room', name='Room', authority_gateway_id=gateway, members=[
            {'member_id': 'one', 'profile': 'default', 'handle': 'one'},
            {'member_id': 'two', 'profile': 'other', 'handle': 'two'}])
        identity = tasks.TaskIdentity('room', 'task', 'thread', 'turn')
        tasks.admit_task(db.db_path, identity, payload={'target_profile': 'default', 'target_member_id': 'one', 'source_event_seq': 1, 'prompt': 'frozen'}, clock=time.time)
        lease = tasks.acquire_lease(db.db_path, room_id='room', gateway_id=gateway, authority_epoch=1, process_generation='test', ttl_seconds=30, clock=time.time)
        tasks.start_task(db.db_path, identity, lease, expected_cancel_generation=0, clock=time.time)
        later = time.time() + 31
        monkeypatch.setattr(service.runtime, 'clock', lambda: later)
        recovered = service.runtime._ensure_lease(service.bindings()[0])
        tasks.recover_room(db.db_path, recovered, clock=lambda: later)
        # Only the remote canonical boundary is substituted; the room task/lease is real.
        calls = []
        rpc = SimpleNamespace(ref=SimpleNamespace(session_id='session'),
            info=lambda **kw: {'status': 'unknown'},
            discard=lambda **kw: calls.append(kw) or {'discarded': True})
        monkeypatch.setattr(service, '_resolve_member_transport', lambda *args: rpc)
        monkeypatch.setattr(service.runtime, 'publish_terminal', None)
        args = dict(room_id='room', task_id='task', member_id='one', execution_generation=1)
        with monkeypatch.context() as scoped:
            # groups.state reads only the shown (live/retryable) rows, never every retained task.
            scoped.setattr(tasks, 'list_tasks', lambda *a, **k: pytest.fail('status() decoded every task'))
            action, = service.status('room')['pending_actions']
            # A native room lists its exact attempt controls after the approval cards.
            service._set_pending_action(
                'room', 'one', {'kind': 'approval', 'task_id': 'task', 'execution_generation': 1, 'request_id': 'r2'})
            assert [a['kind'] for a in service.status('room')['pending_actions']] == ['approval', 'discard']
            service._set_pending_action('room', 'one', None)
        assert action == {'kind': 'discard', 'member_id': 'one', 'task_id': 'task', 'execution_generation': 1}
        for change in ({'member_id': 'two'}, {'execution_generation': 2}, {'execution_generation': True}):
            with pytest.raises(RuntimeStoreError):
                service.discard_room_task(**{**args, **change})
        with pytest.raises(RuntimeStoreError, match='unknown_execution'):
            service.retry_room_task(**args)
        assert not calls
        # A deferred attempt is explicit retryable work, but a canonical unknown
        # still blocks it; no generation advances on either refusal.
        tasks.defer_indeterminate_task(db.db_path, identity, recovered,
            expected_execution_generation=1, expected_cancel_generation=0,
            reason='test-deferred', clock=lambda: later)
        with pytest.raises(RuntimeStoreError, match='unknown_execution'):
            service.retry_room_task(**args)
        rpc.info = lambda **kw: {'status': 'idle', 'active': False}
        retried = service.retry_room_task(**args)
        assert retried['status'] == 'queued'
        assert retried['execution_generation'] == 1
        tasks.start_task(db.db_path, identity, recovered, expected_cancel_generation=0, clock=lambda: later)
        later += 31
        recovered = service.runtime._ensure_lease(service.bindings()[0])
        tasks.recover_room(db.db_path, recovered, clock=lambda: later)
        args['execution_generation'] = 2
        result = service.discard_room_task(**args)
        assert result['status'] == 'cancelled'
        assert calls[0]['execution_generation'] == 2
        assert calls[0]['expected_task_id'] == 'task'
        assert tasks.get_task(db.db_path, identity)['execution_generation'] == 2
        assert service.discard_room_task(**args)['status'] == 'cancelled'
        assert len(calls) == 1
        with pytest.raises(RuntimeStoreError):
            service.discard_room_task(**{**args, 'execution_generation': 1})


def test_a_deadline_stopped_turn_offers_retry_and_requeues(tmp_path, monkeypatch):
    """The gateway's own room projection offers Retry for a turn its deadline stopped, and it works."""
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway import hosted_room_driver as tasks
    from gateway.hosted_rooms import create_room, local_authority_gateway_id
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))

    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(db=db, profile_id=str(tmp_path), epoch=begin_runtime_epoch(db, instance_id='test'))
        service = CanonicalHostedRoomService(authority, None)
        service.authorize_room('alice', 'room', create=True)
        gateway = local_authority_gateway_id()
        create_room(db.db_path, room_id='room', name='Room', authority_gateway_id=gateway, members=[
            {'member_id': 'one', 'profile': 'default', 'handle': 'one'},
            {'member_id': 'two', 'profile': 'default', 'handle': 'two'}])
        # The service's own runtime generation holds the room lease, exactly as a live gateway does.
        lease = service.runtime._ensure_lease(service.bindings()[0])

        def failed_turn(task_id, reason):
            """Drive one admitted turn to the settlement its own deadline (or a refusal) produces."""
            identity = tasks.TaskIdentity('room', task_id, 'thread', task_id)
            tasks.admit_task(db.db_path, identity, payload={
                'target_profile': 'default', 'target_member_id': 'one',
                'source_event_seq': 1, 'prompt': 'frozen'}, clock=time.time)
            attempt = tasks.start_task(db.db_path, identity, lease, expected_cancel_generation=0, clock=time.time)
            stopping = tasks.begin_task_cancel(
                db.db_path, identity, cancel_id=f'deadline:{attempt.execution_generation}',
                expected_cancel_generation=0, clock=time.time)
            tasks.settle_stopping_task(
                db.db_path, identity, lease,
                expected_execution_generation=attempt.execution_generation,
                expected_cancel_generation=int(stopping['cancel_generation']),
                settlement_id=f'deadline:{attempt.execution_generation}', status='failed',
                result={'error': 'This Group Chat turn exceeded its configured time limit and was stopped.',
                        'reason_code': reason, 'timeout_seconds': 30.0}, clock=time.time)
            return identity

        identity = failed_turn('task', 'turn_deadline_exceeded')
        # A refusal inherits the deadline marker of the attempt it replaced; its own result says otherwise.
        failed_turn('refused', 'member_unavailable')
        # A member receipt can name the same deadline with no Stop marker at all.
        receipt = tasks.TaskIdentity('room', 'receipt', 'thread', 'receipt')
        tasks.admit_task(db.db_path, receipt, payload={
            'target_profile': 'default', 'target_member_id': 'two',
            'source_event_seq': 2, 'prompt': 'frozen'}, clock=time.time)
        tasks.settle_task(
            db.db_path,
            tasks.start_task(db.db_path, receipt, lease, expected_cancel_generation=0, clock=time.time),
            settlement_id='receipt:1', status='failed',
            result={'error': 'time limit', 'reason_code': 'turn_deadline_exceeded'}, clock=time.time)

        # The reason alone selects exactly the timeout settlements, with or without the marker.
        actions = service.status('room')['pending_actions']
        assert {action['task_id'] for action in actions} == {'task', 'receipt'}
        action = next(action for action in actions if action['task_id'] == 'task')
        coordinates = {key: value for key, value in action.items() if key != 'kind'}

        retried = service.retry_room_task('room', **coordinates)
        assert retried['status'] == 'queued'
        requeued = tasks.get_task(db.db_path, identity)
        # The old settlement and Stop marker are gone, so the next start is a genuinely fresh attempt.
        assert (requeued['status'], requeued['result'], requeued['settlement_status'],
                requeued['cancel_id']) == ('queued', None, None, None)
        assert [a['task_id'] for a in service.status('room')['pending_actions']] == ['receipt']
        with pytest.raises(RuntimeStoreError):
            service.retry_room_task('room', member_id='one', task_id='refused', execution_generation=1)
        # The marker-less timeout receipt follows the same Retry path.
        assert service.retry_room_task('room', member_id='two', task_id='receipt',
                                       execution_generation=1)['status'] == 'queued'
        assert service.status('room')['pending_actions'] == []


def test_a_stale_retry_cannot_requeue_a_newer_deadline_attempt(tmp_path, monkeypatch):
    """A Retry naming the replaced attempt is refused before the write and by the write's own fence."""
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway import hosted_room_driver as tasks
    from gateway.hosted_rooms import create_room, local_authority_gateway_id
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))

    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(db=db, profile_id=str(tmp_path), epoch=begin_runtime_epoch(db, instance_id='test'))
        service = CanonicalHostedRoomService(authority, None)
        service.authorize_room('alice', 'room', create=True)
        gateway = local_authority_gateway_id()
        create_room(db.db_path, room_id='room', name='Room', authority_gateway_id=gateway, members=[
            {'member_id': 'one', 'profile': 'default', 'handle': 'one'}])
        lease = service.runtime._ensure_lease(service.bindings()[0])
        identity = tasks.TaskIdentity('room', 'task', 'thread', 'turn')
        tasks.admit_task(db.db_path, identity, payload={
            'target_profile': 'default', 'target_member_id': 'one',
            'source_event_seq': 1, 'prompt': 'frozen'}, clock=time.time)

        def deadline_fail(expected_execution_generation, expected_cancel_generation):
            attempt = tasks.start_task(
                db.db_path, identity, lease,
                expected_cancel_generation=expected_cancel_generation, clock=time.time)
            assert attempt.execution_generation == expected_execution_generation
            stopping = tasks.begin_task_cancel(
                db.db_path, identity, cancel_id=f'deadline:{expected_execution_generation}',
                expected_cancel_generation=expected_cancel_generation, clock=time.time)
            tasks.settle_stopping_task(
                db.db_path, identity, lease,
                expected_execution_generation=expected_execution_generation,
                expected_cancel_generation=int(stopping['cancel_generation']),
                settlement_id=f'deadline:{expected_execution_generation}', status='failed',
                result={'error': 'This Group Chat turn exceeded its configured time limit and was stopped.',
                        'reason_code': 'turn_deadline_exceeded', 'timeout_seconds': 30.0},
                clock=time.time)

        deadline_fail(1, 0)
        assert service.retry_room_task('room', task_id='task', member_id='one',
                                       execution_generation=1)['status'] == 'queued'
        # The retry runs generation 2 (the deadline's cancel generation is kept), and also ends
        # at its own deadline.
        deadline_fail(2, 1)

        with pytest.raises(RuntimeStoreError):
            service.retry_room_task('room', task_id='task', member_id='one', execution_generation=1)
        current = tasks.get_task(db.db_path, identity)
        with pytest.raises(tasks.StaleTaskError):
            tasks.requeue_failed_task(
                db.db_path, identity, lease, expected_execution_generation=1,
                expected_cancel_generation=int(current['cancel_generation']), clock=time.time)
        assert (current['status'], current['execution_generation']) == ('failed', 2)
        [card] = service.status('room')['pending_actions']
        assert (card['task_id'], card['execution_generation']) == ('task', 2)


def test_a_renamed_member_does_not_offer_a_stale_deadline_retry(tmp_path, monkeypatch):
    """A deadline failure frozen under a profile the member no longer uses is not Retryable."""
    from gateway.session_hosted_service import CanonicalHostedRoomService
    from gateway import hosted_room_driver as tasks
    from gateway.hosted_rooms import create_room, local_authority_gateway_id
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))

    with SessionDB(tmp_path / 'state.db') as db:
        authority = SimpleNamespace(db=db, profile_id=str(tmp_path), epoch=begin_runtime_epoch(db, instance_id='test'))
        service = CanonicalHostedRoomService(authority, None)
        service.authorize_room('alice', 'room', create=True)
        gateway = local_authority_gateway_id()
        create_room(db.db_path, room_id='room', name='Room', authority_gateway_id=gateway, members=[
            {'member_id': 'security', 'profile': 'cybersecurity', 'handle': 'cybersecurity'}])
        lease = service.runtime._ensure_lease(service.bindings()[0])
        identity = tasks.TaskIdentity('room', 'task', 'thread', 'turn')
        # The attempt was admitted while the member was still profile `security`; the room now
        # seats the same member as `cybersecurity`.
        tasks.admit_task(db.db_path, identity, payload={
            'target_profile': 'security', 'target_member_id': 'security',
            'source_event_seq': 1, 'prompt': 'frozen'}, clock=time.time)
        attempt = tasks.start_task(db.db_path, identity, lease, expected_cancel_generation=0, clock=time.time)
        stopping = tasks.begin_task_cancel(
            db.db_path, identity, cancel_id='deadline:1',
            expected_cancel_generation=0, clock=time.time)
        tasks.settle_stopping_task(
            db.db_path, identity, lease,
            expected_execution_generation=attempt.execution_generation,
            expected_cancel_generation=int(stopping['cancel_generation']),
            settlement_id='deadline:1', status='failed',
            result={'error': 'time limit', 'reason_code': 'turn_deadline_exceeded'}, clock=time.time)

        assert [a for a in service.status('room')['pending_actions'] if a['kind'] == 'retry'] == []
        with pytest.raises(RuntimeStoreError, match='stale_generation'):
            service.retry_room_task('room', task_id='task', member_id='security', execution_generation=1)
