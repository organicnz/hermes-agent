"""Invariant: a worker's direct self-completion cannot reach ``done`` without a
completed review round (false-completion class t_60130e9e / t_f4635fcb /
t_8e5bb2a9: cards closed straight from ``running`` via ``kanban_complete`` with
no review round, claiming deliverables absent from every ref).

The completion gate routes run-owner self-completions from non-review statuses
to ``review`` and fails closed when the routing is refused. Review approvals,
force overrides and stale-run markers keep today's semantics; every gate
decision leaves a ``completion_routed_review`` event.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_db_workspace as kbw


@pytest.fixture
def conn(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    db_path = kb.kanban_db_path(board="default")
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    kb.init_db()
    with kbc.connect() as c:
        yield c


def _worker_card(conn, title: str = "implementation") -> tuple[str, int]:
    """A claimed running card whose live worker is this process — the dispatcher
    worker shape the gate must intercept."""
    tid = kb.create_task(conn, title=title, assignee="builder")
    claimed = kb.claim_task(conn, tid, claimer="builder:test")
    assert claimed is not None
    kbd._set_worker_pid(conn, tid, os.getpid())
    return tid, kb._current_run_id(conn, tid)


def _gate_events(conn, tid):
    return [e for e in kb.list_events(conn, tid) if e.kind == "completion_routed_review"]


def test_worker_self_completion_routes_to_review(conn):
    tid, run_id = _worker_card(conn)
    task = kb.get_task(conn, tid)
    ws = kbw.resolve_workspace(task)
    kbw.set_workspace_path(conn, tid, ws)
    artifact = ws / "evidence.json"
    artifact.write_bytes(b'{"ok": true}')

    assert kb.complete_task(
        conn, tid,
        result="deliverable shipped",
        summary="handoff: built and tested",
        metadata={"artifacts": [str(artifact)]},
        expected_run_id=run_id,
    ) is True

    landed = kb.get_task(conn, tid)
    assert landed.status == "review"  # not done: a review round must close it
    assert landed.completed_at is None
    assert landed.result is None  # the routed path leaves tasks.result untouched
    assert landed.current_run_id is None and landed.claim_lock is None

    run = kb.latest_run(conn, tid)
    assert run.ended_at is not None and run.outcome == "review_requested"
    assert run.summary == "handoff: built and tested"

    events = kb.list_events(conn, tid)
    gate = _gate_events(conn, tid)[-1]
    assert gate.payload["decision"] == "routed"
    assert gate.payload["reason"] == "worker_direct_completion"
    assert gate.payload["prior_status"] == "running"
    assert gate.run_id == run_id
    handoff = [e for e in events if e.kind == "review_requested"][-1]
    assert handoff.payload["implementer"] == "builder"
    assert not [e for e in events if e.kind == "completed"]

    # The declared deliverable is staged durably for the reviewer, exactly as a
    # voluntary request_review handoff stages it.
    attachments = kb.list_attachments(conn, tid)
    assert [a.filename for a in attachments] == ["evidence.json"]
    staged = Path(handoff.payload["artifacts"][0])
    assert staged.exists() and staged.read_bytes() == b'{"ok": true}'
    assert staged.parent == kb.task_attachments_dir(tid)


def test_reviewer_approval_from_review_completes_unchanged(conn):
    tid = kb.create_task(conn, title="approved", assignee="builder")
    implementation = kb.claim_task(conn, tid, claimer="builder:1")
    assert implementation is not None
    assert kb.request_review(
        conn, tid, summary="ready", reviewer="reviewer",
        expected_run_id=implementation.current_run_id,
    )
    review = kb.claim_review_task(conn, tid, claimer="reviewer:1")
    assert review is not None

    assert kb.complete_task(
        conn, tid, summary="Approved after independent verification.",
        expected_run_id=review.current_run_id,
    ) is True

    landed = kb.get_task(conn, tid)
    assert landed.status == "done" and landed.completed_at is not None
    run = kb.latest_run(conn, tid)
    assert run.outcome == "completed"
    assert _gate_events(conn, tid) == []  # the compliant path never meets the gate


def test_fail_closed_on_malformed_re_review_provenance(conn):
    tid = kb.create_task(conn, title="rework", assignee="builder")
    implementation = kb.claim_task(conn, tid, claimer="builder:1")
    assert implementation is not None
    assert kb.request_review(
        conn, tid, summary="v1", reviewer="reviewer",
        expected_run_id=implementation.current_run_id,
    )
    review = kb.claim_review_task(conn, tid, claimer="reviewer:1")
    assert review is not None
    assert kb.request_changes(
        conn, tid, reason="fix it", expected_run_id=review.current_run_id,
    ) == (True, "builder")
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_events SET payload = '{}' "
            "WHERE task_id = ? AND kind = 'changes_requested'",
            (tid,),
        )
    retry = kb.claim_task(conn, tid, claimer="builder:2")
    assert retry is not None

    # Routing is refused (no recoverable reviewer provenance): the completion
    # must fail closed, never fall through to done.
    assert kb.complete_task(
        conn, tid, summary="fixed", expected_run_id=retry.current_run_id,
    ) is False

    landed = kb.get_task(conn, tid)
    assert landed.status == "running"
    assert landed.current_run_id == retry.current_run_id
    gate = _gate_events(conn, tid)[-1]
    assert gate.payload["decision"] == "refused"
    assert gate.payload["detail"]  # the refusal reason is recorded


def test_force_override_still_completes(conn):
    tid, _run_id = _worker_card(conn)
    assert kb.complete_task(conn, tid, result="operator override", force=True) is True
    assert kb.get_task(conn, tid).status == "done"
    assert _gate_events(conn, tid) == []


def test_stale_run_marker_keeps_todays_refusal(conn):
    tid, run_id = _worker_card(conn)
    assert kb.complete_task(
        conn, tid, result="stale worker", expected_run_id=run_id + 999,
    ) is False
    landed = kb.get_task(conn, tid)
    assert landed.status == "running" and landed.current_run_id == run_id
    assert kb.latest_run(conn, tid).ended_at is None
    assert _gate_events(conn, tid) == []
