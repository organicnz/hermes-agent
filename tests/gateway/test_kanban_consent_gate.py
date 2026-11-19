"""Consent-gate skip: machine sweeps leave operator-gated cards at rest.

Covers the spec §5 edge matrix for ``kanban.skip_consent_gated`` (plan
CONSENT_GATE_SKIP_PLAN_2026-11-19.md; spec comment id 59 on t_c197a151):
episode-scoped predicate, operator release signals, fail-open directions,
live kill switch, auto-decompose budget, dispatch lanes, spawnable telemetry,
and bounded ``consent_gate_hold`` observability.

Every test runs on a scratch board under an isolated HERMES_HOME (tmp_path) —
never the live default board. Gate states are built through the real kernel
APIs (block_task / unblock_task / claim / request_review / specify), so the
rig exercises the verified churn loop, not a hand-waved lookalike.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace

import yaml

from hermes_cli import kanban_consent_gate as cg
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd
from hermes_cli import kanban_decompose
from gateway import kanban_watchers_dispatcher as kwd


# ---------------------------------------------------------------------------
# Rig helpers
# ---------------------------------------------------------------------------


def _isolated_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _set_kanban_config(home, **values):
    (home / "config.yaml").write_text(
        yaml.safe_dump({"kanban": values}), encoding="utf-8"
    )


def _roster(monkeypatch, *names):
    from hermes_cli import profiles

    monkeypatch.setattr(
        profiles, "list_profiles",
        lambda **kw: [SimpleNamespace(name=n) for n in names],
    )


def _gate_from_ready(conn, tid, kind="needs_input"):
    """block -> unblock -> block: the breaker escalates into triage with a gate."""
    assert kb.block_task(conn, tid, reason="q1", kind=kind)
    assert kb.unblock_task(conn, tid)
    assert kb.block_task(conn, tid, reason="q2", kind=kind)


def _make_gated_triage(conn, assignee="default"):
    tid = kb.create_task(conn, title="gated triage", assignee=assignee, triage=True)
    assert kb.specify_triage_task(conn, tid, author="auto-decomposer")
    _gate_from_ready(conn, tid)
    assert kb.get_task(conn, tid).status == "triage"
    return tid


def _make_gated_in_lane(conn, lane, assignee="default"):
    """Drive the verified churn loop until the gated card rests in *lane*.

    ready:  claim -> block -> unblock -> claim -> block (triage) -> specify.
    review: same, but claimed from review so the gate payload carries
    source_status=review and recompute_ready resumes into the review lane.
    """
    assert lane in ("ready", "review")
    tid = kb.create_task(conn, title=f"gated {lane}", assignee=assignee)
    if lane == "review":
        assert kb.claim_task(conn, tid) is not None
        run_id = kb.get_task(conn, tid).current_run_id
        assert kb.request_review(conn, tid, summary="handoff", expected_run_id=run_id)
    claim = kb.claim_review_task if lane == "review" else kb.claim_task
    for i in range(2):
        assert claim(conn, tid) is not None
        assert kb.block_task(conn, tid, reason="operator question", kind="needs_input")
        if i == 0:
            assert kb.unblock_task(conn, tid)
    assert kb.get_task(conn, tid).status == "triage"
    assert kb.specify_triage_task(conn, tid, author="auto-decomposer")
    assert kb.get_task(conn, tid).status == lane
    return tid


def _make_gated_blocked(conn, assignee="default"):
    tid = kb.create_task(conn, title="gated blocked", assignee=assignee)
    assert kb.block_task(conn, tid, reason="operator question", kind="needs_input")
    assert kb.get_task(conn, tid).status == "blocked"
    return tid


def _gate_event(conn, tid):
    return conn.execute(
        "SELECT id, payload FROM task_events WHERE task_id = ? "
        "AND kind IN ('blocked','block_loop_detected') ORDER BY id DESC LIMIT 1",
        (tid,),
    ).fetchone()


def _holds(conn, tid):
    return [e for e in kb.list_events(conn, tid) if e.kind == "consent_gate_hold"]


def _spawn_stub():
    calls = []

    def spawn(task, workspace):
        calls.append(task.id)
        return None

    return spawn, calls


def _fake_aux(calls):
    def fake_aux(verb, task_id, **kwargs):
        calls.append(task_id)
        return (
            json.dumps({"fanout": False, "title": "specified", "body": "spec body", "assignee": None}),
            "",
        )

    return fake_aux


def _dispatcher():
    settings = kwd._DispatcherSettings(
        interval=15, max_spawn=None, max_in_progress=None,
        failure_limit=kbd.DEFAULT_FAILURE_LIMIT, stale_timeout_seconds=0,
        reconcile_orphans=True, default_assignee=None, max_in_progress_per_profile=None,
    )
    return kwd._KanbanDispatcher(kb, settings)


def _to_review(conn, tid):
    assert kb.claim_task(conn, tid) is not None
    run_id = kb.get_task(conn, tid).current_run_id
    assert kb.request_review(conn, tid, summary="handoff", expected_run_id=run_id)


def _backdate_comment(conn, tid, author, seconds=3600):
    """Move the newest comment by *author* back in time.

    created_at is epoch SECONDS; the rig compresses minutes of real activity
    into one second. Backdating models the real ordering (an operator answer
    precedes the worker's next question by minutes) so episode-scoping is
    tested deterministically instead of by second-boundary luck.
    """
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_comments SET created_at = created_at - ? "
            "WHERE id = (SELECT MAX(id) FROM task_comments WHERE task_id = ? AND author = ?)",
            (seconds, tid, author),
        )


def _advance_event(conn, event_id, seconds=3600):
    """Move an event forward in time (models a worker question arriving minutes
    after the operator answer that released the previous episode)."""
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE task_events SET created_at = created_at + ? WHERE id = ?",
            (seconds, event_id),
        )


# ---------------------------------------------------------------------------
# Predicate: episodes, releases, fail-open
# ---------------------------------------------------------------------------


def test_fresh_triage_decomposes_gated_holds(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        fresh = kb.create_task(conn, title="fresh", triage=True)
        gated = _make_gated_triage(conn)
        assert cg.consent_gated_ids(conn, [fresh, gated]) == {gated}
    assert kanban_decompose.list_auto_decomposable_triage_ids() == [fresh]
    # Manual enumeration is untouched: operators still see the gated card.
    assert gated in kanban_decompose.list_triage_ids()


def test_operator_comment_after_gate_releases(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        gated = _make_gated_triage(conn)
        assert cg.consent_gated_ids(conn, [gated]) == {gated}
        kb.add_comment(conn, gated, "human-operator", "here is the answer")
        assert cg.consent_gated_ids(conn, [gated]) == set()
    assert kanban_decompose.list_auto_decomposable_triage_ids() == [gated]


def test_machine_only_comments_hold(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    _roster(monkeypatch, "default", "architect", "verifier")
    with kbc.connect() as conn:
        gated = _make_gated_triage(conn)
        for author in (
            "default", "architect", "verifier",  # live roster names
            "auto-decomposer", "specifier", "decomposer", "kanban-empowered",  # fixed set
        ):
            kb.add_comment(conn, gated, author, "machine note")
        assert cg.consent_gated_ids(conn, [gated]) == {gated}


def test_bare_cli_profile_name_comment_holds(tmp_path, monkeypatch):
    """Documented limitation: the bare CLI writes the profile name as author;
    a 'default'-authored comment cannot release the gate (roster member)."""
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        gated = _make_gated_triage(conn)
        kb.add_comment(conn, gated, "default", "operator via bare CLI")
        assert cg.consent_gated_ids(conn, [gated]) == {gated}


def test_unknown_author_fails_open(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        gated = _make_gated_triage(conn)
        kb.add_comment(conn, gated, "someone-not-in-the-roster", "human answer")
        assert cg.consent_gated_ids(conn, [gated]) == set()


def test_dispatcher_and_worker_comments_hold(tmp_path, monkeypatch):
    """Security consult t_10d01d05 item 1: 'dispatcher' (reconciliation
    comments) and 'worker' (_persisted_identity fallback) are live machine
    authors — their comments must not release an operator gate."""
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        gated = _make_gated_triage(conn)
        kb.add_comment(conn, gated, "dispatcher", "reconciliation: requeued to ready")
        kb.add_comment(conn, gated, "worker", "progress note")
        assert cg.consent_gated_ids(conn, [gated]) == {gated}


def test_review_reopened_after_gate_releases(tmp_path, monkeypatch):
    """Security consult item 2: reopen_review_task (CLI reopen-review /
    dashboard drag out of review) writes review_reopened — an operator-only
    release signal the plan's original set missed."""
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        tid = _make_gated_in_lane(conn, "review")
        assert cg.consent_gated_ids(conn, [tid]) == {tid}
        assert kb.reopen_review_task(conn, tid)
        assert cg.consent_gated_ids(conn, [tid]) == set()
        spawn, calls = _spawn_stub()
        kbd.dispatch_once(conn, spawn_fn=spawn)
        assert tid in calls


def test_unblock_after_gate_releases(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        tid = _make_gated_blocked(conn)
        assert cg.consent_gated_ids(conn, [tid]) == {tid}
        assert kb.unblock_task(conn, tid)
        assert cg.consent_gated_ids(conn, [tid]) == set()
        assert kb.get_task(conn, tid).status == "ready"


def test_promote_after_gate_releases(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        tid = _make_gated_blocked(conn)
        ok, reason = kb.promote_task(conn, tid, actor="operator")
        assert ok, reason
        assert cg.consent_gated_ids(conn, [tid]) == set()
        assert kb.get_task(conn, tid).status == "ready"


def test_dashboard_drag_status_event_releases(tmp_path, monkeypatch):
    """The dashboard drag endpoint is the only 'status' event writer; mimic
    its exact effect (status flip + 'status' event) on a gated card."""
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        tid = _make_gated_blocked(conn)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET status = 'ready' WHERE id = ?", (tid,))
            kb._append_event(conn, tid, "status", {"from": "blocked", "to": "ready"})
        assert cg.consent_gated_ids(conn, [tid]) == set()


def test_release_signals_before_gate_do_not_release(tmp_path, monkeypatch):
    """Episode-scoped, NOT lifetime-zero: an older operator comment and an
    older unblock predate the newest gate and must not release it."""
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="episode", assignee="default")
        kb.add_comment(conn, tid, "human-operator", "old answer")
        _backdate_comment(conn, tid, "human-operator")
        _gate_from_ready(conn, tid)  # writes unblocked BEFORE the final gate
        assert cg.consent_gated_ids(conn, [tid]) == {tid}


def test_non_needs_input_blocks_unaffected(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        for kind in ("capability", "transient", None):
            tid = kb.create_task(conn, title=f"block {kind}", assignee="default")
            assert kb.block_task(conn, tid, reason="x", kind=kind)
            assert cg.consent_gated_ids(conn, [tid]) == set()
            # Breaker escalation keeps payload.kind != needs_input.
            assert kb.unblock_task(conn, tid)
            assert kb.block_task(conn, tid, reason="y", kind=kind)
            assert kb.get_task(conn, tid).status == "triage"
            assert cg.consent_gated_ids(conn, [tid]) == set()
        # dependency blocks never carry a gate kind; block_kind != needs_input.
        dep = kb.create_task(conn, title="dep", assignee="default", triage=True)
        with kb.write_txn(conn):
            conn.execute("UPDATE tasks SET block_kind = 'dependency' WHERE id = ?", (dep,))
            kb._append_event(conn, dep, "dependency_wait", {"kind": "dependency", "reason": "parent"})
        assert cg.consent_gated_ids(conn, [dep]) == set()


def test_completed_card_unaffected(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        tid = _make_gated_blocked(conn)
        assert kb.complete_task(conn, tid, result="resolved by operator")
        assert kb.get_task(conn, tid).status == "done"
        assert conn.execute(
            "SELECT block_kind FROM tasks WHERE id = ?", (tid,)
        ).fetchone()["block_kind"] is None
        assert cg.consent_gated_ids(conn, [tid]) == set()


def test_newest_gate_wins(tmp_path, monkeypatch):
    """A later non-needs_input block supersedes the operator gate."""
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        tid = _make_gated_blocked(conn)
        assert cg.consent_gated_ids(conn, [tid]) == {tid}
        assert kb.unblock_task(conn, tid)
        assert kb.block_task(conn, tid, reason="cap", kind="capability")
        assert cg.consent_gated_ids(conn, [tid]) == set()


def test_malformed_gate_payload_not_gated(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        for payload in (None, "not json", '{"kind": "capability"}'):
            tid = kb.create_task(conn, title="malformed", assignee="default")
            with kb.write_txn(conn):
                conn.execute(
                    "UPDATE tasks SET block_kind = 'needs_input' WHERE id = ?", (tid,)
                )
                if payload is None:
                    kb._append_event(conn, tid, "block_loop_detected", None)
                else:
                    conn.execute(
                        "INSERT INTO task_events (task_id, run_id, kind, payload, created_at) "
                        "VALUES (?, NULL, 'block_loop_detected', ?, 1)",
                        (tid, payload),
                    )
            assert cg.consent_gated_ids(conn, [tid]) == set()


def test_empty_ids(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        assert cg.consent_gated_ids(conn, []) == set()


# ---------------------------------------------------------------------------
# Kill switch: default, live toggle, error directions
# ---------------------------------------------------------------------------


def test_kill_switch_false_restores_old_behavior(tmp_path, monkeypatch):
    home = _isolated_home(tmp_path, monkeypatch)
    _set_kanban_config(home, skip_consent_gated=False)
    with kbc.connect() as conn:
        gated = _make_gated_in_lane(conn, "review")
        assert cg.consent_gated_ids(conn, [gated]) == set()
        spawn, calls = _spawn_stub()
        kbd.dispatch_once(conn, spawn_fn=spawn)
        assert gated in calls  # RED: pre-feature churn behavior


def test_live_toggle_between_ticks(tmp_path, monkeypatch):
    home = _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        gated = _make_gated_in_lane(conn, "review")
        spawn, calls = _spawn_stub()
        # Default (no config file) -> enabled -> held.
        kbd.dispatch_once(conn, spawn_fn=spawn)
        assert calls == []
        assert kb.get_task(conn, gated).status == "review"
        # Operator flips the flag; the NEXT tick honors it — no restart.
        _set_kanban_config(home, skip_consent_gated=False)
        kbd.dispatch_once(conn, spawn_fn=spawn)
        assert calls == [gated]
        # Flip back on with a fresh gated card -> held again.
        _set_kanban_config(home, skip_consent_gated=True)
        other = _make_gated_in_lane(conn, "ready")
        kbd.dispatch_once(conn, spawn_fn=spawn)
        assert calls == [gated]
        assert kb.get_task(conn, other).status == "ready"


def test_config_read_error_disables_skip(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    from hermes_cli import config

    def boom():
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(config, "load_config_readonly", boom)
    assert cg.skip_consent_gated_enabled() is False
    with kbc.connect() as conn:
        gated = _make_gated_in_lane(conn, "review")
        assert cg.consent_gated_ids(conn, [gated]) == set()
        spawn, calls = _spawn_stub()
        kbd.dispatch_once(conn, spawn_fn=spawn)
        assert gated in calls  # fail-open to old behavior


def test_roster_error_fails_open(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    from hermes_cli import profiles

    def boom(**kw):
        raise RuntimeError("roster unreadable")

    monkeypatch.setattr(profiles, "list_profiles", boom)
    assert cg.machine_authors() == {
        "auto-decomposer", "specifier", "decomposer", "kanban-empowered",
        "dispatcher", "worker",
    }
    with kbc.connect() as conn:
        gated = _make_gated_triage(conn)
        # 'default' is now an UNKNOWN author -> operator -> released.
        kb.add_comment(conn, gated, "default", "bare cli note")
        assert cg.consent_gated_ids(conn, [gated]) == set()


def test_predicate_error_fails_open_and_tick_survives(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        gated = _make_gated_in_lane(conn, "review")

        def boom(conn_, ids):
            raise RuntimeError("predicate exploded")

        monkeypatch.setattr(cg, "_gated_details", boom)
        assert cg.consent_gated_ids(conn, [gated]) == set()
        spawn, calls = _spawn_stub()
        kbd.dispatch_once(conn, spawn_fn=spawn)  # must not crash
        assert gated in calls
        assert _holds(conn, gated) == []


# ---------------------------------------------------------------------------
# Auto-decompose: budget, manual paths
# ---------------------------------------------------------------------------


def test_budget_not_consumed_by_held_cards(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(kanban_decompose, "_call_aux", _fake_aux(calls))
    with kbc.connect() as conn:
        g1 = _make_gated_triage(conn)
        g2 = _make_gated_triage(conn)
        fresh = kb.create_task(conn, title="fresh", triage=True)
    # per_tick=3 with 2 held + 1 fresh: the fresh card decomposes in ONE tick
    # and the held cards consume no budget (zero aux calls for them).
    assert _dispatcher().auto_decompose_tick(3) == 1
    assert calls == [fresh]
    with kbc.connect() as conn:
        assert kb.get_task(conn, g1).status == "triage"
        assert kb.get_task(conn, g2).status == "triage"
        assert kb.get_task(conn, fresh).status in ("ready", "review")
        assert len(_holds(conn, g1)) == 1
        assert len(_holds(conn, g2)) == 1


def test_kill_switch_false_budget_spends_on_gated(tmp_path, monkeypatch):
    home = _isolated_home(tmp_path, monkeypatch)
    _set_kanban_config(home, skip_consent_gated=False)
    calls = []
    monkeypatch.setattr(kanban_decompose, "_call_aux", _fake_aux(calls))
    with kbc.connect() as conn:
        g1 = _make_gated_triage(conn)
        g2 = _make_gated_triage(conn)
        fresh = kb.create_task(conn, title="fresh", triage=True)
    assert _dispatcher().auto_decompose_tick(3) == 3
    assert set(calls) == {g1, g2, fresh}  # RED: old behavior spends on gated


def test_manual_decompose_never_skips(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(kanban_decompose, "_call_aux", _fake_aux(calls))
    with kbc.connect() as conn:
        gated = _make_gated_triage(conn)
    assert gated in kanban_decompose.list_triage_ids()
    outcome = kanban_decompose.decompose_task(gated, author="operator")
    assert outcome.ok, outcome.reason
    assert calls == [gated]
    with kbc.connect() as conn:
        assert kb.get_task(conn, gated).status in ("ready", "review")


# ---------------------------------------------------------------------------
# Dispatch lanes + telemetry
# ---------------------------------------------------------------------------


def test_dispatch_skips_gated_both_lanes_claims_controls(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        gr = _make_gated_in_lane(conn, "ready")
        gv = _make_gated_in_lane(conn, "review")
        cr = kb.create_task(conn, title="control ready", assignee="default")
        cv = kb.create_task(conn, title="control review", assignee="default")
        _to_review(conn, cv)
        spawn, calls = _spawn_stub()
        claimed_before = {
            tid: [e.id for e in kb.list_events(conn, tid) if e.kind == "claimed"]
            for tid in (gr, gv)
        }
        res = kbd.dispatch_once(conn, spawn_fn=spawn)
        assert set(calls) == {cr, cv}
        for tid, lane in ((gr, "ready"), (gv, "review")):
            task = kb.get_task(conn, tid)
            assert task.status == lane
            assert task.claim_lock is None
            # Construction legitimately claimed these cards; the dispatch tick
            # must add NO new claim.
            assert [e.id for e in kb.list_events(conn, tid) if e.kind == "claimed"] == claimed_before[tid]
            assert len(_holds(conn, tid)) == 1
        assert res.skipped_nonspawnable == []
        assert res.respawn_guarded == []


def test_dry_run_writes_no_holds(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        gated = _make_gated_in_lane(conn, "review")
        spawn, calls = _spawn_stub()
        kbd.dispatch_once(conn, spawn_fn=spawn, dry_run=True)
        assert calls == []
        assert _holds(conn, gated) == []
        assert kb.get_task(conn, gated).status == "review"


def test_has_spawnable_false_on_only_gated_board(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        gr = _make_gated_in_lane(conn, "ready")
        gv = _make_gated_in_lane(conn, "review")
        assert kbd.has_spawnable_ready(conn) is False
        assert kbd.has_spawnable_review(conn) is False
        # Telemetry is a pure predicate: no hold events from these probes.
        assert _holds(conn, gr) == []
        assert _holds(conn, gv) == []
        kb.create_task(conn, title="control", assignee="default")
        assert kbd.has_spawnable_ready(conn) is True


def test_100_ticks_one_hold_then_regate_second_hold(tmp_path, monkeypatch):
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        tid = _make_gated_in_lane(conn, "review")
        gate1 = _gate_event(conn, tid)["id"]
        spawn, calls = _spawn_stub()
        for _ in range(100):
            kbd.dispatch_once(conn, spawn_fn=spawn)
        assert calls == []
        holds = _holds(conn, tid)
        assert len(holds) == 1
        assert holds[0].payload == {"gate_id": gate1, "kind": "needs_input"}
        assert kb.get_task(conn, tid).status == "review"
        # Operator releases; the dispatcher claims; the worker re-gates.
        kb.add_comment(conn, tid, "human-operator", "go ahead")
        kbd.dispatch_once(conn, spawn_fn=spawn)
        assert calls == [tid]
        assert kb.block_task(conn, tid, reason="new question", kind="needs_input")
        assert kb.get_task(conn, tid).status == "triage"
        gate2 = _gate_event(conn, tid)["id"]
        assert gate2 > gate1
        # The new question arrives minutes AFTER the operator answer (epoch
        # seconds would otherwise tie in this compressed rig).
        _advance_event(conn, gate2)
        assert kb.specify_triage_task(conn, tid, author="auto-decomposer")
        assert kb.get_task(conn, tid).status in ("ready", "review")
        # The pre-gate operator comment does NOT release the new episode.
        kbd.dispatch_once(conn, spawn_fn=spawn)
        holds = _holds(conn, tid)
        assert len(holds) == 2
        assert holds[1].payload == {"gate_id": gate2, "kind": "needs_input"}
        assert calls == [tid]  # never claimed again


def test_filter_site_error_fails_open(tmp_path, monkeypatch):
    """Security consult item 4: even if the predicate API itself raises at a
    filter site, the dispatch lane and telemetry fall back to old behavior —
    never stall, never crash the tick."""
    _isolated_home(tmp_path, monkeypatch)
    with kbc.connect() as conn:
        gated = _make_gated_in_lane(conn, "review")

        def boom(conn_, ids):
            raise RuntimeError("site-level failure")

        monkeypatch.setattr(cg, "consent_gated_ids", boom)
        assert kbd.has_spawnable_review(conn) is True  # legacy probe
        spawn, calls = _spawn_stub()
        kbd.dispatch_once(conn, spawn_fn=spawn)
        assert gated in calls
        assert _holds(conn, gated) == []


# ---------------------------------------------------------------------------
# Source pins (security consult t_10d01d05 items 1/5 + spec decision 1)
# ---------------------------------------------------------------------------


def _repo_root():
    return Path(__file__).resolve().parents[2]


def test_status_event_writers_are_dashboard_only():
    """Pin the 'status' release-signal writer set: exactly two writers today,
    both dashboard-drag-only (plugin_api drag endpoint + its descendant-
    invalidation helper, whose only caller is that endpoint). A future machine
    writer of 'status' events would silently auto-release gates — this test
    forces a conscious review of any such change."""
    root = _repo_root()
    append_writers = set()
    insert_writers = set()
    invalidate_callers = set()
    for subdir in ("hermes_cli", "gateway", "plugins", "tools", "agent"):
        for path in sorted((root / subdir).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            rel = path.relative_to(root).as_posix()
            if re.search(r"_append_event\(\s*[^,]+,\s*[^,]+,\s*[\"']status[\"']", text):
                append_writers.add(rel)
            if re.search(
                r"INSERT\s+INTO\s+task_events\s*\([^)]*\)\s*VALUES\s*\([^)]*[\"']status[\"']",
                text,
            ):
                insert_writers.add(rel)
            if (
                "invalidate_descendants_for_parent_reopen(" in text
                and rel != "hermes_cli/kanban_db.py"  # definition site
            ):
                invalidate_callers.add(rel)
    assert append_writers == {"hermes_cli/kanban_db.py"}, append_writers
    assert insert_writers == {"plugins/kanban/dashboard/plugin_api.py"}, insert_writers
    assert invalidate_callers == {"plugins/kanban/dashboard/plugin_api.py"}, invalidate_callers


def test_module_import_surface_pinned():
    """Spec decision 1: the new module never imports the dead refactor
    siblings, and its module-level import surface stays stdlib-only
    (profiles/config/kanban_db are late-bound inside functions)."""
    src = (_repo_root() / "hermes_cli" / "kanban_consent_gate.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert "kanban_blocked_triage" not in name, name
            assert "kanban_db_review" not in name, name
    allowed = {"__future__", "json", "logging", "sqlite3", "typing"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert all(a.name.split(".")[0] in allowed for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] in allowed
