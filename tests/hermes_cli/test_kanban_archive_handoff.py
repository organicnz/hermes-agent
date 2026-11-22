"""``archive_task`` refuses a terminal transition with no record on the board.

Measured class (4/12 archived unguarded): drain parents went
``promoted -> archived`` with no ``completed`` event at all. An archive now
requires a handoff (a ``completed`` event with a substantive summary, or a
stored result) or a named PIN exception, which is itself recorded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


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


def _kinds(conn, tid) -> list[str]:
    return [r["kind"] for r in conn.execute(
        "SELECT kind FROM task_events WHERE task_id = ? ORDER BY id", (tid,)
    ).fetchall()]


def test_archive_without_handoff_refused(conn):
    tid = kb.create_task(conn, title="drain parent", assignee="coder")
    before = conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"]
    assert kb.archive_task(conn, tid) is False
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == before
    assert "archive_blocked_no_handoff" in _kinds(conn, tid)
    assert "archived" not in _kinds(conn, tid)


def test_archive_with_handoff_allowed(conn):
    tid = kb.create_task(conn, title="completed card", assignee="coder")
    assert kb.claim_task(conn, tid, claimer=kb._claimer_id()) is not None
    assert kb.complete_task(conn, tid, summary="the handoff on the record") is True
    assert kb.archive_task(conn, tid) is True
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "archived"


def test_archive_pin_recorded_and_allowed(conn):
    import json
    tid = kb.create_task(conn, title="housekeeping card", assignee="coder")
    assert kb.archive_task(conn, tid, pin="operator_cleanup_20261121") is True
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "archived"
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'archive_pinned'", (tid,)
    ).fetchall()
    assert len(rows) == 1
    assert json.loads(rows[0]["payload"])["pin"] == "operator_cleanup_20261121"


def test_refusal_then_pin_allows(conn):
    tid = kb.create_task(conn, title="retry card", assignee="coder")
    assert kb.archive_task(conn, tid) is False
    assert kb.archive_task(conn, tid, pin="drain_20261121") is True
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "archived"


def test_stored_result_counts_as_handoff(conn):
    tid = kb.create_task(conn, title="result-only card", assignee="coder")
    conn.execute("UPDATE tasks SET result = ? WHERE id = ?", ("recorded outcome", tid))
    conn.commit()
    assert kb.archive_task(conn, tid) is True
    assert conn.execute("SELECT status FROM tasks WHERE id = ?", (tid,)).fetchone()["status"] == "archived"


def test_whitespace_handoff_does_not_count(conn):
    tid = kb.create_task(conn, title="whitespace card", assignee="coder")
    conn.execute("UPDATE tasks SET result = ? WHERE id = ?", ("   ", tid))
    conn.commit()
    assert kb.archive_task(conn, tid) is False
    assert "archive_blocked_no_handoff" in _kinds(conn, tid)
