"""Completion gates for the kanban lifecycle kernel.

``_gate_empty_completion`` refuses evidence-free completions (review approvals
exempt). ``route_worker_completion`` is the completion review gate: a worker's
direct self-completion from a non-review status cannot reach ``done`` — it is
machine-routed through ``request_review`` so a review round closes the card
(false-completion class t_60130e9e / t_f4635fcb / t_8e5bb2a9). Every gate
decision leaves a ``completion_routed_review`` event; a refused routing fails
closed (the caller must not fall through to ``done``).
"""

from __future__ import annotations

import sqlite3
from typing import Optional

from hermes_cli.kanban_db_connect import write_txn


def _substantive_text(value: Optional[str]) -> bool:
    return bool(value is not None and str(value).strip())


def _gate_empty_completion(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    result: Optional[str],
    summary: Optional[str],
) -> None:
    """Refuse a completion that would leave the card with no evidence.

    Review approvals are exempt: a human vouches for the card and
    ``_REVIEW_APPROVED_NOTE`` is the documented record.
    """
    row = conn.execute(
        "SELECT status, result FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if row is None:
        return
    if row["status"] == "review":
        return
    stored = row["result"]
    if _substantive_text(result) or _substantive_text(summary) or _substantive_text(stored):
        return
    with write_txn(conn):
        _kb._append_event(
            conn, task_id, "completion_blocked_empty_result",
            {
                "result_preview": _kb._first_line(result, 200) or None,
                "summary_preview": _kb._first_line(summary, 200) or None,
            },
        )
    raise _kb.EmptyCompletionError(task_id)


def route_worker_completion(
    conn: sqlite3.Connection,
    task_id: str,
    *,
    summary: Optional[str],
    metadata: Optional[dict],
    expected_run_id: Optional[int],
) -> Optional[bool]:
    """Route a run-owner's self-completion to ``review`` instead of ``done``.

    Returns ``None`` when the gate does not apply (no worker run marker, a
    review-lane run — the review round itself — or a stale marker: the caller
    completes normally), ``True`` when the card landed in ``review``,
    ``False`` when the routing was refused: fail-closed, the caller must not
    fall through to ``done``.
    Reviewer resolution stays inside ``request_review`` (``reviewer=None``):
    a re-review without durable provenance refuses rather than route around
    the refusal into self-approval.
    """
    if expected_run_id is None:
        return None
    row = conn.execute(
        "SELECT status, current_run_id FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    if row is None or row["status"] not in ("running", "ready", "blocked"):
        return None
    if row["current_run_id"] != int(expected_run_id):
        return None
    # A run claimed from the review lane IS the review round: its completion
    # is the compliant approval path (the claim moves review -> running, so
    # status alone cannot tell it apart). _retry_status_for_run is the one
    # place that reads the claim provenance, crash/reclaim-safe.
    if _kb._retry_status_for_run(conn, task_id, int(expected_run_id)) == "review":
        return None
    prior_status = row["status"]
    try:
        ok, detail = _kb.request_review(
            conn, task_id, summary=summary, metadata=metadata,
            expected_run_id=expected_run_id, with_reason=True,
        )
    except Exception as exc:
        _record_gate_decision(conn, task_id, "error", prior_status,
                              type(exc).__name__, expected_run_id)
        raise
    _record_gate_decision(
        conn, task_id, "routed" if ok else "refused", prior_status,
        None if ok else detail, expected_run_id,
    )
    return ok


def _record_gate_decision(
    conn: sqlite3.Connection, task_id: str, decision: str,
    prior_status: str, detail: Optional[str], run_id: int,
) -> None:
    """The gate's decision is auditable on every firing — routed, refused, or
    error — in its own transaction after the delegation attempt settled."""
    with write_txn(conn):
        _kb._append_event(
            conn, task_id, "completion_routed_review",
            {
                "decision": decision,
                "reason": "worker_direct_completion",
                "prior_status": prior_status,
                "detail": detail,
            },
            run_id=run_id,
        )


# Late-bound origin namespace (see module docstring).
from hermes_cli import kanban_db as _kb
