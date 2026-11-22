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

import json
import re
import sqlite3
import subprocess
from pathlib import Path
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


# ---------------------------------------------------------------------------
# Completion integrity: the declared landing must be on the named ref
# ---------------------------------------------------------------------------

_LANDING_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
_LANDING_REF_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
_LS_REMOTE_TIMEOUT_S = 30
_FETCH_TIMEOUT_S = 60


def _extract_landings(metadata: Optional[dict]) -> list[dict]:
    """Return the git-landing declarations from a completion's metadata.

    Contract: a card that lands a commit declares it in its completion
    metadata as ``landed`` — a dict, or a list of dicts, each carrying
    ``repo`` (path to a local checkout whose ``origin`` remote is the
    canonical remote), ``ref`` (the named ref, e.g. ``main``) and ``sha``
    (the landed commit, full or unambiguous short). The gate applies to every
    declared landing; a completion that declares none is not a landing card
    and passes untouched (the board-level audit catches declared-as-prose
    landings heuristically).
    """
    if not isinstance(metadata, dict):
        return []
    raw = metadata.get("landed")
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    return [item for item in items if isinstance(item, dict)]


def _git(repo: Path, *args: str, timeout: int) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        return 124, "", f"git {args[0]} unavailable: {exc}"
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def _last_line(text: str, fallback: str) -> str:
    lines = [line for line in str(text).splitlines() if line.strip()]
    return lines[-1] if lines else fallback


def _verify_landing_on_ref(landing: dict) -> tuple[bool, str, str]:
    """Verify one declared landing: the ref exists on origin and the sha is
    the ref's tip or an ancestor of it. Returns ``(ok, origin_tip, reason)``.

    Fail-closed by construction: any step the check cannot prove (bad input,
    missing repo, network failure, unknown object) is a refusal. The
    dangling case — a worktree commit never pushed — resolves locally but is
    not an ancestor of the origin tip, so it is refused as ``sha_not_on_ref``;
    the same commit after its worktree is garbage-collected refuses as
    ``sha_unknown``.
    """
    ref = str(landing.get("ref") or "").strip()
    sha = str(landing.get("sha") or "").strip().lower()
    if not ref or not _LANDING_REF_RE.match(ref):
        return False, "", "malformed_ref"
    if not _LANDING_SHA_RE.match(sha):
        return False, "", "malformed_sha"
    repo = Path(str(landing.get("repo") or "")).expanduser()
    if not repo.is_dir():
        return False, "", "repo_not_found"
    code, _, err = _git(repo, "rev-parse", "--git-dir", timeout=10)
    if code != 0:
        return False, "", "repo_not_git"

    # 1. The named ref must exist on origin, and the check must see its tip.
    code, out, err = _git(repo, "ls-remote", "origin", ref, timeout=_LS_REMOTE_TIMEOUT_S)
    if code != 0:
        return False, "", f"ls-remote_failed: {_last_line(err, 'git rc=%d' % code)}"
    tip = out.split()[0] if out else ""
    if not tip:
        return False, "", "ref_not_on_origin"
    if len(sha) == 40 and tip == sha:
        return True, tip, ""

    # 2. Tip or ancestor: the ref's history must be present locally.
    code, _, err = _git(repo, "fetch", "--quiet", "--no-tags", "origin", ref, timeout=_FETCH_TIMEOUT_S)
    if code != 0:
        return False, tip, f"fetch_failed: {_last_line(err, 'git rc=%d' % code)}"
    resolved = sha
    if len(sha) < 40:
        code, out, err = _git(repo, "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}", timeout=10)
        if code != 0:
            return False, tip, "sha_unknown"
        resolved = out
    code, _, err = _git(repo, "cat-file", "-e", f"{resolved}^{{commit}}", timeout=10)
    if code != 0:
        return False, tip, "sha_unknown"
    code, _, err = _git(repo, "merge-base", "--is-ancestor", resolved, tip, timeout=10)
    if code != 0:
        return False, tip, "sha_not_on_ref"
    return True, tip, ""


def _gate_unpushed_artifact(
    conn: sqlite3.Connection, task_id: str, *, metadata: Optional[dict],
) -> None:
    """Refuse a completion whose declared landing is not on the named ref.

    The refused class (t_b4510271 / dangling-42c2b998): a card closed
    ``done`` with its commit only on a local worktree — ``git ls-remote
    origin <ref>`` never resolved to the landed SHA — then the worktree was
    discarded and a downstream sync copied the orphaned commit into a second
    repository. Every declared ``metadata["landed"]`` line must verify or the
    completion is refused fail-closed with an auditable
    ``completion_blocked_unpushed_artifact`` event; the task stays where it
    was and the worker can push and retry.
    """
    for landing in _extract_landings(metadata):
        ok, tip, reason = _verify_landing_on_ref(landing)
        if ok:
            continue
        with write_txn(conn):
            _kb._append_event(
                conn, task_id, "completion_blocked_unpushed_artifact",
                {
                    "repo": str(landing.get("repo") or ""),
                    "ref": str(landing.get("ref") or ""),
                    "sha": str(landing.get("sha") or "").lower(),
                    "origin_tip": tip or None,
                    "reason": reason,
                },
            )
        raise _kb.UnpushedArtifactError(
            task_id,
            ref=str(landing.get("ref") or ""),
            sha=str(landing.get("sha") or "").lower(),
            reason=reason,
        )


def _gate_archive_handoff(
    conn: sqlite3.Connection, task_id: str, *, pin: Optional[str] = None,
) -> bool:
    """Refuse an archive that would leave a terminal transition with no
    record of what happened.

    Measured class (4/12 archived unguarded): drain parents went
    ``promoted -> archived`` with no ``completed`` event at all. A handoff
    exists when the card carries a ``completed`` event with a substantive
    ``summary`` or a substantive stored ``result``; otherwise the archive
    needs a named PIN exception (``pin``), which is itself recorded as an
    ``archive_pinned`` event so the exception is on the record. Returns True
    to allow the archive, False after an auditable
    ``archive_blocked_no_handoff`` event (the caller must not fall through).
    """
    row = conn.execute(
        "SELECT result FROM tasks WHERE id = ?", (task_id,)
    ).fetchone()
    if row is None:
        return False
    completed = conn.execute(
        "SELECT payload FROM task_events "
        "WHERE task_id = ? AND kind = 'completed' ORDER BY id DESC LIMIT 1",
        (task_id,),
    ).fetchone()
    summary: Optional[str] = None
    if completed and completed["payload"]:
        try:
            payload = json.loads(completed["payload"])
            if isinstance(payload, dict):
                summary = payload.get("summary")
        except (TypeError, ValueError):
            summary = None
    if _substantive_text(summary) or _substantive_text(row["result"]):
        return True
    pin_name = str(pin or "").strip()
    with write_txn(conn):
        if pin_name:
            _kb._append_event(conn, task_id, "archive_pinned", {"pin": pin_name})
            return True
        _kb._append_event(
            conn, task_id, "archive_blocked_no_handoff",
            {
                "result_preview": _kb._first_line(row["result"], 200) or None,
            },
        )
    return False


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
