"""Consent-gate hold: let operator-gated kanban cards rest until a human acts.

A card is CONSENT-GATED iff ALL of:

  (a) ``tasks.block_kind = 'needs_input'``;
  (b) its newest ``blocked``/``block_loop_detected`` event (the "gate") carries
      ``payload.kind = 'needs_input'`` — an operator gate, not a dependency /
      capability / transient / failure block;
  (c) no operator activity at/after that gate: no comment with
      ``created_at >= gate.created_at`` whose author is outside the machine
      author set, and no ``unblocked`` / ``status`` / ``promoted_manual`` /
      ``review_reopened`` event with ``id > gate.id`` (dashboard drag is the
      only ``status`` writer; ``review_reopened`` comes only from
      ``reopen_review_task`` — CLI reopen-review / dashboard drag out of
      review).

The predicate is EPISODE-scoped: machine comments and release events that
predate the newest gate do not release it. Unknown comment authors count as
operators (fail-open: a card we cannot classify is never held).

Machine sweeps (auto-decompose tick, ready/review dispatch lanes, spawnable
telemetry) skip gated cards so they rest operator-visible instead of churning
block_loop_detected -> triage -> specify -> re-promote -> re-claim. Manual
paths (CLI single-id / --all decompose, dashboard specify/decompose endpoints)
are deliberately NOT filtered — an operator naming a card is itself consent.

Kill switch: ``kanban.skip_consent_gated`` (bool, default ``true``), re-read
live from config on every sweep — flipping it takes effect on the next tick
without a gateway restart, mirroring ``kanban.auto_decompose`` (#49638). A
config-read error disables the skip (fail-open to the old behavior). No
config.yaml edit is required for the default.

Held cards get ONE deduped ``consent_gate_hold`` event per gate episode
(``{"gate_id": <gate event id>, "kind": "needs_input"}``) so ``kanban tail`` /
the dashboard can explain the silence. The kind is intentionally absent from
the notifier's TERMINAL_KINDS — holds never notify.

The predicate core depends only on sqlite3 + hermes_cli.profiles +
hermes_cli.config (late-bound inside functions, sibling pattern); the hold
writer late-binds kanban_db for ``write_txn``/``_append_event`` at call time.
The dead refactor siblings ``kanban_blocked_triage`` / ``kanban_db_review``
are never imported.

Trust model: the hold is a churn control for COOPERATIVE automation, not an
adversarial boundary. A worker running as the same OS user can already
self-release today via the CLI or a raw import (pre-existing capability, no
new privilege); the predicate only stops the machine sweeps from re-promoting
and re-claiming cards whose operator gate is unacknowledged.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# Fixed machine authors unioned with the live profile roster. Kept in sync
# with kanban_decompose._build_roster's extras by convention (spec decision 1),
# plus the two live machine writers the security consult verified (t_10d01d05):
# 'dispatcher' — reconciliation comments (kanban_db_dispatch.reconcile_orphaned_running);
# 'worker'     — tools/kanban_tools._persisted_identity fallback when no profile is named.
_FIXED_MACHINE_AUTHORS = frozenset(
    {"auto-decomposer", "specifier", "decomposer", "kanban-empowered", "dispatcher", "worker"}
)

# Event kinds that can carry an operator gate (payload.kind decides).
_GATE_KINDS = ("blocked", "block_loop_detected")

# Event kinds that release a gate when written AFTER it (id > gate.id).
# 'status' is written only by the dashboard drag endpoint (and its
# descendant-invalidation helper, itself called only by that endpoint);
# 'promoted_manual' only by promote_task; 'unblocked' only by unblock_task;
# 'review_reopened' only by reopen_review_task (CLI reopen-review / dashboard
# drag out of review). Every operator path that lands a gated card in a
# dispatchable lane writes one of these. The writer set is pinned by
# tests/gateway/test_kanban_consent_gate.py so a future machine writer cannot
# silently auto-release gates.
_RELEASE_KINDS = ("unblocked", "status", "promoted_manual", "review_reopened")

HOLD_EVENT_KIND = "consent_gate_hold"


def skip_consent_gated_enabled() -> bool:
    """Live read of ``kanban.skip_consent_gated``; default True.

    Mirrors gateway/kanban_watchers_common._resolve_auto_decompose_settings:
    re-read every sweep so an operator flip applies next tick, and a config
    read error returns False (fail-open to the pre-feature behavior).
    """
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
    except Exception:
        return False
    kcfg = cfg.get("kanban", {}) if isinstance(cfg, dict) else {}
    return bool(kcfg.get("skip_consent_gated", True))


def machine_authors() -> set[str]:
    """Live profile roster names ∪ fixed machine set. Read once per sweep.

    A roster read failure keeps only the fixed set: profile-name authors then
    classify as unknown -> operator -> released (fail-open, never held).
    ``lazy_skill_count`` keeps this polled path off the skill-tree walk.
    """
    authors = set(_FIXED_MACHINE_AUTHORS)
    try:
        from hermes_cli import profiles as profiles_mod

        authors.update(p.name for p in profiles_mod.list_profiles(lazy_skill_count=True))
    except Exception:
        logger.warning(
            "kanban consent-gate: profile roster unreadable; "
            "using fixed machine authors only",
            exc_info=True,
        )
    return authors


def _payload_dict(raw: Any) -> dict:
    """Tolerant JSON payload parse: NULL / malformed / non-object -> {}."""
    if not raw:
        return {}
    try:
        val = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return val if isinstance(val, dict) else {}


def _gated_details(conn: sqlite3.Connection, ids: list[str]) -> dict[str, dict]:
    """Batch predicate core: ``{task_id: {"gate_id": int, "kind": "needs_input"}}``.

    Pure read, parameterized, no LLM. Four bounded queries (candidates, newest
    gate per candidate, comments at/after gates, release events after gates);
    the roster is read once. Callers guard the kill switch and fail open.
    """
    placeholders = ",".join("?" for _ in ids)
    # (a) Only needs_input-blocked cards can carry an operator gate.
    candidates = [
        r["id"]
        for r in conn.execute(
            f"SELECT id FROM tasks WHERE id IN ({placeholders}) "
            "AND block_kind = 'needs_input'",
            ids,
        )
    ]
    if not candidates:
        return {}
    cph = ",".join("?" for _ in candidates)
    # (b) Newest blocked/block_loop_detected event per candidate (MAX(id) is
    # the insert-order newest; created_at is epoch seconds and can tie).
    gate_rows = conn.execute(
        "SELECT e.task_id, e.id AS gate_id, e.created_at AS gate_at, e.payload "
        "FROM task_events e JOIN ("
        "  SELECT task_id, MAX(id) AS mid FROM task_events "
        f"  WHERE task_id IN ({cph}) AND kind IN ({','.join('?' for _ in _GATE_KINDS)}) "
        "  GROUP BY task_id"
        ") g ON g.task_id = e.task_id AND g.mid = e.id",
        (*candidates, *_GATE_KINDS),
    ).fetchall()
    gates: dict[str, dict] = {}
    for r in gate_rows:
        if _payload_dict(r["payload"]).get("kind") == "needs_input":
            gates[r["task_id"]] = {"gate_id": r["gate_id"], "gate_at": r["gate_at"]}
    if not gates:
        return {}
    # (c) Operator activity at/after the gate releases the episode.
    authors = machine_authors()
    gated_ids = list(gates)
    gph = ",".join("?" for _ in gated_ids)
    released: set[str] = set()
    min_gate_at = min(g["gate_at"] for g in gates.values())
    for r in conn.execute(
        "SELECT task_id, author, created_at FROM task_comments "
        f"WHERE task_id IN ({gph}) AND created_at >= ?",
        (*gated_ids, min_gate_at),
    ):
        gate = gates[r["task_id"]]
        # Unknown author -> operator -> released (fail-open).
        if r["created_at"] >= gate["gate_at"] and (r["author"] or "") not in authors:
            released.add(r["task_id"])
    for r in conn.execute(
        "SELECT task_id, id FROM task_events "
        f"WHERE task_id IN ({gph}) "
        f"AND kind IN ({','.join('?' for _ in _RELEASE_KINDS)})",
        (*gated_ids, *_RELEASE_KINDS),
    ):
        if r["id"] > gates[r["task_id"]]["gate_id"]:
            released.add(r["task_id"])
    return {
        tid: {"gate_id": g["gate_id"], "kind": "needs_input"}
        for tid, g in gates.items()
        if tid not in released
    }


def consent_gated_ids(conn: sqlite3.Connection, task_ids: Iterable[str]) -> set[str]:
    """Subset of *task_ids* that are consent-gated. Batch SQL, pure read.

    Kill switch off -> empty set. Any internal error -> empty set (fail-open:
    a broken predicate must never hold cards nor crash a dispatch tick).
    """
    try:
        ids = list(dict.fromkeys(str(t) for t in task_ids))
        if not ids or not skip_consent_gated_enabled():
            return set()
        return set(_gated_details(conn, ids))
    except Exception:
        logger.warning(
            "kanban consent-gate: predicate failed; failing open (no holds)",
            exc_info=True,
        )
        return set()


def record_consent_gate_holds(conn: sqlite3.Connection, task_ids: Iterable[str]) -> None:
    """Append ONE deduped ``consent_gate_hold`` event per gate episode.

    Best-effort observability for the sweeps that hold cards: writes only when
    the newest event is not already a hold for the same gate id (dedup pattern
    of ``skipped_nonspawnable`` in kanban_db_dispatch). A benign cross-writer
    race may yield <= 2 events per episode; that is acceptable. Never raises —
    a failed hold write must not break the sweep that called it.
    """
    try:
        ids = list(dict.fromkeys(str(t) for t in task_ids))
        if not ids or not skip_consent_gated_enabled():
            return
        details = _gated_details(conn, ids)
        if not details:
            return
        # Late-bound at call time: by now kanban_db is fully populated, and the
        # predicate core keeps its sqlite3/profiles/config-only import surface.
        from hermes_cli import kanban_db as kb

        for task_id, gate in details.items():
            payload = {"gate_id": gate["gate_id"], "kind": "needs_input"}
            with kb.write_txn(conn):
                last = conn.execute(
                    "SELECT kind, payload FROM task_events WHERE task_id = ? "
                    "ORDER BY created_at DESC, id DESC LIMIT 1",
                    (task_id,),
                ).fetchone()
                if last is not None and last["kind"] == HOLD_EVENT_KIND:
                    if _payload_dict(last["payload"]).get("gate_id") == gate["gate_id"]:
                        continue
                kb._append_event(conn, task_id, HOLD_EVENT_KIND, payload)
    except Exception:
        logger.warning(
            "kanban consent-gate: hold-event write failed; continuing",
            exc_info=True,
        )
