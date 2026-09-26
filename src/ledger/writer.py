"""Action Ledger — the v2 audit spine.

Every agent action writes a row to `agent_actions` **before** the output is
emitted to the customer or console. This is the architecture's core invariant:

    No ledger row, no output.

The ledger is the single source of truth for:
  - What the AI did
  - On whose data (evidence_ids)
  - Whether the action succeeded
  - How long it took (duration_ms)
  - Whether a human took over (agent='supervisor')

Qubot v2's post-contact audit reads this ledger to verify groundedness and
produce the contact audit report.
"""

from __future__ import annotations

import json
import logging
import time
from src.ids import new_ulid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.data.warehouse import ops_con

logger = logging.getLogger(__name__)


# ── Ledger health (item 17) ─────────────────────────────────────────────────
# The main action INSERT raises on failure (callers always know the core
# audit write failed — never a silent success). Auxiliary writes (Merkle
# leaf, claim rows, evidence pins) degrade instead of breaking the contact,
# but degradation is OBSERVABLE: a process-wide counter, a structured log
# record, and the per-call degradation list (see record_action's
# on_degraded hook / record_action_with_status).

import threading as _threading

_health_lock = _threading.Lock()
_degraded_total = 0
_degraded_by_reason: dict[str, int] = {}
_last_degraded: dict[str, Any] | None = None


def _note_degraded(reason: str, *, action_id: str, interaction_id: str, error: str) -> None:
    """Record one auxiliary-ledger degradation (structured + counted)."""
    global _degraded_total, _last_degraded
    with _health_lock:
        _degraded_total += 1
        _degraded_by_reason[reason] = _degraded_by_reason.get(reason, 0) + 1
        _last_degraded = {
            "reason": reason,
            "action_id": action_id,
            "interaction_id": interaction_id,
            "error": error,
        }
    logger.warning(
        "ledger auxiliary write degraded",
        extra={
            "ledger_reason": reason,
            "action_id": action_id,
            "interaction_id": interaction_id,
            "error": error,
        },
    )
    try:
        from src.observability.metrics import inc as _inc

        _inc("audit_degraded", reason=reason)
    except Exception:
        pass


def ledger_health() -> dict[str, Any]:
    """Health state for the audit spine (wired to /ops/slo job alerts)."""
    with _health_lock:
        total = _degraded_total
        by_reason = dict(_degraded_by_reason)
        last = dict(_last_degraded) if _last_degraded else None
    return {
        "degraded_total": total,
        "degraded_by_reason": by_reason,
        "last_degraded": last,
        # Alert threshold (item 17): any sustained auxiliary failure rate
        # deserves a page — single incidents are warnings, 10+ is alerting.
        "alert_threshold": 10,
        "alerting": total >= 10,
    }


def reset_ledger_health() -> None:
    """Reset auxiliary degradation counters (between test runs or recovery)."""
    global _degraded_total, _last_degraded
    with _health_lock:
        _degraded_total = 0
        _degraded_by_reason.clear()
        _last_degraded = None


def _now() -> datetime:
    from src.data.timeutil import utc_now

    return utc_now()


def _ts_for_storage(ts: datetime | None) -> datetime:
    from src.data.timeutil import to_naive_utc

    return to_naive_utc(ts) if ts is not None else _now()


def _ulid() -> str:
    return new_ulid()


# ── Action type vocabulary ──────────────────────────────────────────────────
# These are the canonical action_type values. The orchestrator and agents
# MUST use these so Qubot's retrievers can aggregate them deterministically.

ACTION_TYPES = {
    # Orchestrator lifecycle
    "interaction_started",
    "greeting_emitted",
    "goodbye_emitted",
    "handoff_offer_emitted",
    "handoff_accepted",                # customer accepted; waiting in supervisor queue
    "handoff_unfulfilled",             # SLA timeout with no supervisor pickup
    "case_attached_existing",          # returning-customer attach instead of new case
    "start_deduped",                   # duplicate start replayed, no new interaction
    "escalation_script_emitted",
    "state_transition",
    "interaction_ended",
    "interaction_abandoned",
    "takeover_started",
    "takeover_released",
    "human_turn",                      # supervisor turn (also ledgered)
    "human_override",                  # supervisor slot/severity edit + reason (board)
    "supervised_turn_observed",        # customer turn received during SUPERVISED (safety-processed, generation suppressed)
    "supervised_safety_raised",        # kill-switch/safety hit during SUPERVISED (surfaced to console, not auto-emitted)
    "alert_sent",
    "connector_dispatched",
    "signal_exported",
    # Intake
    "slot_extracted",
    "question_asked",
    "safety_flag_raised",
    "intake_completed",
    # Sentiment
    "turn_scored",
    "frustration_flagged",
    # Triage
    "severity_scored",
    "priority_assigned",
    # Sentinel
    "advisory_check",
    "advisory_notified",
    "escalated",
    # Investigator
    "similar_search",
    "cluster_matched",
    "embedding_recorded",
    "spike_checked",
    "brief_written",
    # Case
    "case_created",
    "followup_drafted",
    "remedy_offered",
    "investigation_linked",
    "investigation_opened",
    "case_status_updated",
    "case_note_added",
    "followup_saved",
    "investigation_status_updated",
    # Multi-issue / four-eyes / coach (platform extensions)
    "multi_issue_attach_failed",
    "approval_requested",
    "approval_decided",
    "callback_scheduled",
    "appointment_booked",
    "coach_whisper",
    "spoken_confirmation",
    "diagnostic_asked",
    "live_intercept",
    "reproduced",
    # Voice delivery and barge-in audit types
    "spoken_turn_truncated",
    "spoken_turn_reconciled",
    "playback_started",
    "playback_completed",
    "playback_interrupted",
    "voice_turn_accepted",
    "slot_confirmed",
    "slot_corrected",
    # Tier-A voice wiring (consent / drive-mode / frame confirmation / VIN)
    "voice_consent_required",
    "voice_consent_obtained",
    "drive_mode_offered",
    "frame_confirmation_asked",
    "frame_confirmation_evaluated",
    "vin_captured",
    "vin_retry_asked",
}


# ── Public dataclass ────────────────────────────────────────────────────────


@dataclass
class AgentAction:
    """A single auditable agent action.

    Use `record_action()` to write one of these to the ledger.
    """

    interaction_id: str
    agent: str                       # intake | sentiment | triage | sentinel | investigator | case | orchestrator | supervisor
    action_type: str                 # one of ACTION_TYPES
    input_summary: str = ""
    output_summary: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    claims: list[Any] = field(default_factory=list)
    case_id: str | None = None
    ok: bool = True
    error: str | None = None
    duration_ms: int | None = None
    action_id: str = field(default_factory=_ulid)
    ts: datetime = field(default_factory=_now)
    hash_version: int = 2
    content_hash: str | None = None


# ── Writer ──────────────────────────────────────────────────────────────────


def _insert_action_row(con, action: AgentAction) -> str:
    """Insert one agent_actions row on an existing connection (no open/close)."""
    if action.action_type not in ACTION_TYPES:
        raise ValueError(
            f"unknown action_type {action.action_type!r}; "
            f"must be one of the ledger ACTION_TYPES vocabulary"
        )
    from src.ledger.chain import GENESIS, compute_content_hash, compute_row_hash

    evidence_json = json.dumps(action.evidence_ids)
    claims_json = (
        action.claims
        if isinstance(action.claims, str)
        else json.dumps(action.claims or [], sort_keys=True)
    )
    ts_stored = _ts_for_storage(action.ts)
    # Hash chain: previous row for this interaction (append-only order by ts/action_id).
    prev_row = con.execute(
        """
        SELECT row_hash, ts FROM agent_actions
        WHERE interaction_id = ?
        ORDER BY ts DESC, action_id DESC
        LIMIT 1
        """,
        [action.interaction_id],
    ).fetchone()
    prev_hash = (prev_row[0] if prev_row and prev_row[0] else GENESIS) or GENESIS
    if prev_row and prev_row[1]:
        from datetime import timedelta
        prev_ts = _ts_for_storage(prev_row[1])
        if ts_stored <= prev_ts:
            ts_stored = prev_ts + timedelta(microseconds=100)

    hash_v = getattr(action, "hash_version", 2) or 2
    row_payload = {
        "action_id": action.action_id,
        "interaction_id": action.interaction_id,
        "case_id": action.case_id,
        "agent": action.agent,
        "action_type": action.action_type,
        "input_summary": (action.input_summary or "")[:500],
        "output_summary": (action.output_summary or "")[:500],
        "evidence_ids": evidence_json,
        "claims": claims_json,
        "ok": action.ok,
        "error": action.error,
        "duration_ms": action.duration_ms,
        "ts": str(ts_stored),
        "hash_version": hash_v,
    }
    content_hash = compute_content_hash(row_payload, version=hash_v)
    row_payload["content_hash"] = content_hash
    action.content_hash = content_hash
    row_hash = compute_row_hash(row_payload, prev_hash, version=hash_v)

    con.execute(
        """
        INSERT INTO agent_actions (
            action_id, interaction_id, case_id, agent, action_type,
            input_summary, output_summary, evidence_ids,
            ok, error, duration_ms, ts, prev_hash, row_hash,
            hash_version, content_hash, claims
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            action.action_id,
            action.interaction_id,
            action.case_id,
            action.agent,
            action.action_type,
            (action.input_summary or "")[:500],
            (action.output_summary or "")[:500],
            evidence_json,
            action.ok,
            action.error,
            action.duration_ms,
            ts_stored,
            prev_hash,
            row_hash,
            hash_v,
            content_hash,
            claims_json,
        ],
    )
    return row_hash


_PACK_BY_INTERACTION: dict[str, str] = {}


def remember_interaction_pack(interaction_id: str, pack_id: str) -> None:
    if interaction_id and pack_id:
        _PACK_BY_INTERACTION[interaction_id] = pack_id


def record_action(action: AgentAction, *, on_degraded=None) -> str:
    """Write an action to the ledger. Returns the action_id.

    This is the ONLY way an agent may produce a side effect. The row is
    written BEFORE any output (text turn, console card) is emitted — this is
    the trust invariant Qubot v2 audits.

    Failure contract (item 17 — never silently swallow):
    - the core ``agent_actions`` INSERT raises: callers know persistence
      failed and must NOT report an audited operation;
    - auxiliary writes (Merkle leaf, claims, pins) degrade WITHOUT breaking
      the contact, but each degradation is structured-logged, counted in the
      ``audit_degraded`` metric / ``ledger_health()``, and reported to the
      caller via ``on_degraded(reasons)`` (or
      :func:`record_action_with_status`).

    Insert, Merkle leaf, pins, and optional claim rows share one ops connection
    so concurrent enrichment agents are not serialized on extra DuckDB locks.

    Safety exception (audit X.5 — explicit integrity-over-life decision):
    when the DB itself is down, SAFETY-class actions (escalation scripts,
    supervised safety hits, handoff offers) are appended to a local durable
    WAL (fsync) instead of raising, the contact is marked degraded_ledger,
    and the caller proceeds to deliver the safety output. Everything else
    still raises (fail-closed).     ``replay_ledger_wal()`` re-inserts WAL rows
    into the chain on recovery.
    """
    pack_id = _PACK_BY_INTERACTION.get(action.interaction_id)
    degraded: list[str] = []
    snaps: dict[str, dict[str, Any]] = {}
    if action.evidence_ids and pack_id:
        try:
            from src.qubot.evidence_pin import fetch_rows_by_type

            snaps = fetch_rows_by_type(pack_id, [str(e) for e in action.evidence_ids])
        except Exception as e:
            degraded.append(f"evidence_prefetch:{type(e).__name__}")
            _note_degraded(
                "evidence_prefetch",
                action_id=action.action_id,
                interaction_id=action.interaction_id,
                error=f"{type(e).__name__}: {e}",
            )
            snaps = {}
    try:
        return _record_action_inner(action, pack_id, degraded, snaps, on_degraded)
    except Exception as e:
        if action.action_type not in SAFETY_ACTION_TYPES:
            raise
        _wal_append(action, e)
        _mark_degraded_ledger(action.interaction_id)
        try:
            from src.observability.degradation import step_down

            step_down("merkle_anchor", reason="wal_fallback")
        except Exception:
            pass
        _note_degraded(
            "wal_fallback",
            action_id=action.action_id,
            interaction_id=action.interaction_id,
            error=f"{type(e).__name__}: {e}",
        )
        return action.action_id


# ── Safety WAL fallback (audit X.5) ─────────────────────────────────────────

#: Action classes whose delivery must survive a ledger DB outage. Everything
#: else stays fail-closed.
SAFETY_ACTION_TYPES = frozenset({
    "escalation_script_emitted",
    "supervised_safety_raised",
    "handoff_offer_emitted",
})


def _wal_path() -> Any:
    from pathlib import Path as _P

    from src.config import settings as _settings

    try:
        base = _settings.frontline_db_path.parent
    except Exception:
        base = _P(".")
    return base / "ledger_wal.jsonl"


def _wal_append(action: AgentAction, cause: BaseException) -> None:
    """Append one action to the durable WAL with fsync. Raises only on disk failure."""
    import os as _os

    from src.data.timeutil import utc_now as _now

    line = json.dumps(
        {
            "wal_ts": _now().isoformat() + "Z",
            "first_error": f"{type(cause).__name__}: {cause}",
            "action": {
                "action_id": action.action_id,
                "interaction_id": action.interaction_id,
                "case_id": action.case_id,
                "agent": action.agent,
                "action_type": action.action_type,
                "input_summary": action.input_summary,
                "output_summary": action.output_summary,
                "evidence_ids": list(action.evidence_ids or []),
                "ok": action.ok,
                "error": action.error,
                "duration_ms": action.duration_ms,
                "ts": str(action.ts),
            },
        },
        default=str,
    )
    path = _wal_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        _os.fsync(fh.fileno())


def _mark_degraded_ledger(interaction_id: str) -> None:
    try:
        with ops_con() as con:
            try:
                con.execute(
                    "UPDATE interactions SET degraded_ledger = TRUE WHERE interaction_id = ?",
                    [interaction_id],
                )
            except Exception:
                pass  # DB is down — that is why we are here
    except Exception:
        pass


def replay_ledger_wal(*, limit: int = 1000) -> dict[str, Any]:
    """Re-insert WAL rows into the chain after recovery.

    Returns {replayed, failed, remaining}. Replay goes through a persistence-only
    path that raises on failure and confirms commit (never falling back to appending
    to the same WAL). On full success the WAL file is rotated aside.
    """
    path = _wal_path()
    try:
        raw_text = path.read_text(encoding="utf-8")
        lines = raw_text.splitlines()
    except FileNotFoundError:
        return {"replayed": 0, "failed": 0, "remaining": 0}
    except OSError as e:
        return {"replayed": 0, "failed": 1, "remaining": -1, "error": str(e)}

    replayed, failed = 0, []
    kept: list[str] = []
    processed_iids: set[str] = set()
    succeeded_lines: set[str] = set()

    for line in lines[:limit]:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            a = entry.get("action") or {}
            act = AgentAction(
                interaction_id=a.get("interaction_id", "unknown"),
                agent=a.get("agent", "orchestrator"),
                action_type=a.get("action_type", "state_transition"),
                input_summary=a.get("input_summary") or "",
                output_summary=((a.get("output_summary") or "") + " [replayed-from-wal]"),
                evidence_ids=list(a.get("evidence_ids") or []),
                case_id=a.get("case_id"),
                ok=bool(a.get("ok", True)),
                error=a.get("error"),
                duration_ms=a.get("duration_ms"),
            )
            act.action_id = a.get("action_id") or act.action_id
            if "ts" in a and a["ts"]:
                try:
                    from datetime import datetime

                    act.ts = datetime.fromisoformat(str(a["ts"]).replace("Z", "+00:00"))
                except Exception:
                    pass

            # Persistence-only path: check idempotency then insert directly via _record_action_inner.
            # Must raise on failure rather than catching and re-appending to WAL.
            already_persisted = False
            try:
                with ops_con(read_only=True) as con:
                    existing = con.execute(
                        "SELECT 1 FROM agent_actions WHERE action_id = ?",
                        [act.action_id],
                    ).fetchone()
                    if existing:
                        already_persisted = True
            except Exception:
                pass

            if not already_persisted:
                _record_action_inner(act, None, [], {}, None)

            replayed += 1
            succeeded_lines.add(line)
            if act.interaction_id:
                processed_iids.add(act.interaction_id)
        except Exception as e:
            failed.append(f"{type(e).__name__}: {e}")
            kept.append(line)

    # Safely handle concurrent appends during replay
    try:
        current_lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        rest = []
        for ln in current_lines:
            if ln in succeeded_lines:
                continue
            rest.append(ln)
    except Exception:
        rest = kept + [ln for ln in lines[limit:] if ln.strip()]

    try:
        if rest:
            path.write_text("\n".join(rest) + "\n", encoding="utf-8")
        else:
            rotated = path.with_name(path.name + f".replayed-{int(time.time())}")
            path.rename(rotated)
    except OSError:
        pass

    if replayed and not kept:
        # Clear degraded_ledger ONLY for interactions that have NO remaining entries in rest
        remaining_iids = set()
        for ln in rest:
            try:
                iid = (json.loads(ln).get("action") or {}).get("interaction_id")
                if iid:
                    remaining_iids.add(iid)
            except Exception:
                pass
        clear_iids = {iid for iid in processed_iids if iid not in remaining_iids}
        if clear_iids:
            try:
                with ops_con() as con:
                    for iid in clear_iids:
                        con.execute(
                            "UPDATE interactions SET degraded_ledger = FALSE "
                            "WHERE interaction_id = ? AND degraded_ledger = TRUE",
                            [iid],
                        )
            except Exception:
                pass

    return {
        "replayed": replayed,
        "failed": len(failed),
        "remaining": len(rest),
        "errors": failed[:5],
    }


def _record_action_inner(action, pack_id, degraded, snaps, on_degraded) -> str:
    with ops_con() as con:
        row_hash = _insert_action_row(con, action)
        try:
            from src.ledger.merkle import append_action_leaf_on_con

            append_action_leaf_on_con(
                con, action.interaction_id, action.action_id, str(row_hash)
            )
        except Exception as e:
            degraded.append(f"merkle_leaf:{type(e).__name__}")
            _note_degraded(
                "merkle_leaf",
                action_id=action.action_id,
                interaction_id=action.interaction_id,
                error=f"{type(e).__name__}: {e}",
            )
        if getattr(action, "claims", None):
            try:
                from src.qubot.claims import store_bound_claims_on_con

                store_bound_claims_on_con(con, action.action_id, action.claims)
            except Exception as e:
                degraded.append(f"claims:{type(e).__name__}")
                _note_degraded(
                    "claims",
                    action_id=action.action_id,
                    interaction_id=action.interaction_id,
                    error=f"{type(e).__name__}: {e}",
                )
        if action.evidence_ids and not pack_id:
            row = con.execute(
                "SELECT pack_id FROM interactions WHERE interaction_id = ?",
                [action.interaction_id],
            ).fetchone()
            if row and row[0]:
                pack_id = str(row[0])
                remember_interaction_pack(action.interaction_id, pack_id)
        if snaps and pack_id:
            try:
                from src.qubot.evidence_pin import insert_pins_on_con

                insert_pins_on_con(
                    con,
                    action_id=action.action_id,
                    interaction_id=action.interaction_id,
                    pack_id=str(pack_id),
                    rows=snaps,
                )
            except Exception as e:
                degraded.append(f"evidence_pin:{type(e).__name__}")
                _note_degraded(
                    "evidence_pin",
                    action_id=action.action_id,
                    interaction_id=action.interaction_id,
                    error=f"{type(e).__name__}: {e}",
                )
    if action.evidence_ids and pack_id and not snaps:
        _pin_cited_best_effort(action, pack_id=str(pack_id))
    if degraded and on_degraded is not None:
        try:
            on_degraded(list(degraded))
        except Exception:
            pass
    return action.action_id


def record_action_with_status(action: AgentAction) -> tuple[str, dict[str, Any]]:
    """Like :func:`record_action` but also returns the persistence status.

    Returns ``(action_id, {"ok": bool, "degraded": [...]})``. ``ok`` is False
    only when the core INSERT raised (in which case the exception propagates
    AND ``ok`` is False — callers must treat the operation as unaudited).
    ``degraded`` lists auxiliary writes that failed but were contained.
    """
    degraded: list[str] = []
    try:
        action_id = record_action(action, on_degraded=degraded.extend)
    except Exception:
        return action.action_id, {"ok": False, "degraded": ["core_insert_failed"]}
    return action_id, {"ok": True, "degraded": degraded}


def _pin_cited_best_effort(action: AgentAction, *, pack_id: str | None = None) -> None:
    if not action.evidence_ids:
        return
    try:
        if not pack_id:
            with ops_con(read_only=True) as con:
                row = con.execute(
                    "SELECT pack_id FROM interactions WHERE interaction_id = ?",
                    [action.interaction_id],
                ).fetchone()
                if row:
                    pack_id = row[0]
        if not pack_id:
            return
        from src.qubot.evidence_pin import pin_cited_records

        pin_cited_records(
            action_id=action.action_id,
            interaction_id=action.interaction_id,
            pack_id=str(pack_id),
            evidence_ids=list(action.evidence_ids),
        )
    except Exception as e:
        logger.warning("best-effort pin failed for %s: %s", action.action_id, e)
        return


def record_action_on_con(con, action: AgentAction) -> str:
    """Write an action on a caller-owned ops connection (same transaction)."""
    _insert_action_row(con, action)
    return action.action_id


class action_timer:
    """Context manager that times a block and records the action.

    Usage:
        with action_timer(interaction_id, "sentinel", "advisory_check",
                          input_summary="...", evidence_ids=[...]) as act:
            ... do work ...
            act.output_summary = "Matched advisory 19V-XXX"
            act.evidence_ids = ["19V-XXX"]
            # on exit, the action is written to the ledger.

    If the block raises, ok=False, error=str(exc) is recorded (and the
    exception re-raised).
    """

    def __init__(
        self,
        interaction_id: str,
        agent: str,
        action_type: str,
        input_summary: str = "",
        case_id: str | None = None,
    ) -> None:
        self._action = AgentAction(
            interaction_id=interaction_id,
            agent=agent,
            action_type=action_type,
            case_id=case_id,
            input_summary=input_summary,
        )
        self._start = 0.0

    def __enter__(self) -> "action_timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._action.duration_ms = int((time.perf_counter() - self._start) * 1000)
        if exc is not None:
            self._action.ok = False
            self._action.error = f"{type(exc).__name__}: {exc}"
        record_action(self._action)
        return False  # don't suppress

    @property
    def action(self) -> AgentAction:
        return self._action

    @property
    def output_summary(self) -> str:
        return self._action.output_summary

    @output_summary.setter
    def output_summary(self, v: str) -> None:
        self._action.output_summary = v

    @property
    def evidence_ids(self) -> list[str]:
        return self._action.evidence_ids

    @evidence_ids.setter
    def evidence_ids(self, v: list[str]) -> None:
        self._action.evidence_ids = v


# ── Reader ──────────────────────────────────────────────────────────────────


def list_actions(interaction_id: str) -> list[dict[str, Any]]:
    """Return all actions for an interaction, ordered by ts."""
    with ops_con(read_only=True) as con:
        cur = con.execute(
            "SELECT * FROM agent_actions WHERE interaction_id = ? ORDER BY ts, action_id",
            [interaction_id],
        )
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall()
    out = []
    for r in rows:
        d = dict(zip(cols, r))
        # parse JSON columns
        if d.get("evidence_ids"):
            try:
                d["evidence_ids"] = json.loads(d["evidence_ids"])
            except (json.JSONDecodeError, TypeError):
                d["evidence_ids"] = []
        else:
            d["evidence_ids"] = []
        if d.get("claims"):
            try:
                if isinstance(d["claims"], str):
                    d["claims"] = json.loads(d["claims"])
            except (json.JSONDecodeError, TypeError):
                pass
        out.append(d)
    return out


__all__ = [
    "AgentAction",
    "action_timer",
    "record_action",
    "record_action_with_status",
    "ledger_health",
    "list_actions",
    "ACTION_TYPES",
    "SAFETY_ACTION_TYPES",
    "replay_ledger_wal",
]
