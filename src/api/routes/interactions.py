"""REST + WS routes for interactions.

Endpoints:
  POST   /api/interactions/start              → create + open WS
  GET    /api/interactions                    → list (filter by status)
  GET    /api/interactions/{id}               → header + turns + actions
  POST   /api/interactions/{id}/end           → end the interaction
  POST   /api/interactions/{id}/takeover     → supervisor takes over
  POST   /api/interactions/{id}/release      → supervisor releases
  WS     /ws/interaction/{id}                 → bidirectional contact stream
  WS     /ws/console                          → live console feed
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect

from src.agents.orchestrator import (
    Orchestrator,
    OrchestratorHooks,
    create_interaction,
)
from src.api.auth import authenticate_websocket, require_api_key
from src.api.frontline_gate import frontline_enabled
from src.api.limiter import limiter
from src.api.rbac import (
    actor_from_websocket,
    get_actor,
    get_role,
    has_perm,
    require_perm,
    require_perm_dep,
    role_from_websocket,
)
from src.channels.web_voice import WebVoiceChannel
from src.data.warehouse import ops_con, ops_in_thread

router = APIRouter(
    prefix="/api/interactions",
    tags=["interactions"],
    # PII surface: list/detail require key when FRONTLINE_API_KEY is set.
    dependencies=[Depends(require_api_key)],
)
# WebSocket routes live at the root path (/ws/interaction/{id} and /ws/console)
# and are registered directly on the app in src/api/main.py — there is no
# separate WS APIRouter (a previous ws_router here was never populated/mounted).


# ── In-memory registry of active orchestrators (one per interaction) ─────────
# Exclusive customer WS attach; per-entry lock for turn/takeover races;
# orphan TTL reaps start-without-WS entries; a reconnect grace window keeps a
# dropped-but-not-hung-up contact resumable so the browser can re-attach.

_ORPHAN_TTL_S = 300.0  # 5 minutes without customer WS attach


def _reconnect_grace_s() -> float:
    """Seconds a dropped customer WS may stay resumable before the contact is closed.

    The dashboard retries with backoff for ~15s across 5 attempts, so the default
    leaves plenty of head-room for a tab throttle / wifi blip. Set to 0 to restore
    the old behaviour (drop == hangup).
    """
    import os

    raw = os.getenv("FRONTLINE_WS_RECONNECT_GRACE_S", "").strip()
    if not raw:
        return 120.0
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 120.0


@dataclass
class ActiveEntry:
    orch: Orchestrator
    created_at: float = field(default_factory=time.monotonic)
    ws_attached: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # monotonic time the customer WS dropped without a hangup (None = never attached
    # or currently attached). Entries past the reconnect grace are reaped.
    detached_at: float | None = None
    # Tier-A: durable turn sequencing survives reconnect + restart via turn_dedup.
    turn_seq: int = 0
    owner_subject: str = ""
    capability_token: str = ""


_active: dict[str, ActiveEntry] = {}
_active_lock = asyncio.Lock()


def _json_safe(value: Any) -> Any:
    """Make DuckDB / Python values JSON-serializable for WebSocket frames (UTC Z)."""
    from src.api.jsonutil import json_safe as _js

    return _js(value)


def normalize_activity_payload(
    payload: dict[str, Any], *, interaction_id: str | None = None
) -> dict[str, Any]:
    """One wire shape for live broadcast and DB-poll ``agent_activity`` frames."""
    summary = (
        payload.get("output_summary")
        or payload.get("summary")
        or payload.get("input_summary")
        or ""
    )
    iid = interaction_id or payload.get("interaction_id") or ""
    frame = {
        "type": "agent_activity",
        "interaction_id": iid,
        "action_id": payload.get("action_id") or "",
        "agent": payload.get("agent"),
        "action_type": payload.get("action_type"),
        "summary": summary,
        "output_summary": summary,
        "input_summary": payload.get("input_summary") or "",
        "evidence_ids": payload.get("evidence_ids") or [],
        "ok": payload.get("ok", True),
        "ts": payload.get("ts"),
        "duration_ms": payload.get("duration_ms"),
    }
    return {k: _json_safe(v) for k, v in frame.items() if v is not None}


def activity_frame_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """Build a console ``agent_activity`` payload safe for ``send_json``."""
    r = dict(row)
    ev = r.get("evidence_ids")
    if isinstance(ev, str):
        try:
            r["evidence_ids"] = json.loads(ev)
        except json.JSONDecodeError:
            r["evidence_ids"] = []
    return normalize_activity_payload(r, interaction_id=str(r.get("interaction_id") or ""))


def fetch_agent_actions_since(
    since: datetime, limit: int = 50, after_action_id: str = ""
) -> list[dict[str, Any]]:
    """Load agent_actions newer than ``(since, after_action_id)`` (console catch-up)."""
    from src.data.timeutil import to_naive_utc

    # Compare on naive-UTC basis so session TZ cannot drop recent rows.
    since_n = to_naive_utc(since)
    after_id = after_action_id or ""
    with ops_con(read_only=True) as con:
        cur = con.execute(
            """
            SELECT action_id, interaction_id, agent, action_type,
                   output_summary, evidence_ids, ts
            FROM agent_actions
            WHERE ts > ? OR (ts = ? AND action_id > ?)
            ORDER BY ts, action_id
            LIMIT ?
            """,
            [since_n, since_n, after_id, limit],
        )
        cols = [d[0] for d in con.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def _sweep_stale_active_db_rows(registry_ids: set[str] | None = None) -> list[str]:
    """Mark ops DB rows still ``status='active'`` but not in the live registry.

    Covers zombie contacts left after hangup-while-supervised failures or
    process restarts where the in-memory registry is empty.
    """
    live = registry_ids if registry_ids is not None else set(_active.keys())
    closed: list[str] = []
    try:
        with ops_con() as con:
            rows = con.execute(
                "SELECT interaction_id FROM interactions WHERE status = 'active'"
            ).fetchall()
            from src.data.timeutil import utc_now

            now = utc_now()
            for (iid,) in rows:
                if iid in live:
                    continue
                con.execute(
                    """
                    UPDATE interactions
                    SET status = 'failed',
                        ended_at = COALESCE(ended_at, ?),
                        outcome = COALESCE(outcome, 'orphan_db_sweep')
                    WHERE interaction_id = ? AND status = 'active'
                    """,
                    [now, iid],
                )
                closed.append(iid)
    except Exception:
        pass
    return closed


def _is_reapable(entry: ActiveEntry, now: float, grace_s: float) -> bool:
    """True when an unattached entry has outlived its window.

    Two distinct windows:
      * never attached a customer WS   → ``_ORPHAN_TTL_S`` from creation
      * WS dropped without a hangup    → ``grace_s`` from the drop (reconnect window)
    """
    if entry.ws_attached:
        return False
    if entry.detached_at is not None:
        return (now - entry.detached_at) > grace_s
    return (now - entry.created_at) > _ORPHAN_TTL_S


def classify_customer_ws_attach(
    entry: ActiveEntry | None,
    *,
    now: float,
    grace_s: float,
) -> str:
    """Classify a customer-WS attach without I/O or the registry lock.

    Returns one of:
      ``missing`` / ``busy`` / ``ended`` / ``not_resumable`` /
      ``ok_new`` / ``ok_resume``.
    """
    if entry is None:
        return "missing"
    if entry.ws_attached:
        return "busy"
    state = getattr(getattr(entry, "orch", None), "ctx", None)
    st = getattr(state, "state", None)
    if st in ("DONE", "ABANDONED"):
        return "ended"
    if _is_reapable(entry, now, grace_s):
        return "not_resumable"
    if entry.detached_at is not None:
        return "ok_resume"
    return "ok_new"


def _pop_reapable_unlocked() -> list[tuple[str, ActiveEntry, str]]:
    """Pop TTL/grace-expired entries. Caller holds ``_active_lock``. No hangup."""
    now = time.monotonic()
    grace_s = _reconnect_grace_s()
    dead: list[tuple[str, ActiveEntry, str]] = []
    for iid, entry in list(_active.items()):
        if not _is_reapable(entry, now, grace_s):
            continue
        popped = _active.pop(iid, None)
        if popped is None:
            continue
        reason = (
            "reconnect_timeout" if popped.detached_at is not None else "orphan_timeout"
        )
        dead.append((iid, popped, reason))
    return dead


async def _finalize_reaped(dead: list[tuple[str, ActiveEntry, str]]) -> None:
    """Hangup + DB mark **off** ``_active_lock`` so WS attach cannot stall."""
    for iid, entry, reason in dead:
        try:
            if entry.orch.ctx.state not in ("DONE", "ABANDONED"):
                await entry.orch.hangup()
        except Exception:
            pass
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(ops_in_thread(_mark_interaction_failed_sync, iid, reason))
        except Exception:
            pass


async def _reap_orphans_unlocked(*, sweep_db: bool = False) -> list[str]:
    """Drop registry entries past their orphan TTL / reconnect grace.

    Caller must **not** hold ``_active_lock`` across this function: hangup and
    DuckDB run here and used to stall ``/ws/interaction/{id}`` accept (~30s).
    """
    async with _active_lock:
        dead = _pop_reapable_unlocked()
    await _finalize_reaped(dead)
    pending_handoffs: list[ActiveEntry] = []
    async with _active_lock:
        pending_handoffs = [
            e
            for e in list(_active.values())
            if getattr(getattr(e.orch, "ctx", None), "state", None) == "HANDOFF_PENDING"
        ]
    for entry in pending_handoffs:
        try:
            # Same per-contact lock as takeover. Do not hold the registry lock here.
            async with entry.lock:
                if getattr(entry.orch.ctx, "state", None) == "HANDOFF_PENDING":
                    await entry.orch.sweep_handoff_timeout()
        except Exception:
            pass
    ids = [iid for iid, _, _ in dead]
    if sweep_db:
        # Recapture AFTER hangup: a POST /start during _finalize_reaped is
        # already in `_active` and must not be swept as a zombie.
        async with _active_lock:
            live = set(_active.keys())
        try:
            extra = await ops_in_thread(_sweep_stale_active_db_rows, live)
            ids.extend(extra)
        except Exception:
            pass
    return ids


async def reap_orphans() -> list[str]:
    """Public reaper (startup / tests): registry TTL + DB active-row sweep."""
    return await _reap_orphans_unlocked(sweep_db=True)


async def reaper_loop(interval_s: float = 30.0) -> None:
    """Background sweep so a contact whose client never returns still finalizes.

    Without this the reaper only runs when something else touches the registry,
    so a dropped call with no follow-up traffic would sit ``status='active'``
    until the next start/end request.
    """
    while True:
        try:
            await asyncio.sleep(interval_s)
            await reap_orphans()
        except asyncio.CancelledError:
            raise
        except Exception:
            continue


async def _get_entry(interaction_id: str) -> ActiveEntry:
    async with _active_lock:
        entry = _active.get(interaction_id)
    if not entry:
        raise HTTPException(
            status_code=404, detail=f"active interaction not found: {interaction_id}"
        )
    return entry


async def _register(
    interaction_id: str,
    orch: Orchestrator,
    *,
    owner_subject: str = "",
    capability_token: str = "",
) -> None:
    async with _active_lock:
        # Register only. Orphan hangup/DB sweep is the reaper's job — never
        # hold ``_active_lock`` across hangup (that stalled WS attach ~30s).
        _active[interaction_id] = ActiveEntry(
            orch=orch,
            owner_subject=owner_subject,
            capability_token=capability_token,
        )


async def _unregister(interaction_id: str) -> None:
    async with _active_lock:
        _active.pop(interaction_id, None)


async def _attach_customer_ws(interaction_id: str) -> tuple[ActiveEntry, bool]:
    """Exclusive attach: second *concurrent* customer WS is rejected (409).

    A socket that dropped without a hangup leaves the entry in the registry with
    ``ws_attached=False``; re-attaching within the reconnect grace resumes the
    same orchestrator instead of 404-ing the caller.

    Returns ``(entry, resumed)`` where ``resumed`` is True for a re-attach.

    Hangup of *other* expired contacts is not awaited here — that used to
    block accept until DuckDB I/O finished (~30s under lock contention).
    """
    pending_dead: list[tuple[str, ActiveEntry, str]] = []
    error: HTTPException | None = None
    result: tuple[ActiveEntry, bool] | None = None

    from src.frontline.dsr import is_interaction_erased

    if is_interaction_erased(interaction_id):
        async with _active_lock:
            popped = _active.pop(interaction_id, None)
            if popped is not None and getattr(popped, "orch", None) is not None:
                popped.orch._erased = True
                if getattr(popped.orch, "ctx", None) is not None:
                    popped.orch.ctx.slots.clear()
                    if hasattr(popped.orch.ctx, "turns") and popped.orch.ctx.turns is not None:
                        popped.orch.ctx.turns.clear()
        raise HTTPException(
            status_code=404, detail=f"active interaction not found: {interaction_id}"
        )

    async with _active_lock:
        entry = _active.get(interaction_id)
        if entry is not None and getattr(entry.orch, "_erased", False):
            _active.pop(interaction_id, None)
            entry = None
        now = time.monotonic()
        grace_s = _reconnect_grace_s()
        decision = classify_customer_ws_attach(entry, now=now, grace_s=grace_s)
        if decision == "missing":
            error = HTTPException(
                status_code=404, detail=f"active interaction not found: {interaction_id}"
            )
        elif decision == "busy":
            error = HTTPException(
                status_code=409,
                detail=f"interaction already has an active customer WebSocket: {interaction_id}",
            )
        elif decision == "ended":
            _active.pop(interaction_id, None)
            error = HTTPException(
                status_code=404, detail=f"interaction already ended: {interaction_id}"
            )
        elif decision == "not_resumable":
            popped = _active.pop(interaction_id, None)
            if popped is not None:
                reason = (
                    "reconnect_timeout"
                    if popped.detached_at is not None
                    else "orphan_timeout"
                )
                pending_dead.append((interaction_id, popped, reason))
            error = HTTPException(
                status_code=404, detail=f"active interaction not found: {interaction_id}"
            )
        else:
            assert entry is not None
            resumed = decision == "ok_resume"
            entry.ws_attached = True
            entry.detached_at = None
            result = (entry, resumed)
    if pending_dead:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_finalize_reaped(pending_dead))
        except Exception:
            pass
    if error is not None:
        raise error
    assert result is not None
    return result


async def _detach_customer_ws(interaction_id: str) -> None:
    """Release the exclusive attach but keep the contact resumable.

    Used when the socket drops without an explicit hangup (wifi blip, tab
    throttle, proxy idle timeout). The reconnect grace window in the reaper
    finalizes the contact if the browser never comes back.
    """
    async with _active_lock:
        entry = _active.get(interaction_id)
        if entry is None:
            return
        entry.ws_attached = False
        entry.detached_at = time.monotonic()


def _mark_interaction_failed_sync(interaction_id: str, reason: str) -> None:
    from src.data.timeutil import utc_now

    with ops_con() as con:
        con.execute(
            """
            UPDATE interactions
            SET status = 'failed',
                ended_at = COALESCE(ended_at, ?),
                outcome = COALESCE(outcome, ?)
            WHERE interaction_id = ? AND status = 'active'
            """,
            [utc_now(), reason[:200], interaction_id],
        )


def _mark_failed_sync(interaction_id: str, reason: str) -> None:
    from src.data.timeutil import utc_now

    with ops_con() as con:
        con.execute(
            """
            UPDATE interactions
            SET status = 'failed',
                ended_at = COALESCE(ended_at, ?),
                outcome = ?
            WHERE interaction_id = ?
              AND status IN ('active', 'abandoned')
            """,
            [utc_now(), reason[:200], interaction_id],
        )


async def _mark_failed(interaction_id: str, reason: str) -> None:
    try:
        await ops_in_thread(_mark_failed_sync, interaction_id, reason)
    except Exception:
        pass


# ── REST ─────────────────────────────────────────────────────────────────────


@router.post("/start")
@limiter.limit("30 per minute")
async def start_interaction(
    request: Request,
    channel: str = "web_voice",
    customer_ref: str | None = None,
    channel_session_id: str | None = None,
    region: str | None = None,
    locale: str | None = None,
    _auth: bool = Depends(require_api_key),
    _role: str = Depends(require_perm_dep("contact:write")),
    actor: str = Depends(get_actor),
) -> dict[str, Any]:
    """Create an interaction + return the WS URL + greeting.

    Idempotent (audit 1.1): pass ``Idempotency-Key`` (or a stable
    ``channel_session_id`` from the widget/telephony webhook) and retries
    return the existing interaction instead of creating a second greeting.
    The dedup is ledgered as ``start_deduped`` on the original interaction.
    ``customer_ref`` (phone/email/account, hashed at rest) links returning
    customers for cross-contact case attach (audit 1.1).
    """
    import hashlib as _hashlib

    from src.api.idempotency import check_idempotency, store_idempotent_result
    from src.domains.active_pack import resolve_active_pack_id
    from src.ops.drain import ServiceDrainingError

    pack_id = resolve_active_pack_id()
    raw_ref = (customer_ref or "").strip()
    ref_hash = _hashlib.sha256(raw_ref.encode()).hexdigest() if raw_ref else None
    material = f"channel={channel}&pack={pack_id}&session={channel_session_id or ''}&ref={ref_hash or ''}"
    replay = check_idempotency(request, "interaction_start", material)
    if replay is not None:
        try:
            from src.ledger import AgentAction, record_action

            record_action(AgentAction(
                interaction_id=replay.get("interaction_id", "unknown"),
                agent="orchestrator",
                action_type="start_deduped",
                input_summary="duplicate start deduplicated",
                output_summary=f"replayed {replay.get('interaction_id')}",
            ))
        except Exception:
            pass
        return {**replay, "deduped": True}

    # ── Phase 1 Live Pilot Routing Gate & Circuit Breaker ────────────────────
    import os
    from src.routing import get_circuit_breaker, get_traffic_gate

    phase1_active = (
        request.headers.get("X-Phase1-Live-Pilot") == "1"
        or request.query_params.get("phase1") == "1"
        or os.getenv("PHASE1_LIVE_PILOT_ENABLED") == "1"
    )

    if phase1_active:
        cb = get_circuit_breaker()
        if cb.is_tripped:
            return {
                "routed_to": "human_control",
                "admitted": False,
                "reason": f"circuit_breaker_tripped: {cb.trip_reason}",
                "phase1_admitted": False,
                "interaction_id": None,
            }

        from src.ids import new_ulid

        tg = get_traffic_gate()
        raw_hash = request.headers.get("X-Call-Hash")
        allow_hash_override = (
            os.getenv("FRONTLINE_ALLOW_CALL_HASH_OVERRIDE", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if allow_hash_override and raw_hash is not None and raw_hash.strip().isdigit():
            admitted, reason = tg.evaluate_ingress(
                call_hash=int(raw_hash.strip()),
                domain=pack_id,
                circuit_breaker=cb,
            )
        else:
            seed_source = (
                request.headers.get("X-Call-Sid")
                or channel_session_id
                or customer_ref
                or f"int_{new_ulid()}"
            )
            admitted, reason = tg.evaluate_ingress(
                seed_source=seed_source,
                domain=pack_id,
                circuit_breaker=cb,
            )

        if not admitted:
            return {
                "routed_to": "human_control",
                "admitted": False,
                "reason": reason,
                "phase1_admitted": False,
                "daily_routed_count": tg.daily_routed_count,
                "interaction_id": None,
            }

    try:
        orch, greeting = await create_interaction(
            channel=channel, pack_id=pack_id, customer_ref=ref_hash,
            region=(region or request.query_params.get("region")),
            locale=(locale or request.query_params.get("locale")),
        )
    except ServiceDrainingError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e) or "service_draining: not accepting new contacts",
        ) from e
    cap_token = secrets.token_urlsafe(32)
    await _register(
        orch.ctx.interaction_id,
        orch,
        owner_subject=actor,
        capability_token=cap_token,
    )
    pack = orch.ctx.pack
    try:
        _server_stt = bool(
            (os.getenv("STT_API_URL") or "").strip()
            or (os.getenv("FASTER_WHISPER_MODEL") or "").strip()
        )
    except Exception:
        _server_stt = False
    try:
        from src.voice.policy import consent_requirement as _cr0

        _creq0 = _cr0(orch.ctx.region or "") if orch.ctx.region else {}
    except Exception:
        _creq0 = {}
    out = {
        "interaction_id": orch.ctx.interaction_id,
        "ws_url": f"/ws/interaction/{orch.ctx.interaction_id}",
        "twilio_ws_url": f"/ws/twilio/{orch.ctx.interaction_id}",
        "capability_token": cap_token,
        "greeting_text": greeting,
        "pack": {
            "id": pack.id,
            "display_name": pack.display_name,
            "entity_labels": pack.entity_labels,
            "pack_version": pack.pack_version,
            "locale": getattr(pack, "locale", "en-US") or "en-US",
        },
        "locale": getattr(orch.ctx, "locale", "en-US") or "en-US",
        "region": getattr(orch.ctx, "region", None),
        "voice_consent_required": bool(getattr(orch.ctx, "voice_consent_required", False)),
        "consent_script": getattr(orch.ctx, "voice_consent_script", "") or str((_creq0 or {}).get("script") or ""),
        "server_stt_available": _server_stt,
    }
    if phase1_active:
        out["phase1_admitted"] = True
        out["routed_to"] = "live_voice_agent"
        out["daily_routed_count"] = tg.daily_routed_count

    store_idempotent_result(request, "interaction_start", out)
    return out


@router.post("/ingest")
@limiter.limit("30 per minute")
async def ingest_interaction(
    request: Request,
    body: dict[str, Any] | None = None,
    _auth: bool = Depends(require_api_key),
    _role: str = Depends(require_perm_dep("contact:write")),
) -> dict[str, Any]:
    """Non-conversational complaint ingest (ticket/webhook path).

    Idempotent (item 45) via ``Idempotency-Key``: retries return the stored
    result instead of ingesting twice.
    """
    from src.frontline.ingest import ingest_complaint

    payload = body if isinstance(body, dict) else {}
    if not payload:
        try:
            payload = await request.json()
        except Exception:
            payload = {}
    import json as _json

    from src.api.idempotency import check_idempotency, store_idempotent_result

    material = _json.dumps(payload, sort_keys=True, default=str)
    replay = check_idempotency(request, "ingest", material)
    if replay is not None:
        return replay
    try:
        result = await ingest_complaint(payload)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid ingest payload") from None
    except Exception:
        raise HTTPException(status_code=400, detail="ingest failed") from None
    if isinstance(result, dict):
        store_idempotent_result(request, "ingest", result)
        return result
    return {"result": result}


@router.get("")
async def list_interactions(
    status: str | None = None,
    limit: int = 50,
    offset: int = 0,
    cursor: str | None = None,
    scrub_pii: bool = True,
    _role: str = Depends(get_role),
) -> dict[str, Any]:
    """List interactions (optionally filter by status). Paginated.

    Free-text columns are PII-redacted by default. ``scrub_pii=false``
    requires ``dsr:export``.
    """
    from src.api.jsonutil import json_safe_rows
    from src.api.pagination import clamp_limit, page_meta, resolve_offset
    from src.security.pii import redact_dict

    lim = clamp_limit(limit)
    off = resolve_offset(offset=offset, cursor=cursor)
    sql = "SELECT * FROM interactions"
    params: list[Any] = []
    if status:
        sql += " WHERE status = ?"
        params.append(status)
    sql += " ORDER BY started_at DESC LIMIT ? OFFSET ?"
    params.extend([lim + 1, off])

    def _load() -> list[dict[str, Any]]:
        with ops_con(read_only=True) as con:
            cur = con.execute(sql, params)
            cols = [d[0] for d in con.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    rows = await ops_in_thread(_load)
    has_extra = len(rows) > lim
    page = rows[:lim]
    async with _active_lock:
        live = {iid: entry for iid, entry in _active.items()}
    for row in page:
        entry = live.get(row.get("interaction_id"))
        if entry is None:
            continue
        ctx = entry.orch.ctx
        row["supervised"] = bool(ctx.supervised)
        row["control_generation"] = int(getattr(ctx, "control_generation", 0) or 0)
        row["takeover_claimed_by"] = ctx.takeover_claimed_by
    if scrub_pii:
        page = [redact_dict(row) for row in page]
    else:
        require_perm(_role, "dsr:export")
    total = None if has_extra else off + len(page)
    return {
        "interactions": json_safe_rows(page),
        "count": len(page),
        "pagination": page_meta(limit=lim, offset=off, returned=len(page), total=total),
    }


@router.get("/{interaction_id}")
@limiter.limit("120 per minute")
async def get_interaction(
    request: Request,
    interaction_id: str,
    scrub_pii: bool = True,
    _role: str = Depends(require_perm_dep("case:read")),
) -> dict[str, Any]:
    """Return the interaction header + turns + actions for replay.

    Free text is PII-redacted by default, including the header, the nested
    case, turns and actions. ``scrub_pii=false`` requires ``dsr:export``.
    """
    from src.qubot.retrievers import contact_audit
    from src.security.pii import redact_dict

    try:
        data = contact_audit(interaction_id)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404, detail=f"interaction not found: {interaction_id}"
        ) from e
    if not scrub_pii:
        require_perm(_role, "dsr:export")
        return data
    return redact_dict(data)


@router.post("/{interaction_id}/end")
async def end_interaction(
    interaction_id: str,
    _auth: bool = Depends(require_api_key),
    _role: str = Depends(require_perm_dep("contact:write")),
) -> dict[str, Any]:
    """End the interaction (customer hung up via REST instead of WS)."""
    entry = await _get_entry(interaction_id)
    async with entry.lock:
        await entry.orch.hangup()
    await _unregister(interaction_id)
    return {"interaction_id": interaction_id, "status": entry.orch.ctx.state}


@router.post("/{interaction_id}/takeover")
async def takeover(
    interaction_id: str,
    _auth: bool = Depends(require_api_key),
    _role: str = Depends(require_perm_dep("takeover")),
    actor: str = Depends(get_actor),
) -> dict[str, Any]:
    """Supervisor takes over the conversation (first claimant wins)."""
    entry = await _get_entry(interaction_id)
    async with entry.lock:
        already = entry.orch.ctx.state == "SUPERVISED"
        claimed_by = entry.orch.ctx.takeover_claimed_by
        await entry.orch.takeover(claimed_by=actor)
        lost = bool(already and claimed_by and claimed_by != actor)
    body = {
        "interaction_id": interaction_id,
        "state": entry.orch.ctx.state,
        "supervised": entry.orch.ctx.state == "SUPERVISED",
        "claimed_by": entry.orch.ctx.takeover_claimed_by,
        "already_claimed": lost,
        "generation": int(entry.orch.ctx.control_generation or 0),
    }
    if lost:
        raise HTTPException(status_code=409, detail=body)
    return body


@router.post("/{interaction_id}/handoff/accept")
async def handoff_accept(
    interaction_id: str,
    body: dict[str, Any] | None = None,
    _auth: bool = Depends(require_api_key),
) -> dict[str, Any]:
    """Queue a customer-requested handoff for supervisor pickup.

    ``force`` is reserved for the caller-facing "Get me a human" control. A
    caller may request a person before the automated flow has separately made
    an offer; treating that as an error made the UI claim a handoff existed
    while no supervisor queue row had been created.
    """
    entry = await _get_entry(interaction_id)
    payload = body if isinstance(body, dict) else {}
    async with entry.lock:
        out = await entry.orch.accept_handoff(
            force=bool(payload.get("force")),
            source="caller_control" if payload.get("force") else "handoff_offer",
        )
    return {"interaction_id": interaction_id, **out}


@router.post("/{interaction_id}/outcome")
async def record_outcome(
    interaction_id: str,
    body: dict[str, Any],
    _auth: bool = Depends(require_api_key),
) -> dict[str, Any]:
    """Post-call outcome signal (board #9): was it actually resolved? CSAT 1-5.

    Ground truth for fix proof on the contact side. Accepted from the widget
    after close; validated (csat 1-5, resolved bool), stored on the
    interaction row. Unknown interactions 404.
    """
    from src.data.warehouse import ops_con

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    csat = body.get("csat", None)
    if csat is not None:
        try:
            csat = int(csat)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="csat must be 1-5") from None
        if csat < 1 or csat > 5:
            raise HTTPException(status_code=400, detail="csat must be 1-5")
    resolved = body.get("resolved", None)
    if resolved is not None and not isinstance(resolved, bool):
        raise HTTPException(status_code=400, detail="resolved must be boolean")
    with ops_con() as con:
        try:
            row = con.execute(
                "SELECT 1 FROM interactions WHERE interaction_id = ?",
                [interaction_id],
            ).fetchone()
        except Exception:
            row = None
        if not row:
            raise HTTPException(status_code=404, detail="interaction not found")
        sets, params = [], []
        if csat is not None:
            sets.append("csat = ?")
            params.append(csat)
        if resolved is not None:
            sets.append("customer_resolved = ?")
            params.append(resolved)
        if not sets:
            raise HTTPException(status_code=400, detail="csat or resolved required")
        try:
            con.execute(
                f"UPDATE interactions SET {', '.join(sets)} WHERE interaction_id = ?",
                [*params, interaction_id],
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"outcome not recorded: {type(e).__name__}") from e
    return {"interaction_id": interaction_id, "csat": csat, "resolved": resolved}


@router.patch("/{interaction_id}/override")
async def human_override(
    interaction_id: str,
    body: dict[str, Any],
    _auth: bool = Depends(require_api_key),
    _role: str = Depends(require_perm_dep("takeover")),
    actor: str = Depends(get_actor),
) -> dict[str, Any]:
    """Supervisor override of slots/severity with a reason (board).

    Overrides are a DISTINCT ledgered action (``human_override`` + reason)
    so the Stage-7 audit never flags them as agent mismatches: the audit
    trail shows a human, not the model, set the value.
    """
    from src.ledger import AgentAction, record_action

    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="JSON object required")
    reason = str(body.get("reason") or "").strip()
    if not reason:
        raise HTTPException(status_code=400, detail="reason required for overrides")
    slots = body.get("slots") if isinstance(body.get("slots"), dict) else {}
    severity = body.get("severity")
    if severity is not None and severity not in ("Low", "Medium", "Critical"):
        raise HTTPException(status_code=400, detail="severity must be Low|Medium|Critical")
    if not slots and severity is None:
        raise HTTPException(status_code=400, detail="nothing to override")
    entry = await _get_entry(interaction_id)
    async with entry.lock:
        orch = entry.orch
        applied: dict[str, Any] = {}
        for k, v in slots.items():
            if k in ("entity_1", "entity_2", "entity_3", "category", "description"):
                orch.ctx.slots[k] = str(v)[:500]
                applied[k] = orch.ctx.slots[k]
        if severity is not None:
            orch.ctx.severity = severity
            orch.ctx.severity_source = "human"
            applied["severity"] = severity
        record_action(AgentAction(
            interaction_id=interaction_id,
            agent="supervisor",
            action_type="human_override",
            input_summary=f"actor={actor} reason={reason[:200]}",
            output_summary=f"overrides={applied}",
            case_id=orch.ctx.case_id,
        ))
    return {"interaction_id": interaction_id, "applied": applied, "actor": actor}


@router.post("/{interaction_id}/release")
async def release(
    interaction_id: str,
    _auth: bool = Depends(require_api_key),
    _role: str = Depends(require_perm_dep("takeover")),
    actor: str = Depends(get_actor),
    override: bool = False,
) -> dict[str, Any]:
    """Supervisor releases; AI resumes. Only the claimant (or an admin override) may."""
    if override and _role != "admin":
        raise HTTPException(status_code=403, detail="admin override required")
    entry = await _get_entry(interaction_id)
    async with entry.lock:
        claimed_by = (entry.orch.ctx.takeover_claimed_by or "").strip()
        if claimed_by and claimed_by != actor:
            if not (override and _role == "admin"):
                raise HTTPException(
                    status_code=403,
                    detail=f"call claimed by {claimed_by}; only claimant or admin override may release",
                )
        result = await entry.orch.release(actor=actor, admin_override=bool(override and _role == "admin"))
    if not result.get("ok"):
        code = result.get("code")
        status_code = 403 if code == "not_owner" else (400 if code == "not_supervised" else 409)
        raise HTTPException(
            status_code=status_code,
            detail={
                "code": result.get("code"),
                "claimed_by": result.get("claimed_by"),
                "state": result.get("state"),
            },
        )
    return {
        "interaction_id": interaction_id,
        "state": entry.orch.ctx.state,
        "supervised": bool(entry.orch.ctx.supervised),
        "generation": int(entry.orch.ctx.control_generation or 0),
    }


# ── Phase 1 Live Pilot Routing & Circuit Breaker Controls ────────────────────


@router.get("/routing/status")
async def get_routing_status(_auth: bool = Depends(require_api_key)) -> dict[str, Any]:
    """Inspect current Phase 1 live pilot traffic gate and circuit breaker status."""
    from src.routing import get_circuit_breaker, get_traffic_gate

    cb = get_circuit_breaker()
    tg = get_traffic_gate()
    return {
        "circuit_breaker": cb.status(),
        "traffic_gate": {
            "target_ratio": tg.target_ratio,
            "daily_max_calls": tg.daily_max_calls,
            "daily_routed_count": tg.daily_routed_count,
            "start_hour_utc": tg.start_hour,
            "end_hour_utc": tg.end_hour,
            "allowed_domain": tg.allowed_domain,
        },
    }


@router.post("/routing/panic")
async def trigger_panic_switch(
    request: Request,
    note: str = "",
    actor: str = Depends(get_actor),
    _auth: bool = Depends(require_api_key),
    _role: str = Depends(require_perm_dep("routing:control", open_mode_ok=True)),
) -> dict[str, Any]:
    """Single-click supervisor panic kill-switch; evacuates 100% of live traffic to human."""
    from src.routing import get_circuit_breaker

    cb = get_circuit_breaker()
    cb.panic(supervisor_id=actor or "supervisor", note=note)
    return {"status": "tripped", "is_tripped": True, "reason": cb.trip_reason}


@router.post("/routing/reset")
async def reset_circuit_breaker(
    request: Request,
    actor: str = Depends(get_actor),
    _auth: bool = Depends(require_api_key),
    _role: str = Depends(require_perm_dep("routing:control", open_mode_ok=True)),
) -> dict[str, Any]:
    """Supervisor manually resets circuit breaker after incident resolution."""
    from src.routing import get_circuit_breaker

    cb = get_circuit_breaker()
    cb.reset(supervisor_id=actor or "supervisor")
    return {"status": "reset", "is_tripped": False}


@router.post("/routing/record-latency")
async def record_latency_metric(
    latency_ms: float,
    _auth: bool = Depends(require_api_key),
) -> dict[str, Any]:
    """Record turn latency to monitor against 1,000ms ceiling."""
    from src.routing import get_circuit_breaker

    cb = get_circuit_breaker()
    cb.record_turn_latency(latency_ms)
    return {"is_tripped": cb.is_tripped, "trip_reason": cb.trip_reason}


@router.post("/routing/record-safety")
async def record_safety_metric(
    is_false_positive: bool = False,
    missed_safety: bool = False,
    _auth: bool = Depends(require_api_key),
) -> dict[str, Any]:
    """Record safety evaluation outcomes to enforce zero missed safety and max 3 consecutive false positives."""
    from src.routing import get_circuit_breaker

    cb = get_circuit_breaker()
    cb.record_safety_evaluation(is_false_positive=is_false_positive, missed_safety=missed_safety)
    return {"is_tripped": cb.is_tripped, "trip_reason": cb.trip_reason}


@router.post("/{interaction_id}/authorize-remedy")
async def authorize_remedy(
    interaction_id: str,
    _auth: bool = Depends(require_api_key),
    _role: str = Depends(require_perm_dep("approval:decide", open_mode_ok=True)),
) -> dict[str, Any]:
    """Stage 5 Remedy Offer HITL: supervisor authorizes proposed remedy before agent speaks it."""
    entry = await _get_entry(interaction_id)
    async with entry.lock:
        orch = entry.orch
        orch.ctx.remedy_authorized = True
        pending = getattr(orch.ctx, "pending_remedy_offer", None)
        if pending and pending.get("customer_text"):
            await orch.hooks._maybe(
                orch.hooks.emit_customer_turn,
                pending["customer_text"],
                {"speaker": "agent", "fast_path": True, "llm_used": False},
            )
            orch.ctx.record_turn("agent", pending["customer_text"], llm_used=False)
            orch.ctx.pending_remedy_offer = None
    return {"interaction_id": interaction_id, "authorized": True}


# ── WebSocket: per-interaction + console fan-out ─────────────────────────────

_CONSOLE_QUEUE_MAX = 32
_console_clients: dict[int, "_ConsoleClient"] = {}
_console_sub_lock = asyncio.Lock()


class _ConsoleClient:
    """One console socket with a bounded outbound queue and its own sender."""

    def __init__(self, ws: WebSocket, role: str) -> None:
        self.ws = ws
        self.role = role
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=_CONSOLE_QUEUE_MAX)
        self.task: asyncio.Task[None] | None = None


def _strip_restricted_ledger(msg: dict[str, Any]) -> dict[str, Any]:
    """Hide raw supervisor speech from roles that cannot read the ledger."""
    out = dict(msg)
    restricted = (
        str(out.get("action_type") or "") == "human_turn"
        or out.get("speaker") == "supervisor"
    )
    if not restricted:
        return out
    for key in ("summary", "output_summary", "input_summary", "text"):
        if out.get(key):
            out[key] = "[restricted]"
    return out


def _prepare_console_frame(msg: dict[str, Any], role: str) -> dict[str, Any]:
    from src.security.pii import redact_dict

    frame = msg
    if not has_perm(role, "ledger:read"):
        frame = _strip_restricted_ledger(frame)
    if not has_perm(role, "dsr:export"):
        frame = redact_dict(frame)
    return frame


async def _drop_console_client(client: _ConsoleClient) -> None:
    async with _console_sub_lock:
        _console_clients.pop(id(client.ws), None)
    task = client.task
    if task is not None and task is not asyncio.current_task():
        task.cancel()
    try:
        await client.ws.close()
    except Exception:
        pass


async def _console_sender(client: _ConsoleClient) -> None:
    try:
        while True:
            msg = await client.queue.get()
            frame = _prepare_console_frame(msg, client.role)
            await asyncio.wait_for(client.ws.send_json(frame), timeout=1.0)
    except asyncio.CancelledError:
        raise
    except Exception:
        await _drop_console_client(client)


async def _enqueue_console(client: _ConsoleClient, msg: dict[str, Any]) -> None:
    try:
        client.queue.put_nowait(msg)
    except asyncio.QueueFull:
        await _drop_console_client(client)


async def _broadcast_console(msg: dict[str, Any]) -> None:
    """Enqueue a frame for every console. Does not wait on subscriber I/O."""
    async with _console_sub_lock:
        clients = list(_console_clients.values())
    for client in clients:
        try:
            client.queue.put_nowait(msg)
        except asyncio.QueueFull:
            asyncio.create_task(_drop_console_client(client))


class _WSHooks(OrchestratorHooks):
    """Hooks that bridge the orchestrator to the customer WS + console fan-out."""

    def __init__(self, channel: WebVoiceChannel, interaction_id: str) -> None:
        # Do NOT call OrchestratorHooks.__init__: the dataclass would set every
        # hook field to None and shadow the methods defined below.
        self._channel = channel
        self._interaction_id = interaction_id

    @staticmethod
    async def _to_customer(coro: Any) -> None:
        """Send to the customer socket, tolerating a dropped/reconnecting client.

        The contact now survives a WS drop (reconnect grace), so the customer
        channel can legitimately be dead while the orchestrator and the
        supervisor console keep running. A failed customer send must never
        abort the turn or kill the console fan-out.
        """
        try:
            await coro
        except Exception:
            pass

    async def emit_customer_turn(self, text: str, meta: dict[str, Any]) -> None:
        # Remember this utterance for barge-in. A later generation must not
        # replace the one the caller is already hearing.
        try:
            entry = _active.get(self._interaction_id)
            if entry is not None:
                uid = entry.orch.note_emitted_utterance(
                    text,
                    utterance_id=str(meta.get("utterance_id") or meta.get("turn_id") or ""),
                    action_id=str(meta.get("action_id") or ""),
                    turn_id=str(meta.get("turn_id") or ""),
                )
                if uid and not meta.get("utterance_id"):
                    meta["utterance_id"] = uid
                if meta.get("action_id") in (None, "") and entry.orch.ctx.last_agent_action_id:
                    if entry.orch.ctx.playing_utterance_id == uid:
                        meta["action_id"] = entry.orch.ctx.last_agent_action_id
        except Exception:
            pass
        await self._to_customer(
            self._channel.send_turn(text, speaker=meta.get("speaker", "agent"), meta=meta)
        )
        await _broadcast_console({
            "type": "agent_turn",
            "interaction_id": self._interaction_id,
            "speaker": meta.get("speaker", "agent"),
            "text": text,
            "turn_id": meta.get("turn_id"),
            "utterance_id": meta.get("utterance_id"),
            "action_id": meta.get("action_id") or "",
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    async def emit_heard_turn(self, text: str, meta: dict[str, Any]) -> None:
        """Fan-out a customer utterance to the supervisor console."""
        await _broadcast_console({
            "type": "customer_turn",
            "interaction_id": self._interaction_id,
            "speaker": meta.get("speaker", "customer"),
            "text": text,
            "turn_id": meta.get("turn_id"),
            "ts": datetime.now(timezone.utc).isoformat(),
        })

    async def emit_activity(self, payload: dict[str, Any]) -> None:
        frame = normalize_activity_payload(payload, interaction_id=self._interaction_id)
        await self._to_customer(self._channel.send_activity(frame))
        await _broadcast_console(frame)

    async def emit_slots_update(self, slots: dict[str, str]) -> None:
        await self._to_customer(self._channel.send_slots_update(slots))
        await _broadcast_console({
            "type": "slots_update",
            "interaction_id": self._interaction_id,
            "slots": dict(slots),
        })

    async def emit_handoff_offer(self) -> None:
        await self._to_customer(self._channel.send_handoff_offer())
        await _broadcast_console({
            "type": "handoff_offer",
            "interaction_id": self._interaction_id,
        })

    async def emit_interaction_ended(self, payload: dict[str, Any]) -> None:
        await self._to_customer(self._channel.send_interaction_ended(payload))
        await _broadcast_console({
            "type": "interaction_ended",
            "interaction_id": self._interaction_id,
            **{k: _json_safe(v) for k, v in (payload or {}).items()},
        })

    async def emit_frustration_update(self, value: float) -> None:
        await _broadcast_console({
            "type": "frustration_update",
            "interaction_id": self._interaction_id,
            "value": float(value),
        })
        # P3 fix: the caller-facing widget is an operator surface — mirror the
        # meter here (was console-only in LiveContactConsole). Channels without
        # a control-frame surface ignore it.
        await self._to_customer(self._channel.send_control({
            "type": "frustration",
            "value": float(value),
        }))

    async def emit_turn_latency(self, payload: dict[str, Any]) -> None:
        frame = {"type": "turn_latency", "interaction_id": self._interaction_id,
                 **{k: _json_safe(v) for k, v in (payload or {}).items()}}
        await self._to_customer(self._channel.send_control(frame))
        await _broadcast_console(frame)

    async def emit_consent_required(self, payload: dict[str, Any]) -> None:
        frame = {"type": "consent_required", "interaction_id": self._interaction_id,
                 **{k: _json_safe(v) for k, v in (payload or {}).items()}}
        await self._to_customer(self._channel.send_control(frame))
        await _broadcast_console(frame)

    async def emit_supervisor_whisper(self, payload: dict[str, Any]) -> None:
        # Supervisor-only: console hears the coaching packet, caller never does.
        await _broadcast_console({
            "type": "supervisor_whisper",
            "interaction_id": self._interaction_id,
            **{k: _json_safe(v) for k, v in (payload or {}).items()},
        })

    async def emit_control_state(self, payload: dict[str, Any]) -> None:
        frame = {
            "type": "control_state",
            "interaction_id": self._interaction_id,
            **{k: _json_safe(v) for k, v in (payload or {}).items() if k != "type"},
        }
        await self._to_customer(self._channel.send_control(frame))
        await _broadcast_console(frame)


def _contact_scope_allows(websocket: WebSocket, interaction_id: str) -> bool:
    """A session that names contacts may attach only to those contacts.

    Customer socket must also match the initiating principal or have the capability token.
    """
    try:
        from src.api.auth import ws_principal

        principal = ws_principal(websocket)
        session = principal.session if principal is not None else None
    except Exception:
        principal = None
        session = None

    if isinstance(session, dict):
        scoped = session.get("interaction_id") or session.get("contact")
        if scoped and str(scoped) != interaction_id:
            return False
        contacts = session.get("contacts")
        if isinstance(contacts, list) and contacts:
            if interaction_id not in {str(item) for item in contacts}:
                return False

    entry = _active.get(interaction_id)
    if entry is not None:
        token = (
            websocket.query_params.get("token")
            or websocket.query_params.get("capability_token")
            or websocket.headers.get("x-capability-token")
        )
        if token and entry.capability_token and secrets.compare_digest(str(token).strip(), entry.capability_token):
            return True

        if entry.owner_subject:
            actor = actor_from_websocket(websocket)
            if actor and actor == entry.owner_subject:
                return True
            role = role_from_websocket(websocket)
            if role == "admin":
                return True
            return False

    return True


async def interaction_ws(websocket: WebSocket, interaction_id: str) -> None:
    """Bidirectional contact stream at ``/ws/interaction/{id}``.

    Client → server: optional auth frame, then
      {"type":"user_turn","text":"...","final":true} | barge_in | hangup
    Server → client: agent_turn | agent_activity | slots_update | handoff_offer
                     | interaction_ended | resumed | error

    Reconnect: a socket that closes **without** a ``hangup`` frame leaves the
    contact resumable for ``FRONTLINE_WS_RECONNECT_GRACE_S`` (default 120s).
    Re-attaching within that window replays the server transcript in a
    ``resumed`` frame; past it the ``error`` frame carries
    ``recoverable: false`` so the client stops retrying.

    Auth: prefer first-message ``{"type":"auth","api_key":"..."}`` or headers.
    Query ``?api_key=`` is deprecated.
    """
    if not frontline_enabled():
        await websocket.close(code=1013)
        return

    from src.api.limiter import check_ws_connect_rate

    ws_key = websocket.client.host if websocket.client else "unknown"
    if not check_ws_connect_rate(f"ix:{ws_key}"):
        try:
            await websocket.close(code=1013)
        except Exception:
            pass
        return

    try:
        await authenticate_websocket(websocket)
    except HTTPException:
        try:
            if websocket.client_state.name != "CONNECTED":
                await websocket.accept()
            await websocket.close(code=1008)
        except Exception:
            pass
        return

    if websocket.client_state.name != "CONNECTED":
        await websocket.accept()

    role = role_from_websocket(websocket)
    if not has_perm(role, "contact:write"):
        await websocket.send_json({
            "type": "error",
            "detail": f"role {role} lacks contact:write",
            "code": "forbidden",
            "recoverable": False,
        })
        await websocket.close(code=1008)
        return
    if not _contact_scope_allows(websocket, interaction_id):
        await websocket.send_json({
            "type": "error",
            "detail": "credential is not allowed to attach to this contact",
            "code": "forbidden",
            "recoverable": False,
        })
        await websocket.close(code=1008)
        return

    try:
        entry, resumed = await _attach_customer_ws(interaction_id)
    except HTTPException as e:
        detail = e.detail if isinstance(e.detail, str) else str(e.detail)
        await websocket.send_json({
            "type": "error",
            "detail": detail,
            "code": "interaction_busy" if e.status_code == 409 else "interaction_not_resumable",
            # Tell the client whether retrying can ever succeed, so it does not
            # spin through its whole reconnect budget on a dead contact.
            "recoverable": False,
        })
        await websocket.close(code=1008 if e.status_code == 409 else 1000)
        return

    orch = entry.orch
    from src.frontline.dsr import is_interaction_erased
    if getattr(orch, "_erased", False) or is_interaction_erased(interaction_id):
        await websocket.close(code=1008)
        return
    channel = WebVoiceChannel(websocket)
    orch.hooks = _WSHooks(channel, interaction_id)
    # Track whether the customer left on purpose; only then do we drop the
    # registry entry in `finally` (otherwise the contact stays resumable).
    hung_up = False

    if resumed:
        # Re-sync the client after a drop. The server transcript is
        # authoritative: the client may have missed turns while it was away,
        # so it replaces its local list rather than appending to it.
        try:
            await websocket.send_json({
                "type": "resumed",
                "interaction_id": interaction_id,
                "state": orch.ctx.state,
                "supervised": bool(orch.ctx.supervised),
                "generation": int(orch.ctx.control_generation or 0),
                "claimed_by": orch.ctx.takeover_claimed_by,
                "turns": [
                    {
                        "id": t.get("turn_id"),
                        "speaker": t.get("speaker"),
                        "text": t.get("text", ""),
                    }
                    for t in (orch.ctx.turns or [])
                    if t.get("text")
                ],
            })
            await channel.send_slots_update(dict(orch.ctx.slots))
        except Exception:
            pass

    try:
        while True:
            try:
                msg = await websocket.receive_json()
            except WebSocketDisconnect:
                raise
            except Exception:
                try:
                    await websocket.send_json({
                        "type": "error",
                        "detail": "invalid JSON",
                        "code": "bad_message",
                        "recoverable": True,
                    })
                except Exception:
                    pass
                continue
            if not isinstance(msg, dict):
                try:
                    await websocket.send_json({
                        "type": "error",
                        "detail": "JSON object required",
                        "code": "bad_message",
                        "recoverable": True,
                    })
                except Exception:
                    pass
                continue
            mtype = msg.get("type")
            if mtype is not None and not isinstance(mtype, str):
                await websocket.send_json({
                    "type": "error",
                    "detail": "type must be a string",
                    "code": "bad_message",
                    "recoverable": True,
                })
                continue
            if mtype == "user_turn":
                # Durable replay dedup lives in ONE place: the orchestrator's
                # _seen_client_turn (turn_dedup PK). A second pre-insert here
                # would make every first turn look like a replay (the row it
                # just wrote) and silently eat 100% of widget turns — so this
                # route must NOT touch turn_dedup before handle_customer_turn.
                # Telephony replays are sequenced separately in twilio_ws.
                if not isinstance(msg.get("text", ""), str):
                    await websocket.send_json({
                        "type": "error",
                        "detail": "text must be a string",
                        "code": "bad_message",
                        "recoverable": True,
                    })
                    continue
                t0 = time.monotonic()
                logger.debug("turn_recv iid=%s", interaction_id)
                # Normalize ASR confidence: client may send overall float or dict.
                _asr = msg.get("asr_confidence")
                _conf = msg.get("confidence")
                try:
                    _conf_f = float(_conf) if _conf is not None else None
                except (TypeError, ValueError):
                    _conf_f = None
                if not isinstance(_asr, dict):
                    _asr = None
                async with entry.lock:
                    try:
                        await orch.handle_customer_turn(
                            msg.get("text", ""), final=msg.get("final", True),
                            client_turn_id=msg.get("turn_id", msg.get("client_turn_id")),
                            asr_confidence=_asr,
                            confidence=_conf_f,
                            region=msg.get("region"),
                            dtmf=msg.get("dtmf"),
                            turn_seq=msg.get("turn_seq", msg.get("seq")),
                            drive_hint=msg.get("drive_hint", msg.get("driving")),
                            locale=msg.get("locale"),
                        )
                    except Exception as turn_exc:
                        from src.agents.base import TurnPersistenceError

                        if isinstance(turn_exc, TurnPersistenceError):
                            await websocket.send_json({
                                "type": "error",
                                "detail": "That turn could not be saved. Please repeat it.",
                                "code": "turn_not_durable",
                                "recoverable": True,
                            })
                            continue
                        raise
                elapsed_ms = (time.monotonic() - t0) * 1000.0
                try:
                    entry.turn_seq = int(entry.turn_seq or 0) + 1
                except Exception:
                    pass
                from src.routing import get_circuit_breaker
                cb = get_circuit_breaker()
                cb.record_turn_latency(elapsed_ms)
                if cb.is_tripped:
                    await websocket.send_json({
                        "type": "circuit_breaker_tripped",
                        "action": "redirect_to_human",
                        "reason": cb.trip_reason,
                    })
                if orch.ctx.state in ("DONE", "ABANDONED"):
                    break
            elif mtype == "barge_in":
                # P1 fix: reconcile the ledger to the audible prefix instead of
                # claiming the full prompt was spoken.
                try:
                    _elapsed = msg.get("elapsed_ms", msg.get("elapsedMs", 0))
                    _utt = msg.get("utterance_id") or msg.get("turn_id")
                    async with entry.lock:
                        _audible = await orch.handle_barge_in_interrupt(
                            int(float(_elapsed or 0)),
                            utterance_id=str(_utt) if _utt else None,
                        )
                    try:
                        await websocket.send_json({
                            "type": "barge_in_reconciled",
                            "audible_text": _audible,
                            "estimated": True,
                        })
                    except Exception:
                        pass
                except Exception:
                    pass
            elif mtype == "hangup":
                hung_up = True
                async with entry.lock:
                    await orch.hangup()
                break
            elif mtype == "human_turn":
                role = role_from_websocket(websocket)
                try:
                    require_perm(role, "takeover")
                except HTTPException:
                    await websocket.send_json(
                        {"type": "error", "detail": "role lacks takeover", "code": "forbidden"}
                    )
                    continue
                if not isinstance(msg.get("text", ""), str):
                    await websocket.send_json({
                        "type": "error",
                        "detail": "text must be a string",
                        "code": "bad_message",
                        "recoverable": True,
                    })
                    continue
                actor = actor_from_websocket(websocket)
                async with entry.lock:
                    claimed_by = (orch.ctx.takeover_claimed_by or "").strip()
                    if claimed_by and claimed_by != actor and role != "admin":
                        await websocket.send_json({
                            "type": "human_turn_result",
                            "ok": False,
                            "code": "not_owner",
                            "claimed_by": claimed_by,
                            "interaction_id": interaction_id,
                        })
                        continue
                    result = await orch.human_turn(
                        msg.get("text", ""),
                        actor=actor,
                        message_id=str(msg.get("message_id") or "") or None,
                    )
                await websocket.send_json({
                    "type": "human_turn_result",
                    "interaction_id": interaction_id,
                    **{k: _json_safe(v) for k, v in result.items()},
                })
    except WebSocketDisconnect:
        # Socket dropped without a hangup. Do NOT end the contact here: the
        # dashboard reconnects with backoff, and tearing the orchestrator down
        # made every reconnect 404 ("active interaction not found"). The reaper
        # closes it if the browser never re-attaches within the grace window.
        pass
    except Exception as e:
        # H1: any agent/turn crash must not leave status='active' forever.
        hung_up = True
        try:
            if orch.ctx.state not in ("DONE", "ABANDONED"):
                async with entry.lock:
                    await orch.hangup()
        except Exception:
            pass
        await _mark_failed(interaction_id, f"ws_error:{type(e).__name__}")
        try:
            await websocket.send_json({
                "type": "error",
                "detail": f"server error: {type(e).__name__}",
                "code": "server_error",
                "recoverable": False,
            })
        except Exception:
            pass
    finally:
        if hung_up or orch.ctx.state in ("DONE", "ABANDONED"):
            await _unregister(interaction_id)
        else:
            await _detach_customer_ws(interaction_id)
        try:
            await websocket.close()
        except Exception:
            pass


async def twilio_ws(websocket: WebSocket, interaction_id: str) -> None:
    """Twilio Media Streams socket at ``/ws/twilio/{id}`` (Tier-A #8).

    Uses TelephonyMediaBridge (μ-law ⇄ PCM) + drive_mode_guard on live PCM +
    ServerSpeechStack STT, then the SAME orchestrator path as web_voice — no
    agent changes. Barge-in clears the Twilio buffer and reconciles the ledger.
    """
    if not frontline_enabled():
        await websocket.close(code=1013)
        return

    try:
        await authenticate_websocket(websocket)
    except HTTPException:
        try:
            if websocket.client_state.name != "CONNECTED":
                await websocket.accept()
            await websocket.close(code=1008)
        except Exception:
            pass
        return

    if websocket.client_state.name != "CONNECTED":
        await websocket.accept()

    role = role_from_websocket(websocket)
    if not has_perm(role, "contact:write"):
        try:
            await websocket.send_json({
                "type": "error",
                "detail": f"role {role} lacks contact:write",
                "code": "forbidden",
                "recoverable": False,
            })
            await websocket.close(code=1008)
        except Exception:
            pass
        return

    if not _contact_scope_allows(websocket, interaction_id):
        try:
            await websocket.send_json({
                "type": "error",
                "detail": "credential is not allowed to attach to this contact",
                "code": "forbidden",
                "recoverable": False,
            })
            await websocket.close(code=1008)
        except Exception:
            pass
        return

    try:
        entry, _resumed = await _attach_customer_ws(interaction_id)
    except HTTPException as e:
        detail = e.detail if isinstance(e.detail, str) else str(e.detail)
        try:
            await websocket.send_json({"type": "error", "detail": detail,
                                       "recoverable": False})
        except Exception:
            pass
        try:
            await websocket.close(code=1008)
        except Exception:
            pass
        return
    orch = entry.orch
    try:
        from src.channels.stt_tts import ServerSpeechStack
        from src.voice.wiring import check_drive_mode, telephony_bridge_for

        speech = ServerSpeechStack()
    except Exception:
        speech = None  # type: ignore
        telephony_bridge_for = None  # type: ignore
        check_drive_mode = None  # type: ignore

    import hashlib as _hashlib
    import time as _ttime
    _twilio_seq = 0

    async def _on_speech(pcm_16k: bytes) -> None:
        nonlocal _twilio_seq
        try:
            driving = bool(check_drive_mode(pcm_chunk=pcm_16k)) if check_drive_mode else False
            text = None
            try:
                # Media frames arrive as PCM here; STT stack handles text pilot frames.
                text = speech.transcribe_chunk("") if speech else None
            except Exception:
                text = None
            if text:
                # Telephony replay guard: server-side monotonic key per frame.
                # No client_turn_id is forwarded, so the orchestrator's web
                # dedup never sees these keys — exactly one durable check.
                _twilio_seq += 1
                try:
                    from src.voice.wiring import acquire_durable_turn_slot

                    _key = "tw_" + _hashlib.sha256(
                        f"{interaction_id}|{_twilio_seq}|{text}".encode()
                    ).hexdigest()[:20]
                    _own = await acquire_durable_turn_slot(
                        interaction_id, _twilio_seq, _key, str(text))
                    if not _own:
                        return
                except Exception:
                    pass
                async with entry.lock:
                    await orch.handle_customer_turn(
                        str(text), final=True,
                        client_turn_id=None, drive_hint=driving or None,
                    )
        except Exception:
            pass

    async def _on_barge() -> None:
        try:
            async with entry.lock:
                await orch.handle_barge_in_interrupt(600)
        except Exception:
            pass

    bridge = None
    try:
        if telephony_bridge_for is not None:
            bridge = telephony_bridge_for(websocket, interaction_id, _on_speech, _on_barge)
            await bridge.run()
        else:
            await websocket.close()
    except Exception:
        pass
    finally:
        await _detach_customer_ws(interaction_id)
        try:
            await websocket.close()
        except Exception:
            pass


@router.get("/diagnostics/voice-latency")
async def voice_latency_diagnostics(
    _auth: bool = Depends(require_api_key),
) -> dict[str, Any]:
    """Live one-trial pipeline proof (benchmark_latency prod caller)."""
    try:
        from src.voice.wiring import run_single_trial_diagnostics

        return await run_single_trial_diagnostics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"diagnostics failed: {type(e).__name__}") from e


async def console_ws(websocket: WebSocket) -> None:
    """Live console feed at ``/ws/console``.

    Server → console: agent_activity (+ lifecycle) for all active interactions.
    Console → server (after takeover): {"type":"human_turn","interaction_id":"...","text":"..."}

    Auth: first-message auth frame preferred; query api_key deprecated.
    """
    if not frontline_enabled():
        await websocket.close(code=1013)
        return

    from src.api.limiter import check_ws_connect_rate

    ws_key = websocket.client.host if websocket.client else "unknown"
    if not check_ws_connect_rate(f"console:{ws_key}"):
        try:
            await websocket.close(code=1013)
        except Exception:
            pass
        return

    try:
        await authenticate_websocket(websocket)
    except HTTPException:
        try:
            if websocket.client_state.name != "CONNECTED":
                await websocket.accept()
            await websocket.close(code=1008)
        except Exception:
            pass
        return

    role = role_from_websocket(websocket)
    try:
        require_perm(role, "case:read", open_mode_ok=True)
    except HTTPException:
        try:
            if websocket.client_state.name != "CONNECTED":
                await websocket.accept()
            await websocket.close(code=1008)
        except Exception:
            pass
        return

    if websocket.client_state.name != "CONNECTED":
        await websocket.accept()
    client = _ConsoleClient(websocket, role)
    client.task = asyncio.create_task(_console_sender(client))
    async with _console_sub_lock:
        _console_clients[id(websocket)] = client
    # Look back so a console that connects mid-call still sees recent activity
    # (and so tests that ledger-then-connect are not deadlocked waiting forever).
    last_ts: datetime = datetime.now(timezone.utc) - timedelta(hours=1)
    last_action_id = ""

    async def _catch_up_loop() -> None:
        nonlocal last_ts, last_action_id
        import os

        while True:
            try:
                poll_s = float(os.getenv("FRONTLINE_CONSOLE_POLL_S", "2") or "2")
            except ValueError:
                poll_s = 2.0
            rows = await asyncio.to_thread(
                fetch_agent_actions_since, last_ts, 50, last_action_id
            )
            for r in rows:
                raw_ts = r.get("ts")
                if isinstance(raw_ts, datetime):
                    last_ts = raw_ts
                    last_action_id = str(r.get("action_id") or "")
                try:
                    await asyncio.wait_for(
                        client.queue.put(activity_frame_from_row(r)),
                        timeout=5.0,
                    )
                except Exception:
                    await _drop_console_client(client)
                    return
            await asyncio.sleep(max(0.05, poll_s))

    catch_task = asyncio.create_task(_catch_up_loop())
    try:
        while True:
            try:
                msg = await websocket.receive_json()
            except WebSocketDisconnect:
                raise
            except Exception:
                await _enqueue_console(client, {
                    "type": "error",
                    "detail": "invalid JSON",
                    "code": "bad_message",
                })
                continue
            if not isinstance(msg, dict):
                await _enqueue_console(client, {
                    "type": "error",
                    "detail": "JSON object required",
                    "code": "bad_message",
                })
                continue
            if msg.get("type") == "release":
                iid = msg.get("interaction_id")
                override = bool(msg.get("override", False))
                try:
                    entry = await _get_entry(iid)
                    actor = actor_from_websocket(websocket)
                    role = role_from_websocket(websocket)
                    async with entry.lock:
                        claimed_by = (entry.orch.ctx.takeover_claimed_by or "").strip()
                        if claimed_by and claimed_by != actor and not (override and role == "admin"):
                            await _enqueue_console(client, {
                                "type": "release_result",
                                "ok": False,
                                "code": "not_owner",
                                "claimed_by": claimed_by,
                                "interaction_id": iid,
                            })
                            continue
                        result = await entry.orch.release(actor=actor, admin_override=bool(override and role == "admin"))
                    await _enqueue_console(client, {
                        "type": "release_result",
                        "interaction_id": iid,
                        **{k: _json_safe(v) for k, v in result.items()},
                    })
                except Exception as ex:
                    await _enqueue_console(client, {
                        "type": "error",
                        "detail": str(ex),
                        "code": "release_failed",
                    })
                continue
            if msg.get("type") != "human_turn":
                continue
            role = role_from_websocket(websocket)
            try:
                require_perm(role, "takeover")
            except HTTPException:
                await _enqueue_console(client, {
                    "type": "error",
                    "detail": "role lacks takeover",
                    "code": "forbidden",
                })
                continue
            iid = msg.get("interaction_id")
            text = msg.get("text", "")
            if not isinstance(text, str) or not isinstance(iid, str):
                await _enqueue_console(client, {
                    "type": "human_turn_result",
                    "ok": False,
                    "code": "bad_message",
                    "interaction_id": iid if isinstance(iid, str) else "",
                    "message_id": str(msg.get("message_id") or ""),
                })
                continue
            try:
                entry = await _get_entry(iid)
                actor = actor_from_websocket(websocket)
                async with entry.lock:
                    claimed_by = (entry.orch.ctx.takeover_claimed_by or "").strip()
                    if claimed_by and claimed_by != actor and role != "admin":
                        await _enqueue_console(client, {
                            "type": "human_turn_result",
                            "ok": False,
                            "code": "not_owner",
                            "claimed_by": claimed_by,
                            "interaction_id": iid,
                            "message_id": str(msg.get("message_id") or ""),
                        })
                        continue
                    result = await entry.orch.human_turn(
                        text,
                        actor=actor,
                        message_id=str(msg.get("message_id") or "") or None,
                    )
                await _enqueue_console(client, {
                    "type": "human_turn_result",
                    "interaction_id": iid,
                    **{k: _json_safe(v) for k, v in result.items()},
                })
            except HTTPException:
                await _enqueue_console(client, {
                    "type": "error",
                    "detail": f"interaction not active: {iid}",
                    "code": "not_active",
                })
    except WebSocketDisconnect:
        return
    finally:
        catch_task.cancel()
        try:
            await catch_task
        except (asyncio.CancelledError, Exception):
            pass
        await _drop_console_client(client)


__all__ = [
    "router",
    "interaction_ws",
    "twilio_ws",
    "console_ws",
    "_active",
    "reap_orphans",
    "reaper_loop",
    "_sweep_stale_active_db_rows",
    "ActiveEntry",
    "_ORPHAN_TTL_S",
    "_reconnect_grace_s",
    "_is_reapable",
    "classify_customer_ws_attach",
    "_attach_customer_ws",
    "_detach_customer_ws",
    "_json_safe",
    "activity_frame_from_row",
    "normalize_activity_payload",
    "fetch_agent_actions_since",
]
