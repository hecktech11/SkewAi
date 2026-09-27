"""Outbound case/investigation connector — pilot generic HTTP + dry-run outbox.

This is **not** Salesforce / ServiceNow / Twilio product integration. It pushes
structured JSON for case-created / investigation-opened events to:

  1. A local dry-run outbox under ``data/connectors/outbox/`` (always when enabled)
  2. An optional HTTP webhook (``CONNECTOR_WEBHOOK_URL`` or runtime config)

Delivery is fire-and-forget from the contact path: failures are recorded and
never raised into the orchestrator. Failed HTTP deliveries stay ``pending`` for
ops replay (same pilot pattern as alert dead-letters).

Config precedence:
  1. Runtime file ``data/connectors/config.json`` (set via PUT admin API)
  2. Env / Settings defaults (``CONNECTOR_ENABLED``, ``CONNECTOR_WEBHOOK_URL``,
     ``CONNECTOR_SHARED_SECRET``)
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

from src.config import REPO_ROOT, settings
from src.data.warehouse import ops_con
from src.ids import new_ulid
from src.ledger import AgentAction, record_action

logger = logging.getLogger(__name__)

CONNECTOR_RETRY_ATTEMPTS = 3
CONNECTOR_RETRY_BASE_S = 0.02

CONFIG_PATH = REPO_ROOT / "data" / "connectors" / "config.json"
OUTBOX_DIR = REPO_ROOT / "data" / "connectors" / "outbox"

_config_lock = threading.Lock()
_runtime_config: dict[str, Any] | None = None


# ── Config ────────────────────────────────────────────────────────────────────


def _env_defaults() -> dict[str, Any]:
    return {
        "enabled": bool(settings.connector_enabled),
        "webhook_url": (settings.connector_webhook_url or "").strip(),
        "shared_secret": (settings.connector_shared_secret or "").strip(),
    }


def _load_file_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {}
    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return {}
        return raw
    except Exception as e:
        logger.warning("connector config read failed: %s", e)
        return {}


def get_connector_config(*, include_secret: bool = False) -> dict[str, Any]:
    """Merged runtime + env config. Secrets redacted unless include_secret=True."""
    global _runtime_config
    with _config_lock:
        if _runtime_config is None:
            _runtime_config = _load_file_config()
        file_cfg = dict(_runtime_config)
    base = _env_defaults()
    # File overrides only keys that are explicitly present.
    for k in ("enabled", "webhook_url", "shared_secret"):
        if k in file_cfg and file_cfg[k] is not None:
            base[k] = file_cfg[k]
    base["enabled"] = bool(base.get("enabled"))
    base["webhook_url"] = (base.get("webhook_url") or "").strip()
    base["shared_secret"] = (base.get("shared_secret") or "").strip()
    if not include_secret:
        secret = base.get("shared_secret") or ""
        base["shared_secret_set"] = bool(secret)
        base["shared_secret"] = ""
        url = base.get("webhook_url") or ""
        base["webhook_url_redacted"] = _redact_url(url)
        base["webhook_url_set"] = bool(url)
        # Never echo full URL or secret in list/status responses.
        base["webhook_url"] = base["webhook_url_redacted"]
    return base


def set_connector_config(
    *,
    enabled: bool | None = None,
    webhook_url: str | None = None,
    shared_secret: str | None = None,
) -> dict[str, Any]:
    """Persist runtime connector config. Returns redacted status."""
    global _runtime_config
    from src.security.url_guard import validate_outbound_url

    with _config_lock:
        current = dict(_runtime_config) if _runtime_config is not None else _load_file_config()
        if enabled is not None:
            current["enabled"] = bool(enabled)
        if webhook_url is not None:
            cleaned = webhook_url.strip()
            if cleaned:
                from urllib.parse import urlparse

                p = urlparse(cleaned)
                if p.username or p.password:
                    raise ValueError("Webhook URL must not contain embedded user credentials")
                # Reject SSRF targets at config time (not only at dispatch).
                validate_outbound_url(cleaned)
            old_url = current.get("webhook_url") or ""
            current["webhook_url"] = cleaned
            if cleaned != old_url and shared_secret is None:
                current["shared_secret"] = ""
        if shared_secret is not None:
            current["shared_secret"] = shared_secret.strip()
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _runtime_config = current
    return get_connector_config(include_secret=False)


def reset_connector_config_cache() -> None:
    """Test helper: drop in-memory cache so next read reloads file/env."""
    global _runtime_config
    with _config_lock:
        _runtime_config = None


def _redact_url(url: str) -> str:
    if not url:
        return ""
    # Keep scheme + host (excluding userinfo), drop path/query secrets.
    try:
        from urllib.parse import urlparse

        p = urlparse(url)
        h = p.hostname or "…"
        host = f"{h}:{p.port}" if p.port else h
        return f"{p.scheme}://{host}/…" if p.scheme else f"{host}/…"
    except Exception:
        return url[:24] + "…" if len(url) > 24 else url


# ── Payload ───────────────────────────────────────────────────────────────────


def build_payload(
    event: str,
    *,
    case_id: str | None = None,
    investigation_id: str | None = None,
    interaction_id: str | None = None,
    pack_id: str | None = None,
    cluster_id: int | str | None = None,
    title: str | None = None,
    severity: str | None = None,
    priority: int | None = None,
    category: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pure structured payload for outbound sinks. Unit-testable without I/O."""
    from src.data.timeutil import to_iso_z, utc_now

    payload: dict[str, Any] = {
        "schema_version": 1,
        "event": event,
        "ts": to_iso_z(utc_now()),
        "source": "frontline_v2",
        "case_id": case_id,
        "investigation_id": investigation_id,
        "interaction_id": interaction_id,
        "pack_id": pack_id,
        "cluster_id": cluster_id,
        "title": title,
        "severity": severity,
        "priority": priority,
        "category": category,
    }
    if extra:
        payload["extra"] = extra
    # Drop nulls for a cleaner wire format, keep event/ts/source always.
    return {k: v for k, v in payload.items() if v is not None or k in ("event", "ts", "source")}


def _safe_title(raw: Any) -> str | None:
    text = (str(raw) if raw is not None else "")[:200]
    if not text:
        return None
    try:
        from src.security.pii import redact_pii

        return redact_pii(text)
    except Exception:
        return text


def build_case_payload_from_db(case_id: str) -> dict[str, Any] | None:
    """Load a case row and build a manual_export payload. None if missing."""
    with ops_con(read_only=True) as con:
        cur = con.execute(
            """
            SELECT case_id, interaction_id, pack_id, investigation_id,
                   cluster_match_id, category, severity, priority,
                   description_summary, status
            FROM cases WHERE case_id = ?
            """,
            [case_id],
        )
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        c = dict(zip(cols, row))
    from src.security.pii import decrypt_case_row

    c = decrypt_case_row(c) or c
    return build_payload(
        "manual_export",
        case_id=c.get("case_id"),
        investigation_id=c.get("investigation_id"),
        interaction_id=c.get("interaction_id"),
        pack_id=c.get("pack_id"),
        cluster_id=c.get("cluster_match_id"),
        severity=c.get("severity"),
        priority=c.get("priority"),
        category=c.get("category"),
        title=_safe_title(c.get("description_summary")),
        extra={"status": c.get("status")},
    )


# ── Outbox + HTTP ─────────────────────────────────────────────────────────────


def _write_outbox(delivery_id: str, payload: dict[str, Any]) -> str:
    OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTBOX_DIR / f"{delivery_id}.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


async def _post_once(
    url: str,
    payload: dict[str, Any],
    *,
    shared_secret: str = "",
) -> tuple[bool, int | None, str | None]:
    """Single POST. Returns (ok, http_status, error). Never raises."""
    from src.security.url_guard import validate_outbound_url

    try:
        url = validate_outbound_url(url)
    except ValueError as e:
        return False, None, f"ssrf_blocked:{e}"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if shared_secret:
        headers["X-Connector-Secret"] = shared_secret
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code >= 400:
            return False, resp.status_code, f"http_{resp.status_code}: {resp.text[:200]}"
        return True, resp.status_code, None
    except Exception as e:
        return False, None, f"{type(e).__name__}: {e}"


async def _post_with_retries(
    url: str,
    payload: dict[str, Any],
    *,
    shared_secret: str = "",
    max_attempts: int | None = None,
    base_delay_s: float | None = None,
) -> tuple[bool, int, int | None, str | None]:
    """Returns (ok, attempts, last_http_status, last_error)."""
    attempts_n = max_attempts if max_attempts is not None else CONNECTOR_RETRY_ATTEMPTS
    base = base_delay_s if base_delay_s is not None else CONNECTOR_RETRY_BASE_S
    last_err: str | None = None
    last_status: int | None = None
    for i in range(1, attempts_n + 1):
        ok, status, err = await _post_once(url, payload, shared_secret=shared_secret)
        last_status = status
        if ok:
            return True, i, status, None
        last_err = err
        if i < attempts_n:
            await asyncio.sleep(base * (2 ** (i - 1)))
    return False, attempts_n, last_status, last_err


def _record_delivery(
    *,
    delivery_id: str,
    event: str,
    ref_id: str,
    interaction_id: str | None,
    case_id: str | None,
    investigation_id: str | None,
    payload: dict[str, Any],
    sink: str,
    status: str,
    attempts: int,
    error: str | None,
    outbox_path: str | None,
    http_status: int | None,
) -> None:
    from src.data.timeutil import utc_now

    now = utc_now()
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO connector_deliveries
            (delivery_id, event, ref_id, interaction_id, case_id, investigation_id,
             payload_json, sink, status, attempts, error, outbox_path, http_status,
             created_at, last_attempt_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                delivery_id,
                event,
                str(ref_id),
                interaction_id,
                case_id,
                investigation_id,
                json.dumps(payload, default=str),
                sink,
                status,
                attempts,
                (error or "")[:1000] if error else None,
                outbox_path,
                http_status,
                now,
                now,
            ],
        )


def list_deliveries(
    limit: int = 50,
    status: str | None = None,
) -> list[dict[str, Any]]:
    """List connector delivery rows (newest first)."""
    limit = min(max(int(limit), 1), 200)
    with ops_con(read_only=True) as con:
        if status:
            cur = con.execute(
                """
                SELECT delivery_id, event, ref_id, interaction_id, case_id,
                       investigation_id, payload_json, sink, status, attempts,
                       error, outbox_path, http_status, created_at, last_attempt_at
                FROM connector_deliveries
                WHERE status = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                [status, limit],
            )
        else:
            cur = con.execute(
                """
                SELECT delivery_id, event, ref_id, interaction_id, case_id,
                       investigation_id, payload_json, sink, status, attempts,
                       error, outbox_path, http_status, created_at, last_attempt_at
                FROM connector_deliveries
                ORDER BY created_at DESC
                LIMIT ?
                """,
                [limit],
            )
        cols = [d[0] for d in cur.description]
        rows = []
        for r in cur.fetchall():
            d = dict(zip(cols, r))
            for k in ("created_at", "last_attempt_at"):
                if isinstance(d.get(k), datetime):
                    d[k] = d[k].isoformat()
            rows.append(d)
    return rows


# ── Dispatch ──────────────────────────────────────────────────────────────────


def _ledger_row_hash(action_id: str) -> str | None:
    with ops_con(read_only=True) as con:
        row = con.execute(
            "SELECT row_hash FROM agent_actions WHERE action_id = ?",
            [action_id],
        ).fetchone()
    return str(row[0]) if row and row[0] else None


def export_audited_signal(
    event: str,
    *,
    case_id: str | None = None,
    investigation_id: str | None = None,
    interaction_id: str | None = None,
    pack_id: str | None = None,
    cluster_id: int | str | None = None,
    title: str | None = None,
    severity: str | None = None,
    priority: int | None = None,
    category: str | None = None,
    extra: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Explicit audited export surface: ledger + outbox, always recorded.

    Unlike ``dispatch_event`` this does not skip when the connector webhook is
    disabled. The outbox file is the delivery artifact; the payload always
    carries event identity and ledger linkage (action_id + row_hash).
    """
    delivery_id = "cdl_" + new_ulid()
    ref_id = case_id or investigation_id or interaction_id or delivery_id
    ix = interaction_id or "platform"

    action_id = record_action(
        AgentAction(
            interaction_id=ix,
            agent="orchestrator",
            action_type="signal_exported",
            input_summary=f"event={event}, ref_id={ref_id}",
            output_summary=f"delivery_id={delivery_id}",
            evidence_ids=[str(ref_id)],
            case_id=case_id,
            ok=True,
        )
    )
    row_hash = _ledger_row_hash(action_id)

    body = payload or build_payload(
        event,
        case_id=case_id,
        investigation_id=investigation_id,
        interaction_id=interaction_id,
        pack_id=pack_id,
        cluster_id=cluster_id,
        title=title,
        severity=severity,
        priority=priority,
        category=category,
        extra=extra,
    )
    body = dict(body)
    audit = {
        "delivery_id": delivery_id,
        "ledger_action_id": action_id,
        "ledger_row_hash": row_hash,
        "ref_id": str(ref_id),
    }
    body["audit"] = audit

    outbox_path = _write_outbox(delivery_id, body)
    _record_delivery(
        delivery_id=delivery_id,
        event=event,
        ref_id=str(ref_id),
        interaction_id=interaction_id,
        case_id=case_id,
        investigation_id=investigation_id,
        payload=body,
        sink="outbox",
        status="success",
        attempts=1,
        error=None,
        outbox_path=outbox_path,
        http_status=None,
    )
    return {
        "ok": True,
        "delivery_id": delivery_id,
        "status": "success",
        "sink": "outbox",
        "outbox_path": outbox_path,
        "ledger_action_id": action_id,
        "ledger_row_hash": row_hash,
        "event": event,
        "ref_id": str(ref_id),
    }


async def dispatch_event(
    event: str,
    *,
    case_id: str | None = None,
    investigation_id: str | None = None,
    interaction_id: str | None = None,
    pack_id: str | None = None,
    cluster_id: int | str | None = None,
    title: str | None = None,
    severity: str | None = None,
    priority: int | None = None,
    category: str | None = None,
    extra: dict[str, Any] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch a structured event. Never raises into the caller.

    Returns a small status dict (ok, delivery_id, status, sink, error).
    """
    cfg = get_connector_config(include_secret=True)
    if not cfg.get("enabled"):
        return {"ok": False, "skipped": True, "reason": "connector_disabled"}

    delivery_id = "cdl_" + new_ulid()
    body = payload or build_payload(
        event,
        case_id=case_id,
        investigation_id=investigation_id,
        interaction_id=interaction_id,
        pack_id=pack_id,
        cluster_id=cluster_id,
        title=title,
        severity=severity,
        priority=priority,
        category=category,
        extra=extra,
    )
    ref_id = case_id or investigation_id or interaction_id or delivery_id

    outbox_path: str | None = None
    http_ok = True
    http_status: int | None = None
    http_err: str | None = None
    attempts = 0
    sinks: list[str] = []

    try:
        outbox_path = _write_outbox(delivery_id, body)
        sinks.append("outbox")
    except Exception as e:
        logger.warning("connector outbox write failed: %s", e)
        try:
            _record_delivery(
                delivery_id=delivery_id,
                event=event,
                ref_id=str(ref_id),
                interaction_id=interaction_id,
                case_id=case_id,
                investigation_id=investigation_id,
                payload=body,
                sink="outbox",
                status="failed",
                attempts=0,
                error=f"outbox: {e}",
                outbox_path=None,
                http_status=None,
            )
        except Exception as e2:
            logger.warning("connector delivery record failed: %s", e2)
        return {
            "ok": False,
            "delivery_id": delivery_id,
            "status": "failed",
            "error": f"outbox: {e}",
        }

    url = (cfg.get("webhook_url") or "").strip()
    secret = (cfg.get("shared_secret") or "").strip()
    if url:
        # Validate before any network I/O (SSRF guard).
        from src.security.url_guard import validate_outbound_url

        try:
            url = validate_outbound_url(url)
            sinks.append("http")
            http_ok, attempts, http_status, http_err = await _post_with_retries(
                url, body, shared_secret=secret
            )
        except ValueError as e:
            http_ok = False
            http_err = f"ssrf_blocked:{e}"
            attempts = 0
            sinks.append("http")

    sink = "+".join(sinks) if sinks else "none"
    if url and not http_ok:
        status = "pending"
        ok = False
        err = http_err
    else:
        status = "success"
        ok = True
        err = None
        attempts = attempts or 1

    try:
        _record_delivery(
            delivery_id=delivery_id,
            event=event,
            ref_id=str(ref_id),
            interaction_id=interaction_id,
            case_id=case_id,
            investigation_id=investigation_id,
            payload=body,
            sink=sink,
            status=status,
            attempts=attempts,
            error=err,
            outbox_path=outbox_path,
            http_status=http_status,
        )
    except Exception as e:
        logger.warning("connector delivery record failed: %s", e)

    try:
        record_action(
            AgentAction(
                interaction_id=interaction_id or "platform",
                agent="orchestrator",
                action_type="connector_dispatched",
                input_summary=f"event={event}, ref_id={ref_id}",
                output_summary=(
                    f"delivery_id={delivery_id}; status={status}; "
                    f"sink={sink}; attempts={attempts}"
                ),
                evidence_ids=[str(ref_id)] if ref_id else [],
                case_id=case_id,
                ok=ok,
                error=err,
            )
        )
    except Exception as e:
        logger.warning("connector ledger failed: %s", e)

    return {
        "ok": ok,
        "delivery_id": delivery_id,
        "status": status,
        "sink": sink,
        "outbox_path": outbox_path,
        "http_status": http_status,
        "attempts": attempts,
        "error": err,
    }


async def dispatch_case_created(
    *,
    case_id: str,
    interaction_id: str | None = None,
    pack_id: str | None = None,
    investigation_id: str | None = None,
    cluster_id: int | str | None = None,
    severity: str | None = None,
    priority: int | None = None,
    category: str | None = None,
) -> dict[str, Any]:
    return await dispatch_event(
        "case_created",
        case_id=case_id,
        interaction_id=interaction_id,
        pack_id=pack_id,
        investigation_id=investigation_id,
        cluster_id=cluster_id,
        severity=severity,
        priority=priority,
        category=category,
    )


async def dispatch_investigation_opened(
    *,
    investigation_id: str,
    cluster_id: int | str | None = None,
    title: str | None = None,
    case_id: str | None = None,
    interaction_id: str | None = None,
    pack_id: str | None = None,
) -> dict[str, Any]:
    return await dispatch_event(
        "investigation_opened",
        investigation_id=investigation_id,
        cluster_id=cluster_id,
        title=title,
        case_id=case_id,
        interaction_id=interaction_id,
        pack_id=pack_id,
    )


async def replay_delivery(delivery_id: str) -> bool:
    """Re-POST a pending delivery once (no retry cascade). Marks replayed on success.

    Outbox-only pending rows (no URL) re-write outbox and mark replayed if enabled.
    """
    with ops_con(read_only=True) as con:
        row = con.execute(
            """
            SELECT payload_json, status, event, ref_id
            FROM connector_deliveries
            WHERE delivery_id = ?
            """,
            [delivery_id],
        ).fetchone()
    if not row:
        return False
    payload_json, status, _event, _ref = row[0], row[1], row[2], row[3]
    if status not in ("pending", "failed"):
        return False
    try:
        payload = json.loads(payload_json)
    except json.JSONDecodeError:
        return False

    cfg = get_connector_config(include_secret=True)
    url = (cfg.get("webhook_url") or "").strip()
    secret = (cfg.get("shared_secret") or "").strip()
    ok = False
    http_status: int | None = None
    err: str | None = None

    if url:
        ok, http_status, err = await _post_once(url, payload, shared_secret=secret)
    else:
        # Dry-run: re-write outbox file as the replay action.
        try:
            _write_outbox(delivery_id + "_replay", payload)
            ok = True
        except Exception as e:
            err = str(e)

    from src.data.timeutil import utc_now

    now = utc_now()
    if ok:
        with ops_con() as con:
            con.execute(
                """
                UPDATE connector_deliveries
                SET status = 'replayed',
                    last_attempt_at = ?,
                    http_status = COALESCE(?, http_status),
                    attempts = attempts + 1,
                    error = NULL
                WHERE delivery_id = ?
                """,
                [now, http_status, delivery_id],
            )
    else:
        with ops_con() as con:
            con.execute(
                """
                UPDATE connector_deliveries
                SET last_attempt_at = ?,
                    attempts = attempts + 1,
                    error = ?,
                    http_status = COALESCE(?, http_status)
                WHERE delivery_id = ?
                """,
                [
                    now,
                    (err or "")[:1000],
                    http_status,
                    delivery_id,
                ],
            )
    return ok


def delivery_counts() -> dict[str, int]:
    """Simple status histogram for the admin status panel."""
    with ops_con(read_only=True) as con:
        rows = con.execute(
            """
            SELECT status, COUNT(*) AS n
            FROM connector_deliveries
            GROUP BY status
            """
        ).fetchall()
    return {str(r[0]): int(r[1]) for r in rows}


__all__ = [
    "build_payload",
    "build_case_payload_from_db",
    "export_audited_signal",
    "dispatch_event",
    "dispatch_case_created",
    "dispatch_investigation_opened",
    "list_deliveries",
    "replay_delivery",
    "get_connector_config",
    "set_connector_config",
    "reset_connector_config_cache",
    "delivery_counts",
    "OUTBOX_DIR",
    "CONFIG_PATH",
    "CONNECTOR_RETRY_ATTEMPTS",
]
