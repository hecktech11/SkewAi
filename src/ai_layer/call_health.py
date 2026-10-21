"""Meta call health: latency, success/failure and fallback-to-rules per call type.  [AI-LAYER]

One row per Meta Model API call that actually tried to reach the network
(``ai_call_health`` in the ops DB, created on the first recorded call). The
wrapper is installed by an additive ``[AI-LAYER]`` hook at the bottom of
``src/ai_layer/meta_client.py`` around ``chat_json`` / ``sam_segment`` /
``image_generate``; it times the call and re-raises exactly what the wrapped
function raised, so caller behaviour is unchanged.

Definitions (documented, not inferred):

* ``ok``: the adapter returned a validated value.
* ``failed``: the adapter raised (HTTP error, network error / timeout, bad
  envelope, schema-invalid output, daily cap). Every structured caller in
  ``src.ai_layer`` catches that and keeps the deterministic value, so each
  failure is also counted as ``fallback_to_rules``.
* ``caller_fallback_events``: decision-store rows (``ai_layer_events``) whose
  reason starts with ``fallback:``; this also catches a caller-side
  ``asyncio.wait_for`` timeout where the adapter thread finished later.
* A call rejected with ``meta_disabled`` (no key) never left the process and
  is NOT recorded, so with no key every count is 0 and the status reads
  "not run".

Rows hold no customer text and no interaction id (only call type, model id,
latency, outcome and a log-safe reason), so there is nothing to DSR-erase.
"""

from __future__ import annotations

import functools
import statistics
import time
from typing import Any, Callable

from src.data.timeutil import utc_now
from src.data.warehouse import ops_con
from src.ids import new_ulid

CALL_TYPES = (
    "intake", "normalize", "severity", "vision", "segmentation", "brief",
    "narration", "explain", "video", "image", "other",
)

_SCHEMA_TO_TYPE = {
    "intake_slots": "intake",
    "normalize": "normalize",
    "severity": "severity",
    "evidence_vision": "vision",
    "grounded_brief": "brief",
    "vehicle_risk_explain": "explain",
    "video_evidence": "video",
}

#: Reasons that mean "no network call was attempted"; never recorded.
NOT_A_CALL = frozenset({"meta_disabled"})

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS ai_call_health (
        call_id           VARCHAR PRIMARY KEY,
        call_type         VARCHAR NOT NULL,
        model_id          VARCHAR,
        ok                BOOLEAN NOT NULL,
        error_reason      VARCHAR,
        latency_ms        DOUBLE NOT NULL,
        fallback_to_rules BOOLEAN NOT NULL,
        created_at        TIMESTAMP NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ai_call_health_type ON ai_call_health(call_type, created_at)",
]


def ensure_tables(con) -> None:
    for stmt in _DDL:
        con.execute(stmt)


def classify(kind: str, kwargs: dict[str, Any]) -> str:
    """Call type from the adapter entry point and its schema name."""
    if kind == "sam":
        return "segmentation"
    if kind == "image":
        return "image"
    name = str(kwargs.get("schema_name") or "")
    if name in _SCHEMA_TO_TYPE:
        return _SCHEMA_TO_TYPE[name]
    if name.startswith("narration_"):
        return "narration"
    if kwargs.get("images"):
        return "vision"
    return "other"


def _model_for(kind: str, kwargs: dict[str, Any]) -> str:
    try:
        from src.ai_layer import config

        if kwargs.get("model"):
            return str(kwargs["model"])
        if kind == "sam":
            return config.model_sam()
        if kind == "image":
            return config.model_image()
        return config.model_vision() if kwargs.get("images") else config.model_text()
    except Exception:
        return ""


def record_call(*, call_type: str, ok: bool, latency_ms: float, error_reason: str = "",
                model_id: str = "") -> str | None:
    """Best-effort insert; never raises into the caller's path."""
    if not ok and error_reason in NOT_A_CALL:
        return None
    try:
        cid = "aic_" + new_ulid()
        with ops_con() as con:
            ensure_tables(con)
            con.execute(
                "INSERT INTO ai_call_health (call_id, call_type, model_id, ok, error_reason, latency_ms, "
                "fallback_to_rules, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [cid, (call_type or "other")[:32], (model_id or "")[:80], bool(ok), (error_reason or "")[:160],
                 float(round(latency_ms, 2)), (not ok), utc_now()],
            )
        return cid
    except Exception:
        return None


def instrument(fn: Callable[..., Any], *, kind: str) -> Callable[..., Any]:
    """Wrap one adapter entry point. Transparent: same return, same exception."""
    if getattr(fn, "_ai_call_health", False):
        return fn

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t0 = time.perf_counter()
        try:
            out = fn(*args, **kwargs)
        except Exception as e:
            reason = str(getattr(e, "reason", "") or type(e).__name__)
            record_call(call_type=classify(kind, kwargs), ok=False, error_reason=reason,
                        latency_ms=(time.perf_counter() - t0) * 1000.0, model_id=_model_for(kind, kwargs))
            raise
        record_call(call_type=classify(kind, kwargs), ok=True,
                    latency_ms=(time.perf_counter() - t0) * 1000.0, model_id=_model_for(kind, kwargs))
        return out

    wrapper._ai_call_health = True  # type: ignore[attr-defined]
    return wrapper


def _relation_exists(con, name: str) -> bool:
    return con.execute("SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]).fetchone() is not None


def _pct(vals: list[float], q: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return round(s[k], 1)


def summary() -> dict[str, Any]:
    """Per call type: calls, ok, failed, fallback_to_rules, latency (ms)."""
    rows: list[tuple] = []
    caller_fallbacks: dict[str, int] = {}
    try:
        with ops_con() as con:
            if _relation_exists(con, "ai_call_health"):
                rows = con.execute(
                    "SELECT call_type, ok, latency_ms, fallback_to_rules, error_reason, created_at "
                    "FROM ai_call_health ORDER BY created_at"
                ).fetchall()
            if _relation_exists(con, "ai_layer_events"):
                for kind, n in con.execute(
                    "SELECT kind, COUNT(*) FROM ai_layer_events WHERE reason LIKE 'fallback:%' GROUP BY kind"
                ).fetchall():
                    caller_fallbacks[str(kind)] = int(n)
    except Exception as e:  # unreadable DB -> say so, no numbers
        return {"status": "unavailable", "error": type(e).__name__, "by_type": {}, "totals": None}
    by: dict[str, dict[str, Any]] = {}
    for ctype, ok, lat, fb, reason, created in rows:
        b = by.setdefault(str(ctype), {"calls": 0, "ok": 0, "failed": 0, "fallback_to_rules": 0,
                                       "timeouts": 0, "_lat": [], "last_error": None, "last_call_at": None})
        b["calls"] += 1
        b["ok"] += int(bool(ok))
        b["failed"] += int(not ok)
        b["fallback_to_rules"] += int(bool(fb))
        if reason and "timeout" in str(reason).lower():
            b["timeouts"] += 1
        if not ok and reason:
            b["last_error"] = str(reason)
        b["_lat"].append(float(lat))
        b["last_call_at"] = str(created)
    out: dict[str, Any] = {}
    for ctype in CALL_TYPES:
        b = by.pop(ctype, None)
        out[ctype] = _finish(b)
    for ctype, b in by.items():  # unknown types still shown
        out[ctype] = _finish(b)
    total_calls = sum(v["calls"] for v in out.values())
    totals = {
        "calls": total_calls,
        "ok": sum(v["ok"] for v in out.values()),
        "failed": sum(v["failed"] for v in out.values()),
        "fallback_to_rules": sum(v["fallback_to_rules"] for v in out.values()),
        "caller_fallback_events": sum(caller_fallbacks.values()),
    }
    return {
        "status": "not run" if total_calls == 0 else "recorded",
        "note": ("no Meta call has been made (no key configured, or the AI layer was never exercised)"
                 if total_calls == 0 else "counts since the ops DB was created"),
        "by_type": out,
        "totals": totals,
        "caller_fallback_events_by_kind": caller_fallbacks,
        "definitions": {
            "failed": "adapter raised (HTTP/network/timeout/bad envelope/schema-invalid/daily cap)",
            "fallback_to_rules": "failed calls; every caller keeps the deterministic value on failure",
            "caller_fallback_events": "ai_layer_events rows with reason 'fallback:*' (includes caller-side timeouts)",
        },
    }


def _finish(b: dict[str, Any] | None) -> dict[str, Any]:
    if not b or not b["calls"]:
        return {"calls": 0, "status": "not run", "ok": 0, "failed": 0, "fallback_to_rules": 0, "timeouts": 0,
                "success_rate": None, "latency_ms": None, "last_error": None, "last_call_at": None}
    lat = b.pop("_lat")
    return {
        **b,
        "status": "recorded",
        "success_rate": round(b["ok"] / b["calls"], 3),
        "latency_ms": {"mean": round(statistics.mean(lat), 1), "p50": _pct(lat, 0.5), "p95": _pct(lat, 0.95),
                       "max": round(max(lat), 1), "n": len(lat)},
    }


__all__ = ["CALL_TYPES", "classify", "ensure_tables", "instrument", "record_call", "summary"]
