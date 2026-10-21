"""Manufacturer safety-alert feed.  [AI-LAYER]

When an open investigation crosses a high-risk threshold, an alert is created
that carries the grounded brief (claims cite real record / advisory IDs and
pass the existing checks) and the recommended inspection / repair actions.

Thresholds (env):
  AI_ALERT_MIN_CASES       default 3   cases needed for a "high" safety-risk slice
  AI_ALERT_ELEVATED_CASES  default 10  cases needed for an "elevated" slice

Delivery: NOTHING is sent anywhere. Alerts are listed at GET /api/ai/alerts,
and published as a pull feed (JSON Feed 1.1) at GET /api/ai/alerts/feed.json.
Each alert also carries ``webhook_payload``, the exact body a manufacturer
webhook would receive, for integration testing. It is marked "not sent".
Alerts are deduplicated per investigation and level, and a level only ever
escalates.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

_DDL = """
CREATE TABLE IF NOT EXISTS ai_alerts (
    alert_id         VARCHAR PRIMARY KEY,
    investigation_id VARCHAR NOT NULL,
    pack_id          VARCHAR,
    created_at       TIMESTAMP NOT NULL,
    level            VARCHAR NOT NULL,
    case_count       INTEGER,
    title            VARCHAR,
    payload          JSON,
    status           VARCHAR NOT NULL DEFAULT 'open',
    acked_at         TIMESTAMP
)
"""
_RANK = {"normal": 0, "elevated": 1, "high": 2}


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def thresholds() -> dict[str, int]:
    return {"high_min_cases": _int_env("AI_ALERT_MIN_CASES", 3),
            "elevated_min_cases": _int_env("AI_ALERT_ELEVATED_CASES", 10)}


def qualifies(level: str, case_count: int, th: dict[str, int] | None = None) -> bool:
    th = th or thresholds()
    if level == "high":
        return case_count >= th["high_min_cases"]
    if level == "elevated":
        return case_count >= th["elevated_min_cases"]
    return False


def _new_id() -> str:
    from src.ids import new_ulid  # existing ULID helper

    return "alrt_" + new_ulid()


def _existing(con) -> dict[str, str]:
    con.execute(_DDL)
    rows = con.execute("SELECT investigation_id, level FROM ai_alerts").fetchall()
    best: dict[str, str] = {}
    for inv, lvl in rows:
        if _RANK.get(lvl, 0) > _RANK.get(best.get(inv, "normal"), 0):
            best[inv] = lvl
    return best


def build_payload(inv: dict[str, Any], brief: dict[str, Any] | None) -> dict[str, Any]:
    b = brief or {}
    return {
        "event": "skewai.safety_alert",
        "delivery": "not sent: pull feed / preview only (no external sending configured)",
        "investigation_id": inv["investigation_id"],
        "title": inv.get("title"),
        "slice": {"category": inv.get("slice_category"), "make": inv.get("slice_entity_2"), "model": inv.get("slice_entity_3")},
        "case_count": inv.get("case_count"),
        "safety_risk": inv.get("safety_risk"),
        "headline": b.get("headline"),
        "claims": b.get("claims") or [],
        "recommended_actions": b.get("recommended_actions") or [],
        "brief_mode": b.get("mode"),
        "powered_by": b.get("powered_by") or [],
        "links": {"issue": f"#issue/{inv['investigation_id']}", "brief": f"/api/ai/investigations/{inv['investigation_id']}/grounded-brief"},
    }


def scan_alerts(limit: int = 200) -> dict[str, Any]:
    from src.ai_layer.grounded_brief import build_grounded_brief
    from src.ai_layer.safety_board import safety_board
    from src.data.warehouse import ops_con

    th = thresholds()
    board = safety_board(limit)
    with ops_con(read_only=False) as con:
        have = _existing(con)
    created = []
    for inv in board.get("investigations") or []:
        level = (inv.get("safety_risk") or {}).get("level", "normal")
        cases = int(inv.get("case_count") or 0)
        if not qualifies(level, cases, th):
            continue
        if _RANK[level] <= _RANK.get(have.get(inv["investigation_id"], "normal"), 0) and inv["investigation_id"] in have:
            continue
        try:
            brief = build_grounded_brief(inv["investigation_id"])
        except Exception:
            brief = None
        payload = build_payload(inv, brief)
        # [AI-LAYER] fault codes: corroborating-code case count (evidence only; thresholds unchanged)
        try:
            from src.ai_layer.fault_codes import matching_for_investigation

            payload["cases_with_matching_fault_code"] = matching_for_investigation(inv["investigation_id"])
        except Exception:
            pass
        row = {"alert_id": _new_id(), "investigation_id": inv["investigation_id"], "pack_id": inv.get("pack_id"),
               "created_at": datetime.now(timezone.utc).replace(tzinfo=None), "level": level, "case_count": cases,
               "title": inv.get("title"), "payload": payload}
        with ops_con(read_only=False) as con:
            con.execute(_DDL)
            con.execute("INSERT INTO ai_alerts (alert_id, investigation_id, pack_id, created_at, level, case_count, title, payload) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        [row["alert_id"], row["investigation_id"], row["pack_id"], row["created_at"], level, cases,
                         row["title"], json.dumps(payload, default=str)])
        have[inv["investigation_id"]] = level
        created.append(row["alert_id"])
    return {"created": created, "thresholds": th, "scanned": len(board.get("investigations") or [])}


def list_alerts(status: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    from src.data.warehouse import ops_con

    with ops_con(read_only=False) as con:
        con.execute(_DDL)
        q = "SELECT alert_id, investigation_id, pack_id, created_at, level, case_count, title, payload, status, acked_at FROM ai_alerts"
        args: list[Any] = []
        if status:
            q += " WHERE status = ?"
            args.append(status)
        q += " ORDER BY created_at DESC LIMIT ?"
        cur = con.execute(q, args + [int(limit)])
        cols = [d[0] for d in cur.description]
        out = []
        for r in cur.fetchall():
            d = dict(zip(cols, r))
            d["payload"] = json.loads(d["payload"]) if isinstance(d["payload"], str) else d["payload"]
            for k in ("created_at", "acked_at"):
                if d.get(k) is not None and hasattr(d[k], "isoformat"):
                    d[k] = d[k].isoformat()
            out.append(d)
        return out


def ack_alert(alert_id: str) -> bool:
    from src.data.warehouse import ops_con

    with ops_con(read_only=False) as con:
        con.execute(_DDL)
        n = con.execute("SELECT COUNT(*) FROM ai_alerts WHERE alert_id = ?", [alert_id]).fetchone()[0]
        if n:
            con.execute("UPDATE ai_alerts SET status='acknowledged', acked_at=? WHERE alert_id = ?",
                        [datetime.now(timezone.utc).replace(tzinfo=None), alert_id])
        return bool(n)


def json_feed(base_url: str = "") -> dict[str, Any]:
    items = []
    for a in list_alerts(limit=100):
        p = a["payload"] or {}
        items.append({
            "id": a["alert_id"], "title": f"[{a['level'].upper()}] {a.get('title') or a['investigation_id']}",
            "content_text": p.get("headline") or "", "date_published": a["created_at"],
            "url": f"{base_url}/ui/#issue/{a['investigation_id']}", "_skewai": p,
        })
    return {"version": "https://jsonfeed.org/version/1.1", "title": "SkewAi manufacturer safety alerts",
            "description": "Pull feed. Nothing is pushed externally.", "items": items}


__all__ = ["ack_alert", "build_payload", "json_feed", "list_alerts", "qualifies", "scan_alerts", "thresholds"]
