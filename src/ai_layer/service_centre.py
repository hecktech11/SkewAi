"""Service-centre inspection call-in queue.  [AI-LAYER]

Ranks open cases (one per recorded vehicle complaint) by a transparent,
deterministic score so a workshop can decide whom to call in first. Every
point is explained in ``reasons``; nothing is AI-generated.

  severity            Critical 60 / Medium 30 / Low 10   (existing triage)
  safety flag raised  +30                                  (existing kill-switch)
  component tier      critical +20 / elevated +10          (src.ai_layer.safety)
  recall advisory     +15                                  (existing advisory match)
  part of an investigation (cluster) +10
  waiting time        +1 per day open, max +10
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from src.ai_layer.safety import component_tier
from src.ai_layer.vehicle_risk import _ADVICE, _DEFAULT_ADVICE

_SEV = {"critical": 60, "medium": 30, "low": 10}


def score_case(c: dict[str, Any], now: datetime | None = None) -> tuple[int, list[str]]:
    now = now or datetime.now(timezone.utc).replace(tzinfo=None)
    reasons: list[str] = []
    sev = str(c.get("severity") or "").lower()
    pts = _SEV.get(sev, 0)
    reasons.append(f"severity {c.get('severity') or 'unknown'} (+{pts})")
    flags = c.get("safety_flags")
    if isinstance(flags, str):
        try:
            flags = json.loads(flags)
        except ValueError:
            flags = {}
    raised = sorted(k for k, v in (flags or {}).items() if v and k != "escalation")
    if raised or (flags or {}).get("escalation"):
        pts += 30
        reasons.append(f"safety flag: {', '.join(raised) or 'escalation'} (+30)")
    tier = component_tier(c.get("category"))
    if tier == "critical":
        pts += 20
        reasons.append(f"safety-critical component {c.get('category')} (+20)")
    elif tier == "elevated":
        pts += 10
        reasons.append(f"elevated-risk component {c.get('category')} (+10)")
    if c.get("advisory_match_id"):
        pts += 15
        reasons.append(f"matches recall advisory {c['advisory_match_id']} (+15)")
    if c.get("investigation_id"):
        pts += 10
        reasons.append(f"part of investigation {c['investigation_id']} (+10)")
    created = c.get("created_at")
    if isinstance(created, datetime):
        days = max(0, int((now - created).total_seconds() // 86400))
        if days:
            add = min(10, days)
            pts += add
            reasons.append(f"waiting {days} day(s) (+{add})")
    return pts, reasons


def call_in_action(c: dict[str, Any]) -> str:
    cat = str(c.get("category") or "").upper()
    base = _ADVICE.get(cat, _DEFAULT_ADVICE)
    if c.get("advisory_match_id"):
        return f"Call in: check recall {c['advisory_match_id']} coverage for this VIN. {base}"
    return f"Call in for inspection. {base}" if cat else "Call back to complete the vehicle details."


def service_queue(limit: int = 50, include_simulated: bool = True) -> dict[str, Any]:
    from src.data.warehouse import ops_con

    kinds = ("customer", "simulated") if include_simulated else ("customer",)
    with ops_con(read_only=True) as con:
        cur = con.execute(
            f"""
            SELECT c.case_id, c.created_at, c.category, c.severity, c.priority, c.safety_flags, c.advisory_match_id,
                   c.investigation_id, c.status, c.case_kind, i.entity_1, i.entity_2, i.entity_3, i.channel
            FROM cases c LEFT JOIN interactions i ON i.interaction_id = c.interaction_id
            WHERE c.status IN ('open', 'pending_followup') AND c.case_kind IN ({','.join('?' * len(kinds))})
            ORDER BY c.created_at DESC LIMIT 2000
            """,
            list(kinds),
        )
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    out = []
    for r in rows:
        pts, reasons = score_case(r, now)
        out.append({
            "case_id": r["case_id"], "score": pts, "reasons": reasons, "action": call_in_action(r),
            "vehicle": " ".join(str(x) for x in (r.get("entity_1"), r.get("entity_2"), r.get("entity_3")) if x) or None,
            "category": r.get("category"), "severity": r.get("severity"), "advisory_match_id": r.get("advisory_match_id"),
            "investigation_id": r.get("investigation_id"), "case_kind": r.get("case_kind"), "channel": r.get("channel"),
            "created_at": r["created_at"].isoformat() if hasattr(r.get("created_at"), "isoformat") else r.get("created_at"),
        })
    out.sort(key=lambda x: (-x["score"], x["created_at"] or ""))
    return {"queue": out[: max(1, int(limit))], "open_cases": len(out),
            "method": "deterministic points from existing triage, safety flags, component tier, recall match and wait time"}


__all__ = ["call_in_action", "score_case", "service_queue"]
