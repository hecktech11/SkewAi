"""Safety-prioritised early-warning board.  [AI-LAYER]

Lists open / monitoring investigations from the ops warehouse, attaches the
deterministic safety-risk indicator for each slice (``src.ai_layer.safety``)
and orders them high -> elevated -> normal, then by case count. It reads the
same tables the existing early-warning board reads and changes nothing.

Also exposes the committed backtest headline (``reports/nhtsa/backtest_safety.json``)
for the demo "hero moment"; when that file is absent the field is null rather
than a made-up number.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.ai_layer.safety import TIER_ORDER, slice_safety

BACKTEST_JSON = Path(__file__).resolve().parents[2] / "reports" / "nhtsa" / "backtest_safety.json"


def _investigations(limit: int) -> list[dict[str, Any]]:
    from src.data.warehouse import ops_con

    with ops_con(read_only=True) as con:
        cur = con.execute(
            """
            SELECT investigation_id, pack_id, cluster_id, title, status, opened_at, last_case_at,
                   case_count, slice_category, slice_entity_2, slice_entity_3
            FROM investigations WHERE status IN ('open', 'monitoring')
            ORDER BY case_count DESC, opened_at DESC LIMIT ?
            """,
            [int(limit)],
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def backtest_headline() -> dict[str, Any] | None:
    try:
        data = json.loads(BACKTEST_JSON.read_text())
    except (OSError, ValueError):
        return None
    return data.get("headline") if isinstance(data, dict) else None


def safety_board(limit: int = 50) -> dict[str, Any]:
    try:
        invs = _investigations(limit)
    except Exception:
        invs = []
    rows = []
    for inv in invs:
        risk = slice_safety(str(inv["pack_id"]), inv.get("slice_category"), inv.get("slice_entity_2"),
                            inv.get("slice_entity_3"), limit=500)
        rows.append({
            **{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in inv.items()},
            "safety_risk": risk,
        })
    rows.sort(key=lambda r: (TIER_ORDER.get(r["safety_risk"]["level"], 9), -int(r.get("case_count") or 0)))
    return {
        "investigations": rows,
        "count": len(rows),
        "counts_by_level": {lvl: sum(1 for r in rows if r["safety_risk"]["level"] == lvl) for lvl in TIER_ORDER},
        "backtest_headline": backtest_headline(),
        "method": "component tier + keyword scan of real complaint narratives; ordering heuristic, not a crash prediction",
    }


__all__ = ["backtest_headline", "safety_board"]
