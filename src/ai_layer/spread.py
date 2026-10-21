"""Fault spread by region and over time.  [AI-LAYER]

Pure SQL aggregation over the pack's ``records`` table: complaint counts per
region (NHTSA ``state``) and per month (``received_at``), plus when each region
first reported the fault. Counts are raw: they are NOT normalised for
population or fleet size (no such data is in the warehouse).
"""

from __future__ import annotations

from typing import Any


def _where(make, model, component, year_min, year_max):
    w, p = ["region IS NOT NULL", "received_at IS NOT NULL"], []
    if make:
        w.append("upper(entity_2) = ?"); p.append(make.strip().upper())
    if model:
        w.append("upper(entity_3) = ?"); p.append(model.strip().upper())
    if component:
        w.append("upper(category) LIKE ?"); p.append(f"%{component.strip().upper()}%")
    if year_min is not None:
        w.append("TRY_CAST(entity_1 AS INTEGER) >= ?"); p.append(int(year_min))
    if year_max is not None:
        w.append("TRY_CAST(entity_1 AS INTEGER) <= ?"); p.append(int(year_max))
    return " AND ".join(w), p


def fault_spread(make: str | None = None, model: str | None = None, component: str | None = None,
                 year_min: int | None = None, year_max: int | None = None, *, pack_id: str | None = None,
                 top_regions: int = 15, months: int = 60) -> dict[str, Any]:
    from src.data.warehouse import domain_con

    if not pack_id:
        from src.domains.active_pack import resolve_active_pack_id
        pack_id = resolve_active_pack_id()
    where, p = _where(make, model, component, year_min, year_max)
    with domain_con(pack_id, read_only=True) as con:
        total = con.execute(f"SELECT count(*) FROM records WHERE {where}", p).fetchone()[0]
        regions = con.execute(
            f"SELECT region, count(*) n, min(received_at) first_seen FROM records WHERE {where} GROUP BY 1 ORDER BY 2 DESC, 1",
            p).fetchall()
        monthly = con.execute(
            f"""SELECT strftime(date_trunc('month', received_at), '%Y-%m') m, count(*) n, count(DISTINCT region) r
                FROM records WHERE {where} GROUP BY 1 ORDER BY 1 DESC LIMIT ?""", p + [int(months)]).fetchall()
        heat = con.execute(
            f"""SELECT region, strftime(date_trunc('year', received_at), '%Y') y, count(*) n
                FROM records WHERE {where} GROUP BY 1, 2""", p).fetchall()
    monthly = list(reversed(monthly))
    top = [r[0] for r in regions[: max(1, int(top_regions))]]
    years = sorted({y for _, y, _ in heat})
    grid = {reg: {y: 0 for y in years} for reg in top}
    for reg, y, n in heat:
        if reg in grid:
            grid[reg][y] = int(n)
    first = sorted(((r[0], r[2]) for r in regions), key=lambda x: x[1])
    return {
        "filters": {"make": make, "model": model, "component": component, "year_min": year_min, "year_max": year_max},
        "pack_id": pack_id, "total_complaints": int(total), "regions_reporting": len(regions),
        "by_region": [{"region": r[0], "count": int(r[1]), "first_seen": r[2].date().isoformat()} for r in regions[: int(top_regions)]],
        "other_regions_count": int(sum(r[1] for r in regions[int(top_regions):])),
        "by_month": [{"month": m, "count": int(n), "regions": int(r)} for m, n, r in monthly],
        "region_year_heat": {"years": years, "rows": [{"region": reg, "counts": [grid[reg][y] for y in years]} for reg in top]},
        "first_seen_order": [{"region": reg, "first_seen": d.date().isoformat()} for reg, d in first[:20]],
        "note": "Raw complaint counts by NHTSA reporting state and received month; not normalised for population or fleet size.",
    }


__all__ = ["fault_spread"]
