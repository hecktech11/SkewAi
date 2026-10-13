"""Product brief for one investigation, keyed by its slice.

The Investigator already retrieves similar records, association lift, and
cross-source links. This module runs those same queries for the slice the
fleet scan stored (category, make, model) so an engineer can open the case
without a live call.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.backtest.engine import _parse_ts, _weeks_between
from src.data.warehouse import domain_con, ops_con
from src.ml_runtime.anomalies import _week_monday

_SOURCE_ORDER = ("NHTSA", "FRONTLINE", "WARRANTY", "SERVICE", "FIELD", "TELEMATICS")
_POOL_LIMIT = 80


def _norm(val: Any) -> str:
    return str(val or "").strip().upper()


def _iso(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.isoformat()
    return str(val)


def _num(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _load_investigation(investigation_id: str) -> dict[str, Any]:
    with ops_con(read_only=True) as con:
        cols = {
            str(r[0])
            for r in con.execute(
                "SELECT name FROM pragma_table_info('investigations')"
            ).fetchall()
        }
        want = [
            "investigation_id",
            "pack_id",
            "cluster_id",
            "title",
            "status",
            "slice_category",
            "slice_entity_2",
            "slice_entity_3",
        ]
        sel = [c for c in want if c in cols]
        cur = con.execute(
            f"SELECT {', '.join(sel)} FROM investigations WHERE investigation_id = ?",
            [investigation_id],
        )
        row = cur.fetchone()
        if not row:
            raise LookupError(f"investigation not found: {investigation_id}")
        names = [d[0] for d in cur.description]
        inv = dict(zip(names, row))
    inv.setdefault("slice_category", None)
    inv.setdefault("slice_entity_2", None)
    inv.setdefault("slice_entity_3", None)
    return inv


def _slice_from_cluster(pack_id: str, cluster_id: int) -> tuple[str, str, str]:
    """Category from the cluster, make and model from its modal member."""
    with domain_con(pack_id, read_only=True) as con:
        cat_row = con.execute(
            "SELECT category FROM clusters WHERE cluster_id = ?",
            [int(cluster_id)],
        ).fetchone()
        mode = con.execute(
            """
            SELECT r.entity_2, r.entity_3, COUNT(*) AS n
            FROM cluster_assignments a
            JOIN records r ON r.record_id = a.record_id
            WHERE a.cluster_id = ?
            GROUP BY r.entity_2, r.entity_3
            ORDER BY n DESC
            LIMIT 1
            """,
            [int(cluster_id)],
        ).fetchone()
    category = str(cat_row[0] or "") if cat_row else ""
    entity_2 = str(mode[0] or "") if mode else ""
    entity_3 = str(mode[1] or "") if mode else ""
    return category, entity_2, entity_3


def resolve_slice(inv: dict[str, Any]) -> tuple[str, str, str]:
    category = str(inv.get("slice_category") or "").strip()
    entity_2 = str(inv.get("slice_entity_2") or "").strip()
    entity_3 = str(inv.get("slice_entity_3") or "").strip()
    if category or entity_2:
        return category, entity_2, entity_3
    try:
        cid = int(inv.get("cluster_id") or 0)
    except (TypeError, ValueError):
        cid = 0
    pack_id = str(inv.get("pack_id") or "")
    if cid > 0 and pack_id:
        try:
            return _slice_from_cluster(pack_id, cid)
        except Exception:
            return "", "", ""
    return "", "", ""


def _same_scope(
    category: str,
    entity_2: str,
    entity_3: str,
    advisory: dict[str, Any],
) -> bool:
    if _norm(category) != _norm(advisory.get("scope_category")):
        return False
    if _norm(entity_2) != _norm(advisory.get("scope_entity_2")):
        return False
    scope_model = _norm(advisory.get("scope_entity_3"))
    row_model = _norm(entity_3)
    if scope_model and row_model and scope_model != row_model:
        return False
    return True


def _series_match(row: dict[str, Any], category: str, entity_2: str, entity_3: str) -> bool:
    if _norm(row.get("category")) != _norm(category):
        return False
    if _norm(row.get("entity_2")) != _norm(entity_2):
        return False
    model = _norm(entity_3)
    row_model = _norm(row.get("entity_3"))
    if model and row_model and model != row_model:
        return False
    if model and row_model and model == row_model:
        return True
    if not model and row_model:
        return False
    return True


def _first_flag_lead(
    series: list[dict[str, Any]],
    advisory: dict[str, Any],
    *,
    category: str,
    entity_2: str,
    entity_3: str,
) -> int | None:
    issued = _parse_ts(advisory.get("issued_at"))
    if issued is None:
        return None
    if not _same_scope(category, entity_2, entity_3, advisory):
        return None
    first: datetime | None = None
    for row in series:
        if not row.get("is_anomaly") or row.get("censored"):
            continue
        if not _series_match(row, category, entity_2, entity_3):
            continue
        start = _week_monday(str(row.get("iso_week") or ""))
        if start is None or not start < issued:
            continue
        if first is None or start < first:
            first = start
    if first is None:
        return None
    lead = _weeks_between(first, issued)
    return lead if lead > 0 else None


def _load_advisories(con) -> list[dict[str, Any]]:
    try:
        rows = con.execute(
            """
            SELECT advisory_id, issued_at, scope_category, scope_entity_2,
                   scope_entity_3, summary, remedy, url
            FROM advisories
            """
        ).fetchall()
    except Exception:
        return []
    return [
        {
            "advisory_id": r[0],
            "issued_at": r[1],
            "scope_category": r[2],
            "scope_entity_2": r[3],
            "scope_entity_3": r[4],
            "summary": r[5],
            "remedy": r[6],
            "url": r[7],
        }
        for r in rows
    ]


def _load_series(con, pack_id: str, category: str, entity_2: str, entity_3: str) -> list[dict[str, Any]]:
    info = {
        str(r[0])
        for r in con.execute(
            "SELECT name FROM pragma_table_info('weekly_anomalies')"
        ).fetchall()
    }
    if "iso_week" not in info:
        return []
    censored = "COALESCE(censored, FALSE)" if "censored" in info else "FALSE"
    model_col = "COALESCE(entity_3, '')" if "entity_3" in info else "''"
    model = entity_3 or ""
    rows = con.execute(
        f"""
        SELECT iso_week, category, entity_2, {model_col}, record_count,
               baseline_mean, z_score, is_anomaly, {censored}
        FROM weekly_anomalies
        WHERE pack_id = ?
          AND category = ?
          AND entity_2 = ?
          AND (
                (? = '' AND {model_col} = '')
             OR (? <> '' AND ({model_col} = ? OR {model_col} = ''))
          )
        ORDER BY iso_week
        """,
        [pack_id, category, entity_2, model, model, model],
    ).fetchall()
    out = []
    for iso_week, cat, ent, model_v, count, baseline, z, anomaly, held in rows:
        out.append(
            {
                "iso_week": iso_week,
                "category": cat,
                "entity_2": ent,
                "entity_3": model_v or "",
                "record_count": int(count or 0),
                "baseline_mean": _num(baseline),
                "z_score": _num(z),
                "is_anomaly": bool(anomaly),
                "censored": bool(held),
            }
        )
    return out


def _public_recall(advisory: dict[str, Any], lead: int | None) -> dict[str, Any]:
    return {
        "advisory_id": advisory.get("advisory_id"),
        "issued_at": _iso(advisory.get("issued_at")),
        "summary": advisory.get("summary"),
        "remedy": advisory.get("remedy"),
        "url": advisory.get("url"),
        "scope_category": advisory.get("scope_category"),
        "scope_entity_2": advisory.get("scope_entity_2"),
        "scope_entity_3": advisory.get("scope_entity_3"),
        "lead_time_weeks": lead,
    }


def matched_recall(
    con,
    *,
    pack_id: str,
    category: str,
    entity_2: str,
    entity_3: str,
    cluster_id: int = 0,
    series: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """The recall this slice rose ahead of, with its lead in weeks.

    A matched cluster backtest wins, because that is the Investigator's
    lead query. Otherwise the first flagged week on this slice is compared
    with each advisory NHTSA had not yet published.
    """
    from src.agents.investigator import _BACKTEST_SQL

    advisories = _load_advisories(con)
    by_id = {str(a["advisory_id"]): a for a in advisories}
    if cluster_id > 0:
        try:
            bt = con.execute(_BACKTEST_SQL, [int(cluster_id)]).fetchone()
        except Exception:
            bt = None
        if bt and bt[1] is not None:
            lead = int(bt[2]) if bt[2] is not None else None
            advisory = by_id.get(str(bt[1]))
            if advisory is None:
                advisory = {
                    "advisory_id": bt[1],
                    "issued_at": None,
                    "summary": None,
                    "remedy": None,
                    "url": None,
                }
            if lead is not None and lead > 0:
                return _public_recall(advisory, lead)
    weeks = series if series is not None else _load_series(
        con, pack_id, category, entity_2, entity_3
    )
    best: dict[str, Any] | None = None
    best_lead = 0
    for advisory in advisories:
        lead = _first_flag_lead(
            weeks,
            advisory,
            category=category,
            entity_2=entity_2,
            entity_3=entity_3,
        )
        if lead is None or lead <= best_lead:
            continue
        best_lead = lead
        best = _public_recall(advisory, lead)
    return best


def lead_weeks_for_slice(
    pack_id: str,
    category: str | None,
    entity_2: str | None,
    entity_3: str | None,
    *,
    cluster_id: int = 0,
    series: list[dict[str, Any]] | None = None,
) -> int | None:
    """Lead weeks to pass into the early-warning alert. None when no recall matches."""
    try:
        with domain_con(pack_id, read_only=True) as con:
            recall = matched_recall(
                con,
                pack_id=pack_id,
                category=category or "",
                entity_2=entity_2 or "",
                entity_3=entity_3 or "",
                cluster_id=int(cluster_id or 0),
                series=series,
            )
    except Exception:
        return None
    if not recall:
        return None
    lead = recall.get("lead_time_weeks")
    try:
        return int(lead) if lead is not None and int(lead) > 0 else None
    except (TypeError, ValueError):
        return None


def _public_record(row: dict[str, Any]) -> dict[str, Any]:
    source = _norm(row.get("source")) or "UNKNOWN"
    text = str(row.get("text") or "")
    return {
        "record_id": row.get("record_id"),
        "source": source,
        "tag": source,
        "category": row.get("category"),
        "entity_1": row.get("entity_1"),
        "entity_2": row.get("entity_2"),
        "entity_3": row.get("entity_3"),
        "entity_key": row.get("entity_key"),
        "received_at": _iso(row.get("received_at")),
        "text": text[:400],
        "assoc_score": row.get("assoc_score"),
    }


def _association_lift(population, category: str, entity_2: str) -> float:
    from src.ml_runtime.association import population_support

    support = population_support(population, category=category, entity_2=entity_2)
    p_cat = float(support.get("p_cat") or 0)
    p_ent = float(support.get("p_ent") or 0)
    if p_cat <= 0 or p_ent <= 0:
        return 0.0
    return float(support.get("support") or 0) / (p_cat * p_ent)


def build_investigation_brief(investigation_id: str) -> dict[str, Any]:
    """Weekly series, per-source records, links, lift, and the matched recall."""
    from src.agents.investigator import _SIMILAR_SQL, _cross_source_links
    from src.ml_runtime.association import (
        get_pack_population_index,
        rank_by_association,
    )

    inv = _load_investigation(investigation_id)
    pack_id = str(inv.get("pack_id") or "")
    try:
        cluster_id = int(inv.get("cluster_id") or 0)
    except (TypeError, ValueError):
        cluster_id = 0
    category, entity_2, entity_3 = resolve_slice(inv)
    empty_sources = {name: [] for name in _SOURCE_ORDER}
    brief: dict[str, Any] = {
        "investigation_id": investigation_id,
        "pack_id": pack_id,
        "cluster_id": cluster_id,
        "slice": {
            "category": category or None,
            "entity_2": entity_2 or None,
            "entity_3": entity_3 or None,
        },
        "weekly": [],
        "flagged_weeks": [],
        "records_by_source": empty_sources,
        "similar_records": [],
        "cross_source_links": [],
        "association_lift": 0.0,
        "recall": None,
    }
    if not pack_id or not (category or entity_2):
        return brief
    model = entity_3 or None
    try:
        with domain_con(pack_id, read_only=True) as con:
            weekly = _load_series(con, pack_id, category, entity_2, entity_3)
            recall = matched_recall(
                con,
                pack_id=pack_id,
                category=category,
                entity_2=entity_2,
                entity_3=entity_3,
                cluster_id=cluster_id,
                series=weekly,
            )
            population = get_pack_population_index(con, pack_id=pack_id)
            pool: list[dict[str, Any]] = []
            if category or entity_2:
                cur = con.execute(
                    _SIMILAR_SQL,
                    [
                        category or None,
                        category or None,
                        entity_2 or None,
                        entity_2 or None,
                        model,
                        model,
                        None,
                        None,
                        _POOL_LIMIT,
                    ],
                )
                cols = [d[0] for d in cur.description]
                pool = [dict(zip(cols, r)) for r in cur.fetchall()]
    except FileNotFoundError:
        return brief
    query = {"category": category, "entity_2": entity_2, "entity_3": entity_3}
    ranked = rank_by_association(query, pool, population, top_k=5) if pool else []
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in _SOURCE_ORDER}
    for row in pool:
        public = _public_record(row)
        src = public["source"]
        grouped.setdefault(src, []).append(public)
    for src in list(grouped):
        grouped[src] = grouped[src][:25]
    brief["weekly"] = [
        {
            "iso_week": w["iso_week"],
            "record_count": w["record_count"],
            "baseline_mean": w["baseline_mean"],
            "z_score": w["z_score"],
            "is_anomaly": w["is_anomaly"],
            "censored": w["censored"],
        }
        for w in weekly
    ]
    brief["flagged_weeks"] = [w for w in brief["weekly"] if w["is_anomaly"]]
    brief["records_by_source"] = grouped
    brief["similar_records"] = [_public_record(r) for r in ranked]
    brief["cross_source_links"] = _cross_source_links(pool)
    brief["association_lift"] = _association_lift(population, category, entity_2)
    brief["recall"] = recall
    return brief


__all__ = [
    "build_investigation_brief",
    "lead_weeks_for_slice",
    "matched_recall",
    "resolve_slice",
]
