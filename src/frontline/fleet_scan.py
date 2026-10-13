"""Scheduled fleet scan (Axion slice, Phase 3).

Contact-triggered intercepts only see slices a customer just touched. This
scan runs on a CADENCE (cron / compose scheduler / job queue) over the whole
pack so the reliability engineer learns about a spike nobody called about
yet — the Axion demo moment:

    1. recompute weekly anomalies for the full pack (observed rows only);
    2. optionally rebuild clusters (heavy; default off, nightly cadence);
    3. refresh the backtest lead-time table;
    4. for every anomalous (category, entity_2) slice, resolve the top
       cluster, or open the slice itself when no cluster matches;
    5. fire early-warning alerts for newly opened investigations,
       including the recall lead time when one was measured.

Nothing is fabricated: investigations open only on computed anomalies with
real record counts, and the report lists exactly what was evaluated.
"""

from __future__ import annotations

from typing import Any

from src.data.warehouse import domain_con


def _top_cluster_for_slice(
    pack_id: str, category: str | None, entity_2: str | None
) -> dict[str, Any] | None:
    """Best cluster for a slice: fused relevance without a contact.

    Same scoring philosophy as the investigator (category fit + entity
    overlap + size as tie-break only) so scan-opened investigations agree
    with contact-opened ones.
    """
    with domain_con(pack_id) as con:
        try:
            rows = con.execute(
                """
                SELECT cluster_id, top_terms, category, record_count
                FROM clusters WHERE pack_id = ?
                """,
                [pack_id],
            ).fetchall()
        except Exception:
            return None
    best: dict[str, Any] | None = None
    best_score = -1.0
    for cid, _top, cat, count in rows:
        score = 0.0
        if category and str(cat or "").upper() == str(category).upper():
            score += 2.0
        try:
            with domain_con(pack_id) as con2:
                modes = con2.execute(
                    """
                    SELECT r.entity_2 AS e2, COUNT(*) AS n
                    FROM cluster_assignments a
                    JOIN records r ON r.record_id = a.record_id
                    WHERE a.cluster_id = ?
                    GROUP BY r.entity_2 ORDER BY n DESC LIMIT 1
                    """,
                    [cid],
                ).fetchone()
        except Exception:
            modes = None
        if modes and entity_2 and str(modes[0] or "").upper() == str(entity_2).upper():
            score += 3.0
        import math as _math

        try:
            score += _math.log1p(float(count or 0)) * 0.1
        except (TypeError, ValueError):
            pass
        if score > best_score:
            best_score = score
            best = {"cluster_id": cid, "category": cat, "record_count": count}
    return best if best_score > 0 else None


def _anomaly_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(row.get("iso_week") or ""),
        str(row.get("category") or ""),
        str(row.get("entity_2") or ""),
        str(row.get("entity_3") or ""),
    )


def select_flagged_slices(
    combined: list[dict[str, Any]],
    by_source: dict[str, list[dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    """Slices to open, ranked by how many sources flag them.

    ``by_source is None`` only when the source list could not be read.
    An empty map opens nothing: a spike that exists only after mixing
    feeds is not a flag from any one source. One source is enough.
    """
    if by_source is None:
        rows: list[dict[str, Any]] = []
        for row in combined:
            if not row.get("is_anomaly"):
                continue
            copied = dict(row)
            copied["sources_flagged"] = []
            rows.append(copied)
        return rows
    flagged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for src, rows in by_source.items():
        for row in rows:
            if not row.get("is_anomaly"):
                continue
            key = _anomaly_key(row)
            slot = flagged.get(key)
            if slot is None:
                slot = dict(row)
                slot["sources_flagged"] = []
                flagged[key] = slot
            names: list[str] = slot["sources_flagged"]
            if src not in names:
                names.append(src)
            try:
                better = abs(float(row.get("z_score") or 0)) > abs(float(slot.get("z_score") or 0))
            except (TypeError, ValueError):
                better = False
            if better:
                kept = list(names)
                slot.clear()
                slot.update(row)
                slot["sources_flagged"] = kept
    out = list(flagged.values())
    out.sort(
        key=lambda r: (
            -len(r.get("sources_flagged") or []),
            -abs(float(r.get("z_score") or 0)),
        )
    )
    return out


def _observed_sources(pack_id: str) -> list[str]:
    with domain_con(pack_id, read_only=True) as con:
        rows = con.execute(
            """
            SELECT DISTINCT source FROM records
            WHERE source IS NOT NULL AND TRIM(CAST(source AS VARCHAR)) <> ''
              AND (provenance IS NULL OR provenance NOT IN ('inferred', 'fixture'))
            """
        ).fetchall()
    return sorted({str(r[0]) for r in rows if r and r[0]})


def run_fleet_scan(
    pack_id: str,
    *,
    rebuild_clusters: bool = False,
    alert: bool = True,
) -> dict[str, Any]:
    """Run one full-pack fleet scan. Returns the scan report."""
    from src.backtest.engine import run_backtest
    from src.frontline.investigation_brief import lead_weeks_for_slice
    from src.frontline.live_intercept import (
        find_open_investigation,
        open_or_link_investigation,
    )
    from src.ml_runtime.anomalies import recompute_weekly_anomalies

    report: dict[str, Any] = {
        "pack_id": pack_id,
        "anomaly_slices": 0,
        "investigations_opened": [],
        "investigations_linked": [],
        "alerts_fired": 0,
        "novel_open": 0,
        "errors": [],
    }
    import os as _os

    try:
        _lag = int(_os.getenv("FRONTLINE_REPORTING_LAG_DAYS", "0"))
    except ValueError:
        _lag = 0
    try:
        scored = recompute_weekly_anomalies(pack_id, censor_recent_days=_lag)
        report["reporting_lag_days"] = _lag
        report["censored_slices"] = sum(1 for r in scored if r.get("censored"))
    except Exception as e:
        report["errors"].append(f"anomalies:{type(e).__name__}:{e}")
        return report
    if rebuild_clusters:
        try:
            from src.ml_runtime.clustering import rebuild_clusters as _rebuild

            report["cluster_rebuild"] = _rebuild(pack_id)
        except Exception as e:
            report["errors"].append(f"clusters:{type(e).__name__}:{e}")
    try:
        run_backtest(pack_id)
    except Exception as e:
        report["errors"].append(f"backtest:{type(e).__name__}:{e}")
    by_source: dict[str, list[dict[str, Any]]] | None = None
    try:
        by_source = {}
        for src in _observed_sources(pack_id):
            try:
                by_source[src] = recompute_weekly_anomalies(
                    pack_id,
                    censor_recent_days=_lag,
                    source=src,
                    persist=False,
                )
            except Exception as e:
                report["errors"].append(f"source:{src}:{type(e).__name__}:{e}")
    except Exception as e:
        report["errors"].append(f"sources:{type(e).__name__}:{e}")
        by_source = None
    anomalous = select_flagged_slices(scored, by_source)
    report["anomaly_slices"] = len(anomalous)
    seen: set[tuple] = set()
    for row in anomalous:
        cat, ent = row.get("category"), row.get("entity_2")
        model = row.get("entity_3") or ""
        try:
            top = _top_cluster_for_slice(pack_id, cat, ent)
        except Exception as e:
            report["errors"].append(f"resolve:{type(e).__name__}:{e}")
            top = None
        cid = 0
        if top:
            try:
                cid = int(top["cluster_id"])
            except (TypeError, ValueError):
                cid = 0
        if cid < 0:
            cid = 0
        # Cluster 0 is not an identity. Two unmatched slices must not fold
        # into one investigation just because neither resolved a cluster.
        if cid > 0:
            dedup = ("cluster", cid)
        else:
            dedup = ("slice", str(cat or "").upper(), str(ent or "").upper(), str(model).upper())
        if dedup in seen:
            continue
        seen.add(dedup)
        is_new = False
        try:
            before = find_open_investigation(pack_id, cid, cat, ent, model or None)
            inv_id = open_or_link_investigation(
                pack_id=pack_id,
                cluster_id=cid,
                title=f"Fleet scan {cat or ''} {ent or ''}".strip(),
                category=cat,
                entity_2=ent,
                entity_3=model or None,
            )
            is_new = before is None
        except Exception as e:
            report["errors"].append(f"investigation:{type(e).__name__}:{e}")
            continue
        lead_weeks = None
        try:
            lead_weeks = lead_weeks_for_slice(
                pack_id,
                cat,
                ent,
                model or None,
                cluster_id=cid,
                series=scored,
            )
        except Exception as e:
            report["errors"].append(f"lead:{type(e).__name__}:{e}")
        entry = {
            "investigation_id": inv_id,
            "cluster_id": cid,
            "category": cat,
            "entity_2": ent,
            "entity_3": model,
            "z_score": row.get("z_score"),
            "lead_time_weeks": lead_weeks,
            "sources_flagged": list(row.get("sources_flagged") or []),
        }
        if is_new:
            report["investigations_opened"].append(entry)
            if alert:
                try:
                    from src.frontline.alerts import alert_early_warning

                    import anyio as _anyio

                    async def _fire(cid=cid, count=int(row.get("record_count") or 0), lead=lead_weeks) -> None:
                        try:
                            await alert_early_warning(
                                cid,
                                count,
                                lead,
                                pack_id=pack_id,
                            )
                        except Exception:
                            pass

                    try:
                        import asyncio as _aio

                        _aio.get_running_loop().create_task(_fire())
                        report["alerts_fired"] += 1
                    except RuntimeError:
                        _anyio.run(_fire)
                        report["alerts_fired"] += 1
                except Exception:
                    pass
        else:
            report["investigations_linked"].append(entry)
    try:
        from src.observability.slo import record_slo_sample

        record_slo_sample(
            "fleet_scan_anomalous_slices",
            numerator=float(len(report["investigations_opened"])
                            + len(report["investigations_linked"])),
            denominator=float(len(anomalous) or 1),
            objective=0.99,
        )
    except Exception:
        pass
    try:
        from src.data.warehouse import ops_con as _ops_con

        with _ops_con(read_only=True) as _con:
            report["novel_open"] = int(_con.execute(
                "SELECT COUNT(*) FROM novel_candidates WHERE status = 'open'"
            ).fetchone()[0])
            emerging = _con.execute(
                """
                SELECT COALESCE(category, 'unknown'), COALESCE(entity_2, 'unknown'), COUNT(*) as cnt
                FROM novel_candidates
                WHERE status = 'open'
                GROUP BY 1, 2
                HAVING COUNT(*) >= 3
                ORDER BY cnt DESC
                """
            ).fetchall()
            report["novel_emerging_clusters"] = [
                {"category": r[0], "entity_2": r[1], "count": int(r[2])}
                for r in emerging
            ]
    except Exception:
        report.setdefault("novel_emerging_clusters", [])
    return report


__all__ = ["run_fleet_scan", "select_flagged_slices"]
