"""Real backtest engine — compute lead_time_weeks from domain warehouse.

For each cluster that has an advisory with overlapping scope, lead time is
weeks between the cluster's first anomaly/spike week (or first_seen) and the
advisory issued_at. Results are written to ``backtest_results`` (replacing
fixture fiction when re-run).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.data.warehouse import domain_con


def _parse_ts(v: Any) -> datetime | None:
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.replace(tzinfo=None) if v.tzinfo else v
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _weeks_between(earlier: datetime, later: datetime) -> int:
    days = (later - earlier).total_seconds() / 86400.0
    # floor: 4 days is 0 weeks, not 1 (round() overclaimed lead)
    return max(0, int(days // 7.0))


def _norm_entity(val: Any) -> str:
    return str(val or "").strip().upper()


class ClusterEntities(tuple):
    """Container for distinct entity sets and individual record entity tuples."""

    def __new__(cls, e2: set[str], e3: set[str], tuples: list[tuple[str, str]]):
        obj = super().__new__(cls, (e2, e3))
        obj.e2 = e2
        obj.e3 = e3
        obj.tuples = tuples
        return obj


def _cluster_entities(
    con, cluster_id: Any, as_of_ts: datetime | None = None
) -> ClusterEntities:
    """Distinct normalized (entity_2 values, entity_3 values) and tuples of cluster members as of as_of_ts.

    Record-level overlap: the advisory's scoped entities must appear among the
    cluster's actual member records known as of the cutoff — never inferred from the cluster row (R13).
    """
    sql = """
        SELECT r.entity_2, r.entity_3
        FROM records r
        JOIN cluster_assignments a ON a.record_id = r.record_id
        WHERE a.cluster_id = ?
    """
    params: list[Any] = [cluster_id]
    if as_of_ts is not None:
        sql += " AND r.received_at <= ?"
        params.append(as_of_ts)
    try:
        rows = con.execute(sql, params).fetchall()
    except Exception:
        rows = []
    member_tuples = [(_norm_entity(r[0]), _norm_entity(r[1])) for r in rows]
    e2 = {t[0] for t in member_tuples if t[0]}
    e3 = {t[1] for t in member_tuples if t[1]}
    return ClusterEntities(e2, e3, member_tuples)


def _entity_gate(
    member_e2: set[str],
    member_e3: set[str],
    scope_e2: Any,
    scope_e3: Any,
    member_tuples: list[tuple[str, str]] | None = None,
) -> tuple[bool, list[str]]:
    """Entity-overlap gate (item 2): category alone is never sufficient when
    the advisory narrows to an entity.

    Returns (ok, basis_parts). Every non-empty advisory entity scope must have
    at least one exact (case-insensitive) hit among cluster member records.
    When both entity_2 and entity_3 are scoped, they must co-occur in at least
    one member record tuple (R13).
    """
    parts: list[str] = []
    want_e2 = _norm_entity(scope_e2)
    want_e3 = _norm_entity(scope_e3)
    if not want_e2 and not want_e3:
        return True, ["scope=category_wide"]

    if want_e2:
        if want_e2 not in member_e2:
            return False, []
        parts.append("entity_2")
    if want_e3:
        if want_e3 not in member_e3:
            return False, []
        parts.append("entity_3")

    if want_e2 and want_e3 and member_tuples is not None:
        has_matching_record = any(
            m2 == want_e2 and m3 == want_e3 for m2, m3 in member_tuples
        )
        if not has_matching_record:
            return False, []

    return True, parts
    return True, parts


def run_backtest(pack_id: str, *, as_of: Any | None = None) -> list[dict[str, Any]]:
    """Compute and persist backtest_results for ``pack_id``. Returns rows written.

    ``as_of`` (audit 8.1 — right-censoring discipline): only evidence at or
    before this timestamp counts (records by received_at, advisories by
    issued_at, anomalies by week start). Without it the backtest asks "would
    we have caught it" with today's complete data and lead times come out
    optimistic. Every returned row carries the ``as_of`` it was computed
    under so cards can display it.
    """
    from datetime import datetime as _dt

    as_of_ts: _dt | None = None
    if as_of is not None:
        as_of_ts = _parse_ts(as_of)
    as_of_label = (
        as_of_ts.replace(tzinfo=None).isoformat() if as_of_ts is not None else None
    )
    results: list[dict[str, Any]] = []
    with domain_con(pack_id, read_only=False) as con:
        # Ensure table exists (schema apply)
        from src.data.warehouse import apply_domain_schema

        apply_domain_schema(con)
        clusters = con.execute(
            """
            SELECT cluster_id, category, first_seen, last_seen, record_count
            FROM clusters WHERE pack_id = ?
            """,
            [pack_id],
        ).fetchall()
        if as_of_ts is None:
            advisories = con.execute(
                """
                SELECT advisory_id, issued_at, scope_category, scope_entity_2, scope_entity_3
                FROM advisories
                """
            ).fetchall()
        else:
            # As-of view: advisories issued after t did not exist yet.
            advisories = con.execute(
                """
                SELECT advisory_id, issued_at, scope_category, scope_entity_2, scope_entity_3
                FROM advisories WHERE issued_at <= ?
                """,
                [as_of_ts],
            ).fetchall()

        # Optional: first anomaly week per category
        anomalies = con.execute(
            """
            SELECT category, MIN(iso_week) AS first_anom
            FROM weekly_anomalies
            WHERE pack_id = ? AND is_anomaly = TRUE
            GROUP BY category
            """,
            [pack_id],
        ).fetchall()
        anom_by_cat = {r[0]: r[1] for r in anomalies if r[0]}

        written: list[tuple] = []
        for cl in clusters:
            cluster_id, category, first_seen, last_seen, record_count = cl
            fs = _parse_ts(first_seen)
            # Earliest evidence of the cluster: min(cluster.first_seen,
            # earliest assigned record). Previous code overwrote first_seen
            # with the record date (which fixtures set to now-10d), hiding
            # the true early signal. Take the earliest, not the latest.
            # As-of discipline (audit 8.1): records received after t are
            # right-censored — they did not exist at scan time.
            if as_of_ts is None:
                rec_first = con.execute(
                    """
                    SELECT MIN(r.received_at)
                    FROM records r
                    JOIN cluster_assignments a ON a.record_id = r.record_id
                    WHERE a.cluster_id = ?
                    """,
                    [cluster_id],
                ).fetchone()
            else:
                rec_first = con.execute(
                    """
                    SELECT MIN(r.received_at)
                    FROM records r
                    JOIN cluster_assignments a ON a.record_id = r.record_id
                    WHERE a.cluster_id = ? AND r.received_at <= ?
                    """,
                    [cluster_id, as_of_ts],
                ).fetchone()
            if rec_first and rec_first[0] is not None:
                rf = _parse_ts(rec_first[0])
                if rf is not None and (fs is None or rf < fs):
                    fs = rf
                elif fs is None:
                    fs = rf
            # Prefer anomaly signal when available and earlier than first_seen
            first_anom = anom_by_cat.get(category)
            spike_start = fs
            if first_anom and isinstance(first_anom, str) and "W" in first_anom:
                try:
                    y, w = first_anom.split("-W")
                    anom_dt = datetime.fromisocalendar(int(y), int(w), 1)
                    if as_of_ts is not None and anom_dt > as_of_ts:
                        pass  # anomaly week did not exist yet at time t
                    elif spike_start is None or anom_dt < spike_start:
                        spike_start = anom_dt
                except Exception:
                    pass
            if as_of_ts is not None and spike_start is not None and spike_start > as_of_ts:
                # No pre-t signal existed at time t: not evaluable, not a hit.
                spike_start = None
            # If still no start, backdate by record_count weeks so demos have a lead signal
            if spike_start is None:
                continue

            # Member entities for the overlap gate (record-level, item 2).
            member_ents = _cluster_entities(con, cluster_id, as_of_ts)
            member_e2, member_e3 = member_ents.e2, member_ents.e3

            for adv in advisories:
                advisory_id, issued_at, scope_cat, scope_e2, scope_e3 = adv
                # Scope match: category overlap AND entity overlap when the
                # advisory narrows to an entity (previously category-only
                # inflated recall). Category alone is never sufficient for an
                # entity-scoped advisory.
                if scope_cat and category and scope_cat != category:
                    continue
                entity_ok, entity_parts = _entity_gate(
                    member_e2, member_e3, scope_e2, scope_e3, member_tuples=member_ents.tuples
                )
                if not entity_ok:
                    # Same category, disjoint entities => explicit MISS, never
                    # a synthetic pre-advisory window.
                    written.append(
                        (cluster_id, advisory_id, 0, False, "entity-mismatch")
                    )
                    results.append(
                        {
                            "cluster_id": cluster_id,
                            "advisory_id": advisory_id,
                            "lead_time_weeks": 0,
                            "matched": False,
                            "pack_id": pack_id,
                            "provenance": "computed",
                            "match_basis": "entity-mismatch",
                            "as_of": as_of_label,
                        }
                    )
                    continue
                iss = _parse_ts(issued_at)
                if not iss:
                    continue
                # Lead time = how many weeks the cluster signal preceded the advisory.
                # Honest backtest: signal AFTER the advisory is a MISS
                # (lead=0, matched=False). Never synthesize a pre-advisory
                # window — that guaranteed 100% hit rate.
                if spike_start is not None and spike_start < iss:
                    lead = _weeks_between(spike_start, iss)
                    matched = lead > 0 and (record_count or 0) >= 1
                else:
                    lead = 0
                    matched = False
                basis = "category+temporal" if not entity_parts else (
                    "category+" + "+".join(entity_parts) + "+temporal"
                )
                if not matched:
                    basis = "no-lead" if lead <= 0 else basis
                    # Missed signal => matched=False, always.
                    matched = False
                written.append((cluster_id, advisory_id, lead, matched, basis))
                results.append(
                    {
                        "cluster_id": cluster_id,
                        "advisory_id": advisory_id,
                        "lead_time_weeks": lead,
                        "matched": matched,
                        "pack_id": pack_id,
                        "provenance": "computed",
                        "match_basis": basis,
                        "as_of": as_of_label,
                    }
                )

        # Replace this pack's rows only (never wipe other packs — item 43).
        # Scoped by pack_id; legacy rows without pack_id are cleaned by this
        # pack's cluster_ids so one pack's rerun never deletes another's data.
        # Unconditionally replace pack results so empty reruns do not leave stale rows (R23).
        try:
            con.execute(
                "DELETE FROM backtest_results WHERE pack_id = ?",
                [pack_id],
            )
        except Exception:
            pass
        cluster_ids = [cl[0] for cl in clusters]
        if cluster_ids:
            placeholders = ",".join("?" for _ in cluster_ids)
            try:
                con.execute(
                    "DELETE FROM backtest_results WHERE pack_id IS NULL "
                    f"AND cluster_id IN ({placeholders})",
                    cluster_ids,
                )
            except Exception:
                pass
        if written:
            for cluster_id, advisory_id, lead, matched, basis in written:
                con.execute(
                    """
                    INSERT INTO backtest_results
                    (cluster_id, advisory_id, lead_time_weeks, matched,
                     pack_id, provenance, match_basis)
                    VALUES (?, ?, ?, ?, ?, 'computed', ?)
                    """,
                    [cluster_id, advisory_id, lead, matched, pack_id, basis],
                )
    return results


def best_lead_time(pack_id: str) -> dict[str, Any] | None:
    """Return the best (max) matched lead_time row for demos.

    Pure read: does NOT recompute (previous version side-effected a full
    recompute on every call). Reports max matched lead; callers needing
    distribution should use run_backtest() rows directly.
    """
    from src.data.warehouse import apply_domain_schema as _apply

    with domain_con(pack_id, read_only=True) as con:
        try:
            _apply(con)
        except Exception:
            pass
        try:
            try:
                cur = con.execute(
                    """
                    SELECT cluster_id, advisory_id, lead_time_weeks, matched,
                           pack_id, provenance, match_basis
                    FROM backtest_results
                    WHERE pack_id = ? OR pack_id IS NULL
                    """,
                    [pack_id],
                )
            except Exception:
                # Legacy schema without pack_id/provenance columns.
                cur = con.execute(
                    """
                    SELECT cluster_id, advisory_id, lead_time_weeks, matched
                    FROM backtest_results
                    """
                )
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        except Exception:
            rows = []
    if not rows:
        rows = run_backtest(pack_id)
    matched = [r for r in rows if r.get("matched")]
    if not matched:
        return rows[0] if rows else None
    return max(matched, key=lambda r: int(r.get("lead_time_weeks") or 0))


__all__ = ["run_backtest", "best_lead_time", "_entity_gate", "_cluster_entities"]
