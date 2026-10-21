"""Weekly volume anomalies computed from ``records`` (not seed literals).

Method (item 10) — quasi-Poisson standardized residuals on zero-filled
weekly buckets, with Benjamini-Hochberg FDR control:

- Event clock: ``occurred_at`` is the event time; ``received_at`` (ingestion
  time) is only a fallback when ``occurred_at`` is missing. Ingestion lag
  must never masquerade as an event spike.
- Zero-fill: every (category, entity_2) series is expanded to the continuous
  Monday-to-Monday week range it spans; missing weeks count 0. Without this,
  quiet weeks vanish and any observed week looks anomalous.
- Prior-only baseline: week *w* is scored against earlier weeks only.
  Later weeks are not in the baseline, so adding future records cannot
  change an alert decision that was already due. The current week is also
  left out, so a spike cannot inflate its own baseline.
- Quasi-Poisson z: ``z = (x - mu) / max(sample_std, sqrt(mu))``. Counts are
  not Gaussian: Poisson variance grows with the mean, so dividing a raw
  count difference by an empirical std mixes units (the old ``z = n - mean``
  bug) and tiny-variance baselines (1000,1000,1000) produce bogus huge z.
  ``max(std, sqrt(mu))`` degrades gracefully to the Poisson floor and also
  absorbs overdispersion. z is unitless by construction.
- Zero baseline (mu == 0): Poisson tail is degenerate, so only an absolute
  floor can fire: ``x >= MIN_ZERO_BASELINE_COUNT`` (default 5). Documented
  heuristic, not a p-value.
- Minimum history (item 42): full-power flags require at least
  ``MIN_BASELINE_WEEKS`` (default 4) baseline weeks — i.e. 5 total weekly
  observations. With 2-3 baseline weeks only the absolute+relative heuristic
  guard may fire (``method='low-history-heuristic'``); with <2, never.
- Multiple comparisons: one-sided Normal-approx p-values from z are adjusted
  with Benjamini-Hochberg frozen at each week, using only slices whose
  iso_week is at or before that week. A later cutoff does not rewrite an
  earlier week's ``p_bh``. ``is_anomaly`` requires
  ``z >= 2.0`` AND ``p_bh <= 0.05`` AND the absolute/relative lift gates
  (``x - mu >= 5`` AND ``(x - mu)/mu >= 50%`` when ``mu > 0``). The lift
  gates keep trivial relative moves (1000 -> 1002) unflagged even when a
  degenerate baseline yields z >= 2.

Assumptions: weekly counts are conditionally independent given the baseline;
no seasonality adjustment (holiday dips can false-positive — check
``baseline_weeks`` and the raw series before acting); population is the
observed warehouse slice, not an external denominator.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Iterable

from src.data.warehouse import apply_domain_schema, domain_con

ANOMALY_Z = 2.0
FDR_Q = 0.05
MIN_BASELINE_WEEKS = 4
MIN_HISTORY_WEEKS_HEURISTIC = 2
MIN_ABSOLUTE_LIFT = 5
MIN_RELATIVE_LIFT = 0.5
MIN_ZERO_BASELINE_COUNT = 5


def iso_week_label(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None
    y, w, _ = dt.isocalendar()
    return f"{y}-W{w:02d}"


def _parse_dt(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw.replace(tzinfo=None) if raw.tzinfo else raw
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _event_time(record: dict[str, Any]) -> datetime | None:
    """Actual event clock: occurred_at first, ingestion (received_at) fallback."""
    for key in ("occurred_at", "received_at"):
        when = _parse_dt(record.get(key))
        if when is not None:
            return when
    return None


def _clock_time(record: dict[str, Any], clock: str) -> datetime | None:
    """``event`` keeps the live clock. ``received`` counts only received_at."""
    if clock == "received":
        return _parse_dt(record.get("received_at"))
    return _event_time(record)


def count_weekly_slices(
    records: Iterable[dict[str, Any]],
    *,
    clock: str = "event",
    as_of: Any | None = None,
    model_grain: bool = False,
    source: str | None = None,
) -> dict[tuple, int]:
    """Weekly counts.

    Default keys are ``(iso_week, category, entity_2)`` on the event clock,
    which existing callers unpack as a 3-tuple. ``model_grain=True`` adds
    entity_3 so a model-scoped recall can match its own series. ``source``
    keeps warranty or telematics off the complaint series. ``clock='received'``
    and ``as_of`` are the recall proof: a row counts only when NHTSA had
    already received it.
    """
    if clock not in ("event", "received"):
        raise ValueError("clock must be 'event' or 'received'")
    as_of_dt = _parse_dt(as_of)
    out: dict[tuple, int] = defaultdict(int)
    for r in records:
        if source is not None and str(r.get("source") or "") != source:
            continue
        when = _clock_time(r, clock)
        if when is None:
            continue
        if as_of_dt is not None and when > as_of_dt:
            continue
        week = iso_week_label(when)
        if not week:
            continue
        cat = str(r.get("category") or "")
        ent = str(r.get("entity_2") or "")
        try:
            weight = int(r.get("event_count") or 1)
        except (TypeError, ValueError):
            weight = 1
        if weight < 1:
            weight = 1
        if model_grain:
            out[(week, cat, ent, str(r.get("entity_3") or ""))] += weight
        else:
            out[(week, cat, ent)] += weight
    return dict(out)


def _week_monday(label: str) -> datetime | None:
    try:
        y, w = label.split("-W")
        return datetime.fromisocalendar(int(y), int(w), 1)
    except (ValueError, TypeError):
        return None


def _unpack_count_key(key: tuple) -> tuple[str, str, str, str]:
    """3-tuples stay make-level (model ''). 4-tuples carry entity_3."""
    if len(key) >= 4:
        week, cat, ent, model = key[0], key[1], key[2], key[3]
    else:
        week, cat, ent = key[0], key[1], key[2]
        model = ""
    return str(week), str(cat), str(ent), str(model or "")


def zero_fill_weeks(counts: dict[tuple, int]) -> dict[tuple, int]:
    """Expand every series to its continuous week range.

    Missing weeks in between count 0. A silent week is evidence of quiet,
    not a gap in the data. 3-tuple keys stay 3-tuples; 4-tuple keys stay
    4-tuples so model grain is not collapsed back to the make.
    """
    grain4 = any(len(k) >= 4 for k in counts)
    by_group: dict[tuple, set[str]] = defaultdict(set)
    for key in counts:
        week, cat, ent, model = _unpack_count_key(key)
        group = (cat, ent, model) if grain4 else (cat, ent)
        by_group[group].add(week)
    filled = dict(counts)
    for group, weeks in by_group.items():
        starts = sorted(m for w in weeks if (m := _week_monday(w)) is not None)
        if len(starts) < 2:
            continue
        cur, end = starts[0], starts[-1]
        while cur <= end:
            y, w, _ = cur.isocalendar()
            label = f"{y}-W{w:02d}"
            if grain4:
                filled.setdefault((label, group[0], group[1], group[2]), 0)
            else:
                filled.setdefault((label, group[0], group[1]), 0)
            cur += timedelta(weeks=1)
    return filled


def _upper_tail_p(z: float) -> float:
    """One-sided Normal-approx upper-tail p-value for z (spike test)."""
    if z <= 0:
        return 1.0
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def _bh_adjust(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (same order as input)."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    adj = [0.0] * m
    running_min = 1.0
    for rank_rev, idx in enumerate(reversed(order)):
        rank = m - rank_rev  # 1-based rank of this p in ascending order
        val = p_values[idx] * m / rank
        running_min = min(running_min, val)
        adj[idx] = min(1.0, running_min)
    return adj


def set_exposure(
    pack_id: str,
    iso_week: str,
    category: str | None,
    entity_2: str | None,
    exposure_units: float,
    *,
    unit: str = "units",
) -> dict[str, Any]:
    """Record exposure (vehicles-in-operation / active accounts / shipped).

    Audit 8.1: rates beat raw counts. Slice-weeks with exposure score on
    rate (count per 1000 units, Poisson offset semantics); slices without
    exposure score on raw counts and are labeled ``normalized=False``
    (boards must render those as 'unnormalised').
    """
    if exposure_units is None or float(exposure_units) <= 0:
        raise ValueError("exposure_units must be positive")
    with domain_con(pack_id, read_only=False) as con:
        apply_domain_schema(con)
        con.execute(
            """
            INSERT OR REPLACE INTO exposure
            (pack_id, iso_week, category, entity_2, exposure_units, unit)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [pack_id, iso_week, category, entity_2, float(exposure_units), unit],
        )
    return {
        "pack_id": pack_id, "iso_week": iso_week, "category": category,
        "entity_2": entity_2, "exposure_units": float(exposure_units), "unit": unit,
    }


def load_exposure(
    pack_id: str,
    *,
    category: str | None = None,
    entity_2: str | None = None,
) -> dict[tuple[str, str, str], float]:
    """(iso_week, category, entity_2) → exposure units. Empty when unrecorded."""
    with domain_con(pack_id) as con:
        try:
            sql = "SELECT iso_week, category, entity_2, exposure_units FROM exposure WHERE pack_id = ?"
            params: list[Any] = [pack_id]
            if category:
                sql += " AND category = ?"
                params.append(category)
            if entity_2:
                sql += " AND entity_2 = ?"
                params.append(entity_2)
            rows = con.execute(sql, params).fetchall()
        except Exception:
            return {}
    return {
        (str(w), str(c or ""), str(e or "")): float(u)
        for w, c, e, u in rows if u and float(u) > 0
    }


def detect_cusum_change_point(
    series: list[float],
    *,
    baseline_mean: float,
    baseline_std: float,
    k: float = 0.5,
    h: float = 4.0,
) -> dict[str, Any]:
    """Page's upper Cumulative Sum (CUSUM) for detecting sustained upward shifts.

    Distinguishes sustained process step-changes from 1-week transient spikes (audit 4.3).
    - k: reference value (slack parameter, in units of std; default 0.5)
    - h: decision boundary threshold (default 4.0 std devs)
    """
    if not series:
        return {"cusum_values": [], "triggered": False, "change_point_index": None}
    sigma = max(baseline_std, math.sqrt(baseline_mean) if baseline_mean > 0 else 1.0)
    k_val = k * sigma
    h_val = h * sigma
    s_pos = 0.0
    cusum_vals = []
    change_idx = None
    for idx, val in enumerate(series):
        s_pos = max(0.0, s_pos + (val - baseline_mean - k_val))
        cusum_vals.append(s_pos)
        if change_idx is None and s_pos >= h_val:
            change_idx = idx
    return {
        "cusum_values": cusum_vals,
        "triggered": change_idx is not None,
        "change_point_index": change_idx,
    }


def score_weekly_slices(
    counts: dict[tuple[str, str, str], int],
    *,
    pack_id: str,
    in_flight: dict[str, Any] | None = None,
    exposure: dict[tuple[str, str, str], float] | None = None,
) -> list[dict[str, Any]]:
    """Attach z_score / p_value / is_anomaly. Pure; no I/O.

    Zero-fills each series, scores prior-only quasi-Poisson residuals,
    and applies BH-FDR at each week using only information available then.

    ``in_flight`` (item 36) folds ONE in-flight contact into the CURRENT
    week bucket before scoring — ``{"iso_week", "category", "entity_2"}`` —
    so a live voice/text contact can tip the threshold in the same call.
    The +1 is labeled (``in_flight_included``) and never persisted by the
    scorer; persistence paths must pass ``in_flight=None``.

    ``exposure`` (audit 8.1) maps (week, category, entity_2) → exposure
    units. A slice scores on RATE (per 1000 units) only when its own week
    and every baseline week have exposure; otherwise it falls back to raw
    counts with ``normalized=False`` so boards label it 'unnormalised'
    instead of presenting volume-driven flags as rate anomalies.
    """
    if in_flight:
        try:
            from src.data.timeutil import utc_now as _now

            week = in_flight.get("iso_week") or iso_week_label(_now())
            cat = str(in_flight.get("category") or "")
            ent = str(in_flight.get("entity_2") or "")
            grain4 = any(len(k) >= 4 for k in counts)
            if grain4:
                key = (str(week), cat, ent, str(in_flight.get("entity_3") or ""))
            else:
                key = (str(week), cat, ent)
            counts = dict(counts)
            counts[key] = counts.get(key, 0) + 1
        except Exception:
            pass
    filled = zero_fill_weeks(counts)
    grain4 = any(len(k) >= 4 for k in filled)
    by_group: dict[tuple, list[tuple[str, int]]] = defaultdict(list)
    for key, n in filled.items():
        week, cat, ent, model = _unpack_count_key(key)
        group = (cat, ent, model) if grain4 else (cat, ent)
        by_group[group].append((week, n))

    # First pass: per-slice statistics. Baseline and CUSUM use only weeks
    # at or before the week being scored.
    prelim: list[dict[str, Any]] = []
    exp = exposure or {}
    cusum_map: dict[tuple, list[dict[str, Any]]] = {}

    for group, weeks in sorted(by_group.items()):
        if grain4:
            cat, ent, model = group
        else:
            cat, ent = group
            model = ""
        sorted_weeks = sorted(weeks)
        prefix_vals: list[float] = []
        prefix_cusum: list[dict[str, Any]] = []
        for _week, n_raw in sorted_weeks:
            prefix_vals.append(float(n_raw))
            s_mu = sum(prefix_vals) / len(prefix_vals)
            s_var = sum((v - s_mu) ** 2 for v in prefix_vals) / max(1, len(prefix_vals) - 1)
            prefix_cusum.append(
                detect_cusum_change_point(
                    list(prefix_vals),
                    baseline_mean=s_mu,
                    baseline_std=math.sqrt(s_var),
                )
            )
        cusum_map[(cat, ent, model)] = prefix_cusum

        for w_idx, (week, n) in enumerate(sorted_weeks):
            others = [(w, c) for w, c in sorted_weeks if w < week]
            # Rate path (audit 8.1): per-1000-unit rates when exposure covers
            # this week AND every baseline week; else raw counts.
            exp_here = exp.get((week, cat, ent))
            exp_others = [exp.get((w, cat, ent)) for w, _ in others]
            normalized = bool(
                exp_here and exp_others and all(e and e > 0 for e in exp_others)
            )
            if normalized:
                assert exp_here is not None
                n_v = float(n) / float(exp_here) * 1000.0
                base_v = [
                    float(c) / float(e) * 1000.0
                    for (_, c), e in zip(others, exp_others)
                    if e
                ]
            else:
                n_v = float(n)
                base_v = [float(c) for _, c in others]
            mu = sum(base_v) / len(base_v) if base_v else 0.0
            if len(base_v) >= 2:
                var = sum((v - mu) ** 2 for v in base_v) / (len(base_v) - 1)
                std = math.sqrt(var)
            else:
                std = 0.0
            lift_abs = float(n_v - mu)
            lift_rel = (lift_abs / mu) if mu > 0 else (float("inf") if lift_abs > 0 else 0.0)
            hist_n = len(base_v)
            if mu <= 0:
                # Zero baseline: Poisson tail is degenerate. Absolute floor
                # only (documented heuristic).
                if n_v >= MIN_ZERO_BASELINE_COUNT and hist_n >= MIN_HISTORY_WEEKS_HEURISTIC:
                    z, method = ANOMALY_Z, "zero-baseline"
                else:
                    z, method = 0.0, "insufficient-history"
            else:
                denom = max(std, math.sqrt(mu))  # quasi-Poisson floor
                z = lift_abs / denom if denom > 0 else 0.0
                if hist_n >= MIN_BASELINE_WEEKS:
                    method = "quasi-poisson"
                elif hist_n >= MIN_HISTORY_WEEKS_HEURISTIC:
                    method = "low-history-heuristic"
                else:
                    method = "insufficient-history"
            prelim.append(
                {
                    "pack_id": pack_id,
                    "iso_week": week,
                    "category": cat,
                    "entity_2": ent,
                    "entity_3": model,
                    "record_count": n,
                    "baseline_mean": mu,
                    "baseline_std": std,
                    "baseline_weeks": hist_n,
                    "z_score": z,
                    "p_value": _upper_tail_p(z) if mu > 0 else (0.0 if z >= ANOMALY_Z else 1.0),
                    "absolute_lift": lift_abs,
                    "relative_lift": lift_rel if lift_rel != float("inf") else None,
                    "method": method,
                    "normalized": normalized,
                    "exposure_units": exp.get((week, cat, ent)),
                    "w_idx": w_idx,
                }
            )

    # Second pass: freeze each week's FDR using only slices known at that week.
    # p_bh is written for the cutoff week only, so a future week cannot
    # rewrite a decision that was already due.
    candidates_by_cutoff: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(prelim):
        if r["method"] != "insufficient-history":
            candidates_by_cutoff[str(r["iso_week"])].append(i)
    p_bh_map: dict[int, float] = {}
    for cutoff in sorted(candidates_by_cutoff):
        window = [
            i
            for i, r in enumerate(prelim)
            if r["method"] != "insufficient-history" and str(r["iso_week"]) <= cutoff
        ]
        adj = _bh_adjust([prelim[i]["p_value"] for i in window])
        adj_by_i = {idx: a for idx, a in zip(window, adj)}
        for i in candidates_by_cutoff[cutoff]:
            p_bh_map[i] = adj_by_i[i]

    rows: list[dict[str, Any]] = []
    for i, r in enumerate(prelim):
        p_bh = p_bh_map.get(i, 1.0)
        r["p_bh"] = p_bh
        if r.get("normalized"):
            # Rate units: absolute-count gates don't apply; relative lift
            # (unitless) + z + FDR carry the decision.
            gates = (
                r["method"] != "insufficient-history"
                and r["z_score"] >= ANOMALY_Z
                and p_bh <= FDR_Q
                and (r["relative_lift"] or 0.0) >= MIN_RELATIVE_LIFT
            )
        else:
            gates = (
                r["method"] != "insufficient-history"
                and r["z_score"] >= ANOMALY_Z
                and p_bh <= FDR_Q
                and r["absolute_lift"] >= MIN_ABSOLUTE_LIFT
                and (r["baseline_mean"] <= 0 or (r["relative_lift"] or 0.0) >= MIN_RELATIVE_LIFT)
            )
        r["is_anomaly"] = bool(gates)

        # CUSUM sustained shift vs transient spike classification (audit 4.3).
        # The prefix stored for this week does not include later weeks.
        cat_ent = (r["category"], r["entity_2"], r.get("entity_3") or "")
        series = cusum_map.get(cat_ent, [])
        w_idx = r.pop("w_idx", 0)
        c_res = series[w_idx] if isinstance(series, list) and w_idx < len(series) else {}
        c_trig = bool(c_res.get("triggered"))
        c_idx = c_res.get("change_point_index")
        is_sustained = c_trig and (c_idx is not None and w_idx >= c_idx)
        if r["is_anomaly"]:
            r["change_point_type"] = "sustained_shift" if is_sustained else "transient_spike"
            r["cusum_triggered"] = is_sustained
        else:
            r["change_point_type"] = "none"
            r["cusum_triggered"] = False

        r["in_flight_included"] = bool(in_flight) and (
            r["iso_week"] == (in_flight.get("iso_week") or r["iso_week"])
            and r["category"] == str(in_flight.get("category") or "")
            and r["entity_2"] == str(in_flight.get("entity_2") or "")
        )
        r["alert_available_week"] = r["iso_week"]
        r["information_through"] = r["iso_week"]
        r["model_identity"] = "quasi-poisson-causal-v1"
        r["feed_complete"] = True
        rows.append(r)
    return rows


def recompute_weekly_anomalies(
    pack_id: str,
    *,
    category: str | None = None,
    entity_2: str | None = None,
    censor_recent_days: int = 0,
    source: str | None = None,
    model_grain: bool = False,
    clock: str = "event",
    as_of: Any | None = None,
    persist: bool = True,
) -> list[dict[str, Any]]:
    """Read ``records``, replace ``weekly_anomalies`` for *pack_id*, return rows.

    Optional category / entity_2 restrict the recompute to one slice so a live
    contact does not rescan the whole warehouse. Exposure rows (see
    ``set_exposure``) switch scoring to rates where covered; uncovered slices
    stay raw-count with ``normalized=False``.

    ``source`` scores one feed (warranty, telematics) on its own series.
    ``model_grain`` stores the model on ``entity_3`` so a model-scoped recall
    can match. The default remains make-level 3-tuples.

    ``clock='received'`` and ``as_of`` count only rows whose ``received_at``
    is at or before the cutoff. The live scan keeps the event clock.

    ``censor_recent_days`` (board #6 — right-censoring): buckets ending
    within N days of now are systematically under-reported (NHTSA/CFPB lag).
    Censored buckets are still counted but never flagged (``censored=True``),
    and that bit is stored so the board can show which weeks were held back.
    """
    with domain_con(pack_id, read_only=False) as con:
        apply_domain_schema(con)
        sql = (
            "SELECT received_at, occurred_at, category, entity_2, entity_3, source,"
            " COALESCE(event_count, 1) AS event_count"
            " FROM records"
        )
        params: list[Any] = []
        # Fixture rows are demo data. Inferred rows are unfinished calls.
        # Neither is an observed failure.
        wheres: list[str] = [
            "(provenance IS NULL OR provenance NOT IN ('inferred', 'fixture'))"
        ]
        if category:
            wheres.append("category = ?")
            params.append(category)
        if entity_2:
            wheres.append("entity_2 = ?")
            params.append(entity_2)
        if source:
            wheres.append("source = ?")
            params.append(source)
        sql += " WHERE " + " AND ".join(wheres)
        raw = con.execute(sql, params).fetchall()
        cols = [d[0] for d in con.description]
        records = [dict(zip(cols, r)) for r in raw]
        try:
            exp_rows = con.execute(
                "SELECT iso_week, category, entity_2, exposure_units FROM exposure"
                " WHERE pack_id = ?",
                [pack_id],
            ).fetchall()
            exposure_map = {
                (str(w), str(c or ""), str(e or "")): float(u)
                for w, c, e, u in exp_rows if u and float(u) > 0
            } or None
        except Exception:
            exposure_map = None
        scored = score_weekly_slices(
            count_weekly_slices(
                records,
                clock=clock,
                as_of=as_of,
                model_grain=model_grain,
                source=source,
            ),
            pack_id=pack_id,
            exposure=exposure_map,
        )
        if censor_recent_days and censor_recent_days > 0:
            from datetime import timedelta as _td

            from src.data.timeutil import utc_now as _now

            try:
                _cutoff = (_now().replace(tzinfo=None) - _td(days=int(censor_recent_days))).date()
            except Exception:
                _cutoff = None
            if _cutoff is not None:
                for _row in scored:
                    try:
                        _y, _w = str(_row.get("iso_week") or "").split("-W")
                        from datetime import datetime as _dt

                        _start = _dt.fromisocalendar(int(_y), int(_w), 1).date()
                        _end = _start + _td(days=6)
                        if _end > _cutoff:
                            _row["censored"] = True
                            _row["is_anomaly"] = False
                        else:
                            _row.setdefault("censored", False)
                    except Exception:
                        _row.setdefault("censored", False)
        else:
            for _row in scored:
                _row.setdefault("censored", False)
        if persist:
            if category or entity_2 or source:
                dsql = "DELETE FROM weekly_anomalies WHERE pack_id = ?"
                dparams: list[Any] = [pack_id]
                if category:
                    dsql += " AND category = ?"
                    dparams.append(category)
                if entity_2:
                    dsql += " AND entity_2 = ?"
                    dparams.append(entity_2)
                con.execute(dsql, dparams)
            else:
                con.execute("DELETE FROM weekly_anomalies WHERE pack_id = ?", [pack_id])
            _insert_weekly_anomalies(con, scored)
    return scored


def _insert_weekly_anomalies(con, rows: list[dict[str, Any]]) -> None:
    """Persist scored slices, including model grain and the censored bit."""
    for row in rows:
        con.execute(
            """
            INSERT INTO weekly_anomalies (
                pack_id, iso_week, category, entity_2, entity_3, record_count,
                baseline_mean, baseline_std, z_score, is_anomaly,
                p_value, p_bh, baseline_weeks, method, censored
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                row["pack_id"],
                row["iso_week"],
                row["category"],
                row["entity_2"],
                row.get("entity_3") or "",
                row["record_count"],
                row["baseline_mean"],
                row["baseline_std"],
                row["z_score"],
                row["is_anomaly"],
                row.get("p_value"),
                row.get("p_bh"),
                row.get("baseline_weeks"),
                row.get("method"),
                bool(row.get("censored")),
            ],
        )


def persist_scored_anomalies(pack_id: str, rows: list[dict[str, Any]]) -> None:
    """Replace this pack's weekly_anomalies with an already-scored series.

    The recall report scores each as-of date in memory. Persisting that
    series is what makes ``run_backtest(as_of=...)`` read the same weeks,
    instead of a table that still contains later data.
    """
    with domain_con(pack_id, read_only=False) as con:
        apply_domain_schema(con)
        con.execute("DELETE FROM weekly_anomalies WHERE pack_id = ?", [pack_id])
        _insert_weekly_anomalies(con, rows)


def current_spikes(pack_id: str) -> dict[str, Any]:
    """This week's computed spikes, plus weeks the lag censor held back.

    "This week" is the latest iso_week that was not censored. Spikes are the
    anomaly rows in that week. A missing grain column (pre-migration file
    opened read-only) degrades to the columns that exist.
    """
    empty = {"spike_week": None, "spikes": [], "censored_weeks": []}
    try:
        with domain_con(pack_id, read_only=True) as con:
            info = con.execute(
                "SELECT name FROM pragma_table_info('weekly_anomalies')"
            ).fetchall()
            cols = {str(r[0]) for r in info}
            if "iso_week" not in cols:
                return empty
            has_censored = "censored" in cols
            has_model = "entity_3" in cols
            censored_expr = "COALESCE(censored, FALSE)" if has_censored else "FALSE"
            model_expr = "COALESCE(entity_3, '')" if has_model else "''"
            week = con.execute(
                f"""
                SELECT MAX(iso_week) FROM weekly_anomalies
                WHERE pack_id = ? AND {censored_expr} = FALSE
                """,
                [pack_id],
            ).fetchone()
            iso = week[0] if week else None
            spikes: list[dict[str, Any]] = []
            if iso:
                raw = con.execute(
                    f"""
                    SELECT iso_week, category, entity_2, {model_expr},
                           record_count, baseline_mean, z_score
                    FROM weekly_anomalies
                    WHERE pack_id = ? AND iso_week = ? AND is_anomaly = TRUE
                      AND {censored_expr} = FALSE
                    ORDER BY z_score DESC
                    """,
                    [pack_id, iso],
                ).fetchall()
                for iso_week, cat, ent, model, count, baseline, z in raw:
                    spikes.append(
                        {
                            "iso_week": iso_week,
                            "category": cat,
                            "entity_2": ent,
                            "entity_3": model or "",
                            "record_count": count,
                            "baseline_mean": baseline,
                            "z_score": z,
                        }
                    )
            censored_weeks: list[str] = []
            if has_censored:
                held = con.execute(
                    """
                    SELECT DISTINCT iso_week FROM weekly_anomalies
                    WHERE pack_id = ? AND COALESCE(censored, FALSE) = TRUE
                    ORDER BY iso_week
                    """,
                    [pack_id],
                ).fetchall()
                censored_weeks = [str(r[0]) for r in held if r[0]]
            return {
                "spike_week": iso,
                "spikes": spikes,
                "censored_weeks": censored_weeks,
            }
    except Exception:
        return empty


__all__ = [
    "ANOMALY_Z",
    "FDR_Q",
    "MIN_BASELINE_WEEKS",
    "iso_week_label",
    "count_weekly_slices",
    "zero_fill_weeks",
    "score_weekly_slices",
    "recompute_weekly_anomalies",
    "current_spikes",
    "persist_scored_anomalies",
    "set_exposure",
    "load_exposure",
    "detect_cusum_change_point",
]
