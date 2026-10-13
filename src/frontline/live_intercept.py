"""Live cluster intercept: one contact re-scores the slice and can open an investigation."""

from __future__ import annotations

from typing import Any

from src.data.timeutil import utc_now
from src.data.warehouse import ops_con
from src.ids import new_ulid
from src.ledger import AgentAction, record_action
from src.ml_runtime.anomalies import recompute_weekly_anomalies


def slice_is_anomalous(
    pack_id: str,
    category: str | None,
    entity_2: str | None,
    *,
    in_flight: bool = True,
) -> dict[str, Any]:
    """Score this contact's slice with the in-flight contact counted (item 36).

    The current voice/text contact is folded (+1) into its week bucket IN
    MEMORY before scoring — never persisted as a phantom record — so a live
    contact tips the threshold in the same call instead of waiting for the
    next recompute. Channel-agnostic: voice and text tip identically.
    """
    from src.data.warehouse import domain_con
    from src.ml_runtime.anomalies import (
        count_weekly_slices,
        iso_week_label,
        score_weekly_slices,
    )

    with domain_con(pack_id) as con:
        try:
            cur = con.execute(
                "SELECT received_at, occurred_at, category, entity_2,"
                " COALESCE(event_count, 1) AS event_count FROM records "
                "WHERE provenance IS NULL OR provenance NOT IN ('inferred', 'fixture')"
            )
            cols = [d[0] for d in cur.description]
            records = [dict(zip(cols, r)) for r in cur.fetchall()]
        except Exception:
            records = []
    counts = count_weekly_slices(records)
    flight = None
    if in_flight:
        try:
            flight = {
                "iso_week": iso_week_label(utc_now()),
                "category": category or "",
                "entity_2": entity_2 or "",
            }
        except Exception:
            flight = None
    scored = score_weekly_slices(counts, pack_id=pack_id, in_flight=flight)
    hits = [
        r
        for r in scored
        if r.get("is_anomaly")
        and (not category or r.get("category") == category)
        and (not entity_2 or r.get("entity_2") == entity_2)
    ]
    hits.sort(key=lambda r: float(r.get("z_score") or 0), reverse=True)
    top = hits[0] if hits else None
    return {
        "anomalous": bool(top),
        "slice": top,
        "scored": len(scored),
        "in_flight_counted": bool(flight),
    }


def _slice_token(category: str | None, entity_2: str | None, entity_3: str | None) -> str:
    return "|".join(
        str(part or "").strip().upper() for part in (category, entity_2, entity_3)
    )


def _select_open_investigation(
    con,
    pack_id: str,
    cluster_id: int,
    category: str | None,
    entity_2: str | None,
    entity_3: str | None,
):
    """One open investigation for this identity.

    A real cluster id is the identity. Cluster 0 is not: every unmatched
    slice would otherwise collapse into the same row. Those are keyed by
    category, make, and model.
    """
    if int(cluster_id or 0) > 0:
        return con.execute(
            """
            SELECT investigation_id FROM investigations
            WHERE pack_id = ? AND cluster_id = ? AND status = 'open'
            LIMIT 1
            """,
            [pack_id, int(cluster_id)],
        ).fetchone()
    return con.execute(
        """
        SELECT investigation_id FROM investigations
        WHERE pack_id = ? AND cluster_id = 0 AND status = 'open'
          AND UPPER(COALESCE(slice_category, '')) = UPPER(?)
          AND UPPER(COALESCE(slice_entity_2, '')) = UPPER(?)
          AND UPPER(COALESCE(slice_entity_3, '')) = UPPER(?)
        LIMIT 1
        """,
        [pack_id, category or "", entity_2 or "", entity_3 or ""],
    ).fetchone()


def find_open_investigation(
    pack_id: str,
    cluster_id: int,
    category: str | None = None,
    entity_2: str | None = None,
    entity_3: str | None = None,
) -> str | None:
    with ops_con(read_only=True) as con:
        try:
            row = _select_open_investigation(
                con, pack_id, cluster_id, category, entity_2, entity_3
            )
        except Exception:
            return None
    return str(row[0]) if row else None


def open_or_link_investigation(
    *,
    pack_id: str,
    cluster_id: int = 0,
    title: str | None = None,
    category: str | None = None,
    entity_2: str | None = None,
    entity_3: str | None = None,
) -> str:
    """Open or link an investigation, fleet-safe (audit 4.5).

    The slice claim (``inv:{pack}:{cluster}``) guarantees exactly one opener
    across live intercept, CaseAgent, and the fleet scan — in-process locks
    are not enough when a scan process races an API worker. Claim losers
    re-read and link to the winner's row. Cluster 0 claims include the
    slice, so two unmatched components do not share one investigation.
    """
    import os as _os

    cid = int(cluster_id or 0)
    if cid > 0:
        claim_key = f"inv:{pack_id}:{cid}"
    else:
        claim_key = f"inv:{pack_id}:0:{_slice_token(category, entity_2, entity_3)}"
    owner = f"opener-{_os.getpid()}"
    try:
        from src.jobs.registry import acquire_slice_claim

        claimed = acquire_slice_claim(claim_key, owner)
    except Exception:
        claimed = True
    with ops_con() as con:
        row = _select_open_investigation(
            con, pack_id, cid, category, entity_2, entity_3
        )
        if row:
            return str(row[0])
        if not claimed:
            # Lost the claim race after re-check: the winner usually commits
            # within milliseconds — poll briefly, then link. Standing down
            # without linking would strand the contact's evidence.
            import time as _time

            for _ in range(10):
                _time.sleep(0.05)
                row = _select_open_investigation(
                    con, pack_id, cid, category, entity_2, entity_3
                )
                if row:
                    return str(row[0])
            raise RuntimeError(
                "investigation open claimed by another worker; retry to link"
            )
        seq = con.execute("SELECT nextval('investigation_seq')").fetchone()
        inv_id = f"inv_{int(seq[0]):04d}" if seq else "inv_" + new_ulid()[:8]
        con.execute(
            """
            INSERT INTO investigations
            (investigation_id, pack_id, cluster_id, title, status,
             opened_at, last_case_at, case_count,
             slice_category, slice_entity_2, slice_entity_3)
            VALUES (?, ?, ?, ?, 'open', ?, ?, 1, ?, ?, ?)
            """,
            [
                inv_id,
                pack_id,
                cid,
                title or f"Live intercept cluster {cid}",
                utc_now(),
                utc_now(),
                category or None,
                entity_2 or None,
                entity_3 or None,
            ],
        )
    return inv_id


def _slice_key(pack_id: str, cat: str | None, ent: str | None) -> str:
    return f"{pack_id}|{(cat or '').upper()}|{(ent or '').upper()}"


def _slice_cooled_down(pack_id: str, cat: str | None, ent: str | None, cooldown_s: int) -> bool:
    """True when this slice fired within the cooldown window (board).

    Records the firing timestamp on a miss so exactly one firing per window
    proceeds. Best-effort: DB failure means "not cooled" (fail open toward
    alerting, never toward silence).
    """
    from datetime import timedelta as _td

    from src.data.timeutil import utc_now as _now

    key = _slice_key(pack_id or "", cat, ent)
    now = _now().replace(tzinfo=None)
    try:
        with ops_con() as con:
            row = con.execute(
                "SELECT last_fired_at FROM intercept_cooldown WHERE slice_key = ?",
                [key],
            ).fetchone()
            if row and row[0] is not None:
                try:
                    last = row[0].replace(tzinfo=None)
                except Exception:
                    last = None
                if last is not None and (now - last) < _td(seconds=max(1, cooldown_s)):
                    return True
            con.execute(
                "INSERT OR REPLACE INTO intercept_cooldown (slice_key, last_fired_at)"
                " VALUES (?, ?)",
                [key, now],
            )
    except Exception:
        return False
    return False


def intercept_contact(ctx: Any) -> dict[str, Any]:
    """Re-score anomalies for this contact's slice; open/link investigation if spiked.

    Includes the in-flight contact (+1): a z-anomaly OR a count threshold
    crossing (live cases + this call >= FRONTLINE_INVESTIGATION_MIN_CASES)
    opens/links within the same call.
    """
    pack_id = ctx.pack.id if getattr(ctx, "pack", None) else None
    if not pack_id:
        return {"anomalous": False, "investigation_id": None}
    cat = (ctx.slots or {}).get("category")
    ent = (ctx.slots or {}).get("entity_2")
    result = slice_is_anomalous(pack_id, cat, ent)
    result["in_flight_counted"] = True
    try:
        from src.qubot.retrievers import live_risk

        result["live_risk"] = [
            r for r in live_risk(7) if not pack_id or r.get("pack_id") == pack_id
        ]
    except Exception:
        result["live_risk"] = []
    inv_id = None
    # Count-threshold tip-over: live cases in slice + this in-flight call.
    threshold_hit = False
    try:
        if cat:
            import os as _os

            try:
                min_cases = int(_os.getenv("FRONTLINE_INVESTIGATION_MIN_CASES", "3"))
            except ValueError:
                min_cases = 3
            with ops_con() as _con:
                try:
                    _row = _con.execute(
                        """
                        SELECT COUNT(*) FROM cases
                        WHERE pack_id = ? AND category = ?
                          AND created_at >= now() - INTERVAL '7 days'
                        """,
                        [pack_id, cat],
                    ).fetchone()
                    _live = int(_row[0]) if _row else 0
                except Exception:
                    _live = 0
            if _live + 1 >= max(1, min_cases):
                threshold_hit = True
                result["threshold_hit"] = True
                result["live_case_count"] = _live
    except Exception:
        pass
    if result.get("anomalous") or threshold_hit:
        # Per-slice cooldown (board): one firing opens/links + alerts; repeat
        # contacts inside the cooldown window link silently instead of
        # spamming investigations on small counts.
        import os as _os2

        try:
            _cd = int(_os2.getenv("FRONTLINE_INTERCEPT_COOLDOWN_S", "3600"))
        except ValueError:
            _cd = 3600
        if _slice_cooled_down(pack_id, cat, ent, _cd):
            result["investigation_id"] = None
            result["cooled_down"] = True
            return result
        cluster_id = 0
        if getattr(ctx, "investigation_brief", None) and ctx.investigation_brief.get("cluster_id") is not None:
            cluster_id = int(ctx.investigation_brief["cluster_id"])
        ent3 = (ctx.slots or {}).get("entity_3")
        inv_id = open_or_link_investigation(
            pack_id=pack_id,
            cluster_id=cluster_id,
            title=f"Live intercept {cat or ''} {ent or ''}".strip(),
            category=cat,
            entity_2=ent,
            entity_3=ent3,
        )
        ctx.investigation_id = inv_id
        record_action(
            AgentAction(
                interaction_id=ctx.interaction_id,
                agent="investigator",
                action_type="live_intercept",
                input_summary=f"slice={cat}/{ent}",
                output_summary=f"investigation_id={inv_id}; anomalous=true",
                evidence_ids=[inv_id],
                ok=True,
            )
        )
    result["investigation_id"] = inv_id
    return result


__all__ = [
    "slice_is_anomalous",
    "find_open_investigation",
    "open_or_link_investigation",
    "intercept_contact",
]
