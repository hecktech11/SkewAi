"""Pilot ops helpers — case lifecycle, notes, investigation status, metrics, CSV.

Keeps HTTP routes thin. All writes ledger before/with the mutation where
applicable. Never invents CRM workflows: status enums are the pack-agnostic
pilot set documented on the schema.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any

from src.data.warehouse import ops_con
from src.ids import new_ulid
from src.ledger import AgentAction, record_action

CASE_STATUSES = frozenset({"open", "pending_followup", "closed"})
INV_STATUSES = frozenset({"open", "monitoring", "closed"})


def _now() -> datetime:
    from src.data.timeutil import utc_now

    return utc_now()


def _parse_json_field(row: dict[str, Any], key: str) -> None:
    v = row.get(key)
    if isinstance(v, str):
        try:
            row[key] = json.loads(v)
        except Exception:
            pass


def get_case_row(case_id: str) -> dict[str, Any] | None:
    with ops_con(read_only=True) as con:
        cur = con.execute("SELECT * FROM cases WHERE case_id = ?", [case_id])
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        d = dict(zip(cols, row))
    _parse_json_field(d, "safety_flags")
    from src.security.pii import decrypt_case_row

    return decrypt_case_row(d)


def update_case(
    case_id: str,
    *,
    status: str | None = None,
    followup_draft: str | None = None,
    author: str = "operator",
) -> dict[str, Any]:
    """Update case status and/or follow-up draft. Raises ValueError on bad input."""
    row = get_case_row(case_id)
    if row is None:
        raise LookupError(f"case not found: {case_id}")

    from src.security.sql_ident import SAFE_CASE_UPDATE_FIELDS, safe_ident

    fields: list[str] = []
    params: list[Any] = []
    if status is not None:
        if status not in CASE_STATUSES:
            raise ValueError(f"invalid status {status!r}; allowed: {sorted(CASE_STATUSES)}")
        col = safe_ident("status", SAFE_CASE_UPDATE_FIELDS, kind="column")
        fields.append(f"{col} = ?")
        params.append(status)
    if followup_draft is not None:
        col = safe_ident("followup_draft", SAFE_CASE_UPDATE_FIELDS, kind="column")
        fields.append(f"{col} = ?")
        params.append(followup_draft)
    if not fields:
        return row

    params.append(case_id)
    with ops_con() as con:
        con.execute(f"UPDATE cases SET {', '.join(fields)} WHERE case_id = ?", params)

    iid = row.get("interaction_id") or "platform"
    if status is not None and status != row.get("status"):
        record_action(
            AgentAction(
                interaction_id=iid,
                agent="supervisor",
                action_type="case_status_updated",
                input_summary=f"from={row.get('status')}",
                output_summary=f"to={status}; by={author}",
                case_id=case_id,
            )
        )
    if followup_draft is not None:
        record_action(
            AgentAction(
                interaction_id=iid,
                agent="supervisor",
                action_type="followup_saved",
                input_summary=f"by={author}",
                output_summary=(followup_draft or "")[:500],
                case_id=case_id,
            )
        )
    updated = get_case_row(case_id)
    assert updated is not None
    return updated


def add_case_note(
    case_id: str,
    body: str,
    *,
    author: str = "operator",
) -> dict[str, Any]:
    row = get_case_row(case_id)
    if row is None:
        raise LookupError(f"case not found: {case_id}")
    text = (body or "").strip()
    if not text:
        raise ValueError("note body required")
    if len(text) > 4000:
        text = text[:4000]
    note_id = "note_" + new_ulid()
    now = _now()
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO case_notes (note_id, case_id, author, body, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [note_id, case_id, (author or "operator")[:80], text, now],
        )
    record_action(
        AgentAction(
            interaction_id=row.get("interaction_id") or "platform",
            agent="supervisor",
            action_type="case_note_added",
            input_summary=f"author={author}",
            output_summary=text[:500],
            case_id=case_id,
        )
    )
    from src.data.timeutil import to_iso_z

    return {
        "note_id": note_id,
        "case_id": case_id,
        "author": author or "operator",
        "body": text,
        "created_at": to_iso_z(now),
    }


def list_case_notes(case_id: str, limit: int = 50) -> list[dict[str, Any]]:
    limit = min(max(int(limit), 1), 200)
    with ops_con(read_only=True) as con:
        cur = con.execute(
            """
            SELECT note_id, case_id, author, body, created_at
            FROM case_notes
            WHERE case_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            [case_id, limit],
        )
        cols = [d[0] for d in cur.description]
        from src.data.timeutil import to_iso_z

        rows = []
        for r in cur.fetchall():
            d = dict(zip(cols, r))
            if isinstance(d.get("created_at"), datetime):
                d["created_at"] = to_iso_z(d["created_at"])
            rows.append(d)
    return rows


def _resolve_issue_slice(pack_id: str, cluster_id: Any) -> tuple[Any, Any, Any]:
    """Modal (category, entity_2, entity_3) across a cluster's member records.

    Investigation titles are display text; fix measurement must use the
    canonical slice or before/after match nothing.
    """
    try:
        cid = int(cluster_id)
    except (TypeError, ValueError):
        return None, None, None
    try:
        from src.data.warehouse import domain_con as _dcon

        with _dcon(pack_id, read_only=True) as con:
            row = con.execute(
                """
                SELECT r.category, r.entity_2, r.entity_3, COUNT(*) AS n
                FROM cluster_assignments a
                JOIN records r ON r.record_id = a.record_id
                WHERE a.cluster_id = ?
                GROUP BY 1, 2, 3 ORDER BY n DESC LIMIT 1
                """,
                [cid],
            ).fetchone()
    except Exception:
        return None, None, None
    if not row:
        return None, None, None
    return row[0], row[1], row[2]


def update_investigation(
    investigation_id: str,
    *,
    status: str,
    author: str = "operator",
) -> dict[str, Any]:
    if status not in INV_STATUSES:
        raise ValueError(f"invalid status {status!r}; allowed: {sorted(INV_STATUSES)}")
    with ops_con(read_only=True) as con:
        cur = con.execute(
            "SELECT investigation_id, status, pack_id, title FROM investigations WHERE investigation_id = ?",
            [investigation_id],
        )
        row = cur.fetchone()
        if not row:
            raise LookupError(f"investigation not found: {investigation_id}")
        cols = [d[0] for d in cur.description]
        before = dict(zip(cols, row))
    with ops_con() as con:
        con.execute(
            "UPDATE investigations SET status = ? WHERE investigation_id = ?",
            [status, investigation_id],
        )
    record_action(
        AgentAction(
            interaction_id="platform",
            agent="supervisor",
            action_type="investigation_status_updated",
            input_summary=f"from={before.get('status')}; inv={investigation_id}",
            output_summary=f"to={status}; by={author}",
            evidence_ids=[investigation_id],
        )
    )
    with ops_con(read_only=True) as con:
        cur = con.execute(
            "SELECT * FROM investigations WHERE investigation_id = ?",
            [investigation_id],
        )
        r = cur.fetchone()
        cols = [d[0] for d in cur.description]
        out = dict(zip(cols, r))
    # Keep raw datetimes: the response ISO-formats below, but fix
    # measurement needs real datetimes (iso strings crash to_naive_utc,
    # which silently dropped every recorded fix).
    _fixed_raw = out.get("last_case_at") or out.get("opened_at")
    for k, v in list(out.items()):
        if isinstance(v, datetime):
            out[k] = v.isoformat()
    # Close the fix-effectiveness loop: closing an investigation records a fix
    # and attaches before/after recurrence so "prove fixes worked" is automatic.
    if status == "closed":
        try:
            from src.enterprise.fix_effectiveness import measure_effectiveness, record_fix

            _pack = str(out.get("pack_id") or "automotive_nhtsa")
            _cat, _e2, _e3 = _resolve_issue_slice(_pack, out.get("cluster_id"))
            fix = record_fix(
                pack_id=_pack,
                fixed_at=_fixed_raw,
                category=_cat,
                entity_2=_e2,
                entity_3=_e3,
                investigation_id=investigation_id,
                note=f"investigation {investigation_id} closed by {author}",
            )
            try:
                meas = measure_effectiveness(
                    pack_id=_pack,
                    fixed_at=_fixed_raw,
                    category=_cat,
                    entity_2=_e2,
                    entity_3=_e3,
                    window_days=30,
                )
            except Exception:
                meas = {}
            out["fix_effectiveness"] = {"fix": fix, "measured": meas}
        except Exception:
            pass
    return out


def _csv_neutralize(value: str) -> str:
    """Prefix formula-leading cells so Excel/Sheets will not execute them."""
    if value and value[0] in ("=", "+", "-", "@", "\t"):
        return "'" + value
    return value


def build_cases_csv(
    *,
    status: str | None = None,
    severity: str | None = None,
    q: str | None = None,
    cluster: str | None = None,
    limit: int = 500,
    scrub_pii: bool = True,
    include_simulated: bool = False,
) -> tuple[str, int]:
    """Return (csv_text, row_count) for case export.

    ``scrub_pii=True`` (default) redacts PII in free-text columns; callers
    must gate ``scrub_pii=False`` behind ``dsr:export`` (item 16).
    """
    limit = min(max(int(limit), 1), 2000)
    sql = """
        SELECT case_id, interaction_id, pack_id, created_at, category,
               description_summary, severity, priority, status, case_kind,
               cluster_match_id, investigation_id, advisory_match_id
        FROM cases
    """
    params: list[Any] = []
    clauses: list[str] = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if severity:
        clauses.append("severity = ?")
        params.append(severity)
    if q and q.strip():
        like = f"%{q.strip()}%"
        clauses.append(
            "(case_id ILIKE ? OR interaction_id ILIKE ? OR category ILIKE ? "
            "OR description_summary ILIKE ? OR investigation_id ILIKE ?)"
        )
        params.extend([like, like, like, like, like])
    if cluster and str(cluster).strip():
        clauses.append("CAST(cluster_match_id AS VARCHAR) = ?")
        params.append(str(cluster).strip())
    if not include_simulated:
        clauses.append(
            "(interaction_id IS NULL OR interaction_id NOT IN "
            "(SELECT interaction_id FROM interactions WHERE channel = 'simulated'))"
        )
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with ops_con(read_only=True) as con:
        cur = con.execute(sql, params)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    from src.security.pii import decrypt_case_rows

    rows = decrypt_case_rows(rows)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    if scrub_pii:
        from src.security.pii import redact_dict

        rows = [redact_dict(r) for r in rows]
    for row in rows:
        for k, v in list(row.items()):
            if isinstance(v, datetime):
                row[k] = v.isoformat()
            elif isinstance(v, str):
                row[k] = _csv_neutralize(v)
        writer.writerow(row)
    return buf.getvalue(), len(rows)


def ops_metrics(*, window_days: int = 7, include_simulated: bool = False) -> dict[str, Any]:
    """Pilot ops snapshot for dashboard header / health-adjacent views."""
    window_days = min(max(int(window_days), 1), 90)
    sim_ix = "" if include_simulated else " AND COALESCE(channel, '') <> 'simulated'"
    sim_case = (
        ""
        if include_simulated
        else (
            " AND interaction_id NOT IN "
            "(SELECT interaction_id FROM interactions WHERE channel = 'simulated')"
        )
    )
    with ops_con(read_only=True) as con:
        case_open = con.execute(
            f"SELECT COUNT(*) FROM cases WHERE status = 'open'{sim_case}"
        ).fetchone()[0]
        case_pending = con.execute(
            f"SELECT COUNT(*) FROM cases WHERE status = 'pending_followup'{sim_case}"
        ).fetchone()[0]
        case_closed = con.execute(
            f"SELECT COUNT(*) FROM cases WHERE status = 'closed'{sim_case}"
        ).fetchone()[0]
        case_critical = con.execute(
            f"SELECT COUNT(*) FROM cases WHERE status != 'closed' AND severity = 'Critical'{sim_case}"
        ).fetchone()[0]
        inv_open = con.execute(
            "SELECT COUNT(*) FROM investigations WHERE status = 'open'"
        ).fetchone()[0]
        inv_mon = con.execute(
            "SELECT COUNT(*) FROM investigations WHERE status = 'monitoring'"
        ).fetchone()[0]
        active_ix = con.execute(
            f"SELECT COUNT(*) FROM interactions WHERE status = 'active'{sim_ix}"
        ).fetchone()[0]
        cases_window = con.execute(
            f"""
            SELECT COUNT(*) FROM cases
            WHERE created_at >= now() - INTERVAL (? || ' days'){sim_case}
            """,
            [str(window_days)],
        ).fetchone()[0]
        try:
            dl_pending = con.execute(
                "SELECT COUNT(*) FROM alert_dead_letter WHERE status = 'pending'"
            ).fetchone()[0]
        except Exception:
            dl_pending = 0
        try:
            conn_pending = con.execute(
                "SELECT COUNT(*) FROM connector_deliveries WHERE status = 'pending'"
            ).fetchone()[0]
        except Exception:
            conn_pending = 0
        notes_window = con.execute(
            """
            SELECT COUNT(*) FROM case_notes
            WHERE created_at >= now() - INTERVAL (? || ' days')
            """,
            [str(window_days)],
        ).fetchone()[0]

    return {
        "window_days": window_days,
        "cases": {
            "open": int(case_open),
            "pending_followup": int(case_pending),
            "closed": int(case_closed),
            "critical_open": int(case_critical),
            "created_in_window": int(cases_window),
        },
        "investigations": {
            "open": int(inv_open),
            "monitoring": int(inv_mon),
        },
        "interactions_active": int(active_ix),
        "alert_dead_letters_pending": int(dl_pending),
        "connector_deliveries_pending": int(conn_pending),
        "case_notes_in_window": int(notes_window),
        "fix_loop": _fix_loop_stats(window_days=window_days),
        "ts": _now().isoformat(),
    }


def _fix_loop_stats(*, window_days: int = 7) -> dict[str, Any]:
    """Before/after fix effectiveness + reopen rate (item 37).

    Distinguishes before-fix volume, after-fix volume, resolved (improved),
    and reopened (fix recorded but investigation open/monitoring again).
    Bounded: measures at most the 10 most recent fixes.
    """
    try:
        with ops_con(read_only=True) as con:
            try:
                fixes = con.execute(
                    """
                    SELECT fix_id, pack_id, investigation_id, category,
                           entity_2, entity_3, fixed_at
                    FROM recorded_fixes
                    ORDER BY fixed_at DESC LIMIT 10
                    """
                ).fetchall()
            except Exception:
                fixes = []
            open_with_fix = 0
            try:
                open_with_fix = con.execute(
                    """
                    SELECT COUNT(*) FROM investigations i
                    WHERE i.status IN ('open', 'monitoring')
                      AND EXISTS (
                          SELECT 1 FROM recorded_fixes f
                          WHERE f.investigation_id = i.investigation_id
                      )
                    """
                ).fetchone()[0]
            except Exception:
                open_with_fix = 0
    except Exception:
        return {"fixes_recorded": 0, "reopened": 0, "reopen_rate": 0.0,
                "resolved": 0, "series": []}
    series = []
    resolved = 0
    try:
        from src.enterprise.fix_effectiveness import measure_effectiveness

        for fix_id, pack_id, _inv, category, e2, e3, fixed_at in fixes:
            try:
                meas = measure_effectiveness(
                    pack_id=str(pack_id or "automotive_nhtsa"),
                    fixed_at=fixed_at,
                    category=str(category or ""),
                    entity_2=str(e2 or "") or None,
                    entity_3=str(e3 or "") or None,
                    window_days=30,
                )
            except Exception:
                continue
            improved = bool(meas.get("improved"))
            resolved += 1 if improved else 0
            series.append({
                "fix_id": fix_id,
                "category": category,
                "before_count": meas.get("before_count"),
                "after_count": meas.get("after_count"),
                "before_rate": meas.get("before_rate"),
                "after_rate": meas.get("after_rate"),
                "improved": improved,
            })
    except Exception:
        pass
    total = len(fixes)
    reopened = int(open_with_fix or 0)
    return {
        "fixes_recorded": total,
        "resolved": resolved,
        "reopened": reopened,
        "reopen_rate": round(reopened / total, 3) if total else 0.0,
        "series": series,
    }


__all__ = [
    "CASE_STATUSES",
    "INV_STATUSES",
    "get_case_row",
    "update_case",
    "add_case_note",
    "list_case_notes",
    "update_investigation",
    "build_cases_csv",
    "ops_metrics",
]
