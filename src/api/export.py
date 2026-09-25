"""Audit export builder — ledger + auditor verdicts for a date range.

Used by GET /api/frontline/audits/export. Pure functions so tests can call
``build_audit_export`` without HTTP.

Supports JSON (default) and CSV, each with a tamper-evident manifest
(count + SHA256 of the export body bytes).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import date, datetime, time, timezone
from typing import Any

from src.data.warehouse import ops_con
from src.qubot.auditor import REPORTS_DIR, audit_interaction


def _parse_day(day: str, end_of_day: bool = False) -> datetime:
    d = date.fromisoformat(day)
    if end_of_day:
        return datetime.combine(d, time(23, 59, 59), tzinfo=timezone.utc)
    return datetime.combine(d, time(0, 0, 0), tzinfo=timezone.utc)


def build_manifest(body: bytes | str, *, count: int) -> dict[str, Any]:
    """Tamper-evident export manifest: row count + SHA256 of body bytes."""
    raw = body.encode("utf-8") if isinstance(body, str) else body
    return {
        "count": count,
        "byte_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "algorithm": "sha256",
    }


async def build_audit_export(
    *,
    start: str | None = None,
    end: str | None = None,
    interaction_ids: list[str] | None = None,
    limit: int = 50,
    include_markdown: bool = False,
    run_auditor: bool = True,
    redact_pii: bool = True,
) -> dict[str, Any]:
    """Build a JSON-serializable export of contacts + actions + audit verdicts.

    Parameters
    ----------
    start, end:
        Inclusive ISO dates (YYYY-MM-DD). If both omitted and no interaction_ids,
        exports the most recent ``limit`` interactions.
    interaction_ids:
        Optional explicit list; when set, date filters are ignored.
    redact_pii:
        When True (default), emails/phones/SSN/cards in summaries and turns
        are replaced with [EMAIL]/[PHONE]/[SSN]/[CARD] via security.pii.
        Pass False only for privileged DSR/legal export with strict auth.
    """
    params: list[Any] = []
    sql = """
        SELECT interaction_id, pack_id, pack_version, started_at, ended_at,
               channel, status, outcome, supervised, peak_frustration,
               entity_1, entity_2, entity_3
        FROM interactions
    """
    if interaction_ids:
        placeholders = ", ".join("?" for _ in interaction_ids)
        sql += f" WHERE interaction_id IN ({placeholders})"
        params.extend(interaction_ids)
    elif start or end:
        clauses = []
        if start:
            clauses.append("started_at >= ?")
            params.append(_parse_day(start, end_of_day=False))
        if end:
            clauses.append("started_at <= ?")
            params.append(_parse_day(end, end_of_day=True))
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY started_at DESC LIMIT ?"
    params.append(min(max(limit, 1), 500))

    # Load rows with a single connection, then close before auditor opens its own
    # DuckDB handles (avoids "different configuration" connection conflicts).
    with ops_con(read_only=True) as con:
        cur = con.execute(sql, params)
        cols = [d[0] for d in cur.description]
        interactions = [dict(zip(cols, row)) for row in cur.fetchall()]

        for row in interactions:
            iid = row["interaction_id"]
            for k, v in list(row.items()):
                if isinstance(v, datetime):
                    row[k] = v.isoformat()

            act_cur = con.execute(
                """
                SELECT action_id, agent, action_type, input_summary, output_summary,
                       evidence_ids, ok, error, duration_ms, ts
                FROM agent_actions
                WHERE interaction_id = ?
                ORDER BY ts
                """,
                [iid],
            )
            a_cols = [d[0] for d in act_cur.description]
            actions = []
            for a in act_cur.fetchall():
                ad = dict(zip(a_cols, a))
                if isinstance(ad.get("ts"), datetime):
                    ad["ts"] = ad["ts"].isoformat()
                ev = ad.get("evidence_ids")
                if isinstance(ev, str):
                    try:
                        ad["evidence_ids"] = json.loads(ev)
                    except json.JSONDecodeError:
                        ad["evidence_ids"] = []
                actions.append(ad)
            row["actions"] = actions

            report_path = REPORTS_DIR / f"{iid}.md"
            try:
                from src.security.identifiers import assert_under_roots as _assert_roots

                report_path = _assert_roots(report_path, [REPORTS_DIR])
            except Exception:
                row["report_name"] = None
                row["report_url"] = f"/api/frontline/audits/{iid}"
                continue
            row["report_name"] = f"{iid}.md" if report_path.is_file() else None
            row["report_url"] = f"/api/frontline/audits/{iid}"
            if include_markdown and report_path.exists():
                row["report_markdown"] = report_path.read_text(encoding="utf-8")

    # PII redaction (was dead code in security.pii — now wired into exports)
    if redact_pii:
        try:
            from src.security.pii import redact_pii as _redact

            for row in interactions:
                for k in ("entity_1", "entity_2", "entity_3"):
                    if isinstance(row.get(k), str):
                        row[k] = _redact(row[k])
                for a in row.get("actions") or []:
                    for fk in ("input_summary", "output_summary", "error"):
                        if isinstance(a.get(fk), str):
                            a[fk] = _redact(a[fk])
                if isinstance(row.get("report_markdown"), str):
                    row["report_markdown"] = _redact(row["report_markdown"])
        except Exception:
            pass

    for row in interactions:
        iid = row["interaction_id"]
        if run_auditor:
            try:
                result = await audit_interaction(iid, write_report=False)
                row["audit"] = {
                    "overall_verdict": result.overall_verdict,
                    "total_actions": result.total_actions,
                    "grounded_actions": result.grounded_actions,
                    "unverifiable_actions": result.unverifiable_actions,
                    "mismatch_actions": result.mismatch_actions,
                    "peak_frustration": result.peak_frustration,
                    "severity_sane": result.severity_sane,
                }
            except Exception as e:
                row["audit"] = {
                    "overall_verdict": "error",
                    "error": type(e).__name__,
                }
        else:
            row["audit"] = None

    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "start": start,
        "end": end,
        "count": len(interactions),
        "interactions": interactions,
        "format": "json",
    }
    # Canonical body for hashing: interactions list + count (stable, no wall clock).
    canonical = json.dumps(
        {"count": payload["count"], "interactions": interactions},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    payload["manifest"] = build_manifest(canonical, count=payload["count"])
    return payload


def _csv_neutralize(value: object) -> object:
    """Prefix formula-leading cells so Excel/Sheets will not execute them."""
    if isinstance(value, str) and value and value[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def export_to_csv(payload: dict[str, Any]) -> str:
    """Flatten export interactions into CSV (one row per contact)."""
    buf = io.StringIO()
    fields = [
        "interaction_id",
        "pack_id",
        "channel",
        "status",
        "outcome",
        "started_at",
        "ended_at",
        "entity_1",
        "entity_2",
        "entity_3",
        "supervised",
        "peak_frustration",
        "action_count",
        "audit_verdict",
        "audit_total_actions",
        "audit_grounded_actions",
        "audit_mismatch_actions",
    ]
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in payload.get("interactions") or []:
        audit = row.get("audit") or {}
        writer.writerow({
            "interaction_id": _csv_neutralize(row.get("interaction_id", "")),
            "pack_id": _csv_neutralize(row.get("pack_id", "")),
            "channel": _csv_neutralize(row.get("channel", "")),
            "status": _csv_neutralize(row.get("status", "")),
            "outcome": _csv_neutralize(row.get("outcome", "")),
            "started_at": _csv_neutralize(row.get("started_at", "")),
            "ended_at": _csv_neutralize(row.get("ended_at", "")),
            "entity_1": _csv_neutralize(row.get("entity_1", "")),
            "entity_2": _csv_neutralize(row.get("entity_2", "")),
            "entity_3": _csv_neutralize(row.get("entity_3", "")),
            "supervised": _csv_neutralize(row.get("supervised", "")),
            "peak_frustration": _csv_neutralize(row.get("peak_frustration", "")),
            "action_count": len(row.get("actions") or []),
            "audit_verdict": _csv_neutralize(audit.get("overall_verdict", "")),
            "audit_total_actions": audit.get("total_actions", ""),
            "audit_grounded_actions": audit.get("grounded_actions", ""),
            "audit_mismatch_actions": audit.get("mismatch_actions", ""),
        })
    return buf.getvalue()


async def build_audit_export_csv(
    *,
    start: str | None = None,
    end: str | None = None,
    interaction_ids: list[str] | None = None,
    limit: int = 50,
    run_auditor: bool = True,
) -> dict[str, Any]:
    """CSV export + manifest over the CSV body bytes."""
    payload = await build_audit_export(
        start=start,
        end=end,
        interaction_ids=interaction_ids,
        limit=limit,
        include_markdown=False,
        run_auditor=run_auditor,
    )
    csv_body = export_to_csv(payload)
    return {
        "format": "csv",
        "exported_at": payload["exported_at"],
        "start": start,
        "end": end,
        "count": payload["count"],
        "csv": csv_body,
        "manifest": build_manifest(csv_body, count=payload["count"]),
    }


__all__ = [
    "build_audit_export",
    "build_audit_export_csv",
    "export_to_csv",
    "build_manifest",
]
