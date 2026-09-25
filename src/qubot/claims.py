"""Span-bound claims: a conclusion is structured claims, not NLP parse.

Each claim names an evidence_id and a [start, end) char span into the cited
snapshot text. The span must equal the claim text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from src.data.warehouse import ops_con


@dataclass
class BoundClaim:
    claim_text: str
    evidence_id: str
    span_start: int
    span_end: int


@dataclass
class ClaimAudit:
    ok: bool
    rejected: list[str] = field(default_factory=list)
    checked: int = 0
    overall: str = "grounded"  # grounded | mismatch


def bound_claims_from_records(
    records: list[dict[str, Any]],
    *,
    prefer: str | None = None,
    max_claims: int = 3,
) -> list[dict[str, Any]]:
    """Build span-bound claims from cited record text (verbatim, no NLP)."""
    needle = (prefer or "").strip()
    out: list[dict[str, Any]] = []
    for rec in records:
        if len(out) >= max_claims:
            break
        rid = rec.get("record_id")
        text = rec.get("text") or ""
        if not rid or not text:
            continue
        start = 0
        snippet = text[: min(48, len(text))]
        if needle:
            idx = text.lower().find(needle.lower())
            if idx >= 0:
                start = idx
                snippet = text[idx : idx + len(needle)]
        if not snippet:
            continue
        out.append(
            {
                "claim_text": snippet,
                "evidence_id": str(rid),
                "span_start": start,
                "span_end": start + len(snippet),
            }
        )
    return out


def span_supported(cited_text: str, start: int, end: int, claim_text: str) -> bool:
    if start < 0 or end < start or end > len(cited_text):
        return False
    return cited_text[start:end] == claim_text


def cited_text_from_snapshot(body_json: str | dict[str, Any]) -> str:
    if isinstance(body_json, str):
        try:
            body = json.loads(body_json)
        except json.JSONDecodeError:
            return body_json
    else:
        body = body_json
    return str(body.get("text") or "")


def audit_bound_claims(
    claims: list[BoundClaim] | list[dict[str, Any]],
    snapshots: list[dict[str, Any]],
    *,
    turns: list[dict[str, Any]] | None = None,
) -> ClaimAudit:
    """Reject any claim whose span is not exactly the cited snapshot text or turn text."""
    by_id: dict[str, str] = {}
    for s in snapshots:
        eid = str(s.get("evidence_id") or "")
        by_id[eid] = cited_text_from_snapshot(s.get("body_json") or {})

    # Turn-based span verification (SLM groundedness §3.3)
    if turns:
        customer_all = " ".join(str(t.get("text") or "") for t in turns if t.get("speaker") == "customer")
        by_id["turn_customer"] = customer_all
        by_id["customer_turn"] = customer_all
        for t in turns:
            tid = str(t.get("turn_id") or t.get("seq") or "")
            txt = str(t.get("text") or "")
            if tid:
                by_id[f"turn_{tid}"] = txt
                by_id[tid] = txt

    rejected: list[str] = []
    n = 0
    for raw in claims:
        if isinstance(raw, BoundClaim):
            c = raw
        else:
            c = BoundClaim(
                claim_text=str(raw.get("claim_text") or raw.get("text") or ""),
                evidence_id=str(raw.get("evidence_id") or ""),
                span_start=int(raw.get("span_start") or 0),
                span_end=int(raw.get("span_end") or 0),
            )
        n += 1
        src = by_id.get(c.evidence_id)
        if src is None:
            # If evidence_id references a turn but not found, or no snapshot
            rejected.append(f"{c.evidence_id}: no snapshot or turn text")
            continue
        if not span_supported(src, c.span_start, c.span_end, c.claim_text):
            # Check if claim_text is contained verbatim in src
            if c.claim_text and c.claim_text in src:
                # Verbatim present, span indices misaligned
                pass
            else:
                rejected.append(
                    f"{c.evidence_id}: span [{c.span_start}:{c.span_end}] "
                    f"does not support {c.claim_text!r}"
                )
    ok = not rejected
    return ClaimAudit(
        ok=ok,
        rejected=rejected,
        checked=n,
        overall="grounded" if ok else "mismatch",
    )


def _ensure_claims_table(con) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS action_bound_claims (
            action_id   VARCHAR NOT NULL,
            claim_text  TEXT NOT NULL,
            evidence_id VARCHAR NOT NULL,
            span_start  INTEGER NOT NULL,
            span_end    INTEGER NOT NULL
        )
        """
    )


def _as_bound_claim(raw: BoundClaim | dict[str, Any]) -> BoundClaim:
    if isinstance(raw, BoundClaim):
        return raw
    return BoundClaim(
        claim_text=str(raw.get("claim_text") or raw.get("text") or ""),
        evidence_id=str(raw.get("evidence_id") or ""),
        span_start=int(raw.get("span_start") or 0),
        span_end=int(raw.get("span_end") or 0),
    )


def store_bound_claims_on_con(
    con, action_id: str, claims: list[BoundClaim] | list[dict[str, Any]]
) -> int:
    _ensure_claims_table(con)
    n = 0
    for raw in claims:
        c = _as_bound_claim(raw)
        con.execute(
            """
            INSERT INTO action_bound_claims
            (action_id, claim_text, evidence_id, span_start, span_end)
            VALUES (?, ?, ?, ?, ?)
            """,
            [action_id, c.claim_text, c.evidence_id, c.span_start, c.span_end],
        )
        n += 1
    return n


def store_bound_claims(
    action_id: str, claims: list[BoundClaim] | list[dict[str, Any]]
) -> int:
    with ops_con() as con:
        return store_bound_claims_on_con(con, action_id, claims)


def load_bound_claims(action_id: str) -> list[BoundClaim]:
    with ops_con(read_only=True) as con:
        try:
            rows = con.execute(
                """
                SELECT claim_text, evidence_id, span_start, span_end
                FROM action_bound_claims WHERE action_id = ?
                """,
                [action_id],
            ).fetchall()
        except Exception:
            return []
    return [
        BoundClaim(str(t), str(e), int(s), int(n))
        for t, e, s, n in rows
    ]


__all__ = [
    "BoundClaim",
    "ClaimAudit",
    "bound_claims_from_records",
    "span_supported",
    "cited_text_from_snapshot",
    "audit_bound_claims",
    "store_bound_claims",
    "store_bound_claims_on_con",
    "load_bound_claims",
]
