"""Supervisor review of AI severity raises.  [AI-LAYER]

An "AI raise" is a decision-store row (``ai_layer_events``) with
``kind='severity'`` and ``reason='raised'``: Muse Spark proposed a higher
severity than the rules, the proposal passed the confidence + grounding
checks, and the case got ``max(rule, AI)``.

A supervisor may record ONE verdict per raise: ``accept`` (the raise was
right) or ``reject`` (it was not), with an optional note. Verdicts:

* live in a NEW table ``ai_raise_reviews`` and are audit-logged in a NEW
  append-only, hash-chained table ``ai_raise_review_audit`` (the existing
  ledger and its vocabulary are not touched);
* NEVER alter case severity, priority, safety flags or the decision store.
  Nothing in this module issues an UPDATE/DELETE against ``cases`` or
  ``ai_layer_events``; raise-only severity stays raise-only;
* are role-checked by the route (``supervisor`` only; see
  ``src/api/routes/ai_layer_v2.py``).

Agreement rate = accepted / reviewed. With fewer than ``MIN_REVIEWS_FOR_RATE``
(20) reviews no rate is reported: the status reads
"too few reviews to estimate (n = X)". 20 is a documented choice: below it a
95% Wilson interval on a proportion is wider than about +/-0.2, too wide to
quote.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from src.data.timeutil import utc_now
from src.data.warehouse import ops_con
from src.ids import new_ulid

MIN_REVIEWS_FOR_RATE = 20
VERDICTS = ("accept", "reject")
NOTE_MAX = 500
GENESIS = "0" * 64

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS ai_raise_reviews (
        review_id    VARCHAR PRIMARY KEY,
        event_id     VARCHAR NOT NULL UNIQUE,
        interaction_id VARCHAR,
        verdict      VARCHAR NOT NULL,
        note         VARCHAR,
        reviewer     VARCHAR NOT NULL,
        reviewer_role VARCHAR NOT NULL,
        rule_severity VARCHAR,
        ai_severity  VARCHAR,
        created_at   TIMESTAMP NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ai_raise_review_audit (
        seq          BIGINT PRIMARY KEY,
        audit_id     VARCHAR NOT NULL,
        action       VARCHAR NOT NULL,
        event_id     VARCHAR,
        review_id    VARCHAR,
        actor        VARCHAR NOT NULL,
        actor_role   VARCHAR NOT NULL,
        detail       JSON,
        created_at   TIMESTAMP NOT NULL,
        prev_hash    VARCHAR NOT NULL,
        row_hash     VARCHAR NOT NULL
    )
    """,
]


class ReviewError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def ensure_tables(con) -> None:
    for stmt in _DDL:
        con.execute(stmt)


def _relation_exists(con, name: str) -> bool:
    return con.execute("SELECT 1 FROM information_schema.tables WHERE table_name = ?", [name]).fetchone() is not None


def _json(v: Any) -> Any:
    if isinstance(v, str):
        try:
            return json.loads(v)
        except (TypeError, json.JSONDecodeError):
            return v
    return v


def _row_hash(prev: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((prev + "|" + raw).encode("utf-8")).hexdigest()


def _audit(con, *, action: str, actor: str, actor_role: str, event_id: str | None,
           review_id: str | None, detail: dict[str, Any]) -> str:
    last = con.execute("SELECT seq, row_hash FROM ai_raise_review_audit ORDER BY seq DESC LIMIT 1").fetchone()
    seq = (int(last[0]) + 1) if last else 1
    prev = str(last[1]) if last else GENESIS
    created = utc_now()
    aid = "arv_" + new_ulid()
    payload = {"seq": seq, "audit_id": aid, "action": action, "event_id": event_id, "review_id": review_id,
               "actor": actor, "actor_role": actor_role, "detail": detail, "created_at": str(created)}
    h = _row_hash(prev, payload)
    con.execute(
        "INSERT INTO ai_raise_review_audit (seq, audit_id, action, event_id, review_id, actor, actor_role, detail, "
        "created_at, prev_hash, row_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [seq, aid, action, event_id, review_id, actor, actor_role, json.dumps(detail, default=str), created, prev, h],
    )
    return aid


def verify_audit_chain() -> dict[str, Any]:
    with ops_con() as con:
        if not _relation_exists(con, "ai_raise_review_audit"):
            return {"ok": True, "rows": 0}
        rows = con.execute(
            "SELECT seq, audit_id, action, event_id, review_id, actor, actor_role, detail, created_at, prev_hash, "
            "row_hash FROM ai_raise_review_audit ORDER BY seq"
        ).fetchall()
    prev = GENESIS
    for r in rows:
        payload = {"seq": int(r[0]), "audit_id": r[1], "action": r[2], "event_id": r[3], "review_id": r[4],
                   "actor": r[5], "actor_role": r[6], "detail": _json(r[7]), "created_at": str(r[8])}
        if r[9] != prev or _row_hash(prev, payload) != r[10]:
            return {"ok": False, "rows": len(rows), "broken_at_seq": int(r[0])}
        prev = r[10]
    return {"ok": True, "rows": len(rows)}


def _raise_rows(con, event_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    if not _relation_exists(con, "ai_layer_events"):
        return []
    sql = ("SELECT event_id, interaction_id, case_id, deterministic, ai_proposal, accepted, model_id, created_at "
           "FROM ai_layer_events WHERE kind = 'severity' AND reason = 'raised'")
    args: list[Any] = []
    if event_id:
        sql += " AND event_id = ?"
        args.append(event_id)
    sql += " ORDER BY created_at DESC LIMIT ?"
    args.append(max(1, min(1000, int(limit))))
    out = []
    for r in con.execute(sql, args).fetchall():
        det, prop, acc = _json(r[3]) or {}, _json(r[4]) or {}, _json(r[5]) or {}
        out.append({
            "event_id": r[0], "interaction_id": r[1], "case_id": r[2],
            "rule_severity": (det or {}).get("severity") if isinstance(det, dict) else None,
            "ai_severity": (prop or {}).get("severity") if isinstance(prop, dict) else None,
            "final_severity": (acc or {}).get("severity") if isinstance(acc, dict) else None,
            "ai_reason": str((prop or {}).get("reason") or "")[:200] if isinstance(prop, dict) else "",
            "model_id": r[6], "created_at": str(r[7]),
        })
    return out


def _case_for(con, interaction_id: str | None) -> dict[str, Any] | None:
    if not interaction_id or not _relation_exists(con, "cases"):
        return None
    row = con.execute(
        "SELECT case_id, severity, priority FROM cases WHERE interaction_id = ? ORDER BY created_at DESC LIMIT 1",
        [interaction_id],
    ).fetchone()
    return {"case_id": row[0], "severity": row[1], "priority": row[2]} if row else None


def list_raises(limit: int = 200) -> list[dict[str, Any]]:
    with ops_con() as con:
        raises = _raise_rows(con, limit=limit)
        reviews: dict[str, dict[str, Any]] = {}
        if _relation_exists(con, "ai_raise_reviews") and raises:
            cur = con.execute("SELECT * FROM ai_raise_reviews")
            cols = [d[0] for d in cur.description]
            for r in cur.fetchall():
                d = dict(zip(cols, r))
                d["created_at"] = str(d["created_at"])
                reviews[d["event_id"]] = d
        for r in raises:
            r["case"] = _case_for(con, r["interaction_id"])
            rv = reviews.get(r["event_id"])
            r["review"] = ({k: rv[k] for k in ("review_id", "verdict", "note", "reviewer", "created_at")}
                           if rv else None)
    return raises


def submit_review(*, event_id: str, verdict: str, note: str | None, reviewer: str, reviewer_role: str) -> dict[str, Any]:
    """Store one verdict. Role must already have been checked by the caller;
    this re-checks it as defence in depth. Never touches case severity."""
    if reviewer_role != "supervisor":
        raise ReviewError(403, f"role {reviewer_role} cannot review AI raises (supervisor only)")
    v = str(verdict or "").strip().lower()
    if v not in VERDICTS:
        raise ReviewError(422, "verdict must be 'accept' or 'reject'")
    clean_note = None
    if note:
        from src.security.pii import redact_pii

        clean_note = redact_pii(str(note))[:NOTE_MAX]
    with ops_con() as con:
        ensure_tables(con)
        rows = _raise_rows(con, event_id=event_id, limit=1)
        if not rows:
            raise ReviewError(404, "no AI severity raise with that event_id")
        raise_row = rows[0]
        if con.execute("SELECT 1 FROM ai_raise_reviews WHERE event_id = ?", [event_id]).fetchone():
            _audit(con, action="review_rejected_duplicate", actor=reviewer, actor_role=reviewer_role,
                   event_id=event_id, review_id=None, detail={"verdict": v})
            raise ReviewError(409, "this raise already has a verdict")
        rid = "rvw_" + new_ulid()
        con.execute(
            "INSERT INTO ai_raise_reviews (review_id, event_id, interaction_id, verdict, note, reviewer, reviewer_role, "
            "rule_severity, ai_severity, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [rid, event_id, raise_row["interaction_id"], v, clean_note, reviewer[:80], reviewer_role,
             raise_row["rule_severity"], raise_row["ai_severity"], utc_now()],
        )
        aid = _audit(con, action="review_recorded", actor=reviewer[:80], actor_role=reviewer_role,
                     event_id=event_id, review_id=rid,
                     detail={"verdict": v, "note_present": bool(clean_note),
                             "rule_severity": raise_row["rule_severity"], "ai_severity": raise_row["ai_severity"],
                             "severity_changed": False})
        case = _case_for(con, raise_row["interaction_id"])
    return {"review_id": rid, "audit_id": aid, "event_id": event_id, "verdict": v, "note": clean_note,
            "case_severity_unchanged": True, "case": case}


def record_denied(*, event_id: str, actor: str, actor_role: str) -> None:
    """Audit a 403 attempt (best-effort)."""
    try:
        with ops_con() as con:
            ensure_tables(con)
            _audit(con, action="review_denied_role", actor=actor[:80], actor_role=actor_role,
                   event_id=event_id, review_id=None, detail={})
    except Exception:
        pass


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)


def stats(min_n: int = MIN_REVIEWS_FOR_RATE) -> dict[str, Any]:
    with ops_con() as con:
        raises = len(_raise_rows(con, limit=1000))
        acc = rej = 0
        if _relation_exists(con, "ai_raise_reviews"):
            for verdict, n in con.execute("SELECT verdict, COUNT(*) FROM ai_raise_reviews GROUP BY verdict").fetchall():
                if verdict == "accept":
                    acc = int(n)
                elif verdict == "reject":
                    rej = int(n)
    n = acc + rej
    out: dict[str, Any] = {
        "ai_raises": raises, "reviewed": n, "accepted": acc, "rejected": rej,
        "pending": max(0, raises - n), "min_reviews_for_rate": min_n, "n_label": f"n = {n}",
    }
    if n == 0:
        out.update(agreement_rate=None, status="no reviews yet", ci95=None)
    elif n < min_n:
        out.update(agreement_rate=None, status=f"too few reviews to estimate (n = {n} < {min_n})", ci95=None)
    else:
        out.update(agreement_rate=round(acc / n, 3), status="estimated", ci95=wilson(acc, n))
    return out


__all__ = ["MIN_REVIEWS_FOR_RATE", "ReviewError", "list_raises", "record_denied", "stats", "submit_review",
           "verify_audit_chain", "wilson"]
