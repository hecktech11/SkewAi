"""OBD-II fault codes (DTCs) as optional intake evidence.  [AI-LAYER]

Small, deterministic, no key needed.

* Format: ``^[PCBU][0-3][0-9A-F]{3}$`` (e.g. P0300, C0035). Anything else is
  rejected as invalid input.
* Meaning: ``src/ai_layer/data/dtc_lookup.json``, generic SAE J2012-style
  meanings only (source + confidence per entry). A code that is not listed is
  ``unknown code`` and has NO effect on anything.
* Severity: a known *safety* code (ABS wheel-speed sensor, restraint loop, lost
  ABS / steering / restraint module) gives a floor of the pack's safety-system
  level (Medium on the NHTSA pack). final = max(current, floor): it can only
  RAISE severity, never lower it.
* Needs review: when a known code's component differs from the complaint's
  category, the code row is flagged ``needs review`` (nothing else changes).
* Counting: ``cases with matching fault code`` = cases whose stored code
  component equals the case category (corroborating). Reported next to the
  existing vehicle-risk / alert / queue numbers; existing scores and thresholds
  are not changed.

There is NO public dataset linking DTCs to recalls, so no accuracy or backtest
claim is made for any of this.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

DTC_RE = re.compile(r"^[PCBU][0-3][0-9A-F]{3}$")
_TOKEN = re.compile(r"(?<![A-Za-z0-9])[PCBUpcbu][0-3][0-9A-Fa-f]{3}(?![A-Za-z0-9])")
_PREFIX = re.compile(r"(?:fault\s*codes?|codes?|dtcs?|कोड)\s*[:=]\s*(.+)", re.IGNORECASE)
_DATA = Path(__file__).resolve().parent / "data" / "dtc_lookup.json"
ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
SAFETY_FLOOR = "Medium"
MAX_CODES = 20

_DDL = """
CREATE TABLE IF NOT EXISTS ai_fault_codes (
    row_id VARCHAR PRIMARY KEY,
    case_id VARCHAR,
    interaction_id VARCHAR,
    code VARCHAR NOT NULL,
    known BOOLEAN,
    component VARCHAR,
    safety VARCHAR,
    meaning VARCHAR,
    confidence VARCHAR,
    complaint_category VARCHAR,
    needs_review BOOLEAN,
    review_reason VARCHAR,
    source VARCHAR,
    created_at TIMESTAMP
)
"""


@lru_cache(maxsize=1)
def table() -> dict[str, Any]:
    return json.loads(_DATA.read_text(encoding="utf-8"))


def _key(code: str) -> tuple[str, int]:
    return code[0], int(code[1:], 16)


def is_valid(code: str) -> bool:
    return bool(DTC_RE.match(code or ""))


def split_codes(value: str | Iterable[str] | None) -> tuple[list[str], list[str]]:
    """(valid, invalid) from a list or a free string such as 'C0035, p0300'. Upper-cased, de-duplicated."""
    if value is None:
        return [], []
    items = re.split(r"[\s,;]+", value) if isinstance(value, str) else [str(v) for v in value]
    valid: list[str] = []
    invalid: list[str] = []
    for raw in items:
        c = (raw or "").strip().upper()
        if not c:
            continue
        (valid if is_valid(c) else invalid).append(c)
    dedup = list(dict.fromkeys(valid))[:MAX_CODES]
    return dedup, list(dict.fromkeys(invalid))


def codes_in_text(text: str) -> list[str]:
    """Codes a customer typed after a 'codes:' / 'fault codes:' / 'DTC:' / 'कोड:' marker (e.g. WhatsApp)."""
    out: list[str] = []
    for line in (text or "").splitlines():
        m = _PREFIX.search(line)
        if m:
            out += [t.upper() for t in _TOKEN.findall(m.group(1))]
    return list(dict.fromkeys(out))[:MAX_CODES]


def lookup(code: str) -> dict[str, Any]:
    c = (code or "").strip().upper()
    if not is_valid(c):
        return {"code": c, "valid": False, "known": False, "meaning": "invalid code format", "component": None,
                "safety": None, "confidence": None}
    t = table()
    for r in t["ranges"]:
        if r["from"][0] == c[0] and _key(r["from"]) <= _key(c) <= _key(r["to"]):
            return {"code": c, "valid": True, "known": True, "meaning": r["meaning"], "component": r["component"],
                    "safety": r["safety"], "confidence": r["confidence"], "family": t["families"].get(c[0])}
    return {"code": c, "valid": True, "known": False, "meaning": "unknown code", "component": None, "safety": None,
            "confidence": None, "family": t["families"].get(c[0])}


def assess(codes: Iterable[str], complaint_category: str | None) -> dict[str, Any]:
    """Pure: per-code lookup + severity floor + needs-review flags."""
    items = []
    cat = (complaint_category or "").strip().upper() or None
    for c in codes:
        info = lookup(c)
        nr = bool(info["known"] and cat and info["component"] != cat)
        info["needs_review"] = nr
        info["review_reason"] = (f"code component {info['component']} differs from complaint category {cat}" if nr else None)
        info["corroborates"] = bool(info["known"] and cat and info["component"] == cat)
        items.append(info)
    safety = [i for i in items if i["known"] and i["safety"] == "safety"]
    return {
        "codes": items,
        "floor": SAFETY_FLOOR if safety else None,
        "floor_reason": (f"safety-system fault code(s): {', '.join(i['code'] + ' ' + str(i['component']) for i in safety)}"
                         if safety else None),
        "needs_review": any(i["needs_review"] for i in items),
        "unknown": [i["code"] for i in items if not i["known"]],
    }


def raise_only(current: str | None, floor: str | None, allowed: set[str] | None = None) -> str | None:
    cur = current if current in ORDER else None
    if not floor or floor not in ORDER or (allowed is not None and floor not in allowed):
        return current
    if cur is None:
        return current  # no severity yet: nothing to raise (triage decides first)
    return floor if ORDER[floor] > ORDER[cur] else current


# ── storage ─────────────────────────────────────────────────────────────────

def _ensure(con) -> None:
    con.execute(_DDL)


def store(codes: list[str], *, case_id: str | None = None, interaction_id: str | None = None,
          complaint_category: str | None = None, source: str = "api") -> dict[str, Any]:
    from src.data.warehouse import ops_con
    from src.ids import new_ulid

    a = assess(codes, complaint_category)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with ops_con() as con:
        _ensure(con)
        have = {r[0] for r in con.execute(
            "SELECT code FROM ai_fault_codes WHERE COALESCE(case_id,'') = ? AND COALESCE(interaction_id,'') = ?",
            [case_id or "", interaction_id or ""]).fetchall()}
        for i in a["codes"]:
            if i["code"] in have:
                continue
            con.execute(
                "INSERT INTO ai_fault_codes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [f"dtc_{new_ulid()}", case_id, interaction_id, i["code"], i["known"], i["component"], i["safety"],
                 i["meaning"], i["confidence"], complaint_category, i["needs_review"], i["review_reason"], source, now])
    return a


def codes_for_interaction(interaction_id: str) -> list[str]:
    from src.data.warehouse import ops_con

    with ops_con() as con:
        _ensure(con)
        return [r[0] for r in con.execute(
            "SELECT DISTINCT code FROM ai_fault_codes WHERE interaction_id = ? ORDER BY code", [interaction_id]).fetchall()]


def for_case(case_id: str) -> dict[str, Any]:
    """Codes attached to the case directly or to its interaction, re-assessed against the case category."""
    from src.data.warehouse import ops_con

    with ops_con() as con:
        _ensure(con)
        row = con.execute("SELECT interaction_id, category, severity FROM cases WHERE case_id = ?", [case_id]).fetchone()
        iid, cat, sev = (row or (None, None, None))
        codes = [r[0] for r in con.execute(
            "SELECT DISTINCT code FROM ai_fault_codes WHERE case_id = ? OR (interaction_id IS NOT NULL AND interaction_id = ?) "
            "ORDER BY code", [case_id, iid or ""]).fetchall()]
    out = assess(codes, cat)
    out.update({"case_id": case_id, "case_found": row is not None, "category": cat, "severity": sev,
                "label": "fault codes (customer-supplied OBD-II DTCs; generic meanings only)"})
    return out


def _priority(pack_id: str | None, severity: str) -> int | None:
    try:
        from src.domains.loader import load_pack

        m = load_pack(pack_id or "automotive_nhtsa").manifest.severity.priority_matrix or {}
        return int(m[severity]) if severity in m else None
    except Exception:
        return None


def _allowed(pack_id: str | None) -> set[str]:
    try:
        from src.domains.loader import load_pack

        m = load_pack(pack_id or "automotive_nhtsa").manifest.severity.priority_matrix or {}
        return {k for k in m if k in ORDER} or {"Low", "Medium", "Critical"}
    except Exception:
        return {"Low", "Medium", "Critical"}


def attach_to_case(case_id: str, value: Any) -> dict[str, Any]:
    """Validate + store codes on an existing case; raise-only severity on that case."""
    from src.data.warehouse import ops_con

    valid, invalid = split_codes(value)
    if invalid:
        return {"error": "invalid", "invalid": invalid, "format": DTC_RE.pattern}
    with ops_con() as con:
        row = con.execute("SELECT interaction_id, category, severity, pack_id FROM cases WHERE case_id = ?", [case_id]).fetchone()
    if row is None:
        return {"error": "not_found"}
    iid, cat, sev, pack_id = row
    a = store(valid, case_id=case_id, complaint_category=cat, source="api_case")
    final = raise_only(sev, a["floor"], _allowed(pack_id))
    raised = bool(final and final != sev)
    if raised:
        prio = _priority(pack_id, final)
        with ops_con() as con:
            # raise-only guard in SQL too: only rows still below the floor are touched
            lower = [k for k, v in ORDER.items() if v < ORDER[final]]
            con.execute(f"UPDATE cases SET severity = ?, priority = COALESCE(?, priority) WHERE case_id = ? "
                        f"AND severity IN ({','.join('?' * len(lower))})", [final, prio, case_id, *lower])
        _ledger(iid, sev, final, a["floor_reason"])
    _event(iid, case_id, sev, final, a)
    out = for_case(case_id)
    out.update({"severity_before": sev, "severity_after": final if raised else sev, "raised": raised})
    return out


def _ledger(iid: str | None, rule: str | None, final: str, why: str | None) -> None:
    if not iid:
        return
    try:
        from src.ledger import record_action
        from src.ledger.writer import AgentAction

        record_action(AgentAction(
            interaction_id=iid, agent="ai_layer", action_type="severity_scored",
            input_summary=f"severity={rule}; fault-code floor={SAFETY_FLOOR}",
            output_summary=f"severity raised {rule}->{final} (max(rule, fault-code floor)); reason={str(why)[:160]}",
        ))
    except Exception:
        pass


def _event(iid, case_id, before, after, a) -> None:
    try:
        from src.ai_layer.store import record_event

        record_event(kind="fault_codes", interaction_id=iid, case_id=case_id, mode="deterministic",
                     deterministic={"severity": before, "codes": [i["code"] for i in a["codes"]]},
                     accepted={"severity": after, "needs_review": a["needs_review"]},
                     reason="raised_by_fault_code" if after != before else ("needs_review" if a["needs_review"] else "evidence_only"))
    except Exception:
        pass


# ── live intake hooks (orchestrator, [AI-LAYER]) ─────────────────────────────

async def maybe_capture_fault_codes(ctx, text: str) -> None:
    """Store codes typed after a 'codes:' marker on this interaction. Never raises."""
    try:
        codes = codes_in_text(text or "")
        if codes:
            store(codes, interaction_id=ctx.interaction_id, complaint_category=(ctx.slots or {}).get("category"),
                  source=f"message:{getattr(ctx, 'channel', '')}")
    except Exception:
        pass


async def maybe_raise_severity_dtc(ctx, tres: dict[str, Any]) -> dict[str, Any]:
    """Post-triage, raise-only: final = max(rule, fault-code floor). Never lowers, never raises an exception."""
    if not isinstance(tres, dict) or tres.get("skipped"):
        return tres
    try:
        codes = codes_for_interaction(ctx.interaction_id)
        if not codes:
            return tres
        from src.ai_layer.severity import _allowed_levels

        a = assess(codes, (ctx.slots or {}).get("category"))
        rule = str(getattr(ctx, "severity", None) or tres.get("severity") or "Low")
        final = raise_only(rule, a["floor"], _allowed_levels(ctx))
        _event(ctx.interaction_id, None, rule, final, a)
        new = dict(tres)
        new["fault_codes"] = {"codes": [i["code"] for i in a["codes"]], "needs_review": a["needs_review"],
                              "unknown": a["unknown"]}
        if final == rule:
            return new
        from src.agents.triage import _priority_from_matrix

        ctx.severity = final
        ctx.priority = _priority_from_matrix(final, ctx)
        _ledger(ctx.interaction_id, rule, final, a["floor_reason"])
        new.update({"severity": final, "priority": ctx.priority})
        new["fault_codes"]["raised"] = {"from": rule, "to": final, "reason": a["floor_reason"]}
        return new
    except Exception:
        return tres


# ── counts for vehicle risk / alerts / queue ─────────────────────────────────

def _case_codes(con, where: str, params: list[Any]) -> list[tuple[str, str | None, str]]:
    """(case_id, case_category, code) for cases matching ``where`` (on alias c / i)."""
    _ensure(con)
    return con.execute(
        f"""SELECT c.case_id, c.category, f.code
            FROM cases c LEFT JOIN interactions i ON i.interaction_id = c.interaction_id
            JOIN ai_fault_codes f ON (f.case_id = c.case_id OR (f.interaction_id IS NOT NULL AND f.interaction_id = c.interaction_id))
            WHERE {where}""", params).fetchall()


def _matching(rows) -> dict[str, Any]:
    by_case: dict[str, set[str]] = {}
    by_comp: dict[str, int] = {}
    for case_id, cat, code in rows:
        info = lookup(code)
        if info["known"] and cat and info["component"] == str(cat).upper():
            if case_id not in by_case:
                by_comp[info["component"]] = by_comp.get(info["component"], 0) + 1
            by_case.setdefault(case_id, set()).add(code)
    return {"cases_with_matching_fault_code": len(by_case), "by_component": by_comp,
            "label": "cases with matching fault code: corroborating code component = case category (evidence only; scores unchanged)"}


def matching_for_vehicle(make: str, model: str, year: str) -> dict[str, Any]:
    from src.data.warehouse import ops_con

    try:
        with ops_con() as con:
            rows = _case_codes(con, "upper(i.entity_2) = upper(?) AND upper(i.entity_3) = upper(?) AND CAST(i.entity_1 AS VARCHAR) = ?",
                               [make, model, str(year)])
        return _matching(rows)
    except Exception:
        return {"cases_with_matching_fault_code": 0, "by_component": {}, "label": "fault-code store unavailable"}


def matching_for_investigation(investigation_id: str) -> int:
    from src.data.warehouse import ops_con

    try:
        with ops_con() as con:
            rows = _case_codes(con, "c.investigation_id = ?", [investigation_id])
        return _matching(rows)["cases_with_matching_fault_code"]
    except Exception:
        return 0


def codes_by_case(case_ids: list[str]) -> dict[str, list[str]]:
    from src.data.warehouse import ops_con

    if not case_ids:
        return {}
    try:
        with ops_con() as con:
            rows = _case_codes(con, f"c.case_id IN ({','.join('?' * len(case_ids))})", list(case_ids))
    except Exception:
        return {}
    out: dict[str, list[str]] = {}
    for case_id, _cat, code in rows:
        out.setdefault(case_id, [])
        if code not in out[case_id]:
            out[case_id].append(code)
    return out


def queue_bonus(codes: list[str]) -> tuple[int, str | None]:
    """+10 when any known safety-system code is on the case; explained. 0 otherwise."""
    safety = [lookup(c) for c in codes or []]
    safety = [i for i in safety if i["known"] and i["safety"] == "safety"]
    if not safety:
        return 0, None
    names = ", ".join(f"{i['code']} ({i['component']})" for i in safety[:3])
    return 10, f"safety-system fault code {names} (+10)"


__all__ = ["DTC_RE", "assess", "attach_to_case", "codes_by_case", "codes_in_text", "for_case", "is_valid", "lookup",
           "matching_for_investigation", "matching_for_vehicle", "maybe_capture_fault_codes", "maybe_raise_severity_dtc",
           "queue_bonus", "raise_only", "split_codes", "store"]
