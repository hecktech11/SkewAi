"""Grounded investigation brief for early-warning clusters.  [AI-LAYER]

Starts from the existing deterministic ``build_investigation_brief`` (weekly
series, per-source records, matched recall). Muse Spark may then write a short
brief as a list of claims, where EVERY claim must cite source record ids.

A claim survives only if all of these hold:

1. it cites at least one id, and every cited id was in the facts we sent
   (no invented ids);
2. every cited id passes the Qubot verifier ``_verify_evidence_id`` against the
   pack warehouse and the slice scope (the same check the post-contact audit
   runs);
3. every id-shaped token in the text (Qubot ``_extract_ids_from_text``) is
   among its citations;
4. every number in the text appears in the facts or the cited records
   (no invented metrics);
5. ``narration_violations`` (promises, remedies, lead-time) finds nothing.

Failing claims are dropped and counted. With no key, or when no claim
survives, the result is the deterministic template brief: the existing
``phrase_investigation_brief`` sentence plus deterministic claims built only
from the warehouse facts.
"""

from __future__ import annotations

import re
from typing import Any

from src.ai_layer import config
from src.ai_layer.meta_client import MetaError, chat_json
from src.ai_layer.store import record_event

BRIEF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headline": {"type": "string", "maxLength": 200},
        "recommended_action": {"type": "string", "maxLength": 400},
        "claims": {
            "type": "array",
            "maxItems": 8,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "maxLength": 400},
                    "record_ids": {"type": "array", "maxItems": 8, "items": {"type": "string", "maxLength": 80}},
                },
                "required": ["text", "record_ids"],
            },
        },
    },
    "required": ["headline", "claims"],
}

_NUM = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?")


def _records(brief: dict[str, Any]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for rec in brief.get("similar_records") or []:
        if rec.get("record_id"):
            seen.setdefault(str(rec["record_id"]), rec)
    for rows in (brief.get("records_by_source") or {}).values():
        for rec in rows or []:
            if rec.get("record_id"):
                seen.setdefault(str(rec["record_id"]), rec)
    return list(seen.values())


def _fmt_nums(value: Any, out: set[str]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, int):
        out.add(str(value))
    elif isinstance(value, float):
        out.update({str(round(value, 1)), str(round(value, 2)), str(int(round(value)))})
        if float(value).is_integer():
            out.add(str(int(value)))
    elif isinstance(value, str):
        out.update(_NUM.findall(value))
    elif isinstance(value, dict):
        for v in value.values():
            _fmt_nums(v, out)
    elif isinstance(value, (list, tuple)):
        for v in value:
            _fmt_nums(v, out)


def facts_for_prompt(brief: dict[str, Any], limit: int = 20) -> dict[str, Any]:
    recs = _records(brief)[:limit]
    flagged = brief.get("flagged_weeks") or []
    return {
        "investigation_id": brief.get("investigation_id"),
        "cluster_id": brief.get("cluster_id"),
        "slice": brief.get("slice"),
        "flagged_weeks": [
            {"iso_week": w.get("iso_week"), "record_count": w.get("record_count"),
             "baseline_mean": round(float(w.get("baseline_mean") or 0), 2), "z_score": round(float(w.get("z_score") or 0), 2)}
            for w in flagged[:12]
        ],
        "flagged_week_count": len(flagged),
        "record_count_in_scope": len(_records(brief)),
        "association_lift": round(float(brief.get("association_lift") or 0), 2),
        "recall": brief.get("recall"),
        "records": [
            {"record_id": r.get("record_id"), "source": r.get("source"), "received_at": r.get("received_at"),
             "entity_1": r.get("entity_1"), "text": str(r.get("text") or "")[:300]}
            for r in recs
        ],
    }


def deterministic_claims(brief: dict[str, Any]) -> list[dict[str, Any]]:
    """Template claims built only from warehouse facts (always grounded)."""
    claims: list[dict[str, Any]] = []
    recs = _records(brief)
    sl = brief.get("slice") or {}
    scope = " ".join(str(sl.get(k)) for k in ("entity_2", "entity_3", "category") if sl.get(k))
    if recs:
        ids = [str(r["record_id"]) for r in recs[:5]]
        claims.append({"text": f"{len(recs)} source records in scope for {scope or 'this slice'}.",
                       "record_ids": ids, "verified": True, "source": "deterministic"})
    flagged = brief.get("flagged_weeks") or []
    if flagged and recs:
        weeks = ", ".join(str(w.get("iso_week")) for w in flagged[:3])
        claims.append({"text": f"{len(flagged)} flagged week(s) above baseline, first: {weeks}.",
                       "record_ids": [str(recs[0]["record_id"])], "verified": True, "source": "deterministic"})
    recall = brief.get("recall") or {}
    if recall.get("advisory_id"):
        claims.append({"text": f"Matched recall advisory {recall['advisory_id']}.",
                       "record_ids": [str(recall["advisory_id"])], "verified": True, "source": "deterministic"})
    return claims


def template_summary(brief: dict[str, Any]) -> str:
    """Existing template brief sentence (``src.ai.narration``), unchanged."""
    from src.ai.narration import phrase_investigation_brief

    recs = _records(brief)
    recall = brief.get("recall") or {}
    sl = brief.get("slice") or {}
    res = phrase_investigation_brief(
        similar_count=len(recs),
        cluster_id=brief.get("cluster_id"),
        lead_time_weeks=recall.get("lead_time_weeks") if isinstance(recall, dict) else None,
        keyword=str(sl.get("category") or ""),
        evidence_ids=[str(r["record_id"]) for r in recs[:5]],
    )
    return res.text


def verify_claim(claim: dict[str, Any], *, brief: dict[str, Any], allowed_ids: set[str],
                 allowed_numbers: set[str], records_by_id: dict[str, dict[str, Any]]) -> tuple[bool, str]:
    from src.ai.provider import narration_violations
    from src.qubot.auditor import _extract_ids_from_text, _verify_evidence_id

    text = str(claim.get("text") or "").strip()
    cited = [str(x).strip() for x in (claim.get("record_ids") or []) if str(x).strip()]
    if not text:
        return False, "empty_text"
    if not cited:
        return False, "no_citation"
    for rid in cited:
        if rid not in allowed_ids:
            return False, f"invented_id:{rid[:40]}"
    sl = brief.get("slice") or {}
    scope = {k: sl.get(k) for k in ("category", "entity_2", "entity_3") if sl.get(k)}
    pack_id = str(brief.get("pack_id") or "")
    for rid in cited:
        if rid == str(brief.get("investigation_id")):
            ok, msg = _verify_evidence_id(pack_id, rid, scope)
        elif rid in records_by_id:
            ok, msg = _verify_evidence_id(pack_id, rid, scope)
        else:  # advisory / cluster ids
            ok, msg = _verify_evidence_id(pack_id, rid, {})
        if not ok:
            return False, f"qubot_reject:{rid[:40]}:{msg[:60]}"
    cited_upper = {c.upper() for c in cited} | set(cited)
    for tok in _extract_ids_from_text(text):
        if tok not in cited_upper and tok.upper() not in cited_upper:
            return False, f"uncited_id_in_text:{tok[:40]}"
    local_numbers = set(allowed_numbers)
    for rid in cited:
        _fmt_nums(records_by_id.get(rid, {}).get("text"), local_numbers)
        _fmt_nums(rid, local_numbers)
    for num in _NUM.findall(text):
        if num not in local_numbers and num.rstrip("0").rstrip(".") not in local_numbers:
            return False, f"unsupported_number:{num}"
    lead = (brief.get("recall") or {}).get("lead_time_weeks") if isinstance(brief.get("recall"), dict) else None
    viol = narration_violations(text, {"evidence_ids": sorted(allowed_ids), "lead_time_weeks": lead})
    if viol:
        return False, "narration_violation:" + ",".join(viol)
    return True, "ok"


def build_grounded_brief(investigation_id: str) -> dict[str, Any]:
    from src.frontline.investigation_brief import build_investigation_brief

    from src.ai_layer.safety import recommended_actions, slice_safety

    brief = build_investigation_brief(investigation_id)  # LookupError -> 404 upstream
    det_claims = deterministic_claims(brief)
    summary = template_summary(brief)
    sl = brief.get("slice") or {}
    safety = slice_safety(str(brief.get("pack_id") or ""), sl.get("category"), sl.get("entity_2"), sl.get("entity_3"))
    actions = recommended_actions(sl.get("category"), brief.get("recall") if isinstance(brief.get("recall"), dict) else None)
    result: dict[str, Any] = {
        "investigation_id": investigation_id,
        "pack_id": brief.get("pack_id"),
        "cluster_id": brief.get("cluster_id"),
        "slice": brief.get("slice"),
        "mode": "deterministic",
        "headline": summary,
        "claims": det_claims,
        "dropped_claims": [],
        "model_id": "deterministic",
        "fallback_reason": "no_key" if not config.meta_enabled() else None,
        "safety_risk": safety,
        "recommended_actions": actions,
        "powered_by": [],
    }
    if not config.meta_enabled():
        return result
    recs = _records(brief)
    if not recs:
        result["fallback_reason"] = "no_records_to_cite"
        return result
    facts = facts_for_prompt(brief)
    facts["safety_risk"] = {k: safety[k] for k in ("level", "component_tier", "mentions", "records_scanned")}
    records_by_id = {str(r["record_id"]): r for r in recs}
    allowed_ids = set(records_by_id) | {str(investigation_id)}
    recall = brief.get("recall") or {}
    if isinstance(recall, dict) and recall.get("advisory_id"):
        allowed_ids.add(str(recall["advisory_id"]))
    allowed_numbers: set[str] = set()
    _fmt_nums({k: v for k, v in facts.items() if k != "records"}, allowed_numbers)
    system = (
        "You write an investigation brief for an automotive quality engineer. Return JSON only. "
        "Write a headline (no numbers or ids) and up to 6 claims. Every claim MUST cite record_ids taken "
        "only from the provided facts, and must only use numbers present in the facts or cited records. "
        "Lead with crash, fire and injury risk when the records show it: the goal is getting faults repaired "
        "before they cause accidents. recommended_action is ONE inspection/repair step for the manufacturer's "
        "engineers (no numbers, no ids, no promises to customers). "
        "No speculation about root cause beyond what the records say."
    )
    import json as _json

    user = "FACTS:\n" + _json.dumps(facts, default=str)[:12000]
    try:
        out = chat_json(system=system, user=user, schema=BRIEF_SCHEMA, schema_name="grounded_brief",
                        timeout=max(config.timeout_s(), 15.0), max_tokens=1200)
    except MetaError as e:
        result["fallback_reason"] = e.reason
        record_event(kind="grounded_brief", reason=f"fallback:{e.reason}", deterministic={"investigation_id": investigation_id})
        return result
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for claim in out.get("claims") or []:
        ok, why = verify_claim(claim, brief=brief, allowed_ids=allowed_ids,
                               allowed_numbers=allowed_numbers, records_by_id=records_by_id)
        row = {"text": str(claim.get("text") or "")[:400], "record_ids": list(claim.get("record_ids") or [])[:8]}
        if ok:
            kept.append({**row, "verified": True, "source": "meta_muse_spark"})
        else:
            dropped.append({**row, "reason": why})
    headline = str(out.get("headline") or "").strip()
    if headline and (_NUM.search(headline) or verify_headline_ids(headline)):
        headline = ""  # headline must not carry numbers / ids (they are uncited)
    record_event(kind="grounded_brief", model_id=config.model_text(),
                 deterministic={"investigation_id": investigation_id, "claims": det_claims},
                 ai_proposal=out, accepted={"kept": len(kept), "dropped": [d["reason"] for d in dropped]},
                 reason="ok" if kept else "all_claims_dropped")
    result["dropped_claims"] = dropped
    ai_action = str(out.get("recommended_action") or "").strip()
    if ai_action and kept and action_ok(ai_action, allowed_ids):
        result["recommended_actions"] = actions + [{
            "action": ai_action, "source": "meta_muse_spark", "record_ids": [],
            "note": "AI-suggested; requires engineering review",
        }]
    if not kept:
        result["fallback_reason"] = "all_claims_failed_verification"
        return result
    result.update({
        "mode": "enhanced",
        "headline": headline or summary,
        "claims": kept,
        "model_id": config.model_text(),
        "fallback_reason": None,
        "powered_by": [config.model_text()],
    })
    return result


def action_ok(text: str, allowed_ids: set[str]) -> bool:
    """AI action text: no numbers, no ids, no promises / remedies to customers."""
    from src.ai.provider import narration_violations

    if _NUM.search(text) or verify_headline_ids(text):
        return False
    return not narration_violations(text, {"evidence_ids": sorted(allowed_ids)})


def verify_headline_ids(text: str) -> bool:
    from src.qubot.auditor import _extract_ids_from_text

    return bool(_extract_ids_from_text(text))


__all__ = ["BRIEF_SCHEMA", "build_grounded_brief", "deterministic_claims", "facts_for_prompt", "verify_claim"]
