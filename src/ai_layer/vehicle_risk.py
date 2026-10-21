"""Per-vehicle fault risk from real complaint + recall data.  [AI-LAYER]

Input: make, model, year (mileage optional). Output: one row per component
with the counts behind it, a deterministic risk level and plain-language
owner advice ("Get your brakes inspected ..."). Nothing is predicted beyond
what the warehouse shows:

* ``recalls``: recall advisories whose scope matches make + model + year
  (real NHTSA rows in the pack warehouse). Any matching recall on a
  safety-critical component is "act now" (the data cannot tell whether a
  given VIN's remedy was already done, so the advice says to ask a dealer).
* ``complaints``: complaints filed for this exact make/model/year and
  component, and the share of this vehicle's complaints that the component
  makes up, compared with the same component's share across every vehicle of
  that model year in the warehouse (``relative_rate``).
* ``hazard mentions``: crash/fire/injury words in those complaint narratives
  (keyword scan, not verified events).

Mileage: the warehouse has no mileage field, so mileage is echoed back and
NOT used (said explicitly in the response).

With a Meta key, Muse Spark may write a short explanation of the facts. It is
rejected if it contains any number that is not in the facts, and the
deterministic levels and advice are never changed by it.
"""

from __future__ import annotations

import re
from typing import Any

from src.ai_layer.safety import component_tier, hazard_mentions

_ADVICE = {
    "SERVICE BRAKES": "Get your brakes inspected: pedal feel, pads, discs, fluid and lines.",
    "PARKING BRAKE": "Have the parking brake checked so it holds on a slope.",
    "STEERING": "Get the steering checked: rack, tie rods and power-steering faults.",
    "AIR BAGS": "Have the airbag system scanned for fault codes and check for open recalls.",
    "SEAT BELTS": "Have seat-belt buckles, retractors and pretensioners checked.",
    "FUEL SYSTEM": "Get the fuel system checked for leaks; a fuel smell is a fire risk, do not ignore it.",
    "FUEL/PROPULSION SYSTEM": "Get the fuel/propulsion system checked for leaks and loss of power.",
    "VEHICLE SPEED CONTROL": "Have the accelerator pedal and cruise control checked.",
    "ELECTRONIC STABILITY CONTROL": "Have the stability-control system scanned for fault codes.",
    "FORWARD COLLISION AVOIDANCE": "Have the forward-collision sensors checked and recalibrated if needed.",
    "TIRES": "Check tyre wear, pressure and sidewalls; replace damaged tyres.",
    "WHEELS": "Have wheel nuts, studs and bearings checked.",
    "ELECTRICAL SYSTEM": "Have wiring and battery connections checked for overheating or chafing.",
    "ENGINE": "Get the engine checked for stalling and oil or coolant leaks.",
    "POWER TRAIN": "Get the transmission checked for slipping, harsh shifts or failing to hold in park.",
    "SUSPENSION": "Have the suspension (arms, joints, springs, struts) inspected.",
    "EXTERIOR LIGHTING": "Check that headlamps and brake lamps work.",
}
_DEFAULT_ADVICE = "Mention this at your next service so the workshop can check it."
_LEVEL_ORDER = {"act_now": 0, "high": 1, "elevated": 2, "watch": 3, "low": 4}


def _advice(category: str, level: str, recall_ids: list[str]) -> str:
    base = _ADVICE.get(category.upper(), _DEFAULT_ADVICE)
    if recall_ids:
        camps = sorted({r.split("|")[0] for r in recall_ids})
        return (f"Recall on record for this model year ({', '.join(camps[:3])}): ask a dealer whether your VIN is "
                f"covered and whether the remedy was done. " + base)
    if level in ("high", "elevated"):
        return base
    return "No unusual pattern in the data. " + base if level == "watch" else "No signal in the data for this component."


def classify(category: str, n: int, relative_rate: float | None, recall_ids: list[str], hazard_records: int) -> str:
    tier = component_tier(category)
    if recall_ids and tier == "critical":
        return "act_now"
    if recall_ids:
        return "high"
    rr = relative_rate or 0.0
    if tier == "critical" and n >= 10 and (rr >= 1.5 or hazard_records >= 3):
        return "high"
    if n >= 10 and rr >= 2.0:
        return "high"
    if n >= 5 and (rr >= 1.5 or tier == "critical"):
        return "elevated"
    if n > 0:
        return "watch"
    return "low"


def _query(pack_id: str, make: str, model: str, year: str) -> dict[str, Any]:
    from src.data.warehouse import domain_con

    with domain_con(pack_id, read_only=True) as con:
        rows = con.execute(
            "SELECT upper(category), count(*) FROM records WHERE upper(entity_2)=upper(?) AND upper(entity_3)=upper(?) "
            "AND entity_1=? AND category IS NOT NULL GROUP BY 1",
            [make, model, str(year)],
        ).fetchall()
        peer = con.execute(
            "SELECT upper(category), count(*) FROM records WHERE entity_1=? AND category IS NOT NULL GROUP BY 1",
            [str(year)],
        ).fetchall()
        texts = con.execute(
            "SELECT upper(category), text FROM records WHERE upper(entity_2)=upper(?) AND upper(entity_3)=upper(?) "
            "AND entity_1=? AND category IS NOT NULL ORDER BY received_at DESC LIMIT 3000",
            [make, model, str(year)],
        ).fetchall()
        recalls = con.execute(
            "SELECT advisory_id, upper(scope_category), issued_at, remedy, url FROM advisories "
            "WHERE upper(scope_entity_2)=upper(?) AND upper(scope_entity_3)=upper(?) "
            "AND (scope_entity_1 IS NULL OR scope_entity_1 = '' OR scope_entity_1=?) ORDER BY issued_at DESC",
            [make, model, str(year)],
        ).fetchall()
    return {"rows": rows, "peer": peer, "texts": texts, "recalls": recalls}


def vehicle_risk(make: str, model: str, year: str | int, mileage: int | None = None, *,
                 pack_id: str | None = None, explain: bool = True) -> dict[str, Any]:
    from src.ai_layer import config

    if not pack_id:
        from src.domains.active_pack import resolve_active_pack_id
        pack_id = resolve_active_pack_id()
    make, model, year = str(make or "").strip(), str(model or "").strip(), str(year or "").strip()
    out: dict[str, Any] = {
        "vehicle": {"make": make.upper(), "model": model.upper(), "year": year, "mileage": mileage},
        "mileage_used": False,
        "mileage_note": "The complaint warehouse has no mileage field, so mileage does not change these results.",
        "pack_id": pack_id, "components": [], "mode": config.ai_mode(), "powered_by": [],
        "method": ("Counts of real complaints and recall advisories for this make/model/year in the pack warehouse; "
                   "relative_rate = this vehicle's share of complaints for the component / that component's share "
                   "across all vehicles of the same model year. Not a probability of failure."),
    }
    if not (make and model and year):
        out["error"] = "make, model and year are required"
        return out
    try:
        q = _query(pack_id, make, model, year)
    except Exception as e:  # missing warehouse etc.
        out["error"] = f"warehouse unavailable ({type(e).__name__})"
        return out
    total = sum(int(c) for _, c in q["rows"])
    peer_total = sum(int(c) for _, c in q["peer"])
    peer_share = {cat: (int(c) / peer_total) for cat, c in q["peer"]} if peer_total else {}
    hazards: dict[str, dict[str, int]] = {}
    for cat, text in q["texts"]:
        h = hazards.setdefault(cat, {"records_scanned": 0, "records_with_hazard_mention": 0, "fire": 0, "crash": 0, "injury": 0})
        h["records_scanned"] += 1
        hits = hazard_mentions(str(text or ""))
        if hits:
            h["records_with_hazard_mention"] += 1
        for k in hits:
            h[k] += 1
    recalls_by_cat: dict[str, list[dict[str, Any]]] = {}
    for aid, cat, issued, remedy, url in q["recalls"]:
        recalls_by_cat.setdefault(cat or "", []).append({
            "advisory_id": aid, "issued_at": issued.isoformat() if hasattr(issued, "isoformat") else issued,
            "remedy": remedy, "url": url})
    cats = {c for c, _ in q["rows"]} | {c for c in recalls_by_cat if c}
    comps = []
    for cat in cats:
        n = next((int(c) for c2, c in q["rows"] if c2 == cat), 0)
        share = (n / total) if total else 0.0
        ps = peer_share.get(cat)
        rr = round(share / ps, 2) if (ps and total) else None
        rec = recalls_by_cat.get(cat, [])
        rec_ids = [r["advisory_id"] for r in rec]
        hz = hazards.get(cat, {"records_scanned": 0, "records_with_hazard_mention": 0, "fire": 0, "crash": 0, "injury": 0})
        level = classify(cat, n, rr, rec_ids, hz["records_with_hazard_mention"])
        comps.append({
            "component": cat, "component_tier": component_tier(cat), "level": level,
            "advice": _advice(cat, level, rec_ids),
            "data": {"complaints": n, "vehicle_complaints_total": total, "share_of_vehicle_complaints": round(share, 3),
                     "model_year_peer_share": round(ps, 4) if ps else None, "relative_rate": rr,
                     "model_year_peer_complaints_total": peer_total, "hazard": hz, "recalls": rec[:5],
                     "recall_count": len(rec)},
        })
    comps.sort(key=lambda c: (_LEVEL_ORDER[c["level"]], -c["data"]["complaints"]))
    out["components"] = comps
    out["summary"] = {"complaints_total": total, "components_with_data": len(comps),
                      "act_now": sum(1 for c in comps if c["level"] == "act_now"),
                      "high": sum(1 for c in comps if c["level"] == "high")}
    if explain and config.meta_enabled() and comps:
        exp = _explain(out)
        if exp:
            out["explanation"] = exp
            out["powered_by"] = [config.model_text()]
    return out


_NUM = re.compile(r"\d+(?:\.\d+)?")
EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {"explanation": {"type": "string"}},
    "required": ["explanation"],
}


def _facts(out: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"component": c["component"], "level": c["level"], "complaints": c["data"]["complaints"],
             "relative_rate": c["data"]["relative_rate"], "recalls": [r["advisory_id"] for r in c["data"]["recalls"]],
             "advice": c["advice"]} for c in out["components"][:6]]


def explanation_ok(text: str, facts: list[dict[str, Any]], vehicle: dict[str, Any]) -> bool:
    if not text or len(text) > 900:
        return False
    allowed = set(_NUM.findall(str(facts) + " " + str(vehicle)))
    return all(n in allowed for n in _NUM.findall(text))


def _explain(out: dict[str, Any]) -> str | None:
    from src.ai_layer.meta_client import MetaError, chat_json

    facts = _facts(out)
    try:
        res = chat_json(
            system=("Explain these vehicle fault-risk facts to a car owner in 2-4 plain sentences. Lead with any open "
                    "recall or safety-critical component. Use only the numbers given. Do not promise safety. JSON only."),
            user=str({"vehicle": out["vehicle"], "facts": facts}),
            schema=EXPLAIN_SCHEMA, schema_name="vehicle_risk_explain",
        )
    except MetaError:
        return None
    text = str(res.get("explanation") or "").strip()
    return text if explanation_ok(text, facts, out["vehicle"]) else None


__all__ = ["classify", "explanation_ok", "vehicle_risk"]
