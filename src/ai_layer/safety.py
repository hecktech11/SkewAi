"""Safety-risk indicator and manufacturer inspection actions.  [AI-LAYER]

Deterministic, offline, and derived only from real warehouse rows:

* ``component_tier`` - is the slice's component one whose failure is directly
  linked to crashes or fires (brakes, steering, air bags, fuel, ...)?
* ``hazard mentions`` - how many complaint narratives in the slice mention a
  fire, a crash or an injury (simple keyword match on the complaint text the
  owner filed; a mention is NOT a verified event, and negations such as "no
  injuries" are excluded only for the obvious "no/not/without" prefixes).

The level (high / elevated / normal) is a documented heuristic for ordering a
quality team's queue. It never changes case severity and it is not a
prediction of crashes.
"""

from __future__ import annotations

import re
from typing import Any

CRITICAL_COMPONENTS = frozenset({
    "SERVICE BRAKES", "PARKING BRAKE", "STEERING", "AIR BAGS", "SEAT BELTS",
    "FUEL SYSTEM", "FUEL/PROPULSION SYSTEM", "VEHICLE SPEED CONTROL",
    "ELECTRONIC STABILITY CONTROL", "FORWARD COLLISION AVOIDANCE", "TIRES", "WHEELS",
    "BACKOVER PREVENTION", "CHILD SEAT",
})
ELEVATED_COMPONENTS = frozenset({
    "SUSPENSION", "POWER TRAIN", "ELECTRICAL SYSTEM", "HYBRID PROPULSION SYSTEM",
    "STRUCTURE", "EXTERIOR LIGHTING", "VISIBILITY", "VISIBILITY/WIPER", "ENGINE",
    "LANE DEPARTURE", "TRACTION CONTROL SYSTEM", "LATCHES/LOCKS/LINKAGES",
})

_HAZARDS = {
    "fire": re.compile(r"\b(fire|flames?|caught fire|burn(?:ed|ing|t)?|smoke|smoking)\b", re.I),
    "crash": re.compile(r"\b(crash(?:ed|es)?|collision|collided|accident|rear[- ]ended|hit (?:a|the|another)|wreck(?:ed)?)\b", re.I),
    "injury": re.compile(r"\b(injur(?:y|ies|ed)|hurt|hospital(?:ized)?|ambulance|killed|died|death|fatal(?:ity)?)\b", re.I),
}
_NEGATED = re.compile(r"\b(no|not|without|never|zero)\s+(?:\w+\s+){0,2}$", re.I)

# Inspection-oriented next steps for a manufacturer quality team. They are
# recommendations for engineering review, never promises to a customer.
_ACTIONS = {
    "SERVICE BRAKES": "Inspect the hydraulic brake path (lines, hoses, calipers, master cylinder, ABS module) on in-scope VINs; pull warranty and dealer service records for matching complaints.",
    "PARKING BRAKE": "Inspect parking-brake actuator, cables and electronic park-brake software on in-scope VINs.",
    "STEERING": "Inspect steering rack, column, tie rods and EPS motor/software on in-scope VINs; review EPS fault codes in dealer data.",
    "AIR BAGS": "Pull SRS fault codes and inspect inflator and clock-spring lots for in-scope VINs; check supplier lot traceability.",
    "SEAT BELTS": "Inspect retractor, buckle and pretensioner function on in-scope VINs; check supplier lots.",
    "FUEL SYSTEM": "Inspect fuel lines, rail, pump and connectors for leaks on in-scope VINs; treat any fuel smell or leak as a fire risk.",
    "FUEL/PROPULSION SYSTEM": "Inspect fuel/propulsion components for leaks and loss-of-power faults on in-scope VINs.",
    "VEHICLE SPEED CONTROL": "Inspect throttle/accelerator pedal assembly and cruise-control software; review unintended-acceleration narratives.",
    "ELECTRONIC STABILITY CONTROL": "Review ESC module fault codes and wheel-speed sensors on in-scope VINs.",
    "FORWARD COLLISION AVOIDANCE": "Review forward-sensor calibration and false/missed braking events in telematics where available.",
    "TIRES": "Inspect tire wear, sidewall and tread separation on in-scope builds; check tire lot / DOT codes.",
    "WHEELS": "Inspect wheel studs, nuts, bearings and hubs on in-scope VINs.",
    "ELECTRICAL SYSTEM": "Inspect wiring harness routing, connectors and fuses for chafing or overheating (fire risk) on in-scope VINs.",
    "ENGINE": "Inspect engine for stall and oil/coolant leaks; review stall-while-driving narratives as a loss-of-motive-power risk.",
    "POWER TRAIN": "Inspect transmission for unexpected shifts, loss of drive and park-engagement failures on in-scope VINs.",
    "SUSPENSION": "Inspect control arms, ball joints, springs and struts for cracks or separation on in-scope VINs.",
    "EXTERIOR LIGHTING": "Inspect headlamp/tail-lamp modules and connectors for loss of lighting.",
}
_DEFAULT_ACTION = "Pull matching warranty and dealer service records for in-scope VINs and inspect the reported component."


def component_tier(category: str | None) -> str:
    cat = (category or "").strip().upper()
    if cat in CRITICAL_COMPONENTS:
        return "critical"
    if cat in ELEVATED_COMPONENTS:
        return "elevated"
    return "other"


def hazard_mentions(text: str) -> set[str]:
    out: set[str] = set()
    for name, rx in _HAZARDS.items():
        for m in rx.finditer(text or ""):
            if _NEGATED.search((text or "")[max(0, m.start() - 30):m.start()]):
                continue
            out.add(name)
            break
    return out


def score_records(category: str | None, records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"fire": 0, "crash": 0, "injury": 0}
    any_hazard = 0
    for rec in records:
        hits = hazard_mentions(str(rec.get("text") or ""))
        for h in hits:
            counts[h] += 1
        if hits:
            any_hazard += 1
    n = len(records)
    share = (any_hazard / n) if n else 0.0
    tier = component_tier(category)
    if tier == "critical" and share >= 0.10:
        level = "high"
    elif tier == "critical" or share >= 0.05:
        level = "elevated"
    else:
        level = "normal"
    return {
        "level": level,
        "component_tier": tier,
        "records_scanned": n,
        "records_with_hazard_mention": any_hazard,
        "hazard_mention_share": round(share, 3),
        "mentions": counts,
        "method": "deterministic keyword scan of real complaint narratives + component tier; mentions are not verified events",
    }


def slice_safety(pack_id: str, category: str | None, entity_2: str | None, entity_3: str | None,
                 *, limit: int = 2000) -> dict[str, Any]:
    from src.data.warehouse import domain_con

    where = ["1=1"]
    args: list[Any] = []
    for col, val in (("category", category), ("entity_2", entity_2), ("entity_3", entity_3)):
        if val:
            where.append(f"upper({col}) = upper(?)")
            args.append(val)
    try:
        with domain_con(pack_id, read_only=True) as con:
            rows = con.execute(
                f"SELECT text FROM records WHERE {' AND '.join(where)} ORDER BY received_at DESC LIMIT ?",
                args + [int(limit)],
            ).fetchall()
    except (FileNotFoundError, Exception):
        rows = []
    out = score_records(category, [{"text": r[0]} for r in rows])
    out["slice"] = {"category": category, "entity_2": entity_2, "entity_3": entity_3}
    return out


def recommended_actions(category: str | None, recall: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Deterministic manufacturer actions; the matched public recall remedy
    (real NHTSA text) is listed first when one exists."""
    actions: list[dict[str, Any]] = []
    if recall and recall.get("remedy"):
        actions.append({
            "action": f"Matched public recall remedy: {str(recall['remedy'])[:400]}",
            "source": "nhtsa_recall", "record_ids": [str(recall.get("advisory_id"))],
        })
    cat = (category or "").strip().upper()
    actions.append({"action": _ACTIONS.get(cat, _DEFAULT_ACTION), "source": "deterministic_playbook",
                    "record_ids": []})
    if component_tier(cat) == "critical":
        actions.append({
            "action": "Safety-critical component: escalate to the product-safety committee and evaluate whether a Part 573 defect report is warranted.",
            "source": "deterministic_playbook", "record_ids": []})
    return actions


TIER_ORDER = {"high": 0, "elevated": 1, "normal": 2}

__all__ = [
    "CRITICAL_COMPONENTS",
    "TIER_ORDER",
    "component_tier",
    "hazard_mentions",
    "recommended_actions",
    "score_records",
    "slice_safety",
]
