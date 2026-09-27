"""NHTSA ``COMPDESC`` → pack category vocabulary.

Why this exists
---------------
NHTSA writes the failed part as a colon-delimited hierarchy, most specific
last::

    SERVICE BRAKES, HYDRAULIC:FOUNDATION COMPONENTS:DISC:ROTOR
    ENGINE AND ENGINE COOLING:COOLING SYSTEM
    FORWARD COLLISION AVOIDANCE: AUTOMATIC EMERGENCY BRAKING

The pack's ``category`` slot — the one a live caller fills — is validated
against ``domains/automotive_nhtsa/gazetteers/categories.csv``. Early warning
only works if a complaint and a phone call about the same failure land in the
*same* slice, so the complaint's component has to be reduced to that same
controlled vocabulary before it reaches ``records.category``.

The top level alone is not enough: NHTSA's real vocabulary has grown wording
variants over 30 years that mean the same system (``ENGINE`` vs ``ENGINE AND
ENGINE COOLING``, ``SERVICE BRAKES`` vs ``SERVICE BRAKES, HYDRAULIC``). Those
are aliased here. Genuinely-unmappable values become ``UNKNOWN OR OTHER``.

Measured against the full ODI corpus (``FLAT_CMPL.zip``, 2,250,304 complaint
rows, retrieved 2026-09-27): 54 distinct top-level values, of which 23 already
match the gazetteer verbatim. :data:`COMPONENT_ALIASES` covers the other 31.
The row counts in the comments below are from that corpus and are what justify
each routing decision; they are documentation, not logic.

Anything NHTSA adds later that is not listed here degrades to
``UNKNOWN OR OTHER`` rather than inventing a category. The full raw
``COMPDESC`` is preserved separately in ``records.subcategory``, so a value
that grows enough volume to deserve its own slice can be promoted by editing
``categories.csv`` and this table, then reloading — no data is lost in the
meantime.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path

from src.config import settings

#: Bucket for components with no defensible home in the gazetteer.
UNKNOWN = "UNKNOWN OR OTHER"

_WS_RE = re.compile(r"\s+")

# ── raw NHTSA top-level value → canonical gazetteer value ────────────────────
#
# Keys are matched after :func:`component_top_level` normalization (upper-cased,
# whitespace-collapsed, hierarchy and semicolon tails removed). Only values that
# do NOT already match the gazetteer verbatim need an entry.
COMPONENT_ALIASES: dict[str, str] = {
    # ── wording variants of a system the gazetteer already names ────────────
    # NHTSA's older name for the same top-level system.
    "ENGINE AND ENGINE COOLING": "ENGINE",                      # 121,391
    # Brake-circuit subtypes: all one system for slicing purposes.
    "SERVICE BRAKES, HYDRAULIC": "SERVICE BRAKES",              # 107,289
    "SERVICE BRAKES, AIR": "SERVICE BRAKES",                    #   4,563
    "SERVICE BRAKES, ELECTRIC": "SERVICE BRAKES",               #   2,647
    # Fuel-type subtypes of the legacy gasoline-era fuel system.
    "FUEL SYSTEM, GASOLINE": "FUEL SYSTEM",                     #  56,275
    "FUEL SYSTEM, OTHER": "FUEL SYSTEM",                        #   5,935
    "FUEL SYSTEM, DIESEL": "FUEL SYSTEM",                       #   3,206
    # The gazetteer spells this without the space; NHTSA uses one.
    "BACK OVER PREVENTION": "BACKOVER PREVENTION",              #   8,416
    # Parenthetical form carries 98% of ESC volume (21,288 vs 335 bare).
    "ELECTRONIC STABILITY CONTROL (ESC)": "ELECTRONIC STABILITY CONTROL",
    # Interior lighting is an electrical failure, NOT exterior lighting —
    # aliasing it to EXTERIOR LIGHTING would corrupt a real safety slice.
    "INTERIOR LIGHTING": "ELECTRICAL SYSTEM",                   #   2,707
    # Horn / telematics / infotainment.
    "COMMUNICATION": "ELECTRICAL SYSTEM",                       #     111
    "COMMUNICATIONS": "ELECTRICAL SYSTEM",                      #       1
    # Frame-mounted towing attachments.
    "TRAILER HITCHES": "STRUCTURE",                             #   1,027
    "TRAILER HARDWARE": "STRUCTURE",                            #       1
    # Adaptive/mobility conversions are aftermarket equipment.
    "EQUIPMENT ADAPTIVE/MOBILITY": "EQUIPMENT",                 #   1,725

    # ── child-seat part names (NHTSA's car-seat complaint form) ─────────────
    # These arrive in Title Case and name a part of a child seat rather than a
    # vehicle system, so they belong in the CHILD SEAT slice the gazetteer
    # already has — not in UNKNOWN OR OTHER, which would hide 1,445 car-seat
    # complaints from the same slice as a caller reporting a car seat.
    "CHEST CLIP, BUCKLE, HARNESS": "CHILD SEAT",                #     661
    "CARRY HANDLE, SHELL, BASE": "CHILD SEAT",                  #     370
    "INSERT, PADDING": "CHILD SEAT",                            #     169
    "TETHER, LOWER ANCHOR (ON CAR SEAT OR VEHICLE)": "CHILD SEAT",  # 154
    "I SUSPECT THE CAR SEAT IS COUNTERFEIT": "CHILD SEAT",      #      91

    # ── not a component: incident attributes and form escape hatches ───────
    "OTHER/I AM NOT SURE": UNKNOWN,                             #     443
    # A hazard flag, not a failed part.
    "FIRERELATED": UNKNOWN,                                     #     175
    "NONE": UNKNOWN,                                            #      14
    "OTHER/UNKNOWN": UNKNOWN,                                   #      10
    "OTHER": UNKNOWN,                                           #       6
    # An outcome, not a component.
    "ROLLOVER": UNKNOWN,                                        #       2
}

# Top-level values added to categories.csv because they name a real vehicle
# system with no gazetteer equivalent and carry enough volume to produce a
# detectable anomaly on their own. Listed here only so the validation below can
# assert they were actually added:
#   FUEL/PROPULSION SYSTEM   62,750   (NHTSA's modern fuel + EV propulsion)
#   EQUIPMENT                33,123
#   VISIBILITY/WIPER         24,185
#   LATCHES/LOCKS/LINKAGES   20,771
REQUIRED_NEW_CATEGORIES = (
    "FUEL/PROPULSION SYSTEM",
    "EQUIPMENT",
    "VISIBILITY/WIPER",
    "LATCHES/LOCKS/LINKAGES",
)


def categories_path(pack_id: str = "automotive_nhtsa") -> Path:
    """Path to the pack's category gazetteer (``pack_dir`` applies the id jail)."""
    return settings.pack_dir(pack_id) / "gazetteers" / "categories.csv"


@lru_cache(maxsize=8)
def category_values(pack_id: str = "automotive_nhtsa") -> frozenset[str]:
    """The pack's controlled category vocabulary, read straight from the CSV.

    Deliberately does not go through :class:`~src.domains.loader.Gazetteer`:
    normalization must not depend on slot-matching behaviour, and this runs in
    the ingest path where loading a whole pack is unnecessary.
    """
    path = categories_path(pack_id)
    values: set[str] = set()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader, None)  # header
        for row in reader:
            if not row:
                continue
            value = row[0].strip()
            if value and not value.startswith("#"):
                values.add(value)
    if not values:
        raise ValueError(f"category gazetteer is empty: {path}")
    return frozenset(values)


def component_top_level(compdesc: str) -> str:
    """Reduce a raw ``COMPDESC`` to its top-level system, upper-cased.

    Strips the ``:``-delimited hierarchy tail, then a ``;``-delimited tail
    (8 rows use ``SERVICE BRAKES, HYDRAULIC; AUTOHOLD BRAKE SYSTEM/BRAKE
    HOLD``), collapses internal whitespace, and upper-cases. The comma is NOT
    a separator — ``SERVICE BRAKES, HYDRAULIC`` is one value, so splitting on
    commas here would turn it into the non-existent component ``HYDRAULIC``.
    """
    head = str(compdesc or "").split(":", 1)[0]
    head = head.split(";", 1)[0]
    return _WS_RE.sub(" ", head).strip().upper()


def normalize_component(compdesc: str, *, pack_id: str = "automotive_nhtsa") -> str:
    """Map a raw ``COMPDESC`` onto the pack's category vocabulary.

    Always returns a value present in ``categories.csv`` — never the raw input
    and never the empty string. Unmappable input becomes ``UNKNOWN OR OTHER``,
    so ``records.category`` can be relied on as a controlled vocabulary.
    """
    vocabulary = category_values(pack_id)
    top = component_top_level(compdesc)
    if not top:
        return UNKNOWN
    if top in vocabulary:
        return top
    aliased = COMPONENT_ALIASES.get(top)
    if aliased and aliased in vocabulary:
        return aliased
    return UNKNOWN


def validate_alias_table(pack_id: str = "automotive_nhtsa") -> None:
    """Fail loudly when the alias table and the gazetteer disagree.

    Called at ingest start. Without this, editing ``categories.csv`` would
    silently reroute every affected alias to ``UNKNOWN OR OTHER`` and the
    ingest would still "succeed" — the exact failure this module exists to
    prevent.
    """
    vocabulary = category_values(pack_id)
    missing_targets = sorted(
        {t for t in COMPONENT_ALIASES.values() if t not in vocabulary}
    )
    if missing_targets:
        raise ValueError(
            f"COMPONENT_ALIASES targets absent from {categories_path(pack_id)}: "
            f"{missing_targets}"
        )
    if UNKNOWN not in vocabulary:
        raise ValueError(
            f"{UNKNOWN!r} must be present in {categories_path(pack_id)}: it is "
            "the fallback for every unmappable component"
        )
    missing_new = sorted(c for c in REQUIRED_NEW_CATEGORIES if c not in vocabulary)
    if missing_new:
        raise ValueError(
            f"categories absent from {categories_path(pack_id)}: {missing_new}. "
            "These NHTSA components have no alias target, so dropping them "
            "would silently route ~140,000 complaints to UNKNOWN OR OTHER."
        )
    # An alias whose key already IS a gazetteer value is dead code: the exact
    # match in normalize_component wins first. Catch it so the table stays honest.
    shadowed = sorted(set(COMPONENT_ALIASES) & vocabulary)
    if shadowed:
        raise ValueError(
            f"COMPONENT_ALIASES keys shadowed by real gazetteer values "
            f"(unreachable): {shadowed}"
        )


__all__ = [
    "UNKNOWN",
    "COMPONENT_ALIASES",
    "REQUIRED_NEW_CATEGORIES",
    "categories_path",
    "category_values",
    "component_top_level",
    "normalize_component",
    "validate_alias_table",
]
