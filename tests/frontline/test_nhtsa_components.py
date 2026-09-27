"""NHTSA `COMPDESC` normalization onto the pack category vocabulary.

Phase 0 step 1. The invariant under test is the one every later phase rests
on: a complaint and a phone call about the same failure must land in the same
`records.category` slice, so every ingested component has to reduce to a value
that `domains/automotive_nhtsa/gazetteers/categories.csv` declares.

`ALL_TOP_LEVEL` below is the complete measured vocabulary — all 54 distinct
top-level `COMPDESC` values in the full ODI corpus (`FLAT_CMPL.zip`, 2,250,304
complaint rows, retrieved 2026-09-27) with their row counts. Pinning the whole
set, not a sample, is what makes this a guard: if someone edits the alias table
or the gazetteer and silently sends 121,391 engine complaints to
`UNKNOWN OR OTHER`, this test says so.
"""

from __future__ import annotations

import pytest

from src.domains.nhtsa_components import (
    UNKNOWN,
    category_values,
    component_top_level,
    normalize_component,
    validate_alias_table,
)

# (raw top-level COMPDESC, corpus row count, expected normalized category)
ALL_TOP_LEVEL: tuple[tuple[str, int, str], ...] = (
    # ── already exact gazetteer values (23) ─────────────────────────────────
    ("ELECTRICAL SYSTEM", 249833, "ELECTRICAL SYSTEM"),
    ("POWER TRAIN", 219361, "POWER TRAIN"),
    ("ENGINE", 170265, "ENGINE"),
    ("AIR BAGS", 156788, "AIR BAGS"),
    ("STEERING", 144523, "STEERING"),
    ("UNKNOWN OR OTHER", 118249, "UNKNOWN OR OTHER"),
    ("STRUCTURE", 104546, "STRUCTURE"),
    ("SUSPENSION", 90118, "SUSPENSION"),
    ("VEHICLE SPEED CONTROL", 85382, "VEHICLE SPEED CONTROL"),
    ("TIRES", 81372, "TIRES"),
    ("SERVICE BRAKES", 79115, "SERVICE BRAKES"),
    ("EXTERIOR LIGHTING", 63094, "EXTERIOR LIGHTING"),
    ("VISIBILITY", 53309, "VISIBILITY"),
    ("SEAT BELTS", 39333, "SEAT BELTS"),
    ("SEATS", 28912, "SEATS"),
    ("FORWARD COLLISION AVOIDANCE", 28718, "FORWARD COLLISION AVOIDANCE"),
    ("WHEELS", 24678, "WHEELS"),
    ("CHILD SEAT", 16817, "CHILD SEAT"),
    ("LANE DEPARTURE", 8031, "LANE DEPARTURE"),
    ("PARKING BRAKE", 4941, "PARKING BRAKE"),
    ("TRACTION CONTROL SYSTEM", 2345, "TRACTION CONTROL SYSTEM"),
    ("HYBRID PROPULSION SYSTEM", 576, "HYBRID PROPULSION SYSTEM"),
    ("ELECTRONIC STABILITY CONTROL", 335, "ELECTRONIC STABILITY CONTROL"),
    # ── values added to categories.csv for Phase 0 (4) ──────────────────────
    ("FUEL/PROPULSION SYSTEM", 62750, "FUEL/PROPULSION SYSTEM"),
    ("EQUIPMENT", 33123, "EQUIPMENT"),
    ("VISIBILITY/WIPER", 24185, "VISIBILITY/WIPER"),
    ("LATCHES/LOCKS/LINKAGES", 20771, "LATCHES/LOCKS/LINKAGES"),
    # ── wording variants aliased onto an existing value (16) ────────────────
    ("ENGINE AND ENGINE COOLING", 121391, "ENGINE"),
    ("SERVICE BRAKES, HYDRAULIC", 107289, "SERVICE BRAKES"),
    ("FUEL SYSTEM, GASOLINE", 56275, "FUEL SYSTEM"),
    ("ELECTRONIC STABILITY CONTROL (ESC)", 21288, "ELECTRONIC STABILITY CONTROL"),
    ("BACK OVER PREVENTION", 8416, "BACKOVER PREVENTION"),
    ("FUEL SYSTEM, OTHER", 5935, "FUEL SYSTEM"),
    ("SERVICE BRAKES, AIR", 4563, "SERVICE BRAKES"),
    ("FUEL SYSTEM, DIESEL", 3206, "FUEL SYSTEM"),
    ("INTERIOR LIGHTING", 2707, "ELECTRICAL SYSTEM"),
    ("SERVICE BRAKES, ELECTRIC", 2647, "SERVICE BRAKES"),
    ("EQUIPMENT ADAPTIVE/MOBILITY", 1725, "EQUIPMENT"),
    ("TRAILER HITCHES", 1027, "STRUCTURE"),
    ("COMMUNICATION", 111, "ELECTRICAL SYSTEM"),
    ("TRAILER HARDWARE", 1, "STRUCTURE"),
    ("COMMUNICATIONS", 1, "ELECTRICAL SYSTEM"),
    ("SERVICE BRAKES, HYDRAULIC; AUTOHOLD BRAKE SYSTEM/BRAKE HOLD", 8, "SERVICE BRAKES"),
    # ── child-seat part names, Title Case in the source (5) ─────────────────
    ("Chest Clip, Buckle, Harness", 661, "CHILD SEAT"),
    ("Carry Handle, Shell, Base", 370, "CHILD SEAT"),
    ("Insert, Padding", 169, "CHILD SEAT"),
    ("Tether, Lower Anchor (on car seat or vehicle)", 154, "CHILD SEAT"),
    ("I suspect the car seat is counterfeit", 91, "CHILD SEAT"),
    # ── not a component: form escape hatches and incident attributes (6) ────
    ("Other/I am not sure", 443, UNKNOWN),
    ("FIRERELATED", 175, UNKNOWN),
    ("NONE", 14, UNKNOWN),
    ("Other/Unknown", 10, UNKNOWN),
    ("OTHER", 6, UNKNOWN),
    ("ROLLOVER", 2, UNKNOWN),
)


def test_measured_vocabulary_is_complete():
    """Guards the corpus arithmetic the alias table was built from."""
    assert len(ALL_TOP_LEVEL) == 54
    # 2,250,304 rows less the 149 with a blank COMPDESC.
    assert sum(count for _, count, _ in ALL_TOP_LEVEL) == 2_250_155


@pytest.mark.parametrize(
    "raw,expected",
    [(raw, expected) for raw, _, expected in ALL_TOP_LEVEL],
    ids=[raw[:44] for raw, _, _ in ALL_TOP_LEVEL],
)
def test_every_real_top_level_value_normalizes(raw: str, expected: str):
    assert normalize_component(raw) == expected


def test_no_real_value_falls_through_unintentionally():
    """Only the 6 known non-components may reach UNKNOWN OR OTHER.

    `UNKNOWN OR OTHER` is itself a real NHTSA value (118,249 rows), so it is
    a legitimate target too — but nothing else may silently land there.
    """
    unknowns = {
        raw for raw, _, expected in ALL_TOP_LEVEL if expected == UNKNOWN
    }
    assert unknowns == {
        "UNKNOWN OR OTHER",
        "Other/I am not sure",
        "FIRERELATED",
        "NONE",
        "Other/Unknown",
        "OTHER",
        "ROLLOVER",
    }
    # The volume routed to UNKNOWN stays small: the 6 junk values are 650 rows
    # (443 + 175 + 14 + 10 + 6 + 2), 0.03% of the corpus.
    junk = sum(
        count
        for raw, count, expected in ALL_TOP_LEVEL
        if expected == UNKNOWN and raw != "UNKNOWN OR OTHER"
    )
    assert junk == 650


def test_output_is_always_in_the_gazetteer():
    """The Phase 0 "done when": every records.category comes from categories.csv."""
    vocabulary = category_values()
    for raw, _, _ in ALL_TOP_LEVEL:
        assert normalize_component(raw) in vocabulary
    # Including inputs the corpus never contained.
    for junk in ("", "   ", "WARP CORE", "SERVICE BRAKES, PNEUMATIC", "123"):
        assert normalize_component(junk) in vocabulary


# ── component_top_level ─────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        # The ':' hierarchy tail is dropped, however deep.
        ("SERVICE BRAKES, HYDRAULIC:FOUNDATION COMPONENTS:DISC:ROTOR",
         "SERVICE BRAKES, HYDRAULIC"),
        ("ENGINE AND ENGINE COOLING:COOLING SYSTEM", "ENGINE AND ENGINE COOLING"),
        # A space after the colon is common in the source.
        ("FORWARD COLLISION AVOIDANCE: AUTOMATIC EMERGENCY BRAKING",
         "FORWARD COLLISION AVOIDANCE"),
        # ';' is a second separator — 8 corpus rows use it, always with a ':'
        # tail as well, so both splits must apply and ':' must go first.
        ("SERVICE BRAKES, HYDRAULIC; AUTOHOLD BRAKE SYSTEM/BRAKE HOLD:SOFTWARE",
         "SERVICE BRAKES, HYDRAULIC"),
        # The comma is NOT a separator: splitting on it would yield the
        # non-existent component "HYDRAULIC".
        ("SERVICE BRAKES, HYDRAULIC", "SERVICE BRAKES, HYDRAULIC"),
        # Case and whitespace are normalized.
        ("  Chest Clip, Buckle, Harness  ", "CHEST CLIP, BUCKLE, HARNESS"),
        ("ENGINE   AND\tENGINE  COOLING", "ENGINE AND ENGINE COOLING"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_component_top_level(raw: str, expected: str):
    assert component_top_level(raw) == expected


def test_blank_and_none_become_unknown():
    """149 corpus rows have a blank COMPDESC; they must not blank the category."""
    for blank in ("", "   ", "\t", None):
        assert normalize_component(blank) == UNKNOWN  # type: ignore[arg-type]


def test_hierarchy_does_not_change_the_category():
    """A deeper COMPDESC must not split a slice — that would defeat detection."""
    assert (
        normalize_component("SERVICE BRAKES, HYDRAULIC:FOUNDATION COMPONENTS:DISC:ROTOR")
        == normalize_component("SERVICE BRAKES")
        == "SERVICE BRAKES"
    )


# ── alias table integrity ───────────────────────────────────────────────────


def test_alias_table_validates_against_the_live_gazetteer():
    """Every alias target exists, the 4 new categories were added, no dead keys."""
    validate_alias_table()


def test_alias_targets_are_not_themselves_aliased():
    """One hop only — an alias must resolve directly to a gazetteer value."""
    from src.domains.nhtsa_components import COMPONENT_ALIASES

    vocabulary = category_values()
    for key, target in COMPONENT_ALIASES.items():
        assert target in vocabulary, f"{key} -> {target} is not a gazetteer value"
        assert target not in COMPONENT_ALIASES, f"{key} -> {target} chains"
