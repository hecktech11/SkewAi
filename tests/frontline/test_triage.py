"""Triage agent tests (per blueprint §15.2).

Covers:
  - Rules path: safety flag → Critical
  - Description-only severity (fire/smoke/crash in description → Critical)
  - Category-based severity (SERVICE BRAKES → Medium, etc.)
  - Priority matrix (P1 for safety, P2 for Medium, P3 for Low)
  - Triage skips when category+description aren't filled
"""

from __future__ import annotations

import pytest

from src.agents.base import InteractionContext
from src.agents.triage import TriageAgent, _eval_condition


def _ctx(pack, **slots) -> InteractionContext:
    ctx = InteractionContext(interaction_id="int_test_triage", pack=pack)
    ctx.slots.update(slots)
    return ctx


# ── Condition evaluator ──────────────────────────────────────────────────────


def test_eval_condition_default(pack):
    ctx = _ctx(pack)
    assert _eval_condition("default", ctx) is True
    assert _eval_condition("always", ctx) is True
    assert _eval_condition("true", ctx) is True
    assert _eval_condition("else", ctx) is True


def test_eval_condition_safety_any(pack):
    ctx = _ctx(pack)
    ctx.safety_flags["escalation"] = False
    assert _eval_condition("safety.any", ctx) is False
    ctx.safety_flags["escalation"] = True
    assert _eval_condition("safety.any", ctx) is True


def test_eval_condition_safety_specific(pack):
    ctx = _ctx(pack)
    ctx.safety_flags["fire"] = False
    assert _eval_condition("safety.fire", ctx) is False
    ctx.safety_flags["fire"] = True
    assert _eval_condition("safety.fire", ctx) is True


def test_eval_condition_category_in(pack):
    ctx = _ctx(pack)
    ctx.slots["category"] = "SERVICE BRAKES"
    assert _eval_condition("category in [SERVICE BRAKES, AIR BAGS]", ctx) is True
    assert _eval_condition("category in [ENGINE, FUEL SYSTEM]", ctx) is False


def test_eval_condition_description_matches(pack):
    ctx = _ctx(pack)
    ctx.slots["description"] = "I smelled smoke and then the car caught fire"
    assert _eval_condition("description_matches_fire|smoke|crash", ctx) is True
    ctx.slots["description"] = "Just a mild rattle"
    assert _eval_condition("description_matches_fire|smoke|crash", ctx) is False


# ── Rules path ─────────────────────────────────────────────────────────────────


async def test_triage_safety_flag_yields_critical(pack):
    """A safety flag set → severity=Critical (the first rule in the pack)."""
    ctx = _ctx(pack, category="SERVICE BRAKES", description="Brakes feel soft")
    ctx.safety_flags["escalation"] = True
    res = await TriageAgent(ctx).run()
    assert res["severity"] == "Critical"
    assert res["severity_source"] in ("rules", "model")  # model falls back to rules
    assert res["priority"] == 1  # safety → P1


async def test_triage_description_smoke_yields_critical(pack):
    """Description contains 'smoke' → Critical per the description_matches rule."""
    ctx = _ctx(pack, category="ENGINE", description="I saw smoke from the engine")
    res = await TriageAgent(ctx).run()
    assert res["severity"] == "Critical"


async def test_triage_service_brakes_yields_medium(pack):
    """SERVICE BRAKES is in the safety-critical-systems Medium bucket."""
    ctx = _ctx(pack, category="SERVICE BRAKES", description="Mild grinding noise")
    res = await TriageAgent(ctx).run()
    assert res["severity"] == "Medium"
    assert res["priority"] == 2


async def test_triage_electrical_yields_medium(pack):
    """ENGINE is a Medium-severity drivetrain component."""
    ctx = _ctx(pack, category="ENGINE", description="Engine misfires at idle")
    res = await TriageAgent(ctx).run()
    assert res["severity"] == "Medium"
    assert res["priority"] == 2


async def test_triage_electrical_system_yields_medium(pack):
    """The canonical category is 'ELECTRICAL SYSTEM', and the rule must match it.

    The pack listed `category in [ELECTRICAL, ENGINE, POWER TRAIN]`, but
    `_eval_condition` compares by exact uppercase membership, so the
    `ELECTRICAL` arm could never fire and ELECTRICAL SYSTEM fell through to
    the default Low rule — the largest category in the corpus (249,833 NHTSA
    complaints, and every live electrical call).

    Asserted against `_apply_rule_severity` rather than the full agent because
    a severity model artifact is configured for this pack and masks the defect
    end-to-end. The rules are the documented fallback whenever that model
    fails (SEVERITY_FALLBACK_POLICY), so this is the layer that must be right.
    """
    from src.agents.triage import _apply_rule_severity

    ctx = _ctx(
        pack, category="ELECTRICAL SYSTEM", description="Dash warning lights flicker"
    )
    severity, source, reason = _apply_rule_severity(ctx)
    assert (severity, source) == ("Medium", "rules")
    assert reason != "Default severity"


async def test_triage_fuel_propulsion_system_is_safety_critical(pack):
    """FUEL/PROPULSION SYSTEM is NHTSA's modern name for the fuel system.

    It carries 62,750 corpus rows and covers EV propulsion. FUEL SYSTEM was
    already listed as safety-critical; the newer spelling must be too, or the
    same failure scores differently depending on which name the source used.
    """
    from src.agents.triage import _apply_rule_severity

    ctx = _ctx(
        pack,
        category="FUEL/PROPULSION SYSTEM",
        description="Fuel smell under the hood",
    )
    severity, source, reason = _apply_rule_severity(ctx)
    assert (severity, source) == ("Medium", "rules")
    assert reason == "Safety-critical system"


async def test_every_severity_rule_category_is_a_real_gazetteer_value(pack):
    """No severity rule may name a category the gazetteer does not declare.

    A rule arm that names a non-existent value is dead code that silently
    downgrades severity — exactly the ELECTRICAL bug above. This guards the
    whole rule set against the next occurrence.
    """
    import re

    from src.domains.nhtsa_components import category_values

    vocabulary = category_values()
    for rule in pack.manifest.severity.rules:
        m = re.search(r"category in \[(.*)\]", rule.when)
        if not m:
            continue
        named = [c.strip().strip("'\"") for c in m.group(1).split(",")]
        unknown = [c for c in named if c and c.upper() not in vocabulary]
        assert unknown == [], (
            f"severity rule {rule.when!r} names categories absent from "
            f"categories.csv: {unknown}"
        )


async def test_triage_default_low(pack):
    """Categories not covered by any rule default to Low."""
    ctx = _ctx(pack, category="STRUCTURE", description="Paint is chipping")
    res = await TriageAgent(ctx).run()
    assert res["severity"] == "Low"
    assert res["priority"] == 3


# ── Priority matrix ───────────────────────────────────────────────────────────


async def test_priority_p1_for_safety(pack):
    ctx = _ctx(pack, category="STRUCTURE", description="Paint chipping")
    ctx.safety_flags["fire"] = True
    res = await TriageAgent(ctx).run()
    assert res["priority"] == 1


async def test_priority_p1_for_critical(pack):
    """Critical (non-safety) → P1."""
    ctx = _ctx(pack, category="ENGINE", description="Engine caught fire")
    res = await TriageAgent(ctx).run()
    assert res["severity"] == "Critical"
    assert res["priority"] == 1


async def test_priority_p2_for_medium(pack):
    ctx = _ctx(pack, category="SERVICE BRAKES", description="Brakes squeal")
    res = await TriageAgent(ctx).run()
    assert res["priority"] == 2


async def test_priority_p3_for_low(pack):
    ctx = _ctx(pack, category="STRUCTURE", description="Paint chipping")
    res = await TriageAgent(ctx).run()
    assert res["priority"] == 3


# ── Skip when precondition missing ──────────────────────────────────────────


async def test_triage_skipped_without_category(pack):
    ctx = _ctx(pack, description="Just a noise")
    res = await TriageAgent(ctx).run()
    assert res.get("skipped") is True


async def test_triage_skipped_without_description(pack):
    ctx = _ctx(pack, category="SERVICE BRAKES")
    res = await TriageAgent(ctx).run()
    assert res.get("skipped") is True
