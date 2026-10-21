"""Severity: final = max(rule, AI), raise-only, grounded.  [AI-LAYER]"""

from __future__ import annotations

import pytest

from src.ai_layer.severity import combine, maybe_raise_severity

ALLOWED = {"Low", "Medium", "Critical"}


def test_combine_is_max_and_never_lowers():
    assert combine("Low", "Critical", allowed=ALLOWED) == "Critical"
    assert combine("Medium", "Low", allowed=ALLOWED) == "Medium"
    assert combine("Critical", "Low", allowed=ALLOWED) == "Critical"
    assert combine("Low", "High", allowed=ALLOWED) == "Low"  # level the pack does not use
    assert combine("Low", None, allowed=ALLOWED) == "Low"
    assert combine("Low", "bogus", allowed=ALLOWED) == "Low"


async def _ctx_with_triage(orchestrator_factory, text):
    from src.agents.intake import IntakeAgent
    from src.agents.triage import TriageAgent

    orch, _ = orchestrator_factory()
    await orch.start()
    orch.ctx.record_turn("customer", text)
    await IntakeAgent(orch.ctx).run(customer_turn=text)
    orch.ctx.slots.setdefault("category", "EXTERIOR LIGHTING")
    orch.ctx.slots.setdefault("description", text)
    tres = await TriageAgent(orch.ctx).run()
    return orch, tres


@pytest.mark.asyncio
async def test_no_key_noop(no_meta, orchestrator_factory):
    orch, tres = await _ctx_with_triage(orchestrator_factory, "my 2019 Honda Civic headlight flickers")
    out = await maybe_raise_severity(orch.ctx, tres)
    assert out is tres and no_meta == []


@pytest.mark.asyncio
async def test_raise_when_grounded(fake_meta, orchestrator_factory):
    text = "my 2019 Honda Civic headlight goes dark at night on the highway and I nearly hit a truck"
    orch, tres = await _ctx_with_triage(orchestrator_factory, text)
    rule = orch.ctx.severity
    assert rule != "Critical"
    fake_meta.chat["severity"] = lambda p: {"severity": "Critical", "confidence": 0.9,
                                             "reason": "loss of lighting at night with near collision",
                                             "evidence": "nearly hit a truck"}
    out = await maybe_raise_severity(orch.ctx, tres)
    assert orch.ctx.severity == "Critical"
    assert out["ai_layer"]["rule"] == rule
    assert out["priority"] == orch.ctx.priority == 1


@pytest.mark.asyncio
async def test_never_lowers(fake_meta, orchestrator_factory):
    orch, tres = await _ctx_with_triage(orchestrator_factory, "my 2019 Honda Civic brakes grind badly")
    orch.ctx.severity = "Medium"
    fake_meta.chat["severity"] = lambda p: {"severity": "Low", "confidence": 0.99, "reason": "minor",
                                             "evidence": "brakes grind"}
    await maybe_raise_severity(orch.ctx, tres)
    assert orch.ctx.severity == "Medium"


@pytest.mark.asyncio
async def test_ungrounded_or_low_confidence_rejected(fake_meta, orchestrator_factory):
    orch, tres = await _ctx_with_triage(orchestrator_factory, "my 2019 Honda Civic headlight flickers")
    rule = orch.ctx.severity
    fake_meta.chat["severity"] = lambda p: {"severity": "Critical", "confidence": 0.99, "reason": "fire",
                                             "evidence": "the car caught fire"}
    await maybe_raise_severity(orch.ctx, tres)
    assert orch.ctx.severity == rule
    fake_meta.chat["severity"] = lambda p: {"severity": "Critical", "confidence": 0.2, "reason": "x",
                                             "evidence": "headlight flickers"}
    await maybe_raise_severity(orch.ctx, tres)
    assert orch.ctx.severity == rule


@pytest.mark.asyncio
async def test_error_falls_back(fake_meta, orchestrator_factory):
    orch, tres = await _ctx_with_triage(orchestrator_factory, "my 2019 Honda Civic headlight flickers")
    rule = orch.ctx.severity
    fake_meta.raise_exc = OSError("down")
    out = await maybe_raise_severity(orch.ctx, tres)
    assert out is tres and orch.ctx.severity == rule
