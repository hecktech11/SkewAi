"""Severity eval harness: rules-only vs rules+Meta.  [AI-LAYER]  All Meta calls faked."""

from __future__ import annotations

import pytest

from eval.enhanced import run_eval_modes as evm

CALLS = [
    {"id": "en_t1", "lang": "en", "kill_switch": False,
     "text": "My 2019 Honda CR-V - the brakes make a loud grinding noise when I stop."},
    {"id": "hl_t1", "lang": "hinglish", "kill_switch": False,
     "text": "Meri Honda CR-V 2019 mein problem hai, rukte waqt brakes se bahut grinding ki awaaz aati hai."},
    {"id": "hi_t1", "lang": "hi", "kill_switch": False,
     "text": "मेरी 2019 की होंडा CR-V में समस्या है, रुकते समय ब्रेक से तेज़ घिसने की आवाज़ आती है।"},
]


def _intake_reply(p):
    return {"language": "en", "normalized_en": "", "normalization_confidence": 0.0, "slots": [],
            "frustration": 0.1, "frustration_reason": "calm"}


@pytest.mark.asyncio
async def test_no_key_rules_only_runs_and_enhanced_not_run(no_meta, pack, reset_ops_db):
    out = await evm.run_all(CALLS)
    assert out["rules_plus_meta"] == "not run" and out["meta"]["enhanced_run"] is False
    assert out["rules_only"]["all"]["n"] == 3 and out["rules_only"]["all"]["meta_calls"] == 0
    assert out["rules_only"]["all"]["pipeline_errors"] == 0
    md = evm.render(out["meta"], out["rules_only"], None)
    assert "NOT RUN" in md and md.count("| not run |") >= 4
    assert no_meta == []


@pytest.mark.asyncio
async def test_enhanced_raise_counted_never_lowers(fake_meta, pack, reset_ops_db):
    fake_meta.chat["intake_slots"] = _intake_reply
    fake_meta.chat["severity"] = lambda p: {"severity": "Critical", "confidence": 0.95,
                                             "reason": "grinding brakes", "evidence": "grinding"}
    out = await evm.run_all(CALLS[:2])
    c = out["rules_plus_meta"]
    assert isinstance(c, dict)
    assert c["all"]["lowers"] == 0
    assert c["all"]["raises"] >= 1 and c["all"]["meta_calls"] >= 1
    assert c["all"]["meta_failures_fallback_to_rules"] == 0
    for row in out["per_call"]:
        r, e = row["rules_only"]["severity"], row["rules_plus_meta"]["severity"]
        if r and e:
            assert evm.ORDER[e] >= evm.ORDER[r]


@pytest.mark.asyncio
async def test_enhanced_failures_fall_back_to_rules(fake_meta, pack, reset_ops_db):
    # no handlers registered -> FakeMeta answers 500 on every chat call
    out = await evm.run_all(CALLS[:2], force_enhanced=True)
    c = out["rules_plus_meta"]
    assert c["all"]["raises"] == 0 and c["all"]["lowers"] == 0
    assert c["all"]["meta_failures_fallback_to_rules"] >= 1 and c["all"]["pipeline_errors"] == 0
    assert c["all"]["severity_agreement"] in (1.0, None)


@pytest.mark.asyncio
async def test_enhanced_timeout_falls_back(fake_meta, pack, reset_ops_db):
    fake_meta.raise_exc = TimeoutError("slow")
    out = await evm.run_all(CALLS[:1], force_enhanced=True)
    c = out["rules_plus_meta"]["all"]
    assert c["pipeline_errors"] == 0 and c["lowers"] == 0 and c["meta_failures_fallback_to_rules"] >= 1
