"""eval/enhanced harness.  [AI-LAYER]"""

from __future__ import annotations

import pytest

from eval.enhanced.build_calls import build
from eval.enhanced.run_eval import render, run_call, score


def test_call_set_is_synthetic_balanced_and_labelled():
    rows = build()
    assert len(rows) == 60 and all(r["synthetic"] for r in rows)
    assert {r["lang"] for r in rows} == {"en", "hi", "hinglish"}
    assert sum(r["lang"] == "hi" for r in rows) == 20
    assert all(set(r["expected"]) == {"entity_2", "entity_3", "entity_1", "category"} for r in rows)
    assert sum(r["kill_switch"] for r in rows) == 12


def test_score_and_render_not_run():
    calls = [{"lang": "en", "kill_switch": False, "expected": {"entity_2": "HONDA", "entity_3": "CR-V", "entity_1": "2019", "category": "SERVICE BRAKES"}},
             {"lang": "en", "kill_switch": True, "expected": {"entity_2": "HONDA", "entity_3": "CR-V", "entity_1": "2019", "category": "SERVICE BRAKES"}}]
    res = [{"slots": {"entity_2": "HONDA", "entity_3": "CR-V", "entity_1": "2019", "category": "ENGINE"}, "kill_switch": False},
           {"slots": {}, "kill_switch": True}]
    s = score(calls, res)
    assert s["en"]["slot_accuracy"] == {"make": 1.0, "model": 1.0, "year": 1.0, "component": 0.0}
    assert s["en"]["kill_switch_recall"] == 1.0 and s["en"]["kill_switch_false_alarms"] == 0
    md = render(s, None, {"n": 2, "model": "x", "generated_at": "t"})
    assert "not run (no META_API_KEY configured)" in md


@pytest.mark.asyncio
async def test_run_call_det_vs_enhanced_hindi(fake_meta, pack, reset_ops_db):
    call = {"id": "hi_t", "lang": "hi", "kill_switch": True, "text": "मेरी 2019 की होंडा CR-V में समस्या है, हाईवे पर ब्रेक पूरी तरह फेल हो गए।",
            "expected": {"entity_2": "HONDA", "entity_3": "CR-V", "entity_1": "2019", "category": "SERVICE BRAKES"}}
    det = await run_call(pack, call, enhanced=False)
    assert det["kill_switch"] is False
    fake_meta.chat["intake_slots"] = lambda p: {
        "language": "hi", "normalized_en": "My 2019 Honda CR-V has a problem, on the highway the brakes failed completely.",
        "normalization_confidence": 0.95, "slots": [], "frustration": 0.3, "frustration_reason": "x"}
    enh = await run_call(pack, call, enhanced=True)
    assert enh["kill_switch"] is True
