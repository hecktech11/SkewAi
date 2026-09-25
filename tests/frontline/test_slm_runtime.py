"""Tests for SLM runtime (Phase 1 zero-shot category, mode gating, and shadow telemetry)."""

from __future__ import annotations

import os
import pytest

from src.agents.base import InteractionContext
from src.domains.loader import load_pack
from src.ml_runtime.slm_runtime import (
    category_confidence_floor,
    extract_category_with_slm,
    predict_category_zero_shot,
    record_slm_shadow_comparison,
    reset_slm_cache,
    slm_kill_switch,
    slm_mode,
)
from src.observability.degradation import get_ladder, reset_all_ladders, step_down
from src.qubot.claims import BoundClaim, audit_bound_claims


@pytest.fixture(autouse=True)
def clean_slm_env(monkeypatch):
    monkeypatch.delenv("FRONTLINE_SLM_MODE", raising=False)
    monkeypatch.delenv("FRONTLINE_SLM_KILL", raising=False)
    monkeypatch.delenv("FRONTLINE_SLM_CATEGORY_FLOOR", raising=False)
    reset_slm_cache()
    reset_all_ladders()
    yield
    reset_slm_cache()
    reset_all_ladders()


def test_slm_mode_defaults_to_legacy():
    assert slm_mode() == "legacy"
    assert not slm_kill_switch()


def test_slm_mode_gating(monkeypatch):
    monkeypatch.setenv("FRONTLINE_SLM_MODE", "shadow")
    assert slm_mode() == "shadow"

    monkeypatch.setenv("FRONTLINE_SLM_MODE", "live")
    assert slm_mode() == "live"

    # Kill switch overrides any mode to legacy immediately
    monkeypatch.setenv("FRONTLINE_SLM_KILL", "1")
    assert slm_kill_switch()
    assert slm_mode() == "legacy"


def test_zero_shot_category_prediction(pack):
    # Automotive pack test
    res = predict_category_zero_shot("the belt doesnt retract properly", pack)
    assert res.category == "SEAT BELTS"
    assert res.score >= 0.30
    assert res.span_text == "the belt doesnt retract properly" or "belt" in res.span_text
    assert res.latency_ms > 0.0


def test_closed_vocabulary_enforcement(pack):
    # A prediction not in the pack's gazetteer must be rejected
    cat_vectors = predict_category_zero_shot("my checking account was debited twice", pack)
    # Automotive pack does not have financial categories
    assert cat_vectors.category in (None, "UNKNOWN OR OTHER") or cat_vectors.category in pack.gazetteer_for_slot("category").values


def test_confidence_floor_rejection(pack):
    # Extremely high floor rejects uncertain predictions
    res = predict_category_zero_shot(
        "weird noise under the hood",
        pack,
        floor=0.99,
    )
    assert res.category is None
    assert res.score < 0.99


def test_extract_category_legacy_mode(pack, monkeypatch):
    monkeypatch.setenv("FRONTLINE_SLM_MODE", "legacy")
    ctx = InteractionContext(interaction_id="int_test_legacy", pack=pack)
    cat, meta = extract_category_with_slm("brakes failed to stop", ctx)
    assert cat == "SERVICE BRAKES"
    assert meta["source"] == "rules"
    assert meta["slm_mode"] == "legacy"


def test_extract_category_shadow_mode_no_behavior_change(pack, monkeypatch):
    # In shadow mode, the caller receives the EXACT rules extraction
    monkeypatch.setenv("FRONTLINE_SLM_MODE", "shadow")
    ctx = InteractionContext(interaction_id="int_test_shadow", pack=pack)

    # Phrasing where rules extract nothing (None)
    cat, meta = extract_category_with_slm("the belt doesnt retract properly", ctx)
    # Rules return None because 'retract' is not in _CATEGORY_SYNONYMS
    assert cat is None
    assert meta["source"] == "rules"
    assert meta["slm_mode"] == "shadow"
    # But shadow SLM extracted SEAT BELTS in parallel
    assert meta["shadow_slm_category"] == "SEAT BELTS"
    assert meta["shadow_score"] >= 0.30


def test_extract_category_live_mode_promotes_slm(pack, monkeypatch):
    monkeypatch.setenv("FRONTLINE_SLM_MODE", "live")
    ctx = InteractionContext(interaction_id="int_test_live", pack=pack)

    cat, meta = extract_category_with_slm("the belt doesnt retract properly", ctx)
    assert cat == "SEAT BELTS"
    assert meta["source"] == "slm"
    assert meta["slm_mode"] == "live"
    assert meta["score"] >= 0.30


def test_extract_category_live_mode_fallback_to_rules_below_floor(pack, monkeypatch):
    monkeypatch.setenv("FRONTLINE_SLM_MODE", "live")
    monkeypatch.setenv("FRONTLINE_SLM_CATEGORY_FLOOR", "0.99")
    ctx = InteractionContext(interaction_id="int_test_live_fallback", pack=pack)

    cat, meta = extract_category_with_slm("my brakes squeak", ctx)
    # SLM falls below floor, falls back to rules which extract SERVICE BRAKES
    assert cat == "SERVICE BRAKES"
    assert meta["source"] == "rules"
    assert meta.get("fallback_reason") == "slm_below_floor_or_none"


def test_slm_degradation_ladder():
    from src.observability.degradation import init_default_ladders

    init_default_ladders()
    ladder = get_ladder("slm_understanding")
    assert ladder is not None
    assert ladder.current_level == 0
    assert ladder.current.name == "full"

    lev = step_down("slm_understanding", reason="latency_timeout")
    assert lev.name == "rules_fallback"
    assert ladder.current_level == 1

    lev2 = step_down("slm_understanding", reason="repeated_failure")
    assert lev2.name == "fail_closed"
    assert ladder.current_level == 2


def test_qubot_span_groundedness_verification():
    turn_text = "the belt doesnt retract properly when pulled"
    turns = [
        {"seq": 1, "speaker": "customer", "text": turn_text, "turn_id": "turn_001"}
    ]

    # Grounded claim: cited verbatim span matches turn text
    valid_claim = BoundClaim(
        claim_text="the belt doesnt retract properly",
        evidence_id="turn_customer",
        span_start=0,
        span_end=32,
    )
    audit = audit_bound_claims([valid_claim], [], turns=turns)
    assert audit.ok
    assert audit.overall == "grounded"
    assert len(audit.rejected) == 0

    # Mismatched claim: cited claim text does not exist in turn
    tampered_claim = BoundClaim(
        claim_text="engine exploded into flames",
        evidence_id="turn_customer",
        span_start=0,
        span_end=27,
    )
    bad_audit = audit_bound_claims([tampered_claim], [], turns=turns)
    assert not bad_audit.ok
    assert bad_audit.overall == "mismatch"
    assert len(bad_audit.rejected) == 1
