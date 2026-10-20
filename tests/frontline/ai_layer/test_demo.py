"""Seeded India demo scenario.  [AI-LAYER]"""

from __future__ import annotations

import pytest


def test_synthetic_png_is_valid_and_labelled():
    from src.ai_layer.demo import synthetic_rotor_png
    from src.ai_layer.evidence import validate_image

    png = synthetic_rotor_png(120, 90)
    meta = validate_image(png, "image/png")
    assert (meta["width"], meta["height"]) == (120, 90)
    assert b"SYNTHETIC" in png


@pytest.mark.asyncio
async def test_seed_without_key_runs_real_pipeline(no_meta, pack, reset_ops_db, monkeypatch, tmp_path):
    from src.ai_layer.demo import seed_india_scenario
    from src.data.warehouse import ops_con

    monkeypatch.setenv("AI_EVIDENCE_DIR", str(tmp_path / "ev"))
    out = await seed_india_scenario(pack_id=pack.id)
    assert no_meta == []  # deterministic: no network
    assert out["ai_mode"] == "deterministic"
    assert out["interaction_id"] and out["state"] in ("DONE", "ABANDONED", "SUPERVISED", "HANDOFF_PENDING")
    assert out["evidence"]["analysis_status"] == "not_run"
    assert out["evidence"]["source"] == "synthetic_drawing"
    with ops_con() as con:
        ch = con.execute("SELECT channel FROM interactions WHERE interaction_id = ?", [out["interaction_id"]]).fetchone()
        n = con.execute("SELECT COUNT(*) FROM case_evidence WHERE evidence_id = ?", [out["evidence"]["evidence_id"]]).fetchone()[0]
    assert ch[0] == "simulated" and n == 1
    # the English turn makes the deterministic path fill the vehicle slots
    assert out["slots"].get("entity_2") == "HONDA"


def test_demo_route_disabled_by_default(reset_ops_db, monkeypatch, no_meta):
    from fastapi.testclient import TestClient

    from src.api.main import app

    monkeypatch.delenv("AI_DEMO_ENABLED", raising=False)
    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    assert TestClient(app).post("/api/ai/demo/seed").status_code == 403


def test_muse_image_photo_none_without_key(no_meta):
    from src.ai_layer.demo import muse_image_photo

    assert muse_image_photo() is None


def test_muse_image_photo_with_fake_key(fake_meta):
    import base64
    import json

    from src.ai_layer import demo
    from src.ai_layer.meta_client import HttpResponse, set_transport

    png = demo.synthetic_rotor_png(8, 6)

    def transport(url, body, headers, timeout):
        assert url.endswith("/images/generations")
        return HttpResponse(status=200, body=json.dumps({"data": [{"b64_json": base64.b64encode(png).decode()}]}).encode())

    set_transport(transport)
    got = demo.muse_image_photo()
    assert got and got[0] == png


@pytest.mark.asyncio
async def test_seed_with_fake_meta_escalates_hinglish_brake_failure(fake_meta, pack, reset_ops_db, monkeypatch, tmp_path):
    """Hero moment: deterministic misses 'brakes fail ho gaye'; Muse Spark's
    normalisation lets the EXISTING pack kill-switch lexicon fire."""
    from src.ai_layer.demo import seed_india_scenario

    monkeypatch.setenv("AI_EVIDENCE_DIR", str(tmp_path / "ev"))

    def intake(p):
        user = p["messages"][-1]["content"] if isinstance(p["messages"][-1]["content"], str) else ""
        failed = "fail ho gaye" in user
        return {"language": "hinglish",
                "normalized_en": ("Namaste. My 2019 Honda CR-V brakes failed, the pedal goes to the floor and they grind."
                                  if failed else "ok"),
                "normalization_confidence": 0.95 if failed else 0.1,
                "slots": [], "frustration": 0.5, "frustration_reason": "worried"}

    fake_meta.chat["intake_slots"] = intake
    fake_meta.chat["severity"] = lambda p: {"severity": "Critical", "confidence": 0.9,
                                             "reason": "brake failure", "evidence": "brakes fail"}
    fake_meta.chat["evidence_vision"] = lambda p: {"component_category": "SERVICE BRAKES", "confidence": 0.9,
                                                    "defect_description": "scored rotor", "damaged_part": "brake rotor",
                                                    "visible_hazard": True, "hazard_reason": "deep grooves"}
    fake_meta.sam = lambda p: "0<|box;x1=10;y1=10;x2=50;y2=40;w=120;h=90|>"
    out = await seed_india_scenario(pack_id=pack.id)
    assert out["ai_mode"] == "enhanced"
    assert "escalation" in out["safety_flags"]
    assert out["severity"] == "Critical"
    assert any("do not" in t["text"].lower() for t in out["transcript"] if t["speaker"] == "agent")
    assert out["evidence"]["analysis_status"] in ("ok", "partial")


def test_safety_contrast_without_key_is_identity(no_meta):
    from src.ai_layer.demo import INDIA_SCENARIO_TURNS, safety_contrast

    out = safety_contrast(INDIA_SCENARIO_TURNS[0], "automotive_nhtsa")
    assert out["lang"] == "hinglish"
    assert out["original_kill_switch"] is None
    assert out["translated_by_meta"] is False
    assert out["final_kill_switch"] is None


def test_safety_contrast_english_hits_existing_lexicon(no_meta):
    from src.ai_layer.demo import safety_contrast

    out = safety_contrast("my brakes failed on the highway", "automotive_nhtsa")
    assert out["original_kill_switch"] == "brakes failed"
    assert out["final_kill_switch"] == "brakes failed"


def test_safety_contrast_with_fake_meta_translation_fires_lexicon(fake_meta):
    from src.ai_layer.demo import INDIA_SCENARIO_TURNS, safety_contrast

    fake_meta.chat["normalize"] = lambda p: {
        "normalized_en": "Hello. The brakes failed on my 2019 Honda CR-V, the pedal goes all the way down and there is a grinding sound.",
        "confidence": 0.92,
    }
    out = safety_contrast(INDIA_SCENARIO_TURNS[0], "automotive_nhtsa")
    assert out["translated_by_meta"] is True
    assert out["original_kill_switch"] is None
    assert out["normalized_kill_switch"] == "brakes failed"
    assert out["final_kill_switch"] == "brakes failed"


def test_safety_contrast_route(reset_ops_db, no_meta, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app

    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)

    r = TestClient(app).post("/api/ai/demo/safety-contrast", json={"text": "my brakes failed"})
    assert r.status_code == 200, r.text
    assert r.json()["final_kill_switch"] == "brakes failed"
