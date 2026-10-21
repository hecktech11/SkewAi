"""Grounded investigation brief (F) + safety risk / repair actions.  [AI-LAYER]"""

from __future__ import annotations

import pytest

INV = "inv_test_0001"


def _brief(category="SERVICE BRAKES"):
    recs = [
        {"record_id": "NHTSA-11111", "source": "nhtsa_complaint", "received_at": "2024-01-02",
         "entity_1": "HONDA", "text": "Brake pedal went to the floor and the car crashed into a pole. 2 people injured."},
        {"record_id": "NHTSA-22222", "source": "nhtsa_complaint", "received_at": "2024-01-09",
         "entity_1": "HONDA", "text": "Brakes failed at 40 mph, no crash."},
    ]
    return {
        "investigation_id": INV, "pack_id": "automotive_nhtsa", "cluster_id": "cl_1",
        "slice": {"category": category, "entity_2": "CIVIC", "entity_3": "2019"},
        "flagged_weeks": [{"iso_week": "2024-W02", "record_count": 9, "baseline_mean": 2.0, "z_score": 3.5}],
        "similar_records": recs, "records_by_source": {"nhtsa_complaint": recs},
        "association_lift": 2.5,
        "recall": {"advisory_id": "24V123000", "remedy": "Dealers will replace the brake master cylinder.",
                   "lead_time_weeks": 6},
    }


@pytest.fixture
def patched(monkeypatch):
    b = _brief()
    monkeypatch.setattr("src.frontline.investigation_brief.build_investigation_brief", lambda iid: b)
    monkeypatch.setattr("src.qubot.auditor._verify_evidence_id",
                        lambda pack, rid, scope: (rid in {"NHTSA-11111", "NHTSA-22222", INV, "24V123000"}, "x"))
    monkeypatch.setattr("src.ai_layer.safety.slice_safety",
                        lambda pack, c, e2, e3, **k: {**__import__("src.ai_layer.safety", fromlist=["x"]).score_records(
                            c, b["similar_records"]), "slice": {"category": c}})
    monkeypatch.setattr("src.ai_layer.store.record_event", lambda **k: None)
    monkeypatch.setattr("src.ai_layer.grounded_brief.record_event", lambda **k: None)
    return b


def test_no_key_is_deterministic_with_safety_and_actions(no_meta, patched):
    from src.ai_layer.grounded_brief import build_grounded_brief

    out = build_grounded_brief(INV)
    assert out["mode"] == "deterministic" and out["fallback_reason"] == "no_key"
    assert out["powered_by"] == [] and no_meta == []
    assert all(c["verified"] and c["record_ids"] for c in out["claims"])
    assert out["safety_risk"]["level"] == "high"
    assert out["safety_risk"]["mentions"]["crash"] == 1  # "no crash" is negated
    assert out["safety_risk"]["mentions"]["injury"] == 1
    srcs = [a["source"] for a in out["recommended_actions"]]
    assert srcs[0] == "recall_advisory" and "deterministic_playbook" in srcs
    assert any("Part 573" in a["action"] for a in out["recommended_actions"])


def test_enhanced_keeps_only_verified_claims(fake_meta, patched):
    from src.ai_layer.grounded_brief import build_grounded_brief

    fake_meta.chat["grounded_brief"] = lambda p: {
        "headline": "Brake failures with crash reports on Civic",
        "recommended_action": "Inspect master cylinder seals on affected builds before customer delivery.",
        "claims": [
            {"text": "An owner reported a crash after brake pedal went to the floor.", "record_ids": ["NHTSA-11111"]},
            {"text": "Invented id claim.", "record_ids": ["NHTSA-99999"]},
            {"text": "No citation.", "record_ids": []},
            {"text": "17 owners were injured.", "record_ids": ["NHTSA-11111"]},
            {"text": "Also see NHTSA-22222.", "record_ids": ["NHTSA-11111"]},
            {"text": "We will fix it for free.", "record_ids": ["NHTSA-22222"]},
        ],
    }
    out = build_grounded_brief(INV)
    assert out["mode"] == "enhanced" and out["powered_by"]
    assert [c["record_ids"] for c in out["claims"]] == [["NHTSA-11111"]]
    reasons = sorted(d["reason"].split(":")[0] for d in out["dropped_claims"])
    assert reasons == ["invented_id", "narration_violation", "no_citation", "uncited_id_in_text", "unsupported_number"]
    ai = [a for a in out["recommended_actions"] if a["source"] == "meta_muse_spark"]
    assert ai and "engineering review" in ai[0]["note"]


def test_ai_action_with_numbers_rejected_and_all_dropped_falls_back(fake_meta, patched):
    from src.ai_layer.grounded_brief import build_grounded_brief

    fake_meta.chat["grounded_brief"] = lambda p: {
        "headline": "x", "recommended_action": "Recall 5000 cars.",
        "claims": [{"text": "bad", "record_ids": ["FAKE-1"]}]}
    out = build_grounded_brief(INV)
    assert out["mode"] == "deterministic"
    assert out["fallback_reason"] == "all_claims_failed_verification"
    assert all(a["source"] != "meta_muse_spark" for a in out["recommended_actions"])


def test_transport_error_falls_back(fake_meta, patched):
    from src.ai_layer.grounded_brief import build_grounded_brief

    fake_meta.raise_exc = OSError("down")
    out = build_grounded_brief(INV)
    assert out["mode"] == "deterministic" and out["fallback_reason"]
    assert out["claims"] and out["safety_risk"]["level"] == "high"


def test_route_returns_brief_and_404(no_meta, patched, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app

    c = TestClient(app)
    r = c.get(f"/api/ai/investigations/{INV}/grounded-brief")
    assert r.status_code == 200 and r.json()["safety_risk"]["level"] == "high"

    def _missing(iid):
        raise LookupError(iid)

    monkeypatch.setattr("src.frontline.investigation_brief.build_investigation_brief", _missing)
    assert c.get("/api/ai/investigations/nope/grounded-brief").status_code == 404
