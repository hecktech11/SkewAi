"""Safety-risk indicator + manufacturer actions.  [AI-LAYER]"""

from src.ai_layer.safety import component_tier, hazard_mentions, recommended_actions, score_records


def test_component_tiers():
    assert component_tier("service brakes") == "critical"
    assert component_tier("AIR BAGS") == "critical"
    assert component_tier("ELECTRICAL SYSTEM") == "elevated"
    assert component_tier("UNKNOWN OR OTHER") == "other"
    assert component_tier(None) == "other"


def test_hazard_mentions_and_negation():
    assert hazard_mentions("the car caught fire on the highway") == {"fire"}
    assert hazard_mentions("rear-ended another car, driver was hospitalized") == {"crash", "injury"}
    assert hazard_mentions("no injuries and no crash, thankfully") == set()
    assert hazard_mentions("") == set()


def test_levels():
    fire = [{"text": "engine fire"}] + [{"text": "noise"}] * 4
    assert score_records("SERVICE BRAKES", fire)["level"] == "high"
    assert score_records("SERVICE BRAKES", [{"text": "noise"}] * 20)["level"] == "elevated"
    assert score_records("PAINT", fire)["level"] == "elevated"
    assert score_records("PAINT", [{"text": "noise"}] * 20)["level"] == "normal"
    empty = score_records("PAINT", [])
    assert empty["level"] == "normal" and empty["records_scanned"] == 0


def test_actions():
    acts = recommended_actions("STEERING", {"advisory_id": "24V1", "remedy": "Replace rack"})
    assert acts[0]["source"] == "recall_advisory" and acts[0]["record_ids"] == ["24V1"]
    assert any("Part 573" in a["action"] for a in acts)
    assert len(recommended_actions("PAINT")) == 1
    assert len(recommended_actions("STEERING", {"advisory_id": "x", "remedy": None})) == 2
