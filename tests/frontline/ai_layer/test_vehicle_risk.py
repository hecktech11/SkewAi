"""Per-vehicle fault risk: counts come from the warehouse, no invented numbers.  [AI-LAYER]"""

from __future__ import annotations

from datetime import datetime

import pytest


@pytest.fixture
def vr_pack(tmp_path, monkeypatch):
    from src.data.warehouse import apply_domain_schema, domain_con

    monkeypatch.setenv("DOMAIN_DB_PATH", str(tmp_path))
    pack = "vr_test"
    with domain_con(pack, read_only=False) as con:
        apply_domain_schema(con)
        rid = 0

        def add(make, model, year, cat, n, text="noise"):
            nonlocal rid
            for _ in range(n):
                rid += 1
                con.execute("INSERT INTO records (record_id, occurred_at, received_at, entity_1, entity_2, entity_3, category, text, source) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'NHTSA')",
                            [f"R{rid}", datetime(2020, 1, 1), datetime(2020, 1, 1), year, make, model, cat, text])

        add("HONDA", "CR-V", "2019", "SERVICE BRAKES", 12, "brakes failed and we had a crash")
        add("HONDA", "CR-V", "2019", "ENGINE", 3)
        add("TOYOTA", "CAMRY", "2019", "ENGINE", 40)
        add("TOYOTA", "CAMRY", "2019", "SERVICE BRAKES", 5)
        con.execute("INSERT INTO advisories (advisory_id, issued_at, scope_entity_1, scope_entity_2, scope_entity_3, scope_category, summary, remedy, url, source) "
                    "VALUES ('20V1|2019|HONDA|CR-V|AIR BAGS', ?, '2019', 'HONDA', 'CR-V', 'AIR BAGS', 's', 'Replace inflator.', 'https://www.nhtsa.gov', 'NHTSA')",
                    [datetime(2020, 5, 1)])
    return pack


def test_vehicle_risk_counts_levels_and_advice(vr_pack, no_meta):
    from src.ai_layer.vehicle_risk import vehicle_risk

    out = vehicle_risk("honda", "cr-v", 2019, 50000, pack_id=vr_pack)
    by = {c["component"]: c for c in out["components"]}
    assert out["summary"]["complaints_total"] == 15
    assert out["mileage_used"] is False and out["vehicle"]["mileage"] == 50000
    assert by["AIR BAGS"]["level"] == "act_now" and by["AIR BAGS"]["data"]["recall_count"] == 1
    assert "20V1" in by["AIR BAGS"]["advice"]
    b = by["SERVICE BRAKES"]
    assert b["data"]["complaints"] == 12
    # brake share 12/15 vs model-year share 17/60
    assert b["data"]["relative_rate"] == round((12 / 15) / (17 / 60), 2)
    assert b["level"] == "high" and "brakes inspected" in b["advice"]
    assert b["data"]["hazard"]["crash"] == 12
    assert by["ENGINE"]["level"] == "watch"
    assert out["components"][0]["component"] == "AIR BAGS"
    assert "explanation" not in out and out["mode"] == "deterministic"


def test_vehicle_risk_unknown_vehicle_and_missing_input(vr_pack, no_meta):
    from src.ai_layer.vehicle_risk import vehicle_risk

    assert vehicle_risk("FIAT", "X", 2001, pack_id=vr_pack)["components"] == []
    assert "error" in vehicle_risk("", "CR-V", 2019, pack_id=vr_pack)


def test_explanation_rejects_invented_numbers():
    from src.ai_layer.vehicle_risk import explanation_ok

    facts = [{"component": "SERVICE BRAKES", "complaints": 12, "relative_rate": 2.82}]
    veh = {"year": "2019"}
    assert explanation_ok("Your 2019 car has 12 brake complaints.", facts, veh)
    assert not explanation_ok("Your car has a 40% chance of brake failure.", facts, veh)


def test_vehicle_risk_with_fake_meta_explanation(vr_pack, fake_meta):
    from src.ai_layer.vehicle_risk import vehicle_risk

    fake_meta.chat["vehicle_risk_explain"] = lambda p: {"explanation": "There is a recall on record for the air bags; 12 brake complaints."}
    out = vehicle_risk("HONDA", "CR-V", "2019", pack_id=vr_pack)
    assert out["explanation"].startswith("There is a recall")
    assert out["powered_by"]
    fake_meta.chat["vehicle_risk_explain"] = lambda p: {"explanation": "90% of these cars fail."}
    out2 = vehicle_risk("HONDA", "CR-V", "2019", pack_id=vr_pack)
    assert "explanation" not in out2
    assert [c["level"] for c in out2["components"]] == [c["level"] for c in out["components"]]


def test_vehicle_risk_route(vr_pack, no_meta, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app

    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    monkeypatch.setattr("src.domains.active_pack.resolve_active_pack_id", lambda: vr_pack)
    c = TestClient(app)
    r = c.get("/api/ai/vehicle-risk", params={"make": "HONDA", "model": "CR-V", "year": "2019"})
    assert r.status_code == 200, r.text
    assert r.json()["summary"]["complaints_total"] == 15
    assert c.get("/api/ai/vehicle-risk", params={"make": "HONDA", "model": "CR-V", "year": "abc"}).status_code == 400
