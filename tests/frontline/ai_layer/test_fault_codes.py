"""OBD-II fault codes (DTCs).  [AI-LAYER]  No key, no network.

All codes below are clearly labelled EXAMPLE codes used only in tests; generic
meanings come from src/ai_layer/data/dtc_lookup.json.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.ai_layer import fault_codes as fc

EXAMPLE_SAFETY = "C0035"   # EXAMPLE: wheel speed sensor -> SERVICE BRAKES (safety)
EXAMPLE_MISFIRE = "P0300"  # EXAMPLE: random misfire -> ENGINE (elevated, no raise)
EXAMPLE_UNKNOWN = "P0999"  # EXAMPLE: valid format, not in the table


def _case(con, cid, sev, cat, *, inv=None, make="HONDA", model="CR-V", year="2019"):
    now = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=1)
    iid = "int_" + cid
    con.execute("INSERT INTO interactions (interaction_id, pack_id, pack_version, started_at, channel, status, entity_1, entity_2, entity_3) "
                "VALUES (?, 'automotive_nhtsa', 'v', ?, 'web_text', 'closed', ?, ?, ?)", [iid, now, year, make, model])
    con.execute("INSERT INTO cases (case_id, interaction_id, pack_id, created_at, category, severity, severity_source, priority, "
                "safety_flags, status, case_kind, investigation_id) VALUES (?, ?, 'automotive_nhtsa', ?, ?, ?, 'rules', 3, '{}', 'open', 'customer', ?)",
                [cid, iid, now, cat, sev, inv])
    return iid


def test_format_validation_and_parsing():
    assert fc.is_valid("C0035") and fc.is_valid("U0121") and fc.is_valid("P0A0F")
    for bad in ("X0035", "C4035", "C003", "C00350", "c0035 ", "P0G00"):
        assert not fc.is_valid(bad), bad
    valid, invalid = fc.split_codes("c0035, P0300 ; ZZ99")
    assert valid == ["C0035", "P0300"] and invalid == ["ZZ99"]
    assert fc.codes_in_text("brakes grinding. codes: C0035 P0300") == ["C0035", "P0300"]
    assert fc.codes_in_text("I drove past exit P0300 yesterday") == []  # no 'codes:' marker -> nothing


def test_lookup_known_unknown_and_ranges():
    assert fc.lookup("C0050")["component"] == "SERVICE BRAKES" and fc.lookup("C0050")["safety"] == "safety"
    assert fc.lookup("P0306")["component"] == "ENGINE" and fc.lookup("P0306")["safety"] == "elevated"
    u = fc.lookup(EXAMPLE_UNKNOWN)
    assert u["known"] is False and u["meaning"] == "unknown code" and u["component"] is None
    assert fc.lookup("C0055")["known"] is False  # outside C0035-C0050
    t = fc.table()
    assert t["_source"] and all(r["confidence"] in ("high", "medium") for r in t["ranges"])


def test_assess_floor_and_needs_review():
    a = fc.assess([EXAMPLE_SAFETY, EXAMPLE_UNKNOWN], "SERVICE BRAKES")
    assert a["floor"] == "Medium" and not a["needs_review"] and a["unknown"] == [EXAMPLE_UNKNOWN]
    b = fc.assess([EXAMPLE_MISFIRE], "SERVICE BRAKES")
    assert b["floor"] is None and b["needs_review"] is True and "ENGINE" in b["codes"][0]["review_reason"]
    assert fc.assess([EXAMPLE_UNKNOWN], "ENGINE")["floor"] is None  # unknown -> no effect


def test_raise_only_never_lowers():
    assert fc.raise_only("Low", "Medium") == "Medium"
    assert fc.raise_only("Critical", "Medium") == "Critical"
    assert fc.raise_only("Medium", None) == "Medium"
    assert fc.raise_only("Low", "Medium", {"Low", "Critical"}) == "Low"  # floor not an allowed level


def test_attach_to_case_raises_and_flags(reset_ops_db):
    from src.data.warehouse import ops_con

    with ops_con() as con:
        _case(con, "case_fc_low", "Low", "SERVICE BRAKES")
        _case(con, "case_fc_crit", "Critical", "SERVICE BRAKES")
    out = fc.attach_to_case("case_fc_low", [EXAMPLE_SAFETY])
    assert out["raised"] and out["severity_after"] == "Medium"
    crit = fc.attach_to_case("case_fc_crit", [EXAMPLE_SAFETY, EXAMPLE_MISFIRE])
    assert crit["raised"] is False and crit["severity_after"] == "Critical" and crit["needs_review"] is True
    with ops_con() as con:
        sev = dict(con.execute("SELECT case_id, severity FROM cases").fetchall())
    assert sev == {"case_fc_low": "Medium", "case_fc_crit": "Critical"}
    assert fc.attach_to_case("case_missing", [EXAMPLE_SAFETY])["error"] == "not_found"
    assert fc.attach_to_case("case_fc_low", ["BAD1"])["error"] == "invalid"


def test_counts_queue_bonus_vehicle_and_investigation(reset_ops_db):
    from src.ai_layer.service_centre import score_case, service_queue
    from src.data.warehouse import ops_con

    with ops_con() as con:
        _case(con, "case_a", "Medium", "SERVICE BRAKES", inv="inv_fc")
        _case(con, "case_b", "Medium", "SERVICE BRAKES", inv="inv_fc")
        _case(con, "case_c", "Medium", "ENGINE", inv="inv_fc")
    fc.store([EXAMPLE_SAFETY], case_id="case_a", complaint_category="SERVICE BRAKES")
    fc.store([EXAMPLE_SAFETY], interaction_id="int_case_c", complaint_category="ENGINE")  # disagrees -> not matching
    v = fc.matching_for_vehicle("HONDA", "CR-V", "2019")
    assert v["cases_with_matching_fault_code"] == 1 and v["by_component"] == {"SERVICE BRAKES": 1}
    assert fc.matching_for_investigation("inv_fc") == 1
    base, _ = score_case({"severity": "Medium", "category": "SERVICE BRAKES"})
    pts, reasons = score_case({"severity": "Medium", "category": "SERVICE BRAKES", "fault_codes": [EXAMPLE_SAFETY]})
    assert pts == base + 10 and any("C0035" in r and "(+10)" in r for r in reasons)
    assert score_case({"severity": "Medium", "category": "ENGINE", "fault_codes": [EXAMPLE_MISFIRE]})[0] == \
        score_case({"severity": "Medium", "category": "ENGINE"})[0]
    q = {x["case_id"]: x for x in service_queue()["queue"]}
    assert q["case_a"]["fault_codes"] == [EXAMPLE_SAFETY] and q["case_b"]["fault_codes"] == []
    assert q["case_a"]["score"] == q["case_b"]["score"] + 10


@pytest.mark.asyncio
async def test_live_hooks_capture_and_raise(no_meta, pack, reset_ops_db):
    from src.agents.base import InteractionContext

    ctx = InteractionContext(interaction_id="int_fc_live", pack=pack, channel="whatsapp_mock")
    await fc.maybe_capture_fault_codes(ctx, "brake light on. codes: C0035 P0999")
    assert fc.codes_for_interaction("int_fc_live") == ["C0035", "P0999"]
    ctx.slots["category"] = "SERVICE BRAKES"
    ctx.severity = "Low"
    out = await fc.maybe_raise_severity_dtc(ctx, {"severity": "Low", "priority": 3})
    assert out["severity"] == "Medium" and ctx.severity == "Medium" and out["fault_codes"]["unknown"] == ["P0999"]
    ctx2 = InteractionContext(interaction_id="int_fc_none", pack=pack, channel="web_text")
    tres = {"severity": "Low"}
    assert (await fc.maybe_raise_severity_dtc(ctx2, tres)) is tres  # no codes -> unchanged
    assert no_meta == []


def test_routes_validate_and_show(reset_ops_db, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app
    from src.data.warehouse import ops_con

    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    with ops_con() as con:
        _case(con, "case_r", "Low", "SERVICE BRAKES")
    c = TestClient(app)
    assert c.post("/api/ai/fault-codes/lookup", json={"fault_codes": "C0035 XYZ"}).status_code == 422
    r = c.post("/api/ai/fault-codes/lookup", json={"fault_codes": ["C0035"]})
    assert r.status_code == 200 and r.json()["codes"][0]["component"] == "SERVICE BRAKES"
    assert c.post("/api/ai/cases/nope/fault-codes", json={"fault_codes": ["C0035"]}).status_code == 404
    r = c.post("/api/ai/cases/case_r/fault-codes", json={"fault_codes": ["C0035", "P0300"]})
    assert r.status_code == 201 and r.json()["severity_after"] == "Medium" and r.json()["needs_review"] is True
    g = c.get("/api/ai/cases/case_r/fault-codes").json()
    assert [x["code"] for x in g["codes"]] == ["C0035", "P0300"]
    r = c.post("/api/ai/interactions/int_x/fault-codes", json={"fault_codes": "U0121"})
    assert r.status_code == 201 and fc.codes_for_interaction("int_x") == ["U0121"]


@pytest.mark.asyncio
async def test_whatsapp_mock_accepts_codes(no_meta, pack, reset_ops_db):
    from src.ai_layer import whatsapp_mock as wm

    wm.reset_sessions()
    body = {"entry": [{"changes": [{"value": {"messages": [
        {"from": "919999000001", "id": "wamid.EXAMPLE1", "type": "text",
         "text": {"body": "My 2019 Honda CR-V ABS light is on. codes: C0035 P0300"}}]}}]}]}
    out = await wm.handle_webhook(body, pack_id="automotive_nhtsa")
    res = out["results"][0]
    assert [x["code"] for x in res["fault_codes"]] == ["C0035", "P0300"]
    assert res["fault_codes"][0]["component"] == "SERVICE BRAKES"
