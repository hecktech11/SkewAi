"""Service-centre call-in queue: transparent deterministic ranking.  [AI-LAYER]"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone


def _case(con, cid, sev, cat, *, flags=None, adv=None, inv=None, days=0, status="open", kind="customer"):
    now = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    iid = "int_" + cid
    con.execute("INSERT INTO interactions (interaction_id, pack_id, pack_version, started_at, channel, status, entity_1, entity_2, entity_3) "
                "VALUES (?, 'automotive_nhtsa', 'v', ?, 'web_text', 'closed', '2019', 'HONDA', 'CR-V')", [iid, now])
    con.execute("INSERT INTO cases (case_id, interaction_id, pack_id, created_at, category, severity, severity_source, priority, "
                "safety_flags, advisory_match_id, investigation_id, status, case_kind) VALUES (?, ?, 'automotive_nhtsa', ?, ?, ?, 'rules', 2, ?, ?, ?, ?, ?)",
                [cid, iid, now, cat, sev, json.dumps(flags or {}), adv, inv, status, kind])


def test_score_case_explains_every_point():
    from src.ai_layer.service_centre import score_case

    now = datetime(2026, 1, 10)
    pts, reasons = score_case({"severity": "Critical", "safety_flags": '{"escalation": true, "smoke": true}',
                               "category": "SERVICE BRAKES", "advisory_match_id": "19V1", "investigation_id": "inv_1",
                               "created_at": datetime(2026, 1, 5)}, now)
    assert pts == 60 + 30 + 20 + 15 + 10 + 5
    assert len(reasons) == 6 and any("smoke" in r for r in reasons)


def test_queue_ranks_and_filters(reset_ops_db):
    from src.ai_layer.service_centre import service_queue
    from src.data.warehouse import ops_con

    with ops_con() as con:
        _case(con, "case_low", "Low", "PAINT")
        _case(con, "case_brk", "Medium", "SERVICE BRAKES", adv="19V383000")
        _case(con, "case_crit", "Critical", "ENGINE", flags={"escalation": True, "smoke": True})
        _case(con, "case_closed", "Critical", "AIR BAGS", status="closed")
        _case(con, "case_sim", "Critical", "AIR BAGS", kind="simulated")
    q = service_queue()
    ids = [x["case_id"] for x in q["queue"]]
    assert ids[0] in ("case_crit", "case_sim") and "case_closed" not in ids and ids[-1] == "case_low"
    assert "case_sim" not in [x["case_id"] for x in service_queue(include_simulated=False)["queue"]]
    brk = next(x for x in q["queue"] if x["case_id"] == "case_brk")
    assert "19V383000" in brk["action"] and "brakes inspected" in brk["action"]
    assert brk["vehicle"] == "2019 HONDA CR-V"


def test_queue_route(reset_ops_db, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app

    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    r = TestClient(app).get("/api/ai/service-centre/queue")
    assert r.status_code == 200 and r.json()["queue"] == []
