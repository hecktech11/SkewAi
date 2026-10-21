"""Manufacturer alert feed: thresholds, dedupe, escalation, no sending.  [AI-LAYER]"""

from __future__ import annotations

from datetime import datetime, timezone


def _inv(con, iid, cat, cases):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    con.execute(
        "INSERT INTO investigations (investigation_id, pack_id, cluster_id, title, status, opened_at, case_count, "
        "slice_category, slice_entity_2, slice_entity_3) VALUES (?, 'automotive_nhtsa', 1, ?, 'open', ?, ?, ?, 'HONDA', '')",
        [iid, f"t {iid}", now, cases, cat])


def _setup(monkeypatch, levels):
    monkeypatch.setattr("src.ai_layer.safety_board.slice_safety",
                        lambda pack, c, e2, e3, **k: {"level": levels[c], "mentions": {}})
    monkeypatch.setattr("src.ai_layer.grounded_brief.build_grounded_brief",
                        lambda iid: {"headline": f"brief {iid}", "claims": [{"text": "c", "evidence_ids": ["R1"]}],
                                     "recommended_actions": [{"action": "Inspect brakes", "source": "playbook"}],
                                     "mode": "deterministic", "powered_by": []})


def test_qualifies_thresholds(monkeypatch):
    from src.ai_layer.alerts import qualifies

    monkeypatch.delenv("AI_ALERT_MIN_CASES", raising=False)
    assert qualifies("high", 3) and not qualifies("high", 2)
    assert qualifies("elevated", 10) and not qualifies("elevated", 9)
    assert not qualifies("normal", 999)


def test_scan_creates_dedupes_and_escalates(reset_ops_db, monkeypatch, no_meta):
    from src.ai_layer.alerts import list_alerts, scan_alerts
    from src.data.warehouse import ops_con

    levels = {"SERVICE BRAKES": "high", "AIR BAGS": "elevated", "PAINT": "normal"}
    _setup(monkeypatch, levels)
    with ops_con() as con:
        _inv(con, "inv_b", "SERVICE BRAKES", 4)
        _inv(con, "inv_c", "AIR BAGS", 12)
        _inv(con, "inv_p", "PAINT", 80)
    out = scan_alerts()
    assert len(out["created"]) == 2
    alerts = list_alerts()
    by = {a["investigation_id"]: a for a in alerts}
    assert set(by) == {"inv_b", "inv_c"}
    p = by["inv_b"]["payload"]
    assert p["headline"] == "brief inv_b" and p["recommended_actions"][0]["action"] == "Inspect brakes"
    assert p["delivery"].startswith("not sent")
    assert scan_alerts()["created"] == []  # dedupe
    levels["AIR BAGS"] = "high"
    assert len(scan_alerts()["created"]) == 1  # escalation elevated -> high
    assert len(list_alerts()) == 3


def test_alert_routes_and_feed(reset_ops_db, monkeypatch, no_meta):
    from fastapi.testclient import TestClient

    from src.api.main import app
    from src.data.warehouse import ops_con

    _setup(monkeypatch, {"SERVICE BRAKES": "high"})
    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    with ops_con() as con:
        _inv(con, "inv_b", "SERVICE BRAKES", 5)
    c = TestClient(app)
    assert c.post("/api/ai/alerts/scan").json()["created"]
    body = c.get("/api/ai/alerts").json()
    assert body["count"] == 1 and "nothing is sent" in body["delivery"]
    aid = body["alerts"][0]["alert_id"]
    feed = c.get("/api/ai/alerts/feed.json").json()
    assert feed["version"].startswith("https://jsonfeed.org") and feed["items"][0]["id"] == aid
    assert c.post(f"/api/ai/alerts/{aid}/ack").status_code == 200
    assert c.get("/api/ai/alerts", params={"status": "open"}).json()["count"] == 0
    assert c.post("/api/ai/alerts/alrt_NOPE/ack").status_code == 404
    assert c.post("/api/ai/alerts/bad id/ack").status_code in (400, 404)
