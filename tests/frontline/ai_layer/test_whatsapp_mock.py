"""WhatsApp intake MOCK: Cloud-API-shaped body -> existing orchestrator; nothing sent.  [AI-LAYER]"""

from __future__ import annotations

import pytest


def _body(wa_id, text, mid, mtype="text"):
    m = {"from": wa_id, "id": mid, "timestamp": "1700000000", "type": mtype}
    if mtype == "text":
        m["text"] = {"body": text}
    return {"object": "whatsapp_business_account", "entry": [{"id": "MOCK", "changes": [{"field": "messages", "value": {
        "messaging_product": "whatsapp", "contacts": [{"wa_id": wa_id, "profile": {"name": "Test"}}], "messages": [m]}}]}]}


def test_parse_webhook_shapes():
    from src.ai_layer.whatsapp_mock import parse_webhook

    assert parse_webhook({"nope": 1}) == [] and parse_webhook("x") == []
    m = parse_webhook(_body("919800000001", "hello", "wamid.1"))
    assert m == [{"from": "919800000001", "id": "wamid.1", "type": "text", "text": "hello", "name": "Test"}]
    assert parse_webhook(_body("9198", None, "wamid.2", "image"))[0]["text"] is None


@pytest.mark.asyncio
async def test_mock_feeds_existing_intake(no_meta, pack, reset_ops_db):
    from src.ai_layer.whatsapp_mock import CHANNEL, handle_webhook, reset_sessions
    from src.data.warehouse import ops_con

    reset_sessions()
    r1 = await handle_webhook(_body("919800000001", "My 2019 Honda CR-V brakes are grinding", "wamid.a"), pack_id=pack.id)
    assert r1["mock"] is True and "no WhatsApp" in r1["label"]
    res = r1["results"][0]
    assert res["interaction_id"] and res["replies"] and all(x["status"] == "mock_not_sent" for x in res["replies"])
    dup = await handle_webhook(_body("919800000001", "My 2019 Honda CR-V brakes are grinding", "wamid.a"), pack_id=pack.id)
    assert dup["results"][0]["duplicate"] is True
    r2 = await handle_webhook(_body("919800000001", "there is smoke and a burning smell from the engine", "wamid.b"), pack_id=pack.id)
    assert r2["results"][0]["interaction_id"] == res["interaction_id"]
    with ops_con() as con:
        ch, ref = con.execute("SELECT channel, customer_ref FROM interactions WHERE interaction_id = ?", [res["interaction_id"]]).fetchone()
    assert ch == CHANNEL and ref and "919800000001" not in ref
    assert no_meta == []
    img = await handle_webhook(_body("919800000002", None, "wamid.c", "image"), pack_id=pack.id)
    assert "only accepts text" in img["results"][0]["replies"][0]["text"]["body"]


def test_mock_route_rejects_non_json(reset_ops_db, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app

    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    r = TestClient(app).post("/api/ai/whatsapp-mock/webhook", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 400
    r = TestClient(app).post("/api/ai/whatsapp-mock/webhook", json={"object": "x"})
    assert r.status_code == 200 and r.json()["received"] == 0
