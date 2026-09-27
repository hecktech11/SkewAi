"""L3: dead-letter list + replay HTTP under API key."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.frontline import alerts as alerts_mod
from src.frontline.alerts import fire_alert, list_dead_letters


KEY = "dl-admin-secret"
HDR = {"X-API-Key": KEY}
_HOOK = "https://hooks.example.test/slack"


@pytest.fixture
def locked(reset_ops_db, seed_automotive_pack, monkeypatch):
    monkeypatch.setenv("FRONTLINE_API_KEY", KEY)
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")
    monkeypatch.setenv("FRONTLINE_SERVICE_IS_ADMIN", "1")
    with TestClient(app) as c:
        yield c
    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)


@pytest.fixture
def open_client(reset_ops_db, seed_automotive_pack, monkeypatch):
    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    with TestClient(app) as c:
        yield c


@pytest.mark.asyncio
async def test_seed_dead_letter_then_list_and_replay(locked):
    fail = AsyncMock(return_value=(False, "down"))
    ok = AsyncMock(return_value=(True, None))
    with patch.object(alerts_mod, "settings") as mock_settings:
        mock_settings.alert_webhook_url = _HOOK
        with patch.object(alerts_mod, "_post_webhook_once", new=fail):
            await fire_alert(
                event="safety_escalation",
                summary="seed dl",
                ref_id="int_dl_admin",
                interaction_id="int_dl_admin",
            )

    assert locked.get("/api/frontline/alerts/dead-letter").status_code == 401

    r = locked.get("/api/frontline/alerts/dead-letter", headers=HDR)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    dl_id = body["dead_letters"][0]["dead_letter_id"]

    # Replay without key
    assert locked.post(f"/api/frontline/alerts/dead-letter/{dl_id}/replay").status_code == 401

    with patch.object(alerts_mod, "settings") as mock_settings:
        mock_settings.alert_webhook_url = _HOOK
        with patch.object(alerts_mod, "_post_webhook_once", new=ok):
            r2 = locked.post(
                f"/api/frontline/alerts/dead-letter/{dl_id}/replay",
                headers=HDR,
            )
    assert r2.status_code == 200
    assert r2.json()["ok"] is True


def test_dead_letter_list_open_mode(open_client):
    r = open_client.get("/api/frontline/alerts/dead-letter")
    assert r.status_code == 200
    assert "dead_letters" in r.json()
