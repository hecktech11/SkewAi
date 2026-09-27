"""Outbound connector: payload, outbox, HTTP dispatch, API, replay, auth."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.frontline import connectors as conn_mod
from src.frontline.connectors import (
    build_payload,
    dispatch_event,
    get_connector_config,
    list_deliveries,
    replay_delivery,
    reset_connector_config_cache,
    set_connector_config,
)


KEY = "connector-admin-secret"
HDR = {"X-API-Key": KEY}


@pytest.fixture
def conn_env(tmp_path, reset_ops_db, monkeypatch):
    """Isolate connector config + outbox under tmp_path."""
    cfg = tmp_path / "config.json"
    outbox = tmp_path / "outbox"
    outbox.mkdir()
    monkeypatch.setattr(conn_mod, "CONFIG_PATH", cfg)
    monkeypatch.setattr(conn_mod, "OUTBOX_DIR", outbox)
    # Patch REPO_ROOT relative path logic for outbox write
    monkeypatch.setattr(conn_mod, "REPO_ROOT", tmp_path)
    reset_connector_config_cache()
    # Disable via runtime file by default; tests enable explicitly.
    set_connector_config(enabled=False, webhook_url="", shared_secret="")
    yield {"config": cfg, "outbox": outbox, "tmp": tmp_path}
    reset_connector_config_cache()


@pytest.fixture
def locked(conn_env, monkeypatch):
    monkeypatch.setenv("FRONTLINE_API_KEY", KEY)
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")
    monkeypatch.setenv("FRONTLINE_SERVICE_IS_ADMIN", "1")
    with TestClient(app) as c:
        yield c
    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)


@pytest.fixture
def open_client(conn_env, monkeypatch):
    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    with TestClient(app) as c:
        yield c


def test_build_payload_shape():
    p = build_payload(
        "case_created",
        case_id="case_abc",
        interaction_id="int_1",
        pack_id="automotive_nhtsa",
        severity="Critical",
        priority=1,
    )
    assert p["event"] == "case_created"
    assert p["case_id"] == "case_abc"
    assert p["interaction_id"] == "int_1"
    assert p["schema_version"] == 1
    assert p["source"] == "frontline_v2"
    assert "ts" in p


@pytest.mark.asyncio
async def test_dispatch_disabled_skips(conn_env):
    set_connector_config(enabled=False)
    r = await dispatch_event("case_created", case_id="case_x")
    assert r.get("skipped") is True
    assert list(conn_env["outbox"].iterdir()) == []


@pytest.mark.asyncio
async def test_dispatch_outbox_only_success(conn_env):
    set_connector_config(enabled=True, webhook_url="")
    r = await dispatch_event(
        "case_created",
        case_id="case_out",
        interaction_id="int_out",
        pack_id="automotive_nhtsa",
    )
    assert r["ok"] is True
    assert r["status"] == "success"
    assert "outbox" in r["sink"]
    files = list(conn_env["outbox"].glob("*.json"))
    assert len(files) == 1
    body = json.loads(files[0].read_text())
    assert body["event"] == "case_created"
    assert body["case_id"] == "case_out"
    rows = list_deliveries()
    assert len(rows) >= 1
    assert rows[0]["status"] == "success"


@pytest.mark.asyncio
async def test_dispatch_http_success_recorded(conn_env):
    set_connector_config(
        enabled=True,
        webhook_url="https://example.com/conn",
        shared_secret="sec-1",
    )
    ok_post = AsyncMock(return_value=(True, 200, None))
    with patch.object(conn_mod, "_post_once", new=ok_post):
        r = await dispatch_event(
            "investigation_opened",
            investigation_id="inv_0001",
            case_id="case_y",
            cluster_id=3,
            title="Cluster 3",
        )
    assert r["ok"] is True
    assert r["status"] == "success"
    assert "http" in r["sink"]
    ok_post.assert_awaited()
    # Secret header path exercised inside _post_once when not fully mocked —
    # here we only assert dispatch recorded success.
    rows = list_deliveries(status="success")
    assert any(x["event"] == "investigation_opened" for x in rows)


@pytest.mark.asyncio
async def test_dispatch_http_failure_pending_no_raise(conn_env):
    set_connector_config(
        enabled=True,
        webhook_url="https://example.com/down",
    )
    fail = AsyncMock(return_value=(False, 500, "http_500: boom"))
    with patch.object(conn_mod, "_post_once", new=fail):
        r = await dispatch_event("case_created", case_id="case_fail")
    assert r["ok"] is False
    assert r["status"] == "pending"
    # Outbox still written on HTTP failure
    assert list(conn_env["outbox"].glob("*.json"))
    pending = list_deliveries(status="pending")
    assert any(x["case_id"] == "case_fail" for x in pending)


@pytest.mark.asyncio
async def test_replay_pending_http(conn_env):
    set_connector_config(
        enabled=True,
        webhook_url="https://example.com/conn",
    )
    fail = AsyncMock(return_value=(False, 503, "down"))
    with patch.object(conn_mod, "_post_once", new=fail):
        r = await dispatch_event("case_created", case_id="case_rp")
    dl_id = r["delivery_id"]
    ok = AsyncMock(return_value=(True, 200, None))
    with patch.object(conn_mod, "_post_once", new=ok):
        assert await replay_delivery(dl_id) is True
    rows = list_deliveries()
    match = next(x for x in rows if x["delivery_id"] == dl_id)
    assert match["status"] == "replayed"


def test_config_api_redacts_secret(locked):
    assert locked.get("/api/frontline/connectors/status").status_code == 401

    r = locked.put(
        "/api/frontline/connectors/config",
        headers={**HDR, "Content-Type": "application/json"},
        json={
            "enabled": True,
            "webhook_url": "https://example.com/secret-path?token=xyz",
            "shared_secret": "super-secret",
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body.get("shared_secret") in ("", None)
    assert body.get("shared_secret_set") is True
    red = body.get("webhook_url_redacted") or body.get("webhook_url") or ""
    assert "xyz" not in red
    assert "super-secret" not in json.dumps(body)

    st = locked.get("/api/frontline/connectors/status", headers=HDR)
    assert st.status_code == 200
    assert st.json()["enabled"] is True
    assert "Salesforce" not in st.json()["note"] or "Not" in st.json()["note"]


@pytest.mark.asyncio
async def test_list_deliveries_api_auth(locked):
    set_connector_config(enabled=True, webhook_url="")
    await dispatch_event("case_created", case_id="case_api_seed")

    assert locked.get("/api/frontline/connectors/deliveries").status_code == 401
    r = locked.get("/api/frontline/connectors/deliveries", headers=HDR)
    assert r.status_code == 200
    body = r.json()
    assert "deliveries" in body
    assert body["count"] >= 1


@pytest.mark.asyncio
async def test_api_export_and_replay_flow(locked, conn_env, reset_ops_db):
    """Full path: enable → seed case row → export → force fail → replay."""
    from src.data.warehouse import ops_con
    from datetime import datetime, timezone

    set_connector_config(enabled=True, webhook_url="")

    with ops_con() as con:
        con.execute(
            """
            INSERT INTO cases (
                case_id, interaction_id, pack_id, created_at,
                category, description_summary, onset, severity,
                severity_source, priority, safety_flags,
                advisory_match_id, cluster_match_id, similar_record_count,
                investigation_id, status, followup_draft
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "case_export_1",
                "int_export_1",
                "automotive_nhtsa",
                datetime.now(timezone.utc),
                "brakes",
                "soft pedal",
                datetime.now(timezone.utc),
                "Medium",
                "rules",
                2,
                "{}",
                None,
                None,
                0,
                None,
                "open",
                "draft",
            ],
        )

    r = locked.post(
        "/api/frontline/connectors/export/case_export_1",
        headers=HDR,
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] is True
    assert body["status"] == "success"
    assert list(Path(conn_env["outbox"]).glob("*.json"))

    # Force a pending delivery and replay
    set_connector_config(
        enabled=True,
        webhook_url="https://example.com/x",
    )
    fail = AsyncMock(return_value=(False, 500, "err"))
    with patch.object(conn_mod, "_post_once", new=fail):
        r2 = await dispatch_event("case_created", case_id="case_pending_api")
    dl_id = r2["delivery_id"]

    assert (
        locked.post(
            f"/api/frontline/connectors/deliveries/{dl_id}/replay"
        ).status_code
        == 401
    )
    ok = AsyncMock(return_value=(True, 200, None))
    with patch.object(conn_mod, "_post_once", new=ok):
        r3 = locked.post(
            f"/api/frontline/connectors/deliveries/{dl_id}/replay",
            headers=HDR,
        )
    assert r3.status_code == 200
    assert r3.json()["ok"] is True


def test_open_mode_status(open_client):
    r = open_client.get("/api/frontline/connectors/status")
    assert r.status_code == 200
    assert "enabled" in r.json()


def test_config_redaction_unit(conn_env):
    set_connector_config(
        enabled=True,
        webhook_url="https://example.com/path?tok=abc",
        shared_secret="shh",
    )
    cfg = get_connector_config(include_secret=False)
    assert cfg["shared_secret"] == ""
    assert cfg["shared_secret_set"] is True
    assert "abc" not in (cfg.get("webhook_url") or "")
    assert "abc" not in (cfg.get("webhook_url_redacted") or "")
    secret_cfg = get_connector_config(include_secret=True)
    assert secret_cfg["shared_secret"] == "shh"
    assert "example.com" in secret_cfg["webhook_url"]
