"""Tests for F-005 (scoped key minting & write surface authorization) and F-006 (hardening routes authentication)."""

from __future__ import annotations

import os
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.rbac import issue_session

TEST_API_KEY = "test-secret-key-32-chars-minimum-auth!!"


@pytest.fixture
def auth_env(monkeypatch):
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", TEST_API_KEY)
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret-distinct-32b!!")
    monkeypatch.delenv("FRONTLINE_OPEN_MODE", raising=False)
    monkeypatch.delenv("FRONTLINE_SERVICE_IS_ADMIN", raising=False)


def test_f006_hardening_routes_require_authentication(auth_env, reset_ops_db):
    """F-006: Hardening endpoints must fail with 401 when unauthenticated."""
    with TestClient(app) as client:
        # Without auth header -> must be 401
        for path in [
            "/api/frontline/hardening/slos",
            "/api/frontline/hardening/retention",
        ]:
            resp = client.get(path)
            assert resp.status_code == 401, f"Expected 401 on {path}, got {resp.status_code}"

        resp = client.post("/api/frontline/hardening/fairness", json={"cases": []})
        assert resp.status_code == 401, f"Expected 401 on fairness, got {resp.status_code}"

        # With valid auth header -> 200
        headers = {"X-API-Key": TEST_API_KEY}
        resp = client.get("/api/frontline/hardening/slos", headers=headers)
        assert resp.status_code == 200


def test_f005_scoped_key_minting_forbidden_for_service_and_unauth(auth_env, reset_ops_db):
    """F-005: POST /keys must refuse minting for unauthenticated and non-admin principals."""
    with TestClient(app) as client:
        # 1. Unauthenticated -> 401
        resp = client.post("/api/frontline/keys", json={"scopes": ["kpi:read"], "tenant_id": "cust_123"})
        assert resp.status_code == 401

        # 2. Service principal (default role for FRONTLINE_API_KEY) -> 403 Forbidden
        service_headers = {"X-API-Key": TEST_API_KEY}
        resp = client.post(
            "/api/frontline/keys",
            headers=service_headers,
            json={"scopes": ["kpi:read", "billing:read"], "tenant_id": "cust_123"},
        )
        assert resp.status_code == 403, f"Expected 403 for service principal on /keys, got {resp.status_code}"

        # 3. Agent session -> 403 Forbidden
        agent_session = issue_session("agent_user", "agent", issuer_role="admin")
        agent_headers = {
            "X-API-Key": TEST_API_KEY,
            "X-Frontline-Session": agent_session["token"],
        }
        resp = client.post(
            "/api/frontline/keys",
            headers=agent_headers,
            json={"scopes": ["kpi:read"], "tenant_id": "cust_123"},
        )
        assert resp.status_code == 403, f"Expected 403 for agent on /keys, got {resp.status_code}"

        # 4. Admin session -> 200 OK
        admin_session = issue_session("admin_user", "admin", issuer_role="admin")
        admin_headers = {
            "X-API-Key": TEST_API_KEY,
            "X-Frontline-Session": admin_session["token"],
        }
        resp = client.post(
            "/api/frontline/keys",
            headers=admin_headers,
            json={"scopes": ["kpi:read"], "tenant_id": "default"},
        )
        assert resp.status_code == 200, f"Expected 200 for admin on /keys, got {resp.status_code}"
        body = resp.json()
        assert "token" in body
        assert body["token"].startswith("sk_live_")
        assert body["tenant_id"] == "default"


def test_f005_marketplace_install_requires_admin(auth_env, reset_ops_db):
    """F-005: Installing marketplace packs requires admin role (not bare service)."""
    with TestClient(app) as client:
        # Service principal lacks marketplace:install
        service_headers = {"X-API-Key": TEST_API_KEY}
        resp = client.post(
            "/api/frontline/marketplace/install/some_pack",
            headers=service_headers,
        )
        assert resp.status_code == 403


def test_f005_seats_assign_requires_supervisor_or_admin(auth_env, reset_ops_db):
    """F-005: Assigning seats requires supervisor/admin role."""
    with TestClient(app) as client:
        # Agent session -> 403
        agent_session = issue_session("agent_user", "agent", issuer_role="admin")
        agent_headers = {
            "X-API-Key": TEST_API_KEY,
            "X-Frontline-Session": agent_session["token"],
        }
        resp = client.post(
            "/api/frontline/seats",
            headers=agent_headers,
            json={"tenant_id": "t1", "user_id": "u1", "role": "agent"},
        )
        assert resp.status_code == 403
