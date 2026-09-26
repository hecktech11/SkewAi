"""Regression tests for 2026-09 security assessment fixes.

Covers:
- V-001: hardcoded backdoor credentials removed (401, no admin mint)
- V-002: signup self-elevation blocked (extra['role'] cannot overwrite claim;
  signup always issues agent; display title stored separately)
- V-003: audit report path traversal rejected (400) + jailed reads
- V-004: CSV formula injection neutralized in audit export
"""

from __future__ import annotations

import uuid

from fastapi.testclient import TestClient

from src.api.export import export_to_csv
from src.api.main import app
from src.api.rbac import issue_session, require_perm, verify_session


def _signup(client: TestClient, role: str) -> dict:
    email = f"reg_{uuid.uuid4().hex[:8]}@example.com"
    r = client.post(
        "/api/frontline/auth/signup",
        json={
            "name": "Reg Test",
            "email": email,
            "company": "Acme",
            "role": role,
            "password": "LongPassword123!",
        },
    )
    assert r.status_code == 200, r.text
    return r


def test_backdoor_credentials_removed(reset_ops_db):
    with TestClient(app) as client:
        for email, pwd in [
            ("pilot.sre@skew.ai", "Pilot2026!Secure"),
            ("alex.mercer@skew.ai", "password123"),
            ("alex.mercer@skew.ai", "secret"),
        ]:
            r = client.post(
                "/api/frontline/auth/login",
                json={"email": email, "password": pwd},
            )
            assert r.status_code == 401, f"backdoor {email} must fail, got {r.status_code}"


def test_signup_cannot_mint_admin(reset_ops_db):
    with TestClient(app) as client:
        r = _signup(client, "admin")
        body = r.json()
        # Token must not leak in JSON body (HttpOnly cookie only)
        assert "token" not in body
        # Session cookie must carry role=agent, never admin
        sess = None
        for k, v in r.cookies.items():
            if "session" in k:
                sess = v
        assert sess, "session cookie must be set"
        raw = sess.strip('"')
        # Unescape quoted-string cookie encoding for assertion only
        assert '"role\\":\\"agent\\"' in raw or '"role":"agent"' in raw.replace("\\", "")


def test_issue_session_extra_cannot_overwrite_role():
    from src.api.rbac import issue_session

    sess = issue_session("u", "agent", extra={"role": "admin", "sub": "evil"})
    assert sess["role"] == "agent"
    assert verify_session(sess["token"])["role"] == "agent"
    try:
        require_perm(verify_session(sess["token"])["role"], "dsr:delete")
    except Exception as exc:
        assert getattr(exc, "status_code", 403) == 403
    else:  # pragma: no cover
        raise AssertionError("spoofed admin must not pass dsr:delete")


def test_audit_traversal_rejected(reset_ops_db):
    with TestClient(app) as client:
        r = client.get("/api/frontline/audits/%2e%2e")
        assert r.status_code == 400
        r2 = client.get("/api/frontline/audits/int_valid123")
        # Valid-format unknown id → 404 (not 400, not 500, no file leak)
        assert r2.status_code == 404


def test_export_csv_neutralizes_formulas():
    payload = {
        "interactions": [
            {
                "interaction_id": "int_1",
                "pack_id": "p",
                "channel": "web",
                "status": "done",
                "outcome": "x",
                "started_at": "",
                "ended_at": "",
                "entity_1": "=cmd|'/c calc'!A0",
                "entity_2": "+2+2",
                "entity_3": "@SUM(1+1)",
                "supervised": "",
                "peak_frustration": "",
                "actions": [],
                "audit": {},
            }
        ]
    }
    out = export_to_csv(payload)
    assert "'=cmd" in out
    assert "'+2+2" in out
    assert "'@SUM" in out


def test_auth_verify_open_mode_ok_without_key(reset_ops_db):
    """Route-guard probe passes in open mode (callers must check auth_required)."""
    with TestClient(app) as client:
        r = client.get("/api/frontline/auth/verify")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["auth_required"] is False


def test_auth_verify_hardened_enforces_key(reset_ops_db, monkeypatch):
    """Route-guard probe: 401 without key, ok + service role with key."""
    key = "test-secret-key-32-chars-minimum-auth!!"
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", key)
    monkeypatch.setenv("SESSION_SECRET", key)
    monkeypatch.delenv("FRONTLINE_OPEN_MODE", raising=False)
    with TestClient(app) as client:
        assert client.get("/api/frontline/auth/verify").status_code == 401
        r = client.get("/api/frontline/auth/verify", headers={"X-API-Key": key})
        assert r.status_code == 200
        body = r.json()
        assert body == {"ok": True, "auth_required": True, "role": "service"}


def _hard_key(monkeypatch):
    key = "test-secret-key-32-chars-minimum-auth!!"
    session = "test-session-secret-32-chars-diff!!"
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", key)
    monkeypatch.setenv("SESSION_SECRET", session)
    monkeypatch.delenv("FRONTLINE_OPEN_MODE", raising=False)
    return key


def test_issue_reads_require_key_when_hardened(reset_ops_db, monkeypatch):
    """Issue-page reads (records/lots/scorecards/workspace/cluster) fail closed."""
    key = _hard_key(monkeypatch)
    with TestClient(app) as client:
        for path in [
            "/api/frontline/records?limit=1",
            "/api/frontline/suppliers/lots",
            "/api/frontline/suppliers/scorecards",
            "/api/frontline/suppliers/capas",
            "/api/frontline/investigations/inv_x/workspace",
            "/api/frontline/clusters/14/context",
        ]:
            assert client.get(path).status_code == 401, path
        headers = {"X-API-Key": key}
        # Unknown ids → 404 (never 500, never a DB created from input).
        r = client.get("/api/frontline/clusters/424242/context", headers=headers)
        assert r.status_code == 404
        r = client.get("/api/frontline/investigations/inv_x/workspace", headers=headers)
        assert r.status_code in (400, 404)
        r = client.get("/api/frontline/records?pack_id=..%2F..%2Fetc&limit=1", headers=headers)
        assert r.status_code in (400, 404)


def test_issue_path_traversal_rejected(reset_ops_db):
    """Identifier guard + handler validation block traversal on new routes."""
    with TestClient(app) as client:
        r = client.get("/api/frontline/clusters/%2e%2e/context")
        assert r.status_code in (400, 404)
        r = client.get("/api/frontline/records?pack_id=%2e%2e&limit=1")
        assert r.status_code in (400, 404)


def test_supplier_capa_open_and_list(reset_ops_db, monkeypatch):
    """CAPA write path: 400 on missing fields, round-trip otherwise."""
    from src.api.rbac import issue_session

    key = _hard_key(monkeypatch)
    monkeypatch.setenv("FRONTLINE_BOOTSTRAP_ADMIN", "1")
    admin = issue_session("tester", "admin", issuer_role="admin")["token"]
    headers = {
        "X-API-Key": key,
        "X-Frontline-Session": admin,
        "Content-Type": "application/json",
    }
    with TestClient(app) as client:
        # Shared service key alone lacks case:write → 403 (RBAC working).
        r = client.post(
            "/api/frontline/suppliers/capas",
            headers={"X-API-Key": key, "Content-Type": "application/json"},
            json={"supplier": "TestCo", "request": "x"},
        )
        assert r.status_code == 403
        r = client.post("/api/frontline/suppliers/capas", headers=headers, json={})
        assert r.status_code == 400
        r = client.post(
            "/api/frontline/suppliers/capas",
            headers=headers,
            json={"supplier": "TestCo", "lot_id": "L-0", "request": "contain lot"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "open" and body["capa_id"].startswith("capa_")
        r = client.get("/api/frontline/suppliers/capas?supplier=TestCo", headers=headers)
        assert r.status_code == 200
        assert any(c["capa_id"] == body["capa_id"] for c in r.json()["capas"])


def test_cookie_session_authenticates_protected_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    """R10: Cookie-only login can authenticate protected HTTP flows without API key header."""
    _hard_key(monkeypatch)
    session = issue_session("tester", "agent")["token"]
    with TestClient(app) as client:
        # Without cookie or API key -> 401
        r = client.get("/api/frontline/auth/me")
        assert r.status_code == 401

        # With session cookie only -> 200
        client.cookies.set("frontline_session", session)
        r = client.get("/api/frontline/auth/me")
        assert r.status_code == 200
        data = r.json()
        assert data.get("signed_in") is True
        assert data.get("subject") == "tester"
