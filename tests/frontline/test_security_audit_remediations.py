"""Tests verifying remediations for the 2026-09-14 Security Audit findings."""

from __future__ import annotations

import os
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.rbac import issue_session, verify_session
from src.api.auth import auth_required, is_open_mode
from src.security.harden import validate_startup_security


def test_audit_finding_1_wildcard_bind_requires_auth(monkeypatch):
    """Wildcard binds (0.0.0.0, ::) require auth by default unless ACK is set."""
    monkeypatch.setenv("API_HOST", "0.0.0.0")
    monkeypatch.delenv("FRONTLINE_OPEN_BIND_ACK", raising=False)
    monkeypatch.delenv("FRONTLINE_AUTH_REQUIRED", raising=False)
    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    monkeypatch.delenv("PILOT_HARDENED", raising=False)
    monkeypatch.delenv("ENV", raising=False)

    assert auth_required() is True
    assert is_open_mode() is False


def test_audit_finding_2_oidc_handoff_uses_fragment(reset_ops_db, seed_automotive_pack, monkeypatch):
    """OIDC callback must place handoff in URL fragment, not query string."""
    import base64
    import json
    import time
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
    from src.frontline.oidc import begin_login, set_http_hooks

    monkeypatch.setenv("GOOGLE_CLIENT_ID", "client_id")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "secret")
    monkeypatch.setenv("OIDC_REDIRECT_URI", "http://127.0.0.1:8000/api/frontline/auth/oidc/callback")
    monkeypatch.setenv("FRONTLINE_OPEN_MODE", "1")
    monkeypatch.setenv("API_HOST", "127.0.0.1")

    def _b64url(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub = priv.public_key().public_numbers()
    jwks = {
        "keys": [
            {
                "kty": "RSA",
                "kid": "k1",
                "n": _b64url(pub.n.to_bytes((pub.n.bit_length() + 7) // 8, "big")),
                "e": _b64url(pub.e.to_bytes((pub.e.bit_length() + 7) // 8, "big")),
            }
        ]
    }

    begun = begin_login(next_url="http://127.0.0.1:8000/ui/#command")
    now = int(time.time())
    header = _b64url(json.dumps({"alg": "RS256", "kid": "k1", "typ": "JWT"}, separators=(",", ":")).encode())
    claims = _b64url(
        json.dumps(
            {
                "iss": "https://accounts.google.com",
                "sub": "g_123",
                "email": "test@gmail.com",
                "email_verified": True,
                "aud": "client_id",
                "nonce": begun["nonce"],
                "exp": now + 300,
                "iat": now,
            },
            separators=(",", ":"),
        ).encode()
    )
    sig = priv.sign(f"{header}.{claims}".encode("ascii"), padding.PKCS1v15(), hashes.SHA256())
    id_token = f"{header}.{claims}.{_b64url(sig)}"

    set_http_hooks(lambda *_a, **_k: {"id_token": id_token}, lambda *_a, **_k: jwks)

    try:
        with TestClient(app) as client:
            cb = client.get(
                "/api/frontline/auth/oidc/callback",
                params={"code": "4/ok", "state": begun["state"]},
                follow_redirects=False,
            )
            assert cb.status_code == 302
            loc = cb.headers["location"]
            # Handoff must be in fragment, NOT query parameters
            assert "#" in loc
            assert "handoff=" in loc
            url_part, frag_part = loc.split("#", 1)
            assert "handoff" not in url_part
            assert "handoff=" in frag_part
    finally:
        set_http_hooks(None, None)


def test_audit_finding_3_session_secret_cannot_alias_api_key_in_production(monkeypatch):
    """Production mode refuses to sign sessions with the API key or aliased secret."""
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("FRONTLINE_API_KEY", "prod-api-key-at-least-32-bytes-long!")
    monkeypatch.setenv("SESSION_SECRET", "prod-api-key-at-least-32-bytes-long!")  # Same!

    with pytest.raises(RuntimeError, match="SESSION_SECRET must not match FRONTLINE_API_KEY"):
        validate_startup_security()


def test_audit_finding_3_bootstrap_admin_disabled_in_production(monkeypatch):
    """FRONTLINE_BOOTSTRAP_ADMIN is disabled in production-like mode."""
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("FRONTLINE_API_KEY", "prod-api-key-at-least-32-bytes-long!")
    monkeypatch.setenv("SESSION_SECRET", "different-session-secret-32-bytes!!")
    monkeypatch.setenv("FRONTLINE_BOOTSTRAP_ADMIN", "1")

    with pytest.raises(RuntimeError, match="FRONTLINE_BOOTSTRAP_ADMIN must not be enabled"):
        validate_startup_security()


def test_audit_finding_4_public_signup_disabled_in_production(reset_ops_db, seed_automotive_pack, monkeypatch):
    """Public signup is disabled in production unless explicitly enabled."""
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("FRONTLINE_API_KEY", "prod-api-key-at-least-32-bytes-long!")
    monkeypatch.setenv("SESSION_SECRET", "different-session-secret-32-bytes!!")
    monkeypatch.delenv("FRONTLINE_ALLOW_PUBLIC_SIGNUP", raising=False)

    with TestClient(app) as client:
        r = client.post(
            "/api/frontline/auth/signup",
            json={
                "name": "Attacker",
                "email": "attacker@example.com",
                "password": "Password123456!",
            },
        )
        assert r.status_code == 403
        assert "Public registration is disabled" in r.text


def test_audit_finding_4_scrypt_password_hashing(reset_ops_db, seed_automotive_pack, monkeypatch):
    """Password hashing uses scrypt and verifies correctly."""
    monkeypatch.setenv("FRONTLINE_ALLOW_PUBLIC_SIGNUP", "1")

    with TestClient(app) as client:
        r = client.post(
            "/api/frontline/auth/signup",
            json={
                "name": "Scrypt User",
                "email": "scrypt_user@example.com",
                "password": "SecurePassword123456!",
            },
        )
        assert r.status_code == 200

        # Login verifies scrypt
        login = client.post(
            "/api/frontline/auth/login",
            json={
                "email": "scrypt_user@example.com",
                "password": "SecurePassword123456!",
            },
        )
        assert login.status_code == 200
        assert login.json()["subject"] == "scrypt_user@example.com"


def test_audit_finding_5_four_eyes_identity_bound_to_actor(reset_ops_db, seed_automotive_pack, monkeypatch):
    """Approvals endpoint binds requester to authenticated actor and requires authenticated reviewer."""
    api_key = "test-api-key-at-least-32-bytes-long!"
    sec_key = "test-secret-at-least-32-bytes-long!"
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", api_key)
    monkeypatch.setenv("SESSION_SECRET", sec_key)
    alice_tok = issue_session("alice@example.com", "supervisor", issuer_role="admin")["token"]
    bob_tok = issue_session("bob@example.com", "supervisor", issuer_role="admin")["token"]

    with TestClient(app) as client:
        headers = {"x-api-key": api_key}
        # Request approval with alice's session
        req = client.post(
            "/api/frontline/approvals",
            json={"action_type": "open_investigation", "resource_id": "inv_sec_1", "requested_by": "spoofed_bob"},
            cookies={"frontline_session": alice_tok},
            headers=headers,
        )
        assert req.status_code == 200
        app_id = req.json()["approval_id"]
        # Authenticated requester overrides body spoof
        assert req.json()["requested_by"] == "alice@example.com"

        # Alice cannot approve her own request
        decide = client.post(
            f"/api/frontline/approvals/{app_id}/decide",
            json={"approve": True},
            cookies={"frontline_session": alice_tok},
            headers=headers,
        )
        assert decide.status_code == 400
        assert "reviewer must differ from requester" in decide.text.lower()

        # Bob (independent supervisor) can approve
        bob_decide = client.post(
            f"/api/frontline/approvals/{app_id}/decide",
            json={"approve": True},
            cookies={"frontline_session": bob_tok},
            headers=headers,
        )
        assert bob_decide.status_code == 200
        assert bob_decide.json()["status"] == "approved"
        assert bob_decide.json()["reviewer"] == "bob@example.com"


def test_audit_finding_6_stripe_webhook_unsigned_fails_in_production(monkeypatch):
    """Stripe webhook fails closed if secret is unset in production-like mode."""
    monkeypatch.setenv("ENV", "production")
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)

    from src.frontline.billing import stripe_webhook

    with pytest.raises(PermissionError, match="STRIPE_WEBHOOK_SECRET must be configured"):
        stripe_webhook({"id": "evt_test", "type": "invoice.paid"})


def test_audit_finding_7_client_tenant_id_ignored_on_billing(reset_ops_db, seed_automotive_pack, monkeypatch):
    """Billing endpoints ignore client-supplied tenant_id and use process tenant."""
    monkeypatch.setenv("FRONTLINE_TENANT_ID", "corp_tenant")

    with TestClient(app) as client:
        r = client.get("/api/frontline/billing/usage?tenant_id=attacker_tenant")
        assert r.status_code == 200
        assert r.json()["tenant_id"] == "corp_tenant"


def test_audit_finding_10_csp_connect_src_restricts_websockets(reset_ops_db, seed_automotive_pack):
    """CSP connect-src restricts websockets to local endpoints and eliminates bare wss:."""
    with TestClient(app) as client:
        r = client.get("/health")
        csp = r.headers.get("content-security-policy", "")
        connect_src = csp.split("connect-src")[1].split(";")[0]
        # Bare wss: must not be present
        assert "wss:" not in connect_src
        assert "ws://127.0.0.1:*" in connect_src or "ws://localhost:*" in connect_src


def test_audit_finding_11_health_hides_internal_details_when_unauthenticated_in_prod(reset_ops_db, seed_automotive_pack, monkeypatch):
    """Public health endpoint hides internal details when unauthenticated in production."""
    monkeypatch.setenv("ENV", "production")
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", "prod-key-at-least-32-bytes-long!")
    monkeypatch.setenv("SESSION_SECRET", "prod-session-at-least-32-bytes-long!")

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert "status" in body
        assert "active_pack" in body
        # Sensitive internal details must not be disclosed
        assert "worker_registry" not in body
        assert "orchestrator_registry" not in body
        assert "worker_count" not in body
        assert "single_worker" not in body
        assert "readiness" not in body


def test_audit_finding_12_oidc_status_masks_client_id_when_unauthenticated(reset_ops_db, seed_automotive_pack):
    """OIDC status masks client_id for unauthenticated callers."""
    with TestClient(app) as client:
        r = client.get("/api/frontline/auth/oidc/status")
        assert r.status_code == 200
        body = r.json()
        assert body["client_id"] in ("", "***configured***")


def test_validation_1_llm_outbound_ssrf_blocked(monkeypatch):
    """Outbound LLM calls block private/metadata addresses."""
    monkeypatch.setenv("FRONTLINE_LLM_ENABLED", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://169.254.169.254/v1")

    from src.ai.provider import _http_chat_openai

    with pytest.raises(ValueError, match="blocked"):
        _http_chat_openai("system", "user", "model")


def test_session_revocation_denylist_on_logout(reset_ops_db, seed_automotive_pack, monkeypatch):
    """Logout adds the active session token to the revocation denylist."""
    monkeypatch.setenv("SESSION_SECRET", "test-secret-at-least-32-bytes-long!")
    tok_info = issue_session("user@example.com", "agent")
    tok = tok_info["token"]

    # Before logout, token is valid
    assert verify_session(tok)["sub"] == "user@example.com"

    with TestClient(app) as client:
        # Call logout with the session cookie
        logout_resp = client.post(
            "/api/frontline/auth/logout",
            cookies={"frontline_session": tok},
        )
        assert logout_resp.status_code == 200
        assert logout_resp.json()["signed_in"] is False

    # After logout, token is revoked and fails verify_session
    with pytest.raises(HTTPException) as exc_info:
        verify_session(tok)
    assert exc_info.value.status_code == 401
    assert "revoked" in exc_info.value.detail.lower()


def test_four_eyes_rejects_anonymous_requester_and_reviewer(reset_ops_db, seed_automotive_pack):
    """Approvals endpoint strictly rejects unauthenticated / anonymous users."""
    with TestClient(app) as client:
        # Request approval without session or credentials
        req = client.post(
            "/api/frontline/approvals",
            json={"action_type": "open_investigation", "resource_id": "inv_anon", "requested_by": "spoofed_actor"},
        )
        assert req.status_code == 401
        assert "Authentication required" in req.text


def test_stripe_webhook_unsigned_fails_in_open_mode(reset_ops_db, seed_automotive_pack, monkeypatch):
    """Stripe webhook HTTP endpoint fails closed against unsigned payloads even in open mode."""
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    monkeypatch.setenv("FRONTLINE_OPEN_MODE", "1")

    with TestClient(app) as client:
        r = client.post(
            "/api/frontline/billing/webhook",
            json={"type": "checkout.session.completed", "id": "evt_fake"},
        )
        assert r.status_code == 401
        assert "STRIPE_WEBHOOK_SECRET must be configured" in r.text


def test_session_signing_refuses_api_key_everywhere(monkeypatch):
    """Session signing refuses to alias to FRONTLINE_API_KEY in non-production as well."""
    monkeypatch.setenv("FRONTLINE_API_KEY", "test-api-key-32-bytes-long-here!!")
    monkeypatch.setenv("SESSION_SECRET", "test-api-key-32-bytes-long-here!!")

    from src.security.rotation import sign_session

    with pytest.raises(RuntimeError, match="SESSION_SECRET must not match FRONTLINE_API_KEY"):
        sign_session("test_message")


def test_llm_key_not_sent_to_unapproved_base_url_host(monkeypatch):
    """A rogue/ambient *_BASE_URL must not redirect the provider key.

    The SSRF guard clears any public host, so without a credential allowlist an
    ambient ANTHROPIC_BASE_URL (common on developer boxes) silently ships
    CLAUDE_API_KEY to a third party.
    """
    import urllib.request

    from src.ai import provider as prov

    monkeypatch.setenv("FRONTLINE_LLM_ENABLED", "1")
    monkeypatch.delenv("FRONTLINE_LLM_ALLOWED_HOSTS", raising=False)
    monkeypatch.setattr(prov, "_openai_key", lambda: "")
    monkeypatch.setattr(prov, "_claude_key", lambda: "sk-ant-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://evil-proxy.example.com")

    def _boom(req, timeout=8.0):  # pragma: no cover - must never run
        raise AssertionError(f"key was sent to {req.full_url}")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)

    with pytest.raises(ValueError, match="blocked"):
        prov._http_chat("sys", "user", "")

    # Same guard on the OpenAI path.
    monkeypatch.setattr(prov, "_openai_key", lambda: "sk-openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://evil-proxy.example.com/v1")
    with pytest.raises(ValueError, match="blocked"):
        prov._http_chat("sys", "user", "")


def test_llm_allowed_hosts_opt_in_permits_gateway(monkeypatch):
    """An operator can still route through a gateway by naming it explicitly."""
    import json
    import io
    import urllib.request

    from src.ai import provider as prov

    monkeypatch.setenv("FRONTLINE_LLM_ENABLED", "1")
    monkeypatch.setattr(prov, "_openai_key", lambda: "")
    monkeypatch.setattr(prov, "_claude_key", lambda: "sk-ant-secret")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.example.com")
    monkeypatch.setenv("FRONTLINE_LLM_ALLOWED_HOSTS", "gateway.example.com")

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *a):
            return json.dumps({"content": [{"type": "text", "text": "ok"}]}).encode()

    seen: dict = {}

    def _urlopen(req, timeout=8.0):
        seen["url"] = req.full_url
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    assert prov._http_chat("sys", "user", "") == "ok"
    assert seen["url"] == "https://gateway.example.com/v1/messages"
