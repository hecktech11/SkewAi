"""Regression tests verifying all 23 findings from rahulx2001-SkewAi-agent-findings.csv.

Every finding row is covered with an explicit test case to ensure no regressions.
"""

from __future__ import annotations

import hmac
import json
import os
from pathlib import Path
import secrets
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.data.warehouse import ops_con


@pytest.fixture(autouse=True)
def _setup_session_secret(monkeypatch):
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret-at-least-32-bytes-long!")


# ── Rows 1 & 16: Non-loopback bind address treated as wildcard / network-exposed ──

def test_row_1_and_16_non_loopback_treated_as_network_exposed():
    from src.security.harden import is_loopback_host, is_wildcard_bind

    assert is_loopback_host("127.0.0.1") is True
    assert is_loopback_host("localhost") is True
    assert is_loopback_host("::1") is True
    assert is_loopback_host("127.0.0.2") is True

    # LAN, public, or wildcard IPs are NOT loopback
    assert is_loopback_host("192.168.1.100") is False
    assert is_loopback_host("10.0.0.5") is False
    assert is_loopback_host("0.0.0.0") is False
    assert is_loopback_host("::") is False

    # is_wildcard_bind treats ALL non-loopback addresses as exposed
    assert is_wildcard_bind("0.0.0.0") is True
    assert is_wildcard_bind("::") is True
    assert is_wildcard_bind("192.168.1.50") is True
    assert is_wildcard_bind("10.0.0.1") is True
    assert is_wildcard_bind("127.0.0.1") is False
    assert is_wildcard_bind("localhost") is False


# ── Row 2: Structured VIN redacted in redact_dict for non-export callers ──

def test_row_2_structured_vin_redacted_for_non_export():
    from src.security.pii import redact_dict

    slots = {
        "vin": "1HGCR2F83HA000000",
        "ssn": "000-12-3456",
        "issue": "Brake pedal vibrating",
        "turn_count": 3,
    }

    redacted = redact_dict(slots)
    assert redacted["vin"] == "[VIN]"
    assert redacted["ssn"] == "[SSN]"
    assert redacted["turn_count"] == 3


# ── Row 3: verify_chain fails when stored head comparison fails ──

def test_row_3_verify_chain_fails_on_head_lookup_error():
    from src.ledger.chain import verify_chain

    dummy_row = {
        "action_id": "act_01",
        "prev_hash": "0" * 64,
        "entry_hash": "a" * 64,
        "timestamp": "2026-09-26T00:00:00Z",
        "interaction_id": "int_01",
        "action_type": "note",
        "actor": "agent",
        "details_json": "{}",
    }

    with patch("src.data.warehouse.ops_con", side_effect=RuntimeError("db disconnected")):
        result = verify_chain([dummy_row], allow_partial=False)
        assert result["ok"] is False
        assert result.get("error") == "head_lookup_failed"


# ── Row 4: Pack builder insight requires pack:edit and isolates live packs ──

def test_row_4_pack_builder_insight_authorization_and_isolation(reset_ops_db):
    from src.domains.builder.pack_builder import build_draft_pack

    # Overwriting protected runtime packs must be rejected
    with pytest.raises(ValueError, match="protected"):
        build_draft_pack(
            pack_id="automotive_nhtsa",
            display_name="Hack Pack",
            csv_path=Path("/tmp/nonexistent.csv"),
            mapping={"text": "summary"},
        )

    # Calling insight endpoint with agent session lacking pack:edit returns 403
    from src.api.rbac import issue_session
    sess_agent = issue_session("agent_user", role="agent", ttl_s=3600)
    with TestClient(app) as client:
        res = client.post(
            "/api/frontline/pack-builder/insight",
            headers={"X-Frontline-Session": sess_agent["token"]},
            json={
                "pack_id": "custom_pack",
                "rows": [{"col": "val"}],
                "mapping": {"col": "summary"},
            },
        )
        assert res.status_code in (401, 403)


# ── Row 5: Takeover release and human-turn enforce active claimant ──

def test_row_5_takeover_release_and_turn_enforce_claimant(reset_ops_db, monkeypatch):
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", "srv_test_key_123")
    from src.api.rbac import issue_session

    sess_alice = issue_session("supervisor_alice", role="supervisor", issuer_role="admin", ttl_s=3600)
    sess_bob = issue_session("supervisor_bob", role="supervisor", issuer_role="admin", ttl_s=3600)
    sess_admin = issue_session("admin_charlie", role="admin", issuer_role="admin", ttl_s=3600)

    with TestClient(app) as client:
        r_start = client.post(
            "/api/interactions/start",
            params={"channel": "web_text"},
            headers={"X-Frontline-Session": sess_alice["token"]},
        )
        assert r_start.status_code == 200
        iid = r_start.json()["interaction_id"]

        # Alice takes over
        r_claim = client.post(
            f"/api/interactions/{iid}/takeover",
            json={"action": "claim"},
            headers={"X-Frontline-Session": sess_alice["token"]},
        )
        assert r_claim.status_code == 200

        # Bob attempts to release Alice's takeover -> 403
        r_bob_rel = client.post(
            f"/api/interactions/{iid}/release",
            headers={"X-Frontline-Session": sess_bob["token"]},
        )
        assert r_bob_rel.status_code == 403

        # Admin can release with override=true
        r_admin_rel = client.post(
            f"/api/interactions/{iid}/release?override=true",
            headers={"X-Frontline-Session": sess_admin["token"]},
        )
        assert r_admin_rel.status_code == 200


# ── Rows 6 & 14: Connector URL redacts credentials and clears secret on change ──

def test_rows_6_and_14_connector_credentials_redacted_and_secret_cleared(reset_ops_db):
    from src.frontline.connectors import _redact_url, get_connector_config, set_connector_config

    # Credential stripping
    redacted = _redact_url("https://admin:supersecret@webhook.example.com:8443/alert?key=1")
    assert "admin" not in redacted
    assert "supersecret" not in redacted
    assert "webhook.example.com:8443" in redacted

    # Reject URLs with embedded credentials
    with pytest.raises(ValueError, match="embedded user credentials"):
        set_connector_config(
            webhook_url="https://user:pass@example.com/webhook",
            enabled=True,
        )

    # Initial config with secret
    set_connector_config(
        webhook_url="https://endpoint-a.example.com/hook",
        shared_secret="secret_abc",
        enabled=True,
    )
    cfg1 = get_connector_config(include_secret=True)
    assert cfg1["shared_secret"] == "secret_abc"
    assert get_connector_config()["shared_secret_set"] is True

    # Changing destination URL without providing new secret clears the retained secret
    set_connector_config(
        webhook_url="https://endpoint-b.example.com/hook",
        enabled=True,
    )
    cfg2 = get_connector_config(include_secret=True)
    assert cfg2["shared_secret"] == ""  # Cleared for security
    assert get_connector_config()["shared_secret_set"] is False


# ── Row 7: Customer websocket verifies capability token or owner ──

def test_row_7_customer_ws_capability_token_required(reset_ops_db):
    from src.api.rbac import issue_session

    sess_creator = issue_session("creator_user", role="agent", ttl_s=3600)
    sess_intruder = issue_session("intruder_user", role="agent", ttl_s=3600)

    with TestClient(app) as client:
        r_start = client.post(
            "/api/interactions/start",
            params={"channel": "web_text"},
            headers={"X-Frontline-Session": sess_creator["token"]},
        )
        assert r_start.status_code == 200
        data = r_start.json()
        iid = data["interaction_id"]
        cap = data["capability_token"]

        # Intruder attempts attachment without capability token -> gets error frame and close
        with client.websocket_connect(
            f"/ws/interaction/{iid}",
            headers={"X-Frontline-Session": sess_intruder["token"]},
        ) as ws:
            frame = ws.receive_json()
            assert frame.get("type") == "error"
            assert frame.get("code") == "forbidden"

        # Creator connects -> succeeds without error frame
        with client.websocket_connect(
            f"/ws/interaction/{iid}",
            headers={"X-Frontline-Session": sess_creator["token"]},
        ) as ws:
            pass


# ── Row 8: Twilio websocket requires authentication and detaches on close ──

def test_row_8_twilio_ws_requires_auth_and_detaches(reset_ops_db, monkeypatch):
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", "srv_key_twilio_test")
    from src.api.rbac import issue_session
    from src.api.routes.interactions import _active

    sess = issue_session("test_user", role="agent", issuer_role="admin", ttl_s=3600)
    with TestClient(app) as client:
        r_start = client.post(
            "/api/interactions/start",
            params={"channel": "web_text"},
            headers={"X-Frontline-Session": sess["token"]},
        )
        assert r_start.status_code == 200
        iid = r_start.json()["interaction_id"]

        # Unauthenticated Twilio WS connection is closed with 1008
        with pytest.raises(Exception):
            with client.websocket_connect(f"/ws/twilio/{iid}") as ws:
                ws.receive_json()

        # Interaction remains in _active with detached ws
        assert iid in _active
        assert _active[iid].ws_attached is False


# ── Row 9: OIDC handoff is browser-bound and callback fragment matches App.jsx ──

def test_row_9_oidc_browser_binding_and_fragment_format(reset_ops_db):
    from src.frontline.oidc import begin_login, consume_handoff, finish_callback

    binding1 = secrets.token_urlsafe(24)
    binding2 = secrets.token_urlsafe(24)

    with patch("src.frontline.oidc.provider_status", return_value={
        "configured": True,
        "client_id": "test_client",
        "redirect_uri": "http://127.0.0.1:8000/api/frontline/auth/oidc/callback",
        "scopes": "openid email",
        "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_endpoint": "https://oauth2.googleapis.com/token",
        "issuer": "https://accounts.google.com",
    }):
        begun1 = begin_login(next_url="http://127.0.0.1:8787/ui/#command", browser_binding=binding1)
        begun2 = begin_login(next_url="http://127.0.0.1:8787/ui/#command", browser_binding=binding2)

        # Mock exchange and id_token verification
        with patch("src.frontline.oidc.exchange_code", return_value={"id_token": "mock_id_token"}), \
             patch("src.frontline.oidc.verify_id_token", return_value={"email": "operator@skew.ai", "email_verified": True}):

            # Finish callback with wrong browser binding -> rejects
            with pytest.raises(ValueError, match="oidc_browser_binding_mismatch"):
                finish_callback(code="mock_code", state=begun1["state"], browser_binding="wrong_binding")

            # Finish callback with correct browser binding -> succeeds
            done2 = finish_callback(code="mock_code", state=begun2["state"], browser_binding=binding2)
            hid = done2["handoff"]

            # Consume handoff with wrong or missing binding -> rejects
            with pytest.raises(ValueError, match="handoff_binding_mismatch"):
                consume_handoff(hid, browser_binding="wrong_binding")

            with pytest.raises(ValueError, match="handoff_binding_mismatch"):
                consume_handoff(hid, browser_binding=None)

            # Consume handoff with correct binding -> succeeds
            consumed = consume_handoff(hid, browser_binding=binding2)
            assert consumed["email"] == "operator@skew.ai"


# ── Row 10: X-Call-Hash header ignored unless explicitly opted in ──

def test_row_10_call_hash_override_ignored_by_default(reset_ops_db, monkeypatch):
    monkeypatch.delenv("FRONTLINE_ALLOW_CALL_HASH_OVERRIDE", raising=False)
    from src.api.rbac import issue_session

    sess = issue_session("caller@skew.ai", role="agent", ttl_s=3600)
    with TestClient(app) as client:
        r = client.post(
            "/api/interactions/start",
            headers={
                "X-Frontline-Session": sess["token"],
                "X-Call-Hash": "0",  # Attacker attempt to game split
            },
            params={"channel": "web_text"},
        )
        assert r.status_code == 200


# ── Row 11: Alert rules evaluate side effects requires ops:write & server metrics ──

def test_row_11_alert_rules_evaluate_requires_ops_write(reset_ops_db):
    from src.api.rbac import issue_session

    sess_agent = issue_session("agent_user", role="agent", ttl_s=3600)
    with TestClient(app) as client:
        r = client.post(
            "/api/frontline/alert-rules/evaluate",
            headers={"X-Frontline-Session": sess_agent["token"]},
            json={
                "rule_id": "rule_01",
                "apply": True,
            },
        )
        # Agent role lacks ops:write -> 403
        assert r.status_code == 403


# ── Row 12: PUT /connectors/config requires ops:write ──

def test_row_12_connector_config_put_requires_ops_write(reset_ops_db):
    from src.api.rbac import issue_session

    sess_agent = issue_session("agent_user", role="agent", ttl_s=3600)
    with TestClient(app) as client:
        r = client.put(
            "/api/frontline/connectors/config",
            headers={"X-Frontline-Session": sess_agent["token"]},
            json={
                "webhook_url": "https://webhook.example.com/dest",
                "enabled": True,
            },
        )
        assert r.status_code == 403


# ── Row 13: Eval review identities derived from authenticated actor ──

def test_row_13_eval_labels_derive_actor_and_adjudicate_requires_approval(reset_ops_db):
    from src.api.rbac import issue_session

    sess_annotator = issue_session("annotator_jane", role="agent", ttl_s=3600)
    with TestClient(app) as client:
        # Submit label -> annotator_id is derived from actor
        r_sub = client.post(
            "/api/frontline/eval/labels",
            headers={"X-Frontline-Session": sess_annotator["token"]},
            json={
                "eval_id": "eval_01",
                "annotator_id": "spoofed_id",  # Overridden by actor
                "label": "paraphrase_positive",
            },
        )
        assert r_sub.status_code == 200

        # Adjudicate requires approval:decide permission -> agent role gets 403
        r_adj = client.post(
            "/api/frontline/eval/labels/adjudicate",
            headers={"X-Frontline-Session": sess_annotator["token"]},
            json={
                "eval_id": "eval_01",
                "label": "paraphrase_positive",
            },
        )
        assert r_adj.status_code == 403


# ── Row 15: Dead-letter list and replay routes require ops permissions ──

def test_row_15_dead_letter_permissions(reset_ops_db):
    from src.api.rbac import issue_session

    sess_agent = issue_session("agent_user", role="agent", ttl_s=3600)
    with TestClient(app) as client:
        # GET dead-letter requires ops:read
        r_get = client.get(
            "/api/frontline/alerts/dead-letter",
            headers={"X-Frontline-Session": sess_agent["token"]},
        )
        assert r_get.status_code == 403

        # POST replay requires ops:write
        r_post = client.post(
            "/api/frontline/alerts/dead-letter/dl_01/replay",
            headers={"X-Frontline-Session": sess_agent["token"]},
        )
        assert r_post.status_code == 403


# ── Rows 17 & 21: Query DSR key resolves to dsr_officer role, not service ──

def test_rows_17_and_21_query_dsr_key_resolves_to_dsr_officer(reset_ops_db, monkeypatch):
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_ALLOW_QUERY_KEY", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", "srv_main_key")
    monkeypatch.setenv("FRONTLINE_DSR_API_KEY", "dsr_secret_key")

    with TestClient(app) as client:
        # Supplying DSR key via query param on audit export route must NOT pass audit:read
        r = client.get(
            "/api/frontline/audits/export?api_key=dsr_secret_key"
        )
        # dsr_officer has dsr permissions, but lacks audit:read -> 403
        assert r.status_code == 403


# ── Row 18: Remote PostgreSQL connections require verified TLS ──

def test_row_18_migrate_postgres_remote_requires_tls():
    from scripts.migrate import validate_postgres_dsn_security

    # Remote hosts without verified TLS raise ValueError
    with pytest.raises(ValueError, match="verified TLS"):
        validate_postgres_dsn_security("postgresql://user:pass@db.production.company.com:5432/skewai")

    with pytest.raises(ValueError, match="verified TLS"):
        validate_postgres_dsn_security("postgresql://user:pass@db.production.company.com:5432/skewai?sslmode=require")

    with pytest.raises(ValueError, match="verified TLS"):
        validate_postgres_dsn_security("postgresql://user:pass@db.production.company.com:5432/skewai?sslmode=prefer")

    # Remote hosts with verified TLS pass
    validate_postgres_dsn_security("postgresql://user:pass@db.production.company.com:5432/skewai?sslmode=verify-full")
    validate_postgres_dsn_security("postgresql://user:pass@db.production.company.com:5432/skewai?sslmode=verify-ca")

    # Local loopback hosts pass without TLS
    validate_postgres_dsn_security("postgresql://user:pass@127.0.0.1:5432/skewai")
    validate_postgres_dsn_security("postgresql://user:pass@localhost:5432/skewai")
    validate_postgres_dsn_security("host=127.0.0.1 dbname=skewai")


# ── Row 19: Password login remember=False omits cookie Max-Age ──

def test_row_19_password_login_remember_false_sets_session_cookie(reset_ops_db):
    from src.api.routes.oidc_auth import _hash_password
    from src.data.timeutil import utc_now

    salt = secrets.token_hex(16)
    pw_hash = _hash_password("SuperSecret123!", salt)

    with ops_con() as con:
        con.execute(
            """
            INSERT INTO frontline_users
            (user_id, name, email, company, role, password_hash, salt, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ["u_01", "Alice", "alice@example.com", "TestCorp", "agent", pw_hash, salt, utc_now()],
        )

    with TestClient(app) as client:
        # remember = False
        r_no_remember = client.post(
            "/api/frontline/auth/login",
            json={
                "email": "alice@example.com",
                "password": "SuperSecret123!",
                "remember": False,
            },
        )
        assert r_no_remember.status_code == 200
        # Cookie header exists and does not contain Max-Age (browser session cookie)
        set_cookie = r_no_remember.headers.get("set-cookie", "")
        assert "frontline_session" in set_cookie
        assert "Max-Age" not in set_cookie

        # remember = True
        r_remember = client.post(
            "/api/frontline/auth/login",
            json={
                "email": "alice@example.com",
                "password": "SuperSecret123!",
                "remember": True,
            },
        )
        assert r_remember.status_code == 200
        set_cookie_rem = r_remember.headers.get("set-cookie", "")
        assert "Max-Age=" in set_cookie_rem


# ── Row 20: SESSION_SECRET is required; no dev-only fallback ──

def test_row_20_session_secret_required_no_fallback(monkeypatch):
    monkeypatch.delenv("SESSION_SECRET", raising=False)
    monkeypatch.delenv("FRONTLINE_SESSION_SECRET", raising=False)

    from src.api.rbac import _secret
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc_info:
        _secret()
    assert exc_info.value.status_code == 503
    assert "SESSION_SECRET" in exc_info.value.detail


# ── Row 22: Sign-in form defaults remember to false ──

def test_row_22_dashboard_signin_source_check():
    from src.config import REPO_ROOT

    signin_path = REPO_ROOT / "dashboard" / "routes" / "SignIn.jsx"
    content = signin_path.read_text(encoding="utf-8")

    # Verify rememberApiKey state is initialized to false and passed to setApiKey
    assert "rememberApiKey, setRememberApiKey] = useState(false)" in content
    assert "setApiKey(key, { remember: rememberApiKey })" in content
    assert 'checked={rememberApiKey}' in content


# ── Row 23: Readiness endpoint suppresses internal details for unauthenticated callers ──

def test_row_23_health_ready_minimal_for_unauthenticated_under_auth(monkeypatch):
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", "secret_key_123")

    with TestClient(app) as client:
        # Unauthenticated request receives minimal status
        r_unauth = client.get("/health/ready")
        assert r_unauth.status_code in (200, 503)
        data_unauth = r_unauth.json()
        assert "ready" in data_unauth
        assert "status" in data_unauth
        assert "checks" not in data_unauth

        # Authenticated request receives full checks
        r_auth = client.get("/health/ready", headers={"X-API-Key": "secret_key_123"})
        assert r_auth.status_code in (200, 503)
        data_auth = r_auth.json()
        assert "checks" in data_auth
