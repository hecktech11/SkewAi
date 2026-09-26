"""Remediation regression tests — HIGH items 11-25."""

from __future__ import annotations

import pytest


# ── ITEM 11: WS query-key bypass ─────────────────────────────────────────────


def test_item11_ws_query_key_rejected_by_default():
    import anyio

    from src.api.auth import require_ws_api_key
    from fastapi import HTTPException

    # Query-only key with no opt-in must raise (parity with HTTP).
    with pytest.raises(HTTPException) as exc:
        anyio.run(require_ws_api_key, "s3cret", None, None)
    assert exc.value.status_code == 401


def test_item11_ws_query_key_opt_in_only(monkeypatch):
    import anyio

    from src.api.auth import require_ws_api_key

    monkeypatch.setenv("FRONTLINE_ALLOW_QUERY_KEY", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", "ws-query-key")
    assert anyio.run(require_ws_api_key, "ws-query-key", None, None) is True


def test_item11_ws_header_key_accepted(monkeypatch):
    import anyio

    from src.api.auth import require_ws_api_key

    monkeypatch.setenv("FRONTLINE_API_KEY", "ws-header-key")
    assert anyio.run(require_ws_api_key, None, None, "ws-header-key") is True


# ── ITEM 12: OIDC config authorization ───────────────────────────────────────


def test_item12_oidc_config_needs_admin(reset_ops_db, seed_automotive_pack, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app

    key = "oidc-rbac-key-32-bytes-long!!!!!!"
    monkeypatch.setenv("FRONTLINE_API_KEY", key)
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.delenv("FRONTLINE_OPEN_MODE", raising=False)
    body = {"client_id": "x", "client_secret": "y"}
    with TestClient(app) as c:
        # Unauthenticated (relative to hardened gate) is rejected…
        r0 = c.put("/api/frontline/auth/oidc/config", json=body)
        assert r0.status_code == 401
        # …an ordinary API key (service principal) is FORBIDDEN…
        r1 = c.put(
            "/api/frontline/auth/oidc/config", json=body, headers={"X-API-Key": key}
        )
        assert r1.status_code == 403, (
            f"service key must not rewrite IdP config: {r1.status_code} {r1.text[:200]}"
        )
        from src.api.rbac import issue_session

        monkeypatch.setenv("SESSION_SECRET", "test-session-secret-distinct-32b!!")
        monkeypatch.setenv("FRONTLINE_BOOTSTRAP_ADMIN", "1")
        admin = issue_session("tester", "admin", issuer_role="admin")["token"]
        r2 = c.put(
            "/api/frontline/auth/oidc/config",
            json={"client_id": "bad", "client_secret": "x"},
            headers={"X-API-Key": key, "X-Frontline-Session": admin},
        )
        assert r2.status_code in (200, 400), r2.text[:200]
        assert r2.status_code != 403  # auth passed if not 403-for-role


def test_item12_oidc_config_malformed_payload_admin(reset_ops_db, seed_automotive_pack, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app

    key = "oidc-rbac-key-32-bytes-long!!!!!!"
    monkeypatch.setenv("FRONTLINE_API_KEY", key)
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret-distinct-32b!!")
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_BOOTSTRAP_ADMIN", "1")
    monkeypatch.delenv("FRONTLINE_OPEN_MODE", raising=False)
    from src.api.rbac import issue_session

    admin = issue_session("tester", "admin", issuer_role="admin")["token"]
    with TestClient(app) as c:
        r = c.put(
            "/api/frontline/auth/oidc/config",
            json={},
            headers={"X-API-Key": key, "X-Frontline-Session": admin},
        )
        assert r.status_code == 400, "malformed payload must be 400, not 500"


# ── ITEM 13: billing webhook ─────────────────────────────────────────────────


def _sig(secret: str, body: bytes, ts: int) -> str:
    import hashlib
    import hmac

    v1 = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return f"t={ts},v1={v1}"


def test_item13_webhook_valid_and_replay_idempotent(reset_ops_db, monkeypatch):
    import json
    import time

    from src.frontline.billing import stripe_checkout, stripe_webhook

    secret = "whsec-test-secret-32-bytes-long!!"
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", secret)
    co = stripe_checkout("acme", plan="pilot")
    payload = {
        "id": "evt_replay_1",
        "type": "checkout.session.completed",
        "stripe_session": co["stripe_session"],
        "tenant_id": "acme",
        "plan": "pilot",
    }
    raw = json.dumps(payload).encode()
    sig = _sig(secret, raw, int(time.time()))
    first = stripe_webhook(payload, raw_body=raw, signature=sig)
    assert first["ok"] is True and first["plan"] == "pilot"
    second = stripe_webhook(payload, raw_body=raw, signature=sig)
    assert second["ok"] is True and second.get("replayed") is True


def test_item13_webhook_rejects_bad_stale_unknown(reset_ops_db, monkeypatch):
    import json
    import time

    import pytest

    from src.frontline.billing import stripe_checkout, stripe_webhook

    secret = "whsec-test-secret-32-bytes-long!!"
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", secret)
    co = stripe_checkout("acme", plan="pilot")
    base = {
        "type": "checkout.session.completed",
        "stripe_session": co["stripe_session"],
        "tenant_id": "acme",
        "plan": "pilot",
    }
    raw = json.dumps(base).encode()
    # Invalid signature.
    with pytest.raises(PermissionError):
        stripe_webhook(base, raw_body=raw, signature="t=1,v1=deadbeef")
    # Expired signature (replay of an old but validly-signed body).
    old_sig = _sig(secret, raw, int(time.time()) - 3600)
    with pytest.raises(PermissionError, match="[Ss]tale"):
        stripe_webhook(base, raw_body=raw, signature=old_sig)
    # Unknown session.
    unk = dict(base, stripe_session="cs_test_forged")
    unk_raw = json.dumps(unk).encode()
    with pytest.raises(LookupError):
        stripe_webhook(
            unk, raw_body=unk_raw,
            signature=_sig(secret, unk_raw, int(time.time())),
        )


def test_item13_webhook_ignores_client_plan_and_tenant(reset_ops_db, monkeypatch):
    import json
    import time

    import pytest

    from src.frontline.billing import get_account, stripe_checkout, stripe_webhook

    secret = "whsec-test-secret-32-bytes-long!!"
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", secret)
    co = stripe_checkout("acme", plan="pilot")
    # Attacker claims enterprise plan + a different tenant for acme's session.
    evil = {
        "id": "evt_evil_1",
        "type": "checkout.session.completed",
        "stripe_session": co["stripe_session"],
        "tenant_id": "attacker",
        "plan": "enterprise",
    }
    evil_raw = json.dumps(evil).encode()
    with pytest.raises(PermissionError, match="[Tt]enant"):
        stripe_webhook(
            evil, raw_body=evil_raw,
            signature=_sig(secret, evil_raw, int(time.time())),
        )
    # Correct tenant but inflated plan: server-side invoice plan wins.
    Upgrade = dict(evil, tenant_id="acme", id="evt_evil_2")
    up_raw = json.dumps(Upgrade).encode()
    out = stripe_webhook(
        Upgrade, raw_body=up_raw, signature=_sig(secret, up_raw, int(time.time()))
    )
    assert out["plan"] == "pilot", "client plan must never escalate"
    assert get_account("acme")["plan"] == "pilot"


# ── ITEM 14: CSV jail ────────────────────────────────────────────────────────


def test_item14_csv_jail_traversal_and_symlink(reset_ops_db, seed_automotive_pack, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from src.api.main import app
    from src.api.routes.pipeline import _resolve_builder_csv
    from fastapi import HTTPException

    monkeypatch.setenv("FRONTLINE_API_KEY", "jail-key")
    # /etc/passwd rejected.
    with pytest.raises(HTTPException) as e:
        _resolve_builder_csv("/etc/passwd")
    assert e.value.status_code == 400
    # ../ traversal rejected.
    with pytest.raises(HTTPException):
        _resolve_builder_csv("data/uploads/../../src/api/main.py")
    # Non-csv rejected.
    with pytest.raises(HTTPException):
        _resolve_builder_csv("data/uploads/evil.exe")
    # Symlink escape rejected.
    from src.config import REPO_ROOT

    link = REPO_ROOT / "data" / "uploads" / "test_escape_link.csv"
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to("/etc/hostname")
        with pytest.raises(HTTPException):
            _resolve_builder_csv(str(link))
    finally:
        if link.is_symlink() or link.exists():
            link.unlink()
    with TestClient(app) as c:
        r = c.post(
            "/api/frontline/pack-builder/insight",
            json={"csv_path": "/etc/passwd"},
            headers={"X-API-Key": "jail-key"},
        )
        assert r.status_code in (400, 404)


# ── ITEM 15: upload limits ───────────────────────────────────────────────────


def test_item15_upload_validation(reset_ops_db, seed_automotive_pack, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app

    monkeypatch.setenv("FRONTLINE_API_KEY", "upload-key")
    with TestClient(app) as c:
        h = {"X-API-Key": "upload-key"}
        # Oversized upload rejected.
        big = b"a,b\n" + b"x,y\n" * (6 * 1024 * 1024 // 4)
        r = c.post("/api/frontline/pack-builder/profile",
                   files={"file": ("big.csv", big, "text/csv")}, headers=h)
        assert r.status_code == 400
        # Executable renamed .csv rejected.
        r = c.post("/api/frontline/pack-builder/profile",
                   files={"file": ("evil.csv", b"MZ" + b"\x90" * 100, "text/csv")},
                   headers=h)
        assert r.status_code == 400
        # NUL bytes rejected.
        r = c.post("/api/frontline/pack-builder/profile",
                   files={"file": ("nul.csv", b"a,b\n\x00\x01\x02", "text/csv")},
                   headers=h)
        assert r.status_code == 400
        # Valid CSV accepted with a randomized server-side path.
        r = c.post("/api/frontline/pack-builder/profile",
                   files={"file": ("../../../etc/evil.csv", b"a,b\n1,2\n", "text/csv")},
                   headers=h)
        assert r.status_code == 200, r.text[:300]
        body = r.json()
        assert body["csv_path"].endswith(".csv")
        assert "etc" not in body["csv_path"].replace("builder", "")


# ── ITEM 16: PII redaction ───────────────────────────────────────────────────


def test_item16_pii_redaction_unit():
    from src.security.pii import find_pii, redact_dict, redact_pii, redact_turns

    text = "Call me at jane.doe@example.com or 415-555-0132, 123 Main Street, SSN 123-45-6789"
    red = redact_pii(text)
    assert "jane.doe@example.com" not in red and "[EMAIL]" in red
    assert "415-555-0132" not in red and "[PHONE]" in red
    assert "123 Main Street" not in red and "[ADDRESS]" in red
    assert "123-45-6789" not in red and "[SSN]" in red
    assert set(find_pii(text)) >= {"email", "phone", "ssn", "address"}
    row = redact_dict({"description_summary": text, "severity": "High"})
    assert "[EMAIL]" in row["description_summary"] and row["severity"] == "High"
    turns = redact_turns([{"speaker": "customer", "text": text}])
    assert "[PHONE]" in turns[0]["text"]


def test_item16_list_cases_redacted_by_default(reset_ops_db, seed_automotive_pack, monkeypatch):
    from fastapi.testclient import TestClient

    from src.api.main import app
    from src.data.warehouse import ops_con
    from src.data.timeutil import utc_now

    key = "pii-list-key"
    monkeypatch.setenv("FRONTLINE_API_KEY", key)
    with ops_con() as con:
        con.execute(
            """INSERT INTO cases (case_id, interaction_id, pack_id, created_at,
               category, description_summary, onset, severity, severity_source,
               priority, safety_flags, status)
               VALUES ('case_pii_1', 'int_pii_1', 'automotive_nhtsa', ?, 'X',
               'email jane.doe@example.com phone 415-555-0132', ?, 'Medium',
               'rules', 3, '{}', 'open')""",
            [utc_now(), utc_now()],
        )
    with TestClient(app) as c:
        h = {"X-API-Key": key}
        r = c.get("/api/frontline/cases", headers=h)
        assert r.status_code == 200
        descs = [x.get("description_summary") or "" for x in r.json()["cases"]]
        assert not any("jane.doe@example.com" in d for d in descs), "default must redact"
        assert any("[EMAIL]" in d for d in descs)
        # Explicit opt-out without dsr:export is forbidden (agent session)…
        from src.api.rbac import issue_session

        monkeypatch.setenv("SESSION_SECRET", "test-session-secret-distinct-32b!!")
        agent = issue_session("reader", "agent")["token"]
        r2 = c.get(
            "/api/frontline/cases?scrub_pii=false",
            headers={**h, "X-Frontline-Session": agent},
        )
        assert r2.status_code == 403


# ── ITEM 17: ledger health ───────────────────────────────────────────────────


def test_item17_ledger_failure_observable(reset_ops_db):
    from src.ledger import AgentAction, ledger_health, record_action
    from src.ledger.writer import _note_degraded

    iid = "int_ledger_health"
    before = ledger_health()["degraded_total"]
    _note_degraded("test-reason", action_id="a1", interaction_id=iid, error="boom")
    after = ledger_health()
    assert after["degraded_total"] == before + 1
    assert after["degraded_by_reason"].get("test-reason") == 1
    assert after["last_degraded"]["action_id"] == "a1"
    assert after["alert_threshold"] == 10
    # Core insert failure still raises (never a silent success).
    with pytest.raises(Exception):
        record_action(AgentAction(
            interaction_id=iid, agent="x", action_type="not_a_real_action_type",
        ))


def test_item17_record_with_status_reports_degraded(reset_ops_db, seed_automotive_pack):
    from src.ledger import AgentAction, record_action_with_status
    from src.ledger.writer import remember_interaction_pack

    remember_interaction_pack("int_ledger_status", "automotive_nhtsa")
    aid, status = record_action_with_status(AgentAction(
        interaction_id="int_ledger_status", agent="investigator",
        action_type="similar_search", output_summary="ok",
        evidence_ids=["NHTSA-100001"],
    ))
    assert aid and status["ok"] is True
    assert status["degraded"] == []


# ── ITEM 18: job queue ───────────────────────────────────────────────────────


def test_item18_atomic_claim_and_crash_recovery(reset_ops_db):
    from datetime import timedelta

    from src.data.timeutil import utc_now
    from src.data.warehouse import ops_con
    from src.jobs.queue import _ensure, enqueue, run_next

    j = enqueue("audit_contact", {"interaction_id": "int_crash"})
    done = run_next(worker_id="w1")
    assert done is not None and done["status"] == "done", "claim must succeed first try"
    # Simulate a crash: running job with an expired lease, owner gone.
    j2 = enqueue("audit_contact", {"interaction_id": "int_crash2"})
    claimed = run_next(worker_id="crasher", lease_s=1)
    assert claimed is not None
    # Force-expire the lease in the past (worker died without finishing).
    with ops_con() as con:
        _ensure(con)
        # Requeue as running+expired (run_next completed it, so craft directly).
        con.execute(
            """INSERT INTO job_queue (job_id, job_type, status, payload_json,
               attempts, lease_owner, lease_expires)
               VALUES ('job_crashed_1', 'audit_contact', 'running', '{}', 1,
               'dead-worker', ?)""",
            [utc_now() - timedelta(seconds=60)],
        )
    recovered = run_next(worker_id="w2")
    assert recovered is not None
    assert recovered["job_id"] == "job_crashed_1"
    assert recovered["status"] == "done"


def test_item18_idempotent_enqueue_and_completion(reset_ops_db):
    from src.jobs.queue import _finish, enqueue, run_next

    a = enqueue("audit_contact", {"interaction_id": "int_x1"}, idempotency_key="idem-1")
    b = enqueue("audit_contact", {"interaction_id": "int_x1"}, idempotency_key="idem-1")
    assert b["job_id"] == a["job_id"] and b.get("duplicate") is True
    done = run_next(worker_id="w9")
    assert done["status"] == "done"
    # Duplicate completion returns the stored result, no double-apply.
    dup = _finish(a["job_id"], "w9", ok=True, result={"ok": True})
    assert dup.get("duplicate_completion") is True


def test_r19_lease_heartbeat_prevents_duplicate_execution(reset_ops_db):
    """A long-running job with lease_s=1 must renew its lease via heartbeat so worker 2 cannot claim it."""
    import time
    from src.jobs.queue import enqueue, register_handler, run_next

    side_effects = []

    def slow_handler(payload):
        side_effects.append("started")
        time.sleep(1.2)  # longer than lease_s=1
        side_effects.append("finished")
        return {"ok": True}

    register_handler("test_slow_job", slow_handler)

    try:
        j = enqueue("test_slow_job", {})
        # Worker 1 runs in a thread
        import threading
        w1_result = []

        t1 = threading.Thread(target=lambda: w1_result.append(run_next(worker_id="w1", lease_s=1)))
        t1.start()

        # While worker 1 is executing (after 0.5s, half the lease):
        time.sleep(0.5)
        # Worker 2 attempts to claim
        w2_res = run_next(worker_id="w2", lease_s=1)
        assert w2_res is None, "Worker 2 must NOT steal the job while worker 1 is alive and heartbeating"

        t1.join(timeout=3.0)
        assert len(w1_result) == 1 and w1_result[0]["status"] == "done"
        assert side_effects == ["started", "finished"]
    finally:
        from src.jobs.queue import _HANDLERS
        _HANDLERS.pop("test_slow_job", None)


def test_r20_failed_job_result_not_marked_done(reset_ops_db):
    """When a job handler returns ok=False or success=False, run_next must not mark it done."""
    from src.jobs.queue import enqueue, register_handler, run_next

    def failing_handler(payload):
        return {"ok": False, "error": "CSV file does not exist", "terminal": True}

    register_handler("test_fail_job", failing_handler)

    try:
        j = enqueue("test_fail_job", {})
        res = run_next(worker_id="w_fail")
        assert res is not None
        assert res["status"] == "failed"
        assert "CSV file does not exist" in str(res.get("error"))
    finally:
        from src.jobs.queue import _HANDLERS
        _HANDLERS.pop("test_fail_job", None)



# ── ITEM 19: cookies ─────────────────────────────────────────────────────────


def test_item19_cookie_attributes_and_rotation(monkeypatch):
    from fastapi.responses import JSONResponse

    from src.api.routes.oidc_auth import _session_cookie, auth_logout
    from src.api.rbac import session_cookie_name, session_token_from_cookies
    import anyio

    monkeypatch.setenv("PILOT_HARDENED", "1")
    monkeypatch.delenv("ENV", raising=False)
    assert session_cookie_name() == "__Host-frontline_session"
    resp = JSONResponse({"ok": True})
    _session_cookie(resp, "tok123")
    cookie = resp.headers.get("set-cookie", "")
    assert "__Host-frontline_session" in cookie
    assert "Secure" in cookie and "HttpOnly" in cookie and "SameSite=lax" in cookie
    # Readers accept both names, __Host- wins.
    assert session_token_from_cookies(
        {"frontline_session": "legacy", "__Host-frontline_session": "host"}
    ) == "host"
    assert session_token_from_cookies({"frontline_session": "legacy"}) == "legacy"
    # Logout clears both.
    out = anyio.run(auth_logout)
    cleared = out.headers.getlist("set-cookie")
    assert any("__Host-frontline_session" in c for c in cleared)
    assert any("frontline_session" in c for c in cleared)

    monkeypatch.delenv("PILOT_HARDENED", raising=False)
    assert session_cookie_name() == "frontline_session"


# ── ITEM 20: dashboard auth ──────────────────────────────────────────────────


def test_item20_dashboard_auth_files():
    from pathlib import Path

    from src.config import REPO_ROOT

    src = (REPO_ROOT / "dashboard" / "src" / "apiAuth.js").read_text()
    # Logout invalidates memory + disk credentials (not just subject).
    assert "clearApiKey()" in src.split("export async function signOut()")[1].split(
        "export function isKeyless"
    )[0]
    assert "isKeyless" in src
    # No component reads credentials directly; ui/hooks localStorage is prefs.
    import subprocess

    out = subprocess.run(
        ["grep", "-rn", "frontline_api_key", "dashboard/src", "--include=*.jsx"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert out.stdout.strip() == "", f"direct JSX credential reads: {out.stdout[:300]}"


# ── ITEM 21: embeddings ──────────────────────────────────────────────────────


def test_item21_dim_idf_collisions():
    from src.ml_runtime.embeddings import (
        cosine,
        embed_text,
        embedding_dim,
        fit_idf,
        idf_status,
        measure_collisions,
        reset_idf,
    )

    assert embedding_dim() >= 512
    assert len(embed_text("hello world")) == embedding_dim()
    reset_idf()
    fit_idf([
        "brake grinding noise on honda cr-v",
        "airbag warning light on toyota camry",
        "power window failure on ford f-150",
        "brake grinding noise on honda cr-v when cold",
    ])
    assert idf_status()["features"] > 0
    # Deterministic with fitted IDF.
    assert embed_text("brake grinding") == embed_text("brake grinding")
    # Length bias: repeated content must not outrank by magnitude (L2 norm).
    short = embed_text("brake failure")
    long = embed_text(" ".join(["brake failure"] * 50))
    assert cosine(short, long) > 0.9, "direction must survive repetition"
    import math

    assert abs(math.sqrt(sum(v * v for v in long)) - 1.0) < 1e-9
    rep = measure_collisions([
        "brake grinding noise on honda cr-v",
        "airbag warning light on toyota camry",
        "power window failure on ford f-150",
        "transmission slipping on nissan altima",
    ])
    assert rep["dim"] == embedding_dim()
    assert rep["collision_rate"] == 0.0
    reset_idf()


# ── ITEM 22: cluster model selection ─────────────────────────────────────────


def test_item22_k_selection_and_small_data():
    from src.ml_runtime.clustering import select_k, silhouette_score

    # Two well-separated groups -> k=2, not forced 1.
    vecs = [
        [1.0, 0.0, 0.0], [0.95, 0.05, 0.0], [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0], [0.0, 0.95, 0.05], [0.05, 0.9, 0.1],
    ]
    k, score, method = select_k(vecs, k_max=4)
    assert k == 2 and method == "silhouette" and score > 0
    assert silhouette_score(vecs, [0, 0, 0, 1, 1, 1]) > 0.5
    # Degenerate inputs.
    assert select_k([], k_max=5)[0] == 0
    assert select_k([[1.0, 0.0]], k_max=5)[0] == 1
    k2, _, m2 = select_k([[1.0, 0.0], [0.0, 1.0]], k_max=5)
    assert (k2, m2) == (2, "pairwise")
    k3, _, _ = select_k([[1.0, 0.0], [1.0, 0.0]], k_max=5)
    assert k3 == 1


# ── ITEM 23: association lift ────────────────────────────────────────────────


def test_item23_lift_known_contingency_table():
    from src.ml_runtime.association import (
        association_score,
        lift_for_pair,
        population_support,
        rank_by_association,
    )

    # 100 rows: 40 BRAKES/HONDA, 40 BRAKES/FORD, 10 AIR/HONDA, 10 AIR/FORD.
    pop = (
        [{"category": "BRAKES", "entity_2": "HONDA"}] * 40
        + [{"category": "BRAKES", "entity_2": "FORD"}] * 40
        + [{"category": "AIR", "entity_2": "HONDA"}] * 10
        + [{"category": "AIR", "entity_2": "FORD"}] * 10
    )
    sup = population_support(pop, category="BRAKES", entity_2="HONDA")
    assert sup == {
        "n": 100.0, "n_cat": 80.0, "n_ent": 50.0, "n_both": 40.0,
        "support": 0.4, "p_cat": 0.8, "p_ent": 0.5,
    }
    lift = lift_for_pair(
        {"category": "BRAKES", "entity_2": "HONDA"},
        {"category": "BRAKES", "entity_2": "HONDA"},
        pop,
    )
    assert lift == pytest.approx(0.4 / (0.8 * 0.5))  # == 1.0
    rare_lift = lift_for_pair(
        {"category": "AIR", "entity_2": "HONDA"},
        {"category": "AIR", "entity_2": "HONDA"},
        pop,
    )
    assert rare_lift == pytest.approx(0.1 / (0.2 * 0.5))  # == 1.0
    # Candidate-side vs population-side distinct: a dependent shortlist
    # yields different lift than the independent full corpus.
    skewed = [{"category": "AIR", "entity_2": "HONDA"}] * 9 + [
        {"category": "BRAKES", "entity_2": "FORD"}
    ]
    skewed_lift = lift_for_pair(
        {"category": "AIR", "entity_2": "HONDA"},
        {"category": "AIR", "entity_2": "HONDA"},
        skewed,
    )
    assert skewed_lift == pytest.approx(0.9 / (0.9 * 0.9))
    assert skewed_lift != pytest.approx(rare_lift), (
        "population denominators must change lift — candidate pool is not the corpus"
    )
    # Recency is a tie-break only: score has no time component.
    q = {"category": "AIR", "entity_2": "HONDA"}
    old = {"record_id": "O", "category": "AIR", "entity_2": "HONDA",
           "received_at": "2020-01-01"}
    new = {"record_id": "N", "category": "AIR", "entity_2": "HONDA",
           "received_at": "2026-01-01"}
    assert association_score(q, old, pop) == association_score(q, new, pop)
    ranked = rank_by_association(q, [old, new], pop, top_k=2)
    assert ranked[0]["record_id"] == "N", "ties break newest-first"
    assert ranked[0]["assoc_population"] == "full-corpus"
    fb = rank_by_association(q, [old, new], None, top_k=2)
    assert fb[0]["assoc_population"] == "candidates-fallback"


# ── ITEM 24: entity resolution ───────────────────────────────────────────────


def test_item24_blocking_checksum_namespaces():
    from src.ml_runtime.entity_resolution import (
        identity_key,
        is_valid_vin,
        resolve_matches,
        same_entity,
    )

    assert is_valid_vin("1HGCM82633A004352") is True
    assert is_valid_vin("1HGCM82633A004353") is False  # bad check digit
    assert is_valid_vin("1HGCM8263") is False  # not a VIN
    assert is_valid_vin("1HGCM82633A00435I") is False  # I disallowed
    # Invalid 17-char VIN never links, even to itself.
    bad = {"vin": "1HGCM82633A004353", "entity_2": "HONDA", "entity_3": "ACCORD"}
    assert identity_key(bad) == ""
    assert same_entity(bad, dict(bad)) is False
    # Namespace separation preserved.
    assert identity_key({"serial": "123"}) == "serial:123"
    assert identity_key({"vin": "123"}) == "vin:123"
    # Blocking correctness on a larger set: same pairs as brute force would give.
    recs = []
    for i in range(60):
        recs.append({"vin": "1HGCM82633A004352", "entity_2": "HONDA",
                     "entity_3": "ACCORD", "n": i})
    for i in range(60):
        recs.append({"serial": f"SN-{i}", "entity_2": "FORD",
                     "entity_3": "F-150", "n": i})
    pairs = resolve_matches(recs)
    assert len(pairs) == 60 * 59 // 2, "all 60 hondas pairwise linked, fords distinct"
    assert all(i < 60 and j < 60 for i, j in pairs)


# ── ITEM 25: severity fallback ───────────────────────────────────────────────


def test_item25_model_source_and_fallback_matrix(reset_ops_db, seed_automotive_pack, tmp_path):
    import json

    from src.agents.base import InteractionContext
    from src.agents.triage import SEVERITY_FALLBACK_POLICY, model_health, score_severity
    from src.domains.loader import load_pack

    assert SEVERITY_FALLBACK_POLICY == "safe-fallback-with-ledger"
    pack = load_pack("automotive_nhtsa", reload=True)

    def _ctx(**slots):
        ctx = InteractionContext(
            interaction_id="int_sev", pack=pack, channel="web_text"
        )
        ctx.slots.update(slots)
        return ctx

    # Rules path for every severity level resolves without a model.
    pack.manifest.severity.model_artifact = None
    for cat, desc, want in [
        ("AIR BAGS", "airbag light is on", "Medium"),
        ("SERVICE BRAKES", "brakes grind a bit", "Medium"),
        ("SEATS", "seat fabric is worn", "Low"),
    ]:
        sev, src, _ = score_severity(_ctx(category=cat, description=desc))
        assert src == "rules" and sev == want
    # Configured model that works -> source MUST be model.
    overlay = tmp_path / "sev.json"
    overlay.write_text(json.dumps({"rules": [{"keyword": "meltdown", "severity": "High"}]}))
    pack.manifest.severity.model_artifact = str(overlay)
    sev, src, _ = score_severity(
        _ctx(category="ENGINE", description="total meltdown observed")
    )
    assert (sev, src) == ("High", "model"), "valid model result must report source=model"
    assert model_health()[str(overlay)] is True
    # Configured model that FAILS -> rules source + model_error ledger + unhealthy.
    pack.manifest.severity.model_artifact = "/nonexistent/severity.json"
    sev2, src2, reason = score_severity(
        _ctx(category="ENGINE", description="total meltdown observed")
    )
    assert src2 == "rules", "failed model must never report source=model"
    assert "model fallback" in reason
    assert model_health()["/nonexistent/severity.json"] is False
    from src.data.warehouse import ops_con

    with ops_con(read_only=True) as con:
        rows = con.execute(
            """SELECT COUNT(*) FROM agent_actions WHERE agent = 'triage'
               AND ok = FALSE AND output_summary LIKE 'model_error%'"""
        ).fetchone()
    assert rows[0] >= 1, "model failure must be ledgered"
