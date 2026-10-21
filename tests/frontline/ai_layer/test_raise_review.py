"""Supervisor review of AI severity raises.  [AI-LAYER]

Role checks use the existing RBAC (signed sessions). A verdict must never
change case severity or the decision store.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from src.ai_layer import raise_review
from src.ai_layer.raise_review import MIN_REVIEWS_FOR_RATE, ReviewError, stats, submit_review

ADMIN_SECRET = "test-admin-secret-key-32-chars-auth!!"
SERVICE_KEY = "test-service-secret-key-32-chars-auth!"


def _seed_raise(i: int = 0, *, with_case: bool = True) -> tuple[str, str]:
    from src.ai_layer.store import record_event
    from src.data.warehouse import ops_con

    iid = f"int_rr_{i:04d}"
    eid = record_event(kind="severity", interaction_id=iid, model_id="muse-spark-test",
                       deterministic={"severity": "Medium", "source": "rules"},
                       ai_proposal={"severity": "Critical", "confidence": 0.9, "reason": "brake failure at speed",
                                    "evidence": "brakes failed"},
                       accepted={"severity": "Critical"}, reason="raised")
    assert eid
    if with_case:
        with ops_con() as con:
            con.execute(
                "INSERT INTO cases (case_id, interaction_id, pack_id, created_at, severity, severity_source, priority) "
                "VALUES (?, ?, 'automotive_nhtsa', ?, 'Critical', 'rules', 1)",
                [f"case_rr_{i:04d}", iid, datetime.now(timezone.utc).replace(tzinfo=None)],
            )
    return eid, iid


def _snapshot():
    from src.data.warehouse import ops_con

    with ops_con() as con:
        cases = con.execute("SELECT case_id, severity, priority, severity_source FROM cases ORDER BY case_id").fetchall()
        events = con.execute("SELECT event_id, kind, reason, accepted FROM ai_layer_events ORDER BY event_id").fetchall()
    return cases, events


def test_supervisor_accept_and_reject_never_change_severity(reset_ops_db):
    e1, _ = _seed_raise(1)
    e2, _ = _seed_raise(2)
    before = _snapshot()
    r1 = submit_review(event_id=e1, verdict="accept", note="correct, brake loss", reviewer="sup1", reviewer_role="supervisor")
    r2 = submit_review(event_id=e2, verdict="reject", note=None, reviewer="sup1", reviewer_role="supervisor")
    assert r1["case_severity_unchanged"] and r1["case"]["severity"] == "Critical"
    assert r2["verdict"] == "reject" and r2["case"]["severity"] == "Critical"  # reject does NOT lower
    assert _snapshot() == before
    rows = {r["event_id"]: r for r in raise_review.list_raises()}
    assert rows[e1]["review"]["verdict"] == "accept" and rows[e2]["review"]["verdict"] == "reject"
    assert rows[e1]["rule_severity"] == "Medium" and rows[e1]["final_severity"] == "Critical"
    assert raise_review.verify_audit_chain() == {"ok": True, "rows": 2}


def test_non_supervisor_roles_refused_in_module(reset_ops_db):
    e1, _ = _seed_raise(1)
    for role in ("agent", "service", "auditor", "admin", "dsr_officer"):
        with pytest.raises(ReviewError) as ex:
            submit_review(event_id=e1, verdict="accept", note=None, reviewer="x", reviewer_role=role)
        assert ex.value.status == 403


def test_duplicate_invalid_and_unknown(reset_ops_db):
    e1, _ = _seed_raise(1)
    submit_review(event_id=e1, verdict="accept", note=None, reviewer="s", reviewer_role="supervisor")
    with pytest.raises(ReviewError) as ex:
        submit_review(event_id=e1, verdict="reject", note=None, reviewer="s", reviewer_role="supervisor")
    assert ex.value.status == 409
    with pytest.raises(ReviewError) as ex:
        submit_review(event_id=e1, verdict="maybe", note=None, reviewer="s", reviewer_role="supervisor")
    assert ex.value.status == 422
    with pytest.raises(ReviewError) as ex:
        submit_review(event_id="aie_nope", verdict="accept", note=None, reviewer="s", reviewer_role="supervisor")
    assert ex.value.status == 404
    chain = raise_review.verify_audit_chain()
    assert chain["ok"] and chain["rows"] == 2  # recorded + duplicate attempt


def test_non_raise_events_are_not_reviewable(reset_ops_db):
    from src.ai_layer.store import record_event

    eid = record_event(kind="severity", interaction_id="int_kept", deterministic={"severity": "Low"},
                       accepted={"severity": "Low"}, reason="ai_not_higher")
    with pytest.raises(ReviewError) as ex:
        submit_review(event_id=eid, verdict="accept", note=None, reviewer="s", reviewer_role="supervisor")
    assert ex.value.status == 404


def test_note_is_pii_redacted_and_capped(reset_ops_db):
    e1, _ = _seed_raise(1)
    out = submit_review(event_id=e1, verdict="accept", note="call me at jane.doe@example.com " + "x" * 900,
                        reviewer="s", reviewer_role="supervisor")
    assert "jane.doe@example.com" not in out["note"] and len(out["note"]) <= raise_review.NOTE_MAX


def test_small_n_makes_no_rate_claim(reset_ops_db):
    assert stats()["status"] == "no reviews yet" and stats()["agreement_rate"] is None
    for i in range(3):
        eid, _ = _seed_raise(i, with_case=False)
        submit_review(event_id=eid, verdict="accept" if i else "reject", note=None, reviewer="s", reviewer_role="supervisor")
    s = stats()
    assert s["n_label"] == "n = 3" and s["agreement_rate"] is None and s["ci95"] is None
    assert s["status"].startswith("too few reviews to estimate") and "n = 3" in s["status"]
    assert (s["accepted"], s["rejected"], s["reviewed"], s["ai_raises"], s["pending"]) == (2, 1, 3, 3, 0)


def test_rate_reported_from_threshold(reset_ops_db):
    n = MIN_REVIEWS_FOR_RATE
    for i in range(n):
        eid, _ = _seed_raise(i, with_case=False)
        submit_review(event_id=eid, verdict="accept" if i % 4 else "reject", note=None, reviewer="s",
                      reviewer_role="supervisor")
    s = stats()
    assert s["status"] == "estimated" and s["reviewed"] == n
    assert s["agreement_rate"] == round(15 / 20, 3)
    lo, hi = s["ci95"]
    assert lo < 0.75 < hi
    assert stats(min_n=n + 1)["agreement_rate"] is None


def test_audit_tamper_detected(reset_ops_db):
    from src.data.warehouse import ops_con

    e1, _ = _seed_raise(1)
    submit_review(event_id=e1, verdict="accept", note=None, reviewer="s", reviewer_role="supervisor")
    with ops_con() as con:
        con.execute("UPDATE ai_raise_review_audit SET detail = ? WHERE seq = 1", [json.dumps({"verdict": "reject"})])
    assert raise_review.verify_audit_chain()["ok"] is False


# ── routes + existing RBAC ──────────────────────────────────────────────────


@pytest.fixture
def auth_client(reset_ops_db, monkeypatch, no_meta):
    from fastapi.testclient import TestClient

    from src.api.main import app

    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", SERVICE_KEY)
    monkeypatch.setenv("SESSION_SECRET", ADMIN_SECRET)
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")
    monkeypatch.delenv("FRONTLINE_OPEN_MODE", raising=False)
    monkeypatch.delenv("FRONTLINE_SERVICE_IS_ADMIN", raising=False)
    with TestClient(app) as c:
        yield c


def _hdr(role: str | None):
    from src.api.rbac import issue_session

    h = {"X-API-Key": SERVICE_KEY}
    if role:
        h["X-Frontline-Session"] = issue_session(f"{role}_user", role, issuer_role="admin")["token"]
    return h


def test_route_supervisor_ok_others_403(auth_client):
    e1, _ = _seed_raise(1)
    before = _snapshot()
    for role in ("agent", "auditor", "admin", None):  # None = shared service key principal
        r = auth_client.post(f"/api/ai/raises/{e1}/review", json={"verdict": "accept"}, headers=_hdr(role))
        assert r.status_code == 403, (role, r.status_code, r.text)
    r = auth_client.post(f"/api/ai/raises/{e1}/review", json={"verdict": "accept", "note": "ok"},
                         headers=_hdr("supervisor"))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["verdict"] == "accept" and body["case_severity_unchanged"] is True
    assert _snapshot() == before
    again = auth_client.post(f"/api/ai/raises/{e1}/review", json={"verdict": "reject"}, headers=_hdr("supervisor"))
    assert again.status_code == 409
    listed = auth_client.get("/api/ai/raises", headers=_hdr("agent")).json()
    assert listed["count"] == 1 and listed["raises"][0]["review"]["reviewer"] == "supervisor_user"
    st = auth_client.get("/api/ai/raises/stats", headers=_hdr("agent")).json()
    assert st["n_label"] == "n = 1" and st["agreement_rate"] is None
    v = auth_client.get("/api/ai/raises/audit/verify", headers=_hdr("auditor")).json()
    assert v["ok"] is True and v["rows"] == 6  # 4 denied + 1 recorded + 1 duplicate
    assert auth_client.get("/api/ai/raises/audit/verify", headers=_hdr("agent")).status_code == 403


def test_route_requires_api_key(auth_client):
    e1, _ = _seed_raise(1)
    assert auth_client.post(f"/api/ai/raises/{e1}/review", json={"verdict": "accept"}).status_code == 401


def test_open_mode_cannot_review(reset_ops_db, monkeypatch, no_meta):
    from fastapi.testclient import TestClient

    from src.api.main import app

    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")
    e1, _ = _seed_raise(1)
    with TestClient(app) as c:
        r = c.post(f"/api/ai/raises/{e1}/review", json={"verdict": "accept"},
                   headers={"X-Frontline-Role": "supervisor"})  # bare header never elevates
        assert r.status_code == 403
        assert c.get("/api/ai/raises/stats").status_code == 200
