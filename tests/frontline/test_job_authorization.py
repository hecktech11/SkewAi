"""Job enqueue authorization contract (R01).

Enqueueing a job is equivalent to invoking the operation: a background worker
executes the payload with full process authority. So `POST /api/frontline/jobs`
must apply the same permission gate as running the job, validate the payload,
and jail any file reference it carries.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.rbac import issue_session
from src.data.warehouse import ops_con

TEST_SERVICE_KEY = "test-service-secret-key-32-chars-job!"
TEST_SESSION_SECRET = "test-session-secret-key-32-chars-job!"


@pytest.fixture
def auth_env(monkeypatch):
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", TEST_SERVICE_KEY)
    monkeypatch.setenv("SESSION_SECRET", TEST_SESSION_SECRET)
    monkeypatch.delenv("FRONTLINE_OPEN_MODE", raising=False)
    monkeypatch.delenv("FRONTLINE_SERVICE_IS_ADMIN", raising=False)
    monkeypatch.delenv("FRONTLINE_DSR_API_KEY", raising=False)


@pytest.fixture
def client(auth_env, reset_ops_db):
    with TestClient(app) as c:
        yield c


def _admin_headers() -> dict[str, str]:
    token = issue_session("admin_usr", "admin", issuer_role="admin")["token"]
    return {"X-API-Key": TEST_SERVICE_KEY, "X-Frontline-Session": token}


def _agent_headers() -> dict[str, str]:
    token = issue_session("agent_usr", "agent", issuer_role="admin")["token"]
    return {"X-API-Key": TEST_SERVICE_KEY, "X-Frontline-Session": token}


def _queued() -> list[dict]:
    from src.jobs.queue import list_jobs

    return list_jobs()


def test_service_principal_denied_run_next_is_also_denied_enqueue(client):
    """The R01 repro: the same principal must not be able to schedule what it
    is refused permission to execute."""
    headers = {"X-API-Key": TEST_SERVICE_KEY}

    run = client.post("/api/frontline/jobs/run-next", headers=headers)
    assert run.status_code == 403

    enq = client.post(
        "/api/frontline/jobs",
        headers=headers,
        json={"job_type": "ingest_source", "payload": {"pack_id": "automotive_nhtsa"}},
    )
    assert enq.status_code == 403, f"service principal enqueued a privileged job: {enq.text}"
    assert _queued() == []


def test_agent_session_cannot_enqueue_privileged_job(client):
    resp = client.post(
        "/api/frontline/jobs",
        headers=_agent_headers(),
        json={"job_type": "rebuild_clusters", "payload": {"pack_id": "automotive_nhtsa"}},
    )
    assert resp.status_code == 403
    assert _queued() == []


def test_admin_may_enqueue(client, tmp_path):
    csv = tmp_path / "source.csv"
    csv.write_text("record_id,text\nR-1,brake noise\n", encoding="utf-8")

    resp = client.post(
        "/api/frontline/jobs",
        headers=_admin_headers(),
        json={
            "job_type": "ingest_source",
            "payload": {
                "pack_id": "automotive_nhtsa",
                "source": "service",
                "csv_path": str(csv),
            },
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "pending"
    assert body["job_type"] == "ingest_source"


def test_enqueue_rejects_csv_path_outside_jail(client):
    """Job payloads reach local file handling; they need the same jail as uploads."""
    resp = client.post(
        "/api/frontline/jobs",
        headers=_admin_headers(),
        json={
            "job_type": "ingest_source",
            "payload": {
                "pack_id": "automotive_nhtsa",
                "source": "service",
                "csv_path": "/etc/passwd",
            },
        },
    )
    assert resp.status_code == 400, f"unjailed csv_path accepted: {resp.text}"
    assert _queued() == []


def test_enqueue_rejects_traversal_mapping_path(client, tmp_path):
    csv = tmp_path / "source.csv"
    csv.write_text("record_id,text\nR-1,brake noise\n", encoding="utf-8")

    resp = client.post(
        "/api/frontline/jobs",
        headers=_admin_headers(),
        json={
            "job_type": "ingest_source",
            "payload": {
                "pack_id": "automotive_nhtsa",
                "source": "service",
                "csv_path": str(csv),
                "mapping_path": "/etc/hosts",
            },
        },
    )
    assert resp.status_code == 400, f"unjailed mapping_path accepted: {resp.text}"
    assert _queued() == []


def test_enqueue_rejects_malformed_pack_id(client, tmp_path):
    csv = tmp_path / "source.csv"
    csv.write_text("record_id,text\nR-1,x\n", encoding="utf-8")

    resp = client.post(
        "/api/frontline/jobs",
        headers=_admin_headers(),
        json={
            "job_type": "ingest_source",
            "payload": {
                "pack_id": "../../etc",
                "source": "service",
                "csv_path": str(csv),
            },
        },
    )
    assert resp.status_code == 400
    assert _queued() == []


def test_unknown_job_type_rejected(client):
    resp = client.post(
        "/api/frontline/jobs",
        headers=_admin_headers(),
        json={"job_type": "totally_made_up_job", "payload": {}},
    )
    assert resp.status_code == 400
    assert _queued() == []


def test_enqueue_persists_initiating_principal(client):
    resp = client.post(
        "/api/frontline/jobs",
        headers=_admin_headers(),
        json={"job_type": "build_digest", "payload": {}},
    )
    assert resp.status_code == 200, resp.text
    jid = resp.json()["job_id"]

    with ops_con(read_only=True) as con:
        row = con.execute(
            "SELECT requested_by, requested_role FROM job_queue WHERE job_id = ?",
            [jid],
        ).fetchone()
    assert row is not None
    assert row[0] == "admin_usr"
    assert row[1] == "admin"


def test_execution_refuses_job_whose_principal_lacks_permission(reset_ops_db):
    """Policy is enforced at execution too, so a row inserted by any other
    path cannot borrow the worker's authority."""
    from src.jobs.queue import _ensure, register_handler, run_next

    ran: list[dict] = []
    register_handler("ingest_scale", lambda payload: ran.append(payload) or {"ok": True})
    try:
        with ops_con() as con:
            _ensure(con)
            con.execute(
                """
                INSERT INTO job_queue
                    (job_id, job_type, status, payload_json, requested_by, requested_role)
                VALUES ('job_forged', 'ingest_scale', 'pending', ?, 'agent_usr', 'agent')
                """,
                [json.dumps({"pack_id": "automotive_nhtsa"})],
            )

        result = run_next()

        assert ran == [], "worker executed a job its initiating principal could not authorize"
        assert result is not None
        assert result["status"] in {"failed", "dead"}, result
        with ops_con(read_only=True) as con:
            status = con.execute(
                "SELECT status FROM job_queue WHERE job_id = 'job_forged'"
            ).fetchone()[0]
        assert status in {"failed", "dead"}, status
    finally:
        from src.jobs.queue import _HANDLERS

        _HANDLERS.pop("ingest_scale", None)


def test_internal_system_enqueue_still_works(reset_ops_db):
    """In-process callers (orchestrator backfill, erasure drill) are not
    behind the HTTP boundary and must keep working."""
    from src.jobs.queue import enqueue

    job = enqueue("reenrich", {"interaction_id": "int_abc", "case_id": "case_abc"})

    assert job["status"] == "pending"
    with ops_con(read_only=True) as con:
        row = con.execute(
            "SELECT requested_role FROM job_queue WHERE job_id = ?", [job["job_id"]]
        ).fetchone()
    assert row[0] == "system"
