"""Ordinary read routes must not return customer PII (R02).

The interaction list returned SELECT * with no redaction. Interaction detail
scrubbed turns and actions but left the header and the nested case. Case notes
store free text in ``body``, which was not a redacted field. Explainability
returned slots, follow-up drafts and action summaries with no dsr:export gate.
A service credential could read a customer email through all four.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from src.api.main import app
from src.data.warehouse import ops_con

TEST_SERVICE_KEY = "test-service-secret-key-32-chars-auth!"
TEST_DSR_KEY = "test-dsr-officer-secret-key-32-chars-auth!"
SECRET = "ada.r02@example.com"
IID = "int_r02_read"
CID = "case_r02_read"


def _auth(monkeypatch) -> None:
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", TEST_SERVICE_KEY)
    monkeypatch.setenv("FRONTLINE_DSR_API_KEY", TEST_DSR_KEY)
    monkeypatch.setenv("SESSION_SECRET", "test-admin-secret-key-32-chars-auth!!")
    monkeypatch.delenv("FRONTLINE_OPEN_MODE", raising=False)
    monkeypatch.delenv("FRONTLINE_SERVICE_IS_ADMIN", raising=False)


def _seed() -> None:
    now = datetime.now(timezone.utc)
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions
            (interaction_id, pack_id, pack_version, started_at, channel, status, description)
            VALUES (?, 'automotive_nhtsa', 't', ?, 'web_text', 'completed', ?)
            """,
            [IID, now, f"caller {SECRET} reported a brake issue"],
        )
        con.execute(
            """
            INSERT INTO interaction_turns
            (turn_id, interaction_id, seq, speaker, text, ts)
            VALUES ('turn_r02', ?, 1, 'customer', ?, ?)
            """,
            [IID, f"email me at {SECRET}", now],
        )
        con.execute(
            """
            INSERT INTO cases
            (case_id, interaction_id, pack_id, created_at, category, description_summary,
             onset, severity, severity_source, priority, safety_flags, status, followup_draft)
            VALUES (?, ?, 'automotive_nhtsa', ?, 'brakes', ?, ?, 'Low', 'rules', 3, '{}', 'open', ?)
            """,
            [CID, IID, now, f"summary {SECRET}", now, f"draft reply to {SECRET}"],
        )
        con.execute(
            """
            INSERT INTO case_notes (note_id, case_id, author, body, created_at)
            VALUES ('note_r02', ?, 'operator', ?, ?)
            """,
            [CID, f"customer email {SECRET}", now],
        )
        con.execute(
            """
            INSERT INTO agent_actions
            (action_id, interaction_id, case_id, agent, action_type,
             input_summary, output_summary, evidence_ids, ok, ts)
            VALUES ('act_r02', ?, ?, 'case', 'case_created', ?, ?, '[]', TRUE, ?)
            """,
            [IID, CID, f"in {SECRET}", f"out {SECRET}", now],
        )


def _paths() -> list[str]:
    return [
        "/api/interactions",
        f"/api/interactions/{IID}",
        f"/api/frontline/cases/{CID}/notes",
        f"/api/frontline/explain/{IID}",
    ]


def test_service_key_reads_are_redacted(reset_ops_db, monkeypatch):
    _auth(monkeypatch)
    _seed()
    headers = {"X-API-Key": TEST_SERVICE_KEY}
    with TestClient(app) as client:
        for path in _paths():
            resp = client.get(path, headers=headers)
            assert resp.status_code == 200, path
            assert SECRET not in resp.text, path
            assert "[EMAIL]" in resp.text, path


def test_service_key_cannot_disable_redaction(reset_ops_db, monkeypatch):
    _auth(monkeypatch)
    _seed()
    headers = {"X-API-Key": TEST_SERVICE_KEY}
    with TestClient(app) as client:
        for path in _paths():
            resp = client.get(path + "?scrub_pii=false", headers=headers)
            assert resp.status_code == 403, path
            assert "dsr:export" in resp.text


def test_dsr_export_can_read_raw(reset_ops_db, monkeypatch):
    _auth(monkeypatch)
    _seed()
    headers = {"X-API-Key": TEST_DSR_KEY}
    with TestClient(app) as client:
        resp = client.get(f"/api/interactions/{IID}?scrub_pii=false", headers=headers)
        assert resp.status_code == 200
        assert SECRET in resp.text
