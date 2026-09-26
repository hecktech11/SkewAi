"""Tests covering the Top 5 Production Review Priorities:

1. Alternate PII read paths (timeline scrub_pii, console_ws authorization & redaction)
2. Atomic key revocation (SubjectKeyStore thread safety & revocation state)
3. Prevent writes after erasure (Orchestrator _record_state / _finalize_interaction erasure barrier)
4. Include archived exports in erasure (purge_audit_archives in delete_interaction & tombstone_interaction)
5. Repair fresh-volume startup (schema convergence before dependent index creation)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

import duckdb
import pytest
from fastapi.testclient import TestClient

from src.agents.orchestrator import Orchestrator, OrchestratorHooks
from src.domains.loader import load_pack
from src.api.main import app
from src.api.rbac import issue_session, session_cookie_name
from src.api.routes.interactions import _broadcast_console, activity_frame_from_row
from src.data.warehouse import apply_ops_schema, init_ops_db, ops_con
from src.frontline.archive import build_audit_archive, purge_audit_archives
from src.frontline.dsr import (
    delete_interaction,
    is_interaction_erased,
    tombstone_interaction,
)
from src.ledger.writer import AgentAction, record_action
from src.security.pii import SubjectKeyStore, encrypt_subject_pii


# ── Priority #1: Close alternate PII read paths ───────────────────────────────


def test_enterprise_timeline_redaction_and_authorization(reset_ops_db, monkeypatch):
    """Timeline scrubs PII by default; scrub_pii=False requires dsr:export."""
    service_key = "test-service-key-32-chars-long!!"
    session_secret = "test-session-secret-32-chars-long!"
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", service_key)
    monkeypatch.setenv("SESSION_SECRET", session_secret)
    monkeypatch.setenv("FRONTLINE_BOOTSTRAP_ADMIN", "1")
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")

    iid = "int_rev_pii_timeline_1"
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions (interaction_id, pack_id, pack_version, started_at, channel, status)
            VALUES (?, 'automotive_nhtsa', '1.0', ?, 'web_text', 'active')
            """,
            [iid, now],
        )
        from src.data.turns import persist_turn

        persist_turn(
            iid,
            {
                "turn_id": "turn_rev_1",
                "seq": 1,
                "speaker": "customer",
                "text": "My email is sensitive.user@example.com and phone is 555-123-4567.",
                "ts": now,
            },
        )

    admin_sess = issue_session("admin_1", "admin")["token"]
    auditor_sess = issue_session("auditor_1", "auditor", issuer_role="admin")["token"]
    service_headers = {"X-API-Key": service_key}

    with TestClient(app) as client:
        # Default with auditor session (has ledger:read, has dsr:export)
        client.cookies.set(session_cookie_name(), auditor_sess)
        res = client.get(f"/api/frontline/enterprise/timeline/{iid}", headers={"X-API-Key": service_key})
        assert res.status_code == 200
        data = res.json()
        turn_event = next(e for e in data["events"] if e.get("kind") == "turn")
        turn_text = turn_event["detail"]["text"]
        assert "sensitive.user@example.com" not in turn_text
        assert "[EMAIL]" in turn_text
        assert "555-123-4567" not in turn_text
        assert "[PHONE]" in turn_text

        # Explicit scrub_pii=false with auditor session (has dsr:export) -> reveals plaintext
        res_raw = client.get(f"/api/frontline/enterprise/timeline/{iid}?scrub_pii=false", headers={"X-API-Key": service_key})
        assert res_raw.status_code == 200
        turn_raw = next(e for e in res_raw.json()["events"] if e.get("kind") == "turn")
        assert "sensitive.user@example.com" in turn_raw["detail"]["text"]

        # Explicit scrub_pii=false with service session (lacks dsr:export) -> 403 Forbidden
        client.cookies.clear()
        res_forbidden = client.get(f"/api/frontline/enterprise/timeline/{iid}?scrub_pii=false", headers=service_headers)
        assert res_forbidden.status_code == 403
        assert "dsr:export" in res_forbidden.json()["detail"]

        # Default with service role (has ledger:read, lacks dsr:export) -> 200 OK, redacted
        res_service_default = client.get(f"/api/frontline/enterprise/timeline/{iid}", headers=service_headers)
        assert res_service_default.status_code == 200
        turn_serv = next(e for e in res_service_default.json()["events"] if e.get("kind") == "turn")
        assert "sensitive.user@example.com" not in turn_serv["detail"]["text"]
        assert "[EMAIL]" in turn_serv["detail"]["text"]


def test_console_ws_enforces_read_perm_and_redacts(reset_ops_db, monkeypatch):
    """Console WS gates subscription on read perm and redacts replay/broadcast frames."""
    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")

    iid = "int_rev_ws_1"
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions (interaction_id, pack_id, pack_version, started_at, channel, status)
            VALUES (?, 'automotive_nhtsa', '1.0', ?, 'web_text', 'active')
            """,
            [iid, now],
        )

    record_action(
        AgentAction(
            interaction_id=iid,
            agent="orchestrator",
            action_type="state_transition",
            input_summary="customer email is contact@example.com",
            output_summary="phone is 555-987-6543 for contact@example.com",
            ok=True,
        )
    )

    agent_sess = issue_session("agent_sub", "agent")["token"]

    with TestClient(app) as client:
        client.cookies.set(session_cookie_name(), agent_sess)
        with client.websocket_connect("/ws/console") as ws:
            frames: list[dict] = []
            for _ in range(5):
                try:
                    frames.append(ws.receive_json())
                except Exception:
                    break
                if any(f.get("interaction_id") == iid for f in frames):
                    break

            act = next((f for f in frames if f.get("interaction_id") == iid), None)
            assert act is not None
            assert "contact@example.com" not in act.get("output_summary", "")
            assert "[EMAIL]" in act.get("output_summary", "")
            assert "[PHONE]" in act.get("output_summary", "")


# ── Priority #2: Make key revocation atomic ───────────────────────────────────


def test_subject_keystore_atomic_concurrency(reset_ops_db):
    """Concurrent lookup vs shred_dek must never leak a shredded key back into cache."""
    SubjectKeyStore.clear()
    subject_id = "subj_atomic_test_99"

    key = SubjectKeyStore.get_or_create_dek(subject_id)
    assert len(key) == 32
    assert SubjectKeyStore.has_dek(subject_id)

    errors: list[Exception] = []
    shredded_barrier = threading.Barrier(4)
    stop_event = threading.Event()

    def reader_loop():
        shredded_barrier.wait()
        while not stop_event.is_set():
            try:
                SubjectKeyStore.get_or_create_dek(subject_id)
            except KeyError:
                pass
            except Exception as e:
                errors.append(e)

    def shredder_thread():
        shredded_barrier.wait()
        time.sleep(0.01)
        res = SubjectKeyStore.shred_dek(subject_id)
        assert res is True
        stop_event.set()

    threads = [
        threading.Thread(target=reader_loop),
        threading.Thread(target=reader_loop),
        threading.Thread(target=reader_loop),
        threading.Thread(target=shredder_thread),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not errors, f"Unexpected reader errors: {errors}"
    assert not SubjectKeyStore.has_dek(subject_id)
    assert subject_id not in SubjectKeyStore._keys
    with pytest.raises(KeyError, match="shredded"):
        SubjectKeyStore.get_or_create_dek(subject_id)


# ── Priority #3: Prevent writes after erasure ─────────────────────────────────


def test_prevent_writes_after_erasure(reset_ops_db, seed_automotive_pack):
    """Hangup and close after erasure barrier must never restore tombstoned text."""
    iid = "int_erasure_barrier_test"
    pack = load_pack("automotive_nhtsa")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions (
                interaction_id, pack_id, pack_version, started_at, channel,
                status, entity_1, category, description, supervised, llm_calls
            )
            VALUES (?, 'automotive_nhtsa', '1.0', ?, 'web_text', 'active', 'Ford', 'brakes', 'Customer secret symptom', FALSE, 0)
            """,
            [iid, now],
        )

    orch = Orchestrator(iid, pack, channel="web_text", hooks=OrchestratorHooks())
    orch.ctx.slots = {
        "entity_1": "Ford",
        "category": "brakes",
        "description": "Customer secret symptom",
    }

    res = tombstone_interaction(iid)
    assert res["ok"]
    assert is_interaction_erased(iid)

    with ops_con(read_only=True) as con:
        row = con.execute(
            "SELECT description, entity_1, erased FROM interactions WHERE interaction_id = ?",
            [iid],
        ).fetchone()
        assert row is not None
        assert row[0] is None or row[0] == ""
        assert row[1] is None
        assert row[2] is True

    asyncio.run(orch.hangup())

    with ops_con(read_only=True) as con:
        row_after = con.execute(
            "SELECT description, entity_1, erased FROM interactions WHERE interaction_id = ?",
            [iid],
        ).fetchone()
        assert row_after is not None
        assert row_after[0] is None or row_after[0] == ""
        assert row_after[1] is None
        assert row_after[2] is True

    assert not orch.ctx.slots.get("description")


# ── Priority #4: Include archived exports in erasure ──────────────────────────


def test_archived_exports_purged_on_erasure(reset_ops_db, tmp_path):
    """Archived export bundles and manifests are purged on DSR erasure."""
    iid = "int_archive_purge_test"
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions (interaction_id, pack_id, pack_version, started_at, channel, status)
            VALUES (?, 'automotive_nhtsa', '1.0', ?, 'web_text', 'completed')
            """,
            [iid, now],
        )

    record_action(
        AgentAction(
            interaction_id=iid,
            agent="orchestrator",
            action_type="state_transition",
            output_summary="action for archive test",
            ok=True,
        )
    )

    out_dir = tmp_path / "archives"
    archive = build_audit_archive(iid, out_dir=out_dir)
    bundle_path = Path(archive["bundle_path"])
    manifest_path = Path(archive["manifest_path"])

    assert bundle_path.is_file(), "Archive bundle must exist before erasure"
    assert manifest_path.is_file(), "Manifest must exist before erasure"

    del_res = delete_interaction(iid, mode="erase")
    assert del_res["ok"]

    assert not bundle_path.exists(), "Archive bundle must be purged on erasure"
    assert not manifest_path.exists(), "Manifest must be purged on erasure"


# ── Priority #5: Repair fresh-volume startup ──────────────────────────────────


def test_fresh_volume_schema_convergence(tmp_path):
    """Fresh volume with 001_ops_init skeleton converges cleanly with entity_1 and indexes."""
    db_file = tmp_path / "fresh_ops.duckdb"
    con = duckdb.connect(str(db_file))

    migration_001 = Path(__file__).resolve().parents[2] / "migrations" / "001_ops_init.sql"
    con.execute(migration_001.read_text(encoding="utf-8"))

    apply_ops_schema(con)

    con.execute(
        """
        INSERT INTO interactions (
            interaction_id, pack_id, pack_version, started_at, channel,
            status, entity_1, entity_2, entity_3, category, description, erased
        )
        VALUES ('int_fresh_1', 'automotive_nhtsa', '1.0', CURRENT_TIMESTAMP, 'web_text',
                'active', 'Chevy', 'Bolt', '2022', 'battery', 'charge fault', FALSE)
        """
    )
    row = con.execute(
        "SELECT entity_1, entity_2, entity_3, erased FROM interactions WHERE interaction_id = 'int_fresh_1'"
    ).fetchone()
    assert row == ("Chevy", "Bolt", "2022", False)

    idx_row = con.execute(
        "SELECT index_name FROM duckdb_indexes() WHERE index_name = 'idx_interactions_entities'"
    ).fetchone()
    assert idx_row is not None
    con.close()
