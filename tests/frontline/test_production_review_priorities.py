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


# ── N05: Learning review requires approval:decide and uses verified actor ─────


def test_learning_review_requires_approval_perm(reset_ops_db, monkeypatch):
    """POST learning review without approval:decide should be rejected."""
    from src.api.main import app

    service_key = "test-service-key-32-chars-long!!"
    session_secret = "test-session-secret-32-chars-long!"
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", service_key)
    monkeypatch.setenv("SESSION_SECRET", session_secret)
    monkeypatch.setenv("FRONTLINE_BOOTSTRAP_ADMIN", "1")
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")

    # Agent role does NOT have approval:decide
    agent_sess = issue_session("review_agent", "agent")["token"]

    with TestClient(app) as client:
        resp = client.post(
            "/api/v3/learning/proposals/fake-id/review",
            json={"status": "approved"},
            headers={"X-API-Key": service_key},
            cookies={session_cookie_name(): agent_sess},
        )
        # Should be 403 — agent doesn't have approval:decide
        assert resp.status_code == 403


def test_learning_review_uses_verified_actor(reset_ops_db, monkeypatch):
    """learning_review should use the authenticated actor, not body.reviewed_by."""
    from unittest.mock import patch

    from src.api.main import app

    service_key = "test-service-key-32-chars-long!!"
    session_secret = "test-session-secret-32-chars-long!"
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", service_key)
    monkeypatch.setenv("SESSION_SECRET", session_secret)
    monkeypatch.setenv("FRONTLINE_BOOTSTRAP_ADMIN", "1")
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")

    # Supervisor has approval:decide
    sup_sess = issue_session("real_reviewer", "supervisor", issuer_role="admin")["token"]
    captured_args: dict = {}

    def mock_review_proposal(proposal_id, *, status, reviewed_by, review_note):
        captured_args["reviewed_by"] = reviewed_by
        return {"proposal_id": proposal_id, "status": status, "reviewed_by": reviewed_by}

    with patch("src.v3.learning.review_proposal", side_effect=mock_review_proposal):
        with TestClient(app) as client:
            resp = client.post(
                "/api/v3/learning/proposals/p-1/review",
                json={"status": "approved", "reviewed_by": "impersonator"},
                headers={"X-API-Key": service_key},
                cookies={session_cookie_name(): sup_sess},
            )
            assert resp.status_code == 200
            # The reviewed_by should be the authenticated actor, NOT "impersonator"
            assert captured_args["reviewed_by"] == "real_reviewer"


# ── N07: Mutable prefix swap guard ───────────────────────────────────────────


def test_legacy_xor_rejects_prefix_downgrade_in_live_reads(reset_ops_db):
    """Live reads accept enc:v1: only; any token with enc:x1: prefix (even short 28-char) is rejected (N07)."""
    import base64
    import secrets

    from src.security.pii import (
        LEGACY_XOR_PREFIX,
        PiiIntegrityError,
        decrypt_subject_pii,
    )

    SubjectKeyStore.clear()
    subject = "subj_xor_prefix_downgrade"
    SubjectKeyStore.get_or_create_dek(subject)

    # Even a short payload (e.g. 16 bytes / 28 chars) is rejected in live reads
    nonce = secrets.token_bytes(12)
    fake_cipher = secrets.token_bytes(16)
    payload = nonce + fake_cipher
    fake_token = LEGACY_XOR_PREFIX + base64.urlsafe_b64encode(payload).decode("ascii")

    with pytest.raises(PiiIntegrityError, match="not permitted in live reads"):
        decrypt_subject_pii(subject, fake_token)


# ── N08: Legacy XOR token migration script ───────────────────────────────────


def test_migrate_legacy_xor_tokens_script(reset_ops_db):
    """migrate_legacy_xor_tokens.py finds enc:v1: tokens failing AES, converts to enc:x1:, and re-encrypts to AES-GCM."""
    import base64
    import hashlib
    import secrets
    from scripts.migrate_legacy_xor_tokens import migrate
    from src.security.pii import TOKEN_PREFIX, SubjectKeyStore, decrypt_subject_pii

    SubjectKeyStore.clear()
    iid = "int_legacy_mig_1"
    turn_id = "turn_legacy_mig_1"
    dek = SubjectKeyStore.get_or_create_dek(iid)

    # Encode plaintext using legacy XOR under enc:v1:
    plain_text = "legacy.user@example.com"
    nonce = secrets.token_bytes(12)
    plain_bytes = plain_text.encode("utf-8")
    stream_key = hashlib.blake2b(dek, key=nonce, digest_size=len(plain_bytes)).digest()
    cipher_bytes = bytes(b ^ k for b, k in zip(plain_bytes, stream_key))
    raw = nonce + cipher_bytes
    legacy_enc_v1 = TOKEN_PREFIX + base64.urlsafe_b64encode(raw).decode("ascii")

    # Insert into interaction_turns
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions (interaction_id, pack_id, pack_version, started_at, channel, status)
            VALUES (?, 'automotive_nhtsa', '1.0', CURRENT_TIMESTAMP, 'web_text', 'active')
            """,
            [iid],
        )
        con.execute(
            """
            INSERT INTO interaction_turns (turn_id, interaction_id, seq, speaker, text, ts)
            VALUES (?, ?, 1, 'customer', ?, CURRENT_TIMESTAMP)
            """,
            [turn_id, iid, legacy_enc_v1],
        )

    # Run migration with explicit provenance
    stats = migrate(dry_run=False, proven_legacy_turn_ids=[turn_id])
    assert stats["legacy"] >= 1
    assert stats["migrated"] >= 1
    assert stats["errors"] == 0

    # Fetch updated token
    with ops_con() as con:
        row = con.execute("SELECT text FROM interaction_turns WHERE turn_id = ?", [turn_id]).fetchone()
    assert row is not None
    migrated_token = row[0]
    assert migrated_token.startswith(TOKEN_PREFIX)
    # Ensure it now decrypts cleanly via AES-GCM
    decrypted = decrypt_subject_pii(iid, migrated_token)
    assert decrypted == plain_text


# ── N09: DSR delete logs failure when result is not ok ────────────────────────


def test_dsr_delete_logs_failure_for_not_ok_result(reset_ops_db, monkeypatch):
    """When delete_interaction returns ok=False, the API should log failure and return 500."""
    from unittest.mock import patch

    from src.api.main import app

    service_key = "test-service-key-32-chars-long!!"
    session_secret = "test-session-secret-32-chars-long!"
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", service_key)
    monkeypatch.setenv("SESSION_SECRET", session_secret)
    monkeypatch.setenv("FRONTLINE_BOOTSTRAP_ADMIN", "1")
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")

    admin_sess = issue_session("dsr_admin", "admin")["token"]
    logged_events: list[dict] = []

    def mock_delete(interaction_id, mode="tombstone"):
        return {"ok": False, "reason": "row not found"}

    def mock_security_event(event, *, outcome="", **kwargs):
        logged_events.append({"event": event, "outcome": outcome, **kwargs})

    with (
        patch("src.frontline.dsr.delete_interaction", side_effect=mock_delete),
        patch("src.security.audit_log.security_event", side_effect=mock_security_event),
    ):
        with TestClient(app) as client:
            resp = client.delete(
                "/api/frontline/dsr/int_fail_1",
                params={"mode": "tombstone"},
                headers={"X-API-Key": service_key},
                cookies={session_cookie_name(): admin_sess},
            )
            # Should be 500 because the deletion wasn't successful
            assert resp.status_code == 500
            # The security_event should have been called with outcome="failure"
            assert any(e["outcome"] == "failure" for e in logged_events), (
                f"Expected a failure log, got: {logged_events}"
            )


# ── N10: Bootstrap seed decision uses populated flag ─────────────────────────


def test_bootstrap_seeds_when_file_exists_but_empty():
    """bootstrap_actions should trigger seed even when file exists, if it's empty."""
    from src.data.bootstrap import bootstrap_actions

    # File exists but is not populated (empty DB from migrations)
    actions = bootstrap_actions(
        ops_exists=True,
        automotive_exists=True,
        finance_exists=True,
        automotive_populated=False,
        finance_populated=False,
        seed_demo=True,
    )
    assert "seed-automotive" in actions
    assert "seed-finance" in actions


def test_bootstrap_skips_seed_when_populated():
    """bootstrap_actions should NOT seed when the DB has data."""
    from src.data.bootstrap import bootstrap_actions

    actions = bootstrap_actions(
        ops_exists=True,
        automotive_exists=True,
        finance_exists=True,
        automotive_populated=True,
        finance_populated=True,
        seed_demo=True,
    )
    assert "seed-automotive" not in actions
    assert "seed-finance" not in actions
    assert "init-automotive" in actions
    assert "init-finance" in actions


def test_bootstrap_backward_compat_no_populated_flag():
    """When populated flags are not supplied, falls back to exists behavior."""
    from src.data.bootstrap import bootstrap_actions

    actions = bootstrap_actions(
        ops_exists=True,
        automotive_exists=False,
        finance_exists=False,
        seed_demo=True,
    )
    assert "seed-automotive" in actions
    assert "seed-finance" in actions


# ── N11: Job submission security log includes actor ──────────────────────────


def test_jobs_enqueue_security_event_includes_actor(reset_ops_db, monkeypatch):
    """security_event for jobs.enqueue must include actor= from the authenticated principal."""
    from unittest.mock import patch

    from src.api.main import app

    service_key = "test-service-key-32-chars-long!!"
    session_secret = "test-session-secret-32-chars-long!"
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_API_KEY", service_key)
    monkeypatch.setenv("SESSION_SECRET", session_secret)
    monkeypatch.setenv("FRONTLINE_BOOTSTRAP_ADMIN", "1")
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")

    admin_sess = issue_session("job_submitter", "admin")["token"]
    logged_events: list[dict] = []

    def mock_enqueue(job_type, payload, *, role, principal):
        return {"job_id": "j-1234", "status": "queued"}

    def mock_security_event(event, *, outcome="", **kwargs):
        logged_events.append({"event": event, "outcome": outcome, **kwargs})

    with (
        patch("src.jobs.queue.enqueue", side_effect=mock_enqueue),
        patch("src.security.audit_log.security_event", side_effect=mock_security_event),
    ):
        with TestClient(app) as client:
            resp = client.post(
                "/api/frontline/jobs",
                json={"job_type": "test-job", "payload": {}},
                headers={"X-API-Key": service_key},
                cookies={session_cookie_name(): admin_sess},
            )
            assert resp.status_code == 200
            # Check that security_event was called with actor=
            evt = next((e for e in logged_events if e["event"] == "jobs.enqueue"), None)
            assert evt is not None, f"No jobs.enqueue event logged: {logged_events}"
            assert evt.get("actor") == "job_submitter", (
                f"Expected actor='job_submitter', got actor={evt.get('actor')!r}"
            )


# ── R16 / R07: Reject PostgreSQL mode as unsupported ─────────────────────────


def test_postgres_mode_rejected_as_unsupported(monkeypatch):
    """Setting FRONTLINE_OPS_DSN must raise RuntimeError rejecting PostgreSQL as unsupported in all environments."""
    monkeypatch.setenv("FRONTLINE_OPS_DSN", "postgresql://user:pass@localhost:5432/db")
    from src.data.postgres_backend import ops_connection
    from src.data.warehouse import ops_con

    with pytest.raises(RuntimeError, match="unsupported in this release"):
        with ops_con():
            pass

    with pytest.raises(RuntimeError, match="unsupported in this release"):
        with ops_connection():
            pass


# ── R15: All eligible records assigned to clusters ───────────────────────────


def test_clustering_assigns_all_eligible_records(monkeypatch, tmp_path):
    """rebuild_clusters assigns all eligible records to cluster_assignments, even if distant (>0.85)."""
    import datetime
    from src.data.warehouse import apply_domain_schema, domain_con
    from src.ml_runtime.clustering import rebuild_clusters
    from src.ml_runtime.embeddings import embedding_dim

    pack = "test_r15_pack"
    db_file = tmp_path / f"{pack}.duckdb"
    monkeypatch.setenv(f"FRONTLINE_{pack.upper()}_DB", str(db_file))
    dim = embedding_dim()

    with domain_con(pack, read_only=False) as con:
        apply_domain_schema(con)
        base_time = datetime.datetime(2025, 1, 1, 12, 0, 0)
        # 10 records: 5 close to each other, 5 with orthogonal/distinct vectors
        rows = []
        for i in range(10):
            v = [0.1] * dim if i < 5 else [0.9 if j == (i % dim) else 0.0 for j in range(dim)]
            rows.append(
                (
                    f"rec_{i}",
                    f"Complaint about brake issue {i}",
                    "BRAKES",
                    "Acme",
                    "Sedan",
                    v,
                    (base_time + datetime.timedelta(minutes=i)).isoformat(),
                )
            )
        con.executemany(
            """
            INSERT INTO records (record_id, text, category, entity_2, entity_3, embedding, received_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    res = rebuild_clusters(pack, k=2)
    assert res["clusters"] >= 1
    with domain_con(pack, read_only=True) as con:
        assigned_count = con.execute("SELECT COUNT(*) FROM cluster_assignments").fetchone()[0]
        assert assigned_count == 10, f"Expected all 10 records to be assigned, got {assigned_count}"


# ── R12: Pack IDF persistence across process restarts ────────────────────────


def test_pack_idf_persistence_across_restarts(tmp_path, monkeypatch):
    """fit_idf persists weights to disk, reloaded across process memory clears."""
    from src.ml_runtime import embeddings

    pack_id = "test_r12_pack"
    embeddings.reset_idf(pack_id)

    texts = [
        "brake fluid leakage in rear caliper",
        "brake pedal feels spongy and soft",
        "transmission slipping between gears",
    ]
    fitted = embeddings.fit_idf(texts, pack_id=pack_id)
    assert len(fitted) > 0
    dig_before = embeddings.idf_digest(pack_id)
    assert dig_before != "uniform"

    # Simulate process restart by clearing in-memory dicts
    embeddings._PACK_IDF.pop(pack_id, None)
    embeddings._PACK_IDF_DOCS.pop(pack_id, None)
    assert pack_id not in embeddings._PACK_IDF

    # idf_digest and embed_text should reload from disk
    dig_after = embeddings.idf_digest(pack_id)
    assert dig_after == dig_before
    assert pack_id in embeddings._PACK_IDF

    status = embeddings.idf_status(pack_id)
    assert status["features"] == len(fitted)
    assert status["docs"] == 3

    # reset_idf should clean up the file
    embeddings.reset_idf(pack_id)
    assert embeddings.idf_digest(pack_id) == "uniform"


# ── R21: Grouped SQL population index and caching ────────────────────────────


def test_population_index_sql_grouping_cache(tmp_path):
    """get_pack_population_index uses SQL GROUP BY and caches result."""
    import duckdb
    from src.ml_runtime.association import _POPULATION_CACHE, get_pack_population_index

    con = duckdb.connect(":memory:")
    con.execute(
        """
        CREATE TABLE records (
            category VARCHAR,
            entity_2 VARCHAR
        )
        """
    )
    con.executemany(
        "INSERT INTO records VALUES (?, ?)",
        [
            ("BRAKES", "Civic"),
            ("BRAKES", "Civic"),
            ("BRAKES", "Accord"),
            ("ENGINE", "Civic"),
        ],
    )

    pack_id = "test_pop_cache_pack"
    _POPULATION_CACHE.pop(pack_id, None)

    idx1 = get_pack_population_index(con, pack_id=pack_id)
    assert idx1.n == 4
    assert idx1.n_cat.get("BRAKES") == 3
    assert idx1.n_cat.get("ENGINE") == 1
    assert idx1.n_ent.get("CIVIC") == 3
    assert idx1.n_ent.get("ACCORD") == 1
    assert idx1.n_both.get(("BRAKES", "CIVIC")) == 2

    assert pack_id in _POPULATION_CACHE

    # Second call returns cached index
    idx2 = get_pack_population_index(con, pack_id=pack_id)
    assert idx2 is idx1


# ── R22: Hard timeout enforcement on category model embedding ─────────────────


def test_category_zeroshot_hard_timeout_enforcement():
    """predict_category_zero_shot times out when embedding takes longer than timeout_ms."""
    import time
    from src.ml_runtime.category_zeroshot import predict_category_zero_shot

    class SlowEmbedder:
        def embed(self, text: str):
            time.sleep(0.1)  # 100ms
            raise RuntimeError("Should have timed out")

    centroids = {
        "brakes": [0.1] * 384,
        "engine": [0.2] * 384,
    }

    t0 = time.monotonic()
    res = predict_category_zero_shot(
        "test text",
        centroids,
        embedder=SlowEmbedder(),
        timeout_ms=10.0,  # 10ms budget
    )
    elapsed = time.monotonic() - t0
    # Should return cleanly without raising, with category=None and within bounded time
    assert res.category is None
    assert elapsed < 0.2
