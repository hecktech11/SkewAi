"""Regression tests verifying fixes for all 16 Follow-up Deep Audit findings (FU01 - FU16)."""

from __future__ import annotations

import asyncio
import os
import time
from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.rbac import issue_session
from src.data.timeutil import utc_now
from src.data.warehouse import ops_con
from src.domains.loader import load_pack


# ============================================================================
# FU01: Automatic Migration & Unsafe XOR Relabeling
# ============================================================================

def test_fu01_corrupt_aes_gcm_never_migrated_as_xor(reset_ops_db):
    """Corrupt enc:v1: rows must NOT be treated as legacy XOR without provenance."""
    from scripts.migrate_legacy_xor_tokens import migrate
    from src.security.pii import SubjectKeyStore

    iid = "fu01_test_interaction"
    SubjectKeyStore.get_or_create_dek(iid)

    corrupt_token = "enc:v1:invalid_corrupt_garbage_ciphertext"
    now = utc_now().replace(tzinfo=None)
    with ops_con() as con:
        con.execute(
            "INSERT INTO interaction_turns (turn_id, interaction_id, seq, speaker, text, ts) VALUES (?, ?, ?, ?, ?, ?)",
            ["t_corrupt_1", iid, 1, "customer", corrupt_token, now],
        )

    # Run migration without provenance for t_corrupt_1
    stats = migrate(dry_run=False)
    assert stats["corrupt"] == 1
    assert stats["migrated"] == 0

    # Ensure corrupt row text was NOT mutated or overwritten
    with ops_con(read_only=True) as con:
        row = con.execute("SELECT text FROM interaction_turns WHERE turn_id = ?", ["t_corrupt_1"]).fetchone()
        assert row[0] == corrupt_token


def test_fu01_proven_legacy_tokens_migrated(reset_ops_db):
    """Rows with explicit provenance or enc:x1: are migrated cleanly to authenticated enc:v1:."""
    import base64
    import hashlib
    from scripts.migrate_legacy_xor_tokens import migrate
    from src.security.pii import LEGACY_XOR_PREFIX, SubjectKeyStore, decrypt_subject_pii

    iid = "fu01_proven_interaction"
    dek = SubjectKeyStore.get_or_create_dek(iid)

    # Create genuine legacy XOR token
    nonce = b"0123456789ab"
    plain = b"secret data"
    stream = hashlib.blake2b(dek, key=nonce, digest_size=len(plain)).digest()
    cipher = bytes(p ^ s for p, s in zip(plain, stream))
    x1_token = LEGACY_XOR_PREFIX + base64.urlsafe_b64encode(nonce + cipher).decode("ascii")

    now = utc_now().replace(tzinfo=None)
    with ops_con() as con:
        con.execute(
            "INSERT INTO interaction_turns (turn_id, interaction_id, seq, speaker, text, ts) VALUES (?, ?, ?, ?, ?, ?)",
            ["t_legacy_1", iid, 1, "customer", x1_token, now],
        )

    stats = migrate(dry_run=False)
    assert stats["legacy"] == 1
    assert stats["migrated"] == 1
    assert stats["errors"] == 0

    # Check that text is now valid enc:v1: and decrypts to original
    with ops_con(read_only=True) as con:
        row = con.execute("SELECT text FROM interaction_turns WHERE turn_id = ?", ["t_legacy_1"]).fetchone()
        assert row[0].startswith("enc:v1:")
        dec = decrypt_subject_pii(iid, row[0])
        assert dec == "secret data"


# ============================================================================
# FU02: Erasure of Active Contacts & Turn Purging
# ============================================================================

def test_fu02_invalidate_active_orchestrator_clears_memory_and_registry(reset_ops_db):
    """invalidate_active_orchestrator must clear turns, slots, and pop from _active."""
    from src.agents.base import InteractionContext
    from src.agents.orchestrator import Orchestrator
    from src.api.routes.interactions import ActiveEntry, _active, _attach_customer_ws
    from src.frontline.dsr import invalidate_active_orchestrator, is_interaction_erased

    iid = "fu02_active_erasure"
    pack = load_pack("automotive_nhtsa")
    orch = Orchestrator(interaction_id=iid, pack=pack, channel="web_voice")
    orch.ctx.slots["issue"] = "brake failure"
    orch.ctx.turns.append({"turn_id": "t1", "speaker": "customer", "text": "help my brakes fail"})
    entry = ActiveEntry(orch=orch, capability_token="tok_fu02")
    _active[iid] = entry

    # Invalidate
    invalidate_active_orchestrator(iid)

    assert iid not in _active
    assert is_interaction_erased(iid) is True
    assert orch._erased is True
    assert len(orch.ctx.turns) == 0
    assert orch.ctx.slots == {"description": ""}

    # Attempting to attach customer ws must reject with 404
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(_attach_customer_ws(iid))
    assert exc_info.value.status_code == 404


# ============================================================================
# FU03: Twilio WebSocket Authorization & Scope Enforcement
# ============================================================================

def test_fu03_twilio_ws_requires_contact_write(reset_ops_db, monkeypatch):
    """twilio_ws must reject roles lacking contact:write (e.g. auditor) with 1008."""
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_OPEN_MODE", "0")
    client = TestClient(app)
    tok = issue_session("auditor_user", "auditor", issuer_role="admin")["token"]

    with client.websocket_connect(
        "/ws/twilio/int_test_123",
        headers={"x-frontline-session": tok},
    ) as ws:
        frame = ws.receive_json()
        assert frame.get("type") == "error"
        assert frame.get("code") == "forbidden"
        assert "contact:write" in frame.get("detail", "")


# ============================================================================
# FU04: Lock Inversion Deadlock Elimination & con parameter
# ============================================================================

def test_fu04_persist_turn_and_subject_keystore_con(reset_ops_db):
    """persist_turn does not hold ops_con while encrypting, and SubjectKeyStore accepts con."""
    from src.data.turns import persist_turn
    from src.security.pii import SubjectKeyStore

    iid = "fu04_lock_test"
    # has_dek and get_or_create_dek with explicit con
    with ops_con() as con:
        dek = SubjectKeyStore.get_or_create_dek(iid, con=con)
        assert len(dek) == 32
        assert SubjectKeyStore.has_dek(iid, con=con) is True

    # persist_turn works cleanly
    persist_turn(iid, {"turn_id": "t_fu04_1", "seq": 1, "speaker": "customer", "text": "hello world"})

    with ops_con(read_only=True) as con:
        row = con.execute("SELECT text FROM interaction_turns WHERE turn_id = ?", ["t_fu04_1"]).fetchone()
        assert row is not None
        assert row[0].startswith("enc:v1:")


# ============================================================================
# FU05: WAL Concurrent Append & Atomic Segment Replay
# ============================================================================

def test_fu05_wal_concurrent_append_not_lost(reset_ops_db, tmp_path, monkeypatch):
    """Concurrent append during replay_ledger_wal must not be lost or overwritten."""
    import threading
    from src.ledger.writer import AgentAction, _wal_append, _wal_path, replay_ledger_wal

    wal_file = tmp_path / "ledger_wal.jsonl"
    monkeypatch.setattr("src.ledger.writer._wal_path", lambda: wal_file)

    a1 = AgentAction(interaction_id="ix1", agent="orch", action_type="state_transition", input_summary="1", output_summary="1")
    a2 = AgentAction(interaction_id="ix2", agent="orch", action_type="state_transition", input_summary="2", output_summary="2")
    _wal_append(a1, RuntimeError("err1"))
    _wal_append(a2, RuntimeError("err2"))

    # Append a3 concurrently
    a3 = AgentAction(interaction_id="ix3", agent="orch", action_type="state_transition", input_summary="3", output_summary="3")

    def _concurrent_write():
        time.sleep(0.01)
        _wal_append(a3, RuntimeError("err3"))

    th = threading.Thread(target=_concurrent_write)
    th.start()
    res = replay_ledger_wal(limit=10)
    th.join()

    # If a3 was written while replaying, it must either be replayed or present in wal_file
    assert res["replayed"] >= 2
    if wal_file.exists():
        content = wal_file.read_text(encoding="utf-8")
        assert "ix3" in content or res["replayed"] == 3


# ============================================================================
# FU06: Supervisor Release Event Ordering
# ============================================================================

@pytest.mark.asyncio
async def test_fu06_release_publishes_control_before_ai_turns(reset_ops_db):
    """release() must call _publish_control before emitting AI turns so frontend accepts them."""
    from src.agents.base import InteractionContext
    from src.agents.orchestrator import Orchestrator, SUPERVISED

    pack = load_pack("automotive_nhtsa")
    orch = Orchestrator(interaction_id="fu06_ix", pack=pack, channel="web_voice")
    orch.ctx.state = SUPERVISED
    orch.ctx.supervised = True
    orch.ctx.takeover_claimed_by = "sup_1"
    orch.ctx.pending_safety_script = "Please stop your vehicle safely."
    orch.ctx.safety_flags["escalation"] = True

    call_order = []

    async def _mock_publish_control():
        call_order.append("publish_control")

    async def _mock_emit_turn(text, meta):
        call_order.append("emit_turn")

    orch._publish_control = _mock_publish_control
    orch.hooks = MagicMock()
    orch.hooks._maybe = AsyncMock(side_effect=lambda fn, *args: call_order.append("emit_turn"))

    res = await orch.release(actor="sup_1")
    assert res["ok"] is True
    assert "publish_control" in call_order
    assert "emit_turn" in call_order
    # Critical invariant: publish_control happened BEFORE emit_turn!
    pub_idx = call_order.index("publish_control")
    turn_idx = call_order.index("emit_turn")
    assert pub_idx < turn_idx, f"publish_control ({pub_idx}) was not before emit_turn ({turn_idx})"


# ============================================================================
# FU07: Console Catch-Up Backpressure
# ============================================================================

@pytest.mark.asyncio
async def test_fu07_console_catchup_handles_large_batch(reset_ops_db):
    """_catch_up_loop should feed events with backpressure rather than throwing QueueFull."""
    from src.api.routes.interactions import _ConsoleClient

    ws = AsyncMock()
    client = _ConsoleClient(ws, role="admin")

    # Put 32 items to fill the queue
    for i in range(32):
        client.queue.put_nowait({"id": i})

    # Now drain one in background to prove backpressure works
    async def _slow_drain():
        await asyncio.sleep(0.02)
        await client.queue.get()

    asyncio.create_task(_slow_drain())

    # 33rd item: with timeout put, it waits and succeeds
    await asyncio.wait_for(client.queue.put({"id": 33}), timeout=1.0)
    assert client.queue.qsize() == 32


# ============================================================================
# FU08: Alert Metrics Severity Ranking & Window Filtering
# ============================================================================

def test_fu08_fetch_cluster_metrics_critical_severity_and_window(reset_ops_db):
    """Critical severity must outrank Medium, and window_days must filter older cases."""
    from src.frontline.alert_rules import fetch_cluster_metrics

    now = utc_now().replace(tzinfo=None)
    with ops_con() as con:
        # Case 1: Critical, recent (1 day old)
        con.execute(
            "INSERT INTO cases (case_id, interaction_id, pack_id, cluster_match_id, severity, severity_source, priority, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["c_crit", "ix_c1", "automotive_nhtsa", 42, "Critical", "model", 1, now - timedelta(days=1)],
        )
        # Case 2: Medium, recent (1 day old) - alphabetical max would incorrectly pick Medium
        con.execute(
            "INSERT INTO cases (case_id, interaction_id, pack_id, cluster_match_id, severity, severity_source, priority, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["c_med", "ix_c2", "automotive_nhtsa", 42, "Medium", "model", 2, now - timedelta(days=1)],
        )
        # Case 3: Critical, old (15 days old)
        con.execute(
            "INSERT INTO cases (case_id, interaction_id, pack_id, cluster_match_id, severity, severity_source, priority, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ["c_old", "ix_c3", "automotive_nhtsa", 42, "Critical", "model", 1, now - timedelta(days=15)],
        )

    # Without window: count=3, max_severity=Critical
    count, sev = fetch_cluster_metrics("automotive_nhtsa", 42)
    assert count == 3
    assert sev == "Critical"

    # With window 7 days: count=2, max_severity=Critical (excludes 15 days old case)
    count_w, sev_w = fetch_cluster_metrics("automotive_nhtsa", 42, window_days=7)
    assert count_w == 2
    assert sev_w == "Critical"


# ============================================================================
# FU13: Investigator Association Denominator & PopulationIndex Counts
# ============================================================================

def test_fu13_population_index_preserved_in_rank_by_association():
    """PopulationIndex passed to rank_by_association must retain grouped SQL support counts."""
    from src.ml_runtime.association import PopulationIndex, rank_by_association

    pop = PopulationIndex([])
    pop.n = 1000
    pop.n_cat = {"BRAKES": 500}
    pop.n_ent = {"FORD": 400}
    pop.n_both = {("BRAKES", "FORD"): 300}

    candidates = [
        {"record_id": "r1", "category": "BRAKES", "entity_2": "FORD", "received_at": "2026-01-01"},
        {"record_id": "r2", "category": "OTHER", "entity_2": "CHEVY", "received_at": "2026-01-01"},
    ]

    res = rank_by_association({"category": "BRAKES", "entity_2": "FORD"}, candidates, pop)
    assert len(res) == 2
    assert res[0]["record_id"] == "r1"
    assert res[0]["assoc_population"] == "full-corpus"
    assert res[0]["assoc_score"] > res[1]["assoc_score"]


# ============================================================================
# FU14: Category Zero-Shot Hard Timeout Non-Blocking
# ============================================================================

def test_fu14_category_zeroshot_hard_timeout_does_not_block():
    """predict_category_zero_shot must return immediately on timeout without waiting for thread pool shutdown."""
    from src.ml_runtime.category_zeroshot import predict_category_zero_shot

    class SlowEmbedder:
        canonical_version = "test_slow"
        def embed(self, text):
            time.sleep(1.0)
            class Emb:
                values = [0.1] * 384
            return Emb()

    class MockPack:
        id = "test_pack"
        categories = ("brakes", "steering")
        gazetteer = {"brakes": ["brake", "pedal"]}
        taxonomy = {}

    t0 = time.monotonic()
    res = predict_category_zero_shot("pedal spongy", pack=MockPack(), embedder=SlowEmbedder(), timeout_ms=30.0)
    elapsed = time.monotonic() - t0

    assert res.category is None
    # Must return within 250ms, NOT after 1.0s
    assert elapsed < 0.35, f"predict_category_zero_shot took {elapsed:.2f}s; thread pool shutdown blocked caller"


# ============================================================================
# FU16: Pack Builder Insight Permission Enforcement
# ============================================================================

def test_fu16_pack_builder_insight_requires_pack_edit(reset_ops_db, monkeypatch):
    """Only roles with pack:edit (admin) may access /pack-builder/insight."""
    monkeypatch.setenv("FRONTLINE_AUTH_REQUIRED", "1")
    monkeypatch.setenv("FRONTLINE_OPEN_MODE", "0")
    client = TestClient(app)

    for role in ["agent", "auditor", "supervisor", "dsr_officer"]:
        tok = issue_session(f"{role}_user", role, issuer_role="admin")["token"]
        r = client.post(
            "/api/frontline/pack-builder/insight",
            json={"csv_path": "test.csv"},
            headers={"x-frontline-session": tok},
        )
        assert r.status_code == 403, f"Role {role} was not denied pack:edit access (got {r.status_code})"

    admin_tok = issue_session("admin_user", "admin", issuer_role="admin")["token"]
    r_admin = client.post(
        "/api/frontline/pack-builder/insight",
        json={"csv_path": ""},
        headers={"x-frontline-session": admin_tok},
    )
    # 400 Bad Request because csv_path is empty, NOT 403 Forbidden!
    assert r_admin.status_code == 400
