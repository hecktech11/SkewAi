import copy
import pytest
from src.data.warehouse import ops_con
from src.ledger import AgentAction, record_action
from src.ledger.chain import GENESIS, compute_row_hash, verify_chain
from src.ledger.writer import list_actions


def test_chain_rejects_truncated_prefix(reset_ops_db):
    """Building 5 rows and passing rows 2-4 must fail with chain_starts_midway."""
    iid = "int_trunc_prefix_1"
    for i in range(5):
        record_action(
            AgentAction(
                interaction_id=iid,
                agent="orchestrator",
                action_type="state_transition",
                input_summary=f"in_{i}",
                output_summary=f"out_{i}",
            )
        )

    rows = list_actions(iid)
    assert len(rows) == 5

    # Pass rows 2-4 (index 1 to 3)
    sub_chain = rows[1:4]
    res = verify_chain(sub_chain)
    assert res["ok"] is False
    assert res.get("error") == "chain_starts_midway"


def test_chain_rejects_missing_tail(reset_ops_db):
    """Building 5 rows and passing rows 1-3 must fail with chain_incomplete when DB has 5 rows."""
    iid = "int_missing_tail_1"
    for i in range(5):
        record_action(
            AgentAction(
                interaction_id=iid,
                agent="orchestrator",
                action_type="state_transition",
                input_summary=f"in_{i}",
                output_summary=f"out_{i}",
            )
        )

    rows = list_actions(iid)
    assert len(rows) == 5

    # Pass rows 1-3 (index 0 to 2)
    sub_chain = rows[:3]
    res = verify_chain(sub_chain)
    assert res["ok"] is False
    assert res.get("error") == "chain_incomplete"

    # With allow_partial=True, verification of prefix should pass
    res_allowed = verify_chain(sub_chain, allow_partial=True)
    assert res_allowed["ok"] is True


def test_chain_accepts_complete_chain(reset_ops_db):
    """Building 5 rows and passing all 5 rows must pass verification."""
    iid = "int_complete_chain_1"
    for i in range(5):
        record_action(
            AgentAction(
                interaction_id=iid,
                agent="orchestrator",
                action_type="state_transition",
                input_summary=f"in_{i}",
                output_summary=f"out_{i}",
            )
        )

    rows = list_actions(iid)
    assert len(rows) == 5

    res = verify_chain(rows)
    assert res["ok"] is True


def test_erased_row_still_content_verified(reset_ops_db):
    """Erased row with content_hash in hash_version=2 must pass verification."""
    iid = "int_erased_verified_1"
    for i in range(3):
        record_action(
            AgentAction(
                interaction_id=iid,
                agent="orchestrator",
                action_type="state_transition",
                input_summary=f"in_{i}",
                output_summary=f"out_{i}",
            )
        )

    rows = list_actions(iid)
    assert len(rows) == 3

    # Erase row 1 (index 1)
    erased_rows = copy.deepcopy(rows)
    target = erased_rows[1]
    target["erased"] = True
    target["input_summary"] = "[ERASED]"
    target["output_summary"] = "[ERASED]"
    # ensure target has content_hash and hash_version=2
    assert "content_hash" in target and target["content_hash"]
    target["hash_version"] = 2

    res = verify_chain(erased_rows, allow_partial=True)
    assert res["ok"] is True


def test_erased_row_without_content_hash_fails(reset_ops_db):
    """Erased row without content_hash in hash_version=2 must fail with erased_row_missing_content_hash."""
    iid = "int_erased_missing_hash_1"
    for i in range(3):
        record_action(
            AgentAction(
                interaction_id=iid,
                agent="orchestrator",
                action_type="state_transition",
                input_summary=f"in_{i}",
                output_summary=f"out_{i}",
            )
        )

    rows = list_actions(iid)
    erased_rows = copy.deepcopy(rows)
    target = erased_rows[1]
    target["erased"] = True
    target["input_summary"] = "[ERASED]"
    target["output_summary"] = "[ERASED]"
    target["hash_version"] = 2
    target["content_hash"] = None

    res = verify_chain(erased_rows, allow_partial=True)
    assert res["ok"] is False
    assert res.get("error") == "erased_row_missing_content_hash"


def test_tampered_content_hash_fails(reset_ops_db):
    """Erased row with tampered content_hash must fail verification."""
    iid = "int_tampered_content_hash_1"
    for i in range(3):
        record_action(
            AgentAction(
                interaction_id=iid,
                agent="orchestrator",
                action_type="state_transition",
                input_summary=f"in_{i}",
                output_summary=f"out_{i}",
            )
        )

    rows = list_actions(iid)
    erased_rows = copy.deepcopy(rows)
    target = erased_rows[1]
    target["erased"] = True
    target["input_summary"] = "[ERASED]"
    target["output_summary"] = "[ERASED]"
    target["hash_version"] = 2
    target["content_hash"] = "f" * 64  # tampered

    res = verify_chain(erased_rows, allow_partial=True)
    assert res["ok"] is False


def test_claims_included_in_hash_v2():
    """Two actions with different claims under hash_version=2 must produce different row_hashes."""
    from src.ledger.chain import compute_row_hash

    action1 = {
        "action_id": "act_1",
        "interaction_id": "int_claims_1",
        "case_id": None,
        "agent": "investigator",
        "action_type": "similar_search",
        "input_summary": "query",
        "output_summary": "result",
        "evidence_ids": [],
        "claims": [{"claim_id": "c1", "text": "claim A"}],
        "ok": True,
        "error": None,
        "duration_ms": 10,
        "ts": "2026-01-01T00:00:00",
        "hash_version": 2,
    }
    action2 = copy.deepcopy(action1)
    action2["claims"] = [{"claim_id": "c2", "text": "claim B"}]

    hash1 = compute_row_hash(action1, GENESIS, version=2)
    hash2 = compute_row_hash(action2, GENESIS, version=2)

    assert hash1 != hash2, "Different claims must produce different row hashes under hash_version=2"


def test_legacy_rows_still_verify():
    """Legacy rows with hash_version=1 must verify using legacy behavior."""
    from src.ledger.chain import compute_row_hash, verify_chain

    # Row 0
    row0 = {
        "action_id": "act_leg_0",
        "interaction_id": "int_leg_1",
        "case_id": None,
        "agent": "orchestrator",
        "action_type": "state_transition",
        "input_summary": "in_0",
        "output_summary": "out_0",
        "evidence_ids": [],
        "claims": [],
        "ok": True,
        "error": None,
        "duration_ms": 5,
        "ts": "2026-01-01T00:00:00",
        "prev_hash": GENESIS,
        "hash_version": 1,
    }
    row0["row_hash"] = compute_row_hash(row0, GENESIS, version=1)

    # Row 1 (erased without content_hash, which was permitted in v1)
    row1 = {
        "action_id": "act_leg_1",
        "interaction_id": "int_leg_1",
        "case_id": None,
        "agent": "orchestrator",
        "action_type": "state_transition",
        "input_summary": "[ERASED]",
        "output_summary": "[ERASED]",
        "evidence_ids": [],
        "claims": [],
        "ok": True,
        "error": None,
        "duration_ms": 5,
        "ts": "2026-01-01T00:00:01",
        "prev_hash": row0["row_hash"],
        "hash_version": 1,
        "erased": True,
        "content_hash": None,
    }
    row1["row_hash"] = compute_row_hash(row1, row0["row_hash"], version=1)

    # Row 2
    row2 = {
        "action_id": "act_leg_2",
        "interaction_id": "int_leg_1",
        "case_id": None,
        "agent": "orchestrator",
        "action_type": "state_transition",
        "input_summary": "in_2",
        "output_summary": "out_2",
        "evidence_ids": [],
        "claims": [{"different": "ignored in v1"}],
        "ok": True,
        "error": None,
        "duration_ms": 5,
        "ts": "2026-01-01T00:00:02",
        "prev_hash": row1["row_hash"],
        "hash_version": 1,
    }
    row2["row_hash"] = compute_row_hash(row2, row1["row_hash"], version=1)

    chain = [row0, row1, row2]
    res = verify_chain(chain, allow_partial=True)
    assert res["ok"] is True


def test_v1_erased_cannot_hide_content_tamper():
    """F-003 leftover: erased v1 rows still have their current content hashed."""
    from src.ledger.chain import compute_row_hash, verify_chain

    row0 = {
        "action_id": "act_tamper_0",
        "interaction_id": "int_tamper_v1",
        "case_id": None,
        "agent": "orchestrator",
        "action_type": "state_transition",
        "input_summary": "in_0",
        "output_summary": "out_0",
        "evidence_ids": [],
        "claims": [],
        "ok": True,
        "error": None,
        "duration_ms": 5,
        "ts": "2026-01-01T00:00:00",
        "prev_hash": GENESIS,
        "hash_version": 1,
    }
    row0["row_hash"] = compute_row_hash(row0, GENESIS, version=1)
    row1 = {
        "action_id": "act_tamper_1",
        "interaction_id": "int_tamper_v1",
        "case_id": None,
        "agent": "orchestrator",
        "action_type": "state_transition",
        "input_summary": "secret original",
        "output_summary": "out_1",
        "evidence_ids": [],
        "claims": [],
        "ok": True,
        "error": None,
        "duration_ms": 5,
        "ts": "2026-01-01T00:00:01",
        "prev_hash": row0["row_hash"],
        "hash_version": 1,
    }
    row1["row_hash"] = compute_row_hash(row1, row0["row_hash"], version=1)
    row1["erased"] = True
    row1["input_summary"] = "attacker rewrite"
    res = verify_chain([row0, row1], allow_partial=True)
    assert res["ok"] is False


def test_r09_tampered_metadata_on_erased_row_fails(reset_ops_db):
    """Setting erased=True and editing non-erasable metadata must fail verification."""
    iid = "int_r09_meta_tamper"
    record_action(
        AgentAction(
            interaction_id=iid,
            agent="orchestrator",
            action_type="state_transition",
            input_summary="customer info",
            output_summary="greeting emitted",
        )
    )
    rows = list_actions(iid)
    assert len(rows) == 1
    # Legitimate erasure passes
    erased_row = copy.deepcopy(rows[0])
    erased_row["erased"] = True
    erased_row["input_summary"] = "[ERASED 2026-09-26 per erasure request]"
    erased_row["output_summary"] = "[ERASED 2026-09-26 per erasure request]"
    res = verify_chain([erased_row], allow_partial=True)
    assert res["ok"] is True

    # Tampering with agent on erased row fails
    tampered_agent = copy.deepcopy(erased_row)
    tampered_agent["agent"] = "rogue_agent"
    res_agent = verify_chain([tampered_agent], allow_partial=True)
    assert res_agent["ok"] is False
    assert res_agent["error"] == "metadata_commitment_mismatch"

    # Tampering with action_type on erased row fails
    tampered_type = copy.deepcopy(erased_row)
    tampered_type["action_type"] = "unauthorized_export"
    res_type = verify_chain([tampered_type], allow_partial=True)
    assert res_type["ok"] is False
    assert res_type["error"] == "metadata_commitment_mismatch"


def test_r09_forged_erased_flag_on_content_fails(reset_ops_db):
    """Setting erased=True without exact tombstone shape must fail verification."""
    iid = "int_r09_content_tamper"
    record_action(
        AgentAction(
            interaction_id=iid,
            agent="orchestrator",
            action_type="state_transition",
            input_summary="customer info",
            output_summary="greeting emitted",
        )
    )
    rows = list_actions(iid)
    # Attacker keeps content_hash and row_hash, changes output_summary, sets erased=True
    tampered = copy.deepcopy(rows[0])
    tampered["output_summary"] = "attacker forged output summary"
    tampered["erased"] = True
    res = verify_chain([tampered], allow_partial=True)
    assert res["ok"] is False
    assert res["error"] == "invalid_tombstone_shape"
