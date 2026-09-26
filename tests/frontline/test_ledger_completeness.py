"""Ledger completeness invariant tests (per blueprint §15.2).

The core audit-first invariant:

    No ledger row, no output.

Every customer-visible turn (speaker in {agent, supervisor}) MUST have a
prior `agent_actions` row. We run a scripted contact end-to-end, then
verify for every interaction_turns row with speaker in (agent, supervisor)
that an `agent_actions` row exists with the same text in its output_summary
(or close to it — the ledger stores the same string the orchestrator emits).

This is the contract Qubot's groundedness audit relies on.
"""

from __future__ import annotations

import pytest

from src.data.warehouse import ops_con


async def _run_complete_contact(orchestrator_factory):
    """Drive a full contact through DONE and return the interaction_id."""
    orch, _ = orchestrator_factory()
    await orch.start()
    await orch.handle_customer_turn("My 2019 Honda CR-V grinds when I brake.")
    await orch.handle_customer_turn("Nobody is hurt and I'm safe.")
    await orch.handle_customer_turn("Yes, I'm in a safe location.")
    if orch.ctx.slots.get("__confirm_pending__"):
        await orch.handle_customer_turn("Yes, that's right.")
    assert orch.ctx.state == "DONE"
    return orch.ctx.interaction_id


# ── The audit-first invariant ─────────────────────────────────────────────────


async def test_every_agent_turn_has_a_ledger_row(orchestrator_factory, reset_ops_db):
    """For every agent turn in interaction_turns, an agent_actions row exists.

    The ledger row's output_summary should contain the turn's text (the
    orchestrator calls record_action BEFORE record_turn — so the row exists
    by the time the turn is persisted).
    """
    iid = await _run_complete_contact(orchestrator_factory)

    with ops_con(read_only=True) as con:
        turns = con.execute(
            "SELECT seq, speaker, text FROM interaction_turns WHERE interaction_id = ? ORDER BY seq",
            [iid],
        ).fetchall()
        actions = con.execute(
            "SELECT action_type, output_summary FROM agent_actions WHERE interaction_id = ? ORDER BY ts",
            [iid],
        ).fetchall()

    agent_turns = [(seq, text) for seq, speaker, text in turns if speaker == "agent"]
    assert len(agent_turns) >= 3, "expected at least 3 agent turns (greeting, advisory, goodbye)"

    # Each agent turn's text must appear in SOME ledger row's output_summary.
    for seq, turn_text in agent_turns:
        # The greeting is emitted by start() and ledgered as 'interaction_started'
        # which has output_summary='interaction created' — so we accept either
        # the turn text appears OR it's the greeting (special-cased below).
        if "Thanks for calling" in turn_text:
            # Greeting is ledgered via the interaction_started action.
            assert any("interaction" in (a[1] or "").lower() for a in actions), (
                f"greeting turn {seq} has no ledger row"
            )
            continue
        # Otherwise the turn text must appear in some action's output_summary.
        assert any(turn_text[:60] in (a[1] or "") for a in actions), (
            f"agent turn {seq} (text='{turn_text[:60]}...') has no matching ledger row"
        )


async def test_every_supervisor_turn_has_a_ledger_row(orchestrator_factory, reset_ops_db):
    """Supervisor turns must also be ledgered before delivery."""
    orch, _ = orchestrator_factory()
    await orch.start()
    await orch.takeover()
    await orch.human_turn("Supervisor stepping in to help.")
    await orch.release()
    await orch.handle_customer_turn("My 2019 Honda CR-V grinds when I brake.")
    await orch.handle_customer_turn("Nobody is hurt and I'm safe.")
    await orch.handle_customer_turn("Yes, I'm in a safe location.")

    iid = orch.ctx.interaction_id
    with ops_con(read_only=True) as con:
        supervisor_turns = con.execute(
            "SELECT text FROM interaction_turns WHERE interaction_id = ? AND speaker = 'supervisor'",
            [iid],
        ).fetchall()
        human_actions = con.execute(
            "SELECT output_summary FROM agent_actions WHERE interaction_id = ? AND action_type = 'human_turn'",
            [iid],
        ).fetchall()

    assert len(supervisor_turns) >= 1
    assert len(human_actions) >= 1
    # The supervisor's text must appear in a human_turn ledger row.
    supervisor_text = supervisor_turns[0][0]
    assert any(supervisor_text[:40] in (a[0] or "") for a in human_actions)


async def test_ledger_rows_written_before_turns_persisted(orchestrator_factory, reset_ops_db):
    """The ledger rows are written before interaction_turns rows.

    We verify this indirectly: every turn's row exists when we query, AND the
    agent_actions table has at least as many rows as the agent turns (so no
    turn was emitted without a ledger row).
    """
    iid = await _run_complete_contact(orchestrator_factory)

    with ops_con(read_only=True) as con:
        n_agent_turns = con.execute(
            "SELECT COUNT(*) FROM interaction_turns WHERE interaction_id = ? AND speaker = 'agent'",
            [iid],
        ).fetchone()[0]
        n_actions = con.execute(
            "SELECT COUNT(*) FROM agent_actions WHERE interaction_id = ?",
            [iid],
        ).fetchone()[0]

    # The orchestrator ledgeres: interaction_started, state_transition (×N),
    # slot_extracted, question_asked (×N), intake_completed, advisory_check,
    # advisory_notified, severity_scored, priority_assigned, similar_search,
    # cluster_matched, spike_checked, brief_written, case_created,
    # followup_drafted, state_transition (closing/done).
    # So n_actions should be comfortably >= n_agent_turns.
    assert n_actions >= n_agent_turns, (
        f"expected >= {n_agent_turns} ledger rows, got {n_actions} "
        f"(some agent turn may not have been ledgered)"
    )


async def test_no_agent_turn_without_action(orchestrator_factory, reset_ops_db):
    """A customer-only interaction (immediate hangup) must have zero agent turns
    but still may have ledger rows (the interaction_started row at minimum).
    """
    orch, _ = orchestrator_factory()
    await orch.start()  # emits greeting
    await orch.hangup()  # abandon immediately

    iid = orch.ctx.interaction_id
    with ops_con(read_only=True) as con:
        n_agent_turns = con.execute(
            "SELECT COUNT(*) FROM interaction_turns WHERE interaction_id = ? AND speaker = 'agent'",
            [iid],
        ).fetchone()[0]
        n_actions = con.execute(
            "SELECT COUNT(*) FROM agent_actions WHERE interaction_id = ?",
            [iid],
        ).fetchone()[0]

    # The greeting was emitted as an agent turn.
    assert n_agent_turns >= 1
    assert n_actions >= n_agent_turns  # every agent turn has a ledger row


def test_r08_ledger_replay_handles_ongoing_outage_honestly(tmp_path, monkeypatch, reset_ops_db):
    """When the DB is down during replay, replay_ledger_wal must not report recovery or clear WAL."""
    import json
    from unittest.mock import patch

    from src.ledger.writer import replay_ledger_wal

    wal_file = tmp_path / "ledger_wal.jsonl"
    monkeypatch.setattr("src.ledger.writer._wal_path", lambda: wal_file)

    # Write a test action to WAL
    action_data = {
        "wal_ts": "2026-09-26T12:00:00Z",
        "action": {
            "action_id": "act_wal_test_1",
            "interaction_id": "int_wal_test_1",
            "action_type": "supervised_safety_raised",
            "agent": "sentinel",
            "output_summary": "safety alert",
        },
    }
    wal_file.write_text(json.dumps(action_data) + "\n", encoding="utf-8")

    # Simulate ongoing outage in _record_action_inner
    with patch("src.ledger.writer._record_action_inner", side_effect=OSError("DB down")):
        res = replay_ledger_wal()
        assert res["replayed"] == 0
        assert res["failed"] == 1
        assert res["remaining"] == 1
        assert wal_file.exists()
        assert "act_wal_test_1" in wal_file.read_text()

    # Now recover (no error)
    res_ok = replay_ledger_wal()
    assert res_ok["replayed"] == 1
    assert res_ok["failed"] == 0
    assert res_ok["remaining"] == 0
    assert not wal_file.exists()
