"""Audit findings: WAL segment durability and erasure during a live turn."""

from __future__ import annotations

import json

import pytest


def test_wal_segment_stays_until_remaining_records_are_fsynced(
    reset_ops_db, tmp_path, monkeypatch
):
    """A failed durable restore must keep the recoverable segment."""
    from src.ledger.writer import replay_ledger_wal

    wal_file = tmp_path / "ledger_wal.jsonl"
    monkeypatch.setattr("src.ledger.writer._wal_path", lambda: wal_file)
    wal_file.write_text(
        json.dumps(
            {
                "action": {
                    "action_id": "act_keep_segment",
                    "interaction_id": "ix_keep_segment",
                    "action_type": "state_transition",
                    "agent": "orchestrator",
                    "output_summary": "still-recoverable",
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def _db_down(*_a, **_k):
        raise OSError("db down")

    def _fsync_down(_path):
        raise OSError("fsync failed before segment release")

    monkeypatch.setattr("src.ledger.writer._record_action_inner", _db_down)
    monkeypatch.setattr("src.ledger.writer._fsync_file", _fsync_down)

    with pytest.raises(OSError, match="fsync failed"):
        replay_ledger_wal()

    segments = list(tmp_path.glob("*.replaying-*"))
    assert len(segments) == 1
    assert "act_keep_segment" in segments[0].read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_erasure_stops_inflight_turn_before_later_writes(
    orchestrator_factory, tmp_path, monkeypatch
):
    """Erasure during intake blocks later transcript, action, case, and archive writes."""
    from src.agents.case_agent import CaseAgent
    from src.agents.intake import IntakeAgent
    from src.data.turns import persist_turn
    from src.data.warehouse import ops_con
    from src.frontline.archive import build_audit_archive
    from src.ledger.writer import AgentAction, record_action

    orch, _hooks = orchestrator_factory()
    await orch.start()
    iid = orch.ctx.interaction_id
    orch.ctx.slots.update(
        {
            "entity_1": "2018",
            "entity_2": "Honda",
            "entity_3": "CR-V",
            "category": "SERVICE BRAKES",
            "description": "grinding noise when stopping",
            "__confirmed__": "1",
            "__diag_asked": "1",
        }
    )

    async def _erase_mid_turn(self, customer_turn="", **kwargs):
        from src.frontline.dsr import tombstone_interaction

        tombstone_interaction(self.ctx.interaction_id)
        return {"question": ""}

    monkeypatch.setattr(IntakeAgent, "run", _erase_mid_turn)

    try:
        await orch.handle_customer_turn("the brakes grind when I stop")
    except Exception:
        pass

    with ops_con(read_only=True) as con:
        cases = con.execute(
            "SELECT COUNT(*) FROM cases WHERE interaction_id = ?",
            [iid],
        ).fetchone()[0]
        fresh_actions = con.execute(
            """
            SELECT COUNT(*) FROM agent_actions
            WHERE interaction_id = ?
              AND action_type = 'case_created'
              AND COALESCE(erased, FALSE) = FALSE
            """,
            [iid],
        ).fetchone()[0]
    assert cases == 0
    assert fresh_actions == 0

    with pytest.raises(RuntimeError, match="erased"):
        record_action(
            AgentAction(
                interaction_id=iid,
                agent="orchestrator",
                action_type="state_transition",
                output_summary="post-erasure action",
            )
        )
    with pytest.raises(RuntimeError, match="erased"):
        persist_turn(
            iid,
            {
                "turn_id": "turn_after_erasure",
                "seq": 99,
                "speaker": "agent",
                "text": "post-erasure transcript",
            },
        )
    with pytest.raises(RuntimeError, match="erased"):
        await CaseAgent(orch.ctx).run()
    with pytest.raises(RuntimeError, match="erased"):
        build_audit_archive(iid, out_dir=tmp_path)

    assert list(tmp_path.glob("*.json")) == []
    with ops_con(read_only=True) as con:
        leaked = con.execute(
            "SELECT text FROM interaction_turns WHERE turn_id = 'turn_after_erasure'"
        ).fetchone()
    assert leaked is None
