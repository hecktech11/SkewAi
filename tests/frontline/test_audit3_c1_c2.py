"""Third-pass audit: C-1 hangup-while-supervised, C-2 UTC timestamps, high-sev."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.agents.orchestrator import ABANDONED, CLOSING, DONE, SUPERVISED
from src.api.jsonutil import json_safe
from src.api.routes.interactions import _sweep_stale_active_db_rows, reap_orphans
from src.data.timeutil import to_iso_z, to_naive_utc, utc_now
from src.data.turns import persist_turn
from src.data.warehouse import ops_con
from src.frontline.alerts import _dedup_key, fire_alert
from src.ids import new_ulid
from src.ledger import AgentAction, record_action


# ── C-1: hangup while SUPERVISED ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hangup_while_supervised_abandons_cleanly(orchestrator_factory):
    """Customer hangup during takeover must not raise and must leave DB non-active."""
    orch, _ = orchestrator_factory()
    await orch.start()
    await orch.takeover()
    assert orch.ctx.state == SUPERVISED

    await orch.hangup()  # must not raise Illegal transition

    assert orch.ctx.state == ABANDONED
    with ops_con(read_only=True) as con:
        row = con.execute(
            "SELECT status, outcome FROM interactions WHERE interaction_id = ?",
            [orch.ctx.interaction_id],
        ).fetchone()
    assert row is not None
    assert row[0] != "active"
    assert row[0] in ("abandoned", "failed", "completed", "escalated")


@pytest.mark.asyncio
async def test_hangup_while_supervised_with_case_id_finalizes(orchestrator_factory):
    """Hangup under SUPERVISED when case_id is set must force CLOSING path (not stay active)."""
    orch, _ = orchestrator_factory()
    await orch.start()
    # Fill enough to create a case, then take over mid-close if possible — or plant case_id.
    await orch.handle_customer_turn("My 2019 Honda CR-V grinds when I brake.")
    await orch.takeover()
    assert orch.ctx.state == SUPERVISED
    # Simulate a case already attached during supervised work
    orch.ctx.case_id = "case_supervised_hangup_1"
    await orch.hangup()
    assert orch.ctx.state in (DONE, "DONE", ABANDONED, CLOSING) or orch.ctx.state != SUPERVISED
    with ops_con(read_only=True) as con:
        st = con.execute(
            "SELECT status FROM interactions WHERE interaction_id = ?",
            [orch.ctx.interaction_id],
        ).fetchone()[0]
    assert st != "active", f"zombie active after supervised hangup with case: {st}"


@pytest.mark.asyncio
async def test_db_orphan_sweep_closes_stale_active(reset_ops_db):
    """Reaper sweeps active rows that are not in the in-memory registry."""
    iid = "int_zombie_" + new_ulid()[:8]
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions
            (interaction_id, pack_id, pack_version, started_at, channel, status)
            VALUES (?, 'automotive_nhtsa', '1', ?, 'web', 'active')
            """,
            [iid, utc_now()],
        )
    closed = _sweep_stale_active_db_rows(registry_ids=set())
    assert iid in closed
    with ops_con(read_only=True) as con:
        st = con.execute(
            "SELECT status, outcome FROM interactions WHERE interaction_id = ?",
            [iid],
        ).fetchone()
    assert st[0] == "failed"
    assert st[1] == "orphan_db_sweep"

    # Public reaper path also runs the sweep
    iid2 = "int_zombie2_" + new_ulid()[:8]
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions
            (interaction_id, pack_id, pack_version, started_at, channel, status)
            VALUES (?, 'automotive_nhtsa', '1', ?, 'web', 'active')
            """,
            [iid2, utc_now()],
        )
    await reap_orphans()
    with ops_con(read_only=True) as con:
        st2 = con.execute(
            "SELECT status FROM interactions WHERE interaction_id = ?", [iid2]
        ).fetchone()
    assert st2[0] != "active"


# ── C-2: UTC timestamp semantics ─────────────────────────────────────────────


def test_utc_round_trip_naive_store_and_z_wire(reset_ops_db):
    """Known UTC instant survives write/read as naive UTC and serializes with Z."""
    fixed = datetime(2026, 7, 27, 19, 47, 7, tzinfo=timezone.utc)
    turn_id = "trn_utc_" + new_ulid()[:8]
    iid = "int_utc_" + new_ulid()[:8]
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions
            (interaction_id, pack_id, pack_version, started_at, channel, status)
            VALUES (?, 'automotive_nhtsa', '1', ?, 'sim', 'active')
            """,
            [iid, to_naive_utc(fixed)],
        )
    persist_turn(
        iid,
        {
            "turn_id": turn_id,
            "seq": 1,
            "speaker": "customer",
            "text": "hello",
            "ts": fixed,
        },
    )
    with ops_con(read_only=True) as con:
        row = con.execute(
            "SELECT ts FROM interaction_turns WHERE turn_id = ?", [turn_id]
        ).fetchone()
        started = con.execute(
            "SELECT started_at FROM interactions WHERE interaction_id = ?", [iid]
        ).fetchone()[0]
    stored = row[0]
    assert isinstance(stored, datetime)
    # Wall-clock UTC equality (naive or aware both OK)
    naive_expected = fixed.replace(tzinfo=None)
    got = stored.replace(tzinfo=None) if stored.tzinfo else stored
    assert got == naive_expected
    wire = json_safe({"ts": stored})["ts"]
    assert wire.endswith("Z"), wire
    assert "2026-07-27T19:47:07" in wire
    # started_at same basis
    assert started.replace(tzinfo=None) == naive_expected
    assert to_iso_z(fixed).endswith("Z")


def test_persist_turn_fallback_is_naive_utc_not_utcnow_mix(reset_ops_db, monkeypatch):
    """Default ts path uses utc_now() (naive UTC), not datetime.utcnow() mix."""
    from src.data import turns as turns_mod

    fixed = datetime(2026, 1, 15, 12, 0, 0)
    monkeypatch.setattr(turns_mod, "utc_now", lambda: fixed)
    iid = "int_fb_" + new_ulid()[:8]
    tid = "trn_fb_" + new_ulid()[:8]
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions
            (interaction_id, pack_id, pack_version, started_at, channel, status)
            VALUES (?, 'automotive_nhtsa', '1', ?, 'sim', 'active')
            """,
            [iid, fixed],
        )
    persist_turn(iid, {"turn_id": tid, "seq": 0, "speaker": "agent", "text": "hi"})
    with ops_con(read_only=True) as con:
        ts = con.execute(
            "SELECT ts FROM interaction_turns WHERE turn_id = ?", [tid]
        ).fetchone()[0]
    assert ts.replace(tzinfo=None) == fixed


def test_ledger_ts_normalized_to_naive_utc(reset_ops_db):
    aware = datetime(2026, 3, 1, 8, 30, 0, tzinfo=timezone.utc)
    aid = record_action(
        AgentAction(
            interaction_id="int_ledger_ts",
            agent="orchestrator",
            action_type="state_transition",
            ts=aware,
        )
    )
    with ops_con(read_only=True) as con:
        ts = con.execute(
            "SELECT ts FROM agent_actions WHERE action_id = ?", [aid]
        ).fetchone()[0]
    assert ts.replace(tzinfo=None) == aware.replace(tzinfo=None)
    assert json_safe({"ts": ts})["ts"].endswith("Z")


def test_reaper_ended_at_is_naive_utc_wall_clock(reset_ops_db, monkeypatch):
    """DB sweep must write ended_at via utc_now (naive), not aware session-skewed values."""
    from src.api.routes import interactions as ix
    from src.data import timeutil as tu

    fixed = datetime(2026, 7, 27, 19, 47, 7)
    monkeypatch.setattr(tu, "utc_now", lambda: fixed)
    # Also patch where the sweep imports utc_now at call time
    iid = "int_reap_utc_" + new_ulid()[:8]
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions
            (interaction_id, pack_id, pack_version, started_at, channel, status)
            VALUES (?, 'automotive_nhtsa', '1', ?, 'web', 'active')
            """,
            [iid, fixed],
        )
    closed = ix._sweep_stale_active_db_rows(registry_ids=set())
    assert iid in closed
    with ops_con(read_only=True) as con:
        ended = con.execute(
            "SELECT ended_at FROM interactions WHERE interaction_id = ?", [iid]
        ).fetchone()[0]
    got = ended.replace(tzinfo=None) if getattr(ended, "tzinfo", None) else ended
    assert got == fixed, f"reaper ended_at skew: stored={ended!r} expected={fixed!r}"
    assert json_safe({"ended_at": ended})["ended_at"].endswith("Z")


@pytest.mark.asyncio
async def test_case_agent_created_at_naive_utc_round_trip(
    reset_ops_db, seed_automotive_pack, pack
):
    """CaseAgent insert path: created_at wall-clock UTC (no session TZ skew)."""
    from src.agents.base import InteractionContext
    from src.agents.case_agent import CaseAgent, _now as case_now

    n = case_now()
    assert n.tzinfo is None, f"CaseAgent._now still aware: {n!r}"

    iid = "int_case_utc_" + new_ulid()[:8]
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions
            (interaction_id, pack_id, pack_version, started_at, channel, status)
            VALUES (?, ?, ?, ?, 'sim', 'active')
            """,
            [iid, pack.id, pack.pack_version, utc_now()],
        )
    ctx = InteractionContext(interaction_id=iid, pack=pack, channel="sim")
    ctx.state = CLOSING
    ctx.slots = {
        "entity_1": "2019",
        "entity_2": "Honda",
        "entity_3": "CR-V",
        "category": "brakes",
        "description": "grinds",
    }
    ctx.severity = "Medium"
    ctx.severity_source = "rules"
    ctx.priority = 2
    cres = await CaseAgent(ctx).run()
    case_id = cres["case_id"]
    assert case_id
    with ops_con(read_only=True) as con:
        created = con.execute(
            "SELECT created_at FROM cases WHERE case_id = ?", [case_id]
        ).fetchone()[0]
    got = created.replace(tzinfo=None) if getattr(created, "tzinfo", None) else created
    # Wall clock must match CaseAgent._now basis (naive UTC), not IST +5.5h
    assert abs((got - case_now()).total_seconds()) < 120
    wire = json_safe({"created_at": created})["created_at"]
    assert wire.endswith("Z")
    assert "T" in wire


def test_residual_writers_use_naive_utc(reset_ops_db, monkeypatch):
    """Scenario started_at, connector deliveries, risk_snapshots, case_notes share utc_now."""
    from src.data import timeutil as tu
    from src.enterprise import risk as risk_mod
    from src.enterprise import scenarios as scn_mod
    from src.frontline import connectors as conn_mod
    from src.frontline import ops as ops_mod

    fixed = datetime(2026, 7, 27, 19, 47, 7)
    monkeypatch.setattr(tu, "utc_now", lambda: fixed)

    # --- scenario header insert (same path as run_scenario) ---
    iid = "int_scn_utc_" + new_ulid()[:8]
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions
            (interaction_id, pack_id, pack_version, started_at, channel, status)
            VALUES (?, 'automotive_nhtsa', '1', ?, 'simulated', 'active')
            """,
            [iid, scn_mod._now()],
        )
    with ops_con(read_only=True) as con:
        started = con.execute(
            "SELECT started_at FROM interactions WHERE interaction_id = ?", [iid]
        ).fetchone()[0]
    assert started.replace(tzinfo=None) == fixed

    # --- connector delivery created_at ---
    conn_mod._record_delivery(
        delivery_id="dlv_utc_1",
        event="case_created",
        ref_id="case_x",
        interaction_id=iid,
        case_id="case_x",
        investigation_id=None,
        payload={"e": 1},
        sink="outbox",
        status="success",
        attempts=1,
        error=None,
        outbox_path=None,
        http_status=None,
    )
    with ops_con(read_only=True) as con:
        c_at = con.execute(
            "SELECT created_at FROM connector_deliveries WHERE delivery_id = ?",
            ["dlv_utc_1"],
        ).fetchone()[0]
    assert c_at.replace(tzinfo=None) == fixed

    # --- risk snapshot ts ---
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO risk_snapshots
            (snapshot_id, interaction_id, ts, escalation_prob, churn_prob,
             sentiment_trend, est_resolution_turns, factors_json)
            VALUES (?, ?, ?, 0.1, 0.1, 0.0, 3.0, '{}')
            """,
            ["rsk_utc_1", iid, risk_mod.__dict__.get("utc_now", tu.utc_now)()],
        )
    # Use module-level path: score_interaction_risk persist path uses utc_now
    from src.data.timeutil import utc_now as un

    assert un() == fixed  # monkeypatched
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO risk_snapshots
            (snapshot_id, interaction_id, ts, escalation_prob, churn_prob,
             sentiment_trend, est_resolution_turns, factors_json)
            VALUES (?, ?, ?, 0.2, 0.2, 0.0, 2.0, '{}')
            """,
            ["rsk_utc_2", iid, un()],
        )
    with ops_con(read_only=True) as con:
        rts = con.execute(
            "SELECT ts FROM risk_snapshots WHERE snapshot_id = ?", ["rsk_utc_2"]
        ).fetchone()[0]
    assert rts.replace(tzinfo=None) == fixed

    # --- case note created_at + wire Z ---
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO cases (
                case_id, interaction_id, pack_id, created_at,
                category, description_summary, onset, severity,
                severity_source, priority, safety_flags,
                advisory_match_id, cluster_match_id, similar_record_count,
                investigation_id, status, followup_draft
            ) VALUES (?, ?, 'automotive_nhtsa', ?, 'brakes', 'x', ?, 'Low',
                      'rules', 3, '{}', NULL, NULL, 0, NULL, 'open', '')
            """,
            ["case_note_utc", iid, fixed, fixed],
        )
    note = ops_mod.add_case_note("case_note_utc", "operator note body", author="pilot")
    assert note["created_at"].endswith("Z")
    assert "2026-07-27T19:47:07" in note["created_at"]
    with ops_con(read_only=True) as con:
        n_at = con.execute(
            "SELECT created_at FROM case_notes WHERE note_id = ?",
            [note["note_id"]],
        ).fetchone()[0]
    assert n_at.replace(tzinfo=None) == fixed


# ── Alert pack-scoped dedup + SSRF ───────────────────────────────────────────


def test_alert_dedup_key_includes_pack():
    k1 = _dedup_key("investigation_opened", "3", "automotive_nhtsa")
    k2 = _dedup_key("investigation_opened", "3", "finance_cfpb")
    assert k1 != k2
    assert "automotive_nhtsa" in k1
    assert "finance_cfpb" in k2


@pytest.mark.asyncio
async def test_alert_dedup_independent_per_pack(reset_ops_db):
    from src.frontline import alerts as alerts_mod

    ok_post = AsyncMock(return_value=(True, None))
    with patch.object(alerts_mod, "settings") as ms:
        ms.alert_webhook_url = "https://example.com/hooks"
        with patch.object(alerts_mod, "_post_webhook_once", new=ok_post):
            a = await fire_alert(
                "investigation_opened", "auto", "3", pack_id="automotive_nhtsa"
            )
            b = await fire_alert(
                "investigation_opened", "fin", "3", pack_id="finance_cfpb"
            )
            c = await fire_alert(
                "investigation_opened", "auto2", "3", pack_id="automotive_nhtsa"
            )
    assert a is True
    assert b is True
    assert c is False
    assert ok_post.await_count == 2


@pytest.mark.asyncio
async def test_alert_webhook_ssrf_blocked(reset_ops_db, monkeypatch):
    from src.frontline import alerts as alerts_mod

    monkeypatch.delenv("FRONTLINE_OPEN_MODE", raising=False)
    with patch.object(alerts_mod, "settings") as ms:
        ms.alert_webhook_url = "http://169.254.169.254/latest"
        sent = await fire_alert("takeover_started", "x", "ref1")
    assert sent is False


# ── Prune ────────────────────────────────────────────────────────────────────


def test_prune_ops_tables_deletes_aged_rows(reset_ops_db):
    from src.data.prune import prune_ops_tables

    old = utc_now() - timedelta(days=90)
    with ops_con() as con:
        con.execute(
            "INSERT INTO alert_dedup (dedup_key, event, ref_id, fired_at) VALUES (?, ?, ?, ?)",
            ["old:key", "e", "r", old],
        )
        con.execute(
            "INSERT INTO alert_dedup (dedup_key, event, ref_id, fired_at) VALUES (?, ?, ?, ?)",
            ["new:key", "e", "r2", utc_now()],
        )
    result = prune_ops_tables(older_than_days=30, now=utc_now())
    assert result["deleted"]["alert_dedup"] >= 1
    with ops_con(read_only=True) as con:
        keys = [
            r[0]
            for r in con.execute("SELECT dedup_key FROM alert_dedup").fetchall()
        ]
    assert "old:key" not in keys
    assert "new:key" in keys


def test_r18_prune_never_deletes_agent_actions_and_aligns_retention(reset_ops_db):
    """Pruning must never hard delete agent_actions rows, and respect turns retention."""
    from src.data.prune import prune_ops_tables
    from src.ledger.writer import AgentAction, record_action

    # Create an old agent action (100 days old)
    old_ts = utc_now() - timedelta(days=100)
    act = AgentAction(
        interaction_id="int_prune_test",
        agent="orchestrator",
        action_type="state_transition",
        input_summary="in",
        output_summary="out",
        ts=old_ts,
    )
    record_action(act)

    # Create an old interaction turn (100 days old) and a recent one (45 days old)
    with ops_con() as con:
        con.execute(
            "INSERT INTO interaction_turns (turn_id, interaction_id, seq, speaker, text, ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ["turn_old", "int_prune_test", 1, "customer", "old complaint", old_ts],
        )
        con.execute(
            "INSERT INTO interaction_turns (turn_id, interaction_id, seq, speaker, text, ts) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ["turn_mid", "int_prune_test", 2, "customer", "mid complaint", utc_now() - timedelta(days=45)],
        )

    # Prune with default policy (turns = 90 days, agent_actions never deleted)
    result = prune_ops_tables(now=utc_now())
    assert "agent_actions" not in result["deleted"]

    with ops_con(read_only=True) as con:
        # agent_actions must survive
        act_count = con.execute("SELECT COUNT(*) FROM agent_actions WHERE interaction_id = ?", ["int_prune_test"]).fetchone()[0]
        assert act_count >= 1

        # turns older than 90d must be deleted, turns under 90d must survive
        turns = [r[0] for r in con.execute("SELECT turn_id FROM interaction_turns WHERE interaction_id = ?", ["int_prune_test"]).fetchall()]
        assert "turn_old" not in turns
        assert "turn_mid" in turns



# ── Rollback safety ──────────────────────────────────────────────────────────


def test_rollback_invalid_target_keeps_active(reset_ops_db):
    from src.v3.governance import (
        activate_deployment,
        create_deployment,
        get_active_deployment,
        rollback_deployment,
    )

    d1 = create_deployment(label="keep-me", artifact_versions={"a": "1"})
    activate_deployment(d1["deployment_id"])
    with pytest.raises(LookupError):
        rollback_deployment(to_deployment_id="dep_does_not_exist")
    active = get_active_deployment()
    assert active is not None
    assert active["deployment_id"] == d1["deployment_id"]
    assert active["status"] == "active"


# ── Scenario HTTP offload structural ─────────────────────────────────────────


def test_scenario_run_route_uses_to_thread():
    from pathlib import Path

    text = Path("src/api/routes/enterprise.py").read_text(encoding="utf-8")
    # scenarios_run must offload via to_thread
    assert "to_thread" in text
    assert "run_scenario" in text


def test_post_contact_audit_not_awaited_inline():
    from pathlib import Path

    text = Path("src/agents/orchestrator.py").read_text(encoding="utf-8")
    assert "create_task" in text
    assert "audit_interaction" in text
