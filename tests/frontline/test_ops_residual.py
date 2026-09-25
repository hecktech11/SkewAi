"""H4 simulate offload, H6 console fan-out shapes, M1 max-turns wrap-up."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from src.agents.base import InteractionContext
from src.agents.intake import IntakeAgent
from src.agents.orchestrator import Orchestrator, OrchestratorHooks
from src.api.routes import interactions as ir
from src.api.routes.frontline import simulate as simulate_route
from src.config import settings
from src.data.warehouse import ops_con
from src.ids import new_ulid


def test_simulate_route_uses_to_thread():
    """Shipped simulate handler must offload via asyncio.to_thread (H4)."""
    src = inspect.getsource(simulate_route)
    assert "to_thread" in src
    # Static: frontline module source
    path = Path(__file__).resolve().parents[2] / "src/api/routes/frontline.py"
    text = path.read_text(encoding="utf-8")
    assert "asyncio.to_thread" in text
    assert "_simulate_in_thread" in text


@pytest.mark.asyncio
async def test_console_broadcast_frame_shapes(reset_ops_db, pack):
    """_WSHooks (or broadcast helper) emit frames LiveConsole already handles."""
    received: list[dict] = []

    async def capture(msg):
        received.append(msg)

    # Drive broadcast directly with the same payloads hooks send
    orig = ir._broadcast_console

    async def fake_broadcast(msg):
        received.append(msg)

    ir._broadcast_console = fake_broadcast  # type: ignore
    try:
        from src.channels.base import ChannelAdapter, ChannelCapabilities
        from src.channels.web_voice import WebVoiceChannel

        class DummyWS:
            async def send_json(self, obj):
                pass

        # Minimal channel stand-in. Subclasses ChannelAdapter on purpose: hooks
        # must work against the declared contract, not one adapter's privates.
        class Chan(ChannelAdapter):
            def capabilities(self):
                return ChannelCapabilities()

            async def send_turn(self, *a, **k):
                pass

            async def send_activity(self, *a, **k):
                pass

            async def send_slots_update(self, *a, **k):
                pass

            async def send_handoff_offer(self, *a, **k):
                pass

            async def send_interaction_ended(self, *a, **k):
                pass

            async def hangup(self, *a, **k):
                pass

        hooks = ir._WSHooks(Chan(), "int_fanout_test")  # type: ignore
        await hooks.emit_customer_turn("hello", {"speaker": "agent"})
        await hooks.emit_slots_update({"entity_1": "2019"})
        await hooks.emit_frustration_update(0.42)
    finally:
        ir._broadcast_console = orig  # type: ignore

    types = {m.get("type") for m in received}
    assert "agent_turn" in types
    assert "slots_update" in types
    assert "frustration_update" in types
    turn = next(m for m in received if m["type"] == "agent_turn")
    assert turn["interaction_id"] == "int_fanout_test"
    assert turn["text"] == "hello"
    fr = next(m for m in received if m["type"] == "frustration_update")
    assert fr["value"] == 0.42


@pytest.mark.asyncio
async def test_max_turns_forces_wrap_up(reset_ops_db, seed_automotive_pack, pack, monkeypatch):
    """When max_turns is hit with incomplete slots, orchestrator closes the case."""
    from src.agents import intake as intake_mod
    from types import SimpleNamespace
    from datetime import datetime, timezone

    # Settings is frozen; rebind the intake module reference only.
    monkeypatch.setattr(
        intake_mod,
        "settings",
        SimpleNamespace(max_turns=2),
    )

    iid = "int_max_" + new_ulid()[:8]
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO interactions
            (interaction_id, pack_id, pack_version, started_at, channel, status, supervised, llm_calls)
            VALUES (?, ?, 't', ?, 'web_text', 'active', FALSE, 0)
            """,
            [iid, pack.id, datetime.now(timezone.utc)],
        )

    orch = Orchestrator(iid, pack, channel="web_text", hooks=OrchestratorHooks())
    await orch.start()
    # Vague turns that do not fill all slots
    await orch.handle_customer_turn("um hello", final=True)
    await orch.handle_customer_turn("not sure really", final=True)
    # Third customer turn should hit max_turns and wrap
    await orch.handle_customer_turn("still thinking", final=True)

    assert orch.ctx.state in ("DONE", "CLOSING", "ABANDONED") or orch.ctx.case_id is not None
    # Prefer case created via wrap-up
    if orch.ctx.case_id:
        with ops_con(read_only=True) as con:
            row = con.execute(
                "SELECT status FROM cases WHERE case_id = ?", [orch.ctx.case_id]
            ).fetchone()
        assert row is not None


@pytest.mark.asyncio
async def test_control_frames_survive_non_websocket_channels(reset_ops_db, pack):
    """Control frames must not crash channels without a frame surface.

    `_send` exists only on WebVoiceChannel; emitting frustration/latency/consent
    over SMS, email or telephony used to raise AttributeError. That error was
    built *before* `_to_customer` ran, so its try/except could not absorb it.
    """
    from src.channels.messaging import SmsChannel

    orig = ir._broadcast_console
    received: list[dict] = []

    async def fake_broadcast(msg):
        received.append(msg)

    ir._broadcast_console = fake_broadcast  # type: ignore
    try:
        chan = SmsChannel(to_address="+15550001111")
        hooks = ir._WSHooks(chan, "int_sms_control")  # type: ignore
        await hooks.emit_frustration_update(0.7)
        await hooks.emit_turn_latency({"ms": 120})
        await hooks.emit_consent_required({"reason": "recording"})
    finally:
        ir._broadcast_console = orig  # type: ignore

    # Console still gets every frame; the SMS channel simply ignores them.
    assert {m["type"] for m in received} == {
        "frustration_update",
        "turn_latency",
        "consent_required",
    }
