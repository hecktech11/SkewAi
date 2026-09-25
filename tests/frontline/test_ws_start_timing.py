"""Voice WS start must attach in well under 30s (no lock+hangup stall)."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from src.agents.orchestrator import Orchestrator, create_interaction
from src.api.main import app
from src.api.routes import interactions as ir
from src.data.warehouse import ops_con
from src.domains.active_pack import clear_active_pack_override


@pytest.fixture
def client(reset_ops_db, seed_automotive_pack, monkeypatch):
    monkeypatch.delenv("FRONTLINE_API_KEY", raising=False)
    monkeypatch.setenv("FRONTLINE_ENABLED", "1")
    clear_active_pack_override()
    ir._active.clear()
    with TestClient(app) as c:
        yield c
    ir._active.clear()


def _start(client) -> tuple[str, str, str]:
    r = client.post("/api/interactions/start", params={"channel": "web_text"})
    assert r.status_code == 200
    body = r.json()
    return body["interaction_id"], body["ws_url"], body["greeting_text"]


def _wait_ws_attached(iid: str, *, timeout_s: float = 3.0) -> None:
    """Poll the shipped registry: attach runs after accept() on the WS route."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        entry = ir._active.get(iid)
        if entry is not None and entry.ws_attached:
            return
        time.sleep(0.01)
    entry = ir._active.get(iid)
    attached = getattr(entry, "ws_attached", None)
    raise AssertionError(
        f"ws_attached not True within {timeout_s}s for {iid} (entry={entry!r} attached={attached})"
    )


def test_classify_customer_ws_attach_matrix():
    now = 1_000.0
    grace = 120.0
    assert ir.classify_customer_ws_attach(None, now=now, grace_s=grace) == "missing"

    orch = SimpleNamespace(ctx=SimpleNamespace(state="COLLECTING"))
    entry = ir.ActiveEntry(orch=orch, created_at=now)
    entry.ws_attached = True
    assert ir.classify_customer_ws_attach(entry, now=now, grace_s=grace) == "busy"

    entry.ws_attached = False
    orch.ctx.state = "DONE"
    assert ir.classify_customer_ws_attach(entry, now=now, grace_s=grace) == "ended"
    orch.ctx.state = "ABANDONED"
    assert ir.classify_customer_ws_attach(entry, now=now, grace_s=grace) == "ended"

    orch.ctx.state = "COLLECTING"
    entry.detached_at = now - (grace + 10)
    assert ir.classify_customer_ws_attach(entry, now=now, grace_s=grace) == "not_resumable"

    entry.detached_at = now - 10
    assert ir.classify_customer_ws_attach(entry, now=now, grace_s=grace) == "ok_resume"

    entry.detached_at = None
    assert ir.classify_customer_ws_attach(entry, now=now, grace_s=grace) == "ok_new"


def test_ws_attach_ready_under_3s(client):
    """POST start then WS attach (ws_attached) is usable in <3s on the real route."""
    t_start = time.perf_counter()
    iid, ws_path, greeting = _start(client)
    start_elapsed = time.perf_counter() - t_start
    assert iid.startswith("int_")
    assert ws_path == f"/ws/interaction/{iid}"
    assert greeting
    assert start_elapsed < 3.0, f"POST /start took {start_elapsed:.3f}s (budget 3s)"

    t0 = time.perf_counter()
    with client.websocket_connect(ws_path) as ws:
        _wait_ws_attached(iid, timeout_s=3.0)
        elapsed = time.perf_counter() - t0
        assert elapsed < 3.0, f"WS attach (ws_attached) took {elapsed:.3f}s (budget 3s)"
        ws.send_json({"type": "hangup"})
    assert iid not in ir._active


def test_ws_accepts_user_turn_without_hanging_loop(client):
    """user_turn is accepted on the live socket (no extra receive — TestClient deadlock)."""
    iid, ws_path, _ = _start(client)
    with client.websocket_connect(ws_path) as ws:
        _wait_ws_attached(iid, timeout_s=3.0)
        ws.send_json(
            {"type": "user_turn", "text": "My 2019 Honda CR-V grinds when I brake", "final": True}
        )
    assert iid in ir._active


def test_ws_attach_not_blocked_by_slow_orphan_hangup(client, monkeypatch):
    """Expired contact hangup must not delay start(B) or B's ws_attached flag."""
    iid_a, _, _ = _start(client)
    ir._active[iid_a].created_at = time.monotonic() - (ir._ORPHAN_TTL_S + 10)

    orig = Orchestrator.hangup

    async def slow_hangup(self, *args, **kwargs):
        if getattr(getattr(self, "ctx", None), "interaction_id", None) != iid_a:
            return await orig(self, *args, **kwargs)
        await asyncio.sleep(4.0)
        return await orig(self, *args, **kwargs)

    monkeypatch.setattr(Orchestrator, "hangup", slow_hangup)

    portal = client.portal
    assert portal is not None
    portal.start_task_soon(ir.reap_orphans)
    time.sleep(0.05)

    t_start = time.perf_counter()
    iid_b, ws_path, _ = _start(client)
    start_elapsed = time.perf_counter() - t_start
    assert iid_b != iid_a
    assert start_elapsed < 3.0, f"start(B) waited on orphan hangup: {start_elapsed:.3f}s"

    t0 = time.perf_counter()
    with client.websocket_connect(ws_path) as ws:
        _wait_ws_attached(iid_b, timeout_s=3.0)
        elapsed = time.perf_counter() - t0
        assert elapsed < 3.0, f"B ws_attached blocked by orphan hangup: {elapsed:.3f}s"
        ws.send_json({"type": "hangup"})


def test_reap_does_not_fail_contact_started_during_finalize(client, monkeypatch):
    """A start that registers during orphan hangup must stay status='active'."""
    iid_a, _, _ = _start(client)
    ir._active[iid_a].created_at = time.monotonic() - (ir._ORPHAN_TTL_S + 10)

    sneaky: dict[str, str] = {}
    orig = ir._finalize_reaped

    async def finalize_then_register(dead):
        await orig(dead)
        orch, _greeting = await create_interaction(channel="web_text")
        await ir._register(orch.ctx.interaction_id, orch)
        sneaky["iid"] = orch.ctx.interaction_id

    monkeypatch.setattr(ir, "_finalize_reaped", finalize_then_register)
    portal = client.portal
    assert portal is not None
    portal.call(ir.reap_orphans)

    iid_b = sneaky["iid"]
    assert iid_b in ir._active
    with ops_con(read_only=True) as con:
        row = con.execute(
            "SELECT status FROM interactions WHERE interaction_id = ?", [iid_b]
        ).fetchone()
    assert row is not None
    assert row[0] == "active", f"start-during-reap swept as {row[0]}"
