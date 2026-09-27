"""Shared fixtures for the Frontline v2 test suite.

Every test that writes to the ops warehouse should depend on the `reset_ops_db`
fixture so the warehouse starts empty. Tests that exercise Sentinel / Investigator
agents (which read the domain warehouse) should also depend on `seed_automotive_pack`.

All fixtures are deterministic and LLM-free — the test suite never calls an LLM
provider, even when one is configured.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from src.agents.base import InteractionContext
from src.agents.orchestrator import Orchestrator, OrchestratorHooks
from src.config import REPO_ROOT
from src.data.warehouse import init_ops_db as _init_ops_db, reset_ops_db as _reset_ops_db_impl
from src.domains.loader import load_pack
from src.ids import new_ulid


# ── Isolate ops/domain DBs from pilot data/ (CRITICAL: no wipe of pilot DBs) ─


@pytest.fixture(scope="session", autouse=True)
def _isolate_test_databases(tmp_path_factory):
    """Point FRONTLINE_DB_PATH / DOMAIN_DB_PATH at a session temp tree.

    Prevents ``reset_ops_db`` / domain seed rebuilds from deleting the
    developer's ``data/frontline.duckdb`` or ``data/domains/*.duckdb``.
    """
    root = tmp_path_factory.mktemp("skewai_pytest_data")
    ops = root / "frontline.duckdb"
    domains = root / "domains"
    domains.mkdir(parents=True, exist_ok=True)
    os.environ["FRONTLINE_TEST_ISOLATION"] = "1"
    os.environ["FRONTLINE_DB_PATH"] = str(ops)
    os.environ["DOMAIN_DB_PATH"] = str(domains)
    os.environ["OIDC_LOCAL_PATH"] = str(root / "oidc_local.json")
    # Pack marketplace + evidence lockers write JSON; keep those off the git tree.
    dest_reg = root / "registry.json"
    src_reg = REPO_ROOT / "domains" / "registry.json"
    if src_reg.is_file():
        dest_reg.write_text(src_reg.read_text(encoding="utf-8"), encoding="utf-8")
    os.environ["PACK_REGISTRY_PATH"] = str(dest_reg)
    lockers = root / "lockers"
    lockers.mkdir(parents=True, exist_ok=True)
    os.environ["QUBOT_LOCKER_DIR"] = str(lockers)
    os.environ.setdefault("SESSION_SECRET", "test_secret_key_frontline_v2_12345")
    os.environ.setdefault("FRONTLINE_ALLOW_CALL_HASH_OVERRIDE", "1")
    # Never allow accidental wipe of pilot path during tests.
    os.environ.pop("FRONTLINE_ALLOW_DEFAULT_DB_RESET", None)
    yield root


# ── Hermetic DNS for outbound URL validation ─────────────────────────────────


@pytest.fixture(scope="session", autouse=True)
def _hermetic_url_resolver():
    """Stub DNS so connector/url_guard tests do not need live network.

    Public-looking hosts resolve to TEST-NET (192.0.2.1); private/metadata
    literals still fail via the guard's IP/host rules without DNS.
    """
    import socket

    from src.security.url_guard import set_url_resolver

    def _fake_resolve(host: str, port: int):
        h = (host or "").lower()
        # Still surface private names as blocked by resolving to private? No —
        # leave hostname blocklist in the guard; only public hostnames get a public IP.
        if h in {"localhost", "127.0.0.1", "::1", "metadata", "metadata.google.internal"}:
            raise socket.gaierror(8, "nodename nor servname provided, or not known")
        # Well-known public resolver IP (not private/reserved per ipaddress).
        sockaddr = ("8.8.8.8", port)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", sockaddr)]

    set_url_resolver(_fake_resolve)
    yield
    set_url_resolver(None)


# ── Database fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def reset_ops_db():
    """Drop and recreate the ops warehouse before each test that writes to it."""
    _reset_ops_db_impl()
    _init_ops_db()
    yield
    # Leave the DB in place for inspection on failure; the next test resets it.


@pytest.fixture
def seed_automotive_pack():
    """Build the automotive_nhtsa domain fixture (records + advisories + clusters).

    Idempotent: rebuilds the warehouse from scratch every time so tests see a
    deterministic corpus.
    """
    from scripts.seed_domains import build as build_domain

    path = build_domain("automotive_nhtsa")
    return path


@pytest.fixture
def pack(seed_automotive_pack):
    """The loaded automotive_nhtsa pack (cached by load_pack).

    Depends on `seed_automotive_pack` so the domain warehouse is present.
    """
    # `reload=True` so a stale cache from another test can't leak pack_version.
    return load_pack("automotive_nhtsa", reload=True)


# ── Orchestrator factory ────────────────────────────────────────────────────────


class _RecordingHooks(OrchestratorHooks):
    """No-op hooks that capture every emit for assertions.

    Mirrors the SimulatedChannel's recording behaviour but wired directly to the
    orchestrator's hook surface, so tests can drive the orchestrator without
    going through a channel adapter.
    """

    # NOTE: do NOT call super().__init__() — OrchestratorHooks is a @dataclass
    # whose __init__ sets every hook field to None, which would shadow the
    # methods defined below. The same trick is used by StdioHooks in cli.py.

    def __init__(self) -> None:
        self.turns: list[dict[str, Any]] = []
        self.activities: list[dict[str, Any]] = []
        self.slots_updates: list[dict[str, str]] = []
        self.handoff_offers: int = 0
        self.ended: dict[str, Any] | None = None

    async def emit_customer_turn(self, text: str, meta: dict[str, Any]) -> None:
        self.turns.append({"text": text, "speaker": meta.get("speaker", "agent"), "meta": meta})

    async def emit_activity(self, payload: dict[str, Any]) -> None:
        self.activities.append(dict(payload))

    async def emit_slots_update(self, slots: dict[str, str]) -> None:
        # Strip the private __safety_questions_asked__ key if present.
        self.slots_updates.append({k: v for k, v in slots.items() if not k.startswith("__")})

    async def emit_handoff_offer(self) -> None:
        self.handoff_offers += 1

    async def emit_interaction_ended(self, payload: dict[str, Any]) -> None:
        self.ended = dict(payload)

    # ── Assertion helpers ────────────────────────────────────────────────────
    def agent_texts(self) -> list[str]:
        return [t["text"] for t in self.turns if t["speaker"] == "agent"]

    def supervisor_texts(self) -> list[str]:
        return [t["text"] for t in self.turns if t["speaker"] == "supervisor"]

    def full_transcript(self) -> str:
        return "\n".join(f"[{t['speaker']}] {t['text']}" for t in self.turns)


@pytest_asyncio.fixture
async def orchestrator_factory(pack, reset_ops_db):
    """Factory that builds a fresh Orchestrator + recording hooks.

    Returns a callable `factory(interaction_id=None, channel="web_text")`
    which returns `(orch, hooks)`. The orchestrator is NOT started — tests
    that need the greeting should call `await orch.start()`.
    """

    def _factory(
        interaction_id: str | None = None,
        channel: str = "web_text",
    ) -> tuple[Orchestrator, _RecordingHooks]:
        iid = interaction_id or ("int_test_" + new_ulid())
        hooks = _RecordingHooks()
        # Insert a minimal interactions row so the orchestrator's state writes
        # (UPDATE interactions SET ...) don't fail when the row is missing.
        from datetime import datetime, timezone
        from src.data.warehouse import ops_con

        with ops_con() as con:
            con.execute(
                """
                INSERT INTO interactions
                (interaction_id, pack_id, pack_version, started_at, channel,
                 status, supervised, llm_calls)
                VALUES (?, ?, ?, ?, ?, 'active', FALSE, 0)
                """,
                [iid, pack.id, pack.pack_version, datetime.now(timezone.utc), channel],
            )
        # Mirror what `create_interaction` does so tests that assert on the
        # ledger (e.g. greeting row presence) see the same action shape.
        from src.ledger import AgentAction, record_action
        record_action(AgentAction(
            interaction_id=iid,
            agent="orchestrator",
            action_type="interaction_started",
            input_summary=f"pack_id={pack.id}",
            output_summary="interaction created",
        ))
        orch = Orchestrator(iid, pack, channel=channel, hooks=hooks)
        return orch, hooks

    return _factory


# ── Asyncio config ────────────────────────────────────────────────────────────


@pytest.fixture
def event_loop():
    """Fresh event loop per test (pytest-asyncio >= 0.23 supports this)."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
