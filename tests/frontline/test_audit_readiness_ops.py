"""Audit findings: Twilio connect limit, spend remainder, unsupported ops DSNs."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest


def test_spend_reservation_rejects_when_full_cost_exceeds_remainder(
    reset_ops_db, monkeypatch
):
    """The reserved amount is checked against what is still left, then recorded."""
    import dataclasses

    from src.ai.provider import _day_key, _reserve_spend, get_spend
    from src.config import settings
    from src.data.timeutil import utc_now
    from src.data.warehouse import ops_con

    monkeypatch.setenv("FRONTLINE_LLM_ENABLED", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-ant-test-remainder")
    monkeypatch.setattr(
        "src.ai.provider.settings",
        dataclasses.replace(settings, max_daily_claude_cost=2.0, llm_turn_cap=6),
    )
    day = _day_key()
    with ops_con() as con:
        con.execute(
            """
            INSERT INTO llm_spend (day_key, turn_count, cost_units, updated_at)
            VALUES (?, 1, 1.0, ?)
            """,
            [day, utc_now()],
        )

    ok, reason = _reserve_spend(1.5)
    assert ok is False
    assert reason == "daily_cost_cap"
    assert get_spend(day)["cost_units"] == 1.0

    ok_fit, reason_fit = _reserve_spend(1.0)
    assert ok_fit is True
    assert reason_fit == "ok"
    assert get_spend(day)["cost_units"] == 2.0


@pytest.mark.asyncio
async def test_twilio_connection_limiter_rejects_over_budget(monkeypatch):
    """The Twilio socket uses the production connect limiter, not a stand-in."""
    from src.api.limiter import _WS_CONNECT_LIMIT, _ws_hits
    from src.api.routes.interactions import twilio_ws

    monkeypatch.setenv("FRONTLINE_ENABLED", "1")
    host = "203.0.113.55"
    key = f"twilio:{host}"
    _ws_hits[key] = [time.time()] * _WS_CONNECT_LIMIT
    try:
        ws = AsyncMock()
        ws.client = type("Client", (), {"host": host})()
        ws.close = AsyncMock()
        await twilio_ws(ws, "int_twilio_real_limit")
        ws.close.assert_awaited_once_with(code=1013)
        ws.accept.assert_not_called()
    finally:
        _ws_hits.pop(key, None)


def _assert_dsn_rejected_before_open(monkeypatch, env_name: str, value: str) -> None:
    import duckdb

    from src.data.postgres_backend import ops_connection
    from src.data.warehouse import init_ops_db, ops_con
    from src.jobs.registry import prune_stale_workers

    for name in ("FRONTLINE_OPS_DSN", "DATABASE_URL", "POSTGRES_DSN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(env_name, value)
    opened: list[object] = []
    real_connect = duckdb.connect

    def _wrap(*args, **kwargs):
        opened.append(args)
        return real_connect(*args, **kwargs)

    monkeypatch.setattr("src.data.warehouse.duckdb.connect", _wrap)
    with pytest.raises(RuntimeError, match=env_name):
        with ops_con():
            pass
    with pytest.raises(RuntimeError, match=env_name):
        with ops_connection():
            pass
    with pytest.raises(RuntimeError, match=env_name):
        init_ops_db()
    with pytest.raises(RuntimeError, match=env_name):
        prune_stale_workers()
    assert opened == []


def test_database_url_rejected_before_duckdb_opens(monkeypatch):
    _assert_dsn_rejected_before_open(monkeypatch, "DATABASE_URL", "snowflake://acct/ops")


def test_postgres_dsn_rejected_before_duckdb_opens(monkeypatch):
    _assert_dsn_rejected_before_open(
        monkeypatch, "POSTGRES_DSN", "postgresql://localhost/ops"
    )


def test_blank_ops_dsns_do_not_select_postgres(monkeypatch):
    from src.data.postgres_backend import unsupported_ops_dsn

    for name in ("FRONTLINE_OPS_DSN", "DATABASE_URL", "POSTGRES_DSN"):
        monkeypatch.setenv(name, "   ")
    assert unsupported_ops_dsn() is None
