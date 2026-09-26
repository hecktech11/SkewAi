"""Degradation ladder locking contract (R31).

The ladder guards its level with a non-reentrant lock, so every public method
must finish without re-entering that lock. These tests run each call in a
daemon thread with a bounded join: a self-deadlock shows up as a failed
assertion instead of hanging the suite.
"""
from __future__ import annotations

import threading

import pytest

from src.observability.degradation import (
    DegradationLadder,
    DegradationLevel,
    build_llm_ladder,
    build_model_ladder,
    build_semantic_search_ladder,
    build_anchor_ladder,
    build_external_sources_ladder,
    build_slm_ladder,
)

JOIN_TIMEOUT = 2.0


def _call_bounded(fn, *args, **kwargs):
    """Run fn in a daemon thread; fail if it does not return in time."""
    box: dict[str, object] = {}

    def runner():
        try:
            box["value"] = fn(*args, **kwargs)
        except BaseException as exc:  # pragma: no cover - surfaced below
            box["error"] = exc

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join(JOIN_TIMEOUT)
    assert not t.is_alive(), f"{getattr(fn, '__name__', fn)} did not return within {JOIN_TIMEOUT}s (deadlock)"
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box.get("value")


def _bottom_out(ladder: DegradationLadder) -> None:
    """Walk the ladder down to its lowest rung."""
    max_level = max((l.level for l in ladder.levels), default=0)
    for _ in range(max_level):
        _call_bounded(ladder.degrade, reason="setup_step")
    assert ladder.current_level == max_level


def test_degrade_at_bottom_returns_current_without_deadlock():
    ladder = build_llm_ladder()
    _bottom_out(ladder)

    level = _call_bounded(ladder.degrade, reason="bottomed_out")

    assert isinstance(level, DegradationLevel)
    assert level.level == 3
    assert level.name == "fail_closed"
    assert ladder.current_level == 3


def test_repeated_degrade_at_bottom_stays_terminal():
    ladder = build_model_ladder()
    _bottom_out(ladder)

    for _ in range(5):
        level = _call_bounded(ladder.degrade, reason="still_failing")
        assert level is not None
        assert level.level == 3
    assert ladder.current_level == 3
    assert not ladder.is_operational()


def test_degrade_on_empty_ladder_returns_none():
    ladder = DegradationLadder(subsystem="empty")

    assert _call_bounded(ladder.degrade, reason="no_levels") is None
    assert ladder.current_level == 0


@pytest.mark.parametrize(
    "builder",
    [
        build_llm_ladder,
        build_model_ladder,
        build_semantic_search_ladder,
        build_anchor_ladder,
        build_external_sources_ladder,
        build_slm_ladder,
    ],
)
def test_every_registered_subsystem_survives_bottom_level_degrade(builder):
    ladder = builder()
    _bottom_out(ladder)

    level = _call_bounded(ladder.degrade, reason="bottom")

    assert level is not None
    assert level.level == max(l.level for l in ladder.levels)


def test_status_and_reset_are_reachable_after_bottom_degrade():
    ladder = build_semantic_search_ladder()
    _bottom_out(ladder)
    _call_bounded(ladder.degrade, reason="bottom")

    status = _call_bounded(ladder.status)
    assert status["current_level"] == 3
    assert status["level_name"] == "no_search"
    # Setup steps are logged; the bottomed-out call adds no transition.
    assert len(status["degradation_log"]) == 3

    _call_bounded(ladder.reset)
    assert ladder.current_level == 0
    assert ladder.is_operational()


def test_concurrent_degrade_status_reset_do_not_stall():
    ladder = build_anchor_ladder()
    _bottom_out(ladder)
    errors: list[BaseException] = []
    stop = threading.Event()

    def worker(op):
        try:
            while not stop.is_set():
                op()
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=(lambda: ladder.degrade(reason="churn"),), daemon=True),
        threading.Thread(target=worker, args=(ladder.status,), daemon=True),
        threading.Thread(target=worker, args=(ladder.is_operational,), daemon=True),
    ]
    for t in threads:
        t.start()
    stop.wait(0.5)
    stop.set()
    for t in threads:
        t.join(JOIN_TIMEOUT)
        assert not t.is_alive(), "concurrent ladder access stalled"
    assert not errors, f"concurrent ladder access raised: {errors}"
