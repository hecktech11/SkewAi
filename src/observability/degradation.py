"""Degradation ladders — explicit, tested, visible on console.

Each critical subsystem (LLM, model inference, semantic search, ledger anchor,
external data sources) has a DegradationLadder with named levels:

    Level 0: Full capability (primary path)
    Level 1: Reduced capability (first fallback)
    Level 2: Minimal capability (safety minimum)
    Level 3: Fail-closed (cannot proceed safely)

The ladder is monotonic-down during a single contact: once degraded, the system
stays at that level or goes lower, never auto-recovers mid-contact (that would
violate auditability). Recovery happens between contacts.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class DegradationLevel:
    """One rung on a degradation ladder."""
    level: int
    name: str
    description: str
    fallback_fn: Callable[..., Any] | None = None  # the function to call at this level


@dataclass
class DegradationLadder:
    """A named subsystem's fallback chain."""
    subsystem: str
    levels: list[DegradationLevel] = field(default_factory=list)
    _current_level: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _degradation_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def current_level(self) -> int:
        with self._lock:
            return self._current_level

    @property
    def current(self) -> DegradationLevel | None:
        with self._lock:
            for lev in self.levels:
                if lev.level == self._current_level:
                    return lev
        return None

    def degrade(self, *, reason: str) -> DegradationLevel | None:
        """Move to the next lower level. Monotonic-down within a contact."""
        with self._lock:
            next_level = self._current_level + 1
            target = None
            for lev in self.levels:
                if lev.level == next_level:
                    target = lev
                    break
            if target is None:
                # Already at bottom
                logger.warning(
                    "degradation_ladder_bottomed_out",
                    extra={"subsystem": self.subsystem, "reason": reason},
                )
                return self.current
            self._current_level = next_level
            self._degradation_log.append({
                "subsystem": self.subsystem,
                "from_level": next_level - 1,
                "to_level": next_level,
                "level_name": target.name,
                "reason": reason,
                "ts": datetime.utcnow().isoformat(),
            })
            logger.warning(
                "degradation_ladder_step",
                extra={
                    "subsystem": self.subsystem,
                    "ladder_level": next_level,
                    "level_name": target.name,
                    "reason": reason,
                },
            )
            return target

    def reset(self) -> None:
        """Reset to level 0 (between contacts only)."""
        with self._lock:
            self._current_level = 0

    def status(self) -> dict[str, Any]:
        """Console-visible status."""
        cur = self.current
        return {
            "subsystem": self.subsystem,
            "current_level": self._current_level,
            "level_name": cur.name if cur else "unknown",
            "description": cur.description if cur else "",
            "max_level": max((l.level for l in self.levels), default=0),
            "degradation_log": list(self._degradation_log),
        }

    def is_operational(self) -> bool:
        """True if not at fail-closed."""
        with self._lock:
            max_lev = max((l.level for l in self.levels), default=0)
            return self._current_level < max_lev


# ── Pre-built ladders for core subsystems ────────────────────────────────

def build_llm_ladder() -> DegradationLadder:
    """LLM narration fallback chain."""
    return DegradationLadder(
        subsystem="llm_narration",
        levels=[
            DegradationLevel(0, "full", "Primary LLM provider responds normally"),
            DegradationLevel(1, "cached_templates", "LLM timeout/error; use cached response templates"),
            DegradationLevel(2, "rule_based", "Template engine; no LLM; rule-based responses only"),
            DegradationLevel(3, "fail_closed", "Cannot generate safe response; escalate to supervisor"),
        ],
    )


def build_model_ladder() -> DegradationLadder:
    """ML severity model fallback chain."""
    return DegradationLadder(
        subsystem="severity_model",
        levels=[
            DegradationLevel(0, "full", "ML severity model returns prediction"),
            DegradationLevel(1, "rule_severity", "Model failed; use rules-based severity scoring"),
            DegradationLevel(2, "safe_default", "Rules failed; default to Medium severity"),
            DegradationLevel(3, "fail_closed", "Cannot score; escalate for human review"),
        ],
    )


def build_semantic_search_ladder() -> DegradationLadder:
    """Semantic similarity search fallback."""
    return DegradationLadder(
        subsystem="semantic_search",
        levels=[
            DegradationLevel(0, "full", "Embedding + cosine similarity search"),
            DegradationLevel(1, "keyword_search", "Embedding failed; fall back to keyword/TF-IDF match"),
            DegradationLevel(2, "category_only", "Keyword failed; match by category + entity only"),
            DegradationLevel(3, "no_search", "Cannot search corpus; skip similar-case enrichment"),
        ],
    )


def build_anchor_ladder() -> DegradationLadder:
    """Merkle anchor (trust anchor) fallback."""
    return DegradationLadder(
        subsystem="merkle_anchor",
        levels=[
            DegradationLevel(0, "full", "Merkle tree anchored to external timestamping service"),
            DegradationLevel(1, "local_merkle", "External anchor unavailable; local Merkle tree only"),
            DegradationLevel(2, "hash_chain_only", "Merkle tree degraded; hash chain still intact"),
            DegradationLevel(3, "wal_fallback", "DB down; safety WAL fallback active"),
        ],
    )


def build_external_sources_ladder() -> DegradationLadder:
    """External data source (NHTSA / CFPB / advisory feeds) fallback."""
    return DegradationLadder(
        subsystem="external_sources",
        levels=[
            DegradationLevel(0, "full", "All external feeds responding"),
            DegradationLevel(1, "cached", "Feed timeout; using cached corpus (age noted in UI)"),
            DegradationLevel(2, "stale_warning", "Corpus > 7 days old; stale warning shown"),
            DegradationLevel(3, "offline", "No external data; frontline-only corpus"),
        ],
    )


def build_slm_ladder() -> DegradationLadder:
    """SLM understanding fallback chain (replaces English heuristics with safe step-down)."""
    return DegradationLadder(
        subsystem="slm_understanding",
        levels=[
            DegradationLevel(0, "full", "Primary SLM encoder returns predictions"),
            DegradationLevel(1, "rules_fallback", "SLM timeout or error; fall back to deterministic regex rules"),
            DegradationLevel(2, "fail_closed", "Cannot extract; fall through to asking caller"),
        ],
    )


# ── Global registry ─────────────────────────────────────────────────────

_REGISTRY: dict[str, DegradationLadder] = {}
_reg_lock = threading.Lock()


def register_ladder(ladder: DegradationLadder) -> None:
    """Register a ladder in the global registry."""
    with _reg_lock:
        _REGISTRY[ladder.subsystem] = ladder


def get_ladder(subsystem: str) -> DegradationLadder | None:
    """Get a registered ladder by subsystem name."""
    with _reg_lock:
        return _REGISTRY.get(subsystem)


def all_ladders_status() -> list[dict[str, Any]]:
    """Console-visible status for all registered ladders."""
    with _reg_lock:
        return [ladder.status() for ladder in _REGISTRY.values()]


def reset_all_ladders() -> None:
    """Reset all ladders (between contacts)."""
    with _reg_lock:
        for ladder in _REGISTRY.values():
            ladder.reset()


def init_default_ladders() -> None:
    """Register the default set of degradation ladders."""
    for builder in (
        build_llm_ladder,
        build_model_ladder,
        build_semantic_search_ladder,
        build_anchor_ladder,
        build_external_sources_ladder,
        build_slm_ladder,
    ):
        register_ladder(builder())


def step_down(subsystem: str, *, reason: str) -> DegradationLevel | None:
    """Move a named ladder one rung. No-op if the ladder is unknown."""
    if not _REGISTRY:
        init_default_ladders()
    ladder = get_ladder(subsystem)
    if ladder is None:
        return None
    return ladder.degrade(reason=reason)


__all__ = [
    "DegradationLevel",
    "DegradationLadder",
    "build_llm_ladder",
    "build_model_ladder",
    "build_semantic_search_ladder",
    "build_anchor_ladder",
    "build_external_sources_ladder",
    "build_slm_ladder",
    "register_ladder",
    "get_ladder",
    "all_ladders_status",
    "reset_all_ladders",
    "init_default_ladders",
    "step_down",
]
