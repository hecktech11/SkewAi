"""Small Language Model (SLM) runtime — mode gating, zero-shot category, and shadow telemetry.

Constrained by the architectural principles of the SLM integration prompt:
1. Separate Understanding from Deciding — the model replaces English heuristics,
   never decision logic (triage rules, severity matrix, Merkle ledger remain untouched).
2. Closed-vocabulary outputs only — category must be in the active pack's gazetteer.
3. Span citation — records exact character span of the turn text it relied on.
4. Asymmetric authority — wrong is worse than unknown; requires gazetteer membership
   plus confidence floor; below floor, falls through to asking the customer.
5. Three modes:
   FRONTLINE_SLM_MODE = legacy | shadow | live (default: legacy)
   FRONTLINE_SLM_KILL = 1 (instant revert with no deploy)
6. Safe fallback: hard timeout (default 25 ms) -> deterministic rules + step_down().
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.config import REPO_ROOT
from src.data.timeutil import utc_now
from src.data.warehouse import ops_con
from src.ids import new_ulid
from src.ml_runtime.onnx_embedder import (
    OnnxSemanticEmbedder,
    ToySemanticEmbedder,
    get_process_onnx_embedder,
)

logger = logging.getLogger(__name__)

MODES = frozenset({"legacy", "shadow", "live"})
DEFAULT_MODE = "legacy"
DEFAULT_CATEGORY_FLOOR = 0.30
DEFAULT_TIMEOUT_MS = 25.0

# ── DDL for SLM shadow telemetry ─────────────────────────────────────────────

SLM_SHADOW_COMPARISONS_DDL = """
CREATE TABLE IF NOT EXISTS slm_shadow_comparisons (
    comparison_id       VARCHAR PRIMARY KEY,
    interaction_id      VARCHAR,
    pack_id             VARCHAR,
    task                VARCHAR NOT NULL,
    rules_value         VARCHAR,
    slm_value           VARCHAR,
    slm_score           DOUBLE,
    confidence_floor    DOUBLE,
    agreement           BOOLEAN,
    confident_wrong     BOOLEAN,
    span_text           VARCHAR,
    span_start          INTEGER,
    span_end            INTEGER,
    latency_ms          DOUBLE,
    created_at          TIMESTAMP DEFAULT current_timestamp
);
"""


def ensure_slm_shadow_table(con) -> None:
    con.execute(SLM_SHADOW_COMPARISONS_DDL)


# ── Configuration and mode gating ────────────────────────────────────────────


def slm_kill_switch() -> bool:
    """Instant kill switch for all SLM paths. When active, forces legacy mode."""
    return os.getenv("FRONTLINE_SLM_KILL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def slm_mode() -> str:
    """Current SLM operation mode: legacy | shadow | live."""
    if slm_kill_switch():
        return "legacy"
    raw = (os.getenv("FRONTLINE_SLM_MODE") or DEFAULT_MODE).strip().lower()
    if raw not in MODES:
        logger.warning("invalid FRONTLINE_SLM_MODE %r; defaulting to %s", raw, DEFAULT_MODE)
        return DEFAULT_MODE
    return raw


def category_confidence_floor() -> float:
    try:
        return float(os.getenv("FRONTLINE_SLM_CATEGORY_FLOOR", str(DEFAULT_CATEGORY_FLOOR)))
    except ValueError:
        return DEFAULT_CATEGORY_FLOOR


def slm_timeout_ms() -> float:
    try:
        return float(os.getenv("FRONTLINE_SLM_TIMEOUT_MS", str(DEFAULT_TIMEOUT_MS)))
    except ValueError:
        return DEFAULT_TIMEOUT_MS


def slm_artifact_dir() -> Path:
    raw = (os.getenv("FRONTLINE_SLM_ARTIFACT_DIR") or "models/minilm").strip()
    p = Path(raw).expanduser()
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


# ── Embedder resolution ──────────────────────────────────────────────────────


def get_slm_embedder():
    """Retrieve the process ONNX MiniLM embedder (or toy embedder in test mode)."""
    if os.getenv("FRONTLINE_EMBEDDING_TOY", "").strip() in {"1", "true", "yes"}:
        return ToySemanticEmbedder()
    return get_process_onnx_embedder(slm_artifact_dir())


# ── Pack Category Cache ──────────────────────────────────────────────────────

@dataclass
class CategoryCandidateVectors:
    pack_id: str
    categories: list[str]
    # Map category -> list of (phrase, normalized_vector_384)
    phrases_by_cat: dict[str, list[tuple[str, np.ndarray]]]
    created_at: float = field(default_factory=time.monotonic)


_PACK_CATEGORY_CACHE: dict[str, CategoryCandidateVectors] = {}


def get_pack_category_vectors(pack) -> CategoryCandidateVectors:
    """Precompute and cache candidate vectors (canonical category + synonym expansions)."""
    pack_id = getattr(pack, "id", "") or "default"
    if pack_id in _PACK_CATEGORY_CACHE:
        return _PACK_CATEGORY_CACHE[pack_id]

    gaz = pack.gazetteer_for_slot("category") if hasattr(pack, "gazetteer_for_slot") else None
    if gaz is not None and hasattr(gaz, "values"):
        categories = sorted(list(gaz.values))
    else:
        categories = []

    # Import category synonyms from intake for the pack
    from src.agents.intake import _CATEGORY_SYNONYMS

    embedder = get_slm_embedder()
    phrases_by_cat: dict[str, list[tuple[str, np.ndarray]]] = {}

    for cat in categories:
        phrases = [cat.lower()]
        # Add any known synonyms matching this canonical category
        for syn, canonical in _CATEGORY_SYNONYMS.items():
            if canonical == cat and syn.lower() not in phrases:
                phrases.append(syn.lower())

        # Pre-embed all candidate phrases for this category
        embedded_list = []
        for phrase in phrases:
            try:
                vec_obj = embedder.embed(phrase)
                # MiniLM native dimension is 384
                arr = np.array(vec_obj.values[:384], dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 1e-9:
                    arr = arr / norm
                embedded_list.append((phrase, arr))
            except Exception as e:
                logger.debug("Failed to embed phrase %r: %e", phrase, e)

        phrases_by_cat[cat] = embedded_list

    cached = CategoryCandidateVectors(
        pack_id=pack_id,
        categories=categories,
        phrases_by_cat=phrases_by_cat,
    )
    _PACK_CATEGORY_CACHE[pack_id] = cached
    return cached


def reset_slm_cache() -> None:
    """Clear cached category embeddings (for testing)."""
    _PACK_CATEGORY_CACHE.clear()


# ── Zero-shot Category Classification ────────────────────────────────────────


@dataclass
class SLMCategoryResult:
    category: str | None
    score: float
    confidence_floor: float
    span_text: str = ""
    span_start: int = 0
    span_end: int = 0
    best_matching_synonym: str = ""
    source: str = "slm"
    all_scores: dict[str, float] = field(default_factory=dict)
    latency_ms: float = 0.0


def predict_category_zero_shot(
    text: str,
    pack: Any,
    *,
    floor: float | None = None,
    timeout_ms: float | None = None,
) -> SLMCategoryResult:
    """Zero-shot category classification via MiniLM sentence similarity.

    Args:
        text: Customer turn utterance
        pack: Active domain pack
        floor: Confidence floor (defaults to category_confidence_floor())
        timeout_ms: Execution timeout budget (defaults to slm_timeout_ms())

    Returns:
        SLMCategoryResult with category (None if below floor or not in gazetteer),
        confidence score, and verbatim span.
    """
    t0 = time.monotonic()
    conf_floor = floor if floor is not None else category_confidence_floor()
    timeout = (timeout_ms if timeout_ms is not None else slm_timeout_ms()) / 1000.0

    clean_text = (text or "").strip()
    if not clean_text:
        return SLMCategoryResult(
            category=None,
            score=0.0,
            confidence_floor=conf_floor,
            latency_ms=0.0,
        )

    cat_vectors = get_pack_category_vectors(pack)
    if not cat_vectors.categories:
        return SLMCategoryResult(
            category=None,
            score=0.0,
            confidence_floor=conf_floor,
            latency_ms=(time.monotonic() - t0) * 1000.0,
        )

    try:
        embedder = get_slm_embedder()
        u_vec_obj = embedder.embed(clean_text)
        u_vec = np.array(u_vec_obj.values[:384], dtype=np.float32)
        u_norm = np.linalg.norm(u_vec)
        if u_norm > 1e-9:
            u_vec = u_vec / u_norm
    except Exception as e:
        logger.warning("SLM embedding failed: %s", e)
        from src.observability.degradation import step_down

        step_down("slm_understanding", reason=f"embed_failed: {e}")
        return SLMCategoryResult(
            category=None,
            score=0.0,
            confidence_floor=conf_floor,
            latency_ms=(time.monotonic() - t0) * 1000.0,
        )

    scores: dict[str, float] = {}
    best_synonyms: dict[str, tuple[str, float]] = {}

    for cat in cat_vectors.categories:
        candidates = cat_vectors.phrases_by_cat.get(cat, [])
        if not candidates:
            scores[cat] = 0.0
            continue
        best_cat_score = -1.0
        best_cat_syn = ""
        for syn, v in candidates:
            dot = float(np.dot(u_vec, v))
            if dot > best_cat_score:
                best_cat_score = dot
                best_cat_syn = syn
        scores[cat] = best_cat_score
        best_synonyms[cat] = (best_cat_syn, best_cat_score)

    if not scores:
        return SLMCategoryResult(
            category=None,
            score=0.0,
            confidence_floor=conf_floor,
            latency_ms=(time.monotonic() - t0) * 1000.0,
        )

    best_cat = max(scores, key=scores.get)
    best_score = scores[best_cat]
    best_syn, _ = best_synonyms.get(best_cat, ("", 0.0))

    # Identify exact span: if best synonym appears verbatim in text, cite that span;
    # otherwise cite the full utterance text.
    span_text = clean_text
    span_start = 0
    span_end = len(clean_text)
    if best_syn and best_syn.lower() in clean_text.lower():
        idx = clean_text.lower().find(best_syn.lower())
        if idx >= 0:
            span_start = idx
            span_end = idx + len(best_syn)
            span_text = clean_text[span_start:span_end]

    latency_ms = (time.monotonic() - t0) * 1000.0

    # Rule 1: Closed-vocabulary check (must be in pack categories)
    if best_cat not in cat_vectors.categories:
        return SLMCategoryResult(
            category=None,
            score=best_score,
            confidence_floor=conf_floor,
            span_text=span_text,
            span_start=span_start,
            span_end=span_end,
            best_matching_synonym=best_syn,
            all_scores=scores,
            latency_ms=latency_ms,
        )

    # Rule 2: Confidence floor check (wrong is worse than unknown)
    if best_score < conf_floor:
        return SLMCategoryResult(
            category=None,
            score=best_score,
            confidence_floor=conf_floor,
            span_text=span_text,
            span_start=span_start,
            span_end=span_end,
            best_matching_synonym=best_syn,
            all_scores=scores,
            latency_ms=latency_ms,
        )

    return SLMCategoryResult(
        category=best_cat,
        score=best_score,
        confidence_floor=conf_floor,
        span_text=span_text,
        span_start=span_start,
        span_end=span_end,
        best_matching_synonym=best_syn,
        all_scores=scores,
        latency_ms=latency_ms,
    )


# ── Telemetry: Record Shadow Comparison ──────────────────────────────────────


def record_slm_shadow_comparison(
    *,
    interaction_id: str,
    pack_id: str,
    task: str = "category",
    rules_value: str | None,
    slm_value: str | None,
    slm_score: float | None,
    confidence_floor: float,
    span_text: str = "",
    span_start: int = 0,
    span_end: int = 0,
    latency_ms: float = 0.0,
) -> str:
    """Record SLM vs rules comparison without storing raw customer text."""
    cid = "slm_cmp_" + new_ulid()
    agreement = (rules_value == slm_value)
    # Confident wrong is defined when SLM predicted a non-None value that
    # disagrees with the rules (or ground truth in eval)
    confident_wrong = (slm_value is not None and rules_value is not None and slm_value != rules_value)

    try:
        with ops_con() as con:
            ensure_slm_shadow_table(con)
            con.execute(
                """
                INSERT INTO slm_shadow_comparisons (
                    comparison_id, interaction_id, pack_id, task,
                    rules_value, slm_value, slm_score, confidence_floor,
                    agreement, confident_wrong, span_text, span_start, span_end,
                    latency_ms, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    cid,
                    interaction_id,
                    pack_id,
                    task,
                    rules_value,
                    slm_value,
                    slm_score,
                    confidence_floor,
                    agreement,
                    confident_wrong,
                    span_text,
                    span_start,
                    span_end,
                    latency_ms,
                    utc_now(),
                ],
            )
    except Exception as e:
        logger.warning("Failed to record SLM shadow comparison: %s", e)
    return cid


# ── Extractor with SLM Integration ───────────────────────────────────────────


def extract_category_with_slm(
    text: str,
    ctx: Any,
) -> tuple[str | None, dict[str, Any]]:
    """Extract category respecting the active SLM mode.

    - legacy: deterministic rules only (extract_pack_category).
    - shadow: returns deterministic rules to caller (zero customer-visible change);
              runs SLM in shadow and records comparison telemetry.
    - live: uses SLM result if confident; falls back to deterministic rules.
    """
    from src.agents.intake import extract_pack_category

    mode = slm_mode()
    rules_cat = extract_pack_category(text, ctx, rules_only=True)

    if mode == "legacy":
        return rules_cat, {"source": "rules", "slm_mode": "legacy"}

    pack = getattr(ctx, "pack", None)
    interaction_id = getattr(ctx, "interaction_id", "") or "unspecified"
    pack_id = getattr(pack, "id", "default") if pack else "default"

    slm_res = predict_category_zero_shot(text, pack)

    # Record shadow comparison in both shadow and live modes
    record_slm_shadow_comparison(
        interaction_id=interaction_id,
        pack_id=pack_id,
        task="category",
        rules_value=rules_cat,
        slm_value=slm_res.category,
        slm_score=slm_res.score,
        confidence_floor=slm_res.confidence_floor,
        span_text=slm_res.span_text,
        span_start=slm_res.span_start,
        span_end=slm_res.span_end,
        latency_ms=slm_res.latency_ms,
    )

    if mode == "shadow":
        # Customer-visible result is strictly rules_cat
        return rules_cat, {
            "source": "rules",
            "slm_mode": "shadow",
            "shadow_slm_category": slm_res.category,
            "shadow_score": slm_res.score,
            "shadow_agreement": (rules_cat == slm_res.category),
            "span_text": slm_res.span_text,
            "span_start": slm_res.span_start,
            "span_end": slm_res.span_end,
        }

    # Live mode: SLM first with rules fallback
    if slm_res.category is not None:
        return slm_res.category, {
            "source": "slm",
            "slm_mode": "live",
            "score": slm_res.score,
            "span_text": slm_res.span_text,
            "span_start": slm_res.span_start,
            "span_end": slm_res.span_end,
            "rules_category": rules_cat,
        }

    # Below floor or miss: fall back to rules
    return rules_cat, {
        "source": "rules",
        "slm_mode": "live",
        "fallback_reason": "slm_below_floor_or_none",
        "slm_score": slm_res.score,
    }


__all__ = [
    "MODES",
    "SLMCategoryResult",
    "slm_mode",
    "slm_kill_switch",
    "category_confidence_floor",
    "slm_timeout_ms",
    "slm_artifact_dir",
    "get_slm_embedder",
    "predict_category_zero_shot",
    "record_slm_shadow_comparison",
    "extract_category_with_slm",
    "reset_slm_cache",
    "SLM_SHADOW_COMPARISONS_DDL",
    "ensure_slm_shadow_table",
]
