"""Phase 1 Zero-Shot Category Extraction — Local MiniLM Centroids & Dual Gates.

Implements Task 3 of the SLM Integration Work Order:
1. Label embeddings from domain pack — centroids built from canonical labels,
   gazetteer synonyms (non-numeric aliases), and taxonomy descriptions.
   Cached by (pack_id, embedding_canonical_version).
2. Dual-gate scoring:
   - FRONTLINE_CATEGORY_ZS_FLOOR (default 0.35) — absolute cosine similarity.
   - FRONTLINE_CATEGORY_ZS_MARGIN (default 0.05) — top-1 minus top-2 score.
3. Closed-vocabulary enforcement — reject any label not in pack's gazetteer.
4. Below either gate -> returns None ("Unknown" is cheap; wrong is expensive).
5. Gating & Telemetry:
   - FRONTLINE_SLM_MODE = legacy | shadow | live (default: legacy)
   - FRONTLINE_SLM_KILL = 1 (instant revert to legacy)
   - Shadow telemetry logged to slm_shadow_comparisons (IDs, scores, labels only — no complaint text).
   - Source stamping: rules | zeroshot | rules+zeroshot.
   - Hard 25 ms timeout with graceful step_down().
"""

from __future__ import annotations

import logging
import os
import time
import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_EMBED_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="cat_zs_embed"
)

from src.config import REPO_ROOT
from src.data.warehouse import ops_con
from src.ids import new_ulid
from src.ml_runtime.onnx_embedder import (
    OnnxSemanticEmbedder,
    ToySemanticEmbedder,
    get_process_onnx_embedder,
)
from src.observability.degradation import step_down

logger = logging.getLogger(__name__)

MODES = frozenset({"legacy", "shadow", "live"})
DEFAULT_MODE = "legacy"
DEFAULT_CATEGORY_ZS_FLOOR = 0.35
DEFAULT_CATEGORY_ZS_MARGIN = 0.05
DEFAULT_CATEGORY_ZS_TIMEOUT_MS = 25.0

# ── Configuration Helpers ───────────────────────────────────────────────────


def slm_kill_switch() -> bool:
    """Instant kill switch for SLM paths. When active, forces legacy mode."""
    return os.getenv("FRONTLINE_SLM_KILL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def slm_mode() -> str:
    """Current SLM operation mode: legacy | shadow | live (default: legacy)."""
    if slm_kill_switch():
        return "legacy"
    raw = (os.getenv("FRONTLINE_SLM_MODE") or DEFAULT_MODE).strip().lower()
    if raw not in MODES:
        logger.warning("invalid FRONTLINE_SLM_MODE %r; defaulting to %s", raw, DEFAULT_MODE)
        return DEFAULT_MODE
    return raw


def category_zs_floor() -> float:
    """Absolute similarity floor. Below this, category is considered unknown."""
    try:
        return float(os.getenv("FRONTLINE_CATEGORY_ZS_FLOOR", str(DEFAULT_CATEGORY_ZS_FLOOR)))
    except ValueError:
        return DEFAULT_CATEGORY_ZS_FLOOR


def category_zs_margin() -> float:
    """Top-1 vs top-2 margin gate against confident-wrong assignments."""
    try:
        return float(os.getenv("FRONTLINE_CATEGORY_ZS_MARGIN", str(DEFAULT_CATEGORY_ZS_MARGIN)))
    except ValueError:
        return DEFAULT_CATEGORY_ZS_MARGIN


def category_zs_timeout_ms() -> float:
    """Execution timeout budget in milliseconds (default 25.0 ms)."""
    try:
        return float(os.getenv("FRONTLINE_CATEGORY_ZS_TIMEOUT_MS", str(DEFAULT_CATEGORY_ZS_TIMEOUT_MS)))
    except ValueError:
        return DEFAULT_CATEGORY_ZS_TIMEOUT_MS


def slm_artifact_dir() -> Path:
    raw = (os.getenv("FRONTLINE_SLM_ARTIFACT_DIR") or "models/minilm").strip()
    p = Path(raw).expanduser()
    return p.resolve() if p.is_absolute() else (REPO_ROOT / p).resolve()


def get_category_embedder():
    """Resolve process embedder (MiniLM INT8 ONNX or Toy embedder for test mode)."""
    if os.getenv("FRONTLINE_EMBEDDING_TOY", "").strip() in {"1", "true", "yes"}:
        return ToySemanticEmbedder()
    return get_process_onnx_embedder(slm_artifact_dir())


# ── Category Centroid Cache ─────────────────────────────────────────────────


@dataclass(frozen=True)
class CategoryCentroids:
    pack_id: str
    canonical_version: str
    categories: tuple[str, ...]
    centroids: dict[str, np.ndarray]
    category_set: frozenset[str]


_CENTROIDS_CACHE: dict[tuple[str, str], CategoryCentroids] = {}


def get_category_centroids(pack: Any, embedder: Any | None = None) -> CategoryCentroids:
    """Compute and cache category centroids for the active pack.

    Centroid construction:
      - Canonical category label lowercased
      - Label with underscores converted to spaces
      - Gazetteer aliases from pack (filtering out numeric frequency values)
      - Taxonomy group name associated with the category
    Vectors are L2-normalized, averaged, and re-normalized to unit length.
    """
    pack_id = getattr(pack, "id", "") or "default"
    if embedder is None:
        embedder = get_category_embedder()
    canonical_version = getattr(embedder, "canonical_version", "unknown")
    cache_key = (pack_id, canonical_version)
    if cache_key in _CENTROIDS_CACHE:
        return _CENTROIDS_CACHE[cache_key]

    gaz = pack.gazetteer_for_slot("category") if hasattr(pack, "gazetteer_for_slot") else None
    if gaz is not None and hasattr(gaz, "values"):
        categories = tuple(gaz.values)
    else:
        categories = ()

    category_set = frozenset(categories)
    centroids: dict[str, np.ndarray] = {}

    tax_groups: dict[str, str] = {}
    if hasattr(pack, "taxonomy") and isinstance(pack.taxonomy, dict):
        for g in pack.taxonomy.get("groups", []):
            gname = g.get("name", "")
            for c in g.get("categories", []):
                tax_groups[str(c)] = str(gname)

    for cat in categories:
        phrases: list[str] = [cat.lower()]
        w = cat.lower().replace("_", " ")
        if w not in phrases:
            phrases.append(w)

        if gaz is not None and hasattr(gaz, "canonical"):
            for alias, c in gaz.canonical.items():
                if c == cat and not alias.isdigit() and alias.lower() not in phrases:
                    phrases.append(alias.lower())

        if cat in tax_groups:
            gp = f"{tax_groups[cat].lower()} {w}"
            if gp not in phrases:
                phrases.append(gp)

        vecs: list[np.ndarray] = []
        for p in phrases:
            try:
                emb = embedder.embed(p)
                v = np.array(emb.values[:384], dtype=np.float32)
                norm = float(np.linalg.norm(v))
                if norm > 1e-9:
                    vecs.append(v / norm)
            except Exception as e:
                logger.debug("Failed to embed centroid phrase %r for %s: %s", p, cat, e)

        if vecs:
            c = np.mean(vecs, axis=0)
            cnorm = float(np.linalg.norm(c))
            centroids[cat] = (c / cnorm if cnorm > 1e-9 else c).astype(np.float32)
        else:
            centroids[cat] = np.zeros(384, dtype=np.float32)

    cached = CategoryCentroids(
        pack_id=pack_id,
        canonical_version=canonical_version,
        categories=categories,
        centroids=centroids,
        category_set=category_set,
    )
    _CENTROIDS_CACHE[cache_key] = cached
    return cached


def reset_category_centroids_cache() -> None:
    """Clear centroid cache (for testing)."""
    _CENTROIDS_CACHE.clear()


# ── Scoring & Gate Evaluation ───────────────────────────────────────────────


@dataclass
class ZeroShotCategoryResult:
    category: str | None
    top1_category: str | None
    top1_score: float
    top2_category: str | None
    top2_score: float
    margin: float
    floor: float
    margin_gate: float
    passed_floor: bool
    passed_margin: bool
    is_member: bool
    latency_ms: float
    extraction_source: str
    all_scores: dict[str, float] = field(default_factory=dict)


def predict_category_zero_shot(
    text: str,
    pack: Any,
    *,
    floor: float | None = None,
    margin: float | None = None,
    timeout_ms: float | None = None,
    embedder: Any | None = None,
) -> ZeroShotCategoryResult:
    """Predict category zero-shot against centroid embeddings with dual gating.

    Requires:
      1. top-1 score >= floor (FRONTLINE_CATEGORY_ZS_FLOOR, default 0.35)
      2. top-1 minus top-2 >= margin (FRONTLINE_CATEGORY_ZS_MARGIN, default 0.05)
      3. top-1 category in pack gazetteer categories
    Returns category=None below either gate.
    """
    t0 = time.monotonic()
    conf_floor = floor if floor is not None else category_zs_floor()
    conf_margin = margin if margin is not None else category_zs_margin()
    max_ms = timeout_ms if timeout_ms is not None else category_zs_timeout_ms()

    clean = (text or "").strip()
    if not clean:
        return ZeroShotCategoryResult(
            category=None,
            top1_category=None,
            top1_score=0.0,
            top2_category=None,
            top2_score=0.0,
            margin=0.0,
            floor=conf_floor,
            margin_gate=conf_margin,
            passed_floor=False,
            passed_margin=False,
            is_member=False,
            latency_ms=0.0,
            extraction_source="none",
        )

    if embedder is None:
        try:
            embedder = get_category_embedder()
        except Exception as e:
            step_down("slm_understanding", reason=f"embedder_unavailable: {e}")
            return ZeroShotCategoryResult(
                category=None,
                top1_category=None,
                top1_score=0.0,
                top2_category=None,
                top2_score=0.0,
                margin=0.0,
                floor=conf_floor,
                margin_gate=conf_margin,
                passed_floor=False,
                passed_margin=False,
                is_member=False,
                latency_ms=(time.monotonic() - t0) * 1000.0,
                extraction_source="none",
            )

    cc = get_category_centroids(pack, embedder)
    if not cc.categories:
        return ZeroShotCategoryResult(
            category=None,
            top1_category=None,
            top1_score=0.0,
            top2_category=None,
            top2_score=0.0,
            margin=0.0,
            floor=conf_floor,
            margin_gate=conf_margin,
            passed_floor=False,
            passed_margin=False,
            is_member=False,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            extraction_source="none",
        )

    timeout_s = max(0.001, max_ms / 1000.0)
    fut = None
    try:
        fut = _EMBED_EXECUTOR.submit(embedder.embed, clean)
        try:
            emb = fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            step_down("slm_understanding", reason=f"latency_breach: embedding timed out after {max_ms:.1f}ms")
            return ZeroShotCategoryResult(
                category=None,
                top1_category=None,
                top1_score=0.0,
                top2_category=None,
                top2_score=0.0,
                margin=0.0,
                floor=conf_floor,
                margin_gate=conf_margin,
                passed_floor=False,
                passed_margin=False,
                is_member=False,
                latency_ms=(time.monotonic() - t0) * 1000.0,
                extraction_source="none",
            )
        u_v = np.array(emb.values[:384], dtype=np.float32)
        norm = float(np.linalg.norm(u_v))
        if norm > 1e-9:
            u_v = u_v / norm
    except Exception as e:
        if fut is not None:
            fut.cancel()
        step_down("slm_understanding", reason=f"embed_text_failed: {e}")
        return ZeroShotCategoryResult(
            category=None,
            top1_category=None,
            top1_score=0.0,
            top2_category=None,
            top2_score=0.0,
            margin=0.0,
            floor=conf_floor,
            margin_gate=conf_margin,
            passed_floor=False,
            passed_margin=False,
            is_member=False,
            latency_ms=(time.monotonic() - t0) * 1000.0,
            extraction_source="none",
        )

    scores = {c: float(np.dot(u_v, cc.centroids[c])) for c in cc.categories}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top1_cat, top1_score = ranked[0]
    top2_cat, top2_score = ranked[1] if len(ranked) > 1 else ("", 0.0)
    diff = top1_score - top2_score

    passed_floor = top1_score >= conf_floor
    passed_margin = diff >= conf_margin
    is_member = top1_cat in cc.category_set

    latency_ms = (time.monotonic() - t0) * 1000.0

    if latency_ms > max_ms:
        step_down("slm_understanding", reason=f"latency_breach: {latency_ms:.1f}ms > {max_ms:.1f}ms")
        chosen_cat = None
        source = "none"
    elif passed_floor and passed_margin and is_member:
        chosen_cat = top1_cat
        source = "zeroshot"
    else:
        chosen_cat = None
        source = "none"

    return ZeroShotCategoryResult(
        category=chosen_cat,
        top1_category=top1_cat,
        top1_score=top1_score,
        top2_category=top2_cat,
        top2_score=top2_score,
        margin=diff,
        floor=conf_floor,
        margin_gate=conf_margin,
        passed_floor=passed_floor,
        passed_margin=passed_margin,
        is_member=is_member,
        latency_ms=latency_ms,
        extraction_source=source,
        all_scores=scores,
    )


predict_category_zeroshot = predict_category_zero_shot


# ── Shadow Telemetry ─────────────────────────────────────────────────────────


def record_category_shadow_comparison(
    *,
    interaction_id: str,
    pack_id: str,
    rules_label: str | None,
    zs_label: str | None,
    zs_score: float,
    margin: float,
    latency_ms: float,
    agreement: bool,
    confident_wrong: bool,
) -> None:
    """Record shadow comparison. No complaint text is ever recorded."""
    try:
        from src.ml_runtime.slm_runtime import ensure_slm_shadow_table

        with ops_con() as con:
            ensure_slm_shadow_table(con)
            comparison_id = f"slm_cmp_{new_ulid()}"
            con.execute(
                """
                INSERT INTO slm_shadow_comparisons (
                    comparison_id, interaction_id, pack_id, task,
                    rules_value, slm_value, slm_score, confidence_floor,
                    agreement, confident_wrong, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    comparison_id,
                    interaction_id,
                    pack_id,
                    "category",
                    rules_label,
                    zs_label,
                    zs_score,
                    category_zs_floor(),
                    agreement,
                    confident_wrong,
                    latency_ms,
                ],
            )
    except Exception as e:
        logger.debug("Failed to record category shadow comparison: %s", e)


# ── Unified Extractor with Mode Gating & Source Stamping ─────────────────────


def extract_category_with_mode(
    text: str,
    ctx: Any,
    rules_cat: str | None,
) -> tuple[str | None, str, dict[str, Any]]:
    """Extract category respecting FRONTLINE_SLM_MODE (legacy | shadow | live).

    Returns:
        (category, extraction_source, metadata)
        extraction_source is strictly one of: 'rules', 'zeroshot', 'rules+zeroshot', 'none'.
    """
    mode = slm_mode()
    pack = getattr(ctx, "pack", None)
    interaction_id = getattr(ctx, "interaction_id", "") or "unknown"
    pack_id = getattr(pack, "id", "") if pack else "unknown"

    meta: dict[str, Any] = {
        "mode": mode,
        "rules_cat": rules_cat,
        "zs_cat": None,
        "zs_score": 0.0,
        "margin": 0.0,
        "latency_ms": 0.0,
        "source": "rules" if rules_cat else "none",
    }

    if mode == "legacy" or pack is None:
        return rules_cat, ("rules" if rules_cat else "none"), meta

    # Run zero-shot scoring
    t0 = time.monotonic()
    zs_res = predict_category_zero_shot(text, pack)
    latency_ms = (time.monotonic() - t0) * 1000.0

    meta.update({
        "zs_cat": zs_res.category,
        "zs_score": zs_res.top1_score,
        "margin": zs_res.margin,
        "latency_ms": latency_ms,
    })

    agreement = (rules_cat == zs_res.category)
    confident_wrong = bool(rules_cat and zs_res.category and rules_cat != zs_res.category)

    # Shadow telemetry logged in both shadow and live modes
    record_category_shadow_comparison(
        interaction_id=interaction_id,
        pack_id=pack_id,
        rules_label=rules_cat,
        zs_label=zs_res.category,
        zs_score=zs_res.top1_score,
        margin=zs_res.margin,
        latency_ms=latency_ms,
        agreement=agreement,
        confident_wrong=confident_wrong,
    )

    if mode == "shadow":
        # In shadow mode: AI extractions are observed only; deterministic rules remain in effect.
        return rules_cat, ("rules" if rules_cat else "none"), meta

    if mode == "live":
        # In live mode: zero-shot category flows through if gated, with honest source stamping.
        # Below either gate -> return None ("Unknown" is cheap; wrong is expensive).
        if zs_res.category:
            if rules_cat and rules_cat == zs_res.category:
                source = "rules+zeroshot"
            else:
                source = "zeroshot"
            meta["source"] = source
            return zs_res.category, source, meta
        else:
            meta["source"] = "none"
            return None, "none", meta

    return rules_cat, ("rules" if rules_cat else "none"), meta
