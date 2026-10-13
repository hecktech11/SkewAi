"""Similarity helpers that refuse cross-version cosine.

Used by InvestigatorAgent and shadow evaluation. Missing semantic vectors
never fall back to hash and never become novelty on their own.
"""

from __future__ import annotations

import json
import time
from typing import Any

from src.ml_runtime.embedding_runtime import (
    active_cluster_build_id,
    customer_visible_mode,
    hash_embedder,
    shadow_enabled,
    try_semantic_embedder,
)
from src.ml_runtime.embedding_space import (
    HASH_EMBEDDING_VERSION,
    EmbeddedVector,
    as_embedded,
    compare_embeddings,
)
from src.ml_runtime.embedding_store import get_embedding, get_embeddings_for_records
from src.ml_runtime.embeddings import rank_by_similarity


def rank_hash(query: str, candidates: list[dict[str, Any]], *, top_k: int = 5) -> list[dict[str, Any]]:
    return rank_by_similarity(query, candidates, top_k=top_k, embedder=hash_embedder())


def rank_semantic(
    pack_id: str,
    query: str,
    candidates: list[dict[str, Any]],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Semantic rank over sidecar vectors. Never hashes on failure."""
    embedder = try_semantic_embedder()
    if embedder is None:
        return {"status": "skipped", "reason": "embedder_unavailable", "ranked": []}
    try:
        q = embedder.embed(query)
    except Exception:
        return {"status": "failed", "reason": "embed_failed", "ranked": []}
    if q.is_zero:
        return {"status": "skipped", "reason": "empty_query", "ranked": []}
    rids = [str(c.get("record_id")) for c in candidates if c.get("record_id")]
    found = get_embeddings_for_records(pack_id, rids, embedder.version)
    scored: list[tuple[float, dict[str, Any]]] = []
    missing = 0
    for c in candidates:
        rid = str(c.get("record_id") or "")
        vec = found.get(rid)
        if vec is None:
            missing += 1
            continue
        try:
            score = compare_embeddings(q, vec)
        except Exception:
            missing += 1
            continue
        row = dict(c)
        row["sim_score"] = score
        row["embedding_version"] = q.embedding_version
        scored.append((score, row))
    if not scored:
        return {
            "status": "skipped",
            "reason": "no_compatible_vectors",
            "ranked": [],
            "missing": missing,
            "query_version": q.embedding_version,
        }
    scored.sort(key=lambda x: x[0], reverse=True)
    return {
        "status": "performed",
        "ranked": [r for _, r in scored[:top_k]],
        "missing": missing,
        "query_version": q.embedding_version,
        "candidate_version": embedder.version,
    }


def semantic_cluster_score(
    pack_id: str,
    query: str,
    cluster_row: dict[str, Any],
) -> dict[str, Any]:
    """Score a versioned cluster centroid against the query. No hash fallback."""
    embedder = try_semantic_embedder()
    build_id = active_cluster_build_id()
    if embedder is None or not build_id:
        return {"status": "skipped", "score": 0.0, "reason": "unavailable"}
    centroid = cluster_row.get("centroid")
    version = cluster_row.get("embedding_version")
    if not centroid or not version:
        return {"status": "skipped", "score": 0.0, "reason": "no_centroid"}
    if version != embedder.version:
        return {"status": "skipped", "score": 0.0, "reason": "version_mismatch"}
    try:
        q = embedder.embed(query)
        model_key = str(cluster_row.get("model_key") or q.model_key or "")
        padding_strategy = str(
            cluster_row.get("padding_strategy") or q.padding_strategy or ""
        )
        cvec = as_embedded(
            centroid,
            version,
            native_dimension=embedder.native_dimension,
            output_dimension=embedder.output_dimension,
            model_key=model_key,
            padding_strategy=padding_strategy,
        )
        score = compare_embeddings(q, cvec)
    except Exception:
        return {"status": "failed", "score": 0.0, "reason": "compare_failed"}
    return {
        "status": "performed",
        "score": score,
        "query_version": q.embedding_version,
        "candidate_version": version,
        "cluster_build_id": build_id,
    }


def maybe_run_shadow(
    *,
    interaction_id: str,
    pack_id: str,
    query: str,
    candidates: list[dict[str, Any]],
    hash_ranked: list[dict[str, Any]],
    hash_cluster_id: int | None,
    hash_novel: bool,
) -> dict[str, Any] | None:
    if not shadow_enabled() or customer_visible_mode() != "hash":
        return None
    t0 = time.perf_counter()
    sem = rank_semantic(pack_id, query, candidates, top_k=5)
    latency = int((time.perf_counter() - t0) * 1000)
    sem_ids = [r.get("record_id") for r in sem.get("ranked") or []]
    sem_score = None
    ranked = sem.get("ranked") or []
    if ranked:
        sem_score = float(ranked[0].get("sim_score") or 0)
    # Shadow cluster match is evaluation-only and never opens investigations.
    semantic_novel = False
    if sem.get("status") == "performed" and ranked:
        semantic_novel = float(ranked[0].get("sim_score") or 0) < 0.15
    elif sem.get("status") != "performed":
        semantic_novel = False  # missing embedding != novel
    try:
        from src.ml_runtime.embedding_shadow import record_shadow_comparison

        record_shadow_comparison(
            interaction_id=interaction_id,
            pack_id=pack_id,
            query_embedding_version=sem.get("query_version") or "",
            candidate_embedding_version=sem.get("candidate_version"),
            cluster_build_id=active_cluster_build_id() or None,
            hash_top_ids=[r.get("record_id") for r in hash_ranked],
            semantic_top_ids=sem_ids,
            hash_top_score=(
                float(hash_ranked[0]["sim_score"])
                if hash_ranked and hash_ranked[0].get("sim_score") is not None
                else None
            ),
            semantic_top_score=sem_score,
            hash_cluster_id=hash_cluster_id,
            semantic_cluster_id=None,
            hash_novel=hash_novel,
            semantic_novel=semantic_novel,
            semantic_status=str(sem.get("status") or "skipped"),
            latency_ms=latency,
        )
    except Exception:
        pass
    return sem


def format_match_meta(
    *,
    query_version: str,
    candidate_version: str | None,
    match_status: str,
    cluster_build_id: str | None = None,
) -> str:
    """Compact ledger metadata. Truncated to fit input_summary."""
    parts = [
        f"emb_q={query_version}",
        f"emb_c={candidate_version or '-'}",
        f"match={match_status}",
    ]
    if cluster_build_id:
        parts.append(f"cluster_build={cluster_build_id}")
    return " ".join(parts)


__all__ = [
    "rank_hash",
    "rank_semantic",
    "semantic_cluster_score",
    "maybe_run_shadow",
    "format_match_meta",
]
