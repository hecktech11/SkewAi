"""Lightweight bag-of-hash embeddings for semantic similarity (no heavy deps).

Produces fixed-dim float vectors from text via hashed character n-grams.
Stored in ``records.embedding`` when present; investigator ranks by cosine
similarity and falls back to ILIKE when embeddings are missing.

Weighting (item 21): sublinear TF (``1 + log(tf)``) × corpus IDF, preserving
the whole-token (2.0) / char-bigram (0.5) vote ratio. IDF comes from
:func:`fit_idf` over the pack corpus (called by cluster rebuilds); without a
fitted table, uniform IDF is used so ``embed_text`` stays deterministic and
dependency-free. L2 normalization removes document-length bias; sublinear TF
additionally damps repetition inside long documents.

Migration: :data:`_DIM` is 512. Vectors stored at another dimension are
stale — :func:`rank_by_similarity` and cluster rebuilds recompute them from
text instead of comparing across dimensions (never a false 0-tie).
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Sequence

_DIM = 512
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)

_TOKEN_VOTE = 2.0
_BIGRAM_VOTE = 0.5

# Corpus IDF table: feature ("tok:x" / "bi:xy") -> idf weight. Empty means
# "unfitted": embed_text falls back to uniform weights (deterministic).
_IDF: dict[str, float] = {}
_IDF_DOCS = 0
_PACK_IDF: dict[str, dict[str, float]] = {}
_PACK_IDF_DOCS: dict[str, int] = {}


def idf_digest(pack_id: str | None = None) -> str:
    """Return an 8-char hex digest of the active IDF table (or 'uniform')."""
    table = _PACK_IDF.get(pack_id) if pack_id else _IDF
    if not table:
        return "uniform"
    items = sorted(table.items())
    raw = ";".join(f"{k}:{round(v, 6)}" for k, v in items).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:8]


def hash_embedding_version(pack_id: str | None = None) -> str:
    from src.ml_runtime.embedding_space import HASH_EMBEDDING_VERSION

    digest = idf_digest(pack_id)
    if digest == "uniform":
        return HASH_EMBEDDING_VERSION
    return f"{HASH_EMBEDDING_VERSION}:idf-{digest}"


def embedding_dim() -> int:
    """Current embedding dimension (512)."""
    return _DIM


def fit_idf(
    texts: Sequence[str], *, pack_id: str | None = None, dim: int = _DIM
) -> dict[str, float]:
    """Fit the IDF table from a corpus (deterministic, pack-scoped when pack_id given).

    Returns the fitted table (feature -> ``log((N+1)/(df+1)) + 1``).
    Call before bulk embedding (cluster rebuilds do this) so frequent
    bigrams (th/he/in) and corpus-dominant tokens don't dominate the norm.
    """
    global _IDF, _IDF_DOCS
    df: dict[str, int] = {}
    docs = [t for t in (texts or []) if t]
    for text in docs:
        seen: set[str] = set()
        for tok in _TOKEN_RE.findall(text.lower()):
            seen.add(f"tok:{tok}")
            for i in range(max(0, len(tok) - 1)):
                seen.add(f"bi:{tok[i:i+2]}")
        for feat in seen:
            df[feat] = df.get(feat, 0) + 1
    n = max(1, len(docs))
    fitted = {feat: math.log((n + 1) / (c + 1)) + 1.0 for feat, c in df.items()}
    if pack_id:
        _PACK_IDF[pack_id] = fitted
        _PACK_IDF_DOCS[pack_id] = n
    else:
        _IDF = fitted
        _IDF_DOCS = n
    return dict(fitted)


def reset_idf(pack_id: str | None = None) -> None:
    """Clear the fitted IDF table (tests / corpus switches)."""
    global _IDF, _IDF_DOCS
    if pack_id:
        _PACK_IDF.pop(pack_id, None)
        _PACK_IDF_DOCS.pop(pack_id, None)
    else:
        _IDF = {}
        _IDF_DOCS = 0
        _PACK_IDF.clear()
        _PACK_IDF_DOCS.clear()


def idf_status(pack_id: str | None = None) -> dict[str, Any]:
    table = _PACK_IDF.get(pack_id) if pack_id else _IDF
    docs = _PACK_IDF_DOCS.get(pack_id, 0) if pack_id else _IDF_DOCS
    return {"features": len(table or {}), "docs": docs, "digest": idf_digest(pack_id)}


def _stable_bucket(gram: str, dim: int) -> int:
    """Process-stable bucket. Python's hash() is salted per process."""
    digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dim


def embed_text(text: str, dim: int = _DIM, pack_id: str | None = None) -> list[float]:
    """Hashing trick embedding — deterministic across processes, no model download.

    Whole-token votes 2.0, char-bigram votes 0.5 (ratio preserved from the
    original weighting), scaled by sublinear TF × fitted IDF when available.
    Empty input -> zero vector (callers treat it as "no signal", never a
    false-identical match).
    """
    vec = [0.0] * dim
    tokens = _TOKEN_RE.findall((text or "").lower())
    if not tokens:
        return vec
    from collections import Counter

    idf_table = _PACK_IDF.get(pack_id) if pack_id else _IDF
    counts = Counter(tokens)
    for tok, tf_raw in counts.items():
        tf = 1.0 + math.log(tf_raw)  # sublinear: repetition damps out
        h = _stable_bucket(f"tok:{tok}", dim)
        weight = idf_table.get(f"tok:{tok}", 1.0) if idf_table else 1.0
        vec[h] += _TOKEN_VOTE * tf * weight
        for i in range(max(0, len(tok) - 1)):
            gram = tok[i:i+2]
            hb = _stable_bucket(f"bi:{gram}", dim)
            b_weight = idf_table.get(f"bi:{gram}", 1.0) if idf_table else 1.0
            vec[hb] += _BIGRAM_VOTE * tf * b_weight
    # L2 normalize — removes document-length bias (long docs don't outrank
    # short ones by magnitude; only direction matters for cosine).
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def measure_collisions(
    texts: Sequence[str], *, threshold: float = 0.99
) -> dict[str, float]:
    """Collision benchmark: share of DISTINCT text pairs with cosine >= threshold.

    Hashing-trick vectors collide by construction; this quantifies it for the
    fitted corpus so dimension/weighting regressions are caught. Lower is
    better; identical texts are excluded (self-similarity is 1.0 by design).
    """
    items = [t for t in (texts or []) if (t or "").strip()]
    vecs = [embed_text(t) for t in items]
    pairs = hits = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i].strip().lower() == items[j].strip().lower():
                continue
            pairs += 1
            try:
                if cosine(vecs[i], vecs[j]) >= threshold:
                    hits += 1
            except ValueError:
                continue
    return {
        "pairs": float(pairs),
        "collisions": float(hits),
        "collision_rate": (hits / pairs) if pairs else 0.0,
        "threshold": float(threshold),
        "dim": float(len(vecs[0])) if vecs else 0.0,
    }


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """True cosine similarity. Raises on dim mismatch (fail loudly, not 0-tie)."""
    if not a or not b:
        return 0.0
    if len(a) != len(b):
        raise ValueError(f"embedding dim mismatch: {len(a)} != {len(b)}")
    dot = float(sum(x * y for x, y in zip(a, b)))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (na * nb)))


def rank_by_similarity(
    query: str,
    candidates: list[dict],
    *,
    text_key: str = "text",
    embedding_key: str = "embedding",
    top_k: int = 5,
    embedder: Any | None = None,
    allow_recompute: bool | None = None,
    pack_id: str | None = None,
) -> list[dict]:
    """Return candidates sorted by cosine sim to query (adds ``sim_score``).

    Empty/whitespace query -> [] (never return input-order as "similar").
    Default embedder is the hash provider. Stored anonymous lists are treated
    as hash vectors (records.embedding is hash-era). Semantic callers must
    pass versioned ``EmbeddedVector`` values and ``allow_recompute=False`` so
    a missing semantic vector is never replaced by a hash vector.
    """
    from src.ml_runtime.embedding_space import (
        HASH_EMBEDDING_VERSION,
        EmbeddedVector,
        EmbeddingVersionMismatch,
        as_embedded,
        compare_embeddings,
    )
    from src.ml_runtime.hash_embedder import HashEmbedder

    if not (query or "").strip():
        return []
    active = embedder or HashEmbedder(pack_id=pack_id)
    recompute = (
        active.version.startswith(HASH_EMBEDDING_VERSION)
        if allow_recompute is None
        else bool(allow_recompute)
    )
    q = active.embed(query)
    if q.is_zero:
        return []
    scored: list[tuple[float, dict]] = []
    for c in candidates:
        emb = c.get(embedding_key)
        cand: EmbeddedVector | None = None
        if isinstance(emb, EmbeddedVector):
            cand = emb
        elif isinstance(emb, (list, tuple)) and len(emb) == q.output_dimension and any(emb):
            if active.version.startswith(HASH_EMBEDDING_VERSION):
                cand = as_embedded(emb, active.version)
        if cand is None:
            if recompute:
                cand = active.embed(str(c.get(text_key) or ""))
            else:
                continue
        try:
            score = compare_embeddings(q, cand)
        except EmbeddingVersionMismatch:
            if recompute:
                cand = active.embed(str(c.get(text_key) or ""))
                score = compare_embeddings(q, cand)
            else:
                continue
        except ValueError:
            if recompute:
                cand = active.embed(str(c.get(text_key) or ""))
                score = compare_embeddings(q, cand)
            else:
                continue
        row = dict(c)
        row["sim_score"] = score
        row["embedding_version"] = q.embedding_version
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_k]]


__all__ = [
    "embed_text",
    "cosine",
    "rank_by_similarity",
    "fit_idf",
    "reset_idf",
    "idf_status",
    "idf_digest",
    "hash_embedding_version",
    "embedding_dim",
    "measure_collisions",
    "_DIM",
    "_stable_bucket",
]
