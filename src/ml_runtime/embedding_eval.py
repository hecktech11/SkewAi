"""Offline evaluation harness for hash vs semantic retrieval.

Labels are never invented. Seed pairs carry an explicit label_source.
Macro-F1 is not reported without human failure-mode labels.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from src.config import REPO_ROOT
from src.ml_runtime.embedding_space import (
    compare_embeddings,
    native_prefix,
)
from src.ml_runtime.hash_embedder import HashEmbedder

DEFAULT_DATASET = REPO_ROOT / "eval" / "embeddings" / "pairs_v1.jsonl"


def load_pairs(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or DEFAULT_DATASET
    rows: list[dict[str, Any]] = []
    if not p.is_file():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(json.loads(line))
    return rows


def _auc(scores: list[tuple[float, int]]) -> float | None:
    """ROC-AUC from (score, label) pairs. None if a class is missing."""
    pos = [s for s, y in scores if y == 1]
    neg = [s for s, y in scores if y == 0]
    if not pos or not neg:
        return None
    correct = 0.0
    ties = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                correct += 1
            elif p == n:
                ties += 1
    return (correct + 0.5 * ties) / (len(pos) * len(neg))


def _average_precision(scores: list[tuple[float, int]]) -> float | None:
    ranked = sorted(scores, key=lambda x: x[0], reverse=True)
    hits = 0
    precisions: list[float] = []
    for i, (_, y) in enumerate(ranked, start=1):
        if y == 1:
            hits += 1
            precisions.append(hits / i)
    if not precisions:
        return None
    return sum(precisions) / sum(y for _, y in ranked)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def evaluate_pairs(
    embedders: dict[str, Any],
    pairs: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare providers on labeled pairs. Does not claim human ground truth."""
    data = list(pairs) if pairs is not None else load_pairs()
    report: dict[str, Any] = {
        "dataset_version": "pairs_v1",
        "n_pairs": len(data),
        "label_sources": sorted({str(r.get("label_source") or "unspecified") for r in data}),
        "providers": {},
        "acceptance_eligible": False,
        "git_sha": _git_sha(),
        "limitations": [
            "pairs_v1 is a coding-agent development fixture and is not an acceptance gate.",
            "Human labels with provenance, κ ≥ 0.70, and quota are required before cutover.",
            "Do not treat Macro-F1 as available: no failure-mode ground truth ships.",
            "canonical_identity is not used as a positive-pair signal.",
        ],
    }
    hash_e = HashEmbedder()
    for name, embedder in embedders.items():
        pos_scores: list[float] = []
        neg_scores: list[float] = []
        labeled: list[tuple[float, int]] = []
        native_vs_padded: list[float] = []
        for row in data:
            a = embedder.embed(str(row.get("query") or ""))
            b = embedder.embed(str(row.get("candidate") or ""))
            score = compare_embeddings(a, b)
            y = _row_polarity(row)
            if y is None:
                continue
            if y == 1:
                pos_scores.append(score)
            else:
                neg_scores.append(score)
            labeled.append((score, y))
            if a.native_dimension != a.output_dimension:
                na = native_prefix(a.values, a.native_dimension)
                nb = native_prefix(b.values, b.native_dimension)
                from src.ml_runtime.embeddings import cosine as numeric_cosine

                native_s = numeric_cosine(na, nb)
                native_vs_padded.append(abs(native_s - score))
        report["providers"][name] = {
            "version": getattr(embedder, "version", None),
            "positive_mean": round(_mean(pos_scores), 4),
            "negative_mean": round(_mean(neg_scores), 4),
            "positive_n": len(pos_scores),
            "negative_n": len(neg_scores),
            "roc_auc": None if _auc(labeled) is None else round(_auc(labeled) or 0.0, 4),
            "average_precision": None
            if _average_precision(labeled) is None
            else round(_average_precision(labeled) or 0.0, 4),
            "max_native_padded_delta": round(max(native_vs_padded), 8) if native_vs_padded else 0.0,
        }
    # Hash is always reported even if the caller did not pass it.
    if "hash" not in report["providers"]:
        report["providers"]["hash"] = evaluate_pairs({"hash": hash_e}, data)["providers"]["hash"]
    report["per_class"] = {
        name: per_class_scorecard(embedder, data) for name, embedder in embedders.items()
    }
    return report


def _git_sha() -> str:
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        sha = (out.stdout or "").strip()
        return sha or "not-a-git-repository"
    except Exception:
        return "not-a-git-repository"


def bootstrap_mean_ci(
    values: Sequence[float], *, n_boot: int = 1000, seed: int = 0
) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = __import__("random").Random(seed)
    means: list[float] = []
    seq = list(values)
    for _ in range(int(n_boot)):
        sample = [seq[rng.randrange(len(seq))] for _ in range(len(seq))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return round(lo, 4), round(hi, 4)


def _row_polarity(row: dict[str, Any]) -> int | None:
    """1 positive, 0 negative, None when the row has no usable label.

    ``novel_false`` is a negative class. Unlabeled rows are excluded from
    scored metrics so they cannot become false negatives.
    """
    cls = str(row.get("eval_class") or "")
    lab = str(row.get("label") or "").lower()
    if cls == "novel_false":
        return 0
    if cls in {"paraphrase_positive", "safety_true_positive"}:
        return 1
    if lab in {"positive", "pos", "1", "true"}:
        return 1
    if lab in {"negative", "neg", "0", "false", "hard_negative"}:
        return 0
    return None


def _is_positive_row(row: dict[str, Any]) -> bool:
    return _row_polarity(row) == 1


def per_class_scorecard(embedder: Any, pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    from collections import defaultdict

    by: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for row in pairs:
        cls = str(row.get("eval_class") or "unspecified")
        a = embedder.embed(str(row.get("query") or ""))
        b = embedder.embed(str(row.get("candidate") or ""))
        score = compare_embeddings(a, b)
        y = _row_polarity(row)
        if y is None:
            continue
        by[cls].append((score, y))
    out: dict[str, Any] = {}
    for cls, scored in sorted(by.items()):
        ranked = sorted(scored, key=lambda x: x[0], reverse=True)
        pos = sum(y for _, y in scored)
        rec5 = _recall_at_k(ranked, 5)
        rec10 = _recall_at_k(ranked, 10)
        prec5 = _precision_at_k(ranked, 5)
        ap = _average_precision(scored)
        pos_scores = [s for s, y in scored if y == 1]
        lo, hi = bootstrap_mean_ci(pos_scores)
        out[cls] = {
            "n": len(scored),
            "positives": pos,
            "recall_at_5": rec5,
            "recall_at_10": rec10,
            "precision_at_5": prec5,
            "average_precision": None if ap is None else round(ap, 4),
            "positive_mean_ci95": [lo, hi],
        }
    return out


def _recall_at_k(ranked: list[tuple[float, int]], k: int) -> float | None:
    total_pos = sum(y for _, y in ranked)
    if total_pos == 0:
        return None
    hit = sum(y for _, y in ranked[:k])
    return round(hit / total_pos, 4)


def _precision_at_k(ranked: list[tuple[float, int]], k: int) -> float | None:
    top = ranked[:k]
    if not top:
        return None
    return round(sum(y for _, y in top) / len(top), 4)


def paraphrase_audit(
    embedder: Any,
    left: str = "vehicle stalled at 65 mph",
    right: str = "engine died on highway at speed",
) -> dict[str, Any]:
    """Required audit example. Hash vs native vs padded semantic cosine."""
    from src.ml_runtime.embeddings import cosine as numeric_cosine
    from src.ml_runtime.hash_embedder import HashEmbedder as _H

    h = _H()
    hv = h.embed(left)
    hw = h.embed(right)
    hash_s = compare_embeddings(hv, hw)
    a = embedder.embed(left)
    b = embedder.embed(right)
    padded_s = compare_embeddings(a, b)
    native_s = numeric_cosine(
        native_prefix(a.values, a.native_dimension),
        native_prefix(b.values, b.native_dimension),
    )
    return {
        "left": left,
        "right": right,
        "hash_similarity": hash_s,
        "semantic_native_similarity": native_s,
        "semantic_padded_similarity": padded_s,
        "hash_version": h.version,
        "semantic_version": embedder.version,
        "native_padded_delta": abs(native_s - padded_s),
    }


__all__ = ["load_pairs", "evaluate_pairs", "paraphrase_audit", "DEFAULT_DATASET"]
