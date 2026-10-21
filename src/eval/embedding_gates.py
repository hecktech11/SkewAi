"""Load pre-registered decision rules and refuse ineligible eval data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from src.config import REPO_ROOT
from src.eval.embedding_labels import (
    INELIGIBLE_SOURCES,
    KAPPA_MIN,
    LABEL_CLASSES,
    agreement_report,
    class_quota_counts,
)

RULES_PATH = REPO_ROOT / "eval" / "embeddings" / "decision_rules_v1.yaml"


def load_decision_rules(path: Path | None = None) -> dict[str, Any]:
    p = path or RULES_PATH
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def label_source_eligible(source: str | None) -> bool:
    return (source or "").strip() not in INELIGIBLE_SOURCES


def acceptance_eligibility(rules: dict[str, Any] | None = None) -> dict[str, Any]:
    """Classes that may be used as gates. Empty until humans label + agree."""
    rules = rules or load_decision_rules()
    agr = agreement_report()
    quotas = class_quota_counts()
    wanted = rules.get("class_quotas") or {}
    eligible_classes: list[str] = []
    blocked: dict[str, str] = {}
    for cls in LABEL_CLASSES:
        info = (agr.get("per_class") or {}).get(cls) or {}
        if not info.get("eligible"):
            blocked[cls] = f"kappa below {KAPPA_MIN} or fewer than 2 annotators"
            continue
        need = wanted.get(cls)
        have = quotas.get(cls, 0)
        if need == "all_available":
            eligible_classes.append(cls)
            continue
        try:
            need_n = int(need)
        except (TypeError, ValueError):
            need_n = 0
        if have < need_n:
            blocked[cls] = f"quota {have} < {need_n}"
            continue
        eligible_classes.append(cls)
    return {
        "rules_version": rules.get("version"),
        "primary_metric": rules.get("primary_metric"),
        "eligible_classes": eligible_classes,
        "blocked_classes": blocked,
        "agreement": agr,
        "quotas": quotas,
        "acceptance_ready": (
            "paraphrase_positive" in eligible_classes
            and "hard_negative_same_category" in eligible_classes
        ),
    }


def filter_acceptance_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        if not label_source_eligible(r.get("label_source")):
            continue
        if r.get("acceptance_eligible") is False:
            continue
        if r.get("adjudication_status") not in {"agreed", "adjudicated"}:
            continue
        out.append(r)
    return out


def semantic_promotion_allowed(
    pack_id: str,
    embedding_version: str,
    *,
    dataset_id: str | None = None,
    thresholds: dict[str, Any] | None = None,
    model_key: str | None = None,
) -> bool:
    """Same gate ``activate_cluster_build`` uses. Does not approve by itself."""
    from src.ml_runtime.cluster_builds import retrieval_scorecard_approved

    return retrieval_scorecard_approved(
        pack_id,
        embedding_version,
        dataset_id=dataset_id,
        thresholds=thresholds,
        model_key=model_key,
    )
