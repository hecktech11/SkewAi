"""Three LLM call sites with deterministic templates as graceful fallback."""

from __future__ import annotations

from typing import Any

from src.ai.provider import NarrationResult, narrate
from src.security.input_validation import sanitize_prompt_variable
from src.security.pii import redact_pii


def phrase_intake_question(
    *,
    slot_label: str,
    template: str,
    customer_last: str = "",
) -> NarrationResult:
    fallback = template or f"Could you tell me the {slot_label}?"
    clean_cust = sanitize_prompt_variable(redact_pii(customer_last[:200]), tag_name="last_customer")
    return narrate(
        site="intake_phrasing",
        system=(
            "You rephrase a single customer-support intake question. "
            "Stay under 40 words. No inventing facts or IDs."
        ),
        user=f"slot={slot_label}\ntemplate={template}\n{clean_cust}",
        fallback=fallback,
    )


def phrase_investigation_brief(
    *,
    similar_count: int,
    cluster_id: Any,
    lead_time_weeks: int | None,
    keyword: str,
    evidence_ids: list[str],
) -> NarrationResult:
    lt = f"{lead_time_weeks} weeks" if lead_time_weeks is not None else "n/a"
    fallback = (
        f"Found {similar_count} similar records"
        + (f" for keyword '{keyword}'" if keyword else "")
        + (f"; cluster {cluster_id}" if cluster_id is not None else "")
        + f"; historical lead-time {lt}."
        + (f" Evidence: {', '.join(evidence_ids[:5])}." if evidence_ids else "")
    )
    clean_kw = sanitize_prompt_variable(redact_pii(keyword[:100]), tag_name="keyword")
    return narrate(
        site="investigator_brief",
        system=(
            "You write a two-sentence RCA brief for a supervisor console. "
            "Only use the numbers given. Cite evidence IDs if provided. No speculation."
        ),
        user=(
            f"similar_count={similar_count} cluster_id={cluster_id} "
            f"lead_time_weeks={lead_time_weeks} {clean_kw} "
            f"evidence_ids={evidence_ids[:8]}"
        ),
        fallback=fallback,
    )


def phrase_followup_draft(
    *,
    case_id: str,
    category: str,
    severity: str,
    description: str,
) -> NarrationResult:
    fallback = (
        f"Case {case_id}: follow up on {category or 'reported issue'} "
        f"(severity {severity}). Customer reported: {(description or '')[:160]}"
    )
    clean_desc = sanitize_prompt_variable(redact_pii((description or "")[:300]), tag_name="customer_description")
    return narrate(
        site="followup_draft",
        system=(
            "You draft a short operator follow-up note (≤60 words). "
            "Do not invent case numbers, remedies, or promises."
        ),
        user=(
            f"case_id={case_id} category={category} severity={severity}\n"
            f"{clean_desc}"
        ),
        fallback=fallback,
    )


__all__ = [
    "phrase_intake_question",
    "phrase_investigation_brief",
    "phrase_followup_draft",
]
