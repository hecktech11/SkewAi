"""Grounded remedy / next-step offer from advisory or pack fields."""

from __future__ import annotations

from typing import Any

from src.ledger import AgentAction, record_action


def build_remedy_offer(
    *,
    advisory_match: dict[str, Any] | None,
    case_id: str | None = None,
    pack_display_name: str = "support",
) -> dict[str, Any] | None:
    """Return customer-facing next step grounded only in advisory fields."""
    adv = advisory_match or {}
    if not adv.get("advisory_id") and not adv.get("remedy"):
        return None
    remedy = (adv.get("remedy") or "").strip()
    summary = (adv.get("summary") or adv.get("scope_summary") or "").strip()
    url = (adv.get("url") or "").strip()
    aid = adv.get("advisory_id") or "known-issue"
    lines = [
        f"We found a matching known issue ({aid}).",
    ]
    if summary:
        lines.append(summary)
    if remedy:
        lines.append(f"Recommended next step: {remedy}")
    else:
        lines.append(
            f"A {pack_display_name} specialist will follow up with the free remedy path."
        )
    if url:
        lines.append(f"Details: {url}")
    if case_id:
        lines.append(f"Your case number is {case_id}.")
    text = " ".join(lines)
    return {
        "advisory_id": aid,
        "remedy": remedy or None,
        "url": url or None,
        "summary": summary or None,
        "customer_text": text,
        "case_id": case_id,
    }


def offer_and_ledger(
    interaction_id: str,
    *,
    advisory_match: dict[str, Any] | None,
    case_id: str | None = None,
    pack_display_name: str = "support",
) -> dict[str, Any] | None:
    offer = build_remedy_offer(
        advisory_match=advisory_match,
        case_id=case_id,
        pack_display_name=pack_display_name,
    )
    if not offer:
        return None
    # Ledger failure must propagate. Callers that still want to continue
    # without an offer have to catch this themselves — returning the offer
    # after a failed write would speak unaudited commercial text.
    record_action(
        AgentAction(
            interaction_id=interaction_id,
            agent="case",
            action_type="followup_drafted",
            input_summary=f"remedy_offer advisory={offer.get('advisory_id')}",
            output_summary=offer["customer_text"][:500],
            evidence_ids=[str(offer["advisory_id"])] if offer.get("advisory_id") else [],
            case_id=case_id,
        )
    )
    return offer


__all__ = ["build_remedy_offer", "offer_and_ledger"]
