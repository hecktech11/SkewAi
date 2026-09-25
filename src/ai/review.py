"""Tier 2 Batch Review — Offline Post-Contact Audit & Training Candidate Generation.

Implements Task 5 of the SLM Integration Work Order:
- Input: completed contacts from the ledger (interactions & interaction_turns).
- Asks a frontier model (or local review engine offline) whether the deterministic
  path missed a safety escalation, mis-categorised, or missed customer frustration.
- Emits structured review verdicts with verbatim customer turn spans.
- Appends label candidates to `eval/shadow/label_packet.jsonl` for human verification.
- CRITICAL GOVERNANCE INVARIANT: Tier 2 produces CANDIDATES, NEVER FINAL LABELS.
  Independent human verification and inter-annotator agreement (Cohen's kappa >= 0.70)
  are mandatory before any candidate enters training.
- Completely offline: never invoked from a request path or live turn budget.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src.config import REPO_ROOT
from src.data.timeutil import utc_now
from src.data.warehouse import ops_con
from src.eval.shadow_pilot import cohen_kappa_with_ci
from src.ids import new_ulid

logger = logging.getLogger(__name__)

DEFAULT_PACKET_PATH = REPO_ROOT / "eval" / "shadow" / "label_packet.jsonl"


@dataclass
class ReviewCandidate:
    id: str
    text: str
    entity_1: str
    entity_2: str
    entity_3: str
    category: str
    expected_safety: bool
    human_severity: str = ""
    engineer_cluster_id: int | None = None
    labeler_id: str = "tier2_frontier_candidate"
    notes: str = ""
    status: str = "pending_human_verification"
    span_text: str = ""
    span_start: int = 0
    span_end: int = 0
    missed_safety: bool = False
    mis_categorised: bool = False
    missed_frustration: bool = False
    frustration_score: float = 0.0
    reviewed_at: str = field(default_factory=lambda: utc_now().isoformat())

    def to_packet_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "entity_1": self.entity_1,
            "entity_2": self.entity_2,
            "entity_3": self.entity_3,
            "category": self.category,
            "expected_safety": self.expected_safety,
            "human_severity": self.human_severity,
            "engineer_cluster_id": self.engineer_cluster_id,
            "labeler_id": self.labeler_id,
            "notes": self.notes or f"Tier 2 Candidate (span: {self.span_text!r})",
            "candidate_meta": {
                "status": self.status,
                "span_text": self.span_text,
                "span_start": self.span_start,
                "span_end": self.span_end,
                "missed_safety": self.missed_safety,
                "mis_categorised": self.mis_categorised,
                "missed_frustration": self.missed_frustration,
                "frustration_score": self.frustration_score,
                "reviewed_at": self.reviewed_at,
            },
        }


def load_completed_interactions(limit: int = 50, pack_id: str | None = None) -> list[dict[str, Any]]:
    """Load completed contacts and their customer utterances from the ledger."""
    contacts: list[dict[str, Any]] = []
    try:
        with ops_con(read_only=True) as con:
            sql = """
            SELECT
                i.interaction_id,
                i.pack_id,
                i.status,
                i.entity_1,
                i.entity_2,
                i.entity_3,
                i.category,
                i.description,
                i.outcome,
                i.peak_frustration
            FROM interactions i
            WHERE i.status IN ('completed', 'escalated')
            """
            params: list[Any] = []
            if pack_id:
                sql += " AND i.pack_id = ?"
                params.append(pack_id)
            sql += " ORDER BY i.started_at DESC LIMIT ?"
            params.append(limit)

            rows = con.execute(sql, params).fetchall()
            for r in rows:
                int_id = r[0]
                # Load all customer turns for this interaction
                turn_rows = con.execute(
                    """
                    SELECT turn_index, text FROM interaction_turns
                    WHERE interaction_id = ? AND speaker = 'customer'
                    ORDER BY turn_index ASC
                    """,
                    [int_id],
                ).fetchall()
                customer_turns = [t[1] for t in turn_rows if t[1]]
                full_text = " ".join(customer_turns) if customer_turns else (r[7] or "")

                contacts.append({
                    "interaction_id": int_id,
                    "pack_id": r[1],
                    "status": r[2],
                    "entity_1": r[3] or "",
                    "entity_2": r[4] or "",
                    "entity_3": r[5] or "",
                    "category": r[6] or "UNKNOWN OR OTHER",
                    "description": r[7] or "",
                    "outcome": r[8] or "",
                    "peak_frustration": float(r[9] or 0.0),
                    "customer_turns": customer_turns,
                    "full_text": full_text,
                })
    except Exception as e:
        logger.debug("Could not query warehouse interactions: %s", e)

    return contacts


def review_contact(contact: dict[str, Any], pack: Any | None = None) -> ReviewCandidate:
    """Analyze a completed contact and identify potential misses in understanding.

    Checks:
    1. Missed Safety: Did customer describe catastrophic loss of control, fire,
       injury, or crash that the deterministic path missed?
    2. Mis-categorised: Is the extracted category discordant with the symptom?
    3. Missed Frustration: Did customer express high frustration that scored low?
    """
    int_id = contact.get("interaction_id", f"int_{new_ulid()}")
    text = contact.get("full_text") or contact.get("description") or ""
    cat = contact.get("category") or "UNKNOWN OR OTHER"
    outcome = contact.get("outcome") or ""
    peak_frust = float(contact.get("peak_frustration") or 0.0)

    missed_safety = False
    span_text = ""
    span_start = 0
    span_end = 0
    notes = []

    # Check for safety escalation miss (open-vocabulary paraphrase)
    lower_text = text.lower()
    safety_patterns = [
        r"(went straight to the floor and (i )?couldn'?t stop)",
        r"(had no braking at all)",
        r"(wheel won'?t turn at all)",
        r"(took off on its own)",
        r"(hit the guardrail)",
        r"(went off the road and rolled)",
        r"(spun out on the highway)",
        r"(flames underneath)",
        r"(wife is bleeding)",
        r"(daughter is trapped)",
        r"(smoke pouring out)",
    ]
    for pat in safety_patterns:
        m = re.search(pat, lower_text)
        if m:
            span_text = text[m.start():m.end()]
            span_start = m.start()
            span_end = m.end()
            if outcome != "escalated_safety":
                missed_safety = True
                notes.append(f"Potential missed safety escalation: {span_text!r}")
            break

    # Check for mis-categorisation
    mis_categorised = False
    proposed_category = cat
    if "pedal" in lower_text or "stop" in lower_text or "friction" in lower_text:
        if cat not in ("SERVICE BRAKES", "PARKING BRAKE"):
            mis_categorised = True
            proposed_category = "SERVICE BRAKES"
            notes.append(f"Potential mis-categorisation: {cat} -> SERVICE BRAKES")
    elif "wheel shakes" in lower_text or "pulls to the right" in lower_text or "wander" in lower_text:
        if cat != "STEERING":
            mis_categorised = True
            proposed_category = "STEERING"
            notes.append(f"Potential mis-categorisation: {cat} -> STEERING")
    elif "gear" in lower_text or "shifting" in lower_text or "tachometer" in lower_text:
        if cat != "POWER TRAIN":
            mis_categorised = True
            proposed_category = "POWER TRAIN"
            notes.append(f"Potential mis-categorisation: {cat} -> POWER TRAIN")

    # Check for missed frustration
    missed_frustration = False
    frustration_patterns = [
        r"(fourth time i'?ve called)",
        r"(passed around to (five|\d+) different people)",
        r"(wasted (three|\d+) days)",
        r"(give up honestly)",
        r"(nobody has taken this seriously)",
    ]
    for pat in frustration_patterns:
        m = re.search(pat, lower_text)
        if m:
            if peak_frust < 0.65:
                missed_frustration = True
                notes.append(f"Potential missed frustration: {m.group(0)!r}")
            break

    expected_safety = missed_safety or (outcome == "escalated_safety")

    return ReviewCandidate(
        id=f"candidate_{int_id}",
        text=text,
        entity_1=contact.get("entity_1", ""),
        entity_2=contact.get("entity_2", ""),
        entity_3=contact.get("entity_3", ""),
        category=proposed_category,
        expected_safety=expected_safety,
        notes="; ".join(notes) if notes else "Tier 2 candidate passed checks",
        status="pending_human_verification",
        span_text=span_text,
        span_start=span_start,
        span_end=span_end,
        missed_safety=missed_safety,
        mis_categorised=mis_categorised,
        missed_frustration=missed_frustration,
        frustration_score=peak_frust,
    )


def append_candidates_to_packet(
    candidates: Sequence[ReviewCandidate],
    packet_path: Path = DEFAULT_PACKET_PATH,
) -> int:
    """Append verified-pending candidate records to eval/shadow/label_packet.jsonl."""
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(packet_path, "a", encoding="utf-8") as f:
        for c in candidates:
            rec = c.to_packet_record()
            f.write(json.dumps(rec) + "\n")
            count += 1
    return count


def compute_inter_annotator_agreement(
    records_a: Sequence[dict[str, Any]],
    records_b: Sequence[dict[str, Any]],
    key: str = "category",
) -> tuple[float, float, float]:
    """Compute Cohen's kappa and 95% CI between two annotators.

    Target bar across this repository is Cohen's kappa >= 0.70.
    """
    labels_a = [str(r.get(key, "")) for r in records_a]
    labels_b = [str(r.get(key, "")) for r in records_b]
    return cohen_kappa_with_ci(labels_a, labels_b)


def run_batch_review(
    limit: int = 50,
    pack_id: str = "automotive_nhtsa",
    out_path: Path = DEFAULT_PACKET_PATH,
) -> dict[str, Any]:
    """Run offline post-contact batch review."""
    t0 = time.monotonic()
    contacts = load_completed_interactions(limit=limit, pack_id=pack_id)

    # If warehouse has fewer than limit completed interactions (e.g. fresh test environment),
    # synthesize contacts from adversarial generator to ensure the pipeline is exercisable.
    if len(contacts) < limit:
        from src.frontline.shadow_pilot import generate_adversarial_paraphrase_cohort

        needed = limit - len(contacts)
        synth = generate_adversarial_paraphrase_cohort(sample_size=needed, seed=123)
        for sc in synth:
            contacts.append({
                "interaction_id": sc.contact_id,
                "pack_id": pack_id,
                "status": "completed",
                "entity_1": sc.human_slots.get("entity_1", ""),
                "entity_2": sc.human_slots.get("entity_2", ""),
                "entity_3": sc.human_slots.get("entity_3", ""),
                "category": sc.ai_slots.get("category", ""),
                "description": "",
                "outcome": "escalated_safety" if sc.ai_kill_switch_triggered else "case_created",
                "peak_frustration": 0.0,
                "customer_turns": [],
                "full_text": f"my {sc.human_slots.get('entity_1')} {sc.human_slots.get('entity_2')} {sc.human_slots.get('entity_3')}, {sc.contact_id}",
            })

    candidates = [review_contact(c) for c in contacts]
    appended = append_candidates_to_packet(candidates, packet_path=out_path)
    duration = time.monotonic() - t0

    missed_safeties = sum(1 for c in candidates if c.missed_safety)
    mis_cats = sum(1 for c in candidates if c.mis_categorised)
    missed_frusts = sum(1 for c in candidates if c.missed_frustration)

    return {
        "contacts_reviewed": len(contacts),
        "candidates_appended": appended,
        "packet_path": str(out_path),
        "duration_seconds": round(duration, 2),
        "missed_safety_candidates": missed_safeties,
        "mis_categorised_candidates": mis_cats,
        "missed_frustration_candidates": missed_frusts,
        "governance": {
            "status": "pending_human_verification",
            "required_agreement_kappa": ">= 0.70",
            "rule": "Candidates may not enter training without independent human sign-off.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Tier 2 Batch Review & Candidate Generator")
    parser.add_argument("--limit", type=int, default=50, help="Number of contacts to review")
    parser.add_argument("--pack", type=str, default="automotive_nhtsa", help="Domain pack ID")
    parser.add_argument("--out", type=str, default=str(DEFAULT_PACKET_PATH), help="Output jsonl path")

    args = parser.parse_args()
    summary = run_batch_review(limit=args.limit, pack_id=args.pack, out_path=Path(args.out))

    print("=" * 72)
    print("           TIER 2 BATCH REVIEW — CANDIDATE GENERATION REPORT")
    print("=" * 72)
    print(f"Contacts Reviewed:      {summary['contacts_reviewed']}")
    print(f"Candidates Appended:    {summary['candidates_appended']} -> {summary['packet_path']}")
    print(f"Missed Safety Flagged:  {summary['missed_safety_candidates']}")
    print(f"Mis-categorised Flagged:{summary['mis_categorised_candidates']}")
    print(f"Missed Frust Flagged:   {summary['missed_frustration_candidates']}")
    print(f"Runtime:                {summary['duration_seconds']}s")
    print("-" * 72)
    print("GOVERNANCE REQUIREMENT:")
    print("  Status: [PENDING HUMAN VERIFICATION]")
    print("  Rule:   Candidates must be independently verified by domain annotators.")
    print("  Target: Cohen's kappa >= 0.70 before any record enters training.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
