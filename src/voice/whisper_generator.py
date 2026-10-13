"""Supervisor whisper and warm handoff payload generator."""

from __future__ import annotations

from typing import Any
from xml.sax.saxutils import escape

from src.frontline.intake import IntakeSlots
from src.triage.triage_agent import TriageResult


def generate_supervisor_whisper(
    interaction_id: str,
    slots: IntakeSlots,
    triage: TriageResult,
    kill_switch_terms: list[str],
) -> dict[str, Any]:
    """
    Synthesizes an immediate audio whisper packet piped exclusively to the
    supervisor's headset during a SIP warm bridge.
    """
    safety_alert = ""
    if triage.priority == "P1":
        terms_str = ", ".join(kill_switch_terms) if kill_switch_terms else "immediate hazard"
        safety_alert = f"Warning: Critical P1 safety escalation. Trigger words: {terms_str}."

    whisper_text = (
        f"Incoming transfer for interaction {interaction_id[-6:]}. "
        f"Customer has a {slots.entity_1} {slots.entity_2} {slots.entity_3}. "
        f"Primary issue: {slots.category}. {safety_alert} "
        f"Customer confirmed details. Connecting you now."
    ).strip()

    whisper_ssml = (
        "<speak><prosody rate='115%'>"
        f"{escape(whisper_text)}</prosody></speak>"
    )

    year_val = int(slots.entity_1) if str(slots.entity_1).isdigit() else 0

    return {
        "interaction_id": interaction_id,
        "canonical_identity": {
            "vin": slots.vin or "UNKNOWN",
            "make": slots.entity_2,
            "model": slots.entity_3,
            "year": year_val,
        },
        "triage_state": {
            "severity": triage.severity,
            "priority": triage.priority,
            "safety_flag_tripped": triage.priority == "P1" or triage.safety_flag_tripped,
            "trigger_terms": kill_switch_terms,
        },
        "diagnostic_summary": {
            "reported_symptom": slots.description,
            "elicitation_answers": {},
            "sentiment_score_peak": triage.sentiment_peak,
        },
        "whisper_audio_ssml": whisper_ssml,
    }
