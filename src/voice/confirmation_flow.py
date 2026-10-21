"""Mandatory conversational confirmation protocol."""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional, Tuple

from src.frontline.intake import IntakeSlots


class DialoguePhase(Enum):
    COLLECTION = "collection"
    CONFIRMATION_PENDING = "confirmation_pending"
    CORRECTION = "correction"
    ENRICHMENT_READY = "enrichment_ready"


class ConfirmationOutcome(Enum):
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"
    REJECTED = "rejected"
    UNKNOWN = "unknown"
    DECLINED_TO_PROVIDE = "declined_to_provide"
    NO_RESPONSE = "no_response"
    HUMAN_REQUESTED = "human_requested"
    SAFETY_PREEMPTED = "safety_preempted"
    CALL_DISCONNECTED = "call_disconnected"


class SlotConfirmationProtocol:
    """
    Enforces a synchronous verification turn before Stage 4 enrichment can begin
    on live conversational channels (voice/telephony), while allowing batch/API
    channels to bypass or auto-confirm.
    """

    def __init__(
        self,
        slots: IntakeSlots,
        *,
        channel: str = "voice",
        skip_conversational_confirmation: bool = False,
    ) -> None:
        self.slots = slots
        self.channel = channel.lower()
        self.skip_conversational_confirmation = skip_conversational_confirmation
        self.phase = DialoguePhase.COLLECTION
        self.confirmation_attempts = 0
        self.last_outcome: Optional[ConfirmationOutcome] = None

    def should_trigger_confirmation(self) -> bool:
        """
        Triggers when core entity and symptom slots are non-empty,
        and confirmation has not yet occurred on live conversational channels.
        Non-conversational channels (batch, API, email) auto-advance.
        """
        if self.skip_conversational_confirmation or self.channel not in ("voice", "web_voice", "telephony", "twilio_media"):
            return False

        critical_slots = [
            self.slots.entity_1,  # Model Year
            self.slots.entity_2,  # Make
            self.slots.entity_3,  # Model
            self.slots.category,  # Functional Area
        ]
        return all(bool(s) for s in critical_slots) and self.phase == DialoguePhase.COLLECTION

    def build_confirmation_prompt(self) -> str:
        self.phase = DialoguePhase.CONFIRMATION_PENDING
        make = self.slots.entity_2.title() if self.slots.entity_2.isupper() else self.slots.entity_2
        model = self.slots.entity_3.title() if self.slots.entity_3.isupper() else self.slots.entity_3
        cat = (self.slots.category or "").lower()
        return (
            f"I want to make sure our technical team has the exact details. "
            f"You have a {self.slots.entity_1} {make} {model}, "
            f"and you're experiencing an issue with your {cat}, "
            f"specifically: {self.slots.description}. Did I get that completely right?"
        )

    def evaluate_customer_confirmation(self, turn_text: str) -> Tuple[bool, str]:
        """
        Evaluates the user's response to the confirmation prompt with explicit
        support for corrections, unknown slots, human requests, and safety preemptions.
        """
        cleaned = (turn_text or "").lower().strip()
        if not cleaned:
            self.last_outcome = ConfirmationOutcome.NO_RESPONSE
            self.phase = DialoguePhase.CORRECTION
            self.confirmation_attempts += 1
            return False, "I didn't hear anything. Could you please confirm if those vehicle details are correct?"

        tokens = set(re.findall(r"[a-zA-Z0-9']+", cleaned))

        # 1. Safety preemption (driving / fire / hazard)
        safety_words = {"driving", "crash", "fire", "smoke", "hazard", "emergency"}
        if bool(tokens & safety_words):
            self.last_outcome = ConfirmationOutcome.SAFETY_PREEMPTED
            self.phase = DialoguePhase.ENRICHMENT_READY
            return True, "Safety is our top priority. Let us address that immediately."

        # 2. Human escalation request
        human_words = {"human", "person", "representative", "supervisor", "agent", "someone"}
        if bool(tokens & human_words) and not bool(tokens & {"yes", "correct"}):
            self.last_outcome = ConfirmationOutcome.HUMAN_REQUESTED
            self.phase = DialoguePhase.ENRICHMENT_READY
            return False, "I can request a person. You are not connected yet."

        # 3. Unknown / declined slot (e.g. unknown VIN)
        unknown_phrases = ["do not know", "dont know", "no idea", "not sure", "decline", "not have"]
        if any(p in cleaned for p in unknown_phrases):
            self.last_outcome = ConfirmationOutcome.UNKNOWN
            self.phase = DialoguePhase.ENRICHMENT_READY
            return True, "Understood, we will proceed with the details you have provided."

        # 4. Affirmative confirmation
        affirmative = {"yes", "yeah", "correct", "that's right", "accurate", "yep", "exact", "right"}
        negative = {"no", "nope", "wrong", "incorrect", "actually", "wait", "not right"}

        has_affirmative = bool(tokens & affirmative or "that's right" in cleaned or "thats right" in cleaned)
        has_negative = bool(tokens & negative or "not right" in cleaned or "no," in cleaned or cleaned.startswith("no "))

        if has_affirmative and not has_negative:
            self.last_outcome = ConfirmationOutcome.CONFIRMED
            self.phase = DialoguePhase.ENRICHMENT_READY
            self.slots.is_confirmed = True
            return True, "Great, I am logging that into our technical investigation queue now."

        # 5. Slot correction
        self.confirmation_attempts += 1
        if any(w in cleaned for w in ("actually", "instead", "model", "it's", "its")):
            self.last_outcome = ConfirmationOutcome.CORRECTED
        else:
            self.last_outcome = ConfirmationOutcome.REJECTED

        self.phase = DialoguePhase.CORRECTION
        return False, "My mistake. What specific part did I get wrong?"
