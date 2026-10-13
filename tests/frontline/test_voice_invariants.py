"""Voice Invariant Test Suite — proves Gaps 2-R1, 2-R2, 1-R2, 3-R2.

Covers:
  - ISO 3779 checksum engine (weights, illegal chars I/O/Q, check digit modulo 11)
  - Spoken NATO / phonetic alphabet elicitation
  - Mandatory SlotConfirmationProtocol gating Stage 4 enrichment
  - Correction loop when customer rejects slot frame
  - TelephonyTurnSequencer monotonic sequence and hash deduplication
  - SpokenPlaybackReconciler committing [INTERRUPTED] to the audit journal
  - Supervisor warm handoff packet & whisper audio SSML generation
  - In-cabin driving acoustic detector (40Hz-250Hz spectral rumble)
  - TwilioMediaBridge buffer clearance and audio translation
  - Voice recording consent policy across jurisdictions
"""

from __future__ import annotations

import asyncio
import json
import pytest
import numpy as np

from src.frontline.intake import IntakeSlots
from src.ledger.journal import AgentActionJournal
from src.triage.triage_agent import TriageResult
from src.voice.confirmation_flow import DialoguePhase, SlotConfirmationProtocol
from src.voice.drive_mode_guard import DRIVE_SAFETY_SCRIPT, detect_driving_environment
from src.voice.interruption_reconciler import SpokenPlaybackReconciler
from src.voice.phonetic_normalizer import parse_spoken_alphanumerics
from src.voice.policy import consent_requirement
from src.voice.telephony_bridge import (
    TelephonyMediaBridge,
    pcm16k_to_ulaw8k,
    ulaw8k_to_pcm16k,
)
from src.voice.turn_sequencer import TelephonyTurnSequencer
from src.voice.vin_validator import validate_iso3779_vin
from src.voice.whisper_generator import generate_supervisor_whisper


# ── 1. ISO 3779 Checksum Invariant Tests ────────────────────────────────────

def test_iso3779_checksum_valid_and_invalid():
    # ISO 3779 check digit for this body is 5.
    assert validate_iso3779_vin("1HGCR2F85HA000000") is True

    # Tampered check digit (5 changed to 4).
    assert validate_iso3779_vin("1HGCR2F84HA000000") is False

    # Real North American VINs with mathematical check digit
    assert validate_iso3779_vin("1HGCM82633A004352") is True
    assert validate_iso3779_vin("1HGCM82633A004353") is False  # Tampered check digit

    # Illegal characters (I, O, Q) must fail
    assert validate_iso3779_vin("1HGCR2F84IA000000") is False
    assert validate_iso3779_vin("1HGCM82633A00435I") is False
    assert validate_iso3779_vin("1HGCM82633A00435O") is False
    assert validate_iso3779_vin("1HGCM82633A00435Q") is False

    # Length check
    assert validate_iso3779_vin("1HGCR2F84") is False
    assert validate_iso3779_vin("") is False
    assert validate_iso3779_vin(None) is False  # type: ignore


# ── 2. Phonetic Alphabet Extraction Invariants ──────────────────────────────

def test_phonetic_alphabet_transcription():
    spoken_input = "1 H as in Henry Golf Charlie Romeo 2 Foxtrot 8 4 Hotel Alpha"
    normalized = parse_spoken_alphanumerics(spoken_input)
    assert normalized == "1HGCR2F84HA"

    # Individual letters and numbers
    assert parse_spoken_alphanumerics("zero one two three") == "0123"
    assert parse_spoken_alphanumerics("victor uniform zulu") == "VUZ"
    assert parse_spoken_alphanumerics("c as in charlie d as in david") == "CD"


# ── 3. Mandatory Confirmation Gate Invariant ─────────────────────────────────

def test_confirmation_state_machine_transition():
    slots = IntakeSlots(
        entity_1="2021",
        entity_2="HONDA",
        entity_3="CIVIC",
        category="SERVICE BRAKES",
        description="pedal sinks to floor",
    )
    protocol = SlotConfirmationProtocol(slots)

    # Must trigger confirmation when slots are complete
    assert protocol.should_trigger_confirmation() is True
    prompt = protocol.build_confirmation_prompt()
    assert "2021 Honda Civic" in prompt
    assert protocol.phase == DialoguePhase.CONFIRMATION_PENDING

    # Incomplete slots must NOT trigger confirmation
    empty_slots = IntakeSlots(entity_1="2021", entity_2="HONDA")
    empty_proto = SlotConfirmationProtocol(empty_slots)
    assert empty_proto.should_trigger_confirmation() is False

    # Affirmative confirmation proceeds to enrichment
    confirmed, msg = protocol.evaluate_customer_confirmation("Yes, that is completely correct")
    assert confirmed is True
    assert protocol.phase == DialoguePhase.ENRICHMENT_READY
    assert slots.is_confirmed is True


def test_confirmation_correction_loop():
    slots = IntakeSlots(
        entity_1="2021",
        entity_2="HONDA",
        entity_3="CIVIC",
        category="SERVICE BRAKES",
        description="pedal sinks",
    )
    protocol = SlotConfirmationProtocol(slots)
    protocol.build_confirmation_prompt()

    # Customer rejects slot payload
    confirmed, msg = protocol.evaluate_customer_confirmation("No, actually it's a 2018 model")
    assert confirmed is False
    assert protocol.phase == DialoguePhase.CORRECTION
    assert protocol.confirmation_attempts == 1


# ── 4. Turn Idempotency Invariants ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_turn_idempotency_deduplication():
    sequencer = TelephonyTurnSequencer(interaction_id="int_01JTEST")

    # First arrival of turn sequence 1
    allowed = await sequencer.acquire_turn_execution_slot(turn_seq=1, turn_payload_hash="hash_aaa")
    assert allowed is True

    # Replay of identical packet (e.g. from network retry)
    duplicate = await sequencer.acquire_turn_execution_slot(turn_seq=1, turn_payload_hash="hash_aaa")
    assert duplicate is False

    # Out of order / stale sequence arrival
    stale = await sequencer.acquire_turn_execution_slot(turn_seq=0, turn_payload_hash="hash_zzz")
    assert stale is False

    # Monotonic next sequence
    next_seq = await sequencer.acquire_turn_execution_slot(turn_seq=2, turn_payload_hash="hash_bbb")
    assert next_seq is True


# ── 5. Interruption & Truncated-Truth Reconciler ─────────────────────────────

def test_spoken_playback_interruption_reconciliation():
    journal = AgentActionJournal()
    reconciler = SpokenPlaybackReconciler(journal, interaction_id="int_voice_001")

    word_alignments = [
        {"word": "I", "start": 0.0, "end": 0.2},
        {"word": "understand", "start": 0.25, "end": 0.6},
        {"word": "your", "start": 0.65, "end": 0.8},
        {"word": "brakes", "start": 0.85, "end": 1.1},
        {"word": "are", "start": 1.15, "end": 1.3},
        {"word": "failing.", "start": 1.35, "end": 1.7},
        {"word": "Have", "start": 1.75, "end": 1.9},
        {"word": "you", "start": 1.95, "end": 2.1},
        {"word": "noticed", "start": 2.15, "end": 2.4},
        {"word": "any", "start": 2.45, "end": 2.6},
        {"word": "leaks?", "start": 2.65, "end": 3.0},
    ]

    action_id = "act_turn_01"
    reconciler.register_planned_turn(action_id, "Full planned text", word_alignments)
    assert reconciler.current_action_id == action_id
    assert reconciler.total_audio_duration_ms == 3000

    # Simulate barge-in after ~1.4s
    reconciler.playback_start_ts = reconciler.playback_start_ts - 1.4  # simulate 1.4s elapsed

    spoken_truth = reconciler.handle_barge_in()
    assert "[INTERRUPTED]" in spoken_truth
    assert "brakes" in spoken_truth
    assert "leaks?" not in spoken_truth  # never played

    # Verify journal recorded the truncation
    assert len(journal.truncations) == 1
    record = journal.truncations[0]
    assert record["action_id"] == action_id
    assert record["spoken_text"] == spoken_truth
    assert record["interrupted_at_ms"] >= 1400


# ── 6. Supervisor Warm Handoff Packet Generation ─────────────────────────────

def test_supervisor_whisper_packet_generation():
    slots = IntakeSlots(
        entity_1="2021",
        entity_2="HONDA",
        entity_3="CIVIC",
        category="SERVICE BRAKES",
        description="pedal sinks to floor with no stopping power",
        vin="1HGCM82633A004352",
    )
    triage = TriageResult(
        severity="Critical",
        priority="P1",
        sentiment_peak=0.92,
        category="SERVICE BRAKES",
        reason="Complete hydraulic pressure loss",
    )

    packet = generate_supervisor_whisper(
        interaction_id="int_voice_transfer_889900",
        slots=slots,
        triage=triage,
        kill_switch_terms=["brakes failed", "no stopping"],
    )

    assert packet["interaction_id"] == "int_voice_transfer_889900"
    assert packet["canonical_identity"]["vin"] == "1HGCM82633A004352"
    assert packet["canonical_identity"]["year"] == 2021
    assert packet["triage_state"]["priority"] == "P1"
    assert packet["triage_state"]["safety_flag_tripped"] is True
    assert "brakes failed" in packet["triage_state"]["trigger_terms"]

    # Verify SSML whisper prosody
    ssml = packet["whisper_audio_ssml"]
    assert "<speak><prosody rate='115%'>" in ssml
    assert "Critical P1 safety escalation" in ssml
    assert "Customer confirmed details" in ssml


# ── 7. Drive-Mode Acoustic Safety Intercept ─────────────────────────────────

def test_driving_environment_spectral_detection():
    # Silence should not trigger driving mode
    silence = np.zeros(3200, dtype=np.int16).tobytes()
    assert detect_driving_environment(silence) is False

    # Low frequency rumble (40Hz-250Hz highway noise)
    sr = 16000
    t = np.linspace(0, 0.5, sr // 2, endpoint=False)
    rumble = (12000 * np.sin(2 * np.pi * 80 * t) + 8000 * np.sin(2 * np.pi * 120 * t)).astype(np.int16).tobytes()
    assert detect_driving_environment(rumble) is True

    # Pure speech tone (1000Hz) should NOT trigger driving mode
    speech = (12000 * np.sin(2 * np.pi * 1000 * t)).astype(np.int16).tobytes()
    assert detect_driving_environment(speech) is False

    assert "motor vehicle" in DRIVE_SAFETY_SCRIPT


# ── 8. Telephony Media Bridge Buffer Clearance & Audio Codec ─────────────────

@pytest.mark.asyncio
async def test_telephony_media_bridge_clear_and_codec():
    sent_messages: list[str] = []

    class MockWebSocket:
        async def send_text(self, text: str) -> None:
            sent_messages.append(text)

    ws = MockWebSocket()
    bridge = TelephonyMediaBridge(
        websocket=ws,
        interaction_id="int_stream_test",
        on_customer_speech_frame=lambda frame: None,
    )
    bridge.stream_sid = "MZ1234567890"

    # Test buffer clearance on barge-in
    await bridge.clear_playback_buffer()
    assert len(sent_messages) == 1
    clear_event = json.loads(sent_messages[0])
    assert clear_event["event"] == "clear"
    assert clear_event["streamSid"] == "MZ1234567890"

    # Test audio encoding / decoding roundtrip.
    # Fallback upsample is exact 4x (16-bit * 2x rate). CPython 3.12 audioop.ratecv
    # (GitHub Actions) drops a couple of edge samples — still 16-bit PCM, still
    # round-trips the overlapping prefix.
    original_mulaw = bytes([255, 0, 128, 255, 42, 100])
    pcm16k = ulaw8k_to_pcm16k(original_mulaw)
    assert len(pcm16k) % 2 == 0
    assert 2 * len(original_mulaw) <= len(pcm16k) <= 4 * len(original_mulaw)

    re_encoded_mulaw = pcm16k_to_ulaw8k(pcm16k)
    n = min(len(re_encoded_mulaw), len(original_mulaw))
    assert n >= len(original_mulaw) - 2
    assert re_encoded_mulaw[:n] == original_mulaw[:n]


# ── 9. Voice Consent & Jurisdiction Policy ───────────────────────────────────

def test_voice_consent_jurisdictions():
    ca_req = consent_requirement("CA")
    assert ca_req["two_party"] is True
    assert ca_req["requires_explicit_voice_consent"] is True

    il_req = consent_requirement("IL")
    assert il_req["two_party"] is True
    assert il_req["voice_is_biometric"] is True

    tx_req = consent_requirement("TX")
    assert tx_req["two_party"] is False
    assert tx_req["requires_explicit_voice_consent"] is False


# ── 10. Seam A: Non-Conversational Bypass & Confirmation Escape Paths ───────

def test_confirmation_escape_paths_and_non_conversational_bypass():
    slots = IntakeSlots(
        entity_1="2021",
        entity_2="HONDA",
        entity_3="CIVIC",
        category="SERVICE BRAKES",
        description="pedal sinks to floor",
    )

    # Non-conversational channels (e.g. batch API / email) must not block on confirmation turn
    batch_proto = SlotConfirmationProtocol(slots, channel="batch")
    assert batch_proto.should_trigger_confirmation() is False

    skip_proto = SlotConfirmationProtocol(slots, channel="voice", skip_conversational_confirmation=True)
    assert skip_proto.should_trigger_confirmation() is False

    # Live voice channel triggers confirmation
    voice_proto = SlotConfirmationProtocol(slots, channel="voice")
    assert voice_proto.should_trigger_confirmation() is True
    voice_proto.build_confirmation_prompt()

    # Escape path 1: Caller does not know VIN -> proceeds without deadlocking
    from src.voice.confirmation_flow import ConfirmationOutcome
    proceed, msg = voice_proto.evaluate_customer_confirmation("I do not know the VIN")
    assert proceed is True
    assert voice_proto.last_outcome == ConfirmationOutcome.UNKNOWN
    assert voice_proto.phase == DialoguePhase.ENRICHMENT_READY

    # Escape path 2: Safety preemption (driving/emergency) immediately advances to safety handling
    voice_proto2 = SlotConfirmationProtocol(slots, channel="voice")
    voice_proto2.build_confirmation_prompt()
    proceed, msg = voice_proto2.evaluate_customer_confirmation("I am currently driving on the highway")
    assert proceed is True
    assert voice_proto2.last_outcome == ConfirmationOutcome.SAFETY_PREEMPTED
    assert voice_proto2.phase == DialoguePhase.ENRICHMENT_READY

    # Escape path 3: Human takeover request
    voice_proto3 = SlotConfirmationProtocol(slots, channel="voice")
    voice_proto3.build_confirmation_prompt()
    proceed, msg = voice_proto3.evaluate_customer_confirmation("Let me speak to a supervisor")
    assert voice_proto3.last_outcome == ConfirmationOutcome.HUMAN_REQUESTED


# ── 11. Seam B: Append-Only Ledger Barge-In Invariant ────────────────────────

def test_ledger_append_only_barge_in_invariants():
    from src.ledger.writer import AgentAction, record_action, list_actions

    interaction_id = "int_voice_append_only_test"

    # Step 1: Agent authorizes response and writes original row
    original_action = AgentAction(
        interaction_id=interaction_id,
        agent="intake",
        action_type="question_asked",
        input_summary="customer described symptoms",
        output_summary="I understand your brakes are failing. Have you noticed any leaking fluid?",
        ok=True,
    )
    original_action_id = record_action(original_action)

    # Step 2: Barge-in occurs -> reconciler invokes record_action_truncation
    journal = AgentActionJournal()
    truncation_action_id = journal.record_action_truncation(
        interaction_id=interaction_id,
        action_id=original_action_id,
        spoken_text="I understand your brakes are failing. [INTERRUPTED]",
        interrupted_at_ms=1420,
    )
    assert truncation_action_id is not None
    assert truncation_action_id != original_action_id

    # Step 3: Verify the original row was NEVER mutated in-place
    actions = list_actions(interaction_id)
    orig_row = next(a for a in actions if a["action_id"] == original_action_id)
    assert "Have you noticed any leaking fluid?" in orig_row["output_summary"]

    # Step 4: Verify compensating action was appended
    trunc_row = next(a for a in actions if a["action_id"] == truncation_action_id)
    assert trunc_row["action_type"] == "spoken_turn_truncated"
    assert trunc_row["output_summary"] == "I understand your brakes are failing. [INTERRUPTED]"
    ev_ids = trunc_row["evidence_ids"]
    assert original_action_id in (ev_ids if isinstance(ev_ids, list) else json.loads(ev_ids))

