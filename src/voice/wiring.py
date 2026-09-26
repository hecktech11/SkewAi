"""Production wiring for Tier-A voice modules.

Every helper here is called from the live contact path (orchestrator intake,
WS route, or Twilio media path) — not just from unit tests. This is what
converts the eight dormant modules into shipped behaviour:

  confirmation_flow, drive_mode_guard, interruption_reconciler,
  phonetic_normalizer, telephony_bridge, turn_sequencer, vin_validator,
  whisper_generator, benchmark_latency, policy.readback
"""

from __future__ import annotations

import hashlib
import re
from typing import Any


# ── TTS word alignment (browser has no word timestamps) ──────────────────────

def estimate_word_alignment(text: str, *, ms_per_word: int = 320) -> list[dict[str, Any]]:
    """Even-spacing alignment so the reconciler has timestamps to truncate.

    Browser speechSynthesis exposes no word boundaries; telephony TTS may.
    Even spacing is honest about being an estimate and is far better than
    claiming the full 30-word prompt was audible after a word-4 barge-in.
    """
    words = [w for w in re.split(r"\s+", (text or "").strip()) if w]
    out: list[dict[str, Any]] = []
    t = 0
    for w in words:
        # Longer words get proportionally more time (cheap prosody model).
        dur = int(ms_per_word * (0.7 + 0.3 * min(len(w), 10) / 5.0))
        out.append({"word": w, "start": t / 1000.0, "end": (t + dur) / 1000.0})
        t += dur
    return out


def build_reconciler(interaction_id: str):
    """Production caller for interruption_reconciler."""
    from src.ledger.journal import AgentActionJournal
    from src.voice.interruption_reconciler import SpokenPlaybackReconciler

    return SpokenPlaybackReconciler(AgentActionJournal(), interaction_id)


def reconcile_barge_in(
    interaction_id: str,
    action_id: str | None,
    text: str,
    word_markers: list[dict[str, Any]],
    elapsed_ms: int,
) -> str:
    """Production caller for SpokenPlaybackReconciler.handle_barge_in.

    Returns the audible-prefix text (with [INTERRUPTED]) and appends the
    compensating `spoken_turn_truncated` ledger row. Never raises.
    """
    try:
        rec = build_reconciler(interaction_id)
        if action_id and word_markers:
            rec.register_planned_turn(
                action_id=action_id, text=text, word_alignment=word_markers
            )
            # Playback started elapsed_ms ago: backdate the start timestamp.
            import time as _time

            rec.playback_start_ts = _time.monotonic() - (max(0, elapsed_ms) / 1000.0)
            out = rec.handle_barge_in()
            if out:
                return out
    except Exception:
        pass
    # Fallback: prefix by elapsed time at ~300ms/word so the ledger never
    # claims audio the caller could not have heard.
    words = [w for w in re.split(r"\s+", (text or "").strip()) if w]
    n = max(0, min(len(words), int(max(0, elapsed_ms) // 300)))
    audible = " ".join(words[:n]) if n else ""
    return (audible + " [INTERRUPTED]").strip() if audible else "[INTERRUPTED]"


# ── Durable turn sequencing ──────────────────────────────────────────────────

def _payload_hash(interaction_id: str, text: str, client_turn_id: str) -> str:
    h = hashlib.sha256(f"{interaction_id}|{client_turn_id}|{text or ''}".encode()).hexdigest()[:24]
    return client_turn_id or f"h_{h}"


async def acquire_durable_turn_slot(
    interaction_id: str,
    turn_seq: int | None,
    client_turn_id: str | None,
    text: str,
) -> bool:
    """Production caller for TelephonyTurnSequencer (durable turn_dedup).

    Returns True when this turn owns its execution slot; False means duplicate
    / replay and the caller must drop the turn without side effects.
    """
    try:
        from src.voice.turn_sequencer import TelephonyTurnSequencer

        seq = TelephonyTurnSequencer(interaction_id)
        try:
            seq_num = int(turn_seq) if turn_seq is not None else 0
        except (TypeError, ValueError):
            seq_num = 0
        # Web clients send monotonic seq; telephony replays may not — relax
        # ordering but keep durable payload dedup in both cases.
        return await seq.acquire_turn_execution_slot(
            seq_num,
            _payload_hash(interaction_id, text or "", (client_turn_id or "").strip()),
            bypass_sequencing=True,
        )
    except Exception:
        return True


# ── VIN / phonetic capture ───────────────────────────────────────────────────

_VIN_CANDIDATE_RE = re.compile(r"\b[A-HJ-NPR-Z0-9][A-HJ-NPR-Z0-9\s\-]{15,21}\b", re.I)


def detect_vin_in_text(text: str) -> dict[str, Any]:
    """Production caller for phonetic_normalizer + vin_validator.

    Strict on purpose: ordinary sentences ("the vehicle has been started
    ...") squash to 17-letter runs and must NEVER hijack the turn. A VIN
    candidate counts only when it carries a digit (all real VINs do —
    "HASBEENSTARTEDTHE" has none) AND an explicit signal: the word "vin"/
    "chassis" nearby, a phonetic "X as in ..." spelling, or a long
    contiguous alphanumeric token (typed VIN, no spaces).
    Returns {vin, valid, prompt}: valid VINs are storable; 17-char invalid
    VINs need an immediate re-ask; {} means no VIN-like content.
    """
    try:
        from src.voice.phonetic_normalizer import parse_spoken_alphanumerics
        from src.voice.vin_validator import validate_iso3779_vin
    except Exception:
        return {}
    raw = (text or "").strip()
    if not raw:
        return {}
    lower = raw.lower()
    has_vin_word = bool(re.search(r"\b(vin|vins|chassis(\s*number|no\.?)?|vehicle\s*id)\b", lower))
    has_phonetic = " as in " in lower
    candidates: list[tuple[str, bool]] = []  # (candidate, explicit_signal)

    def _digit_count(s: str) -> int:
        return sum(1 for c in s if c.isdigit())

    # 1. Contiguous tokens (typed VINs). A 17-digit account number has no
    # letters and no VIN word — it is not a vehicle identifier.
    for m in re.finditer(r"\b[A-HJ-NPR-Z0-9][A-HJ-NPR-Z0-9\-]{15,17}\b", raw.upper()):
        squashed = re.sub(r"[\-]", "", m.group(0))
        has_letter = any(c.isalpha() for c in squashed)
        if 16 <= len(squashed) <= 18 and _digit_count(squashed) >= 2 and (has_vin_word or has_letter):
            candidates.append((squashed, True))
    # 2. Phonetic path only with an explicit spelling signal.
    if has_phonetic or has_vin_word:
        try:
            phonetic = parse_spoken_alphanumerics(raw)
        except Exception:
            phonetic = ""
        if phonetic and 16 <= len(phonetic) <= 18 and _digit_count(phonetic) >= 2:
            candidates.append((phonetic, True))
    # 3. Spaced runs only when the caller said "vin" ( "...vin 1 H G ..." ).
    if has_vin_word:
        for m in _VIN_CANDIDATE_RE.finditer(raw.upper()):
            squashed = re.sub(r"[\s\-]", "", m.group(0))
            if 16 <= len(squashed) <= 18 and _digit_count(squashed) >= 2 \
                    and squashed not in [c for c, _ in candidates]:
                candidates.append((squashed, True))
    for cand, _explicit in candidates:
        clean = re.sub(r"[^A-HJ-NPR-Z0-9]", "", cand.upper())
        if len(clean) == 17 and _digit_count(clean) >= 2:
            valid = bool(validate_iso3779_vin(clean))
            if valid:
                return {"vin": clean, "valid": True, "prompt": ""}
            return {
                "vin": clean,
                "valid": False,
                "prompt": (
                    f"That VIN didn't pass the check digit — {clean}. "
                    "Please read it again slowly, or say the letters phonetically "
                    "like 'F as in Frank'."
                ),
            }
    return {}


# ── Slot confirmation protocol ───────────────────────────────────────────────

def confirmation_prompt_for(slots: dict[str, Any], *, channel: str) -> str | None:
    """Production caller for confirmation_flow.SlotConfirmationProtocol."""
    try:
        from src.frontline.intake import IntakeSlots
        from src.voice.confirmation_flow import SlotConfirmationProtocol
    except Exception:
        return None
    try:
        model = IntakeSlots(
            entity_1=str(slots.get("entity_1") or ""),
            entity_2=str(slots.get("entity_2") or ""),
            entity_3=str(slots.get("entity_3") or ""),
            category=str(slots.get("category") or ""),
            description=str(slots.get("description") or ""),
            vin=slots.get("vin"),
        )
        proto = SlotConfirmationProtocol(model, channel=channel)
        if proto.should_trigger_confirmation():
            return proto.build_confirmation_prompt()
    except Exception:
        return None
    return None


def evaluate_confirmation_reply(slots: dict[str, Any], *, channel: str, text: str) -> dict[str, Any]:
    """Production caller for SlotConfirmationProtocol.evaluate_customer_confirmation."""
    try:
        from src.frontline.intake import IntakeSlots
        from src.voice.confirmation_flow import SlotConfirmationProtocol
    except Exception:
        return {"ok": False, "reply": "", "outcome": "unknown"}
    try:
        model = IntakeSlots(
            entity_1=str(slots.get("entity_1") or ""),
            entity_2=str(slots.get("entity_2") or ""),
            entity_3=str(slots.get("entity_3") or ""),
            category=str(slots.get("category") or ""),
            description=str(slots.get("description") or ""),
            vin=slots.get("vin"),
        )
        proto = SlotConfirmationProtocol(model, channel=channel)
        # Re-enter pending phase: the prompt was already spoken.
        from src.voice.confirmation_flow import DialoguePhase

        proto.phase = DialoguePhase.CONFIRMATION_PENDING
        ok, reply = proto.evaluate_customer_confirmation(text or "")
        outcome = str(getattr(proto.last_outcome, "value", "unknown"))
        return {"ok": bool(ok), "reply": reply, "outcome": outcome}
    except Exception:
        return {"ok": False, "reply": "", "outcome": "unknown"}


# ── Drive mode ───────────────────────────────────────────────────────────────

def drive_hint_from_text(text: str) -> bool:
    t = (text or "").lower()
    return bool(re.search(r"\b(driving|on the road|highway|behind the wheel|pull over)\b", t))


def check_drive_mode(*, pcm_chunk: bytes | None = None, text_hint: str | None = None,
                     explicit_hint: bool | None = None) -> bool:
    """Production caller for drive_mode_guard.detect_driving_environment."""
    if explicit_hint is True:
        return True
    if text_hint and drive_hint_from_text(text_hint):
        return True
    if pcm_chunk:
        try:
            from src.voice.drive_mode_guard import detect_driving_environment

            if detect_driving_environment(pcm_chunk):
                return True
        except Exception:
            pass
    return False


def drive_safety_script() -> str:
    from src.voice.drive_mode_guard import DRIVE_SAFETY_SCRIPT

    return DRIVE_SAFETY_SCRIPT


# ── Supervisor whisper ───────────────────────────────────────────────────────

def build_supervisor_whisper(
    interaction_id: str,
    slots: dict[str, Any],
    *,
    severity: str = "Low",
    priority: str = "P3",
    safety_tripped: bool = False,
    kill_terms: list[str] | None = None,
    sentiment_peak: float = 0.0,
) -> dict[str, Any] | None:
    """Production caller for whisper_generator.generate_supervisor_whisper."""
    try:
        from src.frontline.intake import IntakeSlots
        from src.triage.triage_agent import TriageResult
        from src.voice.whisper_generator import generate_supervisor_whisper

        model = IntakeSlots(
            entity_1=str(slots.get("entity_1") or ""),
            entity_2=str(slots.get("entity_2") or ""),
            entity_3=str(slots.get("entity_3") or ""),
            category=str(slots.get("category") or ""),
            description=str(slots.get("description") or ""),
            vin=slots.get("vin"),
        )
        triage = TriageResult(
            severity=severity or "Low",
            priority=priority or "P3",
            sentiment_peak=float(sentiment_peak or 0.0),
            safety_flag_tripped=bool(safety_tripped),
        )
        return generate_supervisor_whisper(
            interaction_id, model, triage, list(kill_terms or [])
        )
    except Exception:
        return None


# ── Telephony bridge factory ─────────────────────────────────────────────────

def telephony_bridge_for(websocket: Any, interaction_id: str, on_speech, on_barge=None):
    """Production caller for TelephonyMediaBridge (Twilio WS path)."""
    from src.voice.telephony_bridge import TelephonyMediaBridge

    return TelephonyMediaBridge(
        websocket,
        interaction_id,
        on_customer_speech_frame=on_speech,
        on_barge_in_detected=on_barge,
    )


# ── Latency budget (live HUD + benchmark proof) ──────────────────────────────

def live_turn_budget_verdict(elapsed_ms: float) -> dict[str, Any]:
    """Per-turn verdict against the 180ms first-audio / 1500ms TTS budgets."""
    try:
        from src.voice.policy import latency_verdict, tts_latency_budget_ms
    except Exception:
        return {"elapsed_ms": elapsed_ms, "over_budget": False}
    return latency_verdict(int(elapsed_ms), budget_ms=tts_latency_budget_ms())


async def run_single_trial_diagnostics() -> dict[str, Any]:
    """Production caller for benchmark_latency (diagnostics endpoint).

    Runs ONE trial per stage so ops can prove the pipeline shape live without
    paying for 50 trials on the contact path.
    """
    from src.voice.benchmark_latency import TelephonyLatencyBenchmarker

    bench = TelephonyLatencyBenchmarker(trials=1)
    return await bench.run_full_benchmark()
