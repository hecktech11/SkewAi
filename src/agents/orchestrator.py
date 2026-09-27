"""Contact Orchestrator — explicit state machine.

States:
    GREETING → COLLECTING → (SAFETY_ESCALATION | SUPERVISED) → ENRICHING → CLOSING → DONE
    ABANDONED reachable from any state on hangup/timeout.

Guards:
    COLLECTING → SAFETY_ESCALATION:  kill-switch matched.
    COLLECTING → ENRICHING:          required slots filled.
    ENRICHING → CLOSING:             Triage + Sentinel + Investigator done
                                     (or FRONTLINE_ENRICH_TIMEOUT_S).
    SAFETY_ESCALATION → CLOSING:     escalation script delivered.
    CLOSING → DONE:                  case created + goodbye.

SUPERVISED (takeover): reachable from GREETING / COLLECTING / ENRICHING.
AI stops generating turns; enrichment agents keep running; every supervisor
message is ledgered (agent='supervisor', action_type='human_turn') before
delivery. Release → returns to the prior state with slot state intact.

Concurrency: enrichment agents run as asyncio tasks while Intake keeps
talking (Sentinel's entity-only advisory check can fire as soon as entity
slots fill).

Every transition writes an agent_actions row. After DONE: enqueue Qubot
post_contact_audit.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Optional

from src.agents.base import InteractionContext
from src.agents.case_agent import CaseAgent
from src.agents.intake import IntakeAgent
from src.agents.investigator import InvestigatorAgent
from src.agents.sentiment import SentimentAgent
from src.agents.sentinel import SentinelAgent
from src.agents.triage import TriageAgent
from src.config import settings
from src.domains.loader import LoadedPack, load_pack
from src.ids import new_ulid
from src.ledger import record_action

logger = logging.getLogger(__name__)



# ── States ───────────────────────────────────────────────────────────────

GREETING = "GREETING"
COLLECTING = "COLLECTING"
SAFETY_ESCALATION = "SAFETY_ESCALATION"
SUPERVISED = "SUPERVISED"
ENRICHING = "ENRICHING"
HANDOFF_PENDING = "HANDOFF_PENDING"
CLOSING = "CLOSING"
DONE = "DONE"
ABANDONED = "ABANDONED"

# Handoff SLA: supervisor pickup window before the customer is told sorry.
HANDOFF_SLA_S = 75


def _caller_requested_human(text: str) -> bool:
    """Recognise an explicit request for a live person without over-matching."""
    value = (text or "").lower()
    if not value:
        return False
    person = r"(?:human|person|representative|supervisor|live agent|real person|someone)"
    return bool(re.search(
        rf"\b(?:talk|speak|connect|transfer|get|need|want)\b.{{0,28}}\b{person}\b"
        rf"|\b{person}\b\s+(?:please|now|help)\b",
        value,
    ))


def _pack_captures_vin(pack: Any) -> bool:
    """VIN/plate capture is an automotive capability, not a global voice step."""
    pid = str(getattr(pack, "id", "") or "")
    if pid.startswith("automotive"):
        return True
    try:
        names = {s.name for s in pack.manifest.slot_frame}
    except Exception:
        names = set()
    if "vin" in names or "plate" in names:
        return True
    try:
        labels = " ".join(str(v).lower() for v in pack.entity_labels.values())
    except Exception:
        labels = ""
    return "vin" in labels or "vehicle" in labels and "year" in labels

# Deterministic wrap-up when FRONTLINE_MAX_TURNS is hit (not an abrupt drop).
BUDGET_WRAP_SCRIPT = (
    "We've captured what we have and will file this now so nothing is lost."
)

# Allowed transitions (excluding SUPERVISED which has its own enter/exit logic).
_TRANSITIONS = {
    GREETING: {COLLECTING, SUPERVISED, ABANDONED},
    # CLOSING allowed from COLLECTING for max-turns force wrap-up (partial slots).
    COLLECTING: {SAFETY_ESCALATION, ENRICHING, SUPERVISED, ABANDONED, CLOSING, HANDOFF_PENDING},
    SAFETY_ESCALATION: {CLOSING, SUPERVISED, ABANDONED},
    ENRICHING: {CLOSING, SUPERVISED, ABANDONED, HANDOFF_PENDING},
    HANDOFF_PENDING: {SUPERVISED, COLLECTING, ENRICHING, ABANDONED, CLOSING},
    CLOSING: {DONE, ABANDONED},
    DONE: set(),
    ABANDONED: set(),
    SUPERVISED: set(),                # populated dynamically — see _enter_supervised
}


# ── Hooks (for the channel / API layer) ──────────────────────────────────

@dataclass
class OrchestratorHooks:
    """Callbacks the orchestrator uses to emit output to the customer/console.

    All hooks are async. None are required — defaults are no-ops.
    """
    emit_customer_turn: Optional[Callable[[str, dict[str, Any]], Awaitable[None]]] = None
    emit_heard_turn: Optional[Callable[[str, dict[str, Any]], Awaitable[None]]] = None
    emit_activity: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None
    emit_slots_update: Optional[Callable[[dict[str, str]], Awaitable[None]]] = None
    emit_handoff_offer: Optional[Callable[[], Awaitable[None]]] = None
    emit_interaction_ended: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None
    emit_frustration_update: Optional[Callable[[float], Awaitable[None]]] = None
    emit_turn_latency: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None
    emit_consent_required: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None
    emit_supervisor_whisper: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None
    emit_control_state: Optional[Callable[[dict[str, Any]], Awaitable[None]]] = None

    async def _maybe(self, fn: Optional[Callable], *args: Any) -> None:
        if fn is not None:
            await fn(*args)


# ── Orchestrator ──────────────────────────────────────────────────────────


class Orchestrator:
    """Drives a single contact through its state machine."""

    def __init__(
        self,
        interaction_id: str,
        pack: LoadedPack,
        channel: str = "web_text",
        hooks: OrchestratorHooks | None = None,
    ) -> None:
        self.ctx = InteractionContext(
            interaction_id=interaction_id,
            pack=pack,
            channel=channel,
            case_kind="simulated" if channel == "simulated" else "customer",
        )
        self.hooks = hooks or OrchestratorHooks()
        self._pre_supervised_state: str | None = None

    # ── Public API ────────────────────────────────────────────────────────
    async def start(self, *, region: str | None = None, locale: str | None = None) -> str:
        """Emit the greeting. Returns the greeting text."""
        try:
            if region:
                self.ctx.region = str(region).strip() or None
            pack_locale = getattr(self.ctx.pack, "locale", None) or "en-US"
            self.ctx.locale = (locale or pack_locale or "en-US").strip() or "en-US"
            if self.ctx.region:
                from src.voice.policy import consent_requirement as _cr2

                _creq = _cr2(self.ctx.region)
                if _creq.get("requires_explicit_voice_consent"):
                    self.ctx.voice_consent_required = True
                    self.ctx.voice_consent_script = str(_creq.get("script") or "")
        except Exception:
            pass
        self._transition(GREETING, COLLECTING)
        greeting = self.ctx.pack.greeting
        try:
            from src.frontline.consent import with_consent

            greeting = with_consent(greeting)
        except Exception:
            pass
        # Ledger the greeting turn BEFORE emitting it (audit-first invariant:
        # no ledger row, no output). The state_transition row above covers the
        # transition; this row covers the customer-visible greeting text.
        record_action(self._orchestrator_action(
            "greeting_emitted",
            input_summary="pack greeting (fast-path, no LLM)",
            output_summary=greeting,
        ))
        await self.hooks._maybe(
            self.hooks.emit_customer_turn,
            greeting,
            {"speaker": "agent", "fast_path": True, "llm_used": False},
        )
        self.ctx.record_turn("agent", greeting, llm_used=False, latency_ms=0)
        self._record_state(COLLECTING)
        return greeting

    async def handle_customer_turn(
        self, text: str, final: bool = True, client_turn_id: str | None = None,
        asr_confidence: dict[str, float] | None = None,
        region: str | None = None, dtmf: str | None = None,
        confidence: float | None = None, turn_seq: int | None = None,
        drive_hint: bool | None = None, locale: str | None = None,
    ) -> None:
        """Process a customer turn (text from STT or text input).

        ``client_turn_id`` (board #4): reconnect retries re-send the same
        turn; a seen id is dropped after recording nothing — no double
        sentiment scoring, no double elicitation answers.
        ``asr_confidence`` 2.1: {slot: 0..1} forwarded to intake readback.
        ``confidence``: overall Web Speech confidence 0..1, retained for
        diagnostics only. It is not entity-level evidence and must never be
        guessed onto the next missing slot.
        ``region`` 2.4: two-party/biometric consent branching. ``dtmf`` 2.4.
        ``turn_seq``: monotonic client sequence for durable dedup diagnostics.
        ``drive_hint``: client-reported driving state for drive_mode_guard.
        ``locale``: BCP-47 client locale; pack locale wins when set.
        """
        # 2.1/2.4 observability: turn latency + budget
        import time as _t
        _turn_start = _t.perf_counter()
        try:
            from src.observability.metrics import observe as _obs
            _obs("turn_received", 1.0)
        except Exception:
            pass
        # 2.4 + P0 fix: region-aware consent is a GATE, not a ledger note.
        # When the region requires explicit voice consent, the contact cannot
        # proceed to intake until the caller affirmatively consents. The script
        # is spoken (customer turn) and surfaced to the UI (consent frame).
        if region:
            self.ctx.region = str(region).strip() or self.ctx.region
        if locale:
            try:
                pack_locale = getattr(self.ctx.pack, "locale", None) or "en-US"
                # Pack locale is authoritative; client locale is a fallback hint.
                if (pack_locale or "en-US") == "en-US":
                    self.ctx.locale = str(locale).strip() or self.ctx.locale
                else:
                    self.ctx.locale = pack_locale
            except Exception:
                pass
        if self.ctx.region:
            try:
                from src.voice.policy import consent_requirement as _cr
                _creq = _cr(self.ctx.region)
                if _creq.get("requires_explicit_voice_consent"):
                    self.ctx.voice_consent_required = True
                    self.ctx.voice_consent_script = str(_creq.get("script") or "")
            except Exception:
                pass
        # Web Speech supplies a confidence for the whole utterance, not for a
        # recognised vehicle entity. Mapping it to the next missing slot made
        # an ordinary report such as "my car is broken" look like a dubious
        # model year, then repeated the same bogus readback forever. Keep the
        # score for diagnostics; only channels that supply per-slot confidence
        # may trigger an entity readback.
        if (not asr_confidence) and (confidence is not None):
            try:
                c = float(confidence)
                if 0.0 <= c <= 1.0:
                    self.ctx.last_asr_confidence = c
            except Exception:
                pass
        elif isinstance(asr_confidence, dict) and asr_confidence:
            try:
                vals = [float(x) for x in asr_confidence.values()
                        if isinstance(x, (int, float))]
                if vals:
                    self.ctx.last_asr_confidence = min(vals)
            except Exception:
                pass
        if turn_seq is not None:
            try:
                self.ctx.last_turn_seq = int(turn_seq)  # diagnostics only
            except Exception:
                pass
        if self.ctx.state in (DONE, ABANDONED):
            logger.debug("turn early return: state=%s iid=%s", self.ctx.state, self.ctx.interaction_id)
            return
        if client_turn_id and self._seen_client_turn(client_turn_id):
            try:
                from src.observability.metrics import inc as _inc

                _inc("duplicate_turn_dropped")
            except Exception:
                pass
            logger.debug("duplicate client turn dropped: %s", client_turn_id)
            return

        # SUPERVISED (item 6): suppress only customer-facing *generation*.
        # Safety/scoring/ledger processing below still runs for every turn —
        # an emergency ("I'M BLEEDING") or kill-switch during takeover must
        # never be silently dropped. `supervised=True` threads through each
        # stage to withhold emissions while recording everything.
        supervised = self.ctx.state == SUPERVISED

        # ── Security: validate input before any agent logic ────────────────
        from src.security import validate_input
        v = validate_input(text)
        if not v.ok:
            record_action(self._orchestrator_action(
                "state_transition",
                input_summary="customer input rejected by security"
                + (" (supervised: generation suppressed)" if supervised else ""),
                output_summary=f"reason: {v.reason}",
            ))
            if supervised:
                return
            # Politely decline and ask for a rephrase; never surface the reason.
            decline = "I'm sorry, I didn't catch that clearly. Could you rephrase?"
            record_action(self._orchestrator_action(
                "goodbye_emitted",  # reuse a customer-turn action type
                input_summary="security rejection; canned rephrase request",
                output_summary=decline,
            ))
            await self.hooks._maybe(
                self.hooks.emit_customer_turn,
                decline,
                {"speaker": "agent", "fast_path": True, "llm_used": False},
            )
            self.ctx.record_turn("agent", decline, llm_used=False)
            return
        text = v.text

        # Record the customer turn only after it is durable. A failed insert
        # releases the dedup slot so the same client id can be retried.
        try:
            customer_turn = self.ctx.record_turn("customer", text)
        except Exception:
            if client_turn_id:
                self._forget_client_turn(str(client_turn_id))
            raise
        await self.hooks._maybe(
            getattr(self.hooks, "emit_heard_turn", None),
            text,
            {"speaker": "customer", "turn_id": customer_turn.get("turn_id")},
        )

        # ── P0: explicit voice-consent gate (two-party/biometric regions) ──
        # Until consent is obtained, intake/slots/enrichment are blocked. The
        # script is spoken once; an affirmative reply unlocks the contact.
        if self.ctx.voice_consent_required and not self.ctx.voice_consent_obtained:
            try:
                from src.frontline.elicitation import looks_like_confirmation as _llc

                if _llc(text) or text.strip().lower() in (
                    "i consent", "i agree", "yes i consent", "consent",
                ):
                    self.ctx.voice_consent_obtained = True
                    record_action(self._orchestrator_action(
                        "voice_consent_obtained",
                        input_summary=f"region={self.ctx.region} version={self.ctx.voice_consent_version}",
                        output_summary="caller consented; contact unlocked",
                    ))
                    # Consent state is recorded while a human owns the call.
                    # The acknowledgement is AI speech and must not resume intake.
                    if self.ctx.state == SUPERVISED or self.ctx.supervised:
                        return
                    ack = "Thanks — recording your consent. Now, "
                    try:
                        ack += self.ctx.pack.manifest.slot_frame[0].prompt
                    except Exception:
                        ack += "what's going on with your vehicle?"
                    await self._emit_spoken(
                        ack,
                        action_type="question_asked",
                        input_summary="consent acknowledged; next slot",
                    )
                    return
            except Exception:
                pass
            script = self.ctx.voice_consent_script or (
                "This call may be recorded for quality and safety. "
                "Say 'I consent' to continue, or type to use text instead."
            )
            record_action(self._orchestrator_action(
                "voice_consent_required",
                input_summary=f"region={self.ctx.region} gate blocks intake",
                output_summary=script[:500],
            ))
            if not supervised:
                await self.hooks._maybe(
                    self.hooks.emit_customer_turn,
                    script,
                    {"speaker": "agent", "fast_path": True, "llm_used": False,
                     "consent_required": True},
                )
                self.ctx.record_turn("agent", script, llm_used=False)
            await self.hooks._maybe(
                getattr(self.hooks, "emit_consent_required", None),
                {"region": self.ctx.region, "script": script,
                 "version": self.ctx.voice_consent_version},
            )
            return

        # A spoken request should have the same effect as the caller-facing
        # button. Previously only a prior automated offer could create the
        # handoff row, so "talk to a human" was treated as ordinary intake
        # text and the caller was left without a real pickup request.
        if not supervised and _caller_requested_human(text):
            await self.accept_handoff(force=True, source="spoken_request")
            return

        # ── Drive-mode guard (automotive safety): spoken + ledgered once ────
        try:
            from src.voice.wiring import check_drive_mode, drive_safety_script

            _driving = bool(check_drive_mode(
                text_hint=text, explicit_hint=bool(drive_hint) if drive_hint is not None else None,
            ))
            if _driving and not self.ctx.drive_mode_offered:
                self.ctx.drive_mode = True
                self.ctx.drive_mode_offered = True
                _dscript = drive_safety_script()
                record_action(self._orchestrator_action(
                    "drive_mode_offered",
                    input_summary="driving language or client drive_hint",
                    output_summary=_dscript[:500],
                ))
                await self.hooks._maybe(self.hooks.emit_activity, {
                    "agent": "drive_mode_guard",
                    "action_type": "drive_mode_detected",
                    "summary": _dscript,
                    "ok": True,
                })
                if not supervised:
                    await self.hooks._maybe(
                        self.hooks.emit_customer_turn,
                        _dscript,
                        {"speaker": "agent", "fast_path": True, "llm_used": False,
                         "drive_mode": True},
                    )
                    self.ctx.record_turn("agent", _dscript, llm_used=False)
                # Do not return: driving callers still get intake, but prompts
                # downstream stay short and the callback offer is queued.
        except Exception:
            pass

        # ── Spoken VIN / plate capture (phonetic + ISO-3779) ───────────────
        # Only packs that declare vehicle identity run this. A finance account
        # number is a 17-digit string and must stay on the finance intake path.
        try:
            from src.voice.wiring import detect_vin_in_text

            if not _pack_captures_vin(self.ctx.pack):
                _vin = {}
            else:
                _vin = detect_vin_in_text(text)
            if _vin.get("vin") and _vin.get("valid") and not self.ctx.slots.get("vin"):
                self.ctx.slots["vin"] = str(_vin["vin"])
                self.ctx.vin = str(_vin["vin"])
                record_action(self._orchestrator_action(
                    "slot_extracted",
                    input_summary=f"spoken VIN normalized: '{text[:120]}'",
                    output_summary=f"vin={_vin['vin']} iso3779=valid",
                ))
                await self.hooks._maybe(self.hooks.emit_activity, {
                    "agent": "intake",
                    "action_type": "vin_captured",
                    "summary": f"VIN {_vin['vin']} captured and check-digit verified.",
                    "ok": True,
                })
                await self.hooks._maybe(
                    self.hooks.emit_slots_update, dict(self.ctx.slots))
            elif _vin.get("vin") and not _vin.get("valid"):
                record_action(self._orchestrator_action(
                    "slot_extracted",
                    input_summary=f"spoken VIN failed check digit: {_vin.get('vin')}",
                    output_summary=_vin.get("prompt", "")[:500],
                ))
                if not supervised:
                    await self.hooks._maybe(
                        self.hooks.emit_customer_turn,
                        str(_vin.get("prompt") or "Please read the VIN again slowly."),
                        {"speaker": "agent", "fast_path": True, "llm_used": False,
                         "vin_retry": True},
                    )
                    self.ctx.record_turn(
                        "agent", str(_vin.get("prompt") or ""), llm_used=False)
                return
        except Exception:
            pass

        # Spoken confirmation of a prior diagnostic read-back.
        # Any answer (yes/no or free text) to a pending diagnostic question is
        # ledgered as customer-confirmed evidence and attached for audit.
        pending_q = (self.ctx.slots or {}).get("__diag_qid")
        if pending_q:
            from src.frontline.elicitation import (
                looks_like_confirmation,
                record_spoken_confirmation,
            )

            # Bare confirmations ("yes", "correct") close the question; longer
            # agreements ("Yes, only on cold mornings") carry evidence and
            # take the free-text path so the content is preserved.
            if looks_like_confirmation(text) and len(text.strip()) <= 24:
                record_spoken_confirmation(
                    self.ctx.interaction_id,
                    text,
                    confirmed=True,
                    question_id=pending_q,
                )
                self.ctx.slots.pop("__diag_qid", None)
            elif text and text.strip():
                # Free-text diagnostic answer: confirmed evidence with content
                record_spoken_confirmation(
                    self.ctx.interaction_id,
                    text,
                    confirmed=True,
                    question_id=pending_q,
                )
                # Persist answer provenance (item 35) — slots alone are
                # volatile; diagnostic_answers survives for brief/hypotheses.
                try:
                    from src.frontline.elicitation import record_elicitation_answer

                    record_elicitation_answer(
                        self.ctx.interaction_id,
                        str(pending_q),
                        text.strip(),
                        prompt="",
                        confirmed=True,
                    )
                except Exception:
                    pass
                # Surface the Q&A on the console activity feed (UI card).
                # Customer text is escaped: custom renderers don't auto-escape
                # (board: stored XSS); ledger/turns keep verbatim evidence.
                try:
                    from src.security.input_validation import escape_untrusted

                    await self.hooks._maybe(self.hooks.emit_activity, {
                        "agent": "intake",
                        "action_type": "diagnostic_answered",
                        "summary": f"Q {pending_q}: {escape_untrusted(text.strip()[:200])}",
                        "ok": True,
                    })
                except Exception:
                    pass
                # Keep the answer on the context for the investigation brief
                try:
                    diag_answers = self.ctx.slots.get("__diag_answers") or {}
                    if not isinstance(diag_answers, dict):
                        diag_answers = {}
                    diag_answers[str(pending_q)] = text.strip()[:1000]
                    self.ctx.slots["__diag_answers"] = diag_answers
                except Exception:
                    pass
                self.ctx.slots.pop("__diag_qid", None)
                # Re-run semantics (board #4): answers arriving AFTER
                # enrichment silently refresh the investigator brief (version
                # bump, ledgered by the agent itself) instead of leaving a
                # stale brief on the close path. Sentinel/Triage are NOT
                # re-run — severity/advisory were already decided.
                if self.ctx.enrichment_done and self.ctx.investigation_brief:
                    try:
                        refreshed = await InvestigatorAgent(self.ctx).run()
                        if isinstance(refreshed, dict) and refreshed.get("brief"):
                            prev = self.ctx.investigation_brief.get("brief_version", 1)
                            self.ctx.investigation_brief["brief_version"] = int(prev) + 1
                    except Exception:
                        pass

        # ── Sentiment (always runs; cheap, deterministic) ───────────────
        sentiment = SentimentAgent(self.ctx)
        sres = await sentiment.run(customer_turn=text)
        # Fan-out live frustration for the supervisor console (H6).
        await self.hooks._maybe(
            getattr(self.hooks, "emit_frustration_update", None),
            float(self.ctx.frustration_score or 0.0),
        )
        if sres.get("threshold_crossed") and not self.ctx.handoff_offered:
            self.ctx.handoff_offered = True
            offer = sres["handoff_offer"]
            # Audit-first: ledger the handoff offer BEFORE emitting it.
            record_action(self._orchestrator_action(
                "handoff_offer_emitted",
                input_summary="frustration threshold crossed; empathy + human offer"
                + (" (supervised: console only)" if supervised else ""),
                output_summary=offer,
            ))
            if supervised:
                # Supervisor already owns the conversation: surface on the
                # console activity feed instead of generating a customer turn.
                await self.hooks._maybe(self.hooks.emit_activity, {
                    "agent": "sentiment",
                    "action_type": "frustration_flagged",
                    "summary": offer,
                    "ok": True,
                })
            else:
                await self.hooks._maybe(
                    self.hooks.emit_customer_turn,
                    offer,
                    {"speaker": "agent", "fast_path": True, "llm_used": False},
                )
                self.ctx.record_turn("agent", offer, llm_used=False)
            await self.hooks._maybe(self.hooks.emit_handoff_offer)

        # ── Intake ────────────────────────────────────────────────────────
        # 2.3: budget_remaining logged every turn for audit of elicitation conflicts
        try:
            _ccount = self.ctx.count_turn()
            from src.config import settings as _s
            record_action(self._orchestrator_action(
                "state_transition",
                input_summary=f"turn budget: used={_ccount}/{_s.max_turns}",
                output_summary=f"budget_remaining={max(0, _s.max_turns - _ccount)}",
            ))
        except Exception:
            pass
        # 6.4: per-agent flags (intake kill → safe close, not crash)
        try:
            from src.ops.pilot import agent_enabled as _ae
            if not _ae("intake"):
                record_action(self._orchestrator_action(
                    "state_transition",
                    input_summary="intake disabled by flag",
                    output_summary="skip to closing",
                ))
                await self._move_to_closing()
                return
        except Exception:
            pass
        intake = IntakeAgent(self.ctx)
        ires = await intake.run(customer_turn=text,
                                asr_confidence=asr_confidence, dtmf=dtmf)
        # 2.1/6.3: turn latency observe vs 1200ms SLO
        try:
            import time as _t2
            from src.observability.metrics import observe as _obs2
            _obs2("turn_latency_ms", (_t2.perf_counter() - _turn_start) * 1000.0)
        except Exception:
            pass

        # Supervised safety path (item 6): kill-switch/escalation language is
        # fully processed — severity, ledger, sentinel, alert, console
        # surfacing — while staying in SUPERVISED (the supervisor, already in
        # control, decides what the customer hears).
        if supervised and ires.get("kill_switch"):
            # Keep the escalation script: if the supervisor releases with
            # safety pending, release() emits it (audit 3.3) instead of
            # dropping the escalation.
            if ires.get("question"):
                self.ctx.pending_safety_script = str(ires["question"])[:500]
            await self._supervised_safety_hit(ires["kill_switch"], text)
            return

        # Kill-switch → SAFETY_ESCALATION
        if ires.get("kill_switch"):
            self._transition(self.ctx.state, SAFETY_ESCALATION)
            # Enforce Critical/P1 before CaseAgent (Triage is skipped on this path).
            self.ctx.severity = "Critical"
            self.ctx.severity_source = "rules"
            self.ctx.priority = 1
            # Audit-first: ledger the escalation script BEFORE emitting it.
            record_action(self._orchestrator_action(
                "escalation_script_emitted",
                input_summary=f"kill-switch term: '{ires['kill_switch']}'",
                output_summary=ires["question"],
            ))
            await self.hooks._maybe(
                self.hooks.emit_customer_turn,
                ires["question"],                # the escalation script
                {"speaker": "agent", "fast_path": True, "llm_used": False},
            )
            self.ctx.record_turn("agent", ires["question"], llm_used=False)
            self._register_spoken_turn(ires["question"])
            # Supervisor whisper (Tier-A): coach on the P1 without the caller hearing.
            try:
                from src.voice.wiring import build_supervisor_whisper

                _wh = build_supervisor_whisper(
                    self.ctx.interaction_id, dict(self.ctx.slots),
                    severity="Critical", priority="P1", safety_tripped=True,
                    kill_terms=[str(ires.get("kill_switch") or "")],
                    sentiment_peak=float(self.ctx.peak_frustration or 0.0),
                )
                if _wh:
                    await self.hooks._maybe(self.hooks.emit_activity, {
                        "agent": "supervisor",
                        "action_type": "supervisor_whisper",
                        "summary": str(_wh.get("whisper_audio_ssml") or "")[:500],
                        "ok": True,
                    })
                    await self.hooks._maybe(
                        getattr(self.hooks, "emit_supervisor_whisper", None), _wh)
            except Exception:
                pass
            # Sentinel escalation in parallel
            await SentinelAgent(self.ctx).run(trigger="escalation")
            # Fire P1 safety escalation alert (non-blocking)
            try:
                from src.frontline.alerts import alert_safety_escalation
                await alert_safety_escalation(self.ctx.interaction_id, ires["kill_switch"])
            except Exception:
                pass
            try:
                from src.routing import get_circuit_breaker

                get_circuit_breaker().record_safety_evaluation(missed_safety=False)
            except Exception:
                pass
            # Best-effort enrichment on the escalation path (audit 4.1): a
            # turn-1 kill-switch fires before slots complete, and the
            # Critical case that follows is exactly where evidence matters
            # most. Short timeout, partial results kept, never blocking.
            try:
                await asyncio.wait_for(
                    self._run_enrichment(), timeout=settings.enrich_timeout_s
                )
                self.ctx.enrichment_done = True
            except asyncio.TimeoutError:
                self.ctx.enrichment_partial = True
                self.ctx.enrichment_done = True
            except Exception:
                self.ctx.enrichment_partial = True
                self.ctx.enrichment_done = True
            await self._move_to_closing()
            return

        # Normal supervised input (item 6): the turn was validated, scored,
        # and slot-processed above; record that safety processing ran and
        # return WITHOUT generating a customer-facing turn.
        if supervised:
            record_action(self._orchestrator_action(
                "supervised_turn_observed",
                input_summary=f"customer turn safety-processed during takeover",
                output_summary=(
                    f"frustration={self.ctx.frustration_score} "
                    f"peak={self.ctx.peak_frustration} "
                    f"kill_switch={bool(ires.get('kill_switch'))}; "
                    "generation suppressed"
                ),
            ))
            return

        # Max customer turns: force wrap-up (case + goodbye), not silent empty questions.
        # count_turn() is the single budget counter: confirmation, readback, and
        # elicitation replies count; supervisor turns do not.
        over_budget = self.ctx.count_turn() >= settings.max_turns
        if ires.get("max_turns_reached") or (over_budget and not self.ctx.has_required_slots()):
            await self._force_budget_wrap()
            return
        if over_budget and self.ctx.has_required_slots():
            # Slots filled at the cap: skip extra confirmation/elicitation loops.
            await self._enter_enriching()
            return

        # Confidence-gated escalate → same handoff surface as frustration (not text-only).
        if ires.get("escalate_low_confidence"):
            if not self.ctx.handoff_offered:
                self.ctx.handoff_offered = True
            offer = ires.get("question") or (
                "I'm having trouble capturing that clearly. "
                "Let me connect you with a specialist who can help."
            )
            record_action(self._orchestrator_action(
                "handoff_offer_emitted",
                input_summary=(
                    f"low confidence on slot "
                    f"{ires.get('low_confidence_slot') or 'unknown'}"
                ),
                output_summary=offer[:500],
            ))
            try:
                from src.frontline.validation_queue import enqueue_insight

                enqueue_insight(
                    kind="low_confidence",
                    interaction_id=self.ctx.interaction_id,
                    summary=(
                        f"low confidence on slot "
                        f"{ires.get('low_confidence_slot') or 'unknown'}"
                    ),
                )
            except Exception:
                pass
            await self.hooks._maybe(
                self.hooks.emit_customer_turn,
                offer,
                {
                    "speaker": "agent",
                    "fast_path": True,
                    "llm_used": False,
                    "escalate_low_confidence": True,
                },
            )
            self.ctx.record_turn("agent", offer, llm_used=False)
            await self.hooks._maybe(self.hooks.emit_handoff_offer)
            await self.hooks._maybe(self.hooks.emit_slots_update, dict(self.ctx.slots))
            return

        # Emit the next question (or empty if all slots filled)
        question = ires.get("question", "")
        safety_pending = bool(
            ires.get("safety_question")
            or (self.ctx.slots or {}).get("__safety_pending__")
            or ires.get("kill_switch")
        )
        if (
            self.ctx.has_required_slots()
            and not self.ctx.slots.get("__diag_asked")
            and not safety_pending
        ):
            from src.frontline.elicitation import next_question
            from src.ledger import record_action as _ra

            dq = next_question(
                self.ctx.pack.id, self.ctx.slots,
                interaction_id=self.ctx.interaction_id,
            )
            if dq:
                question = dq["prompt"]
                self.ctx.slots["__diag_qid"] = dq["question_id"]
                self.ctx.slots["__diag_asked"] = "1"
                _ra(
                    self._orchestrator_action(
                        "diagnostic_asked",
                        input_summary=dq["question_id"],
                        output_summary=question[:500],
                    )
                )
        if question:
            await self.hooks._maybe(
                self.hooks.emit_customer_turn,
                question,
                {"speaker": "agent", "fast_path": True, "llm_used": False},
            )
            self.ctx.record_turn("agent", question, llm_used=False)
            await self.hooks._maybe(self.hooks.emit_slots_update, dict(self.ctx.slots))

        # If slots complete AND no question is pending (safety or slot) → ENRICHING.
        # A pending question means the agent is still collecting (e.g. a safety
        # question was just asked and we must wait for the customer's answer
        # before transitioning). This preserves the one-question-per-turn policy.
        #
        # Tier-A confirmation_flow: full-frame readback on voice channels before
        # enrichment (wired, not just tested). Bounded: one prompt + one repair.
        if (
            not question
            and self.ctx.has_required_slots()
            and self.ctx.state == COLLECTING
            and not supervised
            and self.ctx.channel in ("web_voice", "voice", "telephony", "twilio_media")
        ):
            if await self._confirm_frame_gate(text):
                return
        # Slot confirmation gate (board #4): a mis-heard entity compounds
        # through cluster pick → case → corpus. One readback turn with a
        # bounded single correction round precedes enrichment.
        if (
            not question
            and self.ctx.has_required_slots()
            and self.ctx.state == COLLECTING
            and not supervised
        ):
            if await self._confirm_slots_gate(text):
                return
        if (
            not question
            and self.ctx.has_required_slots()
            and self.ctx.state == COLLECTING
        ):
            await self._enter_enriching()
        # Live latency HUD: every turn reports elapsed vs budget (non-blocking).
        try:
            import time as _t3
            from src.voice.wiring import live_turn_budget_verdict

            _elapsed = (_t3.perf_counter() - _turn_start) * 1000.0
            self.ctx.last_turn_latency_ms = float(_elapsed)
            await self.hooks._maybe(
                getattr(self.hooks, "emit_turn_latency", None),
                {"elapsed_ms": round(float(_elapsed), 1),
                 **live_turn_budget_verdict(float(_elapsed))},
            )
        except Exception:
            pass

    def note_emitted_utterance(
        self,
        text: str,
        *,
        utterance_id: str = "",
        action_id: str = "",
        turn_id: str = "",
    ) -> str:
        """Remember one generated utterance without stealing the one in playback.

        Returns the utterance id stored. A later generation is kept in the map
        but does not replace ``playing_utterance_id`` or the audible text.
        """
        try:
            from src.voice.wiring import estimate_word_alignment

            markers = estimate_word_alignment(text or "")
        except Exception:
            markers = []
        aid = (action_id or "").strip() or self._action_id_matching_text(text)
        uid = (utterance_id or turn_id or aid or "").strip()
        if not uid:
            uid = f"utt_{len(self.ctx.emitted_utterances) + 1}"
        # A transcript turn id is not a ledger action id.
        if aid and turn_id and aid == turn_id:
            aid = self._action_id_matching_text(text)
        self.ctx.emitted_utterances[uid] = {
            "text": str(text or ""),
            "action_id": aid,
            "turn_id": turn_id or "",
            "markers": markers,
        }
        if self.ctx.playing_utterance_id and self.ctx.playing_utterance_id != uid:
            return uid
        self.ctx.playing_utterance_id = uid
        self.ctx.last_agent_text = str(text or "")
        self.ctx.last_agent_action_id = aid or None
        self.ctx.last_agent_word_markers = list(markers or [])
        return uid

    def _action_id_matching_text(self, text: str) -> str:
        needle = (text or "").strip()[:80]
        if len(needle) < 8:
            return ""
        try:
            from src.ledger import list_actions

            for row in reversed(list_actions(self.ctx.interaction_id)):
                out = str(row.get("output_summary") or "")
                if needle[:40] in out:
                    return str(row.get("action_id") or "")
        except Exception:
            return ""
        return ""

    def _register_spoken_turn(self, text: str) -> None:
        """Backward-compatible alias. Does not overwrite an in-flight utterance."""
        self.note_emitted_utterance(text)

    async def handle_barge_in_interrupt(
        self, elapsed_ms: int = 0, utterance_id: str | None = None,
    ) -> str:
        """Reconcile a barge-in against the utterance the caller was hearing.

        ``utterance_id`` is the id the browser started playing. When it is
        absent, the server uses the utterance it marked as playing — not the
        newest generated text.
        """
        try:
            from src.voice.wiring import reconcile_barge_in

            uid = (utterance_id or self.ctx.playing_utterance_id or "").strip()
            rec = (self.ctx.emitted_utterances or {}).get(uid) if uid else None
            if rec:
                text = str(rec.get("text") or "")
                action_id = str(rec.get("action_id") or "")
                markers = list(rec.get("markers") or [])
            else:
                text = self.ctx.last_agent_text or ""
                action_id = str(self.ctx.last_agent_action_id or "")
                markers = list(self.ctx.last_agent_word_markers or [])
            audible = reconcile_barge_in(
                self.ctx.interaction_id,
                action_id,
                text,
                markers,
                int(elapsed_ms or 0),
            )
            record_action(self._orchestrator_action(
                "spoken_turn_reconciled",
                input_summary=(
                    f"barge_in at {int(elapsed_ms or 0)}ms"
                    f" utterance={uid or '-'} action={action_id or '-'}"
                ),
                output_summary=(audible or "[INTERRUPTED]")[:500],
            ))
            await self.hooks._maybe(self.hooks.emit_activity, {
                "agent": "voice_reconciler",
                "action_type": "spoken_turn_truncated",
                "summary": audible or "[INTERRUPTED]",
                "ok": True,
                "evidence_ids": [action_id] if action_id else [],
            })
            if uid and uid == self.ctx.playing_utterance_id:
                self.ctx.playing_utterance_id = None
            return audible or "[INTERRUPTED]"
        except Exception:
            return "[INTERRUPTED]"

    async def _confirm_frame_gate(self, text: str) -> bool:
        """Full-frame confirmation via confirmation_flow (Tier-A #7).

        Returns True when this turn is consumed by the confirmation prompt or
        its repair. Bounded to one prompt + one evaluation.
        """
        try:
            from src.voice.wiring import confirmation_prompt_for, evaluate_confirmation_reply
        except Exception:
            return False
        if self.ctx.confirmation_phase == "done":
            return False
        if self.ctx.confirmation_phase == "pending":
            try:
                res = evaluate_confirmation_reply(
                    dict(self.ctx.slots), channel=self.ctx.channel, text=text or "")
            except Exception:
                return False
            outcome = str(res.get("outcome") or "unknown")
            reply = str(res.get("reply") or "")
            record_action(self._orchestrator_action(
                "frame_confirmation_evaluated",
                input_summary=f"outcome={outcome} reply={text[:200]}",
                output_summary=reply[:500],
            ))
            if outcome in ("confirmed", "unknown", "declined_to_provide"):
                self.ctx.confirmation_phase = "done"
                return False
            if outcome == "safety_preempted":
                self.ctx.confirmation_phase = "done"
                return False
            if outcome == "human_requested":
                self.ctx.confirmation_phase = "done"
                try:
                    await self.hooks._maybe(self.hooks.emit_handoff_offer)
                except Exception:
                    pass
                return False
            # corrected / rejected: speak the repair once, then proceed.
            self.ctx.confirmation_phase = "done"
            if reply and not res.get("ok"):
                try:
                    await self.hooks._maybe(
                        self.hooks.emit_customer_turn, reply,
                        {"speaker": "agent", "fast_path": True, "llm_used": False},
                    )
                    self.ctx.record_turn("agent", reply, llm_used=False)
                    return True
                except Exception:
                    return False
            return False
        # collection → emit the frame prompt once.
        try:
            prompt = confirmation_prompt_for(dict(self.ctx.slots), channel=self.ctx.channel)
        except Exception:
            prompt = None
        if not prompt:
            return False
        self.ctx.confirmation_phase = "pending"
        self.ctx.confirmation_attempts = int(self.ctx.confirmation_attempts or 0) + 1
        record_action(self._orchestrator_action(
            "frame_confirmation_asked",
            input_summary="full slot frame readback before enrichment",
            output_summary=prompt[:500],
        ))
        await self.hooks._maybe(
            self.hooks.emit_customer_turn, prompt,
            {"speaker": "agent", "fast_path": True, "llm_used": False,
             "frame_confirmation": True},
        )
        self.ctx.record_turn("agent", prompt, llm_used=False)
        self._register_spoken_turn(prompt)
        await self.hooks._maybe(self.hooks.emit_slots_update, dict(self.ctx.slots))
        return True

    def _confirmation_summary(self) -> str:
        s = self.ctx.slots or {}
        parts = [str(s.get(k) or "").strip() for k in ("entity_1", "entity_2", "entity_3")]
        vehicle = " ".join(p for p in parts if p) or "your vehicle"
        cat = str(s.get("category") or "the issue").strip()
        return (
            f"Just to confirm before I dig in: {vehicle} — {cat}. Is that right?"
        )

    async def _confirm_slots_gate(self, text: str) -> bool:
        """Confirm-before-enrich readback. Returns True when this turn is consumed."""
        from src.frontline.elicitation import looks_like_confirmation

        slots = self.ctx.slots or {}
        if slots.get("__confirmed__"):
            return False
        if slots.get("__confirm_pending__"):
            if looks_like_confirmation(text):
                slots.pop("__confirm_pending__", None)
                slots["__confirmed__"] = "1"
                record_action(self._orchestrator_action(
                    "slot_extracted",
                    input_summary="slot confirmation accepted",
                    output_summary="slots confirmed; proceeding to enrichment",
                ))
                return False
            # Correction path: overwrite policy via targeted re-extraction.
            rounds = 0
            try:
                rounds = int(slots.get("__confirm_rounds") or 0)
            except (TypeError, ValueError):
                rounds = 0
            try:
                changed = IntakeAgent(self.ctx).reextract_overwrite(text)
            except Exception:
                changed = {}
            if changed and rounds < 1:
                slots["__confirm_rounds"] = str(rounds + 1)
                summary = self._confirmation_summary()
                record_action(self._orchestrator_action(
                    "question_asked",
                    input_summary=f"slots corrected: {sorted(changed)}",
                    output_summary=summary,
                ))
                await self.hooks._maybe(
                    self.hooks.emit_customer_turn,
                    summary,
                    {"speaker": "agent", "fast_path": True, "llm_used": False},
                )
                self.ctx.record_turn("agent", summary, llm_used=False)
                return True
            # No change extracted, or second round: proceed (bounded, no loops).
            slots.pop("__confirm_pending__", None)
            slots["__confirmed__"] = "1"
            return False
        summary = self._confirmation_summary()
        slots["__confirm_pending__"] = "1"
        slots["__confirm_rounds"] = "0"
        record_action(self._orchestrator_action(
            "question_asked",
            input_summary="slot confirmation readback",
            output_summary=summary,
        ))
        await self.hooks._maybe(
            self.hooks.emit_customer_turn,
            summary,
            {"speaker": "agent", "fast_path": True, "llm_used": False},
        )
        self.ctx.record_turn("agent", summary, llm_used=False)
        await self.hooks._maybe(self.hooks.emit_slots_update, dict(self.ctx.slots))
        return True

    async def hangup(self) -> None:
        """Customer hung up. Close with a case when there is evidence to keep.

        Rule (audit 6.1 — the worst failure mode is dropping a safety
        contact): safety_flag OR enrichment_done OR an existing case_id routes
        to Stage 5 (case created, outcome escalated_safety/case_created).
        Only a contact with no case, no safety flag, and no enrichment may
        abandon. Either way turns + interaction row persist.
        """
        if self.ctx.state == DONE:
            return
        if self.ctx.state == SUPERVISED:
            # Customer left; supervisor control ends so we can finalize.
            self.ctx.supervised = False
            self.ctx.closed_while_supervised = True
            self._pre_supervised_state = None
        must_close = bool(
            self.ctx.case_id
            or self.ctx.enrichment_done
            or any(self.ctx.safety_flags.values())
        )
        if self.ctx.state == HANDOFF_PENDING:
            # Customer left while queued: cancel the handoff row.
            try:
                from src.data.warehouse import ops_con as _ops_con2

                with _ops_con2() as _con:
                    _con.execute(
                        "UPDATE handoff_requests SET status = 'cancelled'"
                        " WHERE interaction_id = ? AND status = 'pending'",
                        [self.ctx.interaction_id],
                    )
            except Exception:
                pass
        if must_close:
            if not self.ctx.case_id:
                record_action(self._orchestrator_action(
                    "state_transition",
                    input_summary="hangup with safety/enrichment evidence",
                    output_summary="routing to close so a case is created",
                ))
            await self._move_to_closing(force=True)
        else:
            self._transition(self.ctx.state, ABANDONED)
            self._record_state(ABANDONED)
            await self._finalize_interaction(outcome=self._outcome())

    async def takeover(self, claimed_by: str | None = None) -> None:
        """Supervisor takes over. Save the pre-state; AI stops generating turns.

        Simultaneous takeovers (board): the first claimant wins; a second
        supervisor gets the live state back with the existing claimant named
        instead of silently seizing the call. Takeover stays idempotent.
        A finished contact (DONE/ABANDONED) cannot be seized.
        """
        if self.ctx.state in (SUPERVISED, DONE, ABANDONED):
            return
        was_handoff = self.ctx.state == HANDOFF_PENDING
        self.ctx.control_generation = int(self.ctx.control_generation or 0) + 1
        self.ctx.takeover_claimed_by = (claimed_by or "").strip() or None
        self.ctx.playing_utterance_id = None
        self._pre_supervised_state = self.ctx.state
        self.ctx.supervised = True
        self._transition(self.ctx.state, SUPERVISED)
        record_action(self._orchestrator_action(
            "takeover_started",
            input_summary=f"prior_state={self._pre_supervised_state}",
            output_summary="supervisor now controls the conversation",
        ))
        if was_handoff:
            # Supervisor claimed the queued handoff: close the queue row so
            # the SLA sweep never fires for a served customer.
            try:
                from src.data.warehouse import ops_con as _ops_con

                with _ops_con() as _con:
                    _con.execute(
                        "UPDATE handoff_requests SET status = 'claimed'"
                        " WHERE interaction_id = ? AND status = 'pending'",
                        [self.ctx.interaction_id],
                    )
            except Exception:
                pass
        # Fire ops alert (non-blocking, deduped per interaction per day)
        try:
            from src.frontline.alerts import alert_takeover_started
            await alert_takeover_started(self.ctx.interaction_id)
        except Exception:
            pass  # alerts must never break a call
        try:
            self._record_state(SUPERVISED)
        except Exception:
            pass
        await self._publish_control()

    def _actor_may_control(self, actor: str | None, *, admin_override: bool = False) -> bool:
        """Owner check for send/release.

        A claim with no subject (historical or service-only) stays controllable
        so existing calls are not stranded. A named claimant is exclusive.
        """
        if admin_override:
            return True
        owner = (self.ctx.takeover_claimed_by or "").strip()
        if not owner:
            return True
        return (actor or "").strip() == owner

    async def release(self, actor: str | None = None, *, admin_override: bool = False) -> dict[str, Any]:
        """Supervisor releases. Next state is recomputed from facts (audit 3.3).

        Restoring the literal pre-takeover state is stale when enrichment
        finished or safety fired while supervised: safety escalation pending
        → SAFETY_ESCALATION (stored script delivered); enrichment done →
        CLOSING; slots complete → ENRICHING; else the prior state.
        """
        if self.ctx.state != SUPERVISED:
            return {"ok": False, "code": "not_supervised", "state": self.ctx.state}
        if not self._actor_may_control(actor, admin_override=admin_override):
            return {
                "ok": False,
                "code": "not_owner",
                "claimed_by": self.ctx.takeover_claimed_by,
                "state": self.ctx.state,
            }
        self.ctx.control_generation = int(self.ctx.control_generation or 0) + 1
        if self.ctx.safety_flags.get("escalation"):
            target: str | None = "SAFETY_ESCALATION"
        elif self.ctx.enrichment_done:
            target = "CLOSING"
        elif self.ctx.has_required_slots():
            target = "ENRICHING"
        else:
            target = self._pre_supervised_state or COLLECTING
        if target == "CLOSING":
            # _move_to_closing owns the SUPERVISED→CLOSING transition itself.
            self.ctx.supervised = False
            self.ctx.takeover_claimed_by = None
            self._pre_supervised_state = None
            record_action(self._orchestrator_action(
                "takeover_released",
                input_summary="returning to state=CLOSING (recomputed)",
                output_summary="AI resumes",
            ))
            # FU06: Publish control update FIRST so browser humanControlRef is cleared
            # before the closing turn is emitted to the customer.
            await self._publish_control()
            await self._move_to_closing(force=True)
            return {
                "ok": True,
                "state": self.ctx.state,
                "generation": self.ctx.control_generation,
            }
        self._transition(SUPERVISED, target)
        # Clear the supervised flag so the console / audit reflects that the
        # AI has resumed. (takeover sets this to True; release must reset it.)
        self.ctx.supervised = False
        self.ctx.takeover_claimed_by = None
        self._pre_supervised_state = None
        record_action(self._orchestrator_action(
            "takeover_released",
            input_summary=f"returning to state={target} (recomputed) actor-released",
            output_summary="AI resumes",
        ))
        try:
            self._record_state(target)
        except Exception:
            pass
        # FU06: Publish control update FIRST so browser humanControlRef is cleared
        # before the AI emits safety scripts or resumed conversational turns.
        await self._publish_control()
        if target == "SAFETY_ESCALATION":
            await self._deliver_pending_safety_script()
        elif target == "ENRICHING":
            await self._enter_enriching()
        return {
            "ok": True,
            "state": self.ctx.state,
            "generation": self.ctx.control_generation,
        }

    async def _deliver_pending_safety_script(self) -> None:
        """Emit the stored escalation script post-release (audit 3.3)."""
        script = (self.ctx.pending_safety_script or "").strip() or (
            "For your safety, please pull over somewhere safe right away. "
            "If anyone is hurt, call emergency services now."
        )
        self.ctx.pending_safety_script = None
        self.ctx.severity = "Critical"
        self.ctx.severity_source = "rules"
        self.ctx.priority = 1
        aid = record_action(self._orchestrator_action(
            "escalation_script_emitted",
            input_summary="safety escalation raised while supervised; delivered on release",
            output_summary=script,
        ))
        turn = self.ctx.record_turn("agent", script, llm_used=False, action_id=aid)
        await self.hooks._maybe(
            self.hooks.emit_customer_turn,
            script,
            {
                "speaker": "agent",
                "fast_path": True,
                "llm_used": False,
                "action_id": aid,
                "turn_id": turn.get("turn_id"),
            },
        )
        await self._move_to_closing()

    async def human_turn(
        self,
        text: str,
        *,
        actor: str | None = None,
        message_id: str | None = None,
    ) -> dict[str, Any]:
        """Supervisor typed a message. Ledger it BEFORE emitting.

        Supervisor turns pass the same input validation as customer turns
        before broadcast — the audit-first invariant applies to humans too.
        Returns an explicit acceptance or rejection. Duplicate message ids
        do not speak twice.
        """
        mid = (message_id or "").strip()
        if self.ctx.state != SUPERVISED:
            return {"ok": False, "code": "not_supervised", "message_id": mid}
        if not self._actor_may_control(actor):
            return {
                "ok": False,
                "code": "not_owner",
                "claimed_by": self.ctx.takeover_claimed_by,
                "message_id": mid,
            }
        if mid and mid in self.ctx.human_message_ids:
            return {"ok": True, "duplicate": True, "message_id": mid}
        from src.security import validate_input
        v = validate_input(text)
        if not v.ok:
            record_action(self._orchestrator_action(
                "state_transition",
                input_summary=f"actor={actor or 'unknown'} supervisor input rejected",
                output_summary=f"reason: {v.reason}",
            ))
            return {"ok": False, "code": "rejected", "detail": v.reason, "message_id": mid}
        text = v.text
        who = (actor or "unknown").strip() or "unknown"
        action_id = record_action(self._orchestrator_action(
            "human_turn",
            input_summary=f"actor={who} message_id={mid or '-'}",
            output_summary=text,
        ))
        turn = self.ctx.record_turn("supervisor", text, action_id=action_id)
        if mid:
            self.ctx.human_message_ids.add(mid)
        await self._emit_spoken(
            text,
            action_type="human_turn",
            input_summary=f"actor={who}",
            speaker="supervisor",
            ledger=False,
            action_id=action_id,
            turn=turn,
        )
        return {"ok": True, "message_id": mid, "action_id": action_id, "duplicate": False}

    def _forget_client_turn(self, client_turn_id: str) -> None:
        """Drop a dedup row when the turn was not durably accepted."""
        tid = (client_turn_id or "").strip()
        if not tid:
            return
        from src.data.warehouse import ops_con

        try:
            with ops_con() as con:
                con.execute(
                    "DELETE FROM turn_dedup WHERE interaction_id = ? AND client_turn_id = ?",
                    [self.ctx.interaction_id, tid],
                )
        except Exception:
            pass

    def _seen_client_turn(self, client_turn_id: str) -> bool:
        """Idempotency check for retried turns (board #4).

        INSERT-or-detect on ``turn_dedup``: the PK makes concurrent retries
        collapse too (loser hits IntegrityError → treated as duplicate).
        Returns True when this turn was already processed.
        """
        tid = (client_turn_id or "").strip()
        if not tid:
            return False
        from src.data.timeutil import utc_now
        from src.data.warehouse import ops_con

        try:
            with ops_con() as con:
                seen = con.execute(
                    "SELECT 1 FROM turn_dedup WHERE interaction_id = ? AND client_turn_id = ?",
                    [self.ctx.interaction_id, tid],
                ).fetchone()
                if seen:
                    return True
                con.execute(
                    "INSERT INTO turn_dedup (interaction_id, client_turn_id, seen_at)"
                    " VALUES (?, ?, ?)",
                    [self.ctx.interaction_id, tid, utc_now()],
                )
            return False
        except Exception:
            # PK race lost → duplicate. Any other DB failure → process the
            # turn (dropping customer input is worse than double-processing).
            try:
                with ops_con(read_only=True) as con:
                    seen = con.execute(
                        "SELECT 1 FROM turn_dedup WHERE interaction_id = ? AND client_turn_id = ?",
                        [self.ctx.interaction_id, tid],
                    ).fetchone()
                    return bool(seen)
            except Exception:
                return False

    # ── Handoff queue (audit 2.4) ──────────────────────────────────────────
    async def accept_handoff(
        self, *, force: bool = False, source: str = "handoff_offer"
    ) -> dict[str, Any]:
        """Customer accepted the handoff offer: park in HANDOFF_PENDING.

        Pushes a supervisor-queue row with a 60–90s SLA and notifies the
        console. A supervisor claims it via takeover(); if nobody picks up,
        the reaper sweep restores the prior state, apologises, and ledgers
        handoff_unfulfilled — the offer is never a dead end.
        """
        if self.ctx.state == HANDOFF_PENDING:
            # Re-clicking the visible handoff control must not make duplicate
            # queue rows or tell the caller that their request failed.
            return {
                "accepted": True,
                "already_pending": True,
                "handoff_id": None,
                "sla_s": HANDOFF_SLA_S,
            }
        if self.ctx.state not in (COLLECTING, ENRICHING):
            return {"accepted": False, "reason": f"state={self.ctx.state}"}
        if not self.ctx.handoff_offered:
            if not force:
                return {"accepted": False, "reason": "no offer outstanding"}
            # A caller pressing "Get me a human" is an explicit request, not a
            # failed attempt to accept an offer that has not been shown yet.
            self.ctx.handoff_offered = True
            record_action(self._orchestrator_action(
                "handoff_offer_emitted",
                input_summary=f"explicit human request via {source}",
                output_summary="caller requested a human specialist",
            ))
        from datetime import timedelta as _td

        from src.data.timeutil import utc_now
        from src.data.warehouse import ops_con
        from src.ids import new_ulid

        sla_due = utc_now() + _td(seconds=HANDOFF_SLA_S)
        hid = "hfr_" + new_ulid()
        with ops_con() as con:
            con.execute(
                """
                INSERT INTO handoff_requests
                (handoff_id, interaction_id, status, prior_state, sla_due_at,
                 claimed_by, created_at)
                VALUES (?, ?, 'pending', ?, ?, NULL, ?)
                """,
                [hid, self.ctx.interaction_id, self.ctx.state, sla_due, utc_now()],
            )
        self._pre_supervised_state = self.ctx.state
        self._transition(self.ctx.state, HANDOFF_PENDING)
        self._record_state(HANDOFF_PENDING)
        record_action(self._orchestrator_action(
            "handoff_accepted",
            input_summary="customer accepted human handoff",
            output_summary=f"queued for supervisor pickup; SLA {HANDOFF_SLA_S}s",
        ))
        await self.hooks._maybe(self.hooks.emit_activity, {
            "agent": "orchestrator",
            "action_type": "handoff_accepted",
            "summary": (
                f"Customer accepted handoff — pickup SLA {HANDOFF_SLA_S}s "
                f"(queue {hid})."
            ),
            "ok": True,
        })
        # Notify both caller and console that this is now a live queue item.
        # This also keeps a spoken request visually consistent with the button.
        await self.hooks._maybe(self.hooks.emit_handoff_offer)
        await self.hooks._maybe(
            self.hooks.emit_customer_turn,
            "Connecting you to a human specialist now — thanks for holding.",
            {"speaker": "agent", "fast_path": True, "llm_used": False},
        )
        self.ctx.record_turn(
            "agent", "Connecting you to a human specialist now.", llm_used=False
        )
        return {"accepted": True, "handoff_id": hid, "sla_s": HANDOFF_SLA_S}

    async def sweep_handoff_timeout(self) -> bool:
        """Expire our pending handoff when the SLA lapsed (audit 2.4).

        Returns True when a timeout was processed. Restores the prior state,
        apologises to the customer, ledgers handoff_unfulfilled, and resumes
        the AI flow — no dead end.
        """
        if self.ctx.state != HANDOFF_PENDING:
            return False
        from src.data.timeutil import utc_now
        from src.data.warehouse import ops_con

        with ops_con() as con:
            row = con.execute(
                """
                SELECT handoff_id, prior_state, sla_due_at FROM handoff_requests
                WHERE interaction_id = ? AND status = 'pending'
                ORDER BY created_at DESC LIMIT 1
                """,
                [self.ctx.interaction_id],
            ).fetchone()
        if not row:
            return False
        hid, prior, sla_due = row[0], row[1], row[2]
        try:
            expired = utc_now().replace(tzinfo=None) >= sla_due.replace(tzinfo=None)
        except Exception:
            expired = True
        if not expired:
            return False
        with ops_con() as con:
            con.execute(
                "UPDATE handoff_requests SET status = 'unfulfilled' WHERE handoff_id = ?",
                [hid],
            )
        resume = prior or COLLECTING
        try:
            self._transition(HANDOFF_PENDING, resume)
        except ValueError:
            self._transition(HANDOFF_PENDING, COLLECTING)
            resume = COLLECTING
        self._record_state(resume)
        record_action(self._orchestrator_action(
            "handoff_unfulfilled",
            input_summary=f"no supervisor pickup within {HANDOFF_SLA_S}s",
            output_summary=f"resumed {resume}; callback offered",
        ))
        generation = int(self.ctx.control_generation or 0)
        await self.hooks._maybe(self.hooks.emit_activity, {
            "agent": "orchestrator",
            "action_type": "handoff_unfulfilled",
            "summary": "Handoff SLA expired with no pickup; AI resumed.",
            "ok": False,
        })
        if (
            self.ctx.state == SUPERVISED
            or self.ctx.supervised
            or int(self.ctx.control_generation or 0) != generation
        ):
            return True
        apology = (
            "I'm sorry for the wait — no specialist was available. "
            "I can keep helping, or arrange a callback at a better time."
        )
        await self._emit_spoken(
            apology,
            action_type="question_asked",
            input_summary="handoff SLA expired",
        )
        if (
            self.ctx.state == SUPERVISED
            or self.ctx.supervised
            or int(self.ctx.control_generation or 0) != generation
        ):
            return True
        # The callback offer is a real queue row, not just words: staff work
        # it from the existing callback queue (board: no dead-end offers).
        try:
            from src.frontline.callback import schedule_callback

            schedule_callback(
                interaction_id=self.ctx.interaction_id,
                pack_id=self.ctx.pack.id,
                phone_or_channel="console-callback",
                case_id=self.ctx.case_id,
                reason="handoff_unfulfilled",
            )
        except Exception:
            pass
        return True

    # ── Supervised safety (item 6) ─────────────────────────────────────────
    async def _supervised_safety_hit(self, kill_term: str, text: str) -> None:
        """Process a kill-switch/safety hit observed during SUPERVISED takeover.

        Mirrors the autonomous SAFETY_ESCALATION scoring (Critical/P1,
        ledger, sentinel, alert) but stays in SUPERVISED: the supervisor is
        already live on the conversation and decides what the customer hears.
        The hit is surfaced on the console activity feed — never dropped.
        """
        from src.security.input_validation import escape_untrusted as _escape_untrusted

        self.ctx.severity = "Critical"
        self.ctx.severity_source = "rules"
        self.ctx.priority = 1
        self.ctx.safety_flags["escalation"] = True
        record_action(self._orchestrator_action(
            "supervised_safety_raised",
            input_summary=f"kill-switch term during takeover: '{kill_term}'",
            output_summary=(
                "severity=Critical P1 recorded; supervisor owns response; "
                "no autonomous customer turn generated"
            ),
        ))
        await self.hooks._maybe(self.hooks.emit_activity, {
            "agent": "sentinel",
            "action_type": "escalated",
            "summary": (
                f"SAFETY during takeover — customer said: "
                f"{_escape_untrusted(text[:200])} "
                f"(trigger: '{kill_term}'). Severity set Critical/P1."
            ),
            "ok": True,
        })
        await self.hooks._maybe(
            getattr(self.hooks, "emit_frustration_update", None),
            float(self.ctx.frustration_score or 0.0),
        )
        await SentinelAgent(self.ctx).run(trigger="escalation")
        try:
            from src.frontline.alerts import alert_safety_escalation
            await alert_safety_escalation(self.ctx.interaction_id, kill_term)
        except Exception:
            pass  # alerts must never break safety processing

    # ── Enrichment ────────────────────────────────────────────────────────
    async def _force_budget_wrap(self) -> None:
        """Close at FRONTLINE_MAX_TURNS with a ledgered canned script."""
        record_action(self._orchestrator_action(
            "goodbye_emitted",
            input_summary=f"max_turns={settings.max_turns} force wrap-up",
            output_summary=BUDGET_WRAP_SCRIPT,
        ))
        await self.hooks._maybe(
            self.hooks.emit_customer_turn,
            BUDGET_WRAP_SCRIPT,
            {"speaker": "agent", "fast_path": True, "llm_used": False, "max_turns_reached": True},
        )
        self.ctx.record_turn("agent", BUDGET_WRAP_SCRIPT, llm_used=False)
        await self._move_to_closing()

    async def _enter_enriching(self) -> None:
        # Idempotent entry (audit 3.3 release path may already sit here).
        if self.ctx.state != ENRICHING:
            self._transition(self.ctx.state, ENRICHING)
        self._record_state(ENRICHING)

        # Sentinel can fire as soon as entities are present (already true here).
        # Triage + Investigator run after category+description are present.
        try:
            await asyncio.wait_for(
                self._run_enrichment(), timeout=settings.enrich_timeout_s
            )
        except asyncio.TimeoutError:
            self.ctx.enrichment_partial = True
            record_action(self._orchestrator_action(
                "state_transition",
                input_summary=f"enrichment timeout after {settings.enrich_timeout_s}s",
                output_summary="moving to CLOSING with partial enrichment enrichment_partial=true",
            ))
        finally:
            # Enrichment ran (even partially/timed-out): a later hangup must
            # close with a case, never abandon the evidence (audit 6.1).
            self.ctx.enrichment_done = True

        await self._move_to_closing()

    async def _run_enrichment(self) -> None:
        """Run independent specialists concurrently (P1: was sequential awaits).

        Sentinel, Triage, and Investigator write distinct ctx fields and do not
        depend on each other's outputs for the pilot path.

        Failure isolation (item 8): each agent runs under structured result
        handling (``return_exceptions=True``). A DB/ML/network failure in one
        agent is ledgered as that agent's failed action and the other agents'
        partial results are still used — the contact is never stranded in
        ENRICHING because of a single dependency.
        """
        from src.ledger import AgentAction
        from src.ledger.writer import remember_interaction_pack
        from src.observability.otel import start_span

        remember_interaction_pack(self.ctx.interaction_id, self.ctx.pack.id)
        sentinel = SentinelAgent(self.ctx)
        triage = TriageAgent(self.ctx)
        inv = InvestigatorAgent(self.ctx)

        with start_span(
            "orchestrator.enrichment",
            attributes={"interaction_id": self.ctx.interaction_id},
        ):
            raw = await asyncio.gather(
                sentinel.run(trigger="entities_complete"),
                triage.run(),
                inv.run(),
                return_exceptions=True,
            )

        def _settle(
            agent: str,
            action_type: str,
            value: Any,
        ) -> dict[str, Any]:
            if isinstance(value, BaseException):
                # Isolated failure: observable, partial-results-safe.
                try:
                    record_action(AgentAction(
                        interaction_id=self.ctx.interaction_id,
                        agent=agent,
                        action_type=action_type,
                        input_summary="enrichment agent raised",
                        output_summary=(
                            f"partial enrichment: {agent} failed; "
                            "siblings preserved"
                        ),
                        ok=False,
                        error=f"{type(value).__name__}: {value}"[:500],
                        case_id=self.ctx.case_id,
                    ))
                except Exception:
                    pass
                self.ctx.enrichment_partial = True
                self.ctx.enrichment_degraded = True
                if agent == "investigator":
                    # Audit 4.1: a failed investigator must not leave the case
                    # evidence-free forever — enqueue a leased, retried
                    # backfill that patches the brief post-close.
                    try:
                        from src.jobs.queue import enqueue

                        enqueue(
                            "reenrich",
                            {
                                "interaction_id": self.ctx.interaction_id,
                                "case_id": self.ctx.case_id,
                                "reason": f"{type(value).__name__}: {value}"[:200],
                            },
                        )
                    except Exception:
                        pass
                return {
                    "skipped": True,
                    "reason": f"{agent} failed: {type(value).__name__}",
                    "error": f"{type(value).__name__}: {value}"[:500],
                }
            if isinstance(value, dict):
                return value
            return {"skipped": True, "reason": f"{agent} returned no result"}

        sres = _settle("sentinel", "advisory_check", raw[0])
        tres = _settle("triage", "severity_scored", raw[1])
        ires = _settle("investigator", "brief_written", raw[2])

        if sres.get("advisory_match"):
            notice = sres["notice"]
            await self.hooks._maybe(
                self.hooks.emit_customer_turn,
                notice,
                {"speaker": "agent", "fast_path": True, "llm_used": False},
            )
            self.ctx.record_turn("agent", notice, llm_used=False)
            await self.hooks._maybe(self.hooks.emit_activity, {
                "agent": "sentinel",
                "action_type": "advisory_check",
                "summary": notice,
                "evidence_ids": sres.get("evidence_ids", []),
                "ok": True,
            })

        if not tres.get("skipped"):
            await self.hooks._maybe(self.hooks.emit_activity, {
                "agent": "triage",
                "action_type": "severity_scored",
                "summary": f"severity={tres['severity']} (P{tres['priority']})",
                "ok": True,
            })

        if not ires.get("skipped"):
            await self.hooks._maybe(self.hooks.emit_activity, {
                "agent": "investigator",
                "action_type": "brief_written",
                "summary": ires.get("narration", ""),
                "evidence_ids": ires.get("evidence_ids", []),
                "ok": True,
            })

        # Live intercept: re-score this slice; open investigation before close.
        hit: dict = {}
        try:
            from src.frontline.live_intercept import intercept_contact

            hit = await asyncio.to_thread(intercept_contact, self.ctx) or {}
        except Exception:
            hit = {}
        if hit.get("investigation_id"):
            try:
                from src.frontline.alerts import alert_early_warning, alert_investigation_opened

                sl = hit.get("slice") or {}
                cid = 0
                if getattr(self.ctx, "investigation_brief", None):
                    cid = int(self.ctx.investigation_brief.get("cluster_id") or 0)
                await alert_investigation_opened(
                    hit["investigation_id"],
                    cid,
                    f"Live intercept {hit.get('investigation_id')}",
                    pack_id=self.ctx.pack.id,
                )
                risk = hit.get("live_risk") or []
                if risk:
                    top = risk[0]
                    score = float(top.get("live_risk_score") or 0.0)
                    if score >= float(settings.early_warning_alert_threshold):
                        await alert_early_warning(
                            int(top.get("cluster_id") or cid or 0),
                            int(top.get("live_case_count") or 0),
                            top.get("lead_time_weeks"),
                            pack_id=self.ctx.pack.id,
                        )
            except Exception:
                pass
            try:
                from src.frontline.remedy import build_remedy_offer

                pack_name = getattr(self.ctx.pack, "display_name", None) or self.ctx.pack.id
                offer = build_remedy_offer(
                    advisory_match=self.ctx.advisory_match,
                    case_id=self.ctx.case_id,
                    pack_display_name=str(pack_name),
                )
                if offer and offer.get("customer_text"):
                    await self._emit_ledgered_remedy(offer)
            except Exception:
                pass

    # ── Closing ───────────────────────────────────────────────────────────
    async def _move_to_closing(self, *, force: bool = False) -> None:
        if self.ctx.state == CLOSING or self.ctx.state == DONE:
            return
        if self.ctx.state == ABANDONED:
            # Customer already gone and finalized: never create a case after
            # abandon (late enrichment completions land here). Idempotent.
            return
        if self.ctx.state == SUPERVISED and not force:
            return  # supervisor controls; don't auto-close mid-call
        if self.ctx.state == SUPERVISED and force:
            self.ctx.supervised = False
        # Distributed close claim (item 31): exactly one worker in the fleet
        # may run the close body. In-process state guards cover one process;
        # this covers replicas sharing the ops DB (Redis NX when configured).
        from src.jobs.registry import acquire_close_claim_info, heartbeat_close_claim

        _claim_owner = f"orch-{self.ctx.interaction_id}"
        _claim_takeover = False
        try:
            _claim = acquire_close_claim_info(self.ctx.interaction_id, _claim_owner)
            if not _claim["acquired"]:
                record_action(self._orchestrator_action(
                    "state_transition",
                    input_summary="close already claimed by another worker",
                    output_summary="standing down; no duplicate case",
                ))
                return
            _claim_takeover = bool(_claim["takeover"])
            # Heartbeat so a slow-but-alive closer keeps ownership while a
            # truly dead worker's claim expires and becomes re-acquirable
            # (audit 5.1).
            heartbeat_close_claim(self.ctx.interaction_id, _claim_owner)
        except Exception:
            pass
        if _claim_takeover and not self.ctx.case_id:
            # Crash-retry (audit 5.1): the previous worker may have inserted
            # the case after claiming. Reuse it instead of allocating new.
            try:
                from src.data.warehouse import ops_con as _ops_con2

                with _ops_con2(read_only=True) as _con:
                    _row = _con.execute(
                        "SELECT case_id FROM cases WHERE interaction_id = ?"
                        " ORDER BY created_at LIMIT 1",
                        [self.ctx.interaction_id],
                    ).fetchone()
                if _row and _row[0]:
                    self.ctx.case_id = str(_row[0])
            except Exception:
                pass
        self._transition(self.ctx.state, CLOSING)
        self._record_state(CLOSING)

        # Pre-close self-critique (feature #20) — soft by default
        try:
            from src.agents.self_critique import run_self_critique

            critique = run_self_critique(self.ctx, strict=False)
            self.ctx.self_critique = critique
        except Exception:
            pass

        # Case agent
        case_agent = CaseAgent(self.ctx)
        # Returning-customer attach (audit 1.1): same hashed identity +
        # same entity/category with an OPEN case in the window → attach this
        # contact's turns to the existing case instead of opening a second
        # one (which would double-count the corpus and split context).
        attached = self._attach_returning_case()
        if attached:
            self.ctx.case_id = attached
        cres = await case_agent.run()
        case_id = cres["case_id"]

        # Goodbye with case number + investigation mention
        goodbye = self.ctx.pack.manifest.goodbye.replace("{case_id}", case_id)
        if cres.get("investigation_opened"):
            goodbye += " You're not alone — this issue is under active investigation."
        # Customer-facing remedy/next-step when advisory matched
        offer = cres.get("remedy_offer") or getattr(self.ctx, "remedy_offer", None)
        goodbye_evidence: list[str] = []
        if isinstance(offer, dict) and offer.get("customer_text"):
            goodbye = f"{goodbye} {offer['customer_text']}"
            if offer.get("advisory_id"):
                goodbye_evidence.append(str(offer["advisory_id"]))
        linked = cres.get("linked_issues") or []
        if len(linked) > 1:
            goodbye += f" We also opened {len(linked) - 1} linked issue(s) from this contact."

        # Audit-first invariant: ledger the goodbye BEFORE emitting it.
        # Cite advisory IDs present in remedy text so Qubot uncited-ID gate stays green.
        from src.ledger import AgentAction

        record_action(
            AgentAction(
                interaction_id=self.ctx.interaction_id,
                agent="orchestrator",
                action_type="goodbye_emitted",
                input_summary="pack goodbye (fast-path, no LLM)",
                output_summary=goodbye[:500],
                evidence_ids=goodbye_evidence,
                case_id=case_id,
            )
        )
        await self.hooks._maybe(
            self.hooks.emit_customer_turn,
            goodbye,
            {"speaker": "agent", "fast_path": True, "llm_used": False},
        )
        self.ctx.record_turn("agent", goodbye, llm_used=False)

        self._transition(CLOSING, DONE)
        self._record_state(DONE)
        await self._finalize_interaction(outcome=self._outcome())

        await self.hooks._maybe(self.hooks.emit_interaction_ended, {
            "case_id": case_id,
            "investigation_id": cres.get("investigation_id"),
            "investigation_opened": cres.get("investigation_opened"),
            "audit_pending": True,
        })

    # ── Helpers ────────────────────────────────────────────────────────────
    def _transition(self, from_state: str, to_state: str) -> None:
        allowed = _TRANSITIONS.get(from_state, set())
        if from_state == SUPERVISED:
            # Release may return to the pre-takeover state; hangup must always
            # be able to leave SUPERVISED (docstring: ABANDONED from any state).
            # Release recomputes from facts (audit 3.3), so escalation and
            # enrichment targets are legal exits too.
            resume = {self._pre_supervised_state} if self._pre_supervised_state else {
                COLLECTING, ENRICHING, CLOSING
            }
            allowed = resume | {
                ABANDONED, CLOSING, DONE, SAFETY_ESCALATION, ENRICHING,
                COLLECTING, HANDOFF_PENDING,
            }
        if to_state not in allowed and from_state != to_state:
            raise ValueError(f"Illegal transition: {from_state} → {to_state}")
        self.ctx.state = to_state
        record_action(self._orchestrator_action(
            "state_transition",
            input_summary=f"from={from_state}",
            output_summary=f"to={to_state}",
        ))

    def _record_state(self, state: str) -> None:
        # Persist the latest state + slot values + frustration on the interaction row.
        from src.data.warehouse import ops_con
        from src.frontline.dsr import is_interaction_erased

        if getattr(self, "_erased", False) or is_interaction_erased(self.ctx.interaction_id):
            self._erased = True
            self.ctx.slots.clear()
            with ops_con() as con:
                con.execute(
                    """
                    UPDATE interactions
                    SET status = ?, peak_frustration = ?, last_frustration = ?,
                        supervised = ?
                    WHERE interaction_id = ? AND NOT COALESCE(erased, FALSE)
                    """,
                    [
                        "abandoned" if state == ABANDONED else ("active" if state not in (DONE,) else "completed"),
                        self.ctx.peak_frustration,
                        self.ctx.frustration_score,
                        self.ctx.supervised,
                        self.ctx.interaction_id,
                    ],
                )
            return

        with ops_con() as con:
            con.execute(
                """
                UPDATE interactions
                SET status = ?, peak_frustration = ?, last_frustration = ?,
                    supervised = ?, entity_1 = ?, entity_2 = ?, entity_3 = ?,
                    category = ?, description = ?
                WHERE interaction_id = ? AND NOT COALESCE(erased, FALSE)
                """,
                [
                    "abandoned" if state == ABANDONED else ("active" if state not in (DONE,) else "completed"),
                    self.ctx.peak_frustration,
                    self.ctx.frustration_score,
                    self.ctx.supervised,
                    self.ctx.slots.get("entity_1"),
                    self.ctx.slots.get("entity_2"),
                    self.ctx.slots.get("entity_3"),
                    self.ctx.slots.get("category"),
                    (self.ctx.slots.get("description") or "")[:500],
                    self.ctx.interaction_id,
                ],
            )

    def _outcome(self) -> str:
        # Human-touch outcomes (board) take precedence over the plain close:
        # a claimed handoff that ended in a case was handed off; a case
        # forced closed while supervised was human-resolved.
        if self.ctx.case_id and self._handoff_was_claimed():
            return "handed_off"
        if self.ctx.case_id and self.ctx.closed_while_supervised:
            return "human_resolved"
        if self.ctx.safety_flags.get("escalation"):
            return "escalated_safety"
        if self.ctx.advisory_match:
            return "advisory_notified"
        if self.ctx.case_id:
            return "case_created"
        if self._abandoned_has_projectable_slots():
            return "abandoned_with_slots"
        return "incomplete"

    def _abandoned_has_projectable_slots(self) -> bool:
        slots = self.ctx.slots or {}
        if not (slots.get("category") or "").strip():
            return False
        return bool(
            (slots.get("entity_1") or "").strip()
            or (slots.get("entity_2") or "").strip()
            or (slots.get("description") or "").strip()
        )

    def _handoff_was_claimed(self) -> bool:
        try:
            from src.data.warehouse import ops_con as _ops_con

            with _ops_con(read_only=True) as _con:
                row = _con.execute(
                    "SELECT 1 FROM handoff_requests WHERE interaction_id = ?"
                    " AND status IN ('claimed', 'done')",
                    [self.ctx.interaction_id],
                ).fetchone()
                return bool(row)
        except Exception:
            return False

    async def _finalize_interaction(self, outcome: str) -> None:
        from src.data.timeutil import utc_now
        from src.data.warehouse import ops_con, ops_in_thread
        from src.frontline.dsr import is_interaction_erased

        erased = getattr(self, "_erased", False) or is_interaction_erased(self.ctx.interaction_id)
        if erased:
            self._erased = True
            self.ctx.slots.clear()

        # Determine final status from outcome: 'incomplete' → 'abandoned',
        # otherwise 'completed'. 'escalated_safety' stays 'escalated' for the
        # console's red-flag view.
        if outcome in ("incomplete", "abandoned_with_slots"):
            final_status = "abandoned"
        elif outcome == "escalated_safety":
            final_status = "escalated"
        else:
            final_status = "completed"

        if erased:
            def _write_erased() -> None:
                with ops_con() as con:
                    try:
                        con.execute(
                            """
                            UPDATE interactions
                            SET ended_at = ?, status = ?, outcome = ?,
                                peak_frustration = ?, supervised = ?, llm_calls = ?,
                                enrichment_partial = ?
                            WHERE interaction_id = ? AND NOT COALESCE(erased, FALSE)
                            """,
                            [
                                utc_now(),
                                final_status,
                                outcome,
                                self.ctx.peak_frustration,
                                self.ctx.supervised,
                                self.ctx.llm_calls,
                                bool(self.ctx.enrichment_partial),
                                self.ctx.interaction_id,
                            ],
                        )
                    except Exception:
                        con.execute(
                            """
                            UPDATE interactions
                            SET ended_at = ?, status = ?, outcome = ?,
                                peak_frustration = ?, supervised = ?, llm_calls = ?
                            WHERE interaction_id = ? AND NOT COALESCE(erased, FALSE)
                            """,
                            [
                                utc_now(),
                                final_status,
                                outcome,
                                self.ctx.peak_frustration,
                                self.ctx.supervised,
                                self.ctx.llm_calls,
                                self.ctx.interaction_id,
                            ],
                        )

            await ops_in_thread(_write_erased)
            return

        payload = [
            utc_now(),
            final_status,
            outcome,
            self.ctx.peak_frustration,
            self.ctx.supervised,
            self.ctx.llm_calls,
            self.ctx.slots.get("entity_1"),
            self.ctx.slots.get("entity_2"),
            self.ctx.slots.get("entity_3"),
            self.ctx.slots.get("category"),
            (self.ctx.slots.get("description") or "")[:500],
            bool(self.ctx.enrichment_partial),
            self.ctx.interaction_id,
        ]

        def _write() -> None:
            with ops_con() as con:
                try:
                    con.execute(
                        """
                        UPDATE interactions
                        SET ended_at = ?, status = ?, outcome = ?,
                            peak_frustration = ?, supervised = ?, llm_calls = ?,
                            entity_1 = ?, entity_2 = ?, entity_3 = ?,
                            category = ?, description = ?,
                            enrichment_partial = ?
                        WHERE interaction_id = ? AND NOT COALESCE(erased, FALSE)
                        """,
                        payload,
                    )
                except Exception:
                    con.execute(
                        """
                        UPDATE interactions
                        SET ended_at = ?, status = ?, outcome = ?,
                            peak_frustration = ?, supervised = ?, llm_calls = ?,
                            entity_1 = ?, entity_2 = ?, entity_3 = ?,
                            category = ?, description = ?
                        WHERE interaction_id = ? AND NOT COALESCE(erased, FALSE)
                        """,
                        payload[:-2] + payload[-1:],
                    )

        await ops_in_thread(_write)
        # Cross-contact entity memory (enterprise) — never raise into contact path.
        try:
            from src.enterprise.memory import upsert_memory_from_interaction

            upsert_memory_from_interaction(self.ctx.interaction_id)
        except Exception:
            pass
        # v3 governance stamp + continuous learning proposal (best-effort).
        try:
            from src.v3.governance import stamp_and_maybe_learn

            stamp_and_maybe_learn(self.ctx.interaction_id)
        except Exception:
            pass
        # Persist any turns not yet flushed mid-contact (idempotent by turn_id).
        try:
            from src.data.turns import persist_turn

            for t in self.ctx.turns:
                persist_turn(self.ctx.interaction_id, t)
        except Exception:
            pass

        # ── Step 5.6: project the contact into the domain warehouse ────────
        # The loop only compounds if closed contacts reach the table the
        # fleet scan reads. Best-effort, never raises into finalize.
        if outcome in ("case_created", "advisory_notified", "escalated_safety"):
            try:
                self._project_contact_to_corpus(outcome)
            except Exception:
                pass
        elif outcome in ("incomplete", "abandoned_with_slots") and self._abandoned_has_projectable_slots():
            # Abandoned WITH slots: keep the signal as inferred. Anomaly
            # scoring excludes provenance='inferred' until verified.
            try:
                self._project_contact_to_corpus(outcome, provenance="inferred")
            except Exception:
                pass

        # ── Enqueue Qubot post_contact_audit (non-blocking) ────────────────
        # Never await the auditor on the contact/WS hot path (H4 / audit3).
        # Failures are ledgered; retry via `make audit ID=...` if needed.
        try:
            from src.qubot.auditor import audit_interaction

            iid = self.ctx.interaction_id

            async def _run_audit() -> None:
                try:
                    await audit_interaction(iid, write_report=True)
                except Exception as e:
                    try:
                        record_action(self._orchestrator_action(
                            "state_transition",
                            input_summary="post_contact_audit",
                            output_summary=f"audit failed: {type(e).__name__}: {e}",
                        ))
                    except Exception:
                        pass

            try:
                loop = asyncio.get_running_loop()
                # Fire-and-forget: hangup/finalize returns without waiting.
                task = loop.create_task(_run_audit())
                # Prevent "task exception never retrieved" noise if suite exits early.
                task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            except RuntimeError:
                # No running loop (CLI/sync): skip; callers can run audit explicitly.
                pass
        except Exception as e:
            record_action(self._orchestrator_action(
                "state_transition",
                input_summary="post_contact_audit",
                output_summary=f"audit schedule failed: {type(e).__name__}: {e}",
            ))

        # Feature #51: release drain registry slot so deploy can exit when idle.
        try:
            from src.ops.drain import DRAIN

            DRAIN.unregister(self.ctx.interaction_id)
        except Exception:
            pass

    def _attach_returning_case(self) -> str | None:
        """Find an open case for a returning customer (audit 1.1).

        Matches sha256 customer_ref + same entity_2/category on a case still
        being worked (``open`` or ``pending_followup`` — a Qubot
        needs_review flag must not hide the case from the returning customer)
        updated within FRONTLINE_RETURNING_CUSTOMER_DAYS (default 14).
        Returns the case_id to attach to, or None. The preset case_id flows
        into CaseAgent.run()'s idempotent re-close path, and the attach is
        ledgered as case_attached_existing.
        """
        import os as _os

        ref = (self.ctx.customer_ref or "").strip()
        if not ref or self.ctx.case_id:
            return None
        try:
            window_days = int(_os.getenv("FRONTLINE_RETURNING_CUSTOMER_DAYS", "14"))
        except ValueError:
            window_days = 14
        entity_2 = (self.ctx.slots.get("entity_2") or "").strip()
        category = (self.ctx.slots.get("category") or "").strip()
        if not entity_2 and not category:
            return None
        try:
            from src.data.timeutil import utc_now
            from src.data.warehouse import ops_con

            cutoff = utc_now() - __import__("datetime").timedelta(
                days=max(1, window_days)
            )
            with ops_con(read_only=True) as con:
                row = con.execute(
                    """
                    SELECT case_id FROM cases c
                    JOIN interactions i ON i.interaction_id = c.interaction_id
                    WHERE c.customer_ref = ?
                      AND c.status IN ('open', 'pending_followup')
                      AND c.created_at >= ?
                      AND (i.entity_2 = ? OR ? = '')
                      AND (i.category = ? OR ? = '')
                    ORDER BY c.created_at DESC LIMIT 1
                    """,
                    [ref, cutoff, entity_2, entity_2, category, category],
                ).fetchone()
        except Exception:
            return None
        if not row:
            return None
        try:
            record_action(self._orchestrator_action(
                "case_attached_existing",
                input_summary=f"returning customer_ref match (window {window_days}d)",
                output_summary=f"attached to {row[0]} instead of opening a new case",
            ))
        except Exception:
            pass
        return str(row[0])

    def _project_contact_to_corpus(self, outcome: str, *, provenance: str = "observed") -> str | None:
        """Write this contact as a domain record (source='FRONTLINE').

        Returns the record_id, or None when there is nothing projectable
        (no category). record_id is namespaced per source so a contact can
        never collide with an ingested row. Evidence closes project as
        ``observed``; abandoned-with-slots contacts project as ``inferred``
        (board: lost signal without polluting observed statistics).
        """
        from src.data.timeutil import utc_now
        from src.data.warehouse import apply_domain_schema, domain_con

        slots = self.ctx.slots or {}
        category = (slots.get("category") or "").strip()
        if not category:
            return None
        e1 = (slots.get("entity_1") or "").strip().upper()
        e2 = (slots.get("entity_2") or "").strip().upper()
        e3 = (slots.get("entity_3") or "").strip().upper()
        entity_key = "|".join([p for p in (e1, e2, e3) if p])
        now = utc_now().replace(tzinfo=None)
        rid = f"FRONTLINE-{self.ctx.interaction_id}"
        with domain_con(self.ctx.pack.id, read_only=False) as con:
            apply_domain_schema(con)
            # 1.1: embed at close so fleet scan sees vectors immediately
            _text = (slots.get("description") or "")[:1000]
            try:
                from src.ml_runtime.embeddings import embed_text as _embed
                _emb = _embed(_text)
            except Exception:
                _emb = None
            sec_raw = slots.get("secondary_categories")
            sec_label = ", ".join(sec_raw) if isinstance(sec_raw, list) else (str(sec_raw) if sec_raw else None)
            con.execute(
                """
                INSERT OR REPLACE INTO records (
                    record_id, occurred_at, received_at, entity_1, entity_2,
                    entity_3, category, subcategory, text, severity_label,
                    region, source, entity_key, provenance, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'FRONTLINE', ?, ?, ?)
                """,
                [
                    rid,
                    now,
                    now,
                    slots.get("entity_1"),
                    slots.get("entity_2"),
                    slots.get("entity_3"),
                    category,
                    sec_label,
                    _text,
                    self.ctx.severity or None,
                    entity_key or None,
                    provenance,
                    _emb,
                ],
            )
        # Sidecar semantic vector (shadow/semantic). Never replaces records.embedding.
        # Failure is pending/failed + retry job; never delays safety or case close.
        try:
            from src.ml_runtime.embedding_runtime import embedding_mode, try_semantic_embedder
            from src.ml_runtime.embedding_store import mark_embedding_status, upsert_embedding

            _mode = embedding_mode()
            if _mode in {"shadow", "semantic"}:
                _sem = try_semantic_embedder()
                if _sem is None:
                    try:
                        from src.observability.degradation import step_down

                        step_down("semantic_search", reason="embedder_unavailable")
                    except Exception:
                        pass
                    mark_embedding_status(
                        self.ctx.pack.id,
                        rid,
                        "unloaded",
                        status="pending",
                        error_code="unavailable",
                        error_message="semantic embedder not ready",
                    )
                else:
                    try:
                        _svec = _sem.embed(_text)
                        _sha = ""
                        _meta = _sem.artifact_metadata()
                        if _meta:
                            _sha = _meta.model_sha256 or ""
                        upsert_embedding(
                            self.ctx.pack.id,
                            rid,
                            _svec,
                            artifact_sha256=_sha,
                            status="complete",
                        )
                        _status = "complete"
                    except Exception as _emb_err:
                        mark_embedding_status(
                            self.ctx.pack.id,
                            rid,
                            _sem.version,
                            status="failed",
                            error_code="inference",
                            error_message=type(_emb_err).__name__,
                            native_dimension=_sem.native_dimension,
                            output_dimension=_sem.output_dimension,
                        )
                        _status = "failed"
                        try:
                            from src.jobs.queue import enqueue as _enq_emb

                            _enq_emb(
                                "embedding_backfill",
                                {
                                    "pack_id": self.ctx.pack.id,
                                    "embedding_version": _sem.version,
                                    "limit": 8,
                                },
                                idempotency_key=f"emb-retry-{rid}",
                            )
                        except Exception:
                            pass
                    try:
                        from src.ledger import record_action as _ra_emb

                        _ra_emb(
                            self._orchestrator_action(
                                action_type="embedding_recorded",
                                input_summary=f"record_id={rid} mode={_mode}",
                                output_summary=f"status={_status} version={_sem.version}",
                            )
                        )
                    except Exception:
                        pass
        except Exception:
            pass
        # 1.1: gazetteer candidate queue (human approve, never auto-mutate)
        try:
            from src.data.warehouse import ops_con as _ops
            with _ops() as _c:
                _c.execute(
                    "CREATE TABLE IF NOT EXISTS gazetteer_candidates "
                    "(term VARCHAR, slot VARCHAR, pack_id VARCHAR, "
                    "interaction_id VARCHAR, created_at TIMESTAMP DEFAULT current_timestamp)"
                )
                for _slot in ("entity_2", "entity_3", "category"):
                    _v = (slots.get(_slot) or "").strip()
                    if _v:
                        _c.execute(
                            "INSERT INTO gazetteer_candidates (term, slot, pack_id, interaction_id) "
                            "VALUES (?, ?, ?, ?)",
                            [_v, _slot, self.ctx.pack.id, self.ctx.interaction_id],
                        )
        except Exception:
            pass
        # 1.1: invalidate anomaly buckets async + immediate slice recompute (audit 1.1)
        try:
            from src.jobs.queue import enqueue as _enq
            _enq("recompute_anomalies", {
                "pack_id": self.ctx.pack.id,
                "category": category,
                "entity_2": slots.get("entity_2"),
            })
        except Exception:
            pass
        try:
            from src.ml_runtime.anomalies import recompute_weekly_anomalies
            recompute_weekly_anomalies(
                self.ctx.pack.id,
                category=category,
                entity_2=slots.get("entity_2"),
            )
        except Exception:
            pass
        try:
            record_action(self._orchestrator_action(
                "state_transition",
                input_summary="contact projected to domain corpus",
                output_summary=f"record_id={rid} source=FRONTLINE provenance={provenance}",
            ))
        except Exception:
            pass
        # Fix-regression watch (board #6): a new contact matching a recently
        # fixed slice (same category + entity inside 90 days) re-opens the
        # question — alert instead of silently counting it as background.
        try:
            self._watch_fix_regression(category, slots, rid)
        except Exception:
            pass
        return rid

    def _watch_fix_regression(
        self, category: str, slots: dict[str, Any], record_id: str
    ) -> None:
        from datetime import timedelta as _td

        from src.data.timeutil import utc_now
        from src.data.warehouse import ops_con

        e2 = (slots.get("entity_2") or "").strip()
        e3 = (slots.get("entity_3") or "").strip()
        if not category or (not e2 and not e3):
            return
        with ops_con(read_only=True) as con:
            try:
                rows = con.execute(
                    """
                    SELECT fix_id, investigation_id, fixed_at FROM recorded_fixes
                    WHERE pack_id = ? AND category = ?
                      AND (entity_2 = ? OR entity_3 = ?)
                      AND fixed_at >= ?
                    """,
                    [
                        self.ctx.pack.id, category, e2, e3,
                        utc_now() - _td(days=90),
                    ],
                ).fetchall()
            except Exception:
                return
        if not rows:
            return
        fix_id = str(rows[0][0])
        try:
            record_action(self._orchestrator_action(
                "state_transition",
                input_summary=f"contact matches fixed slice (fix {fix_id})",
                output_summary=f"possible fix regression on {category}; record {record_id}",
            ))
        except Exception:
            pass
        try:
            from src.frontline.alerts import fire_alert

            import anyio as _anyio

            async def _fire() -> None:
                try:
                    await fire_alert(
                        event="fix_regression",
                        summary=(
                            f"[P1 CRITICAL] fix regression detected: new {category} contact "
                            f"matches fix {fix_id} (<90d)"
                        ),
                        ref_id=fix_id,
                        pack_id=self.ctx.pack.id,
                        extra={
                            "fix_id": fix_id,
                            "record_id": record_id,
                            "priority": "P1",
                            "severity": "Critical",
                            "regression": True,
                        },
                    )
                except Exception:
                    pass

            try:
                _loop = asyncio.get_running_loop()
                _loop.create_task(_fire())
            except RuntimeError:
                _anyio.run(_fire)
        except Exception:
            pass

    async def _emit_spoken(
        self,
        text: str,
        *,
        action_type: str,
        input_summary: str = "",
        speaker: str = "agent",
        ledger: bool = True,
        action_id: str = "",
        turn: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Ledger (unless already done), persist, then emit one spoken turn.

        AI speech is suppressed once the contact is supervised. Supervisor
        speech is not. Ledger or transcript failure skips the emit.
        """
        if speaker != "supervisor" and (self.ctx.state == SUPERVISED or self.ctx.supervised):
            return None
        aid = action_id or ""
        if ledger:
            aid = record_action(self._orchestrator_action(
                action_type, input_summary, (text or "")[:500],
            ))
        if turn is None:
            turn = self.ctx.record_turn(speaker, text, llm_used=False, action_id=aid or None)
        meta = {
            "speaker": speaker,
            "fast_path": True,
            "llm_used": False,
            "action_id": aid,
            "turn_id": (turn or {}).get("turn_id"),
            "utterance_id": (turn or {}).get("turn_id"),
            **(extra or {}),
        }
        await self.hooks._maybe(self.hooks.emit_customer_turn, text, meta)
        return meta

    async def _emit_ledgered_remedy(self, offer: dict[str, Any]) -> None:
        """Commit a remedy action and the transcript, then speak. No silent bypass."""
        from src.ledger import AgentAction

        spoken = str(offer.get("customer_text") or "")
        if not spoken:
            return
        evidence = [str(offer["advisory_id"])] if offer.get("advisory_id") else []
        action_id = record_action(AgentAction(
            interaction_id=self.ctx.interaction_id,
            agent="orchestrator",
            action_type="remedy_offered",
            input_summary=f"remedy_offer advisory={offer.get('advisory_id')}",
            output_summary=spoken[:500],
            evidence_ids=evidence,
            case_id=self.ctx.case_id,
        ))
        turn = self.ctx.record_turn("agent", spoken, llm_used=False, action_id=action_id)
        await self.hooks._maybe(
            self.hooks.emit_customer_turn,
            spoken,
            {
                "speaker": "agent",
                "fast_path": True,
                "llm_used": False,
                "action_id": action_id,
                "turn_id": turn.get("turn_id"),
                "utterance_id": turn.get("turn_id"),
            },
        )

    async def _publish_control(self) -> None:
        payload = {
            "type": "control_state",
            "interaction_id": self.ctx.interaction_id,
            "state": self.ctx.state,
            "supervised": bool(self.ctx.supervised),
            "generation": int(self.ctx.control_generation or 0),
            "claimed_by": self.ctx.takeover_claimed_by,
        }
        await self.hooks._maybe(getattr(self.hooks, "emit_control_state", None), payload)

    def _orchestrator_action(self, action_type: str, input_summary: str = "", output_summary: str = ""):
        from src.ledger import AgentAction
        return AgentAction(
            interaction_id=self.ctx.interaction_id,
            agent="orchestrator",
            action_type=action_type,
            input_summary=input_summary,
            output_summary=output_summary,
            case_id=self.ctx.case_id,
        )


# ── Factory ───────────────────────────────────────────────────────────────


def new_interaction_id() -> str:
    return "int_" + new_ulid()


async def create_interaction(
    channel: str = "web_text",
    pack_id: str | None = None,
    hooks: OrchestratorHooks | None = None,
    customer_ref: str | None = None,
    region: str | None = None,
    locale: str | None = None,
) -> tuple[Orchestrator, str]:
    """Create an interaction row + orchestrator. Returns (orchestrator, greeting).

    Feature #51: refuses new contacts while the process is draining (SIGTERM /
    blue-green). Raises ``ServiceDrainingError`` so API callers return 503.

    ``customer_ref`` is an already-hashed identity (sha256 hex of
    phone/email/account — never raw PII) used for returning-customer case
    attach (audit 1.1).
    """
    from src.data.warehouse import ops_con, ops_in_thread
    from src.domains.active_pack import resolve_active_pack_id
    from src.ops.drain import DRAIN, ServiceDrainingError

    pack = load_pack(pack_id or resolve_active_pack_id())
    interaction_id = new_interaction_id()

    # Register before DB insert so drain rejects never leave orphan rows.
    if not DRAIN.register_active(interaction_id):
        raise ServiceDrainingError()

    from src.data.timeutil import utc_now

    try:
        insert_row = [
            interaction_id,
            pack.id,
            pack.pack_version,
            utc_now(),
            channel,
            customer_ref,
        ]

        def _insert() -> None:
            with ops_con() as con:
                try:
                    con.execute(
                        """
                        INSERT INTO interactions
                        (interaction_id, pack_id, pack_version, started_at, channel,
                         status, supervised, llm_calls, customer_ref)
                        VALUES (?, ?, ?, ?, ?, 'active', FALSE, 0, ?)
                        """,
                        insert_row,
                    )
                except Exception:
                    # Pre-customer_ref schema: column added by later migrate.
                    con.execute(
                        """
                        INSERT INTO interactions
                        (interaction_id, pack_id, pack_version, started_at, channel,
                         status, supervised, llm_calls)
                        VALUES (?, ?, ?, ?, ?, 'active', FALSE, 0)
                        """,
                        insert_row[:5],
                    )

        await ops_in_thread(_insert)
        record_action(_build_started_action(interaction_id, pack.id))

        # Initialize subject DEK for crypto-shredding (GDPR/CCPA Art. 17 compliance, audit F-008)
        try:
            from src.security.pii import SubjectKeyStore

            SubjectKeyStore.get_or_create_dek(interaction_id)
        except Exception:
            pass

        orch = Orchestrator(interaction_id, pack, channel=channel, hooks=hooks)
        orch.ctx.customer_ref = customer_ref
        greeting = await orch.start(region=region, locale=locale)
        return orch, greeting
    except Exception:
        DRAIN.unregister(interaction_id)
        raise


def _build_started_action(interaction_id: str, pack_id: str):
    from src.ledger import AgentAction
    return AgentAction(
        interaction_id=interaction_id,
        agent="orchestrator",
        action_type="interaction_started",
        input_summary=f"pack_id={pack_id}",
        output_summary="interaction created",
    )
