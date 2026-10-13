/**
 * Pure helpers for the Voice agent call surface.
 * Kept free of DOM/WebSocket so unit tests can drive the real shipped logic.
 */

export const CALL_STATE = {
  IDLE: "idle",
  CONNECTING: "connecting",
  LISTENING: "listening",
  THINKING: "thinking",
  AGENT_SPEAKING: "agent-speaking",
  ENDED: "ended",
};

export const STATE_LABELS = {
  [CALL_STATE.IDLE]: "Ready to start",
  [CALL_STATE.CONNECTING]: "Connecting…",
  [CALL_STATE.LISTENING]: "Listening to you",
  [CALL_STATE.THINKING]: "Thinking…",
  [CALL_STATE.AGENT_SPEAKING]: "Agent speaking",
  [CALL_STATE.ENDED]: "Call ended",
};

/**
 * Human label for call machine state.
 * Greeting TTS uses AGENT_SPEAKING with phase "greeting" — distinct copy so
 * operators know barge-in is off until the pack greeting finishes.
 */
export function callStateLabel(state, opts = {}) {
  if (opts.textOnly && (state === CALL_STATE.LISTENING || state === CALL_STATE.CONNECTING)) {
    return state === CALL_STATE.CONNECTING ? "Connecting…" : "Text only";
  }
  if (state === CALL_STATE.AGENT_SPEAKING && opts.speakPhase === "greeting") {
    return "Greeting you";
  }
  return STATE_LABELS[state] || String(state || "unknown");
}

/** Short status line under the hero label (barge-in honesty). */
export function callPhaseHint(state, opts = {}) {
  const phase = opts.speakPhase || "normal";
  if (opts.textOnly && state === CALL_STATE.LISTENING) {
    return "Microphone unavailable · type your reply below";
  }
  if (state === CALL_STATE.AGENT_SPEAKING && phase === "greeting") {
    return "Barge-in off · full greeting plays through";
  }
  if (state === CALL_STATE.AGENT_SPEAKING) {
    return "Barge-in after ~1.8s · speak to interrupt";
  }
  if (state === CALL_STATE.LISTENING) {
    return "Your turn · speak or type below";
  }
  if (state === CALL_STATE.THINKING) {
    return "Working on your answer… mic paused";
  }
  if (state === CALL_STATE.CONNECTING) {
    return "Opening contact · greeting next";
  }
  return "";
}

/**
 * Merge a turn into the transcript list.
 * Interim replaces trailing interim; final drops trailing interim then appends.
 * @param {Array<{speaker:string,text:string,interim?:boolean,id?:string}>} turns
 * @param {{speaker:string,text:string,interim?:boolean,id?:string}} entry
 */
export function mergeTranscriptTurn(turns, entry) {
  const list = Array.isArray(turns) ? turns : [];
  if (!entry || !entry.text) return list;
  if (entry.interim && list.length && list[list.length - 1].interim) {
    return [...list.slice(0, -1), entry];
  }
  if (entry.interim) return [...list, entry];
  const cleaned = list.length && list[list.length - 1].interim ? list.slice(0, -1) : list;
  return [...cleaned, { ...entry, interim: false }];
}

/**
 * Map a server WS frame for interaction_ended into a summary object.
 * Server sends flat fields on the message (not nested under payload).
 * @param {Record<string, unknown>} msg
 */
export function mapInteractionEnded(msg) {
  if (!msg || typeof msg !== "object") {
    return { reason: "ended" };
  }
  const nested = msg.payload && typeof msg.payload === "object" ? msg.payload : {};
  return {
    reason: nested.reason || msg.reason || "ended",
    case_id: nested.case_id ?? msg.case_id ?? null,
    investigation_id: nested.investigation_id ?? msg.investigation_id ?? null,
    investigation_opened: Boolean(
      nested.investigation_opened ?? msg.investigation_opened ?? false
    ),
    audit_pending: Boolean(nested.audit_pending ?? msg.audit_pending ?? true),
  };
}

/**
 * Should SpeechRecognition process a result right now?
 * Ignore customer STT while agent TTS is playing (unless barge-in already stopped TTS).
 */
const ANSWER_MARKERS = new Set([
  "yes", "yeah", "yep", "yup", "yea", "no", "nope", "nah", "not", "nobody", "none",
]);

const EMERGENCY_UTTERANCE = /\b(fire|smoke|crash|hurt|hurts|bleeding|injured|injury|help|emergency|ambulance|trapped|cut)\b/;

function _hasCallerAnswerMarker(heardNorm, agentNorm) {
  const agentWords = new Set(String(agentNorm || "").split(" ").filter(Boolean));
  return String(heardNorm || "")
    .split(" ")
    .some((word) => ANSWER_MARKERS.has(word) && !agentWords.has(word));
}

/**
 * Keep the words that interrupted playback.
 * A short "No!" or an emergency phrase survives the post-TTS cooldown.
 * Ordinary bleed during that cooldown is still dropped.
 */
export function shouldRetainBargeInUtterance(text, opts = {}) {
  const raw = String(text || "").trim();
  if (!raw) return "";
  const norm = _normUtterance(raw);
  const shortNo = /^(no|nope|nah|stop|wait)$/.test(norm);
  const emergency = EMERGENCY_UTTERANCE.test(norm);
  const interrupting = Boolean(opts.interrupting);
  // The utterance that stopped playback is the caller's, including "No!".
  if (interrupting) return raw;
  const ts = opts.now || Date.now();
  const ready = opts.listenReadyAt || 0;
  const inCooldown = Boolean(ready) && ts < ready;
  if (inCooldown && (shortNo || emergency)) return raw;
  return "";
}

export function shouldAcceptSpeechResult({
  speaking,
  wsOpen,
  listenReadyAt = 0,
  now = 0,
  text = "",
  interrupting = false,
}) {
  if (!wsOpen) return false;
  if (speaking && !interrupting) return false;
  const ts = now || Date.now();
  const inCooldown = Boolean(listenReadyAt) && ts < listenReadyAt;
  if (inCooldown && !interrupting) {
    if (text && shouldRetainBargeInUtterance(text, { now: ts, listenReadyAt })) return true;
    return false;
  }
  if (interrupting && String(text || "").trim()) {
    const kept = shouldRetainBargeInUtterance(text, {
      interrupting: true,
      now: ts,
      listenReadyAt,
    });
    return Boolean(kept);
  }
  if (inCooldown) return false;
  return true;
}

/** Settle after TTS so the mic does not hear the agent's last words as the customer. */
export const POST_TTS_COOLDOWN_MS = 650;

/** Join Chrome's split final STT chunks ("my car has" then "been stuck") into one turn. */
export const STT_FINAL_DEBOUNCE_MS = 1400;

/** Chrome throws if recognition.start() runs in the same tick as stop()/onend. */
export const RECOG_RESTART_MS = 280;

function _normUtterance(s) {
  return String(s || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

/**
 * True when heard STT is the agent talking (speaker bleed / TTS echo).
 * Prevents "is anyone hurt?" being captured as a customer "yes/hurt" answer.
 */
export function looksLikeAgentEcho(heard, agentText) {
  const a = _normUtterance(heard);
  const b = _normUtterance(agentText);
  if (!a || !b || a.length < 4) return false;
  if (a === b) return true;
  // Verbatim playback of the agent line, or a long slice of it.
  if (b.includes(a) && a.length >= 6) return true;
  if (a.includes(b) && b.length >= 10) return true;
  // A real answer that adds yes/no/not the agent did not say is not echo,
  // even when it repeats the question's own words.
  if (_hasCallerAnswerMarker(a, b)) return false;
  const ta = a.split(" ").filter((w) => w.length > 3);
  const tb = new Set(b.split(" ").filter((w) => w.length > 3));
  if (!ta.length || !tb.size) return false;
  let hit = 0;
  for (const w of ta) if (tb.has(w)) hit += 1;
  return hit / ta.length >= 0.65;
}

/**
 * Fold consecutive SpeechRecognition finals into one utterance.
 * Returns { buffer, flush } — flush is the joined text when debounce elapsed.
 */
export function coalesceSpeechFinals(buffer, nextText, now, debounceMs = STT_FINAL_DEBOUNCE_MS) {
  const piece = String(nextText || "").trim();
  const ts = now || Date.now();
  const wait = debounceMs == null ? STT_FINAL_DEBOUNCE_MS : debounceMs;
  if (!piece) {
    return { buffer: buffer || { text: "", at: 0 }, flush: null };
  }
  const prev = buffer && buffer.text ? buffer : { text: "", at: 0 };
  const joined = prev.text ? `${prev.text} ${piece}` : piece;
  return { buffer: { text: joined, at: ts }, flush: null, wait };
}

/** Default settle before barge-in can cancel TTS (ms). */
export const BARGE_IN_GRACE_MS = 1800;

/**
 * Whether barge-in may cancel the current agent utterance.
 * - Disabled entirely for the initial pack greeting (phase === 'greeting').
 * - Disabled until graceMs after TTS start (speaker bleed is not barge-in).
 * - Requires sustained high RMS samples (not a single tick).
 *
 * @param {{
 *   phase?: string,
 *   elapsedMs: number,
 *   graceMs?: number,
 *   rms: number,
 *   rmsThreshold?: number,
 *   highRmsStreak?: number,
 *   streakNeeded?: number,
 * }} opts
 */
export function shouldAllowBargeIn(opts) {
  const {
    phase = "normal",
    elapsedMs = 0,
    graceMs = BARGE_IN_GRACE_MS,
    rms = 0,
    rmsThreshold = 0.14,
    highRmsStreak = 0,
    streakNeeded = 4,
  } = opts || {};
  if (phase === "greeting") return false;
  if (elapsedMs < graceMs) return false;
  if (rms < rmsThreshold) return false;
  // highRmsStreak is the count of consecutive high-RMS frames already observed.
  if ((highRmsStreak || 0) < streakNeeded) return false;
  return true;
}

/** Preserve greeting text for transcript + TTS (no truncation). */
export function normalizeGreetingText(text) {
  if (text == null) return "";
  return String(text).trim();
}

/**
 * Decide what string to pass to speechSynthesis for the start greeting.
 * Always the full start response field — never a partial interim.
 */
export function greetingSpeakText(greetingText) {
  return normalizeGreetingText(greetingText);
}

/**
 * Build slot display rows from pack entity_labels + live slots map.
 */
export function buildSlotEntries(pack, slots) {
  const s = slots && typeof slots === "object" ? slots : {};
  const labels = pack?.entity_labels;
  if (labels && !Array.isArray(labels) && typeof labels === "object") {
    const rows = Object.entries(labels).map(([key, label]) => ({
      key,
      label: label || key,
      value: s[key] ?? "",
    }));
    for (const [k, v] of Object.entries(s)) {
      if (k.startsWith("__")) continue;
      if (!rows.some((r) => r.key === k)) {
        rows.push({ key: k, label: k, value: v });
      }
    }
    return rows;
  }
  if (Array.isArray(labels) && labels.length) {
    return labels.map((label, i) => {
      const key = `entity_${i + 1}`;
      return {
        key,
        label,
        value: s[key] ?? s[label] ?? "",
      };
    });
  }
  return Object.entries(s)
    .filter(([k]) => !k.startsWith("__"))
    .map(([k, v]) => ({ key: k, label: k, value: v }));
}

/** Filled / total for progress meter. */
export function slotProgress(entries) {
  const list = Array.isArray(entries) ? entries : [];
  const total = list.length;
  const filled = list.filter((e) => String(e.value || "").trim()).length;
  return { filled, total, pct: total ? Math.round((filled / total) * 100) : 0 };
}

/**
 * Capability honesty flags for the side panel.
 */
export function capabilitySnapshot({ hasSpeechRecognition, hasSpeechSynthesis, hasMicStream }) {
  return {
    stt: hasSpeechRecognition ? "Web Speech" : "text only",
    sttOk: !!hasSpeechRecognition,
    tts: hasSpeechSynthesis ? "Browser TTS" : "unavailable",
    ttsOk: !!hasSpeechSynthesis,
    bargeIn: hasMicStream ? "armed" : hasSpeechRecognition ? "needs mic permission" : "n/a",
    bargeOk: !!hasMicStream,
    textFallback: true,
  };
}

/**
 * After a normal interaction_ended (or user hangup), WS onclose must not reconnect.
 * @param {{ intentionalClose: boolean, callEnded: boolean }} flags
 */
export function shouldReconnectOnClose({ intentionalClose, callEnded }) {
  if (intentionalClose) return false;
  if (callEnded) return false;
  return true;
}

/**
 * On WS open after reconnect: only resume listening if the call is still live.
 */
export function shouldResumeListeningOnOpen({ intentionalClose, callEnded, isReconnect }) {
  if (intentionalClose || callEnded) return false;
  // Fresh connect may start with greeting path instead; reconnect always resumes listen.
  return !!isReconnect;
}

/**
 * Is a server `error` frame fatal for this call?
 *
 * The server tags unrecoverable errors with `recoverable: false` (the contact is
 * gone, already ended, or held by another socket). Retrying those just burns the
 * reconnect budget and shows the raw 404 detail to the operator, so the widget
 * ends the call cleanly instead. Older servers only send `detail` — fall back to
 * matching the known unrecoverable details.
 *
 * @param {{ recoverable?: boolean, code?: string, detail?: string }} msg
 */
export function isFatalWsError(msg) {
  if (!msg || typeof msg !== "object") return false;
  if (msg.recoverable === false) return true;
  if (msg.recoverable === true) return false;
  const detail = String(msg.detail || msg.message || "").toLowerCase();
  return (
    detail.includes("active interaction not found") ||
    detail.includes("interaction already ended") ||
    detail.includes("already has an active customer websocket") ||
    detail.startsWith("server error")
  );
}

/** Operator-facing copy for a fatal WS error (never the raw interaction id). */
export function fatalWsErrorMessage(msg) {
  const code = msg && msg.code;
  if (code === "interaction_busy") {
    return "This contact is already open in another tab or window.";
  }
  if (code === "server_error") {
    return "The contact hit a server error and was closed. Start a new call.";
  }
  return "This contact could not be resumed — it was closed while you were disconnected.";
}

/**
 * Rebuild the transcript from the server's authoritative turn list on resume.
 * The client may have missed turns while its socket was down, so the server
 * list replaces the local one rather than being appended to it.
 *
 * @param {Array<{id?:string,turn_id?:string,speaker?:string,text?:string}>} turns
 */
export function transcriptFromResume(turns) {
  if (!Array.isArray(turns)) return null;
  const rows = turns
    .filter((t) => t && String(t.text || "").trim())
    .map((t, i) => ({
      id: t.id || t.turn_id || `r${i + 1}`,
      speaker: t.speaker || "agent",
      text: String(t.text),
      interim: false,
    }));
  return rows.length ? rows : null;
}

/** Quiet listen this long with no speech before the agent asks again. */
export const IDLE_LISTEN_REPROMPT_MS = 8000;

/**
 * End-of-turn follows speech activity, not only a final-result timer.
 * An interim that is still inside the debounce must not flush.
 * A listen with no speech can ask the caller to continue.
 */
export function decideSpeechFinal({
  lastSpeechAt = 0,
  lastInterimAt = 0,
  committedAt = 0,
  now = 0,
  debounceMs = 1400,
  idleRepromptMs = IDLE_LISTEN_REPROMPT_MS,
  listeningSince = 0,
} = {}) {
  const interimAt = Number(lastInterimAt) || 0;
  const committed = Number(committedAt) || 0;
  const latest = Math.max(interimAt, committed, Number(lastSpeechAt) || 0);
  const interimOpen = interimAt > 0 && interimAt >= committed && (now - interimAt) < debounceMs;
  if (interimOpen) {
    return { flush: false, reprompt: false, reason: "interim" };
  }
  if (latest > 0 && (now - latest) >= debounceMs) {
    return { flush: true, reprompt: false, reason: "committed" };
  }
  const idle = latest === 0 && Number(listeningSince) > 0 && (now - Number(listeningSince)) >= idleRepromptMs;
  if (idle) {
    return { flush: false, reprompt: true, reason: "idle" };
  }
  return { flush: false, reprompt: false, reason: "waiting" };
}

/**
 * Browser turn queue. A send stays pending until the server ack.
 * A retry keeps the same turn id. A duplicate after ack returns that outcome.
 */
export function createTurnQueue() {
  const turns = new Map();
  return {
    submit(turnId, text) {
      const id = String(turnId || "").trim();
      if (!id) {
        return { status: "rejected", accepted: false, reason: "missing_turn_id" };
      }
      const existing = turns.get(id);
      if (existing && existing.accepted) {
        return {
          status: "duplicate",
          accepted: true,
          turnId: id,
          text: existing.text,
          outcome: existing.outcome,
        };
      }
      if (existing && !existing.accepted) {
        return {
          status: "pending",
          accepted: false,
          turnId: id,
          text: existing.text,
        };
      }
      const row = {
        turnId: id,
        text: String(text || ""),
        status: "pending",
        accepted: false,
        outcome: null,
      };
      turns.set(id, row);
      return { status: "pending", accepted: false, turnId: id, text: row.text };
    },
    acknowledge(turnId, outcome) {
      const id = String(turnId || "").trim();
      const existing = turns.get(id);
      if (!existing) {
        return { status: "rejected", accepted: false, reason: "unknown_turn" };
      }
      const status = String((outcome && outcome.status) || "accepted");
      existing.accepted = true;
      existing.outcome = outcome || { status };
      existing.status = status;
      turns.set(id, existing);
      return {
        status,
        accepted: true,
        turnId: id,
        outcome: existing.outcome,
        duplicate: status === "duplicate",
      };
    },
  };
}

/** Live only after the socket is open, authenticated, and attached. */
export function socketIsLive({ open = false, authenticated = false, attached = false } = {}) {
  return Boolean(open && authenticated && attached);
}

const PLAYBACK_STATES = new Set(["queued", "started", "completed", "failed", "interrupted"]);

/** Playback state machine. Unknown events leave the previous state in place. */
export function advancePlayback(current, event) {
  const prev = current && typeof current === "object" ? current : { status: "", text: "", utteranceId: "" };
  const status = String((event && (event.status || event.event)) || "");
  if (!PLAYBACK_STATES.has(status)) return { ...prev };
  const text = event && event.text != null ? String(event.text) : String(prev.text || "");
  const utteranceId = String(
    (event && (event.utteranceId || event.utterance_id)) || prev.utteranceId || ""
  );
  return { status, text, utteranceId };
}
