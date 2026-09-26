import { useEffect, useRef, useState } from "react";
import { apiHeaders, sendWsAuth } from "../src/apiAuth.js";
import { openCases } from "../src/ui/opsActions.js";
import { IconHangup, IconMic, IconSpeaker } from "../src/icons.jsx";
import {
  BARGE_IN_GRACE_MS,
  CALL_STATE,
  buildSlotEntries,
  callPhaseHint,
  callStateLabel,
  capabilitySnapshot,
  fatalWsErrorMessage,
  greetingSpeakText,
  isFatalWsError,
  mapInteractionEnded,
  mergeTranscriptTurn,
  shouldAcceptSpeechResult,
  shouldAllowBargeIn,
  shouldReconnectOnClose,
  shouldResumeListeningOnOpen,
  slotProgress,
  transcriptFromResume,
  POST_TTS_COOLDOWN_MS,
  STT_FINAL_DEBOUNCE_MS,
  RECOG_RESTART_MS,
  looksLikeAgentEcho,
  coalesceSpeechFinals,
} from "../src/voiceHelpers.js";
import { capabilityLabel } from "../src/ui/labels.js";

// Backoff caps at 5s, so 8 tries ≈ 25s of retrying — comfortably inside the
// server's reconnect grace (FRONTLINE_WS_RECONNECT_GRACE_S, default 120s).
const RECONNECT_MAX_ATTEMPTS = 8;

function getSpeechRecognition() {
  if (typeof window === "undefined") return null;
  return window.SpeechRecognition || window.webkitSpeechRecognition || null;
}

export default function CallWidget() {
  const [state, setState] = useState(CALL_STATE.IDLE);
  const [error, setError] = useState(null);
  const [info, setInfo] = useState(null);
  const [transcript, setTranscript] = useState([]);
  const [slots, setSlots] = useState({});
  const [pack, setPack] = useState(null);
  const [interactionId, setInteractionId] = useState(null);
  const [handoff, setHandoff] = useState(false);
  const [ended, setEnded] = useState(null);
  const [wsStatus, setWsStatus] = useState("idle");
  const [textFallback, setTextFallback] = useState("");
  const [activity, setActivity] = useState(null);
  const [micGranted, setMicGranted] = useState(null);
  const [turnCount, setTurnCount] = useState(0);
  const [speakPhase, setSpeakPhase] = useState("normal"); // 'greeting' | 'normal'
  const [muted, setMuted] = useState(false);
  const [driving, setDriving] = useState(false);
  const [frustration, setFrustration] = useState(0);
  const [latency, setLatency] = useState(null);
  const [consent, setConsent] = useState(null);
  const [safetyMode, setSafetyMode] = useState(false);
  const [callSecs, setCallSecs] = useState(0);
  const [survey, setSurvey] = useState({ csat: 0, resolved: null, sent: false });
  const [region, setRegion] = useState(() => {
    try { return localStorage.getItem("skew_region") || ""; } catch { return ""; }
  });

  const hasSR = !!getSpeechRecognition();
  const hasTTS = typeof window !== "undefined" && "speechSynthesis" in window;

  const wsRef = useRef(null);
  const recogRef = useRef(null);
  const speakingRef = useRef(false);
  const speakPhaseRef = useRef("normal"); // 'greeting' | 'normal'
  const ttsStartedAtRef = useRef(0);
  const bargeStreakRef = useRef(0);
  const audioCtxRef = useRef(null);
  const analyserRef = useRef(null);
  const micStreamRef = useRef(null);
  const bargeRafRef = useRef(null);
  const transcriptEndRef = useRef(null);
  const transcriptScrollRef = useRef(null);
  const intentionalCloseRef = useRef(false);
  const callEndedRef = useRef(false);
  const reconnectAttemptRef = useRef(0);
  const reconnectTimerRef = useRef(null);
  const activeWsUrlRef = useRef(null);
  const interactionIdRef = useRef(null);
  const turnSeqRef = useRef(0);
  const speakWatchdogRef = useRef(null);
  const lastAgentTextRef = useRef("");
  const listenReadyAtRef = useRef(0);
  const speakQueueRef = useRef([]);
  const drainingSpeakRef = useRef(false);
  const sttBufferRef = useRef({ text: "", at: 0 });
  const sttFlushTimerRef = useRef(null);
  const recogRestartTimerRef = useRef(null);
  const lastInterimRef = useRef({ text: "", at: 0 });
  const thinkingTimeoutRef = useRef(null);
  const stateRef = useRef(CALL_STATE.IDLE);
  const wsSeqRef = useRef(0);
  const callStartRef = useRef(0);
  const callTimerRef = useRef(null);
  const callGenRef = useRef(0);
  const mutedRef = useRef(false);
  const voiceDeniedRef = useRef(false);
  const utteranceSeqRef = useRef(0);
  const currentUtteranceRef = useRef(null);
  const humanControlRef = useRef(false);
  const controlGenRef = useRef(0);
  const pendingTerminalRef = useRef(null);
  const [voiceDenied, setVoiceDenied] = useState(false);
  const [humanControl, setHumanControl] = useState(false);
  useEffect(() => {
    stateRef.current = state;
  }, [state]);
  useEffect(() => {
    if (state === CALL_STATE.IDLE || state === CALL_STATE.ENDED) {
      return undefined;
    }
    if (!callStartRef.current) callStartRef.current = Date.now();
    const id = window.setInterval(() => {
      setCallSecs(Math.floor((Date.now() - callStartRef.current) / 1000));
    }, 1000);
    callTimerRef.current = id;
    return () => {
      window.clearInterval(id);
      if (callTimerRef.current === id) callTimerRef.current = null;
    };
  }, [state]);

  function packLocale() {
    try {
      const l = pack && (pack.locale || pack?.pack?.locale);
      if (l) return l;
      if (pack && pack.id && typeof pack.id === "string") return "en-US";
    } catch { /* ignore */ }
    return "en-US";
  }
  function newTurnId() {
    try {
      if (window.crypto && window.crypto.randomUUID) return window.crypto.randomUUID();
    } catch { /* ignore */ }
    return `t_${Date.now().toString(36)}_${Math.floor(Math.random() * 1e6)}`;
  }

  function clearSpeakWatchdog() {
    if (speakWatchdogRef.current) {
      clearTimeout(speakWatchdogRef.current);
      speakWatchdogRef.current = null;
    }
  }

  /**
   * Recovery if the TTS engine never fires onend/onerror (throttled tab,
   * headset switch, dropped utterance). Without this the call wedges in
   * AGENT_SPEAKING with the mic gated off forever.
   */
  function armSpeakWatchdog(full) {
    clearSpeakWatchdog();
    if (typeof window === "undefined") return;
    const ms = Math.min(
      45000,
      Math.max(10000, 6000 + String(full || "").length * 120),
    );
    speakWatchdogRef.current = window.setTimeout(() => {
      speakWatchdogRef.current = null;
      if (!speakingRef.current) return;
      try {
        window.speechSynthesis.cancel();
      } catch {
        /* ignore */
      }
      speakingRef.current = false;
      speakPhaseRef.current = "normal";
      setSpeakPhase("normal");
      drainingSpeakRef.current = false;
      speakQueueRef.current = [];
      armListenCooldown();
      setInfo("Voice output stalled — mic is live, keep speaking or type below");
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        setState(CALL_STATE.LISTENING);
        startRecognitionSafe();
      }
    }, ms);
  }

  function pickVoice() {
    try {
      const vs = window.speechSynthesis.getVoices() || [];
      return (
        vs.find((v) => v.default) ||
        vs.find((v) => v.lang && v.lang.startsWith("en")) ||
        null
      );
    } catch {
      return null;
    }
  }

  function markCallTerminal() {
    // Prevent onclose reconnect from treating a normal server hangup as a drop.
    callGenRef.current += 1;
    intentionalCloseRef.current = true;
    callEndedRef.current = true;
    activeWsUrlRef.current = null;
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (thinkingTimeoutRef.current) {
      clearTimeout(thinkingTimeoutRef.current);
      thinkingTimeoutRef.current = null;
    }
  }

  /** Ask the server to finalize the contact (REST path, works with no socket). */
  function releaseInteraction() {
    const iid = interactionIdRef.current;
    if (!iid) return;
    interactionIdRef.current = null;
    fetch(`/api/interactions/${iid}/end`, {
      method: "POST",
      headers: apiHeaders(),
    }).catch(() => {});
  }

  function closeWsQuietly() {
    try {
      wsRef.current?.close();
    } catch {
      /* ignore */
    }
  }

  function pushTurn(entry) {
    turnSeqRef.current += 1;
    const withId = { ...entry, id: entry.id || `t${turnSeqRef.current}` };
    setTranscript((t) => mergeTranscriptTurn(t, withId));
    if (!entry.interim) setTurnCount((n) => n + 1);
  }

  function stopRecognition() {
    try {
      recogRef.current?.stop();
    } catch {
      /* ignore */
    }
  }

  function clearThinkingTimeout() {
    if (thinkingTimeoutRef.current) {
      clearTimeout(thinkingTimeoutRef.current);
      thinkingTimeoutRef.current = null;
    }
  }

  function enterThinking() {
    // Mic pauses while the server works so background noise during the
    // 5-10s enrichment window cannot pile up duplicate user_turns.
    stopRecognition();
    setState(CALL_STATE.THINKING);
    clearThinkingTimeout();
    thinkingTimeoutRef.current = window.setTimeout(() => {
      thinkingTimeoutRef.current = null;
      if (callEndedRef.current || intentionalCloseRef.current) return;
      if (stateRef.current !== CALL_STATE.THINKING) return;
      setInfo("Still working — if the agent stays quiet, speak or type again");
      setState(CALL_STATE.LISTENING);
      startRecognitionSafe();
    }, 12000);
  }

  function startRecognitionSafe() {
    if (mutedRef.current || voiceDeniedRef.current) return;
    if (callEndedRef.current || pendingTerminalRef.current) return;
    if (!recogRef.current) return;
    if (speakingRef.current) return;
    if (drainingSpeakRef.current) return;
    if (stateRef.current === CALL_STATE.THINKING) return;
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    const now = Date.now();
    if (listenReadyAtRef.current && now < listenReadyAtRef.current) {
      const wait = listenReadyAtRef.current - now;
      if (recogRestartTimerRef.current) clearTimeout(recogRestartTimerRef.current);
      recogRestartTimerRef.current = window.setTimeout(() => {
        recogRestartTimerRef.current = null;
        startRecognitionSafe();
      }, wait + 20);
      return;
    }
    if (recogRestartTimerRef.current) {
      clearTimeout(recogRestartTimerRef.current);
      recogRestartTimerRef.current = null;
    }
    try {
      recogRef.current.start();
    } catch {
      /* already started */
    }
  }

  function armListenCooldown() {
    listenReadyAtRef.current = Date.now() + POST_TTS_COOLDOWN_MS;
  }

  function flushCustomerUtterance() {
    if (mutedRef.current || voiceDeniedRef.current) {
      if (sttFlushTimerRef.current) {
        clearTimeout(sttFlushTimerRef.current);
        sttFlushTimerRef.current = null;
      }
      sttBufferRef.current = { text: "", at: 0 };
      lastInterimRef.current = { text: "", at: 0 };
      return;
    }
    if (sttFlushTimerRef.current) {
      clearTimeout(sttFlushTimerRef.current);
      sttFlushTimerRef.current = null;
    }
    const buf = sttBufferRef.current;
    sttBufferRef.current = { text: "", at: 0 };
    // Single-word answers ("yes") often arrive as interim only and would be
    // lost forever — promote the last interim to a final on flush.
    const fallback = lastInterimRef.current && lastInterimRef.current.text
      ? lastInterimRef.current.text
      : "";
    lastInterimRef.current = { text: "", at: 0 };
    const raw = (buf && buf.text ? buf.text : "") || fallback;
    const text = raw.replace(/\s+/g, " ").trim();
    if (!text) return;
    if (looksLikeAgentEcho(text, lastAgentTextRef.current)) {
      setInfo("Ignored speaker echo — say your answer after the agent finishes");
      return;
    }
    wsSeqRef.current += 1;
    const sent = sendWs({
      type: "user_turn", text, final: true,
      turn_id: newTurnId(), turn_seq: wsSeqRef.current,
      // Web Speech exposes confidence for an entire phrase, not for a vehicle
      // slot. Do not send it: older API servers treated it as entity_1 and
      // looped on a confirmation of every partial phrase.
      locale: packLocale(),
      region: (region || "").trim() || undefined,
      drive_hint: driving || undefined,
    });
    if (!sent) {
      setError("Connection was lost before your reply could be sent. Please reconnect and try again.");
      setState(CALL_STATE.LISTENING);
      startRecognitionSafe();
      return;
    }
    pushTurn({ speaker: "customer", text });
    enterThinking();
  }

  function queueCustomerFinal(txt) {
    const piece = String(txt || "").trim();
    const prev = String((sttBufferRef.current && sttBufferRef.current.text) || "").trim();
    if (piece && prev === piece) return;
    const { buffer } = coalesceSpeechFinals(sttBufferRef.current, txt, Date.now(), STT_FINAL_DEBOUNCE_MS);
    sttBufferRef.current = buffer;
    if (sttFlushTimerRef.current) clearTimeout(sttFlushTimerRef.current);
    sttFlushTimerRef.current = window.setTimeout(() => {
      sttFlushTimerRef.current = null;
      flushCustomerUtterance();
    }, STT_FINAL_DEBOUNCE_MS);
  }

  /**
   * Speak agent text. opts.phase === 'greeting' disables barge-in so speaker
   * bleed cannot cancel the pack greeting mid-word ("than" from "Thanks").
   * Queue later agent turns instead of cancel+speak — Chrome drops the next
   * question when cancel and speak run back-to-back.
   */
  function enqueueSpeak(text, opts = {}) {
    const full = greetingSpeakText(text);
    if (!full) return;
    const speaker = opts.speaker || "agent";
    if (humanControlRef.current && speaker !== "supervisor") return;
    lastAgentTextRef.current = full;
    speakQueueRef.current.push({
      full,
      phase: opts.phase || "normal",
      speaker,
      utteranceId: opts.utteranceId || null,
    });
    if (!drainingSpeakRef.current && !speakingRef.current) {
      drainSpeakQueue();
    }
  }

  function drainSpeakQueue() {
    const next = speakQueueRef.current.shift();
    if (!next) {
      drainingSpeakRef.current = false;
      if (pendingTerminalRef.current) {
        const summary = pendingTerminalRef.current;
        pendingTerminalRef.current = null;
        setEnded(summary);
        setState(CALL_STATE.ENDED);
        setWsStatus("disconnected");
        cleanupCall();
        closeWsQuietly();
        return;
      }
      armListenCooldown();
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN && !humanControlRef.current) {
        setState(CALL_STATE.LISTENING);
        startRecognitionSafe();
      } else if (humanControlRef.current && wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        setState(CALL_STATE.LISTENING);
        startRecognitionSafe();
      }
      return;
    }
    drainingSpeakRef.current = true;
    speakNow(next.full, next.phase, next);
  }

  function speakNow(full, phase, meta = {}) {
    const token = ++utteranceSeqRef.current;
    currentUtteranceRef.current = meta.utteranceId || null;
    speakPhaseRef.current = phase;
    setSpeakPhase(phase);
    bargeStreakRef.current = 0;

    if (!hasTTS) {
      clearSpeakWatchdog();
      speakingRef.current = false;
      speakPhaseRef.current = "normal";
      setSpeakPhase("normal");
      drainSpeakQueue();
      return;
    }
    speakingRef.current = true;
    setState(CALL_STATE.AGENT_SPEAKING);
    stopRecognition();
    armSpeakWatchdog(full);

    const finishSpeaking = () => {
      if (token !== utteranceSeqRef.current) return;
      clearSpeakWatchdog();
      speakingRef.current = false;
      speakPhaseRef.current = "normal";
      setSpeakPhase("normal");
      bargeStreakRef.current = 0;
      if (bargeRafRef.current) {
        cancelAnimationFrame(bargeRafRef.current);
        bargeRafRef.current = null;
      }
      drainSpeakQueue();
    };

    window.setTimeout(() => {
      if (token !== utteranceSeqRef.current) return;
      if (!speakingRef.current) return;
      const u = new SpeechSynthesisUtterance(full);
      u.rate = 1.0;
      try { u.lang = packLocale() || "en-US"; } catch { u.lang = "en-US"; }
      const voice = pickVoice();
      if (voice) u.voice = voice;
      u.onstart = () => {
        if (token !== utteranceSeqRef.current) return;
        ttsStartedAtRef.current = Date.now();
        bargeStreakRef.current = 0;
        if (phase !== "greeting") {
          startBargeWatch();
        }
      };
      u.onend = finishSpeaking;
      u.onerror = finishSpeaking;
      try {
        window.speechSynthesis.speak(u);
      } catch {
        finishSpeaking();
      }
      if (phase !== "greeting" && !ttsStartedAtRef.current) {
        ttsStartedAtRef.current = Date.now();
        startBargeWatch();
      }
    }, 60);
  }

  function cancelAiSpeech() {
    utteranceSeqRef.current += 1;
    currentUtteranceRef.current = null;
    speakQueueRef.current = speakQueueRef.current.filter((item) => item.speaker === "supervisor");
    clearSpeakWatchdog();
    if (hasTTS) {
      try { window.speechSynthesis.cancel(); } catch { /* ignore */ }
    }
    speakingRef.current = false;
    drainingSpeakRef.current = false;
    speakPhaseRef.current = "normal";
    setSpeakPhase("normal");
    if (bargeRafRef.current) {
      cancelAnimationFrame(bargeRafRef.current);
      bargeRafRef.current = null;
    }
    if (speakQueueRef.current.length) drainSpeakQueue();
  }

  function startBargeWatch() {
    const ctx = audioCtxRef.current;
    const analyser = analyserRef.current;
    if (!ctx || !analyser) return;
    if (bargeRafRef.current) cancelAnimationFrame(bargeRafRef.current);
    const data = new Uint8Array(analyser.frequencyBinCount);
    const started = ttsStartedAtRef.current || Date.now();

    const tick = () => {
      if (!speakingRef.current) {
        bargeRafRef.current = null;
        bargeStreakRef.current = 0;
        return;
      }
      // Never barge during pack greeting.
      if (speakPhaseRef.current === "greeting") {
        bargeRafRef.current = requestAnimationFrame(tick);
        return;
      }
      analyser.getByteTimeDomainData(data);
      let sum = 0;
      for (let i = 0; i < data.length; i += 8) {
        const v = (data[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / (data.length / 8));
      const elapsedMs = Date.now() - started;
      const high = rms >= 0.14;
      if (high) bargeStreakRef.current += 1;
      else bargeStreakRef.current = 0;

      if (
        shouldAllowBargeIn({
          phase: speakPhaseRef.current,
          elapsedMs,
          graceMs: BARGE_IN_GRACE_MS,
          rms,
          rmsThreshold: 0.14,
          highRmsStreak: bargeStreakRef.current,
          streakNeeded: 4,
        })
      ) {
        utteranceSeqRef.current += 1;
        try {
          window.speechSynthesis.cancel();
        } catch {
          /* ignore */
        }
        clearSpeakWatchdog();
        speakingRef.current = false;
        speakPhaseRef.current = "normal";
        setSpeakPhase("normal");
        bargeStreakRef.current = 0;
        try {
          const elapsedMs = Date.now() - (ttsStartedAtRef.current || Date.now());
          sendWs({
            type: "barge_in",
            elapsed_ms: Math.max(0, elapsedMs),
            utterance_id: currentUtteranceRef.current || undefined,
          });
        } catch {
          sendWs({ type: "barge_in", utterance_id: currentUtteranceRef.current || undefined });
        }
        speakQueueRef.current = [];
        drainingSpeakRef.current = false;
        armListenCooldown();
        setState(CALL_STATE.LISTENING);
        setInfo("Barge-in — listening again");
        startRecognitionSafe();
        bargeRafRef.current = null;
        return;
      }
      bargeRafRef.current = requestAnimationFrame(tick);
    };
    bargeRafRef.current = requestAnimationFrame(tick);
  }

  function sendWs(obj) {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify(obj));
        return true;
      } catch {
        return false;
      }
    }
    return false;
  }

  function applyControl(msg) {
    const gen = Number(msg.generation || 0);
    if (gen && controlGenRef.current && gen < controlGenRef.current) return;
    if (gen) controlGenRef.current = gen;
    const supervised = msg.supervised === true || msg.state === "SUPERVISED";
    humanControlRef.current = supervised;
    setHumanControl(supervised);
    if (supervised) {
      cancelAiSpeech();
      setInfo("A human specialist has this call");
    }
  }

  function handleWsMessage(msg) {
    switch (msg.type) {
      case "agent_turn": {
        const speaker = msg.speaker || "agent";
        if (humanControlRef.current && speaker !== "supervisor") break;
        const agentText = greetingSpeakText(msg.text || "");
        clearThinkingTimeout();
        if (msg.drive_mode) setDriving(true);
        if (msg.consent_required) setConsent((c) => c || { script: agentText });
        // Safety escalation dominates the UI (Tier-B #13).
        try {
          const t = (agentText || "").toLowerCase();
          if (t.includes("safety specialist") || t.includes("emergency services")) {
            setSafetyMode(true);
          }
        } catch { /* ignore */ }
        pushTurn({ speaker, text: agentText });
        if (speaker === "supervisor") setInfo("A human specialist is responding");
        enqueueSpeak(agentText, {
          phase: msg.closing ? "normal" : "normal",
          speaker,
          utteranceId: msg.utterance_id || msg.turn_id || null,
        });
        break;
      }
      case "control_state": {
        applyControl(msg);
        break;
      }
      case "frustration": {
        try { setFrustration(Number(msg.value || 0)); } catch { /* ignore */ }
        break;
      }
      case "turn_latency": {
        try {
          setLatency({
            elapsed_ms: Number(msg.elapsed_ms || 0),
            over_budget: !!msg.over_budget,
            budget_ms: Number(msg.budget_ms || 1500),
          });
        } catch { /* ignore */ }
        break;
      }
      case "consent_required": {
        try {
          setConsent({ script: String(msg.script || ""), region: msg.region || "", version: msg.version || "" });
          setInfo("Recording consent required — say 'I consent' or tap Consent");
        } catch { /* ignore */ }
        break;
      }
      case "barge_in_reconciled": {
        try {
          setActivity({ agent: "voice_reconciler", action_type: "spoken_turn_truncated", summary: String(msg.audible_text || "[INTERRUPTED]") });
        } catch { /* ignore */ }
        break;
      }
      case "duplicate_dropped": {
        // Proven replay (same turn_id seen twice) — stay live and ask for a
        // fresh turn instead of wedging in THINKING until the 12s timeout.
        clearThinkingTimeout();
        setState(CALL_STATE.LISTENING);
        setInfo("That turn arrived twice — please say it once more");
        startRecognitionSafe();
        break;
      }
      case "agent_activity": {
        setActivity({
          agent: msg.agent,
          action_type: msg.action_type,
          summary: msg.summary || msg.output_summary || "",
        });
        try {
          if (msg.action_type === "supervisor_whisper") setInfo("Supervisor coaching live on console");
          if (msg.action_type === "drive_mode_detected") setDriving(true);
          if (msg.action_type === "vin_captured") setInfo(String(msg.summary || "VIN captured"));
        } catch { /* ignore */ }
        break;
      }
      case "slots_update": {
        setSlots(msg.slots || {});
        break;
      }
      case "handoff_offer": {
        setHandoff(true);
        setInfo("A supervisor can take over this contact");
        break;
      }
      case "interaction_ended": {
        // Server sends flat case_id / investigation_id — not nested payload.
        // Mark terminal BEFORE close so onclose does not reconnect / wipe summary.
        // Closing audio already queued is allowed to finish; user hangup still cancels.
        const summary = mapInteractionEnded(msg);
        markCallTerminal();
        interactionIdRef.current = null;
        stopRecognition();
        pendingTerminalRef.current = summary;
        setEnded(summary);
        if (!speakingRef.current && !drainingSpeakRef.current && speakQueueRef.current.length === 0) {
          pendingTerminalRef.current = null;
          setState(CALL_STATE.ENDED);
          setWsStatus("disconnected");
          cleanupCall();
          closeWsQuietly();
        }
        break;
      }
      case "resumed": {
        // Server accepted the re-attach. Its turn list is authoritative — we may
        // have missed turns while the socket was down.
        applyControl(msg);
        const rows = transcriptFromResume(msg.turns);
        if (rows) {
          setTranscript(rows);
          setTurnCount(rows.length);
          turnSeqRef.current = rows.length;
        }
        setError(null);
        setInfo("Reconnected — the contact is still live");
        break;
      }
      case "error": {
        if (isFatalWsError(msg)) {
          // Retrying cannot help: stop the reconnect loop and close out cleanly
          // instead of looping the raw "active interaction not found" detail.
          markCallTerminal();
          setError(fatalWsErrorMessage(msg));
          setInfo(null);
          setEnded((prev) => prev || { reason: "connection_lost" });
          setState(CALL_STATE.ENDED);
          setWsStatus("disconnected");
          cleanupCall();
          closeWsQuietly();
          break;
        }
        setError(msg.detail || msg.message || "Server error");
        break;
      }
      default:
        break;
    }
  }

  function cleanupCall() {
    utteranceSeqRef.current += 1;
    stopRecognition();
    clearSpeakWatchdog();
    clearThinkingTimeout();
    if (sttFlushTimerRef.current) {
      clearTimeout(sttFlushTimerRef.current);
      sttFlushTimerRef.current = null;
    }
    if (recogRestartTimerRef.current) {
      clearTimeout(recogRestartTimerRef.current);
      recogRestartTimerRef.current = null;
    }
    speakQueueRef.current = [];
    drainingSpeakRef.current = false;
    pendingTerminalRef.current = null;
    sttBufferRef.current = { text: "", at: 0 };
    lastInterimRef.current = { text: "", at: 0 };
    if (bargeRafRef.current) cancelAnimationFrame(bargeRafRef.current);
    bargeRafRef.current = null;
    if (hasTTS) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        /* ignore */
      }
    }
    speakingRef.current = false;
    if (micStreamRef.current) {
      micStreamRef.current.getTracks().forEach((t) => t.stop());
      micStreamRef.current = null;
    }
    if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
      audioCtxRef.current.close().catch(() => {});
    }
    audioCtxRef.current = null;
    analyserRef.current = null;
  }

  async function setupMic(gen) {
    let stream = null;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (gen !== callGenRef.current || callEndedRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        return false;
      }
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const src = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      src.connect(analyser);
      if (gen !== callGenRef.current || callEndedRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        ctx.close().catch(() => {});
        return false;
      }
      micStreamRef.current = stream;
      audioCtxRef.current = ctx;
      analyserRef.current = analyser;
      setMicGranted(true);
      return true;
    } catch {
      if (stream) stream.getTracks().forEach((t) => t.stop());
      if (gen !== callGenRef.current || callEndedRef.current) return false;
      setMicGranted(false);
      setInfo("Microphone blocked — use the text box to talk");
      return false;
    }
  }

  function retrySpeechInput() {
    voiceDeniedRef.current = false;
    setVoiceDenied(false);
    setError(null);
    setInfo("Trying speech recognition again");
    if (!recogRef.current) setupSpeechRecognition();
    if (stateRef.current === CALL_STATE.LISTENING && !pendingTerminalRef.current) {
      startRecognitionSafe();
    }
  }

  function toggleMute() {
    const next = !mutedRef.current;
    mutedRef.current = next;
    setMuted(next);
    try {
      const tracks = (micStreamRef.current && micStreamRef.current.getAudioTracks()) || [];
      tracks.forEach((t) => { t.enabled = !next; });
    } catch { /* ignore */ }
    if (next) {
      if (sttFlushTimerRef.current) {
        clearTimeout(sttFlushTimerRef.current);
        sttFlushTimerRef.current = null;
      }
      if (recogRestartTimerRef.current) {
        clearTimeout(recogRestartTimerRef.current);
        recogRestartTimerRef.current = null;
      }
      sttBufferRef.current = { text: "", at: 0 };
      lastInterimRef.current = { text: "", at: 0 };
      setTranscript((rows) => (Array.isArray(rows) ? rows.filter((t) => !t.interim) : rows));
      stopRecognition();
    } else if (stateRef.current === CALL_STATE.LISTENING) startRecognitionSafe();
  }

  function setupSpeechRecognition() {
    const SR = getSpeechRecognition();
    if (!SR) return null;
    const recog = new SR();
    recog.continuous = true;
    recog.interimResults = true;
    try { recog.lang = packLocale() || "en-US"; } catch { recog.lang = "en-US"; }
    recog.onresult = (ev) => {
      const accept = shouldAcceptSpeechResult({
        speaking: speakingRef.current,
        wsOpen: !!(wsRef.current && wsRef.current.readyState === WebSocket.OPEN),
        listenReadyAt: listenReadyAtRef.current,
        now: Date.now(),
      });
      // While thinking the mic is paused — drop late STT so it cannot stack
      // a second user_turn behind the one the server is already answering.
      if (stateRef.current === CALL_STATE.THINKING) return;
      if (mutedRef.current || voiceDeniedRef.current) return;
      if (!accept) return;

      let interim = "";
      for (let i = ev.resultIndex; i < ev.results.length; i++) {
        const res = ev.results[i];
        const txt = (res[0].transcript || "").trim();
        if (!txt) continue;
        if (res.isFinal) {
          if (looksLikeAgentEcho(txt, lastAgentTextRef.current)) continue;
          lastInterimRef.current = { text: "", at: 0 };
          queueCustomerFinal(txt);
        } else {
          interim += (interim ? " " : "") + txt;
        }
      }
      if (interim && !looksLikeAgentEcho(interim, lastAgentTextRef.current)) {
        lastInterimRef.current = { text: interim, at: Date.now() };
        pushTurn({ speaker: "customer", text: interim, interim: true });
      }
    };
    recog.onerror = (e) => {
      if (e.error === "no-speech" || e.error === "aborted") return;
      if (e.error === "not-allowed" || e.error === "service-not-allowed") {
        voiceDeniedRef.current = true;
        setVoiceDenied(true);
        setMicGranted(false);
        setInfo("Speech recognition denied — type your replies below");
        stopRecognition();
        return;
      }
      setError(`Speech: ${e.error}`);
    };
    recog.onend = () => {
      if (mutedRef.current || voiceDeniedRef.current) return;
      if (speakingRef.current) return;
      if (speakPhaseRef.current === "greeting") return;
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
      if (intentionalCloseRef.current) return;
      if (stateRef.current === CALL_STATE.THINKING) {
        // A short answer ("yes") may end the session with only interim text
        // buffered — promote it to a final instead of dropping the turn.
        if (sttBufferRef.current.text || lastInterimRef.current.text) {
          flushCustomerUtterance();
        }
        return;
      }
      if (!sttBufferRef.current.text && lastInterimRef.current.text) {
        queueCustomerFinal(lastInterimRef.current.text);
        lastInterimRef.current = { text: "", at: 0 };
      }
      // Chrome may end a recognition session between pieces of one spoken
      // sentence. Keep the debounce window alive and restart recognition so
      // "my car has ... been stuck" becomes one turn, not two competing turns.
      if (recogRestartTimerRef.current) clearTimeout(recogRestartTimerRef.current);
      recogRestartTimerRef.current = window.setTimeout(() => {
        recogRestartTimerRef.current = null;
        startRecognitionSafe();
      }, RECOG_RESTART_MS);
    };
    recogRef.current = recog;
    return recog;
  }

  async function startCall() {
    const gen = ++callGenRef.current;
    setError(null);
    setInfo(null);
    setTranscript([]);
    setSlots({});
    setHandoff(false);
    setEnded(null);
    setActivity(null);
    setTurnCount(0);
    turnSeqRef.current = 0;
    wsSeqRef.current = 0;
    speakPhaseRef.current = "normal";
    setSpeakPhase("normal");
    setMicGranted(null);
    setMuted(false);
    mutedRef.current = false;
    voiceDeniedRef.current = false;
    setVoiceDenied(false);
    humanControlRef.current = false;
    setHumanControl(false);
    controlGenRef.current = 0;
    pendingTerminalRef.current = null;
    setDriving(false);
    setFrustration(0);
    setLatency(null);
    setConsent(null);
    setSafetyMode(false);
    setSurvey({ csat: 0, resolved: null, sent: false });
    callStartRef.current = Date.now();
    setCallSecs(0);
    sttBufferRef.current = { text: "", at: 0 };
    lastInterimRef.current = { text: "", at: 0 };
    if (thinkingTimeoutRef.current) {
      clearTimeout(thinkingTimeoutRef.current);
      thinkingTimeoutRef.current = null;
    }
    setWsStatus("connecting");
    setState(CALL_STATE.CONNECTING);

    // Ask for microphone access in parallel with contact creation. Waiting for
    // getUserMedia before opening the WebSocket made a permission prompt (or a
    // slow Bluetooth device) look like a slow connection.
    const micReady = setupMic(gen);
    let startRes;
    try {
      try { localStorage.setItem("skew_region", region || ""); } catch { /* ignore */ }
      const qs = new URLSearchParams({ channel: "web_voice" });
      if ((region || "").trim()) qs.set("region", region.trim());
      const r = await fetch(`/api/interactions/start?${qs.toString()}`, {
        method: "POST",
        headers: apiHeaders(),
      });
      if (!r.ok) {
        const detail = await r.text().catch(() => "");
        throw new Error(`Could not start call (${r.status})${detail ? `: ${detail.slice(0, 120)}` : ""}`);
      }
      startRes = await r.json();
    } catch (e) {
      if (gen !== callGenRef.current) return;
      setError(String(e.message || e));
      setState(CALL_STATE.IDLE);
      void micReady.then(() => cleanupCall());
      return;
    }
    if (gen !== callGenRef.current) {
      const staleId = startRes && startRes.interaction_id;
      if (staleId) {
        fetch(`/api/interactions/${encodeURIComponent(staleId)}/end`, {
          method: "POST",
          headers: apiHeaders(),
        }).catch(() => {});
      }
      return;
    }

    setInteractionId(startRes.interaction_id);
    interactionIdRef.current = startRes.interaction_id;
    setPack(startRes.pack || null);
    intentionalCloseRef.current = false;
    callEndedRef.current = false;
    reconnectAttemptRef.current = 0;

    // Warm up the TTS voice list inside the user gesture so the first
    // utterance is not dropped by engines that load voices lazily.
    try {
      if (hasTTS) {
        window.speechSynthesis.getVoices();
        window.speechSynthesis.onvoiceschanged = () => {
          try {
            window.speechSynthesis.getVoices();
          } catch {
            /* ignore */
          }
        };
      }
    } catch {
      /* ignore */
    }

    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${proto}//${window.location.host}${startRes.ws_url}`;
    activeWsUrlRef.current = wsUrl;

    function bindWs(ws, { isReconnect }) {
      const boundGen = gen;
      wsRef.current = ws;
      ws.onopen = () => {
        if (boundGen !== callGenRef.current) {
          try { ws.close(); } catch { /* ignore */ }
          return;
        }
        if (
          !shouldResumeListeningOnOpen({
            intentionalClose: intentionalCloseRef.current,
            callEnded: callEndedRef.current,
            isReconnect,
          }) &&
          isReconnect
        ) {
          // Call already ended while reconnect was queued — stay quiet.
          setWsStatus("disconnected");
          closeWsQuietly();
          return;
        }
        sendWsAuth(ws);
        setWsStatus("live");
        reconnectAttemptRef.current = 0;
        if (!isReconnect && startRes.greeting_text) {
          // Full pack greeting — transcript + TTS must match start response.
          const greet = greetingSpeakText(startRes.greeting_text);
          pushTurn({ speaker: "agent", text: greet });
          enqueueSpeak(greet, { phase: "greeting" });
        } else if (
          shouldResumeListeningOnOpen({
            intentionalClose: intentionalCloseRef.current,
            callEnded: callEndedRef.current,
            isReconnect: true,
          }) ||
          !isReconnect
        ) {
          // Reconnect resume OR first open without greeting path.
          if (isReconnect || !startRes.greeting_text) {
            speakingRef.current = false;
            setState(CALL_STATE.LISTENING);
            startRecognitionSafe();
          }
        }
      };
      ws.onmessage = (ev) => {
        if (boundGen !== callGenRef.current) return;
        try {
          handleWsMessage(JSON.parse(ev.data));
        } catch {
          /* ignore bad frames */
        }
      };
      ws.onerror = () => {
        if (boundGen !== callGenRef.current) return;
        if (callEndedRef.current || intentionalCloseRef.current) return;
        // onerror fires on every failed reconnect attempt; do not overwrite
        // the "reconnecting…" info line. onclose owns the terminal banner.
        if (reconnectAttemptRef.current > 0 || isReconnect) return;
        setError("WebSocket connection error");
      };
      ws.onclose = () => {
        if (boundGen !== callGenRef.current) return;
        if (
          !shouldReconnectOnClose({
            intentionalClose: intentionalCloseRef.current,
            callEnded: callEndedRef.current,
          })
        ) {
          setWsStatus("disconnected");
          return;
        }
        if (reconnectAttemptRef.current >= RECONNECT_MAX_ATTEMPTS) {
          setWsStatus("disconnected");
          setState(CALL_STATE.ENDED);
          setEnded((prev) => prev || { reason: "connection_lost" });
          markCallTerminal();
          cleanupCall();
          // Release the contact server-side now instead of leaving it parked
          // until the reconnect grace expires.
          releaseInteraction();
          return;
        }
        setWsStatus("reconnecting");
        setInfo(
          `Connection dropped — reconnecting (${reconnectAttemptRef.current + 1}/${RECONNECT_MAX_ATTEMPTS})…`
        );
        const delay = Math.min(5000, 400 * Math.pow(1.6, reconnectAttemptRef.current));
        reconnectAttemptRef.current += 1;
        reconnectTimerRef.current = setTimeout(() => {
          if (
            !shouldReconnectOnClose({
              intentionalClose: intentionalCloseRef.current,
              callEnded: callEndedRef.current,
            }) ||
            !activeWsUrlRef.current
          ) {
            return;
          }
          bindWs(new WebSocket(activeWsUrlRef.current), { isReconnect: true });
        }, delay);
      };
    }

    bindWs(new WebSocket(wsUrl), { isReconnect: false });
    // Recognition itself does not require a MediaStream, but wait until the
    // microphone request settles so barge-in analysis and browser permissions
    // are ready before we begin listening.
    void micReady.finally(() => {
      if (gen !== callGenRef.current || callEndedRef.current || intentionalCloseRef.current) return;
      setupSpeechRecognition();
      if (stateRef.current === CALL_STATE.LISTENING) startRecognitionSafe();
    });
  }

  function endCall() {
    markCallTerminal();
    setWsStatus("disconnected");
    sendWs({ type: "hangup" });
    setState(CALL_STATE.ENDED);
    setEnded((prev) => prev || { reason: "user_hangup" });
    cleanupCall();
    releaseInteraction();
    closeWsQuietly();
  }

  // Cleanup on unmount: arm intentional close + clear reconnect timer (no ghost reconnects).
  useEffect(() => {
    return () => {
      markCallTerminal();
      cleanupCall();
      // Leaving the page is a deliberate hangup — do not park the contact for
      // the whole reconnect grace window.
      releaseInteraction();
      closeWsQuietly();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-scroll transcript to newest turn
  useEffect(() => {
    const el = transcriptScrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [transcript]);

  function submitTextFallback(e) {
    e.preventDefault();
    const text = textFallback.trim();
    if (!text) return;
    if (callEndedRef.current || pendingTerminalRef.current) return;
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      setError("Not connected — start a call first");
      return;
    }
    // If agent is speaking, cancel TTS so we don't talk over ourselves.
    // speechSynthesis.cancel often does not fire utterance onend — resume STT
    // explicitly and clear speakPhase so recog.onend is not stuck on greeting.
    if (speakingRef.current && hasTTS) {
      utteranceSeqRef.current += 1;
      try {
        window.speechSynthesis.cancel();
      } catch {
        /* ignore */
      }
      clearSpeakWatchdog();
      speakingRef.current = false;
    }
    speakQueueRef.current = [];
    drainingSpeakRef.current = false;
    speakPhaseRef.current = "normal";
    setSpeakPhase("normal");
    wsSeqRef.current += 1;
    const sent = sendWs({
      type: "user_turn", text, final: true,
      turn_id: newTurnId(), turn_seq: wsSeqRef.current,
      locale: packLocale(), region: (region || "").trim() || undefined,
      drive_hint: driving || undefined,
    });
    if (!sent) {
      setError("Connection was lost before your reply could be sent. Please reconnect and try again.");
      return;
    }
    pushTurn({ speaker: "customer", text });
    setTextFallback("");
    lastInterimRef.current = { text: "", at: 0 };
    enterThinking();
  }

  function sendConsent() {
    if (callEndedRef.current || pendingTerminalRef.current) return;
    wsSeqRef.current += 1;
    const txt = "I consent";
    const sent = sendWs({
      type: "user_turn", text: txt, final: true,
      turn_id: newTurnId(), turn_seq: wsSeqRef.current,
      locale: packLocale(), region: (region || "").trim() || undefined,
    });
    if (!sent) {
      setError("Connection was lost before your consent could be sent. Please reconnect and try again.");
      return;
    }
    pushTurn({ speaker: "customer", text: txt });
    setConsent(null);
    enterThinking();
  }

  async function requestHuman() {
    const iid = interactionIdRef.current;
    if (!iid) {
      setError("Not connected — start a call first");
      return;
    }
    try {
      const r = await fetch(`/api/interactions/${iid}/handoff/accept`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...apiHeaders() },
        body: JSON.stringify({ force: true }),
      });
      const body = await r.json().catch(() => ({}));
      if (!r.ok || body.accepted === false) {
        throw new Error(body.reason || `handoff failed (${r.status})`);
      }
      setHandoff(true);
      setInfo(body.already_pending
        ? "A human specialist is already being paged. Please stay on this call."
        : "A human specialist has been paged. Please stay on this call while they join.");
    } catch (e) {
      setError(`Could not page a human specialist: ${String((e && e.message) || e).slice(0, 120)}`);
    }
  }

  function openLiveConsole() {
    try {
      const iid = interactionIdRef.current || interactionId;
      const base = window.location.href.split("#")[0];
      const target = `${base}#console${iid ? `?id=${encodeURIComponent(iid)}` : ""}`;
      const opened = window.open(target, "_blank", "noopener");
      if (opened) {
        setInfo("Supervisor console opened in a new tab. This call remains connected here.");
      } else {
        setInfo("Your browser blocked the new supervisor tab. This call remains connected here.");
      }
      return;
    } catch { /* ignore */ }
    setInfo("Unable to open the supervisor console. This call remains connected here.");
  }

  function copyReceipt() {
    try {
      const lines = transcript.map((t) => `[${t.speaker}] ${t.text}`).join("\n");
      const body = `Skew AI contact ${interactionId || ""}\n${lines}\nCase: ${ended?.case_id || "—"}`;
      navigator.clipboard.writeText(body).then(
        () => setInfo("Transcript copied — receipt ready to email"),
        () => setInfo("Copy blocked — select the transcript manually"),
      );
    } catch { /* ignore */ }
  }

  async function sendSurvey() {
    if (!interactionId || survey.sent) return;
    try {
      await fetch(`/api/interactions/${interactionId}/outcome`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...apiHeaders() },
        body: JSON.stringify({ csat: survey.csat || null, resolved: survey.resolved }),
      });
      setSurvey((s) => ({ ...s, sent: true }));
      setInfo("Thanks — survey recorded");
    } catch {
      setError("Survey failed to send");
    }
  }

  const slotEntries = buildSlotEntries(pack, slots);
  const progress = slotProgress(slotEntries);
  const caps = capabilitySnapshot({
    hasSpeechRecognition: hasSR,
    hasSpeechSynthesis: hasTTS,
    hasMicStream: !!micStreamRef.current || micGranted === true,
  });

  const inCall = state !== CALL_STATE.IDLE && state !== CALL_STATE.ENDED;
  const canStart = state === CALL_STATE.IDLE || state === CALL_STATE.ENDED;
  const showCompose = true;

  return (
    <div className="voice-page page-enter">
      <header className="page-header">
        <div>
          <h1>Voice agent</h1>
          <p className="sub">
            Live contact with the Skew AI orchestrator. Wait for the agent to finish, then speak or
            type. The mic stays off during TTS so the next question is not lost to echo.
            {pack ? (
              <>
                {" "}
                Pack <span className="mono">{pack.id}</span>
                {pack.display_name ? ` · ${pack.display_name}` : ""}.
              </>
            ) : null}
          </p>
        </div>
        <div className="page-actions">
          {!canStart && (
            <button type="button" className="danger voice-cta" onClick={endCall}>
              <IconHangup />
              End call
            </button>
          )}
        </div>
      </header>

      {error && (
        <div className="banner banner-error" role="alert">
          {error}{" "}
          <button type="button" className="ghost" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}
      {info && !error && (
        <div className="banner banner-ok" role="status">
          {info}{" "}
          <button type="button" className="ghost" onClick={() => setInfo(null)}>
            Dismiss
          </button>
        </div>
      )}
      {consent && (
        <div className="banner banner-warn" role="alert">
          {(consent.script || "This call may be recorded. Say 'I consent' to continue.")}{" "}
          <button type="button" className="primary" onClick={sendConsent}>I consent</button>
        </div>
      )}
      {safetyMode && inCall && (
        <div className="banner banner-error" role="alert">
          Safety escalation active — follow the specialist script. Do not drive until reviewed.
        </div>
      )}
      {(!hasSR || voiceDenied) && !canStart && (
        <div className="banner banner-warn" role="status">
          Speech recognition is unavailable — type your replies below. This call sends text only and does not upload audio.
          {voiceDenied && hasSR ? (
            <>
              {" "}
              <button type="button" className="ghost" onClick={retrySpeechInput}>
                Try microphone again
              </button>
            </>
          ) : null}
        </div>
      )}

      <div className="call-layout">
        <div className="call-stage">
          <div className="call-hero">
            <div
              className={
                "call-ring " +
                state +
                (speakPhase === "greeting" ? " greeting" : "")
              }
              aria-hidden="true"
            >
              <div className="mic-orbit">
                <span className="call-aura call-aura-a" />
                <span className="call-aura call-aura-b" />
                <span className="call-aura call-aura-c" />
                <button
                  type="button"
                  className={
                    "mic-btn " +
                    state +
                    (speakPhase === "greeting" ? " greeting" : "")
                  }
                  onClick={canStart ? startCall : endCall}
                  disabled={state === CALL_STATE.CONNECTING}
                  aria-label={canStart ? "Start call" : "End call"}
                >
                  {state === CALL_STATE.CONNECTING || state === CALL_STATE.THINKING ? (
                    <span className="spinner" aria-hidden="true" />
                  ) : canStart ? (
                    <IconMic width={32} height={32} />
                  ) : state === CALL_STATE.LISTENING ? (
                    <IconMic width={32} height={32} />
                  ) : state === CALL_STATE.AGENT_SPEAKING ? (
                    <IconSpeaker width={32} height={32} />
                  ) : (
                    <IconHangup width={32} height={32} />
                  )}
                </button>
              </div>
              {(state === CALL_STATE.LISTENING ||
                state === CALL_STATE.AGENT_SPEAKING ||
                state === CALL_STATE.THINKING) && (
                <div className="voice-waves-wrap">
                  <div
                    className={
                      "voice-waves" +
                      (state === CALL_STATE.LISTENING ? " listening" : " speaking") +
                      (speakPhase === "greeting" ? " greeting" : "")
                    }
                  >
                    <span /><span /><span /><span /><span />
                  </div>
                </div>
              )}
            </div>
            <div
              className={
                "call-state " +
                state +
                (speakPhase === "greeting" ? " greeting" : "")
              }
            >
              <span className="label">{callStateLabel(state, { speakPhase, textOnly: voiceDenied || !hasSR })}</span>
              {callPhaseHint(state, { speakPhase, textOnly: voiceDenied || !hasSR }) && (
                <span className="phase-hint">{callPhaseHint(state, { speakPhase, textOnly: voiceDenied || !hasSR })}</span>
              )}
              <span className="call-meta">
                <span className={"status-pip " + (wsStatus === "live" ? "ok" : wsStatus === "reconnecting" ? "warn" : "")} />
                {wsStatus === "idle" ? "idle" : `ws ${wsStatus}`}
                {interactionId && (
                  <>
                    {" · "}
                    <span className="mono" title={interactionId}>
                      {interactionId.slice(0, 16)}…
                    </span>
                  </>
                )}
                {turnCount > 0 && <> · {turnCount} turns</>}
                {inCall && <> · {Math.floor(callSecs / 60)}:{String(callSecs % 60).padStart(2, "0")}</>}
                {humanControl && inCall && <> · human specialist</>}
                {latency && <> · {latency.elapsed_ms}ms{latency.over_budget ? " over budget" : ""}</>}
                {frustration > 0 && <> · frustration {frustration.toFixed(2)}</>}
              </span>
              {inCall && (
                <div className="row" style={{ marginTop: 8, flexWrap: "wrap", gap: 8 }}>
                  <button type="button" className="ghost" onClick={toggleMute}>{muted ? "Unmute" : "Mute"}</button>
                  <button type="button" className={driving ? "primary" : "ghost"} onClick={() => setDriving((d) => !d)}>{driving ? "Driving: on" : "I'm driving"}</button>
                  <button type="button" className="ghost" onClick={requestHuman}>Get me a human</button>
                  <label className="mono faint" style={{ display: "flex", gap: 4, alignItems: "center" }}>
                    Region
                    <input aria-label="Region for consent" value={region} onChange={(e) => setRegion(e.target.value)} placeholder="CA" style={{ width: 56 }} />
                  </label>
                </div>
              )}
            </div>
            {progress.total > 0 && (
              <div className="slot-meter" aria-label={`Slots ${progress.filled} of ${progress.total}`}>
                <div className="slot-meter-track">
                  <div className="slot-meter-fill" style={{ width: `${progress.pct}%` }} />
                </div>
                <span className="slot-meter-label mono">
                  {progress.filled}/{progress.total} slots
                </span>
              </div>
            )}
          </div>

          {activity && inCall && (
            <div className="activity-strip" aria-live="polite">
              <span className="mono faint">{activity.agent}</span>
              <span>{activity.summary || activity.action_type}</span>
            </div>
          )}

          <div
            className="transcript"
            ref={(node) => {
              transcriptScrollRef.current = node;
              transcriptEndRef.current = node;
            }}
            aria-live="polite"
            aria-relevant="additions"
          >
            {transcript.length === 0 && (
              <div className="empty-state">
                <p className="empty-state-text">
                  {state === CALL_STATE.IDLE
                    ? hasSR
                      ? "Start a call, allow the microphone, then speak naturally. You can always type below."
                      : "This browser has no Web Speech API. Start a call and type every customer turn."
                    : state === CALL_STATE.CONNECTING
                      ? "Opening the contact and waiting for the greeting…"
                      : "Waiting for the first turn…"}
                </p>
              </div>
            )}
            {transcript.map((t) => (
              <div
                key={t.id || `${t.speaker}-${t.text.slice(0, 12)}`}
                className={"turn " + t.speaker + (t.interim ? " interim" : "")}
              >
                <div className="who">
                  {t.speaker}
                  {t.interim ? " · listening" : ""}
                </div>
                <div>{t.text}</div>
              </div>
            ))}
          </div>

          {showCompose && (
            <form className="call-compose" onSubmit={submitTextFallback}>
              <input
                aria-label="Type a customer turn"
                placeholder={
                  canStart
                    ? hasSR
                      ? "Start a call, then type or speak"
                      : "Start a text contact, then type here"
                    : hasSR
                      ? "Type a reply (or speak) — Enter to send"
                      : "Type your reply — Enter to send"
                }
                value={textFallback}
                onChange={(e) => setTextFallback(e.target.value)}
                disabled={state === CALL_STATE.ENDED || state === CALL_STATE.CONNECTING || canStart}
                autoComplete="off"
              />
              <button
                type="submit"
                className={inCall && textFallback.trim() ? "primary" : "ghost"}
                disabled={
                  !inCall ||
                  state === CALL_STATE.ENDED ||
                  state === CALL_STATE.CONNECTING ||
                  !textFallback.trim()
                }
              >
                Send
              </button>
            </form>
          )}

          {handoff && inCall && (
            <div className="banner banner-ok" role="status" style={{ marginTop: 12 }}>
              Supervisor handoff queued — pickup SLA 75s.{" "}
              <button type="button" className="primary" onClick={openLiveConsole}>
                Open supervisor console in new tab
              </button>{" "}
              <button type="button" className="ghost" onClick={requestHuman}>
                Re-page supervisor
              </button>
            </div>
          )}

          {ended && state === CALL_STATE.ENDED && (
            <div className="result-card" style={{ marginTop: 14 }}>
              <div className="result-card-title">Call ended</div>
              <div className="kvs">
                <span className="k">reason</span>
                <span className="v mono">{ended.reason || "ended"}</span>
                <span className="k">case</span>
                <span className="v mono">{ended.case_id || "—"}</span>
                <span className="k">investigation</span>
                <span className="v mono">
                  {ended.investigation_id ||
                    (ended.investigation_opened ? "opened" : "—")}
                </span>
                <span className="k">audit</span>
                <span className="v">{ended.audit_pending ? "pending" : "n/a"}</span>
              </div>
              <div className="row" style={{ marginTop: 12 }}>
                <button type="button" className="primary" onClick={startCall}>
                  New call
                </button>
                {ended.case_id && (
                  <button
                    type="button"
                    className="ghost"
                    onClick={() => openCases({ caseId: ended.case_id, status: "open" })}
                  >
                    Open case queue
                  </button>
                )}
                <button type="button" className="ghost" onClick={copyReceipt}>Copy receipt</button>
              </div>
              <div className="row" style={{ marginTop: 12, flexWrap: "wrap", gap: 8 }}>
                <span className="mono faint">Did we get your issue right?</span>
                {[1, 2, 3, 4, 5].map((n) => (
                  <button key={n} type="button" className={survey.csat === n ? "primary" : "ghost"} onClick={() => setSurvey((s) => ({ ...s, csat: n }))}>{n}</button>
                ))}
                <button type="button" className={survey.resolved === true ? "primary" : "ghost"} onClick={() => setSurvey((s) => ({ ...s, resolved: true }))}>Resolved</button>
                <button type="button" className={survey.resolved === false ? "primary" : "ghost"} onClick={() => setSurvey((s) => ({ ...s, resolved: false }))}>Not resolved</button>
                <button type="button" className="ghost" disabled={survey.sent} onClick={sendSurvey}>{survey.sent ? "Sent" : "Send survey"}</button>
              </div>
              {interactionId && (
                <p className="muted small" style={{ marginTop: 8 }}>
                  Resume link: <span className="mono">{window.location.origin}{window.location.pathname}#/call?resume={interactionId.slice(0, 16)}…</span>
                </p>
              )}
            </div>
          )}
        </div>

        <aside className="stack">
          <div className="panel">
            <h2>Slot frame</h2>
            {slotEntries.length === 0 ? (
              <div className="empty-state">
                <p className="empty-state-text">
                  Slots appear as the agent fills the pack frame from your answers.
                </p>
              </div>
            ) : (
              slotEntries.map((s) => (
                <div className="slot-row" key={s.key || s.label}>
                  <span className="k">{s.label}</span>
                  <span className={"v" + (s.value ? "" : " empty")}>{s.value || "—"}</span>
                </div>
              ))
            )}
          </div>

          <details className="panel">
            <summary style={{ cursor: "pointer", fontWeight: 600 }}>This browser</summary>
            <div className="kvs" style={{ marginTop: 12 }}>
              <span className="k">{capabilityLabel("stt")}</span>
              <span className="v">
                {caps.sttOk ? (
                  <span className="ok-text">{caps.stt}</span>
                ) : (
                  <span className="warn-text">{caps.stt}</span>
                )}
              </span>
              <span className="k">{capabilityLabel("tts")}</span>
              <span className="v">
                {caps.ttsOk ? (
                  <span className="ok-text">{caps.tts}</span>
                ) : (
                  <span className="warn-text">{caps.tts}</span>
                )}
              </span>
              <span className="k">{capabilityLabel("barge_in")}</span>
              <span className="v">
                {caps.bargeOk ? (
                  <span className="ok-text">{caps.bargeIn}</span>
                ) : (
                  <span className="faint">{caps.bargeIn}</span>
                )}
              </span>
              <span className="k">{capabilityLabel("mic")}</span>
              <span className="v">
                {micGranted === true ? (
                  <span className="ok-text">Granted</span>
                ) : micGranted === false ? (
                  <span className="warn-text">Denied / blocked</span>
                ) : (
                  <span className="faint">Not requested</span>
                )}
              </span>
              <span className="k">{capabilityLabel("text path")}</span>
              <span className="v ok-text">Always available in-call</span>
            </div>
            <p className="muted small" style={{ marginTop: 12 }}>
              While the agent speaks, recognition is paused so TTS is not captured as a customer
              turn. Interrupt by speaking or typing.
            </p>
          </details>

          <div className="panel">
            <h2>Related</h2>
            <button type="button" className="ghost" onClick={openLiveConsole}>
              Live console
            </button>
          </div>
        </aside>
      </div>
    </div>
  );
}
