// [AI-LAYER] Small fetch helpers for the optional Meta AI layer routes.
import { apiHeaders } from "../apiAuth.js";

export async function aiGet(path) {
  const r = await fetch(path, { headers: apiHeaders(), credentials: "same-origin" });
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}

export async function aiPostJson(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: apiHeaders({ "Content-Type": "application/json" }),
    credentials: "same-origin",
    body: JSON.stringify(body || {}),
  });
  if (!r.ok) throw new Error(`${r.status}`);
  return r.json();
}

export async function aiUpload(path, file) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(path, { method: "POST", headers: apiHeaders(), credentials: "same-origin", body: fd });
  let body = null;
  try { body = await r.json(); } catch { body = null; }
  if (!r.ok) throw new Error((body && body.detail) || `${r.status}`);
  return body;
}

let _statusPromise = null;
/** Cached GET /api/ai/status (one request per page load). */
export function aiStatus({ refresh = false } = {}) {
  if (!_statusPromise || refresh) {
    _statusPromise = aiGet("/api/ai/status").catch(() => ({ mode: "deterministic", unavailable: true }));
  }
  return _statusPromise;
}

export const SPEECH_LANG_KEY = "skewai.ai.speechLang";

/** Optional speech-recognition language override (hi-IN). null = existing behaviour. */
export function aiSpeechLang() {
  try {
    const v = window.localStorage.getItem(SPEECH_LANG_KEY);
    return v === "hi-IN" || v === "en-IN" ? v : null;
  } catch {
    return null;
  }
}

export function setAiSpeechLang(v) {
  try {
    if (!v || v === "en-US") window.localStorage.removeItem(SPEECH_LANG_KEY);
    else window.localStorage.setItem(SPEECH_LANG_KEY, v);
  } catch { /* ignore */ }
}
