/**
 * Shared ops actions used by Command Center + Command Palette.
 * Keeps deep-links and demo traffic one source of truth.
 */
import { apiHeaders } from "../apiAuth.js";

export const SS = {
  caseSev: "frontline:case_filter_sev",
  caseStatus: "frontline:case_filter_status",
  caseSelect: "frontline:case_select_id",
  consoleSelect: "frontline:console_select_id",
  caseSearch: "frontline:case_search_q",
};

export const INCLUDE_SIM_KEY = "fl.includeSimulated";

/** Retired hashes → canonical route + tab. */
export const HASH_ALIASES = {
  insights: { id: "warning", tab: "insights" },
  economics: { id: "warning", tab: "economics" },
  trust: { id: "audits", tab: "reports" },
  labels: { id: "audits", tab: "labels" },
  builder: { id: "settings", tab: "builder" },
  studio: { id: "settings", tab: "lab" },
  enterprise: { id: "platform", tab: "ops" },
};

export function parseLocationHash() {
  const raw = window.location.hash.slice(1) || "command";
  const qIndex = raw.indexOf("?");
  const path = (qIndex === -1 ? raw : raw.slice(0, qIndex)).replace(/^\/+/, "") || "command";
  const qs = qIndex === -1 ? "" : raw.slice(qIndex + 1);
  return { id: path, params: new URLSearchParams(qs) };
}

export function hashQueryObject() {
  return Object.fromEntries(parseLocationHash().params.entries());
}

export function setHash(id, params = {}) {
  const sp = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v != null && v !== "") sp.set(k, String(v));
  });
  const q = sp.toString();
  window.location.hash = q ? `${id}?${q}` : id;
}

export function goHash(id) {
  window.location.hash = id;
}

export function patchHashQuery(updates) {
  const { id, params } = parseLocationHash();
  Object.entries(updates).forEach(([k, v]) => {
    if (v == null || v === "") params.delete(k);
    else params.set(k, String(v));
  });
  const q = params.toString();
  const dest = q ? `${id}?${q}` : id;
  if (window.location.hash.slice(1) !== dest) window.location.hash = dest;
}

export function canonicalizeHash() {
  const { id, params } = parseLocationHash();
  const alias = HASH_ALIASES[id];
  if (!alias) return id;
  if (alias.tab && !params.get("tab")) params.set("tab", alias.tab);
  const q = params.toString();
  const dest = `${alias.id}${q ? `?${q}` : ""}`;
  if (window.location.hash.slice(1) !== dest) window.location.hash = dest;
  return alias.id;
}

export function openCases({ severity, status, caseId, q, clusterId } = {}) {
  try {
    if (severity) sessionStorage.setItem(SS.caseSev, severity);
    else sessionStorage.removeItem(SS.caseSev);
    if (status) sessionStorage.setItem(SS.caseStatus, status);
    else sessionStorage.removeItem(SS.caseStatus);
    if (caseId) sessionStorage.setItem(SS.caseSelect, caseId);
    if (q) sessionStorage.setItem(SS.caseSearch, q);
  } catch {
    /* ignore */
  }
  setHash("cases", {
    severity: severity || undefined,
    status: status || undefined,
    id: caseId || undefined,
    q: q || undefined,
    cluster: clusterId != null ? clusterId : undefined,
  });
}

export function openConsole(interactionId) {
  try {
    if (interactionId) sessionStorage.setItem(SS.consoleSelect, interactionId);
  } catch {
    /* ignore */
  }
  setHash("console", interactionId ? { id: interactionId } : {});
}

export function openWarning({ clusterId, packId, tab } = {}) {
  setHash("warning", {
    tab: tab || undefined,
    cluster: clusterId != null ? clusterId : undefined,
    pack: packId || undefined,
  });
}

export function openIssue(clusterId) {
  if (clusterId === null || clusterId === undefined || clusterId === "") return;
  goHash(`issue/${encodeURIComponent(String(clusterId))}`);
}

export function consumeSession(key) {
  try {
    const v = sessionStorage.getItem(key);
    if (v != null) sessionStorage.removeItem(key);
    return v;
  } catch {
    return null;
  }
}

/** Replay N corpus records as simulated contacts (fills wallboard / analytics). */
export async function simulateTraffic({ count = 15, speed = "instant" } = {}) {
  const params = new URLSearchParams({
    count: String(count),
    speed: String(speed),
  });
  const r = await fetch(`/api/frontline/simulate?${params}`, {
    method: "POST",
    headers: apiHeaders(),
  });
  if (!r.ok) {
    const t = await r.text().catch(() => "");
    throw new Error(`simulate failed (${r.status}) ${t.slice(0, 160)}`);
  }
  return r.json();
}

export async function fetchOpenP1Cases(limit = 10) {
  return fetchCriticalCases(limit);
}

export async function fetchCriticalCases(limit = 10) {
  const r = await fetch(
    `/api/frontline/cases?severity=Critical&status=open&limit=${limit}&include_simulated=${includeSimQuery()}`,
    { headers: apiHeaders() },
  );
  if (!r.ok) return [];
  const d = await r.json();
  return d.cases || [];
}

export function readIncludeSimulated() {
  try {
    const raw = window.localStorage.getItem(INCLUDE_SIM_KEY);
    if (raw == null || raw === "") return false;
    const v = JSON.parse(raw);
    return v === true || v === "true";
  } catch {
    return false;
  }
}

export function includeSimQuery() {
  return readIncludeSimulated() ? "true" : "false";
}
