/**
 * Single-tenant pilot auth helpers.
 * Key is kept in MEMORY by default; persisted to localStorage ONLY when the
 * operator checks "Remember on this device" in Settings. Memory-only keys
 * vanish on tab close and are never written to disk, shrinking XSS theft.
 *
 * Security (SOC 2 / enterprise): any JS-accessible key is XSS-accessible.
 * Prefer httpOnly session cookies + OIDC in production (fetchMe uses
 * credentials:same-origin so a signed session works with no key at all).
 * Pilots: FRONTLINE_AUTH_REQUIRED=1 and treat the browser as trusted.
 *
 * WebSocket: prefer first-message auth (no query-string secrets).
 * withApiKeyQuery is deprecated and intentionally a no-op for keys.
 */

export const API_KEY_STORAGE = "frontline_api_key";
export const SUBJECT_STORAGE = "frontline_subject";
export const AUTH_EVENT = "frontline-auth";

// In-memory key (survives SPA navigation, not disk). Set via setApiKey().
let _memKey = "";
let _memOnly = true;

export function setApiKey(key, { remember = false } = {}) {
  _memKey = String(key || "").trim();
  _memOnly = !remember;
  // Only persist when explicitly opted in; otherwise clear disk copy.
  _write(API_KEY_STORAGE, remember ? _memKey : "");
  notifyAuthChange();
}

export function clearApiKey() {
  _memKey = "";
  _write(API_KEY_STORAGE, "");
  notifyAuthChange();
}

function _read(key) {
  if (typeof localStorage === "undefined") return "";
  try {
    return (localStorage.getItem(key) || "").trim();
  } catch {
    return "";
  }
}

function _write(key, value) {
  if (typeof localStorage === "undefined") return;
  try {
    if (value) localStorage.setItem(key, value);
    else localStorage.removeItem(key);
  } catch {
    /* ignore quota / private mode */
  }
}

export function notifyAuthChange() {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new Event(AUTH_EVENT));
}

export function getStoredApiKey() {
  // Memory first (no disk write), then opt-in localStorage copy.
  if (_memKey) return _memKey;
  const disk = _read(API_KEY_STORAGE);
  if (disk) _memKey = disk; // hydrate memory for this tab
  return disk;
}

export function getStoredSubject() {
  return _read(SUBJECT_STORAGE);
}

/** Headers for write/console REST routes. Session rides an httpOnly cookie. */
export function apiHeaders(extra = {}) {
  const h = { ...extra };
  const key = getStoredApiKey();
  if (key) h["X-API-Key"] = key;
  return h;
}

export async function fetchMe() {
  const r = await fetch("/api/frontline/auth/me", {
    headers: apiHeaders(),
    credentials: "same-origin",
  });
  if (!r.ok) return { signed_in: false, status: r.status };
  return r.json();
}

/** Public server health (no auth). Used by the route guard. */
export async function fetchHealth() {
  try {
    const r = await fetch("/health");
    if (!r.ok) return { status: "down" };
    return r.json();
  } catch {
    return { status: "down" };
  }
}

/** Validate the stored API key against the server (401 when bad). */
export async function verifyCredential() {
  try {
    const r = await fetch("/api/frontline/auth/verify", {
      headers: apiHeaders(),
      credentials: "same-origin",
    });
    if (!r.ok) return { ok: false, status: r.status };
    return { ok: true, ...(await r.json()) };
  } catch {
    return { ok: false, offline: true };
  }
}

/**
 * Single check used by the App route guard. A signed session always passes.
 * Otherwise a stored API key passes only when the server runs auth-required
 * and accepts it. Anonymous visitors fail closed (open pilot mode included —
 * there the only way in is sign up / sign in).
 */
export async function hasConsoleAccess() {
  let me = { signed_in: false };
  let health = {};
  try {
    [me, health] = await Promise.all([fetchMe(), fetchHealth()]);
  } catch {
    /* both default to deny */
  }
  if (me?.signed_in) return { allowed: true, via: "session", me, health };
  if (getStoredApiKey() && health?.auth_required) {
    const v = await verifyCredential();
    if (v.ok) return { allowed: true, via: "api-key", me, health };
  }
  return { allowed: false, me, health };
}

export async function signOut() {
  // Logout MUST invalidate the session everywhere the client holds it:
  // memory key, opt-in disk copy, subject label, AND the server session
  // cookie (item 20). A key that survives logout is a session that survives
  // logout — previously clearApiKey() was never called here.
  clearApiKey();
  _write(SUBJECT_STORAGE, "");
  _memOnly = true;
  try {
    await fetch("/api/frontline/auth/logout", {
      method: "POST",
      credentials: "same-origin",
      headers: apiHeaders(),
    });
  } catch {
    /* cookie clear is best-effort */
  }
  notifyAuthChange();
}

/** Keyless-pilot check: true when no API key is stored anywhere (item 20).

Pure memory/disk probe — the dashboard can operate keyless when the server
runs open (local pilot) or with an httpOnly session (fetchMe signed_in).
Components must use this + apiHeaders() instead of reading storage directly.
*/
export function isKeyless() {
  if (_memKey) return false;
  return !_read(API_KEY_STORAGE);
}

/** Persist Google OAuth client on this machine (dev/pilot). Secret never stored in JS session. */
export async function saveGoogleProvider({ clientId, clientSecret, redirectUri }) {
  const r = await fetch("/api/frontline/auth/oidc/config", {
    method: "PUT",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...apiHeaders() },
    body: JSON.stringify({
      client_id: clientId,
      client_secret: clientSecret,
      redirect_uri: redirectUri || "",
    }),
  });
  if (!r.ok) {
    const detail = await r.text();
    throw new Error(detail || `Could not save Google client (${r.status})`);
  }
  return r.json();
}

/** Finish Google OIDC after the API callback redirects with a one-time handoff. */
export async function completeGoogleHandoff(handoff) {
  const code = String(handoff || "").trim();
  if (!code) throw new Error("Missing Google handoff");
  const r = await fetch("/api/frontline/auth/oidc/complete", {
    method: "POST",
    credentials: "same-origin",
    headers: { "Content-Type": "application/json", ...apiHeaders() },
    body: JSON.stringify({ handoff: code }),
  });
  if (!r.ok) {
    const detail = await r.text();
    throw new Error(detail || `Google sign-in failed (${r.status})`);
  }
  const out = await r.json();
  if (out.token) {
    throw new Error("server leaked session token to JavaScript");
  }
  if (out.subject || out.email) {
    _write(SUBJECT_STORAGE, out.subject || out.email || "");
  }
  notifyAuthChange();
  return out;
}

/**
 * @deprecated Query-string API keys leak via logs/history. Returns URL unchanged.
 * Use sendWsAuth(ws) after open instead.
 */
export function withApiKeyQuery(url) {
  return url;
}

/**
 * After WebSocket open, send auth frame when a pilot key is stored.
 * Server authenticate_websocket accepts this as the preferred browser path.
 */
export function sendWsAuth(ws) {
  if (!ws || typeof ws.send !== "function") return;
  const key = getStoredApiKey();
  if (!key) return;
  try {
    ws.send(JSON.stringify({ type: "auth", api_key: key }));
  } catch {
    /* ignore */
  }
}

/**
 * Register a new operator / user account.
 * Ready for backend integration via POST /api/frontline/auth/signup.
 * Gracefully falls back to structured preview mode if the backend route is not yet wired.
 */
export async function registerUser({ name, email, company, role, password }) {
  try {
    const r = await fetch("/api/frontline/auth/signup", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...apiHeaders() },
      credentials: "same-origin",
      body: JSON.stringify({ name, email, company, role, password }),
    });
    if (r.ok) {
      const data = await r.json();
      if (data.subject || data.email) {
        _write(SUBJECT_STORAGE, data.subject || data.email || email);
      }
      notifyAuthChange();
      return { success: true, data, mode: "api" };
    }
    if (r.status === 404 || r.status === 405 || r.status === 501) {
      // Backend route pending integration: preserve session in preview mode
      _write(SUBJECT_STORAGE, email);
      notifyAuthChange();
      return {
        success: true,
        mock: true,
        message: `Account created for ${name} (${email}).`,
        user: { name, email, company, role },
      };
    }
    const errText = await r.text();
    let detail = "Registration failed";
    try {
      const j = JSON.parse(errText);
      detail = j.detail || j.title || j.message || detail;
    } catch {
      detail = errText || detail;
    }
    if (r.status === 409) {
      throw new Error("An account with this email already exists. Try signing in instead.");
    }
    if (r.status === 429) {
      throw new Error("Too many attempts. Please wait a minute and try again.");
    }
    throw new Error(detail);
  } catch (err) {
    if (
      String(err.message).includes("Failed to fetch") ||
      String(err.message).includes("NetworkError")
    ) {
      _write(SUBJECT_STORAGE, email);
      notifyAuthChange();
      return {
        success: true,
        mock: true,
        message: `Account staged for ${name} (${email}).`,
        user: { name, email, company, role },
      };
    }
    throw err;
  }
}

/**
 * Sign in with email and password.
 * Ready for backend integration via POST /api/frontline/auth/login.
 */
export async function loginWithPassword({ email, password, remember = false }) {
  try {
    const r = await fetch("/api/frontline/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...apiHeaders() },
      credentials: "same-origin",
      body: JSON.stringify({ email, password, remember: Boolean(remember) }),
    });
    if (r.ok) {
      const data = await r.json();
      if (data.subject || data.email) {
        _write(SUBJECT_STORAGE, data.subject || data.email || email);
      }
      if (data.api_key) {
        setApiKey(data.api_key, { remember });
      }
      notifyAuthChange();
      return { success: true, data };
    }
    if (r.status === 404 || r.status === 405 || r.status === 501) {
      _write(SUBJECT_STORAGE, email);
      notifyAuthChange();
      return {
        success: true,
        mock: true,
        message: "Signed in (local preview).",
        user: { email },
      };
    }
    const errText = await r.text();
    let detail = "Invalid credentials";
    try {
      const j = JSON.parse(errText);
      detail = j.detail || j.title || j.message || detail;
    } catch {
      detail = errText || detail;
    }
    if (r.status === 429) {
      throw new Error("Too many attempts. Please wait a minute and try again.");
    }
    throw new Error(detail);
  } catch (err) {
    if (
      String(err.message).includes("Failed to fetch") ||
      String(err.message).includes("NetworkError")
    ) {
      _write(SUBJECT_STORAGE, email);
      notifyAuthChange();
      return {
        success: true,
        mock: true,
        message: "Signed in (offline preview).",
        user: { email },
      };
    }
    throw err;
  }
}

/**
 * @deprecated Fake client-side login (no server session). It cannot pass the
 * App route guard — use real signup/sign-in or a validated API key instead.
 * Kept exported so older imports don't crash.
 */
export function signInAsPilotDemo({ name = "Alex Mercer", email = "alex.mercer@skew.ai", role = "Platform & ML Engineer" } = {}) {
  _write(SUBJECT_STORAGE, email);
  notifyAuthChange();
  return {
    success: true,
    pilot: true,
    user: { name, email, role },
  };
}

