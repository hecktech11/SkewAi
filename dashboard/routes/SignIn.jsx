import { useEffect, useState } from "react";
import {
  completeGoogleHandoff,
  fetchMe,
  getStoredSubject,
  hasConsoleAccess,
  registerUser,
  saveGoogleProvider,
  setApiKey,
  loginWithPassword,
} from "../src/apiAuth.js";

function routeParams() {
  const raw = window.location.hash.slice(1);
  const q = raw.includes("?") ? raw.slice(raw.indexOf("?") + 1) : "";
  const hp = new URLSearchParams(q);
  const sp = new URLSearchParams(window.location.search);
  for (const [k, v] of sp.entries()) {
    if (!hp.has(k)) hp.set(k, v);
  }
  return hp;
}

const iconBase = {
  width: 16,
  height: 16,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  "aria-hidden": true,
};

function GoogleMark() {
  return (
    <svg className="google-mark" viewBox="0 0 48 48" aria-hidden="true">
      <path
        fill="#EA4335"
        d="M24 9.5c3.54 0 6.71 1.22 9.21 3.6l6.85-6.85C35.9 2.38 30.47 0 24 0 14.62 0 6.51 5.38 2.56 13.22l7.98 6.19C12.43 13.72 17.74 9.5 24 9.5z"
      />
      <path
        fill="#4285F4"
        d="M46.98 24.55c0-1.57-.15-3.09-.38-4.55H24v9.02h12.94c-.58 2.96-2.26 5.48-4.78 7.18l7.73 6c4.51-4.18 7.09-10.36 7.09-17.65z"
      />
      <path
        fill="#FBBC05"
        d="M10.53 28.59c-.48-1.45-.76-2.99-.76-4.59s.27-3.14.76-4.59l-7.98-6.19C.92 16.46 0 20.12 0 24c0 3.88.92 7.54 2.56 10.78l7.97-6.19z"
      />
      <path
        fill="#34A853"
        d="M24 48c6.48 0 11.93-2.13 15.89-5.81l-7.73-6c-2.15 1.45-4.92 2.3-8.16 2.3-6.26 0-11.57-4.22-13.47-9.91l-7.98 6.19C6.51 42.62 14.62 48 24 48z"
      />
    </svg>
  );
}

function EyeIcon({ show }) {
  return show ? (
    <svg {...iconBase}>
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  ) : (
    <svg {...iconBase}>
      <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24" />
      <line x1="1" y1="1" x2="23" y2="23" />
    </svg>
  );
}

function IconZap() {
  return (
    <svg {...iconBase} width="20" height="20">
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  );
}

function IconShieldCheck() {
  return (
    <svg {...iconBase} width="20" height="20">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <polyline points="9 12 11 14 15 10" />
    </svg>
  );
}

function IconCrosshair() {
  return (
    <svg {...iconBase} width="20" height="20">
      <circle cx="12" cy="12" r="10" />
      <line x1="22" y1="12" x2="18" y2="12" />
      <line x1="6" y1="12" x2="2" y2="12" />
      <line x1="12" y1="6" x2="12" y2="2" />
      <line x1="12" y1="22" x2="12" y2="18" />
    </svg>
  );
}

function IconCheck() {
  return (
    <svg {...iconBase} width="14" height="14">
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

function IconCopy() {
  return (
    <svg {...iconBase} width="14" height="14">
      <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </svg>
  );
}

function calcStrength(pwd) {
  if (!pwd) return { score: 0, label: "None", color: "transparent", checks: {} };
  const hasLength = pwd.length >= 12;
  const hasUpper = /[A-Z]/.test(pwd);
  const hasNumber = /[0-9]/.test(pwd);
  const hasSymbol = /[^A-Za-z0-9]/.test(pwd);

  let score = 0;
  if (hasLength) score++;
  if (hasUpper) score++;
  if (hasNumber) score++;
  if (hasSymbol) score++;

  let label = "Weak";
  let color = "var(--danger)";
  if (score === 2) {
    label = "Fair";
    color = "var(--warn)";
  } else if (score === 3) {
    label = "Good";
    color = "var(--ok)";
  } else if (score === 4) {
    label = "Strong";
    color = "var(--ok)";
  }

  return {
    score,
    label,
    color,
    checks: { hasLength, hasUpper, hasNumber, hasSymbol },
  };
}

export default function SignIn({ initialMode = "signin", onNavigate }) {
  const [mode, setMode] = useState(initialMode); // "signin" | "signup"
  const [status, setStatus] = useState(null);
  const [me, setMe] = useState(null);
  const [err, setErr] = useState("");
  const [successMsg, setSuccessMsg] = useState("");
  const [busy, setBusy] = useState(false);

  // Common fields
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [remember, setRemember] = useState(true);

  // Sign up fields
  const [name, setName] = useState("");
  const [company, setCompany] = useState("");
  const [role, setRole] = useState("Operations Lead / Commander");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [agreeTerms, setAgreeTerms] = useState(false);
  const [regSuccess, setRegSuccess] = useState(null);

  // Google Pilot / Self-hosted modal assistant
  const [showOidcAssistant, setShowOidcAssistant] = useState(false);
  const [showOidcForm, setShowOidcForm] = useState(false);
  const [copiedRedirect, setCopiedRedirect] = useState(false);
  const [clientId, setClientId] = useState("");
  const [clientSecret, setClientSecret] = useState("");

  // Forgot Password modal
  const [showForgotBox, setShowForgotBox] = useState(false);

  // Operator API-key entry (key-only operators can't reach Settings ungated)
  const [showApiKeyBox, setShowApiKeyBox] = useState(false);
  const [apiKeyInput, setApiKeyInput] = useState("");

  async function handleApiKeySubmit(e) {
    e.preventDefault();
    setErr("");
    setSuccessMsg("");
    const key = apiKeyInput.trim();
    if (!key) {
      setErr("Paste your operator API key first.");
      return;
    }
    setBusy(true);
    try {
      setApiKey(key, { remember: true });
      const access = await hasConsoleAccess();
      if (access.allowed) {
        setApiKeyInput("");
        setShowApiKeyBox(false);
        setSuccessMsg("API key accepted. Opening console…");
        setTimeout(() => goToConsole(), 400);
      } else {
        setErr("That key was not accepted by the server. Check it and try again.");
      }
    } catch (ex) {
      setErr(String(ex.message || ex));
    } finally {
      setBusy(false);
    }
  }

  const redirectUri =
    status?.redirect_uri || "http://127.0.0.1:8000/api/frontline/auth/oidc/callback";

  useEffect(() => {
    if (window.location.hash === "#signup") {
      setMode("signup");
    } else if (window.location.hash === "#signin") {
      setMode("signin");
    } else if (initialMode) {
      setMode(initialMode);
    }
  }, [initialMode]);

  useEffect(() => {
    const params = routeParams();
    const handoff = params.get("handoff");
    const error = params.get("error");
    if (error) setErr(error.replaceAll("_", " "));

    fetch("/api/frontline/auth/oidc/status")
      .then((r) => (r.ok ? r.json() : {}))
      .then(setStatus)
      .catch(() => setStatus({ configured: false }));

    fetchMe()
      .then((m) => {
        setMe(m);
        if (m?.signed_in && !handoff) {
          goToConsole();
        }
      })
      .catch(() => setMe({ signed_in: false }));

    if (handoff) {
      setBusy(true);
      completeGoogleHandoff(handoff)
        .then(() => {
          goToConsole();
        })
        .catch((e) => {
          setErr(String(e.message || e));
          setBusy(false);
        });
    }
  }, []);

  function goToConsole() {
    if (typeof onNavigate === "function") {
      onNavigate("command");
    } else {
      window.location.hash = "command";
    }
  }

  const configured = Boolean(status?.configured);
  const signedIn = Boolean(me?.signed_in);
  const pwdStrength = calcStrength(password);

  function startGoogle() {
    if (!configured) {
      setErr("");
      setShowOidcAssistant(true);
      return;
    }
    const next = `${window.location.origin}/ui/`;
    window.location.href = `/api/frontline/auth/oidc/start?next=${encodeURIComponent(next)}`;
  }

  async function handleCopyRedirect() {
    try {
      await navigator.clipboard.writeText(redirectUri);
      setCopiedRedirect(true);
      setTimeout(() => setCopiedRedirect(false), 2000);
    } catch {
      // fallback
    }
  }

  async function connectAndStart(e) {
    e.preventDefault();
    setErr("");
    setBusy(true);
    try {
      const out = await saveGoogleProvider({
        clientId: clientId.trim(),
        clientSecret: clientSecret.trim(),
        redirectUri,
      });
      setStatus(out);
      setShowOidcAssistant(false);
      startGoogle();
    } catch (ex) {
      setErr(String(ex.message || ex));
      setBusy(false);
    }
  }

  async function handleSignInSubmit(e) {
    e.preventDefault();
    setErr("");
    setSuccessMsg("");
    setBusy(true);
    try {
      await loginWithPassword({
        email: email.trim(),
        password,
        remember,
      });
      setSuccessMsg("Signed in successfully. Launching console…");
      setTimeout(() => {
        goToConsole();
      }, 500);
    } catch (ex) {
      setErr(String(ex.message || ex));
      setBusy(false);
    }
  }

  async function handleSignUpSubmit(e) {
    e.preventDefault();
    setErr("");
    setSuccessMsg("");

    if (password !== confirmPassword) {
      setErr("Passwords do not match. Please re-enter.");
      return;
    }
    if (password.length < 12) {
      setErr("Password must be at least 12 characters long.");
      return;
    }
    if (!agreeTerms) {
      setErr("Please agree to the Terms of Service and Privacy Policy.");
      return;
    }

    setBusy(true);
    try {
      const res = await registerUser({
        name: name.trim(),
        email: email.trim(),
        company: company.trim(),
        role,
        password,
      });

      // Confirm the HttpOnly session cookie actually landed (it won't when
      // cookies are blocked) so the CTA below never dead-ends on console.
      let sessionOk = false;
      try {
        const m = await fetchMe();
        sessionOk = Boolean(m?.signed_in);
      } catch {
        sessionOk = false;
      }

      setRegSuccess({
        name: name.trim(),
        email: email.trim(),
        company: company.trim(),
        role,
        mock: Boolean(res.mock),
        sessionOk,
        message: res.message || "Account created successfully.",
      });
      setBusy(false);
    } catch (ex) {
      setErr(String(ex.message || ex));
      setBusy(false);
    }
  }

  function switchMode(newMode) {
    setMode(newMode);
    setErr("");
    setSuccessMsg("");
    setRegSuccess(null);
    setShowOidcAssistant(false);
    setShowForgotBox(false);
    window.location.hash = newMode;
  }

  return (
    <div className="auth-container">
      {/* ── Left Column: Value Proposition & Live Telemetry Stream ── */}
      <div className="auth-showcase">
        <div className="auth-showcase-content">
          <div className="auth-badge">
            <span className="auth-badge-dot" />
            Skew AI · Enterprise Ops Platform
          </div>

          <h1 className="auth-headline">
            Grounded Voice-of-Customer Intelligence &amp; Root-Cause RCA
          </h1>
          <p className="auth-lead">
            Supervise autonomous voice agent swarms in real-time, detect emerging failure clusters,
            and audit 100% of customer interactions against cryptographic truth ledgers.
          </p>

          {/* ── Live RCA Observability Terminal Preview ── */}
          <div className="telemetry-box" aria-hidden="true">
            <div className="telemetry-head">
              <div className="terminal-dots">
                <span className="dot dot-red" />
                <span className="dot dot-amber" />
                <span className="dot dot-green" />
              </div>
              <span className="terminal-title">telemetry-stream.live</span>
              <span className="terminal-pill">
                <span className="pill-pulse" /> LIVE · 4.2k ev/s
              </span>
            </div>

            <div className="telemetry-metrics">
              <div className="telemetry-kpi">
                <span className="kpi-label">Corpus Integrity</span>
                <span className="kpi-val text-ok">100% SHA-256</span>
              </div>
              <div className="telemetry-kpi">
                <span className="kpi-label">Swarm Latency</span>
                <span className="kpi-val">182ms (P99)</span>
              </div>
              <div className="telemetry-kpi">
                <span className="kpi-label">Anomaly Clusters</span>
                <span className="kpi-val text-accent">3 Emerging</span>
              </div>
              <div className="telemetry-kpi">
                <span className="kpi-label">Evidence Grounding</span>
                <span className="kpi-val text-ok">99.8% High Conf</span>
              </div>
            </div>

            <div className="telemetry-logs">
              <div className="telemetry-log-row">
                <span className="log-time">00:01.2</span>
                <span className="log-badge log-triage">TRIAGE</span>
                <span className="log-msg">Session #4829 clustered: <code>powertrain_stall</code></span>
              </div>
              <div className="telemetry-log-row">
                <span className="log-time">00:04.7</span>
                <span className="log-badge log-audit">AUDIT</span>
                <span className="log-msg">Qubot v2 cryptographic hash chain verified</span>
              </div>
              <div className="telemetry-log-row">
                <span className="log-time">00:08.3</span>
                <span className="log-badge log-supervise">SUPERVISE</span>
                <span className="log-msg">Supervisor takeover ready · zero friction score</span>
              </div>
            </div>
          </div>

          {/* ── 3 Architecture Feature Pillars ── */}
          <div className="auth-features">
            <div className="auth-feature-item">
              <div className="auth-feature-icon">
                <IconZap />
              </div>
              <div>
                <div className="auth-feature-title">Real-Time Swarm Triage</div>
                <div className="auth-feature-desc">
                  Sub-second intent parsing, sentiment friction scoring, and seamless supervisor takeover.
                </div>
              </div>
            </div>

            <div className="auth-feature-item">
              <div className="auth-feature-icon">
                <IconShieldCheck />
              </div>
              <div>
                <div className="auth-feature-title">Cryptographic Audit Ledgers</div>
                <div className="auth-feature-desc">
                  Qubot v2 verifies evidence IDs against DuckDB historical corpora with immutable hash chains.
                </div>
              </div>
            </div>

            <div className="auth-feature-item">
              <div className="auth-feature-icon">
                <IconCrosshair />
              </div>
              <div>
                <div className="auth-feature-title">Early-Warning Anomaly Clustering</div>
                <div className="auth-feature-desc">
                  Live call patterns automatically group into named investigations before customer escalations spike.
                </div>
              </div>
            </div>
          </div>

          <div className="auth-footnote">
            <span>SOC 2 Type II Baseline</span>
            <span>·</span>
            <span>Zero Hallucination Audit Trails</span>
            <span>·</span>
            <span>DuckDB Domain-as-Data</span>
          </div>
        </div>
      </div>

      {/* ── Right Column: Authentication Card (Sign In / Sign Up) ── */}
      <div className="auth-form-pane">
        <div className="auth-card">
          {/* Top Brand / Mode Switcher */}
          <div className="auth-card-head">
            <div className="auth-brand-lockup">
              Skew <em>AI</em>
              <span className="auth-version-tag">v2.4</span>
            </div>
            <div className="auth-tabs" role="tablist" aria-label="Authentication Options">
              <button
                type="button"
                role="tab"
                aria-selected={mode === "signin"}
                className={`auth-tab ${mode === "signin" ? "active" : ""}`}
                onClick={() => switchMode("signin")}
              >
                Sign In
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={mode === "signup"}
                className={`auth-tab ${mode === "signup" ? "active" : ""}`}
                onClick={() => switchMode("signup")}
              >
                Create Account
              </button>
            </div>
          </div>

          {signedIn && !busy && !regSuccess && (
            <div className="banner banner-ok" role="status">
              Currently signed in as <strong>{me.subject || getStoredSubject()}</strong>.{" "}
              <button type="button" className="ghost link-btn" onClick={goToConsole}>
                Open console →
              </button>
            </div>
          )}

          {err && (
            <div className="banner banner-error" role="alert">
              <span>{err}</span>
              <button
                type="button"
                className="banner-close-btn"
                onClick={() => setErr("")}
                aria-label="Dismiss error"
              >
                ×
              </button>
            </div>
          )}

          {successMsg && (
            <div className="banner banner-ok" role="status">
              {successMsg}
            </div>
          )}

          {/* ── Mode: Sign Up Success Confirmation ── */}
          {regSuccess ? (
            <div className="auth-success-box">
              <div className="success-icon-badge">
                <IconCheck />
              </div>
              <h2>Account Created!</h2>
              <p className="sub">
                Welcome to Skew AI, <strong>{regSuccess.name}</strong>. Your profile has been configured for{" "}
                <strong>{regSuccess.company || "your organization"}</strong>.
              </p>
              {!regSuccess.sessionOk && !regSuccess.mock && (
                <p className="sub">
                  Your browser blocked the sign-in cookie — please sign in with your new credentials to continue.
                </p>
              )}

              <div className="reg-summary-card">
                <div className="reg-row">
                  <span className="label">Work Email</span>
                  <span className="val">{regSuccess.email}</span>
                </div>
                <div className="reg-row">
                  <span className="label">Assigned Role</span>
                  <span className="val">{regSuccess.role}</span>
                </div>
                {regSuccess.mock && (
                  <div className="reg-integration-pill">
                    ⚡ <strong>Ready for Backend API:</strong> Plug into <code>POST /api/frontline/auth/signup</code>.
                  </div>
                )}
              </div>

              <div className="auth-actions-group">
                {regSuccess.sessionOk ? (
                  <button
                    type="button"
                    className="btn btn-primary btn-block"
                    onClick={goToConsole}
                  >
                    Launch Console →
                  </button>
                ) : (
                  <button
                    type="button"
                    className="btn btn-primary btn-block"
                    onClick={() => switchMode("signin")}
                  >
                    Continue to Sign In →
                  </button>
                )}
                <button
                  type="button"
                  className="ghost btn-block"
                  onClick={() => switchMode("signin")}
                >
                  Return to Sign In
                </button>
              </div>
            </div>
          ) : (
            <>
              {/* ── SSO Option: Google ── */}
              <button
                type="button"
                className="google-btn"
                onClick={startGoogle}
                disabled={busy}
              >
                <GoogleMark />
                <span>{mode === "signup" ? "Sign up with Google" : "Continue with Google"}</span>
              </button>

              {/* ── Smooth Google SSO Pilot Assistant (when unconfigured locally) ── */}
              {showOidcAssistant && (
                <div className="oidc-assistant-card" role="region" aria-label="SSO Pilot Assistant">
                  <div className="assistant-head">
                    <div className="assistant-title-row">
                      <span className="assistant-badge">Local Pilot</span>
                      <strong>Google Single Sign-On</strong>
                    </div>
                    <button
                      type="button"
                      className="assistant-close"
                      onClick={() => setShowOidcAssistant(false)}
                      aria-label="Close assistant"
                    >
                      ×
                    </button>
                  </div>

                  <p className="assistant-hint">
                    Self-hosted Google OAuth credentials are not configured on this machine yet. Create an account below, or configure real credentials:
                  </p>

                  <div className="assistant-actions">
                    <button
                      type="button"
                      className="btn btn-primary btn-sm"
                      onClick={() => {
                        setShowOidcAssistant(false);
                        switchMode("signup");
                      }}
                    >
                      Create an account instead
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={() => setShowOidcForm(!showOidcForm)}
                    >
                      {showOidcForm ? "Hide Credentials Setup" : "Configure Google Credentials"}
                    </button>
                  </div>

                  {showOidcForm && (
                    <div className="oidc-form-drawer">
                      <div className="redirect-copy-box">
                        <span className="redirect-label">OAuth Redirect Callback URI:</span>
                        <div className="redirect-row">
                          <code className="redirect-code">{redirectUri}</code>
                          <button
                            type="button"
                            className="btn-copy"
                            onClick={handleCopyRedirect}
                            title="Copy redirect URI"
                          >
                            <IconCopy />
                            <span>{copiedRedirect ? "Copied!" : "Copy"}</span>
                          </button>
                        </div>
                      </div>

                      <div className="auth-form-sub">
                        <label className="auth-label">
                          <span>Google Client ID</span>
                          <input
                            type="text"
                            value={clientId}
                            onChange={(e) => setClientId(e.target.value)}
                            placeholder="….apps.googleusercontent.com"
                          />
                        </label>
                        <label className="auth-label">
                          <span>Client Secret</span>
                          <input
                            type="password"
                            value={clientSecret}
                            onChange={(e) => setClientSecret(e.target.value)}
                            placeholder="••••••••••••••••"
                          />
                        </label>
                        <button
                          type="button"
                          className="btn btn-primary btn-block btn-sm"
                          onClick={connectAndStart}
                          disabled={busy || !clientId.trim() || !clientSecret.trim()}
                        >
                          Save &amp; Connect Google SSO
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              <div className="auth-separator">
                <span>or continue with work email</span>
              </div>

              {/* ── Mode: Sign In ── */}
              {mode === "signin" && (
                <form className="auth-form" onSubmit={handleSignInSubmit}>
                  <label className="auth-label">
                    <span>Work Email</span>
                    <input
                      type="email"
                      name="email"
                      autoComplete="username"
                      required
                      placeholder="name@company.com"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                    />
                  </label>

                  <label className="auth-label">
                    <div className="label-head">
                      <span>Password</span>
                      <button
                        type="button"
                        className="faint-link"
                        onClick={() => setShowForgotBox(!showForgotBox)}
                      >
                        Forgot password?
                      </button>
                    </div>
                    <div className="password-input-wrap">
                      <input
                        type={showPassword ? "text" : "password"}
                        name="password"
                        autoComplete="current-password"
                        required
                        placeholder="Enter your password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                      />
                      <button
                        type="button"
                        className="pwd-toggle-btn"
                        onClick={() => setShowPassword(!showPassword)}
                        title={showPassword ? "Hide password" : "Show password"}
                        tabIndex="-1"
                      >
                        <EyeIcon show={showPassword} />
                      </button>
                    </div>
                  </label>

                  {/* Sleek Forgot Password helper note */}
                  {showForgotBox && (
                    <div className="forgot-notice-box" role="status">
                      <div className="forgot-notice-head">
                        <strong>Password Recovery (Self-Hosted Pilot)</strong>
                        <button
                          type="button"
                          className="forgot-close"
                          onClick={() => setShowForgotBox(false)}
                        >
                          ×
                        </button>
                      </div>
                      <p>
                        In pilot mode, operator credentials are local. Create a new account under the{" "}
                        <button
                          type="button"
                          className="inline-link-btn"
                          onClick={() => switchMode("signup")}
                        >
                          Create Account
                        </button>{" "}
                        tab, or ask your admin to reset your password.
                      </p>
                    </div>
                  )}

                  <div className="auth-options-row">
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={remember}
                        onChange={(e) => setRemember(e.target.checked)}
                      />
                      <span>Remember on this device</span>
                    </label>
                  </div>

                  <button
                    type="submit"
                    className="btn btn-primary btn-block"
                    disabled={busy}
                  >
                    {busy ? "Signing in…" : "Sign In to Console"}
                  </button>

                  <div className="auth-switch-prompt">
                    Don't have an account?{" "}
                    <button
                      type="button"
                      className="link-btn"
                      onClick={() => switchMode("signup")}
                    >
                      Create one
                    </button>
                  </div>

                  <div className="oidc-collapsible">
                    <button
                      type="button"
                      className="oidc-toggle-link"
                      onClick={() => setShowApiKeyBox(!showApiKeyBox)}
                      aria-expanded={showApiKeyBox}
                    >
                      {showApiKeyBox ? "Hide API key entry −" : "Have an operator API key? +"}
                    </button>
                    {showApiKeyBox && (
                      <form className="auth-form-sub" onSubmit={handleApiKeySubmit}>
                        <label className="auth-label">
                          <span>Operator API key</span>
                          <input
                            type="password"
                            autoComplete="off"
                            placeholder="Paste key from your admin"
                            value={apiKeyInput}
                            onChange={(e) => setApiKeyInput(e.target.value)}
                          />
                        </label>
                        <button
                          type="submit"
                          className="btn btn-secondary btn-block btn-sm"
                          disabled={busy || !apiKeyInput.trim()}
                        >
                          {busy ? "Verifying…" : "Verify & Continue"}
                        </button>
                      </form>
                    )}
                  </div>
                </form>
              )}

              {/* ── Mode: Sign Up ── */}
              {mode === "signup" && (
                <form className="auth-form" onSubmit={handleSignUpSubmit}>
                  <p className="auth-form-intro">
                    Create your operator account. Passwords need at least 12 characters.
                  </p>
                  <div className="form-row-2">
                    <label className="auth-label">
                      <span>Full Name</span>
                      <input
                        type="text"
                        name="name"
                        autoComplete="name"
                        required
                        placeholder="Alex Mercer"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                      />
                    </label>

                    <label className="auth-label">
                      <span>Work Email</span>
                      <input
                        type="email"
                        name="email"
                        autoComplete="email"
                        required
                        placeholder="alex@acme.com"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                      />
                    </label>
                  </div>

                  <div className="form-row-2">
                    <label className="auth-label">
                      <span>Organization / Company</span>
                      <input
                        type="text"
                        name="company"
                        autoComplete="organization"
                        required
                        placeholder="Acme Mobility"
                        value={company}
                        onChange={(e) => setCompany(e.target.value)}
                      />
                    </label>

                    <label className="auth-label">
                      <span>Primary Role</span>
                      <select
                        value={role}
                        onChange={(e) => setRole(e.target.value)}
                        className="auth-select"
                      >
                        <option value="Operations Lead / Commander">Operations Lead / Commander</option>
                        <option value="Customer Experience Analyst">Customer Experience Analyst</option>
                        <option value="Quality & Compliance Auditor">Quality &amp; Compliance Auditor</option>
                        <option value="Platform & ML Engineer">Platform &amp; ML Engineer</option>
                        <option value="Executive / Risk Officer">Executive / Risk Officer</option>
                      </select>
                    </label>
                  </div>

                  <div className="form-row-2">
                    <label className="auth-label">
                      <span>Password</span>
                      <div className="password-input-wrap">
                        <input
                          type={showPassword ? "text" : "password"}
                          name="password"
                          autoComplete="new-password"
                          required
                          minLength={12}
                          placeholder="Min 12 characters"
                          value={password}
                          onChange={(e) => setPassword(e.target.value)}
                        />
                        <button
                          type="button"
                          className="pwd-toggle-btn"
                          onClick={() => setShowPassword(!showPassword)}
                          title={showPassword ? "Hide password" : "Show password"}
                          tabIndex="-1"
                        >
                          <EyeIcon show={showPassword} />
                        </button>
                      </div>
                    </label>

                    <label className="auth-label">
                      <span>Confirm Password</span>
                      <div className="password-input-wrap">
                        <input
                          type={showConfirmPassword ? "text" : "password"}
                          name="confirmPassword"
                          autoComplete="new-password"
                          required
                          placeholder="Repeat password"
                          value={confirmPassword}
                          onChange={(e) => setConfirmPassword(e.target.value)}
                        />
                        <button
                          type="button"
                          className="pwd-toggle-btn"
                          onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                          title={showConfirmPassword ? "Hide password" : "Show password"}
                          tabIndex="-1"
                        >
                          <EyeIcon show={showConfirmPassword} />
                        </button>
                      </div>
                    </label>
                  </div>

                  {/* Password Strength Indicator & Checklist */}
                  {password && (
                    <div className="pwd-strength-meter" aria-live="polite">
                      <div className="pwd-meter-head">
                        <span>Password Strength:</span>
                        <strong style={{ color: pwdStrength.color }}>{pwdStrength.label}</strong>
                      </div>
                      <div className="pwd-bars">
                        {[1, 2, 3, 4].map((step) => (
                          <div
                            key={step}
                            className="pwd-bar"
                            style={{
                              background:
                                pwdStrength.score >= step
                                  ? pwdStrength.color
                                  : "var(--edge)",
                            }}
                          />
                        ))}
                      </div>
                      <div className="pwd-checklist">
                        <span className={`pwd-check-item ${pwdStrength.checks?.hasLength ? "valid" : ""}`}>
                          {pwdStrength.checks?.hasLength ? "✓" : "•"} 12+ chars
                        </span>
                        <span className={`pwd-check-item ${pwdStrength.checks?.hasUpper ? "valid" : ""}`}>
                          {pwdStrength.checks?.hasUpper ? "✓" : "•"} Uppercase
                        </span>
                        <span className={`pwd-check-item ${pwdStrength.checks?.hasNumber ? "valid" : ""}`}>
                          {pwdStrength.checks?.hasNumber ? "✓" : "•"} Number
                        </span>
                        <span className={`pwd-check-item ${pwdStrength.checks?.hasSymbol ? "valid" : ""}`}>
                          {pwdStrength.checks?.hasSymbol ? "✓" : "•"} Symbol
                        </span>
                      </div>
                    </div>
                  )}

                  <div className="auth-options-row">
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={agreeTerms}
                        onChange={(e) => setAgreeTerms(e.target.checked)}
                        required
                      />
                      <span>
                        I agree to the <a href="#terms" onClick={(e) => e.preventDefault()}>Terms of Service</a> and{" "}
                        <a href="#privacy" onClick={(e) => e.preventDefault()}>Privacy Policy</a>
                      </span>
                    </label>
                  </div>

                  <button
                    type="submit"
                    className="btn btn-primary btn-block"
                    disabled={busy}
                  >
                    {busy ? "Creating Account…" : "Create Skew AI Account"}
                  </button>

                  <div className="auth-switch-prompt">
                    Already have an account?{" "}
                    <button
                      type="button"
                      className="link-btn"
                      onClick={() => switchMode("signin")}
                    >
                      Sign In
                    </button>
                  </div>
                </form>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}

