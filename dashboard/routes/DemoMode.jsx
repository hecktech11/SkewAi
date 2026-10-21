// [AI-LAYER] Hackathon demo path (#demo): the four key screens on one page.
// 1. Call in Hindi/Hinglish + photo  2. Case with evidence  3. Grounded brief
// 4. Safety-first early-warning board. Existing screens are untouched; every
// step links to its full page.
import { useEffect, useState } from "react";
import { aiGet, aiPostJson, aiStatus } from "../src/ai/aiApi.js";
import { SpeechLangSelect } from "../src/ai/CallAiExtras.jsx";
import EvidencePanel from "../src/ai/EvidencePanel.jsx";
import GroundedBriefPanel from "../src/ai/GroundedBriefPanel.jsx";
import MetaTag from "../src/ai/MetaTag.jsx";
import SafetyRiskBoard from "../src/ai/SafetyRiskBoard.jsx";

const LAST_SEED_KEY = "skewai.ai.lastDemoSeed";
const SAMPLE_HINGLISH = "Meri Honda CR-V 2019 ke brakes fail ho rahe hain, pedal neeche tak chala jata hai aur kal almost accident ho gaya.";

function readLastSeed() {
  try { return JSON.parse(window.localStorage.getItem(LAST_SEED_KEY) || "null"); } catch { return null; }
}

function Step({ n, title, children, link, linkLabel }) {
  return (
    <section className="card demo-step" style={{ padding: 14, marginBottom: 14 }} data-testid={`demo-step-${n}`}>
      <h2 style={{ marginTop: 0, display: "flex", gap: 10, alignItems: "baseline" }}>
        <span className="chip">{n}</span> {title}
        {link && <a href={link} className="muted small" style={{ marginLeft: "auto" }}>{linkLabel || "Open full screen"} →</a>}
      </h2>
      {children}
    </section>
  );
}

function NormalizeTry() {
  const [text, setText] = useState(SAMPLE_HINGLISH);
  const [out, setOut] = useState(null);
  const [busy, setBusy] = useState(false);
  async function run() {
    setBusy(true);
    try { setOut(await aiPostJson("/api/ai/language/normalize", { text })); }
    catch (e) { setOut({ error: e.message }); }
    finally { setBusy(false); }
  }
  return (
    <div style={{ marginTop: 8 }}>
      <textarea value={text} onChange={(e) => setText(e.target.value)} style={{ width: "100%", minHeight: 54 }}
        aria-label="Hindi or Hinglish complaint" />
      <button className="primary" onClick={run} disabled={busy || !text.trim()}>{busy ? "Normalising…" : "Detect + normalise to English"}</button>
      {out && !out.error && (
        <div className="small" style={{ marginTop: 6 }}>
          <span className="chip">detected: {out.lang}</span>{" "}
          <MetaTag mode={out.method === "meta_muse_spark" ? "enhanced" : "deterministic"} models={out.method === "meta_muse_spark" ? ["Muse Spark"] : []}
            fallbackReason={out.fallback_reason} />
          <div style={{ marginTop: 4 }}><strong>English:</strong> {out.normalized_en}</div>
          <div className="muted">Original kept alongside: {out.original}</div>
        </div>
      )}
      {out && out.error && <div className="muted small">Normalise failed ({out.error}).</div>}
    </div>
  );
}

// Hero moment: the SAME Hinglish words through the existing kill-switch lexicon,
// first as spoken, then after English normalisation (Muse Spark when a key is
// set). The deterministic lexicon makes the decision in both columns.
function HeroContrast({ text }) {
  const [out, setOut] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    let alive = true;
    aiPostJson("/api/ai/demo/safety-contrast", { text })
      .then((r) => { if (alive) setOut(r); })
      .catch((e) => { if (alive) setErr(e.message); });
    return () => { alive = false; };
  }, [text]);
  if (err) return <div className="muted small">Safety contrast unavailable ({err}).</div>;
  if (!out) return null;
  const hit = (t) => t
    ? <span className="chip sev-critical">kill-switch: "{t}" → Critical, "do not drive"</span>
    : <span className="chip">no kill-switch term matched</span>;
  return (
    <div className="row" style={{ gap: 12, marginTop: 10, flexWrap: "wrap" }} data-testid="hero-contrast">
      <div className="card" style={{ flex: "1 1 280px", padding: 10 }}>
        <div className="small"><strong>Rules only</strong> <MetaTag mode="deterministic" /></div>
        <div className="small muted" style={{ margin: "4px 0" }}>Existing lexicon on the caller's own words ({out.lang})</div>
        {hit(out.original_kill_switch)}
      </div>
      <div className="card" style={{ flex: "1 1 280px", padding: 10 }}>
        <div className="small"><strong>Meta AI enhanced</strong>{" "}
          <MetaTag mode={out.translated_by_meta ? "enhanced" : "deterministic"} models={out.translated_by_meta ? ["Muse Spark"] : []} fallbackReason={out.fallback_reason} />
        </div>
        {out.translated_by_meta ? (
          <>
            <div className="small" style={{ margin: "4px 0" }}><strong>English:</strong> {out.normalized_en}</div>
            {hit(out.normalized_kill_switch)}
          </>
        ) : (
          <div className="small muted" style={{ margin: "4px 0" }}>
            Not run: no Meta key. With a key, Muse Spark translates first and the SAME lexicon runs on the English
            (covered by a mocked-Meta test; not verified against the live API).
          </div>
        )}
      </div>
      <div className="muted small" style={{ flexBasis: "100%" }}>Decision: {out.decided_by}. AI can add a safety flag, never remove one.</div>
    </div>
  );
}

function CaseSummary({ caseId }) {
  const [c, setC] = useState(null);
  useEffect(() => {
    if (!caseId) return;
    aiGet(`/api/frontline/cases/${encodeURIComponent(caseId)}`).then(setC).catch(() => setC(null));
  }, [caseId]);
  if (!caseId) return <div className="muted small">Seed the scenario (or run a call) to get a case.</div>;
  if (!c) return <div className="muted small mono">{caseId}</div>;
  const flags = c.safety_flags && typeof c.safety_flags === "object" ? Object.keys(c.safety_flags).filter((f) => c.safety_flags[f]) : [];
  return (
    <div className="kvs small">
      <span className="k">case</span><span className="v mono">{c.case_id || caseId}</span>
      <span className="k">summary</span><span className="v">{c.description_summary || "—"}</span>
      <span className="k">component</span><span className="v">{c.category || "—"}</span>
      <span className="k">severity</span><span className="v">{c.severity || "—"} · P{c.priority} ({c.severity_source || "rules"})</span>
      <span className="k">safety flags</span><span className="v">{flags.length ? flags.join(", ") : "none"}</span>
      <span className="k">known issue</span><span className="v mono">{c.advisory_match_id || "—"}</span>
      <span className="k">investigation</span><span className="v mono">{c.investigation_id || "—"}</span>
    </div>
  );
}

export default function DemoMode({ hashQuery }) {
  const q = hashQuery || {};
  const [seed, setSeed] = useState(readLastSeed());
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const caseId = q.case || seed?.case_id || null;
  const invId = q.inv || seed?.investigation_id || null;

  useEffect(() => { aiStatus().then(setStatus); }, []);

  async function runSeed() {
    setBusy(true); setErr("");
    try {
      const out = await aiPostJson("/api/ai/demo/seed", {});
      setSeed(out);
      try { window.localStorage.setItem(LAST_SEED_KEY, JSON.stringify(out)); } catch { /* ignore */ }
    } catch (e) {
      setErr(e.message === "403" ? "Demo seeding is off on this server. Run `python -m scripts.ai_demo_seed` or start with AI_DEMO_ENABLED=1." : `Seed failed (${e.message})`);
    } finally { setBusy(false); }
  }

  return (
    <div className="demo-mode" style={{ maxWidth: 1100 }}>
      <header style={{ marginBottom: 14 }}>
        <h1 style={{ marginBottom: 4 }}>Catch vehicle faults before they cause crashes</h1>
        <p className="muted" style={{ marginTop: 0 }}>
          SkewAi records every customer complaint (voice, Hindi or English, with photos), links it to public NHTSA data,
          and flags safety defects early so manufacturers can inspect and repair before accidents happen.
        </p>
        <div className="row" style={{ gap: 8, flexWrap: "wrap", alignItems: "center" }}>
          {status && <MetaTag mode={status.mode} models={status.mode === "enhanced" ? [status.models?.text, status.models?.segmentation].filter(Boolean) : []} />}
          {status && status.mode !== "enhanced" && <span className="muted small">No Meta key: deterministic rules run every step; photo upload still works.</span>}
          <button className="primary" onClick={runSeed} disabled={busy} data-testid="demo-seed">{busy ? "Running scenario…" : "Run seeded India scenario"}</button>
          {seed && <span className="muted small">Last seed: <span className="mono">{seed.interaction_id}</span> · {seed.label}</span>}
        </div>
        {err && <div className="banner banner-warn small" style={{ marginTop: 8 }}>{err}</div>}
      </header>

      <Step n={1} title="A driver in Pune calls in Hinglish and sends a photo" link="#call" linkLabel="Open voice agent">
        <SpeechLangSelect />
        <NormalizeTry />
        <HeroContrast text={seed?.transcript?.find((t) => t.speaker === "customer")?.text || SAMPLE_HINGLISH} />
        {seed?.transcript && (
          <details style={{ marginTop: 8 }} open>
            <summary className="small">Seeded call transcript (synthetic, ran through the real pipeline)</summary>
            <ul className="small" style={{ paddingLeft: 18 }}>
              {seed.transcript.map((t, i) => (
                <li key={i} className={t.speaker === "agent" ? "muted" : ""}><strong>{t.speaker}:</strong> {t.text}</li>
              ))}
            </ul>
            {(seed.safety_flags || []).length > 0 && (
              <div className="chip sev-critical">Kill-switch fired: {seed.safety_flags.join(", ")} → "do not drive" script, safety specialist paged</div>
            )}
          </details>
        )}
      </Step>

      <Step n={2} title="The case, with photo evidence (SAM 3.1 overlay when enabled)" link="#cases" linkLabel="Open case queue">
        <CaseSummary caseId={caseId} />
        {caseId && <EvidencePanel caseId={caseId} />}
      </Step>

      <Step n={3} title="Grounded investigation brief: every claim cites real records" link={invId ? `#issue/${encodeURIComponent(invId)}` : "#warning"} linkLabel="Open issue">
        {invId ? <GroundedBriefPanel investigationId={invId} /> : <div className="muted small">No investigation linked yet: pick one from the board below.</div>}
      </Step>

      <Step n={4} title="Early-warning board, safety-critical first" link="#warning" linkLabel="Open early warning">
        <SafetyRiskBoard />
      </Step>
    </div>
  );
}
