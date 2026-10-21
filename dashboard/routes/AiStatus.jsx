// [AI-LAYER] AI status page (#ai-status): mode, Meta call health per call type,
// and supervisor review of AI severity raises. A verdict never changes a case's
// severity; with no key every call count reads 0 / "not run".
import { useEffect, useState } from "react";
import { aiGet, aiPostJson } from "../src/ai/aiApi.js";

function fmt(v) { return v === null || v === undefined ? "–" : String(v); }

function CallHealth({ h }) {
  if (!h) return null;
  if (h.status === "unavailable") return <div className="muted small">Call health unavailable ({h.error}).</div>;
  const rows = Object.entries(h.by_type || {});
  return (
    <div data-testid="ai-call-health">
      <div className="small" style={{ marginBottom: 6 }}>
        <strong>Meta calls: {fmt(h.totals?.calls)}</strong>{" "}
        <span className="muted">· {h.status === "not run" ? "not run" : `${fmt(h.totals?.failed)} failed · ${fmt(h.totals?.fallback_to_rules)} fell back to rules`}
        {" "}· caller fallback events {fmt(h.totals?.caller_fallback_events)}</span>
      </div>
      {h.status === "not run" && <div className="muted small" style={{ marginBottom: 6 }}>{h.note}</div>}
      <table className="table small">
        <thead><tr><th>Call type</th><th>Calls</th><th>OK</th><th>Failed</th><th>Fallback to rules</th><th>Timeouts</th><th>p50 ms</th><th>p95 ms</th><th>Last error</th></tr></thead>
        <tbody>
          {rows.map(([t, r]) => (
            <tr key={t}>
              <td>{t}</td>
              <td>{r.calls}{r.calls === 0 ? <span className="muted"> · not run</span> : null}</td>
              <td>{r.ok}</td><td>{r.failed}</td><td>{r.fallback_to_rules}</td><td>{r.timeouts}</td>
              <td>{fmt(r.latency_ms?.p50)}</td><td>{fmt(r.latency_ms?.p95)}</td><td className="muted">{fmt(r.last_error)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ReviewStats({ s }) {
  if (!s) return null;
  return (
    <div className="small" data-testid="ai-raise-stats">
      <strong>{s.n_label}</strong> reviewed of {s.ai_raises} AI raises · accepted {s.accepted} · rejected {s.rejected} · pending {s.pending}
      <div className="muted">
        {s.agreement_rate === null
          ? `Agreement rate: ${s.status} (a rate is shown only from n ≥ ${s.min_reviews_for_rate}).`
          : `Agreement rate ${s.agreement_rate} (95% CI ${s.ci95?.[0]}–${s.ci95?.[1]}).`}
      </div>
    </div>
  );
}

export default function AiStatus() {
  const [o, setO] = useState(null);
  const [raises, setRaises] = useState(null);
  const [msg, setMsg] = useState("");
  const [notes, setNotes] = useState({});
  const load = () => {
    aiGet("/api/ai/status/overview").then(setO).catch((e) => setMsg(`status: ${e.message}`));
    aiGet("/api/ai/raises").then(setRaises).catch((e) => setMsg(`raises: ${e.message}`));
  };
  useEffect(() => { load(); }, []);
  async function review(id, verdict) {
    setMsg("");
    try {
      await aiPostJson(`/api/ai/raises/${id}/review`, { verdict, note: notes[id] || null });
      setMsg(`Recorded ${verdict}. Case severity unchanged.`);
      load();
    } catch (e) {
      setMsg(e.message === "403" ? "Only a supervisor session can review AI raises (403)." : `Review failed: ${e.message}`);
    }
  }
  const st = o?.status;
  return (
    <div style={{ maxWidth: 1150 }} data-testid="ai-status-page">
      <h1 style={{ marginBottom: 4 }}>AI status</h1>
      <p className="muted" style={{ marginTop: 0 }}>
        Mode: <strong>{st ? st.mode : "…"}</strong>{st && st.mode !== "enhanced" ? " (no Meta key: the deterministic rules path is running)" : ""}.
        AI may only raise severity; supervisor verdicts below are recorded and audited but never change a case.
      </p>
      {msg && <div className="muted small" style={{ marginBottom: 8 }}>{msg}</div>}
      <section className="card" style={{ padding: 14, marginBottom: 12 }}>
        <h3 style={{ marginTop: 0 }}>Meta call health</h3>
        <CallHealth h={o?.call_health} />
      </section>
      <section className="card" style={{ padding: 14 }}>
        <h3 style={{ marginTop: 0 }}>Review of AI severity raises</h3>
        <ReviewStats s={o?.raise_reviews} />
        {raises && raises.count === 0 && <div className="muted small" style={{ marginTop: 8 }}>No AI severity raises recorded.</div>}
        {raises && raises.raises.map((r) => (
          <div key={r.event_id} className="card" style={{ padding: 10, marginTop: 10 }}>
            <div className="small">
              <strong>{r.rule_severity} → {r.final_severity}</strong> <span className="muted">(AI proposed {r.ai_severity}; {r.created_at})</span>
              {r.case && <span className="muted"> · case {r.case.case_id} severity {r.case.severity}</span>}
            </div>
            {r.ai_reason && <div className="small">{r.ai_reason}</div>}
            {r.review ? (
              <div className="small muted">Verdict: {r.review.verdict} by {r.review.reviewer}{r.review.note ? ` · “${r.review.note}”` : ""}</div>
            ) : (
              <div className="row" style={{ gap: 6, marginTop: 6, alignItems: "center" }}>
                <input placeholder="Optional note" value={notes[r.event_id] || ""} onChange={(e) => setNotes({ ...notes, [r.event_id]: e.target.value })} />
                <button className="primary" onClick={() => review(r.event_id, "accept")}>Accept</button>
                <button className="ghost" onClick={() => review(r.event_id, "reject")}>Reject</button>
              </div>
            )}
          </div>
        ))}
      </section>
    </div>
  );
}
