// [AI-LAYER] Manufacturer safety alerts (pull feed; nothing is sent externally).
import { useEffect, useState } from "react";
import { aiGet, aiPostJson } from "../aiApi.js";
import MetaTag from "../MetaTag.jsx";

export default function Alerts() {
  const [d, setD] = useState(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const load = () => aiGet("/api/ai/alerts").then(setD).catch((e) => setMsg(e.message));
  useEffect(() => { load(); }, []);
  async function scan() {
    setBusy(true); setMsg("");
    try { const r = await aiPostJson("/api/ai/alerts/scan", {}); setMsg(`${r.created.length} new alert(s) from ${r.scanned} open investigations`); await load(); }
    catch (e) { setMsg(e.message); } finally { setBusy(false); }
  }
  async function ack(id) { try { await aiPostJson(`/api/ai/alerts/${id}/ack`, {}); await load(); } catch (e) { setMsg(e.message); } }
  return (
    <div data-testid="alerts">
      <div className="row" style={{ gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        <button className="primary" onClick={scan} disabled={busy}>{busy ? "Scanning…" : "Scan investigations for high-risk clusters"}</button>
        <a className="small" href="/api/ai/alerts/feed.json" target="_blank" rel="noreferrer">JSON feed</a>
        {d && <span className="muted small">Thresholds: high-risk ≥ {d.thresholds.high_min_cases} cases, elevated ≥ {d.thresholds.elevated_min_cases} cases · {d.delivery}</span>}
      </div>
      {msg && <div className="muted small" style={{ marginTop: 6 }}>{msg}</div>}
      {d && d.alerts.length === 0 && <div className="muted small" style={{ marginTop: 8 }}>No alerts yet.</div>}
      {d && d.alerts.map((a) => {
        const p = a.payload || {};
        return (
          <div key={a.alert_id} className="card" style={{ padding: 10, marginTop: 10 }}>
            <div className="row" style={{ gap: 8, alignItems: "center" }}>
              <span className={`chip ${a.level === "high" ? "sev-critical" : "sev-medium"}`}>{a.level}</span>
              <strong>{a.title || a.investigation_id}</strong>
              <span className="muted small">{a.case_count} cases · {a.created_at} · {a.status}</span>
              <MetaTag mode={(p.powered_by || []).length ? "enhanced" : "deterministic"} models={p.powered_by || []} />
              {a.status === "open" && <button className="ghost" style={{ marginLeft: "auto" }} onClick={() => ack(a.alert_id)}>Acknowledge</button>}
            </div>
            {p.headline && <div className="small" style={{ marginTop: 6 }}>{p.headline}</div>}
            {(p.recommended_actions || []).length > 0 && (
              <ol className="small">{p.recommended_actions.map((x, i) => <li key={i}>{x.action || x.text} <span className="muted">{x.source}</span></li>)}</ol>
            )}
            <a className="small" href={p.links?.issue}>Open issue →</a>
            <details className="small"><summary>Webhook payload preview (not sent)</summary><pre style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(p, null, 2)}</pre></details>
          </div>
        );
      })}
    </div>
  );
}
