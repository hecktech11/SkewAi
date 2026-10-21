// [AI-LAYER] Fault spread by region (NHTSA state) and over time. Plain SVG, raw counts.
import { useState } from "react";
import { aiGet } from "../aiApi.js";

function Bars({ rows }) {
  const max = Math.max(1, ...rows.map((r) => r.count));
  const h = 18;
  return (
    <svg width="100%" viewBox={`0 0 420 ${rows.length * h + 4}`} role="img" aria-label="complaints by region">
      {rows.map((r, i) => (
        <g key={r.region} transform={`translate(0,${i * h})`}>
          <text x="0" y="13" fontSize="11" fill="currentColor">{r.region}</text>
          <rect x="34" y="3" height="12" width={(r.count / max) * 320} fill="#d9534f" opacity="0.8" />
          <text x={40 + (r.count / max) * 320} y="13" fontSize="10" fill="currentColor">{r.count}</text>
        </g>
      ))}
    </svg>
  );
}

function Line({ rows }) {
  if (rows.length < 2) return <div className="muted small">Not enough months to plot.</div>;
  const W = 420, H = 140, max = Math.max(1, ...rows.map((r) => r.count));
  const x = (i) => 30 + (i / (rows.length - 1)) * (W - 40);
  const y = (v) => H - 20 - (v / max) * (H - 35);
  const pts = rows.map((r, i) => `${x(i)},${y(r.count)}`).join(" ");
  const rpts = rows.map((r, i) => `${x(i)},${y((r.regions / Math.max(1, ...rows.map((q) => q.regions))) * max)}`).join(" ");
  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="complaints per month">
      <polyline points={pts} fill="none" stroke="#d9534f" strokeWidth="2" />
      <polyline points={rpts} fill="none" stroke="#5b8def" strokeWidth="1.5" strokeDasharray="4 3" />
      <text x="30" y="12" fontSize="10" fill="currentColor">max {max}/month · red = complaints, blue dashed = regions reporting (scaled)</text>
      <text x="30" y={H - 4} fontSize="10" fill="currentColor">{rows[0].month}</text>
      <text x={W - 50} y={H - 4} fontSize="10" fill="currentColor">{rows[rows.length - 1].month}</text>
    </svg>
  );
}

function Heat({ heat }) {
  const max = Math.max(1, ...heat.rows.flatMap((r) => r.counts));
  return (
    <table className="small" style={{ borderCollapse: "collapse" }}>
      <thead><tr><th /> {heat.years.map((y) => <th key={y} style={{ padding: "0 4px" }}>{y.slice(2)}</th>)}</tr></thead>
      <tbody>
        {heat.rows.map((r) => (
          <tr key={r.region}><td>{r.region}</td>
            {r.counts.map((c, i) => <td key={i} title={`${r.region} ${heat.years[i]}: ${c}`} style={{ width: 22, height: 16, background: `rgba(217,83,79,${c ? 0.15 + 0.85 * (c / max) : 0})` }} />)}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function Spread() {
  const [f, setF] = useState({ make: "FORD", model: "", component: "ELECTRICAL", year_min: "", year_max: "" });
  const [d, setD] = useState(null);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  async function run(e) {
    e?.preventDefault(); setBusy(true); setErr("");
    const q = new URLSearchParams(Object.entries(f).filter(([, v]) => String(v).trim()));
    try { setD(await aiGet(`/api/ai/spread?${q}`)); } catch (x) { setErr(x.message); } finally { setBusy(false); }
  }
  const inp = (k, ph, w = 110) => <input value={f[k]} placeholder={ph} style={{ width: w }} onChange={(e) => setF({ ...f, [k]: e.target.value })} />;
  return (
    <div data-testid="spread">
      <form onSubmit={run} className="row" style={{ gap: 6, flexWrap: "wrap" }}>
        {inp("make", "make")}{inp("model", "model")}{inp("component", "component contains", 150)}{inp("year_min", "MY from", 70)}{inp("year_max", "MY to", 70)}
        <button className="primary" disabled={busy}>{busy ? "Loading…" : "Show spread"}</button>
      </form>
      {err && <div className="muted small">{err}</div>}
      {d && (
        <div style={{ marginTop: 10 }}>
          <div className="small"><strong>{d.total_complaints.toLocaleString()}</strong> complaints from <strong>{d.regions_reporting}</strong> regions. <span className="muted">{d.note}</span></div>
          <div className="row" style={{ gap: 16, flexWrap: "wrap", alignItems: "flex-start", marginTop: 8 }}>
            <div style={{ flex: "1 1 360px" }}><h4>By region (top {d.by_region.length}{d.other_regions_count ? `; ${d.other_regions_count} more elsewhere` : ""})</h4><Bars rows={d.by_region} /></div>
            <div style={{ flex: "1 1 360px" }}><h4>Per month (last {d.by_month.length})</h4><Line rows={d.by_month} />
              <h4>Region × year</h4><Heat heat={d.region_year_heat} />
              <div className="muted small">First regions to report: {d.first_seen_order.slice(0, 6).map((r) => `${r.region} (${r.first_seen})`).join(", ")}</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
