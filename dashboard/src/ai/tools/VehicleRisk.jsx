// [AI-LAYER] Per-vehicle fault risk: real complaint + recall counts behind every row.
import { useState } from "react";
import { aiGet } from "../aiApi.js";
import MetaTag from "../MetaTag.jsx";

const LEVEL_CLASS = { act_now: "sev-critical", high: "sev-critical", elevated: "sev-medium", watch: "", low: "" };
const LEVEL_LABEL = { act_now: "Act now", high: "High", elevated: "Elevated", watch: "Watch", low: "Low" };

export default function VehicleRisk() {
  const [f, setF] = useState({ make: "HONDA", model: "CR-V", year: "2019", mileage: "" });
  const [out, setOut] = useState(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  async function run(e) {
    e && e.preventDefault();
    setBusy(true); setErr("");
    const q = new URLSearchParams({ make: f.make, model: f.model, year: f.year });
    if (f.mileage) q.set("mileage", f.mileage);
    try { setOut(await aiGet(`/api/ai/vehicle-risk?${q}`)); } catch (ex) { setErr(ex.message); } finally { setBusy(false); }
  }
  const set = (k) => (e) => setF({ ...f, [k]: e.target.value });
  return (
    <div data-testid="vehicle-risk">
      <form onSubmit={run} className="row" style={{ gap: 8, flexWrap: "wrap", alignItems: "center" }}>
        <input aria-label="Make" value={f.make} onChange={set("make")} placeholder="Make" />
        <input aria-label="Model" value={f.model} onChange={set("model")} placeholder="Model" />
        <input aria-label="Year" value={f.year} onChange={set("year")} placeholder="Year" style={{ width: 80 }} />
        <input aria-label="Mileage" value={f.mileage} onChange={set("mileage")} placeholder="Mileage (optional)" style={{ width: 150 }} />
        <button className="primary" disabled={busy}>{busy ? "Checking…" : "Check fault risk"}</button>
      </form>
      {err && <div className="muted small">Failed ({err}).</div>}
      {out && (
        <div style={{ marginTop: 10 }}>
          {out.error && <div className="muted small">{out.error}</div>}
          <div className="small" style={{ marginBottom: 6 }}>
            <strong>{out.vehicle.year} {out.vehicle.make} {out.vehicle.model}</strong>
            {" · "}{out.summary ? `${out.summary.complaints_total} real NHTSA complaints in the warehouse` : ""}{" "}
            <MetaTag mode={out.powered_by && out.powered_by.length ? "enhanced" : "deterministic"} models={out.powered_by || []} />
          </div>
          {out.explanation && <div className="banner banner-ok small" style={{ marginBottom: 8 }}>{out.explanation} <span className="muted">(AI summary of the counts below)</span></div>}
          {out.vehicle.mileage != null && <div className="muted small">{out.mileage_note}</div>}
          <table className="table small">
            <thead><tr><th>Risk</th><th>Component</th><th>What to do</th><th>Data behind it</th></tr></thead>
            <tbody>
              {out.components.map((c) => (
                <tr key={c.component}>
                  <td><span className={`chip ${LEVEL_CLASS[c.level] || ""}`}>{LEVEL_LABEL[c.level] || c.level}</span></td>
                  <td>{c.component}{c.component_tier === "critical" ? " ⚠" : ""}</td>
                  <td>{c.advice}</td>
                  <td className="muted">
                    {c.data.complaints} complaints ({Math.round(c.data.share_of_vehicle_complaints * 100)}% of this vehicle's)
                    {c.data.relative_rate != null ? ` · ${c.data.relative_rate}× model-year norm` : ""}
                    {c.data.recall_count ? ` · ${c.data.recall_count} recall row(s)` : ""}
                    {c.data.hazard && c.data.hazard.records_with_hazard_mention ? ` · ${c.data.hazard.records_with_hazard_mention} mention crash/fire/injury` : ""}
                    {/* [AI-LAYER] fault codes */}
                    {c.data.cases_with_matching_fault_code != null ? ` · cases with matching fault code: ${c.data.cases_with_matching_fault_code}` : ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="muted small">{out.method}</div>
        </div>
      )}
    </div>
  );
}
