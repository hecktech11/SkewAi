// [AI-LAYER] Early-warning investigations ordered by safety risk, plus the
// committed NHTSA backtest headline (real numbers or nothing).
import { useEffect, useState } from "react";
import { aiGet } from "./aiApi.js";
import SafetyRiskChip from "./SafetyRiskChip.jsx";

export function BacktestHero({ h }) {
  if (!h) return null;
  const hero = h.hero;
  const c = h.campaign_level || {};
  return (
    <div className="banner banner-ok ai-hero" data-testid="backtest-hero" style={{ marginBottom: 10 }}>
      <strong>NHTSA backtest (2015-2024, 5 makes):</strong>{" "}
      {c.campaigns ? <>{c.hits} of {c.campaigns} recall campaigns flagged early (median lead {c.lead_weeks_median} weeks)</>
        : <>{h.hits} of {h.recalls_scored} recall scope rows flagged early (median lead {h.lead_weeks_median} weeks)</>}
      {h.metric ? <span className="muted"> ({h.metric})</span> : null}.
      {" "}In the windows between flag and recall, NHTSA received{" "}
      {h.preventable_window_crash} crash-, {h.preventable_window_fire} fire- and {h.preventable_window_injury_complaints} injury-linked
      complaints on the flagged slices.
      {hero && (
        <> Largest: {hero.scope_entity_2} {hero.scope_category} flagged from {hero.first_flagged_week}, {hero.lead_weeks} weeks before recall {hero.advisory_id}.</>
      )}
      <div className="muted small">Correlation from self-reported complaints, not proof that a crash would have been prevented. Source: {h.source}</div>
    </div>
  );
}

export default function SafetyRiskBoard({ limit = 12 }) {
  const [d, setD] = useState(null);
  useEffect(() => {
    let live = true;
    aiGet(`/api/ai/early-warning/safety?limit=${limit}`).then((v) => live && setD(v)).catch(() => live && setD({ investigations: [], error: true }));
    return () => { live = false; };
  }, [limit]);
  if (!d) return null;
  const rows = d.investigations || [];
  return (
    <section className="ai-safety-board card" data-testid="safety-board" style={{ marginBottom: 14, padding: 12 }}>
      <h3 className="analytics-h">Safety-first queue <span className="muted small">(brakes, steering, air bags, fuel/fire first)</span></h3>
      <BacktestHero h={d.backtest_headline} />
      {rows.length === 0 && <div className="muted small">No open investigations yet.</div>}
      {rows.length > 0 && (
        <table className="table small">
          <thead><tr><th>Risk</th><th>Investigation</th><th>Slice</th><th>Cases</th><th>Crash / fire / injury mentions</th></tr></thead>
          <tbody>
            {rows.map((r) => {
              const m = (r.safety_risk && r.safety_risk.mentions) || {};
              return (
                <tr key={r.investigation_id}>
                  <td><SafetyRiskChip risk={r.safety_risk} /></td>
                  <td><a className="mono" href={`#issue/${encodeURIComponent(r.investigation_id)}`}>{r.investigation_id}</a></td>
                  <td>{[r.slice_entity_2, r.slice_entity_3, r.slice_category].filter(Boolean).join(" ")}</td>
                  <td>{r.case_count}</td>
                  <td>{m.crash || 0} / {m.fire || 0} / {m.injury || 0}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      <div className="muted small">{d.method}</div>
    </section>
  );
}
