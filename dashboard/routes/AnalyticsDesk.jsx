import { useEffect, useRef, useState } from "react";
import { apiHeaders } from "../src/apiAuth.js";
import { AreaChart, BarList, DonutChart, Gauge, HeatMatrix, StackBar } from "../src/ui/Chart.jsx";
import { renderMarkdown } from "./AuditReports.jsx";

const WINDOWS = [1, 7, 30];
const COHORT_FIELDS = ["entity_3", "entity_1", "entity_2", "category", "status", "pack_id"];

function toneForSeverity(sev) {
  const s = String(sev || "").toLowerCase();
  if (s.startsWith("crit")) return "danger";
  if (s.startsWith("med")) return "warn";
  if (s.startsWith("low")) return "ok";
  return "accent";
}

function fmtPct(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  return `${Math.round(Number(x) * 100)}%`;
}

function fmtUSD(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return "—";
  return `$${Math.round(Number(x)).toLocaleString()}`;
}

function isoDay(offsetBack) {
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  d.setDate(d.getDate() - offsetBack);
  return d.toISOString().slice(0, 10);
}

function bucketByDay(rows, days, getDate) {
  const out = [];
  for (let i = days - 1; i >= 0; i--) {
    const key = isoDay(i);
    out.push({ key, label: key.slice(5), value: 0 });
  }
  const map = Object.fromEntries(out.map((o) => [o.key, o]));
  for (const r of rows || []) {
    const k = String(getDate(r) || "").slice(0, 10);
    if (map[k]) map[k].value += 1;
  }
  return out;
}

export default function AnalyticsDesk({ refreshKey }) {
  const [windowDays, setWindowDays] = useState(7);
  const [includeSim, setIncludeSim] = useState(false);
  const [cohortField, setCohortField] = useState("entity_3");
  const [bundle, setBundle] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [audit, setAudit] = useState(null);
  const [auditLoading, setAuditLoading] = useState(false);
  const [showDigest, setShowDigest] = useState(false);

  // ── Primary fetch (fast endpoints) ──────────────────────────
  // StrictMode-safe: each run gets an AbortController; cleanup aborts the
  // stale run so double-invoked effects can't leave requests hanging around,
  // and only the latest run writes state (run-id guard).
  const runRef = useRef(0);
  useEffect(() => {
    const ctrl = new AbortController();
    runRef.current += 1;
    const runId = runRef.current;
    const alive = () => runId === runRef.current && !ctrl.signal.aborted;
    async function load() {
      setLoading(true);
      setError(null);
      setBundle(null);
      const sim = includeSim ? "&include_simulated=true" : "";
      // Aborts resolve to null (never reject) so cleanup aborts stay silent.
      // Genuine HTTP/network failures still reject and count as failed.
      const get = (url) =>
        fetch(url, { headers: apiHeaders(), signal: ctrl.signal }).then(
          (r) => {
            if (!r.ok) throw new Error(`${url} → HTTP ${r.status}`);
            return r.json();
          },
          (e) => {
            if (ctrl.signal.aborted) return null;
            throw e;
          },
        );
      // Small sequential batches: kind to the browser connection pool and
      // to the single-process dev API, and each batch paints progressively.
      const batches = [
        [
          get(`/api/frontline/metrics?window_days=${windowDays}${sim}`),
          get(`/api/frontline/early-warning?window_days=${windowDays}${sim}`),
          get(`/api/frontline/insights/csat?window_days=${windowDays}${sim}`),
          get(`/api/frontline/insights/product-gap?window_days=${windowDays}&limit=8${sim}`),
          get(`/api/frontline/wallboard`),
        ],
        [
          get(`/api/frontline/cases?limit=200${sim}`),
          get(`/api/frontline/digest?window_days=${windowDays}`),
          get(`/api/frontline/analytics/forecast`),
          get(`/api/frontline/analytics/severity-drift`),
          get(`/api/frontline/analytics/cohorts?group_field=${cohortField}`),
        ],
        [
          get(`/api/frontline/analytics/cross-pack`),
          get(`/api/frontline/investigations`),
          get(`/api/frontline/analytics/financial-impact`),
          get(`/api/frontline/analytics/fairness`),
          get(`/api/frontline/analytics/regulator-watch`),
        ],
      ];
      const acc = {};
      const keys = ["metrics", "early", "csat", "gaps", "wall",
        "cases", "digest", "forecast", "drift", "cohorts",
        "crosspack", "invs", "financial", "fairness", "regulator"];
      let failed = 0;
      let ki = 0;
      for (const batch of batches) {
        const settled = await Promise.allSettled(batch);
        if (!alive()) return;
        settled.forEach((s, bi) => {
          acc[keys[ki + bi]] = s.status === "fulfilled" ? s.value : null;
          if (s.status !== "fulfilled") failed += 1;
        });
        ki += batch.length;
        setBundle({ ...acc });
      }
      if (!alive()) return;
      if (failed === keys.length) {
        setError("Telemetry endpoints are unreachable. Is the API running?");
      }
      setLoading(false);
    }
    load();
    return () => {
      ctrl.abort();
    };
  }, [windowDays, includeSim, cohortField, refreshKey]);

  // ── Secondary fetch (audit aggregates — slow, streams in later) ──
  useEffect(() => {
    const ctrl = new AbortController();
    let cancelled = false;
    setAudit(null);
    setAuditLoading(true);
    const sim = includeSim ? "&include_simulated=true" : "";
    const start = isoDay(windowDays - 1);
    const end = isoDay(0);
    fetch(`/api/frontline/audits/export?limit=25&start=${start}&end=${end}${sim}`, {
      headers: apiHeaders(),
      signal: ctrl.signal,
    })
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(
        (d) => {
          if (!cancelled) setAudit(d);
        },
        () => {
          // Aborted on cleanup/unmount — stay silent.
        },
      )
      .catch(() => {
        if (!cancelled) setAudit({ error: true });
      })
      .finally(() => {
        if (!cancelled) setAuditLoading(false);
      });
    return () => {
      cancelled = true;
      ctrl.abort();
    };
  }, [windowDays, includeSim, refreshKey]);

  if (loading && !bundle) return <div className="empty">Loading analytics…</div>;
  if (error && !bundle) return <div className="empty err-text">{error}</div>;
  const { metrics, early, csat, gaps, wall, cases, digest,
    forecast, drift, cohorts, crosspack, invs, financial, fairness, regulator } = bundle || {};

  // ── Aggregates ──────────────────────────────────────────────
  const funnel = early?.funnel || {};
  const funnelBars = [
    { label: "Started", value: funnel.started || 0 },
    { label: "Completed", value: funnel.completed || 0, tone: "ok" },
    { label: "Abandoned", value: funnel.abandoned || 0, tone: "warn" },
    { label: "Escalated", value: funnel.escalated || 0, tone: "danger" },
    { label: "Takeovers", value: funnel.takeovers || 0 },
    { label: "Advisories notified", value: funnel.advisories_notified || 0 },
    { label: "Cases created", value: funnel.cases_created || 0 },
  ];

  const outcomeMix = csat?.outcome_mix || {};
  const outcomeBars = Object.entries(outcomeMix)
    .map(([label, value]) => ({ label, value: Number(value) || 0 }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 8);

  const caseRows = cases?.cases || [];
  const sevCount = {};
  const statusCount = {};
  const catCount = {};
  for (const c of caseRows) {
    sevCount[c.severity || "Unknown"] = (sevCount[c.severity || "Unknown"] || 0) + 1;
    statusCount[c.status || "Unknown"] = (statusCount[c.status || "Unknown"] || 0) + 1;
    if (c.category) catCount[c.category] = (catCount[c.category] || 0) + 1;
  }
  const sevSegs = Object.entries(sevCount).map(([label, value]) => ({
    label,
    value,
    tone: toneForSeverity(label),
  }));
  const statusSegs = Object.entries(statusCount).map(([label, value]) => ({ label, value }));
  const catBars = Object.entries(catCount)
    .map(([label, value]) => ({ label, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 8);

  // Severity × status matrix
  const sevKeys = Object.keys(sevCount).sort();
  const statusKeys = Object.keys(statusCount).sort();
  const matrix = {};
  for (const c of caseRows) {
    const k = `${c.severity || "Unknown"}|${c.status || "Unknown"}`;
    matrix[k] = (matrix[k] || 0) + 1;
  }

  const casesPerDay = bucketByDay(caseRows, windowDays, (c) => c.created_at);

  const clusters = (wall?.top_risk_clusters || []).map((c) => ({
    label: `#${c.cluster_id} · ${c.pack_id || ""}`.trim(),
    value: c.open_cases || 0,
    tone: toneForSeverity(c.max_severity),
  }));

  const topIssues = (gaps?.top_issues || []).map((g) => ({
    label: g.category || g.issue_key || "uncategorized",
    value: g.volume || 0,
    tone: g.getting_worse ? "danger" : toneForSeverity(g.severity_mix && Object.entries(g.severity_mix).sort((a, b) => b[1] - a[1])[0]?.[0]),
  }));
  const rising = gaps?.rising_issues || [];

  const kpis = [
    { label: "Open cases", value: wall?.open_cases ?? metrics?.cases?.open ?? "—", tone: "accent" },
    { label: "Critical open", value: wall?.critical_open ?? metrics?.cases?.critical_open ?? "—", tone: "danger" },
    { label: "Completed contacts", value: wall?.completed_contacts ?? "—", tone: "ok" },
    { label: "Satisfaction proxy", value: csat ? fmtPct(csat.satisfaction_proxy) : "—", tone: "ok" },
    { label: "Active now", value: wall?.active_contacts ?? early ? 0 : "—", tone: "accent" },
    { label: "Open investigations", value: metrics?.investigations?.open ?? "—", tone: "warn" },
    { label: "Portfolio risk", value: financial ? fmtUSD(financial.portfolio_risk_usd) : "—", tone: "danger" },
    { label: "Contacts (csat window)", value: csat?.contact_count ?? "—", tone: "accent" },
  ];

  // Investigations
  const invRows = invs?.investigations || [];
  const aging = { fresh: 0, week: 0, stale: 0 };
  for (const inv of invRows) {
    const d = Number(inv.days_open) || 0;
    if (d <= 1) aging.fresh += 1;
    else if (d <= 7) aging.week += 1;
    else aging.stale += 1;
  }
  const agingSegs = [
    { label: "0–1d", value: aging.fresh, tone: "ok" },
    { label: "2–7d", value: aging.week, tone: "warn" },
    { label: "8d+", value: aging.stale, tone: "danger" },
  ];
  const invTop = [...invRows].sort((a, b) => (b.days_open || 0) - (a.days_open || 0)).slice(0, 8);

  // Forecast
  const forecasts = forecast?.forecasts || [];
  const forecastVol = [...forecasts]
    .sort((a, b) => (b.last_week_volume || 0) - (a.last_week_volume || 0))
    .slice(0, 8)
    .map((f) => ({ label: `#${f.cluster_id}`, value: f.last_week_volume || 0 }));

  // Drift
  const driftRows = drift?.clusters || [];

  // Cohorts
  const cohortBars = ((cohorts?.cohorts || []).map((c) => ({
    label: String(c.cohort ?? "unknown"),
    value: c.volume || 0,
  }))).sort((a, b) => b.value - a.value).slice(0, 10);

  // Cross-pack
  const packSpecific = crosspack?.pack_specific || [];
  const crossPatterns = crosspack?.patterns || [];

  // Financial
  const estimates = financial?.estimates || [];

  // Fairness
  const fairGroups = fairness?.groups || [];

  // Regulator
  const regMatches = regulator?.matches || [];

  // ── Audit deep-dive (secondary payload) ─────────────────────
  const axRows = audit?.interactions || [];
  const verdictCount = {};
  let groundedSum = 0;
  let groundedTotal = 0;
  let supervised = 0;
  const frustBuckets = [0, 0, 0, 0, 0];
  const channelCount = {};
  const hourCount = new Array(24).fill(0);
  for (const r of axRows) {
    const v = r.audit?.overall_verdict || "unknown";
    verdictCount[v] = (verdictCount[v] || 0) + 1;
    groundedSum += Number(r.audit?.grounded_actions) || 0;
    groundedTotal += Number(r.audit?.total_actions) || 0;
    if (r.supervised) supervised += 1;
    const f = Number(r.peak_frustration);
    if (!Number.isNaN(f)) frustBuckets[Math.min(4, Math.floor(f * 5))] += 1;
    channelCount[r.channel || "unknown"] = (channelCount[r.channel || "unknown"] || 0) + 1;
    const h = new Date(r.started_at).getHours();
    if (!Number.isNaN(h)) hourCount[h] += 1;
  }
  const verdictDonut = Object.entries(verdictCount).map(([label, value]) => ({
    label,
    value,
    tone: /mismatch/i.test(label) ? "danger" : /unverif|error/i.test(label) ? "warn" : "ok",
  }));
  const frustBars = ["0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"].map((label, i) => ({
    label,
    value: frustBuckets[i],
    tone: i >= 3 ? "danger" : i === 2 ? "warn" : "ok",
  }));
  const channelSegs = Object.entries(channelCount).map(([label, value]) => ({ label, value }));
  const hourBars = hourCount.map((value, h) => ({
    label: String(h).padStart(2, "0"),
    value,
  }));

  return (
    <div>
      <header className="page-header">
        <div>
          <h1>Analytics</h1>
          <p className="sub">
            Telemetry and data analysis across contacts, cases, audits and early warning.
          </p>
        </div>
        <div className="row" style={{ gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <label className="check-row" style={{ fontSize: 12 }}>
            <input
              type="checkbox"
              checked={includeSim}
              onChange={(e) => setIncludeSim(e.target.checked)}
            />
            <span className="muted">Include test traffic</span>
          </label>
          <div className="tabs" role="tablist" aria-label="Analytics window" style={{ margin: 0 }}>
            {WINDOWS.map((w) => (
              <button
                key={w}
                type="button"
                role="tab"
                aria-selected={windowDays === w}
                className={"tab" + (windowDays === w ? " active" : "")}
                onClick={() => setWindowDays(w)}
              >
                {w}d
              </button>
            ))}
          </div>
        </div>
      </header>

      {loading && <div className="empty">Refreshing…</div>}

      {/* ── KPI cards ── */}
      <div className="analytics-kpis">
        {kpis.map((k) => (
          <div key={k.label} className="analytics-kpi">
            <span className="analytics-kpi-label">{k.label}</span>
            <span className={`analytics-kpi-value tone-${k.tone}`}>{k.value}</span>
          </div>
        ))}
      </div>

      {/* ── Volume trend ── */}
      <div className="panel" style={{ marginTop: 18 }}>
        <h3 className="analytics-h">Cases per day · last {windowDays}d (total {caseRows.length} in sample)</h3>
        <AreaChart data={casesPerDay} label={`Cases per day, total ${caseRows.length}`} />
      </div>

      <div className="analytics-grid2" style={{ marginTop: 18 }}>
        {/* ── Funnel ── */}
        <div className="panel">
          <h3 className="analytics-h">Contact funnel · {windowDays}d</h3>
          <BarList data={funnelBars} emptyText="No contacts in window" />
        </div>

        {/* ── Satisfaction ── */}
        <div className="panel">
          <h3 className="analytics-h">Satisfaction proxy · {windowDays}d</h3>
          {csat ? (
            <div className="row" style={{ gap: 16, alignItems: "center", flexWrap: "wrap" }}>
              <Gauge value={csat.satisfaction_proxy || 0} label="satisfaction" tone="ok" />
              <ul className="analytics-facts">
                <li><span className="muted">Contacts</span> <strong className="mono">{csat.contact_count ?? "—"}</strong></li>
                <li><span className="muted">Avg peak frustration</span> <strong className="mono">{csat.avg_peak_frustration ?? "—"}</strong></li>
                <li><span className="muted">Angry contacts</span> <strong className="mono">{csat.angry_contacts ?? "—"}</strong></li>
                <li><span className="muted">Safety-flagged</span> <strong className="mono">{csat.safety_flagged_cases ?? "—"}</strong></li>
              </ul>
            </div>
          ) : (
            <div className="chart-empty">No satisfaction data</div>
          )}
        </div>
      </div>

      <div className="analytics-grid2" style={{ marginTop: 18 }}>
        {/* ── Severity / status ── */}
        <div className="panel">
          <h3 className="analytics-h">Case severity mix ({caseRows.length} cases)</h3>
          <StackBar segments={sevSegs} label="Case severity distribution" />
          <h3 className="analytics-h" style={{ marginTop: 14 }}>Case status mix</h3>
          <StackBar segments={statusSegs} label="Case status distribution" />
        </div>

        {/* ── Risk clusters ── */}
        <div className="panel">
          <h3 className="analytics-h">Top risk clusters</h3>
          <BarList data={clusters} emptyText="No live cluster risk" />
        </div>
      </div>

      {/* ── Severity × status matrix ── */}
      {sevKeys.length > 0 && statusKeys.length > 0 && (
        <div className="panel" style={{ marginTop: 18 }}>
          <h3 className="analytics-h">Severity × status matrix</h3>
          <HeatMatrix
            rows={sevKeys.map((k) => ({ key: k, label: k }))}
            columns={statusKeys.map((k) => ({ key: k, label: k }))}
            getValue={(sev, st) => matrix[`${sev}|${st}`] || 0}
            label="Case count by severity and status"
          />
        </div>
      )}

      {/* ── Audit deep-dive (streams in separately) ── */}
      <div className="panel" style={{ marginTop: 18 }}>
        <h3 className="analytics-h">
          Audit deep-dive · {axRows.length > 0 ? `${axRows.length} sampled contacts` : "sampling contacts"}
          {auditLoading && <span className="muted"> (loading…)</span>}
        </h3>
        {audit?.error && <div className="chart-empty">Audit aggregates unavailable</div>}
        {!audit?.error && axRows.length === 0 && !auditLoading && (
          <div className="chart-empty">No sampled contacts in window</div>
        )}
        {axRows.length > 0 && (
          <>
            <div className="analytics-grid2">
              <div>
                <h3 className="analytics-h">Audit verdicts</h3>
                <DonutChart data={verdictDonut} label="Audit verdict share" />
                <p className="muted" style={{ fontSize: 12, marginTop: 8 }}>
                  Grounded actions {groundedSum}/{groundedTotal}
                  {groundedTotal > 0 && ` (${Math.round((groundedSum / groundedTotal) * 100)}%)`}
                  {" · "}supervised {supervised}/{axRows.length}
                </p>
              </div>
              <div>
                <h3 className="analytics-h">Peak-frustration histogram</h3>
                <BarList data={frustBars} emptyText="No frustration data" />
              </div>
            </div>
            <div className="analytics-grid2" style={{ marginTop: 14 }}>
              <div>
                <h3 className="analytics-h">Channel mix</h3>
                <StackBar segments={channelSegs} label="Contact channel mix" />
              </div>
              <div>
                <h3 className="analytics-h">Contacts by hour (UTC)</h3>
                <BarList data={hourBars} emptyText="No hourly data" />
              </div>
            </div>
          </>
        )}
      </div>

      <div className="analytics-grid2" style={{ marginTop: 18 }}>
        {/* ── Outcome mix ── */}
        <div className="panel">
          <h3 className="analytics-h">Contact outcomes · {windowDays}d</h3>
          <BarList data={outcomeBars} emptyText="No outcomes in window" />
        </div>

        {/* ── Categories ── */}
        <div className="panel">
          <h3 className="analytics-h">Top case categories</h3>
          <BarList data={catBars} emptyText="No categorized cases" />
        </div>
      </div>

      {/* ── Product gaps ── */}
      <div className="panel" style={{ marginTop: 18 }}>
        <h3 className="analytics-h">Product gaps & rising issues · {windowDays}d</h3>
        {topIssues.length === 0 && <div className="chart-empty">No product gaps detected</div>}
        {topIssues.length > 0 && <BarList data={topIssues} />}
        {rising.length > 0 && (
          <ul className="analytics-rising">
            {rising.map((g, i) => (
              <li key={`${g.issue_key}-${i}`}>
                <span aria-hidden="true">⚠️</span>{" "}
                <strong>{g.category || g.issue_key || "uncategorized"}</strong>
                <span className="muted">
                  {" "}· vol {g.recent_volume} (was {g.prior_volume}) · drift {g.severity_drift}
                </span>
                {g.getting_worse && <span className="chip red" style={{ marginLeft: 8 }}>getting worse</span>}
              </li>
            ))}
          </ul>
        )}
      </div>

      {/* ── Financial impact ── */}
      {financial && estimates.length > 0 && (
        <div className="panel" style={{ marginTop: 18 }}>
          <h3 className="analytics-h">
            Financial impact · portfolio risk {fmtUSD(financial.portfolio_risk_usd)}
            <span className="muted"> (range {fmtUSD(financial.portfolio_risk_range_usd?.[0])}–{fmtUSD(financial.portfolio_risk_range_usd?.[1])})</span>
          </h3>
          <BarList
            data={estimates.map((e) => ({
              label: `#${e.cluster_id} · ${e.case_count} cases (${e.critical_count} critical)`,
              value: e.total_risk_usd || 0,
              tone: e.critical_count > 0 ? "danger" : "warn",
            }))}
            valueFormat={fmtUSD}
          />
        </div>
      )}

      {/* ── Fairness by channel ── */}
      {fairGroups.length > 0 && (
        <div className="panel" style={{ marginTop: 18 }}>
          <h3 className="analytics-h">Fairness by channel</h3>
          <div style={{ overflowX: "auto" }}>
            <table className="analytics-table">
              <thead>
                <tr><th>Group</th><th>Volume</th><th>Escalation rate</th><th>Handoff rate</th></tr>
              </thead>
              <tbody>
                {fairGroups.map((g) => (
                  <tr key={g.group}>
                    <td className="mono">{g.group}</td>
                    <td className="mono">{g.volume}</td>
                    <td>
                      <div className="row" style={{ gap: 8, alignItems: "center" }}>
                        <div className="rate-bar" style={{ flex: 1 }}>
                          <span style={{ width: `${Math.min(100, (g.escalation_rate || 0) * 100)}%` }} />
                        </div>
                        <span className="mono">{fmtPct(g.escalation_rate)}</span>
                      </div>
                    </td>
                    <td>
                      <div className="row" style={{ gap: 8, alignItems: "center" }}>
                        <div className="rate-bar" style={{ flex: 1 }}>
                          <span style={{ width: `${Math.min(100, (g.handoff_proxy_rate || 0) * 100)}%` }} />
                        </div>
                        <span className="mono">{fmtPct(g.handoff_proxy_rate)}</span>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Investigations ── */}
      {invRows.length > 0 && (
        <div className="panel" style={{ marginTop: 18 }}>
          <h3 className="analytics-h">Investigations · aging</h3>
          <StackBar segments={agingSegs} label="Investigation age distribution" />
          <div style={{ overflowX: "auto", marginTop: 12 }}>
            <table className="analytics-table">
              <thead>
                <tr><th>Investigation</th><th>Status</th><th>Days open</th><th>Cases</th><th>Cluster</th></tr>
              </thead>
              <tbody>
                {invTop.map((inv) => (
                  <tr key={inv.investigation_id}>
                    <td>{inv.title || inv.investigation_id}</td>
                    <td><span className="chip">{inv.status}</span></td>
                    <td className="mono">{inv.days_open ?? "—"}</td>
                    <td className="mono">{inv.case_count ?? "—"}</td>
                    <td className="mono">#{inv.cluster_id ?? "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="analytics-grid2" style={{ marginTop: 18 }}>
        {/* ── Forecast ── */}
        <div className="panel">
          <h3 className="analytics-h">Volume forecast (top clusters)</h3>
          {forecastVol.length === 0 && <div className="chart-empty">No forecast data</div>}
          {forecastVol.length > 0 && <BarList data={forecastVol} />}
          {forecasts.length > 0 && (
            <div style={{ overflowX: "auto", marginTop: 12 }}>
              <table className="analytics-table">
                <thead>
                  <tr><th>Cluster</th><th>Last wk</th><th>Slope/wk</th><th>Projected</th><th>Status</th></tr>
                </thead>
                <tbody>
                  {[...forecasts].sort((a, b) => (b.last_week_volume || 0) - (a.last_week_volume || 0)).slice(0, 8).map((f) => (
                    <tr key={f.cluster_id}>
                      <td className="mono">#{f.cluster_id}</td>
                      <td className="mono">{f.last_week_volume ?? "—"}</td>
                      <td className="mono">{f.slope_per_week ?? "—"}</td>
                      <td className="mono">{f.projected_next_week ?? "—"}</td>
                      <td><span className="chip">{f.status}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* ── Severity drift ── */}
        <div className="panel">
          <h3 className="analytics-h">Severity drift by cluster</h3>
          {driftRows.length === 0 && <div className="chart-empty">No drift data</div>}
          {driftRows.length > 0 && (
            <div style={{ overflowX: "auto" }}>
              <table className="analytics-table">
                <thead>
                  <tr><th>Cluster</th><th>Early avg</th><th>Late avg</th><th>Δ</th><th>Trend</th></tr>
                </thead>
                <tbody>
                  {[...driftRows].sort((a, b) => (b.severity_drift || 0) - (a.severity_drift || 0)).slice(0, 8).map((d) => (
                    <tr key={d.cluster_id}>
                      <td className="mono">#{d.cluster_id}</td>
                      <td className="mono">{d.early_avg_severity}</td>
                      <td className="mono">{d.late_avg_severity}</td>
                      <td className="mono">+{d.severity_drift}</td>
                      <td>{d.worsening ? <span className="chip red">▲ worsening</span> : <span className="chip green">stable</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      <div className="analytics-grid2" style={{ marginTop: 18 }}>
        {/* ── Cohorts ── */}
        <div className="panel">
          <div className="row" style={{ justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
            <h3 className="analytics-h" style={{ margin: 0 }}>Cohorts</h3>
            <select
              value={cohortField}
              onChange={(e) => setCohortField(e.target.value)}
              className="auth-select"
              style={{ minHeight: 32, fontSize: 12, width: "auto" }}
              aria-label="Cohort grouping"
            >
              {COHORT_FIELDS.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </div>
          <BarList data={cohortBars} emptyText="No cohorts" />
        </div>

        {/* ── Cross-pack ── */}
        <div className="panel">
          <h3 className="analytics-h">Cross-pack concepts</h3>
          {packSpecific.length === 0 && crossPatterns.length === 0 && (
            <div className="chart-empty">No cross-pack patterns</div>
          )}
          {(packSpecific.length > 0 || crossPatterns.length > 0) && (
            <div style={{ overflowX: "auto" }}>
              <table className="analytics-table">
                <thead>
                  <tr><th>Concept</th><th>Signal</th><th>Packs</th><th>Total</th></tr>
                </thead>
                <tbody>
                  {[...crossPatterns, ...packSpecific].slice(0, 8).map((p, i) => (
                    <tr key={`${p.concept}-${i}`}>
                      <td>{p.concept}</td>
                      <td><span className="chip">{p.signal}</span></td>
                      <td className="mono">{p.pack_count ?? Object.keys(p.packs || {}).length}</td>
                      <td className="mono">{p.total}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* ── Regulator watch ── */}
      {regMatches.length > 0 && (
        <div className="panel" style={{ marginTop: 18 }}>
          <h3 className="analytics-h">Regulator watch · {regMatches.length} matches</h3>
          <div style={{ overflowX: "auto" }}>
            <table className="analytics-table">
              <thead>
                <tr><th>Match</th><th>Detail</th></tr>
              </thead>
              <tbody>
                {regMatches.slice(0, 8).map((m, i) => (
                  <tr key={i}>
                    <td className="mono">{typeof m === "string" ? m : m.title || m.id || JSON.stringify(m).slice(0, 80)}</td>
                    <td className="muted">{typeof m === "string" ? "" : JSON.stringify(m).slice(0, 120)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Digest ── */}
      {digest?.digest_markdown && (
        <div className="panel" style={{ marginTop: 18 }}>
          <button
            type="button"
            className="ghost"
            onClick={() => setShowDigest((v) => !v)}
            aria-expanded={showDigest}
          >
            {showDigest ? "Hide daily digest −" : `Read daily digest (${digest.digest_name}) +`}
          </button>
          {showDigest && <div className="md analytics-digest">{renderMarkdown(digest.digest_markdown)}</div>}
        </div>
      )}
    </div>
  );
}
