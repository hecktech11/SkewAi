import { useCallback, useEffect, useState } from "react";
import { apiHeaders } from "../src/apiAuth.js";
import {
  fetchCriticalCases,
  goHash,
  includeSimQuery,
  openCases,
  openConsole,
  openIssue,
  openWarning,
  readIncludeSimulated,
  simulateTraffic,
} from "../src/ui/opsActions.js";
const fetchOpenP1Cases = fetchCriticalCases;
import { dayGreeting } from "../src/ui/greeting.js";
import { formatSnapshotTs } from "../src/ui/formatTime.js";
import { packLabel } from "../src/ui/labels.js";

function fmtUSD(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return null;
  return `$${Math.round(Number(x)).toLocaleString()}`;
}

/** Hottest fleet issue = max dollars at risk among live clusters, else max live cases. */
function pickHottest(liveRisk, dollarRows, estimates, gaps) {
  const live = Array.isArray(liveRisk) ? liveRisk : [];
  if (!live.length) return null;
  const dolById = new Map((dollarRows || []).map((r) => [String(r.cluster_id), r]));
  const estById = new Map((estimates || []).map((e) => [String(e.cluster_id), e]));
  const gapById = new Map((gaps || []).map((g) => [String(g.cluster_id), g]));
  let best = null;
  let bestKey = -1;
  for (const r of live) {
    const id = String(r.cluster_id);
    const est = estById.get(id);
    const dol = dolById.get(id);
    const dollars = est?.total_risk_usd ?? dol?.dollar_impact ?? 0;
    const key = Number(dollars) * 1e6 + Number(r.live_case_count || 0);
    if (key > bestKey) {
      bestKey = key;
      best = { ...r, _est: est || null, _dol: dol || null, _category: gapById.get(id)?.category || null };
    }
  }
  if (!best || bestKey <= 0) {
    // No dollars anywhere — hottest by live volume, if any cases exist.
    const byVol = [...live].sort((a, b) => (b.live_case_count || 0) - (a.live_case_count || 0))[0];
    if (!byVol || !(byVol.live_case_count > 0)) return null;
    return { ...byVol, _est: null, _dol: null, _category: gapById.get(String(byVol.cluster_id))?.category || null };
  }
  return best;
}

/**
 * Command Center — flagship wallboard.
 * One glance at live ops before diving into voice / cases.
 */
export default function CommandCenter({ refreshKey }) {
  const [health, setHealth] = useState(null);
  const [wall, setWall] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [usage, setUsage] = useState(null);
  const [drain, setDrain] = useState(null);
  const [p1Cases, setP1Cases] = useState([]);
  const [hot, setHot] = useState(null);
  const [cites, setCites] = useState(null);
  const [err, setErr] = useState("");
  const [msg, setMsg] = useState("");
  const [loading, setLoading] = useState(true);
  const [simBusy, setSimBusy] = useState(false);
  const [tick, setTick] = useState(() => new Date());

  const load = useCallback(async () => {
    setErr("");
    setLoading(true);
    try {
      const h = apiHeaders();
      const sim = includeSimQuery();
      // The hero always includes rehearsal traffic (labeled): in production
      // there is none, so this is a no-op there.
      const [a, b, c, d, e, p1, f, g, hh, gg] = await Promise.all([
        fetch("/health"),
        fetch(`/api/frontline/wallboard?include_simulated=${sim}`, { headers: h }),
        fetch(`/api/frontline/metrics?include_simulated=${sim}`, { headers: h }),
        fetch("/api/frontline/usage", { headers: h }),
        fetch("/api/frontline/ops/drain", { headers: h }),
        fetchCriticalCases(8),
        fetch(`/api/frontline/early-warning?window_days=30&include_simulated=true`, { headers: h }),
        fetch(`/api/frontline/copq/rank?window_days=30&include_simulated=true`, { headers: h }),
        fetch(`/api/frontline/analytics/financial-impact`, { headers: h }),
        fetch(`/api/frontline/insights/product-gap?window_days=30&limit=8&include_simulated=true`, { headers: h }),
      ]);
      if (!a.ok) throw new Error(`health ${a.status}`);
      setHealth(await a.json());
      setWall(b.ok ? await b.json() : null);
      setMetrics(c.ok ? await c.json() : null);
      setUsage(d.ok ? await d.json() : null);
      setDrain(e.ok ? await e.json() : null);
      setP1Cases(p1);
      const early = f.ok ? await f.json().catch(() => null) : null;
      const copq = g.ok ? await g.json().catch(() => null) : null;
      const fin = hh.ok ? await hh.json().catch(() => null) : null;
      const gaps = gg.ok ? await gg.json().catch(() => null) : null;
      let hotPick = pickHottest(early?.live_risk, copq?.clusters, fin?.estimates, gaps?.top_issues);
      if (!hotPick) {
        // Corpus fallback: no live traffic yet — hottest product-gap issue.
        const top = [...(gaps?.top_issues || [])].sort((a, b) => (b.volume || 0) - (a.volume || 0))[0];
        if (top && top.volume > 0 && top.cluster_id !== null && top.cluster_id !== undefined) {
          hotPick = {
            cluster_id: top.cluster_id,
            pack_id: null,
            live_case_count: top.volume,
            critical_count: top.severity_mix?.Critical || 0,
            _est: null,
            _dol: null,
            _category: top.category || null,
            _corpus: true,
          };
        }
      }
      if (hotPick) {
        // Corpus context: top terms, record volume, backtest lead vs advisory.
        try {
          const cr = await fetch(`/api/frontline/clusters/${encodeURIComponent(hotPick.cluster_id)}/context`, { headers: h });
          if (cr.ok) hotPick = { ...hotPick, _ctx: await cr.json() };
        } catch {
          /* hero works without context */
        }
        // Dollarized fallback: no costed cases yet → exposure estimate from
        // corpus volume × the pack's own cost-per-case, always labeled est.
        const hasRealDollars = (hotPick._est?.total_risk_usd ?? hotPick._dol?.dollar_impact ?? 0) > 0;
        if (!hasRealDollars) {
          const cpc = Number(fin?.cost_per_case);
          const vol = Number(hotPick._ctx?.record_count);
          if (cpc > 0 && vol > 0) hotPick = { ...hotPick, _estDollars: vol * cpc };
        }
      }
      setHot(hotPick);
      setTick(new Date());
    } catch (ex) {
      setErr(String(ex.message || ex));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const t = setInterval(load, 12000);
    return () => clearInterval(t);
  }, [load, refreshKey]);

  // Grounded-citation count streams in once (slow export — never on the 12s tick).
  useEffect(() => {
    const ctrl = new AbortController();
    fetch(`/api/frontline/audits/export?limit=60`, {
      headers: apiHeaders(),
      signal: ctrl.signal,
    })
      .then(
        (r) => {
          if (!r.ok) throw new Error("bad");
          return r.json();
        },
        () => null,
      )
      .then((d) => {
        if (!d || ctrl.signal.aborted) return;
        let g = 0;
        let t = 0;
        for (const row of d.interactions || []) {
          g += Number(row.audit?.grounded_actions) || 0;
          t += Number(row.audit?.total_actions) || 0;
        }
        if (t > 0) setCites({ grounded: g, total: t });
      })
      .catch(() => {});
    return () => ctrl.abort();
  }, []);

  async function runSimulate() {
    setSimBusy(true);
    setMsg("");
    setErr("");
    try {
      const d = await simulateTraffic({ count: 15 });
      setMsg(
        `Simulated traffic loaded · completed ${d.completed ?? d.count ?? "?"} · cases created ${d.cases_created ?? "—"}`,
      );
      await load();
    } catch (e) {
      setErr(String(e.message || e));
    } finally {
      setSimBusy(false);
    }
  }

  const live = Array.isArray(wall?.live_contacts) ? wall.live_contacts : [];
  const clusters = wall?.top_risk_clusters || wall?.top_clusters || [];
  const liveCount =
    wall?.active_contacts ?? wall?.active_count ?? wall?.live_count ?? live.length ?? 0;
  const p1 = wall?.critical_open ?? wall?.p1_open ?? 0;
  const openCaseCount = wall?.open_cases ?? 0;
  const draining = Boolean(drain?.draining);

  const hotDollarsReal = fmtUSD(hot?._est?.total_risk_usd ?? hot?._dol?.dollar_impact);
  const hotDollars = hotDollarsReal ?? (hot?._estDollars ? `${fmtUSD(hot._estDollars)} est.` : null);
  const hotLead = hot?.lead_time_weeks ?? hot?._dol?.lead_time_weeks ?? hot?._ctx?.backtest?.lead_time_weeks;
  const hotAdv = hot?.matched_advisory ?? hot?._dol?.matched_advisory ?? hot?._ctx?.backtest?.advisory_id;
  const hotCategory = hot?._category || hot?._ctx?.category || `Cluster ${hot?.cluster_id}`;
  const heroBits = hot
    ? [
        `${hotCategory} cluster`,
        hotDollars ? `${hotDollars} at risk` : null,
        hotLead !== null && hotLead !== undefined
          ? `flagged ${hotLead}w before ${hotAdv || "the advisory"}`
          : null,
        cites ? `grounded ${cites.grounded}/${cites.total} citations` : null,
      ].filter(Boolean)
    : [];

  return (
    <div className="page-enter">
      {hot ? (
        <section className="panel cc-issue-hero" aria-label="Hottest fleet issue">
          <p className="muted mono" style={{ margin: "0 0 6px", fontSize: 11 }}>
            HOTTEST FLEET ISSUE · #{hot.cluster_id}
            {hot?.critical_count > 0 && (
              <> · <span className="chip red">{hot.critical_count} critical</span></>
            )}
          </p>
          <h1 className="cc-issue-line">{heroBits.join(" · ")}</h1>
          <p className="muted" style={{ fontSize: 11, margin: "6px 0 0" }}>
            Incl. simulated rehearsal traffic (no-op when there is none).
          </p>
          <div className="row" style={{ marginTop: 12, gap: 8 }}>
            <button type="button" className="btn btn-primary" onClick={() => openIssue(hot.cluster_id)}>
              Open issue →
            </button>
            <button type="button" className="ghost" onClick={() => openWarning({ clusterId: hot.cluster_id, packId: hot.pack_id })}>
              Early warning
            </button>
          </div>
        </section>
      ) : (
        <section className="cc-hero" aria-label="Command center overview">
          <div className="cc-hero-copy">
            <h1 className="cc-greeting">{dayGreeting(tick)}</h1>
          </div>
          <div className="cc-hero-meta">
            <button type="button" className="ghost" onClick={load} disabled={loading}>
              {loading ? "Refreshing…" : "Refresh"}
            </button>
          </div>
        </section>
      )}

      {msg && (
        <div className="banner banner-ok" role="status">
          {msg}
        </div>
      )}

      {err && (
        <div className="banner banner-error" role="alert">
          Could not load command center: <span className="mono">{err}</span>
        </div>
      )}

      {loading && !wall && (
        <div className="cc-skel" aria-hidden="true">
          <div className="skeleton block" />
          <div className="skeleton block" />
          <div className="skeleton block" />
          <div className="skeleton block" />
        </div>
      )}

      {readIncludeSimulated() && (
        <p className="faint" style={{ margin: "0 0 10px" }}>
          Including simulated contacts (toggle on Early warning).
        </p>
      )}

      <div className="stat-grid" style={{ marginBottom: 16 }}>
        <button
          type="button"
          className="stat-card stat-card-btn"
          onClick={() => openConsole()}
          title="Open live console"
        >
          <div className="label">Live contacts</div>
          <div className="value">{liveCount}</div>
          <div className="hint">Active now</div>
        </button>
        <button
          type="button"
          className={`stat-card ${Number(p1) > 0 ? "danger" : ""} stat-card-btn`}
          onClick={() => openCases({ severity: "Critical", status: "open" })}
          title="Open Critical case queue"
        >
          <div className="label">Critical open</div>
          <div className="value">{p1}</div>
          <div className="hint">Needs a look</div>
        </button>
        <button
          type="button"
          className="stat-card stat-card-btn"
          onClick={() => openCases({ status: "open" })}
          title="Open case queue"
        >
          <div className="label">Open cases</div>
          <div className="value">{openCaseCount}</div>
          <div className="hint">Open and follow-up</div>
        </button>
        <button
          type="button"
          className={`stat-card ${draining ? "danger" : "ok"} stat-card-btn`}
          onClick={() => goHash("settings?tab=lab&view=ops")}
          title="Open drain controls in Feature lab"
        >
          <div className="label">Deploy drain</div>
          <div className="value" style={{ fontSize: 24, letterSpacing: "-0.04em" }}>
            {draining ? "Draining" : "Ready"}
          </div>
          <div className="hint">
            {draining
              ? `${drain?.active_count ?? 0} still active`
              : "New contacts accepted"}
          </div>
        </button>
      </div>

      <div className="cc-section-label">
        <h2>Floor view</h2>
        <span>Live contacts · top risk clusters</span>
      </div>

      <div className="grid-2">
        <section className="panel">
          <h2>
            Live contacts
            <span className="faint" style={{ fontWeight: 500, letterSpacing: 0 }}>
              {live.length} active
            </span>
          </h2>
          {live.length === 0 ? (
            <div className="empty-state">
              <p className="empty-state-text">
                No active contacts. Start one from Voice agent.
              </p>
              <div className="row" style={{ marginTop: 12 }}>
                <button type="button" className="primary" onClick={() => (window.location.hash = "call")}>
                  Start a contact
                </button>
              </div>
            </div>
          ) : (
            <div className="live-feed">
              {live.map((row) => (
                <button
                  type="button"
                  className="live-row live-row-btn"
                  key={row.interaction_id}
                  onClick={() => openConsole(row.interaction_id)}
                  title="Open in live console"
                >
                  <span className="dot" aria-hidden="true" />
                  <div className="id" title={row.interaction_id}>
                    {row.interaction_id}
                  </div>
                  <div className="meta">
                    {[row.pack_id, row.category].filter(Boolean).join(" · ") || "intake in progress"}
                  </div>
                  <div className="chan">
                    {row.channel === "simulated" ? (
                      <span className="chip purple" style={{ fontSize: 10, padding: "2px 6px" }}>simulated</span>
                    ) : (
                      row.channel || "open →"
                    )}
                  </div>
                </button>
              ))}
            </div>
          )}
        </section>

        <section className="panel">
          <h2>
            Top risk clusters
            <span className="faint" style={{ fontWeight: 500, letterSpacing: 0 }}>
              by open cases
            </span>
          </h2>
          {clusters.length === 0 ? (
            <div className="empty-state">
              <p className="empty-state-text">
                No open clustered cases yet. Early warning lights up here when similar contacts
                share a cluster.
              </p>
              <button
                type="button"
                className="ghost"
                style={{ marginTop: 12 }}
                onClick={() => (window.location.hash = "warning")}
              >
                Open early warning
              </button>
            </div>
          ) : (
            <div className="risk-list">
              {clusters.map((c, i) => (
                <button
                  type="button"
                  className="risk-row"
                  key={`${c.pack_id}-${c.cluster_id}`}
                  onClick={() => openWarning({ clusterId: c.cluster_id, packId: c.pack_id })}
                >
                  <div className="risk-rank">#{String(i + 1).padStart(2, "0")}</div>
                  <div className="risk-main">
                    <div className="title">
                      Cluster {c.cluster_id}{" "}
                      <span className={`sev ${(c.max_severity || "").toLowerCase()}`}>
                        {c.max_severity || "—"}
                      </span>
                    </div>
                    <div className="hint">{packLabel(c.pack_id)}</div>
                  </div>
                  <div className="risk-count" title="Open cases">
                    {c.open_cases}
                  </div>
                </button>
              ))}
            </div>
          )}
        </section>
      </div>

      <div className="cc-section-label">
        <h2>Critical queue</h2>
        <span>Open Critical cases — click a row to open the case queue</span>
      </div>
      <section className="panel" style={{ marginBottom: 16 }}>
        {p1Cases.length === 0 ? (
          <div className="empty-state">
            <p className="empty-state-text">
              No open Critical cases. Simulate traffic or take live contacts to light this up.
            </p>
            <div className="row" style={{ marginTop: 10 }}>
              <button type="button" className="ghost" onClick={runSimulate} disabled={simBusy}>
                {simBusy ? "Seeding…" : "Seed demo traffic"}
              </button>
              <button type="button" className="ghost" onClick={() => openCases({ status: "open" })}>
                All open cases
              </button>
            </div>
          </div>
        ) : (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Case</th>
                  <th>Category</th>
                  <th>Severity</th>
                  <th>Status</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {p1Cases.map((c) => (
                  <tr
                    key={c.case_id}
                    style={{ cursor: "pointer" }}
                    tabIndex={0}
                    onClick={() =>
                      openCases({
                        severity: "Critical",
                        status: "open",
                        caseId: c.case_id,
                      })
                    }
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        openCases({
                          severity: "Critical",
                          status: "open",
                          caseId: c.case_id,
                        });
                      }
                    }}
                  >
                    <td className="mono">{c.case_id}</td>
                    <td>{c.category || "—"}</td>
                    <td>{c.severity || "Critical"}</td>
                    <td>{c.status}</td>
                    <td>
                      <button
                        type="button"
                        className="ghost"
                        onClick={(e) => {
                          e.stopPropagation();
                          openCases({
                            severity: "Critical",
                            status: "open",
                            caseId: c.case_id,
                          });
                        }}
                      >
                        Open
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <div className="cc-section-label">
        <h2>System readiness</h2>
        <span>Health · metering · pilot metrics</span>
      </div>

      <div className="readiness-strip">
        <div className="readiness-item">
          <div className="k">API status</div>
          <div className={`v ${health?.status === "ok" ? "ok-text" : "warn-text"}`}>
            {health?.status || "connecting"}
          </div>
        </div>
        <div className="readiness-item">
          <div className="k">Auth</div>
          <div className="v">{health?.auth_required ? "API key required" : "open local"}</div>
        </div>
        <div className="readiness-item">
          <div className="k">Database</div>
          <div className="v">{health?.db_ok ? "reachable" : "check connection"}</div>
        </div>
        <div className="readiness-item">
          <div className="k">Pack load</div>
          <div className="v">{health?.pack_ok ? "ok" : "failed"}</div>
        </div>
        <div className="readiness-item">
          <div className="k">LLM</div>
          <div className={`v ${health?.llm_available ? "ok-text" : "warn-text"}`}>
            {health?.llm_available ? "available" : "off"}
          </div>
        </div>
        <div className="readiness-item">
          <div className="k">Embedding</div>
          <div className="v">{health?.embedding?.mode || "—"}</div>
        </div>
      </div>

      <div className="grid-2" style={{ marginTop: 16 }}>
        <section className="panel">
          <h2>Usage metering</h2>
          {usage ? (
            <div className="kvs">
              <span className="k">tenant</span>
              <span className="v mono">{usage.tenant_id}</span>
              <span className="k">month</span>
              <span className="v mono">{usage.month}</span>
              {Object.entries(usage.metrics || {}).map(([k, v]) => (
                <div key={k} style={{ display: "contents" }}>
                  <span className="k">{k}</span>
                  <span className="v mono">{v}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="empty">No usage rows yet — contacts still count from interactions.</div>
          )}
        </section>

        <section className="panel">
          <div className="pilot-head">
            <h2>Pilot metrics</h2>
            {metrics?.ts ? (
              <span className="pilot-asof">{formatSnapshotTs(metrics.ts)}</span>
            ) : null}
          </div>
          {metrics ? (
            <div className="kv-grid">
              {[
                ["Open cases", metrics.cases?.open],
                ["Critical open", metrics.cases?.critical_open],
                ["Pending follow-up", metrics.cases?.pending_followup],
                ["Investigations", metrics.investigations?.open],
                ["Active calls", metrics.interactions_active],
                ["Dead-letters", metrics.alert_dead_letters_pending],
                ["Connector pending", metrics.connector_deliveries_pending],
                ["Case notes", metrics.case_notes_in_window],
                ["Fixes recorded", metrics.fix_loop?.fixes_recorded],
              ]
                .filter(([k, v]) => k !== "ts" && v != null)
                .map(([k, v]) => (
                  <div className="kv-tile" key={k}>
                    <div className="k">{k}</div>
                    <div className="v mono">{String(v)}</div>
                  </div>
                ))}
            </div>
          ) : (
            <div className="empty">No scalar metrics in snapshot.</div>
          )}
        </section>
      </div>
    </div>
  );
}
