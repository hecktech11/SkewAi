import { useEffect, useRef, useState } from "react";
import { apiHeaders } from "../src/apiAuth.js";
import { BarList } from "../src/ui/Chart.jsx";
import { openCases, openConsole } from "../src/ui/opsActions.js";

function fmtUSD(x) {
  if (x === null || x === undefined || Number.isNaN(Number(x))) return null;
  return `$${Math.round(Number(x)).toLocaleString()}`;
}

/** Lead-time ruler: spike weeks → advisory marker. */
function LeadRuler({ weeks, advisoryId, leadW }) {
  const rows = (weeks || []).slice().reverse().slice(-12);
  if (!rows.length) return null;
  const max = Math.max(1, ...rows.map((w) => Number(w.record_count) || 0));
  const W = 600;
  const H = 64;
  const pad = 10;
  const step = rows.length > 1 ? (W - pad * 2 - 60) / (rows.length - 1) : 0;
  const xFor = (i) => pad + i * step;
  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`Lead time ${leadW} weeks before ${advisoryId}`}
      preserveAspectRatio="none"
      style={{ width: "100%", display: "block", marginTop: 10 }}
    >
      <line x1={pad} y1={H / 2} x2={W - pad} y2={H / 2} stroke="var(--edge)" strokeWidth={2} />
      {rows.map((w, i) => (
        <circle
          key={w.iso_week || i}
          cx={xFor(i)}
          cy={H / 2}
          r={3 + (Number(w.record_count) / max) * 7}
          fill={w.is_anomaly ? "var(--danger)" : "var(--accent)"}
          opacity={w.is_anomaly ? 1 : 0.55}
        >
          <title>{`${w.iso_week}: ${w.record_count}${w.is_anomaly ? " (spike)" : ""}`}</title>
        </circle>
      ))}
      {advisoryId && (
        <g>
          <rect
            x={W - pad - 7}
            y={H / 2 - 7}
            width={14}
            height={14}
            transform={`rotate(45 ${W - pad} ${H / 2})`}
            fill="var(--warn)"
          />
          <text x={W - pad - 4} y={H - 4} textAnchor="end" fontSize={11} fill="var(--ink-soft)">
            {advisoryId} · {leadW}w lead
          </text>
        </g>
      )}
    </svg>
  );
}

/** Before rate → fix date → after rate, with reopen count. */
function FixChart({ weeks, fixIdx, before, after, reopened }) {
  const rows = weeks || [];
  if (!rows.length) return <div className="chart-empty">No weekly history</div>;
  const max = Math.max(1, ...rows.map((w) => Number(w.record_count) || 0));
  const W = 600;
  const H = 140;
  const pad = 6;
  const bw = (W - pad * 2) / rows.length;
  const yFor = (v) => H - 18 - (v / max) * (H - 30);
  return (
    <div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Weekly volume, fix marked, before ${before} after ${after}`}
        preserveAspectRatio="none"
        className="chart-svg fix-chart"
      >
        {rows.map((w, i) => {
          const h = Math.max(2, H - 18 - yFor(Number(w.record_count) || 0));
          return (
            <rect
              key={w.iso_week || i}
              x={(pad + i * bw + 1).toFixed(1)}
              y={yFor(Number(w.record_count) || 0).toFixed(1)}
              width={Math.max(1, bw - 2).toFixed(1)}
              height={h.toFixed(1)}
              className={w.is_anomaly ? "fix-bar anomaly" : "fix-bar"}
            >
              <title>{`${w.iso_week}: ${w.record_count}`}</title>
            </rect>
          );
        })}
        {fixIdx >= 0 && fixIdx < rows.length && (
          <g>
            <line
              x1={(pad + fixIdx * bw + bw / 2).toFixed(1)}
              y1={4}
              x2={(pad + fixIdx * bw + bw / 2).toFixed(1)}
              y2={H - 18}
              className="fix-line"
            />
            <text
              x={(pad + fixIdx * bw + bw / 2 + 4).toFixed(1)}
              y={14}
              className="fix-tag"
            >
              fix
            </text>
          </g>
        )}
      </svg>
      <p className="muted" style={{ fontSize: 12, margin: "6px 0 0" }}>
        Before <strong className="mono">{before}</strong>
        {" → "}after <strong className="mono">{after}</strong>
        {" · "}reopened <strong className="mono">{reopened}</strong>
      </p>
    </div>
  );
}

function mode(values) {
  const counts = {};
  for (const v of values || []) {
    if (v === null || v === undefined || v === "") continue;
    counts[v] = (counts[v] || 0) + 1;
  }
  const top = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
  return top ? { value: top[0], count: top[1] } : null;
}

function SourceCards({ rows, chip }) {
  return (rows || []).map((r) => (
    <div className="reg-summary-card" key={`${chip}-${r.record_id}`}>
      <span className="chip">{chip}</span>{" "}
      <span className="mono muted" style={{ fontSize: 12 }}>{r.record_id}</span>
      <p style={{ margin: "8px 0" }}>{r.text}</p>
      <p className="muted" style={{ margin: 0, fontSize: 12 }}>
        {[r.entity_1, r.entity_2, r.entity_3].filter(Boolean).join(" · ")}
        {r.tag && r.tag !== chip.toUpperCase() ? ` · ${r.tag}` : ""}
        {r.received_at ? ` · ${String(r.received_at).slice(0, 10)}` : ""}
      </p>
    </div>
  ));
}

/**
 * One Issue page (#issue/<investigation_id>, or a cluster id from older
 * links): everything known about a single failure. Evidence comes from
 * every loaded source on that slice, including NHTSA complaints and live
 * calls. Every block hides itself when it has no real data.
 */
export default function IssueDesk({ issueId, refreshKey }) {
  const routeId = issueId != null && issueId !== "" ? String(issueId) : null;
  const openedByInvestigation = !!routeId && routeId.startsWith("inv_");
  const [bundle, setBundle] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [corpus, setCorpus] = useState(null);
  const [stamp, setStamp] = useState(null);
  const [stampBusy, setStampBusy] = useState(false);
  const [fixResult, setFixResult] = useState(null);
  const [fixBusy, setFixBusy] = useState(false);
  const [fixDate, setFixDate] = useState("");
  const [fixNote, setFixNote] = useState("");
  const [fixControl, setFixControl] = useState("");
  const [commentBody, setCommentBody] = useState("");
  const [commentBusy, setCommentBusy] = useState(false);
  const [capas, setCapas] = useState(null);
  const [capaReq, setCapaReq] = useState("");
  const [capaBusy, setCapaBusy] = useState(false);
  const [searchQ, setSearchQ] = useState("");
  const [searchRes, setSearchRes] = useState(null);
  const [searchBusy, setSearchBusy] = useState(false);
  const [brief, setBrief] = useState(null);
  const [assigneeDraft, setAssigneeDraft] = useState("");
  const [slaDraft, setSlaDraft] = useState("");
  const [hypDraft, setHypDraft] = useState("");
  const [wsBusy, setWsBusy] = useState(false);
  const [wsError, setWsError] = useState(null);
  const topLotRef = useRef(null);

  useEffect(() => {
    if (routeId == null) return;
    const ctrl = new AbortController();
    let alive = true;
    async function load() {
      setLoading(true);
      setError(null);
      setBundle(null);
      setBrief(null);
      setStamp(null);
      setFixResult(null);
      const get = (url) =>
        fetch(url, { headers: apiHeaders(), signal: ctrl.signal }).then(
          (r) => {
            if (!r.ok) throw new Error(`${url} → HTTP ${r.status}`);
            return r.json();
          },
          () => null,
        );
      // Issue analysis always includes rehearsal traffic (labeled below):
      // in production there is none, so this is a no-op there, while the
      // offline demo's simulated contacts stay visible without toggle-hunting.
      const SIM = "include_simulated=true";
      try {
        let clusterForPanels = openedByInvestigation ? null : routeId;
        let directInv = null;
        if (openedByInvestigation) {
          directInv = await get(`/api/frontline/investigations/${encodeURIComponent(routeId)}`);
          const cid = Number(directInv?.cluster_id);
          if (Number.isFinite(cid) && cid > 0) clusterForPanels = String(cid);
        }
        const clusterQs = clusterForPanels != null
          ? encodeURIComponent(clusterForPanels)
          : "";
        const [early, copq, fin, gaps, cases, invs, metrics, ctx] = await Promise.all([
          get(`/api/frontline/early-warning?window_days=30&${SIM}`),
          get(`/api/frontline/copq/rank?window_days=30&${SIM}`),
          get(`/api/frontline/analytics/financial-impact`),
          get(`/api/frontline/insights/product-gap?window_days=30&limit=8&${SIM}`),
          clusterForPanels
            ? get(`/api/frontline/cases?cluster=${clusterQs}&limit=50&${SIM}`)
            : null,
          openedByInvestigation
            ? null
            : get(`/api/frontline/investigations?window_days=90&limit=50`),
          get(`/api/frontline/metrics?window_days=30&${SIM}`),
          clusterForPanels
            ? get(`/api/frontline/clusters/${clusterQs}/context`).catch(() => null)
            : null,
        ]);
        if (!alive) return;
        const num = Number(clusterForPanels);
        const matchCid = (r) =>
          clusterForPanels != null
          && r
          && (r.cluster_id === num || String(r.cluster_id) === String(clusterForPanels));
        const live = (early?.live_risk || []).find(matchCid) || null;
        const dollars = (copq?.clusters || []).find(matchCid) || null;
        const est = (fin?.estimates || []).find((e) => matchCid(e)) || null;
        const gap = (gaps?.top_issues || []).find((g) => matchCid(g)) || null;
        let inv = directInv;
        if (!inv && clusterForPanels != null) {
          const invList = (invs?.investigations || []).filter(matchCid);
          inv = invList.find((i) => i.status === "open") || invList[0] || null;
        }
        setBundle({
          live, dollars, est, gap, cases, inv, metrics, ctx,
          clusterId: clusterForPanels,
        });
        setLoading(false);
      } catch (e) {
        if (!alive) return;
        setError(String(e.message || e));
        setLoading(false);
      }
    }
    load();
    return () => {
      alive = false;
      ctrl.abort();
    };
  }, [routeId, openedByInvestigation, refreshKey]);

  // Phase 2: category-dependent corpus evidence + workspace + voice turn.
  const casesSample = bundle?.cases?.cases || [];
  const inv = bundle?.inv || null;

  useEffect(() => {
    if (!bundle?.inv?.investigation_id) return;
    const ctrl = new AbortController();
    let alive = true;
    async function load() {
      const cat = bundle.gap?.category || bundle.live?.category || bundle.ctx?.category || null;
      const get = (url) =>
        fetch(url, { headers: apiHeaders(), signal: ctrl.signal }).then(
          (r) => {
            if (!r.ok) throw new Error("bad");
            return r.json();
          },
          () => null,
        );
      const e2 = mode(casesSample.map((c) => c.entity_2 || c.make))?.value || null;
      const invId = bundle.inv.investigation_id;
      const [warranty, service, lots, workspace, nextBrief] = await Promise.all([
        cat ? get(`/api/frontline/records?source=WARRANTY&category=${encodeURIComponent(cat)}&limit=25`) : null,
        cat ? get(`/api/frontline/records?source=SERVICE&category=${encodeURIComponent(cat)}&limit=25`) : null,
        get(`/api/frontline/suppliers/lots?limit=10`),
        get(`/api/frontline/investigations/${encodeURIComponent(invId)}/workspace`),
        get(`/api/frontline/investigations/${encodeURIComponent(invId)}/brief`).catch(() => null),
      ]);
      // Voice evidence: first member case with an interaction → first customer turn.
      let voice = null;
      const withIx = casesSample.find((c) => c.interaction_id);
      if (withIx) {
        const det = await get(`/api/interactions/${encodeURIComponent(withIx.interaction_id)}`);
        const turn = (det?.turns || []).find((t) => t.speaker === "customer" && (t.text || "").trim());
        if (turn) {
          voice = {
            text: turn.text,
            ts: turn.ts,
            interaction_id: withIx.interaction_id,
            case_id: withIx.case_id,
            pack_version: det?.interaction?.pack_version || null,
          };
        }
      }
      if (!alive) return;
      setBrief(nextBrief);
      setCorpus({ warranty, service, lots, workspace, voice, entity2: e2 });
    }
    load();
    return () => {
      alive = false;
      ctrl.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [bundle, routeId]);

  // Open CAPAs load once per issue (filtered client-side to the top lot).
  useEffect(() => {
    if (routeId == null) return;
    const ctrl = new AbortController();
    fetch(`/api/frontline/suppliers/capas?status=open&limit=20`, {
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
        if (!ctrl.signal.aborted) setCapas(d?.capas || []);
      })
      .catch(() => {});
    return () => ctrl.abort();
  }, [routeId]);

  const savedAssignee = corpus?.workspace?.assignee || "";
  const savedSla = corpus?.workspace?.sla_due_at
    ? String(corpus.workspace.sla_due_at).slice(0, 10)
    : "";
  useEffect(() => {
    setAssigneeDraft(savedAssignee);
    setSlaDraft(savedSla);
  }, [inv?.investigation_id, savedAssignee, savedSla]);

  // Qubot stamp for the voice evidence interaction.
  async function verifyVoice() {
    const iid = corpus?.voice?.interaction_id;
    if (!iid || stampBusy) return;
    setStampBusy(true);
    try {
      const r = await fetch(`/api/frontline/audits/${encodeURIComponent(iid)}?rerun=true`, {
        headers: apiHeaders(),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setStamp(await r.json());
    } catch (e) {
      setStamp({ error: String(e.message || e) });
    } finally {
      setStampBusy(false);
    }
  }

  async function recordFix() {
    if (!inv || fixBusy) return;
    setFixBusy(true);
    try {
      const body = { status: "closed" };
      if (fixDate.trim()) body.fixed_at = fixDate.trim();
      if (fixNote.trim()) body.note = fixNote.trim();
      if (fixControl.trim()) body.control_entity_2 = fixControl.trim();
      const r = await fetch(`/api/frontline/investigations/${encodeURIComponent(inv.investigation_id)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json", ...apiHeaders() },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 160)}`);
      setFixResult(await r.json());
    } catch (e) {
      setFixResult({ error: String(e.message || e) });
    } finally {
      setFixBusy(false);
    }
  }

  async function refreshWorkspace() {
    if (!inv) return null;
    try {
      const r = await fetch(
        `/api/frontline/investigations/${encodeURIComponent(inv.investigation_id)}/workspace`,
        { headers: apiHeaders() },
      );
      if (!r.ok) return null;
      const ws = await r.json();
      setCorpus((c) => (c ? { ...c, workspace: ws } : c));
      return ws;
    } catch {
      return null;
    }
  }

  async function postComment(e) {
    e.preventDefault();
    const body = commentBody.trim();
    if (!inv || !body || commentBusy) return;
    setCommentBusy(true);
    try {
      const r = await fetch(
        `/api/frontline/investigations/${encodeURIComponent(inv.investigation_id)}/comments`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json", ...apiHeaders() },
          body: JSON.stringify({ body }),
        },
      );
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setCommentBody("");
      await refreshWorkspace();
    } catch (ex) {
      setCommentBody(body);
      alert(`Comment failed: ${String(ex.message || ex)}`);
    } finally {
      setCommentBusy(false);
    }
  }

  async function refreshCapas() {
    try {
      const r = await fetch(`/api/frontline/suppliers/capas?status=open&limit=20`, {
        headers: apiHeaders(),
      });
      if (r.ok) setCapas((await r.json()).capas || []);
    } catch {
      /* best effort */
    }
  }

  async function openCapa(e) {
    e.preventDefault();
    const req = capaReq.trim();
    if (!req || capaBusy || !topLotRef.current) return;
    setCapaBusy(true);
    try {
      const r = await fetch(`/api/frontline/suppliers/capas`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...apiHeaders() },
        body: JSON.stringify({
          supplier: topLotRef.current.supplier,
          lot_id: topLotRef.current.lot_id,
          request: req,
        }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 160)}`);
      setCapaReq("");
      await refreshCapas();
    } catch (ex) {
      alert(`CAPA failed: ${String(ex.message || ex)}`);
    } finally {
      setCapaBusy(false);
    }
  }

  async function runSearch(q) {
    const query = (q ?? searchQ).trim();
    if (!query || searchBusy) return;
    setSearchBusy(true);
    setSearchRes(null);
    try {
      const r = await fetch(`/api/frontline/enterprise/copilot`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...apiHeaders() },
        body: JSON.stringify({ query, limit: 6 }),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setSearchRes(await r.json());
    } catch (e) {
      setSearchRes({ error: String(e.message || e) });
    } finally {
      setSearchBusy(false);
    }
  }

  async function saveWorkspace(e) {
    e.preventDefault();
    if (!inv || wsBusy) return;
    setWsBusy(true);
    setWsError(null);
    const payload = { assignee: assigneeDraft };
    if (slaDraft) payload.sla_due_at = slaDraft;
    if (hypDraft.trim()) {
      payload.hypotheses = [{ body: hypDraft.trim(), status: "open" }];
    }
    try {
      const r = await fetch(
        `/api/frontline/investigations/${encodeURIComponent(inv.investigation_id)}/workspace`,
        {
          method: "PATCH",
          headers: { "Content-Type": "application/json", ...apiHeaders() },
          body: JSON.stringify(payload),
        },
      );
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 160)}`);
      setHypDraft("");
      await refreshWorkspace();
    } catch (ex) {
      setWsError(String(ex.message || ex));
    } finally {
      setWsBusy(false);
    }
  }

  if (routeId == null) {
    return <div className="empty">No issue selected. Open one from the Command Center.</div>;
  }
  if (loading && !bundle) return <div className="empty">Loading {routeId}…</div>;
  if (error && !bundle) return <div className="empty err-text">{error}</div>;
  if (!bundle) return null;

  const { live, dollars, est, gap, cases, metrics, ctx } = bundle;
  const clusterId = bundle.clusterId ?? null;
  const slice = brief?.slice || null;
  const title = slice?.category || gap?.category || live?.category || ctx?.category || inv?.title || (clusterId ? `Cluster ${clusterId}` : routeId);
  const dollarsUsd = fmtUSD(est?.total_risk_usd ?? dollars?.dollar_impact);
  const leadW = brief?.recall?.lead_time_weeks ?? live?.lead_time_weeks ?? dollars?.lead_time_weeks ?? ctx?.backtest?.lead_time_weeks;
  const advisory = brief?.recall?.advisory_id ?? live?.matched_advisory ?? dollars?.matched_advisory ?? ctx?.backtest?.advisory_id;
  const critical = live?.critical_count ?? 0;
  const liveCount = live?.live_case_count ?? casesSample.length;
  const years = casesSample.map((c) => c.entity_1).filter(Boolean);
  const yearRange = years.length
    ? `${years.slice().sort()[0]}–${years.slice().sort()[years.length - 1]}`
    : null;
  const makeMode = mode(casesSample.map((c) => c.entity_2));
  const modelMode = mode(casesSample.map((c) => c.entity_3));
  const lots = (corpus?.lots?.lots || []).filter((l) =>
    !gap?.category || (l.top_categories || []).some((t) => t.category === gap.category),
  );
  const topLot = lots[0] || null;
  topLotRef.current = topLot;
  const bySource = brief?.records_by_source || {};
  const nhtsaRows = bySource.NHTSA || [];
  const callRows = bySource.FRONTLINE || [];
  const wRows = bySource.WARRANTY?.length ? bySource.WARRANTY : (corpus?.warranty?.records || []);
  const sRows = bySource.SERVICE?.length ? bySource.SERVICE : (corpus?.service?.records || []);
  const fRows = bySource.FIELD || [];
  const tRows = bySource.TELEMATICS || [];
  const voice = corpus?.voice || null;
  const ws = corpus?.workspace || null;
  const hyps = ws?.hypotheses || [];
  const comments = ws?.comments || [];
  const fixLoop = metrics?.fix_loop || null;
  const fixSeries = (fixLoop?.series || []).filter((s) => !gap?.category || s.category === gap.category);
  const fix = fixResult?.fix_effectiveness || fixResult?.measured ? fixResult : null;

  const evidenceCount = (voice ? 1 : 0) + nhtsaRows.length + callRows.length + wRows.length + sRows.length + fRows.length + tRows.length + (topLot ? 1 : 0);

  // ── Correlation timeline: every signal as a node on one story ──
  const timeline = [];
  const pushTl = (ts, source, label, text, extra) => {
    const t = ts ? new Date(ts).getTime() : NaN;
    if (Number.isNaN(t)) return;
    timeline.push({ ts: t, iso: String(ts).slice(0, 10), source, label, text, extra });
  };
  for (const r of nhtsaRows) pushTl(r.occurred_at || r.received_at, "nhtsa", r.category || "Complaint", r.text, r.record_id);
  for (const r of callRows) pushTl(r.occurred_at || r.received_at, "live call", "Live call", r.text, r.record_id);
  for (const r of sRows) pushTl(r.occurred_at || r.received_at, "service", r.subcategory || "Service visit", r.text, r.record_id);
  for (const r of wRows) pushTl(r.occurred_at || r.received_at, "warranty", r.subcategory || "Warranty claim", r.text, r.record_id);
  for (const r of fRows) pushTl(r.occurred_at || r.received_at, "field", r.subcategory || "Field return", r.text, r.record_id);
  for (const r of tRows) pushTl(r.occurred_at || r.received_at, "telematics", r.subcategory || "Fault code", r.text, r.record_id);
  if (voice) pushTl(voice.ts, "voice", "Customer call", voice.text, voice.interaction_id);
  if (inv?.opened_at) pushTl(inv.opened_at, "system", `Investigation ${inv.investigation_id} opened`, inv.title, null);
  const advIssued = ctx?.backtest?.advisory?.issued_at;
  if (ctx?.backtest?.matched && advIssued) {
    pushTl(advIssued, "advisory", `Advisory ${ctx.backtest.advisory_id}`, ctx.backtest.advisory.summary, null);
  }
  timeline.sort((a, b) => a.ts - b.ts);
  const timelineShown = timeline.slice(0, 16);

  // ── Error codes: FAIL_CODE / OP_CODE frequency across corpus rows ──
  const codeCount = {};
  for (const r of [...wRows, ...sRows]) {
    const code = (r.subcategory || "").trim();
    if (code) codeCount[code] = (codeCount[code] || 0) + 1;
  }
  const codeBars = Object.entries(codeCount)
    .map(([label, value]) => ({ label, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 8);

  // ── Fix chart: weekly slice volume around the recorded fix ──
  const fixMeasured = fix?.fix_effectiveness?.measured || null;
  const fixAt = fix?.fix_effectiveness?.fix?.fixed_at || null;
  const fixWeekIdx = (() => {
    if (!fixAt || !ctx?.weekly?.length) return -1;
    const fixTs = new Date(fixAt).getTime();
    if (Number.isNaN(fixTs)) return -1;
    const asc = [...ctx.weekly].reverse();
    let idx = -1;
    asc.forEach((w, i) => {
      const m = /^(\d{4})-W(\d{2})$/.exec(String(w.iso_week || ""));
      if (!m) return;
      const d = new Date(Date.UTC(+m[1], 0, 4));
      d.setUTCDate(d.getUTCDate() - ((d.getUTCDay() + 6) % 7) + (+m[2] - 1) * 7);
      if (d.getTime() <= fixTs) idx = i;
    });
    return idx;
  })();
  const fixWeeksAsc = ctx?.weekly ? [...ctx.weekly].reverse() : [];

  return (
    <div>
      <header className="page-header">
        <div>
          <p className="muted mono" style={{ margin: 0, fontSize: 12 }}>
            {inv ? `ISSUE · ${inv.investigation_id}` : clusterId ? `ISSUE · CLUSTER #${clusterId}` : `ISSUE · ${routeId}`}
          </p>
          <h1>{title}</h1>
          <p className="sub">
            {gap?.getting_worse && <span className="chip red">getting worse</span>}{" "}
            {critical > 0 && <span className="chip red">{critical} critical</span>}{" "}
            {liveCount > 0 && <span className="chip">{liveCount} live cases</span>}{" "}
            {inv && <span className="chip">{inv.status} · {inv.investigation_id}</span>}
          </p>
          <p className="muted" style={{ fontSize: 11, marginTop: 4 }}>
            Counts include simulated rehearsal traffic (no-op when there is none).
          </p>
        </div>
      </header>

      {/* ── 1. What is failing ── */}
      <div className="panel">
        <h3 className="analytics-h">What is failing</h3>
        <div style={{ overflowX: "auto" }}>
          <table className="analytics-table">
            <tbody>
              <tr><th>System</th><td>{slice?.category || gap?.category || live?.category || ctx?.category || "—"}</td></tr>
              {ctx?.top_terms?.length > 0 && (
                <tr><th>Top terms</th><td className="mono">{ctx.top_terms.join(", ")}</td></tr>
              )}
              {ctx?.record_count > 0 && (
                <tr><th>Corpus volume</th><td className="mono">{ctx.record_count} records</td></tr>
              )}
              {makeMode && <tr><th>Make</th><td>{makeMode.value} <span className="muted">({makeMode.count} cases)</span></td></tr>}
              {!makeMode && slice?.entity_2 && <tr><th>Make</th><td>{slice.entity_2}</td></tr>}
              {modelMode && <tr><th>Model</th><td>{modelMode.value} <span className="muted">({modelMode.count} cases)</span></td></tr>}
              {!modelMode && slice?.entity_3 && <tr><th>Model</th><td>{slice.entity_3}</td></tr>}
              {yearRange && <tr><th>Model years</th><td className="mono">{yearRange}</td></tr>}
              {(brief?.flagged_weeks || []).length > 0 && (
                <tr>
                  <th>Flagged weeks</th>
                  <td className="mono">
                    {brief.flagged_weeks.slice(0, 6).map((w) => (
                      <span key={w.iso_week} style={{ display: "inline-block", marginRight: 12 }}>
                        {w.iso_week} · {w.record_count}
                        {w.baseline_mean != null ? ` / ${Number(w.baseline_mean).toFixed(1)}` : ""}
                        {w.z_score != null ? ` · z ${Number(w.z_score).toFixed(1)}` : ""}
                      </span>
                    ))}
                  </td>
                </tr>
              )}
              {topLot && (
                <tr>
                  <th>Top lot</th>
                  <td>
                    <span className="mono">{topLot.supplier} · {topLot.lot_id}</span>{" "}
                    <span className="muted">({topLot.failures} failures)</span>
                  </td>
                </tr>
              )}
              {gap?.severity_mix && (
                <tr>
                  <th>Severity</th>
                  <td className="mono">
                    Critical {gap.severity_mix.Critical ?? 0} · Medium {gap.severity_mix.Medium ?? 0} · Low {gap.severity_mix.Low ?? 0}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        {codeBars.length > 0 && (
          <div style={{ marginTop: 12 }}>
            <p className="muted" style={{ fontSize: 11, margin: "0 0 6px" }}>ERROR CODES (FAIL / OP)</p>
            <BarList data={codeBars} emptyText="" />
          </div>
        )}
      </div>

      {/* ── 2. Why it matters ── */}
      {(dollarsUsd || (leadW !== null && leadW !== undefined) || critical > 0) && (
        <div className="panel" style={{ marginTop: 18 }}>
          <h3 className="analytics-h">Why it matters</h3>
          <div className="analytics-kpis">
            {dollarsUsd && (
              <div className="analytics-kpi">
                <span className="analytics-kpi-label">$ at risk</span>
                <span className="analytics-kpi-value tone-danger">{dollarsUsd}</span>
              </div>
            )}
            {leadW !== null && leadW !== undefined && (
              <div className="analytics-kpi">
                <span className="analytics-kpi-label">Lead vs advisory</span>
                <span className="analytics-kpi-value">{leadW}w{advisory ? ` · ${advisory}` : ""}</span>
              </div>
            )}
            {critical > 0 && (
              <div className="analytics-kpi">
                <span className="analytics-kpi-label">Critical cases</span>
                <span className="analytics-kpi-value tone-danger">{critical}</span>
              </div>
            )}
          </div>
          {est?.total_risk_range_usd && (
            <p className="muted" style={{ fontSize: 12 }}>
              Range {fmtUSD(est.total_risk_range_usd[0])}–{fmtUSD(est.total_risk_range_usd[1])} · {est.case_count} cases in estimate
            </p>
          )}
          {ctx?.backtest?.matched && (
            <LeadRuler weeks={ctx.weekly} advisoryId={advisory} leadW={leadW} />
          )}
        </div>
      )}

      {/* ── 3. Evidence, more than one source ── */}
      {evidenceCount > 0 && (
        <div className="panel" style={{ marginTop: 18 }}>
          <h3 className="analytics-h">Evidence · {evidenceCount} items, {new Set([
            voice && "voice",
            nhtsaRows.length && "nhtsa",
            callRows.length && "live call",
            wRows.length && "warranty",
            sRows.length && "service",
            fRows.length && "field",
            tRows.length && "telematics",
            topLot && "supplier lot",
          ].filter(Boolean)).size} sources</h3>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <SourceCards rows={nhtsaRows} chip="nhtsa" />
            <SourceCards rows={callRows} chip="live call" />
            {voice && (
              <div className="reg-summary-card">
                <span className="chip">voice turn</span>
                <p style={{ margin: "8px 0" }}>“{voice.text}”</p>
                <p className="muted" style={{ margin: 0, fontSize: 12 }}>
                  <span className="mono">{voice.interaction_id}</span>
                  {voice.pack_version && <> · config <span className="mono">pack {String(voice.pack_version).slice(0, 12)}</span></>}
                  {voice.case_id && (
                    <> · <button type="button" className="link-btn" onClick={() => openConsole(voice.interaction_id)}>open in console</button></>
                  )}
                </p>
              </div>
            )}
            <SourceCards rows={wRows} chip="warranty" />
            <SourceCards rows={sRows} chip="service" />
            <SourceCards rows={fRows} chip="field" />
            <SourceCards rows={tRows} chip="telematics" />
            {(brief?.cross_source_links || []).slice(0, 5).map((lk, i) => (
              <p key={lk.entity_key || lk.record_ids?.join("|") || i} style={{ margin: 0, fontSize: 12 }}>
                <span className="chip">{lk.basis}</span>{" "}
                <span className="mono">{lk.entity_key || [lk.category, lk.entity_2, lk.entity_3].filter(Boolean).join(" · ")}</span>
                {" "}· {(lk.sources || []).join(", ")}
              </p>
            ))}
            {topLot && (
              <div className="reg-summary-card">
                <span className="chip">supplier lot</span>
                <p style={{ margin: "8px 0" }}>
                  <span className="mono">{topLot.supplier} · lot {topLot.lot_id}</span> — {topLot.failures} failures
                  {(topLot.top_categories || []).map((t) => ` · ${t.category} (${t.count})`).join("")}
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── 3b. One story: every signal on a timeline ── */}
      {timelineShown.length > 1 && (
        <div className="panel" style={{ marginTop: 18 }}>
          <h3 className="analytics-h">One story · every signal in order</h3>
          <ol className="tl">
            {timelineShown.map((n, i) => (
              <li key={`${n.ts}-${i}`} className="tl-row">
                <span className={`tl-dot tone-${n.source === "voice" ? "accent" : n.source === "warranty" ? "warn" : n.source === "service" ? "ok" : n.source === "advisory" ? "danger" : "muted"}`} aria-hidden="true" />
                <div className="tl-body">
                  <div className="tl-head">
                    <span className="chip">{n.source}</span>
                    <span className="mono muted" style={{ fontSize: 11 }}>{n.iso}</span>
                  </div>
                  <div className="tl-label">{n.label}</div>
                  {n.text && <div className="tl-text">“{String(n.text).slice(0, 220)}”</div>}
                </div>
              </li>
            ))}
          </ol>
          {timeline.length > timelineShown.length && (
            <p className="muted" style={{ fontSize: 12 }}>Showing {timelineShown.length} of {timeline.length} nodes.</p>
          )}
        </div>
      )}

      {/* ── 4. Qubot stamps ── */}      {voice && (
        <div className="panel" style={{ marginTop: 18 }}>
          <h3 className="analytics-h">Qubot stamps</h3>
          {!stamp && (
            <button type="button" className="btn btn-secondary btn-sm" disabled={stampBusy} onClick={verifyVoice}>
              {stampBusy ? "Verifying…" : `Verify voice evidence (${voice.interaction_id.slice(0, 12)}…)`}
            </button>
          )}
          {stamp?.error && <div className="empty err-text">{stamp.error}</div>}
          {stamp && !stamp.error && (
            <div className="row" style={{ gap: 8, alignItems: "center", flexWrap: "wrap" }}>
              {String(stamp.overall_verdict || "").toLowerCase().includes("mismatch") ? (
                <span className="chip red">mismatch</span>
              ) : (
                <span className="chip green">{stamp.overall_verdict || "grounded"}</span>
              )}
              <span className="muted" style={{ fontSize: 12 }}>
                grounded {stamp.grounded_actions ?? "—"}/{stamp.total_actions ?? "—"}
                {stamp.mismatch_actions ? ` · mismatch ${stamp.mismatch_actions}` : ""}
                {stamp.unverifiable_actions ? ` · unverifiable ${stamp.unverifiable_actions}` : ""}
              </span>
              <button type="button" className="ghost" disabled={stampBusy} onClick={verifyVoice}>
                Re-verify
              </button>
            </div>
          )}
        </div>
      )}

      {/* ── 5. Ownership ── */}
      {inv && (
        <div className="panel" style={{ marginTop: 18 }}>
          <h3 className="analytics-h">Ownership · <span className="mono">{inv.investigation_id}</span></h3>
          <div style={{ overflowX: "auto" }}>
            <table className="analytics-table">
              <tbody>
                <tr><th>Assignee</th><td>{ws?.assignee || inv.assignee || <span className="muted">Unassigned</span>}</td></tr>
                <tr><th>SLA due</th><td className="mono">{(ws?.sla_due_at || inv.sla_due_at || "").toString().slice(0, 10) || <span className="muted">—</span>}</td></tr>
                <tr><th>Status</th><td><span className="chip">{inv.status}</span> <span className="muted">· {inv.days_open ?? "—"} days open · {inv.case_count ?? "—"} cases</span></td></tr>
              </tbody>
            </table>
          </div>
          <form className="row" style={{ gap: 8, marginTop: 8, flexWrap: "wrap" }} onSubmit={saveWorkspace}>
            <input
              aria-label="Assignee"
              placeholder="Assignee"
              value={assigneeDraft}
              onChange={(e) => setAssigneeDraft(e.target.value)}
              disabled={wsBusy}
            />
            <input
              aria-label="SLA due"
              type="date"
              value={slaDraft}
              onChange={(e) => setSlaDraft(e.target.value)}
              disabled={wsBusy}
            />
            <input
              aria-label="Hypothesis"
              placeholder="Hypothesis"
              value={hypDraft}
              onChange={(e) => setHypDraft(e.target.value)}
              disabled={wsBusy}
              style={{ flex: 1, minWidth: 160 }}
            />
            <button
              type="submit"
              className="btn btn-secondary btn-sm"
              disabled={wsBusy || (assigneeDraft === savedAssignee && slaDraft === savedSla && !hypDraft.trim())}
            >
              {wsBusy ? "Saving…" : "Save"}
            </button>
          </form>
          {wsError && <div className="empty err-text">{wsError}</div>}
          {hyps.length > 0 && (
            <>
              <h3 className="analytics-h" style={{ marginTop: 14 }}>Hypotheses</h3>
              <ul className="analytics-rising">
                {hyps.map((h) => (
                  <li key={h.hypothesis_id}>
                    {h.body}{" "}
                    <span className={`chip ${h.status === "confirmed" ? "green" : h.status === "rejected" ? "red" : ""}`}>
                      {h.status}
                    </span>
                  </li>
                ))}
              </ul>
            </>
          )}
          <h3 className="analytics-h" style={{ marginTop: 14 }}>Discussion</h3>
          {comments.length === 0 && (
            <p className="muted" style={{ fontSize: 12 }}>No comments yet — note what the next owner should know.</p>
          )}
          {comments.length > 0 && (
            <ul className="analytics-rising">
              {comments.map((c) => (
                <li key={c.comment_id || `${c.author}-${c.created_at}`}>
                  {c.body}{" "}
                  <span className="muted" style={{ fontSize: 11 }}>
                    — {c.author}{c.created_at ? ` · ${String(c.created_at).slice(0, 10)}` : ""}
                  </span>
                </li>
              ))}
            </ul>
          )}
          <form className="row" style={{ gap: 8, marginTop: 8 }} onSubmit={postComment}>
            <input
              aria-label="Add a comment"
              placeholder="Add a comment for the next owner…"
              value={commentBody}
              onChange={(e) => setCommentBody(e.target.value)}
              disabled={commentBusy}
              style={{ flex: 1 }}
            />
            <button type="submit" className="btn btn-secondary btn-sm" disabled={commentBusy || !commentBody.trim()}>
              {commentBusy ? "Posting…" : "Post"}
            </button>
          </form>
          {(topLot || (capas && capas.length > 0)) && (
            <>
              <h3 className="analytics-h" style={{ marginTop: 14 }}>Supplier CAPA</h3>
              {(capas || [])
                .filter((c) => !topLot || c.supplier === topLot.supplier)
                .slice(0, 5)
                .map((c) => (
                  <div className="reg-summary-card" key={c.capa_id}>
                    <span className={`chip ${c.status === "open" ? "red" : "green"}`}>{c.status}</span>{" "}
                    <span className="mono" style={{ fontSize: 12 }}>{c.supplier}{c.lot_id ? ` · lot ${c.lot_id}` : ""}</span>
                    <p style={{ margin: "8px 0" }}>{c.request}</p>
                    {c.response && <p className="muted" style={{ margin: 0, fontSize: 12 }}>Response: {c.response}</p>}
                  </div>
                ))}
              {topLot && (
                <form className="row" style={{ gap: 8, marginTop: 8 }} onSubmit={openCapa}>
                  <input
                    aria-label="CAPA request"
                    placeholder={`Ask ${topLot.supplier} for containment on lot ${topLot.lot_id}…`}
                    value={capaReq}
                    onChange={(e) => setCapaReq(e.target.value)}
                    disabled={capaBusy}
                    style={{ flex: 1 }}
                  />
                  <button type="submit" className="btn btn-secondary btn-sm" disabled={capaBusy || !capaReq.trim()}>
                    {capaBusy ? "Opening…" : "Open CAPA"}
                  </button>
                </form>
              )}
            </>
          )}
        </div>
      )}

      {/* ── 6. Fix ── */}
      {((fixLoop && (fixLoop.reopen_rate !== null && fixLoop.reopen_rate !== undefined)) || fixSeries.length > 0 || inv?.status === "open" || inv?.status === "monitoring" || fix) && (
        <div className="panel" style={{ marginTop: 18 }}>
          <h3 className="analytics-h">Fix</h3>
          {fixLoop && fixLoop.reopen_rate !== null && fixLoop.reopen_rate !== undefined && (
            <p style={{ marginTop: 0 }}>
              Reopen rate <strong className="mono">{Math.round(fixLoop.reopen_rate * 100)}%</strong>
              <span className="muted"> · {fixLoop.resolved ?? 0} resolved · {fixLoop.reopened ?? 0} reopened</span>
            </p>
          )}
          {fixSeries.length > 0 && (
            <div style={{ overflowX: "auto" }}>
              <table className="analytics-table">
                <thead><tr><th>Fix</th><th>Before</th><th>After</th><th>Result</th></tr></thead>
                <tbody>
                  {fixSeries.slice(0, 5).map((s) => (
                    <tr key={s.fix_id}>
                      <td className="mono">{s.fix_id}</td>
                      <td className="mono">{s.before_count} ({s.before_rate})</td>
                      <td className="mono">{s.after_count} ({s.after_rate})</td>
                      <td>{s.improved ? <span className="chip green">improved</span> : <span className="chip">flat</span>}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {fix?.fix_effectiveness?.measured && (
            <div className="table-wrap">
              <table className="analytics-table">
                <tbody>
                  <tr>
                    <th>Status</th>
                    <td>
                      {fix.fix_effectiveness.measured.status === "improved"
                        ? <span className="chip green">improved</span>
                        : <span className="chip">{fix.fix_effectiveness.measured.status || "waiting"}</span>}
                    </td>
                  </tr>
                  <tr>
                    <th>Fixed</th>
                    <td className="mono">{fixAt ? String(fixAt).slice(0, 10) : "—"}</td>
                  </tr>
                  <tr>
                    <th>Measured</th>
                    <td className="mono">
                      {fix.fix_effectiveness.measured.measured_at
                        ? String(fix.fix_effectiveness.measured.measured_at).slice(0, 10)
                        : "—"}
                    </td>
                  </tr>
                  <tr>
                    <th>Before rate</th>
                    <td className="mono">
                      {fix.fix_effectiveness.measured.before_count ?? "—"}
                      {" · "}
                      {fix.fix_effectiveness.measured.before_rate ?? "—"}
                    </td>
                  </tr>
                  <tr>
                    <th>After rate</th>
                    <td className="mono">
                      {fix.fix_effectiveness.measured.after_count ?? "—"}
                      {" · "}
                      {fix.fix_effectiveness.measured.after_rate ?? "—"}
                    </td>
                  </tr>
                  <tr>
                    <th>Control difference</th>
                    <td className="mono">
                      {fix.fix_effectiveness.measured.diff_in_diff ?? "—"}
                    </td>
                  </tr>
                </tbody>
              </table>
              {!fix.fix_effectiveness.measured.conclusive && (
                <p className="muted" style={{ fontSize: 12 }}>
                  Early signal — post window still maturing, watch recurrence.
                </p>
              )}
            </div>
          )}
          {fix?.error && <div className="empty err-text">{fix.error}</div>}
          {fixMeasured && fixWeeksAsc.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <FixChart
                weeks={fixWeeksAsc}
                fixIdx={fixWeekIdx}
                before={fixMeasured.before_count}
                after={fixMeasured.after_count ?? fixMeasured.nowcasted_after_count}
                reopened={fixLoop?.reopened ?? 0}
              />
              {!fixMeasured.conclusive && (
                <p className="muted" style={{ fontSize: 12 }}>
                  Early signal — post window still maturing, watch recurrence.
                </p>
              )}
            </div>
          )}
          {(inv?.status === "open" || inv?.status === "monitoring") && !fix && (
            <form
              className="row"
              style={{ gap: 8, flexWrap: "wrap", marginTop: 8 }}
              onSubmit={(e) => {
                e.preventDefault();
                recordFix();
              }}
            >
              <input
                aria-label="Fix date"
                type="date"
                value={fixDate}
                onChange={(e) => setFixDate(e.target.value)}
                disabled={fixBusy}
              />
              <input
                aria-label="Fix note"
                placeholder="What changed"
                value={fixNote}
                onChange={(e) => setFixNote(e.target.value)}
                disabled={fixBusy}
              />
              <input
                aria-label="Control make"
                placeholder="Control make"
                value={fixControl}
                onChange={(e) => setFixControl(e.target.value)}
                disabled={fixBusy}
              />
              <button type="submit" className="btn btn-primary btn-sm" disabled={fixBusy}>
                {fixBusy ? "Recording…" : `Record fix & close ${inv.investigation_id}`}
              </button>
            </form>
          )}
        </div>
      )}

      {/* ── 7. Search the quality brain ── */}
      <div className="panel" style={{ marginTop: 18 }}>
        <h3 className="analytics-h">Search the quality brain</h3>
        <form
          className="row"
          style={{ gap: 8 }}
          onSubmit={(e) => {
            e.preventDefault();
            runSearch();
          }}
        >
          <input
            aria-label="Search quality history"
            placeholder="e.g. Show conversations similar to case_…"
            value={searchQ}
            onChange={(e) => setSearchQ(e.target.value)}
            disabled={searchBusy}
            style={{ flex: 1 }}
          />
          <button type="submit" className="btn btn-secondary btn-sm" disabled={searchBusy || !searchQ.trim()}>
            {searchBusy ? "Searching…" : "Search"}
          </button>
        </form>
        <div className="row" style={{ gap: 6, marginTop: 8, flexWrap: "wrap" }}>
          {[
            casesSample[0]?.case_id ? `Show conversations similar to ${casesSample[0].case_id}` : null,
            "Show open critical cases",
            "Which customers are most frustrated?",
          ]
            .filter(Boolean)
            .map((preset) => (
              <button
                key={preset}
                type="button"
                className="ghost"
                style={{ fontSize: 12 }}
                disabled={searchBusy}
                onClick={() => {
                  setSearchQ(preset);
                  runSearch(preset);
                }}
              >
                {preset}
              </button>
            ))}
        </div>
        {searchRes?.error && <div className="empty err-text">{searchRes.error}</div>}
        {searchRes?.answer && <p style={{ marginBottom: 6 }}>{searchRes.answer}</p>}
        {Array.isArray(searchRes?.rows) && searchRes.rows.length > 0 && (
          <div style={{ overflowX: "auto" }}>
            <table className="analytics-table">
              <thead>
                <tr>
                  {Object.keys(searchRes.rows[0]).slice(0, 5).map((k) => (
                    <th key={k}>{k}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {searchRes.rows.slice(0, 6).map((row, i) => (
                  <tr key={i}>
                    {Object.entries(row).slice(0, 5).map(([k, v]) => (
                      <td key={k} className="mono" style={{ fontSize: 12 }}>
                        {String(v ?? "—").slice(0, 60)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {searchRes && !searchRes.error && (
          <p className="muted" style={{ fontSize: 11, marginBottom: 0 }}>
            Deterministic warehouse search (intent {searchRes.intent || "—"}) — not a generative LLM.
          </p>
        )}
      </div>

      <div className="row" style={{ marginTop: 18, gap: 8 }}>
        <button type="button" className="ghost" onClick={() => openCases(clusterId ? { clusterId } : { q: inv?.investigation_id })}>
          Open member cases
        </button>
      </div>
    </div>
  );
}
