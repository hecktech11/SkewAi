import { useEffect, useRef, useState } from "react";
import { apiHeaders } from "../src/apiAuth.js";
import { INCLUDE_SIM_KEY, SS, consumeSession, parseLocationHash, patchHashQuery } from "../src/ui/opsActions.js";
import {
  caseDescription,
  cleanFollowupDraft,
  packLabel,
  safetyFlagLabel,
  statusLabel,
} from "../src/ui/labels.js";
import { useDebounced, useFocusTrap, useLocalStorage } from "../src/ui/hooks.js";
// [AI-LAYER] photo evidence panel (optional Meta AI layer; upload works without a key)
import EvidencePanel from "../src/ai/EvidencePanel.jsx";

const SEV = ["All", "Critical", "Medium", "Low"];
const STATUS = ["All", "open", "pending_followup", "closed"];

function SevBadge({ severity }) {
  const s = (severity || "").toLowerCase();
  const cls = s === "critical" ? "red" : s === "medium" ? "orange" : "blue";
  return <span className={"chip " + cls}>{severity || "—"}</span>;
}

function PriorityPill({ priority }) {
  return (
    <span className="chip mono">
      P{priority}
    </span>
  );
}

function ChannelBadge({ channel }) {
  if (!channel) return <span className="faint">—</span>;
  const sim = channel === "simulated";
  return (
    <span className={"chip " + (sim ? "purple" : "teal")}>
      {sim ? "simulated" : channel}
    </span>
  );
}

export default function CaseQueue() {
  const [cases, setCases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [filterSev, setFilterSev] = useState("All");
  const [filterStatus, setFilterStatus] = useState("All");
  const [searchQ, setSearchQ] = useState("");
  const searchDebounced = useDebounced(searchQ, 400);
  const [selectedId, setSelectedId] = useState(null);
  const drawerRef = useRef(null);
  const skipHashWrite = useRef(true);
  useFocusTrap(drawerRef, !!selectedId, () => setSelectedId(null));
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [followupDraft, setFollowupDraft] = useState("");
  const [noteBody, setNoteBody] = useState("");
  const [error, setError] = useState(null);
  const [exportMsg, setExportMsg] = useState(null);
  const [exporting, setExporting] = useState(false);
  const [actionMsg, setActionMsg] = useState(null);
  const [includeSimulated, setIncludeSimulated] = useLocalStorage(INCLUDE_SIM_KEY, false);
  const [clusterFilter, setClusterFilter] = useState("");
  const [hasMore, setHasMore] = useState(false);
  const [nextOffset, setNextOffset] = useState(0);

  async function load({ append = false, offset = 0 } = {}) {
    setLoading(true);
    const params = new URLSearchParams({
      limit: "100",
      offset: String(offset),
      include_simulated: includeSimulated ? "true" : "false",
    });
    if (filterSev !== "All") params.set("severity", filterSev);
    if (filterStatus !== "All") params.set("status", filterStatus);
    if (searchDebounced.trim()) params.set("q", searchDebounced.trim());
    if (clusterFilter) params.set("cluster", clusterFilter);
    try {
      const r = await fetch(`/api/frontline/cases?${params.toString()}`, {
        headers: apiHeaders(),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      const list = d.cases || [];
      setCases((prev) => (append ? [...prev, ...list] : list));
      const pag = d.pagination || {};
      setHasMore(Boolean(pag.has_more));
      setNextOffset(pag.next_offset != null ? pag.next_offset : offset + list.length);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  // Deep-link from hash, then sessionStorage fallback.
  useEffect(() => {
    const { params } = parseLocationHash();
    const sev = params.get("severity") || consumeSession(SS.caseSev);
    const st = params.get("status") || consumeSession(SS.caseStatus);
    const q = params.get("q") || consumeSession(SS.caseSearch);
    const pick = params.get("id") || consumeSession(SS.caseSelect);
    const cluster = params.get("cluster");
    if (sev) setFilterSev(sev);
    if (st) setFilterStatus(st);
    if (q) setSearchQ(q);
    if (cluster) setClusterFilter(cluster);
    if (pick) setTimeout(() => openCase(pick), 0);
    skipHashWrite.current = true;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    load({ append: false, offset: 0 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filterSev, filterStatus, searchDebounced, includeSimulated, clusterFilter]);

  useEffect(() => {
    if (skipHashWrite.current) {
      skipHashWrite.current = false;
      return;
    }
    patchHashQuery({
      id: selectedId || null,
      severity: filterSev !== "All" ? filterSev : null,
      status: filterStatus !== "All" ? filterStatus : null,
      q: searchDebounced.trim() || null,
      cluster: clusterFilter || null,
    });
  }, [selectedId, filterSev, filterStatus, searchDebounced, clusterFilter]);

  useEffect(() => {
    if (!selectedId) return undefined;
    const onEsc = () => setSelectedId(null);
    window.addEventListener("frontline-escape", onEsc);
    const onKey = (e) => {
      if (e.key === "Escape") setSelectedId(null);
    };
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("frontline-escape", onEsc);
      window.removeEventListener("keydown", onKey);
    };
  }, [selectedId]);

  async function downloadCasesCsv() {
    setActionMsg(null);
    try {
      const params = new URLSearchParams({
        limit: "500",
        include_simulated: includeSimulated ? "true" : "false",
      });
      if (filterSev !== "All") params.set("severity", filterSev);
      if (filterStatus !== "All") params.set("status", filterStatus);
      if (searchDebounced.trim()) params.set("q", searchDebounced.trim());
      if (clusterFilter) params.set("cluster", clusterFilter);
      const r = await fetch(`/api/frontline/cases/export?${params.toString()}`, {
        headers: apiHeaders(),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "cases_export.csv";
      a.click();
      URL.revokeObjectURL(url);
      setError(null);
      setActionMsg(`CSV downloaded (${r.headers.get("X-Export-Count") || "?"} rows)`);
    } catch (e) {
      setActionMsg(null);
      setError(String(e));
    }
  }

  async function exportCase(caseId) {
    setExportMsg(null);
    setExporting(true);
    try {
      const r = await fetch(
        `/api/frontline/connectors/export/${encodeURIComponent(caseId)}`,
        { method: "POST", headers: apiHeaders() }
      );
      const text = await r.text();
      let body = {};
      try {
        body = JSON.parse(text);
      } catch {
        body = { detail: text };
      }
      if (!r.ok) {
        throw new Error(body.detail || `HTTP ${r.status}`);
      }
      setExportMsg(
        `Exported ${body.delivery_id || caseId} → ${body.status || "ok"} (${body.sink || "outbox"})`
      );
    } catch (e) {
      setExportMsg(String(e));
    } finally {
      setExporting(false);
    }
  }

  async function openCase(caseId) {
    setSelectedId(caseId);
    setDetail(null);
    setFollowupDraft("");
    setNoteBody("");
    setExportMsg(null);
    setActionMsg(null);
    setDetailLoading(true);
    try {
      const r = await fetch(`/api/frontline/cases/${caseId}`, {
        headers: apiHeaders(),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      setDetail(d);
      setFollowupDraft(cleanFollowupDraft(d.followup_draft || ""));
    } catch (e) {
      setError(String(e));
    } finally {
      setDetailLoading(false);
    }
  }

  async function saveCasePatch(patch) {
    if (!selectedId) return;
    setActionMsg(null);
    try {
      const r = await fetch(`/api/frontline/cases/${encodeURIComponent(selectedId)}`, {
        method: "PATCH",
        headers: { ...apiHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify(patch),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
      const d = await r.json();
      setDetail((prev) => ({ ...prev, ...d, notes: prev?.notes || [] }));
      setError(null);
      setActionMsg("Case updated.");
      await load();
    } catch (e) {
      setActionMsg(null);
      setError(String(e));
    }
  }

  async function addNote() {
    if (!selectedId || !noteBody.trim()) return;
    setActionMsg(null);
    try {
      const r = await fetch(
        `/api/frontline/cases/${encodeURIComponent(selectedId)}/notes`,
        {
          method: "POST",
          headers: { ...apiHeaders(), "Content-Type": "application/json" },
          body: JSON.stringify({ body: noteBody.trim() }),
        }
      );
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
      const note = await r.json();
      setDetail((prev) =>
        prev
          ? { ...prev, notes: [note, ...(prev.notes || [])] }
          : prev
      );
      setNoteBody("");
      setError(null);
      setActionMsg("Note added.");
    } catch (e) {
      setActionMsg(null);
      setError(String(e));
    }
  }

  return (
    <div className="page-enter">
      <header className="page-header">
        <div>
          <h1>Case queue</h1>
          <p className="sub">
            Triage open cases with severity chips, open the drawer for follow-ups, export CSV for ops.
          </p>
        </div>
        <div className="page-actions">
          <span className="build-stamp">Queue filters</span>
          <label className="check-row" style={{ fontSize: 12, margin: 0 }}>
            <input
              type="checkbox"
              checked={includeSimulated}
              onChange={(e) => setIncludeSimulated(e.target.checked)}
              aria-label="Include simulated traffic"
            />
            Include simulated
          </label>
          <button type="button" className="ghost" onClick={downloadCasesCsv}>
            Export CSV
          </button>
          <button type="button" onClick={() => load({ append: false, offset: 0 })} disabled={loading}>
            {loading ? "Loading…" : "Refresh"}
          </button>
        </div>
      </header>

      {error && (
        <div className="banner banner-error" role="alert">
          {error}{" "}
          <button type="button" className="ghost" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}

      {actionMsg && (
        <div className="banner banner-ok" role="status">
          {actionMsg}
        </div>
      )}

      <div className="filter-bar">
        <div className="field" style={{ flex: 1, minWidth: 200 }}>
          <label>Search</label>
          <input
            value={searchQ}
            onChange={(e) => setSearchQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && load()}
            placeholder="case id, category, text…"
            aria-label="Search cases"
          />
        </div>
        <div className="field">
          <label>Severity</label>
          <div className="filter-chips" role="group" aria-label="Severity filter">
            {SEV.map((s) => (
              <button
                key={s}
                type="button"
                className={
                  "filter-chip" +
                  (filterSev === s ? " active" : "") +
                  (s === "Critical" ? " danger" : "")
                }
                onClick={() => setFilterSev(s)}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
        <div className="field">
          <label>Status</label>
          <div className="filter-chips" role="group" aria-label="Status filter">
            {STATUS.map((s) => (
              <button
                key={s}
                type="button"
                className={"filter-chip" + (filterStatus === s ? " active" : "")}
                onClick={() => setFilterStatus(s)}
              >
                {statusLabel(s)}
              </button>
            ))}
          </div>
        </div>
        <button className="primary" onClick={() => load({ append: false, offset: 0 })} disabled={loading}>
          Search
        </button>
        <span className="toolbar-count">{loading ? "loading…" : `${cases.length} cases`}</span>
        {clusterFilter ? (
          <button type="button" className="ghost" onClick={() => setClusterFilter("")}>
            Cluster {clusterFilter} ×
          </button>
        ) : null}
      </div>

      <div className="panel" style={{ padding: 0, overflow: "hidden" }}>
        <div className="table-wrap case-table-desktop">
          <table className="table-clickable">
            <thead>
              <tr>
                <th>Case ID</th>
                <th>Category</th>
                <th>Severity</th>
                <th>Priority</th>
                <th>Safety flags</th>
                <th>Advisory</th>
                <th>Cluster</th>
                <th>Investigation</th>
                <th>Peak frustration</th>
                <th>Channel</th>
              </tr>
            </thead>
            <tbody>
              {cases.length === 0 && !loading && (
                <tr>
                  <td colSpan={10}>
                    <div className="hero-empty" style={{ margin: 16, border: "none" }}>
                      <h3>No cases match</h3>
                      <p>Try clearing filters, or create a contact from Voice agent.</p>
                      <button type="button" className="primary" onClick={() => (window.location.hash = "call")}>
                        Start a contact
                      </button>
                    </div>
                  </td>
                </tr>
              )}
              {cases.map((c) => (
                <tr
                  key={c.case_id}
                  className={c.case_id === selectedId ? "selected" : ""}
                  tabIndex={0}
                  onClick={() => openCase(c.case_id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      openCase(c.case_id);
                    }
                  }}
                >
                  <td className="mono">{c.case_id}</td>
                  <td>{c.category || "—"}</td>
                  <td>
                    <SevBadge severity={c.severity} />
                  </td>
                  <td>
                    <PriorityPill priority={c.priority} />
                  </td>
                  <td>
                    {c.safety_flags && Object.keys(c.safety_flags).length ? (
                      Object.entries(c.safety_flags)
                        .filter(([, v]) => v)
                        .map(([k]) => (
                          <span className="chip red" key={k}>
                            {safetyFlagLabel(k)}
                          </span>
                        ))
                    ) : (
                      <span className="faint">—</span>
                    )}
                  </td>
                  <td className="mono">
                    {c.advisory_match_id ? (
                      <span className="chip green">{c.advisory_match_id}</span>
                    ) : (
                      <span className="faint">—</span>
                    )}
                  </td>
                  <td className="mono">
                    {c.cluster_match_id != null ? (
                      <span className="chip teal">cluster {c.cluster_match_id}</span>
                    ) : (
                      <span className="faint">—</span>
                    )}
                  </td>
                  <td className="mono">
                    {c.investigation_id ? (
                      <span className="chip teal">{c.investigation_id}</span>
                    ) : (
                      <span className="faint">—</span>
                    )}
                  </td>
                  <td className="mono">
                    {c.peak_frustration != null ? c.peak_frustration.toFixed(2) : "—"}
                  </td>
                  <td>
                    <ChannelBadge channel={c.channel} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="case-card-list">
          {cases.length === 0 && !loading && (
            <div className="hero-empty" style={{ margin: 8, border: "none" }}>
              <h3>No cases match</h3>
              <p>Try clearing filters, or start a contact.</p>
              <button type="button" className="primary" onClick={() => (window.location.hash = "call")}>
                Start a contact
              </button>
            </div>
          )}
          {cases.map((c) => (
            <button
              type="button"
              key={`card-${c.case_id}`}
              className={"case-card" + (c.case_id === selectedId ? " selected" : "")}
              onClick={() => openCase(c.case_id)}
            >
              <div className="case-card-top">
                <span className="case-card-cat">{c.category || "Uncategorized"}</span>
                <SevBadge severity={c.severity} />
              </div>
              <div className="case-card-id">{c.case_id}</div>
              <div className="case-card-meta">
                <span>{statusLabel(c.status)}</span>
                {c.cluster_match_id != null ? <span>Cluster {c.cluster_match_id}</span> : null}
                <ChannelBadge channel={c.channel} />
              </div>
            </button>
          ))}
        </div>
        {hasMore && (
          <div className="row" style={{ padding: 12, justifyContent: "center" }}>
            <button
              type="button"
              className="ghost"
              disabled={loading}
              onClick={() => load({ append: true, offset: nextOffset })}
            >
              {loading ? "Loading…" : "Load more"}
            </button>
          </div>
        )}
      </div>

      {selectedId && (
        <div
          className="drawer-scrim"
          onMouseDown={(e) => {
            if (e.target === e.currentTarget) setSelectedId(null);
          }}
        >
          <div className="drawer" ref={drawerRef} role="dialog" aria-modal="true" aria-label="Case detail">
            <header>
              <div>
                <strong>{detail?.category || "Case"}</strong>{" "}
                {detail && <SevBadge severity={detail.severity} />}
                <div className="mono faint" style={{ fontSize: 11 }}>{selectedId}</div>
              </div>
              <button className="ghost icon-btn" onClick={() => setSelectedId(null)} aria-label="Close">
                ×
              </button>
            </header>
            <div className="body">
            {detailLoading && <div className="empty">loading…</div>}
            {!detailLoading && !detail && (
              <div className="empty">Case not found.</div>
            )}
            {detail && (
              <>
                <p className="muted" style={{ marginTop: 0 }}>
                  {detail.severity || "—"} · {statusLabel(detail.status)}
                  {detail.cluster_match_id != null ? ` · cluster ${detail.cluster_match_id}` : ""}
                  {detail.channel === "simulated" ? " · simulated" : ""}
                </p>
                <p>{caseDescription(detail.description_summary) || "No description."}</p>
                <div className="kvs">
                  <span className="k">pack</span>
                  <span className="v">{packLabel(detail.pack_id)}</span>
                  <span className="k">advisory</span>
                  <span className="v mono">{detail.advisory_match_id || "—"}</span>
                </div>

                {detail.safety_flags &&
                  Object.keys(detail.safety_flags).length > 0 && (
                    <>
                      <div className="h-divider" />
                      <h2 style={panelHeading}>Safety Flags</h2>
                      <div className="row">
                        {Object.entries(detail.safety_flags)
                          .filter(([, v]) => v)
                          .map(([k]) => (
                            <span className="chip red" key={k}>
                              {safetyFlagLabel(k)}
                            </span>
                          ))}
                      </div>
                    </>
                  )}

                <div className="h-divider" />
                <h2 style={panelHeading}>Status</h2>
                <div className="row" style={{ gap: 8, flexWrap: "wrap" }}>
                  {["open", "pending_followup", "closed"].map((s) => (
                    <button
                      key={s}
                      className={detail.status === s ? "primary" : "ghost"}
                      onClick={() => saveCasePatch({ status: s })}
                    >
                      {statusLabel(s)}
                    </button>
                  ))}
                </div>

                <div className="h-divider" />
                <h2 style={panelHeading} id="followup-draft-heading">Follow-up Draft</h2>
                <textarea
                  id="followup-draft-input"
                  aria-labelledby="followup-draft-heading"
                  aria-label="Follow-up Draft"
                  value={followupDraft}
                  onChange={(e) => setFollowupDraft(e.target.value)}
                  placeholder="Draft a follow-up message…"
                  style={{ width: "100%" }}
                />
                <button
                  className="ghost"
                  style={{ marginTop: 8 }}
                  onClick={() => saveCasePatch({ followup_draft: followupDraft })}
                >
                  Save follow-up
                </button>

                {/* [AI-LAYER] photo evidence for this case */}
                <EvidencePanel caseId={detail.case_id || selectedId} />

                <div className="h-divider" />
                <h2 style={panelHeading} id="operator-notes-heading">Operator notes</h2>
                <textarea
                  id="operator-notes-input"
                  aria-labelledby="operator-notes-heading"
                  aria-label="Operator notes"
                  value={noteBody}
                  onChange={(e) => setNoteBody(e.target.value)}
                  placeholder="Add a note for the pilot team…"
                  style={{ width: "100%", minHeight: 60 }}
                />
                <button
                  className="primary"
                  style={{ marginTop: 8 }}
                  disabled={!noteBody.trim()}
                  onClick={addNote}
                >
                  Add note
                </button>
                {(detail.notes || []).length > 0 && (
                  <ul style={{ marginTop: 12, paddingLeft: 18, fontSize: 12 }}>
                    {detail.notes.map((n) => (
                      <li key={n.note_id} style={{ marginBottom: 8 }}>
                        <span className="mono muted">{n.created_at}</span>{" "}
                        <strong>{n.author}</strong>: {n.body}
                      </li>
                    ))}
                  </ul>
                )}
              </>
            )}
          </div>
          <div className="footer">
            {exportMsg && (
              <span className="muted" style={{ fontSize: 11, marginRight: 8 }}>
                {exportMsg}
              </span>
            )}
            {detail && (
              <button
                className="ghost"
                disabled={exporting}
                onClick={() => exportCase(detail.case_id || selectedId)}
                title="Push case JSON to connector outbox / webhook"
              >
                {exporting ? "exporting…" : "Export"}
              </button>
            )}
            {detail?.audit_report_url && (
              <a
                className="button ghost"
                style={{ textDecoration: "none" }}
                href={`#audits?tab=reports&id=${encodeURIComponent(detail.interaction_id || "")}`}
                onClick={() => {
                  sessionStorage.setItem(
                    "frontline:audit_interaction_id",
                    detail.interaction_id
                  );
                  setSelectedId(null);
                }}
              >
                View audit report →
              </a>
            )}
            <button className="ghost" onClick={() => setSelectedId(null)}>
              Close
            </button>
          </div>
          </div>
        </div>
      )}
    </div>
  );
}

const panelHeading = {
  margin: "0 0 8px",
  fontSize: 12,
  fontWeight: 600,
  color: "var(--text-dim)",
  textTransform: "uppercase",
  letterSpacing: "0.06em",
};
