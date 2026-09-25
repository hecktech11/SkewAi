import { useEffect, useState } from "react";
import { apiHeaders } from "../src/apiAuth.js";
import { hashQueryObject, patchHashQuery } from "../src/ui/opsActions.js";

// ── Minimal markdown renderer (basic #, ##, ###, lists, tables, code) ──
// Returns an array of React nodes. Intentionally simple — no library.
// Exported for reuse (AnalyticsDesk digest panel).

export function renderMarkdown(md) {
  if (!md) return [<div className="empty" key="empty">No report.</div>];
  const lines = md.split(/\r?\n/);
  const out = [];
  let i = 0;
  let key = 0;

  function push(el) {
    out.push(<div key={key++}>{el}</div>);
  }

  while (i < lines.length) {
    const line = lines[i];

    // Code block (``` ... ```)
    if (line.trim().startsWith("```")) {
      const buf = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith("```")) {
        buf.push(lines[i]);
        i++;
      }
      i++; // skip closing ```
      push(
        <pre>
          <code>{buf.join("\n")}</code>
        </pre>
      );
      continue;
    }

    // Table (line with | followed by separator line of ---|--- )
    if (line.includes("|") && i + 1 < lines.length && /^\s*\|?[\s:|-]+\|?\s*$/.test(lines[i + 1]) && lines[i + 1].includes("-")) {
      const header = splitRow(line);
      i += 2; // skip header + separator
      const rows = [];
      while (i < lines.length && lines[i].includes("|")) {
        rows.push(splitRow(lines[i]));
        i++;
      }
      push(
        <table>
          <thead>
            <tr>
              {header.map((h) => (
                <th key={h}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((r, ri) => (
              <tr key={ri}>
                {r.map((c, ci) => (
                  <td key={ci}>{renderInline(c)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      );
      continue;
    }

    // Headings
    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) {
      const level = h[1].length;
      const text = h[2];
      if (level === 1) push(<h1>{renderInline(text)}</h1>);
      else if (level === 2) push(<h2>{renderInline(text)}</h2>);
      else if (level === 3) push(<h3>{renderInline(text)}</h3>);
      else push(<h3>{renderInline(text)}</h3>);
      i++;
      continue;
    }

    // Blockquote
    if (line.trim().startsWith(">")) {
      const buf = [];
      while (i < lines.length && lines[i].trim().startsWith(">")) {
        buf.push(lines[i].replace(/^\s*>\s?/, ""));
        i++;
      }
      push(<blockquote>{renderInline(buf.join(" "))}</blockquote>);
      continue;
    }

    // Unordered list
    if (/^\s*[-*]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s+/, ""));
        i++;
      }
      push(
        <ul>
          {items.map((it, idx) => (
            <li key={`${it}-${idx}`}>{renderInline(it)}</li>
          ))}
        </ul>
      );
      continue;
    }

    // Ordered list
    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+\.\s+/, ""));
        i++;
      }
      push(
        <ol>
          {items.map((it, idx) => (
            <li key={`${it}-${idx}`}>{renderInline(it)}</li>
          ))}
        </ol>
      );
      continue;
    }

    // Blank line
    if (line.trim() === "") {
      i++;
      continue;
    }

    // Paragraph (accumulate consecutive non-blank, non-special lines)
    const buf = [line];
    i++;
    while (
      i < lines.length &&
      lines[i].trim() !== "" &&
      !/^(#{1,6})\s+/.test(lines[i]) &&
      !/^\s*[-*]\s+/.test(lines[i]) &&
      !/^\s*\d+\.\s+/.test(lines[i]) &&
      !lines[i].includes("|") &&
      !lines[i].trim().startsWith("```") &&
      !lines[i].trim().startsWith(">")
    ) {
      buf.push(lines[i]);
      i++;
    }
    push(<p>{renderInline(buf.join(" "))}</p>);
  }
  return out;
}

function splitRow(line) {
  return line
    .trim()
    .replace(/^\||\|$/g, "")
    .split("|")
    .map((c) => c.trim());
}

// Inline formatting: **bold**, *italic*, `code`, links
// SECURITY: hrefs are allowlisted to http/https/relative only — javascript:,
// data:, vbscript: etc. render as plain text to block stored-XSS via audit
// markdown (which embeds customer/case text).
function isTestAudit(id) {
  return /int_test_|_test_/i.test(String(id || ""));
}

function safeHref(raw) {
  const href = String(raw || "").trim();
  if (!href) return null;
  // Relative links (#anchor, /path) are safe.
  if (href.startsWith("#") || href.startsWith("/") || href.startsWith("./") || href.startsWith("../")) return href;
  try {
    // Use a dummy base so bare "example.com/x" parses; require http/https.
    const u = new URL(href, "http://localhost");
    const proto = (u.protocol || "").toLowerCase();
    if (proto === "http:" || proto === "https:") {
      // Reject protocol-relative //evil and userinfo tricks that smuggle hosts.
      if (!href.toLowerCase().startsWith("http://") && !href.toLowerCase().startsWith("https://")) return null;
      if (href.includes("@") && !href.startsWith("http")) return null;
      return href;
    }
  } catch {
    return null;
  }
  return null;
}

function renderInline(text) {
  if (!text) return null;
  // Tokenize into nodes by walking regex matches.
  const tokens = [];
  const re = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)]+\)|MISMATCH|🚩|✅)/g;
  let last = 0;
  let m;
  let i = 0;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) tokens.push(text.slice(last, m.index));
    const tok = m[0];
    if (tok.startsWith("**")) {
      tokens.push(<strong key={i++}>{tok.slice(2, -2)}</strong>);
    } else if (tok.startsWith("`")) {
      tokens.push(<code key={i++}>{tok.slice(1, -1)}</code>);
    } else if (tok.startsWith("[")) {
      const lm = /\[([^\]]+)\]\(([^)]+)\)/.exec(tok);
      if (lm) {
        const href = safeHref(lm[2]);
        if (href) tokens.push(<a key={i++} href={href} target="_blank" rel="noreferrer noopener">{lm[1]}</a>);
        else tokens.push(tok);
      }
      else tokens.push(tok);
    } else if (tok === "MISMATCH") {
      tokens.push(<span key={i++} className="err-text" style={{ fontWeight: 600 }}>MISMATCH</span>);
    } else {
      tokens.push(tok);
    }
    last = m.index + tok.length;
  }
  if (last < text.length) tokens.push(text.slice(last));
  return <>{tokens}</>;
}

export default function AuditReports({ embedded }) {
  const [audits, setAudits] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedId, setSelectedId] = useState(null);
  const [report, setReport] = useState(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [mismatchOnly, setMismatchOnly] = useState(false);
  const [error, setError] = useState(null);
  const [exportStart, setExportStart] = useState("");
  const [exportEnd, setExportEnd] = useState("");
  const [exporting, setExporting] = useState(false);
  const [hasMore, setHasMore] = useState(false);
  const [nextOffset, setNextOffset] = useState(0);
  const [showTest, setShowTest] = useState(false);

  async function loadAudits({ append = false, offset = 0 } = {}) {
    setLoading(true);
    setError(null);
    try {
      const r = await fetch(`/api/frontline/audits?limit=100&offset=${offset}`, {
        headers: apiHeaders(),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      const list = d.audits || [];
      setAudits((prev) => (append ? [...prev, ...list] : list));
      const pag = d.pagination || {};
      setHasMore(Boolean(pag.has_more));
      setNextOffset(pag.next_offset != null ? pag.next_offset : offset + list.length);
      if (!append) {
        setSelectedId((cur) => {
          if (cur && list.some((a) => a.interaction_id === cur)) return cur;
          const pre = hashQueryObject().id;
          if (pre) return pre;
          const vis = list.filter((a) => !isTestAudit(a.interaction_id));
          return vis[0]?.interaction_id || null;
        });
      }
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  async function downloadExport(fmt = "json") {
    setExporting(true);
    setError(null);
    try {
      const params = new URLSearchParams({ limit: "100", format: fmt });
      if (exportStart) params.set("start", exportStart);
      if (exportEnd) params.set("end", exportEnd);
      const r = await fetch(`/api/frontline/audits/export?${params}`, {
        headers: apiHeaders(),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
      const body = await r.json();
      let blob;
      let filename;
      if (fmt === "csv" && body.csv) {
        blob = new Blob([body.csv], { type: "text/csv" });
        filename = `skewai-audit-export-${exportStart || "all"}-${exportEnd || "now"}.csv`;
      } else {
        blob = new Blob([JSON.stringify(body, null, 2)], {
          type: "application/json",
        });
        filename = `skewai-audit-export-${exportStart || "all"}-${exportEnd || "now"}.json`;
      }
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(String(e));
    } finally {
      setExporting(false);
    }
  }

  useEffect(() => {
    loadAudits();
    const preselect =
      sessionStorage.getItem("frontline:audit_interaction_id") || hashQueryObject().id;
    if (preselect) {
      sessionStorage.removeItem("frontline:audit_interaction_id");
      openAudit(preselect);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (selectedId) patchHashQuery({ id: selectedId });
  }, [selectedId]);

  useEffect(() => {
    if (selectedId) openAudit(selectedId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId]);

  async function openAudit(interactionId) {
    setSelectedId(interactionId);
    setReport(null);
    setReportLoading(true);
    setError(null);
    try {
      const r = await fetch(`/api/frontline/audits/${interactionId}`, {
        headers: apiHeaders(),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const d = await r.json();
      setReport(d.report_markdown || "");
    } catch (e) {
      setError(String(e));
    } finally {
      setReportLoading(false);
    }
  }

  // Filter for mismatch verdict: we don't have a verdict column in the audits
  // list response, so we filter client-side by checking the loaded report
  // for "MISMATCH". For unloaded rows we keep them (so user can click to reveal).
  // Keep unopened rows (no verdict yet). Hide only confirmed non-mismatches.
  const testHidden = audits.filter((a) => isTestAudit(a.interaction_id)).length;
  const visibleAudits = audits.filter((a) => {
    if (!showTest && isTestAudit(a.interaction_id)) return false;
    if (mismatchOnly && a._hasMismatch === false) return false;
    return true;
  });

  // If we just loaded a report and mismatchOnly is on, mark whether it has a mismatch.
  useEffect(() => {
    if (!report || !selectedId) return;
    const hasMismatch = /MISMATCH/i.test(report);
    setAudits((prev) =>
      prev.map((a) =>
        a.interaction_id === selectedId ? { ...a, _hasMismatch: hasMismatch } : a
      )
    );
  }, [report, selectedId]);

  const selectedAudit = audits.find((a) => a.interaction_id === selectedId);

  return (
    <div style={{ display: "grid", gridTemplateColumns: "360px 1fr", gap: 16, minHeight: "70vh" }}>
      {/* ── List ─────────────────────────────────────────────────────── */}
      <div className="panel" style={{ padding: 0, overflow: "hidden", display: "flex", flexDirection: "column" }}>
        {!embedded && (
          <div className="page-header" style={{ padding: 16, margin: 0 }}>
            <h1 style={{ fontSize: 16 }}>Audit Reports</h1>
          </div>
        )}
        <div style={{ padding: "0 16px 12px" }}>
          <label className="check-row" style={{ fontSize: 12 }}>
            <input
              type="checkbox"
              checked={mismatchOnly}
              onChange={(e) => setMismatchOnly(e.target.checked)}
              aria-label="Show only MISMATCH verdicts"
            />
            <span className="muted">Mismatches only</span>
          </label>
          {testHidden > 0 && (
            <label className="check-row" style={{ fontSize: 12, marginTop: 8 }}>
              <input
                type="checkbox"
                checked={showTest}
                onChange={(e) => setShowTest(e.target.checked)}
                aria-label="Show test reports"
              />
              <span className="muted">Show {testHidden} test reports</span>
            </label>
          )}
          <div style={{ marginTop: 10, fontSize: 11 }}>
            <div className="muted" style={{ marginBottom: 4 }}>
              Pilot export (JSON / CSV + SHA256 manifest)
            </div>
            <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
              <input
                type="date"
                value={exportStart}
                onChange={(e) => setExportStart(e.target.value)}
                aria-label="Export start date"
                style={{ fontSize: 11, maxWidth: 130 }}
              />
              <input
                type="date"
                value={exportEnd}
                onChange={(e) => setExportEnd(e.target.value)}
                aria-label="Export end date"
                style={{ fontSize: 11, maxWidth: 130 }}
              />
              <button
                className="ghost"
                disabled={exporting}
                onClick={() => downloadExport("json")}
              >
                {exporting ? "…" : "↓ JSON"}
              </button>
              <button
                className="ghost"
                disabled={exporting}
                onClick={() => downloadExport("csv")}
              >
                {exporting ? "…" : "↓ CSV"}
              </button>
            </div>
          </div>
          {error && (
            <div className="err-text" style={{ fontSize: 12, marginTop: 8 }}>
              {error}
            </div>
          )}
        </div>
        <div className="scroll-y" style={{ flex: 1 }}>
          {loading && <div className="empty">loading…</div>}
          {!loading && visibleAudits.length === 0 && (
            <div className="empty">
              {testHidden > 0 && !showTest ? "Test contacts are hidden." : "No audit reports."}
            </div>
          )}
          {visibleAudits.map((a) => (
            <div
              key={a.interaction_id}
              role="button"
              tabIndex={0}
              className={"interaction-card" + (a.interaction_id === selectedId ? " selected" : "")}
              style={{ margin: "0 12px 8px" }}
              onClick={() => openAudit(a.interaction_id)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  openAudit(a.interaction_id);
                }
              }}
            >
              <div className="top">
                <span className="mono" style={{ fontSize: 11 }}>
                  {a.interaction_id}
                </span>
                {a._hasMismatch && <span className="chip red">MISMATCH</span>}
              </div>
              <div className="meta" style={{ fontSize: 11, color: "var(--ink-faint)" }}>
                {isTestAudit(a.interaction_id) ? "Test contact" : "Contact audit"}
              </div>
            </div>
          ))}
          {hasMore && (showTest || visibleAudits.length > 0) && (
            <div style={{ padding: "8px 12px" }}>
              <button
                type="button"
                className="ghost"
                disabled={loading}
                onClick={() => loadAudits({ append: true, offset: nextOffset })}
              >
                {loading ? "Loading…" : "Load more"}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* ── Report viewer ─────────────────────────────────────────────── */}
      <div className="panel" style={{ overflow: "auto", maxHeight: "78vh" }}>
        {!selectedId && (
          <div className="empty">
            {visibleAudits.length === 0
              ? showTest
                ? "No audit reports."
                : "No operator reports yet. Test contacts are hidden."
              : "Pick a report from the list."}
          </div>
        )}
        {selectedId && reportLoading && <div className="empty">loading report…</div>}
        {selectedId && !reportLoading && report === null && error && (
          <div className="empty err-text">{error}</div>
        )}
        {selectedId && !reportLoading && report !== null && (
          <>
            <div className="row" style={{ marginBottom: 14, justifyContent: "space-between" }}>
              <div className="mono faint" style={{ fontSize: 11 }}>
                {selectedAudit?.interaction_id}
              </div>
              <div className="row" style={{ gap: 6 }}>
                {/MISMATCH/i.test(report) && (
                  <span className="chip red">MISMATCH</span>
                )}
                {/✅|grounded/i.test(report) && !/MISMATCH/i.test(report) && (
                  <span className="chip green">grounded</span>
                )}
              </div>
            </div>
            <div className="md">{renderMarkdown(report)}</div>
          </>
        )}
      </div>
    </div>
  );
}
