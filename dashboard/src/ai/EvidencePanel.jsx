// [AI-LAYER] Photo evidence: upload (JPEG/PNG/WebP), list, SAM 3.1 box overlay,
// Muse Spark defect description mapped to the pack's component categories.
// Without a Meta key the upload still works and analysis shows "not run".
import { useCallback, useEffect, useRef, useState } from "react";
import { apiHeaders } from "../apiAuth.js";
import { aiGet, aiPostJson, aiUpload, aiStatus } from "./aiApi.js";
import MetaTag from "./MetaTag.jsx";

function listPath({ caseId, interactionId, investigationId }) {
  if (caseId) return `/api/cases/${encodeURIComponent(caseId)}/evidence`;
  if (interactionId) return `/api/interactions/${encodeURIComponent(interactionId)}/evidence`;
  if (investigationId) return `/api/ai/investigations/${encodeURIComponent(investigationId)}/evidence`;
  return null;
}

function useEvidenceImage(id) {
  // Fetched with the API key header (an <img src> cannot carry it), shown via an object URL.
  const [url, setUrl] = useState(null);
  useEffect(() => {
    let live = true;
    let obj = null;
    fetch(`/api/evidence/${encodeURIComponent(id)}/file`, { headers: apiHeaders(), credentials: "same-origin" })
      .then((r) => (r.ok ? r.blob() : null))
      .then((b) => { if (b && live) { obj = URL.createObjectURL(b); setUrl(obj); } })
      .catch(() => {});
    return () => { live = false; if (obj) URL.revokeObjectURL(obj); };
  }, [id]);
  return url;
}

function Overlay({ ev }) {
  const src = useEvidenceImage(ev.evidence_id);
  const seg = ev.analysis && ev.analysis.segmentation;
  const boxes = (seg && seg.boxes) || [];
  return (
    <div className="ai-evidence-frame" style={{ position: "relative", display: "inline-block", maxWidth: "100%" }}>
      {src ? <img
        src={src}
        alt={ev.filename || "evidence photo"}
        style={{ maxWidth: "100%", maxHeight: 260, display: "block", borderRadius: 6 }}
      /> : <div className="muted small" style={{ width: 160, height: 90 }}>loading photo…</div>}
      {boxes.filter((b) => b.pct).map((b, i) => (
        <div
          key={i}
          className="ai-sam-box"
          title={`${seg.concept} · SAM 3.1 match ${i + 1}`}
          style={{
            position: "absolute",
            left: `${b.pct.left}%`, top: `${b.pct.top}%`, width: `${b.pct.width}%`, height: `${b.pct.height}%`,
            border: "2px solid #ff4d4f", boxShadow: "0 0 0 1px rgba(0,0,0,.4)", borderRadius: 3, pointerEvents: "none",
          }}
        />
      ))}
    </div>
  );
}

function Analysis({ ev, enabled, onAnalyze, busy }) {
  const a = ev.analysis || {};
  const v = a.vision;
  const seg = a.segmentation;
  const models = [v && v.model_id, seg && seg.model_id].filter(Boolean);
  if (ev.analysis_status === "not_run") {
    return (
      <div className="muted small">
        Stored and attached. {enabled ? "Analysis not run yet." : "Analysis not run (no Meta key) - add META_API_KEY to enable SAM 3.1 + Muse Spark."}
        {enabled && <button className="ghost" style={{ marginLeft: 8 }} disabled={busy} onClick={onAnalyze}>Analyze</button>}
      </div>
    );
  }
  if (ev.analysis_status === "pending") return <div className="muted small">Analyzing with Muse Spark + SAM 3.1…</div>;
  return (
    <div className="small" style={{ marginTop: 6 }}>
      <MetaTag mode={models.length ? "enhanced" : "deterministic"} models={models} fallbackReason={ev.analysis_status} />
      {v && (
        <div style={{ marginTop: 6 }}>
          <div><strong>{v.damaged_part || "part"}</strong>: {v.defect_description}</div>
          <div className="muted">
            Proposed category {v.proposed_category} ({Math.round((v.confidence || 0) * 100)}%)
            {v.mapped_category ? <> → pack category <span className="chip">{v.mapped_category}</span></> : " → not mapped (low confidence or not in pack)"}
          </div>
          {v.visible_hazard && <div className="chip sev-critical" style={{ marginTop: 4 }}>Visible hazard: {v.hazard_reason || "flagged by model"}</div>}
        </div>
      )}
      {seg && <div className="muted">SAM 3.1 · "{seg.concept}" · {seg.count} region(s)</div>}
      {(a.errors || []).length > 0 && <div className="muted">Partial: {a.errors.join(", ")}</div>}
      <div className="muted">Advisory only: the case category and severity are never changed by photo analysis.</div>
    </div>
  );
}

export default function EvidencePanel({ caseId, interactionId, investigationId, readOnly = false, title = "Photo evidence" }) {
  const [rows, setRows] = useState([]);
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const [enabled, setEnabled] = useState(false);
  const fileRef = useRef(null);
  const path = listPath({ caseId, interactionId, investigationId });

  const load = useCallback(async () => {
    if (!path) return;
    try {
      const d = await aiGet(path);
      setRows(d.evidence || []);
    } catch (e) {
      setErr(`Could not load evidence (${e.message})`);
    }
  }, [path]);

  useEffect(() => { load(); aiStatus().then((s) => setEnabled(s.mode === "enhanced")); }, [load]);
  useEffect(() => {
    if (!rows.some((r) => r.analysis_status === "pending")) return undefined;
    const t = setTimeout(load, 2500);
    return () => clearTimeout(t);
  }, [rows, load]);

  async function onFile(e) {
    const f = e.target.files && e.target.files[0];
    if (!f || !path || investigationId) return;
    setBusy(true); setErr("");
    try {
      await aiUpload(path, f);
      await load();
    } catch (ex) {
      setErr(`Upload rejected: ${ex.message}`);
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  async function analyze(id) {
    setBusy(true);
    try { await aiPostJson(`/api/evidence/${encodeURIComponent(id)}/analyze`, {}); await load(); }
    catch (ex) { setErr(`Analysis failed: ${ex.message}`); }
    finally { setBusy(false); }
  }

  if (!path) return null;
  return (
    <section className="ai-evidence-panel" data-testid="evidence-panel" style={{ marginTop: 12 }}>
      <h3 className="analytics-h" style={{ display: "flex", gap: 8, alignItems: "center" }}>
        {title} <span className="muted small">({rows.length})</span>
      </h3>
      {!readOnly && !investigationId && (
        <label className="ghost" style={{ display: "inline-block", cursor: "pointer" }}>
          <input ref={fileRef} type="file" accept="image/jpeg,image/png,image/webp" onChange={onFile} disabled={busy}
            style={{ display: "none" }} aria-label="Upload photo evidence" />
          <span className="chip">{busy ? "Uploading…" : "+ Add photo (JPEG/PNG/WebP, ≤ 8 MB)"}</span>
        </label>
      )}
      {err && <div className="banner banner-warn small" role="alert" style={{ marginTop: 6 }}>{err}</div>}
      {rows.length === 0 && <div className="muted small" style={{ marginTop: 6 }}>No photos attached.</div>}
      {readOnly && rows.length > 3 && <div className="muted small">Showing 3 of {rows.length}; open the cases for the rest.</div>}
      {(readOnly ? rows.slice(0, 3) : rows).map((ev) => (
        <div key={ev.evidence_id} className="ai-evidence-item" style={{ marginTop: 10 }}>
          <Overlay ev={ev} />
          <div className="mono muted small">{ev.evidence_id} · {ev.mime} · {ev.width}×{ev.height}{ev.case_id ? ` · ${ev.case_id}` : ""}</div>
          <Analysis ev={ev} enabled={enabled} busy={busy} onAnalyze={() => analyze(ev.evidence_id)} />
        </div>
      ))}
    </section>
  );
}
