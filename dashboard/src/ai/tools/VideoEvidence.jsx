// [AI-LAYER] Short video evidence (5–10 s) attached to a case. Analysis is off unless
// AI_VIDEO_ANALYSIS=1 and a Meta key are set; Muse Spark video input is UNVERIFIED.
import { useState } from "react";
import { aiGet, aiPostJson, aiUpload } from "../aiApi.js";

export default function VideoEvidence({ hashQuery }) {
  const [caseId, setCaseId] = useState(() => (hashQuery && hashQuery.case) || "");
  const [list, setList] = useState(null);
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);
  const load = async (id = caseId) => { if (id.trim()) setList(await aiGet(`/api/ai/cases/${encodeURIComponent(id.trim())}/video`)); };
  async function upload(e) {
    const f = e.target.files?.[0]; e.target.value = "";
    if (!f || !caseId.trim()) return;
    setBusy(true); setMsg("");
    try { const v = await aiUpload(`/api/ai/cases/${encodeURIComponent(caseId.trim())}/video`, f); setMsg(`Stored ${v.video_id} (${v.duration_s}s, ${v.analysis_status})`); await load(); }
    catch (x) { setMsg(x.message); } finally { setBusy(false); }
  }
  async function analyze(id) { try { await aiPostJson(`/api/ai/video/${id}/analyze`, {}); await load(); } catch (x) { setMsg(x.message); } }
  return (
    <div data-testid="video-evidence">
      <div className="muted small" style={{ marginBottom: 6 }}>mp4 / mov / webm, 1–10 s (aim for 5–10 s), size-capped. Duration is read from the file itself.</div>
      <div className="row" style={{ gap: 6, alignItems: "center", flexWrap: "wrap" }}>
        <input value={caseId} placeholder="case id" onChange={(e) => setCaseId(e.target.value)} style={{ width: 240 }} />
        <button className="ghost" onClick={() => load().catch((x) => setMsg(x.message))}>Load</button>
        <label className="primary" style={{ cursor: "pointer", padding: "4px 10px", borderRadius: 6 }}>
          {busy ? "Uploading…" : "Upload clip"}<input type="file" accept="video/mp4,video/quicktime,video/webm" hidden onChange={upload} disabled={busy || !caseId.trim()} />
        </label>
      </div>
      {msg && <div className="small" style={{ marginTop: 6 }}>{msg}</div>}
      {list && (
        <div style={{ marginTop: 8 }}>
          <div className="muted small">Video analysis: {list.analysis_enabled ? "enabled" : "off (clips stored without analysis)"} · {list.analysis_note}</div>
          {list.videos.length === 0 && <div className="muted small">No clips on this case.</div>}
          {list.videos.map((v) => (
            <div key={v.video_id} className="card" style={{ padding: 8, marginTop: 8 }}>
              <video src={v.file_url} controls width="320" preload="metadata" />
              <div className="small">{v.video_id} · {v.duration_s}s · {(v.size_bytes / 1024).toFixed(0)} KB · {v.analysis_status}</div>
              {v.analysis && <pre className="small" style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(v.analysis, null, 2)}</pre>}
              {list.analysis_enabled && v.analysis_status !== "done" && <button className="ghost" onClick={() => analyze(v.video_id)}>Analyze (unverified)</button>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
