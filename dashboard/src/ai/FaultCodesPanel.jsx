// [AI-LAYER] OBD-II fault codes (DTCs) on the case view: shown as evidence next to photos /
// transcript. Generic SAE J2012-style meanings only; unknown codes have no effect. A safety-system
// code can only RAISE severity; a code that disagrees with the complaint is flagged "needs review".
// No public dataset links DTCs to recalls, so no accuracy claim is made.
import { useCallback, useEffect, useState } from "react";
import { aiGet, aiPostJson } from "./aiApi.js";

export const DTC_RE = /^[PCBU][0-3][0-9A-F]{3}$/;

export function parseCodes(text) {
  const toks = String(text || "").toUpperCase().split(/[\s,;]+/).filter(Boolean);
  return { valid: toks.filter((t) => DTC_RE.test(t)), invalid: toks.filter((t) => !DTC_RE.test(t)) };
}

export default function FaultCodesPanel({ caseId, readOnly = false }) {
  const [data, setData] = useState(null);
  const [text, setText] = useState("");
  const [msg, setMsg] = useState("");
  const [busy, setBusy] = useState(false);
  const path = `/api/ai/cases/${encodeURIComponent(caseId)}/fault-codes`;
  const load = useCallback(() => { aiGet(path).then(setData).catch(() => setData(null)); }, [path]);
  useEffect(() => { load(); }, [load]);

  async function add(e) {
    e.preventDefault();
    const { valid, invalid } = parseCodes(text);
    if (invalid.length) { setMsg(`Invalid code(s): ${invalid.join(", ")} (format like C0035, P0300)`); return; }
    if (!valid.length) return;
    setBusy(true); setMsg("");
    try {
      const out = await aiPostJson(path, { fault_codes: valid });
      setMsg(out.raised ? `Severity raised ${out.severity_before} → ${out.severity_after} (safety-system code)` : "Codes added (severity unchanged)");
      setText(""); load();
    } catch (ex) { setMsg(`Rejected: ${ex.message}`); }
    finally { setBusy(false); }
  }

  const codes = (data && data.codes) || [];
  return (
    <div className="ai-fault-codes" data-testid="fault-codes-panel" style={{ marginTop: 12 }}>
      <h4 className="analytics-h">Fault codes (OBD-II) <span className="muted small">({codes.length})</span></h4>
      {codes.length === 0 && <div className="muted small">No fault codes attached.</div>}
      {codes.length > 0 && (
        <table className="table small">
          <thead><tr><th>Code</th><th>Generic meaning</th><th>Component</th><th>Safety</th><th>Check</th></tr></thead>
          <tbody>
            {codes.map((c) => (
              <tr key={c.code}>
                <td className="mono">{c.code}</td>
                <td>{c.meaning}{c.confidence ? <span className="muted"> ({c.confidence} confidence)</span> : null}</td>
                <td>{c.component || "—"}</td>
                <td>{c.safety || "—"}</td>
                <td>{c.needs_review ? <span className="chip chip-warn" title={c.review_reason || ""}>needs review</span>
                  : c.corroborates ? <span className="chip">matches complaint</span> : <span className="muted">—</span>}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {!readOnly && (
        <form onSubmit={add} style={{ display: "flex", gap: 6, marginTop: 6 }}>
          <input value={text} onChange={(e) => setText(e.target.value)} placeholder="e.g. C0035 P0300" aria-label="Fault codes" disabled={busy} />
          <button className="ghost" type="submit" disabled={busy || !text.trim()}>Add codes</button>
        </form>
      )}
      {msg && <div className="muted small" role="status">{msg}</div>}
      <div className="muted small">Generic meanings only; unknown codes have no effect. No public data links codes to recalls.</div>
    </div>
  );
}
