// [AI-LAYER] WhatsApp intake MOCK. Posts a Cloud-API-shaped webhook body to our own
// endpoint; nothing is sent to WhatsApp/Meta. Replies come from the existing intake agent.
import { useState } from "react";
import { aiPostJson } from "../aiApi.js";

const SAMPLES = [
  "Meri 2019 Honda CR-V ke brakes fail ho gaye, highway pe",
  "My 2019 Honda CR-V brakes are grinding and the pedal goes soft",
  "Yes, it happened yesterday on the highway",
];

export default function WhatsAppMock() {
  const [waId] = useState(() => "91980000" + String(Math.floor(Math.random() * 9000) + 1000));
  const [msgs, setMsgs] = useState([]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [info, setInfo] = useState(null);
  async function send(t) {
    const body = (t ?? text).trim();
    if (!body) return;
    setText(""); setBusy(true);
    setMsgs((m) => [...m, { me: true, text: body }]);
    const payload = { object: "whatsapp_business_account", entry: [{ id: "MOCK", changes: [{ field: "messages", value: {
      messaging_product: "whatsapp", contacts: [{ wa_id: waId, profile: { name: "Demo customer" } }],
      messages: [{ from: waId, id: `wamid.mock.${Date.now()}`, timestamp: String(Math.floor(Date.now() / 1000)), type: "text", text: { body } }] } }] }] };
    try {
      const r = await aiPostJson("/api/ai/whatsapp-mock/webhook", payload);
      const res = r.results[0] || {};
      setInfo({ label: r.label, ...res });
      setMsgs((m) => [...m, ...(res.replies || []).map((x) => ({ me: false, text: x.text.body }))]);
    } catch (e) { setMsgs((m) => [...m, { me: false, text: `[error] ${e.message}` }]); } finally { setBusy(false); }
  }
  return (
    <div data-testid="whatsapp-mock" style={{ maxWidth: 520 }}>
      <div className="chip sev-medium" style={{ display: "inline-block", marginBottom: 8 }}>MOCK — simulated WhatsApp, no message is sent</div>
      <div style={{ background: "#e5ddd5", borderRadius: 8, padding: 10, minHeight: 220, maxHeight: 380, overflowY: "auto" }}>
        <div className="small" style={{ color: "#555", textAlign: "center" }}>Chat with the SkewAi intake agent as +{waId} (mock)</div>
        {msgs.map((m, i) => (
          <div key={i} style={{ display: "flex", justifyContent: m.me ? "flex-end" : "flex-start", margin: "6px 0" }}>
            <div style={{ background: m.me ? "#dcf8c6" : "#fff", color: "#111", padding: "6px 10px", borderRadius: 8, maxWidth: "80%", whiteSpace: "pre-wrap" }}>{m.text}</div>
          </div>
        ))}
      </div>
      <form className="row" style={{ gap: 6, marginTop: 6 }} onSubmit={(e) => { e.preventDefault(); send(); }}>
        <input style={{ flex: 1 }} value={text} onChange={(e) => setText(e.target.value)} placeholder="Type a message" disabled={busy} />
        <button className="primary" disabled={busy}>Send</button>
      </form>
      {/* [AI-LAYER] fault codes hint */}
      <div className="muted small" style={{ marginTop: 4 }}>Optional: add OBD-II fault codes as <span className="mono">codes: C0035 P0300</span></div>
      <div className="row small" style={{ gap: 6, marginTop: 6, flexWrap: "wrap" }}>
        {SAMPLES.map((s) => <button key={s} className="ghost" onClick={() => send(s)} disabled={busy}>{s.slice(0, 38)}…</button>)}
      </div>
      {info && (
        <div className="muted small" style={{ marginTop: 6 }}>
          interaction {info.interaction_id} · state {info.state}{info.case_id ? <> · case <a href={`#cases?case=${info.case_id}`}>{info.case_id}</a></> : null}
          {info.severity ? ` · severity ${info.severity}` : ""}{info.safety_flags?.length ? ` · safety: ${info.safety_flags.join(", ")}` : ""}
          {/* [AI-LAYER] fault codes typed as "codes: C0035 P0300" */}
          {info.fault_codes?.length ? ` · fault codes: ${info.fault_codes.map((c) => `${c.code} (${c.component || "unknown code"})`).join(", ")}` : ""}
        </div>
      )}
    </div>
  );
}
