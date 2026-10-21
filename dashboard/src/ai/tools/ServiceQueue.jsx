// [AI-LAYER] Service-centre inspection call-in queue (deterministic, every point explained).
import { useEffect, useState } from "react";
import { aiGet } from "../aiApi.js";

export default function ServiceQueue() {
  const [d, setD] = useState(null);
  const [sim, setSim] = useState(true);
  const [err, setErr] = useState("");
  useEffect(() => { aiGet(`/api/ai/service-centre/queue?include_simulated=${sim}`).then(setD).catch((e) => setErr(e.message)); }, [sim]);
  return (
    <div data-testid="service-queue">
      <div className="row" style={{ gap: 10, alignItems: "center" }}>
        <label className="small"><input type="checkbox" checked={sim} onChange={(e) => setSim(e.target.checked)} /> include demo/simulated cases</label>
        {d && <span className="muted small">{d.open_cases} open cases · {d.method}</span>}
      </div>
      {err && <div className="muted small">{err}</div>}
      {d && d.queue.length === 0 && <div className="muted small" style={{ marginTop: 8 }}>No open cases.</div>}
      {d && d.queue.length > 0 && (
        <table className="small" style={{ width: "100%", marginTop: 8 }}>
          <thead><tr><th>#</th><th>Score</th><th>Vehicle</th><th>Component</th><th>Why</th><th>Call-in action</th></tr></thead>
          <tbody>
            {d.queue.map((x, i) => (
              <tr key={x.case_id}>
                <td>{i + 1}</td><td><strong>{x.score}</strong></td>
                <td>{x.vehicle || <span className="muted">unknown</span>}<div className="muted">{x.case_id}{x.case_kind === "simulated" ? " · demo" : ""}</div></td>
                <td>{x.category}<div className="muted">{x.severity}</div></td>
                <td><ul style={{ margin: 0, paddingLeft: 16 }}>{x.reasons.map((r, j) => <li key={j}>{r}</li>)}</ul></td>
                <td>{x.action}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
