// [AI-LAYER] Hackathon tools hub (#ai-tools): vehicle risk, manufacturer alerts,
// service-centre queue, fault spread, WhatsApp intake mock. Each tab is a new
// component under dashboard/src/ai/tools; existing screens are untouched.
import { useState } from "react";
import VehicleRisk from "../src/ai/tools/VehicleRisk.jsx";
import Alerts from "../src/ai/tools/Alerts.jsx";
import ServiceQueue from "../src/ai/tools/ServiceQueue.jsx";

const TABS = [
  { id: "vehicle", label: "Vehicle fault risk", el: VehicleRisk },
  { id: "alerts", label: "Manufacturer alerts", el: Alerts },
  { id: "service", label: "Service-centre queue", el: ServiceQueue },
];

export default function AiTools({ hashQuery }) {
  const initial = (hashQuery && hashQuery.tab) || TABS[0].id;
  const [tab, setTab] = useState(TABS.some((t) => t.id === initial) ? initial : TABS[0].id);
  const Cur = (TABS.find((t) => t.id === tab) || TABS[0]).el;
  return (
    <div className="ai-tools" style={{ maxWidth: 1150 }}>
      <h1 style={{ marginBottom: 4 }}>Road-safety tools</h1>
      <p className="muted" style={{ marginTop: 0 }}>Every number on these tabs comes from the warehouse (real NHTSA rows or recorded cases). AI text, when a Meta key is set, is labelled.</p>
      <div className="row" style={{ gap: 6, marginBottom: 12, flexWrap: "wrap" }} role="tablist">
        {TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={tab === t.id} className={tab === t.id ? "primary" : "ghost"} onClick={() => setTab(t.id)}>{t.label}</button>
        ))}
      </div>
      <section className="card" style={{ padding: 14 }}><Cur /></section>
    </div>
  );
}
