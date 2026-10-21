// [AI-LAYER] Grounded investigation brief: every claim cites real record ids
// (Qubot-verified), safety risk first, manufacturer inspection actions.
import { useEffect, useState } from "react";
import { aiGet } from "./aiApi.js";
import MetaTag from "./MetaTag.jsx";
import SafetyRiskChip from "./SafetyRiskChip.jsx";

export default function GroundedBriefPanel({ investigationId }) {
  const [b, setB] = useState(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    if (!investigationId) return;
    let live = true;
    setB(null); setErr("");
    aiGet(`/api/ai/investigations/${encodeURIComponent(investigationId)}/grounded-brief`)
      .then((d) => live && setB(d))
      .catch((e) => live && setErr(e.message));
    return () => { live = false; };
  }, [investigationId]);
  if (!investigationId) return null;
  if (err) return <div className="muted small">Grounded brief unavailable ({err}).</div>;
  if (!b) return <div className="muted small">Building grounded brief…</div>;
  const r = b.safety_risk || {};
  const m = r.mentions || {};
  return (
    <section className="ai-grounded-brief" data-testid="grounded-brief" style={{ marginTop: 12 }}>
      <h3 className="analytics-h" style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
        Grounded brief <SafetyRiskChip risk={r} />
        <MetaTag mode={b.mode} models={b.powered_by} fallbackReason={b.fallback_reason} />
      </h3>
      <p style={{ margin: "6px 0" }}>{b.headline}</p>
      {r.records_scanned > 0 && (
        <p className="muted small">
          Of {r.records_scanned} complaints in scope, {r.records_with_hazard_mention} mention a crash, fire or injury
          (crash {m.crash || 0} · fire {m.fire || 0} · injury {m.injury || 0}). Keyword mentions in owner narratives, not verified events.
        </p>
      )}
      <ul style={{ paddingLeft: 18 }}>
        {(b.claims || []).map((c, i) => (
          <li key={i} style={{ marginBottom: 4 }}>
            {c.text}{" "}
            {(c.record_ids || []).map((id) => <span key={id} className="chip mono" style={{ marginRight: 4 }}>{id}</span>)}
            <span className="muted small"> {c.source === "meta_muse_spark" ? "· Muse Spark, Qubot-verified" : "· deterministic"}</span>
          </li>
        ))}
      </ul>
      {(b.dropped_claims || []).length > 0 && (
        <div className="muted small">{b.dropped_claims.length} AI claim(s) dropped by verification ({[...new Set(b.dropped_claims.map((d) => d.reason.split(":")[0]))].join(", ")}).</div>
      )}
      <h4 style={{ margin: "10px 0 4px" }}>Recommended inspection / repair actions (for the manufacturer)</h4>
      <ol style={{ paddingLeft: 18 }}>
        {(b.recommended_actions || []).map((a, i) => (
          <li key={i} style={{ marginBottom: 4 }}>
            {a.action}{" "}
            <span className="muted small">
              · {a.source === "recall_advisory" ? "recall advisory remedy (warehouse)" : a.source === "meta_muse_spark" ? "Muse Spark suggestion, requires engineering review" : "SkewAi safety playbook"}
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}
