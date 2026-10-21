// [AI-LAYER] Safety-risk indicator (deterministic: component tier + crash/fire/
// injury mentions in real complaint text). Ordering aid, not a crash prediction.
const CLS = { high: "sev-critical", elevated: "sev-high", normal: "" };

export default function SafetyRiskChip({ risk }) {
  if (!risk) return null;
  const m = risk.mentions || {};
  const title = `${risk.component_tier} component · ${risk.records_with_hazard_mention}/${risk.records_scanned} complaints mention crash/fire/injury (crash ${m.crash || 0}, fire ${m.fire || 0}, injury ${m.injury || 0}). ${risk.method || ""}`;
  return (
    <span className={"chip " + (CLS[risk.level] || "")} title={title} data-testid="safety-risk">
      Safety risk: {risk.level}
    </span>
  );
}
