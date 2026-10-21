// [AI-LAYER] Visible model attribution: "Powered by Muse Spark" or the
// deterministic fallback with its reason. Never hides which path produced a value.
export default function MetaTag({ models, mode, fallbackReason, style }) {
  const list = (models || []).filter(Boolean);
  if (mode === "enhanced" && list.length) {
    return (
      <span className="chip ai-tag ai-tag-on" style={style} title="Produced by the optional Meta AI layer; verified before display">
        Powered by Meta · {list.join(" + ")}
      </span>
    );
  }
  return (
    <span className="chip ai-tag ai-tag-off" style={style} title="Deterministic rules path (default and fallback)">
      Deterministic{fallbackReason && fallbackReason !== "no_key" ? ` · fallback: ${fallbackReason}` : ""}
    </span>
  );
}
