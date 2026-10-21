// [AI-LAYER] Call-widget add-ons: optional hi-IN speech recognition and
// mid-call photo upload. Defaults leave the existing call flow untouched.
import { useState } from "react";
import { aiSpeechLang, setAiSpeechLang } from "./aiApi.js";
import EvidencePanel from "./EvidencePanel.jsx";

export function SpeechLangSelect() {
  const [lang, setLang] = useState(aiSpeechLang() || "en-US");
  return (
    <label className="small muted" style={{ display: "inline-flex", gap: 6, alignItems: "center" }}>
      Speech language
      <select value={lang} aria-label="Speech recognition language"
        onChange={(e) => { setLang(e.target.value); setAiSpeechLang(e.target.value); }}>
        <option value="en-US">English (default)</option>
        <option value="en-IN">English (India)</option>
        <option value="hi-IN">हिन्दी / Hinglish (hi-IN)</option>
      </select>
      <span className="faint">applies when the mic starts</span>
    </label>
  );
}

export default function CallAiExtras({ interactionId, caseId }) {
  return (
    <div className="ai-call-extras" style={{ marginTop: 10 }}>
      <SpeechLangSelect />
      {(caseId || interactionId) && (
        <EvidencePanel caseId={caseId || undefined} interactionId={caseId ? undefined : interactionId}
          title="Photo of the fault (optional)" />
      )}
    </div>
  );
}
