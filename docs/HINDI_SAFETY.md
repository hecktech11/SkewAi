# Hindi / Hinglish safety vocabulary layer  [AI-LAYER]

`src/ai_layer/hindi_safety.py` is deterministic. It needs no key and makes no network calls, and `AI_HINDI_SAFETY=0` turns it off. It runs right after the existing `IntakeAgent` for turns that `language.detect` labels `hi` or `hinglish`.

**Gloss.** Devanagari and romanised safety terms are mapped to an English gloss, for example:
- ब्रेक फेल / "brake fail" → "brakes failed"
- आग / aag → "fire"
- धुआं / dhuan → "smoke"
- टक्कर / takkar → "collision"
- चोट / chot → "injury"
- स्टीयरिंग जाम → "steering locked"

The **existing** pack kill-switch check then runs on that gloss. A hit can only ADD an escalation.

**Slots.** Only empty slots are filled, and existing values are never overwritten:
- category, validated against the pack gazetteer;
- make and model from Devanagari transliterations;
- the year;
- the description, which is the caller's own words.

**Severity.** After triage, final = max(rule, floor):
- a hazard term sets a floor of Critical;
- a safety-critical component sets a floor of Medium.

It never lowers severity.

**Negation and hedges.** Negation (कोई घायल नहीं, koi chot nahi, bina accident ke) and hedges (शायद) suppress a hazard term within its clause.

**How the vocabulary was built.** It was written from general vocabulary, not from any eval file's labels. A fresh 24-sentence hold-out (`eval/hindi_fresh/fresh_calls.jsonl`) was committed before the lexicon. Results are in `reports/hindi_safety/HINDI_EVAL.md`.

**Limitations.** The same author wrote the hold-out and the lexicon. On kill-switch calls, slot filling stops at the escalation, as in English.
