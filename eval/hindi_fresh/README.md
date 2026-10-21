# Fresh Hindi / Hinglish hold-out set (SYNTHETIC)

24 synthetic sentences (12 Devanagari Hindi, 12 romanised Hinglish), written by hand for this task
BEFORE the hindi_safety lexicon was written and before any measurement. They are not taken from
`eval/enhanced/calls.jsonl`, and the lexicon was not built from that file's expected labels.

Labels are the author's judgement:
- `kill_switch`: true when the sentence describes an active hazard (fire, smoke, crash, injury,
  brake/steering failure, unintended acceleration, airbag malfunction with injury).
- `expect_category`: the NHTSA pack category the complaint is about, or null when none is
  clear or the complaint has no component.

Negated hazards ("कोई घायल नहीं हुआ", "koi dhuan ya aag nahi") are deliberately included as
`kill_switch: false`.

Caveat: the same author wrote this set and the lexicon, so this hold-out is weaker than an
independently collected one.
