# OBD-II fault codes (DTCs): optional intake evidence  [AI-LAYER]

**Input.** You can supply fault codes in three ways:
- `POST /api/ai/interactions/{id}/fault-codes` during intake;
- `POST /api/ai/cases/{id}/fault-codes` on an existing case (the dashboard case view uses this one);
- typing `codes: C0035 P0300` in any message, including the WhatsApp mock.

Each code must match `^[PCBU][0-3][0-9A-F]{3}$`. Anything else gets a 422 from the API and is ignored in a message.

**Meaning.** Meanings come from `src/ai_layer/data/dtc_lookup.json`:
- Only generic SAE J2012-style meanings the author is confident of are listed. Each entry carries a confidence of `high` or `medium`.
- No manufacturer-specific meanings are included.
- C04xx / C0455 steering codes are deliberately left out, because their meanings vary by manufacturer.
- A valid code that is not listed is shown as `unknown code` and has **no effect**.

**Severity.** A known *safety-system* code sets a floor at the pack's safety-system level. On the NHTSA pack that is Medium. The safety-system codes are:
- ABS wheel-speed sensor C0035 to C0050;
- restraint deployment loop B0001 to B0099;
- lost communication with the ABS, steering or restraint module (U0121, U0126, U0131, U0151).

The final severity is max(current, floor). It can only **raise** severity, never lower it. Every raise is ledgered as `severity_scored` by `ai_layer`.

**Needs review.** A code whose component differs from the complaint's category is flagged `needs review`. Nothing else changes.

**Counts.** "Cases with matching fault code: n" counts cases whose stored code component equals the case category. It appears:
- on vehicle risk (per component and in total);
- on new alerts' payloads;
- on the service-centre queue, as an explained `+10` reason line for a safety-system code.

Existing vehicle-risk levels and alert thresholds are unchanged.

**No accuracy claim.** No public dataset links DTCs to recalls, so there is no backtest or accuracy figure for any of this. The tests use clearly labelled example codes only.
