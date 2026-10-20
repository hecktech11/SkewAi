# Skew AI: Meta AI layer (Meta Global AI Developer Hackathon)

> **Mission: help vehicle manufacturers reduce road accidents by recording
> vehicle faults from every customer contact, in the customer's own language,
> and predicting defects early enough to inspect and repair before they cause
> crashes, fires or injuries.**

Skew AI was already a working deterministic pilot (voice/text intake, safety
kill-switch, triage, NHTSA clustering, early-warning board, Qubot auditor, audit
ledger). For the hackathon we added an **optional** Meta AI layer on branch
`hackathon/ai-layer`. With no key the product behaves exactly as before. With
a key, Meta models make it **hear Hindi/Hinglish, see the broken part, and
write a grounded brief**, and the existing deterministic safety rules still have
the final say.

- Demo storyboard: [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md)
- One-command judge setup: `bash scripts/judge_setup.sh` → `http://127.0.0.1:8000/ui/#demo`
- Single container (not deployed): `docker compose -f docker-compose.hackathon.yml up --build`

## 1. What is new vs. what already existed

| Already existed (unchanged) | New in this branch (`[AI-LAYER]`) |
|---|---|
| Domain packs, slot frames, regex/gazetteer intake | `src/ai_layer/*`: Meta client, config, intake enhancer, language normaliser, severity raise, photo evidence, grounded brief, safety tiering, safety board, demo seeding |
| Kill-switch lexicon, `assess_safety`, Qubot auditor, audit ledger | `src/api/routes/ai_layer.py`: `/api/ai/*`, `/api/cases/{id}/evidence`, `/api/evidence/*` |
| Triage severity rules, NHTSA clustering, investigations | Dashboard: `dashboard/src/ai/*`, `dashboard/routes/DemoMode.jsx` (`#demo`) |
| Recall lead-time backtest (`src/backtest/recall_lead.py`) | `scripts/ai_safety_backtest.py`: full-scale run + preventable-window counts |
| Early-warning board, Issue desk, Case queue, Call widget | `eval/enhanced/`: 60 labelled synthetic calls (en/hi/hinglish) + harness |
| | `scripts/judge_setup.sh`, `scripts/ai_demo_seed.py`, `Dockerfile.hackathon`, `docker-compose.hackathon.yml` |
| | Tests: `tests/frontline/ai_layer/` (every Meta call mocked) |

## 2. How each Meta tool is used

All calls go through `src/ai_layer/meta_client.py`, which does the following:
- uses an allow-listed host (the key only goes to `api.meta.ai` or `META_ALLOWED_HOSTS`);
- applies a timeout, a retry on 429/5xx and a daily call cap;
- validates strict JSON schemas;
- logs nothing secret.
Every result also writes a row to an AI event log that records the deterministic value, the AI proposal and what was accepted.

| Meta model | Where | What it does | Guard-rail |
|---|---|---|---|
| **Muse Spark 1.3** (`muse-spark-1.3`, chat completions) | Live call intake (`intake_enhancer.py`) | Fills only *still-empty* slots; normalises Hindi/Hinglish to English | Values must validate against the pack gazetteer/year range and a confidence floor (0.7). The **existing** kill-switch lexicon is re-run on the English text and may only **add** an escalation |
| Muse Spark 1.3 | Language API (`normalize.py`) | Hindi/Hinglish → English, original kept alongside | Numbers must survive translation; a deterministic glossary is the fallback |
| Muse Spark 1.3 | Severity/sentiment (`severity.py`) | Proposes a severity + caller sentiment | `final = max(rule, AI)`; AI can never lower severity; the Qubot severity sanity check still runs |
| Muse Spark 1.3 (vision) | Photo evidence (`evidence.py`) | Describes the photo, guesses the component, notes visible hazards | May only add flags; never changes case severity by itself |
| **SAM 3.1** (`sam-3.1`, `/v1/responses`) | Photo evidence | Segments the damaged region; boxes drawn as an overlay in `EvidencePanel` | Box tokens parsed strictly; malformed output is dropped |
| Muse Spark 1.3 | Grounded brief (`grounded_brief.py`) | Rewrites the investigation brief, leading with crash/fire/injury risk, plus a suggested inspection step | Every claim must cite a real record/advisory ID and pass **Qubot**; failing claims are dropped and counted. AI actions with numbers/IDs are rejected; accepted ones are labelled "requires engineering review" |
| **Muse Image 1.0** (`muse-image-1.0`) | `scripts/ai_demo_seed.py --muse-image` | Optional *synthetic* brake-rotor photo for the demo | Always labelled SYNTHETIC; the default is a code-generated PNG |
| Muse Spark Open / Muse Glimmer (open weights) | Any OpenAI-compatible gateway | Self-host by pointing `META_API_BASE` at it and listing it in `META_ALLOWED_HOSTS` | Same client and guards |
| Browser Web Speech `hi-IN` | Call widget | Hindi speech recognition (language picker) | Muse Voice Transcribe is **not** wired (see §6) |

Every AI surface in the UI carries a **Meta model tag** (e.g. `Muse Spark 1.3`,
`SAM 3.1`) or a **Deterministic** tag, and the top bar shows the AI-mode badge.

## 3. Safety contract (what AI can never do)

1. Without a key (`META_API_KEY` / `MODEL_API_KEY`) or with `FRONTLINE_AI_LAYER=off`, no network call is made and the output is identical to `main`. This is enforced in tests by a transport that fails on any call.
2. The kill-switch, `assess_safety`, the Qubot auditor and the audit ledger always run and have the final say. AI may only **raise** severity or **add** a safety flag.
3. On any error, timeout, invalid JSON, schema mismatch or low confidence, the layer falls back silently to the deterministic result and logs the reason.
4. No invented data. Brief claims must cite real IDs that pass Qubot. Photos in the demo are synthetic and labelled as such. Backtest numbers come only from real NHTSA files.

## 4. Results (real data, honest caveats)

### 4.1 Recall early-warning backtest (NHTSA)

Full run on public NHTSA ODI data:
- complaints: `FLAT_CMPL`, Honda / Toyota / Ford / Hyundai / Kia, model years 2015–2024, **219,510 complaints**;
- recalls: `FLAT_RCL_POST_2010`, **4,393 recall scope rows = 1,172 campaigns**.

The detector is the **existing** deterministic weekly z-score scan; no AI is involved in these numbers. Each recall is scored only on complaints NHTSA had *received* by the recall report date, and misses stay in.
Full report: [`reports/nhtsa/backtest_safety_2015_2024.md`](reports/nhtsa/backtest_safety_2015_2024.md). Reproduce with `make ai-backtest ZIP=FLAT_CMPL.zip` (needs a side `DOMAIN_DB_PATH` loaded by `scripts/ingest_nhtsa.py` + `scripts/ingest_recalls.py`).

**Headline: actionable early warning** (a flag episode on the same make + component ending within 52 weeks before the recall):

| | Campaigns | Scope rows |
|---|---|---|
| Flagged early (hits) | **535 / 1,172 (45.6%)** | 2,090 / 4,393 (47.6%) |
| Missed | 637 | 2,303 |
| Median / mean lead | **11 / 15.9 weeks** | 10 / 14.8 weeks (IQR 3–23) |

| Flag quality (full series) | Value |
|---|---|
| Flag episodes | 1,540 |
| Confirmed by a later recall of the same make + component | 985, so **precision 0.64** |
| False-alarm episodes (no later recall in the data) | 555 |
| Slice-weeks flagged / confirmed / false alarm | 2,409 / 1,619 / 790 (precision 0.672) |

**Preventable window (correlation only, not "lives saved"):** this counts distinct complaints on the flagged make + component that NHTSA received between the start of the flag episode and the recall report date, using NHTSA's own owner-entered fields:

| CRASH = Y | FIRE = Y | INJURED > 0 | DEATHS > 0 | Any of these |
|---|---|---|---|---|
| 1,379 | 712 | 882 (1,266 persons) | 19 (29 deaths reported) | 2,331 |

**Upper bound we do not quote as the headline:** counting from the *first-ever* flag of a slice gives 735 / 1,172 campaigns (62.7%) with a median lead of 214 weeks. That lead is inflated. A broad slice such as FORD ENGINE is first flagged in 2015 and then "matches" an unrelated 2026 recall. Both metrics are in the report.

Caveats:
- Matching is at make + component grain, so a flag can come from a different defect than the one recalled.
- The 52-week window is a chosen parameter, not a tuned one.
- "False alarm" includes real defects that were never recalled, or that were recalled after the data window (right-censoring).
- Complaints are self-reported.
- The recall date is the Part 573 receipt date, and manufacturers often know earlier.
- Precision uses the full series.
- Only 5 makes were ingested (2,258,118 FLAT_CMPL lines read, 219,510 kept for these makes and model years). The existing trust check marked only 9,198 of them "trusted" (mostly freshness/staleness on historical data). They were loaded with the existing historical-backfill option (`enforce_trust=False`), so every scored complaint is real, but most are not "fresh" by the product's live-data rules.

Memory note: the existing `recall_lead_report` caches every as-of series and ran the box out of RAM at this scale. `scripts/ai_safety_backtest.py` therefore calls the same building blocks but keeps only one series in memory per worker. A test asserts that its output is identical to `recall_lead_report` on a fixture.

### 4.2 Hindi/Hinglish intake eval (`eval/enhanced/REPORT.md`)

There are 60 SYNTHETIC labelled calls (20 English, 20 Hindi, 20 Hinglish; 12 should trip the kill-switch).
Results for the deterministic path, which is the unchanged product:

| Language | Make | Component | Kill-switch recall | False alarms |
|---|---|---|---|---|
| English | 1.0 | 1.0 | 1.0 | 0 |
| Hindi (Devanagari) | 0.0 | 0.0 | **0.0** | 0 |
| Hinglish | 1.0 | 0.875 | **0.5** | 0 |

This is the gap the layer targets: the rules-only path misses **every** Hindi
safety call. The enhanced column reads **"not run"** because there is no key.
We did not fabricate enhanced numbers. To fill it, run
`META_API_KEY=… python -m eval.enhanced.run_eval`.

## 5. Existing files touched (minimal wiring, each marked `[AI-LAYER]`)

| File | Why |
|---|---|
| `src/api/main.py` | `include_router(ai_layer_routes.router)` (2 lines) |
| `src/agents/orchestrator.py` | Two guarded hooks: `maybe_enhance_intake` after deterministic intake; `maybe_raise_severity` after triage. Both return the input unchanged with no key, and each sits inside `try/except: pass` |
| `src/frontline/dsr.py` | DSR erase also deletes the photo evidence rows/files for the erased interaction |
| `.gitignore` | Ignore `data/ai_evidence/` (customer photos) and `data/hackathon/` |
| `docs/openapi.yaml` | Additive entries for the new `/api/ai/*` and evidence paths (the surface test requires it) |
| `Makefile` | Appended targets `hackathon-setup`, `ai-demo-seed`, `ai-eval`, `ai-backtest`, `ai-test` |
| `README.md` | Mission line + "AI Layer (optional)" section |
| `dashboard/src/App.jsx` | `#demo` page entry + AI-mode badge in the top bar |
| `dashboard/routes/CallWidget.jsx` | Speech language picker (`hi-IN`) and photo upload under the compose box |
| `dashboard/routes/CaseQueue.jsx` | `EvidencePanel` above operator notes |
| `dashboard/routes/IssueDesk.jsx` | `GroundedBriefPanel` + read-only evidence panel |
| `dashboard/routes/EarlyWarningBoard.jsx` | `SafetyRiskBoard` under the header |

Nothing in the existing rules, regex, gazetteers, lexicons, tests or data was
modified.

## 6. What is unverified about Meta (no key exists in this project)

- Endpoint defaults follow the public docs at <https://dev.meta.ai/docs> (read 2026-10-21): base `https://api.meta.ai/v1`, Bearer auth, `muse-spark-1.3` chat completions with `response_format: json_schema`, `sam-3.1` via `/v1/responses` streaming box tokens, and `muse-image-1.0` via `/v1/images/generations`. **None of this has been called live.** The request/response shapes are covered only by mocked tests.
- The following are assumptions: `json_schema` strict-mode support on Muse Spark, the exact SAM box-token grammar, image-input formatting for vision, rate limits and pricing. Every one is env-configurable (`META_API_BASE`, `META_MODEL_*`, `META_CHAT_PATH`, `META_SAM_ENDPOINT`, `META_IMAGE_ENDPOINT`, `META_TIMEOUT_S`, …; see `src/ai_layer/config.py`).
- Muse Voice Transcribe (`muse-voice-transcribe-1.0`) exists in the docs but is **not implemented**. Hindi speech uses the browser's `hi-IN` Web Speech recogniser.
- Enhanced-mode accuracy numbers do not exist yet (see §4.2).
- **To verify in about 30 seconds with a key:** `META_API_KEY=… python -m scripts.ai_meta_smoke` (add `--image` for Muse Image). It makes one small call each to text, vision and SAM 3.1 through the product client and prints ok/failed with the reason. Without a key it is skipped and makes no network call.

## 7. Known limitations (honest list)

- The pack kill-switch lexicon is English-only, and that was not changed (by rule). Hindi safety recall depends on the Meta key.
- Existing `classify_yes_no` does not parse Hinglish "nahi"/"haan" as no/yes, so the seeded scenario answers yes/no questions in English. This was left as is (existing logic).
- The British spelling "tyre" is not in the English gazetteer. That was left as is.
- `scripts/seed_issue_demo.py` fails on this box with "path outside allow-root jail" (pre-existing). The judge setup seeds with `seed_frontline_fixtures` + the simulator instead.
- `pyarrow` is used by ingest trust persistence but is not declared in `requirements.txt` (pre-existing). Several baseline test failures come from that.
- The safety-risk tier uses a keyword list over complaint text, so it is a triage aid, not a hazard analysis.
- Backtest caveats are listed in §4.1.
