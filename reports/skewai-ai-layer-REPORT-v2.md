# SkewAi AI layer, v2 report (branch `hackathon/ai-layer`)

All work was done on the box. No Cursor cloud agents were used, nothing was pushed, and no Meta API key was configured.
Every number below is copied from a file under `reports/` in the repo, and the path is cited next to it.

## Head, commits, bundle
- Start head: `945f5b8`. New head: **`9d2c30d`** (35 commits ahead of `main` `151f7ee`).
- v2 commits (oldest first):
  - `e0e51db`: OOS backtest script (tune 2015-2020, frozen hashed config, single 2021-2024 test, refuses re-tune/re-test)
  - `a6816ba`: frozen config from tuning (sha256 `520ec7b0…`), committed before the test run
  - `2a605be`: Meta call health per call type via an additive [AI-LAYER] wrapper
  - `7967f5b`: supervisor-only review of AI severity raises, plus /api/ai/call-health and /api/ai/status/overview
  - `3dd3b29`: eval harness comparing rules-only and rules+Meta
  - `a265375`: dashboard #ai-status page; `npm run build` ok
  - `2fb854e`: rules-only eval reports, pre-change relevant test run, smoke script
  - `052a5d6`: no-key smoke test outputs and the smoke stop() kill fallback
  - `0786621`: OOS 2021-2024 test outputs and after-change relevant tests
  - `9d2c30d`: comment-only `[AI-LAYER]` markers in docs/openapi.yaml
- Bundle: `/workspace/skewai-ai-layer.bundle` (main + hackathon/ai-layer). Regenerated at 11:34 IST (UTC+5:30); `git bundle verify` says "is okay" and reports a complete history.

## Files added
- Code:
  - `src/ai_layer/call_health.py`
  - `src/ai_layer/raise_review.py`
  - `src/api/routes/ai_layer_v2.py`
  - `scripts/ai_oos_backtest.py`
  - `scripts/ai_v2_smoke.sh`
  - `eval/enhanced/run_eval_modes.py`
  - `dashboard/routes/AiStatus.jsx`
- Tests, all mocked with no network (31 tests):
  - `tests/frontline/ai_layer/test_call_health.py` (10)
  - `test_raise_review.py` (11)
  - `test_oos_backtest.py` (6)
  - `test_eval_modes.py` (4)
- Reports:
  - `reports/oos/*`
  - `reports/eval_modes/*`
  - `reports/eval_modes_stdout.txt`
  - `reports/smoke_v2/*`
  - `reports/dashboard_build_v2.txt`
  - `reports/tests_before_relevant.txt`
  - `reports/tests_after_relevant.txt`
  - `reports/tests_before_read_auth_isolated.txt`
  - `reports/tests_after_read_auth_isolated.txt`

## Existing files touched (additive only, each hunk marked `[AI-LAYER]`, 0 deleted lines)
- `src/ai_layer/meta_client.py` (+14): a block just before `__all__` wraps `chat_json`, `sam_segment` and `image_generate` with `call_health.instrument`.
  - The wrapper returns and raises exactly what the wrapped function does.
  - `meta_disabled` (no key) is never recorded.
  - The block sits inside `try/except: pass`, so an import failure leaves the plain functions in place.
- `src/api/routes/ai_layer.py` (+5): `router.include_router(ai_layer_v2.router)` added at the end of the file.
- `dashboard/src/App.jsx` (+3): one import line and one `EXTRA_PAGES` entry `{id:"ai-status", ...}`.
- `docs/openapi.yaml` (+367): 6 new paths (`/api/ai/call-health`, `/api/ai/status/overview`, `/api/ai/raises`, `/api/ai/raises/stats`, `/api/ai/raises/{event_id}/review`, `/api/ai/raises/audit/verify`) and a `ReviewBody` component, plus two `# [AI-LAYER]` comment lines.

No orchestrator, rules, regex, safety check, ledger, Qubot auditor or existing test was changed. Severity is still raise-only: review verdicts are stored in new tables and never write to `cases` or `ai_layer_events`.

## Feature behaviour (no key), from `reports/smoke_v2/summary.txt`
- **Run 1, open mode:**
  - `/health`, `/api/ai/status`, `/api/ai/call-health`, `/api/ai/status/overview`, `/api/ai/raises`, `/api/ai/raises/stats` and `/ui/` all return 200.
  - A review POST returns **403**, because open mode is always the agent role.
  - Call health is `"status":"not run"` with 0 calls for every type (`reports/smoke_v2/open_call_health.json`).
- **Run 2, auth required, one seeded raise** (synthetic seed with no Meta call, `reports/smoke_v2/seed.json`):

  | Who | Review POST result |
  | --- | --- |
  | No credentials | **401** |
  | Agent session | **403** |
  | Shared service key | **403** ("role service cannot review AI raises (supervisor only)") |
  | Supervisor | **200** |
  | Supervisor, duplicate | **409** |

  - Stats: `"n = 1"`, `agreement_rate: null`, `"too few reviews to estimate (n = 1 < 20)"` (`auth_raise_stats.json`).
  - Audit chain verify: `{"ok":true,"rows":5}` (`auth_audit_verify.json`).
  - Case severity after the verdicts: `('Critical', 1)`. That is the same as the seeded `Critical`/priority 1, so it is unchanged.
- Dashboard build: exit 0 (`reports/dashboard_build_v2.txt`). The only warning is a chunk larger than 500 kB.

## Tests, before vs after (relevant suites)
- Command: `pytest tests/frontline/ai_layer test_openapi_surface test_read_auth test_triage test_ledger_completeness test_auth`.

| Run | Failed | Passed | File |
| --- | --- | --- | --- |
| Before (945f5b8) | 2 | 152 | `reports/tests_before_relevant.txt` |
| After | 2 | 183 | `reports/tests_after_relevant.txt` |

- The +31 passed are the 31 new tests.
- The 2 failures are the same in both runs and existed before this work:
  - `test_read_auth::test_read_routes_accept_with_key[/api/frontline/digest]`
  - `test_read_auth::test_digest_regenerate_requires_key`
- They are an ordering issue. `test_read_auth.py` alone passes 15/15 both before and after (`reports/tests_before_read_auth_isolated.txt`, `reports/tests_after_read_auth_isolated.txt`).
- **New failures: 0.**

### Full suite
- A *before* full run (`--timeout 120`) is **still running** in the clean worktree `/workspace/skewai-before-v2`, which is checked out at `945f5b8`. That makes it a genuine pre-change run.
  - It writes to `reports/tests_before_full.txt`, which is not committed and has no final counts yet.
  - The partial output shows 3 failures by about 13%, which matches the pre-existing early failures in the main baseline log.
- After it finishes, the same script (`/workspace/ailayer_logs/v2/full_suites.sh`) runs the *after* full suite in `/workspace/skewai-after-v2` at `9d2c30d` (dist built, BUILD_EXIT 0), writing `reports/tests_after_full.txt`.
- **No full-suite before/after counts are reported here.** For reference only, the earlier main baseline log `/workspace/ailayer_logs/baseline312.txt` (outside reports/) shows 47 failed, 1515 passed, 1 skipped.

## OOS backtest (`reports/oos/OOS_BACKTEST.md`, from `oos_tune_results.json` and `oos_test_results.json`)
- Frozen config: `z2.0_q0.05_abs3_rel0.5`, sha256 `520ec7b0fdb6c18b5fe388102100334bd969e6308ff33b86ebfed0c556159e23` (`reports/oos/oos_config.json`). Production is `abs5`.
- It was tuned once on 2015-2020 (36 configs) and tested **once** on 2021-2024 (`reports/oos/run_log.jsonl`: tune, test_start, then test with `forced:false`, 1130 s).
- Re-gate check: 0 mismatches over 81,386 train rows and over 205,337 test rows.

| Metric | In-sample 2015-2020 | OOS 2021-2024 |
| --- | --- | --- |
| Scope rows / campaigns scored | 1423 / 473 | 1861 / 477 |
| Actionable 52-wk campaigns flagged early | 217 | 299 |
| Campaign recall rate | 0.459 | 0.627 |
| Campaign median lead (wk) | 13 | 8 |
| Scope-row hits / recall | 634 / 0.446 | 1142 / 0.614 |
| Scope-row median / mean lead (wk) | 13.0 / 15.8 | 8.0 / 15.4 |
| Episodes / episode precision | 807 / 0.585 | 1098 / 0.598 |
| Flagged slice-weeks / precision | 1080 / 0.6 | 1699 / 0.634 |
| First-flag (upper bound) recall / median lead (wk) | 0.584 / 128 | 0.883 / 274.0 |
| Preventable window: complaints | 8944 | 23662 |
| Preventable window: crash / fire / injury complaints | 542 / 138 / 386 | 682 / 374 / 452 |
| Preventable window: crash, fire, injury or death | 764 | 1160 |

- In-sample, production thresholds: campaign recall 0.336, episode precision 0.61.
- Caveats:
  - The chosen config sits **at the grid edge** (abs=3 is the lowest value tried).
  - It was a **4-way tie** on the objective (0.5144), broken by the stated rule.
  - OOS recall is *higher* than in-sample. This is not evidence of generalisation gain: the two windows differ in recall mix, history length and censoring.
  - Full limitations are in OOS_BACKTEST.md: RCDATE labels, right-censoring, reporting bias, no denominators, make+component grain, 5 makes, a small grid, and correlation only.
  - There is no OOS run of the production thresholds; one would count as a second test.
- Run-log note: the `test` entry records `git_head 052a5d6` because the head is read when the run ends. Commit `052a5d6` only added smoke outputs and changed no scoring code; `test_start` recorded `e0e51db`.

## Eval, 60 synthetic calls (`reports/eval_modes/EVAL_MODES.md`)
Rules-only results:

| Lang | Calls | Severity | Triage skipped | Errors | Kill-switch hits / expected | Meta calls |
| --- | --- | --- | --- | --- | --- | --- |
| en | 20 | Low 4, Medium 10, Critical 2 | 4 | 0 | 4/4 | 0 |
| hi | 20 | none | 20 | 0 | 0/4 | 0 |
| hinglish | 20 | Low 2, Medium 12, Critical 2 | 4 | 0 | 2/4 | 0 |
| all | 60 | Low 6, Medium 22, Critical 4 | 28 | 0 | 6/12 | 0 |

- **Rules+Meta: NOT RUN** because no META_API_KEY is set. Every enhanced cell (agreement, raises, lowers, failures) reads "not run".
- The calls have no human labels, so "agreement" would only compare the two modes. It is not accuracy.
- Finding: the rules path produces no severity for any Devanagari call (20/20 triage skipped, 0/4 kill-switch). Hinglish catches 2/4 kill-switch cases.

## Unverified / not done
- No live Meta call was made. Enhanced eval, real call-health latencies and the Meta API response shapes are unverified; only mocks were used.
- The full-suite before/after comparison is still pending (see above).
- The OOS test run overlapped with the lightweight relevant-test and smoke runs on the same box, which affects runtime only.
- pyarrow was installed only into `/workspace/ingest_extra` for the NHTSA side-DB ingest. The test venv is unchanged.
- Nothing was pushed (gh is not logged in).
