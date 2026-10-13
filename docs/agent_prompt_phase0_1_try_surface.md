# Agent prompt: Phase 0–1 fleet proof + try surface (MCP + CSV load)

**Audience:** an AI coding agent working in this repository with no prior chat history.  
**Product:** Skew AI (`rahulx2001/SkewAi`) — domain-agnostic voice-of-customer / quality intelligence with Domain Packs, live agents, and a Qubot groundedness auditor. Adjacent to Axion (fleet quality), **not** an Axion clone.  
**Horizon:** ~3–4 weeks of focused work. End state is a report a quality engineer can read, plus two cheap ways to try the product (MCP read tools, CSV that actually loads).  
**Branch context (as of this prompt):** work lives on `fleet-quality-phase0`. Phase 0 commits already landed; Phase 1 and some product-brief pieces exist as WIP. Treat the “Already done” checklist as ground truth — **do not reimplement**. Finish gaps, harden tests, then add MCP + real CSV load.

---

## 0. Why this scope (read before coding)

### Mission

Prove the **early-warning claim** on **public NHTSA data**, then make the product easy to try. The claim to prove:

> Given real complaint traffic that NHTSA had already received before a recall’s report-received date, did Skew’s weekly anomaly pipeline flag the matching make/category (and optionally model) slice **at least one week earlier**, and by how many weeks? Which recalls did it miss? Which spikes never matched a later recall?

That answer is a **lead-time report**, not a dashboard full of connectors.

### Why not the whole Axion-shaped list (warehouse / telematics / SAP / PLM / SSO / multi-tenant)

1. **Years of team work.** Axion raised ~$64M for connectors, analytics, and enterprise features and still uses engineers to validate insights. One developer building SAP, PLM, telematics, and five warehouse connectors on speculation will burn calendar before anyone uses them.
2. **Connectors need customer data.** Warranty, service, field, and telematics are private. Until a prospect hands you an export or access, those connectors have nothing to load and cannot be tested against real files.
3. **The core claim is not yet proven on real scale.** More integrations do not help while the pipeline that would consume them cannot yet show “we saw this early” on real recalls.

### Worth doing now (cheap, no partner required)

| Workstream | Why |
|---|---|
| **Phases 0–1** | Free public NHTSA complaints + recalls. Ends in a report you can show. |
| **Small MCP server (read tools)** | A few days. Lets someone try Skew from Claude/Cursor without adopting another dashboard. |
| **File upload that loads rows** | First customers send CSV exports; simpler than a warehouse connector. |

### Build only when a real prospect asks

Warehouse / telematics / ERP / PLM connectors; Jira / ServiceNow sync; SSO; multi-tenant. Build **the one connector the first customer needs**, not a speculative matrix.

### After the lead-time report

If this becomes a company: talk to 5–10 quality/reliability engineers at mid-size manufacturers, fleet operators, or dealer groups. One shared export beats any connector on a roadmap.  
If this is a portfolio/learning project: Phases 0–1 + MCP already make a strong end-to-end demo; leave the rest on paper.

---

## 1. Product and architecture the agent must respect

### Domain Pack model

- Packs live under `domains/<pack_id>/` (YAML, gazetteers, demo fixtures, mapping).
- Canonical fleet tables live in a **per-pack DuckDB** via `src.data.warehouse.domain_con(pack_id)`.
- Ops / investigations / jobs / novel_candidates live in the **ops** DuckDB via `ops_con()`.
- DuckDB is effectively **single-writer**. Do not open nested writers. Batch writes; avoid one commit per row on large ingests.

### Canonical tables that matter for this work

- `records` — complaints / VoC / warranty-like rows (`category`, `entity_1/2/3`, `received_at`, `occurred_at`, `source`, `provenance`, `text`, …).
- `advisories` — recalls / campaigns (`advisory_id`, `issued_at`, `scope_*`, `summary`, `remedy`, `url`).
- `weekly_anomalies` — scored weekly slices (make × category [× model]).
- `clusters` / `cluster_assignments` — embedding clusters used by investigations and backtest.
- `backtest_results` — lead-time rows from `src.backtest.engine`.
- `investigations` (ops) — fleet-scan opened cases with slice fields.
- `novel_candidates` (ops) — unresolved category-like phrases from clustering.

### Pack for this proof

Primary pack: **`automotive_nhtsa`**.

- Complaints: NHTSA FLAT_CMPL (via `scripts/ingest_nhtsa.py` / mapping ingest).
- Recalls: NHTSA FLAT_RCL (via `scripts/ingest_recalls.py`). Official single `FLAT_RCL.zip` may 404; use the post-2010 zip URL already coded in the ingest script unless NHTSA changes again.
- Category vocabulary: `domains/automotive_nhtsa/data/gazetteers/categories.csv` + `src.domains.nhtsa_components.normalize_component`.

### Non-negotiable semantics

1. **`records.category` must be a closed vocabulary** (normalized component), **not** free-text narrative. Matching advisories, anomalies, clusters, and severity rules use **exact string equality**. A free-text `category` fails silently (matches nothing).
2. **`received_at` is the complaint clock for lead-time honesty.** For recall lead reports, count complaints by `received_at` only, censored to `received_at <= issued_at`. Do not score lead time using event weeks that include post-recall knowledge.
3. **Recall clock is `RCDATE` (Part 573 report-received)**, stored as `advisories.issued_at`. Do **not** use owner notification (`ODATE`) or file-add (`DATEA`) as the primary issued clock.
4. **A hit requires ≥1 week lead.** Same-week or post-issue flags are not hits. Misses and unconfirmed flags stay in the report (honest reporting).
5. **Provenance:** real ingested rows are observed; live FRONTLINE projections may be `observed`/`inferred`. Lead-time scoring should exclude `provenance = 'inferred'` unless a test explicitly wants them.
6. **Qubot / ledger / RBAC:** do not weaken groundedness checks, tamper-evident ledger, or auth gates while adding try-surface tools.

---

## 2. Already done — do not redo (verify, then extend)

Confirm these on branch `fleet-quality-phase0` before writing new code.

### Phase 0 (committed)

| Item | Evidence |
|---|---|
| Normalize NHTSA `COMPDESC` onto gazetteer categories | `src/domains/nhtsa_components.py`, ingest path, commits `e998011` / `a0d2a9a0` family |
| Map `records.category` → normalized `COMPONENT` | `domains/automotive_nhtsa/data/mapping.yaml` (`category: COMPONENT`, `subcategory: COMPDESC`, `text: CDESC`) |
| Stable record key = complaint × vehicle × component | `RECORD_KEY` / `scripts.ingest_nhtsa.record_key` (not bare `CMPLID`) |
| Delete path for a source’s records | purge / source delete path (P0-3a) |
| Batched `upsert_records` | `src/domains/mapping_ingest.py` `upsert_records` |
| Provenance on ingested rows | P0-5 commit |
| `novel_candidates` written to **ops** DB | `persist_novel_candidates` in `src/ml_runtime/clustering.py` uses `ops_con` |

### Phase 1 (mostly present as WIP / untracked — finish & harden, don’t reinvent)

| Item | Paths |
|---|---|
| FLAT_RCL → `advisories` ingest | `scripts/ingest_recalls.py`, Makefile `ingest-recalls` |
| Mapping contract for advisories | `mapping.yaml` `advisories:` block |
| Honest per-recall lead-time report | `src/backtest/recall_lead.py`, `scripts/recall_lead_report.py`, `reports/nhtsa/recall_lead_time.md` |
| Eval gate on a committed sample | `eval/frontline/recall_lead_gate.py`, `eval/frontline/fixtures/recall_lead/*` |
| Anomaly helpers: received clock, censor, `current_spikes` | `src/ml_runtime/anomalies.py` |
| Scheduled scan cadence on worker tick | `src/jobs/worker.py` `maybe_enqueue_scheduled_scans` |
| Early Warning Board spikes UI | `dashboard/routes/EarlyWarningBoard.jsx`, `GET /api/frontline/early-warning` |
| Investigation brief API module | `src/frontline/investigation_brief.py`, `GET` brief route in `frontline.py` |
| Phase 1 tests | `tests/frontline/test_phase1_early_warning.py`, `test_novel_candidates.py`, etc. |

### Still missing / incomplete relative to this prompt’s end state

1. **MCP server** — no FastMCP / MCP package in-repo yet. HTTP brief/spikes exist; MCP wrappers do not.
2. **CSV upload that loads full rows into a pack warehouse** — Pack Builder only **profiles** CSV and seeds ~50 sample rows on draft pack create. There is no first-class “confirm mapping → ingest all rows → run anomalies” API for a prospect CSV.
3. **Phase 1 WIP may be unfinished:** commit remaining Phase 1 files, fix failing tests, regenerate a real (not only sample) lead-time report for a bounded make/year window, document how to reproduce it.
4. **Product polish only as needed for the report/demo** — Issue Desk / brief UI may exist as WIP; do not expand into full Axion Issue 360.

---

## 3. Phase 0 — completion criteria (must stay true)

If any of these regress, fix them before Phase 1 polish.

### 3.1 Category normalization

- Ingest maps NHTSA component hierarchy through `normalize_component` onto `gazetteers/categories.csv`.
- `records.category` is always a vocabulary term (or a documented unknown bucket if you explicitly add one — prefer rejecting/skipping unmappable only with clear metrics, never silently putting CDESC into `category`).
- `records.subcategory` may keep raw `COMPDESC`.
- `records.text` is narrative (`CDESC`).
- Severity / cluster / advisory joins use `category` equality.

**Verify:** `tests/frontline/test_nhtsa_flat_map.py` and a spot check after `make ingest-nhtsa N=…` that `SELECT DISTINCT category FROM records` is a short closed list, not free-text paragraphs.

### 3.2 Identity and volume honesty

- One NHTSA complaint listing multiple components must not explode into N independent “complaints” that inflate weekly counts as if they were N separate ODIs. The composite `RECORD_KEY` policy is the intended fix — keep it.
- Re-ingest is idempotent via upsert on `record_id`.
- Large loads use batched upserts (not per-row commit).

### 3.3 Provenance

- Ingested historical rows tagged so lead-time logic can exclude inferred / projected rows.
- Live call projections into FRONTLINE remain allowed but must not poison recall backtests.

### 3.4 novel_candidates

- Persist only via ops warehouse.
- Domain writers must not assume `novel_candidates` exists in domain DuckDB.

---

## 4. Phase 1 — completion criteria (the proof)

### 4.1 Real recalls in `advisories`

**Script:** `python -m scripts.ingest_recalls` / `make ingest-recalls`.

**Contract (already documented in script + mapping.yaml):**

- Source: NHTSA FLAT_RCL (tab-separated inside zip). Field order is positional; see `FLAT_RCL_FIELDS` in `ingest_recalls.py`.
- `advisory_id` = `CAMPNO|YEARTXT|MAKETXT|MODELTXT|COMPONENT` (one campaign spans many models — that is intentional).
- `issued_at` ← `RCDATE`.
- `scope_category` ← `normalize_component(COMPNAME)` (same function as complaints).
- `summary` ← `DESC_DEFECT`; `remedy` ← `CORRECTIVE_ACTION`; `url` ← NHTSA recall page for campno.
- Filters: `--makes`, `--year-min`, `--year-max`, `--rcl-type` (prefer vehicle recalls `V` for the demo window).

**Acceptance:**

- After ingest aligned to the same make/year window as complaints, `advisories` is non-empty and categories overlap the complaint vocabulary for the demo makes.
- Unit tests in `test_phase1_early_warning.py` for flat-line parsing and mapping contract stay green.

### 4.2 Anomaly / spike honesty

- Support counting by `received_at` only for lead-time runs (`clock='received'`).
- Support right-censoring recent incomplete weeks (`censor_recent_days`) for live board spikes.
- `current_spikes(pack_id)` powers Early Warning Board “this week” without pretending incomplete weeks are full signals.
- Exclude inferred provenance from lead-time complaint sets.

### 4.3 Lead-time report (the deliverable humans read)

**Code:** `src/backtest/recall_lead.py` → `scripts/recall_lead_report.py` → markdown under `reports/nhtsa/`.

**Algorithm (must remain honest):**

1. Load advisories + records (exclude inferred).
2. For each advisory with a parseable `issued_at`:
   - Restrict complaints to `received_at <= issued_at`.
   - Score weekly slices on that censored set (do **not** reuse a global `weekly_anomalies` table that was scored with future weeks in the baseline).
   - Persist or pass that scored series into `run_backtest` so engine and report share one clock.
3. Match slice: same `category` and make (`entity_2` / `scope_entity_2`); optional model grain via flag.
4. **Hit** iff first flagged week is ≥1 week before `issued_at`. Record lead weeks.
5. **Miss** iff no such flag. Keep in report.
6. **Unconfirmed** iff a flagged week/slice has no later matching recall. Keep in report (do not hide false alarms).

**Human output:** markdown table with advisory id, issued date, slice, first flagged week, lead weeks, hit/miss; plus miss list and unconfirmed table.

**Reproduce (document in README or report header):**

```bash
# Example bounded window — adjust makes/years to what fits machine + demo
make ingest-nhtsa N=200000 MAKES=HONDA,TOYOTA,FORD,HYUNDAI,KIA YEAR_MIN=2015 YEAR_MAX=2024
make ingest-recalls MAKES=HONDA,TOYOTA,FORD,HYUNDAI,KIA YEAR_MIN=2015 YEAR_MAX=2024 RCL_TYPE=V
make recall-lead PACK=automotive_nhtsa
```

**Acceptance:**

- Sample eval gate passes: `eval/frontline/recall_lead_gate.py` (committed fixture pack `recall_lead_sample`).
- A regenerated report for the real pack exists at `reports/nhtsa/recall_lead_time.md` (or a dated sibling) with real CAMPNO rows — not only the tiny sample.
- Report explicitly states clock definitions (received_at vs RCDATE) and hit rule (≥1 week).

### 4.4 Scheduled scan

- Worker tick can enqueue `scheduled_scan` per pack on cadence (`FRONTLINE_SCAN_INTERVAL_S`; default daily; `0` disables; test isolation keeps it off unless opted in).
- `scripts/scheduled_scan.py` and job handler share one implementation path (`run_fleet_scan` / equivalent).
- Scan opens/updates investigations with slice fields the brief can read.

### 4.5 Early Warning Board

- Shows current spikes from `current_spikes`.
- Shows cluster/investigation rows with lead-time / matched advisory when backtest data exists.
- Empty states must be honest (“no computed spike”), never fake numbers.

### 4.6 Tests & eval

- `tests/frontline/test_phase1_early_warning.py` — synthetic correctness for recall clock, censoring, cadence.
- `eval/frontline/recall_lead_gate.py` wired into `eval/frontline/run_eval.py` for automotive packs.
- Do not require downloading multi-GB NHTSA files inside CI; use the committed fixture for the gate. Document the full-window report as a manual/make target.

---

## 5. Try surface A — MCP server (read-only)

### Goal

Someone using Claude Desktop, Cursor, or another MCP client can ask “what’s spiking?” and “brief me on investigation X” **without** opening Skew’s dashboard.

### Scope (keep tiny)

Implement a small MCP server process (prefer official Python MCP / FastMCP patterns already common in the ecosystem) that exposes **read tools only**. No write tools in v1 (no create investigation, no ingest, no delete).

### Suggested tools

| Tool | Behavior |
|---|---|
| `list_packs` | List known domain packs (from `list_packs()` / pack loader). |
| `list_spikes` | Args: `pack_id` (default `automotive_nhtsa`). Returns `current_spikes(pack_id)` JSON (spike_week, spikes[], censored_weeks). |
| `list_investigations` | Args: `pack_id`, optional `status`, `limit`. Thin wrapper over existing ops query / frontline list semantics. |
| `get_investigation_brief` | Args: `investigation_id`. Calls `build_investigation_brief` (same payload as HTTP). |
| `get_recall_lead_summary` | Args: `pack_id`. Optional: return last generated report path + top-line counts (hits/misses/unconfirmed) by reading `reports/nhtsa/…` or recomputing if cheap on sample pack only. Prefer reading artifacts for the large pack to avoid multi-minute tool calls. |

### Implementation notes

- Reuse existing Python functions; do **not** duplicate SQL.
- Auth: for local demo, document env vars (e.g. same API token / open mode as the API). Do not invent a second security model. If the HTTP API requires a role, MCP should fail closed when credentials are missing in non-open mode.
- Package layout suggestion: `src/mcp_server/` or `mcp_server/` with `__main__.py` so `python -m …` works; document Claude Desktop / Cursor MCP config snippet in README (stdio).
- Dependencies: add only what is needed; pin reasonably; do not drag a second web framework if stdio MCP suffices.
- Tests: unit-test tool handlers with monkeypatched warehouse (no network).

### Non-goals for MCP v1

- Write tools, file upload via MCP, streaming subscriptions, multi-tenant OAuth, hosting on the public internet.

---

## 6. Try surface B — CSV upload that actually loads rows

### Problem today

`POST /api/pipeline/pack-builder/profile` saves a CSV and proposes a mapping. `build_draft_pack` / `_seed_domain_warehouse` loads only a **small sample**. That is enough for “first insight,” not enough for “here is our warranty export — show spikes.”

### Goal

A prospect (or you) can:

1. Upload a CSV.
2. Confirm / edit column mapping (existing profile + lint).
3. **Ingest all accepted rows** into a pack’s `records` (new pack or existing).
4. Optionally run `recompute_weekly_anomalies` (and optionally cluster rebuild) so Early Warning / MCP `list_spikes` have something real.
5. See row counts, rejected rows, and next actions.

### Design constraints

- Reuse `src.domains.mapping_ingest` / `upsert_records` / pack `mapping.yaml` shape.
- Enforce the same upload safety already in `pack_builder_profile` (`.csv` only, size cap — consider raising the 5 MB demo cap behind an explicit env for pilot loads, or document a CLI path for large files).
- For large files, prefer **enqueue a job** (`src/jobs/queue.py`) over blocking the HTTP request for minutes. Job payload: `pack_id`, `csv_path`, `mapping` (or mapping path), flags for anomaly recompute.
- CLI twin is required for demos without UI: e.g. `python -m scripts.ingest_source --pack … --csv …` (extend existing `scripts/ingest_source.py` if present) so agents and humans are not UI-locked.
- Category column: if the CSV’s category is free text, either (a) require mapping through a gazetteer normalize step for that pack, or (b) document that automotive_nhtsa expects COMPDESC-like values and runs `normalize_component`. Do not write narrative into `category`.
- Idempotency: stable `record_id` from mapping; re-upload upserts.
- Provenance: tag as observed / source name from mapping.

### API / UX sketch (implement the smallest vertical slice)

1. Keep `POST …/pack-builder/profile`.
2. Add `POST …/pack-builder/ingest` (name flexible) with body: `{ pack_id, csv_path, mapping, run_anomalies?: bool }`.
3. Returns: `{ written, skipped, job_id? , anomalies_scored? }`.
4. Dashboard: one button “Load into pack” after profile lint is clean — wire only if low cost; CLI alone is acceptable for Phase 0–1 if UI time is tight, but HTTP ingest should exist so MCP/docs can point at one path.

### Acceptance

- Uploading the committed `eval/frontline/fixtures/recall_lead/complaints.csv` into a temp pack yields row counts matching the fixture and allows `recall_lead_report` / spikes after advisories load.
- Rejects non-CSV and oversize per policy.
- Does not claim warehouse/Snowflake connectivity.

---

## 7. Explicit out of scope (refuse unless the human overrides in writing)

Do **not** implement in this engagement:

- Snowflake / Databricks / BigQuery / S3 batch connectors
- Telematics / CAN / MQTT live streams
- SAP / Oracle ERP, Windchill / Teamcenter PLM
- Jira / ServiceNow two-way sync
- SSO / SAML / SCIM, full multi-tenant isolation productization
- Fix-effectiveness loops that need long post-remedy windows (known gap: short windows → always inconclusive)
- Full Axion-style Issue 360 rebuild
- Cloning Axion marketing surfaces or pretending Skew has Axion’s connector matrix
- Weakening Qubot, ledger chain, or WebSocket/auth fences

If a task looks like “add five connectors,” stop and remind the human of §0.

---

## 8. Working rules for the agent

1. **Search the repo before inventing.** Prefer extending `ingest_recalls`, `recall_lead`, `investigation_brief`, `mapping_ingest`, `anomalies`, `worker` over new parallel systems.
2. **Small diffs.** Match existing style; no drive-by refactors; no new markdown docs unless asked (this prompt file is an exception when the human requested it).
3. **Tests first for logic bugs.** Synthetic unit tests for clocks/censoring; fixture eval gate for lead-time; do not download FLAT files in CI.
4. **Honest empty states.** Never seed fake lead times or hide misses.
5. **DuckDB locking.** One writer; batch; no nested `domain_con` while holding another writer to the same pack.
6. **Secrets.** Never commit `.env`, tokens, or customer CSVs. Customer uploads stay under `data/uploads/` (gitignored).
7. **Commits.** Only when the human asks. Follow repo message style (`feat(P0-…)` / `feat(P1-…)`).
8. **Definition of demo-ready:** (a) sample eval gate green, (b) real-window lead-time markdown generated and readable by a non-engineer quality lead, (c) MCP `list_spikes` + `get_investigation_brief` work locally, (d) one CSV can be loaded end-to-end via CLI or HTTP.

---

## 9. Suggested execution order

```
1. Verify Phase 0 invariants (tests + spot SQL on categories / provenance).
2. Finish & commit Phase 1 WIP: ingest_recalls, recall_lead, eval gate, spikes board, scheduled_scan, tests.
3. Run bounded real ingest + make recall-lead; refresh reports/nhtsa/recall_lead_time.md; sanity-check hits/misses.
4. Implement MCP read server + README config snippet + handler tests.
5. Implement CSV ingest path (job + CLI); prove with recall_lead fixture CSV.
6. Stop. Hand the lead-time report to humans. Do not start connectors.
```

---

## 10. Definition of done (checklist)

- [ ] `records.category` remains closed vocabulary for automotive_nhtsa; narrative stays in `text`.
- [ ] Real FLAT_RCL path loads into `advisories` with RCDATE and normalized COMPNAME.
- [ ] `recall_lead_report` uses received_at ≤ issued_at scoring; hits need ≥1 week lead; misses + unconfirmed retained.
- [ ] `eval/frontline/recall_lead_gate.py` passes in eval runs.
- [ ] `reports/nhtsa/recall_lead_time.md` (or dated) exists for a documented make/year window with reproduction commands.
- [ ] Worker can enqueue scheduled scans; Early Warning shows real spikes or honest empty.
- [ ] MCP server: at least `list_spikes` and `get_investigation_brief` (plus list helpers as needed).
- [ ] CSV → full row load via CLI and/or HTTP job; Pack Builder sample-seed is no longer the only load path.
- [ ] No speculative warehouse/ERP/telematics connectors landed.
- [ ] README or short runbook section: how to regenerate the report and how to attach the MCP server.

---

## 11. One-paragraph brief you can paste to stakeholders

Skew will spend the next few weeks proving early warning on public NHTSA complaints and recalls: correct component categories, real recall campaigns, and a lead-time report that shows which recalls complaint spikes preceded (and which they missed). In parallel we will add a thin MCP read surface and a CSV load path so a quality engineer can try the product from Claude/Cursor or a spreadsheet export. We will not build enterprise connectors until someone brings real private data and asks for a specific system.

---

## 12. First command for the agent

Start by reading:

1. This file.
2. `domains/automotive_nhtsa/data/mapping.yaml`
3. `scripts/ingest_nhtsa.py` (header + record_key + category path)
4. `scripts/ingest_recalls.py` (header + FLAT_RCL contract)
5. `src/backtest/recall_lead.py`
6. `eval/frontline/recall_lead_gate.py`
7. `src/frontline/investigation_brief.py`
8. `src/ml_runtime/anomalies.py` (`recompute_weekly_anomalies`, `current_spikes`)
9. `src/jobs/worker.py` (scheduled scan)
10. `src/api/routes/pipeline.py` (pack-builder upload limits)

Then run the existing frontline tests for Phase 0/1 and the recall lead eval gate. Report what is green vs broken. Only then implement remaining gaps in §9 order.
