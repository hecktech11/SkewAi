# SLM Integration — Implementation Prompt

> Self-contained working prompt for an engineer or coding agent. Assumes no prior
> context on this investigation. Repo: `v2_ai_rca_tool` (Skew AI).
> Authored 2026-09-14 from a measured probe of the live deterministic path.

---

## ROLE

You are implementing a small language model (SLM) into the Skew AI frontline
agent stack. Your job is **not** to add a model wherever one fits. It is to
replace a specific, measured, structurally-capped set of hand-written English
heuristics with a learned component — while leaving the platform's deterministic
decision and audit layers completely untouched.

Read this entire prompt before editing any file. Phase 0 is mandatory and must
land before any model artifact enters the repo.

---

## 1. CONTEXT — what this codebase is and why its design matters

Skew AI is a domain-agnostic voice-of-customer platform. A customer plugs in a
**Domain Pack** (declarative YAML + CSVs) and gets a voice agent, a swarm of
triage/investigation agents, and **Qubot v2**, a deterministic auditor.

Three design commitments constrain everything you do:

1. **"DB computes; agents narrate from templates by default."** Deterministic,
   $0, replayable. Optional LLM narration lives in `src/ai/` behind
   `FRONTLINE_LLM_ENABLED=1` + a provider key, with turn and daily spend caps.
2. **"Domain-as-data, not domain-as-code."** Adding an industry must mean
   writing YAML + CSVs, not Python. This is the commercial thesis.
3. **Audit-first.** Every agent action writes to a hash-chained ledger before
   output is emitted. `src/qubot/auditor.py` re-queries the warehouse to verify
   every cited evidence ID. Its docstring is explicit: *"Deterministic code, not
   LLM judgment."* **This is the moat. Do not put a model inside it.**

### Local-model infrastructure that already exists — reuse it, do not reinvent

| Asset | Location | What it gives you |
|---|---|---|
| ONNX CPU embedder | `src/ml_runtime/onnx_embedder.py` | MiniLM-L6 INT8, **23.7 MB / 156 MB RSS / 1.2 ms p95**. Fail-closed loader: `REQUIRED_FILES`, SHA-256 checksums, manifest, `build_canonical_version()` |
| Shipped artifact | `models/minilm/` | model.onnx + tokenizer + manifest + checksums + LICENSE |
| Mode gating | `src/ml_runtime/embedding_runtime.py` | `FRONTLINE_EMBEDDING_MODE=legacy\|shadow\|live`, `FRONTLINE_EMBEDDING_SHADOW_KILL` |
| Shadow telemetry | `src/ml_runtime/embedding_shadow.py` | `record_shadow_comparison()` — IDs and scores only, no complaint text |
| Model governance | `src/governance/registry.py` | `artifact_version()`, `save_model_card()`, `compare_shadow()`, `fairness_report()` |
| Honest source stamping | `src/agents/triage.py:108` | `SEVERITY_FALLBACK_POLICY` — `source` is `"rules"` (never `"model"`) whenever the model result was not actually used |
| Graceful degradation | `src/observability/degradation.py` | `step_down(component, reason=...)` |

### Budgets you must respect

- Artifact size gate: `FRONTLINE_EMBED_MAX_ARTIFACT_MB` default **80 MB**
- RAM after warmup: `FRONTLINE_EMBED_MAX_RAM_MB` default **512 MB**
- Inference ceilings already in use: **p50 ≤ 15 ms, p99 ≤ 50 ms**
- Live-turn budget for intake: **180 ms** (`intake_llm_first_token`,
  `reports/latency_benchmark.json`)
- Cost headroom: measured **$0.0235/contact** against a **$0.45** target — cost
  is not the constraint. Latency, auditability and artifact size are.
- Deploy constraint: active-call registry is in-process; **one uvicorn worker**.
  A blocking call on the event loop stalls every concurrent call.

---

## 2. THE EVIDENCE — why this work is justified

A 57-case probe was run against the **real** deterministic path (no simulation),
using lowercase, unpunctuated phrasing as the browser Web Speech API actually
delivers it. Results:

| Site | Measured | Target | What `reports/shadow_empirical_01.json` claims |
|---|---|---|---|
| Kill-switch **recall** | **0.350** | ≥ 0.98 | 1.000 |
| Kill-switch **precision** | **1.000** | ≥ 0.85 | 1.000 |
| Category extraction | **0.071** (1/14) | ≥ 0.90 | 1.000 |
| Frustration detection | **0/5** (all scored 0.000) | — | — |
| Safety yes/no | **0.333** (2/6) | — | — |

**The decisive finding.** The 13 missed true emergencies split as:

- **3/13** — a lexicon term *was* present but got suppressed by the fixed
  6-token negator/hedge window in `_term_hedged_or_negated`
  (`src/agents/intake.py:400`). Regex-fixable.
- **10/13** — **no lexicon term was present at all.** The caller paraphrased:
  *"the brake pedal went straight to the floor and i couldnt stop"*,
  *"we hit the guardrail pretty hard"*, *"the wheel wont turn at all"*,
  *"we went off the road and rolled into the ditch"*.

> **Therefore: with perfect negation scoping and no other change, kill-switch
> recall caps at 0.500 against a 0.98 target.** The gap is open-vocabulary
> paraphrase. Closing it with word lists means hand-writing unbounded paraphrase
> per vertical, forever — which is exactly the cost the domain-as-data thesis
> exists to eliminate.

Treat these numbers as **indicative of magnitude and conclusive as to
mechanism**. 57 author-written cases cannot establish a production recall
figure. Phase 0 exists to replace them with a defensible measurement.

### Confirmed defects (fix regardless of the SLM decision)

1. **The baseline is simulated.** `src/frontline/shadow_pilot.py:173` produces
   "AI extraction" as `if rng.random() < 0.95:` — it never calls `IntakeAgent`.
   Every green metric in `reports/shadow_empirical_01.json` is a property of the
   generator. Its transcripts are also built from `_SYMPTOM_TEMPLATES` containing
   the exact gazetteer strings the extractor searches for, which is why category
   scores 150/150.
2. **`classify_yes_no` is blind to which question it answers**
   (`src/agents/intake.py:340`). A caller replying *"im fine"* to
   **"Is anyone hurt?"** maps to polarity `yes`; `escalate_on_for_prompt` returns
   `yes`; the two match → **safety flag + P1 + "do not drive the vehicle"
   escalation script.** A false emergency generated by someone saying they are fine.
3. **Unclear answers re-ask.** A miss returns `None`, so
   *"my son has a cut on his forehead"* causes the agent to ask "Is anyone hurt?"
   again on a live call.
4. **Blocking HTTP on the event loop.** `src/ai/provider.py` uses
   `urllib.request.urlopen(..., timeout=8.0)`, called from `async def run`
   (`src/agents/intake.py:736`) with no `asyncio.to_thread`. On a single-worker
   deploy this can stall all concurrent calls for up to 8 s, against a 180 ms budget.
5. **Dead signals on the flagship channel.** `_ALLCAPS_BUMP` and
   `_EXCLAM_CAPS_BUMP` (`src/agents/sentiment.py:68`) key off capitalisation and
   `!`, which browser STT transcripts largely do not carry.

---

## 3. ARCHITECTURAL PRINCIPLES — non-negotiable

### 3.1 Separate Understanding from Deciding

The codebase currently conflates two jobs inside `intake.py`. Split them:

- **Understanding** — messy speech → structured facts. Open-vocabulary,
  unbounded paraphrase, no ground truth in the DB. **This is the model's only
  territory.**
- **Deciding** — structured facts → severity, priority, escalation, clusters,
  evidence. Closed-world, must replay identically for audit. **Deterministic
  code stays. Do not modify.**

The SLM replaces *English-language heuristics*, never *decision logic*.

### 3.2 Closed-vocabulary outputs only

The model never emits free text. Every output must be checkable against the
active pack:

- **Category** → a label selected from the pack's gazetteer / taxonomy. Reject
  anything not in it.
- **Safety** → `{escalate: bool, concept: <one of pack.safety.escalation_lexicon>,
  span: <exact substring of the turn>}`.
- **Polarity** → `yes | no | unclear`, **conditioned on the question text**.
- **Frustration** → a float in [0,1].

### 3.3 Span citation — extend groundedness to model outputs

Every model-driven extraction must record the **exact span of the turn** it
relied on. Then extend `src/qubot/auditor.py` to verify that span exists verbatim
in the stored turn text. The auditor stays fully deterministic; it simply gains a
new class of claim to verify. This keeps audit-first intact for model outputs.

### 3.4 Asymmetric authority — the rollout safety argument

**Safety / kill-switch — union semantics, widen only:**

```
escalate = regex_hit OR slm_hit
```

The model may only *widen* escalation, never suppress it. The worst case of a bad
SLM is extra false alarms, which are measurable and bounded — it can never
introduce a new miss. And there is room to spend: precision is **1.000** against
a **0.85** floor. That is 15 points of precision available to buy 63 points of
recall. This asymmetry is why the rollout is safe; do not remove it.

**Category — inverted, because wrong is worse than unknown:**

The observed failures are not merely `None`, they are *confidently wrong*:
`"it stalls at traffic lights"` → `EXTERIOR LIGHTING`; `"the wheel shakes in my
hands"` → `WHEELS` instead of `STEERING`. Wrong categories feed the pack's
severity rules (`category in [SERVICE BRAKES, AIR BAGS, FUEL SYSTEM, STEERING]`)
and cluster matching, so they silently misgrade severity and misroute
investigations. Require gazetteer membership **plus** a confidence floor; below
the floor, fall through to asking the customer — `_next_required_slot` already
does this. "Unknown" is cheap.

### 3.5 Files you must NOT put a model into

- `src/qubot/auditor.py` and `src/qubot/retrievers.py` — the deterministic verifier
- `src/agents/triage.py` severity rules and priority matrix
- `src/ledger/*` — hash-chained ledger, Merkle
- The pack YAML decision semantics (`_eval_condition`)

---

## 4. MODEL SELECTION

### Primary: one multi-task **encoder**, 22–70M params

Four heads on a shared trunk:

| Head | Output | Replaces |
|---|---|---|
| Safety escalation | binary + concept + span | `_check_kill_switch` lexicon + hedge window |
| Question-conditioned polarity | yes / no / unclear | `classify_yes_no` |
| Frustration | float [0,1] | `score_text` lexicon |
| Category | multi-label over pack labels | `_CATEGORY_SYNONYMS` (~101 hardcoded terms) |

- Base candidates: `bge-small-en-v1.5`, `e5-small-v2`, `MiniLM-L6`,
  `DeBERTa-v3-small`.
- Export **ONNX INT8** → ~30–90 MB, ~2–8 ms CPU per turn.
- Fits inside the enforced 80 MB artifact / 512 MB RAM / p99 50 ms envelope and
  leaves the 180 ms intake budget essentially untouched.

### Do NOT use a generative SLM on the live path

A 0.5–1.5B model (Qwen3-0.6B, Llama-3.2-1B, Gemma-3-1B class) is the wrong tool
here, and the reasoning must be recorded in the model card:

- Every output is fixed-shape (label, span, score) — generation buys nothing.
- It needs constrained decoding just to guarantee gazetteer membership.
- ~400 MB+ at INT4 **breaches the 80 MB artifact gate**.
- 10–50× slower, and it introduces a hallucination surface.
- An encoder is deterministic (same input → same logits), which the audit story
  depends on.

### Cold start for a new vertical

Three of the four tasks are **domain-universal** — "is this an emergency", "is
this person frustrated", "did they say yes or no" mean the same thing in
automotive and finance. The encoder transfers.

Only **category** is domain-specific, and it can be done **zero-shot with the
MiniLM already in `models/minilm/`**: embed the utterance, embed the pack's
category labels plus gazetteer synonym expansions, take argmax with a confidence
floor. No new artifact, no training run, and the labels come from the pack —
which makes it genuinely domain-as-data. **Do this first (Phase 1).**

### Generative models: offline only

Pack Builder gazetteer/synonym/severity-rule generation from a customer CSV
(`make pack-init`). This is a one-time, per-customer, non-latency-critical job,
so prefer a frontier API model over a local SLM. Never on the live turn path.

---

## 5. CHECKS AND GUARDRAILS

1. **Fix the fake baseline first.** Replace the `rng.random()` simulation in
   `src/frontline/shadow_pilot.py` with real `IntakeAgent` execution. **You may
   not ship a model against a simulated baseline.**
2. **Three modes**, mirroring `embedding_runtime.py` exactly:
   `FRONTLINE_SLM_MODE=legacy|shadow|live`, plus `FRONTLINE_SLM_KILL=1` for
   instant revert with no deploy. Default `legacy`.
3. **Shadow before live.** In shadow mode, log model-vs-rules disagreements
   through the `record_shadow_comparison()` pattern — IDs, labels, scores,
   latency. No customer-visible change. No complaint text in telemetry.
4. **Artifact integrity: copy `onnx_embedder.py` wholesale** — `REQUIRED_FILES`,
   SHA-256 checksum verification, manifest, `build_canonical_version()`,
   fail-closed on any mismatch or unexpected output dimension. Register via
   `governance/registry.py::save_model_card`; write a card alongside
   `docs/models/minilm-l6-onnx-card.md`.
5. **Honest source stamping.** Extend the `triage.py` policy:
   `extraction_source = "rules" | "slm" | "slm+rules"`, where `"slm"` appears
   **only** when a model result actually flowed through. Ledger it per action.
6. **Safe fallback.** Hard timeout ~25 ms → deterministic path + ledger entry +
   `model_error` gauge + `step_down()`. A live call must never block on the model.
7. **Pre-register promotion gates before looking at any result:**
   - Kill-switch recall **≥ 0.98** to promote (currently 0.350)
   - Kill-switch precision floor **≥ 0.85**; alert if shadow drops below 0.90
   - Category: promote only if accuracy rises **and** confident-wrong rate falls.
     Track confident-wrong separately from `None`.
   - Frustration: score against real supervisor-takeover decisions, not lexicon
     agreement.
   - Added latency: **p95 ≤ 25 ms, p99 ≤ 50 ms**; artifact ≤ 80 MB; RSS ≤ 512 MB.
8. **Qubot stays deterministic** and gains the span-verification check from §3.3.
9. **Held-out evaluation.** The Appendix probe set becomes a CI regression gate
   and is therefore training-adjacent. Promotion decisions must use a **separate
   human-labelled set** the model was never tuned against.
10. **Reporting honesty.** Never report a metric derived from synthetic or
    simulated data as a system metric. If labels do not exist, mark the metric
    `waived` and say so — follow the precedent already set for
    `severity_agreement_kappa` and `cluster_agreement` in
    `reports/shadow_empirical_01.json`, and the `n=12` caveat in
    `reports/benchmark_interpretation.md`. Stale green results must be renamed,
    as `shadow_empirical_01.STALE_GREEN.json` already shows.

---

## 6. PHASES AND DEFINITION OF DONE

### Phase 0 — days, no model. **Mandatory gate.**

- Rewire `src/frontline/shadow_pilot.py` to run the real `IntakeAgent` /
  `extract_pack_category` / `_check_kill_switch` / `score_text` path.
- Land the Appendix probe set as a CI regression suite (`make` target + CI job).
- Fix defect **#2** — make polarity classification question-conditioned so
  *"im fine"* answering *"Is anyone hurt?"* resolves to `no`, not an escalation.
- Fix defect **#4** — wrap the blocking provider call in `asyncio.to_thread` with
  a timeout well inside the 180 ms budget.
- Re-run `make eval-frontline` and the shadow harness. Publish the honest numbers,
  regenerate `reports/shadow_empirical_01.json`, and mark the prior file stale.

**Done when:** the shadow harness executes real code, and there is one published,
defensible baseline for every metric in §2.

### Phase 1 — days, no new artifact

- Zero-shot category via the shipped MiniLM: utterance embedding vs. pack label
  and gazetteer embeddings, argmax + confidence floor, gazetteer membership
  enforced, fall through to asking below the floor.
- Run in `shadow` mode. Compare against the Phase 0 baseline.

**Done when:** shadow telemetry shows the category accuracy delta and the
confident-wrong delta, with no customer-visible behaviour change.

### Phase 2 — weeks

- Assemble ~2–4k labelled turns. Bootstrap labels with a frontier model offline,
  then have humans verify — do not hand-label from scratch. Record inter-annotator
  agreement; the existing bar in this repo is Cohen's κ ≥ 0.70.
- Fine-tune the multi-task encoder. Export ONNX INT8 with a full manifest,
  checksums and model card.
- Add `src/ml_runtime/slm_runtime.py` following `onnx_embedder.py` structurally.
- Run in `shadow` mode across both real packs (`automotive_nhtsa`, `finance_cfpb`).

**Done when:** the artifact loads fail-closed, is governed by a model card, and
shadow telemetry covers every §7 gate.

### Phase 3 — promotion

- Promote **per task, independently**. Safety first, with union semantics — the
  recall-only-upside path.
- Exercise `FRONTLINE_SLM_KILL` in `src/routing/day_one_drill.py`.

**Done when:** recall ≥ 0.98 at precision ≥ 0.85 on the held-out human-labelled
set; canonical model version recorded in the ledger for every scored turn; Qubot
verifying spans; kill switch drilled.

### Phase 4 — commercial leverage

- Frontier-model Pack Builder generation from a customer CSV, with mandatory human
  review before `make pack-lint`. Makes new verticals cheap and retires the
  remaining hardcoded English in `intake.py`.

**Done when:** a new vertical can be onboarded without editing Python —
the domain-as-data claim becomes literally true.

---

## 7. NON-GOALS

- Do **not** replace the three narration sites in `src/ai/narration.py` with a
  local model. Cosmetic rephrasing carries hallucination risk for near-zero
  value; the templates are correct as they stand.
- Do **not** move severity or priority scoring to a model.
- Do **not** introduce LangChain or any agent framework — the orchestrator is a
  deliberately explicit, testable state machine.
- Do **not** grow `_CATEGORY_SYNONYMS`, `_FRUSTRATION_LEXICON`, `_NEGATORS` or
  `_HEDGES` as a fix. §2 establishes that this path has a 0.500 recall ceiling.
- Do **not** raise the 80 MB artifact gate to accommodate a generative model.

---

## APPENDIX — seed regression set

Phrased as browser STT delivers it: lowercase, minimal punctuation. Every case
below runs against real code, not a simulation.

### A. True emergencies — kill switch MUST fire (target recall ≥ 0.98)

```
im not sure what happened but theres smoke pouring out of the hood
i dont know if its serious but the engine is on fire right now
could you help me my car just caught fire in the driveway
i think maybe someone is hurt in the other car
we might need an ambulance my wife is bleeding
the brake pedal went straight to the floor and i couldnt stop
i had no braking at all coming down the hill
it wouldnt slow down at all i just kept going through the intersection
the wheel wont turn at all its completely stiff
the car took off on its own and i couldnt stop it
we hit the guardrail pretty hard
the airbag went off and my chest hurts
there was a loud bang and now the steering is gone
my daughter is trapped in the back seat
smoke started coming through the vents while i was driving
i was worried it would catch fire and then it actually did
if you could just note that the car is smoking badly right now
not gonna lie im scared theres flames under the bonnet
we went off the road and rolled into the ditch
i couldnt steer it just went wherever it wanted
```

### B. Must NOT fire — idiom / negation / hedge (precision floor ≥ 0.85)

```
this repair bill is killing me but the brakes just squeak a bit
nobody is hurt and there was no fire just a clicking noise
im worried it might catch fire because of a faint hot smell
the dealer gave me a crash course on the infotainment system
my reading light bulb burned out last week
no smoke no fire just an annoying rattle from the dashboard
i was dying to get this fixed before my road trip
the check engine light came on nothing dramatic happened
the mechanic said the clutch is burning out slowly
i had a fire drill at work so i missed my appointment
```

### C. Category extraction — caller paraphrase (`utterance -> expected label`)

```
theres a grinding noise every time i slow down          -> SERVICE BRAKES
it shudders really bad when i press the pedal           -> SERVICE BRAKES
takes way longer to come to a stop than it used to      -> SERVICE BRAKES
the wheel shakes in my hands on the motorway            -> STEERING
it pulls to the right on a flat road                    -> STEERING
it jerks between gears when im speeding up              -> POWER TRAIN
the dash lights flicker and the screen goes black       -> ELECTRICAL SYSTEM
battery is flat every morning                           -> ELECTRICAL SYSTEM
it wont start and just clicks                           -> ELECTRICAL SYSTEM
theres a knocking sound from under the bonnet when cold -> ENGINE
it stalls at traffic lights                             -> ENGINE
the warning light for the bag stays lit                 -> AIR BAGS
clunking over speed bumps at the front                  -> SUSPENSION
the belt doesnt retract properly                        -> SEAT BELTS
```

### D. Frustration — should cross `FRONTLINE_FRUSTRATION_THRESHOLD` (0.65)

Note these express frustration through **repetition and wasted effort**, not
anger vocabulary — which is why the current lexicon scores all five at 0.000.

```
this is the fourth time ive called about this and nobody calls me back   -> fire
ive been passed around to five different people today                    -> fire
i give up honestly i just want someone to actually help                  -> fire
nobody has taken this seriously since day one                            -> fire
ive wasted three days off work waiting for this                          -> fire
i just want to book it in whenever suits you                             -> no fire
no rush at all just logging it for the record                            -> no fire
```

### E. Question-conditioned polarity — question: **"Is anyone hurt?"** (`yes` escalates)

```
no everyones fine thanks               -> no
im fine                                -> no    (currently yes -> FALSE P1)
everyone is okay                       -> no    (currently yes -> FALSE P1)
were all good no injuries              -> no    (currently None -> re-asks)
my son has a cut on his forehead       -> yes   (currently None -> re-asks)
shes complaining her neck is sore      -> yes   (currently None -> re-asks)
i think shes alright but shes limping  -> yes   (currently None -> re-asks)
```

Same replies against **"Are you in a safe location right now?"** (`no` escalates)
must resolve with the opposite polarity. The current classifier never sees which
question it is answering — that is defect #2.
