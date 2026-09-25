# Frontline Semantic Upgrade: Benchmark & Evaluation Interpretation

**Date:** 2026-09-06  
**Status:** Audit & Evidence Governance  
**Git Provenance:** `b0e923ea375c1eb8f86e3c5ec644cc8605c461f0` (tagged `v1.0-embed-setup`)  
**Runtime Mode:** `legacy` (`FRONTLINE_EMBEDDING_MODE=legacy`)  
**Acceptance Gate Status:** **CLOSED** (`acceptance_ready: false`)

---

## 1. Executive Summary

This document provides the authoritative interpretation of the acoustic and semantic benchmarks recorded during Phase 0 implementation of the frontline embedding engine upgrade (`all-MiniLM-L6-v2` ONNX INT8).

The recorded numbers demonstrate **technical regression readiness and operational safety**, but **do not constitute statistical evidence of domain accuracy**. Specifically:
1. The evaluated pair dataset (`eval/embeddings/pairs_v1.jsonl`) consists of **$n = 12$ author seed pairs**, explicitly flagged with `acceptance_eligible: false`.
2. No production quality, recall ceiling, or domain macro-$F_1$ can be claimed from an $n = 12$ seed fixture.
3. The embedding cutover gate remains **strictly closed** until recruited external human domain experts provide independent labels meeting the pre-registered Cohen's $\kappa \ge 0.70$ inter-annotator agreement threshold.

---

## 2. Benchmark Architecture & Performance Results

Inference benchmarking was performed on host hardware (`arm64`, macOS, 1 execution thread) under `.venv/bin/python` using ONNX Runtime with quantized INT8 weights.

Source: `reports/embedding_benchmark.json`

| Metric | Measured Value | Operational Threshold / Constraint |
| :--- | :--- | :--- |
| **Model Artifact** | `all-MiniLM-L6-v2` INT8 ONNX | `models/minilm/model.onnx` (SHA-256 verified) |
| **Artifact Size** | 23.69 MB (23,687,085 bytes) | Hard cap: < 50 MB |
| **Resident Memory (RSS)** | 156.69 MB (+98.05 MB delta) | Hard ceiling: 250 MB |
| **Cold Initialization** | 64.5 ms | Sub-second process initialization |
| **Warm Single Latency ($p50$)** | 1.00 ms | Ceiling: 25.0 ms |
| **Warm Single Latency ($p95$)** | 1.20 ms | Ceiling: 50.0 ms |
| **Warm Single Latency ($p99$)** | 1.20 ms | Ceiling: 50.0 ms |
| **Batch-8 Latency ($p50$)** | 7.50 ms (0.94 ms / item) | Backfill throughput optimization |
| **Batch-8 Latency ($p95$)** | 7.90 ms | Predictable backfill runtime |
| **Inference Throughput** | 1,051.88 texts / sec | Far exceeds telephony ingestion volume |
| **Baseline Lexical Hash** | 0.062 ms ($62 \mu\text{s}$) / call | Blake2b-512 token/bigram projection |

### Performance Interpretation
- The INT8 ONNX runtime delivers single-vector inference at $1.2\text{ ms } (p95)$, well within the 25ms budget allocated to frontline embedding.
- Cold boot overhead ($64.5\text{ ms}$) and memory footprint ($156.7\text{ MB}$) satisfy single-process deployment constraints on constrained edge/worker nodes.
- Operational latency is verified safe for parallel background shadow execution without degrading telephony turn TTFA.

---

## 3. Evaluation Dataset Caveats & Seed Analysis

Source: `reports/embedding_eval.json` and `eval/embeddings/pairs_v1.jsonl`

### Dataset Constraints
- **Sample Size ($n$):** 12 pairs.
- **Label Source:** `author_seed` / `coding_agent_seed`.
- **Review State:** `ineligible`.
- **Acceptance Gate Eligibility:** `acceptance_eligible: false` on **100% of rows**.

### Relative Comparison Results (MiniLM INT8 vs. Lexical Hash)

| Metric | Lexical Hash (Blake2b-512) | MiniLM INT8 (384-dim zeropad 512) | Relative Improvement |
| :--- | :--- | :--- | :--- |
| **Positive Pair Mean Similarity** | 0.1133 | 0.4046 | $+0.2913$ (+257%) |
| **Negative Pair Mean Similarity** | 0.2184 | 0.2449 | $+0.0265$ |
| **ROC-AUC** | 0.2778 | 0.7222 | $+0.4444$ (+160%) |
| **Average Precision (AP)** | 0.4549 | 0.7052 | $+0.2503$ (+55%) |

### Key Paraphrase Separation Example
- **Query:** `"vehicle stalled at 65 mph"`
- **Candidate:** `"engine died on highway at speed"`
- **Hash Cosine Similarity:** `0.1953` (fails to correlate due to disjoint vocabulary)
- **MiniLM INT8 Cosine Similarity:** `0.4350` (correctly captures semantic alignment)

---

## 4. Why This Is NOT a Production Acceptance Proof

The primary pitfall of early ML evaluations is mistaking smoke-test separation on author fixtures for generalizable production readiness.

1. **Small-$n$ Overfitting & Variance:**
   With $n = 12$, confidence intervals on ROC-AUC ($0.7222$) span across wide margins. Single-digit shifts in pairwise rankings alter metrics drastically.
2. **Author-Generated Bias:**
   Because the test pairs were crafted by the system builders, the examples reflect the authors' mental models of failure modes rather than the noisy, uncurated, dialect-heavy utterances of real automotive callers.
3. **No Negation & Domain-Edge Calibration:**
   While the seed set includes edge cases (such as negated symptoms like *"brakes do not grind they are silent and firm"*), 12 rows cannot calibrate optimal cosine decision thresholds (`CLUSTER_MAX_DISTANCE=0.85`, `NOVELTY_MIN_SCORE=3.0`).
4. **Zero Macro-$F_1$ Metric:**
   The evaluation script explicitly states: `"Not a representative human-labeled study. No Macro-F1."`

---

## 5. Gate Invariants and Pre-Conditions for Cutover

The system enforces programmatic barriers to prevent early cutover:

1. **`src/eval/embedding_gates.py` Acceptance Filter:**
   ```python
   def acceptance_eligibility(rules=None):
       # Checks per-class Cohen's kappa >= 0.70 across >= 2 independent human annotators
       # and verifies quotas from decision_rules_v1.yaml
   ```
   If any row originates from `INELIGIBLE_SOURCES` (`coding_agent_seed`, `author_seed`, `synthetic`), `acceptance_ready` evaluates to `False`.

2. **Configuration Pinning:**
   - Default environment remains `FRONTLINE_EMBEDDING_MODE=legacy`.
   - `FRONTLINE_ACTIVE_CLUSTER_BUILD_ID` remains unset.
   - Startup health check (`validate_startup_config()`) fails closed if semantic mode is attempted without a verified cluster build.

3. **Required Next Steps Prior to Semantic Cutover:**
   - Complete annotator recruitment (1 automotive warranty engineer, 1 contact-center lead, 1 independent adjudicator).
   - Label stratified sample of $\ge 250$ real contact pairs generated by `scripts.eval_labels sample`.
   - Calculate human inter-annotator agreement ($\kappa \ge 0.70$).
   - Run calibrated threshold tuning and cluster stability verification.

---

## 6. Frontline Intake Dual-Stream Scorecard & Adversarial Paraphrase Baseline

**Date:** 2026-09-14  
**Artifact:** `reports/shadow_empirical_02.json` (supersedes `reports/shadow_empirical_01.STALE_GREEN.json`)  
**Verdict:** **RED (BLOCKED)** — driven exclusively by `adversarial_paraphrase` stream

### Why the Templated Stream Cannot Support an Accuracy Claim

The original evaluation pipeline (`reports/shadow_empirical_01.STALE_GREEN.json`) reported near-perfect green metrics:
- Kill-switch recall: `1.000`
- Category slot accuracy: `1.000`
- Overall verdict: `GREEN`

Investigation proved this green scorecard was self-fulfilling:
1. **Keyword Overfitting by Construction:**
   Transcripts in `synthetic_templated` were synthesized from templates containing the exact gazetteer keywords and escalation lexicon searched for by the extractors (e.g. `"brake pedal"`, `"check engine light"`, `"airbag warning light"`, `"flames"`, `"crash"`).
2. **Open-Vocabulary Paraphrase Blindness:**
   Real human callers do not speak in domain pack category labels. They describe sensory symptoms and functional consequences using diverse, unpunctuated phrasing:
   - *"the pedal went straight down to the mat and i could not get the car to halt"* (SERVICE BRAKES)
   - *"the wheel shakes violently in my hands on the motorway"* (STEERING)
   - *"it shudders between gears when im speeding up"* (POWER TRAIN)
   - *"all dashboard needles dropped to zero and displays went dark"* (ELECTRICAL SYSTEM)
3. **The Honest RED Baseline (`reports/shadow_empirical_02.json`):**
   When evaluated on open-vocabulary caller paraphrase without verbatim gazetteer tokens, the real deterministic intake path measures:
   - **Kill-Switch Recall:** `0.5000` [RED] (target $\ge 0.98$)
   - **Kill-Switch Precision:** `1.0000` [GREEN] (target $\ge 0.85$, no false alarms on idiomatic/hedged traps)
   - **Category Extraction Accuracy:** `0.0267` [RED] (target $\ge 0.90$)
   - **Severity Agreement ($\kappa$):** `0.1623` [RED] (target $\ge 0.70$)
   - **Overall Verdict:** `RED`

### Policy: Dual-Stream Reporting Requirement

Moving forward, shadow pilot evaluations must report both streams:
1. `corpus: "synthetic_templated"` — **Integration Smoke Test Only**. Confirms data plumbing, extractor execution, CI pipelines, and unit cost modeling. Explicitly **not** an accuracy claim.
2. `corpus: "adversarial_paraphrase"` — **System Accuracy Claim**. Evaluates real open-vocabulary understanding and drives `overall_verdict`. Live customer pilot traffic remains blocked until learned components (Phase 1 zero-shot category and Phase 2 encoder) lift adversarial accuracy above the pre-registered gates.
