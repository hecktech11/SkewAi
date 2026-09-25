# Model card: all-MiniLM-L6-v2 Zero-Shot Category Extraction

**model_key:** `minilm-l6-zero-shot-category-v1`

## Intended use

Zero-shot candidate category classification by embedding customer turn text
and evaluating cosine similarity against pack category names and gazetteer
synonym expansions. Operates in `shadow` mode by default
(`FRONTLINE_SLM_MODE=shadow`), providing parallel telemetry without changing
customer-visible behavior. Instant kill switch via `FRONTLINE_SLM_KILL=1`.

## Model architecture and artifact

- Base model: `sentence-transformers/all-MiniLM-L6-v2`
- Quantization: ONNX CPU INT8 (`models/minilm/model.onnx`, 22.8 MB)
- Max sequence length: 256 tokens
- Native dimension: 384 (L2 normalized)
- SHA-256 verification: Fail-closed on any checksum or dimension discrepancy
  (`src/ml_runtime/onnx_embedder.py::validate_artifact_dir`).

## Closed-vocabulary and safety constraints

1. **Closed-vocabulary enforcement:** The model never emits free-form text.
   Outputs must be an exact member of `pack.gazetteer_for_slot("category")`.
   Any candidate not in the pack's gazetteer is rejected.
2. **Confidence floor:** Default floor = 0.30 (`FRONTLINE_SLM_CATEGORY_FLOOR`).
   Scores below the floor resolve to `None` and fall through to asking the
   caller via `_next_required_slot`. Unknown is cheap; wrong is expensive.
3. **Asymmetric authority:** In Phase 1, runs in `shadow` mode. Telemetry
   tracks model-vs-rules agreements, accuracy deltas, and confident-wrong rates.
4. **Exact span citation:** Records the character span `(span_start, span_end)`
   and `span_text` of the turn text it relied on for verification by Qubot
   (`src/qubot/auditor.py`).

## Performance and budgets

- Measured inference latency: **p50 ~1.3 ms, p95 ~2.5 ms, p99 ~5.0 ms** on CPU
  (well within the 15 ms p50 / 50 ms p99 ceiling and 180 ms intake budget).
- Hard timeout: 25 ms (`FRONTLINE_SLM_TIMEOUT_MS`).
- Artifact size: 23.7 MB total (gate: <= 80 MB).
- Warm RSS memory: ~156 MB (gate: <= 512 MB).

## Governance and promotion gates

- Must show higher category accuracy and lower confident-wrong rate on held-out
  human-annotated sets before live promotion.
- Pre-registered promotion gates:
  - Accuracy delta > 0 against rules baseline
  - Confident-wrong rate reduction
  - Zero hallucinated labels outside pack taxonomy
  - Latency within budget under concurrent traffic

## Date and author

2026-09-14. Skew AI ML Platform / Frontline Agent Team.
