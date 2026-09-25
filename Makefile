# Frontline v2 Makefile — mirrors the v1 makefile discipline.
# All targets are idempotent and safe to re-run.
#
# Python: prefer an activated venv. If PY is unset, pick python3 then python.

ifndef PY
  # Prefer project venv when present (pytest / deps live there).
  ifneq (,$(wildcard .venv/bin/python))
    PY := .venv/bin/python
  else
    PY := $(shell command -v python3 2>/dev/null || command -v python 2>/dev/null || echo python3)
  endif
endif
PACK ?= automotive_nhtsa
N ?= 25
Q ?=
ID ?=

.PHONY: help frontline-db seed-domains pack-lint pack-init contact simulate eval-frontline \
        audit ask digest test clean ingest-scale ingest-nhtsa backtest verify-chain ci \
        dashboard-build compileall run run-hardened security-test slm-probes

help:
	@echo "Frontline v2 — common targets:"
	@echo "  make frontline-db            Build ops DuckDB + canonical views + fixture"
	@echo "  make seed-domains            Rebuild automotive + finance domain fixtures only"
	@echo "  make pack-lint PACK=<id>     Schema + referential checks on a pack"
	@echo "  make pack-init SRC=csv PACK=<id>   Pack Builder MVP (CSV → draft pack)"
	@echo "  make ingest-scale PACK=<id> N=10000  Synthetic scale corpus + embeddings"
	@echo "  make ingest-nhtsa N=50000 [CSV=file]  Real NHTSA complaints via mapping.yaml"
	@echo "  make backtest PACK=<id>      Recompute backtest lead_time_weeks"
	@echo "  make verify-chain ID=int_xxx Verify ledger hash chain"
	@echo "  make contact                 Run a scripted text interaction (no mic, deterministic)"
	@echo "  make simulate N=25            Replay N corpus records as simulated contacts"
	@echo "  make eval-frontline           Offline persona eval across both packs"
	@echo "  make audit ID=int_xxx         Rerun Qubot post-contact audit on a contact"
	@echo "  make digest                   Generate the daily Frontline ops digest"
	@echo "  make ask Q=\"...\"              Ask the Qubot ask-data layer a question"
	@echo "  make test                    Run tests/frontline/ unit + contract tests"
	@echo "  make ci                      Full local mirror of GitHub Actions core jobs"
	@echo "  make dashboard-build         Production Vite build of dashboard/"
	@echo "  make run-hardened            Fail-closed API on 127.0.0.1:8000 (needs FRONTLINE_API_KEY + SESSION_SECRET)"
	@echo "  make security-test           Auth + SOC2 engineering baseline tests"
	@echo "  make slm-probes              SLM Phase 0 probe suite (57 honest-baseline cases)"
	@echo "  make clean                    Drop DuckDB files and generated fixtures"

# ── Hardened local run (SOC 2 engineering baseline) ─────────────────────────
run-hardened: dashboard-build
	@test -n "$$FRONTLINE_API_KEY" || (echo "Set FRONTLINE_API_KEY"; exit 1)
	@test -n "$$SESSION_SECRET" || (echo "Set SESSION_SECRET"; exit 1)
	PILOT_HARDENED=1 FRONTLINE_AUTH_REQUIRED=1 FRONTLINE_OPEN_MODE=0 \
	$(PY) -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

security-test:
	$(PY) -m pytest tests/frontline/test_security_harden.py tests/frontline/test_soc2_baseline.py tests/frontline/test_auth.py tests/frontline/test_security_audit_remediations.py -q

# Playwright E2E against a running API (default http://127.0.0.1:8000). Needs E2E_API_KEY or FRONTLINE_API_KEY.
e2e:
	cd dashboard && E2E_API_KEY="$${E2E_API_KEY:-$$FRONTLINE_API_KEY}" BASE_URL="$${BASE_URL:-http://127.0.0.1:8000}" npm run test:e2e

# ── Phase 1: data spine ────────────────────────────────────────────────────
# Intentional pilot rebuild only — wipes data/frontline.duckdb + domain DBs after allow flag.
frontline-db:
	FRONTLINE_ALLOW_DEFAULT_DB_RESET=1 $(PY) -m scripts.seed_frontline_fixtures

# Domain fixtures only (automotive + finance). Uses allow flag + --force.
seed-domains:
	FRONTLINE_ALLOW_DEFAULT_DB_RESET=1 $(PY) -m scripts.seed_domains --force
	FRONTLINE_ALLOW_DEFAULT_DB_RESET=1 $(PY) -m scripts.seed_finance_cfpb --force

# ── Phase 2: pack system ───────────────────────────────────────────────────
pack-lint:
	$(PY) -m src.domains.loader --lint $(PACK)

pack-init:
	$(PY) -m src.domains.builder --src $(SRC) \
		$(if $(ADVISORIES),--advisories $(ADVISORIES),) \
		--pack-id $(PACK) --name "$(or $(NAME),$(PACK))"

ingest-scale:
	FRONTLINE_ALLOW_DEFAULT_DB_RESET=1 $(PY) -m scripts.ingest_scale --pack $(PACK) --n $(or $(N),10000) --force

# Dense one-issue demo pack: scale corpus + warranty/service + lots + traffic.
# Wipes pilot DBs. N=N-scale rows (default 10000 in script).
demo-issue:
	FRONTLINE_ALLOW_DEFAULT_DB_RESET=1 $(PY) -m scripts.seed_issue_demo --force

# Real NHTSA complaints via mapping.yaml (not ingest_scale). Optional: CSV=path
ingest-nhtsa:
	$(PY) -m scripts.ingest_nhtsa --pack $(or $(PACK),automotive_nhtsa) --limit $(or $(N),50000) \
		$(if $(CSV),--csv $(CSV),)

backtest:
	$(PY) -c "from src.backtest.engine import run_backtest, best_lead_time; r=run_backtest('$(PACK)'); print(len(r), 'rows'); print('best', best_lead_time('$(PACK)'))"

verify-chain:
	$(if $(ID),$(PY) -m scripts.verify_ledger_chain --interaction $(ID),$(PY) -m scripts.verify_ledger_chain --all)

cluster:
	$(PY) -c "from src.ml_runtime.clustering import rebuild_clusters; print(rebuild_clusters('$(PACK)'))"

prepare-minilm:
	$(PY) -m scripts.prepare_minilm_onnx --out models/minilm

embedding-backfill:
	$(PY) -m scripts.embedding_backfill --pack $(PACK) $(if $(VERSION),--version $(VERSION),) $(if $(DRY),--dry-run,)

cluster-build:
	$(PY) -m scripts.rebuild_cluster_build --pack $(PACK) $(if $(VERSION),--version $(VERSION),) $(if $(DRY),--dry-run,)

embedding-benchmark:
	$(PY) -m scripts.embedding_benchmark

eval-labels-sample:
	$(PY) -m scripts.eval_labels sample --pack $(PACK) --n 20

eval-labels-agreement:
	$(PY) -m scripts.eval_labels agreement

embedding-eval:
	$(PY) -c "from src.ml_runtime.embedding_eval import evaluate_pairs; from src.ml_runtime.hash_embedder import HashEmbedder; from src.ml_runtime.onnx_embedder import ToySemanticEmbedder; import json; print(json.dumps(evaluate_pairs({'hash': HashEmbedder(), 'toy': ToySemanticEmbedder()}), indent=2))"

# ── Phase 3: agent core (text mode) ───────────────────────────────────────
contact:
	$(PY) -m src.frontline.cli contact

# ── Phase 5: Qubot auditor ─────────────────────────────────────────────────
audit:
	$(PY) -m src.qubot.cli audit --interaction-id $(ID)

ask:
	$(PY) -m src.qubot.cli ask "$(Q)"

digest:
	$(PY) -m src.qubot.cli digest

# ── Phase 7: live ops ──────────────────────────────────────────────────────
simulate:
	$(PY) -m src.frontline.simulator --count $(N) --speed instant

# ── Phase 8: eval ──────────────────────────────────────────────────────────
# Isolated temp DBs — never touch pilot data/frontline.duckdb.
eval-frontline:
	@ROOT=$$(mktemp -d "$${TMPDIR:-/tmp}/skewai-eval-XXXXXX"); \
	echo "eval data root: $$ROOT"; \
	FRONTLINE_TEST_ISOLATION=1 \
	FRONTLINE_DB_PATH=$$ROOT/frontline.duckdb \
	DOMAIN_DB_PATH=$$ROOT/domains \
	$(PY) -m eval.frontline.run_eval; \
	STATUS=$$?; rm -rf "$$ROOT"; exit $$STATUS

# Isolated temp DBs — pytest session also isolates via conftest.
test:
	@ROOT=$$(mktemp -d "$${TMPDIR:-/tmp}/skewai-pytest-XXXXXX"); \
	echo "test data root: $$ROOT"; \
	FRONTLINE_TEST_ISOLATION=1 \
	FRONTLINE_DB_PATH=$$ROOT/frontline.duckdb \
	DOMAIN_DB_PATH=$$ROOT/domains \
	$(PY) -m pytest tests/frontline/ -q --tb=line --timeout=30; \
	STATUS=$$?; rm -rf "$$ROOT"; exit $$STATUS

# SLM Phase 0 probe suite — Appendix cases against real intake code.
# Same isolation as `test`; safe to run alongside a live pilot.
slm-probes:
	@ROOT=$$(mktemp -d "$${TMPDIR:-/tmp}/skewai-probes-XXXXXX"); \
	echo "probe data root: $$ROOT"; \
	FRONTLINE_TEST_ISOLATION=1 \
	FRONTLINE_DB_PATH=$$ROOT/frontline.duckdb \
	DOMAIN_DB_PATH=$$ROOT/domains \
	$(PY) -m pytest tests/frontline/test_slm_probes.py -q --tb=short --timeout=60; \
	STATUS=$$?; rm -rf "$$ROOT"; exit $$STATUS

# Tier 2 batch review — offline post-contact audit & candidate generation.
review-batch:
	$(PY) -m src.ai.review --limit $(or $(N),50)

compileall:
	$(PY) -m compileall -q src scripts eval

dashboard-build:
	cd dashboard && npm run build && test -f dist/index.html

# Rebuild UI then serve the one-box console (refuses a missing dist/index.html).
run: dashboard-build
	$(PY) -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# Full local mirror of .github/workflows/frontline.yml core jobs.
# Does NOT call frontline-db (that wipes pilot data); pack-lint is file-based.
# test + eval use isolated temp warehouses.
ci: compileall
	$(MAKE) pack-lint PACK=automotive_nhtsa
	$(MAKE) pack-lint PACK=finance_cfpb
	$(MAKE) test
	$(MAKE) eval-frontline
	@echo "✅ make ci — all core gates green"

clean:
	rm -f data/frontline.duckdb data/frontline.duckdb.wal
	rm -rf data/domains/*.duckdb
	@echo "Cleaned DuckDB files. Rebuild with: make frontline-db"
