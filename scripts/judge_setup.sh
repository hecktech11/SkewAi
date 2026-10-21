#!/usr/bin/env bash
# [AI-LAYER] One-command judge setup for the SkewAi hackathon demo.
#
#   ./scripts/judge_setup.sh            # install, seed the demo, serve on http://127.0.0.1:8000
#   META_API_KEY=... ./scripts/judge_setup.sh   # same, with the optional Meta AI layer on
#   ./scripts/judge_setup.sh --no-serve # install + seed only
#
# Everything lives in data/hackathon/ (never the pilot data/frontline.duckdb).
# Without META_API_KEY the deterministic path runs every step (identical to the
# base product) and photo upload still works; with it, Muse Spark / SAM 3.1 add
# Hindi normalisation, slot filling, grounded briefs and photo analysis.
set -euo pipefail
cd "$(dirname "$0")/.."

SERVE=1
for a in "$@"; do [ "$a" = "--no-serve" ] && SERVE=0; done

PYBIN="${PYTHON:-}"
if [ -z "$PYBIN" ]; then
  for c in python3.12 python3.13 python3; do command -v "$c" >/dev/null 2>&1 && { PYBIN="$c"; break; }; done
fi
[ -n "$PYBIN" ] || { echo "Python 3.12+ is required" >&2; exit 1; }

echo "→ [1/5] Python env (.venv) + dependencies"
[ -x .venv/bin/python ] || "$PYBIN" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements.txt -c requirements-lock.txt

echo "→ [2/5] Dashboard build"
if command -v npm >/dev/null 2>&1; then
  (cd dashboard && { [ -f package-lock.json ] && npm ci --silent || npm install --silent; } && npm run build --silent)
else
  echo "   npm not found: skipping the dashboard build (API still works; UI needs dashboard/dist)"
fi

export FRONTLINE_DB_PATH="$PWD/data/hackathon/frontline.duckdb"
export DOMAIN_DB_PATH="$PWD/data/hackathon/domains"
export AI_EVIDENCE_DIR="$PWD/data/hackathon/ai_evidence"
export FRONTLINE_ENABLED=1 AI_DEMO_ENABLED=1
mkdir -p "$DOMAIN_DB_PATH"

if [ ! -f data/hackathon/.seeded ]; then
  echo "→ [3/5] Demo warehouses (fixture corpus, data/hackathon/)"
  .venv/bin/python -m scripts.seed_frontline_fixtures
  echo "→ [4/5] Simulated traffic (opens early-warning investigations) + seeded India scenario"
  .venv/bin/python -m src.frontline.simulator --count "${SIM_COUNT:-15}" --speed instant
  .venv/bin/python -m scripts.ai_demo_seed --out data/hackathon/india_seed.json >/dev/null
  touch data/hackathon/.seeded
else
  echo "→ [3-4/5] data/hackathon already seeded (delete data/hackathon/ to start over)"
fi

KEY_FILE=data/hackathon/operator_api_key
[ -f "$KEY_FILE" ] || .venv/bin/python -c "import secrets; print(secrets.token_hex(24))" > "$KEY_FILE"
chmod 600 "$KEY_FILE"
export FRONTLINE_API_KEY="$(cat "$KEY_FILE")" FRONTLINE_AUTH_REQUIRED=1

echo
echo "AI layer: $(.venv/bin/python -c 'from src.ai_layer.config import ai_mode; print(ai_mode())')"
echo "Seeded India case: $(.venv/bin/python -c "import json;d=json.load(open('data/hackathon/india_seed.json'));print(d['case_id'], d['severity'], d['investigation_id'])")"
echo "Operator API key (local only, paste on the Sign in page → 'Have an operator API key?'):"
echo "   $FRONTLINE_API_KEY"
echo "Demo path: http://127.0.0.1:8000/ui/#demo"
[ "$SERVE" = "1" ] || exit 0
echo "→ [5/5] Serving on http://127.0.0.1:8000 (Ctrl-C to stop)"
exec .venv/bin/python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
