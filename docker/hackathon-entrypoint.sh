#!/bin/sh
# [AI-LAYER] Hackathon single-container entrypoint: seed the demo once, then
# hand over to the unchanged pilot entrypoint (migrations, bootstrap, uvicorn).
set -e
cd /app
export AI_DEMO_ENABLED="${AI_DEMO_ENABLED:-1}"
export AI_EVIDENCE_DIR="${AI_EVIDENCE_DIR:-/app/data/ai_evidence}"
if [ ! -f /app/data/.hackathon_seeded ]; then
  echo "→ Hackathon demo seed (first start only)…"
  python -m scripts.migrate --db ops || true
  FRONTLINE_SEED_DEMO=1 python -m scripts.container_bootstrap || true
  python -m src.frontline.simulator --count "${SIM_COUNT:-15}" --speed instant || true
  python -m scripts.ai_demo_seed --out /app/data/india_seed.json >/dev/null || true
  touch /app/data/.hackathon_seeded
fi
echo "   AI layer mode: $(python -c 'from src.ai_layer.config import ai_mode; print(ai_mode())')"
echo "   Demo path: http://localhost:${API_PORT:-8000}/ui/#demo"
exec /entrypoint.sh
