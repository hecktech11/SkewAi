#!/usr/bin/env bash
# [AI-LAYER] No-key app smoke for the v2 AI-layer routes (call health, raise review, status page).
# Starts uvicorn on throwaway warehouses, hits the new routes, writes outputs to reports/smoke_v2/.
# Run 1: open mode (no auth). Run 2: auth required, agent vs supervisor sessions on a seeded raise.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/workspace/skewai-venv312/bin/python}
OUT=reports/smoke_v2
PORT=${PORT:-8773}
rm -rf "$OUT"; mkdir -p "$OUT"
TMP=$(mktemp -d /tmp/skewai-smoke-v2-XXXX)
unset META_API_KEY MODEL_API_KEY
export FRONTLINE_DB_PATH="$TMP/frontline.duckdb" DOMAIN_DB_PATH="$TMP/domains" AI_EVIDENCE_DIR="$TMP/ev" FRONTLINE_ENABLED=1
mkdir -p "$DOMAIN_DB_PATH"
B="http://127.0.0.1:$PORT"

start() { "$PY" -m uvicorn src.api.main:app --host 127.0.0.1 --port "$PORT" > "$OUT/$1_server.log" 2>&1 & echo $! > "$TMP/pid";
  for i in $(seq 1 60); do curl -sf "$B/health" >/dev/null && return 0; sleep 1; done; echo "server did not start"; return 1; }
stop() { kill "$(cat "$TMP/pid")" 2>/dev/null; wait "$(cat "$TMP/pid")" 2>/dev/null; sleep 1; }
hit() { # name method path [curl args...]
  local name=$1 m=$2 p=$3; shift 3
  code=$(curl -s -o "$OUT/$name.json" -w '%{http_code}' -X "$m" "$B$p" "$@")
  echo "$m $p -> $code" | tee -a "$OUT/summary.txt"; }

echo "# run 1: open mode, no Meta key  ($(date -u +%FT%TZ))" | tee "$OUT/summary.txt"
start open || exit 1
hit open_health GET /health
hit open_ai_status GET /api/ai/status
hit open_call_health GET /api/ai/call-health
hit open_status_overview GET /api/ai/status/overview
hit open_raises GET /api/ai/raises
hit open_raise_stats GET /api/ai/raises/stats
hit open_review_attempt POST /api/ai/raises/aie_none/review -H 'Content-Type: application/json' -H 'X-Frontline-Role: supervisor' -d '{"verdict":"accept"}'
hit open_ui_page GET /ui/
stop

echo "# seed one AI severity raise + its case into the throwaway ops DB" | tee -a "$OUT/summary.txt"
"$PY" - > "$OUT/seed.json" <<'PYEOF'
import json
from datetime import datetime, timezone
from src.ai_layer.store import record_event
from src.data.warehouse import init_ops_db, ops_con
init_ops_db()
eid = record_event(kind="severity", interaction_id="int_smoke_v2", model_id="muse-spark (SMOKE SEED, not a live call)",
                   deterministic={"severity": "Medium", "source": "rules"},
                   ai_proposal={"severity": "Critical", "confidence": 0.9, "reason": "SMOKE SEED", "evidence": "brakes failed"},
                   accepted={"severity": "Critical"}, reason="raised")
with ops_con() as con:
    con.execute("INSERT INTO cases (case_id, interaction_id, pack_id, created_at, severity, severity_source, priority) "
                "VALUES ('case_smoke_v2', 'int_smoke_v2', 'automotive_nhtsa', ?, 'Critical', 'rules', 1)",
                [datetime.now(timezone.utc).replace(tzinfo=None)])
print(json.dumps({"event_id": eid, "note": "synthetic seed row for the smoke test; no Meta call was made"}))
PYEOF
EID=$("$PY" -c "import json;print(json.load(open('$OUT/seed.json'))['event_id'])")

echo "# run 2: auth required (throwaway key + session secret), no Meta key" | tee -a "$OUT/summary.txt"
export FRONTLINE_AUTH_REQUIRED=1 FRONTLINE_API_KEY="smoke-$(openssl rand -hex 16)" SESSION_SECRET="smoke-$(openssl rand -hex 24)"
unset FRONTLINE_OPEN_MODE FRONTLINE_SERVICE_IS_ADMIN
tok() { "$PY" -c "from src.api.rbac import issue_session; print(issue_session('smoke_$1', '$1', issuer_role='admin')['token'])"; }
AGENT=$(tok agent); SUP=$(tok supervisor); AUD=$(tok auditor)
start auth || exit 1
K=(-H "X-API-Key: $FRONTLINE_API_KEY")
hit auth_no_key_review POST "/api/ai/raises/$EID/review" -H 'Content-Type: application/json' -d '{"verdict":"accept"}'
hit auth_agent_review POST "/api/ai/raises/$EID/review" "${K[@]}" -H "X-Frontline-Session: $AGENT" -H 'Content-Type: application/json' -d '{"verdict":"accept"}'
hit auth_service_review POST "/api/ai/raises/$EID/review" "${K[@]}" -H 'Content-Type: application/json' -d '{"verdict":"accept"}'
hit auth_supervisor_review POST "/api/ai/raises/$EID/review" "${K[@]}" -H "X-Frontline-Session: $SUP" -H 'Content-Type: application/json' -d '{"verdict":"reject","note":"smoke: rule severity was right"}'
hit auth_supervisor_duplicate POST "/api/ai/raises/$EID/review" "${K[@]}" -H "X-Frontline-Session: $SUP" -H 'Content-Type: application/json' -d '{"verdict":"accept"}'
hit auth_raises GET /api/ai/raises "${K[@]}" -H "X-Frontline-Session: $AGENT"
hit auth_raise_stats GET /api/ai/raises/stats "${K[@]}" -H "X-Frontline-Session: $AGENT"
hit auth_status_overview GET /api/ai/status/overview "${K[@]}" -H "X-Frontline-Session: $AGENT"
hit auth_audit_verify GET /api/ai/raises/audit/verify "${K[@]}" -H "X-Frontline-Session: $AUD"
stop
"$PY" - >> "$OUT/summary.txt" <<'PYEOF'
from src.data.warehouse import ops_con
with ops_con() as con:
    print("case severity after verdicts:", con.execute("SELECT severity, priority FROM cases WHERE case_id='case_smoke_v2'").fetchone())
PYEOF
rm -rf "$TMP"
echo "done" | tee -a "$OUT/summary.txt"
