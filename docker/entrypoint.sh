#!/bin/sh
set -e
cd /app

# Step 0.1b (audit 0.5): migrations BEFORE seeding/ingest. Ingest scripts
# refuse to run against a behind-HEAD schema, so ordering is enforced, not
# just documented.
echo "→ Applying database migrations…"
python -m scripts.migrate --db ops || echo "   (ops migrate deferred — app will apply idempotent DDL)"
for _pack in automotive_nhtsa finance_cfpb; do
  DOMAIN_PACK="$_pack" python -m scripts.migrate --db domain --pack "$_pack" || true
done

# Create missing schemas. FRONTLINE_SEED_DEMO=1 fills a domain file only
# when it is absent. This never resets an existing warehouse and never
# treats a non-CFPB record id as corruption.
python -m scripts.container_bootstrap

# Fail-closed when production-like (also enforced in app lifespan).
if [ "${ENV:-}" = "production" ] || [ "${ENV:-}" = "prod" ] \
   || [ "${ENV:-}" = "staging" ] || [ "${ENV:-}" = "stage" ] \
   || [ "${PILOT_HARDENED:-0}" = "1" ] || [ "${SOC2_MODE:-0}" = "1" ]; then
  if [ -z "${FRONTLINE_API_KEY:-}" ]; then
    echo "ERROR: FRONTLINE_API_KEY required when ENV=production/staging or PILOT_HARDENED/SOC2_MODE=1" >&2
    exit 1
  fi
  if [ "${#FRONTLINE_API_KEY}" -lt 32 ]; then
    echo "ERROR: FRONTLINE_API_KEY must be at least 32 bytes in hardened/production mode" >&2
    exit 1
  fi
  if [ -n "${SESSION_SECRET:-}" ] && [ "${#SESSION_SECRET}" -lt 32 ]; then
    echo "ERROR: SESSION_SECRET must be at least 32 bytes in hardened/production mode" >&2
    exit 1
  fi
  if [ "${FRONTLINE_OPEN_MODE:-0}" = "1" ]; then
    echo "ERROR: FRONTLINE_OPEN_MODE=1 is not allowed in hardened/production mode" >&2
    exit 1
  fi
  export FRONTLINE_AUTH_REQUIRED="${FRONTLINE_AUTH_REQUIRED:-1}"
  export FRONTLINE_OPEN_MODE=0
  echo "→ Hardened mode: auth required, open mode off"
fi

# Fail-closed wildcard bind (item 1): API_HOST=0.0.0.0 without explicit auth
# refuses to start, even outside hardened mode. Loopback-only dev keeps the
# intentional open escape hatch; acknowledged local pilots opt in explicitly.
case "${API_HOST:-0.0.0.0}" in
  127.*|localhost|::1)
    ;;
  *)
    if [ "${FRONTLINE_AUTH_REQUIRED:-0}" != "1" ] && [ "${FRONTLINE_OPEN_BIND_ACK:-0}" != "1" ]; then
      echo "ERROR: API_HOST=${API_HOST:-0.0.0.0} binds beyond loopback but FRONTLINE_AUTH_REQUIRED=1 is not set." >&2
      echo "Refusing to start open on a shared network. Set FRONTLINE_AUTH_REQUIRED=1 + a 32-byte FRONTLINE_API_KEY," >&2
      echo "bind API_HOST=127.0.0.1 for local dev, or set FRONTLINE_OPEN_BIND_ACK=1 for an acknowledged local pilot." >&2
      exit 1
    fi
    ;;
esac

echo "→ Starting Skew AI API on ${API_HOST:-0.0.0.0}:${API_PORT:-8000}"
echo "   Dashboard (if built): http://localhost:${API_PORT:-8000}/ui/"
echo "   Workers: 1 (in-process call registry — do not pass --workers / do not scale replicas)"
if [ -n "${FRONTLINE_API_KEY:-}" ]; then
  echo "   Auth: FRONTLINE_API_KEY is set"
else
  echo "   Auth: no FRONTLINE_API_KEY (open local / pilot mode — not for shared networks)"
fi

exec python -m uvicorn src.api.main:app --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}"
