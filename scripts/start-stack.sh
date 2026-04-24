#!/usr/bin/env bash
#
# start-stack.sh
#
# One-command launcher: brings up the AutoCodeAI backend via docker compose,
# waits for its /health endpoint, then execs opencode with the plugin enabled.
#
# Usage:
#   ./scripts/start-stack.sh             # starts stack + launches opencode
#   ./scripts/start-stack.sh --no-tui    # starts stack only (for CI/tests)
#   ./scripts/start-stack.sh --down      # tears down the sidecar containers
#
# Environment:
#   AUTOCODEAI_URL   default http://localhost:8000
#   HEALTH_TIMEOUT   seconds to wait for /health (default 60)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${ROOT_DIR}/docker-compose.integration.yml"
AUTOCODEAI_URL="${AUTOCODEAI_URL:-http://localhost:8000}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-60}"

log()  { printf '\033[1;36m[stack]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[stack]\033[0m %s\n' "$*" >&2; exit 1; }

case "${1:-}" in
  --down)
    log "Tearing down AutoCodeAI sidecar..."
    docker compose -f "$COMPOSE_FILE" down
    exit 0
    ;;
esac

command -v docker  >/dev/null || fail "docker not found on PATH"
command -v opencode >/dev/null || log   "opencode not found on PATH — will skip launching it"

log "Starting AutoCodeAI + ChromaDB..."
OPENCODE_PROJECT_DIR="$(pwd)" docker compose -f "$COMPOSE_FILE" up -d

log "Waiting for ${AUTOCODEAI_URL}/health (timeout: ${HEALTH_TIMEOUT}s)..."
for i in $(seq 1 "$HEALTH_TIMEOUT"); do
  if curl -fsS "${AUTOCODEAI_URL}/health" >/dev/null 2>&1; then
    log "AutoCodeAI is healthy."
    break
  fi
  if [ "$i" -eq "$HEALTH_TIMEOUT" ]; then
    log "Backend did not become healthy in time. Recent logs:"
    docker compose -f "$COMPOSE_FILE" logs --tail=40 autocodeai || true
    fail "Giving up. Try: docker compose -f $COMPOSE_FILE logs -f autocodeai"
  fi
  sleep 1
done

if [ "${1:-}" = "--no-tui" ]; then
  log "Stack up. Not launching opencode (--no-tui)."
  exit 0
fi

if ! command -v opencode >/dev/null; then
  log "Stack up at ${AUTOCODEAI_URL}. Install opencode and re-run, or use --no-tui."
  exit 0
fi

log "Launching opencode..."
exec opencode "$@"
