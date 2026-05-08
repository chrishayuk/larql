#!/usr/bin/env bash
# Boot the CPU/GPU split compose stack and run a one-shot Gemma prompt
# through the router-backed FFN path.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE=(docker compose -f "$ROOT/deploy/docker/docker-compose.yml")

PROMPT="${PROMPT:-The capital of France is}"
HOST_VINDEX_PATH="${HOST_VINDEX_PATH:-$ROOT/output/gemma-3-4b-it-vindex}"
export FFN_HOST_PORT="${FFN_HOST_PORT:-18080}"
export ATTENTION_HOST_PORT="${ATTENTION_HOST_PORT:-18081}"
export ROUTER_HOST_PORT="${ROUTER_HOST_PORT:-18082}"
ROUTER_URL="${ROUTER_URL:-http://127.0.0.1:${ROUTER_HOST_PORT}}"
FFN_REMOTE_TIMEOUT_SECS="${FFN_REMOTE_TIMEOUT_SECS:-10}"

if [[ -d "$HOST_VINDEX_PATH" ]]; then
  export VINDEX_SOURCE="$HOST_VINDEX_PATH"
  echo "[demo] using local vindex: $HOST_VINDEX_PATH"
else
  echo "[demo] local vindex not found at $HOST_VINDEX_PATH"
  echo "[demo] compose will populate the named Docker volume from HF_REPO=${HF_REPO:-default}"
fi

echo "[demo] building compose images"
"${COMPOSE[@]}" build

echo "[demo] booting compose stack"
BOOT_START=$SECONDS
"${COMPOSE[@]}" up -d

wait_http() {
  local name="$1"
  local url="$2"
  local deadline=$((SECONDS + 60))
  until curl -fsS "$url" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      echo "[demo] $name did not become healthy within 60s: $url" >&2
      "${COMPOSE[@]}" ps >&2 || true
      exit 1
    fi
    sleep 2
  done
  echo "[demo] $name healthy"
}

wait_http ffn "http://127.0.0.1:${FFN_HOST_PORT}/v1/health"
wait_http attention "http://127.0.0.1:${ATTENTION_HOST_PORT}/v1/health"
wait_http router "$ROUTER_URL/v1/health"
echo "[demo] all services healthy in $((SECONDS - BOOT_START))s"

if [[ ! -d "$HOST_VINDEX_PATH" ]]; then
  echo "[demo] no host vindex available for the local attention driver; health smoke complete"
  exit 0
fi

if [[ -x "$ROOT/target/release/larql" ]]; then
  LARQL_BIN=("$ROOT/target/release/larql")
else
  LARQL_BIN=(cargo run --release -p larql-cli --)
fi

echo "[demo] running one-shot prompt through router: $PROMPT"
INFER_START_NS=$(date +%s%N)
"${LARQL_BIN[@]}" dev walk \
  --index "$HOST_VINDEX_PATH" \
  --prompt "$PROMPT" \
  --predict \
  --predict-top-k 3 \
  --max-tokens 1 \
  --ffn-remote "$ROUTER_URL" \
  --ffn-remote-timeout-secs "$FFN_REMOTE_TIMEOUT_SECS"
INFER_END_NS=$(date +%s%N)
ELAPSED_S=$(awk -v start="$INFER_START_NS" -v end="$INFER_END_NS" \
  'BEGIN { printf "%.2f", (end - start) / 1000000000 }')
TOK_S=$(awk -v start="$INFER_START_NS" -v end="$INFER_END_NS" \
  'BEGIN { d = (end - start) / 1000000000; printf "%.2f", (d > 0 ? 1 / d : 0) }')
echo "[demo] one-shot complete in ${ELAPSED_S}s (${TOK_S} tok/s for 1 generated token)"
