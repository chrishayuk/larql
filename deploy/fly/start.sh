#!/bin/bash
set -e

VINDEX_DIR="${VINDEX_PATH:-/data/vindex}"
HF_REPO="${HF_REPO:-chrishayuk/gemma-4-26b-a4b-it-vindex-expert-server}"

# Verify the vindex is complete: index.json + embeddings + interleaved FFN + 30 layer files
LAYER_COUNT=$(ls "$VINDEX_DIR/layers/"*.weights 2>/dev/null | wc -l)
HAS_EMBED=$([ -f "$VINDEX_DIR/embeddings.bin" ] && echo yes || echo no)
HAS_FFN=$([ -f "$VINDEX_DIR/interleaved_q4k.bin" ] && echo yes || echo no)
if [ ! -f "$VINDEX_DIR/index.json" ] || [ "$HAS_EMBED" = "no" ] || [ "$HAS_FFN" = "no" ] || [ "$LAYER_COUNT" -lt 30 ]; then
  echo "Vindex incomplete (layers=$LAYER_COUNT/30 embed=$HAS_EMBED ffn=$HAS_FFN) — re-downloading..."
  rm -rf "$VINDEX_DIR"
  mkdir -p "$VINDEX_DIR"
  HF_HUB_ENABLE_HF_TRANSFER=1 python3 - <<PYEOF
import os, sys
from huggingface_hub import snapshot_download

repo_id = os.environ.get("HF_REPO", "chrishayuk/gemma-4-26b-a4b-it-vindex-expert-server")
token   = os.environ.get("HF_TOKEN") or None
dest    = os.environ.get("VINDEX_PATH", "/data/vindex")

print(f"Downloading {repo_id} → {dest}", flush=True)
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    local_dir=dest,
    token=token,
    ignore_patterns=["*.md", ".gitattributes"],
)
print("Download complete.", flush=True)
PYEOF
  echo "Vindex ready at $VINDEX_DIR"
fi

echo "Starting larql-server from $VINDEX_DIR"
echo "  EXPERTS: ${EXPERTS:-all}"
echo "  LAYERS:  ${LAYERS:-all}"

EXTRA_ARGS=""
[ -n "$EXPERTS" ] && EXTRA_ARGS="$EXTRA_ARGS --experts $EXPERTS"
[ -n "$LAYERS"  ] && EXTRA_ARGS="$EXTRA_ARGS --layers $LAYERS"

# --warmup-walk-ffn pre-faults the owned expert pages into RAM at startup.
# This prevents mmap thrashing: pages for the owned shard are hot before the
# first request; pages for other shards are never touched (--experts filter).
# On performance-8x (16 GB), each 64-expert shard needs ~8 GB → 8 GB headroom.
[ "${WARMUP:-0}" = "1" ] && EXTRA_ARGS="$EXTRA_ARGS --warmup-walk-ffn"
[ -n "$GRPC_PORT"  ] && EXTRA_ARGS="$EXTRA_ARGS --grpc-port $GRPC_PORT"

# fly.toml publishes this server's HTTP and gRPC ports on the public
# internet, so it does not start without an API key. Set one with
#   fly secrets set API_KEY=...
# and give clients the same value in LARQL_SHARD_API_KEY.
#
# ALLOW_KEYLESS_PUBLIC=1 starts it with no key. That is for an app whose
# public services have been removed from its fly.toml, so that only the
# private network can reach it; it is never right for the config as shipped.
if [ -n "${API_KEY:-}" ]; then
  AUTH_ARGS=(--api-key "$API_KEY")
elif [ "${ALLOW_KEYLESS_PUBLIC:-0}" = "1" ]; then
  echo "WARNING: starting with no API key (ALLOW_KEYLESS_PUBLIC=1); anyone who can reach this server can use it" >&2
  AUTH_ARGS=(--insecure-public)
else
  echo "refusing to start: API_KEY is not set, and fly.toml publishes this server publicly." >&2
  echo "Run: fly secrets set API_KEY=<key> --app <app>" >&2
  exit 1
fi

# shellcheck disable=SC2086  # EXTRA_ARGS is a list of flags, split on purpose
exec larql-server "$VINDEX_DIR" --port "${PORT:-8080}" --host 0.0.0.0 "${AUTH_ARGS[@]}" $EXTRA_ARGS
