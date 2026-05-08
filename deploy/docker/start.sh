#!/usr/bin/env bash
# Launcher for both the FFN and GPU containers.
# Behaviour is gated by ROLE: ffn | attention | all.
#
# Env:
#   ROLE          ffn | attention | all   (default: all)
#   VINDEX_PATH   /data/vindex            (default)
#   HF_REPO       remote vindex on HF Hub (downloaded if VINDEX_PATH is empty)
#   HF_TOKEN      optional HF token for private repos
#   PORT          HTTP port               (default: 8080)
#   GRPC_PORT     gRPC port               (default: 8081)
#   EXPERTS       FFN expert range        (e.g. "0-63") — ffn / all only
#   LAYERS        attention layer range   (e.g. "0-15") — attention / all only
#   KV_FORMAT     fp16 | iso3 | planar3 | iso4 | planar4 (attention/all only)
#   WARMUP        "1" pre-faults expert pages at boot
#   LARQL_BACKEND override compute backend (cpu | metal | cuda)
#   JOIN          router grid URL, e.g. http://router:50052
#   PUBLIC_URL    URL announced to the router, e.g. http://ffn:8080

set -euo pipefail

ROLE="${ROLE:-all}"
VINDEX_DIR="${VINDEX_PATH:-/data/vindex}"
HF_REPO="${HF_REPO:-chrishayuk/gemma-4-26b-a4b-it-vindex-expert-server}"
export VINDEX_PATH="$VINDEX_DIR"
export HF_REPO

# ---- preflight: GPU role needs CUDA runtime ---------------------------------

if [[ "$ROLE" == "attention" || "$ROLE" == "all" ]]; then
  if [[ "${LARQL_BACKEND:-}" != "cpu" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      if ! nvidia-smi >/dev/null 2>&1; then
        echo "[start.sh] no NVIDIA runtime detected. Run with --gpus=all or set LARQL_BACKEND=cpu." >&2
        exit 1
      fi
    elif [[ ! -e /dev/nvidiactl && ! -e /proc/driver/nvidia/version ]]; then
      echo "[start.sh] no NVIDIA runtime detected. Run with --gpus=all or set LARQL_BACKEND=cpu." >&2
      exit 1
    fi
  fi
fi

# ---- vindex bootstrap (download if absent / incomplete) ---------------------

mkdir -p "$VINDEX_DIR"
HAS_INDEX=$([ -f "$VINDEX_DIR/index.json" ] && echo yes || echo no)
HAS_EMBED=$([ -f "$VINDEX_DIR/embeddings.bin" ] && echo yes || echo no)
HAS_WEIGHT=$(
  if compgen -G "$VINDEX_DIR/layers/*.weights" >/dev/null \
    || [ -f "$VINDEX_DIR/interleaved_q4k.bin" ] \
    || [ -f "$VINDEX_DIR/attn_weights_q4k.bin" ] \
    || [ -f "$VINDEX_DIR/weights.bin" ]; then
    echo yes
  else
    echo no
  fi
)
if [[ "$HAS_INDEX" == "no" || "$HAS_EMBED" == "no" || "$HAS_WEIGHT" == "no" ]]; then
  echo "[start.sh] vindex incomplete (index=$HAS_INDEX embed=$HAS_EMBED weight=$HAS_WEIGHT) — fetching $HF_REPO"
  HF_HUB_ENABLE_HF_TRANSFER=1 python3 - <<PYEOF
import os
from huggingface_hub import snapshot_download
repo  = os.environ["HF_REPO"]
dest  = os.environ["VINDEX_PATH"]
token = os.environ.get("HF_TOKEN") or None
print(f"Downloading {repo} → {dest}", flush=True)
snapshot_download(
    repo_id=repo, repo_type="model", local_dir=dest, token=token,
    ignore_patterns=["*.md", ".gitattributes"],
)
print("Download complete.", flush=True)
PYEOF
fi

# ---- build server arg list per role -----------------------------------------

ARGS=(
  "$VINDEX_DIR"
  --port "${PORT:-8080}"
  --host 0.0.0.0
)

# Translate ROLE (compose-friendly) → --role (server-friendly).
# attention-service-routes change.
case "$ROLE" in
  ffn)         ARGS+=(--role expert) ;;
  attention)   ARGS+=(--role attention) ;;
  all|both|*)  ARGS+=(--role both) ;;
esac

if [[ "$ROLE" == "ffn" || "$ROLE" == "all" ]]; then
  [[ -n "${EXPERTS:-}" ]] && ARGS+=(--experts "$EXPERTS")
  [[ "${WARMUP:-0}" == "1" ]] && ARGS+=(--warmup-walk-ffn)
fi

if [[ "$ROLE" == "attention" || "$ROLE" == "all" ]]; then
  [[ -n "${LAYERS:-}" ]] && ARGS+=(--layers "$LAYERS")
  [[ -n "${KV_FORMAT:-}" ]] && ARGS+=(--kv-format "$KV_FORMAT")
  [[ -n "${ATTN_SESSION_TTL:-}" ]] && ARGS+=(--attention-session-ttl-secs "$ATTN_SESSION_TTL")
  [[ -n "${MAX_ATTN_SESSIONS:-}" ]] && ARGS+=(--max-attention-sessions "$MAX_ATTN_SESSIONS")
fi

[[ -n "${GRPC_PORT:-}" ]] && ARGS+=(--grpc-port "$GRPC_PORT")
[[ -n "${JOIN:-}" ]] && ARGS+=(--join "$JOIN")
[[ -n "${PUBLIC_URL:-}" ]] && ARGS+=(--public-url "$PUBLIC_URL")

echo "[start.sh] role=$ROLE backend=${LARQL_BACKEND:-auto} kv=${KV_FORMAT:-fp16}"
echo "[start.sh] exec: larql-server ${ARGS[*]}"
exec larql-server "${ARGS[@]}"
