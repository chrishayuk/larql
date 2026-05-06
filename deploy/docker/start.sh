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

set -euo pipefail

ROLE="${ROLE:-all}"
VINDEX_DIR="${VINDEX_PATH:-/data/vindex}"
HF_REPO="${HF_REPO:-chrishayuk/gemma-4-26b-a4b-it-vindex-expert-server}"

# ---- preflight: GPU role needs CUDA runtime ---------------------------------

if [[ "$ROLE" == "attention" || "$ROLE" == "all" ]]; then
  if [[ "${LARQL_BACKEND:-}" != "cpu" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    if ! nvidia-smi >/dev/null 2>&1; then
      echo "[start.sh] no NVIDIA runtime detected. Run with --gpus=all or set LARQL_BACKEND=cpu." >&2
      exit 1
    fi
  fi
fi

# ---- vindex bootstrap (download if absent / incomplete) ---------------------

mkdir -p "$VINDEX_DIR"
LAYER_COUNT=$(ls "$VINDEX_DIR/layers/"*.weights 2>/dev/null | wc -l)
HAS_INDEX=$([ -f "$VINDEX_DIR/index.json" ] && echo yes || echo no)
HAS_EMBED=$([ -f "$VINDEX_DIR/embeddings.bin" ] && echo yes || echo no)
if [[ "$HAS_INDEX" == "no" || "$HAS_EMBED" == "no" || "$LAYER_COUNT" -lt 1 ]]; then
  echo "[start.sh] vindex incomplete (index=$HAS_INDEX embed=$HAS_EMBED layers=$LAYER_COUNT) — fetching $HF_REPO"
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
  --role "$ROLE"
)

if [[ "$ROLE" == "ffn" || "$ROLE" == "all" ]]; then
  [[ -n "${EXPERTS:-}" ]] && ARGS+=(--experts "$EXPERTS")
  [[ "${WARMUP:-0}" == "1" ]] && ARGS+=(--warmup-walk-ffn)
fi

if [[ "$ROLE" == "attention" || "$ROLE" == "all" ]]; then
  [[ -n "${LAYERS:-}" ]] && ARGS+=(--layers "$LAYERS")
  [[ -n "${KV_FORMAT:-}" ]] && ARGS+=(--kv-format "$KV_FORMAT")
fi

[[ -n "${GRPC_PORT:-}" ]] && ARGS+=(--grpc-port "$GRPC_PORT")

echo "[start.sh] role=$ROLE backend=${LARQL_BACKEND:-auto} kv=${KV_FORMAT:-fp16}"
echo "[start.sh] exec: larql-server ${ARGS[*]}"
exec larql-server "${ARGS[@]}"
