#!/usr/bin/env bash
set -euo pipefail

repo="https://github.com/johndpope/llama-cpp-turboquant.git"
commit="08e025c06ab521e4fa9e5c08b80af57614543e53"
root="$(git rev-parse --show-toplevel)"
vendor="$root/crates/larql-rotorquant/cuda/upstream"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

git clone --quiet --depth 1 --filter=blob:none "$repo" "$tmp/repo"
git -C "$tmp/repo" fetch --quiet --depth 1 origin "$commit"
git -C "$tmp/repo" checkout --quiet "$commit"

declare -A files=(
  ["ggml/src/ggml-cuda/cpy-planar-iso.cu"]="cpy-planar-iso.cu"
  ["ggml/src/ggml-cuda/cpy-planar-iso.cuh"]="cpy-planar-iso.cuh"
  ["ggml/src/ggml-cuda/planar-iso-constants.cuh"]="planar-iso-constants.cuh"
  ["ggml/src/ggml-cuda/set-rows-planar-iso.cuh"]="set-rows-planar-iso.cuh"
)

status=0
for src in "${!files[@]}"; do
  dst="${files[$src]}"
  if ! diff -u "$tmp/repo/$src" "$vendor/$dst"; then
    status=1
  fi
done

if [[ "$status" -ne 0 ]]; then
  echo "rotorquant-sync: vendored CUDA sources differ from $repo@$commit" >&2
  exit "$status"
fi

echo "rotorquant-sync: vendored CUDA sources match $repo@$commit"
