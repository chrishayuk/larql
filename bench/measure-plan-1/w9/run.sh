#!/usr/bin/env bash
# MEASURE-PLAN-1 W9: R (metal-lowered-f16, source) against C (metal-lowered, NVFP4 pack) on gpt-oss-20b,
# over all 69 Q-BANK-1 sequences. Records; judges nothing.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$here" rev-parse --show-toplevel)"
bin="$repo/target/release/larql"
models="$HOME/chris-models"
sha="$(git -C "$repo" rev-parse HEAD)"
{
  echo "binary: $sha (release) sha256 $(shasum -a 256 "$bin" | cut -d' ' -f1)"
  echo "tree clean outside bench/ and docs/: $(git -C "$repo" status --porcelain -- crates | wc -l | tr -d ' ') changed crate paths"
  date
  uptime
  echo "LARQL_* env: $(env | grep -c '^LARQL_' || true) vars"
} > "$here/meta.txt"
"$bin" vindex3 measure \
  --reference "$models/gpt-oss-20b.vindex3" --reference-backend metal-lowered-f16 \
  --candidate "$models/gpt-oss-20b.nvfp4.vindex3" --candidate-backend metal-lowered --candidate-source stored \
  --bank "$models/qbanks/gpt-oss-20b.quality-bank-1.tokens" --sequences 69 \
  --label w9-gptoss-f16-vs-conservative --output "$here/out" \
  --provenance "git_sha=$sha" 2>&1 | sed "s#$HOME#~#g" | tee "$here/run.txt"
