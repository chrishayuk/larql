#!/usr/bin/env bash
# HEAD-FID-2 (docs/head-fid-2.md): R->C yardstick, R->H decision, C->H head in isolation, one binary, one session.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$here" rev-parse --show-toplevel)"
bin="$repo/target/release/larql"
m="$HOME/chris-models"
bank="$m/qbanks/gpt-oss-20b.quality-bank-1.tokens"
R="$m/gpt-oss-20b.vindex3"; C="$m/gpt-oss-20b.nvfp4.vindex3"; H="$m/gpt-oss-20b.nvfp4-head.vindex3"
sha="$(git -C "$repo" rev-parse HEAD)"
src="${1:?usage: run.sh <commit the binary was built from>}"
# The binary may predate this checkout only by documentation: no source, manifest or lockfile may differ.
git -C "$repo" diff --quiet "$src" HEAD -- crates Cargo.toml Cargo.lock || { echo "binary source $src differs from HEAD in code" >&2; exit 1; }
{
  echo "binary: built from $src sha256 $(shasum -a 256 "$bin" | cut -d' ' -f1)"
  echo "checkout: $sha"
  echo "LARQL_* env: $(env | grep -c '^LARQL_' || true) vars"
} > "$here/meta.txt"
measure() { # label ref ref-backend ref-source cand cand-backend
  { date; uptime; } >> "$here/meta.txt"
  "$bin" vindex3 measure --reference "$2" --reference-backend "$3" --reference-source "$4" \
    --candidate "$5" --candidate-backend "$6" --candidate-source stored \
    --bank "$bank" --sequences 69 --label "$1" --output "$here/$1" --provenance "git_sha=$sha" \
    2>&1 | sed "s#$HOME#~#g" | tee "$here/$1.txt"
  sed -i '' "s#$HOME#~#g" "$here/$1/report.json" "$here/$1/receipt.json"
}
measure r-to-c "$R" metal-lowered-f16 auto   "$C" metal-lowered
measure r-to-h "$R" metal-lowered-f16 auto   "$H" metal-lowered
measure c-to-h "$C" metal-lowered     stored "$H" metal-lowered
