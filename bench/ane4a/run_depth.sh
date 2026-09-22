#!/usr/bin/env bash
# ANE-4A1 — one depth of the ladder: generate, score, delete.
#
#   ./bench/ane4a/run_depth.sh 16
#
# The raw draft planes are large (523 positions x 248,320 f32 ~ 520 MB
# per depth) and the data volume has ~12 GiB free, so a depth's dumps are
# deleted the moment its metrics are banked. Only the scorer output is
# kept — it is small, and it is the part that carries the result.
#
# Backend and format policy are pinned here to what the target bank was
# produced with. The scorer asserts them again from each plane's sidecar
# and refuses on a mismatch; setting them in one place and checking them
# in another is deliberate.
set -euo pipefail

DEPTH="${1:?usage: run_depth.sh <depth>}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONTAINER="${ANE4A_CONTAINER:-$HOME/chris-models/Qwen3.8-27B.vindex3}"
SUBSET="$ROOT/bench/ane4a/subset-v1.json"
LARQL="$ROOT/target/release/larql"
WORK="${ANE4A_WORK:-/tmp/ane4a-work}/d$DEPTH"
OUT="$ROOT/bench/ane4a/depth-$DEPTH.json"

[ -x "$LARQL" ] || { echo "build first: cargo build --release -p larql-cli" >&2; exit 1; }
[ -e "$OUT" ] && { echo "ANE-4A1: $OUT exists — a depth is scored once." >&2; exit 1; }

mkdir -p "$WORK"
echo "== depth $DEPTH: $(python3 -c "import json;print(len(json.load(open('$SUBSET'))['entries']))") prompts =="

python3 - "$SUBSET" <<'PY' > "$WORK/_ids.txt"
import json, sys
for e in json.load(open(sys.argv[1]))["entries"]:
    print(e["id"], ",".join(str(i) for i in e["ids"]))
PY

n=0
while read -r id ids; do
    n=$((n + 1))
    if [ -e "$WORK/$id.f32" ]; then continue; fi
    printf '  [%2d] %-16s ' "$n" "$id"
    LARQL_CPU_MAX_FORMAT=bf16 "$LARQL" vindex3 exec "$CONTAINER" \
        --tokens "$ids" --backend production \
        --logit-dump "$WORK/$id.f32" --draft-depth "$DEPTH" \
        > "$WORK/$id.log" 2>&1 \
        || { echo "FAILED — see $WORK/$id.log"; exit 1; }
    grep -o 'in [0-9.]* s ([0-9]* ms/position)' "$WORK/$id.log" | tail -1
done < "$WORK/_ids.txt"

python3 "$ROOT/bench/ane4a/score_depth.py" "$SUBSET" "$WORK" "$DEPTH" "$OUT"

# Metrics are banked; the planes are not worth 520 MB of a full disk.
rm -f "$WORK"/*.f32
echo "removed raw planes for depth $DEPTH; kept $OUT"
