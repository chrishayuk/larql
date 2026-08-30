#!/usr/bin/env bash
# CPU-5 rescue rungs: what is the smallest set of operands that must be
# RESTORED from Q4 to recover acceptable quality?
#
#   ./cpu5_rescue.sh <container> <screen-outdir> <census-dir>
#
# One axis at a time. R0 is the blanket hypothesis; each later rung
# restores exactly ONE matrix class to Q8 — in the SAME integer
# arithmetic, so the only variable is the weight bits.
#
#     R0  all          blanket Q4
#     R1  attn,ffn     head restored
#     R2  ffn,head     attention restored
#     R3  attn,head    FFN restored
#
# The ORDER these are reported in is not the order they should be read
# in: the point is to measure each family's marginal value, and the
# ordering that matters comes out of the numbers, not out of this list.
#
# Each rung produces both halves of the trade:
#   quality  — KL, top-1, flips at margin, over the SCREEN subset
#   cost     — the byte census and what the cost model predicts from it
set -euo pipefail

CONTAINER="${1:?container}"
SCREEN="${2:?screen outdir}"
CENSUS="${3:?census dir}"
HERE="$(cd "$(dirname "$0")" && pwd)"
export LARQL="${LARQL:-./target/release/larql}"
mkdir -p "$CENSUS"

RUNGS=("R0:all" "R1:attn,ffn" "R2:ffn,head" "R3:attn,head")

for spec in "${RUNGS[@]}"; do
  rung="${spec%%:*}"
  classes="${spec#*:}"
  label="${rung}-$(echo "$classes" | tr ',' '+')"
  echo "===== ${rung}  Q4 classes = ${classes} ====="

  export LARQL_CPU_ARITHMETIC=q4xq8b
  export LARQL_CPU_Q4_CLASSES="$classes"

  # COST first and cheaply: two generated tokens is enough to price one
  # steady step, and the census is deterministic so it does not need a
  # quiet machine the way a timing would.
  "$LARQL" vindex3 exec "$CONTAINER" --tokens 760,6511,314,9338,369 \
      --backend production --generate 2 > "$CENSUS/${label}.census" 2>&1 || {
        echo "census failed for $rung"; tail -5 "$CENSUS/${label}.census"; exit 1; }
  grep -E "predicted from bytes|GB at|non-projection floor|residency:" \
      "$CENSUS/${label}.census" || true

  # QUALITY on the screen subset, against the shared bank reference.
  python3 "$HERE/run_bank.py" compare "$CONTAINER" "$SCREEN" \
      --backend production --source auto --label "$label"
  python3 "$HERE/run_bank.py" report "$SCREEN" --label "$label"
done
