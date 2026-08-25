#!/usr/bin/env bash
# CPU-5 activation geometry sweep.
#
#   ./cpu5_act_sweep.sh <container> <screen-outdir> [block ...]
#
# CPU-5 measured a blocked-Q8[64] activation against EXACT weights at
# KL 0.00061 bits/token — 3.8x the entire accepted cost of Q8 weight
# quantisation, and 1.9x over the G1 gate on its own. Every integer arm
# inherits that floor, so no weight format can clear the gate until the
# activation does.
#
# The activation is ONE VECTOR, so its scale granularity is nearly free:
# at in_dim 5120 its scales are 320 B at block 64 and 1.3 KB at block 16,
# against 14.4 GB of weights per token. The only real cost is one extra
# float multiply-add per sub-block, and SDOT is untouched.
#
# Arm is held at bf16xq8b — EXACT weights — so this measures the
# activation and nothing else.
set -euo pipefail

CONTAINER="${1:?container}"
SCREEN="${2:?screen outdir}"
shift 2
BLOCKS=("$@")
[ ${#BLOCKS[@]} -eq 0 ] && BLOCKS=(64 32 16)

HERE="$(cd "$(dirname "$0")" && pwd)"
export LARQL="${LARQL:-./target/release/larql}"
export LARQL_CPU_ARITHMETIC=bf16xq8b

for b in "${BLOCKS[@]}"; do
  echo "===== activation block ${b} (exact bf16 weights) ====="
  export LARQL_CPU_ACT_BLOCK="$b"
  python3 "$HERE/run_bank.py" compare "$CONTAINER" "$SCREEN" \
      --backend production --source auto --label "act${b}"
  python3 "$HERE/run_bank.py" report "$SCREEN" --label "act${b}"
done
