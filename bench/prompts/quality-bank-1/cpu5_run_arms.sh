#!/usr/bin/env bash
# CPU-5 quality gate: every arm of the Q4 x Q8 programme over Q-BANK-1.
#
#   ./cpu5_run_arms.sh <container> <outdir> [arm ...]
#
# The reference (exact bf16 weights, f32 activation) must already be
# banked with `run_bank.py reference` under LARQL_CPU_MAX_FORMAT=bf16.
# Every arm below is compared against THAT bank, so the canonical arm is
# run once and never re-run — which is what makes a six-arm sweep over a
# 51 GB model affordable.
#
# Arms, and what each one isolates:
#   shipped    Q8 weights x f32 activation      what ships today (CPU-3B)
#   bf16xq8    exact weights x per-TENSOR int8  activation alone
#   q8xq8      Q8 x per-tensor int8             CPU-4X's arithmetic
#   q4xq8      Q4 x per-tensor int8             CPU-4Y's arithmetic
#   bf16xq8b   exact weights x per-BLOCK int8   activation alone, blocked
#   q8xq8b     Q8 x per-block int8
#   q4xq8b     Q4 x per-block int8              the target
set -euo pipefail

CONTAINER="${1:?container}"
OUT="${2:?outdir}"
shift 2
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(shipped bf16xq8b q8xq8b q4xq8b bf16xq8 q8xq8 q4xq8)

HERE="$(cd "$(dirname "$0")" && pwd)"
export LARQL="${LARQL:-./target/release/larql}"

for arm in "${ARMS[@]}"; do
  echo "===== arm ${arm} ====="
  # `shipped` is the default policy: no arm variable at all, so it is the
  # untouched production path rather than an arm that happens to agree
  # with it.
  if [ "$arm" = "shipped" ]; then
    unset LARQL_CPU_ARITHMETIC || true
  else
    export LARQL_CPU_ARITHMETIC="$arm"
  fi
  # `--keep` retains each arm's logits: the interaction analysis needs
  # the ERROR VECTORS, and a per-position KL cannot say whether two
  # perturbations reinforce, cancel, or are independent.
  python3 "$HERE/run_bank.py" compare "$CONTAINER" "$OUT" \
      --backend production --source auto --keep --label "$arm"
  python3 "$HERE/run_bank.py" report "$OUT" --label "$arm"
done
