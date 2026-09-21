#!/bin/zsh
# MAP-7 T2 instrument gate (prereg §4.1).
#
# Runs once train-general (fit) and dev-general (eval) shards exist —
# the fitter and evaluator both refuse on missing shards or identity
# mismatch, so running early is safe.
#
# Gate arm: global rank-30, delta mode (frozen). A small relative-ridge
# sweep {1e-3, 1e-2, 1e-1} is reported; the gate reads the best top-1.
# PASS iff best dev/general top-1 ∈ [0.78, 0.90]. The result must be
# recorded on the map-7-t3-local registry record; only after that may
# t2-gate-pass.json be created (which unblinds the T3 pair in
# map7_boundary_eval).
set -euo pipefail

CAPTURE="$HOME/chris-models/map7-kshape-capture-v1"
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
export LARQL_M7B_VINDEX="$HOME/chris-models/gemma3-4b-f16.vindex"
export LARQL_M7B_CORPUS="$REPO/bench/map7-kshape/corpus-v1"
export LARQL_M7B_OUT="$CAPTURE/results"
export LARQL_M7B_SPLIT=dev
export LARQL_M7B_STRATUM=general

mkdir -p "$CAPTURE/maps" "$CAPTURE/results"

for ridge in 1e-3 1e-2 1e-1; do
  map="$CAPTURE/maps/t2-global-r30-delta-ridge$ridge.safetensors"
  if [ ! -f "$map" ]; then
    python3 "$REPO/bench/map7-kshape/fit_global_map.py" \
      --capture-dir "$CAPTURE" \
      --pair 4,20 --mode delta --rank 30 --ridge "$ridge" \
      --splits train --strata general --positions all \
      --out "$map"
  fi
  export LARQL_M7B_MAP="$map"
  (cd "$REPO" && cargo run --release -p larql-inference --example map7_boundary_eval)
done

echo ""
echo "== T2 gate summaries =="
for f in "$CAPTURE"/results/eval-global-rank30-delta-ridge*-4-20-dev-general.summary.json; do
  [ -f "$f" ] && python3 -c "
import json,sys
s=json.load(open('$f'))
print(f\"{s['map_manifest']['ridge_relative']}: top1 {s['top1_rate']:.4f} faithful {s['faithful_rate']:.4f} medianKL {s['median_kl_nats']:.4f}\")"
done
echo "Gate band: top-1 in [0.78, 0.90] (prereg §4.1)."
echo "Record the result on map-7-t3-local BEFORE creating t2-gate-pass.json."
