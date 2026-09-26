#!/bin/zsh
# MAP-7 driver: build -> capture (dev-general, then train-general) -> T2 gate.
#
# Scoped deliberately to the general stratum only. The T2 instrument gate
# (prereg §4.1) fits a global rank-30 map on train/general and evaluates on
# dev/general; the other four strata are not needed until T3 unblinds, and
# capturing them now would cost ~40GB for nothing.
#
# Resumable: the capture harness is append-only per shard and skips any shard
# file that already exists, so re-running after an interruption continues.
#
# The gate result must be recorded on the map-7-t3-local registry record
# BEFORE t2-gate-pass.json is created. This script deliberately does NOT
# create that file.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
CAPTURE="$HOME/chris-models/map7-kshape-capture-v1"
LOG="$REPO/bench/map7-kshape/map7-t2-run.log"

export LARQL_M7K_VINDEX="$HOME/chris-models/gemma3-4b-f16.vindex"
export LARQL_M7K_CORPUS="$REPO/bench/map7-kshape/corpus-v1"
export LARQL_M7K_OUT="$CAPTURE"
export LARQL_M7K_MODE=capture

say() { print -r -- "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== MAP-7 T2 driver starting ==="
say "repo    $REPO"
say "capture $CAPTURE"
say "free    $(df -g "$HOME" | tail -1 | awk '{print $4}') GB (harness floor is 120 GB)"

say "--- step 1/4: build ---"
if ! (cd "$REPO" && cargo build --release -p larql-inference \
        --example map7_kshape_capture --example map7_boundary_eval) >>"$LOG" 2>&1; then
	say "BUILD FAILED — see $LOG"; exit 1
fi
say "build ok"

for filter in dev-general- train-general-; do
	say "--- step 2/4: capture $filter ---"
	if ! (cd "$REPO" && LARQL_M7K_SHARD_FILTER="$filter" \
	        cargo run --release -p larql-inference --example map7_kshape_capture) >>"$LOG" 2>&1; then
		say "CAPTURE FAILED for $filter — see $LOG"; exit 1
	fi
	say "capture $filter ok  ($(ls "$CAPTURE" | grep -c "^${filter}" || echo 0) shard files)"
	say "capture dir now $(du -sh "$CAPTURE" | cut -f1)"
done

say "--- step 3/4: T2 gate (fit rank-30 global map, 3 ridges, evaluate) ---"
if ! "$REPO/bench/map7-kshape/run_t2_gate.sh" >>"$LOG" 2>&1; then
	say "T2 GATE SCRIPT FAILED — see $LOG"; exit 1
fi

say "--- step 4/4: gate summaries ---"
for f in "$CAPTURE"/results/eval-global-rank30-delta-ridge*-4-20-dev-general.summary.json; do
	[ -f "$f" ] || continue
	python3 -c "
import json,sys
s=json.load(open('$f'))
print(f\"  ridge {s['map_manifest']['ridge_relative']}: top1 {s['top1_rate']:.4f}  faithful {s['faithful_rate']:.4f}  medianKL {s['median_kl_nats']:.4f}\")
" | tee -a "$LOG"
done
say "gate band: top-1 in [0.78, 0.90] (prereg 4.1)"
say "RECORD THE RESULT ON map-7-t3-local BEFORE creating t2-gate-pass.json"
say "=== driver complete ==="
