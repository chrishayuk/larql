#!/usr/bin/env bash
# ANE-3 — the concurrency rung. Three conditions, one coordinator.
#
#   G   GPU alone
#   A   ANE alone
#   GA  both, released from a common barrier
#
# Same 5120 -> 17408 f16 projection as ANE-0b and ANE-1. Battery regime by
# explicit override, matching the frozen ANE-0b baseline — a concurrency
# result must be scored against a control measured in the same regime.
#
#   ANE3_VENV=... ANE3_ALLOW_BATTERY=1 ./bench/ane/run_ane3.sh s1-battery
set -euo pipefail

SESSION="${1:?usage: run_ane3.sh <session-label>}"
DURATION_MS="${ANE3_DURATION_MS:-4000}"
RAMP_S="${ANE3_RAMP_S:-1.5}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="$ROOT/bench/ane"
VENV="${ANE3_VENV:?set ANE3_VENV to the scoped coremltools venv}"
PY="$VENV/bin/python"
GPU_BIN="$ROOT/target/release/examples/ane3_gpu_worker"
RUN_ROOT="${ANE3_RUN_ROOT:-/tmp/ane3-run}/$SESSION"
JSON="$OUT_DIR/ane3-$SESSION.json"
PROV="$OUT_DIR/ane3-$SESSION.provenance.txt"

POWER_REGIME="AC"
if ! pmset -g ps | head -1 | grep -q 'AC Power'; then
    if [ "${ANE3_ALLOW_BATTERY:-0}" != "1" ]; then
        echo "ANE-3: refusing — not on AC power." >&2
        echo "  Set ANE3_ALLOW_BATTERY=1 and use a label containing 'battery'." >&2
        exit 1
    fi
    case "$SESSION" in
        *battery*) ;;
        *) echo "ANE-3: refusing — battery override set but label lacks 'battery'." >&2; exit 1 ;;
    esac
    POWER_REGIME="BATTERY (override; matches ANE-0b's regime)"
fi

[ -x "$GPU_BIN" ] || { echo "ANE-3: build the gpu worker first:" >&2
    echo "  cargo build --release -p larql-compute-metal --example ane3_gpu_worker" >&2; exit 1; }
[ -e "$JSON" ] && { echo "ANE-3: $JSON exists; pick a new label." >&2; exit 1; }

rm -rf "$RUN_ROOT"
mkdir -p "$RUN_ROOT"/{G,A,GA}

{
    echo "session:      $SESSION"
    echo "power regime: $POWER_REGIME"
    echo "duration_ms:  $DURATION_MS per condition"
    echo "order:        ${ANE3_ORDER:-forward}"
    echo "ramp_s:       $RAMP_S both engines, every condition"
    echo "recorded:     $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "git sha:      $(git -C "$ROOT" rev-parse HEAD)"
    echo "host:         $(hostname)"
    echo "os:           $(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
    echo "coremltools:  $("$PY" -c 'import coremltools; print(coremltools.__version__)')"
    echo
    echo "--- power ---"
    pmset -g ps | head -2 || true
    echo
    echo "--- thermal (before) ---"
    pmset -g therm
    echo
    echo "--- busiest processes (before) ---"
    ps -Ao pid,pcpu,comm | sort -k2 -rn | head -10 || true
} > "$PROV"

# Wait for a readiness file, then release. Kills by CAPTURED PID only —
# never a pattern; a pattern has already cost this project a corrupted run.
release_when_ready() {
    local dir="$1"; shift
    local files=("$@")
    local waited=0
    while :; do
        local all=1
        for f in "${files[@]}"; do
            [ -e "$dir/$f" ] || all=0
        done
        [ "$all" = 1 ] && break
        sleep 0.05
        waited=$((waited + 1))
        if [ "$waited" -gt 6000 ]; then
            echo "ANE-3: workers never became ready" >&2
            exit 1
        fi
    done
    # Both engines are now ramping. Hold them there before releasing, so
    # every condition starts from the same clocks.
    sleep "$RAMP_S"
    touch "$dir/go"
}

# ANE-3b: every condition launches BOTH engines and holds them at full
# load for RAMP_MS before the barrier opens, so all three conditions are
# entered from the same power state. The engine that is not part of a
# condition runs with role `ramp` and exits at `go`.
#
# Without this, ANE-3's GPU-alone arm read 233.5 GB/s cold and 257.5 GB/s
# after sustained load — a 10% drift, the same size as the contention
# being measured, which flipped the verdict between condition orders.
condition() {
    local name="$1" gpu_role="$2" ane_role="$3"
    echo "== condition $name: gpu=$gpu_role ane=$ane_role =="
    local dir="$RUN_ROOT/$name"
    "$GPU_BIN" "$dir" "$DURATION_MS" "$gpu_role" &
    local gpid=$!
    "$PY" "$OUT_DIR/ane3_ane_worker.py" "$dir" "$DURATION_MS" "$ane_role" &
    local apid=$!
    release_when_ready "$dir" gpu.ready ane.ready
    wait "$gpid"
    wait "$apid"
}

cond_G()  { condition G  measure ramp; }
cond_A()  { condition A  ramp measure; }
cond_GA() { condition GA measure measure; }

# Order-reversal is the drift falsifier this project trusts: a clean
# interleave CANNOT fail, and has already produced a false +2.4% here,
# whereas running the conditions in both orders can. If an isolated arm is
# slow only because it ran while the SoC was idle, the two orders disagree.
case "${ANE3_ORDER:-forward}" in
    forward) cond_G; cond_A; cond_GA ;;
    reverse) cond_GA; cond_A; cond_G ;;
    *) echo "ANE-3: ANE3_ORDER must be forward|reverse" >&2; exit 1 ;;
esac

echo
"$PY" "$OUT_DIR/ane3_analyze.py" "$RUN_ROOT" "$JSON"

{
    echo
    echo "--- thermal (after) ---"
    pmset -g therm
} >> "$PROV"

echo "banked: $JSON"
echo "        $PROV"
