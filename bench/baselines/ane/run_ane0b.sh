#!/usr/bin/env bash
# ANE-0b — run one session of the frozen GPU-alone baseline.
#
# The protocol is enforced here rather than remembered: this script
# refuses to produce a bankable result on battery power, and it records
# the provenance that makes the number auditable later. Two independent
# sessions are required; run it twice, separately.
#
#   ./bench/baselines/ane/run_ane0b.sh s1
#   ./bench/baselines/ane/run_ane0b.sh s2
#
# Once ANE-3 begins, the JSON files this writes are immutable.
set -euo pipefail

SESSION="${1:?usage: run_ane0b.sh <session-label>}"
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT_DIR="$ROOT/bench/baselines/ane"
BIN="$ROOT/target/release/examples/ane0b_gpu_baseline"
JSON="$OUT_DIR/ane0b-$SESSION.json"
PROV="$OUT_DIR/ane0b-$SESSION.provenance.txt"

# --- protocol gates -------------------------------------------------
# Battery changes GPU clocks, so AC is the default requirement. It can be
# overridden deliberately — what must not happen is an unlabelled battery
# result being banked as if it were an AC one. When overridden, the power
# regime is recorded in the provenance and the session label is required
# to carry it, so a reader of the JSON alone cannot miss it.
POWER_REGIME="AC"
if ! pmset -g ps | head -1 | grep -q 'AC Power'; then
    if [ "${ANE0B_ALLOW_BATTERY:-0}" != "1" ]; then
        echo "ANE-0b: refusing — not on AC power." >&2
        echo "  Plug in, or set ANE0B_ALLOW_BATTERY=1 and use a session" >&2
        echo "  label containing 'battery'." >&2
        exit 1
    fi
    case "$SESSION" in
        *battery*) ;;
        *)
            echo "ANE-0b: refusing — battery override set but session label" >&2
            echo "  '$SESSION' does not say 'battery'. The regime must be" >&2
            echo "  visible in the JSON, not only in the provenance file." >&2
            exit 1
            ;;
    esac
    POWER_REGIME="BATTERY (ANE0B_ALLOW_BATTERY=1 override)"
    echo "ANE-0b: WARNING — measuring on battery by explicit override."
    echo "        ANE-3 must be measured in the SAME regime, or this"
    echo "        baseline must be re-taken. Cross-session agreement is"
    echo "        the instrument that says whether the regime was stable."
fi

if [ ! -x "$BIN" ]; then
    echo "ANE-0b: build first:" >&2
    echo "  cargo build --release -p larql-compute-metal --example ane0b_gpu_baseline" >&2
    exit 1
fi

if [ -e "$JSON" ]; then
    echo "ANE-0b: $JSON already exists. A banked session is immutable —" >&2
    echo "  pick a new session label rather than overwriting it." >&2
    exit 1
fi

# --- provenance -----------------------------------------------------
{
    echo "session:      $SESSION"
    echo "power regime: $POWER_REGIME"
    echo "recorded:     $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "git sha:      $(git -C "$ROOT" rev-parse HEAD)"
    echo "git branch:   $(git -C "$ROOT" rev-parse --abbrev-ref HEAD)"
    dirty="$(git -C "$ROOT" status --porcelain | wc -l | tr -d ' ')"
    echo "git dirty:    $dirty file(s)"
    echo "host:         $(hostname)"
    echo "hw model:     $(sysctl -n hw.model)"
    echo "cpu:          $(sysctl -n machdep.cpu.brand_string)"
    echo "memory:       $(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 )) GB"
    echo "os:           $(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
    echo
    echo "--- power ---"
    # `head` closing the pipe early raises SIGPIPE in the writer, which
    # `pipefail` + `set -e` turn into a silent script death. Every
    # head-terminated pipeline here is therefore tolerated explicitly.
    pmset -g ps | head -2 || true
    echo
    echo "--- thermal (before) ---"
    pmset -g therm
    echo
    echo "--- busiest processes (before) ---"
    ps -Ao pid,pcpu,comm | sort -k2 -rn | head -10 || true
} > "$PROV"

# --- run ------------------------------------------------------------
"$BIN" "$SESSION" "$JSON"

{
    echo
    echo "--- thermal (after) ---"
    pmset -g therm
    echo
    echo "--- power (after) ---"
    pmset -g ps | head -2 || true
} >> "$PROV"

echo "banked: $JSON"
echo "        $PROV"
