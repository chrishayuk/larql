#!/usr/bin/env bash
# ANE-1 — run one session of the placement rung.
#
# Same discipline as ANE-0b: the protocol is enforced here, not
# remembered. Power regime is recorded and must match ANE-0b's, since
# ANE-3 will compare against that baseline.
#
#   ANE1_VENV=/path/to/venv ./bench/ane/run_ane1.sh s1-battery
#
# The venv is scoped and disposable. Recreate with:
#   python3 -m venv <venv> && <venv>/bin/pip install coremltools
set -euo pipefail

SESSION="${1:?usage: run_ane1.sh <session-label>}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT_DIR="$ROOT/bench/ane"
VENV="${ANE1_VENV:?set ANE1_VENV to the scoped coremltools venv}"
PY="$VENV/bin/python"
JSON="$OUT_DIR/ane1-$SESSION.json"
PROV="$OUT_DIR/ane1-$SESSION.provenance.txt"
WORK="${ANE1_WORK_DIR:-/tmp/ane1-work}"

POWER_REGIME="AC"
if ! pmset -g ps | head -1 | grep -q 'AC Power'; then
    if [ "${ANE1_ALLOW_BATTERY:-0}" != "1" ]; then
        echo "ANE-1: refusing — not on AC power." >&2
        echo "  Set ANE1_ALLOW_BATTERY=1 and use a label containing 'battery'." >&2
        exit 1
    fi
    case "$SESSION" in
        *battery*) ;;
        *)
            echo "ANE-1: refusing — battery override set but session label" >&2
            echo "  '$SESSION' does not say 'battery'." >&2
            exit 1
            ;;
    esac
    POWER_REGIME="BATTERY (ANE1_ALLOW_BATTERY=1 override)"
    echo "ANE-1: measuring on battery by explicit override — matches ANE-0b's regime."
fi

if [ ! -x "$PY" ]; then
    echo "ANE-1: no python at $PY" >&2
    exit 1
fi

if [ -e "$JSON" ]; then
    echo "ANE-1: $JSON already exists; pick a new session label." >&2
    exit 1
fi

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
    echo "os:           $(sw_vers -productName) $(sw_vers -productVersion) ($(sw_vers -buildVersion))"
    echo "python:       $("$PY" --version 2>&1)"
    echo "coremltools:  $("$PY" -c 'import coremltools; print(coremltools.__version__)')"
    echo "venv:         $VENV"
    echo
    echo "--- power ---"
    # head closing the pipe raises SIGPIPE in the writer; pipefail + set -e
    # would turn that into a silent death.
    pmset -g ps | head -2 || true
    echo
    echo "--- thermal (before) ---"
    pmset -g therm
    echo
    echo "--- busiest processes (before) ---"
    ps -Ao pid,pcpu,comm | sort -k2 -rn | head -10 || true
} > "$PROV"

ANE1_WORK_DIR="$WORK" "$PY" "$OUT_DIR/ane1_placement.py" "$SESSION" "$JSON"

{
    echo
    echo "--- thermal (after) ---"
    pmset -g therm
} >> "$PROV"

echo "banked: $JSON"
echo "        $PROV"
