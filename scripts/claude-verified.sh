#!/usr/bin/env bash
# claude-verified — launch Claude Code with the codex-as-verifier loop wired up.
#
# Sets LARQL_VERIFIER and friends, runs preflight checks (codex on PATH,
# hook present, schema/prompt readable), then exec's `claude`. Any
# arguments after this script's own flags are passed through to claude
# verbatim, so you can do things like:
#
#   scripts/claude-verified.sh --resume
#   scripts/claude-verified.sh --max-loops 1 -- /init
#
# See docs/verifier-agent.md for the full architecture.

set -euo pipefail

# ---- defaults ---------------------------------------------------------------

ENABLE=1
MAX_LOOPS="${LARQL_VERIFIER_MAX_LOOPS:-3}"
TIMEOUT="${LARQL_VERIFIER_TIMEOUT:-120}"
MODEL="${LARQL_VERIFIER_MODEL:-}"
RUN_SMOKE=0
SKIP_PREFLIGHT=0

# Resolve repo root from the script's location so this works no matter where
# you invoke it from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---- usage ------------------------------------------------------------------

usage() {
    cat <<'EOF'
Usage: claude-verified [options] [-- claude-args...]

Launches `claude` with the codex-as-verifier Stop hook enabled.

Options:
  --off, --no-verifier       Disable the verifier loop for this run.
                             (Equivalent to launching claude directly.)
  --max-loops N              Cap corrective cycles per session (default: 3).
  --timeout SEC              Per-call codex timeout (default: 120).
  --model NAME               Override codex model (e.g. gpt-5, o3).
  --smoke                    Run the verifier smoke test and exit.
  --skip-preflight           Skip codex/PATH/login checks.
  -h, --help                 Show this help.

Anything after `--` (or any unrecognised flag, with --) is forwarded to
claude. Examples:

  claude-verified                          # plain start with verifier on
  claude-verified --resume                 # resume your last session
  claude-verified --max-loops 1            # one correction max
  claude-verified --off -- /init           # disable verifier; run /init
  claude-verified --smoke                  # exercise the hook offline

Environment variables (set automatically by this script, but you can
override them):
  LARQL_VERIFIER             "1" enables the loop; unset disables it.
  LARQL_VERIFIER_MAX_LOOPS   Integer cap.
  LARQL_VERIFIER_TIMEOUT     Seconds.
  LARQL_VERIFIER_MODEL       Codex model id.
  CLAUDE_PROJECT_DIR         Repo root; set automatically.
EOF
}

# ---- arg parsing ------------------------------------------------------------

CLAUDE_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --off|--no-verifier)
            ENABLE=0; shift ;;
        --max-loops)
            MAX_LOOPS="$2"; shift 2 ;;
        --max-loops=*)
            MAX_LOOPS="${1#*=}"; shift ;;
        --timeout)
            TIMEOUT="$2"; shift 2 ;;
        --timeout=*)
            TIMEOUT="${1#*=}"; shift ;;
        --model)
            MODEL="$2"; shift 2 ;;
        --model=*)
            MODEL="${1#*=}"; shift ;;
        --smoke)
            RUN_SMOKE=1; shift ;;
        --skip-preflight)
            SKIP_PREFLIGHT=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        --)
            shift; CLAUDE_ARGS+=("$@"); break ;;
        *)
            # Unknown flag — assume it's for claude.
            CLAUDE_ARGS+=("$1"); shift ;;
    esac
done

# ---- smoke mode -------------------------------------------------------------

if (( RUN_SMOKE )); then
    exec bash "$REPO_ROOT/.claude/verifier/.smoketest/run.sh"
fi

# ---- preflight --------------------------------------------------------------

red()    { printf '\033[31m%s\033[0m\n' "$*" >&2; }
green()  { printf '\033[32m%s\033[0m\n' "$*" >&2; }
yellow() { printf '\033[33m%s\033[0m\n' "$*" >&2; }

preflight_fail=0

check() {
    local label="$1"; shift
    if "$@" >/dev/null 2>&1; then
        green   "  ✓ $label"
    else
        red     "  ✗ $label"
        preflight_fail=1
    fi
}

if (( ! SKIP_PREFLIGHT )); then
    echo "preflight:" >&2
    check "claude on PATH"             command -v claude
    if (( ENABLE )); then
        check "codex on PATH"          command -v codex
        check "verify.sh present"      test -x "$REPO_ROOT/.claude/hooks/verify.sh"
        check "verifier prompt"        test -f "$REPO_ROOT/.claude/verifier/prompt.md"
        check "verifier schema"        test -f "$REPO_ROOT/.claude/verifier/schema.json"
        check "settings.json wired"    grep -q "verify.sh" "$REPO_ROOT/.claude/settings.json"
        check "state dir writable"     test -w "$REPO_ROOT/.claude/verifier/state"
    fi
    if (( preflight_fail )); then
        red "preflight failed; not launching claude. Pass --skip-preflight to override." >&2
        exit 1
    fi
fi

# ---- export env -------------------------------------------------------------

export CLAUDE_PROJECT_DIR="$REPO_ROOT"

if (( ENABLE )); then
    export LARQL_VERIFIER=1
    export LARQL_VERIFIER_MAX_LOOPS="$MAX_LOOPS"
    export LARQL_VERIFIER_TIMEOUT="$TIMEOUT"
    if [[ -n "$MODEL" ]]; then
        export LARQL_VERIFIER_MODEL="$MODEL"
    else
        unset LARQL_VERIFIER_MODEL || true
    fi
    yellow "verifier: ON  (max-loops=$MAX_LOOPS, timeout=${TIMEOUT}s, model=${MODEL:-codex-default})"
else
    unset LARQL_VERIFIER || true
    yellow "verifier: OFF (--off / --no-verifier)"
fi

# ---- launch -----------------------------------------------------------------

# Use exec so signals (Ctrl-C, hangup) propagate cleanly to claude.
exec claude "${CLAUDE_ARGS[@]}"
