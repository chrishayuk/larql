#!/usr/bin/env bash
# verify.sh — Stop hook that runs Codex as an independent verifier of
# Claude's last turn. On PARTIAL/FEEDBACK, blocks the stop and feeds the
# correction back; on PERFECT/VERIFIED, passes through; on FAILED or after
# MAX_LOOPS, escalates to the human (passes through with a warning).
#
# Inspired by https://github.com/disler/the-verifier-agent — the Pi
# Builder/Verifier pattern translated to Claude Code's Stop hook +
# `codex exec`.
#
# Opt-in: set LARQL_VERIFIER=1 in your environment. Without it the
# hook is a no-op so cloning the repo doesn't change anyone's flow.
#
# Diagnostics:
#   .claude/verifier/state/<session>.{count,last,log}
#       count : per-session loop counter (increments on each invocation)
#       last  : last verdict JSON
#       log   : append-only NDJSON of every (timestamp, grade, feedback)
#
# Hook contract (stdin):
#   { "session_id", "transcript_path", "cwd", "hook_event_name",
#     "stop_hook_active", ... }
#
# Hook contract (stdout) — to influence Claude:
#   {"decision": "block", "reason": "<text>"}     → continues with reason
#   (empty stdout, exit 0)                        → stop normally

set -euo pipefail

# ---------- guard rails ------------------------------------------------------

if [[ "${LARQL_VERIFIER:-0}" != "1" ]]; then
    # No-op when not opted in. Pass through silently.
    exit 0
fi

if ! command -v codex >/dev/null 2>&1; then
    # Verifier unavailable — fail open, never block the user.
    >&2 echo "[verifier] codex CLI not on PATH; skipping verification."
    exit 0
fi

PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
VERIFIER_DIR="$PROJECT_DIR/.claude/verifier"
STATE_DIR="$VERIFIER_DIR/state"
mkdir -p "$STATE_DIR"

# ---------- read hook payload ------------------------------------------------

PAYLOAD=$(cat)

# Tiny portable JSON helper. We don't depend on jq existing on every dev box.
# All inputs we consume are produced by Claude Code, so they are valid JSON
# with ASCII-safe keys we control.
json_get() {
    # $1 = field name, reads from $PAYLOAD by default (or $2 if provided)
    local field="$1"
    local src="${2:-$PAYLOAD}"
    python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
v = data.get('$field')
sys.stdout.write('' if v is None else (v if isinstance(v, str) else json.dumps(v)))
" <<< "$src"
}

SESSION_ID=$(json_get session_id)
TRANSCRIPT_PATH=$(json_get transcript_path)
STOP_HOOK_ACTIVE=$(json_get stop_hook_active)

# Re-entrancy guard: if Claude is *already* in a Stop-hook loop (because we
# previously blocked), we let it stop normally to avoid runaway cycles.
# Per Claude Code's docs, stop_hook_active becomes true on the second pass.
if [[ "$STOP_HOOK_ACTIVE" == "true" ]]; then
    exit 0
fi

if [[ -z "$SESSION_ID" || -z "$TRANSCRIPT_PATH" || ! -f "$TRANSCRIPT_PATH" ]]; then
    >&2 echo "[verifier] missing session_id or transcript_path; skipping."
    exit 0
fi

# ---------- loop counter -----------------------------------------------------

MAX_LOOPS=${LARQL_VERIFIER_MAX_LOOPS:-3}
COUNT_FILE="$STATE_DIR/${SESSION_ID}.count"
LOG_FILE="$STATE_DIR/${SESSION_ID}.log"
LAST_FILE="$STATE_DIR/${SESSION_ID}.last"

count=0
[[ -f "$COUNT_FILE" ]] && count=$(cat "$COUNT_FILE")
count=$((count + 1))
echo "$count" > "$COUNT_FILE"

if (( count > MAX_LOOPS )); then
    # Escalate without blocking. The human can read the log to see what
    # the verifier was complaining about across attempts.
    >&2 echo "[verifier] $count corrections attempted; escalating to human (no block)."
    exit 0
fi

# ---------- extract last assistant turn from transcript ---------------------

# Transcript is NDJSON; pull the last assistant message (text content only).
LAST_ASSISTANT=$(python3 - "$TRANSCRIPT_PATH" <<'PY'
import json, sys
path = sys.argv[1]
last = None
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        # Claude Code's transcript stores assistant messages with
        # type=="assistant" and a `message` object whose `content` is a list
        # of blocks. We pick the last one and concatenate text blocks.
        if ev.get("type") == "assistant":
            last = ev
out = []
if last is not None:
    msg = last.get("message", {})
    content = msg.get("content", [])
    if isinstance(content, str):
        out.append(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    out.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    out.append(
                        f"[tool_use {block.get('name','?')} "
                        f"input={json.dumps(block.get('input', {}))[:400]}]"
                    )
print("\n\n".join(out)[:8000])
PY
)

if [[ -z "$LAST_ASSISTANT" ]]; then
    >&2 echo "[verifier] no assistant message found in transcript; skipping."
    exit 0
fi

# ---------- compose verifier prompt ------------------------------------------

PROMPT_FILE="$VERIFIER_DIR/prompt.md"
SCHEMA_FILE="$VERIFIER_DIR/schema.json"

if [[ ! -f "$PROMPT_FILE" || ! -f "$SCHEMA_FILE" ]]; then
    >&2 echo "[verifier] missing prompt.md or schema.json; skipping."
    exit 0
fi

OUT_FILE=$(mktemp -t verifier-out.XXXXXX.txt)
RUN_LOG=$(mktemp -t verifier-run.XXXXXX.log)
trap 'rm -f "$OUT_FILE" "$RUN_LOG"' EXIT

PROMPT=$(cat <<EOF
$(cat "$PROMPT_FILE")

---

# Builder's last turn (verbatim, possibly truncated)

$LAST_ASSISTANT

---

# Working directory

$PROJECT_DIR

---

# Loop attempt

This is verifier attempt $count of $MAX_LOOPS for builder session
$SESSION_ID. If the same issue keeps coming back, weight your feedback
toward "stop trying to fix this; explain what's actually wrong."

EOF
)

# ---------- run codex --------------------------------------------------------

# Read-only sandbox, ephemeral session, schema-validated output.
# Timeout keeps a runaway verifier from blocking the user indefinitely.
TIMEOUT=${LARQL_VERIFIER_TIMEOUT:-120}
CODEX_MODEL=${LARQL_VERIFIER_MODEL:-}
MODEL_ARG=()
[[ -n "$CODEX_MODEL" ]] && MODEL_ARG=(-m "$CODEX_MODEL")

set +e
timeout "$TIMEOUT" codex exec \
    --sandbox read-only \
    --ephemeral \
    --skip-git-repo-check \
    --color never \
    --output-schema "$SCHEMA_FILE" \
    --output-last-message "$OUT_FILE" \
    -C "$PROJECT_DIR" \
    "${MODEL_ARG[@]}" \
    "$PROMPT" \
    > "$RUN_LOG" 2>&1
codex_rc=$?
set -e

if (( codex_rc != 0 )) || [[ ! -s "$OUT_FILE" ]]; then
    >&2 echo "[verifier] codex exit=$codex_rc; failing open."
    >&2 sed -n '1,40p' "$RUN_LOG"
    exit 0
fi

VERDICT=$(cat "$OUT_FILE")
echo "$VERDICT" > "$LAST_FILE"

# Validate that the verdict parses as JSON with a known grade. If not, fail
# open — the verifier should never silently block on its own bug.
GRADE=$(python3 -c "
import json, sys
try:
    v = json.loads(sys.stdin.read())
    g = v.get('grade', '')
    if g in ('PERFECT','VERIFIED','PARTIAL','FEEDBACK','FAILED'):
        print(g)
    else:
        print('UNKNOWN')
except Exception:
    print('UNKNOWN')
" <<< "$VERDICT")

FEEDBACK=$(python3 -c "
import json, sys
try:
    v = json.loads(sys.stdin.read())
    print(v.get('feedback','').strip())
except Exception:
    pass
" <<< "$VERDICT")

# ---------- append log -------------------------------------------------------

python3 - "$LOG_FILE" "$count" "$GRADE" "$VERDICT" <<'PY'
import json, os, sys, time
log, count, grade, verdict = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
entry = {
    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "attempt": int(count),
    "grade": grade,
    "verdict": verdict[:6000],
}
with open(log, "a") as f:
    f.write(json.dumps(entry) + "\n")
PY

# ---------- act on verdict ---------------------------------------------------

case "$GRADE" in
    PERFECT|VERIFIED)
        # Reset counter on a clean pass so future turns start fresh.
        rm -f "$COUNT_FILE"
        exit 0
        ;;
    PARTIAL|FEEDBACK)
        if [[ -z "$FEEDBACK" ]]; then
            FEEDBACK="(verifier returned no feedback text; check $LAST_FILE)"
        fi
        # Block the stop and feed the correction back to Claude as a new
        # user-side prompt. Claude Code reads JSON on stdout: a
        # `decision:"block"` with a `reason` re-runs the model with the
        # reason injected as a user message.
        FEEDBACK="$FEEDBACK" python3 -c '
import json, os
fb = os.environ.get("FEEDBACK", "")
print(json.dumps({
    "decision": "block",
    "reason": "[verifier:codex] " + fb,
}))
'
        exit 0
        ;;
    FAILED|UNKNOWN|*)
        >&2 echo "[verifier] grade=$GRADE; escalating to human (no block)."
        rm -f "$COUNT_FILE"
        exit 0
        ;;
esac
