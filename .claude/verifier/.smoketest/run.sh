#!/usr/bin/env bash
# Smoke test for .claude/hooks/verify.sh — exercises the four code paths
# (PERFECT, PARTIAL, loop-cap escalation, FAILED-fail-open) with a
# stubbed codex CLI. Run from repo root:
#
#   bash .claude/verifier/.smoketest/run.sh
#
# Cleans up its own state at the end.

set -euo pipefail
ROOT=$(pwd)
SMOKE="$ROOT/.claude/verifier/.smoketest"
STATE="$SMOKE/state"
STUB="$SMOKE/stub-bin"

# Use an isolated state dir so the smoke test never touches real session
# state. The hook writes to .claude/verifier/state by default; we
# temporarily redirect by symlinking a fresh dir into place after the
# real one is moved aside.
REAL_STATE="$ROOT/.claude/verifier/state"
BACKUP_STATE=""
if [[ -e "$REAL_STATE" ]]; then
  BACKUP_STATE=$(mktemp -d -p "$ROOT/.claude/verifier" .state-bak.XXXX)
  mv "$REAL_STATE"/* "$BACKUP_STATE"/ 2>/dev/null || true
fi

restore_state() {
  if [[ -n "$BACKUP_STATE" && -d "$BACKUP_STATE" ]]; then
    mv "$BACKUP_STATE"/* "$REAL_STATE"/ 2>/dev/null || true
    rmdir "$BACKUP_STATE" 2>/dev/null || true
  fi
  rm -f "$REAL_STATE"/smoke-* 2>/dev/null || true
}
trap restore_state EXIT

payload_for() {
  local sid="$1"
  local active="${2:-False}"  # python literal
  python3 -c "
import json
print(json.dumps({
  'session_id': '$sid',
  'transcript_path': '$SMOKE/transcript.jsonl',
  'cwd': '$ROOT',
  'hook_event_name': 'Stop',
  'stop_hook_active': $active,
}))
"
}
PAYLOAD=$(payload_for smoke-default)

run_one() {
  local label="$1"
  local verdict="$2"
  local expect_exit="$3"
  local expect_stdout_contains="$4"
  local payload="${5:-$PAYLOAD}"
  echo "─── $label ───"
  local out
  set +e
  out=$(
    PATH="$STUB:$PATH" \
    LARQL_VERIFIER=1 \
    CLAUDE_PROJECT_DIR="$ROOT" \
    STUB_VERDICT="$verdict" \
    bash "$ROOT/.claude/hooks/verify.sh" <<< "$payload" 2>&1
  )
  local rc=$?
  set -e
  echo "  exit=$rc"
  echo "  stdout/err: ${out:0:200}"
  if [[ "$rc" != "$expect_exit" ]]; then
    echo "  ✗ expected exit=$expect_exit"
    return 1
  fi
  if [[ -n "$expect_stdout_contains" ]] && [[ "$out" != *"$expect_stdout_contains"* ]]; then
    echo "  ✗ stdout missing expected substring '$expect_stdout_contains'"
    return 1
  fi
  echo "  ✓ pass"
  return 0
}

# Test 1: PERFECT → pass-through (exit 0, no block JSON)
run_one "Test 1 PERFECT" \
  '{"grade":"PERFECT","feedback":"","details":"All claims verified.","commands_run":["ls"]}' \
  0 ""

# After a PERFECT, count file should be absent (reset).
if [[ -e "$REAL_STATE/smoke-test 1 perfect.count" ]]; then
  echo "  ✗ count file should have been reset on PERFECT"
fi

# Test 2: PARTIAL → block JSON on stdout
run_one "Test 2 PARTIAL" \
  '{"grade":"PARTIAL","feedback":"openspec/specs/foo missing scenario X.","details":"...","commands_run":[]}' \
  0 '"decision": "block"'

# Test 3: schema invalid → fail open (exit 0, no block)
run_one "Test 3 invalid JSON" \
  'not json at all' \
  0 ""

# Test 4: loop cap. Set MAX=2, fire three PARTIALs. The third should not block.
PAYLOAD3=$(payload_for smoke-loopcap)
echo "─── Test 4 loop cap (MAX=2) ───"
for i in 1 2 3; do
  set +e
  out=$(
    PATH="$STUB:$PATH" \
    LARQL_VERIFIER=1 \
    LARQL_VERIFIER_MAX_LOOPS=2 \
    CLAUDE_PROJECT_DIR="$ROOT" \
    STUB_VERDICT='{"grade":"PARTIAL","feedback":"keep trying","details":"x","commands_run":[]}' \
    bash "$ROOT/.claude/hooks/verify.sh" <<< "$PAYLOAD3" 2>&1
  )
  set -e
  echo "  attempt $i: ${out:0:120}"
  if (( i <= 2 )); then
    if [[ "$out" != *'"decision": "block"'* ]]; then
      echo "  ✗ attempt $i should have blocked"
    else
      echo "  ✓ attempt $i blocked"
    fi
  else
    if [[ "$out" == *'"decision": "block"'* ]]; then
      echo "  ✗ attempt $i should have escalated (no block)"
    else
      echo "  ✓ attempt $i escalated"
    fi
  fi
done

# Test 5: stop_hook_active=true → immediate exit (no codex spawn).
PAYLOAD5=$(payload_for smoke-reentrancy True)
run_one "Test 5 re-entrancy guard" \
  '{"grade":"PARTIAL","feedback":"x","details":"x","commands_run":[]}' \
  0 "" "$PAYLOAD5"

# Confirm re-entrancy didn't even spawn codex (no state file should exist
# for that session id).
if [[ -e "$REAL_STATE/smoke-reentrancy.count" ]]; then
  echo "  ✗ re-entrancy path should not have written a count file"
else
  echo "  ✓ re-entrancy short-circuited before state writes"
fi

# Test 6: opt-out (LARQL_VERIFIER unset) → silent no-op.
echo "─── Test 6 opt-out ───"
set +e
out=$(
  PATH="$STUB:$PATH" \
  CLAUDE_PROJECT_DIR="$ROOT" \
  STUB_VERDICT='{"grade":"PARTIAL","feedback":"x","details":"x","commands_run":[]}' \
  bash "$ROOT/.claude/hooks/verify.sh" <<< "$PAYLOAD" 2>&1
)
rc=$?
set -e
if [[ "$rc" == "0" && -z "$out" ]]; then
  echo "  ✓ opt-out is silent no-op"
else
  echo "  ✗ expected silent exit 0; got rc=$rc out=$out"
fi

echo
echo "Smoke test complete."
echo "State files left for inspection:"
ls -la "$REAL_STATE" 2>/dev/null || true
