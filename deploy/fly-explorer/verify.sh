#!/usr/bin/env bash
# Post-deploy gate for the public explorer endpoint.
#
# The capability contract is only worth anything if the deployed server
# keeps it, so this checks the promise against the behaviour rather than
# checking that the endpoint merely answers:
#
#   1. the report is readable    — schema this client knows, profile named
#   2. it promises hf:// planning and refuses local-path planning
#   3. planning a real repo works, is pinned, and is cacheable
#   4. asking again is served from cache
#   5. a local path is refused 403 — the promise in (2), kept
#
# Step 3 reaches Hugging Face and reads ~11 MB of headers. It downloads
# no weights.
#
#   ./deploy/fly-explorer/verify.sh [BASE_URL]

set -euo pipefail

# --revision <sha>: assert the live server IS this commit. Without it
# the gate can only show the server BEHAVES like the build it expects,
# which is circumstantial — a signature, not an identity.
EXPECT_REVISION=""
if [ "${1:-}" = "--revision" ]; then
  EXPECT_REVISION="${2:?--revision needs a commit sha}"
  shift 2
fi

BASE="${1:-https://vindex3-explorer.fly.dev}"
MODEL="${VERIFY_MODEL:-hf://Qwen/Qwen3-0.6B}"
# Never a path that exists: this must be refused by POLICY, before the
# server ever looks at the filesystem.
LOCAL_PROBE="/tmp/not-a-checkpoint-$$"

pass() { printf '  ok    %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1" >&2; exit 1; }

say() { printf '\n%s\n' "$1"; }

# Read one JSON path out of stdin, or "-" when absent or unreadable.
# A server that predates this contract answers 404 with no body, and
# that must read as "does not promise it", not as a crash.
jget() { python3 -c "
import json,sys
try:
    d = json.load(sys.stdin)
except Exception:
    print('-'); sys.exit()
for k in sys.argv[1].split('.'):
    d = d.get(k) if isinstance(d, dict) else None
    if d is None: print('-'); sys.exit()
# JSON spelling, not Python's, so a boolean prints lowercase
# true and never capital-True: the comparisons below are against
# JSON literals and matched neither, which made this gate report a
# capability the server was in fact advertising.
# (No backticks in here - this block is inside a double-quoted
# shell string, where bash would run them as commands.)
print(d if isinstance(d, str) else json.dumps(d))
" "$1"; }

say "== $BASE =="

# ── 1-2. the report ─────────────────────────────────────────────────
CAPS="$(curl -sS --max-time 60 "$BASE/v1/capabilities")"
SCHEMA="$(printf '%s' "$CAPS" | jget schema)"
[ "$SCHEMA" = "-" ] \
  && fail "$BASE/v1/capabilities did not answer with a report — this server predates the capability contract, so the Explorer's PLAN control will stay hidden. Deploy first."
[ "$SCHEMA" = "1" ] \
  || fail "capabilities schema is $SCHEMA; this script reads schema 1 only"
pass "capabilities schema 1"

REVISION="$(printf '%s' "$CAPS" | jget server.revision)"
if [ -n "$EXPECT_REVISION" ]; then
  [ "$REVISION" = "-" ] \
    && fail "the server reports no build revision, so this gate cannot confirm which code is live. Deploy with --build-arg LARQL_SERVER_REVISION=\$(git rev-parse HEAD)."
  [ "$REVISION" = "$EXPECT_REVISION" ] \
    || fail "live server is $REVISION, expected $EXPECT_REVISION — a different build is serving traffic"
  pass "live server IS $EXPECT_REVISION"
elif [ "$REVISION" != "-" ]; then
  pass "build revision $REVISION (pass --revision <sha> to assert it)"
else
  # Not a failure: a build may legitimately not know its commit. But say
  # so, because everything below is then a behavioural signature only.
  pass "build revision not reported — identity unasserted, signature only"
fi

PROFILE="$(printf '%s' "$CAPS" | jget profile)"
[ "$PROFILE" = "public_explorer" ] \
  || fail "profile is '$PROFILE', expected public_explorer"
pass "profile public_explorer"

[ "$(printf '%s' "$CAPS" | jget sources.plan.hf)" = "true" ] \
  || fail "sources.plan.hf is not true — the Explorer's PLAN control will stay hidden"
pass "promises: sources.plan.hf = true"

[ "$(printf '%s' "$CAPS" | jget sources.plan.local)" = "false" ] \
  || fail "sources.plan.local is true on a PUBLIC server — a local path would be a filesystem probe"
pass "promises: sources.plan.local = false"

for forbidden in runtime.execute runtime.lifecycle sources.encode.hf; do
  [ "$(printf '%s' "$CAPS" | jget "$forbidden")" = "false" ] \
    || fail "$forbidden is true on the public surface"
done
pass "public surface executes, binds and encodes nothing"

# ── 3. a real verdict ───────────────────────────────────────────────
say "planning $MODEL (headers only; no weights are downloaded)"
PLAN="$(curl -sS --max-time 180 -X POST "$BASE/v1/plan" \
  -H 'content-type: application/json' \
  -d "{\"sources\":[\"$MODEL\"]}")"

[ "$(printf '%s' "$PLAN" | jget schema)" = "4" ] \
  || fail "plan schema is not 4: $(printf '%s' "$PLAN" | head -c 200)"
pass "plan schema 4"

REV="$(printf '%s' "$PLAN" | jget artifacts)"
printf '%s' "$PLAN" | python3 -c "
import json,sys
d=json.load(sys.stdin)
a=d['artifacts'][0]
assert a['source'].get('revision'), 'verdict is not pinned to a commit'
print('  ok    pinned at', a['source']['revision'][:12] + '...', '(' + a['name'] + ')')
print('  ok    judged by', d['planner']['package'], d['planner']['package_version'],
      '· semantics', d['planner']['semantics_version'])
s=d['staging'][0]
print('  ok    read', s['staged'], 'standing in for', s['stands_in_for'])
" || fail "the verdict did not name its subject"

[ "$(printf '%s' "$PLAN" | jget serving.cacheable)" = "true" ] \
  || fail "a pinned verdict reports cacheable=false"
pass "verdict is cacheable (every artifact pinned)"

# ── 4. the cache ────────────────────────────────────────────────────
AGAIN="$(curl -sS --max-time 180 -X POST "$BASE/v1/plan" \
  -H 'content-type: application/json' -d "{\"sources\":[\"$MODEL\"]}")"
[ "$(printf '%s' "$AGAIN" | jget serving.cached)" = "true" ] \
  || fail "the second ask was not served from cache"
pass "second ask served from the verdict cache"

# ── 5. a hit must not redo the work ─────────────────────────────────
# Latency alone cannot show this: before the cache moved ahead of the
# work, a hit was already fast because hf-hub's disk cache had removed
# the network, while the 39 MB header parse still ran. So measure the
# round trip to a route that does nothing (/v1/health) and subtract it.
# What remains is server-side time.
#
#   pre-fix hit    0.76-1.81 s of work (reparse from local disk)
#   post-fix hit   ~0.25 s             (one ranged GET for the commit)
#
# measured 2026-09-03 on this box. The threshold sits between them with
# room either side; it is a deployment signature, not a perf budget.
MAX_HIT_WORK_S=0.50

floor="$(for _ in 1 2 3; do
  curl -s -o /dev/null -w '%{time_total}\n' --max-time 60 "$BASE/v1/health"
done | sort -n | head -1)"
hit="$(curl -s -o /dev/null --max-time 180 -w '%{time_total}' \
  -X POST "$BASE/v1/plan" -H 'content-type: application/json' \
  -d "{\"sources\":[\"$MODEL\"]}")"

python3 - "$floor" "$hit" "$MAX_HIT_WORK_S" <<'PYEOF' || fail "a cache hit is still doing the work it should have skipped"
import sys
floor, hit, limit = (float(x) for x in sys.argv[1:])
work = hit - floor
print(f"  ok    a hit costs {work:.2f}s of server work "
      f"({hit:.2f}s minus a {floor:.2f}s network floor), under {limit}s"
      if work < limit else
      f"  FAIL  a hit costs {work:.2f}s of server work "
      f"({hit:.2f}s minus a {floor:.2f}s floor) — at or above the {limit}s "
      f"ceiling, which is where a full header reparse lands")
raise SystemExit(0 if work < limit else 1)
PYEOF

# ── 6. the promise, kept ────────────────────────────────────────────
CODE="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 60 \
  -X POST "$BASE/v1/plan" -H 'content-type: application/json' \
  -d "{\"sources\":[\"$LOCAL_PROBE\"]}")"
[ "$CODE" = "403" ] \
  || fail "a local path answered $CODE, expected 403 — the report promised it would be refused"
pass "local-path planning refused 403, as advertised"

say "PASS — the deployed server keeps what its capability report promises."
