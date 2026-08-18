#!/bin/bash
# A-10 — the cost of abstraction: gpt-oss-20b through the generic VINDEX3
# plan (`vindex3 exec --backend metal-lowered*`) against the mature served
# routed Metal path (`bench --routed-from`), bracketed, per context.
#
#   ./scripts/gptoss-vindex3-cost-ladder.sh bench/prompts/gpt-oss/ctx-1024.txt 300
#
# Per prompt, two blocks, each `served / candidate / served`:
#
#   block F16    candidate = metal-lowered-ffn   (f16 attention + head, native MXFP4 experts;
#                            the arm certified interpreter-≡-Metal at rel_rms ≤ 1e-6)
#   block NVFP4  candidate = metal-lowered       (NVFP4 attention; the representation that
#                            flipped one generated token — priced here, not argued)
#
# The unit of measurement is the bracket, not the arm (kv-ladder-bracket.sh
# has the rationale): a candidate counts only if the two served baselines
# around it agree within BRACKET_TOL_PCT, and every arm must complete its
# full decode budget. Rest before every arm.
#
# What this ladder is NOT: an id-parity gate. The served path decodes over
# the Q4_K spine while the container carries BF16 attention, so the two
# trajectories are "same semantics, different representation" (A-9.3) and
# are expected to agree at the start and drift at low-margin steps. The
# ids fingerprint is printed so a *within-arm* change is visible, and the
# lowered arms print their attention route witness so routing and timing
# stay separate assertions.
#
# Statistic caveat, deliberately visible: `bench` reports the mean over all
# post-warmup decode steps; `vindex3 exec --generate` reports the last-half
# mean and has no warmup flag yet. Both are per-token decode means over a
# steady tail at n=256, but they are not the same estimator — treat a
# single-digit-% delta as "within instrument", not as a ranking, until
# `vindex3 exec` grows `--warmup` and both read the same window.
#
# Shared input, checked (feedback: a parity gate blind to shared input has
# cost a rope bug before): the served arm tokenises the prompt text; the
# lowered arms take ids. The ids come from `larql run --emit-ids` over the
# same text on the same spine, and the prompt-token COUNT of the two is
# compared before any arm runs — a mismatch voids the prompt.

set -u

SPINE="${LARQL_LADDER_MODEL:-/Users/christopherhay/chris-models/gpt-oss-20b-q4k.vindex}"
ROUTED="${LARQL_LADDER_ROUTED:-/Users/christopherhay/chris-models/gpt-oss-20b-experts-mxfp4.v3}"
# The A-9 container: system graph + expert banks, encoded from the HF checkpoint.
CONTAINER="${LARQL_LADDER_CONTAINER:?set LARQL_LADDER_CONTAINER to the gpt-oss VINDEX3 container}"
BIN="${LARQL_LADDER_BIN:-./target/release/larql}"

# Pinned by bench/prompts/README.md; changing either changes the number.
WARMUP=16
TOKENS=256
MIN_STEPS=$((TOKENS - 1))
BRACKET_TOL_PCT=1.0

PROMPT_FILE="${1:?usage: gptoss-vindex3-cost-ladder.sh <prompt-file> [rest-seconds]}"
REST="${2:-300}"

[ -r "$PROMPT_FILE" ] || { echo "no such prompt file: $PROMPT_FILE" >&2; exit 2; }
[ -x "$BIN" ] || { echo "no larql binary at $BIN (cargo build --release)" >&2; exit 2; }
[ -d "$CONTAINER" ] || { echo "no container at $CONTAINER" >&2; exit 2; }
PROMPT="$(cat "$PROMPT_FILE")"

# ── shared input: ids for the lowered arms, count-checked against the served tokeniser ──
IDS_FILE="${PROMPT_FILE%.txt}.ids"
emit=$(LARQL_GPU_ROUTE=1 "$BIN" run "$SPINE" --metal --emit-ids --routed-from "$ROUTED" -n 1 "$PROMPT" 2>&1)
served_count=$(echo "$emit" | sed -n 's/^\[ids\] prompt \([0-9]*\) tokens: .*/\1/p' | head -1)
echo "$emit" | sed -n 's/^\[ids\] prompt [0-9]* tokens: \(\[.*\]\)$/\1/p' | head -1 | tr -d '[] ' > "$IDS_FILE"
ids_count=$(tr ',' '\n' < "$IDS_FILE" | grep -c .)
if [ -z "$served_count" ] || [ "$served_count" != "$ids_count" ]; then
  echo "VOID: served prompt tokens (${served_count:-none}) != ids written (${ids_count}) — shared input not established" >&2
  exit 1
fi

served_arm() {
  LARQL_GPU_ROUTE=1 "$BIN" bench "$SPINE" --routed-from "$ROUTED" \
    --prompt "$PROMPT" --warmup "$WARMUP" -n "$TOKENS" 2>&1 \
    | awk '/^ *larql-metal/{
        note=""; for (i=7; i<=NF; i++) note = note $i " ";
        gsub(/ms$/, "", $3);
        print $3, $6, note
      }'
}

# $1 = metal-lowered-ffn | metal-lowered. Prints "mean steps note".
lowered_arm() {
  out=$("$BIN" vindex3 exec "$CONTAINER" --tokens "$(cat "$IDS_FILE")" \
        --backend "$1" --generate "$TOKENS" 2>&1)
  mean=$(echo "$out" | sed -n 's/^steady (last half): \([0-9.]*\) ms\/token.*/\1/p' | head -1)
  # `generated ids: [a, b, …]` — count = steps completed (first token from prefill).
  steps=$(echo "$out" | sed -n 's/^generated ids: \[\(.*\)\]$/\1/p' | tr ',' '\n' | grep -c .)
  ids=$(echo "$out" | grep '^generated ids' | md5 | cut -c1-8)
  wit=$(echo "$out" | sed -n 's/^attention dispatches: //p' | tr ' ' '_')
  metal=$(echo "$out" | grep -c '\[metal\]')
  [ -n "$mean" ] && echo "$mean $((steps > 0 ? steps - 1 : 0)) ids=$ids wit=$wit metal=$metal"
}

run_block() {
  local label="$1" candidate="$2"
  echo "════ BLOCK $label   served / $candidate / served   ${REST}s rest before every arm"
  echo "     prompt $PROMPT_FILE ($served_count tokens)   warmup $WARMUP   n $TOKENS"
  local MEANS=() VOID=0
  for arm in served candidate served; do
    sleep "$REST"
    if [ "$arm" = served ]; then
      read -r mean steps note <<<"$(served_arm)"
    else
      read -r mean steps note <<<"$(lowered_arm "$candidate")"
    fi
    if [ -z "${mean:-}" ]; then
      echo "  $(printf '%-10s' "$arm") NO ROW — arm failed to produce a reading"
      VOID=1; continue
    fi
    flag=""
    if [ "${steps:-0}" -lt "$MIN_STEPS" ]; then
      flag="  ← TRUNCATED: $steps < $MIN_STEPS steps"; VOID=1
    fi
    printf "  %-10s mean %8s ms   steps %5s   %s%s\n" "$arm" "$mean" "$steps" "${note:-}" "$flag"
    MEANS+=("$mean")
  done
  if [ "${#MEANS[@]}" -ne 3 ]; then echo "  VERDICT: VOID — block did not produce three arms"; return 1; fi
  awk -v o1="${MEANS[0]}" -v c="${MEANS[1]}" -v o2="${MEANS[2]}" \
      -v tol="$BRACKET_TOL_PCT" -v void="$VOID" -v lab="$label" '
  BEGIN {
    spread = (o1 > o2 ? o1 - o2 : o2 - o1) / (o1 < o2 ? o1 : o2) * 100
    printf "  brackets %.2f%% apart (tolerance %.1f%%)\n", spread, tol
    if (void)         { print "  VERDICT: VOID — an arm did not complete its decode budget"; exit 1 }
    if (spread > tol) { print "  VERDICT: VOID — brackets disagree; do NOT average across them"; exit 1 }
    base = (o1 + o2) / 2
    printf "  VERDICT: VALID — %s %.2f vs served %.2f ms/token", lab, c, base
    printf "  (%+.1f%% latency = the cost of abstraction for this arm)\n", (c - base) / base * 100
  }'
}

run_block F16   metal-lowered-ffn
run_block NVFP4 metal-lowered
