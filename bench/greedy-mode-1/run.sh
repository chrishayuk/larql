#!/bin/zsh
# GREEDY-MODE-1: why does gpt-oss-20b greedy decode read ~7.95 or ~8.6 ms/token?
# The verify-n revival table showed two effects:
#   (a) prompts <=159 tokens: ~7.95 ms, encode ~2.5 ms; prompts >=213: ~8.57, encode ~0.63
#   (b) the first six long-prompt runs of the session read ~8.05, every later one ~8.57
# Arms (greedy, fixed 128 tokens, no EOS stop, so the decode spans compare):
#   E1 length:  json_edit suffixes of 96..256 tokens (same content family)
#   E2 content: prose_free (84) vs prose_free repeated to 256
#   E3 cores:   json_256 under `taskpolicy -c utility` (background QoS)
# Arm order rotates every round so a time drift cannot alias onto one arm.
# Same quiet gate as the verify-n timing harness (gate-v2).
S=${0:A:h}
C=~/chris-models/gpt-oss-20b.nvfp4-head.vindex3
B=${B:?set B to the release larql binary}
QUIET=4; CPU_MAX=20; WATCH_CPU_MAX=50; WATCH_HITS=2; QUIET_CHECKS=3
CHECK_S=5; WATCH_S=2; GATE_TIMEOUT=7200; MAX_ATTEMPTS=5
ROUNDS=${ROUNDS:-4}; GEN=${GEN:-128}
EXEMPT='^(WindowServer|launchd|kernel_task|mds_stores|mediaanalysisd)$'
COMPILERS='^(rustc|clippy-driver|cargo|cc|clang|ld|swift-frontend)$'
export HF_HUB_OFFLINE=1
cd $S
load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
power() { pmset -g batt | head -1 | grep -o "'.*'" ; }
therm() { pmset -g therm 2>/dev/null | grep -oE 'CPU_Speed_Limit[^0-9]*[0-9]+' | grep -oE '[0-9]+$'; }
competitor() {
  ps -Ao pid=,pcpu=,comm= | awk -v me=$$ -v arm=${1:-0} -v max=${2:-$CPU_MAX} -v re="$COMPILERS" -v ex="$EXEMPT" '
    { n=split($3,p,"/"); c=p[n] }
    $1==me || $1==arm || c ~ ex { next }
    c ~ re { print $1, c, $2"% (compiler)"; exit }
    $2+0 > max { print $1, c, $2"%"; exit }'
}
quiet_now() { (( $(load1) <= QUIET )) && [ "$(power)" = "'AC Power'" ] && [ -z "$(competitor)" ]; }
gate() {
  local w=0 ok=0
  while (( ok < QUIET_CHECKS )); do
    if quiet_now; then ok=$((ok+1)); else ok=0; fi
    if (( ok < QUIET_CHECKS )); then
      sleep $CHECK_S; w=$((w+CHECK_S)); (( w > GATE_TIMEOUT )) && return 1
    fi
  done
}
log() { echo "$(date +%H:%M:%S) $* | load $(load1) | $(power) | cpu-limit $(therm)" >> conditions.txt; }
run_arm() { # name outfile cmd...
  local name=$1 out=$2; shift 2
  gate || { log "$name gate TIMEOUT"; return 1; }
  log "$name start"
  "$@" > $out 2>&1 &
  local pid=$! hit="" c="" streak=0
  while kill -0 $pid 2>/dev/null; do
    if [ -z "$hit" ]; then
      c=$(competitor $pid $WATCH_CPU_MAX)
      if [[ $c == *compiler* ]]; then hit=$c
      elif [ -n "$c" ]; then streak=$((streak+1)); (( streak >= WATCH_HITS )) && hit="$c sustained"
      else streak=0; fi
      [ -z "$hit" ] && [ "$(power)" != "'AC Power'" ] && hit="power $(power)"
    fi
    sleep $WATCH_S
  done
  wait $pid || { log "$name exit nonzero"; return 1; }
  [ -n "$hit" ] && { log "$name CONTAMINATED by [$hit]"; return 1; }
  return 0
}
field() { grep -oE "$2" $1 | head -1 | grep -oE '[0-9.]+' | tail -1; }
ARMS=(json_96 json_128 json_160 json_192 json_224 json_256 prose_84 prose_256 json_256_util)
cmd_for() { # arm -> command words
  local tok=${1%_util} pre=()
  [[ $1 == *_util ]] && pre=(taskpolicy -c utility)
  print -r -- "${pre[@]} $B vindex3 exec $C --backend metal-lowered --tokens $(cat tok_$tok.txt) --generate $GEN"
}
: > conditions.txt
echo "seq round arm time prompt steady gpu encode p50 first32" > table.txt
log "=== GREEDY-MODE-1 === binary $B $(shasum -a 256 $B | cut -c1-16) commit $(git -C ${B:h:h:h} rev-parse --short HEAD)"
run_arm warm warm.txt ${=$(cmd_for json_256)} || exit 1
seq=0
for round in $(seq 1 $ROUNDS); do
  n=${#ARMS}; off=$(( (round-1) * 2 % n ))
  for i in $(seq 0 $((n-1))); do
    a=${ARMS[$(( (i+off) % n + 1 ))]}
    for attempt in $(seq 1 $MAX_ATTEMPTS); do
      o=r${round}_${a}.txt
      if run_arm r${round}_$a $o ${=$(cmd_for $a)}; then
        seq=$((seq+1))
        echo "$seq $round $a $(date +%H:%M:%S) $(field $o 'prompt: [0-9]+') $(field $o 'steady \(last half\): [0-9.]+') $(field $o 'steady GPU span: [0-9.]+') $(field $o 'encode [0-9.]+') $(field $o 'decode p50: [0-9.]+') $(grep -oE 'first [0-9]+ tokens: [0-9.]+' $o | awk '{print $4}')" >> table.txt
        break
      fi
    done
  done
done
log done
