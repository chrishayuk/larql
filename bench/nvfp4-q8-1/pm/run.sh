#!/bin/zsh
# NVFP4-Q8-1 P/M workload, as frozen in docs/nvfp4-q8-1.md: the reconnaissance workload, three interleaved repeats of
# the three arms, 60 s cooldowns, AC power, load recorded per run. It waits for a quiet machine (1-min load < QUIET)
# before starting; it never waits between arms, so the interleave is not re-ordered by load.
S=${0:a:h}; B=./target/release/larql; M=~/chris-models; QUIET=3.0
T=2,105,2364,107,6974,496,9813,4083,529,506,10995,23436,699,1061,37813,531,1061,3798,236761,106,107,105,4368,107
load1() { sysctl -n vm.loadavg | awk '{print $2}'; }
until [ $(echo "$(load1) < $QUIET" | bc) = 1 ]; do sleep 60; done
{ echo "binary: source $(git log -1 --format=%H -- crates) (HEAD $(git rev-parse HEAD)) built $(stat -f %Sm $B)"; echo "LARQL_* env vars: $(env | grep -c '^LARQL_')"; echo "start $(date) load $(sysctl -n vm.loadavg)"; } > $S/meta.txt
arm() { # name container backend
  echo "## $1 rep $rep $(date +%T) load=$(sysctl -n vm.loadavg) power=$(pmset -g batt | head -1 | sed "s/.*'\(.*\)'.*/\1/")" >> $S/runs.txt
  $B vindex3 exec $M/$2 --tokens $T --backend $3 --generate 32 >> $S/runs.txt 2>&1
  sleep 60
}
for rep in 1 2 3; do
  arm nvfp4 gemma3-4b-it.nvfp4.vindex3 production-nvfp4
  arm nvfp4-q8 gemma3-4b-it.nvfp4.vindex3 production-nvfp4-q8
  arm q4k-q8k gemma3-4b-it.q4k.vindex3 production-q4k-q8k
done
echo "end $(date) load $(sysctl -n vm.loadavg)" >> $S/meta.txt
echo DONE >> $S/runs.txt
