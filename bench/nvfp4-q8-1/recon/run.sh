#!/bin/zsh
S=${0:a:h}; B=./target/release/larql; M=~/chris-models
T=2,105,2364,107,6974,496,9813,4083,529,506,10995,23436,699,1061,37813,531,1061,3798,236761,106,107,105,4368,107
arm() { # name env container backend
  local name=$1 envs=$2 c=$3 b=$4
  echo "## $name rep $rep $(date +%T) load=$(uptime | sed 's/.*averages: //') power=$(pmset -g batt | head -1 | sed "s/.*'\(.*\)'.*/\1/")" >> $S/runs.txt
  env ${=envs} $B vindex3 exec $M/$c --tokens $T --backend $b --generate 32 >> $S/runs.txt 2>&1
  sleep 60
}
for rep in 1 2 3; do
  arm bf16 LARQL_CPU_MAX_FORMAT=bf16 gemma3-4b-it.vindex3 production
  arm q8xq8 NOTHING=1 gemma3-4b-it.vindex3 production
  arm nvfp4 NOTHING=1 gemma3-4b-it.nvfp4.vindex3 production-nvfp4
  arm q4k-q8k NOTHING=1 gemma3-4b-it.q4k.vindex3 production-q4k-q8k
done
echo DONE >> $S/runs.txt
