#!/bin/zsh
# Ecosystem comparison, gpt-oss-20b, M3 Max. Six arms interleaved, 3 repeats, 60 s cooldowns.
S=${0:a:h}; B=./target/release/larql; M=~/chris-models
P="Write a detailed history of the Roman Empire from its founding to its fall."
G=$(ls $(ls -d ~/.cache/huggingface/hub/models--ggml-org--gpt-oss-20b-GGUF/snapshots/*)/gpt-oss-20b-MXFP4.gguf)
Q8=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/*)
Q4=$(ls -d ~/.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q4/snapshots/*)
{ echo "larql $(git rev-parse HEAD) binary $(stat -f '%Sm' $B)"; llama-bench --version 2>&1 | tail -1; python3 -c "import mlx_lm,mlx.core as mx;print('mlx_lm',mlx_lm.__version__,'mlx',mx.__version__)"; ollama --version; } > $S/meta.txt 2>&1
run() { # name, command...
  local name=$1; shift
  echo "## $name rep $rep $(date +%T) load=$(uptime | sed 's/.*averages: //') power=$(pmset -g batt | head -1 | sed "s/.*'\(.*\)'.*/\1/")" >> $S/runs.txt
  "$@" >> $S/runs.txt 2>&1
  sleep 60
}
ollama_run() {
  curl -s http://localhost:11434/api/generate -d "{\"model\":\"gpt-oss:20b\",\"prompt\":\"$P\",\"stream\":false,\"keep_alive\":0,\"options\":{\"num_predict\":272}}" \
   | python3 -c "import sys,json;j=json.load(sys.stdin);print('OLLAMA eval_count',j['eval_count'],'eval_s',j['eval_duration']/1e9,'tok_s',round(j['eval_count']/(j['eval_duration']/1e9),2),'prompt_count',j.get('prompt_eval_count'))"
}
for rep in 1 2 3; do
  run larql-H $B bench $M/gpt-oss-20b.nvfp4-head.vindex3 --prompt "$P" --backends metal-lowered --warmup 16 --tokens 256
  run llama.cpp llama-bench -m $G -p 75 -n 256 -d 91 -r 1 -fa 1
  run mlx-Q8 python3 -m mlx_lm benchmark --model $Q8 -p 91 -g 256 -n 1
  run ollama ollama_run
  run larql-mxfp4 $B bench $M/gpt-oss-20b.vindex3 --prompt "$P" --backends metal-lowered-mxfp4 --warmup 16 --tokens 256
  run mlx-Q4 python3 -m mlx_lm benchmark --model $Q4 -p 91 -g 256 -n 1
done
echo DONE >> $S/runs.txt
