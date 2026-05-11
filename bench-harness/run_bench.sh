#!/bin/bash
# Three-config head-to-head: llama.cpp all-GPU vs larql all-GPU vs
# larql --ffn=remote. Measures tok/s AND peak VRAM so the FFN-offload
# VRAM-headroom advantage shows up explicitly.
#
# Usage:
#   ./run_bench.sh <vindex-dir> <gguf-file> [prompt-file]
#
# Example:
#   ./run_bench.sh \
#     /home/ianblenke/github.com/ianblenke/larql/output/gemma-3-4b-it-vindex \
#     /home/ianblenke/github.com/ianblenke/larql/output/gguf-cache/Gemma3-4B/gemma-3-4b-it-Q4_K_M.gguf \
#     prompts/json.txt
#
# The script lays down `results/<timestamp>/` with one subdirectory per
# config containing stdout, stderr, and the nvidia-smi VRAM CSV.

set -euo pipefail

VINDEX="${1:?vindex directory required}"
GGUF="${2:?gguf file required}"
PROMPT_FILE="${3:-$(dirname "$0")/prompts/json.txt}"

if [ ! -d "$VINDEX" ]; then
    echo "ERROR: vindex directory not found: $VINDEX" >&2
    exit 1
fi
if [ ! -f "$GGUF" ]; then
    echo "ERROR: GGUF file not found: $GGUF" >&2
    exit 1
fi
if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: prompt file not found: $PROMPT_FILE" >&2
    exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
LARQL_BIN="${LARQL_BIN:-$HERE/../target/release/larql}"
LLAMA_CLI="${LLAMA_CLI:-/home/ianblenke/3rd-party/llama.cpp/build/bin/llama-cli}"
LLAMA_BENCH="${LLAMA_BENCH:-/home/ianblenke/3rd-party/llama.cpp/build/bin/llama-bench}"

for bin in "$LARQL_BIN" "$LLAMA_CLI" "$LLAMA_BENCH"; do
    if [ ! -x "$bin" ]; then
        echo "ERROR: required binary not found or not executable: $bin" >&2
        exit 1
    fi
done

PROMPT="$(cat "$PROMPT_FILE")"
TOKENS="${TOKENS:-128}"
WARMUP="${WARMUP:-4}"

TS="$(date +%Y%m%d-%H%M%S)"
OUT="$HERE/results/$TS"
mkdir -p "$OUT"
echo "Results landing in: $OUT"

start_poller() {
    local out_csv="$1"
    local sentinel="$2"
    "$HERE/nvidia_smi_poller.sh" "$out_csv" "$sentinel" &
    echo $!
}

stop_poller() {
    local pid="$1"
    local sentinel="$2"
    touch "$sentinel"
    wait "$pid" 2>/dev/null || true
}

peak_vram() {
    local csv="$1"
    # Skip header, take max of column 2.
    awk -F, 'NR>1 {if ($2+0 > max) max=$2+0} END {print max}' "$csv"
}

run_config() {
    local name="$1"; shift
    local cmd=("$@")
    local cfg_dir="$OUT/$name"
    mkdir -p "$cfg_dir"
    local sentinel="$cfg_dir/poller.stop"
    rm -f "$sentinel"
    local vram_csv="$cfg_dir/vram.csv"

    echo
    echo "==== $name ===="
    echo "cmd: ${cmd[*]}" | tee "$cfg_dir/cmd.txt"

    local poller_pid
    poller_pid=$(start_poller "$vram_csv" "$sentinel")

    local start_ts end_ts
    start_ts=$(date +%s.%N)
    "${cmd[@]}" >"$cfg_dir/stdout.log" 2>"$cfg_dir/stderr.log" || true
    end_ts=$(date +%s.%N)

    stop_poller "$poller_pid" "$sentinel"

    local wall=$(awk "BEGIN{printf \"%.3f\", $end_ts - $start_ts}")
    local peak=$(peak_vram "$vram_csv")
    echo "wall:    ${wall}s"   | tee -a "$cfg_dir/summary.txt"
    echo "peak_vram_MiB: $peak" | tee -a "$cfg_dir/summary.txt"
    echo "vram_headroom_MiB: $((24576 - peak))" | tee -a "$cfg_dir/summary.txt"
}

# ── Config 1: llama.cpp all-on-GPU ──
run_config "llama-cpp" \
    "$LLAMA_BENCH" -m "$GGUF" -ngl 999 -p 128 -n "$TOKENS" -r 3

# ── Config 2: larql all-on-GPU ──
run_config "larql-cuda" \
    "$LARQL_BIN" bench "$VINDEX" --backends cuda --tokens "$TOKENS" --warmup "$WARMUP" --prompt "$PROMPT"

# ── Config 3: larql --ffn remote (attention on GPU, FFN on CPU/host) ──
# Requires `larql-server` running with the same vindex on a local port.
# Start it in another terminal before invoking:
#   larql server $VINDEX --port 8080
if curl -sf "http://127.0.0.1:8080/health" >/dev/null 2>&1; then
    run_config "larql-ffn-offload" \
        "$LARQL_BIN" bench "$VINDEX" --backends cuda --tokens "$TOKENS" --warmup "$WARMUP" \
        --prompt "$PROMPT" --ffn "http://127.0.0.1:8080"
else
    echo
    echo "==== larql-ffn-offload (SKIPPED) ===="
    echo "Skipping FFN-offload config: no larql-server on port 8080."
    echo "Start one in another terminal: larql server <vindex> --port 8080"
    mkdir -p "$OUT/larql-ffn-offload"
    echo "SKIPPED — no localhost:8080" > "$OUT/larql-ffn-offload/summary.txt"
fi

echo
echo "==== Summary ($OUT) ===="
for cfg in llama-cpp larql-cuda larql-ffn-offload; do
    if [ -f "$OUT/$cfg/summary.txt" ]; then
        echo "--- $cfg ---"
        cat "$OUT/$cfg/summary.txt"
    fi
done
