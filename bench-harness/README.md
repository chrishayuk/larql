# larql vs llama.cpp bench harness

Three-config head-to-head measuring **tok/s** AND **VRAM headroom** so we
can quantify both the speed parity story and the FFN-offload VRAM-savings
story.

## Goal

Single-token-speed parity isn't the whole point. The latent advantage of
larql's `--ffn` flag is that FFN weights live on CPU/system-RAM while
only attention weights and KV cache live on GPU. That lets larql run a
*bigger model* or a *longer context* on the same VRAM budget than
llama.cpp can. For small models (Gemma 3 4B) this is invisible; for
Qwen 3.6 27B / 35B-A3B it's the whole point.

## Three configs

| | llama.cpp all-on-GPU | larql all-on-GPU | larql `--ffn` remote |
|---|---|---|---|
| Where attention runs | GPU | GPU | GPU |
| Where FFN runs | GPU | GPU | CPU (over PCIe, same host) |
| Where KV cache lives | GPU | GPU | GPU |
| Peak VRAM (expect) | high | high | **low** |
| System RAM | low | low | high (FFN weights) |
| PCIe traffic | none | none | hidden×layers×2 directions per token |

## Metrics

- **decode tok/s** — main throughput metric
- **prefill ms** — time-to-first-token
- **peak VRAM (MiB)** — `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 100` polled into a file in the background; take max
- **system RAM delta (MiB)** — `/proc/$PID/status` `VmRSS` before + during + after
- **VRAM headroom (MiB)** — `24576 - peak_VRAM` — directly translates to "how much more context could fit"

## Test prompts

Three workload classes (each prompts a different α regime for the spec
path; we run with spec OFF for the main comparison to isolate raw decode):

1. **Chat-style** (`prompts/chat.txt`) — Q&A with no echoing. Stresses
   novel-token generation. PLD α ≈ 0 on this; the raw-decode story.
2. **JSON-structured** (`prompts/json.txt`) — partial-JSON completion.
   Highly repetitive. PLD α ≈ 0.7-0.9.
3. **RAG-style** (`prompts/rag.txt`) — passage + question + answer-from-
   passage. Moderate repetition.

## Files

- `run_bench.sh` — main entry point. Runs all 3 configs × 3 prompts.
- `nvidia_smi_poller.sh` — backgrounded VRAM poller; emits CSV.
- `parse_results.py` — collects raw outputs into a single summary table.
- `prompts/*.txt` — the three test prompts.
- `results/` — per-run output (CSV from poller + run logs).

## Run

```bash
# One-time: ensure llama.cpp built at /home/ianblenke/3rd-party/llama.cpp/build/
# and the larql binary at target/release/larql.
./run_bench.sh /path/to/qwen-vindex /path/to/qwen.gguf
```
