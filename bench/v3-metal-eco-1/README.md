# V3-METAL-ECO-1: gpt-oss-20b decode, LARQL V3 against the Apple-silicon ecosystem

Measured 2026-09-23, on one machine in one session.

**Fidelity is unresolved for every row.** These are speeds of different representations of the same model.
They say nothing about which representation is acceptable. LARQL's own head-fidelity question (HEAD-FID-1)
was UNINFORMATIVE by its frozen rule. MEASURE-PLAN-1 (`docs/measure-plan-1.md`) is the procedure that will
decide it, and it cannot score the external runtimes without an external arm, which is outside its frozen scope.

## Result: single-stream decode, 256 tokens

| runtime / arm | tok/s (3 repeats) | mean | ms/token |
|---|---|---:|---:|
| MLX `mlx-community/gpt-oss-20b-MXFP4-Q4` | 122.39, 122.17, 122.21 | **122.3** | 8.18 |
| **LARQL V3 H**: `metal-lowered` on the NVFP4 pack with a compiled NVFP4 head | 116.2, 116.1, 115.9 | **116.1** | 8.62 |
| LARQL V3 `metal-lowered-mxfp4` on the source container | 106.2, 111.7, 113.3 | 110.4 | 9.06 |
| MLX `mlx-community/gpt-oss-20b-MXFP4-Q8` | 101.48, 102.08, 102.19 | 101.9 | 9.81 |
| llama.cpp `ggml-org/gpt-oss-20b-GGUF` MXFP4 | 96.03, 98.60, 96.07 | 96.9 | 10.32 |
| Ollama `gpt-oss:20b` | 77.65, 77.07, 77.18 | 77.3 | 12.94 |

Each LARQL arm produced one generation fingerprint across its three repeats. The `mxfp4` arm drifted upward
across repeats (106 → 113) while the battery was charging from 92%; it is the one noisy row.

### The rows are two precision tiers, not one ranking

The speed tiers follow the output head's precision:
- **4-bit head** (MLX-Q4, H, and `mxfp4`, which quantises the head at load): 110–122 tok/s.
- **Higher-precision head:** MLX-Q8 at 101.9. LARQL's conservative pack C (attention NVFP4, head at source) was
  96.4 tok/s in the separate V3 baseline below.

`metal-lowered-mxfp4` is not a conservative arm: it quantises every class to MXFP4 at load.

## Protocol

- **Machine:** Apple M3 Max, AC power throughout, battery charging from 92%. Load averages 1.4–2.4 at run starts.
- **Order:** six arms interleaved in each repeat (H, llama.cpp, MLX-Q8, Ollama, `mxfp4`, MLX-Q4), three repeats,
  60 s cooldown after every run. `raw/ecosystem-gptoss-run.sh` is the exact script.
- **Window:** every tool was aimed at 256 decode tokens starting at context depth ~91.

| tool | invocation | what its number is |
|---|---|---|
| LARQL | `larql bench --prompt "Write a detailed history of the Roman Empire from its founding to its fall." --warmup 16 --tokens 256` | mean over 256 decode steps after 16 warm-up steps, 75-token templated prompt (positions 91–347) |
| llama.cpp | `llama-bench -p 75 -n 256 -d 91 -r 1 -fa 1` | `tg256 @ d91`: 256 generated tokens starting at depth 91, flash attention on. The same window as LARQL. |
| MLX | `python3 -m mlx_lm benchmark -p 91 -g 256 -n 1` | `generation_tps` over 256 tokens after a 91-token prompt, synthetic prompt ids |
| Ollama | `/api/generate`, the same prompt text, `num_predict 272`, `keep_alive 0` | `eval_count / eval_duration` over all 272 tokens, including the 16 LARQL discards, with Ollama's own chat template (82 prompt tokens) |

## Identities

| item | identity |
|---|---|
| LARQL binary for this run | built from `b77ec0e1` (the MEASURE-PLAN-1 token-bank branch, which touches no decode path); lowered Metal path as on main `5b82e45c` |
| H container | `gpt-oss-20b.nvfp4-head.vindex3`, model `6cee5e81ee83917806bbde320786a8fb61efebee`; `output_head@NVFP4` payload sha256 `a327f987e2bb8ecf…`, `decoder_stack@NVFP4` `ac28010b97fba772…` |
| source container | `gpt-oss-20b.vindex3`, same model; `expert_bank@BF16+MXFP4` `6233d166c8d4a563…`, `output_head@BF16` `b71791ff44723099…` |
| llama.cpp | Homebrew build 9430 (`d48a56eff`); GGUF blob sha256 `27cd6c432c7672cb812a92f611cf3ba7bbc35928262bb1e1253ff4ee6ae35901`, repo revision `ef9b12f2` |
| MLX | mlx_lm 0.29.1, mlx 0.30.1; Q8 revision `773a7da7`, Q4 revision `f356f274` |
| Ollama | 0.33.3, model id `17052f91a42e` |

The H and conservative (C) containers were verified byte-identical, by per-file sha256, except the added
`output_head@NVFP4` segment and three metadata files.

## Companion V3 baselines, same day and same protocol, LARQL only

`raw/v3-gptoss-baseline-runs.txt`: gpt-oss-20b, 3 repeats, arms interleaved, ms/token.

| arm | runs | mean |
|---|---|---:|
| `metal-lowered-mxfp4` | 8.74, 9.20, 9.39 | 9.11 |
| C: conservative pack | 10.07, 10.36, 10.67 | 10.37 |
| H: compiled head | 8.62, 8.38, 8.54 | 8.51 |
| `metal` interpreter | 93.67, 91.15, 87.79 | 90.87 |

The interpreter's split: host 1.6, queue 57.3, GPU 21.3 and glue 10.6 ms/token, at 265 submissions per token.

`raw/v3-gemma-baseline-runs.txt`: Gemma 3 4B NVFP4.
- **`metal-nvfp4` (interpreter):** 49.65 ms/token, of which host 0.90, queue 25.00, GPU 11.11 and glue 12.63, at
  137 submissions per token.
- **`metal-lowered`:** 13.34 ms/token.
- **Stage profile:** `raw/v3-gemma-lowered-profile.txt`.

## Files

- `raw/ecosystem-gptoss-{runs,meta,run.sh}`: the 18 runs above, verbatim except that the home directory is
  rewritten to `~`. The meta file's llama-bench line is help text, because that build takes no `--version`;
  the version above comes from `llama-cli --version`.
- `raw/v3-gptoss-baseline-*`, `raw/v3-gemma-*`: the companion baselines. Each meta records the checkout at run
  time; the Gemma binary was re-verified against its source by a rebuild that reproduced both fingerprints.
- `raw/head-fid-1-*`: HEAD-FID-1's pre-registration and its UNINFORMATIVE result, kept beside the speed it
  failed to license.
