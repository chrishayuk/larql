# CPU Baseline Probes

Short CPU-track probe artifacts for ROADMAP C10.

These files are not full regression gates yet. They capture the exact command
surface and first measured numbers so the CPU baseline work can move from
"not started" to repeatable.

## Current measurement (post-fix)

- LARQL Gemma 3 4B Q4K CPU decode: **24.5 tok/s** (was 0.36; ~68× over baseline).
- llama.cpp Gemma 3 4B Q4_K_M CPU decode: 42.53 tok/s.
- Gap: **1.69× behind** (was 114×).
- Per-step split: 35-37 ms CPU fwd (attention + FFN, Q4_K × Q8_K via NEON sdot, 8 threads)
  + 5-6 ms lm_head (Q4_K × Q8_K matvec on synthesised Q4 lm_head).

Docs:
- `SESSION-2026-05-15.md` — top-level session writeup (start here).
- `DIAGNOSIS.md` — bottleneck history + each fix detailed + remaining work.
- `COMPARISON.md` — apples-to-apples vs llama.cpp on the same hardware.

JSON envelopes (chronological, one per stage of work):
- `gemma3-4b-cpu-probe-2026-05-15.json` (0.36 tok/s, pre-branch baseline + llama.cpp reference).
- `gemma3-4b-cpu-after-cached-direct-2026-05-15.json` (5.4 tok/s, pre-NEON).
- `gemma3-4b-cpu-after-neon-2026-05-15.json` (9.9 tok/s, post-NEON).
- `gemma3-4b-cpu-after-q4-lmhead-2026-05-15.json` (12.6 tok/s, post-Q4-lm-head).
- `gemma3-4b-cpu-final-2026-05-15.json` (13.1 tok/s, pre-rayon-chunks).
- `gemma3-4b-cpu-after-rayon-chunks-2026-05-15.json` (14.5–14.9 tok/s, pre-Q8K).
- `gemma3-4b-cpu-after-q8k-sdot-2026-05-16.json` (18.0–19.4 tok/s, pre-t=8).
- `gemma3-4b-cpu-after-t8-default-2026-05-16.json` (24.5 tok/s, latest).
- `comparison-2026-05-15.json` — comparison envelope.

See `DIAGNOSIS.md` for the six changes that delivered the win
(KV-cache wiring, dequant-free decode, parallel lm_head, NEON Q4_K/Q6_K
matvec, NEON f32_dot, Q4 lm_head) and the remaining work to close the
last 3.2×.

## Current caveats

- `gemma3-4b-cpu-probe-2026-05-15.json` uses a Q4_K_M GGUF quantized locally
  from `output/larql-gemma-3-4b-it.gguf` with `llama-quantize`.
- `llama-bench` must be forced to `-dev BLAS` in this environment; otherwise it
  tries to initialize the Metal backend even with `-ngl 0`.
- The LARQL short runs stop early after 15 decode steps because EOS fires
  on this prompt — the bench reports the measured window only.
- The direct-matvec path is correct but does not bit-match the
  dequant→sgemv path (different summation orders). Parity test
  enforces first-token agreement only.
