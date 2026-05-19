# larql vs llama.cpp — CPU decode on Gemma 3 4B Q4_K

Recorded 2026-05-15 on Apple M3 Max, 12 threads, BLAS / Accelerate enabled,
no GPU. Both engines load the same model weights — `output/larql-gemma-3-4b-it.gguf`
quantized to Q4_K_M for llama.cpp, the matching `output/gemma3-4b-q4k-v2.vindex`
for larql.

## Headline

| Metric | larql | llama.cpp | Ratio |
|---|---:|---:|---:|
| Decode (tg16, tok/s)        | 14.5–14.9 (best 14.9) | 41.37 ± 3.1 | **2.78× behind** |
| Decode mean / token (ms)    | 67–70 ms              | ~24 ms      | 2.78× slower |
| Prefill (pp5, tok/s)        | ~2                    | 106.7       | **55× behind** |
| Time to first token (5-tok prompt) | 2.59 s          | 47 ms       | 55× slower |

Decode is within striking distance. **Prefill is the big remaining gap.**

## Commands

llama.cpp (build `6cd0cf72c` / 7060):
```
llama-bench \
    -m /private/tmp/larql-gemma-3-4b-it-Q4_K_M.gguf \
    -dev BLAS -ngl 0 \
    -p 5 -n 16 -r 3 -t 12
```

larql (this branch):
```
target/release/larql bench \
    output/gemma3-4b-q4k-v2.vindex \
    --cpu --tokens 16 --warmup 1 --profile
```

Same `Prompt: "The capital of France is"` (5 BPE tokens after wrapping).

## Per-stage split

### Decode (per-token average, ms)

| Stage | larql | llama.cpp |
|---|---:|---:|
| Forward pass (attn + FFN, 33 layers) | 67.6 | — |
| LM head + sample                     | 8.7  | — |
| **Total**                            | **76.6** | **~24** |

`larql bench --profile` reports the per-stage breakdown; `llama-bench`
does not expose it. The total of ~24 ms is 1000 / 41.37 tok/s.

### Prefill (5-token prompt, total ms)

| Phase | larql | llama.cpp |
|---|---:|---:|
| Embed + 33 layers (per-layer dequant + dense f32 sgemv on seq=5) | 2 593 | — |
| Total                                                            | 2 593 | 47 |

llama.cpp uses batched gemm: weights read once and applied to all 5
prompt positions in a single Accelerate sgemm call. larql currently
takes the legacy `predict_q4k_prefill` path: dequantise each layer's
Q/K/V/O/gate/up/down to f32 once per layer, then do per-position attention
+ FFN over the prompt. The dequant cost (≈ 75 ms × 33 layers ≈ 2.5 s) is
the dominant prefill cost; the actual matmul work is small at seq_len=5.

## Why the decode gap is 3.16×

larql's decode-step breakdown (from `--profile`):

```
CPU fwd   67.6 ms  (88.6%)   ← 33 layers × 6 Q4_K matvec + 1 Q6_K matvec
lm_head    8.7 ms  (11.4%)   ← 262K-vocab Q4_K matvec
─────────
total     76.3 ms
```

Effective Q4_K weight bandwidth: ~2 GB / 68 ms ≈ **30 GB/s**. llama.cpp on
the same machine achieves ~2 GB / 21 ms ≈ **95 GB/s** — about 3× our
effective bandwidth, which matches the decode-rate ratio almost exactly.

Where the difference comes from (educated guesses based on the llama.cpp
source structure):

1. **Hand-written NEON dispatch** — llama.cpp's `ggml-quants.c` Q4_K
   matvec is a single contiguous NEON routine with prefetch hints and
   aggressive interleaving across multiple super-blocks of one row.
   Ours is one super-block at a time with parity-tested helpers.
2. **No per-matvec rayon overhead** — llama.cpp dispatches one
   "compute graph" per token, walking the graph in a single thread pool
   sweep. We do 198 separate `par_iter_mut` launches per decode token
   (33 layers × 6 Q4_K projections); rayon's join overhead per launch is
   small but adds up.
3. **Pre-formatted Q4_K block layout** — llama.cpp keeps Q4_K blocks
   in a layout that pairs lo / hi nibbles for SIMD without per-block
   shuffling. We share the on-disk GGUF Q4_K layout and unpack inside
   the matvec.

None of these are algorithmic — they're constant-factor kernel work.

## What our eight fixes gave us

| Change | Decode (tok/s) | Δ vs prior | Δ vs baseline |
|---|---:|---:|---:|
| Baseline (legacy O(N²) per-step path)  | 0.36  | —      | 1.0× |
| + KV-cached decode                     | ~1.5  | 4.2×   | 4.2× |
| + Direct Q4_K matvec (skip per-step dequant) | 2.6 | 1.7× | 7.2× |
| + Row-parallel f32 lm_head sgemv       | 5.4   | 2.1×   | 15×  |
| + NEON Q4_K / Q6_K / f32_dot           | 9.9   | 1.8×   | 28×  |
| + Q4_K lm_head (synth from f16 embed)  | 12.6  | 1.27×  | 35×  |
| + 4-way acc NEON + fused gate+up / K+V | 13.1  | 1.04×  | 36×  |
| + par_chunks_mut(32) on rayon outer loop | 14.5–14.9 | 1.10× | 40× |

Closing the remaining **2.78× decode gap** is hand-tuned kernel work —
hand-rolled aarch64 asm matching llama.cpp's effective ~95 GB/s read
bandwidth (we're at ~33 GB/s). No more easy algorithmic or scheduling
wins to find on this branch.

The **55× prefill gap** is the bigger target by absolute time. It's
also more tractable: routing prefill through `q4k_matmul` (multi-row
matvec, weights read once and amortised across seq positions) would
shrink prefill from ~2.6 s to a few hundred ms, then NEON would close
the rest. Estimated effort: comparable to the decode-time work in this
branch.

## Verification

- `cargo test -p larql-compute --lib` — 171 unit tests (incl. 6 NEON
  kernel parity tests against scalar reference).
- `cargo test -p larql-inference --lib` — 910 unit tests.
- `cargo test -p larql-inference --release --test test_q4k_cached_parity -- --ignored`
  — end-to-end cached-vs-uncached bit-match plus direct-matvec vs dequant
  first-token agreement.

## Raw JSON

- `gemma3-4b-cpu-final-2026-05-15.json` — larql full bench output.
- `gemma3-4b-cpu-probe-2026-05-15.json` — original baseline + llama.cpp
  reference (commands + results) from before this branch landed.
