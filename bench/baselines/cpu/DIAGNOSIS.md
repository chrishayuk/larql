# CPU Bottleneck Diagnosis — Gemma 3 4B Q4_K

Originally recorded 2026-05-15 for ROADMAP C10. Last updated 2026-05-15
after the rayon-chunk restructure landed.

Machine: Apple M3 Max, macOS 24.6.0, 12 threads (rayon default), no GPU.
Reference: llama.cpp build `6cd0cf72c` (7060) on the same hardware.

## Headline

| Engine | Model | Quant | Decode (tg16) | Prefill (pp5) |
|---|---|---|---:|---:|
| larql (this branch) | `output/gemma3-4b-q4k-v2.vindex` | Q4_K | **14.5–14.9 tok/s** | ~2 tok/s |
| llama.cpp | `larql-gemma-3-4b-it-Q4_K_M.gguf` | Q4_K_M | 41.37 tok/s | 105.7 tok/s |
| Ratio | | | **2.79× behind** | 55× behind |

Both engines load the same source weights — the GGUF was produced by
`llama-quantize` from the Gemma 3 4B base; the vindex was extracted
from the same base. Both run on CPU only, no Metal.

## Pre-branch baseline (kept for reference)

Pre-fix decode was **0.36 tok/s**, ~114× behind llama.cpp.

Pre-fix per-token split:

| Stage | Time | Share |
|---|---:|---:|
| CPU full-prefix forward | 2547 ms | 91.6% |
| lm_head + top-k         | 233 ms  | 8.4%  |

The fallback ran `predict_q4k` for every generated token over the full
growing prefix — O(N²) decode, no KV cache reuse, and a per-layer
Q4_K → f32 dequant every step. The dequant alone cost ~75 ms × 33
layers = 2.5 s per generated token; the actual matmul was a small
slice of that.

## Progression on this branch

| Stage of work | Decode | Gap |
|---|---:|---:|
| Baseline | 0.36 tok/s | 114× |
| + KV-cached decode | ~1.5 tok/s | 27× |
| + Direct Q4_K / Q6_K matvec (no per-step dequant) | 2.6 tok/s | 16× |
| + Row-parallel f32 lm_head sgemv | 5.4 tok/s | 7.5× |
| + NEON Q4_K / Q6_K / f32_dot kernels | 9.9 tok/s | 4.1× |
| + Q4_K lm_head (synth from f16 embeddings) | 12.6 tok/s | 3.2× |
| + 4-way accumulator NEON inner | 13.1 tok/s | 3.1× |
| + Fused gate+up / K+V dual-matvec | 13.1 tok/s | 3.1× |
| + `par_chunks_mut(32)` outer rayon | 14.5–14.9 tok/s | 2.79× |
| + Q4_K × Q8_K matvec via NEON sdot | 18.0–19.4 tok/s | 2.36× |
| + Auto-default `--threads 8` on Apple silicon | **24.5 tok/s** | **1.69×** |
| llama.cpp Q4_K_M (reference) | 42.53 tok/s | — |

**~40× over baseline.** Cold-machine numbers — sustained-load reruns
throttle to ~3 tok/s per `feedback_thermal_perf_artifacts.md`, so cold
is the load-bearing reading.

## Current per-step decode breakdown

```
target/release/larql bench output/gemma3-4b-q4k-v2.vindex \
    --cpu --tokens 16 --warmup 1 --profile
```

| Stage | Time (ms) | Share |
|---|---:|---:|
| CPU fwd (33 layers × Q4_K/Q6_K × Q8_K matvec, NEON sdot, t=8) | 35–37 | 85–87% |
| LM head (262K-vocab Q4_K × Q8_K matvec) | 5–6 | 13–15% |
| **Total decode mean** | **40–43** | **100%** |
| Prefill (5-token prompt, still on dequant path) | 2510–2660 | (one-shot) |

CPU fwd is the dominant cost; lm_head is no longer a bottleneck.

## What changed — ten independent fixes

1. **KV-cached decode path**
   (`crates/larql-inference/src/vindex/q4k_forward/cached.rs`):
   `predict_q4k_prefill` + `predict_q4k_decode_step` split. Prefill
   captures per-layer K/V into a `CpuKvCache`; decode runs single-row
   attention against the growing cache. O(N²) → O(N) decode work on
   attention/FFN. Dense architectures only — hybrid-MoE and KV-shared
   architectures still fall through `supports_cached_decode` to the
   legacy loop.

2. **Direct Q4_K / Q6_K matvec, skipping per-step dequant**
   (`predict_q4k_decode_step_direct`):
   Every Q/K/V/O and gate/up/down projection routes through
   `backend.quant_matvec` against the vindex's raw Q4_K/Q6_K bytes.
   No more `insert_q4k_layer_tensors` dequant staging per step. The
   CPU backend's `q4k_matvec` was switched from a slow scalar
   reference (`ops::q4k_matvec::dispatch`) to the sumy-precomputed
   `q4_common::q4k_matvec_into`.

3. **Row-parallel f32 lm_head sgemv**
   (`forward/predict/dense.rs::parallel_lm_head_logits`):
   `dot_proj(h, lm_head)` was falling off ndarray's BLAS fast path
   because `lm_head.t()` is a transposed view (non-standard layout)
   — scalar fallback at 10 GB/s on a 2.7 GB head matrix. Hand-rolled
   row-parallel dot over the row-major buffer with rayon.

4. **NEON Q4_K / Q6_K matvec inner loops**
   (`cpu/ops/q4_common.rs::q4_dual_dot_32_neon`,
    `cpu/ops/q6k_matvec.rs::q6_subblock_dot_16_neon`):
   The 32-element nibble dot and the 16-element 6-bit reconstruction
   were pure scalar; autovec was defeated by the nibble unpacking
   pattern. Hand-written intrinsics: `vandq_u8` / `vshrq_n_u8` for
   nibble extraction, `vmovl_u8` → `vmovl_u16` → `vcvtq_f32_u32` to
   widen, `vfmaq_f32` for FMA. Q6_K uses `vqtbl1q_u8` to broadcast
   hi2 bytes across 16 lanes plus per-lane `vshlq_u8` for the
   0/2/4/6-bit shifts.

5. **NEON f32 dot for lm_head**
   (`forward/predict/dense.rs::f32_dot_neon`):
   4 independent f32x4 accumulators (to hide M3's 4-cycle FMA
   latency), 16 f32 per iteration, scalar tail. Used inside the
   row-parallel lm_head driver.

6. **Q4_K lm_head wiring**
   (`forward/predict/dense.rs::logits_to_predictions_q4_lm_head` +
    `layer_graph/generate/cpu.rs::lm_head_predict`):
   The vindex already synthesises a Q4_K view of the LM head from
   the f16 embeddings at load time (for tied-embedding models like
   Gemma 3 / Llama). Decode-path lm_head now reads that Q4_K view
   via `backend.q4k_matvec` instead of the 2.7 GB f32 staging.

7. **4-way accumulator NEON inner loop**
   (`q4_dual_dot_32_neon`, `q6_subblock_dot_16_neon`):
   Replaced single chained-FMA accumulator with 4 independent
   accumulators per side. M3's FMA is 4-cycle latency, 1/cycle
   throughput — a single-acc chain runs at 25% of peak; 4-acc lets
   the FMAs pipeline at 1/cycle.

8. **`par_chunks_mut(32)` on rayon outer loop**
   (`q4k_matvec_into`, `q4k_dual_matvec_into`, `q6k_matvec::dispatch`):
   Each rayon work unit now covers 32 contiguous rows instead of 1.
   Was 198 × `rows` work-stealing units per decode step; now
   198 × `rows/32`. Same load-balance across the 11 perf cores, ~10×
   less work-stealing overhead.

10. **Auto-default `--threads 8` on Apple silicon**
    (`larql-cli/src/commands/primary/bench/run.rs::configure_rayon_threads`):
    Rayon's default uses all 12 P-cores on M3 Max, but the Q4_K matvec
    saturates LPDDR5 channels at ~8 threads — adding more creates
    DRAM contention without throughput. `larql bench --cpu` now
    configures rayon's global pool to 8 threads on aarch64 macOS;
    `--threads N` and `RAYON_NUM_THREADS` override. 19.4 → 24.5
    tok/s. Documented in `DIAGNOSIS-2026-05-16-thread-scaling.md`.

9. **Q4_K × Q8_K matvec via NEON `sdot`**
   (decode path routes through
   `larql_compute::cpu::ops::q4k_q8k_dot::{q4k_q8k_matvec_into,
   q6k_q8k_matvec_into}` instead of the f32-FMA Q4_K matvec):
   Quantise the activation row to Q8_K (per-256-block int8 + f32
   scale + i16 sub-block sums) once per attn/FFN call, then the inner
   loop does int8 × int8 → i32 SDOT instead of nibble × f32 FMA.
   ARMv8.2-A SDOT does 4 × 4 int8 dot products in one instruction
   versus 4 NEON f32 FMAs per cycle — same approach llama.cpp uses
   for `ggml_vec_dot_q4_K_q8_K`. The compute crate's
   `q4k_q8k_matvec_into` was already row-sequential (single-threaded)
   from prior MoE work; we wrap it with `par_chunks_mut(32)` at the
   call site in `cached.rs` + `dense.rs` to scale across the 11 perf
   cores. ~30% step speedup; 14.9 → 19.4 tok/s.

## Tried and dropped

- **Multi-superblock split-accumulator outer unroll** (2026-05-15):
  Tried 2-superblock interleaving with two parallel scalar accumulators
  across the row's super-block loop. Null result; the helper-function
  factoring made LLVM produce the same schedule it would have anyway.
  Reverted.
- **Fused gate+up dual matvec** (kept in code; bit-exact + one fewer
  Vec alloc, but the predicted x-re-stream saving wasn't real): both
  matvecs operate on a 10 KB `h_in_post_norm` that stays in L1 across
  separate calls. No measurable end-to-end win, but the helper is
  numerically aligned with `q4k_matvec_into` so it can stay.
- **Fused K+V dual matvec** (same story — kept for symmetry +
  one-fewer-rayon-dispatch saving, no measurable speedup).

## Remaining gap to llama.cpp

The 1.69× decode gap is now **purely per-core kernel quality** —
matched algorithm, matched thread count, matched hardware:

- larql per-core throughput (t=1): 5.7 tok/s
- llama.cpp per-core throughput (t=1): 9.88 tok/s
- Per-core ratio: **1.73×** — stable across t=1, t=4, t=8 (see
  `DIAGNOSIS-2026-05-16-thread-scaling.md`).

Both algorithms run the same Q4_K × Q8_K SDOT pattern. What differs:
1. **Inner-loop instruction scheduling**. llama.cpp's
   `ggml_vec_dot_q4_K_q8_K` is hand-asm with explicit interleaving
   across two adjacent super-blocks; LLVM emits a more conservative
   schedule from intrinsics.
2. **Software prefetch**. llama.cpp uses `prfm pldl1keep` ahead of
   each super-block's SDOT chain; we rely on M3's hardware prefetcher.
3. **Possibly different activation Q8_K quantisation**. We round-to-
   nearest; llama.cpp's may use a slightly different scale derivation.

Closing the 1.73× per-core gap is hand-tuned kernel work:

1. **Hand-rolled aarch64 asm** for the Q4_K matvec inner. NEON
   intrinsics via rustc/LLVM don't quite produce llama.cpp's
   instruction interleaving. Comparable effort to the rest of this
   branch combined.
2. **Software prefetch** (`prfm pldl1keep`) ahead of the inner FMA
   loop. Likely 5–15% on top of any kernel restructure.
3. **Lower-precision quant** (Q3_K_M, Q2_K): smaller bytes-per-weight,
   less memory traffic. Needs new vindex extraction.

The **55× prefill gap** is a separate target — `predict_q4k_prefill`
still uses the legacy dequantise-then-sgemv path (~75 ms × 33 layers
of dequant). Routing through a batched CPU `q4k_matmul` (multi-row
matvec sharing weight reads across seq positions) would close most of
it. Caveat: `project_prefill_matmul_falsified` memory documents this
failing for Metal, but the Metal case was different (no dequant cost
to amortise — GPU had Q4_K matvec already; the gap was elsewhere). CPU
prefill is bottlenecked exactly on dequant today, so the matmul
amortisation should genuinely help.

## JSON envelopes

Each snapshot is a real `larql bench --output json` run; the file name
encodes which fixes were in place:

- `gemma3-4b-cpu-probe-2026-05-15.json` — baseline + llama.cpp reference (114× gap).
- `gemma3-4b-cpu-after-cached-direct-2026-05-15.json` — after KV cache + direct matvec + parallel lm_head (5.4 tok/s).
- `gemma3-4b-cpu-after-neon-2026-05-15.json` — after NEON kernels (9.9 tok/s).
- `gemma3-4b-cpu-after-q4-lmhead-2026-05-15.json` — after Q4 lm_head wiring (12.6 tok/s).
- `gemma3-4b-cpu-final-2026-05-15.json` — pre-rayon-chunks (13.1 tok/s).
- `gemma3-4b-cpu-after-rayon-chunks-2026-05-15.json` — pre-Q8K (14.5–14.9 tok/s).
- `gemma3-4b-cpu-after-q8k-sdot-2026-05-16.json` — pre-t=8 (18.0–19.4 tok/s).
- `gemma3-4b-cpu-after-t8-default-2026-05-16.json` — latest, t=8 default (24.5 tok/s).
- `comparison-2026-05-15.json` — side-by-side vs llama.cpp.
- `DIAGNOSIS-2026-05-16-thread-scaling.md` — thread-scaling diagnostic.

## Verification

- **End-to-end parity** (requires a real Q4_K vindex; `#[ignore]`d):
  ```
  cargo test -p larql-inference --release --test test_q4k_cached_parity -- --ignored
  ```
  Two tests: `cached_decode_matches_uncached_tokens` (bit-match
  required — cached prefill+decode against the legacy
  `predict_q4k_hidden` per-step path) and
  `direct_matvec_decode_matches_dequant_path` (first-token agreement
  + ≤ 1 disagreement in first 3 tokens; later drift expected from
  different summation orders, both paths are mathematically correct).
- **NEON kernel parity** (no model needed):
  ```
  cargo test -p larql-compute --lib q4_dual_dot_32   # 3 tests
  cargo test -p larql-compute --lib q4k_dual_matvec  # 3 tests
  cargo test -p larql-compute --lib q6_subblock      # 3 tests
  ```
- **f32_dot NEON parity**:
  ```
  cargo test -p larql-inference --lib dot_tests      # 3 tests
  ```
- **Full unit-test sweep**: 1267 tests across cli/inference/compute,
  all green; `cargo fmt --check` clean.
