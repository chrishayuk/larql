# Q4K decode kernel — close the 1.5× gap to llama.cpp

**Status:** Specced 2026-05-16. Not started. CPU-only.
**Driver:** StandardEngine through the dispatch trait now hits 27.6 tok/s on Gemma 3 4B Q4K on M3 Max (8 threads). llama.cpp hits 41.4 tok/s on the same model + hardware. The remaining 1.50× gap is **entirely in the Q4K × Q8K matvec inner loop** — same algorithm, same NEON `vdotq_s32` instructions, but llama.cpp uses hand-written aarch64 inline assembly with explicit instruction interleaving while ours uses Rust intrinsics lowered by LLVM.

## Diagnosis (already recorded)

See `bench/baselines/cpu/COMPARISON.md` (2026-05-15) and `bench/baselines/cpu/DIAGNOSIS-2026-05-16-thread-scaling.md` (2026-05-16).

Per-core thread-scaling table from the diagnosis:

| Threads | larql | llama.cpp | Per-core ratio |
|---:|---:|---:|---:|
| 1 | 5.7 | 9.88 | **1.73×** |
| 4 | 18.4 | 31.86 | 1.73× |
| 8 | 24.6 | 42.13 | 1.71× |

The ratio is **constant across thread counts** — this is a per-core kernel-quality issue, not a scaling problem. Effective Q4K weight bandwidth:

- llama.cpp: ~95 GB/s (≈80% of M3 Max LPDDR5 peak)
- larql today: ~63 GB/s

We use ~66% of llama.cpp's effective bandwidth on the same hardware.

## Root cause

Both engines:
- Use the same Q4K × Q8K dot-product algorithm.
- Use ARM NEON SDOT (`vdotq_s32`) as the primitive multiply-accumulate.
- Read 4-bit weights, dequantise on-the-fly, accumulate into i32 lanes.

The difference is in the inner loop's instruction stream:

| Layer | llama.cpp (`ggml_vec_dot_q4_K_q8_K`) | larql (`q4k_q8k_matvec_neon`) |
|---|---|---|
| Inner kernel | Hand-written inline aarch64 asm | Rust `core::arch::aarch64::vdotq_s32` intrinsics, lowered by LLVM |
| Block interleaving | Two adjacent super-blocks interleaved in the asm to keep both load units busy | One super-block at a time (parity-tested helpers compose) |
| Prefetch | Explicit `prfm pldl1strm` hints ahead of the matvec stream | LLVM emits prefetch heuristically |
| Block layout | Pre-formatted lo/hi nibble pairs ready for `tbl` / `ushr` | GGUF on-disk Q4K layout; unpack inside the matvec each block |

LLVM's instruction scheduling from intrinsic IR is good but not optimal on this kind of byte-unpacking-heavy hot loop. On a wide-issue core like M3 Max where both load ports + 4 SDOT pipes can fire per cycle, **tight scheduling matters**. The intrinsics produce ~33 cycles/super-block on this workload; hand-asm closes it to ~21 cycles/super-block.

## Proposed work

### Phase 1 — Hand-asm Q4K × Q8K matvec (highest leverage)

Replace `crates/larql-compute/src/cpu/ops/q4k_q8k_dot.rs`'s NEON intrinsic path with hand-written `global_asm!` (or `asm!` per-function) aarch64 implementation modeled after llama.cpp's `ggml_vec_dot_q4_K_q8_K`. Two-super-block interleaved, explicit instruction scheduling, prefetch hints on the weight stream.

**Acceptance:**
1. Bit-identical output to today's intrinsic path on the full Gemma 3 4B Q4K test corpus.
2. Single-thread tok/s ≥ 9.5 (closing >95% of the per-core gap to llama.cpp's 9.88).
3. 8-thread tok/s ≥ 38 (closing >90% of the wall-clock gap to llama.cpp's 42).
4. No regression on any other workload that uses `quant_matvec`.
5. Cleanly gated `#[cfg(target_arch = "aarch64")]` with the intrinsics path retained as fallback for other architectures.

**Estimated effort:** 1–2 weeks. Mostly:
- Day 1–2: port llama.cpp's kernel structure into a Rust `global_asm!` block with the same register allocation.
- Day 3–5: parity testing against the intrinsic path on synthetic + real-model corpora. Iterate on scheduling.
- Day 6+: thermal-aware benchmarking, p99 latency check, ensure no scaling regression past 8 threads.

### Phase 2 — Pre-formatted Q4K block layout (secondary)

llama.cpp pairs lo/hi nibbles in the block layout so SIMD doesn't need to shuffle per block. We unpack inside the matvec because we share the on-disk GGUF layout.

Two options:
- **(a)** Change the on-disk vindex Q4K layout to pre-paired. Breaks compatibility with existing `.vindex` files; needs a migration path.
- **(b)** Repack to a paired layout on vindex load (one-time cost per process, kept in RAM/mmap). Adds a few MB of RAM and ~100ms load time but keeps disk compatible.

(b) is the pragmatic choice. Estimated gain: 1.1–1.2× on top of Phase 1.

**Estimated effort:** 3–5 days, mostly the loader + the kernel's tighter scheduling that becomes available without the shuffle.

### Phase 3 — Q6K kernel (Gemma 3 `down` projection)

Gemma 3's `down` projection is Q6K, not Q4K. We have a NEON Q6K matvec but it hasn't received the same SDOT treatment. Same shape as Phase 1 but smaller scope (only one matvec per layer per step).

**Estimated effort:** 2–3 days. Gain: ~1.05× — Q6K is a small fraction of total decode time.

### Phase 4 — Rayon launch overhead (sub-percent, last)

Today the per-token decode does **198 separate `par_iter_mut` launches** (34 layers × 6 Q4K projections). Each launch has rayon's join overhead (~5–10 µs). At 198 launches per token × 7.5 µs = ~1.5 ms of pure rayon overhead per token. That's about 4% of the total.

**Fix:** batch each layer's projections into a single rayon sweep (one `scope` per layer or per attention/FFN block).

**Estimated effort:** 2–3 days. Gain: ~1.04×.

## Non-goals

- **Hand-asm Q4 (non-K) matvec.** Q4_0 is legacy; production models use Q4K. Don't spend effort there until a model needs it.
- **Anything Metal-side.** This is a CPU-track item. Metal Q4K is already fast through `decode_token`.
- **Algorithmic changes.** Q4K dequant → Q8 input → SDOT accumulate is the right algorithm. We're optimizing the constant factor.
- **Switching to a different quant format.** FP4 (per memory `project_exp26_fp4_quantisation_q1.md`) is a separate track; orthogonal to this kernel work.

## Acceptance criteria (final)

Phases 1–4 land in order. After all four:

1. **Single-core ≥ 9.5 tok/s** on Gemma 3 4B Q4K via `larql bench --cpu --engine standard -t 1`.
2. **8-thread ≥ 39 tok/s** (closing 96% of the 41.4 tok/s gap).
3. Effective Q4K weight bandwidth ≥ 90 GB/s.
4. p99 step-time within 1.2× of mean (no scheduling cliffs).
5. Bit-parity tests pass at every phase.

## Architectural notes

This work is **orthogonal to the dispatch-trait redesign**. It lands inside `crates/larql-compute/src/cpu/ops/q4k_q8k_dot.rs` (and friends). The dispatch trait's `coarse_prefill` / `coarse_decode_step` already route through these kernels; no engine code changes when the kernel improves. Same `larql_compute::QuantMatVec::q4k_matvec` entry point.

Phase 2's pre-formatted layout is the only change that potentially touches the vindex layer. If we go with option (b) — repack on load — even that is contained to the loader and doesn't change disk formats or the trait surface.

## What this is NOT

This spec is **not** about getting research engines (MarkovResidual, UnlimitedContext, TurboQuant, Apollo) to production speed. Those have a different problem: their bespoke `prefill_q4k` overrides bypass the dispatch trait's `coarse_*` intents and use slower CPU code paths. That's covered by `kv-dispatch-quantization.md` Phase 2 (engine migration) — separate work item.

After Phases 1–4 here, `StandardEngine` reaches parity with llama.cpp on CPU Q4K decode. Other engines need their own migration to benefit from the same kernels.
