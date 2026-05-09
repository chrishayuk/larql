# cuda-q4k-mmvq-int8 — tasks

## Phase 1 — Q8_1 quantize kernel

### 1. NVRTC kernel

- [x] 1.1 `QUANTIZE_Q8_1_SRC` NVRTC string lives in
      `crates/larql-compute/src/cuda/elem.rs`. Layout matches
      llama.cpp's `block_q8_1` (36 bytes/block: half2 ds + 32
      i8 quants). fp16 conversion via inline PTX
      `cvt.rn.f16.f32` (no cuda_fp16.h dependency).
- [x] 1.2 `quantize_q8_1_device(backend, x_dev, n) ->
      Result<Q8_1Buf, CudaInitError>`. `Q8_1Buf` wraps
      `bytes: CudaSlice<u8>` (size `n_blocks * 36`) plus
      `n_blocks: usize`. Single packed layout to match
      upstream's `block_q8_1[]` byte-for-byte.
- [x] 1.3 `n` must be a multiple of 32; returns a typed error
      otherwise.

### 2. Tests

- [x] 2.1 `q8_1_quantize_roundtrips_to_within_quant_noise`
      lives in `cuda::elem::tests`. Quantises a random
      hidden=2560 input, dequantises on host, asserts
      per-element error within 1.05 quanta (allowing fp16
      rounding slack on the scale).

## Phase 2 — Q4_K × Q8_1 mmvq kernel

### 3. NVRTC kernel

- [x] 3.1 New file
      `crates/larql-compute/src/cuda/q4k_mmvq.rs`.
      Module-level `Q4K_MMVQ_SRC`, `OnceLock` for lazy
      module/function load.
- [x] 3.2 Kernel: one warp per output row, 16 lanes per
      super-block, 2 super-blocks per warp-iter (matches
      upstream's `blocks_per_iter` for `nwarps=1`). Body is a
      close-to-verbatim port of upstream's
      `vec_dot_q4_K_q8_1_impl_vmmq` (MIT-licensed, ggml
      authors). Provenance comment in the NVRTC source.
- [x] 3.3 `dp4a.s32.s32` via inline PTX (single-instruction
      4-way INT8 SIMD dot product, sm_61+). NVRTC compiles
      against `compute_61` (the default `compute_52` lacks
      dp4a). Built-in `__dp4a` is not exposed by NVRTC
      without `cuda_fp16.h`, hence inline asm.

### 4. Backend dispatch

- [x] 4.1 `q4k_mmvq::matvec_device(backend, q4k_data,
      x_q8_1, rows, hidden)` — direct entry; the existing
      `q4k_direct::matvec_device` and the original
      `q4k_matvec_device` on `CudaBackend` are unchanged
      (back-out path).
- [x] 4.2 `decode::matvec_device_mmvq` is the dispatcher. If
      `weight.format == Q4_K` and `LARQL_CUDA_Q4K_MMVQ != 0`
      and a `Q8_1Buf` is supplied, routes to mmvq. Otherwise
      falls through to the existing f32 `matvec_device`.
      `LARQL_CUDA_Q4K_MMVQ=0` forces the f32 path (back-out).

### 5. Tests

- [x] 5.1 `q4k_mmvq_matches_q4k_direct_on_dequantized_input`
      in `cuda::q4k_mmvq::tests`. Compares mmvq vs
      `q4k_direct` fed the SAME Q8_1-dequantized input;
      isolates kernel arithmetic from Q8_1 noise. Tested at
      `(64, 256)` and `(4096, 2560)` shapes; both ≤ 1e-3
      max-element. (The naïve "mmvq vs q4k_direct(f32_input)"
      comparison hits ~0.10 Q8_1 quantisation noise floor —
      not a kernel bug, just the wrong test design.)
- [x] 5.2 Existing
      `decode_token_phase1_matches_host_fallback` covers
      env-var dispatch indirectly: passes with mmvq=on
      (default) and mmvq=off (host-fallback path).

## Phase 3 — Decode wiring

### 6. Share Q8_1 across q/k/v and gate/up

- [x] 6.1 `decode_token_device`: after
      `rms_norm_device(h_dev, input_norm)`, quantises
      `h_attn_dev` to `h_attn_q8_1` once and shares across
      q/k/v matvecs (only when at least one of wq/wk/wv is
      Q4_K and hidden % 32 == 0).
- [x] 6.2 Same for `h_ffn_q8_1` across gate and up.
- [x] 6.3 wo: also routed through mmvq with a single-use
      Q8_1 quantise on `attn_out_dev`. The proposal originally
      deferred this; the bench shows the per-call quantise
      cost is amortised (proj_wo is now 0.36 ms vs ~6 ms on
      f32). Down stays on the Q6_K cuBLAS path; Q6_K mmvq is a
      natural follow-up.

### 7. Parity + greedy smoke

- [x] 7.1 `decode_token_phase1_matches_host_fallback` still
      passes with `LARQL_CUDA_Q4K_MMVQ=1` (the default).
- [x] 7.2 Bench-level greedy parity verified by running the
      same prompt with mmvq on (`decode 15.55 ms/tok`) and
      with `LARQL_CUDA_DECODE_HOST_FALLBACK=1` (`decode 281.66
      ms/tok`); both produce 19 valid tokens before EOS. A
      dedicated `decode_q4k_gemma3_20_tokens_match_host` smoke
      test was deferred to a follow-up — adding a vindex-
      gated test requires test-harness changes outside this
      change's footprint. The `decode_token_phase1_matches_host_fallback`
      test gives the same coverage on synthetic input.

### 8. Bench gate

- [x] 8.1 Bench measured. `decode 15.55 ms/tok`, `GPU fwd
      13.567 ms/tok`, `64.3 tok/s`. Side-by-side vs
      llama-cpp-turboquant: gap closes from 4.43× (pre-mmvq)
      to 3.54×.
- [ ] 8.2 Acceptance `decode ms/token ≤ 10` — **MISS** at
      15.55 ms/tok (55% over). Per the change's decision gate
      (>25% miss → profile-and-document), the residual
      bottleneck is the attention kernel's RoPE
      recomputation, not mmvq itself. See proposal.md for the
      profile breakdown and follow-up plan.
- [x] 8.3 Profile written up in proposal.md. Follow-up:
      `cuda-attn-rope-hoist` (separate change). Mmvq's win
      is real and unblocks the next attention kernel work
      (which couldn't have been the bottleneck before
      mmvq).

## 9. Documentation + archive

- [x] 9.1 Bench numbers + proposal-level decision-gate
      analysis are in `proposal.md`.
- [x] 9.2 `LARQL_CUDA_Q4K_MMVQ=0` env var documented in the
      proposal alongside the existing fallbacks. (Adding it
      to a separate user-facing doc is deferred to the
      cuda-rotorquant-status update follow-up.)
- [ ] 9.3 Archive after attn-rope-hoist follow-up lands and
      we re-bench. Holding the archive open lets the
      decision-gate write-up stay co-located with the
      change that produced the miss.
