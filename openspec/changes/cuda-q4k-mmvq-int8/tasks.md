# cuda-q4k-mmvq-int8 — tasks

## Phase 1 — Q8_1 quantize kernel

### 1. NVRTC kernel

- [ ] 1.1 Add `QUANTIZE_Q8_1_SRC` NVRTC string in
      `crates/larql-compute/src/cuda/elem.rs` (or a new
      `cuda/quantize.rs` module if elem.rs gets too big).
      Layout: 32-element blocks, fp16 scale + fp16 sum-scaled,
      s8 quants. Matches llama.cpp's `block_q8_1` byte layout.
- [ ] 1.2 Add `quantize_q8_1_device(backend, x_dev, n) ->
      Result<Q8_1Buf, CudaInitError>` where `Q8_1Buf` is a
      typed wrapper holding `qs: CudaSlice<i8>` and
      `ds: CudaSlice<u8>` (raw bytes for the fp16×2 scale+sum).
      Both buffers come from `device_alloc_uninit`; kernel
      writes every element.
- [ ] 1.3 The `n` argument MUST be a multiple of 32; assert
      and return a typed error otherwise.

### 2. Tests

- [ ] 2.1 `q8_1_quantize_roundtrips_to_within_quant_noise` —
      random `[hidden=2560]` input, quantize, dequantize on
      host, assert max-element absolute error
      ≤ `(amax / 127.0) * 1.0` (one quantum). Locks the kernel
      to the standard Q8_1 contract.

## Phase 2 — Q4_K × Q8_1 mmvq kernel

### 3. NVRTC kernel

- [ ] 3.1 New file
      `crates/larql-compute/src/cuda/q4k_mmvq.rs`. Module-level
      `Q4K_MMVQ_SRC` const, `OnceLock<(CudaModule,
      CudaFunction)>` for lazy load.
- [ ] 3.2 Kernel: one row per warp (32 threads), strided over
      super-blocks. Body lifts
      `vec_dot_q4_K_q8_1_impl_vmmq` from
      `ggml/src/ggml-cuda/vecdotq.cuh` (MIT-licensed,
      provenance comment in source).
- [ ] 3.3 Use NVRTC's built-in `__dp4a(int, int, int) -> int`
      directly. No inline asm. (Verify NVRTC exposes it; if
      not, use the documented fallback `int dp4a(int, int,
      int) { ... }` written via `__byte_perm` + IMAD as in
      llama.cpp's `common.cuh`.)

### 4. Backend dispatch

- [ ] 4.1 Add `q4k_matvec_device_mmvq` on `CudaBackend`
      analogous to `q4k_matvec_device`.
- [ ] 4.2 `q4k_matvec_device` becomes a dispatcher that checks
      `LARQL_CUDA_Q4K_MMVQ` (default `1` after Phase 3 parity
      verifies; initially `0`). When `0`, calls the existing
      `q4k_direct::matvec_device`. When `1`, quantizes input
      to Q8_1 (if the input is f32) and calls the new mmvq
      entry point.

### 5. Tests

- [ ] 5.1 `q4k_mmvq_matches_q4k_direct` — random Q4_K packed
      weight + random input. Run both kernels; assert
      max-element ≤ 1e-3. Sizes: `(rows=4096, hidden=2560)`
      (Gemma 3 4B q_dim) and a tiny `(64, 256)` shape.
- [ ] 5.2 `q4k_mmvq_dispatch_via_env_var` — set
      `LARQL_CUDA_Q4K_MMVQ=0`, decode-token output via
      `q4k_matvec_device` returns the f32-direct result;
      set `=1`, returns the mmvq result; both within ≤ 1e-3
      of each other.

## Phase 3 — Decode wiring

### 6. Share Q8_1 across q/k/v and gate/up

- [ ] 6.1 `decode_token_device`: after `rms_norm_device(h_dev,
      input_norm)`, call `quantize_q8_1_device` once on
      `h_attn_dev`, store the result in a local
      `h_attn_q8_1`. Pass that to all three Q/K/V
      `q4k_matvec_device_mmvq` calls.
- [ ] 6.2 Same for `h_ffn_q8_1` across gate and up.
- [ ] 6.3 wo and down: keep on the f32 direct path for now.
      Add a TODO comment pointing at the follow-up
      `cuda-q4k-mmvq-extend` (not yet drafted).

### 7. Parity + greedy smoke

- [ ] 7.1 Existing
      `decode_token_phase1_matches_host_fallback` MUST still
      pass with `LARQL_CUDA_Q4K_MMVQ=1` (the default after
      Phase 3 lands).
- [ ] 7.2 New `decode_q4k_gemma3_20_tokens_match_host` —
      `#[ignore]`'d, gated on `LARQL_CUDA_AVAILABLE=1` and the
      real Gemma 3 4B Q4_K vindex on disk. Runs 20 decode
      steps with mmvq, again with
      `LARQL_CUDA_DECODE_HOST_FALLBACK=1`, asserts greedy
      argmax token IDs are identical.

### 8. Bench gate

- [ ] 8.1 Run the standard bench:
      `LARQL_CUDA_AVAILABLE=1 ./target/release/larql bench
      output/gemma-3-4b-it-vindex --backends cuda --tokens 20
      --warmup 3 --verbose`. Record `decode ms/token`,
      `GPU fwd ms/token`, `tok/s`.
- [ ] 8.2 Acceptance: `decode ms/token ≤ 10` AND `GPU fwd
      ms/token ≤ 8`. Compare side-by-side with
      llama-cpp-turboquant's `4.40 ms/tok` / `227.5 tok/s`
      baseline; record the gap-closure ratio in the PR
      description.
- [ ] 8.3 If miss > 25% (i.e., > 12.5 ms/tok), abort: do
      `LARQL_CUDA_DECODE_PROFILE=1` and write up which bucket
      moved the wrong way. Most likely cause: Q8_1 quantize on
      the critical path → fuse into `rms_norm_device` as a
      separate follow-up.

## 9. Documentation + archive

- [ ] 9.1 Update `docs/cuda-rotorquant-status.md` with the
      bench-progress table including the mmvq row and the
      llama-cpp-turboquant comparator.
- [ ] 9.2 Document `LARQL_CUDA_Q4K_MMVQ=0` env var alongside
      the existing `LARQL_CUDA_Q4K_HOST_DEQUANT=1`,
      `LARQL_CUDA_Q6K_HOST_DEQUANT=1`, and
      `LARQL_CUDA_DECODE_HOST_FALLBACK=1` flags.
- [ ] 9.3 If acceptance hit, archive:
      `openspec archive cuda-q4k-mmvq-int8`.
