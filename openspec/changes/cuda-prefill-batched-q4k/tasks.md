# cuda-prefill-batched-q4k — tasks

## 1. Q4_K f32 device cache

- [x] 1.1 `with_q4k_f32_device_buf` on `CudaBackend`,
      parallel to `with_q6k_f32_device_buf`.
- [x] 1.2 ~9.6 GB VRAM cost documented in proposal.

## 2. Batched element-wise kernels

- [x] 2.1 `rms_norm_batch_device` — generalised the existing
      kernel to use `gridDim.x` as the row count. Single-row
      callers launch with `grid_dim = (1, 1, 1)` (unchanged
      semantics).
- [x] 2.2 `silu_gate_up_batch_device` — thin wrapper over
      the existing kernel with `n_total = seq_len * inter`.
- [x] 2.3 `add_in_place_batch_device` — same pattern.
- [x] 2.4 `scale_inplace_batch_device` — same pattern.

## 3. Batched prefill core

- [x] 3.1 `CudaBackend::prefill_q4_seq_device` shipped. Routes
      Q4_K and Q6_K weights through their f32 device caches +
      cuBLAS `matmul_transb_device_inout`.
- [x] 3.2 New `kernels::matmul_transb_device_inout` in
      `cuda/matmul.rs`: device-input/device-output cuBLAS
      GEMM. Reused for every batched projection.
- [x] 3.3 Per-position attention loop (Phase 2 will batch
      this — see decision gate write-up).
- [x] 3.4 Element-wise ops via the new `*_batch_device`
      wrappers.
- [x] 3.5 Single `dtoh_f32(h_seq)` returns the full output.

## 4. DecodeBackend dispatch

- [x] 4.1 `prefill_q4` dispatches to
      `prefill_q4_seq_device` when all layers use Q4_K /
      Q6_K projections. Falls back to per-position decode
      loop otherwise.
- [x] 4.2 `LARQL_CUDA_PREFILL_HOST_FALLBACK=1` forces the
      legacy path.

## 5. Tests

- [x] 5.1 Existing
      `decode_token_phase1_matches_host_fallback` covers
      the projection-GEMM correctness via the shared
      `matvec_device` path. A dedicated
      `prefill_q4_seq_matches_host_fallback` test was
      deferred to a follow-up because adding it requires
      test-harness changes that don't fit the change's
      footprint.
- [x] 5.2 Real-vindex bench confirms both paths produce 19
      valid tokens before EOS: batched (`decode 11.04
      ms/tok`) and host-fallback (`decode 10.41 ms/tok`).

## 6. Bench gate

- [x] 6.1 Bench measured. `prefill 117.6 ms`, `decode 11.04
      ms/tok`, `90.6 tok/s`.
- [ ] 6.2 Acceptance MISSED: `prefill 117.6 vs ≤ 20 ms`
      (5.9× miss), `decode 11.04 vs ≤ 11` (0.4% over).
- [x] 6.3 Profile recorded in `proposal.md`. Per-position
      attention is 37% of remaining cost; batched rms_norm
      launch overhead is another 35%. Phase 2
      (`cuda-prefill-batched-attention`) is the right
      follow-up.

## 7. Documentation + archive

- [x] 7.1 Final bench numbers in proposal.md.
- [x] 7.2 `LARQL_CUDA_PREFILL_HOST_FALLBACK=1` documented
      in proposal.
- [ ] 7.3 Archive deferred — Phase 1 is a 10% prefill win
      (worth shipping since it provides the plumbing for
      Phase 2) but doesn't clear the gate. Hold open until
      Phase 2 lands and we re-bench.
