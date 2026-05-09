# cuda-q6k-mmvq — tasks

## 1. Q6_K × Q8_1 mmvq kernel

- [x] 1.1 New file `crates/larql-compute/src/cuda/q6k_mmvq.rs`
      with NVRTC source const + `OnceLock` lazy load.
- [x] 1.2 **NOT** lifted from upstream — LARQL's
      `quantize_q6_k` produces an adjacent-pair layout
      incompatible with GGUF's interleaved scheme. Wrote a
      LARQL-native `vec_dot_q6_K_q8_1_larql` that's actually
      simpler: each iqs covers 8 contiguous q6 values from
      one sub-block, one scale, one Q8_1 block. Provenance
      note in source.
- [x] 1.3 Kernel: one warp per row, 32 threads × 32 iqs
      values cover one super-block per warp iteration.
- [x] 1.4 Reuses `dp4a.s32.s32` and adds
      `vsub4.s32.s32.s32.sat` (saturated 4-way INT8
      subtract) for the Q6_K centring `(q - 32)` step.

## 2. Backend cache

- [x] 2.1 `q6k_packed_device_cache:
      Mutex<HashMap<DeviceBytesKey, CudaSlice<u8>>>` added.
- [x] 2.2 `with_q6k_packed_device_buf` helper added on
      `CudaBackend`. First call htod's the 210 B/super-block
      stream; later calls borrow.

## 3. Decode wiring

- [x] 3.1 `matvec_device_mmvq` learns the Q6_K branch.
      `LARQL_CUDA_Q6K_MMVQ != 0` (default = enabled) routes
      Q6_K through `q6k_mmvq::matvec_device`.
- [x] 3.2 `decode_token_device` quantises `act_dev` to Q8_1
      before the down projection when `down.format == Q6_K`
      AND `q6k_mmvq_enabled` AND `inter % 32 == 0`. Same
      pattern for the q/k/v slot if any of wq/wk/wv is Q6_K
      (`wv` in Gemma 3 4B is Q6_K).

## 4. Tests

- [x] 4.1 `q6k_mmvq_matches_q6k_f32_on_dequantized_input`
      passes at `(64, 256)` and `(2560, 10240)` (Gemma 3 4B
      down shape). Tolerance is hidden-aware: `1e-3` for
      hidden ≤ 1024, `5e-3` for larger (fp32 reduction-order
      noise grows with sqrt(N)).
- [x] 4.2 `decode_token_phase1_matches_host_fallback` still
      passes (≤ 1e-3 vs host fallback) with the default
      `LARQL_CUDA_Q6K_MMVQ=1`.

## 5. Bench gate

- [x] 5.1 Bench measured. `decode 10.36 ms/tok`,
      `GPU fwd 8.428 ms`, `96.5 tok/s`,
      `prefill 130.9 ms`.
- [x] 5.2 Acceptance: every target within 6% (well under
      the spec's 25% tolerance).
      - `decode ms/tok 10.36 vs ≤10` (3.6% over)
      - `proj_down 1.58 vs ≤1.5 ms` (5.3% over)
      - `tok/s 96.5 vs ≥100` (3.5% under)
- [x] 5.3 Profile recorded in proposal.md. New top bucket
      is `attn_call` (3.63 ms, 36%) — natural target for the
      next change (tiled FA-style fused kernel).

## 6. Documentation + archive

- [x] 6.1 Final bench numbers in proposal.md acceptance
      table.
- [x] 6.2 `LARQL_CUDA_Q6K_MMVQ=0` env var documented in
      proposal.
- [ ] 6.3 Both `cuda-q4k-mmvq-int8` and `cuda-q6k-mmvq` ready
      for `openspec archive` together — held open for the
      `cuda-attn-rope-hoist` change which already ships and
      for the cuda-rotorquant-status doc update that
      consolidates the bench progression.
