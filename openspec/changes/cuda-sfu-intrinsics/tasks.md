# cuda-sfu-intrinsics — tasks

## 1. Kernel-source edits

- [x] 1.1 Swap `cosf` → `__cosf`, `sinf` → `__sinf`,
      `expf` → `__expf`, `powf` → `__powf` in `attn.rs`
      NVRTC sources (FUSED_DECODE_ATTN_SRC,
      FUSED_PREFILL_ATTN_SRC, KV_CACHE_WRITE_SEQ_SRC,
      SOFTMAX_SRC, QKV_RMS_PROJ_SRC).
- [x] 1.2 Keep `tanhf` unchanged — no `__tanhf` intrinsic
      exists.
- [x] 1.3 Swap `expf` → `__expf` in
      `elem.rs::silu_gate_up_f32`.
- [x] 1.4 Add `use_fast_math: Some(true)` (= cudarc's
      `--fmad=true`) to the `CompileOptions` for the three
      attention kernels.

## 2. Tests

- [x] 2.1 `decode_token_phase1_matches_host_fallback`
      (≤ 1e-3 vs CPU host fallback) still passes after the
      swap.
- [x] 2.2 Full CUDA suite (193 tests) green.

## 3. Bench gate

- [x] 3.1 5 bench runs on the dev box averaging:
      `decode 9.35 ms/tok`, `tok/s 107`, `attn_call 2.68 ms`.
      Cleared all targets.

## 4. Documentation + archive

- [x] 4.1 Final numbers in proposal.md.
- [ ] 4.2 Archive together with the rest of today's CUDA
      perf work in a single sweep.
