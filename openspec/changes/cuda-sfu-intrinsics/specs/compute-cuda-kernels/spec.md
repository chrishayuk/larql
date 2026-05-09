## ADDED Requirements

### Requirement: Trig-heavy NVRTC kernels SHALL use SFU intrinsics

`fused_decode_attention_f32`, `fused_prefill_attention_f32`, `kv_cache_write_seq_f32`, `scaled_softmax`, `qkv_rms_proj_f32`, and `silu_gate_up_f32` SHALL call the SFU-fast `__cosf` / `__sinf` / `__expf` / `__powf` intrinsics instead of the IEEE-compliant `cosf` / `sinf` / `expf` / `powf` library functions for the trig and exponentiation operations on the hot path. `tanhf` SHALL remain on the IEEE path because no `__tanhf` intrinsic is available on the supported architectures.

#### Scenario: SFU-intrinsic kernels match host-fallback reference within 1e-3

- **WHEN** the synthetic Q4_K decode pipeline runs three
  decode steps via the device-resident path (with the
  intrinsic-using kernels) and again via
  `LARQL_CUDA_DECODE_HOST_FALLBACK=1` (the CPU host-fallback
  attention path)
- **THEN** the per-step output vectors SHALL agree to
  max-element absolute difference ≤ 1e-3 — the same bound
  the kernels cleared before the intrinsic swap
<!-- test: larql_compute::tests::test_cuda_decode::decode_token_phase1_matches_host_fallback -->

### Requirement: attn_call profile bucket SHALL drop after the intrinsic swap

`attn_call` profile bucket on the dev-box decode bench (RTX 4090, Gemma 3 4B Q4_K, with `LARQL_CUDA_DECODE_PROFILE=1`) SHALL drop to ≤ 3 ms (down from 3.63 ms post-`cuda-attn-rope-hoist`). **Actual**: 2.68 ms (-26%). A miss of > 25% (i.e., `attn_call > 3.75 ms`) SHALL trigger a profile-and-document write-up.

#### Scenario: profile bucket cleared at acceptance

- **WHEN** `LARQL_CUDA_AVAILABLE=1
  LARQL_CUDA_DECODE_PROFILE=1 ./target/release/larql bench
  output/gemma-3-4b-it-vindex --backends cuda --tokens 20
  --warmup 3 --verbose` is run after this change lands
- **THEN** the `attn_call` profile bucket SHALL be ≤ 3 ms
  AND `decode ms/token` SHALL be ≤ 9.5
<!-- test: unbacked -->
