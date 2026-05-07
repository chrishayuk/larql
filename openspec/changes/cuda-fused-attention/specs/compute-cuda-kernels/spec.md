## ADDED Requirements

### Requirement: Scaled-softmax CUDA kernel via NVRTC

The CUDA backend SHALL ship a custom scaled-softmax kernel
compiled via cudarc NVRTC at backend init and cached under
`$XDG_CACHE_HOME/larql/cudarc/<arch>/softmax.cubin`. The kernel
MUST support per-row reduction (one CUDA block per row), arbitrary
row length up to seq_len = 4096, and a `scale: f32` multiplier
applied before the row max.

#### Scenario: softmax kernel matches naive reference at small shape
- **WHEN** a 4×16 random input is softmaxed by both the CUDA kernel and a naive scalar reference (with the same `scale`)
- **THEN** the maximum absolute element difference SHALL be ≤ 1e-3 and cosine similarity ≥ 0.9999 per row
<!-- test: larql_compute::test_cuda_attn::softmax_small_parity -->

#### Scenario: softmax kernel matches reference at long-row shape
- **WHEN** a 32×4096 input is softmaxed on both backends
- **THEN** the maximum absolute element difference SHALL be ≤ 1e-3 per row
<!-- test: larql_compute::test_cuda_attn::softmax_long_row_parity -->

### Requirement: Causal mask and softcap fold into the softmax kernel

The kernel SHALL accept an optional causal mask (positions `j > i`
get `-inf` before max) and an optional softcap (`x = softcap *
tanh(x / softcap)` applied before max). Both knobs reduce to no-op
when off.

#### Scenario: causal mask zeroes future positions
- **WHEN** a 4×4 input is run with causal=true and the resulting row 2 has positions 3 and 0..2 inspected
- **THEN** position 3 SHALL be ≤ 1e-9 (mask applied) and positions 0..2 SHALL sum to ~1.0
<!-- test: larql_compute::test_cuda_attn::softmax_causal_mask -->

#### Scenario: softcap clamps large logits
- **WHEN** an input with very large positive values is softmaxed with `softcap=50.0`
- **THEN** the result SHALL match a reference that applies `x = 50 * tanh(x / 50)` before softmax, within 1e-3 absolute
<!-- test: larql_compute::test_cuda_attn::softmax_softcap_50 -->

### Requirement: `decode_attention` helper chains GEMM + softmax + GEMM

The helper `cuda::attn::decode_attention(drv, q, k, v, n_q, n_kv, head_dim, opts)` SHALL produce `[n_q, head_dim]` row-major output equal to a naive single-head attention reference within 1e-3 absolute and cosine similarity ≥ 0.9999. The helper MUST issue exactly one host roundtrip (one synchronize at the end).

#### Scenario: decode_attention parity at small head shape
- **WHEN** a single-head attention with n_q=8, n_kv=8, head_dim=64 is computed via `decode_attention` and via a naive CPU loop
- **THEN** the outputs SHALL agree within 1e-3 absolute, cosine ≥ 0.9999
<!-- test: larql_compute::test_cuda_attn::decode_attention_small_parity -->

#### Scenario: decode_attention parity at Gemma 4B head shape
- **WHEN** a single-head decode attention with n_q=1, n_kv=2048, head_dim=320 is computed via `decode_attention` and a naive CPU loop
- **THEN** the outputs SHALL agree within 1e-3 absolute, cosine ≥ 0.9999
<!-- test: larql_compute::test_cuda_attn::decode_attention_gemma4b_head_parity -->

### Requirement: Capability::FlashAttentionV2 advertised after this change

`CudaBackend::supports(Capability::FlashAttentionV2)` SHALL return
`true` after `cuda-fused-attention` lands. Other previously-set bits
(`Cuda`, `F32Gemv`, `QuantMatVec`, `Q4VecMat`) MUST stay set.

#### Scenario: capability flips on
- **WHEN** the backend is constructed and `supports` is queried
- **THEN** `FlashAttentionV2` SHALL be `true`; previously-set bits SHALL remain `true`
<!-- test: larql_compute::cuda::backend::tests::supports_fa2_after_fused_attention -->
