## ADDED Requirements

### Requirement: f32 dense matmul matches ndarray reference

The CPU `f32_matmul` and `f32_matmul_transb` kernels SHALL be the
ndarray-routed implementation of dense `A × B` and `A × B^T`. They
MUST delegate to `ndarray::dot` (which itself binds Accelerate or
OpenBLAS depending on platform) and MUST return outputs that match a
pure-CPU `ndarray::dot` reference within 1e-6 absolute tolerance for
random inputs in `[-1, 1]`. Dispatch through `CpuBackend::matmul` /
`matmul_transb` MUST return identical shapes and values.

#### Scenario: matmul shape and value match ndarray dot
- **WHEN** `matmul(A, B)` is called for random `A: [m, k]` and `B: [k, n]`
- **THEN** the result SHALL have shape `[m, n]` and SHALL equal `ndarray::dot(&A, &B)` within 1e-6
<!-- test: larql_compute::cpu::ops::f32_matmul::tests::matmul_correct_shape -->
<!-- test: larql_compute::cpu::ops::f32_matmul::tests::matmul_identity -->
<!-- test: larql_compute::test_correctness::cpu_matmul_matches_ndarray -->

#### Scenario: matmul_transb shape and value match ndarray reference
- **WHEN** `matmul_transb(A, B)` is called for random `A: [m, k]` and `B: [n, k]`
- **THEN** the result SHALL have shape `[m, n]` and SHALL equal `A.dot(&B.t())` within 1e-6
<!-- test: larql_compute::cpu::ops::f32_matmul::tests::matmul_transb_correct_shape -->
<!-- test: larql_compute::test_correctness::cpu_matmul_transb_matches_ndarray -->

### Requirement: Q4_0 / Q8_0 matvec correctness vs dequantise reference

CPU Q4_0 and Q8_0 matvec kernels SHALL produce results that match the
"dequantise → ndarray dot" reference within Q4 / Q8 quantisation noise
(cosine similarity ≥ 0.95). The covered kernels are
`cpu::ops::q4_matvec::dispatch`, `q4_matvec::dispatch_q8`,
`q8_matvec::dispatch`, and the `q4_vecmat` transposed-shape kernel.
The kernels MUST handle zero-input gracefully (return all-zero output
without panic) and MUST emit one output element per row of the weight
matrix. The Q4_0 path MUST internally re-quantise f32 input to Q8 via
`quantize_to_q8`.

#### Scenario: Q4_0 matvec matches dequantise reference
- **WHEN** `q4_matvec` is dispatched on a random Q4_0 weight matrix and Q8 input
- **THEN** the kernel result SHALL agree with the per-row "dequant Q4 then dot Q8" reference within Q4 noise (max relative error ≤ 0.1, cosine ≥ 0.99)
<!-- test: larql_compute::test_q4_x86_correctness::q4_matvec_matches_dequant_reference -->
<!-- test: larql_compute::test_q4_x86_correctness::q4_matvec_vs_raw_f32_matvec_quant_noise -->
<!-- test: larql_compute::cpu::ops::q4_matvec::tests::q4_matvec_produces_output -->
<!-- test: larql_compute::cpu::ops::q4_matvec::tests::q4_matvec_zero_input -->
<!-- test: larql_compute::test_correctness::cpu_q4_matvec_nonzero -->
<!-- test: larql_compute::test_correctness::cpu_has_q4 -->

#### Scenario: Q4 vecmat (down-projection shape) matches reference
- **WHEN** `q4_vecmat(activation, q4_data, …)` is called with a non-zero activation
- **THEN** the result SHALL agree with the dequantise-and-multiply reference within Q4 quant noise
<!-- test: larql_compute::cpu::ops::q4_vecmat::tests::q4_vecmat_produces_output -->
<!-- test: larql_compute::cpu::ops::q4_vecmat::tests::q4_vecmat_zero_activation -->
<!-- test: larql_compute::test_q4_x86_correctness::q4_vecmat_matches_dequant_reference -->
<!-- test: larql_compute::test_correctness::cpu_q4_vecmat_nonzero -->

#### Scenario: Q8 matvec preserves cosine similarity to f32
- **WHEN** `q8_matvec::dispatch` is called against a Q8-quantised version of a f32 matrix
- **THEN** the resulting vector SHALL have cosine similarity ≥ 0.99 with the f32-direct dot product
<!-- test: larql_compute::cpu::ops::q8_matvec::tests::q8_matvec_produces_output -->
<!-- test: larql_compute::cpu::ops::q8_matvec::tests::q8_vs_f32_high_cosine -->

### Requirement: Q4_K / Q4_KF / Q6_K matvec match reference and Ollama layout

CPU Q4_K-family matvec kernels SHALL implement the Ollama-compatible
super-block layout (256 elements per super-block; 8 sub-block scales
+ 8 mins per super-block; f16 super-block scale and min). The covered
kernels are `q4k_matvec`, `q4kf` (Q4_K-flat), `q6k_matvec`, and the
fused `q4k_q8k_matvec*` family. Their outputs MUST match a
deterministic dequantise-then-dot reference within the documented
quantisation noise. Misaligned or truncated input buffers MUST return
an empty vector rather than panic.

#### Scenario: Q4_K matches dequantise-and-dot on production layout
- **WHEN** `q4k_matvec` is run on Q4_K weights produced by `quantize_q4_k` and matched against `dequantize_q4_k(...) → dot`
- **THEN** the relative error SHALL stay within Q4_K quant noise on both single- and multi-superblock shapes
<!-- test: larql_compute::cpu::ops::q4_common::tests::q4k_matvec_matches_dequant_then_matmul -->
<!-- test: larql_compute::cpu::ops::q4_common::tests::q4k_matvec_multi_block_matches_dequant -->
<!-- test: larql_compute::cpu::ops::q4k_matvec::tests::q4k_matches_dequantize_reference_single_superblock -->
<!-- test: larql_compute::cpu::ops::q4k_matvec::tests::q4k_matches_dequantize_reference_multi_superblock -->
<!-- test: larql_compute::test_q4k_parity::q4k_lifted_matches_larql_models_reference -->
<!-- test: larql_compute::test_q4k_parity::q4k_round_trip_within_quant_noise -->

#### Scenario: Q4_K rejects malformed input rather than panicking
- **WHEN** `q4k_matvec` receives a buffer whose length is not a multiple of 256 elements per super-block, or a truncated buffer
- **THEN** the kernel SHALL return an empty `Vec<f32>` rather than panic
<!-- test: larql_compute::cpu::ops::q4_common::tests::q4k_matvec_rejects_non_multiple_of_256 -->
<!-- test: larql_compute::test_q4k_parity::q4k_misaligned_input_returns_empty -->
<!-- test: larql_compute::test_q4k_parity::q4k_truncated_input_returns_empty -->

#### Scenario: Q4_KF flat layout converts losslessly from Q4_K
- **WHEN** `q4k_to_q4kf` is run on Q4_K weights and the resulting Q4_KF data is fed back into matvec
- **THEN** the matvec output SHALL agree with the original Q4_K matvec (same numerical content under a different memory layout)
<!-- test: larql_compute::cpu::ops::q4_common::tests::q4k_to_q4kf_converts_format -->
<!-- test: larql_compute::cpu::ops::q4_common::tests::q4kf_output_size -->

#### Scenario: Q6_K matvec matches dequantise reference
- **WHEN** `q6k_matvec::dispatch` is run against `quantize_q6_k` output
- **THEN** the result SHALL match `dequantize_q6_k(...) → dot` within Q6_K quant noise
<!-- test: larql_compute::cpu::ops::q6k_matvec::tests::q6k_produces_nonzero -->
<!-- test: larql_compute::cpu::ops::q4_common::tests::q6_k_round_trip_via_matvec -->
<!-- test: larql_compute::cpu::ops::q4_common::tests::q6_k_output_size -->

#### Scenario: Q4_K / Q6_K f16-to-f32 conversion is bit-exact and handles edge cases
- **WHEN** `f16_to_f32` is invoked across the full 16-bit input space (or specifically on negative zero, subnormals, infinities, and NaN)
- **THEN** the conversion SHALL match the `powi`-based reference bit-for-bit and SHALL preserve the documented IEEE-half semantics
<!-- test: larql_compute::cpu::ops::q4_common::tests::f16_to_f32_bit_exact_for_all_inputs -->
<!-- test: larql_compute::cpu::ops::q4k_matvec::tests::f16_to_f32_neg_zero -->
<!-- test: larql_compute::cpu::ops::q4k_matvec::tests::f16_to_f32_subnormal_positive -->
<!-- test: larql_compute::cpu::ops::q4k_matvec::tests::f16_to_f32_subnormal_negative -->
<!-- test: larql_compute::cpu::ops::q4k_matvec::tests::f16_to_f32_neg_infinity -->
<!-- test: larql_compute::cpu::ops::q4k_matvec::tests::f16_to_f32_nan -->
<!-- test: larql_compute::cpu::ops::q6k_matvec::tests::f16_to_f32_neg_zero -->
<!-- test: larql_compute::cpu::ops::q6k_matvec::tests::f16_to_f32_subnormal_positive -->
<!-- test: larql_compute::cpu::ops::q6k_matvec::tests::f16_to_f32_subnormal_negative -->
<!-- test: larql_compute::cpu::ops::q6k_matvec::tests::f16_to_f32_neg_infinity -->
<!-- test: larql_compute::cpu::ops::q6k_matvec::tests::f16_to_f32_nan -->

### Requirement: Q4_K · Q8_K fused dot path with NEON / AVX2 SIMD

The `q4k_q8k_matvec_*` and `q6k_q8k_matvec_*` family SHALL provide a
scalar reference plus NEON (aarch64) and AVX2 (x86_64) SIMD
specialisations that produce bit-exact results vs the scalar reference
on the supported architecture. The fused gate+up path
(`q4k_q8k_gate_up_*`) SHALL produce results identical to running gate
and up matvecs separately and combining them. Q8_K activation
quantisation MUST round-trip within one quantisation step on f32
inputs and MUST emit clean zeros on zero input.

#### Scenario: Q8_K activation quantisation round-trips
- **WHEN** `quantize_x_to_q8k` is applied to f32 input and dequantised back
- **THEN** the maximum absolute error SHALL stay within one Q8 quant step, and zero input SHALL produce a clean zero quantisation
<!-- test: larql_compute::cpu::ops::q4k_q8k_dot::tests::q8k_quantize_round_trip_within_quant_step -->
<!-- test: larql_compute::cpu::ops::q4k_q8k_dot::tests::q8k_zero_input_clean -->

#### Scenario: NEON Q4_K·Q8_K matches scalar reference bit-exactly on aarch64
- **WHEN** the NEON path `q4k_q8k_matvec_neon` runs on aarch64 against the scalar reference
- **THEN** results SHALL be bit-equal across single-row, multi-block, two-row, and in-place variants
<!-- test: larql_compute::cpu::ops::q4k_q8k_dot::tests::q8k_matvec_neon_matches_scalar_bit_exact -->
<!-- test: larql_compute::cpu::ops::q4k_q8k_dot::tests::q8k_matvec_2row_matches_single_row_bit_exact -->
<!-- test: larql_compute::cpu::ops::q4k_q8k_dot::tests::q8k_in_place_matches_alloc_version -->

#### Scenario: AVX2 Q4_K·Q8_K matches scalar reference on x86_64
- **WHEN** the AVX2 path runs on x86_64 against the scalar reference
- **THEN** results SHALL agree with the scalar reference within Q4_K noise
<!-- test: larql_compute::cpu::ops::q4k_q8k_dot::tests::q8k_matvec_avx2_matches_scalar -->

#### Scenario: Fused gate+up matches separate matvecs
- **WHEN** `q8k_gate_up_fused` is invoked
- **THEN** its outputs SHALL equal the pair of separate Q4_K·Q8_K matvecs run on the same gate and up weights
<!-- test: larql_compute::cpu::ops::q4k_q8k_dot::tests::q8k_gate_up_fused_matches_separate_matvecs -->

#### Scenario: Q4_K·Q8_K matvec preserves cosine vs f32 cached reference
- **WHEN** `q4k_q8k_matvec_scalar` runs on Q4_K weights and Q8_K activations
- **THEN** the output SHALL stay within Q8 noise of the f32-cached reference, and SHALL handle multi-block shapes
<!-- test: larql_compute::cpu::ops::q4k_q8k_dot::tests::q8k_matvec_matches_f32_cached_within_q8_noise -->
<!-- test: larql_compute::cpu::ops::q4k_q8k_dot::tests::q8k_matvec_multi_block_within_noise -->

#### Scenario: Q4_K·Q8_K matvec rejects degenerate inputs
- **WHEN** `q4k_q8k_matvec_into` receives zero dimensions or a short weight buffer
- **THEN** the helper SHALL return without panic and SHALL leave the output zero
<!-- test: larql_compute::cpu::ops::q4k_q8k_dot::tests::q8k_matvec_zero_dims_returns_zero -->
<!-- test: larql_compute::cpu::ops::q4k_q8k_dot::tests::q8k_matvec_short_weight_buffer_returns_zero -->

### Requirement: GEGLU activation and fused causal attention

The CPU GEGLU helper SHALL implement `out[i] = silu(gate[i]) * up[i]`
both in-place (`geglu_silu`) and allocating (`geglu_silu_alloc`), with
`silu(x) = x * sigmoid(x)`. The CPU causal-attention helper MUST
implement scaled-dot-product attention with a triangular causal mask,
return outputs of shape `[seq_len, num_heads * head_dim]`, refuse to
let later positions attend to earlier positions' future neighbours,
and average value vectors when keys are uniform.

#### Scenario: SiLU and GEGLU produce documented values
- **WHEN** `silu(x)` is called for x = 0, x = 1, or `geglu_silu(gate, up, out)` is called for trivial inputs
- **THEN** the outputs SHALL match the closed-form `x * sigmoid(x)` and `silu(gate) * up` references and the in-place variant SHALL match the allocating variant
<!-- test: larql_compute::cpu::ops::geglu::tests::silu_basic -->
<!-- test: larql_compute::cpu::ops::geglu::tests::geglu_basic -->
<!-- test: larql_compute::cpu::ops::geglu::tests::geglu_in_place -->

#### Scenario: Causal attention masks future positions
- **WHEN** a multi-position causal-attention pass is run with logged keys/values and a comparison reference
- **THEN** the position-i output SHALL depend only on positions ≤ i, output SHALL have shape `[seq_len, num_heads * head_dim]`, and uniform keys SHALL produce a value-average
<!-- test: larql_compute::cpu::ops::attention::tests::single_token_attention -->
<!-- test: larql_compute::cpu::ops::attention::tests::causal_mask -->
<!-- test: larql_compute::cpu::ops::attention::tests::output_shape -->
<!-- test: larql_compute::cpu::ops::attention::tests::uniform_keys_average_values -->
<!-- test: larql_compute::cpu::ops::attention::tests::later_positions_cannot_see_future -->

### Requirement: Vector and outer-combine primitives

The vector helpers (`dot`, `norm`, `cosine`) SHALL implement the
standard linear-algebra reference (Σ aᵢbᵢ; √Σ aᵢ²; aᵢbᵢ / (‖a‖‖b‖)).
`outer_post_norm_residual` SHALL apply RMSNorm with the bias offset
documented per family, then add the residual; it MUST skip the
norm-weight branch when `weight = None` and MUST add the per-element
offset to each weight when applied. `apply_layer_scalar_in_place` MUST
multiply by the layer scalar and MUST be a no-op when the scalar is
0.0 (per Gemma 4 layer-scalar=0 convention) or 1.0.

#### Scenario: Vector primitives match closed-form reference
- **WHEN** `dot`, `norm`, and `cosine` are called on synthetic vectors
- **THEN** outputs SHALL match the closed form (orthogonal vectors → 0, identical → 1, opposite → −1, unit norm → 1)
<!-- test: larql_compute::cpu::ops::vector::tests::dot_basic -->
<!-- test: larql_compute::cpu::ops::vector::tests::dot_orthogonal -->
<!-- test: larql_compute::cpu::ops::vector::tests::norm_unit -->
<!-- test: larql_compute::cpu::ops::vector::tests::cosine_identical -->
<!-- test: larql_compute::cpu::ops::vector::tests::cosine_orthogonal -->
<!-- test: larql_compute::cpu::ops::vector::tests::cosine_opposite -->

#### Scenario: outer_post_norm_residual matches the Metal-equivalent path
- **WHEN** `outer_post_norm_residual` is called with weight Some / None and various offsets
- **THEN** the output SHALL match the handwritten Metal-mirroring reference, the `weight = None` path SHALL skip norm scaling, and norm offsets SHALL be added per-weight
<!-- test: larql_compute::cpu::ops::outer_combine::tests::outer_post_norm_residual_matches_handwritten_metal_logic -->
<!-- test: larql_compute::cpu::ops::outer_combine::tests::outer_post_norm_residual_skips_norm_when_weight_none -->
<!-- test: larql_compute::cpu::ops::outer_combine::tests::norm_offset_is_added_to_each_weight -->

#### Scenario: Layer scalar is a no-op for sentinel values
- **WHEN** `apply_layer_scalar_in_place` is called with scalar = 0.0 or scalar = 1.0
- **THEN** the buffer SHALL be unchanged (these encode "skip" in the Gemma 4 convention)
<!-- test: larql_compute::cpu::ops::outer_combine::tests::apply_layer_scalar_in_place_skips_identity_and_zero -->
<!-- test: larql_compute::cpu::ops::outer_combine::tests::apply_layer_scalar_in_place_multiplies -->

### Requirement: MoE expert kernels and dispatch caching

The `cpu::ops::moe` kernels SHALL implement an MoE forward pass on
CPU: per-expert pre-norm (with optional weight), gated FFN
(`gate · activation · up → down`), top-K + softmax routing, and
weighted combine. The expert cache SHALL dequantise BF16 / Q4_K
weights once and reuse them across token positions, MUST reject
unsupported formats and malformed buffers (truncated, misaligned), and
MUST be safe under parallel hits from multiple threads. The forward
path SHALL produce zero output on zero input and zero experts, MUST
respect both SiLU and GELU-tanh activations, MUST honour per-expert
scale and router-scale vectors, and MUST index expert weights by
expert id (not by top-K position).

#### Scenario: Per-expert math primitives match references
- **WHEN** `bf16_to_f32`, `rms_norm`, `silu`, `top_k`, `softmax`, and `matmul_vec` helpers are exercised on hand-checked inputs
- **THEN** each helper SHALL match its closed-form reference (zero-dim → zeros; rms-norm with empty weight is passthrough; softmax sums to 1; top-K is descending and capped at len)
<!-- test: larql_compute::cpu::ops::moe::math::tests::bf16_to_f32_known_values -->
<!-- test: larql_compute::cpu::ops::moe::math::tests::rms_norm_constant_input -->
<!-- test: larql_compute::cpu::ops::moe::math::tests::rms_norm_empty_weight_passthrough -->
<!-- test: larql_compute::cpu::ops::moe::math::tests::rms_norm_no_weight_normalises_to_unit_rms -->
<!-- test: larql_compute::cpu::ops::moe::math::tests::silu_known_values -->
<!-- test: larql_compute::cpu::ops::moe::math::tests::top_k_descending_with_k_capped_at_len -->
<!-- test: larql_compute::cpu::ops::moe::math::tests::softmax_sums_to_one -->
<!-- test: larql_compute::cpu::ops::moe::math::tests::matmul_vec_matches_scalar_reference -->
<!-- test: larql_compute::cpu::ops::moe::math::tests::matmul_vec_zero_dimensions_returns_zeros -->

#### Scenario: Single-expert forward matches manual pre-norm + gated FFN
- **WHEN** `run_single_expert_with_norm` is called on synthetic weights and compared to a manual implementation
- **THEN** the output SHALL match within float tolerance, zero hidden / zero inter SHALL produce empty / zero output, non-zero weights SHALL produce non-zero output, and GELU-tanh SHALL differ from SiLU
<!-- test: larql_compute::cpu::ops::moe::expert::tests::zero_inter_returns_zero_vec -->
<!-- test: larql_compute::cpu::ops::moe::expert::tests::zero_hidden_returns_empty -->
<!-- test: larql_compute::cpu::ops::moe::expert::tests::nonzero_weights_produce_nonzero_output -->
<!-- test: larql_compute::cpu::ops::moe::expert::tests::with_norm_matches_manual_prenorm -->
<!-- test: larql_compute::cpu::ops::moe::expert::tests::gelu_tanh_differs_from_silu -->

#### Scenario: Expert cache is format-aware and corruption-resistant
- **WHEN** the expert cache dispatches BF16, Q4_K, F32, an unsupported format, or a malformed Q4_K buffer
- **THEN** supported formats SHALL round-trip through the cache, F32 SHALL pass through, unsupported formats SHALL return empty, malformed Q4_K SHALL return empty, and concurrent hits SHALL not deadlock or corrupt
<!-- test: larql_compute::cpu::ops::moe::cache::tests::bf16_dispatch_round_trip -->
<!-- test: larql_compute::cpu::ops::moe::cache::tests::q4k_dispatch_round_trip -->
<!-- test: larql_compute::cpu::ops::moe::cache::tests::f32_dispatch_passthrough -->
<!-- test: larql_compute::cpu::ops::moe::cache::tests::unsupported_format_returns_empty -->
<!-- test: larql_compute::cpu::ops::moe::cache::tests::q4k_truncated_input_returns_empty -->
<!-- test: larql_compute::cpu::ops::moe::cache::tests::q4k_misaligned_length_returns_empty -->
<!-- test: larql_compute::cpu::ops::moe::cache::tests::parallel_hits_do_not_deadlock_or_corrupt -->

#### Scenario: MoE forward respects routing and scaling rules
- **WHEN** `cpu_moe_forward` runs with parameter-free or learned router norm, per-expert scale, router-scale vector, scalar router-input, GELU-tanh activation, identity expert, or zero experts/empty router proj
- **THEN** the result SHALL run without panic, weights SHALL apply per-expert / per-vector / per-scalar, zero-experts and zero-input SHALL produce zero outputs, and identity experts SHALL pass the input through
<!-- test: larql_compute::test_pipeline_and_moe::moe_parameter_free_router_norm_runs_without_panic -->
<!-- test: larql_compute::test_pipeline_and_moe::moe_learned_router_norm_runs_without_panic -->
<!-- test: larql_compute::test_pipeline_and_moe::moe_per_expert_scale_applied -->
<!-- test: larql_compute::test_pipeline_and_moe::moe_router_scale_vector_applied -->
<!-- test: larql_compute::test_pipeline_and_moe::moe_router_input_scalar_nonunit -->
<!-- test: larql_compute::test_pipeline_and_moe::moe_empty_router_proj_returns_zeros -->
<!-- test: larql_compute::test_pipeline_and_moe::moe_zero_num_experts_returns_zeros -->
<!-- test: larql_compute::test_pipeline_and_moe::moe_gelu_tanh_activation_in_forward -->
<!-- test: larql_compute::cpu::ops::moe::tests::test_moe_zero_input_produces_zero -->
<!-- test: larql_compute::cpu::ops::moe::tests::test_moe_identity_expert -->
<!-- test: larql_compute::cpu::ops::moe::tests::cpu_moe_forward_q4k_dispatch -->
<!-- test: larql_compute::cpu::ops::moe::tests::cpu_moe_forward_uses_same_router_input_as_cpu_moe_route -->
<!-- test: larql_compute::cpu::ops::moe::tests::experts_gate_up_indexed_by_expert_id_not_topk_position -->
<!-- test: larql_compute::cpu::ops::moe::tests::per_expert_indexing_routes_correctly -->
<!-- test: larql_compute::cpu::ops::moe::tests::cache_hit_returns_same_arc -->
<!-- test: larql_compute::cpu::ops::moe::tests::cache_eviction_no_panic -->

### Requirement: Cholesky / ridge linear-algebra primitives

The `cpu::ops::linalg` module SHALL provide a Cholesky factorisation
with optional ridge term, a forward/back substitution solver
(`cholesky_solve`), an explicit inverse (`cholesky_inverse`), and a
ridge-decomposition solver (`ridge_decomposition_solve`) with shape
checks. The factorisation MUST return `Err(...)` for non
positive-definite matrices and MUST recover from singular keys when a
non-zero ridge is supplied. Round-trips on realistic shapes MUST be
within numerical tolerance.

#### Scenario: Cholesky factorise / solve / inverse round-trip
- **WHEN** `cholesky` is run on a 2×2 SPD matrix, then `cholesky_solve` and `cholesky_inverse` are exercised against the identity right-hand-side
- **THEN** the factorisation SHALL match a hand-computed L and the inverse SHALL satisfy `A · A⁻¹ = I` within tolerance
<!-- test: larql_compute::cpu::ops::linalg::tests::test_cholesky_2x2 -->
<!-- test: larql_compute::cpu::ops::linalg::tests::test_cholesky_solve_identity -->
<!-- test: larql_compute::cpu::ops::linalg::tests::test_cholesky_inverse -->

#### Scenario: Ridge handles singular keys
- **WHEN** `ridge_decomposition_solve` is given singular keys with a non-zero ridge
- **THEN** the solver SHALL return a finite solution rather than NaN, and SHALL match the round-trip reference on realistic shapes
<!-- test: larql_compute::cpu::ops::linalg::tests::test_cholesky_with_ridge -->
<!-- test: larql_compute::cpu::ops::linalg::tests::test_cholesky_not_positive_definite -->
<!-- test: larql_compute::cpu::ops::linalg::tests::test_ridge_decomposition_round_trip -->
<!-- test: larql_compute::cpu::ops::linalg::tests::test_ridge_decomposition_singular_keys_need_ridge -->
<!-- test: larql_compute::cpu::ops::linalg::tests::test_ridge_decomposition_zero_keys -->
<!-- test: larql_compute::cpu::ops::linalg::tests::test_ridge_decomposition_realistic_shape -->
<!-- test: larql_compute::cpu::ops::linalg::tests::test_ridge_decomposition_shape_mismatch -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_compute::test_correctness::**::* -->
<!-- test: larql_compute::test_q4_x86_correctness::**::* -->
<!-- test: larql_compute::test_q4k_parity::**::* -->
<!-- test: larql_compute::cpu::ops::q4_common::**::* -->
<!-- test: larql_compute::cpu::ops::q4_common::tests::**::* -->
