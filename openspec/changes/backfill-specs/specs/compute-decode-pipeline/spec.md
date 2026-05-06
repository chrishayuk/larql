## ADDED Requirements

### Requirement: FullPipelineLayer carries all per-layer architecture parameters

`larql_compute::FullPipelineLayer<'a>` SHALL be the single per-layer
struct the decode pipeline consumes. It MUST carry quantised attention
and FFN weights (Q/K/V/O, gate, up, down) plus the f32 norm vectors
(input, post-attn, optional pre-FFN, optional post-FFN) and per-layer
architecture parameters (norm offset, QK-norm offset, eps, norm type,
FFN type, activation, attention scale, head dim, num Q/KV heads, RoPE
base and rotary dim, sliding window, V-norm flag, layer scalar, QK
norm weights, optional FFN biases, optional `MoeLayerWeights`,
remote-FFN flag, MoE combined-output norm flag, and outer post-norm).
The struct MUST implement `Default` so that tests can construct
minimal instances via `..Default::default()` accepting locally
borrowed slices, and SHALL provide `is_gated()` and `is_hybrid_moe()`
helpers that mirror the underlying enums / Option fields.

#### Scenario: Default spread accepts stack-local borrowed slices
- **WHEN** `FullPipelineLayer { input_norm: &local_norms, wq: QuantWeight { data: &local_data, ..Default::default() }, …, ..Default::default() }` is constructed in a test
- **THEN** the construction SHALL compile, defaulted fields (`eps`, `norm_type`) SHALL carry through, and the layer SHALL be usable with locally borrowed slices
<!-- test: larql_compute::pipeline::tests::default_layer_accepts_local_borrows_via_spread -->

#### Scenario: is_gated and is_hybrid_moe reflect the underlying fields
- **WHEN** layers are constructed with `FfnType::Gated` vs `FfnType::Standard` and with `moe: Some(...)` vs `moe: None`
- **THEN** `is_gated()` SHALL return true only for `Gated` and `is_hybrid_moe()` SHALL return true only when `moe.is_some()`
<!-- test: larql_compute::pipeline::tests::is_gated_matches_ffn_type -->
<!-- test: larql_compute::pipeline::tests::is_hybrid_moe_reflects_option -->

#### Scenario: Activation conversion from boolean disambiguates SiLU vs GELU-tanh
- **WHEN** `Activation::from(bool)` is invoked
- **THEN** `true` SHALL produce `Activation::GeluTanh` and `false` SHALL produce `Activation::Silu`
<!-- test: larql_compute::pipeline::tests::activation_from_bool -->

### Requirement: QuantFormat taxonomy and packed-matrix sizing

The `QuantFormat` enum SHALL classify weight formats into three
families: Q4_K-family (Q4_K, Q4_KF, Q6_K — 256-element super-blocks),
legacy block-32 Q8 (Q4_0, Q8_0), and float-input formats (BF16, F16,
F32). The `is_q4k_family`, `is_legacy_q8`, and `is_q4kf` classifiers
MUST partition the formats unambiguously, and the
`packed_matrix_bytes(rows, cols)` helper MUST return the on-disk byte
count for each quantised format and `None` for float formats.
Equality MUST distinguish all variants.

#### Scenario: Format classifiers partition the variants
- **WHEN** `is_q4k_family`, `is_legacy_q8`, and `is_q4kf` are called for every `QuantFormat` variant
- **THEN** Q4_K-family classifiers SHALL include `Q4_K`, `Q4_KF`, `Q6_K`; legacy-Q8 SHALL include `Q4_0`, `Q8_0`; float formats SHALL be in neither family; and `is_q4kf` SHALL be true only for `Q4_KF`
<!-- test: larql_compute::pipeline::tests::quant_format_classifiers -->
<!-- test: larql_compute::pipeline::tests::quant_format_equality -->

#### Scenario: packed_matrix_bytes returns documented sizes
- **WHEN** `packed_matrix_bytes(2, 32)` is called for Q4_0, `packed_matrix_bytes(2, 256)` for Q4_K / Q4_KF / Q6_K, and `packed_matrix_bytes(2, 256)` for F16
- **THEN** results SHALL be `Some(36)`, `Some(288)`, `Some(320)`, `Some(420)`, and `None` respectively
<!-- test: larql_compute::pipeline::tests::quant_format_reports_packed_matrix_bytes -->

### Requirement: Single-token decode runs all layers in one command buffer

`DecodeBackend::decode_token` (Metal-only impl) SHALL drive a full
forward pass for one token through every layer using a single Metal
command buffer with a single global encoder, executing the documented
nine-stage layer recipe (fused input-norm + QKV proj; fused QK-norm +
RoPE; batched V-norm where required; fused KV append + attend; O
projection; fused post-attn norm + residual + FFN-norm + h_post_attn
store; fused FFN gate + up; fused GEGLU + down; fused post-FFN norm +
residual add). The decode pass MUST NOT round-trip residuals to CPU
between layers, MUST update the KV cache per layer, and MUST hit the
~306-dispatch/token target for Gemma 3 4B (34 layers × 9 dispatches +
final dispatches). Backends without GPU decode MUST fall back to
returning `None` from `decode_token`.

#### Scenario: KV cache helpers are no-ops on backends without decode
- **WHEN** `populate_kv_layer`, `reset_kv_cache`, `truncate_kv_cache`, `preallocate_kv_cache_per_layer`, `kv_cache_len`, and `decode_token` are invoked on a backend that does not implement decode (e.g. `CpuBackend`)
- **THEN** the side-effecting methods SHALL return without panic, `kv_cache_len` SHALL return 0, and the fallible methods SHALL return `None`
<!-- test: larql_compute::test_backend_matmul_quant::default_decode_stubs -->

#### Scenario: Decode profile timings sum and format correctly
- **WHEN** `DecodeProfile` accumulates per-stage timing buckets, formats a zero-total summary, and formats a per-layer-average summary
- **THEN** `total_ms()` SHALL equal the sum of bucket durations, the zero-total summary SHALL render without divide-by-zero, and the formatted summary SHALL include the per-layer average
<!-- test: larql_compute::metal::decode::profile::tests::total_ms_sums_buckets -->
<!-- test: larql_compute::metal::decode::profile::tests::format_summary_handles_zero_total -->
<!-- test: larql_compute::metal::decode::profile::tests::format_summary_includes_per_layer_average -->

### Requirement: Multi-position prefill populates KV cache atomically

`DecodeBackend::prefill_q4` (Metal-only impl) SHALL accept a
multi-position input `[seq_len * hidden]`, run all layers across all
positions in one submission, store post-RoPE K/V values at the correct
slots in the KV cache, and return the per-position hidden states
`[seq_len * hidden]`. Prefill MUST be invariant when the layer set
includes no MoE, MUST honour MoE layers (mixing dense FFN and expert
block weighted-sum), and MUST hand off cleanly to a single-token
`decode_token` continuation. Prefill output shape MUST be exactly
`seq_len * hidden`.

#### Scenario: prefill_q4 with one MoE layer returns the correct shape
- **WHEN** `prefill_q4` runs with a layer set that includes one MoE layer
- **THEN** the output length SHALL equal `seq_len * hidden` and the MoE block SHALL contribute via its expert weighted-sum
<!-- test: larql_compute::test_pipeline_and_moe::prefill_q4_with_moe_returns_correct_shape -->

#### Scenario: prefill_q4 with all-MoE layers returns the correct shape
- **WHEN** `prefill_q4` runs with every layer being MoE
- **THEN** the output length SHALL equal `seq_len * hidden` and every layer's MoE block SHALL be exercised
<!-- test: larql_compute::test_pipeline_and_moe::prefill_q4_all_moe_layers_returns_correct_shape -->

#### Scenario: prefill_q4 with no MoE layers is unaffected by the MoE plumbing
- **WHEN** `prefill_q4` runs with a layer set that has no MoE blocks
- **THEN** the result SHALL match the dense-only reference and the MoE plumbing SHALL not perturb the output
<!-- test: larql_compute::test_pipeline_and_moe::prefill_q4_no_moe_unaffected -->

### Requirement: MoeLayerWeights routing and combination semantics

`MoeLayerWeights<'a>` SHALL carry the per-layer hybrid-MoE block
parameters consumed by the decode and prefill paths: per-expert
gate+up and down byte slices, expert data format, router projection
(weights, learned per-token scale vector, per-expert output scale,
optional learned router-norm with the parameter-free fallback flag,
post-norm scalar multiplier), expert pre-norm and post-norm vectors
for both dense FFN1 and the expert block, expert count, top-K, expert
intermediate size, and the activation. `decode_token_with_moe` /
`decode_token_with_moe_split` SHALL route the layer's `h_post_attn`
through the supplied `moe_fn` (or fire/collect pair) for each MoE
layer; the split-fire/collect default impl MUST synthesise a single
synchronous closure from the pair. The MoE forward path MUST honour
the documented router-input-scalar / per-expert scale / router-scale
vector / parameter-free vs learned router-norm / SiLU vs GELU-tanh
activation rules. Empty router proj or `num_experts == 0` MUST yield a
zero contribution.

#### Scenario: MoE forward respects routing flags without panic
- **WHEN** the MoE forward path is exercised with parameter-free or learned router norm, per-expert scale, router-scale vector, scalar router-input, GELU-tanh activation, empty router proj, and zero experts
- **THEN** every configuration SHALL run without panic, the scaling factors SHALL apply, the empty router and zero-experts cases SHALL return zeros, and the GELU-tanh activation path SHALL be reached
<!-- test: larql_compute::test_pipeline_and_moe::moe_parameter_free_router_norm_runs_without_panic -->
<!-- test: larql_compute::test_pipeline_and_moe::moe_learned_router_norm_runs_without_panic -->
<!-- test: larql_compute::test_pipeline_and_moe::moe_per_expert_scale_applied -->
<!-- test: larql_compute::test_pipeline_and_moe::moe_router_scale_vector_applied -->
<!-- test: larql_compute::test_pipeline_and_moe::moe_router_input_scalar_nonunit -->
<!-- test: larql_compute::test_pipeline_and_moe::moe_empty_router_proj_returns_zeros -->
<!-- test: larql_compute::test_pipeline_and_moe::moe_zero_num_experts_returns_zeros -->
<!-- test: larql_compute::test_pipeline_and_moe::moe_gelu_tanh_activation_in_forward -->

#### Scenario: Split fire/collect default delegates to combined moe_fn
- **WHEN** a backend that does not override `decode_token_with_moe_split` is invoked
- **THEN** the default impl SHALL synthesise one synchronous closure from `moe_fire_fn` + `moe_collect_fn` and forward to `decode_token_with_moe`
<!-- test: unbacked -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_compute::test_pipeline_and_moe::**::* -->
<!-- test: larql_compute::test_backend_matmul_quant::**::* -->
