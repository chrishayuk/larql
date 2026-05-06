## ADDED Requirements

### Requirement: BLAS-fused online-softmax GQA attention

The `larql_inference::attention` module SHALL implement grouped-query
attention (GQA) using a BLAS-fused online-softmax kernel that never
materialises the `[seq, seq]` attention matrix and that scales
memory linearly in sequence length, per ADR-001. The kernel MUST
produce numerics within tolerance of a naive reference for every
supported `(num_q_heads, num_kv_heads, head_dim)` configuration that
shipping families use, MUST honour an arbitrary GQA ratio
(`num_q_heads % num_kv_heads == 0`), and MUST default the attention
scale to `1.0 / sqrt(head_dim)` when no override is supplied.

#### Scenario: Output shape and finiteness
- **WHEN** `gqa_attention` is invoked on synthetic Q/K/V matrices spanning single-token and longer sequences
- **THEN** the output SHALL have `[seq, num_q_heads * head_dim]` shape and every entry SHALL be finite
<!-- test: larql_inference::test_fused_attention::basic::single_token -->
<!-- test: larql_inference::test_fused_attention::basic::output_shape -->
<!-- test: larql_inference::test_fused_attention::basic::longer_sequence -->
<!-- test: larql_inference::attention::gqa::tests::gqa_output_shape -->
<!-- test: larql_inference::attention::gqa::tests::gqa_output_finite -->
<!-- test: larql_inference::attention::gqa::tests::gqa_single_token -->

#### Scenario: Uniform attention averages V across positions
- **WHEN** `gqa_attention` is invoked with Q and K constructed so that softmax weights are uniform
- **THEN** the output SHALL equal the per-position mean of V within tolerance
<!-- test: larql_inference::test_fused_attention::basic::uniform_attention_averages_v -->
<!-- test: larql_inference::test_fused_attention::basic::single_head_small -->

#### Scenario: Multi-head and head_dim variants match a naive reference
- **WHEN** the fused kernel is run against the naive reference at multiple head counts and head_dim ∈ {single, default, large, 512}
- **THEN** every configuration SHALL agree within tolerance
<!-- test: larql_inference::test_fused_attention::reference_agreement::multi_head -->
<!-- test: larql_inference::test_fused_attention::edge_cases::single_token_single_dim -->
<!-- test: larql_inference::test_fused_attention::edge_cases::large_head_dim -->
<!-- test: larql_inference::test_fused_attention::edge_cases::large_head_dim_512 -->
<!-- test: larql_inference::test_fused_attention::edge_cases::custom_scale -->

### Requirement: Causal mask and attention softcap

The kernel SHALL apply a strict causal mask (query position `i` may
attend only to key positions `j ≤ i`) and SHALL apply a softcap
`tanh(scores / cap) * cap` when a positive cap is supplied,
matching Gemma 2 attention softcap semantics.

#### Scenario: Causal mask blocks future positions
- **WHEN** the kernel is run on a two-token input and `gqa_attention_with_weights` is queried
- **THEN** position 0 SHALL only attend to position 0, position 1 SHALL attend to both positions, captured weights SHALL sum to one per row, and weights SHALL be lower-triangular
<!-- test: larql_inference::test_fused_attention::basic::causal_mask_two_tokens -->
<!-- test: larql_inference::test_fused_attention::capture::captured_weights_sum_to_one -->
<!-- test: larql_inference::test_fused_attention::capture::captured_weights_causal -->
<!-- test: larql_inference::attention::gqa::tests::gqa_causal_last_token_attends_all -->
<!-- test: larql_inference::test_modules::test_attention::gqa_attention_causal_mask -->
<!-- test: larql_inference::test_modules::test_attention::gqa_attention_single_token -->

#### Scenario: Softcap is applied when configured
- **WHEN** `gqa_attention` is called with a positive softcap
- **THEN** the kernel SHALL apply the `tanh`-based cap and the result SHALL agree with a reference that applies the cap to the unfused naive path
<!-- test: larql_inference::test_fused_attention::basic::with_softcap -->

### Requirement: Grouped-query attention (GQA) ratios

The kernel SHALL accept any `num_q_heads / num_kv_heads` ratio that
satisfies `num_q_heads % num_kv_heads == 0`, including the 2x ratio
and the Gemma 3 dimensions used in production. Each KV head SHALL be
shared across exactly `num_q_heads / num_kv_heads` Q heads.

#### Scenario: 2x GQA ratio matches reference
- **WHEN** the kernel is run with `num_q_heads = 2 * num_kv_heads`
- **THEN** the output SHALL agree with a naive reference, the inline GQA tests SHALL exercise the 2x ratio path, and head pairs SHALL share KV
<!-- test: larql_inference::test_fused_attention::reference_agreement::gqa_2x_ratio -->
<!-- test: larql_inference::attention::gqa::tests::gqa_reps_2_output_shape -->
<!-- test: larql_inference::attention::gqa::tests::gqa_reps_2_output_is_finite -->
<!-- test: larql_inference::attention::gqa::tests::gqa_reps_2_head_pairs_share_kv -->

#### Scenario: Gemma 3 GQA dimensions are exercised end-to-end
- **WHEN** the kernel is run with the exact `(num_q_heads, num_kv_heads, head_dim)` Gemma 3 ships
- **THEN** the output SHALL match the naive reference within tolerance
<!-- test: larql_inference::test_fused_attention::reference_agreement::gqa_gemma3_dimensions -->
<!-- test: larql_inference::attention::gqa::tests::gqa_with_weights_captures_softmax -->

### Requirement: Rotary position embeddings (full and partial)

The crate SHALL apply RoPE to Q/K with support for full rotation
(rotary_dim == head_dim), partial rotation (rotary_dim < head_dim,
applied to the leading dimensions), and per-layer base values
(Gemma 3/4 alternate two RoPE bases). The function MUST preserve
shape, MUST be a no-op at position zero when reduced to identity, and
MUST produce different outputs at different positions.

#### Scenario: RoPE preserves shape and head norm
- **WHEN** `apply_rope` is invoked with a positive position
- **THEN** the output SHALL preserve `[seq, num_heads * head_dim]` shape, every entry SHALL be finite, and the per-head L2 norm SHALL be preserved within tolerance
<!-- test: larql_inference::attention::rope::tests::apply_rope_preserves_shape -->
<!-- test: larql_inference::attention::rope::tests::apply_rope_output_is_finite -->
<!-- test: larql_inference::attention::rope::tests::apply_rope_preserves_norm_per_head -->
<!-- test: larql_inference::test_modules::test_attention::apply_rope_preserves_shape -->
<!-- test: larql_inference::test_modules::test_attention::apply_rope_position_zero_is_identity -->
<!-- test: larql_inference::test_modules::test_attention::apply_rope_different_positions_differ -->

#### Scenario: Partial RoPE rotates only the leading rotary_dim dimensions
- **WHEN** `apply_rope_partial` is invoked with `fraction < 1.0`
- **THEN** the rotated dimensions SHALL change, the non-rotated dimensions SHALL be preserved bit-for-bit, fraction == 1.0 SHALL match the full RoPE path, and fraction == 0.0 SHALL be a passthrough
<!-- test: larql_inference::test_fused_attention::rope_tests::partial_rope_fraction_1_matches_full -->
<!-- test: larql_inference::test_fused_attention::rope_tests::partial_rope_preserves_non_rotated_dims -->
<!-- test: larql_inference::test_fused_attention::rope_tests::partial_rope_rotates_correct_dims -->
<!-- test: larql_inference::test_fused_attention::rope_tests::partial_rope_multi_head -->
<!-- test: larql_inference::attention::rope::tests::apply_rope_partial_at_offset -->
<!-- test: larql_inference::attention::rope::tests::apply_rope_partial_fraction_zero_is_passthrough -->
<!-- test: larql_inference::attention::rope::tests::rope_partial_fraction_one_equals_full_rope -->
<!-- test: larql_inference::attention::rope::tests::rope_partial_fraction_between_0_and_1_is_finite -->

#### Scenario: Position offset and base value affect the rotation
- **WHEN** `apply_rope` is invoked at sequential positions and with different RoPE bases
- **THEN** the rotated output SHALL differ between bases, and applying RoPE with a position offset SHALL match a sequential-position reference
<!-- test: larql_inference::attention::rope::tests::apply_rope_different_positions_differ -->
<!-- test: larql_inference::attention::rope::tests::rope_different_base_produces_different_output -->
<!-- test: larql_inference::attention::rope::tests::rope_position_offset_matches_sequential_positions -->

### Requirement: Attention weight capture (no output drift)

`gqa_attention_with_weights` SHALL return the per-head softmax
weights in addition to the attention output without changing the
output. Capture MUST be opt-in: callers that pass `None` SHALL NOT
receive any captured matrices.

#### Scenario: Capture matches the reference and does not change output
- **WHEN** `gqa_attention_with_weights` is invoked alongside `gqa_attention` on the same inputs
- **THEN** the captured weights SHALL be returned, weights SHALL be lower-triangular and row-sum to one, and the attention output SHALL be bit-equivalent to the no-capture path
<!-- test: larql_inference::test_fused_attention::capture::capture_returns_weights -->
<!-- test: larql_inference::test_fused_attention::capture::no_capture_returns_none -->
<!-- test: larql_inference::test_fused_attention::capture::capture_does_not_change_output -->

### Requirement: KV cache surgery and decode

The `attention::decode` module SHALL provide a sliding/full KV cache
plus surgery primitives — `get_layer`, `set_layer`, `clear_layer`,
`clone_layer_from`, `clone_layer_position_range` — so that callers
can extract a layer's KV, splice in a donor layer's KV, or copy a
position range across runs without rebuilding the cache. Out-of-range
operations MUST be no-ops, sliding-window caches MUST clip at the
configured window, and donor copies MUST clamp to the donor's actual
length.

#### Scenario: KV cache lifecycle round-trips and clips correctly
- **WHEN** a fresh `KvCache` is constructed and a layer KV is set / fetched / cleared
- **THEN** the cache SHALL start empty, `set_layer` then `get_layer` SHALL round-trip, out-of-range writes SHALL be no-ops, and a sliding-window cache SHALL clip at the window
<!-- test: larql_inference::attention::decode::tests::kv_cache_starts_empty -->
<!-- test: larql_inference::attention::decode::tests::kv_cache_with_window_clips -->
<!-- test: larql_inference::attention::decode::tests::get_layer_returns_none_when_empty -->
<!-- test: larql_inference::attention::decode::tests::set_layer_then_get_layer_round_trips -->
<!-- test: larql_inference::attention::decode::tests::set_layer_out_of_range_is_noop -->
<!-- test: larql_inference::attention::decode::tests::clear_layer_removes_kv -->

#### Scenario: clone_layer_from and clone_layer_position_range copy donor KV
- **WHEN** `clone_layer_from` is invoked with a valid donor layer, a missing donor layer, or a position range
- **THEN** matching donors SHALL be copied, missing donors SHALL be no-ops, position ranges SHALL slice the donor, the slice SHALL clamp to the donor's length, and an empty slice SHALL be a no-op
<!-- test: larql_inference::attention::decode::tests::clone_layer_from_copies_donor_kv -->
<!-- test: larql_inference::attention::decode::tests::clone_layer_from_missing_donor_layer_is_noop -->
<!-- test: larql_inference::attention::decode::tests::clone_layer_position_range_slices_donor -->
<!-- test: larql_inference::attention::decode::tests::clone_layer_position_range_clamps_to_donor_length -->
<!-- test: larql_inference::attention::decode::tests::clone_layer_position_range_empty_slice_is_noop -->

#### Scenario: decode_step grows KV and produces finite output
- **WHEN** `decode_step` is invoked across all layers with a non-empty prior cache
- **THEN** the output SHALL preserve `[1, hidden]` shape, every entry SHALL be finite, and the KV cache SHALL grow with each step on every layer
<!-- test: larql_inference::attention::decode::tests::decode_step_output_shape -->
<!-- test: larql_inference::attention::decode::tests::decode_step_output_finite -->
<!-- test: larql_inference::attention::decode::tests::decode_step_kv_grows_with_prior -->
<!-- test: larql_inference::attention::decode::tests::decode_step_all_layers_succeed -->

### Requirement: CPU / Metal numeric parity and V-projection

The attention block SHALL produce identical outputs within numeric
tolerance on CPU and Metal backends for every architecture that has
both backends, on a prefill workload. For Gemma 4 global layers
specifically, the on-disk V tensor SHALL load as Q6K and the CPU Q4K
loader SHALL produce distinct W_K and W_V matrices (no shared
projection).

#### Scenario: CPU and Metal prefill agree across families
- **WHEN** the parity harness runs prefill on Gemma 3 4B, Gemma 4 31B dense, Llama 2 7B, and Mistral 7B
- **THEN** every (vindex, backend pair) SHALL match within tolerance, and missing weights SHALL be skipped (not failed) when not in strict mode
<!-- test: larql_inference::test_cpu_metal_parity::parity_gemma3_4b_prefill -->
<!-- test: larql_inference::test_cpu_metal_parity::parity_gemma4_31b_dense_prefill -->
<!-- test: larql_inference::test_cpu_metal_parity::parity_llama2_7b_prefill -->
<!-- test: larql_inference::test_cpu_metal_parity::parity_mistral_7b_prefill -->

#### Scenario: Gemma 4 global layers store V as Q6K and load distinct W_V
- **WHEN** a Gemma 4 dense vindex is loaded
- **THEN** the V tensor SHALL be reported as Q6K-quantized for global layers, and the CPU Q4K loader SHALL produce W_K and W_V matrices that are not bit-identical
<!-- test: larql_inference::test_cpu_v_projection::vindex_stores_v_as_q6k_for_gemma4_global_layers -->
<!-- test: larql_inference::test_cpu_v_projection::cpu_q4k_load_produces_distinct_w_k_and_w_v_for_gemma4_global -->

#### Scenario: Attention block constructs and runs across all layers
- **WHEN** the high-level attention block is run on every layer of a synthetic model and asked to return KV
- **THEN** the output SHALL preserve `[seq, hidden]` shape, every entry SHALL be finite, the single-token path SHALL succeed, and KV-out SHALL be returned when requested
<!-- test: larql_inference::attention::block::tests::attention_block_output_shape -->
<!-- test: larql_inference::attention::block::tests::attention_block_output_finite -->
<!-- test: larql_inference::attention::block::tests::attention_block_single_token -->
<!-- test: larql_inference::attention::block::tests::attention_block_all_layers -->
<!-- test: larql_inference::attention::block::tests::attention_block_with_kv_out_returns_kv -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_inference::test_fused_attention::**::* -->
<!-- test: larql_inference::test_cpu_metal_parity::**::* -->
<!-- test: larql_inference::test_cpu_v_projection::**::* -->
<!-- test: larql_inference::attention::gqa::tests::**::* -->
