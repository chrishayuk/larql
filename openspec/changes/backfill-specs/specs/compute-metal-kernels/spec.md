## ADDED Requirements

### Requirement: f32 tiled matmul and gemv on Metal

The Metal backend SHALL provide tiled f32 sgemm (32×32 work-group)
plus row-per-simdgroup f32_gemv specialisations. The sgemm
dispatchers (`sgemm`, `sgemm_transb`) MUST match the CPU `ndarray::dot`
reference within Metal precision (≤ 1e-4 absolute / ≤ 1e-5 relative
for non-degenerate inputs), MUST handle small / non-square shapes, and
MUST accept transposed B. The `f32_gemv` specialisation MUST match
`ndarray::dot` at vocab-scale (lm-head shape) and MUST allow
`f16_gemv_topk1` / `f16_gemv_topk` to share a kernel with capped K.

#### Scenario: f32 sgemm matches CPU ndarray reference
- **WHEN** `sgemm` and `sgemm_transb` run on synthetic matrices including small shapes
- **THEN** the result SHALL agree with `ndarray::dot` within Metal precision
<!-- test: larql_compute::test_metal_shaders::sgemm_matches_cpu -->
<!-- test: larql_compute::test_metal_shaders::sgemm_transb_matches_cpu -->
<!-- test: larql_compute::test_metal_shaders::sgemm_transb_small_matrix -->

#### Scenario: f32_gemv matches ndarray dot at vocab scale
- **WHEN** `f32_gemv` is dispatched on a vocab-scale weight matrix
- **THEN** the GPU result SHALL match the CPU `ndarray::dot` reference within tolerance
<!-- test: larql_compute::test_kernel_lm_head_gemv::f32_gemv_cpu_vs_metal_at_vocab_scale -->
<!-- test: larql_compute::test_kernel_vindex_integration::f32_gemv_matches_ndarray_dot -->

#### Scenario: f16 gemv top-K and top-1 match CPU references with capacity bounds
- **WHEN** `f16_gemv_topk1`, `f16_gemv_topk`, and the partial-tg helper are called for top-K up to and just past the kernel capacity (`K_TOPK`)
- **THEN** the top-K results SHALL match a full CPU argmax / sorted-top-K, requests beyond capacity SHALL return `None`, partial last threadgroups SHALL be handled, and the capacity ceiling SHALL be enforced
<!-- test: larql_compute::test_metal_shaders::f16_gemv_topk1_matches_full_argmax -->
<!-- test: larql_compute::test_metal_shaders::f16_gemv_topk_matches_cpu_topk -->
<!-- test: larql_compute::test_metal_shaders::topk_capacity_edges_return_none -->
<!-- test: larql_compute::metal::trait_impl::matmul::tests::topk_partial_handles_partial_last_tg -->
<!-- test: larql_compute::metal::trait_impl::matmul::tests::topk_capacity_ceiling_enforced -->
<!-- test: larql_compute::test_kernel_vindex_integration::f16_gemv_matches_f32_gemv_argmax -->

### Requirement: Simdgroup Q4 / Q4_K / Q6_K matvec kernels match CPU

Metal simdgroup quantised matvec shaders SHALL match the CPU
dequantise-and-dot reference within quantisation noise on production
shapes. The covered shaders are `q4_matvec_v4`, `q4k_matvec`,
`q4k_matvec_8sg`, `q4k_matvec_stride32`, `q6k_matvec`,
`q6k_matvec_8sg`, `q4_vecmat`, and `q4_sparse_matvec`. The
8-simdgroup variants MUST be bit-equal to the 4-simdgroup baseline
for both Q4_K and Q6_K. Q4 matvec + GPU top-K / top-1 paths MUST
agree with full argmax. Q4_K matvec MUST correctly handle small N,
misaligned N, and zero input.

#### Scenario: Q4 / Q4_K / Q6_K matvec match CPU references on real shapes
- **WHEN** Metal `q4_matvec`, `q4k_matvec`, and `q6k_matvec` run against the CPU dequantise-and-dot reference for small, multi-superblock, and production shapes
- **THEN** outputs SHALL agree within quant noise and SHALL be non-zero for non-zero inputs
<!-- test: larql_compute::test_metal_shaders::q4_matvec_matches_cpu -->
<!-- test: larql_compute::test_metal_shaders::q4_matvec_small_matrix -->
<!-- test: larql_compute::test_metal_shaders::q4_matvec_zero_input -->
<!-- test: larql_compute::test_metal_shaders::q4k_matvec_matches_cpu -->
<!-- test: larql_compute::test_metal_shaders::q4k_matvec_produces_nonzero -->
<!-- test: larql_compute::test_metal_shaders::q4k_quantize_then_matvec_matches_f32 -->
<!-- test: larql_compute::test_metal_shaders::q4k_single_superblock_matches_dequantize_reference -->
<!-- test: larql_compute::test_metal_shaders::q4k_multi_row_matches_dequantize_reference -->
<!-- test: larql_compute::test_metal_shaders::q6k_matvec_matches_cpu -->
<!-- test: larql_compute::test_metal_shaders::q6k_matvec_produces_nonzero -->
<!-- test: larql_compute::test_metal_shaders::q6k_single_superblock_matches_dequantize_reference -->
<!-- test: larql_compute::test_metal_shaders::q6k_multi_superblock_matches_dequantize_reference -->
<!-- test: larql_compute::test_metal_shaders::q6k_subnormal_d_matches_cpu -->

#### Scenario: 8-simdgroup variants are bit-equal to 4-simdgroup baselines
- **WHEN** `q4k_matvec_8sg` and `q6k_matvec_8sg` are run side-by-side with the 4-simdgroup `q4k_matvec` / `q6k_matvec` baselines and the stride-32 lane-pattern variant
- **THEN** the outputs SHALL be bit-equal across all configurations
<!-- test: larql_compute::test_kernel_q4k_matvec_8sg::q4k_matvec_8sg_matches_4sg_bit_equal -->
<!-- test: larql_compute::test_kernel_q4k_matvec_8sg::q4k_matvec_stride32_matches_cpu -->
<!-- test: larql_compute::test_kernel_q6k_matvec_8sg::q6k_matvec_8sg_matches_4sg_bit_equal -->
<!-- test: larql_compute::test_kernel_q6k_matvec_8sg::q6k_matvec_8sg_perf_vs_4sg -->

#### Scenario: Q4 matvec top-K and top-1 match CPU argmax
- **WHEN** `q4_matvec_topk` and `q4_matvec_topk1` are dispatched on production-scale weights and compared to a full CPU `q4_matvec` + argmax / sort
- **THEN** Metal SHALL return the same top-K indices and scores
<!-- test: larql_compute::test_metal_shaders::q4_matvec_topk_matches_cpu_topk -->
<!-- test: larql_compute::test_metal_shaders::q4_matvec_topk1_matches_full_argmax -->
<!-- test: larql_compute::test_kernel_lm_head_gemv::q4_matvec_cpu_vs_metal_at_vocab_scale -->
<!-- test: larql_compute::test_kernel_lm_head_gemv::q4_matvec_metal_writes_every_row_small_n -->
<!-- test: larql_compute::test_kernel_lm_head_gemv::q4_matvec_metal_writes_every_row_misaligned_n -->
<!-- test: larql_compute::test_kernel_lm_head_gemv::q4_matvec_dispatch_geometry_matches_v4_kernel -->
<!-- test: larql_compute::test_kernel_lm_head_gemv::q4_matvec_pipeline_max_threads_per_tg -->
<!-- test: larql_compute::test_kernel_lm_head_gemv::q4_matvec_cutoff_sweep -->

#### Scenario: Q4 vecmat / sparse / pair / multi-layer paths match references
- **WHEN** the `q4_vecmat`, `q4_sparse_matvec`, `q4_pair_batch`, `q4_f32_matvec`, `q8_matvec`, and `multi_layer_q4_ffn` shaders are dispatched
- **THEN** each SHALL match the CPU dense reference, the dense matvec for sparse, and the per-layer manual reference for the multi-layer path
<!-- test: larql_compute::test_metal_shaders::q4_vecmat_matches_cpu -->
<!-- test: larql_compute::test_metal_shaders::q4_pair_batch_matches_individual -->
<!-- test: larql_compute::test_metal_shaders::q4_f32_matvec_nonzero -->
<!-- test: larql_compute::test_metal_shaders::sparse_matvec_matches_dense -->
<!-- test: larql_compute::test_metal_shaders::q8_matvec_metal_nonzero -->
<!-- test: larql_compute::test_metal_shaders::q8_matvec_metal_matches_cpu_reference -->
<!-- test: larql_compute::test_metal_shaders::multi_layer_q4_produces_output -->
<!-- test: larql_compute::test_metal_shaders::multi_position_q4k_matches_individual -->

### Requirement: Q4_K matmul and FFN gate+up fused kernels

The Metal Q4_K matmul (`q4k_matmul`) SHALL amortise dequantisation
across `seq_len > 1` and MUST match a stacked-matvec reference for
basic, decode-shape (seq_len = 1), seq_len-not-multiple-of-cols-per-tg,
num-rows-not-multiple-of-rows-per-tg, and production O-projection
shapes. The fused FFN gate+up kernels (`q4k_ffn_gate_up`,
`q4k_ffn_gate_up_8sg`, `q4k_ffn_gate_up_f16acc`) MUST cover the
documented shapes (Gemma 3 4B; Gemma 4 26B-A4B MoE; Gemma 4 31B
dense; max-K boundary 4096; just past max-K 4352; smoke 256×64; zero
input). The 8-simdgroup variant MUST be bit-equal to the 4-simdgroup
baseline; the f16-accum variant MUST stay within documented tolerance
of the f32-accum baseline.

#### Scenario: Q4_K matmul matches stacked matvec across shapes
- **WHEN** `q4k_matmul` is invoked across basic, decode (seq_len=1), non-multiple-of-tg-size, and 4B O-projection shapes
- **THEN** the result SHALL equal stacking individual `q4k_matvec` calls and SHALL outperform stacked matvec on prefill shapes
<!-- test: larql_compute::test_kernel_q4k_matmul::q4k_matmul_matches_stacked_matvec_basic -->
<!-- test: larql_compute::test_kernel_q4k_matmul::q4k_matmul_matches_stacked_matvec_seq_len_1_decode_shape -->
<!-- test: larql_compute::test_kernel_q4k_matmul::q4k_matmul_handles_seq_len_not_multiple_of_cols_per_tg -->
<!-- test: larql_compute::test_kernel_q4k_matmul::q4k_matmul_handles_num_rows_not_multiple_of_rows_per_tg -->
<!-- test: larql_compute::test_kernel_q4k_matmul::q4k_matmul_production_shape_4b_o_proj -->
<!-- test: larql_compute::test_kernel_q4k_matmul_perf::q4k_matmul_faster_than_stacked_matvec_on_prefill_shape -->

#### Scenario: FFN gate+up covers production and boundary shapes
- **WHEN** `q4k_ffn_gate_up` runs at smoke, Gemma 3 4B, Gemma 4 26B-A4B MoE, Gemma 4 31B dense, max-K boundary 4096, and just-past-max-K 4352, and zero-input
- **THEN** every shape SHALL match the dequantise-and-multiply reference within Q4_K noise
<!-- test: larql_compute::test_kernel_q4k_ffn_gate_up::q4k_ffn_gate_up_smoke_256x64 -->
<!-- test: larql_compute::test_kernel_q4k_ffn_gate_up::q4k_ffn_gate_up_gemma3_4b -->
<!-- test: larql_compute::test_kernel_q4k_ffn_gate_up::q4k_ffn_gate_up_gemma4_26b_a4b_moe_shape -->
<!-- test: larql_compute::test_kernel_q4k_ffn_gate_up::q4k_ffn_gate_up_max_k_boundary_4096 -->
<!-- test: larql_compute::test_kernel_q4k_ffn_gate_up::q4k_ffn_gate_up_just_past_max_k_4352 -->
<!-- test: larql_compute::test_kernel_q4k_ffn_gate_up::q4k_ffn_gate_up_gemma4_31b_dense -->
<!-- test: larql_compute::test_kernel_q4k_ffn_gate_up::q4k_ffn_gate_up_zero_input -->

#### Scenario: FFN gate+up 8sg / f16-accum variants stay within tolerance
- **WHEN** `q4k_ffn_gate_up_8sg` and `q4k_ffn_gate_up_f16acc` are compared to the 4-simdgroup f32-accum baseline
- **THEN** the 8sg variant SHALL be bit-equal and the f16-accum variant SHALL stay within documented tolerance
<!-- test: larql_compute::test_kernel_q4k_ffn_gate_up_8sg::q4k_ffn_gate_up_8sg_matches_4sg_bit_equal -->
<!-- test: larql_compute::test_kernel_q4k_ffn_gate_up_8sg::q4k_ffn_gate_up_8sg_perf_vs_4sg -->
<!-- test: larql_compute::test_kernel_q4k_ffn_gate_up_f16acc::q4k_ffn_gate_up_f16acc_matches_f32_within_tolerance -->
<!-- test: larql_compute::test_kernel_q4k_ffn_gate_up_f16acc::q4k_ffn_gate_up_f16acc_perf_vs_f32 -->

### Requirement: Fused QKV projection, QK-norm + RoPE, and V-norm

The Metal backend SHALL provide fused QKV projection
(`q4k_qkv_proj`, `q4k_qkv_proj_v2`, `q4kf_qkv_proj`,
`q4k_q6k_qkv_proj`, `q4k_q6k_qkv_proj_normed`), QK-norm in standalone
and `qk` modes, fused QK-norm + RoPE (`qk_norm_rope_fused`), batched
V-norm, and standalone RoPE / partial RoPE shaders. The fused
QKV+norm path MUST match a separate `rms_norm` + `qkv_proj` reference,
including at production hidden sizes. QK-norm MUST honour the
per-family weight offset (Gemma 3 → 1.0; Gemma 4 → 0.0), MUST be
shape-equivalent to `v_norm` when run in v-mode, MUST match a CPU
reference, MUST behave identically in-place vs separate buffers, and
MUST cover smoke + production shapes (Gemma 3 4B; Gemma 4 global
offset0). RoPE MUST cover Llama 2 full, Gemma 3 full-256, Gemma 4
sliding, Gemma 4 global partial-rotary-fraction, partial-rotary
pass-through, and the batched per-head variants.

#### Scenario: Fused QKV projection matches separate norm + proj
- **WHEN** `q4k_q6k_qkv_proj_normed` and the v2 / `q4k_qkv_proj_matches_per_proj_dispatch` paths are dispatched
- **THEN** outputs SHALL match `rms_norm` then `qkv_proj` in separate kernels at smoke and production hidden sizes
<!-- test: larql_compute::test_kernel_new_fused_kernels::q4k_q6k_qkv_proj_normed_matches_separate_norm_and_proj -->
<!-- test: larql_compute::test_kernel_new_fused_kernels::q4k_q6k_qkv_proj_normed_matches_at_production_hidden -->
<!-- test: larql_compute::test_metal_shaders::q4kf_proj_matches_cpu_reference -->
<!-- test: larql_compute::test_metal_shaders::q4kf_proj_matches_cpu_reference_gemma3_shape -->
<!-- test: larql_compute::test_metal_shaders::q4kf_qkv_proj_matches_individual_projections -->
<!-- test: larql_compute::test_kernel_vindex_integration::q4kf_proj_matches_cpu_on_real_vindex_bytes -->
<!-- test: larql_compute::test_kernel_vindex_integration::q4k_qkv_proj_matches_per_proj_dispatch -->

#### Scenario: QK-norm honours family-specific weight offsets
- **WHEN** `qk_norm` is dispatched at offset 1.0 (Gemma 3) and offset 0.0 (Gemma 4 sliding / global)
- **THEN** outputs SHALL match the CPU reference, SHALL agree with `v_norm` in v-mode, SHALL match the CPU `v_norm` reference in v-mode, SHALL be invariant in-place vs separate buffers, and SHALL pass smoke + production-shape checks
<!-- test: larql_compute::test_kernel_qk_norm::qk_norm_gemma3_offset_one -->
<!-- test: larql_compute::test_kernel_qk_norm::qk_norm_gemma4_sliding_offset_zero -->
<!-- test: larql_compute::test_kernel_qk_norm::qk_norm_gemma4_global_offset_zero -->
<!-- test: larql_compute::test_kernel_qk_norm::qk_norm_v_mode_matches_v_norm_gemma4_sliding -->
<!-- test: larql_compute::test_kernel_qk_norm::qk_norm_v_mode_matches_v_norm_gemma4_global -->
<!-- test: larql_compute::test_kernel_qk_norm::qk_norm_v_mode_matches_cpu_v_norm_reference -->
<!-- test: larql_compute::test_kernel_qk_norm::qk_norm_in_place_matches_separate_buffers -->
<!-- test: larql_compute::test_kernel_qk_norm::qk_norm_qk_smoke -->
<!-- test: larql_compute::test_kernel_qk_norm::qk_norm_qk_gemma3_4b -->
<!-- test: larql_compute::test_kernel_qk_norm::qk_norm_qk_gemma4_global_offset0 -->
<!-- test: larql_compute::test_metal_shaders::qk_norm_matches_cpu_reference -->

#### Scenario: V-norm shaders cover production shapes and in-place mode
- **WHEN** `v_norm` is dispatched on a 4×256 all-ones input, on separate buffers across multiple shapes, and in-place
- **THEN** every head SHALL be written, the separate-buffer output SHALL match the reference, and the in-place output SHALL equal the separate-buffer reference
<!-- test: larql_compute::test_kernel_v_norm::all_ones_4x256_writes_every_head -->
<!-- test: larql_compute::test_kernel_v_norm::separate_buffers_match_reference_across_shapes -->
<!-- test: larql_compute::test_kernel_v_norm::in_place_matches_separate_buffer_reference -->
<!-- test: larql_compute::test_metal_shaders::v_norm_matches_cpu -->

#### Scenario: RoPE covers full / sliding / partial / batched configurations
- **WHEN** `rope_at_pos` is dispatched on Llama 2 full, Gemma 3 full-256, Gemma 4 sliding, Gemma 4 global partial, partial-rotary pass-through, and the batched per-head + qk smoke / Gemma 3 4B / partial-rotary variants
- **THEN** every dispatch SHALL match the CPU rope reference (and the partial-rotary tail SHALL be passed through unchanged)
<!-- test: larql_compute::test_kernel_rope_at_pos::rope_at_pos_llama2_full -->
<!-- test: larql_compute::test_kernel_rope_at_pos::rope_at_pos_gemma3_full_256 -->
<!-- test: larql_compute::test_kernel_rope_at_pos::rope_at_pos_gemma4_sliding -->
<!-- test: larql_compute::test_kernel_rope_at_pos::rope_at_pos_gemma4_global_partial -->
<!-- test: larql_compute::test_kernel_rope_at_pos::rope_at_pos_partial_pass_through_preserved -->
<!-- test: larql_compute::test_kernel_rope_at_pos::rope_at_pos_matches_rope_at_pos_batched_one_head -->
<!-- test: larql_compute::test_kernel_rope::rope_at_pos_batched_llama2_full -->
<!-- test: larql_compute::test_kernel_rope::rope_at_pos_batched_gemma3_full_256 -->
<!-- test: larql_compute::test_kernel_rope::rope_at_pos_batched_gemma4_sliding -->
<!-- test: larql_compute::test_kernel_rope::rope_at_pos_batched_gemma4_global_partial -->
<!-- test: larql_compute::test_kernel_rope::rope_at_pos_batched_q_heads_global -->
<!-- test: larql_compute::test_kernel_rope::rope_at_pos_batched_qk_smoke -->
<!-- test: larql_compute::test_kernel_rope::rope_at_pos_batched_qk_gemma3_4b -->
<!-- test: larql_compute::test_kernel_rope::rope_at_pos_batched_qk_partial_rotary -->
<!-- test: larql_compute::test_metal_shaders::rope_apply_matches_cpu -->
<!-- test: larql_compute::test_metal_shaders::rope_apply_partial_rotation -->

### Requirement: KV cache append + attend, GEGLU + down, and fused attention

The Metal backend SHALL provide a fused KV append + attend kernel
(`kv_append_attend_fused`), a fused causal attention (`fused_attention`),
fused GEGLU + down kernels (`q4k_geglu_down`,
`q6k_geglu_gelu_tanh_down_cached`), and standalone activation /
norm shaders (silu, gelu_tanh, layer_norm, scale_vector, residual_add).
The KV append path MUST write only the target slot for production
geometry (Llama 2; Gemma 3 4B; Gemma 4 sliding/global), MUST be
identical to a CPU re-implementation of append on a round-trip, MUST
clear at position 0, and MUST hand off cleanly to a long-context
prefill (n=18 / n=128). The KV attention shader MUST match a CPU
reference at production T (1, 18 sliding/global, 512 long, 2048 long).
The fused attention SHALL match the CPU reference for single-head and
head_dim 512.

#### Scenario: KV append writes only the target slot per family
- **WHEN** `kv_append` is run for Llama 2, Gemma 3 4B, Gemma 4 sliding, and Gemma 4 global, plus position-0 (clears otherwise) and prefill handoff
- **THEN** the append kernel SHALL write only the target slot, the round-trip SHALL match the CPU reference, position-0 SHALL clear and write one slot, and the prefill handoff SHALL match the CPU reference at n=18 / n=128
<!-- test: larql_compute::test_kernel_kv_cache_append::append_writes_only_target_slot_llama2 -->
<!-- test: larql_compute::test_kernel_kv_cache_append::append_writes_only_target_slot_gemma3_4b -->
<!-- test: larql_compute::test_kernel_kv_cache_append::append_writes_only_target_slot_gemma4_sliding -->
<!-- test: larql_compute::test_kernel_kv_cache_append::append_writes_only_target_slot_gemma4_global -->
<!-- test: larql_compute::test_kernel_kv_cache_append::append_at_pos_zero_clears_otherwise_only_writes_one -->
<!-- test: larql_compute::test_kernel_kv_cache_append::append_roundtrip_llama2_t8 -->
<!-- test: larql_compute::test_kernel_kv_cache_append::append_roundtrip_gemma3_4b_t18 -->
<!-- test: larql_compute::test_kernel_kv_cache_append::append_roundtrip_gemma4_sliding_t18 -->
<!-- test: larql_compute::test_kernel_kv_cache_append::append_roundtrip_gemma4_global_t18 -->
<!-- test: larql_compute::test_kernel_kv_cache_append::prefill_handoff_llama2_n18 -->
<!-- test: larql_compute::test_kernel_kv_cache_append::prefill_handoff_gemma3_4b_n18 -->
<!-- test: larql_compute::test_kernel_kv_cache_append::prefill_handoff_gemma4_sliding_n18 -->
<!-- test: larql_compute::test_kernel_kv_cache_append::prefill_handoff_gemma4_global_n18 -->
<!-- test: larql_compute::test_kernel_kv_cache_append::prefill_handoff_long_context_n128 -->

#### Scenario: KV attention matches CPU reference across families
- **WHEN** `kv_attention` is dispatched at T=1 Llama 2, T=18 Gemma 3 / Gemma 4 sliding / Gemma 4 global head_dim=512, T=512 long context, and T=2048 Gemma 4 global long context
- **THEN** the GPU output SHALL match the CPU attention reference within Metal precision
<!-- test: larql_compute::test_kernel_kv_attention::kv_attention_t1_llama2 -->
<!-- test: larql_compute::test_kernel_kv_attention::kv_attention_t18_gemma3 -->
<!-- test: larql_compute::test_kernel_kv_attention::kv_attention_t18_gemma4_sliding -->
<!-- test: larql_compute::test_kernel_kv_attention::kv_attention_t18_gemma4_global_head_dim_512 -->
<!-- test: larql_compute::test_kernel_kv_attention::kv_attention_t512_long_context -->
<!-- test: larql_compute::test_kernel_kv_attention::kv_attention_t2048_gemma4_global_long_context -->

#### Scenario: Fused causal attention matches CPU reference
- **WHEN** `fused_attention` is run at single-token and head_dim 512
- **THEN** outputs SHALL match the CPU `causal_attention` reference
<!-- test: larql_compute::test_kernel_fused_attention::fused_attention_matches_cpu_reference -->
<!-- test: larql_compute::test_kernel_fused_attention::fused_attention_head_dim_512 -->
<!-- test: larql_compute::test_metal_shaders::fused_attention_single_token -->

#### Scenario: GEGLU + down covers SiLU and GELU-tanh activation paths
- **WHEN** the `q4k_geglu_silu_down`, `q4k_geglu_gelu_tanh_down`, `q6k_geglu_silu_down`, and `q6k_geglu_gelu_tanh_down_*` shaders are dispatched at smoke, Gemma 3 4B, Gemma 4 31B, and Llama 2 7B FFN shapes
- **THEN** outputs SHALL match the CPU GEGLU + down reference and SHALL not produce NaN on large gates
<!-- test: larql_compute::test_kernel_q4k_geglu_down::q4k_geglu_silu_down_smoke -->
<!-- test: larql_compute::test_kernel_q4k_geglu_down::q4k_geglu_gelu_tanh_down_smoke -->
<!-- test: larql_compute::test_kernel_q4k_geglu_down::q4k_geglu_silu_down_gemma3_4b_ffn -->
<!-- test: larql_compute::test_kernel_q4k_geglu_down::q4k_geglu_gelu_tanh_down_gemma3_4b_ffn -->
<!-- test: larql_compute::test_kernel_q4k_geglu_down::q4k_geglu_silu_down_gemma4_31b_ffn -->
<!-- test: larql_compute::test_kernel_q4k_geglu_down::q4k_geglu_gelu_tanh_down_gemma4_31b_ffn -->
<!-- test: larql_compute::test_kernel_q6k_geglu_down::q6k_geglu_silu_down_smoke -->
<!-- test: larql_compute::test_kernel_q6k_geglu_down::q6k_geglu_gelu_tanh_down_smoke -->
<!-- test: larql_compute::test_kernel_q6k_geglu_down::q6k_geglu_silu_down_llama2_7b_ffn -->
<!-- test: larql_compute::test_kernel_q6k_geglu_down::q6k_geglu_gelu_tanh_down_gemma3_4b_ffn -->
<!-- test: larql_compute::test_kernel_q6k_geglu_down::q6k_geglu_gelu_tanh_down_gemma4_31b_ffn -->
<!-- test: larql_compute::test_metal_shaders::geglu_matches_cpu -->
<!-- test: larql_compute::test_metal_shaders::geglu_gelu_tanh_no_nan_on_large_gate -->
<!-- test: larql_compute::test_metal_shaders::silu_standalone_matches_cpu -->
<!-- test: larql_compute::test_metal_shaders::gelu_tanh_standalone_matches_cpu -->

### Requirement: Cooperative norm reduction and fused-residual stages

The Metal backend SHALL provide cooperative simd-reduction RMS-norm
shaders (`rms_norm`, `residual_norm_store`,
`post_attn_residual_norm_store`, `post_ffn_norm_residual_add`,
`residual_inject`) that scale to large hidden dimensions, fused
residual + norm + Q8 quantise paths, and a Q8 quantise standalone.
RMS-norm MUST match the CPU reference (and the zero-offset and
"large-vector simd-cooperative" variants), residual-add SHALL match
the CPU reference, Q8 quantise SHALL match the CPU `quantize_to_q8`
reference, and the fused residual-norm + Q8 path SHALL match the
unfused reference. The `residual_norm_store` and
`post_attn_residual_norm_store` paths MUST match the unfused
"residual + norm + raw-sum store" reference. The post-attn / post-ffn
fused stage helpers SHALL match a CPU pre-norm / post-norm reference,
and the Q8 staging emitted from those stages SHALL be round-trippable
through Q8 dequant.

#### Scenario: RMS-norm and residual-add match CPU references
- **WHEN** `rms_norm`, the zero-offset variant, the simd-cooperative large-vector variant, and `residual_add` are dispatched against the CPU references
- **THEN** outputs SHALL match within Metal precision and the simd-cooperative path SHALL agree with the standard path on long vectors
<!-- test: larql_compute::test_kernel_fused_ops_norms::rms_norm_matches_cpu -->
<!-- test: larql_compute::test_kernel_fused_ops_norms::rms_norm_zero_offset -->
<!-- test: larql_compute::test_kernel_fused_ops_norms::rms_norm_large_vector_simd_cooperative -->
<!-- test: larql_compute::test_kernel_fused_ops_norms::residual_norm_large_vector_simd_cooperative -->
<!-- test: larql_compute::test_kernel_fused_ops_norms::residual_add_matches_cpu -->
<!-- test: larql_compute::test_metal_shaders::layer_norm_matches_cpu -->
<!-- test: larql_compute::test_metal_shaders::layer_norm_no_bias_matches_cpu -->
<!-- test: larql_compute::test_metal_shaders::scale_vector_matches_cpu -->
<!-- test: larql_compute::test_metal_shaders::rms_norm_with_different_eps -->
<!-- test: larql_compute::test_metal_shaders::residual_add_correct -->

#### Scenario: Fused residual-norm + Q8 quantise matches separate ops
- **WHEN** `quantize_q8`, `rms_norm_q8`, `residual_norm`, and `residual_norm_store` are dispatched and compared to running the equivalent separate kernels
- **THEN** outputs SHALL match within Metal precision and the residual_norm_store SHALL match the residual_norm + raw-sum reference
<!-- test: larql_compute::test_kernel_fused_ops_norms::quantize_q8_matches_cpu -->
<!-- test: larql_compute::test_kernel_fused_ops_norms::rms_norm_q8_matches_separate_ops -->
<!-- test: larql_compute::test_kernel_fused_ops_norms::residual_norm_matches_separate_ops -->
<!-- test: larql_compute::test_kernel_new_fused_kernels::residual_norm_store_matches_residual_norm_and_raw_sum -->

#### Scenario: Stage encoders match CPU pre-norm / post-norm reference
- **WHEN** `stage_post_attn_pre_norm`, `stage_post_attn_post_norm`, `stage_post_ffn_pre_norm`, `stage_post_ffn_post_norm`, and `stage_quant_matvec` encoders run against the CPU reference, plus the Q8 FFN stage that emits Q8 staging
- **THEN** every stage output SHALL match the CPU reference and the Q8 staging buffer SHALL round-trip through dequantisation
<!-- test: larql_compute::test_kernel_vindex_integration::stage_post_attn_pre_norm_matches_cpu -->
<!-- test: larql_compute::test_kernel_vindex_integration::stage_post_attn_post_norm_matches_cpu -->
<!-- test: larql_compute::test_kernel_vindex_integration::stage_post_ffn_pre_norm_matches_cpu -->
<!-- test: larql_compute::test_kernel_vindex_integration::stage_post_ffn_post_norm_matches_cpu -->
<!-- test: larql_compute::test_kernel_vindex_integration::stage_quant_matvec_routes_format_to_correct_shader -->
<!-- test: larql_compute::test_kernel_vindex_integration::stage_post_attn_q8_ffn_emits_roundtrippable_q8 -->

### Requirement: Buffer cache, calibration, kernel handles, MoE dispatch and shader bench

The Metal backend SHALL maintain a buffer cache that maps slice
identity to GPU buffers, return distinct buffers for distinct slices,
return shared stubs for empty slices (with separate stubs for empty
f32 vs empty bytes), refuse to cache transient buffers, and panic
clearly if a Metal buffer is sized too small. The auto-calibration
helper SHALL return a FLOP threshold inside the documented envelope,
clamp manually-set thresholds to `MIN_FLOP_FLOOR`, and only ever
trigger the GPU above that floor. Kernel handles SHALL satisfy a
`KernelHandle` contract for every public op (Q4 pipelines, k_matvec,
ffn_gate_up, qkv_proj, q8_qkv_proj, geglu_down, gemv) and the Metal
backend's `Capability` truth table SHALL match the implemented
methods. The MoE preselected dispatch SHALL match the CPU MoE
reference, the shader bench parser SHALL read batched timing JSON,
and the KV cache helpers SHALL detect shape mismatches in conflicting
existing layers, copy populate-after-commit only into the target
layer, grow undersized caches, and use Q8 staging buffers sized to
the largest layer's q-dim with proper row-byte rounding.

#### Scenario: Buffer cache identity, transience, and bounds are enforced
- **WHEN** the buffer cache is exercised across slice identity, distinct slices, empty-slice stub, separate empty-f32 vs empty-bytes stubs, transient (non-cached) requests, undersized output buffers, and round-tripping a buffer through Metal
- **THEN** identical slices SHALL share buffers, distinct slices SHALL get distinct buffers, empty inputs SHALL share stubs, transient requests SHALL not be cached, output buffers SHALL be at least the requested size, the round-trip SHALL preserve f32 values, and undersized buffers SHALL panic with the documented message
<!-- test: larql_compute::metal::buffers::tests::get_f32_caches_by_slice_identity -->
<!-- test: larql_compute::metal::buffers::tests::get_f32_distinct_slices_get_distinct_buffers -->
<!-- test: larql_compute::metal::buffers::tests::get_f32_empty_slice_returns_shared_stub -->
<!-- test: larql_compute::metal::buffers::tests::empty_f32_and_empty_bytes_have_separate_stubs -->
<!-- test: larql_compute::metal::buffers::tests::transient_buffers_are_not_cached -->
<!-- test: larql_compute::metal::buffers::tests::output_buffer_is_at_least_requested_size -->
<!-- test: larql_compute::metal::buffers::tests::read_buffer_f32_round_trip -->
<!-- test: larql_compute::test_metal_shaders::buffer_cache_reuses_same_pointer -->

#### Scenario: Calibration returns a legal threshold and clamps manual values
- **WHEN** `calibrate` runs against synthetic test cases and `set_flop_threshold` is called with a value below `MIN_FLOP_FLOOR`
- **THEN** the calibrated threshold SHALL fall in the documented `[MIN_FLOP_FLOOR, DEFAULT_FLOP_THRESHOLD]` envelope and manual values SHALL be clamped to the floor
<!-- test: larql_compute::metal::calibrate::tests::calibrate_returns_threshold_in_legal_envelope -->
<!-- test: larql_compute::metal::calibrate::tests::set_flop_threshold_clamps_to_min_floor -->

#### Scenario: Kernel handles are populated for every op and capabilities match
- **WHEN** the Metal kernel handle contract is checked across q4_pipelines, k_matvec, ffn_gate_up, qkv_proj, q8_qkv_proj, geglu_down, gemv, plus the `MetalBackend` capability truth table
- **THEN** every handle SHALL be present and the truth table SHALL agree with the implemented methods
<!-- test: larql_compute::test_kernel_handle_contract::q4_pipelines_handle_contract -->
<!-- test: larql_compute::test_kernel_handle_contract::k_matvec_handle_contract -->
<!-- test: larql_compute::test_kernel_handle_contract::ffn_gate_up_handle_contract -->
<!-- test: larql_compute::test_kernel_handle_contract::qkv_proj_handle_contract -->
<!-- test: larql_compute::test_kernel_handle_contract::q8_qkv_proj_handle_contract -->
<!-- test: larql_compute::test_kernel_handle_contract::geglu_down_handle_contract -->
<!-- test: larql_compute::test_kernel_handle_contract::gemv_handle_contract -->
<!-- test: larql_compute::test_kernel_handle_contract::metal_backend_capability_truth_table -->

#### Scenario: All shaders compile and required kernel functions exist
- **WHEN** the shader-compilation harness runs across every shader file and the new fused-kernel function table is checked
- **THEN** every shader SHALL compile cleanly and every advertised kernel function SHALL resolve
<!-- test: larql_compute::test_metal_shaders::all_shaders_compile -->
<!-- test: larql_compute::test_metal_shaders::all_kernel_functions_exist -->
<!-- test: larql_compute::test_metal_shaders::all_new_kernel_functions_exist -->
<!-- test: larql_compute::test_metal_shaders::new_kernel_functions_exist -->
<!-- test: larql_compute::test_metal_shaders::metal_backend_implements_trait -->

#### Scenario: MoE preselected dispatch matches CPU MoE reference
- **WHEN** the Metal MoE preselected small Q4_K kernel runs against the CPU MoE reference
- **THEN** the GPU output SHALL match the CPU output within Metal precision
<!-- test: larql_compute::test_kernel_moe_expert_dispatch::metal_moe_preselected_small_q4k_matches_cpu -->

#### Scenario: KV cache helpers detect shape mismatch and grow safely
- **WHEN** populate-after-commit copies into a `None` cache, into an existing layer, into an undersized cache, and into a single layer of a multi-layer cache, plus a conflicting layer-shape detection
- **THEN** the `None` case SHALL be a no-op, the layer-targeted copy SHALL update only that layer, the undersized cache SHALL be grown, the multi-layer one-layer copy SHALL leave other layers untouched, the empty cache SHALL be grown by the one-layer copy, and conflicting shapes SHALL be detected
<!-- test: larql_compute::metal::ops::full_pipeline::kv_copy::tests::populate_kv_after_commit_with_none_cache_is_a_noop -->
<!-- test: larql_compute::metal::ops::full_pipeline::kv_copy::tests::populate_kv_after_commit_copies_into_correct_layer -->
<!-- test: larql_compute::metal::ops::full_pipeline::kv_copy::tests::populate_kv_after_commit_grows_undersized_cache -->
<!-- test: larql_compute::metal::ops::full_pipeline::kv_copy::tests::populate_kv_one_layer_updates_only_target_layer -->
<!-- test: larql_compute::metal::ops::full_pipeline::kv_copy::tests::populate_kv_one_layer_grows_empty_cache -->
<!-- test: larql_compute::metal::ops::kv_cache::tests::shape_mismatch_detects_conflicting_existing_layer -->

#### Scenario: Q8 staging buffers are sized for the largest layer's q-dim
- **WHEN** Q8 staging buffers are allocated across uniform geometry, mixed geometry, empty layers, and row-byte rounding
- **THEN** the buffer size SHALL be the max of `hidden` and `q_dim` for uniform layers, the largest layer's q-dim for mixed layers, the documented fallback for empty layer lists, and the row bytes SHALL round up to a full Q8 block
<!-- test: larql_compute::metal::ops::full_pipeline::buffers::tests::q8_staging_uniform_geometry_picks_max_of_hidden_and_qdim -->
<!-- test: larql_compute::metal::ops::full_pipeline::buffers::tests::q8_staging_mixed_geometry_picks_largest_layer_q_dim -->
<!-- test: larql_compute::metal::ops::full_pipeline::buffers::tests::q8_staging_empty_layers_uses_fallback -->
<!-- test: larql_compute::metal::ops::full_pipeline::buffers::tests::q8s_row_bytes_rounds_up_to_full_block -->

#### Scenario: Shader bench compare-JSON parser reads batched timings
- **WHEN** the shader-bench compare-JSON parser ingests a batched-ms record
- **THEN** it SHALL extract the documented batched_ms timing fields
<!-- test: larql_compute::metal::diag::shader_bench::tests::compare_json_parser_reads_batched_ms -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_compute::test_metal_shaders::**::* -->
