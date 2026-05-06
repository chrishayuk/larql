## ADDED Requirements

### Requirement: f16 / bf16 encode and decode round-trips

`larql_models::quant::half` SHALL provide `encode_f16`, `decode_f16`,
`encode_bf16`, and `decode_bf16` such that any f32 value representable
in the target half format round-trips to itself with at most the
format's quantisation error. The decoders SHALL handle IEEE 754
special values (zero, ±infinity, NaN) without panicking, and known
fixed values (1.0, 2.0, the smallest positive value, etc.) MUST decode
to the published bit-pattern interpretation.

#### Scenario: f16 representable values round-trip
- **WHEN** an f32 value that is exactly representable in binary16 is encoded with `encode_f16` and decoded with `decode_f16`
- **THEN** the resulting f32 SHALL equal the input
<!-- test: larql_models::quant::half::f16_round_trip -->
<!-- test: larql_models::quant::half::f16_encode_decode_round_trip -->

#### Scenario: bf16 representable values round-trip
- **WHEN** an f32 value that is exactly representable in bfloat16 is encoded with `encode_bf16` and decoded with `decode_bf16`
- **THEN** the resulting f32 SHALL equal the input
<!-- test: larql_models::quant::half::bf16_round_trip -->
<!-- test: larql_models::quant::half::bf16_encode_decode_round_trip -->

#### Scenario: f16 special values decode without panicking
- **WHEN** `decode_f16` is given the canonical bit patterns for ±zero, ±infinity, and NaN
- **THEN** decoding SHALL succeed and the f32 outputs SHALL preserve those special-value classes
<!-- test: larql_models::quant::half::f16_special_values -->

#### Scenario: f16 known values decode to documented constants
- **WHEN** `decode_f16` is given the bit patterns for canonical reference values (for example 1.0)
- **THEN** the decoded f32 SHALL equal the documented value
<!-- test: larql_models::quant::half::f16_known_values -->

#### Scenario: bf16 known values decode to documented constants
- **WHEN** `decode_bf16` is given the bit patterns for canonical reference values
- **THEN** the decoded f32 SHALL equal the documented value
<!-- test: larql_models::quant::half::bf16_known_values -->

### Requirement: GGML legacy block formats (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)

`larql_models::quant::ggml` SHALL implement the legacy GGML block
formats Q4_0, Q4_1, Q5_0, Q5_1, and Q8_0 over 32-element blocks with
the documented per-block storage layouts. Dequantising MUST yield the
mathematically expected f32 values for canonical fixtures (including
zero-scale blocks, multi-block arrays, and Q5 blocks whose 5th bit is
non-zero). Round-trip from f32 through `quantize_q4_0` /
`quantize_q8_0` and back MUST preserve every value within the format's
representable step. Each dequantiser MUST reject buffers shorter than
the declared block count and MUST reject `n_elements` values that are
not a multiple of the block size.

#### Scenario: Q4_0 dequantises a known block to the expected values
- **WHEN** `dequantize_q4_0` is called with a hand-crafted block (scale = 1.0, nibbles = 0x12)
- **THEN** the result SHALL match the documented `value = scale * (nibble - 8)` formula
<!-- test: larql_models::quant::ggml::q4_0_basic -->

#### Scenario: Q4_0 zero-scale blocks dequantise to zero
- **WHEN** `dequantize_q4_0` is called on a block whose f16 scale is 0.0
- **THEN** every output value SHALL be `0.0`
<!-- test: larql_models::quant::ggml::q4_0_zero_scale -->

#### Scenario: Q4_0 multi-block arrays dequantise per-block
- **WHEN** `dequantize_q4_0` is called on data containing two consecutive blocks
- **THEN** each block SHALL be decoded independently with its own scale
<!-- test: larql_models::quant::ggml::q4_0_two_blocks -->

#### Scenario: Q4_1 applies the per-block minimum offset
- **WHEN** `dequantize_q4_1` is called on a block with a non-zero minimum
- **THEN** the result SHALL match `value = scale * nibble + min`
<!-- test: larql_models::quant::ggml::q4_1_basic -->
<!-- test: larql_models::quant::ggml::q4_1_with_offset -->

#### Scenario: Q5_0 / Q5_1 incorporate the high bit per element
- **WHEN** `dequantize_q5_0` and `dequantize_q5_1` are called on blocks whose 5th bit is non-zero
- **THEN** the high bits SHALL be combined with the low nibbles before scaling
<!-- test: larql_models::quant::ggml::q5_0_basic -->
<!-- test: larql_models::quant::ggml::q5_0_with_high_bits -->
<!-- test: larql_models::quant::ggml::q5_0_mixed -->
<!-- test: larql_models::quant::ggml::q5_1_basic -->
<!-- test: larql_models::quant::ggml::q5_1_with_high_bits -->

#### Scenario: Q8_0 round-trip is precise within the int8 step
- **WHEN** an f32 array is encoded with `quantize_q8_0` and decoded with `dequantize_q8_0`
- **THEN** every output SHALL be within one quantisation step of the input
<!-- test: larql_models::quant::ggml::q8_0_basic -->
<!-- test: larql_models::quant::ggml::q8_0_round_trip_precise -->
<!-- test: larql_models::quant::ggml::q8_0_round_trip_edges -->

#### Scenario: Q4_0 round-trip preserves values within half a step
- **WHEN** an f32 array is encoded with `quantize_q4_0` and decoded with `dequantize_q4_0`
- **THEN** every output SHALL be within half a quantisation step of the input
<!-- test: larql_models::quant::ggml::q4_0_round_trip_preserves_within_half_step -->
<!-- test: larql_models::quant::ggml::q4_0_round_trip_all_zero -->

#### Scenario: Truncated buffers are rejected for every legacy format
- **WHEN** any of `dequantize_q4_0`, `dequantize_q4_1`, `dequantize_q8_0`, `dequantize_q5_0`, or `dequantize_q5_1` is called with a buffer shorter than `n_elements / 32` blocks
- **THEN** the call SHALL return a parse error rather than panicking
<!-- test: larql_models::quant::ggml::q4_0_rejects_short_buffer -->
<!-- test: larql_models::quant::ggml::q4_1_rejects_short_buffer -->
<!-- test: larql_models::quant::ggml::q8_0_rejects_short_buffer -->
<!-- test: larql_models::quant::ggml::q5_0_rejects_short_buffer -->
<!-- test: larql_models::quant::ggml::q5_1_rejects_short_buffer -->

#### Scenario: Misaligned `n_elements` is rejected
- **WHEN** a legacy dequantiser is called with `n_elements` that is not a multiple of 32
- **THEN** the call SHALL return a parse error
<!-- test: larql_models::quant::ggml::q4_0_rejects_misaligned_n_elements -->

#### Scenario: Top-level dispatch routes by GGML type id
- **WHEN** `dequantize` is called with `TYPE_Q4_0`, `TYPE_Q5_0`, `TYPE_Q5_1`, `TYPE_Q8_0`, `TYPE_Q4_K`, or `TYPE_Q6_K`
- **THEN** dispatch SHALL invoke the corresponding type-specific decoder and produce the same output as calling that decoder directly
<!-- test: larql_models::quant::ggml::q4_0_via_dequantize -->
<!-- test: larql_models::quant::ggml::q8_0_via_dequantize -->
<!-- test: larql_models::quant::ggml::q5_0_via_dequantize -->
<!-- test: larql_models::quant::ggml::q5_1_via_dequantize -->
<!-- test: larql_models::quant::ggml::q4_k_via_dequantize_roundtrips_to_known_output -->
<!-- test: larql_models::quant::ggml::q6_k_via_dequantize -->

#### Scenario: Sizing helpers report the correct byte counts and names
- **WHEN** `tensor_data_size` is called on the supported GGML type ids
- **THEN** it SHALL return the documented byte count, and `type_name` SHALL return the canonical short name
<!-- test: larql_models::quant::ggml::tensor_sizes -->
<!-- test: larql_models::quant::ggml::type_names -->

#### Scenario: F32 / F16 / BF16 passthrough decoders reject short buffers
- **WHEN** a passthrough dequantise is requested for `TYPE_F32`, `TYPE_F16`, or `TYPE_BF16` on a buffer shorter than `n_elements`
- **THEN** the call SHALL return a parse error, and zero-element calls SHALL succeed with an empty output
<!-- test: larql_models::quant::ggml::f32_passthrough -->
<!-- test: larql_models::quant::ggml::passthrough_f32_rejects_short_buffer -->
<!-- test: larql_models::quant::ggml::passthrough_f16_rejects_short_buffer -->
<!-- test: larql_models::quant::ggml::passthrough_bf16_rejects_short_buffer -->
<!-- test: larql_models::quant::ggml::empty_input_ok_when_zero_elements -->

### Requirement: GGML K-quant block formats (Q4_K, Q6_K)

`larql_models::quant::ggml::q4_k` and `q6_k` SHALL dequantise 256-element
super-blocks per the documented Q4_K (144 bytes per super-block) and
Q6_K (210 bytes per super-block) layouts. The fused row operations
`q4k_row_dot` / `q6k_row_dot` MUST agree with full-dequantise-then-dot
within numerical tolerance, and `q4k_row_scaled_add` /
`q6k_row_scaled_add` MUST agree with `alpha * dequantize(row)` added
into the destination. Both formats MUST reject truncated buffers and
misaligned element counts. On aarch64, NEON and scalar implementations
of `q4k_row_dot` / `q6k_row_dot` MUST agree.

#### Scenario: Q4_K dequantises canonical fixtures to the documented values
- **WHEN** `dequantize_q4_k` is called on a fixture super-block with non-zero values
- **THEN** the resulting f32 array SHALL match the published reference values
<!-- test: larql_models::quant::ggml::q4_k_dequantize_known_nonzero_values -->

#### Scenario: Q4_K row-dot agrees with dequantise-then-dot
- **WHEN** `q4k_row_dot` is called on a row whose dequantisation is also computed independently
- **THEN** the fused dot-product SHALL agree with `dot(dequantize_q4_k(row), x)` within tolerance
<!-- test: larql_models::quant::ggml::q4k_row_dot_matches_dequantized_dot -->

#### Scenario: NEON and scalar Q4_K row-dot agree on aarch64
- **WHEN** `q4k_row_dot` is invoked on aarch64 hosts for single-block and multi-block rows
- **THEN** the NEON path SHALL produce the same value as the scalar fallback
<!-- test: larql_models::quant::ggml::q4k_row_dot_neon_matches_scalar_single_block -->
<!-- test: larql_models::quant::ggml::q4k_row_dot_neon_matches_scalar_multi_block -->

#### Scenario: Q4_K scaled-add agrees with alpha-scaled dequant
- **WHEN** `q4k_row_scaled_add(row, alpha, &mut out)` is called
- **THEN** the resulting `out` SHALL equal `out_initial + alpha * dequantize_q4_k(row)`
<!-- test: larql_models::quant::ggml::q4k_row_scaled_add_matches_alpha_times_deq -->

#### Scenario: Q4_K rejects truncated and misaligned inputs
- **WHEN** Q4_K is called on a buffer shorter than the declared super-block count or with `n_elements` not a multiple of 256
- **THEN** the call SHALL return a parse error
<!-- test: larql_models::quant::ggml::q4_k_rejects_short_buffer -->
<!-- test: larql_models::quant::ggml::q4k_row_scaled_add_rejects_misaligned -->

#### Scenario: Q6_K row-dot matches dequantise-then-dot
- **WHEN** `q6k_row_dot` is called on a row whose dequantisation is also computed independently
- **THEN** the fused dot-product SHALL agree with `dot(dequantize_q6_k(row), x)` within tolerance
<!-- test: larql_models::quant::ggml::q6k_row_dot_matches_dequantized_dot -->

#### Scenario: NEON and scalar Q6_K row-dot agree on aarch64
- **WHEN** `q6k_row_dot` is invoked on aarch64 hosts for single-block and multi-block rows
- **THEN** the NEON path SHALL produce the same value as the scalar fallback
<!-- test: larql_models::quant::ggml::q6k_row_dot_neon_matches_scalar_single_block -->
<!-- test: larql_models::quant::ggml::q6k_row_dot_neon_matches_scalar_multi_block -->

#### Scenario: Q6_K scaled-add agrees with alpha-scaled dequant
- **WHEN** `q6k_row_scaled_add(row, alpha, &mut out)` is called
- **THEN** the resulting `out` SHALL equal `out_initial + alpha * dequantize_q6_k(row)`
<!-- test: larql_models::quant::ggml::q6k_row_scaled_add_matches_alpha_times_deq -->

#### Scenario: Q6_K rejects truncated and misaligned inputs
- **WHEN** Q6_K is called on a buffer shorter than the declared super-block count or with `n_elements` not a multiple of 256
- **THEN** the call SHALL return a parse error
<!-- test: larql_models::quant::ggml::q6_k_rejects_short_buffer -->
<!-- test: larql_models::quant::ggml::q6_k_rejects_misaligned_n_elements -->
<!-- test: larql_models::quant::ggml::q6k_row_scaled_add_rejects_misaligned -->

### Requirement: MXFP4 (e8m0 + FP4) format for packed MoE experts

`larql_models::quant::mxfp4` SHALL implement the MXFP4 microscaling
format used for GPT-OSS / OpenAI packed MoE expert weights:
32-element groups, one e8m0 scale byte per group, plus 16 bytes of
packed FP4 nibbles. The e8m0 helper `e8m0_to_f32` MUST encode `2^(exp
- 127)` for normal exponents (returning `1.0` at exponent 127 and
mapping the all-ones byte to NaN). The 16-entry FP4 nibble decode
table MUST match the documented positive and negative values. The
expert dequantisers MUST split 4-bit nibbles, multiply by the per-group
scale, and SHALL reject truncated `blocks` or `scales` buffers. The
fused-gate-up split SHALL produce two independent expert weight sets
of the same shape.

#### Scenario: e8m0 zero exponent dequantises to a finite tiny scale
- **WHEN** `e8m0_to_f32(0)` is called
- **THEN** it SHALL return a finite value equal to `2^-127`
<!-- test: larql_models::quant::mxfp4::e8m0_zero -->

#### Scenario: e8m0 exponent 127 dequantises to 1.0
- **WHEN** `e8m0_to_f32(127)` is called
- **THEN** it SHALL return `1.0`
<!-- test: larql_models::quant::mxfp4::e8m0_one -->

#### Scenario: e8m0 powers-of-two dequantise correctly
- **WHEN** `e8m0_to_f32` is called on representative power-of-two exponents
- **THEN** the result SHALL equal the corresponding `2^k`
<!-- test: larql_models::quant::mxfp4::e8m0_powers_of_two -->

#### Scenario: e8m0 NaN encoding maps to NaN
- **WHEN** `e8m0_to_f32(255)` is called
- **THEN** the result SHALL be NaN
<!-- test: larql_models::quant::mxfp4::e8m0_nan -->

#### Scenario: FP4 positive nibble table matches the documented values
- **WHEN** the positive half (nibbles `0x0..=0x7`) of the FP4 decode table is read
- **THEN** the values SHALL be `[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]`
<!-- test: larql_models::quant::mxfp4::table_positive -->

#### Scenario: FP4 negative nibble table mirrors the positive half
- **WHEN** the negative half (nibbles `0x8..=0xF`) of the FP4 decode table is read
- **THEN** each value SHALL be the negation of the corresponding positive value
<!-- test: larql_models::quant::mxfp4::table_negative -->

#### Scenario: Dequantising all-ones nibbles produces the documented FP4 value
- **WHEN** `dequantize_expert` is called on a group whose nibbles are all `0x1` and whose scale is `1.0`
- **THEN** every output SHALL be `0.5`
<!-- test: larql_models::quant::mxfp4::dequant_all_ones -->

#### Scenario: Dequantising with a non-unit scale multiplies the FP4 values
- **WHEN** `dequantize_expert` is called with a non-unit e8m0 scale
- **THEN** every output SHALL equal `scale * fp4_value`
<!-- test: larql_models::quant::mxfp4::dequant_with_scale -->
<!-- test: larql_models::quant::mxfp4::dequant_negative -->

#### Scenario: Zero scale collapses every output to zero
- **WHEN** `dequantize_expert` is called on a group whose scale byte is the e8m0 zero encoding
- **THEN** the output SHALL be all zeros within tolerance
<!-- test: larql_models::quant::mxfp4::dequant_zero_scale -->

#### Scenario: Mixed-nibble groups dequantise per nibble
- **WHEN** `dequantize_expert` is called on a group with mixed nibbles
- **THEN** each output position SHALL match `scale * fp4_table[nibble]`
<!-- test: larql_models::quant::mxfp4::dequant_mixed_nibbles -->
<!-- test: larql_models::quant::mxfp4::dequant_two_groups -->

#### Scenario: Multi-expert dequant slices scales and blocks per expert
- **WHEN** `dequantize_all_experts` is called for two experts with distinct scales
- **THEN** each expert's output SHALL use only its own scale slice
<!-- test: larql_models::quant::mxfp4::dequant_two_experts -->
<!-- test: larql_models::quant::mxfp4::dequant_all_experts_slices_scales_per_expert -->

#### Scenario: Truncated expert blocks or scales are rejected
- **WHEN** `dequantize_expert` or `dequantize_all_experts` is called on `blocks` or `scales` shorter than the declared shape
- **THEN** the call SHALL return a parse error rather than slicing OOB
<!-- test: larql_models::quant::mxfp4::dequant_expert_rejects_short_blocks -->
<!-- test: larql_models::quant::mxfp4::dequant_expert_rejects_short_scales -->
<!-- test: larql_models::quant::mxfp4::dequant_all_experts_rejects_short_blocks -->
<!-- test: larql_models::quant::mxfp4::dequant_all_experts_rejects_short_scales -->

#### Scenario: Zero-expert dequant succeeds with an empty result
- **WHEN** `dequantize_all_experts` is called with `num_experts = 0`
- **THEN** the call SHALL return an empty `Vec<Vec<f32>>` without error
<!-- test: larql_models::quant::mxfp4::dequant_zero_experts_ok -->

#### Scenario: Fused gate_up tensor splits at the row midpoint
- **WHEN** `split_gate_up_experts` is called with even `out_features`
- **THEN** it SHALL produce two expert weight sets of identical shape covering the gate and up halves
<!-- test: larql_models::quant::mxfp4::split_gate_up_even_split -->
<!-- test: larql_models::quant::mxfp4::split_gate_up_two_experts -->

### Requirement: FP4 nibble layout, table, and rounding

`larql_models::quant::fp4` SHALL share the same 16-entry FP4 decode
table as MXFP4, SHALL pack two nibbles per byte with the lower nibble
holding the even-index element and the upper nibble holding the
odd-index element, SHALL round to nearest even when encoding, and SHALL
saturate to `±6.0` for inputs that exceed the representable range.
NaN inputs MUST encode to the FP4 zero nibble, infinities MUST
saturate, and signed zero MUST be preserved.

#### Scenario: FP4 table matches the MXFP4 table
- **WHEN** the FP4 decode table is compared to the MXFP4 nibble table
- **THEN** the two tables SHALL be identical
<!-- test: larql_models::quant::fp4::fp4_table_matches_mxfp4 -->

#### Scenario: Representable values round-trip exactly
- **WHEN** an f32 value that is exactly representable in FP4 is encoded and decoded
- **THEN** the result SHALL equal the input
<!-- test: larql_models::quant::fp4::fp4_representable_round_trip -->

#### Scenario: Out-of-range values saturate to ±6.0
- **WHEN** an f32 value larger than 6.0 (or smaller than -6.0) is encoded
- **THEN** the decoded value SHALL be `6.0` or `-6.0`
<!-- test: larql_models::quant::fp4::fp4_saturation -->
<!-- test: larql_models::quant::fp4::fp4_inf_saturates -->

#### Scenario: Encoding rounds halfway values to nearest even
- **WHEN** an f32 value lies exactly halfway between two representable FP4 values
- **THEN** the encoder SHALL round to the even neighbour
<!-- test: larql_models::quant::fp4::fp4_rounding_to_nearest_even -->

#### Scenario: Nibble packing places the lower-index element in the lower nibble
- **WHEN** a pair of FP4 nibbles is packed and then unpacked
- **THEN** the lower nibble SHALL hold the even-indexed element and the upper nibble SHALL hold the odd-indexed element, and the round-trip SHALL preserve the original sequence
<!-- test: larql_models::quant::fp4::nibble_pack_unpack_round_trip -->
<!-- test: larql_models::quant::fp4::nibble_pack_order_lower_is_even_index -->
<!-- test: larql_models::quant::fp4::fp4_nibble_packing_assorted_lengths -->

#### Scenario: `decode_fp4_into` matches the static table
- **WHEN** `decode_fp4_into` is called over every nibble value
- **THEN** the output SHALL match the documented FP4 decode table
<!-- test: larql_models::quant::fp4::decode_fp4_into_matches_table -->

#### Scenario: Special-value encoding follows the documented rules
- **WHEN** the encoder is given NaN, ±infinity, subnormal-like, or signed-zero inputs
- **THEN** NaN SHALL map to the FP4 zero nibble, infinities SHALL saturate, subnormals SHALL flush per the documented thresholds, and signed zero SHALL be preserved
<!-- test: larql_models::quant::fp4::fp4_nan_input_maps_to_zero -->
<!-- test: larql_models::quant::fp4::fp4_subnormal_like_values -->
<!-- test: larql_models::quant::fp4::fp4_signed_zero -->

### Requirement: FP4 / FP8 block layout (`fp4_block.rs`)

`larql_models::quant::fp4_block` SHALL implement block-quantised FP4
and FP8 layouts whose byte sizes are exactly 137 bytes per FP4 block
and 257 bytes per FP8 block. Block round-trip MUST preserve a Gaussian
distribution within the format's accuracy bound, MUST handle all-zero
blocks, MUST tolerate pathological dynamic-range ratios within a
block, MUST flush below-subnormal magnitudes to zero (FP8) and SHALL
preserve a single outlier inside an otherwise-quiet block. Layer-level
helpers MUST round-trip 2D feature matrices.

#### Scenario: FP4 block byte size is exactly 137 bytes
- **WHEN** an FP4 block is laid out
- **THEN** its serialised size SHALL be 137 bytes
<!-- test: larql_models::quant::fp4_block::fp4_block_size_is_137_bytes -->

#### Scenario: FP8 block byte size is exactly 257 bytes
- **WHEN** an FP8 block is laid out
- **THEN** its serialised size SHALL be 257 bytes
<!-- test: larql_models::quant::fp4_block::fp8_block_size_is_257_bytes -->

#### Scenario: FP4 block round-trips a Gaussian distribution
- **WHEN** a Gaussian f32 vector is encoded and decoded as FP4 blocks
- **THEN** the decoded vector SHALL stay within the documented accuracy bound of the input
<!-- test: larql_models::quant::fp4_block::fp4_block_round_trip_gaussian -->

#### Scenario: FP4 block round-trips a pathological dynamic-range ratio
- **WHEN** a block contains values spanning many orders of magnitude
- **THEN** encode then decode SHALL still keep every element within the documented FP4 accuracy bound
<!-- test: larql_models::quant::fp4_block::fp4_block_round_trip_pathological_ratio -->

#### Scenario: FP4 all-zero block decodes to all zeros
- **WHEN** an FP4 block is encoded from an all-zero input
- **THEN** decoding SHALL return all zeros
<!-- test: larql_models::quant::fp4_block::fp4_block_all_zeros -->

#### Scenario: FP4 block sparsity preserves a single outlier
- **WHEN** a block is mostly zero with a single outlier value
- **THEN** the decoded block SHALL preserve the outlier within the format's quantisation step
<!-- test: larql_models::quant::fp4_block::fp4_block_single_outlier_preserved -->
<!-- test: larql_models::quant::fp4_block::fp4_block_sparse_single_element -->

#### Scenario: FP4 block tolerates mixed zero and non-zero sub-blocks
- **WHEN** a block contains a mix of all-zero and non-zero sub-blocks
- **THEN** encode then decode SHALL preserve every sub-block independently
<!-- test: larql_models::quant::fp4_block::fp4_block_mixed_zero_and_nonzero_sub_blocks -->

#### Scenario: FP4 block NaN inputs map to zero elements
- **WHEN** an FP4 block contains NaN inputs
- **THEN** the corresponding decoded element SHALL be `0.0`
<!-- test: larql_models::quant::fp4_block::fp4_block_nan_input_maps_to_zero_element -->

#### Scenario: FP8 block round-trips a Gaussian distribution
- **WHEN** a Gaussian f32 vector is encoded and decoded as FP8 blocks
- **THEN** the decoded vector SHALL stay within the documented FP8 accuracy bound of the input
<!-- test: larql_models::quant::fp4_block::fp8_block_round_trip_gaussian -->

#### Scenario: FP8 small-magnitude FFN-down-style values round-trip
- **WHEN** typical FFN-down-style small magnitudes are encoded and decoded as FP8
- **THEN** decoded values SHALL be within FP8 accuracy of the input
<!-- test: larql_models::quant::fp4_block::fp8_block_small_magnitude_like_ffn_down -->

#### Scenario: FP8 saturation values survive an encode-decode round trip
- **WHEN** values at the FP8 saturation bound are encoded and decoded
- **THEN** the output SHALL still saturate to those bounds rather than becoming NaN
<!-- test: larql_models::quant::fp4_block::fp8_block_saturation_values_round_trip -->

#### Scenario: FP8 below-subnormal magnitudes flush to zero
- **WHEN** an FP8 block contains values below the format's subnormal threshold
- **THEN** the decoded values SHALL be `0.0`
<!-- test: larql_models::quant::fp4_block::fp8_block_below_subnormal_flushes_to_zero -->

#### Scenario: FP4 / FP8 feature-vector helpers round-trip 2560 elements
- **WHEN** `fp4_feature_round_trip_2560` and `fp8_feature_round_trip_2560` encode and decode a 2560-element feature vector
- **THEN** the round-tripped vector SHALL stay within the format's accuracy bound
<!-- test: larql_models::quant::fp4_block::fp4_feature_round_trip_2560 -->
<!-- test: larql_models::quant::fp4_block::fp8_feature_round_trip_2560 -->

#### Scenario: FP4 / FP8 layer-level helpers round-trip a small layer
- **WHEN** the layer-level helpers encode and decode a small 2D layer
- **THEN** the decoded layer SHALL match the input within the format's accuracy bound
<!-- test: larql_models::quant::fp4_block::fp4_layer_round_trip_small -->
<!-- test: larql_models::quant::fp4_block::fp8_layer_round_trip_small -->

#### Scenario: FP4 typical 4-bit distribution round-trips
- **WHEN** a typical 4-bit-distribution test vector is encoded and decoded
- **THEN** the result SHALL stay within the documented FP4 accuracy bound
<!-- test: larql_models::quant::fp4_block::fp4_block_typical_4b_distribution -->

### Requirement: FP8 (E4M3) scaling and dequant

`larql_models::quant::fp8` SHALL implement IEEE-style E4M3 FP8 with the
documented canonical values, representable round-trip, saturation
short of NaN, infinity-saturation, signed-NaN preservation, subnormal
flushing, and round-to-nearest semantics. Bulk encode/decode over the
representable set MUST be lossless.

#### Scenario: E4M3 canonical values match the documented bit patterns
- **WHEN** canonical values (e.g. zero, one, the smallest normal, the largest finite) are encoded and decoded
- **THEN** the round-tripped values SHALL match the published E4M3 reference table
<!-- test: larql_models::quant::fp8::e4m3_canonical_values -->

#### Scenario: E4M3 representable values round-trip
- **WHEN** an f32 value exactly representable in E4M3 is encoded and decoded
- **THEN** the output SHALL equal the input
<!-- test: larql_models::quant::fp8::e4m3_round_trip_representable -->
<!-- test: larql_models::quant::fp8::e4m3_bulk_representable_round_trip -->

#### Scenario: E4M3 saturates short of NaN for very large magnitudes
- **WHEN** an f32 value larger than the largest finite E4M3 magnitude is encoded
- **THEN** the decoded value SHALL saturate to the largest finite E4M3 value rather than NaN
<!-- test: larql_models::quant::fp8::e4m3_saturation -->
<!-- test: larql_models::quant::fp8::e4m3_saturates_short_of_nan -->
<!-- test: larql_models::quant::fp8::e4m3_infinity_saturates -->

#### Scenario: E4M3 tiny magnitudes flush to zero
- **WHEN** an f32 magnitude below the smallest E4M3 subnormal is encoded
- **THEN** the decoded value SHALL be `0.0`
<!-- test: larql_models::quant::fp8::e4m3_tiny_flush_to_zero -->

#### Scenario: E4M3 rounds to nearest
- **WHEN** an f32 value falls between two representable E4M3 values
- **THEN** encoding SHALL round to the nearest representable value
<!-- test: larql_models::quant::fp8::e4m3_rounding_to_nearest -->

#### Scenario: E4M3 subnormal sweep matches the published step
- **WHEN** a sweep across the E4M3 subnormal range is encoded and decoded
- **THEN** every step SHALL match the documented subnormal grid, and the subnormal-to-normal boundary SHALL behave consistently
<!-- test: larql_models::quant::fp8::e4m3_subnormal_sweep -->
<!-- test: larql_models::quant::fp8::e4m3_subnormal_normal_boundary -->

#### Scenario: E4M3 preserves negative NaN sign
- **WHEN** a negative-NaN f32 is encoded and decoded
- **THEN** the result SHALL be a negative NaN
<!-- test: larql_models::quant::fp8::e4m3_negative_nan_preserved -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_models::quant::ggml::tests::**::* -->
<!-- test: larql_models::quant::mxfp4::tests::**::* -->
<!-- test: larql_models::quant::fp4_block::tests::**::* -->
<!-- test: larql_models::quant::fp4::tests::**::* -->
<!-- test: larql_models::quant::fp8::tests::**::* -->
<!-- test: larql_models::quant::half::tests::**::* -->
