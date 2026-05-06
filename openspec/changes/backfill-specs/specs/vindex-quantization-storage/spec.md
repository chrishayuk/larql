## ADDED Requirements

### Requirement: FP4 E2M1 nibble-packed block format

`larql_vindex::format::fp4_codec` SHALL encode FFN projections as
FP4 E2M1 values, four-bit nibble-packed two values per byte, in
256-element superblocks composed of eight 32-element sub-blocks.
Each superblock MUST carry a per-superblock `fp8_e4m3` master scale
plus eight per-sub-block `fp8_e4m3` scales, in the canonical
`fp4_e2m1_mxfp4_nibble_order` value encoding. Per-layer layouts
MUST place every layer at a deterministic byte offset so the
on-disk file size SHALL equal the sum of per-layer block bytes
plus per-layer scale bytes.

#### Scenario: FP4 projection round-trips through the codec
- **WHEN** a synthetic projection is encoded with `write_fp4_projection` and read back with `read_fp4_projection`
- **THEN** the decoded values SHALL agree with the source within FP4 quantisation error per row
<!-- test: larql_vindex::format::fp4_codec::tests::fp4_projection_round_trip -->
<!-- test: larql_vindex::format::fp4_codec::tests::fp4_projection_non_uniform_widths -->

#### Scenario: FP4 file size matches the published format spec
- **WHEN** `fp4_layer_layouts` is computed for a set of per-layer feature counts at a fixed hidden size
- **THEN** the projected on-disk file size SHALL match the spec's per-layer byte budget exactly
<!-- test: larql_vindex::format::fp4_codec::tests::fp4_file_size_matches_spec -->
<!-- test: larql_vindex::format::fp4_codec::tests::fp4_layer_layouts_matches_file_offsets -->

#### Scenario: Reader rejects mismatched payloads
- **WHEN** `read_fp4_projection` is given a buffer whose size disagrees with the layout it computed
- **THEN** the reader SHALL return a structured error rather than read past the end of the buffer
<!-- test: larql_vindex::format::fp4_codec::tests::fp4_reader_rejects_wrong_size -->

### Requirement: FP8 down-projection storage

The codec SHALL provide an FP8 `e4m3` row-major variant for the
down projection in Option B mixed-precision policies, with the
same per-layer offset discipline as the FP4 path. FP8 round-trip
error MUST be lower than the FP4 round-trip error on the same
input so callers can pick FP8 when accuracy is the dominant
constraint.

#### Scenario: FP8 projection round-trips with bounded error
- **WHEN** a synthetic projection is encoded via `write_fp8_projection` and decoded via `read_fp8_projection`
- **THEN** the result SHALL match the source within FP8 quantisation error and the on-disk size SHALL match `fp8_layer_layouts`
<!-- test: larql_vindex::format::fp4_codec::tests::fp8_projection_round_trip -->
<!-- test: larql_vindex::format::fp4_codec::tests::fp8_file_size_matches_spec -->

### Requirement: Q4_K and Q6_K streaming weight writers

`larql_vindex::format::weights::write_q4k` SHALL stream-quantise
attention and FFN projections into GGML-compatible Q4_K
(144-byte block, 256 elements) and Q6_K (210-byte block, 256
elements) files without buffering the full weight matrix. The
writer MUST pad rows that are not block-aligned with zeros, MUST
fall back from V to K weights when the architecture declares
`v_shares_k`, and MUST reject architectures (such as MLA) that
the streaming writer does not support.

#### Scenario: V projection falls back to K when V shares K
- **WHEN** `resolve_v_weights` is called on a layer whose architecture reports `v_shares_k(layer) == true` and the safetensors shard contains only `k_proj`
- **THEN** the writer SHALL emit V weights derived from the K weights, not panic on a missing key
<!-- test: larql_vindex::format::weights::write_q4k::tests::resolve_v_falls_back_to_k_when_v_shared -->
<!-- test: larql_vindex::format::weights::write_q4k::tests::resolve_v_returns_v_when_present -->
<!-- test: larql_vindex::format::weights::write_q4k::tests::resolve_v_none_when_missing_and_not_shared -->
<!-- test: larql_vindex::format::weights::write_q4k::tests::resolve_v_none_when_v_missing_and_k_missing -->

#### Scenario: Block padding is exact on every alignment boundary
- **WHEN** a row of arbitrary length is passed through `pad_to_block`
- **THEN** the padded length SHALL be the smallest multiple of `K_QUANT_BLOCK_ELEMS` that fits, with zeros in the trailing slots
<!-- test: larql_vindex::format::weights::write_q4k::tests::pad_to_block_noop_when_exact_multiple -->
<!-- test: larql_vindex::format::weights::write_q4k::tests::pad_to_block_zero_fills_to_next_block -->
<!-- test: larql_vindex::format::weights::write_q4k::tests::pad_to_block_handles_one_below_multiple -->
<!-- test: larql_vindex::format::weights::write_q4k::tests::pad_to_block_handles_one_above_multiple -->
<!-- test: larql_vindex::format::weights::write_q4k::tests::pad_to_block_empty_input_stays_empty -->

#### Scenario: MLA architectures are rejected up front
- **WHEN** `validate_capabilities` is called on a DeepSeek-style architecture that uses MLA attention
- **THEN** the writer SHALL return an error before opening any output file
<!-- test: larql_vindex::format::weights::capabilities::tests::mla_architecture_is_rejected -->
<!-- test: larql_vindex::format::weights::capabilities::tests::standard_attention_accepts_llama -->

#### Scenario: GGML round-trip preserves block payloads
- **WHEN** Q4_K and Q6_K block payloads are encoded and decoded via the registry-backed kernels
- **THEN** the recovered values SHALL match the source within the published block error bound, with Q6_K strictly more accurate than Q4_K on the same input
<!-- test: larql_vindex::quant_roundtrip::q4_k_roundtrip_one_block -->
<!-- test: larql_vindex::quant_roundtrip::q4_k_roundtrip_many_blocks -->
<!-- test: larql_vindex::quant_roundtrip::q6_k_roundtrip_one_block -->
<!-- test: larql_vindex::quant_roundtrip::q6_k_roundtrip_many_blocks -->
<!-- test: larql_vindex::quant_roundtrip::q6_k_more_accurate_than_q4_k -->
<!-- test: larql_vindex::quant_roundtrip::q4_0_roundtrip_one_block -->
<!-- test: larql_vindex::quant_roundtrip::q4_0_roundtrip_many_blocks -->

### Requirement: Quant format registry

`larql_vindex::quant::registry` SHALL be the single source of
truth that maps a string tag (e.g. `"Q4_K"`, `"Q6_K"`) to its
block geometry, byte stride, dequantize function pointer, and
matmul/row-dot kernels. Tags MUST be unique, lookup MUST be
case-sensitive, `bytes_per_row` MUST require column counts to be
block-aligned, and `expected_bytes` MUST reject any 2-D shape
whose column count is not block-aligned. The registry MUST NOT
report the legacy 148-byte Q4_K stride as a valid expected size,
so that stale vindexes are caught at load time rather than
producing NaN GPU output.

#### Scenario: Registry has no duplicate tags
- **WHEN** the in-tree registry is enumerated
- **THEN** each tag SHALL appear exactly once
<!-- test: larql_vindex::quant::registry::tests::registry_tags_unique -->

#### Scenario: Lookup is case-sensitive and rejects unknown tags
- **WHEN** `lookup` is called with a known mixed-case tag, a typo, or empty string
- **THEN** only the exact published spelling SHALL return `Some`, all others return `None`
<!-- test: larql_vindex::quant::registry::tests::lookup_known_formats -->
<!-- test: larql_vindex::quant::registry::tests::lookup_unknown_returns_none -->

#### Scenario: Byte sizing matches Gemma 3 4B layer-0 projections
- **WHEN** `expected_bytes` is queried with the Gemma 3 4B q/k/v shapes against Q4_K and Q6_K
- **THEN** the byte count SHALL equal `rows * (cols / 256) * bytes_per_block` and SHALL reject non-block-aligned columns or non-2-D shapes
<!-- test: larql_vindex::quant::registry::tests::bytes_per_row_block_aligned -->
<!-- test: larql_vindex::quant::registry::tests::expected_bytes_q4k_gemma3_4b_q_proj -->
<!-- test: larql_vindex::quant::registry::tests::expected_bytes_q4k_gemma3_4b_k_proj -->
<!-- test: larql_vindex::quant::registry::tests::expected_bytes_q6k_v_proj -->
<!-- test: larql_vindex::quant::registry::tests::expected_bytes_rejects_non_2d_shape -->
<!-- test: larql_vindex::quant::registry::tests::expected_bytes_rejects_non_block_aligned_cols -->

#### Scenario: Legacy 148-byte stride is not silently accepted
- **WHEN** an `expected_bytes` value computed against the current 144-byte stride is compared with the legacy 148-byte stride for the same shape
- **THEN** the values SHALL differ, so the loader's `expected != length` check fires on stale vindexes
<!-- test: larql_vindex::quant::registry::tests::expected_bytes_does_not_match_legacy_148_byte_stride -->

### Requirement: Q1 compliance gate (read-only measurement)

`larql_vindex::quant::scan` SHALL be a pure measurement pass that
reports per-projection compliance fractions against a set of
ratio thresholds without mutating any vindex artefact. Bucket
geometry (tile size, top-k offenders, threshold list) SHALL be
pinned via `ScanConfig::default()` so that scan reports are
reproducible across runs.

#### Scenario: Bucket compliance counts all-zero blocks as compliant
- **WHEN** a bucket containing some non-zero ratios and at least one all-zero block is queried at a ratio threshold
- **THEN** `compliance_at(threshold)` SHALL include the all-zero blocks in the compliant numerator
<!-- test: larql_vindex::quant::scan::tests::bucket_compliance_fraction -->

#### Scenario: Empty bucket yields safe quantiles
- **WHEN** `quantiles()` is called on a default (empty) bucket
- **THEN** it SHALL return `total_blocks == 0` and a NaN mean rather than panic
<!-- test: larql_vindex::quant::scan::tests::bucket_quantiles_empty_ok -->

#### Scenario: Default scan config pins geometry
- **WHEN** `ScanConfig::default()` is constructed
- **THEN** the tile size, top-k offenders count, and threshold list lengths SHALL match the published defaults
<!-- test: larql_vindex::quant::scan::tests::config_defaults_pin_geometry -->

### Requirement: FP4/FP8 precision policies and conversion

`larql_vindex::quant::convert` SHALL expose Policies A
(all-fp4), B (default: gate=source, up=fp4, down=fp8), and C
(gate=source, up=fp4, down=f16) and convert a base vindex into
its FP4 form via `vindex_to_fp4`. Conversion MUST refuse to
overwrite an existing destination unless `force` is set, MUST
hardlink unchanged base files when the filesystem supports it,
SHALL emit a `fp4_compliance.json` sidecar by default, and SHALL
honour the `--no-sidecar` toggle. `vindex_to_q4k` SHALL refuse
sources lacking model weights, refuse already-quantised sources,
and SHALL default to the Q4K-M mix (Q4_K for q/k/up/gate; Q6_K
for v/down) with a feature-major down round-trip path enabled
on demand.

#### Scenario: Policy precisions keep gate at source dtype
- **WHEN** `Policy::A`, `Policy::B`, and `Policy::C` are queried with a source dtype
- **THEN** the gate precision in each policy SHALL equal the source dtype, while up/down SHALL match the policy's published table (B = fp4 up + fp8 down; C = fp4 up + f16 down)
<!-- test: larql_vindex::quant::convert::tests::policy_precisions_keep_gate_source -->
<!-- test: larql_vindex::quant::convert::tests::policy_b_is_fp4_up_fp8_down -->

#### Scenario: Policy parser accepts short and long forms
- **WHEN** `Policy::parse` is called with `"a"`, `"A"`, `"option-b"`, or an unknown string
- **THEN** valid spellings SHALL parse to their policy and unknown spellings SHALL return an error
<!-- test: larql_vindex::quant::convert::tests::policy_parse_accepts_short_forms -->

#### Scenario: Default config is Option B with full sidecar emission
- **WHEN** `Fp4ConvertConfig::default()` is constructed
- **THEN** the policy SHALL be B, compliance floor 0.99, threshold 16.0, sidecar emission enabled, and force/strict flags off
<!-- test: larql_vindex::quant::convert::tests::default_config_is_option_b -->

#### Scenario: vindex_to_fp4 refuses to overwrite without force
- **WHEN** `vindex_to_fp4` is invoked against a destination directory that already exists
- **THEN** conversion SHALL error unless `force` is set, and SHALL succeed when `force` is set
<!-- test: larql_vindex::test_vindex_to_fp4::vindex_to_fp4_refuses_existing_output -->
<!-- test: larql_vindex::test_vindex_to_fp4::vindex_to_fp4_force_overwrites_existing -->

#### Scenario: vindex_to_fp4 emits or skips the sidecar per config
- **WHEN** `vindex_to_fp4` is run end-to-end against a synthetic source vindex with the default config and again with sidecar emission disabled
- **THEN** the default run SHALL produce `fp4_compliance.json` and the disabled run SHALL omit it
<!-- test: larql_vindex::test_vindex_to_fp4::vindex_to_fp4_option_b_smoke -->
<!-- test: larql_vindex::test_vindex_to_fp4::vindex_to_fp4_no_sidecar_skips_emission -->

#### Scenario: vindex_to_q4k refuses unsuitable sources
- **WHEN** `vindex_to_q4k` is invoked against a vindex that lacks model weights, or that is already quantised, or whose destination already exists without `force`
- **THEN** conversion SHALL error in each of those cases
<!-- test: larql_vindex::test_vindex_to_q4k::q4k_refuses_existing_output_without_force -->
<!-- test: larql_vindex::test_vindex_to_q4k::q4k_refuses_source_without_model_weights -->
<!-- test: larql_vindex::test_vindex_to_q4k::q4k_refuses_already_quantised_source -->

#### Scenario: vindex_to_q4k defaults match the Q4K-M mix
- **WHEN** `Q4kConvertConfig::default()` is constructed
- **THEN** `down_q4k` SHALL be false (Q6_K stays on the down projection) and the opt-in flag SHALL toggle correctly when set
<!-- test: larql_vindex::test_vindex_to_q4k::q4k_config_defaults_match_q4k_m_mix -->
<!-- test: larql_vindex::quant::convert_q4k::tests::default_config_is_q4k_m_mix -->
<!-- test: larql_vindex::quant::convert_q4k::tests::down_q4k_opt_in_toggles_flag -->

#### Scenario: vindex_to_q4k end-to-end produces a feature-major down round-trip
- **WHEN** `vindex_to_q4k` is run against a synthetic Llama safetensors source and `add_feature_major_down` is invoked on the result
- **THEN** the conversion SHALL succeed and the feature-major down file SHALL round-trip back to the source within Q4_K error
<!-- test: larql_vindex::test_vindex_to_q4k::q4k_end_to_end_from_synthetic_safetensors -->
<!-- test: larql_vindex::test_vindex_to_q4k::q4k_feature_major_down_round_trip -->

### Requirement: FP4 storage and per-projection precision dispatch

`larql_vindex::index::storage::fp4_store::Fp4Storage` SHALL load
mmap-backed FP4/FP8 layer files described by a manifest-driven
`Fp4Config`, expose a per-component (`gate`, `up`, `down`)
precision query, and dispatch `dequant_row_into`, `row_dot`, and
`row_scaled_add` to the matching backend. Loading MUST validate
file sizes against the layout, reject missing files, and reject
out-of-range layer/feature indices. F16 down projections SHALL
fall through to the legacy path rather than mmap'ing as FP8.

#### Scenario: Loader rejects missing or wrong-sized files
- **WHEN** `Fp4Storage::load` is invoked with the gate or up file missing, or with a file whose size disagrees with the layout
- **THEN** it SHALL return an error
<!-- test: larql_vindex::index::storage::fp4_store::tests::load_rejects_missing_files -->
<!-- test: larql_vindex::index::storage::fp4_store::tests::load_validates_file_sizes -->

#### Scenario: Per-component precision and mmap dispatch is correct
- **WHEN** `precision(component)` and the internal `mmap_for(component)` are queried for gate, up, and down on an Option B storage
- **THEN** gate and up SHALL report FP4 with mmaps present and down SHALL report FP8 with its own mmap
<!-- test: larql_vindex::index::storage::fp4_store::tests::precision_and_mmap_dispatch_per_component -->

#### Scenario: Feature byte ranges match the format spec
- **WHEN** `feature_byte_range` is computed for several layers and feature offsets
- **THEN** the returned ranges SHALL match the per-layer offset table from `fp4_layer_layouts`
<!-- test: larql_vindex::index::storage::fp4_store::tests::feature_byte_range_matches_format_spec -->

#### Scenario: Dequant matches source within FP4 error
- **WHEN** `dequant_row_into` is called for a synthetic feature whose source vector is known
- **THEN** the decoded buffer SHALL match the source within FP4 quantisation error, and SHALL be rejected with a structured error when the output buffer is the wrong length or the layer/feature index is out of range
<!-- test: larql_vindex::index::storage::fp4_store::tests::dequant_row_into_matches_source -->
<!-- test: larql_vindex::index::storage::fp4_store::tests::dequant_row_into_rejects_bad_out_length -->
<!-- test: larql_vindex::index::storage::fp4_store::tests::dequant_row_into_rejects_out_of_range -->

#### Scenario: row_dot and row_scaled_add agree with manual dequant
- **WHEN** `row_dot` and `row_scaled_add` are computed against an externally-decoded reference
- **THEN** the values SHALL agree within float tolerance and the kernels SHALL reject mismatched x/out lengths
<!-- test: larql_vindex::index::storage::fp4_store::tests::row_dot_agrees_with_dequant_plus_manual_dot -->
<!-- test: larql_vindex::index::storage::fp4_store::tests::row_dot_rejects_wrong_x_length -->
<!-- test: larql_vindex::index::storage::fp4_store::tests::row_scaled_add_accumulates_correctly -->
<!-- test: larql_vindex::index::storage::fp4_store::tests::row_scaled_add_rejects_bad_out_length -->

#### Scenario: F16 down projections use the legacy path
- **WHEN** an Option C storage with f16 down is loaded
- **THEN** the down mmap SHALL be absent and `dequant_row_into` for the down component SHALL fall through (return false) so the legacy path can handle it
<!-- test: larql_vindex::index::storage::fp4_store::tests::load_handles_f16_projection_tag_without_mmap -->

#### Scenario: Non-uniform per-layer feature counts dequantise correctly
- **WHEN** `dequant_row_into` is exercised across layers with different feature counts (E2B-style)
- **THEN** every per-layer feature SHALL decode to the source values within FP4 error
<!-- test: larql_vindex::index::storage::fp4_store::tests::non_uniform_layer_widths_dequant_correctly -->

### Requirement: Q4_K compute dispatch

`larql_vindex::index::compute::q4k_dispatch` SHALL provide
matmul, FFN row-dot, FFN row-scaled-add, FFN row-into, and
feature-scaled-add kernels that operate directly on Q4_K block
storage. The dispatch SHALL refuse component identifiers it does
not handle, reject mismatched output buffer lengths, and SHALL
report a soft failure (`false`) when the underlying down store
is not loaded so callers can fall back to a different backend.

#### Scenario: row_scaled_add only accepts gate (0) and up (1) components
- **WHEN** `q4k_ffn_row_scaled_add` is invoked with component id 2 (down)
- **THEN** it SHALL reject the call with a structured error
<!-- test: larql_vindex::index::compute::q4k_dispatch::tests::q4k_ffn_row_scaled_add_rejects_component_2 -->

#### Scenario: row_scaled_add validates output length
- **WHEN** `q4k_ffn_row_scaled_add` is invoked with an out buffer whose length does not match the projection width
- **THEN** it SHALL reject the call with a structured error
<!-- test: larql_vindex::index::compute::q4k_dispatch::tests::q4k_ffn_row_scaled_add_rejects_wrong_out_len -->

#### Scenario: down-feature add returns false when down store unloaded
- **WHEN** `q4k_down_feature_scaled_add` is invoked on a vindex whose down store has not been loaded
- **THEN** it SHALL return false rather than panic, signalling fallback to the legacy path
<!-- test: larql_vindex::index::compute::q4k_dispatch::tests::q4k_down_feature_scaled_add_returns_false_when_unloaded -->

### Requirement: Quant manifest, precision config, and compliance gate config

`larql_vindex::format::weights::manifest::QuantManifest` SHALL be
the on-disk descriptor for streamed Q4_K/Q6_K weights, recording
the format tag, padded width, and per-tensor metadata. Format
tags MUST match the on-disk strings, missing fields MUST fail to
parse, and round-trip serialisation MUST equal the writer's wire
shape. `larql_vindex::config::quantization` SHALL define
`QuantFormat`, `Precision`, and `Fp4Config` with stable
serde casing (`fp4`, `fp8`, `f16`, `f32`) and Option B defaults.
`larql_vindex::config::compliance::LayerBands` SHALL classify
layers into syntax / knowledge / output bands per family with a
deterministic fallback for unknown families and SHALL refuse to
classify models with too few layers.

#### Scenario: QuantManifest round-trips and rejects malformed input
- **WHEN** a manifest is round-tripped through serialisation and when a manifest missing the `format` field is parsed
- **THEN** the round-trip SHALL match the writer's wire shape, format tags SHALL match the published on-disk strings, padded width SHALL extract the second dim, and the missing-format input SHALL fail to parse
<!-- test: larql_vindex::format::weights::manifest::tests::round_trip_matches_writer_wire_shape -->
<!-- test: larql_vindex::format::weights::manifest::tests::format_tag_matches_on_disk_strings -->
<!-- test: larql_vindex::format::weights::manifest::tests::padded_width_extracts_second_dim -->
<!-- test: larql_vindex::format::weights::manifest::tests::missing_format_field_fails_parse -->

#### Scenario: QuantFormat and Precision serde are stable
- **WHEN** `QuantFormat` and `Precision` values are serialised and deserialised
- **THEN** the on-disk casing SHALL be `none`, `q4k`, `fp4`, `fp8`, `f16`, `f32` and the default `QuantFormat` SHALL be `None`
<!-- test: larql_vindex::config::quantization::tests::quant_format_default_is_none -->
<!-- test: larql_vindex::config::quantization::tests::quant_format_display -->
<!-- test: larql_vindex::config::quantization::tests::quant_format_serde_round_trip -->
<!-- test: larql_vindex::config::quantization::tests::precision_display_all_variants -->
<!-- test: larql_vindex::config::quantization::tests::precision_serde_snake_case -->

#### Scenario: Fp4Config v1 defaults pin block geometry and Option B
- **WHEN** `Fp4Config::v1_defaults` and `Fp4Config::option_b_default` are constructed
- **THEN** the v1 defaults SHALL pin block geometry to 256/32 elements with `fp8_e4m3` scales and `fp4_e2m1_mxfp4_nibble_order` value encoding, Option B projections SHALL be fp4 gate / fp4 up / fp8 down, the compliance gate SHALL fall back to fp8 with a positive minimum compliant fraction, and the sidecar filename SHALL be `fp4_compliance.json`
<!-- test: larql_vindex::config::quantization::tests::fp4_config_v1_defaults_block_geometry -->
<!-- test: larql_vindex::config::quantization::tests::fp4_config_option_b_projection_precisions -->
<!-- test: larql_vindex::config::quantization::tests::fp4_config_compliance_gate_defaults -->
<!-- test: larql_vindex::config::quantization::tests::fp4_config_compliance_report_filename -->

#### Scenario: LayerBands classify per family and fall back gracefully
- **WHEN** `LayerBands::for_family` is queried for `gemma3` (34 layers), `llama` (32), `gpt2` (12), an unknown family with sufficient layers, and a model with too few layers
- **THEN** the published bands SHALL match for known families, the unknown family SHALL fall back to a partition that covers `[0, num_layers-1]`, the too-few-layers case SHALL return `None`, and out-of-range queries SHALL report `"unknown"`
<!-- test: larql_vindex::config::compliance::tests::gemma3_34_layer_bands -->
<!-- test: larql_vindex::config::compliance::tests::llama_32_layer_bands -->
<!-- test: larql_vindex::config::compliance::tests::gpt2_12_layer_bands -->
<!-- test: larql_vindex::config::compliance::tests::unknown_family_with_sufficient_layers_uses_fallback -->
<!-- test: larql_vindex::config::compliance::tests::too_few_layers_returns_none -->
<!-- test: larql_vindex::config::compliance::tests::band_for_layer_gemma3 -->
<!-- test: larql_vindex::config::compliance::tests::band_for_layer_out_of_range_is_unknown -->
<!-- test: larql_vindex::config::compliance::tests::layer_bands_serde_round_trip -->
<!-- test: larql_vindex::config::compliance::tests::compliance_gate_serde_round_trip -->

### Requirement: FP4 storage end-to-end with real and synthetic vindexes

A vindex saved with FP4 storage SHALL load with `Fp4Storage`
populated, dispatch FFN row dots through the FP4 backend for
gate/up and through the FP8 backend for down, and SHALL preserve
its FP4 storage across `Clone`. A vindex without FP4 sidecars
(legacy / non-Option-B) SHALL load without an FP4 store and
SHALL gracefully refuse to dispatch to FP4 kernels.

#### Scenario: Real-fixture vindex dispatches gate/up via FP4
- **WHEN** an FP4 fixture vindex is loaded and `row_dot` and `row_scaled_add` are exercised against the source f32 baseline
- **THEN** the FP4 backend SHALL produce values within FP4 quantisation error of the baseline
<!-- test: larql_vindex::test_fp4_storage::fp4_storage_loads_from_real_vindex -->
<!-- test: larql_vindex::test_fp4_storage::fp4_row_dot_matches_source_f32_baseline -->
<!-- test: larql_vindex::test_fp4_storage::fp4_row_scaled_add_matches_source_baseline -->

#### Scenario: Legacy vindex has no FP4 storage
- **WHEN** a non-FP4 vindex is loaded
- **THEN** `Fp4Storage` SHALL be absent on the result
<!-- test: larql_vindex::test_fp4_storage::fp4_storage_absent_on_legacy_vindex -->

#### Scenario: Synthetic vindex round-trips FP4 storage and dispatch
- **WHEN** a minimal synthetic vindex is built and reloaded from disk
- **THEN** `Fp4Storage` SHALL be present, gate/up FFN row dots SHALL dispatch to the FP4 backend, the down row dot SHALL dispatch to FP8, `row_scaled_add` and `row_into` SHALL match the source within error, OOB feature lookups SHALL return None, and the FP4 store SHALL survive cloning
<!-- test: larql_vindex::test_fp4_synthetic::minimal_synthetic_vindex_loads_fp4_storage -->
<!-- test: larql_vindex::test_fp4_synthetic::synthetic_ffn_row_dot_uses_fp4_backend -->
<!-- test: larql_vindex::test_fp4_synthetic::synthetic_ffn_row_dot_down_uses_fp8_backend -->
<!-- test: larql_vindex::test_fp4_synthetic::synthetic_ffn_row_scaled_add_matches_source -->
<!-- test: larql_vindex::test_fp4_synthetic::synthetic_ffn_row_into_decodes_correctly -->
<!-- test: larql_vindex::test_fp4_synthetic::synthetic_ffn_row_returns_none_on_oob -->
<!-- test: larql_vindex::test_fp4_synthetic::synthetic_num_features_never_zero_on_fp4_vindex -->
<!-- test: larql_vindex::test_fp4_synthetic::synthetic_cloned_index_preserves_fp4_storage -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: larql_vindex::test_fp4_storage::**::* -->
<!-- test: larql_vindex::test_fp4_synthetic::**::* -->
<!-- test: larql_vindex::test_vindex_to_fp4::**::* -->
<!-- test: larql_vindex::test_vindex_to_q4k::**::* -->
<!-- test: larql_vindex::quant_roundtrip::**::* -->
<!-- test: larql_vindex::format::fp4_codec::tests::**::* -->
<!-- test: larql_vindex::format::weights::write_q4k::**::* -->
<!-- test: larql_vindex::quant::registry::tests::**::* -->
<!-- test: larql_vindex::quant::scan::tests::**::* -->
<!-- test: larql_vindex::quant::convert::tests::**::* -->
<!-- test: larql_vindex::quant::convert_q4k::tests::**::* -->
<!-- test: larql_vindex::index::storage::fp4_store::tests::**::* -->
<!-- test: larql_vindex::index::compute::q4k_dispatch::tests::**::* -->
<!-- test: larql_vindex::config::quantization::tests::**::* -->
<!-- test: larql_vindex::config::compliance::tests::**::* -->
