## ADDED Requirements

### Requirement: KV cache becomes parameterised by KvFormat

The KV cache type used by `larql_inference::attention` SHALL accept
a `KvFormat` parameter (`Fp16` (default), `Iso3`, `Planar3`, `Iso4`,
`Planar4`). The format is fixed for the lifetime of a cache instance;
changing it MUST require allocating a fresh cache.

KV-surgery operations (`get_layer`, `set_layer`,
`clone_layer_position_range`) MUST work transparently across formats:
get/clone returns reconstructed FP16, set takes FP16 and quantises on
insert.

#### Scenario: Iso3 cache reads back as FP16
- **WHEN** a layer is read from an Iso3 cache via `get_layer(layer_id)`
- **THEN** the result SHALL be an FP16 tensor reconstructed from the quantised storage
<!-- test: larql_inference::attention::decode::tests::kv_cache_format_is_fixed_at_construction -->
<!-- test: larql_inference::attention::decode::tests::get_layer_returns_fp32_after_compress -->
<!-- test: larql_inference::attention::decode::tests::quantize_then_dequantize_roundtrip_preserves_direction -->

#### Scenario: Cross-format cloning is allowed via FP16 round-trip
- **WHEN** a position range is cloned from an Iso3 cache into a Planar3 cache via the surgery API
- **THEN** the source range SHALL be dequantised to FP16, then quantised into the target format; the destination SHALL pass round-trip cosine ≥ 0.99
<!-- test: larql_inference::attention::decode::tests::clone_layer_position_range_slices_donor -->
<!-- test: larql_inference::attention::decode::tests::clone_layer_position_range_cross_format_round_trips -->

### Requirement: Attention forward pass dispatches the format-specific path on the active backend

The attention forward pass SHALL dispatch the format-specific
quantize-on-write and dequantize-with-inverse-rotation-on-read
kernels when the cache format is `Iso3` / `Planar3` / `Iso4` /
`Planar4`. The CPU backend MAY fall back to a slow scalar reference
implementation to keep correctness tests passing without a GPU.

#### Scenario: CUDA backend uses the production kernel for Iso3
- **WHEN** the active backend is CUDA and the cache format is Iso3
- **THEN** the forward pass SHALL invoke the vendored RotorQuant Iso3 kernels (verified by inspecting the kernel-launch trace)
<!-- test: unbacked -->

#### Scenario: CPU backend uses the scalar reference for Iso3 (slow but correct)
- **WHEN** the active backend is CPU and the cache format is Iso3
- **THEN** the forward pass SHALL produce the same hidden state to within 1e-3 absolute, at significantly reduced throughput
<!-- test: larql_inference::attention::decode::tests::set_layer_quantizes_when_format_active -->
