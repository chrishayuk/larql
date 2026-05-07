## ADDED Requirements

### Requirement: KvCache MUST expose a KvFormat parameter

`larql_inference::attention::decode::KvCache` SHALL gain a
`kv_format: Option<KvFormat>` field plus a `set_kv_format(format)`
setter. The parameter governs which `larql_rotorquant` format is
used by `quantize_layer`. `kv_format` MUST be `None` after
construction (`with_layers` / `with_window`) to preserve the FP32
default.

#### Scenario: quantize_layer is a no-op when kv_format is unset
- **WHEN** a layer is filled with FP32 data and `quantize_layer(layer)` is called without first calling `set_kv_format`
- **THEN** the call SHALL return `false`, the FP32 slot SHALL remain populated, and `is_layer_compressed(layer)` SHALL return `false`
<!-- test: larql_inference::attention::decode::tests::quantize_layer_no_op_when_format_unset -->

### Requirement: quantize_layer + dequantize_layer round-trip preserves direction

The KvCache SHALL take the FP32 slot (set to `None`) and store a `(QuantizedKv, QuantizedKv)` in the parallel `quantized_kv` side-table after `set_kv_format(format)` and `quantize_layer(layer)`. A subsequent `dequantize_layer(layer)` SHALL produce an FP32 `SharedKV` whose K and V cosine similarity to the original input are each ≥ 0.95.

#### Scenario: Iso3 quantize/dequantize roundtrip
- **WHEN** an FP32 layer is set, the cache's format set to `KvFormat::Iso3`, the layer compressed, then dequantised
- **THEN** the recovered K and V tensors SHALL match the original within cosine ≥ 0.95
<!-- test: larql_inference::attention::decode::tests::quantize_then_dequantize_roundtrip_preserves_direction -->

### Requirement: promote_layer_to_fp32 inverts quantize_layer

`promote_layer_to_fp32(layer)` SHALL move a compressed layer back
into the FP32 `layers` slot and clear the corresponding
`quantized_kv` slot, restoring the cache to a pre-`quantize_layer`
state (modulo the codebook / rotation noise from the round-trip).

#### Scenario: promote restores FP32 slot and clears compressed
- **WHEN** a layer is compressed via `quantize_layer` and then promoted via `promote_layer_to_fp32`
- **THEN** `is_layer_compressed(layer)` SHALL return `false`, `get_layer(layer)` SHALL return `Some`, and the layer SHALL be available for the existing FP32 attention paths
<!-- test: larql_inference::attention::decode::tests::promote_layer_to_fp32_restores_layers_slot -->
