## ADDED Requirements

### Requirement: CudaKvCache MUST expose a KvFormat parameter

`larql_compute::cuda::decode::CudaKvCache` SHALL gain a
`kv_format: Option<KvFormat>` field plus a `set_kv_format(format)`
setter. The parameter governs which `larql_rotorquant` format is
used by `quantize_layer`. `kv_format` MUST default to `None` after
construction (`new_device` / `preallocate_kv_cache_per_layer`) so
the canonical f16 path is bit-exact preserved at every existing
call site.

#### Scenario: quantize_layer is a no-op when kv_format is unset

- **WHEN** a `CudaKvCache` layer is populated with f16 K/V data and `quantize_layer(layer)` is called without first calling `set_kv_format`
- **THEN** the call SHALL return `false`, the f16 slab SHALL remain populated, and `is_layer_compressed(layer)` SHALL return `false`
<!-- test: larql_compute::test_cuda_kv_rotorquant::cuda_kvcache_quantize_layer_no_op_when_format_unset -->

### Requirement: CudaKvCache quantize_layer + dequantize_layer round-trip MUST preserve direction

`CudaKvCache::quantize_layer` SHALL move a populated f16 layer into the parallel `quantized_kv` side-table as a `(DeviceQuantizedKv, DeviceQuantizedKv)` pair and SHALL release the corresponding f16 slab's device memory. A subsequent `dequantize_layer(layer)` SHALL produce f16 K and V slices whose cosine similarity to the original input data is each ≥ 0.95 for 3-bit packed formats (Iso3, Planar3) and ≥ 0.98 for 4-bit packed formats (Iso4, Planar4).

#### Scenario: Iso3 quantize then dequantize round-trip on CudaKvCache preserves direction

- **WHEN** an f16 layer of synthetic Gemma-shaped data is set, the cache's format set to `KvFormat::Iso3`, the layer compressed via `quantize_layer`, then read back via `dequantize_layer`
- **THEN** the recovered K and V slices SHALL match the original within cosine ≥ 0.95
<!-- test: larql_compute::test_cuda_kv_rotorquant::cuda_kvcache_iso3_quantize_then_dequantize_roundtrip_preserves_direction -->

#### Scenario: Planar3 quantize then dequantize round-trip on CudaKvCache preserves direction

- **WHEN** an f16 layer of synthetic Gemma-shaped data is set, the cache's format set to `KvFormat::Planar3`, the layer compressed via `quantize_layer`, then read back via `dequantize_layer`
- **THEN** the recovered K and V slices SHALL match the original within cosine ≥ 0.95
<!-- test: larql_compute::test_cuda_kv_rotorquant::cuda_kvcache_planar3_quantize_then_dequantize_roundtrip_preserves_direction -->

### Requirement: CudaKvCache promote_layer_to_fp32 inverts quantize_layer

`promote_layer_to_fp32(layer)` SHALL move a compressed layer back
into the f16 `layers` slot (allocating fresh device memory of the
original shape) and clear the corresponding `quantized_kv` slot,
restoring the cache to a pre-`quantize_layer` state (modulo the
codebook / rotation noise from the round-trip).

#### Scenario: promote restores f16 slot and clears compressed entry

- **WHEN** a `CudaKvCache` layer is compressed via `quantize_layer` and then promoted via `promote_layer_to_fp32`
- **THEN** `is_layer_compressed(layer)` SHALL return `false`, the f16 slab in `layers[layer]` SHALL be repopulated, and `quantized_kv[layer]` SHALL be `None`
<!-- test: larql_compute::test_cuda_kv_rotorquant::cuda_kvcache_promote_layer_to_fp32_restores_f16_slot -->

### Requirement: CUDA decode_attention MUST transparently dequantize compressed layers on read

`larql_compute::cuda::attn::decode_attention` SHALL route compressed layer reads through `CudaKvCache::dequantize_layer`, producing a transient f16 scratch buffer, before invoking the existing attention kernel. The output of the attention kernel SHALL match (cosine ≥ 0.95) the output that would have been produced by reading the same layer's data uncompressed.

#### Scenario: attention output with one compressed layer matches uncompressed within cosine 0.95

- **WHEN** a `CudaKvCache` layer is populated with synthetic f16 K/V and attention is run twice — once with the layer left as f16 and once with the same data compressed via `set_kv_format(KvFormat::Iso3)` + `quantize_layer`
- **THEN** the cosine similarity between the two attention output tensors SHALL be ≥ 0.95
<!-- test: larql_compute::test_cuda_kv_rotorquant::cuda_decode_attention_with_compressed_layer_matches_uncompressed_within_cosine_0_95 -->

### Requirement: decode_tokens_speculative MUST leave compressed layers unaffected

`DecodeBackend::decode_tokens_speculative` SHALL continue to write to and read from the f16 KV slab on `CudaKvCache`. Compressed layers SHALL be treated as read-only views of past committed K/V; speculative writes during the helper's internal advance-and-rollback dance SHALL NOT disturb the `quantized_kv` side-table.

#### Scenario: speculative dispatch with a compressed layer leaves the compressed slot intact

- **WHEN** a `CudaKvCache` has one layer compressed (`is_layer_compressed(L) == true`) and `decode_tokens_speculative` is invoked with N tokens
- **THEN** after the call returns, `is_layer_compressed(L)` SHALL still return `true` and `quantized_kv[L]` SHALL be byte-identical to its pre-call state
<!-- test: unbacked -->
