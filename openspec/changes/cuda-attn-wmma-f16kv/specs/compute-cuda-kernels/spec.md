## ADDED Requirements

### Requirement: CUDA decode K/V cache SHALL store f16 elements

`cuda::decode::CudaKvLayer::{k, v}` SHALL be typed
`CudaSlice<half::f16>`. `CudaKvCache::new_device` SHALL allocate
each layer's K/V slab via `stream.alloc_zeros::<half::f16>(n)`
where `n = max_seq * num_kv_heads * head_dim`. The on-device
HBM footprint of each slab MUST be `n × 2` bytes (vs the
legacy `n × 4` for f32).

#### Scenario: cache slab is half the size of the legacy f32 slab

- **WHEN** a `CudaKvCache` is allocated for Gemma 3 4B
  (max_seq = 4096, 34 layers × (4 kv heads × 256 head_dim))
- **THEN** the total K + V device footprint SHALL be
  ≈ 660 MB (was ≈ 1.32 GB pre-change)
<!-- test: unbacked -->

### Requirement: fused decode attention SHALL read/write the f16 cache via cvt PTX

`FUSED_DECODE_ATTN_SRC` and `FUSED_PREFILL_ATTN_SRC` SHALL take
`unsigned short*` (or `const unsigned short*`) for `k_cache` and
`v_cache`. Cache reads SHALL go through a `cvt.f32.f16` device
helper (`ld_kvcache` for decode, `ld_kvc_pf` for prefill) that
returns f32. Cache writes (decode: `st_kvcache`; prefill writer
`KV_CACHE_WRITE_SEQ_SRC`'s `st_kv_seq`) SHALL use
`cvt.rn.f16.f32`.

#### Scenario: parity holds against the f32-cache reference within 5e-3

- **WHEN** `fused_decode_attention_matches_cpu_reference` runs
  with the new f16-cache CUDA path
- **THEN** the K cache, V cache, and attention output SHALL
  match the host f32 reference within 5e-3 max-element
  absolute difference (loosened from the legacy 1e-3 / 1e-6
  bounds to absorb f16 quantisation noise)
<!-- test: larql_compute::tests::test_cuda_attn::fused_decode_attention_matches_cpu_reference -->

### Requirement: legacy host-fallback decode path SHALL bridge f32 and f16 via host-boundary helpers

`CudaBackend` SHALL expose `htod_f32_as_f16_into_slice` (host f32
slice into device f16 buffer with round-to-nearest convert) and
`dtoh_f16_as_f32` (device f16 buffer into host f32 vector with
convert). The DecodeBackend `populate_kv_layer` impl and the
legacy host-fallback `decode_token` path SHALL use these to
preserve their existing host-f32 contract over the new f16
cache storage.

#### Scenario: host-fallback decode preserves bit-equivalent output

- **WHEN** `decode_token_phase1_matches_host_fallback` runs
  with the new f16 cache (default decode path) and again with
  `LARQL_CUDA_DECODE_HOST_FALLBACK=1` (legacy host-fallback
  path through the f16↔f32 bridge)
- **THEN** per-step max-element absolute difference SHALL be
  ≤ 1e-3
<!-- test: larql_compute::tests::test_cuda_decode::decode_token_phase1_matches_host_fallback -->
