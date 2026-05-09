## ADDED Requirements

### Requirement: preallocate_kv_cache_per_layer SHALL be idempotent

`CudaBackend::preallocate_kv_cache_per_layer` SHALL re-use the existing K/V cache when the requested shape matches. Specifically, when the cache exists AND its `(num_kv_heads, head_dim)` per-layer shapes AND its `max_seq` match the requested values, the function SHALL only reset `cache.len` to 0; it SHALL NOT re-allocate the device-resident slabs. Mismatched shape SHALL still trigger a fresh allocation.

#### Scenario: repeated prealloc with same shape skips allocation

- **WHEN** `preallocate_kv_cache_per_layer` is called twice
  in a row with the same `shapes` and `max_seq` (the bench
  harness's pattern: every prefill_start triggers one)
- **THEN** the second call SHALL NOT allocate fresh K/V
  device buffers (verified empirically: prefill drops from
  97.3 ms to 18.0 ms when the per-call ~38 ms PCIe transfer
  is eliminated)
<!-- test: unbacked -->

### Requirement: K/V cache zero-init SHALL use device-side memset

`CudaKvCache::new_device` SHALL initialise the per-layer K/V slabs via `Driver::device_alloc` (`cuMemAllocAsync` + `memset_d8_async`, HBM-bound) rather than `htod_f32(&zeros)` (PCIe-bound). On the dev box's 1.088 GB Gemma 3 4B cache size this saves ~36 ms on every fresh allocation (38 ms PCIe → ~2 ms HBM).

#### Scenario: device-side zero-init replaces host-zero htod

- **WHEN** `CudaKvCache::new_device` is called for the first
  time in a session
- **THEN** zero-initialisation SHALL go through
  `Driver::device_alloc` (HBM-bound) rather than
  `Driver::device_buf_from(&vec![0.0; n])` (PCIe-bound)
<!-- test: unbacked -->

### Requirement: prefill_q4_seq_device SHALL fast-path seq_len=1 to decode_token_device

`prefill_q4_seq_device` SHALL detect `seq_len == 1` and delegate to `decode_token_device` after resetting the K/V cache. The batched cuBLAS f32 GEMM path is significantly slower than the per-vector `__dp4a` mmvq kernel for M=1; the bench harness's first-token prefill (the hot case) is therefore covered by the optimised decode path.

#### Scenario: seq_len=1 prefill produces decode-equivalent output

- **WHEN** `prefill_q4_seq_device` is called with
  `seq_len = 1`
- **THEN** the returned hidden state SHALL bit-equal the
  output of `decode_token_device` called on the same input
  with a freshly-reset K/V cache
<!-- test: unbacked -->
