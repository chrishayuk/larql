# cuda-prefill-alloc-fixes — tasks

## 1. Idempotent K/V cache prealloc

- [x] 1.1 `CudaKvCache::matches_shape` helper added.
- [x] 1.2 `preallocate_kv_cache_per_layer` checks the
      existing cache first; reuses with `len = 0` when the
      shape matches, allocates fresh only on mismatch.
- [x] 1.3 `CudaKvCache::new_device` uses `device_alloc`
      (HBM-bound memset) instead of `htod_f32(&zeros)`
      (PCIe-bound transfer).

## 2. seq_len=1 fast path

- [x] 2.1 `prefill_q4_seq_device` short-circuits to
      `decode_token_device` for `seq_len = 1`. The bench
      harness only prefills the first token via this entry
      point.

## 3. Tests + bench

- [x] 3.1 `decode_token_phase1_matches_host_fallback` still
      passes.
- [x] 3.2 Full CUDA test suite (193 tests) green.
- [x] 3.3 Bench measured: `prefill 18.0 ms` (was 97.3),
      `decode 10.4 ms/tok` (was 10.36), `tok/s 96.2`.

## 4. Documentation + archive

- [x] 4.1 Bench numbers in proposal.md.
- [ ] 4.2 Archive together with the related prefill changes.
