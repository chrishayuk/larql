## Why

The post-`cuda-prefill-batched-attention` bench reported
prefill = 97.3 ms, but the per-section profile only
accounted for ~40 ms. Investigating the 57 ms gap revealed
two correctness-grade inefficiencies:

1. **The bench harness re-allocates the K/V cache on every
   prefill_start.** `larql-inference/.../gpu.rs:127`
   unconditionally calls `backend.preallocate_kv_cache_per_layer`
   inside the prefill timing scope. The CUDA backend's impl
   was non-idempotent — it always replaced the cache with
   a fresh allocation. For Gemma 3 4B at `max_seq=4096 × 4
   KV-heads × 256 head-dim × f32 × 2 (K+V) × 34 layers`
   that's **1 088 MB** allocated and zero-initialised per
   prefill. With our previous `htod_f32(&zeros)` zero-init
   path that's a PCIe-bound ~38 ms transfer.

2. **`prefill_q4_seq_device` for `seq_len=1` uses cuBLAS f32
   GEMM-with-M=1 instead of the mmvq path.** The bench
   harness only prefills the first token via `prefill_q4`
   (the remaining prompt tokens go through the decode loop
   with forced-token feedback), so `seq_len=1` is the hot
   case. cuBLAS GEMM on M=1 is significantly slower than
   the per-vector `__dp4a` mmvq kernel.

Combined fix:

| | Pre-fix | Post-fix |
|---|---:|---:|
| `prefill ms` (bench, 6-tok prompt, seq_len=1 first-token) | 97.3 | **18.0** |
| `decode ms/tok` | 10.36 | 10.4 (no regression) |
| `tok/s` | 96.6 | 96.2 |

5.4× prefill speedup. Combined with the rest of today's CUDA
work, total prefill speedup vs the pre-LARQL-CUDA baseline
(1100.7 ms) is **61.2×**.

## What Changes

### Single phase — two small fixes

- MODIFY `CudaBackend::preallocate_kv_cache_per_layer` to
  be idempotent: when the existing cache already matches
  the requested shape, just reset `cache.len = 0` instead
  of allocating a fresh `CudaKvCache::new_device`.
- MODIFY `CudaKvCache::new_device` to use `device_alloc`
  (= `cuMemAllocAsync` + `memset_d8_async`, HBM-bound at
  ~1.8 ms for the 1 GB cache) instead of
  `htod_f32(&zeros)` (PCIe-bound at ~38 ms). On first
  allocation the saving is real even when the idempotent
  guard misses.
- ADD a `seq_len=1` fast path in
  `prefill_q4_seq_device`: delegate to
  `decode_token_device` directly. The bench harness only
  prefills the first token via this entry point.

### Out of scope

- A general `seq_len ≤ N` threshold for switching between
  mmvq and batched-GEMM paths. The bench-relevant case is
  `seq_len=1`; longer prompts benefit from batched GEMM
  even at small M. Left for a follow-up if real-world
  prompts show a regression.

## Impact

- `crates/larql-compute/src/cuda/decode.rs` only.
- No API changes; no env-var flags. The fixes are
  unconditional improvements.

## Risks and back-out

- **Idempotent prealloc** could mask a shape mismatch if
  callers expected the cache to be cleared. Mitigated by
  the explicit `matches_shape` check — only true matches
  skip the allocation; any mismatch still re-allocates.
- **device_alloc instead of htod**: `device_alloc` uses
  `alloc_zeros` which is `unsafe alloc + memset_d8_async`.
  Both are well-trodden cudarc paths.
- **seq_len=1 fast path**: just calls into
  `decode_token_device`, which is the same code path that
  produces correct output for every decode call. Parity
  is therefore at the same bound as decode itself
  (≤ 1e-3 vs host fallback).

## Acceptance bar

Final numbers measured on the dev box (RTX 4090, CUDA 12.5,
Gemma 3 4B Q4_K vindex, 6-token prompt, 20 decode tokens
after 3 warmup):

| Metric | Pre-change | **Actual** | Target |
|---|---:|---:|---:|
| `prefill ms / 6 tokens` | 97.3 | **18.0** | ≤ 25 ✓ |
| `decode ms/token` | 10.36 | 10.4 | ≤ 11 ✓ |
| `tok/s` | 96.6 | 96.2 | ≥ 90 ✓ |
| Bit parity vs host fallback | passes | **passes** | ≤ 1e-3 |

Combined progress vs the pre-LARQL-CUDA-work baseline
(prefill 1100.7 ms, decode 162.72 ms/tok, 6.1 tok/s):

| | Baseline | **Now** | Speedup |
|---|---:|---:|---:|
| prefill ms | 1100.7 | **18.0** | **61.2×** |
| decode ms/tok | 162.72 | 10.4 | 15.6× |
| tok/s | 6.1 | 96.2 | 15.8× |

Closes the prefill gap with llama-cpp-turboquant
(~5.6 ms / 1073 tok/s on the same hardware) from 17× to
**~3.2×**. The remaining gap is mostly the projection-compute
floor — a CPU-equivalent tuned implementation would need
batched mmq or Tensor Cores to close further.
