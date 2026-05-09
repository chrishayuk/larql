## Why

After `cuda-fused-norm-add` LARQL is at 8.22 ms/tok on Gemma 3 4B
Q4_K vs llama-cpp-turboquant's 4.34 ms/tok — a 1.89× gap. Reading
llama.cpp's `fattn-mma-f16.cuh` and `fattn-tile.cu` surfaced the
single biggest gap: their attention path runs over an **f16 K/V
cache** with **WMMA / MMA-fragment Tensor Cores**, while ours
runs pure f32 SIMT with an f32 K/V cache.

This change is **Phase 1**: switch the K/V cache storage to f16.
Phase 2 (WMMA-based attention compute) follows separately.

## Why Phase 1 alone is worth shipping

- **Halves the K/V slab footprint** — Gemma 3 4B at max_seq=4096
  goes from 1.3 GB → 660 MB. Useful headroom for longer contexts.
- **Halves K/V read bandwidth** in the fused attention kernel.
  At short context (n_ctx ≈ 20–30) this is a small fraction of
  total per-token bandwidth, so the bench-numbers win is modest.
  At long context (n_ctx in the thousands) it's significant.
- **Prerequisite for Phase 2 WMMA**. WMMA fragments natively
  consume f16 inputs; an f16 K/V cache is the right input format
  for `wmma::load_matrix_sync` directly off the cache.
- **Architecturally aligned with TensorRT-LLM and llama.cpp**, both
  of which use f16 (or smaller) K/V caches by default.

## Relationship to RotorQuant

LARQL has `larql_rotorquant` (Iso3/Planar3/Iso4/Planar4) for the
inference-side host KV cache, with 3-4 bit compression and
rotation-based reconstruction. That's the **deep-compression
long-context** tier and is unaffected by this change. The CUDA
backend's own `CudaKvCache` (used by `decode_token_device_*`) is
a separate, **fast-path short-/medium-context** tier — Phase 1
moves it from f32 to f16. The two tiers are complementary and can
coexist.

## What Changes

### Phase 1A: storage type

- MODIFY `cuda::decode::CudaKvLayer` so that
  `k: CudaSlice<f32>` becomes `k: CudaSlice<half::f16>` (and
  same for `v`). `CudaKvCache::new_device` allocates with
  `stream.alloc_zeros::<half::f16>(n)` instead of
  `device_alloc(n)` (which targeted f32). All zero values are
  byte-identical between f16 and f32, so the semantics survive
  unchanged.

### Phase 1B: kernel f16 read/write helpers

- MODIFY `FUSED_DECODE_ATTN_SRC` (the captured-decode attention
  kernel) so `k_cache` / `v_cache` arguments become
  `unsigned short*` and ADD device-helper inlines:
  - `ld_kvcache(p)` → `cvt.f32.f16` PTX inline.
  - `st_kvcache(p, f)` → `cvt.rn.f16.f32` PTX inline.
- MODIFY `KV_CACHE_WRITE_SEQ_SRC` (prefill K/V writer) similarly:
  add `st_kv_seq` writer.
- MODIFY `FUSED_PREFILL_ATTN_SRC` similarly: add `ld_kvc_pf`
  reader. Cache args become `const unsigned short*`.
- All read/write sites in those three kernels SHALL go through
  the helpers; raw `k_cache[idx]` indexing is removed.

### Phase 1C: Rust wrapper signature plumbing

- MODIFY every wrapper that touched
  `&mut CudaSlice<f32>` for K/V cache args to take
  `&mut CudaSlice<half::f16>`:
  - `attn::fused_decode_attention_device_kv`
  - `attn::fused_decode_attention_device_kv_into`
  - `attn::fused_prefill_attention_seq_device`
- The host-wrapper paths that take `&[f32]` for K/V cache
  (`fused_decode_attention`, `fused_decode_attention_device`)
  internally convert host f32 → f16 once, htod the f16 buffer,
  run the kernel, dtoh the f16 result, and convert back to
  `Vec<f32>` for the legacy host-fallback contract. These paths
  are back-out / parity only — production decode uses
  `decode_token_device_graph_attempt`, which holds the f16
  cache device-resident across all decode steps.

### Phase 1D: host-boundary helpers

- ADD `CudaBackend::htod_f32_as_f16_into_slice(src: &[f32],
  dst: &mut CudaSlice<half::f16>, offset: usize)` — element-wise
  host f32→f16 then `memcpy_htod`.
- ADD `CudaBackend::dtoh_f16_as_f32(dev: &CudaSlice<half::f16>)
  → Vec<f32>` — `clone_dtoh` + element-wise convert.
- MODIFY `populate_kv_layer` (DecodeBackend trait impl) to use
  `htod_f32_as_f16_into_slice`.
- MODIFY the legacy `decode_token` host-fallback path to use
  `dtoh_f16_as_f32` for cache reads and
  `htod_f32_as_f16_into_slice` for cache writes.

### Phase 1E: parity-test tolerance bump

- MODIFY `tests/test_cuda_attn.rs::fused_decode_attention_matches_cpu_reference`
  to relax the K cache bound from 1e-3 to 5e-3 and the V cache
  bound from 1e-6 to 5e-3. The new f16 cache stores values with
  ~5e-4 absolute error per unit-magnitude element; round-tripping
  amplifies that to a few × 1e-4 max-element. 5e-3 is comfortable
  headroom while still tight enough to catch arithmetic bugs.

## Out of scope (Phase 2 follow-up)

- **WMMA / MMA-fragment-based attention compute**. The kernel
  body still runs as f32 SIMT; only the cache storage is f16.
  Phase 2 will reshape the per-block attention computation around
  16×16 WMMA fragments, making the K^T @ Q and (softmax) @ V
  matmuls run on Tensor Cores. That's a kernel rewrite (~3-5
  days) and benefits long-context decode much more than the
  short-context bench used here.

## Capabilities

### Modified Capabilities

- `compute-cuda-kernels` — switches the CUDA decode K/V cache
  storage type from f32 to f16.

## Impact

- **Affected files**:
  - `crates/larql-compute/src/cuda/decode.rs` —
    `CudaKvLayer` storage type, `new_device`, `populate_kv_layer`,
    legacy `decode_token` host-fallback path.
  - `crates/larql-compute/src/cuda/attn.rs` — three NVRTC kernels'
    cache args + helpers, all four attention wrappers, two
    f32↔f16 host-boundary conversions in the host wrappers.
  - `crates/larql-compute/src/cuda/backend.rs` —
    `htod_f32_as_f16_into_slice`, `dtoh_f16_as_f32`.
  - `crates/larql-compute/tests/test_cuda_attn.rs` — parity bound
    bump.
- **Affected systems**: GPU only.

## Risks and back-out

- **Numerical drift**: f16 has ~5e-4 absolute error per
  unit-magnitude element. The K/V cache stores rotated K
  (unit-magnitude after RMSNorm) and raw V (similarly bounded
  for normalised activations), so the per-token attention output
  drift is ≤ a few × 1e-3 max-element. Verified by the
  existing 1e-3 parity tests passing
  (`decode_token_phase1_matches_host_fallback`,
  `decode_token_graph_matches_per_call_over_5_steps`).
- **Memory layout shift**: existing cache slabs (f32) become
  invalid across this change. New backends and any code that
  serialises/deserialises CUDA-side cache slabs has to
  account for the type change. Internal caches only — no
  on-disk format affected.
- **No env-var back-out**: this is a structural type change,
  not an opt-in path. The legacy host-fallback decode
  (`LARQL_CUDA_DECODE_HOST_FALLBACK=1`) still works, just runs
  through the f16↔f32 conversion bridge.

## Acceptance bar

Measured on the dev box (RTX 4090, CUDA 12.5, Gemma 3 4B Q4_K,
6-token prompt, 20 decode tokens after 3 warmup, 5-run average,
on top of every prior optimisation, with
`LARQL_CUDA_PREFILL_TENSOR_CORES=1`):

| Metric | Pre-change | **Actual** | Comparator |
|---|---:|---:|---:|
| `decode ms/token` | 8.22 | **8.04** | llama.cpp 4.34 |
| `tok/s` | 121.7 | **124.4** | llama.cpp 230.2 |
| Run-to-run variance | 8.05–8.52 | **8.03–8.05** | — |
| Parity (1e-3) | passes | **passes** | — |
| Generated text | identical | **identical** | — |

The headline isn't the 0.18 ms / 2.2% mean improvement — it's the
**collapse of run-to-run variance** (from 0.47 ms range to 0.02
ms). f16 cache reduces L2 working-set pressure enough that
scheduling jitter visibly drops. Combined with all earlier wins:

| | Pre-session | Post-this-change | llama.cpp |
|---|---:|---:|---:|
| prefill ms | 18.0 | **10.70** | 6.25 |
| decode ms/tok | 9.62 | **8.04** | 4.34 |
| decode tok/s | 103.9 | **124.4** | 230.2 |
| Decode gap closed | — | **27%** | — |
