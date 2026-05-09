## Why

This change is the result of asking *"what does TensorRT (and llama.cpp,
which inherits the same patterns from upstream NVIDIA mmvq work) do
that we don't?"*. Comparing our `q4k_mmvq` kernel against
`ggml/src/ggml-cuda/mmvq.cu` (llama-cpp-turboquant tip) surfaced the
single biggest architectural gap:

- **Our existing kernel**: `(WARP_SIZE × ROWS_PER_BLOCK) = (32 × 4)`
  threads per block, **1 warp per output row**, 4 rows per block.
  Each warp computes its row's `n_super_blocks / 2` iterations
  serially.
- **llama.cpp's mmvq (NVIDIA / GENERIC table, ncols_dst = 1)**:
  `nwarps = 4`, `rows_per_cuda_block = 1`. **All 4 warps cooperate
  on ONE row**. With `blocks_per_iter = vdr × nwarps × warp_size /
  qi = 8` for Q4_K, the inner-loop count drops from
  `n_super_blocks / 2` to `n_super_blocks / 8`. On the Gemma 3 4B
  `down` projection (40 super-blocks/row) that's 5 iters/warp
  vs our 20.

The cross-warp shared-memory reduction at the end is cheap; the
parallelism win is dominant on long rows.

## What Changes

### Phase 1: cooperative-warp kernel

- ADD `mul_mat_vec_q4_K_q8_1_f32_coop` in `Q4K_MMVQ_SRC`.
  - Grid: `(rows, 1, 1)` — one block per output row.
  - Block: `(WARP_SIZE = 32, NWARPS = 4, 1)` — 128 threads
    cooperating on the single row.
  - `tid = blockDim.x × threadIdx.y + threadIdx.x` (0..127).
  - `kbx_lane = tid / 16` (0..7) — 8 super-blocks worked on per
    iter; `iqs = 2 × (tid % 16)` — 16 iqs slices per super-block.
  - Loop: `for (kbx_base = 0; kbx_base + kbx_lane < n_sb;
    kbx_base += 8)` — `blocks_per_iter = 8`.
  - Final reduction: warp-internal `__shfl_xor_sync` first, then
    cross-warp via `extern __shared__ float warp_sums[NWARPS]`,
    then `dst[row] = sum(warp_sums)`.

### Phase 2: Rust wrapper

- ADD `matvec_device_into_with_dev_coop` — launches with grid
  `(rows, 1, 1)`, block `(32, 4, 1)`, `shared_mem_bytes = 16`.

### Phase 3: shape-aware dispatcher

The empirical sweep
(`q4k_mmvq_legacy_vs_coop_sweep` on RTX 4090, Gemma 3 4B Q4_K,
200 iters after warmup) shows **the coop kernel is NOT a uniform
win**:

| Shape (rows × hidden) | n_sb | legacy | coop | speedup |
|---|---:|---:|---:|---:|
| q     ( 2048 ×  2560) | 10 |  5.4 µs |  5.5 µs | **0.98×** |
| kv    ( 1024 ×  2560) | 10 |  5.3 µs |  3.8 µs | **1.39×** |
| wo    ( 2560 ×  2048) |  8 |  4.8 µs |  4.8 µs | **1.01×** |
| gate  (10240 ×  2560) | 10 | 12.2 µs | 14.2 µs | **0.86×** |
| up    (10240 ×  2560) | 10 | 12.2 µs | 14.2 µs | **0.86×** |
| down  ( 2560 × 10240) | 40 | 16.2 µs | 12.8 µs | **1.26×** |

Coop wins on `kv` (rows ≤ 1024 — legacy doesn't saturate the chip)
and `down` (n_sb ≥ 16 — long rows have enough work to amortise
the cross-warp reduction). Coop loses on `gate`/`up` (high row
count + short row → cross-warp overhead dominates).

`q4k_mmvq_use_coop(rows, hidden)` returns `true` when
`n_super_blocks ≥ 16 || rows ≤ 1024`. Override:
`LARQL_CUDA_Q4K_COOP=1` (force coop everywhere) or
`LARQL_CUDA_Q4K_COOP=0` (force legacy everywhere).

## Capabilities

### Modified Capabilities

- `compute-cuda-kernels` — adds the cooperative-warp kernel and
  the shape-aware dispatch contract.

## Impact

- **Affected files**:
  - `crates/larql-compute/src/cuda/q4k_mmvq.rs` — the new kernel
    `mul_mat_vec_q4_K_q8_1_f32_coop`, its Rust wrapper
    `matvec_device_into_with_dev_coop`, the dispatcher
    `q4k_mmvq_use_coop`, and a `q4k_mmvq_legacy_vs_coop_sweep`
    ignored microbench that records the per-shape speedup table.
- **Affected systems**: GPU only; Metal unaffected.

## Risks and back-out

- **Numerical drift**: the cooperative kernel computes the same
  partial dot products as the legacy kernel — just split across
  4 warps instead of 1. The final reduction order changes (warp
  partials first, then cross-warp summed), introducing minor
  fp32 reduction-order noise. The existing 1e-3 parity tests
  (`q4k_mmvq_matches_q4k_direct_on_dequantized_input`,
  `decode_token_phase1_matches_host_fallback`,
  `decode_token_graph_matches_per_call_over_5_steps`) all pass
  with the dispatcher's default settings.
- **Throughput regression**: the `gate`/`up` shapes regress 14%
  in the per-shape microbench. The dispatcher routes them to
  the legacy kernel, so production isn't affected.
  `LARQL_CUDA_Q4K_COOP=0` reverts to the all-legacy path.

## Acceptance bar

Measured on the dev box (RTX 4090, CUDA 12.5, Gemma 3 4B Q4_K
vindex, 6-token prompt, 20 decode tokens after 3 warmup, 5-run
average, on top of `cuda-decode-cuda-graph` + `cuda-attn-grid-split`
+ `cuda-prefill-tensor-cores`):

| Metric | Pre-change | **Actual** | Comparator |
|---|---:|---:|---:|
| `decode ms/token` | 8.50 | **8.23** | llama.cpp 4.41 |
| `tok/s` | 117.6 | **121.5** | llama.cpp 226.8 |
| Prefill (unchanged) | 10.7 | **10.7** | llama.cpp 5.6 |
| Per-shape parity | — | **passes (1e-3)** | — |
