## Why

`fused_decode_attention_f32` launches with `grid_dim = (num_q_heads,
1, 1)`. For Gemma 3 4B that's 8 blocks × 256 threads = 8 SMs of
RTX 4090's 128 (~6% chip occupancy). The kernel is genuinely
under-parallelised; even if compute time per call is small, exposing
more grid parallelism gives the scheduler more freedom to overlap
the attention kernel with the surrounding mmvq queue.

## What Changes

- ADD a `d_split` kernel arg (`int`) and grow the launch grid to
  `(num_q_heads, d_split, 1)`. Each block computes a contiguous
  `[d_start, d_end)` slice of `out[qh, :]` where
  `d_per_chunk = head_dim / d_split`.
- The Q/K reductions, Q-rotation, score loop, softmax, and inverse-
  sum reduction are recomputed redundantly in every chunk — they
  don't depend on `d`, so duplicating them is the price for the
  parallelism. The redundant work scales with `n_ctx`, but for
  decode (n_ctx grows from 1 to a few hundred) it stays small
  relative to the per-chunk output sum.
- K/V cache writes are gated to `dchunk == 0` so multiple chunks
  for the same `(qh, kvh)` pair don't double-write.
- `choose_attn_d_split(num_q_heads, head_dim)` picks a default
  targeting ~32 grid blocks (RTX 4090 has 128 SMs; 32 leaves room
  for the rest of the decode pipeline to overlap on the other
  SMs). For Gemma 3 4B with 8 q_heads → `d_split = 4`.
- `LARQL_CUDA_ATTN_DSPLIT=N` (1, 2, 4, 8, 16) overrides the default;
  `=1` is the back-out (single-block-per-head, legacy behaviour).
  `head_dim % d_split != 0` falls back to 1.

## Impact

- **Affected files**:
  - `crates/larql-compute/src/cuda/attn.rs` — kernel signature
    (`+ int d_split`), `dchunk = blockIdx.y` plumbing, K/V write
    gate, output-loop range; `choose_attn_d_split` helper; all
    four wrappers (`fused_decode_attention`, `_device`, `_device_kv`,
    `_device_kv_into`) updated to pass the new arg + grid dim.
- **Affected systems**: GPU only; Metal unaffected.

## Risks and back-out

- **Numerical drift**: none expected — same arithmetic, same
  reduction order per block; only the d range each block writes
  changes. Confirmed by the existing parity suite
  (`decode_token_phase1_matches_host_fallback`,
  `decode_token_graph_matches_per_call_over_5_steps`).
- **Throughput regression**: redundant work in the score-loop /
  Q-rotation steps is paid `d_split` times. For very long context
  (n_ctx in the thousands), the duplication may outweigh the
  parallelism win. Mitigation: `LARQL_CUDA_ATTN_DSPLIT=1` reverts
  to the legacy single-block path.

## Capabilities

### Modified Capabilities

- `compute-cuda-kernels` — adds the `d_split` parallelism contract
  for `fused_decode_attention_f32`.

## Acceptance bar

Measured on the dev box (RTX 4090, CUDA 12.5, Gemma 3 4B Q4_K
vindex, 6-token prompt, 20–50 decode tokens after 3–5 warmup,
graph path on, 3–5 run average):

| Metric | Pre-change | Target | **Actual** | Comparator |
|---|---:|---:|---:|---:|
| `decode ms/token` | 8.33 | ≤ 8.2 | **8.21** | llama.cpp 4.40 |
| `tok/s` | 120.0 | ≥ 122 | **121.8** | llama.cpp 227.5 |
| Parity vs `LARQL_CUDA_ATTN_DSPLIT=1` | — | ≤ 1e-3 | **passes** | — |

The win is real but small (~0.12 ms / 1.5%). The fused decode-attention
kernel is not the dominant per-token cost in graph mode — the bulk of
8.21 ms is HBM-bandwidth-limited mmvq projections. A bigger move
(Tensor Cores, two-pass score+output split) would have to attack
those.
