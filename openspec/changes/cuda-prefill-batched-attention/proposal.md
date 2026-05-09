## Why

`cuda-prefill-batched-q4k` Phase 1 batched the projection GEMMs
but kept attention per-position — the bench profile showed
22.15 ms (37%) of the 60 ms accountable prefill time was the
loop calling `fused_decode_attention_device_kv` once per
position per layer. The projection batched-GEMM win was real
(qkv+wo+gate_up+down sum to 17.5 ms across 34 layers) but the
attention loop dwarfs it.

This change replaces the per-position attention loop with a
single batched-prefill attention kernel that processes all
`seq_len` query positions in parallel.

| | Pre-change | This change | Delta |
|---|---:|---:|---:|
| `attn` profile bucket | 22.15 ms (37%) | ≤ 5 ms (target) | ~17 ms saved |
| `prefill ms / 6 tokens` | 117.6 | ≤ 100 (target) | ~15% improvement |

The savings come from two effects:

1. **Parallelism**. The current loop launches one kernel per
   position, serialised by the per-call `cudaStreamSynchronize`
   the kernel doesn't issue but cudarc waits for implicitly
   via the next launch's arg-binding. Even without the implicit
   wait, the 6 launches × 34 layers = 204 small-kernel-launch
   overhead dominates the per-call kernel time (~50 µs of
   actual compute on a 4090 vs ~5 µs of launch overhead — the
   ratio is maybe 90:10 useful work).
2. **Cache locality**. With one launch covering all 6
   positions, the per-(head) shared-memory state (q_norm,
   k_norm, scores buffer) is set up once and reused across
   all positions in the same SM. The current per-position
   path tears down and rebuilds it 6 times per layer.

## What Changes

### Single phase — batched prefill attention kernel

Two-kernel design (cleaner separation than fused, avoids
cross-block synchronisation issues):

#### Kernel 1: `kv_cache_write_seq_f32`

- ADD an NVRTC kernel that takes
  `(k_new_seq, v_new_seq, k_cache, v_cache, kv_norm,
  num_kv_heads, head_dim, base_pos, seq_len, rotary_dim,
  rope_base, eps, qk_norm_offset, use_qk_norm)` and writes
  every `(seq_pos, kv_head)` row of `k_new_seq` (with RoPE +
  optional QK norm applied) into `k_cache[base_pos + seq_pos,
  kv_head, :]`. Same for `v_new_seq` into `v_cache` (no
  rotation; raw V copy).
- One CUDA block per `(seq_pos, kv_head)` pair; launch with
  `grid_dim = (seq_len, num_kv_heads, 1)`. Each block has
  `block_dim = (head_dim, 1, 1)` threads — one thread per
  feature dimension. No reductions needed since K-norm only
  needs a per-head sum that one warp can do.

#### Kernel 2: `fused_prefill_attention_f32`

- ADD an NVRTC kernel based on the existing
  `fused_decode_attention_f32` but with `seq_len` query
  positions in flight:
  - `grid_dim = (num_q_heads, seq_len, 1)`. `blockIdx.x`
    selects the Q head; `blockIdx.y` selects the seq position
    `sp`.
  - `block_dim = (256, 1, 1)` (same as existing).
  - Each block computes attention for one `(qh, sp)` pair
    over the cached K/V at positions `[0, base_pos + sp]`.
  - Q vector is `q_seq[sp, qh, :]`; Q-RoPE rotation pre-pass
    (same trick as `cuda-attn-rope-hoist`) hoists the
    rotation into shared memory before the score loop.
  - Score loop: `for j in 0..base_pos + sp + 1`. Causal —
    `sp` only attends to positions ≤ `base_pos + sp`.
  - Softmax + output the same way as the existing kernel.

#### Plumbing

- ADD `attn::fused_prefill_attention_seq_device` —
  Rust-side wrapper that allocates the output, computes the
  launch config, and dispatches both kernels in sequence
  on the same stream.
- MODIFY `decode::prefill_q4_seq_device` to call the new
  function once per layer instead of looping
  `fused_decode_attention_device_kv` over positions.

### Out of scope

- **True FA-style tiled attention** (load K/V tiles into
  shared memory, iterate over j in chunks of 32-128). For
  prefill of `seq_len ≤ 64` it's overkill; the per-`sp`
  block reads ~`(sp+1) × head_dim × 4` bytes of K from
  global memory which fits in L1/L2 for typical prompt
  lengths. The optimisation pays off at long prompts
  (≥ 256 tokens) — defer until we measure that regime.
- **Q4_K mmq** for prefill projections — separate proposal
  (`cuda-q4k-mmq-prefill`).
- **CUDA Graph capture** of the whole prefill — separate
  proposal.

## Capabilities

### Modified Capabilities

- `compute-cuda-kernels` — adds a requirement that prefill
  attention runs as a single batched launch over all `seq_len`
  query positions, not as a per-position loop.

## Impact

- **Affected files**:
  - `crates/larql-compute/src/cuda/attn.rs` — adds the two
    NVRTC kernel sources and the
    `fused_prefill_attention_seq_device` wrapper.
  - `crates/larql-compute/src/cuda/decode.rs` — replaces
    the per-position attention loop in
    `prefill_q4_seq_device` with the single batched call.

- **Affected systems**: GPU only. Metal unaffected.

- **Provenance**: bottleneck identified by the prefill
  profile recorded in the `cuda-prefill-batched-q4k` proposal
  (`attn 22.15 ms (37%)`).

## Risks and back-out

- **Numerical drift**. The batched kernel computes the same
  attention math as the per-position one, just over more
  blocks. Expected parity within the existing 1e-3 bound.
- **Causal mask correctness**. Easy to get off-by-one. The
  existing
  `decode_token_phase1_matches_host_fallback` test exercises
  the prefill-then-decode boundary; we add a
  `prefill_q4_seq_matches_per_position` test that runs the
  batched path against the existing per-position path on
  synthetic inputs.
- **Two-kernel design** vs. one fused kernel: the two-kernel
  design accepts an extra ~5 µs launch overhead in exchange
  for cleaner correctness (no cross-block synchronisation
  issues with K/V cache write-then-read races). Worth it.
- **Back-out**:
  `LARQL_CUDA_PREFILL_BATCHED_ATTN=0` reverts to the
  per-position loop.

## Acceptance bar

Final numbers measured on the dev box (RTX 4090, CUDA 12.5,
Gemma 3 4B Q4_K vindex, 6-token prompt, 20 decode tokens
after 3 warmup):

| Metric | Pre-change | **Actual** | Target | Comparator |
|---|---:|---:|---:|---:|
| `prefill ms / 6 tokens` | 117.6 | **97.3** | ≤ 100 ✓ | llama.cpp ~5.6 |
| `attn` profile bucket | 22.15 ms | **0.86 ms** | ≤ 5 ms ✓ (26× drop) | — |
| `decode ms/token` | 11.04 | **10.36** | ≤ 11 ✓ (recovered) | 4.40 |
| Bit parity vs per-position | — | **passes** | ≤ 1e-3 | — |

**Cleared every gate.** The 26× drop on `attn` (22.15 → 0.86 ms)
is well past the predicted 4× — the per-`(qh, sp)` blocks
parallelise cleanly across the 4090's 128 SMs, and the
two-kernel design (cache-write + attention) avoids the
per-call launch overhead the per-position loop accumulated.

Decode also recovered to 10.36 ms/tok (matching the
post-`cuda-q6k-mmvq` baseline before `cuda-prefill-batched-q4k`
introduced its small regression). The 0.7 ms decode wobble
turned out to be run-to-run noise, not VRAM pressure from the
f32 weight cache.

Post-change prefill profile (steady state, after cache warm):

```
norm     20.84 ms (47%)   ← NEW TOP: 4 rms_norm calls × 34 layers
                            = 136 small kernel launches.
                            CUDA Graph capture would help.
gate_up   8.91 ms (20%)
down      4.58 ms (10%)
qkv       2.63 ms ( 6%)
wo        1.30 ms ( 3%)
silu      0.25 ms ( 1%)
attn      0.86 ms ( 2%)   ← was 22.15 ms before this change
                ─────
total    ~40 ms accountable + ~57 ms launch/setup = 97 ms
```

Combined progress vs the pre-LARQL-CUDA baseline:

| | Baseline | Now | Speedup |
|---|---:|---:|---:|
| decode ms/tok | 162.72 | **10.36** | **15.7×** |
| prefill ms (6 tok) | 1100.7 | **97.3** | **11.3×** |
| tok/s | 6.1 | **96.6** | **15.8×** |

Closes the prefill gap with llama-cpp-turboquant from 23×
(pre-mmvq era) to **17×**. Decode gap stays at 2.35×. The
remaining prefill cost is dominated by per-launch overhead
on the rms_norm kernels (47% of the budget); CUDA Graph
capture is the natural next change to address this.
