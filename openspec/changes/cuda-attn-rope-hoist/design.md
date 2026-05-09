# cuda-attn-rope-hoist — design

## Current kernel shape

`fused_decode_attention_f32` (in `cuda/attn.rs`'s NVRTC source)
runs one CUDA block per Q head, `block_dim = 256` threads.
Shared memory layout:

```text
smem[0 .. max_seq]                  ← scores[]   (per-token attention scores)
smem[max_seq .. max_seq + block_dim] ← scratch[] (warp reduction temp)
```

Total: `(max_seq + block_dim) * 4` bytes ≈ 17 KB at
`max_seq = 4096`.

Per-thread work:

1. Compute `q_inv` (rsqrt of mean Q²) — warp reduction across
   the head, written to scratch.
2. Compute `k_inv` similarly.
3. If `(qh % group) == 0`, append the rotated K and raw V to
   the cache at position `pos`. (Single-thread / single-warp
   work; not on the critical path.)
4. Score loop: each thread takes one `j ∈ [0, n_ctx)` and
   iterates `d` from 0 to `head_dim - 1`, computing
   `dot += qv(d, j) × kv(d, j)`. Inside the d loop the Q
   rotation is recomputed every iteration even though it's
   only a function of `(d, pos)`.
5. Softmax over `scores[]`, output via V aggregation.

## The redundancy

Step 4 contains:

```cuda
if (d < rdim) {
    int pair = d % hdim;
    bool imag = d >= hdim;
    float re = q_head[pair];
    float im = q_head[pair + hdim];
    if (use_qk_norm) {
        re *= q_inv * (q_norm[pair]      + qk_norm_offset);
        im *= q_inv * (q_norm[pair+hdim] + qk_norm_offset);
    }
    float freq  = 1.0f / powf(rope_base, (float)(2 * pair) / (float)rdim);
    float angle = (float)pos * freq;
    float c = cosf(angle);
    float s = sinf(angle);
    qv = imag ? (re * s + im * c) : (re * c - im * s);
}
```

This block runs once per `(thread → j, d)`. Across the warp,
all threads with valid `j` are doing the same work for the same
`d`. There are `n_ctx ≤ 25` valid threads, each doing
`head_dim = 256` rotations → ~6 400 redundant rotations per
Q-head per layer call.

## Hoisted shape

```cuda
// After q_inv reduction, before the score loop, do a one-pass
// pre-rotation of the Q vector. Each thread computes one (or
// more) d values; result lives in shared memory.
extern __shared__ float smem[];
float* scores  = smem;
float* scratch = smem + max_seq;
float* q_rot   = smem + max_seq + bdim;   // ← new region

for (int d = tid; d < head_dim; d += bdim) {
    float qv = q_head[d];
    if (use_qk_norm) qv *= q_inv * (q_norm[d] + qk_norm_offset);

    int rdim = (rotary_dim == 0) ? head_dim : min(rotary_dim, head_dim);
    if (d < rdim) {
        int hdim = rdim / 2;
        int pair = d % hdim;
        bool imag = d >= hdim;
        float re = q_head[pair];
        float im = q_head[pair + hdim];
        if (use_qk_norm) {
            re *= q_inv * (q_norm[pair]      + qk_norm_offset);
            im *= q_inv * (q_norm[pair+hdim] + qk_norm_offset);
        }
        float freq  = 1.0f / powf(rope_base, (float)(2 * pair) / (float)rdim);
        float angle = (float)pos * freq;
        float c = cosf(angle);
        float s = sinf(angle);
        qv = imag ? (re * s + im * c) : (re * c - im * s);
    }
    q_rot[d] = qv;
}
__syncthreads();
```

Then the score loop simplifies to:

```cuda
for (int j = tid; j < n_ctx; j += bdim) {
    float dot = 0.f;
    for (int d = 0; d < head_dim; d++) {
        float qv = q_rot[d];               // ← was 10+ lines

        // K rotation logic stays (j == pos rotates inline; otherwise
        // reads from k_cache which is pre-rotated)
        float kv;
        if (j == pos) { /* compute k_rot from k_head */ }
        else          { kv = k_cache[((size_t)j * num_kv_heads + kvh) * head_dim + d]; }

        dot += qv * kv;
    }
    float logit = dot * attn_scale;
    if (softcap > 0.f) logit = softcap * tanhf(logit / softcap);
    scores[j] = logit;
}
```

## Shared memory budget

- Pre-change: `(max_seq + bdim) * 4 = (4096 + 256) * 4` = 17 408 B.
- Post-change: `(max_seq + bdim + head_dim) * 4 = (4096 + 256 +
  256) * 4` = 18 432 B.
- sm_89 dynamic shared memory limit: 100 KB per block (after
  `cudaFuncSetAttribute`); default cap without that call is
  48 KB. Both pre- and post-change fit comfortably under 48 KB.

The Rust-side launch config in `attn.rs` already computes
`shared_mem_bytes` from `(max_seq + block_dim)`. It needs to be
extended by `head_dim`.

## Numerical contract

Pre-change vs post-change is *bit-equivalent in expectation* —
the same arithmetic in the same fp32 precision, just performed
once instead of `n_ctx` times. The only nondeterminism source
is fp32 reduction order, which is unaffected (the score-loop
reduction is unchanged).

The existing parity test in
`crates/larql-compute/tests/test_cuda_decode.rs::decode_token_phase1_matches_host_fallback`
asserts `≤ 1e-3` max-element diff vs the CPU host-fallback
attention. That bound holds before and after this change.

## Test plan

| Layer | Test | Status |
|---|---|---|
| Integration | `decode_token_phase1_matches_host_fallback` | exists, must still pass |
| Integration | `fused_decode_attention_matches_cpu_reference` (in `test_cuda_attn.rs`) | exists, must still pass |
| Bench | `larql bench output/gemma-3-4b-it-vindex --backends cuda --tokens 20 --warmup 3 --verbose` | new acceptance run |

No new tests added; the existing ones already cover the
attention path against a CPU reference. Adding a hoist-specific
unit test would just duplicate the parity coverage.

## Bench plan

Same as `cuda-q4k-mmvq-int8`'s. Acceptance:

- `decode ms/token ≤ 13`
- `attn_call` (with `LARQL_CUDA_DECODE_PROFILE=1`) ≤ 4 ms

If the bucket drops by < 1 ms we know the trig calls were
already absorbed by the SFU pipeline and the remaining cost is
elsewhere (e.g., the dot-product itself or memory bandwidth on
the K/V cache). In that case the next move is a tiled FA-style
kernel rewrite, separate change.

## Decision gates

- After implementation: if `attn_call > 5 ms`, the hoist
  didn't pay. Profile with `nvprof` to determine whether trig
  was the actual bottleneck or whether the inner-loop K-cache
  reads dominate.
- If `decode ms/tok ≤ 13` AND `attn_call ≤ 4 ms`: ship and
  archive.
- If `decode ms/tok` improves but stays > 13 (i.e., we hit
  the `attn_call` target but compute moved to other buckets):
  document; the budget moved cleanly and the next change can
  attack the new top bucket.
