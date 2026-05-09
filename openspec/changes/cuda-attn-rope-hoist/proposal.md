## Why

After `cuda-q4k-mmvq-int8` Phase 3 the bench profile (RTX 4090,
Gemma 3 4B Q4_K, 20 tok / 3 warmup, with the corrected
`sync_if_profile` after `attn_call`) reports:

```
attn_call       6.35 ms (41%)   ← THIS CHANGE'S TARGET
proj_down       4.10 ms (26%)
proj_gate_up    1.39 ms ( 9%)
residual_cpu    1.23 ms ( 8%)
proj_qkv        1.02 ms ( 7%)
norm_cpu        1.06 ms ( 7%)
proj_wo         0.36 ms ( 2%)
htod/dtoh       ~0.02 ms
```

`attn_call` is `fused_decode_attention_device_kv` from
`cuda-decode-device-resident` Phase 3. Its score-computation
loop has the shape:

```cuda
for (int j = tid; j < n_ctx; j += bdim) {
    float dot = 0.f;
    for (int d = 0; d < head_dim; d++) {
        float qv = q_head[d];
        if (use_qk_norm) qv *= q_inv * (q_norm[d] + qk_norm_offset);
        if (d < rdim) {
            // ── Q-vector RoPE rotation ───────────────────
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
        // K rotation + dot product...
    }
}
```

**The Q-vector rotation depends only on `pos`, not on `j`.** It
gets recomputed identically for every `j` iteration. With
Gemma 3 4B at `head_dim = 256`, `rdim = 256`, `pos ≈ 25` during
decode, that's 25 × 256 = **6 400 redundant `(cosf, sinf,
powf)` triples per Q head per layer call**, multiplied by
`num_q_heads = 8` and `num_layers = 34`:

```
6 400 × 8 × 34 ≈ 1.74 M redundant trig triples per token
```

At ~1 ns per `cosf`/`sinf` on RTX 4090's SFU pipeline, that's
~3.5 ms/token of pure redundant arithmetic — and matches the
gap between the observed 6.35 ms `attn_call` and a back-of-
envelope minimum of ~3 ms for the actual unique work
(QK-norm + score dot + softmax + output).

Hoisting the Q rotation out of the `j` loop and into a one-
pass pre-rotation written to shared memory should ~halve
`attn_call`. The change is small (~30 lines of CUDA, under
shared-memory budget), strictly numerically equivalent (no
new precision drift — same arithmetic, just performed once
instead of `n_ctx` times), and fits cleanly inside the
existing `fused_decode_attention_device_kv` kernel.

## What Changes

### Single phase — kernel modification

- MODIFY the NVRTC kernel
  `fused_decode_attention_f32` (in
  `crates/larql-compute/src/cuda/attn.rs`):
  - After the `q_inv` reduction (which is already shared
    across all `j` iterations), add a one-pass loop where
    each thread computes the rotated Q value at one or more
    `d` indices and writes to a new shared-memory buffer
    `q_rot[head_dim]`.
  - Add `__syncthreads()` between the pre-rotation and the
    score loop.
  - Inside the score loop, replace the inline Q-rotation
    block with a single `q_rot[d]` load.
- KEEP the K-vector rotation as-is. The `j != pos` case
  reads from the k_cache (pre-rotated when stored on prior
  decode steps), and the `j == pos` case is computed by only
  the lane handling that single `j` — no redundancy.
- KEEP the kernel-level signature, dispatch, and existing
  `fused_decode_attention_device_kv` Rust wrapper unchanged.
  The change is internal to the NVRTC source.
- ADD shared-memory budget: `head_dim * sizeof(float)` extra
  per block (1 KB at `head_dim = 256`). Current usage is
  `(max_seq + block_dim) * 4` ≈ 17 KB/block at `max_seq =
  4096`; adding 1 KB lands at ≈ 18 KB, well inside sm_89's
  100 KB per-block ceiling.

### Out of scope

- The K-vector rotation and the K-cache append path are
  already computed once each. No changes there.
- Reducing `cosf`/`sinf` further via lookup tables or
  approximations — out of scope; the trig calls are now ~1/n_ctx
  of what they were, which is fine.
- Multi-warp parallelisation of the attention kernel — bigger
  change, separate proposal.
- Replacing `powf` for the per-pair `freq` with a precomputed
  table — small win, not worth the complexity here.

## Capabilities

### New Capabilities

(none — extends `compute-cuda-kernels`.)

### Modified Capabilities

- `compute-cuda-kernels` — adds a requirement that the
  attention kernel's Q-RoPE rotation run once per `(pos, head)`
  pair, not once per `(pos, head, j)`.

## Impact

- **Affected files**:
  - `crates/larql-compute/src/cuda/attn.rs` — kernel source
    modification.
  - Existing tests in
    `crates/larql-compute/tests/test_cuda_attn.rs` and
    `tests/test_cuda_decode.rs` exercise the attention path
    and serve as the parity gates. No new tests required —
    the change is numerically equivalent to the prior code.

- **Affected systems**: GPU container only. Metal backend
  unaffected.

- **Provenance**: bottleneck identified by direct profiling
  with `LARQL_CUDA_DECODE_PROFILE=1` after the
  `cuda-q4k-mmvq-int8` change exposed it (the corrected
  `sync_if_profile` after `attn_call` revealed that
  `attn_call` was actually 6.35 ms/tok, previously
  misattributed to `proj_wo`).

- **Out-of-scope notes**: A more aggressive refactor would
  also hoist the per-`pair` `freq` and `c/s` precomputation
  into the pre-rotation loop. Doing that is a small further
  win but adds an additional shared-memory float per pair
  (`hdim * 2 * sizeof(float)` = 2 KB). Easy to add later if
  the profile shows trig is still material.

## Risks and back-out

- **Numerical drift.** None expected. The pre-rotation
  computes the *same* values the inline rotation computed,
  just earlier and stored. fp32 arithmetic is associative
  enough that the order doesn't matter at 1e-3 tolerance.
- **Kernel correctness.** The rewrite touches the most
  complex single kernel in the backend; the pre-rotation loop
  is small but easy to mis-index. Mitigation: existing
  `decode_token_phase1_matches_host_fallback` test (≤ 1e-3
  vs the host-fallback path that uses CPU-side attention)
  serves as the parity gate. The test uses `rotary_dim =
  head_dim = 256` and `use_qk_norm` true/false coverage.
- **Back-out:** the change is contained in
  `fused_decode_attention_f32`'s NVRTC source. Reverting is a
  single file revert; no API surface changes; no env var
  flag needed.

## Acceptance bar

Measured on the dev box (RTX 4090, CUDA 12.5, Gemma 3 4B Q4_K
vindex, 20 tokens after 3 warmup):

| Metric | Pre-change (post-mmvq) | Target |
|---|---:|---:|
| `decode ms/token` | 15.55 | ≤ 13 |
| `GPU fwd ms/token` | 13.567 | ≤ 11 |
| `tok/s` | 64.3 | ≥ 77 |
| `attn_call` (profile) | 6.35 ms | ≤ 4 ms |
| Bit parity vs host fallback | ≤ 1e-3 | ≤ 1e-3 |

The attention-kernel-internal change targets a **2-3 ms/token
decode reduction**. If the profile shows `attn_call` did not
materially drop (e.g., < 1 ms reduction), the rotation was
not on the critical path after all and we should profile
deeper before further work — likely cause would be `cosf`/
`sinf` already being well-pipelined by the SFU even with
redundancy, in which case the residual cost is the dot-product
itself and the next move is to fuse Q × K dot and softmax into
a tiled FlashAttention-style kernel (separate change).
