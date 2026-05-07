## Context

`cuda-q4-matvec` covers Q4 dispatch end-to-end. The attention block
is composed of:

1. `Q @ K^T` — small-K GEMM (head_dim × seq_len for decode, larger for prefill).
2. **Softmax** — row-wise: subtract max, exp, divide by sum. cuBLAS doesn't ship this.
3. `attn @ V` — small-K GEMM.

cuBLAS handles 1 and 3. We need a custom CUDA kernel for 2.

cudarc 0.19 provides NVRTC compilation of CUDA C source strings at
runtime. The output is cached on disk; subsequent runs are near-free.
`cuda::cache` already provides the directory layout from
`cuda-f32-baseline`.

Goal: ship the kernel + a small helper. Integration into the actual
inference forward pass is a separate change because the inference
side has its own concerns (per-layer norms, RoPE, KV cache surgery,
GQA reshape, sliding window, softcap on attn vs logits, …).

## Goals / Non-Goals

**Goals:**

- A working scaled-softmax kernel that handles seq_len up to 8192,
  optional causal mask, optional softcap.
- A `decode_attention` helper that chains GEMM → softmax → GEMM in
  one host roundtrip.
- Numeric parity to a CPU naive reference within 1e-3 absolute,
  cosine ≥ 0.9999 on Gemma-shaped inputs.
- The PTX module is compiled once per process and cached on disk for
  warm-restart speed.

**Non-Goals:**

- FlashAttention v2's tiled online-softmax algorithm — that's later.
- GQA head reshape. Helper accepts already-reshaped Q/K/V.
- KV-cache. Helper takes flat tensors.
- f16 path. f32 only this change; f16 is a follow-up.
- Hooking into `decode_token`. Trait surface unchanged.

## Decisions

### D1 — Softmax kernel: one block per row

Rows are independent. With seq_len ≤ 1024 we can fit each row in
one CUDA block (max 1024 threads/block). For longer rows we'd need
inter-block reduction; we cap the helper at seq_len ≤ 1024 in this
change and document a follow-up to lift it.

For Gemma 4B at the typical 2k-8k context lengths we'd need to handle
longer rows. Pragmatic stopgap: chunk the softmax across multiple
1024-element blocks per row, with a final inter-block reduction.

Actually simpler: use a one-block-per-row kernel that loops with
`stride = blockDim.x` over the full row. Each thread handles
`ceil(seq_len / blockDim.x)` elements. blockDim.x stays at 1024.

This is the standard pattern. Three passes over the row per call:
1. Find max (with mask + softcap pre-applied).
2. Compute `exp(x - max)`, accumulate sum.
3. Normalise: `x[i] /= sum`.

Each pass uses warp + block reduction.

### D2 — Causal mask + softcap fused into the kernel

Rather than apply mask + softcap in a separate kernel, fold them
into the softmax kernel:

```c
__global__ void scaled_softmax(float *x, int n_rows, int n_cols,
                               float scale, float softcap, int causal) {
    int row = blockIdx.x;
    if (row >= n_rows) return;
    float *r = x + row * n_cols;
    // Pass 1: max
    float m = -INFINITY;
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x) {
        float v = r[j] * scale;
        if (softcap > 0.f) v = softcap * tanhf(v / softcap);
        if (causal && j > row) v = -INFINITY;
        r[j] = v;            // store back so passes 2-3 don't redo
        m = fmaxf(m, v);
    }
    m = block_reduce_max(m);
    // Pass 2: exp + sum
    float s = 0.f;
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x) {
        float e = expf(r[j] - m);
        r[j] = e;
        s += e;
    }
    s = block_reduce_sum(s);
    // Pass 3: normalise
    float inv = 1.f / s;
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x) r[j] *= inv;
}
```

Three passes is fine; rows are short enough for L1 to absorb them.

### D3 — `decode_attention` helper signature

```rust
pub struct AttentionOpts {
    pub causal: bool,
    pub softcap: Option<f32>,
}

pub(crate) fn decode_attention(
    drv: &Driver,
    q: &[f32],          // [n_q, head_dim]
    k: &[f32],          // [n_kv, head_dim]
    v: &[f32],          // [n_kv, head_dim]
    n_q: usize,
    n_kv: usize,
    head_dim: usize,
    opts: AttentionOpts,
) -> Result<Vec<f32>, CudaInitError>;
```

Single head, single batch. The caller is responsible for splitting
heads, broadcasting GQA, etc. The output shape is `[n_q, head_dim]`.

### D4 — PTX caching layout

`cuda::cache::cache_dir().join(arch).join("softmax.cubin")`. Compile
on first call via cudarc's `compile_ptx` helper; persist to disk;
subsequent calls reload from disk. Cache key includes the cudarc
version + the kernel source SHA-256 so an upgrade invalidates the
cache automatically.

### D5 — Capability::FlashAttentionV2 is now true

It's a stretch — we don't have FlashAttention-2's tiled algorithm —
but the bit semantically means "the backend can run the attention
block end-to-end on device." We satisfy that. A future sub-change
that ships true FA-2 doesn't need to flip the bit; it just makes the
kernel faster.

## Risks / Trade-offs

- **Risk: softmax kernel incorrectness silently breaks attention.**
  → Mitigation: parity tests against a naive scalar reference;
  abort the change if any seq_len doesn't match within 1e-3.
- **Risk: NVRTC compile cost.** First-call latency is ~150 ms.
  → Mitigation: compile at backend init, cache to disk.
- **Risk: row > 1024 elements.** Standard 1-block-per-row kernels
  handle this via the strided loop above. We test up to seq_len = 4096
  to confirm.
- **Risk: numerical differences in softmax sum reductions.** Block
  reductions accumulate in different order than the reference loop.
  → 1e-3 absolute tolerance covers this comfortably.

## Migration Plan

Land. Helper is unused by inference today; it's verified through
direct tests. The integration change (`inference-cuda-attention-integration`)
plumbs it into the per-layer dispatch.

Rollback: revert. cuda-q4-matvec still works.

## Open Questions

- **Q1: Should we also ship a fused QK^T+softmax kernel?** Saves one
  sync. Decision: no — the helper does its own end-of-call sync, so
  there's only one host roundtrip already. The micro-optimisation is
  follow-up work.
- **Q2: f16 path.** Decode-time attention often uses f16 for the
  intermediate `attn` tensor to halve memory bandwidth. We stick to
  f32 here for correctness; f16 is a follow-up.
