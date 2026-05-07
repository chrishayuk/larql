## Why

Third sub-change of [`cuda-and-rotorquant-kv`](../cuda-and-rotorquant-kv/proposal.md).
With f32 GEMM/GEMV and Q4 matvec running on the RTX 4090, the
attention block is the next thing that needs to leave the CPU. cuBLAS
covers two of the three pieces (QK^T and attn·V are GEMMs), but
softmax has no cuBLAS counterpart — we need a custom CUDA kernel.

This sub-change ships:

- a **scaled / masked softmax** CUDA kernel via cudarc NVRTC, with
  tunable scale, optional causal mask, and optional softcap (Gemma-2
  style), and
- a **decode-time attention** helper `cuda::attn::decode_attention`
  that chains cuBLAS GEMM → our softmax → cuBLAS GEMM into a single
  device-resident sequence with one host roundtrip.

The helper is a side-channel — the trait-level `decode_token` is
intentionally left as `None` because that path requires per-architecture
norms / RoPE / KV-cache management that is too much to bite off in one
sub-change. A later `inference-cuda-attention-integration` change
plumbs `cuda::attn::decode_attention` into `larql-inference`'s forward
pass.

## What Changes

- ADD `larql_compute::cuda::attn` — softmax kernel + decode helper.
- ADD `cuda::ptx_softmax::SOFTMAX_PTX` — a small NVRTC source that
  compiles to a row-per-block scaled-softmax kernel with optional
  causal mask + softcap. Cached under
  `$XDG_CACHE_HOME/larql/cudarc/<arch>/softmax.cubin`.
- ADD `cuda::attn::softmax_inplace` — load + launch + sync helper.
- ADD `cuda::attn::decode_attention(q, k, v, opts) -> Vec<f32>` —
  cuBLAS GEMM (QK^T) → softmax → cuBLAS GEMM (attn·V), single sync.
- MODIFY `CudaBackend::supports` to advertise
  `Capability::FlashAttentionV2` (semantically, "we can run the
  attention block end-to-end on device").
- ADD parity tests in `tests/test_cuda_attn.rs` against a naive
  CPU reference (loops, no BLAS).

This is non-breaking. The trait surface is unchanged. The default
`decode_token` still returns `None`. Existing inference paths keep
their current dispatch.

## Capabilities

### New Capabilities

(none — implements scenarios already declared on
`compute-cuda-kernels` via the parent change.)

### Modified Capabilities

- `compute-cuda-kernels`: scenarios for fused decode-time attention
  (declared in the parent delta with `<!-- test: unbacked -->`) get
  real test annotations. Two new requirements cover the softmax kernel
  contract and the cudarc PTX cache layout.

## Impact

- **Affected files**: new `crates/larql-compute/src/cuda/attn.rs`
  and `cuda/ptx_softmax.rs`; `cuda::backend.rs` capability flip;
  new `crates/larql-compute/tests/test_cuda_attn.rs`.
- **Affected systems**: CUDA-only. CPU + Metal builds identical.
- **Performance**: a single decode-time attention call (Gemma 4B
  shapes: hidden=2560, head_dim=320, seq_len up to 8k) does two
  cuBLAS GEMMs and one softmax — well within the budget for first
  real measurement. No micro-optimisation in this change.
- **Out of scope**: prefill batching, FlashAttention-2's full
  algorithm (online softmax with tiling), KV-cache management,
  RoPE, QK norm, softcap-on-attn, GQA expansion. All belong to
  `inference-cuda-attention-integration` and later perf changes.
