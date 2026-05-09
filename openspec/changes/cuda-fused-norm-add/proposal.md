## Why

TensorRT-LLM's `RMSNormPlugin` fuses `rms_norm + add_residual` into
a single kernel. Our captured-decode pipeline ran them as two
separate kernels (`rms_norm_device_into` writing to
`scratch.attn_normed` / `scratch.ffn_normed`, then
`add_in_place_device(h, normed)`). That intermediate buffer was
written, read, and never used again — pure waste in the graph.

For Gemma 3 4B at decode steady state: 2 fusions × 34 layers =
68 intermediate writes per token × `hidden = 2560` × 4 B = 680 KB
of avoidable HBM traffic, plus 68 redundant launches that the
captured graph still has to plumb through.

## What Changes

- ADD `RMS_NORM_ADD_SRC` PTX in `cuda::elem` with one kernel:
  `rms_norm_add_f32(dst, src, weight, n, has_weight, eps,
   norm_offset, scale)`. Computes
  `dst[i] += rms_norm(src, weight)[i] * scale` in a single
  pass. Single-block (grid = 1, block = 1024) with shared-memory
  reduction for the row sum-of-squares.
- ADD `elem::rms_norm_add_device` Rust wrapper.
- MODIFY `decode::run_decode_pipeline_into_scratch` to call
  `rms_norm_add_device` in place of the legacy
  `rms_norm_device_into` + `add_in_place_device` pair at the
  post-attn and post-ffn residual sites. The post-FFN
  `layer_scalar` (Gemma 4) folds in via the kernel's `scale`
  argument, eliminating a third launch (`scale_inplace_device`).
- The pre-attn / pre-ffn norms (which feed Q8_1 quant for the
  next mmvq) stay on the existing `rms_norm_device_into` path —
  there's no `add` to fuse with there.

## Out of scope

- **`norm + quantize_q8_1` fusion**: the pre-attn norm followed
  immediately by Q8_1 quantize is the next obvious candidate.
  Skipped here because the Q8_1 quantize is a 1-warp-per-32-elem
  pattern with a different launch geometry; the fusion would
  require either splitting the norm work-distribution (less
  efficient for the reduce phase) or a complex two-stage block
  layout. Future work.
- **FP16 KV cache**: rotorquant
  (`openspec/specs/kv-cache-rotorquant/`) is LARQL's answer for
  KV cache compression and goes further (3-4 bit). Adding a
  parallel FP16-cache path on the CUDA side would duplicate
  work for marginal gain — KV bandwidth is < 1% of decode time
  at this short-context bench. Punted.

## Capabilities

### Modified Capabilities

- `compute-cuda-kernels` — adds the fused norm+add+scale
  contract for the captured-decode pipeline.

## Impact

- **Affected files**:
  - `crates/larql-compute/src/cuda/elem.rs` — `RMS_NORM_ADD_SRC`,
    `RMS_NORM_ADD_FUNC`, `rms_norm_add_device` wrapper.
  - `crates/larql-compute/src/cuda/decode.rs` —
    `run_decode_pipeline_into_scratch` post-attn and post-ffn
    branches.
- **Affected systems**: GPU only.

## Risks and back-out

- **Numerical drift**: the fused kernel computes the same
  arithmetic (rms_norm followed by add) but in a different
  reduction-of-fp32 order. Empirically the existing 1e-3
  parity tests
  (`decode_token_phase1_matches_host_fallback`,
  `decode_token_graph_matches_per_call_over_5_steps`) pass.
- **No env-var back-out**: the fusion is a pure win on the
  captured-graph path with no behaviour difference; reverting
  would require re-introducing the intermediate buffers. The
  legacy non-graph path (`decode_token_device_legacy`) keeps
  the unfused `rms_norm + add_in_place + scale_inplace`
  sequence as the implicit back-out.

## Acceptance bar

Measured on the dev box (RTX 4090, CUDA 12.5, Gemma 3 4B Q4_K
vindex, 6-token prompt, 20 decode tokens after 3 warmup, 10-run
average, on top of `cuda-decode-cuda-graph` +
`cuda-attn-grid-split` + `cuda-prefill-tensor-cores` +
`cuda-q4k-mmvq-warp-cooperative`, with
`LARQL_CUDA_PREFILL_TENSOR_CORES=1`):

| Metric | Pre-change | **Actual** | Comparator |
|---|---:|---:|---:|
| `decode ms/token` | 8.23 | **8.19** (8.09 excl. one outlier) | llama.cpp 4.41 |
| `tok/s` | 121.5 | **122.1** (123.6 excl.) | llama.cpp 226.8 |
| Per-shape parity | — | **passes (1e-3)** | — |

Marginal but real (~0.04 ms / 0.5%). The bench is short-context
where these intermediate writes are a small fraction of total
HBM traffic — the win shows up bigger at longer contexts and
is, more importantly, a cleaner architectural baseline for
future fusion work.
