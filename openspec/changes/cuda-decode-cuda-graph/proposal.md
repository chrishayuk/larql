## Why

After `cuda-sfu-intrinsics` decode is at 9.35 ms/tok / 107 tok/s.
The remaining gap with llama-cpp-turboquant (4.40 ms/tok /
227.5 tok/s) is 2.13×. The decode profile shows compute is
spread across many small kernels:

```
attn_call       2.68 ms (29%)
proj_down       1.65 ms (18%)
proj_gate_up    1.43 ms (15%)
norm_cpu        1.15 ms (12%)
residual_cpu    1.07 ms (12%)
proj_qkv        0.91 ms (10%)
proj_wo         0.39 ms ( 4%)
                ─────
total ~9.3 ms
```

A single decode token issues ~16 kernel launches per layer ×
34 layers ≈ **544 launches per token**. cuLaunchKernel host
overhead is typically 5-10 µs each → **~3-5 ms of pure launch
overhead per token** that the GPU spends idle between
kernels.

CUDA Graphs let us capture a sequence of launches once and
replay them as a single submission. The launch-overhead
budget collapses from ~3-5 ms to ~10 µs (a single
`cuGraphLaunch` call).

Predicted savings: 2-4 ms/token (~25-40% improvement). Decode
target: 6-7 ms/tok / 140-170 tok/s. Closes most of the
remaining gap with llama-cpp-turboquant.

## What Changes

This is a multi-phase refactor. The phases ship together
because they're interdependent — graph capture requires
stable buffer addresses AND device-side `pos`.

### Phase 1: DecodeScratch — pre-allocated intermediate buffers

CUDA Graphs need consistent device pointers across replays.
Currently each kernel call allocates a fresh
`CudaSlice<f32>` via `device_alloc_uninit`; cudarc's
`cuMemAllocAsync` may return the same virtual address for
repeated same-shape allocations, but it's not guaranteed.

- ADD `cuda::scratch::DecodeScratch` holding all per-decode
  intermediate buffers sized for a fixed shape:
  `h_dev`, `h_attn_dev`, `q_dev`, `k_dev`, `v_dev`,
  `attn_out_dev`, `attn_delta_dev`, `normed_dev`,
  `h_ffn_dev`, `gate_dev`, `up_dev`, `act_dev`,
  `ffn_delta_dev`, plus the four `Q8_1Buf` scratches
  (`h_attn_q8_1`, `h_ffn_q8_1`, `attn_out_q8_1`,
  `act_q8_1`) and the `pos_dev: CudaSlice<i32>`.
- ADD `CudaBackend::ensure_decode_scratch(shape)` —
  lazy-allocates the scratch on first call; reuses on
  subsequent calls if the shape matches.

### Phase 2: Device-side `pos`

Currently `fused_decode_attention_device_kv` takes `pos:
i32` as a kernel argument, which gets baked into the
captured graph. To allow replay with a new `pos`, the
kernel must read it from device memory.

- MODIFY the kernel signature:
  `int pos` → `const int* pos_dev`. The kernel reads
  `int pos = *pos_dev` at the top.
- MODIFY the Rust wrapper to take a
  `pos_dev: &CudaSlice<i32>` and dispatch the kernel arg.
- The caller updates `*pos_dev` via
  `htod_into_slice(&[new_pos], pos_dev, 0)` between
  graph replays.

### Phase 3: Write-into kernel wrappers

All the per-call kernel wrappers currently allocate fresh
output buffers. To pin allocations to scratch, each gets
an `_into` variant that writes into a pre-allocated buffer.

- `q4k_mmvq::matvec_device_into(out: &mut CudaSlice<f32>, ...)`
- `q6k_mmvq::matvec_device_into(out, ...)`
- `q4k_direct::matvec_device_into(out, ...)` (fallback path)
- `elem::rms_norm_device_into(out, ...)`
- `elem::silu_gate_up_device_into(out, ...)`
- `elem::quantize_q8_1_device_into(out: &mut Q8_1Buf, ...)`
- `attn::fused_decode_attention_device_kv_into(out, ...)`
- `matmul::gemv_device_inout_into(out, ...)` (existing
  variant already writes into `&mut`, just needs the
  helper)

The existing `matvec_device` etc. become thin wrappers
that allocate then call `_into`.

### Phase 4: decode_token_device refactor

- MODIFY `CudaBackend::decode_token_device` to:
  - Take a `DecodeScratch` parameter (or look it up via
    `ensure_decode_scratch`).
  - Use scratch buffers everywhere instead of letting
    matvec/elem helpers allocate.
  - Initial input: `htod_into_slice(x, &mut scratch.h_dev, 0)`.
  - Final output: `dtoh_f32(&scratch.h_dev)`.
- KEEP the existing path under a `LARQL_CUDA_DECODE_NO_SCRATCH=1`
  back-out env var for parity verification.

### Phase 5: CUDA Graph capture + replay

- ADD `DecodeGraph` struct holding a `CudaGraph` and the
  `DecodeScratch` it was captured with.
- ADD `CudaBackend::capture_decode_graph(layers, scratch)`:
  - `stream.begin_capture(...)` → run `decode_token_device`
    once → `stream.end_capture(...)`.
- MODIFY `decode_token_device` to:
  - On first call: run normally, then capture into a graph.
  - On subsequent calls with same shape AND same layer
    set: write `pos_dev`, write `h_dev` from input,
    `graph.launch()`, `dtoh_f32(scratch.h_dev)`.
- ADD `LARQL_CUDA_DECODE_GRAPH=0` env var to disable the
  graph path (back-out).

## Out of scope

- **Per-layer-set graph caching** — for now, capture a
  single graph for the layer set seen first. Different
  models or different layer counts cause re-capture.
- **Prefill graph capture** — prefill's `seq_len` varies
  per call, requiring re-capture each time. Not worth
  it. Decode's fixed shape is the natural target.
- **Tensor Cores / BF16** — orthogonal next step.

## Capabilities

### Modified Capabilities

- `compute-cuda-kernels` — adds requirements for graph
  capture, scratch buffers, and the device-side `pos`
  contract.

## Impact

- **Affected files**:
  - `crates/larql-compute/src/cuda/scratch.rs` (new) —
    `DecodeScratch` and the per-shape allocator.
  - `crates/larql-compute/src/cuda/backend.rs` —
    `ensure_decode_scratch`, `DecodeGraph` cache.
  - `crates/larql-compute/src/cuda/attn.rs` — kernel
    signature change for `pos_dev` + `_into` wrapper.
  - `crates/larql-compute/src/cuda/q4k_mmvq.rs`,
    `q6k_mmvq.rs`, `q4k_direct.rs`, `elem.rs`,
    `matmul.rs` — `_into` wrapper variants.
  - `crates/larql-compute/src/cuda/decode.rs` — new
    scratch-using `decode_token_device`, graph
    capture/replay logic.

- **Affected systems**: GPU only. Metal unaffected.

## Risks and back-out

- **Graph capture failure**. cudarc's `begin_capture` /
  `end_capture` may not support all our kernels (cuBLAS
  GEMV inside the matvec path, NVRTC-compiled kernels).
  Mitigation: env-var back-out + parity test gate.
- **Stale graph after layer-set change**. If the model
  layers change (different model loaded, or shape
  mismatch), the cached graph is invalid. Mitigation:
  cache by shape + layer pointers; invalidate on
  mismatch.
- **Numerical drift**. None expected — same kernels, same
  arithmetic, just different scheduling.
- **Back-out**: `LARQL_CUDA_DECODE_GRAPH=0` reverts to
  the per-call kernel launch path. The new infrastructure
  (scratch buffers, device-side pos, write-into wrappers)
  is a strict superset of the old, so the back-out is
  always available.

## Acceptance bar

Measured on the dev box (RTX 4090, CUDA 12.5, Gemma 3 4B Q4_K
vindex, 6-token prompt, 20 decode tokens after 3 warmup):

| Metric | Pre-change | Target | Comparator |
|---|---:|---:|---:|
| `decode ms/token` | 9.35 | ≤ 7 | llama.cpp 4.40 |
| `tok/s` | 107 | ≥ 140 | llama.cpp 227.5 |
| Bit parity vs no-graph path | — | ≤ 1e-3 max-element | — |

If the decode improvement is < 1 ms (i.e., > 8.5 ms post-
change), launch overhead is not the dominant cost and the
profile will show where the actual wall-clock time goes.
Tensor Cores are then the next move.
