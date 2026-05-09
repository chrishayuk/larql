## Why

LARQL prefill on Gemma 3 4B Q4_K runs at 18.0 ms for a 6-token
prompt vs llama-cpp-turboquant's 5.6 ms — a 3.2× gap. The prefill
projection chain is dominated by f32 cuBLAS GEMM (`sgemm` → no
Tensor Cores). On Ada / Ampere / Hopper, dispatching the same GEMM
through `hgemm` (f16 inputs, f32 accumulator) routes through Tensor
Cores at ~2-4× the SGEMM throughput.

Decode (single-token mmvq via `__dp4a`) is HBM-bandwidth-bound and
already efficient — Tensor Cores don't help it. The prefill path
is the right target for f16/Tensor-Core work.

## What Changes

### Phase 1: f32 ↔ f16 conversion kernels

- ADD `elem::f32_to_f16_device` / `elem::f16_to_f32_device` —
  element-wise PTX `cvt.rn.f16.f32` / `cvt.f32.f16` kernels. One
  thread per element, 256-threads per block.

### Phase 2: f16 weight cache

- ADD `q4k_f16_device_cache: HashMap<DeviceBytesKey, Arc<CudaSlice<half::f16>>>`
  + `with_q4k_f16_device_buf(host, n_elements, |w_dev| ...)` — on
  first call dequant Q4_K to f32, downcast each element to f16 on
  the host, htod the f16 buffer. Halves the device memory of the
  equivalent `q4k_f32_device_cache` (4 B/elem → 2 B/elem).
- Same for Q6_K (`q6k_f16_device_cache`).

### Phase 3: f16 cuBLAS GEMM

- ADD `matmul::matmul_transb_device_inout_f16` — `cublasGemmEx`
  with `CUDA_R_16F` inputs and `CUBLAS_COMPUTE_32F` accumulator
  (Tensor Core dispatch on supported GPUs).

### Phase 4: prefill dispatch

- MODIFY `decode::gemm_proj_seq` to gate on
  `LARQL_CUDA_PREFILL_TENSOR_CORES=1`:
  1. Convert `x_seq` (f32) → fresh f16 buffer.
  2. Lookup cached f16 weight via `with_q4k_f16_device_buf`.
  3. cuBLAS hgemm.
  4. Convert result (f16) → fresh f32 buffer.
- The default keeps the existing f32 path so the f16 cache only
  populates when the user opts in (it doubles the cached projection
  weights' memory footprint over the existing f32 cache, since both
  caches can coexist).

## Out of scope

- **Decode-time Tensor Cores**: decode mmvq is bandwidth-bound on
  the Q4_K read; HMMA wouldn't speed it up. INT4-Tensor-Core
  matvec (sm_80+ IMMA) is a separate, larger change.
- **f16 throughout the prefill pipeline**: this change keeps the
  norm / silu / KV-write kernels in f32 and only swaps the
  projection GEMM to f16 + Tensor Cores. The conversion overhead
  is small (one element-wise pass per call) compared to the GEMM
  win.
- **bf16 / int8**: bf16 has the same Tensor Core dispatch as f16
  on Ada+; we use f16 for now because cuBLAS doesn't gate it
  differently and f16 has wider support across older cards.

## Capabilities

### Modified Capabilities

- `compute-cuda-kernels` — adds the f16 prefill GEMM contract +
  `LARQL_CUDA_PREFILL_TENSOR_CORES` env var.

## Impact

- **Affected files**:
  - `crates/larql-compute/src/cuda/elem.rs` — `F32_F16_CONVERT_SRC`
    PTX, `f32_to_f16_device` / `f16_to_f32_device` wrappers.
  - `crates/larql-compute/src/cuda/matmul.rs` —
    `matmul_transb_device_inout_f16`.
  - `crates/larql-compute/src/cuda/backend.rs` — two new caches +
    `with_q4k_f16_device_buf` / `with_q6k_f16_device_buf`.
  - `crates/larql-compute/src/cuda/decode.rs` — env-var gate +
    f16 path inside `gemm_proj_seq`.
  - `crates/larql-compute/Cargo.toml` — `cudarc` `f16` feature +
    direct `half = "2"` dep.
- **Affected systems**: GPU only; Metal unaffected.

## Risks and back-out

- **Numerical drift**: f16 has ~3-4 decimal digits of precision.
  Mitigation: cuBLAS hgemm uses `CUBLAS_COMPUTE_32F` accumulator,
  so per-output-element error is bounded by the f16 input quant
  noise (~5e-4 relative). Empirically the existing 1e-3 parity
  tests pass with the f16 path enabled.
- **Memory pressure**: the f16 cache adds ~6 GB on Gemma 3 4B
  (full Q4_K + Q6_K weight set, half-precision). Coexists with
  the f32 cache, so total prefill cache footprint roughly
  doubles. Mitigation: env-var-gated; the f16 cache only
  populates when `LARQL_CUDA_PREFILL_TENSOR_CORES=1` is set, so
  hosts with tighter memory can skip it.
- **Back-out**: unset `LARQL_CUDA_PREFILL_TENSOR_CORES` (or set to
  any value other than `1`) reverts to the existing f32 path. The
  f16 caches stay allocated for the session but the f16 GEMM
  isn't called.

## Acceptance bar

Measured on the dev box (RTX 4090, CUDA 12.5, Gemma 3 4B Q4_K
vindex, 6-token prompt, 20 decode tokens after 3 warmup, 5-run
average):

| Metric | Pre-change | Target | **Actual** | Comparator |
|---|---:|---:|---:|---:|
| `prefill ms` | 18.0 | ≤ 12 | **10.7** | llama.cpp 5.6 |
| `decode ms/token` | 8.33 | ± noise | **8.46** | llama.cpp 4.40 |
| Generated text parity | — | identical | **identical** | — |

The 40% prefill reduction is the headline. Decode is unaffected
(within run-to-run noise) — the change only replaces the prefill
GEMM. Closes most of the prefill gap with llama-cpp-turboquant;
the residual ~5 ms is likely in the prefill attention path (per-
position fused_decode_attention loop), which the next change can
attack.
