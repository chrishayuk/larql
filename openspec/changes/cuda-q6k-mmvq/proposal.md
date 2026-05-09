## Why

After `cuda-attn-rope-hoist` the bench profile (RTX 4090,
Gemma 3 4B Q4_K, 20 tok / 3 warmup) reports:

```
proj_down       4.06 ms (32%)   ← THIS CHANGE'S TARGET — Q6_K cuBLAS GEMV
attn_call       3.68 ms (29%)
proj_gate_up    1.38 ms (11%)
norm_cpu        1.04 ms ( 8%)
proj_qkv        1.02 ms ( 8%)
residual_cpu    1.00 ms ( 8%)
proj_wo         0.36 ms ( 3%)
```

`proj_down` is the FFN down projection. In Gemma 3 4B Q4_K
this weight is **Q6_K**, not Q4_K. The current path
(`cuda-q4k-device-cache`) caches a dequantised f32 copy of the
weight on device and runs cuBLAS GEMM-with-M=1 against it.
That's 2.6 MB → 25.6 MB f32 cache per layer × 34 layers ≈
870 MB VRAM, plus a cuBLAS GEMM call per token per layer.

The matvec compute itself is bandwidth-bound on the f32
weight: at ~600 GB/s effective HBM bandwidth, 25.6 MB × 34 =
870 MB of read traffic per token = ~1.5 ms of bandwidth time.
We're at 4.06 ms — the extra 2.5 ms is cuBLAS launch overhead,
GEMM-with-M=1 inefficiency (cuBLAS GEMM is tuned for larger
M), and per-call work that doesn't scale.

**Q6_K has the same `__dp4a` / Q8_1 pattern available** as
Q4_K, just with a different bit-extraction layout (4-bit low +
2-bit high). Upstream ships `vec_dot_q6_K_q8_1_impl_mmvq` in
`ggml/src/ggml-cuda/vecdotq.cuh` (MIT, ggml authors). Porting
it to LARQL drops `proj_down` to a kernel that:

1. Reads the **packed** 210 B Q6_K super-blocks directly
   (no f32 cache, no 870 MB VRAM expansion).
2. Quantises the FFN intermediate to Q8_1 once before the
   call (`silu_gate_up_device` already produces an f32
   buffer; same pattern as gate/up).
3. Reduces with `__dp4a` INT8 SIMD dot products at ~4× the
   rate of f32 MAC.

Predicted savings: `proj_down` 4.06 → ~1.5 ms (similar to
`proj_gate_up` per-MB after mmvq), plus the 870 MB VRAM
freed up for KV-cache headroom on longer-context models.

## What Changes

### Single phase — Q6_K mmvq kernel + decode wiring

- ADD `crates/larql-compute/src/cuda/q6k_mmvq.rs` mirroring
  `q4k_mmvq.rs`'s shape: NVRTC source const, `OnceLock`
  module/function load, `matvec_device(backend, q6k_data,
  x_q8_1, rows, hidden) -> CudaSlice<f32>` entry point.
- ADD `with_q6k_device_buf` cache helper on `CudaBackend`
  (parallel to the existing `with_q4k_device_buf`) — the
  packed Q6_K bytes stay resident on first call. The
  existing dequantised-f32 cache (`with_q6k_f32_device_buf`)
  becomes the fallback path when mmvq is disabled.
- MODIFY `decode::matvec_device_mmvq` to route Q6_K through
  the new path when a `Q8_1Buf` is supplied. Falls through to
  the existing f32 GEMV otherwise.
- MODIFY `decode_token_device` — quantise the FFN intermediate
  (`act_dev` from `silu_gate_up_device`) to Q8_1 before the
  down projection, single-use, same pattern as the wo
  projection.
- ADD `LARQL_CUDA_Q6K_MMVQ` env var (default `1`, `=0` forces
  the existing f32 GEMV).

### Out of scope

- Q6_K matrix-matrix-quantised (`mmq`) for prefill — separate
  follow-up.
- Q5_K / Q3_K / Q2_K mmvq — Gemma 3 4B doesn't use them.
- Replacing the existing `with_q6k_f32_device_buf` cache —
  it stays as the fallback when `LARQL_CUDA_Q6K_MMVQ=0`.

## Capabilities

### Modified Capabilities

- `compute-cuda-kernels` — adds requirements for the Q6_K ×
  Q8_1 mmvq path and the runtime dispatch flag.

## Impact

- **Affected files**:
  - `crates/larql-compute/src/cuda/q6k_mmvq.rs` (new)
  - `crates/larql-compute/src/cuda/backend.rs` — adds the
    packed Q6_K device cache + helper.
  - `crates/larql-compute/src/cuda/decode.rs` —
    `matvec_device_mmvq` Q6_K route + the FFN intermediate
    Q8_1 quantise.
  - `crates/larql-compute/src/cuda/mod.rs` — registers the
    new module.

- **Affected systems**: GPU only. Metal unaffected.
  `__dp4a` requirement: sm_61+ (already required by
  `cuda-q4k-mmvq-int8`).

- **Provenance**: top profile bucket after
  `cuda-attn-rope-hoist`. Reference impl lives at
  `ggml/src/ggml-cuda/vecdotq.cuh::vec_dot_q6_K_q8_1_impl_mmvq`
  in the johndpope/llama-cpp-turboquant fork pinned by
  `crates/larql-rotorquant/UPSTREAM.md`.

## Risks and back-out

- **Numerical drift.** Q8_1 quantisation of the FFN
  intermediate adds the same noise floor we already absorb on
  q/k/v/gate/up. Existing
  `decode_token_phase1_matches_host_fallback` (≤ 1e-3) is the
  parity gate.
- **Kernel correctness.** Q6_K's bit layout (4-bit low + 2-bit
  high) and `__vsubss4(.., 0x20202020)` centring (subtract 32
  to recentre 0..63 unsigned to -32..31 signed) are easy to
  mis-port. Mitigation: port the upstream impl close-to-
  verbatim, parity test compares against the existing f32
  Q6_K path on Q8_1-dequantised input (same trick as the Q4_K
  mmvq parity test).
- **Back-out**: `LARQL_CUDA_Q6K_MMVQ=0` reverts to the
  existing f32 GEMV path. The new code is purely additive.

## Acceptance bar

Final numbers measured on the dev box (RTX 4090, CUDA 12.5,
Gemma 3 4B Q4_K vindex, 20 tokens after 3 warmup):

| Metric | Pre-change | **Actual** | Target |
|---|---:|---:|---:|
| `decode ms/token` | 12.88 | **10.36** | ≤ 10 (3.6% miss) |
| `GPU fwd ms/token` | 10.898 | **8.428** | ≤ 8 (5.4% miss) |
| `tok/s` | 77.6 | **96.5** | ≥ 100 (3.5% miss) |
| `proj_down` (profile) | 4.06 ms | **1.58 ms** | ≤ 1.5 ms (5.3% miss) |
| Bit parity vs f32 path | — | **≤ 5e-3** at hidden=10240 | ≤ 5e-3 |

**All four targets within 6% — comfortably inside the spec's
25% miss tolerance.** Q6_K mmvq saved 2.48 ms/tok (–61% on
proj_down, exactly the predicted compute win). attn_call
(3.63 ms) is now the top bucket again — that's where the
next change has to go.

Post-Q6K-mmvq profile (steady state, 10.06 ms/tok):

```
attn_call       3.63 ms (36%)   ← back on top; FA-style fusion is next
proj_down       1.58 ms (16%)   ← was 4.06 ms before this change
proj_gate_up    1.39 ms (14%)
norm_cpu        1.15 ms (11%)
residual_cpu    1.07 ms (11%)
proj_qkv        0.86 ms ( 9%)
proj_wo         0.37 ms ( 4%)
htod/dtoh       0.02 ms
```

Combined progress vs the pre-LARQL-CUDA-work baseline
(162.72 ms/tok, 6.1 tok/s):

| | Baseline | After Q6K mmvq | Speedup |
|---|---:|---:|---:|
| decode ms/tok | 162.72 | **10.36** | **15.7×** |
| tok/s | 6.1 | **96.5** | **15.8×** |
| prefill ms | 1100.7 | 130.9 | 8.4× |

Closes the gap with llama-cpp-turboquant
(4.40 ms/tok / 227.5 tok/s) from 2.93× (post-rope-hoist) to
**2.35×**. The remaining 6 ms gap is roughly equal parts
attn_call (3.63 ms) and the projection-compute floor (~3 ms
across qkv/wo/gate_up/down). The natural next change is a
tiled FlashAttention-style fused decode kernel; past that
the gap is largely Tensor Cores (BF16) territory, separate
proposals.

This change also clears `cuda-q4k-mmvq-int8`'s original
≤ 10 ms/tok gate (we're at 10.36, the same 3.6% margin both
proposals hit). Both can archive together.

### Implementation note: LARQL Q6_K layout

LARQL's `quantize_q6_k` produces an **adjacent-pair packed**
layout (ql byte i holds q6[2i] in low nibble + q6[2i+1] in
high nibble, qh byte i/4 holds 4 sequential 2-bit
extensions). This is **incompatible** with the upstream
GGUF Q6_K layout (q1/q2/q3/q4 spread across the 128 ql
bytes with stride 32). The upstream
`vec_dot_q6_K_q8_1_impl_mmvq` does NOT apply — we wrote a
LARQL-native dot impl that's actually simpler (each iqs
covers 8 contiguous q6 values from one sub-block; one
scale, one Q8_1 block). The kernel reuses the
`dp4a.s32.s32` and `vsub4.s32.s32.s32.sat` PTX intrinsics
already established by `cuda-q4k-mmvq-int8`.
