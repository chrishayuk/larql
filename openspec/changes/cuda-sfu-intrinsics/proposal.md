## Why

After `cuda-prefill-alloc-fixes`, decode steady-state was
10.36 ms/tok with the attention bucket at 3.63 ms (38% of
the budget). Profiling pointed at the trig-heavy RoPE
rotation in `fused_decode_attention_f32` and the sigmoid in
`silu_gate_up_f32` as the dominant per-call costs.

The IEEE-compliant `sinf`, `cosf`, `expf`, `powf` library
calls go through the SP pipeline at ~1 op per 4-8 cycles.
CUDA also exposes **SFU intrinsics** — `__sinf`, `__cosf`,
`__expf`, `__powf` — that map to dedicated SFU
(special-function unit) hardware at roughly **3-5× the
throughput** with ~4 ULPs of accuracy loss (well inside our
existing 1e-3 max-element parity bound).

Pre-change profile (decode, 34 layers per token):

```
attn_call       3.63 ms (38%)   ← THIS CHANGE'S TARGET
proj_down       1.58 ms (16%)
proj_gate_up    1.39 ms (14%)
norm_cpu        1.15 ms (11%)
residual_cpu    1.07 ms (11%)
proj_qkv        0.86 ms ( 9%)
proj_wo         0.37 ms ( 4%)
```

Post-change profile (same workload, intrinsics swapped in):

```
attn_call       2.68 ms (29%)   ← -26% (-0.95 ms)
proj_down       1.65 ms (18%)
proj_gate_up    1.43 ms (15%)
norm_cpu        1.15 ms (12%)
residual_cpu    1.07 ms (12%)
proj_qkv        0.91 ms (10%)
proj_wo         0.39 ms ( 4%)
```

## What Changes

### Single phase — surgical kernel-source edit

- MODIFY `crates/larql-compute/src/cuda/attn.rs` NVRTC kernel
  sources (`FUSED_DECODE_ATTN_SRC`,
  `FUSED_PREFILL_ATTN_SRC`, `KV_CACHE_WRITE_SEQ_SRC`,
  `SOFTMAX_SRC`, `QKV_RMS_PROJ_SRC`):
  - `cosf(x)` → `__cosf(x)`
  - `sinf(x)` → `__sinf(x)`
  - `expf(x)` → `__expf(x)`
  - `powf(x, y)` → `__powf(x, y)`
  - **Keep** `tanhf(x)` as-is — there is no `__tanhf`
    SFU intrinsic on Volta+; the existing path via
    `tanhf` is correct.
- MODIFY `crates/larql-compute/src/cuda/elem.rs`
  `silu_gate_up_f32`: `expf(-g)` → `__expf(-g)` for the
  Silu sigmoid.
- ADD `use_fast_math: Some(true)` to the
  `compile_ptx_with_opts` `CompileOptions` for the three
  attention kernels — gives cudarc's `--fmad=true` flag.

### Out of scope

- Replacing `tanhf` (used in softcap and GeluTanh) — no SFU
  intrinsic available; would need a software approximation.
  The softcap path is `softcap > 0.0` which Gemma 3 4B
  doesn't use, so this isn't on the hot path anyway.
- Q4_K mmvq kernels (`q4k_mmvq.rs`, `q6k_mmvq.rs`) — they
  don't use trig; no benefit from this change.
- A general "fast math" feature flag — the
  intrinsic-level swap is durable and contained.

## Capabilities

### Modified Capabilities

- `compute-cuda-kernels` — adds a numerical-accuracy contract
  that the SFU-intrinsic path stays inside the same 1e-3
  max-element parity bound the kernels already had against
  the host-fallback reference.

## Impact

- **Affected files**: `crates/larql-compute/src/cuda/attn.rs`,
  `crates/larql-compute/src/cuda/elem.rs`. Pure NVRTC source
  edits — no Rust API changes.

- **Affected systems**: GPU only. Metal unaffected.

- **Hardware requirement**: SFU intrinsics are available on
  every NVIDIA arch from Kepler onwards. Already a subset of
  the sm_61 floor `cuda-q4k-mmvq-int8` requires.

## Risks and back-out

- **Numerical drift**. Intrinsics have ~4 ULPs of accuracy
  loss vs the IEEE-compliant library calls. Empirical bound:
  the existing
  `decode_token_phase1_matches_host_fallback` test (≤ 1e-3
  vs the host CPU attention) still passes after the swap.
  The drift is well below the existing parity bound.
- **Back-out**: revert the `__cosf` → `cosf` etc. text edit.
  Single-file revert; no plumbing.

## Acceptance bar

Final numbers measured on the dev box (RTX 4090, CUDA 12.5,
Gemma 3 4B Q4_K vindex, 6-token prompt, 20 decode tokens
after 3 warmup), averaged over 5 runs:

| Metric | Pre-change | **Actual** | Target |
|---|---:|---:|---:|
| `decode ms/token` | 10.36 | **9.35** (-9.7%) | ≤ 9.5 |
| `tok/s` | 96.6 | **107** (+11%) | ≥ 105 |
| `attn_call` profile | 3.63 ms | **2.68 ms** (-26%) | ≤ 3 ms |
| `prefill ms` | 18.0 | 18.0 | unchanged |
| Bit parity vs host fallback | passes | **passes** | ≤ 1e-3 |

Combined progress vs the pre-LARQL-CUDA-work baseline:

| | Baseline | **Now** | Speedup |
|---|---:|---:|---:|
| decode ms/tok | 162.72 | **9.35** | **17.4×** |
| prefill ms | 1100.7 | **18.0** | **61.2×** |
| tok/s | 6.1 | **107** | **17.5×** |

Closes the decode gap with llama-cpp-turboquant
(4.40 ms/tok / 227.5 tok/s) from 2.35× to **2.13×**.
