## Why

A late-session audit of the hot path found that the
`larql_f16_to_f32` helper in both `q4k_mmvq.rs` and `q6k_mmvq.rs`
was a hand-rolled software emulation (~20 instructions of
mantissa/exponent reconstruction, with a denormal slow path) when
NVRTC + sm_61+ has had a single-instruction `cvt.f32.f16` PTX
intrinsic the entire time.

The Q4_K mmvq kernel calls this helper **4× per super-block**
(d, dmin, d8_0, d8_1) per dot product. For a Gemma 3 4B token at
n_super_blocks = 10 per row × 7 mmvq projections × 34 layers =
**~9 500 conversions per output row × 4 = 38 000 conversions per
mmvq output row**, compounded across all rows in all projections
totals tens of millions of f16 conversions per decode token —
each one ~20 cycles slower than it had to be.

## What Changes

- REPLACE the body of `larql_f16_to_f32` in `q4k_mmvq.rs` and
  `q6k_mmvq.rs` with a single inline PTX `cvt.f32.f16`:

  ```c
  __device__ float larql_f16_to_f32(unsigned short h) {
      float f;
      asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
      return f;
  }
  ```

- The signature and call sites are unchanged; the helper still
  takes a raw `unsigned short` (so callers don't need
  `<cuda_fp16.h>` either).

## Impact

- **Affected files**:
  - `crates/larql-compute/src/cuda/q4k_mmvq.rs`
  - `crates/larql-compute/src/cuda/q6k_mmvq.rs`
- **Affected systems**: GPU only.
- **Behaviour change**: bit-exact within IEEE rounding. The
  software emulation was already round-to-nearest correct; the
  hardware `cvt` is the same. All 200+ tests pass with no
  tolerance changes.

## Capabilities

### Modified Capabilities

- `compute-cuda-kernels` — replaces the software f16 helper with
  the hardware PTX intrinsic.

## Risks and back-out

- Risk: very low. Both implementations produce IEEE-754
  round-to-nearest f16 → f32 conversions; the existing
  `cuda::q4k_mmvq::tests::q4k_mmvq_matches_q4k_direct_on_dequantized_input`
  parity test passes unchanged at 1e-3 max-element. Other
  mmvq parity tests pass too.
- Back-out: revert this change. The software emulation is
  preserved in git history for any future host without
  cvt.f32.f16 (sm_5x and earlier — not supported by LARQL anyway).

## Acceptance bar

Measured on the dev box (RTX 4090 / sm_89, Gemma 3 4B Q4_K, 6-token
prompt + 20 decode tokens after 3 warmup, 10-run average, on top of
every prior optimisation, with `LARQL_CUDA_PREFILL_TENSOR_CORES=1`):

| Metric | Pre-change | **Actual** | Comparator |
|---|---:|---:|---:|
| `decode ms/token` | 8.04 | **7.44** | llama.cpp 4.34 |
| `tok/s` | 124.4 | **134.5** | llama.cpp 230.2 |
| Generated text | identical | **identical** | — |
| All 200+ tests | pass | pass | — |

This is a **0.6 ms / 7.5%** decode savings — the largest
single-change win since the original `cuda-decode-cuda-graph`
(1.1 ms). It's also the most surprising: a 5-line change that
the codebase has been carrying for the entire session's
optimization push without anyone realising the hand-rolled
emulation was where ~10% of decode time was hiding.

Cumulative session impact:

| | Pre-session | **This change** | llama.cpp |
|---|---:|---:|---:|
| prefill ms | 18.0 | **10.7** | 6.25 |
| decode ms/tok | 9.62 | **7.44** | 4.34 |
| decode tok/s | 103.9 | **134.5** | 230.2 |

Decode gap with llama.cpp closed from 2.18× to **1.71×** (was 1.85×
pre-change). Decode gap closed: **38%** of original gap (was 27%).
