## ADDED Requirements

### Requirement: mmvq f16 conversion SHALL use hardware cvt PTX

The `larql_f16_to_f32` device helper SHALL emit a single inline
PTX `cvt.f32.f16` instruction. The hand-rolled software emulation
(mantissa/exponent reconstruction with a denormal slow path) is
removed in favour of the hardware intrinsic.

`larql_f16_to_f32` in both `Q4K_MMVQ_SRC` and `Q6K_MMVQ_SRC`
SHALL use a single inline PTX `cvt.f32.f16` instruction, not the
hand-rolled software emulation that previously decoded the f16
mantissa/exponent fields manually. The signature and call sites
SHALL remain unchanged so existing callers don't need to include
`<cuda_fp16.h>`.

#### Scenario: parity holds across the swap to hardware cvt

- **WHEN** the existing parity tests
  (`cuda::q4k_mmvq::tests::q4k_mmvq_matches_q4k_direct_on_dequantized_input`,
  `cuda::q6k_mmvq::tests::q6k_mmvq_matches_q6k_f32_on_dequantized_input`,
  and the full decode/prefill suite)
  are run with the hardware-cvt helper
- **THEN** every test SHALL pass at its existing tolerance (1e-3
  max-element absolute difference) without modification
<!-- test: larql_compute::cuda::q4k_mmvq::tests::q4k_mmvq_matches_q4k_direct_on_dequantized_input -->
<!-- test: larql_compute::cuda::q6k_mmvq::tests::q6k_mmvq_matches_q6k_f32_on_dequantized_input -->

### Requirement: bench MUST show decode speedup vs the software emulation

The hardware-cvt swap MUST produce a measurable decode speedup
on the dev box. Specifically, a 10-run averaged
`larql bench --backends cuda --tokens 20 --warmup 3` MUST report
`decode ms/token ≤ 7.65` (≥ 5% improvement over the previous
branch tip's 8.04 ms/tok).

#### Scenario: 10-run avg decode bench shows the speedup

- **WHEN** `LARQL_CUDA_AVAILABLE=1
  LARQL_CUDA_PREFILL_TENSOR_CORES=1 ./target/release/larql bench
  output/gemma-3-4b-it-vindex --backends cuda --tokens 20
  --warmup 3` is averaged over 10 runs
- **THEN** decode ms/token SHALL be ≤ 7.65 ms (vs the previous
  branch tip of 8.04 ms — a 5%+ speedup; actual measured
  improvement on the dev box was 7.5%)
<!-- test: unbacked -->
