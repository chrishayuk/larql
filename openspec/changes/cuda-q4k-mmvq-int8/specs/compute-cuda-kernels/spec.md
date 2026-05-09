## ADDED Requirements

### Requirement: Q8_1 quantize kernel SHALL match llama.cpp byte layout

A new `quantize_q8_1_device(x_dev, n)` SHALL produce a
device-resident `Q8_1Buf { qs: CudaSlice<i8>, ds: CudaSlice<u8>
}` whose byte layout matches llama.cpp's `block_q8_1` (32-element
groups, `half2 ds` carrying scale + scale*sum, then 32 `int8_t`
quants per group). `n` MUST be a multiple of 32; the kernel
SHALL refuse other sizes.

#### Scenario: Q8_1 quantize → dequantize round-trips to within one quantum

- **WHEN** a synthetic `[hidden=2560]` f32 vector is quantized
  via `quantize_q8_1_device` and the resulting blocks are
  dequantized on host
- **THEN** the per-element absolute error SHALL be ≤
  `(amax_block / 127.0)`, i.e. one Q8_1 quantum, for every
  element of every block
<!-- test: larql_compute::cuda::elem::tests::q8_1_quantize_roundtrips_to_within_quant_noise -->

### Requirement: Q4_K × Q8_1 mmvq kernel SHALL match the existing direct f32 kernel

A new `q4k_matvec_device_mmvq` SHALL produce output that agrees with the existing `q4k_direct::matvec_device` to max-element absolute difference ≤ 1e-3. The mmvq kernel SHALL
use `__dp4a` four-way INT8 SIMD dot products and SHALL accept
input in the Q8_1 layout (not f32). The kernel body SHALL be
ported close-to-verbatim from llama.cpp's
`vec_dot_q4_K_q8_1_impl_vmmq` with provenance recorded in the
NVRTC source comment.

#### Scenario: mmvq output matches the f32 direct kernel within 1e-3

- **WHEN** a random Q4_K packed weight `[rows=4096, hidden=2560]`
  is multiplied by a random f32 input via both
  `q4k_matvec_device` (f32 direct path) and the new
  `q4k_matvec_device_mmvq` path (input quantized to Q8_1 first)
- **THEN** the two output `Vec<f32>`s SHALL agree to max-element
  absolute difference ≤ 1e-3
<!-- test: unbacked -->

### Requirement: Q4_K matvec dispatch SHALL be runtime-selectable

`LARQL_CUDA_Q4K_MMVQ` env var SHALL select the Q4_K matvec
kernel: `0` forces the existing direct-f32 path, `1` (the
default after Phase 3) routes through the new mmvq path. The
new code SHALL be additive — the old kernel MUST stay compiled
and reachable so the env var is a true back-out.

#### Scenario: env var routes between mmvq and direct paths

- **WHEN** `LARQL_CUDA_Q4K_MMVQ=0` is set in the environment
  and the same decode call is made
- **THEN** the existing `q4k_direct::matvec_device` kernel
  SHALL be invoked (verified via the same parity test under
  both flag values)
<!-- test: unbacked -->

### Requirement: decode_token_device SHALL share Q8_1 input across same-input projections

The device-resident decode path SHALL quantize the layer-input
vector `h_attn_dev` to Q8_1 once and pass the cached form to
the q, k, and v projections. Same for `h_ffn_dev` across the
gate and up projections. wo and down MAY remain on the existing
f32 direct path for this change; whether to migrate them is
deferred to a follow-up `cuda-q4k-mmvq-extend` change.

#### Scenario: greedy decode against real Gemma 3 4B vindex matches host fallback under mmvq

- **WHEN** `larql bench output/gemma-3-4b-it-vindex --backends
  cuda --tokens 20` is run with `LARQL_CUDA_Q4K_MMVQ=1` (the
  default) and again with `LARQL_CUDA_DECODE_HOST_FALLBACK=1`
- **THEN** the generated token-id sequences SHALL be identical
  under greedy sampling
<!-- test: unbacked -->

### Requirement: Phase 3 SHALL clear a quantitative bench gate

Phase 3 SHALL clear a quantitative decode-throughput bar before
merging. Measured on the dev box (RTX 4090, CUDA 12.5, Gemma 3
4B Q4_K vindex, 20 tokens after 3 warmup) the bench SHALL
report:

| Metric | Pre-change | Phase 3 target |
|---|---:|---:|
| `decode ms/token` | 19.49 | ≤ 10 |
| `GPU fwd ms/token` | 17.491 | ≤ 8 |
| `tok/s` | 51.3 | ≥ 100 |

A miss by > 25% (decode > 12.5 ms/tok) SHALL trigger a profile
write-up in the PR description before the change is merged or
re-tried.

#### Scenario: bench cleared at acceptance

- **WHEN** `larql bench output/gemma-3-4b-it-vindex --backends
  cuda --tokens 20 --warmup 3 --verbose` is run on the dev box
  after Phase 3 lands and `LARQL_CUDA_Q4K_MMVQ=1` is the default
- **THEN** the reported `decode ms/token` SHALL be ≤ 10 and
  `GPU fwd ms/token` ≤ 8; otherwise the change SHALL not be
  archived
<!-- test: unbacked -->
