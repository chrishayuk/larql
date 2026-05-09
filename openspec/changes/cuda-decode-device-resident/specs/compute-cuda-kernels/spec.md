## ADDED Requirements

### Requirement: CudaBackend MUST expose device-resident matvec variants

The CUDA backend SHALL expose `*_device` companions for every
`QuantMatVec` matvec method (`q4k_matvec_device`,
`q6k_matvec_device`, `q4kf_matvec_device`) plus `f32_gemv_device`.
Each variant SHALL accept a `&CudaSlice<f32>` input and return a
`CudaSlice<f32>` output, leaving intermediate state on the device.
The existing host-input / host-output variants MUST continue to
work and SHALL be implemented as `htod → *_device → dtoh`
wrappers.

#### Scenario: q4k_matvec_device returns same bytes as host variant

- **WHEN** the same Q4_K packed weight + same input are passed to
  both `q4k_matvec(weight, x_host, rows, cols)` and
  `q4k_matvec_device(weight, htod(x_host), rows, cols)` and the
  device output is read back via `dtoh_sync_copy`
- **THEN** the two `Vec<f32>` outputs SHALL be bit-equal
<!-- test: unbacked -->

#### Scenario: device-resident matvec reuses the device weight cache

- **WHEN** `q4k_matvec_device` is called twice with the same
  weight pointer
- **THEN** the second call SHALL not re-upload the packed Q4_K
  bytes (verified by checking the per-backend cache hit metric
  added in `cuda-q4k-device-cache`)
<!-- test: unbacked -->

### Requirement: decode_token MUST keep per-layer state on the device

The CUDA `decode_token` impl SHALL hold the running residual
`h: CudaSlice<f32>` across the layer loop and dispatch the
projection chain via the new `*_device` matvec variants. The
final `Vec<f32>` return SHALL be produced by a single
`dtoh_sync_copy` after the loop completes.

#### Scenario: device-resident decode produces identical output to host fallback

- **WHEN** the same synthetic pipeline-layer input is run through
  `decode_token` and `decode_token` with
  `LARQL_CUDA_DECODE_HOST_FALLBACK=1`
- **THEN** the two output vectors SHALL agree to max-element
  absolute difference ≤ 1e-3
<!-- test: larql_compute::tests::test_cuda_decode::decode_token_phase1_matches_host_fallback -->

#### Scenario: greedy decode against real Gemma 3 4B Q4_K vindex matches host fallback

- **WHEN** `larql bench output/gemma-3-4b-it-vindex --backends
  cuda --tokens 20` is run with the device-resident path and again
  with `LARQL_CUDA_DECODE_HOST_FALLBACK=1`
- **THEN** the generated token-id sequences SHALL be identical
  under greedy sampling
<!-- test: unbacked -->

### Requirement: CUDA decode MUST expose a host-fallback escape hatch

`LARQL_CUDA_DECODE_HOST_FALLBACK=1` (env var) SHALL force the
existing host-bouncing decode path. Default (unset / "0") SHALL
use the new device-resident path.

#### Scenario: env var routes to the host fallback

- **WHEN** `LARQL_CUDA_DECODE_HOST_FALLBACK=1` is set in the
  environment and `decode_token` is called
- **THEN** the host-bouncing implementation
  (`decode_token_host_fallback`) SHALL be invoked
<!-- test: larql_compute::tests::test_cuda_decode::decode_token_phase1_matches_host_fallback -->

### Requirement: GPU rms_norm / silu / add kernels SHALL match the CPU reference

Phase 2 SHALL introduce `rms_norm_vec_device`,
`silu_gate_up_device`, and `add_in_place_device`. Each MUST
produce output within max-element absolute difference ≤ 1e-3 of
the existing CPU helper on synthetic input at Gemma 3 4B
dimensions (hidden = 2560, intermediate = 10240).

#### Scenario: rms_norm_vec_device matches CPU reference

- **WHEN** a random `[hidden=2560]` input is normalised via both
  `rms_norm_vec_device` and `rms_norm_vec`
- **THEN** the per-element absolute difference SHALL be ≤ 1e-3
<!-- test: unbacked -->

#### Scenario: silu_gate_up_device matches CPU reference

- **WHEN** random `gate` + `up` `[inter=10240]` vectors are
  reduced via both `silu_gate_up_device` and `activate(... Silu)`
- **THEN** the per-element absolute difference SHALL be ≤ 1e-3
<!-- test: unbacked -->

#### Scenario: add_in_place_device is bit-equal to CPU add

- **WHEN** two random `[hidden=2560]` vectors are summed via both
  `add_in_place_device` and `add_in_place`
- **THEN** the result SHALL be bit-equal (no reduction order;
  pure pair-wise add)
<!-- test: unbacked -->

### Requirement: K/V cache SHALL live on the device after Phase 3

The `CudaKvCache::layers[*].{k, v}` storage MUST be
`CudaSlice<f32>`, allocated once at
`preallocate_kv_cache_per_layer` time. The
`populate_kv_layer(layer, k_data, v_data, …)` API SHALL accept
host slices and `htod_sync_copy_into` them into the pre-allocated
device buffers.

#### Scenario: populate_kv_layer round-trips through device buffers

- **WHEN** `populate_kv_layer` is called with random K/V data
  and the same layer is later read back via
  `dtoh_sync_copy(&cache.layers[layer].k)`
- **THEN** the read-back data SHALL bit-equal the original input
<!-- test: unbacked -->

#### Scenario: fused_decode_attention_device accepts device K/V cache

- **WHEN** `fused_decode_attention_device` is called with
  `&CudaSlice<f32>` K/V cache pointers (the new contract)
- **THEN** the kernel SHALL not perform an internal H2D copy of
  the K/V slabs (verified by checking the cudarc allocation
  trace for an absent `htod_*` call inside the entry point)
<!-- test: unbacked -->

### Requirement: Bench acceptance gates SHALL match the proposal targets

Each phase MUST clear a quantitative acceptance bar measured on
the dev box (RTX 4090, CUDA 13.1, Gemma 3 4B Q4_K vindex, 20
tokens after 3 warmup) before the next phase MAY merge:

| Phase | `decode ms/token` | `GPU fwd ms/token` |
|---|---:|---:|
| 1 | ≤ 100 | ≤ 95 |
| 2 | ≤ 80 | ≤ 75 |
| 3 | ≤ 60 | ≤ 55 |

A phase that misses by > 25% SHALL not advance to the next
phase; the change owner SHALL profile the residual cost and
document in the PR description before continuing.

#### Scenario: Phase 1 misses gate triggers profile-and-document

- **WHEN** Phase 1 lands and `larql bench …` reports
  `decode ms/token > 125`
- **THEN** Phase 2 work SHALL not be merged until a profile
  document explaining the residual overhead has been added to
  the change's PR description
<!-- test: unbacked -->
