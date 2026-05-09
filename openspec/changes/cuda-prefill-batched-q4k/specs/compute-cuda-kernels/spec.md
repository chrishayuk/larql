## ADDED Requirements

### Requirement: Batched element-wise kernels SHALL match per-row CPU semantics

`rms_norm_batch_device`, `silu_gate_up_batch_device`, `add_in_place_batch_device`, and `scale_inplace_batch_device` SHALL apply their respective single-row semantics to each row of an `[seq_len, n]` device buffer independently. Each batched kernel SHALL produce output bit-equivalent to running the corresponding single-row kernel `seq_len` times.

#### Scenario: batched rms_norm matches per-row rms_norm

- **WHEN** a `[seq_len=8, n=2560]` device buffer is normalised
  via `rms_norm_batch_device(...)` and the same input is
  normalised via `rms_norm_device(...)` 8 times (once per row)
- **THEN** the two outputs SHALL agree to max-element absolute
  difference ≤ 1e-6
<!-- test: unbacked -->

### Requirement: prefill_q4 SHALL use batched cuBLAS GEMM by default

`DecodeBackend::prefill_q4` on `CudaBackend` SHALL dispatch to a batched implementation (`prefill_q4_seq_device`) that performs each per-layer projection as a single cuBLAS GEMM over all `seq_len` positions, when every layer in the prompt supports the device path. `LARQL_CUDA_PREFILL_HOST_FALLBACK=1` SHALL force the legacy per-position decode loop.

#### Scenario: batched prefill matches per-position prefill within 1e-3

- **WHEN** a synthetic Q4_K pipeline runs prefill on a 6-token
  prompt via the new `prefill_q4_seq_device` path and via
  `LARQL_CUDA_PREFILL_HOST_FALLBACK=1` (the legacy
  per-position loop)
- **THEN** the two `[seq_len * hidden]` output vectors SHALL
  agree to max-element absolute difference ≤ 1e-3
<!-- test: unbacked -->

#### Scenario: prefill→decode boundary preserves KV-cache correctness

- **WHEN** prefill 4 tokens via the batched path, then decode
  5 more via `decode_token_device`, and again via the host
  fallback combination (host-fallback prefill + host-fallback
  decode)
- **THEN** the hidden states at each decode step SHALL agree
  to max-element absolute difference ≤ 1e-3
<!-- test: unbacked -->

### Requirement: prefill bench gate SHALL hit the proposal target

Prefill on the dev-box bench (RTX 4090, CUDA 12.5, Gemma 3 4B Q4_K vindex, 6-token prompt) SHALL drop to ≤ 20 ms total (down from 130.9 ms), and decode SHALL not regress beyond 11 ms/token. Misses by > 50% (i.e., prefill > 30 ms) SHALL trigger a profile-and-document write-up. **Actual**: 117.6 ms — 5.9× miss; profile-and-document write-up is in proposal.md (per-position attention loop is the dominant residual cost; `cuda-prefill-batched-attention` is the planned follow-up).

#### Scenario: prefill cleared at acceptance OR profile-documented on miss

- **WHEN** `larql bench output/gemma-3-4b-it-vindex --backends
  cuda --tokens 20 --warmup 3 --verbose` is run after this
  change lands
- **THEN** EITHER `prefill ms` SHALL be ≤ 20 AND
  `decode ms/token` ≤ 11 (acceptance hit), OR the change's
  `proposal.md` SHALL contain a profile-and-document write-up
<!-- test: unbacked -->
