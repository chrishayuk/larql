## ADDED Requirements

### Requirement: decode_token_device SHALL use a pre-allocated scratch buffer

`CudaBackend` SHALL maintain a `DecodeScratch` struct holding all per-decode intermediate buffers (one buffer per role: h, h_attn, q, k, v, attn_out, attn_delta, normed, h_ffn, gate, up, act, ffn_delta) plus the matching Q8_1 scratches and a `pos_dev: CudaSlice<i32>`. The scratch SHALL be lazy-allocated on the first decode call and reused for every subsequent call with the same shape.

#### Scenario: scratch reused across decode calls

- **WHEN** `decode_token_device` is called five times in a row
  with the same model
- **THEN** the second through fifth calls SHALL NOT allocate
  any new device buffers for intermediate state
<!-- test: unbacked -->

### Requirement: fused_decode_attention_device_kv SHALL take device-side pos

The `fused_decode_attention_f32` NVRTC kernel SHALL read its `pos` value from a device-side `int*` argument, NOT as an immediate kernel arg. The Rust wrapper SHALL take a `pos_dev: &CudaSlice<i32>` instead of a host-side `pos: usize`. This allows CUDA Graph replay to update `pos` between launches without re-capture.

#### Scenario: pos updates between graph replays

- **WHEN** a captured decode graph is launched twice with
  different `pos_dev` contents (e.g., positions 5 and 6)
- **THEN** the two replays SHALL produce outputs corresponding
  to those two `pos` values, NOT both producing the
  capture-time `pos`
<!-- test: unbacked -->

### Requirement: decode_token_device SHALL replay a captured CUDA graph after the first call

Once `DecodeScratch` is allocated and a CUDA graph has been captured for a given (shape, layer-set) pair, subsequent calls SHALL replay the graph (`htod` new `pos` + new `x`, then `graph.launch()`, then `dtoh` `h`) instead of re-issuing every kernel launch. `LARQL_CUDA_DECODE_GRAPH=0` SHALL force the legacy per-call launch path.

#### Scenario: bench gate cleared after graph replay

- **WHEN** `larql bench output/gemma-3-4b-it-vindex --backends
  cuda --tokens 20 --warmup 3 --verbose` is run with the
  default `LARQL_CUDA_DECODE_GRAPH=1`
- **THEN** the reported `decode ms/token` SHALL be ≤ 7 AND
  `tok/s` ≥ 140 (target — substantial improvement over the
  9.35 ms / 107 tok/s pre-change baseline)
<!-- test: unbacked -->
