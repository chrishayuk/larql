## ADDED Requirements

### Requirement: rms_norm_add_f32 SHALL fuse rms_norm + add_residual + optional scale

`rms_norm_add_f32` SHALL compute
`dst[i] += rms_norm(src, weight)[i] * scale` in a single
single-block kernel with grid `(1, 1, 1)` and block
`(1024, 1, 1)`, using shared memory for the row sum-of-squares
reduction. `has_weight = 0` SHALL skip the per-element multiply
(`rms_norm` then collapses to `src[i] * inv_rms`). `scale = 1.0`
SHALL leave the residual add unscaled; non-1.0 values fold the
post-FFN per-layer scalar (Gemma 4 `layer_scalar`) into the
fusion.

#### Scenario: parity with the unfused chain

- **WHEN** `decode_token_graph_matches_per_call_over_5_steps`
  runs with the captured-graph pipeline using
  `rms_norm_add_device` and again with the legacy
  per-call kernel-launch path (`LARQL_CUDA_DECODE_GRAPH=0`)
  using the unfused `rms_norm_device + add_in_place_device +
  scale_inplace_device` chain
- **THEN** per-step max-element absolute difference SHALL be
  ≤ 1e-3
<!-- test: larql_compute::tests::test_cuda_decode::decode_token_graph_matches_per_call_over_5_steps -->

### Requirement: captured-decode pipeline SHALL drop the normed intermediate buffers at fusion sites

`decode::run_decode_pipeline_into_scratch` SHALL call
`rms_norm_add_device` (instead of the legacy
`rms_norm_device_into → add_in_place_device` pair) at the
post-attn residual site (`scratch.h += rms_norm(attn_delta,
post_attn_norm)`) and the post-ffn residual site (`scratch.h
+= rms_norm(ffn_delta, post_ffn_norm) * layer_scalar`). The
post-FFN call SHALL pass `layer_scalar` through the kernel's
`scale` arg, eliminating the third `scale_inplace_device`
launch when `layer_scalar != 1.0`.

#### Scenario: scratch buffers no longer hold normed intermediates

- **WHEN** `decode_token_device_graph_attempt` runs the
  captured pipeline
- **THEN** `scratch.attn_normed` and `scratch.ffn_normed` SHALL
  not be written by the production code path (they remain as
  `DecodeScratch` fields for the legacy non-graph path's use,
  but are dead in the graph path)
<!-- test: unbacked -->
