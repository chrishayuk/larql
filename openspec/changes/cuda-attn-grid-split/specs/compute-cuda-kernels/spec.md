## ADDED Requirements

### Requirement: fused_decode_attention_f32 SHALL split each q_head's output across `d_split` blocks

The `fused_decode_attention_f32` NVRTC kernel SHALL accept an
`int d_split` argument and run with grid
`(num_q_heads, d_split, 1)`. Each block SHALL compute the
`[d_start, d_end)` slice of `out[qh, :]` where
`d_per_chunk = head_dim / d_split`,
`d_start = blockIdx.y * d_per_chunk`,
`d_end   = d_start + d_per_chunk`. K/V cache writes SHALL be
gated to `blockIdx.y == 0` so that multiple `d` chunks for the
same `(qh, kvh)` pair do not double-write.

#### Scenario: bench shows non-zero gain over the legacy single-block path

- **WHEN** `LARQL_CUDA_AVAILABLE=1 ./target/release/larql bench
  output/gemma-3-4b-it-vindex --backends cuda --tokens 50 --warmup 5`
  is run twice — once with the default heuristic (`d_split = 4`
  for Gemma 3 4B's 8 q_heads) and once with
  `LARQL_CUDA_ATTN_DSPLIT=1`
- **THEN** the heuristic run SHALL show `decode ms/token`
  ≤ 95% of the `=1` run, averaged over 5 trials each
<!-- test: unbacked -->

### Requirement: d_split=1 SHALL preserve bit-equivalent output

`LARQL_CUDA_ATTN_DSPLIT=1` SHALL produce bit-equivalent output
to the heuristic-chosen `d_split` value (≤ 1e-3 max-element).
This is the parity gate for the back-out contract.

#### Scenario: parity test passes

- **WHEN** `decode_token_graph_matches_per_call_over_5_steps`
  runs with the default heuristic and again with
  `LARQL_CUDA_ATTN_DSPLIT=1`
- **THEN** per-step max-element difference SHALL be ≤ 1e-3
<!-- test: larql_compute::tests::test_cuda_decode::decode_token_graph_matches_per_call_over_5_steps -->
