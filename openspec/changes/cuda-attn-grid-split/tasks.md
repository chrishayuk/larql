# cuda-attn-grid-split — tasks

## 1. Kernel signature

- [x] 1.1 Add `int d_split` arg to `fused_decode_attention_f32`.
- [x] 1.2 Read `dchunk = blockIdx.y` and bail when
      `dchunk >= d_split`.
- [x] 1.3 Compute `d_per_chunk = head_dim / d_split`,
      `d_start = dchunk * d_per_chunk`,
      `d_end = d_start + d_per_chunk`.

## 2. K/V cache write gate

- [x] 2.1 Restrict the K/V cache append to `dchunk == 0` (in
      addition to the existing `qh % group == 0` guard) so
      multiple chunks for the same `(qh, kvh)` don't
      double-write.

## 3. Output loop

- [x] 3.1 Change the output loop bound from
      `for (d = tid; d < head_dim; d += bdim)` to
      `for (d = tid + d_start; d < d_end; d += bdim)` so each
      block writes only its slice.

## 4. Rust wrappers

- [x] 4.1 `choose_attn_d_split(num_q_heads, head_dim)` helper —
      heuristic + `LARQL_CUDA_ATTN_DSPLIT` env-var override +
      divisibility fallback.
- [x] 4.2 Update each of the four wrappers
      (`fused_decode_attention`, `_device`, `_device_kv`,
      `_device_kv_into`) to: compute `d_split`, set
      `grid_dim = (num_q_heads, d_split, 1)`, pass `d_split` as
      the new kernel arg.

## 5. Tests + bench

- [x] 5.1 Existing parity tests cover the change:
      `decode_token_phase1_matches_host_fallback` and
      `decode_token_graph_matches_per_call_over_5_steps` both
      pass with the heuristic-chosen `d_split = 4` (default).
- [x] 5.2 Bench gate: 5-run average, 50 tokens / 5 warmup,
      Gemma 3 4B Q4_K, RTX 4090.
      `LARQL_CUDA_ATTN_DSPLIT=1`: 8.38 ms/tok.
      `LARQL_CUDA_ATTN_DSPLIT=4` (default): 8.31 ms/tok.

## 6. Documentation + archive

- [x] 6.1 `LARQL_CUDA_ATTN_DSPLIT=N` env var documented in
      `proposal.md`.
- [ ] 6.2 Archive when reviewed.
