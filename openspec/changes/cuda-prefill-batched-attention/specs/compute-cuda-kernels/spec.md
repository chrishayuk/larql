## ADDED Requirements

### Requirement: prefill attention SHALL run as a single batched launch

`prefill_q4_seq_device` SHALL compute attention for all `seq_len` query positions in a single batched kernel launch, not as a per-position loop. The launch SHALL use `grid_dim = (num_q_heads, seq_len, 1)` with each block computing one `(head, seq_pos)` attention output. K/V cache writes for all positions SHALL happen in a separate preceding kernel (`kv_cache_write_seq_f32`) to avoid intra-launch race conditions.

#### Scenario: batched-attn output matches per-position loop within 1e-3

- **WHEN** a synthetic Q4_K prefill on a 6-token prompt runs
  through `fused_prefill_attention_seq_device` and again
  through the per-position
  `fused_decode_attention_device_kv` loop with the same
  inputs
- **THEN** the per-position output buffers SHALL agree to
  max-element absolute difference ≤ 1e-3
<!-- test: unbacked -->

### Requirement: attn profile bucket SHALL drop to ≤ 5 ms after batching

After this change ships, the prefill profile (RTX 4090, Gemma 3 4B Q4_K, 6-token prompt, with `LARQL_CUDA_PREFILL_PROFILE=1`) SHALL report `attn ≤ 5 ms` (down from 22.15 ms). **Actual**: 0.86 ms — 26× drop, well past target. A miss of > 60% (i.e., `attn > 8 ms`) SHALL trigger a profile-and-document write-up identifying why the per-position parallelism didn't pay off.

#### Scenario: profile bucket cleared at acceptance OR documented on miss

- **WHEN** `LARQL_CUDA_AVAILABLE=1 LARQL_CUDA_PREFILL_PROFILE=1
  ./target/release/larql bench output/gemma-3-4b-it-vindex
  --backends cuda --tokens 20 --warmup 3 --verbose` is run
  after this change lands
- **THEN** EITHER the prefill `attn` bucket SHALL be ≤ 5 ms
  AND total `prefill ms` ≤ 100 (acceptance hit), OR the
  proposal SHALL contain a profile-and-document write-up
<!-- test: unbacked -->
