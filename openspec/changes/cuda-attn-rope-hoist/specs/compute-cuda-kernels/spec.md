## ADDED Requirements

### Requirement: Q-vector RoPE rotation SHALL run once per attention call

The fused decode-attention kernel SHALL compute the Q-vector
RoPE rotation once per attention call (per `(pos, head)`
pair), not once per `(pos, head, j)` iteration of the score
loop. The rotated Q values SHALL live in a shared-memory
buffer of size `head_dim` floats; the score loop SHALL read
them via a single load instead of recomputing the rotation.

#### Scenario: pre-rotation produces parity output vs the host-fallback attention

- **WHEN** the synthetic Q4_K pipeline runs three decode steps
  with the modified attention kernel and again with
  `LARQL_CUDA_DECODE_HOST_FALLBACK=1` (CPU-side attention)
- **THEN** the per-step output vectors SHALL agree to
  max-element absolute difference ≤ 1e-3, the same bound the
  pre-change kernel cleared
<!-- test: larql_compute::tests::test_cuda_decode::decode_token_phase1_matches_host_fallback -->

### Requirement: attn_call profile bucket SHALL drop materially after the hoist

`attn_call` profile bucket SHALL drop to ≤ 4 ms (down from the post-`cuda-q4k-mmvq-int8` 6.35 ms) on the dev-box bench after this change ships. **Actual**: 3.68 ms — cleared. A
miss by > 25% (i.e., `attn_call > 5 ms`) SHALL trigger a
profile-and-document write-up identifying the residual cost
before the change is archived.

#### Scenario: profile bucket cleared at acceptance OR documented on miss

- **WHEN** `LARQL_CUDA_AVAILABLE=1 LARQL_CUDA_DECODE_PROFILE=1
  ./target/release/larql bench output/gemma-3-4b-it-vindex
  --backends cuda --tokens 20 --warmup 3 --verbose` is run on
  the dev box after this change lands
- **THEN** EITHER `attn_call` SHALL be ≤ 4 ms (acceptance hit
  → archive), OR the change's `proposal.md` SHALL contain a
  profile-and-document write-up explaining why the hoist did
  not pay
<!-- test: unbacked -->
