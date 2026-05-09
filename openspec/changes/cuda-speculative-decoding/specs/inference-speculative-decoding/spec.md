## ADDED Requirements

### Requirement: SpeculativeDecoder produces tokens identical to non-speculative decode

The `larql_inference::speculative::SpeculativeDecoder` SHALL emit
the same token IDs as the non-speculative greedy or
temperature-sampled decode path, given the same prompt, sampling
temperature, top-k/top-p settings, and RNG seed. Equivalence is
asserted on a fixed 256-prompt eval set.

#### Scenario: token-ID parity on 256-prompt eval at temperature 0
- **WHEN** the same 256 prompts are decoded for ≤ 64 tokens with `temperature=0`, once with `LARQL_SPECULATIVE_DECODE=0` and once with `=1` (depth=2, branches=2)
- **THEN** the emitted token sequences SHALL be identical for every prompt
<!-- test: larql_inference::test_speculative_parity::greedy_parity_256_prompts -->

#### Scenario: token-ID parity on 256-prompt eval at temperature 0.7 with seeded RNG
- **WHEN** the same 256 prompts are decoded for ≤ 64 tokens with `temperature=0.7` and a fixed RNG seed, once non-speculative and once speculative
- **THEN** the emitted token sequences SHALL be identical for every prompt
<!-- test: larql_inference::test_speculative_parity::seeded_temp_parity_256_prompts -->

### Requirement: SpeculativeDecoder is gated by LARQL_SPECULATIVE_DECODE

The speculative path SHALL only activate when the environment variable
`LARQL_SPECULATIVE_DECODE` is set to `1`. Any other value (unset,
empty, `0`, anything else) SHALL fall through to the existing
non-speculative `decode_token` path bit-exactly.

#### Scenario: unset env var falls through to existing path
- **WHEN** `LARQL_SPECULATIVE_DECODE` is unset and a single token is decoded
- **THEN** the dispatched path SHALL be the existing non-speculative `decode_token`, and the output SHALL be bit-exactly equal to the pre-speculative behaviour on a fixed input
<!-- test: larql_inference::speculative::tests::unset_env_uses_legacy_path -->

#### Scenario: env=0 also falls through
- **WHEN** `LARQL_SPECULATIVE_DECODE=0` is set
- **THEN** the dispatched path SHALL be the legacy non-speculative path
<!-- test: larql_inference::speculative::tests::env_zero_uses_legacy_path -->

### Requirement: Verification kernel implements exact rejection sampling

`cuda::sampling::verify_tree` SHALL implement the rejection-sampling
acceptance rule from Leviathan et al. 2022:

- For each draft token `d_k`, accept with probability
  `min(1, p_target(d_k) / p_draft(d_k))`.
- On first rejection at position `r`, sample one corrected token
  from the residual distribution `max(0, p_target - p_draft) / Z`,
  where `Z` is the normaliser of the residual.
- On all-accept, additionally sample one bonus token directly from
  `p_target` at the deepest accepted position.

The kernel output SHALL match a CPU reference implementation
bit-exactly on token-ID equality (probability values may differ by
≤ 1e-6 due to f32 softmax ordering).

#### Scenario: verify_tree matches CPU reference on 64 fixed RNG seeds
- **WHEN** synthetic logits + draft probs are generated for tree depth=2, branches=2 across 64 fixed RNG seeds
- **THEN** the GPU `verify_tree` accepted-span SHALL equal the CPU reference accepted-span on every seed
<!-- test: larql_compute::test_cuda_verify_tree::matches_cpu_reference_64_seeds -->

#### Scenario: residual sampling normalises correctly
- **WHEN** a synthetic case is constructed where `p_target = p_draft` exactly so the residual is the zero vector
- **THEN** the kernel SHALL fall back to sampling from `p_target` directly (not divide-by-zero)
<!-- test: larql_compute::test_cuda_verify_tree::residual_zero_falls_back_to_target -->

### Requirement: Speculative depth is bounded by sliding window

The dispatched speculative depth SHALL satisfy
`depth ≤ min(configured_depth, swa_window - cache_len)` whenever the
target model uses sliding-window attention (e.g. Gemma 3). This
prevents a draft token from being accepted under a different
attention mask than the target sees on verification.

#### Scenario: depth clamps near sliding-window boundary
- **WHEN** the cache is at position `swa_window - 1` and a depth-4 speculative window is requested
- **THEN** the actual speculative depth dispatched SHALL be 1 (the remaining slack), and the rest of the configured depth SHALL be silently dropped
<!-- test: larql_inference::test_speculative_swa_clamp::clamps_to_window_remainder -->

### Requirement: Rotorquant supports rolling back speculative KV writes

`larql_rotorquant::compress` SHALL gain a window-lag mode where
compression of slot `s` is delayed by `lag` positions. While in the
lag window, the K/V at slot `s` is kept in f16 (or whatever the
non-compressed representation is) and a rollback to position
`s - 1` is an O(1) pointer update with no tensor recomputation.

#### Scenario: compress_with_window_lag holds slot uncompressed for the lag window
- **WHEN** lag=8 and 4 successive K/V writes are made
- **THEN** all 4 slots SHALL remain in their uncompressed representation, observable via the rotorquant inspect API
<!-- test: larql_rotorquant::test_window_lag::holds_within_lag -->

#### Scenario: rollback restores prior slot count
- **WHEN** 5 K/V writes are made at lag=8 and then `rollback_to(slot - 3)` is called
- **THEN** the rotorquant cache SHALL report `len = slot - 3 + 1` and the next read SHALL return the original f16 K/V at the rolled-back position bit-exactly
<!-- test: larql_rotorquant::test_window_lag::rollback_round_trip -->

### Requirement: Performance gates speculative-rollout default

The project SHALL flip the default of `LARQL_SPECULATIVE_DECODE`
from `0` to `1` only when phase 4 acceptance benchmark reports
both:

- acceptance rate `α ≥ 0.6` on the project-wide eval set
- ms/token ≤ 5.5 on Gemma 3 4B Q4_K_M, RTX 4090

If either gate fails, the default SHALL stay `0` and the result
SHALL be documented in
`openspec/changes/cuda-speculative-decoding/RETROSPECTIVE.md`.

#### Scenario: bench reports gate metrics in machine-readable form
- **WHEN** `bench/decode_speculative.rs --gate` is run
- **THEN** the bench SHALL emit a JSON line `{"alpha": <f32>, "ms_per_token": <f32>, "gate_passed": <bool>}` and exit 0 on pass / 1 on fail
<!-- test: larql_compute::test_decode_speculative_bench::gate_metric_format -->
