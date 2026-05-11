## ADDED Requirements

### Requirement: Phase 4b naive end-to-end speculative decode is the parity oracle

The naive speculative path SHALL be the **parity oracle** for phase
4c's batched implementation. Phase 4c's `target_forward_batched`
SHALL produce the same `AcceptedSpan` (token-ID equality) as phase
4b's `target_forward_naive` on the same `(tree, history, RNG seed)`
inputs across 64 fixed seeds.

The naive path's correctness contract:

1. The dispatch at `gpu.rs:735` SHALL fall through bit-exactly to the
   existing non-speculative `decode_token` path when
   `LARQL_SPECULATIVE_DECODE` is unset OR when no drafter / target
   executor is installed via thread-local APIs.
2. The dispatch SHALL succeed and emit at least one token when env=1
   AND a drafter AND a `SpeculativeTargetExecutor` are installed.
3. The first emitted token SHALL match the baseline non-speculative
   path on the same prompt (bit-exact at position 0).
4. The drafter's internal history SHALL stay in sync with the loop's
   canonical history via re-seeding inside the dispatcher.

#### Scenario: env-OFF preserves baseline bit-exactly

- **WHEN** `larql bench` is run without `--draft-model` (or with `LARQL_SPECULATIVE_DECODE` unset)
- **THEN** the per-stage timings SHALL match the pre-speculative baseline within run-to-run noise (≤ 5% deviation in ms/tok)
<!-- test: unbacked -->

#### Scenario: env-ON dispatch emits tokens through the speculative path

- **WHEN** `larql bench --draft-model <vindex>` is run with `LARQL_SPECULATIVE_DECODE=1` against the same vindex as the bench's main model
- **THEN** the bench output SHALL include the line "Speculative drafter: loaded ... (active)" AND emit ≥ 1 token AND complete without panic
<!-- test: unbacked -->

#### Scenario: first-token parity between speculative and baseline

- **WHEN** `cargo test test_speculative_parity` is run with `LARQL_SPECULATIVE_PARITY_VINDEX` set to a real Gemma 3 4B Q4_K_M vindex
- **THEN** the first emitted token from the speculative path SHALL equal the first emitted token from the baseline path
<!-- test: larql_inference::test_speculative_parity::token_id_parity_speculative_vs_baseline_short_prompt -->
