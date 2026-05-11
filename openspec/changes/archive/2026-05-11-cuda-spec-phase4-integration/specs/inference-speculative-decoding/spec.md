## ADDED Requirements

### Requirement: Dispatch decision tree at the per-token call site

The `larql_inference::layer_graph::generate::generate` function SHALL
gate the speculative path at the per-token decode call site as
follows:

1. If `larql_inference::speculative::enabled()` returns `false` OR
   the optional drafter argument is `None`, the function SHALL fall
   through to the existing `backend.decode_token` path bit-exactly
   (no overhead, no diff).
2. Otherwise, the function SHALL call
   `larql_inference::speculative::maybe_speculative_step` with a
   closure that runs the target model on the constructed draft tree.
3. On `Some(tokens)` from `maybe_speculative_step`, the function
   SHALL emit each token in order, advance the KV cache by
   `tokens.len()` positions, and call `drafter.accept(&tokens)` to
   keep the drafter's history in sync.
4. On `None` from `maybe_speculative_step`, the function SHALL fall
   through to the existing `backend.decode_token` path for that step.

#### Scenario: env unset with drafter passed produces bit-exact legacy output

- **WHEN** `LARQL_SPECULATIVE_DECODE` is unset and `generate(..., Some(&mut drafter))` is called
- **THEN** the emitted token sequence SHALL be identical to `generate(..., None)` for the same prompt
<!-- test: unbacked -->

#### Scenario: env on with drafter advances cache and drafter history together

- **WHEN** speculation accepts a span of length N
- **THEN** target KV cache length SHALL increase by N AND drafter history length SHALL increase by N (in that order)
<!-- test: unbacked -->

### Requirement: Naive sequential target_forward implementation

The naive `target_forward(tree)` implementation SHALL emit one
vocab-sized probability vector per tree node by re-running the
target model's full forward pass on the ancestor sequence of each
node. The re-run SHALL use `predict_full_vocab_probs` rather than
`predict_q4k`'s top-k truncated path.

The naive implementation SHALL be the **parity oracle** for phase 4c's
batched implementation: phase 4c MUST produce the same `AcceptedSpan`
on the same `(tree, history, RNG seed)` inputs.

#### Scenario: predict_full_vocab_probs sums to 1.0 within fp32 tolerance

- **WHEN** `predict_full_vocab_probs(weights, tokenizer, tokens, index)` is called on a real Gemma 3 4B vindex
- **THEN** the returned vector SHALL have length equal to `weights.arch.vocab_size()` AND the elements SHALL sum to 1.0 within 1e-5 absolute
<!-- test: unbacked -->

#### Scenario: target_forward_naive at deepest node matches non-speculative argmax

- **WHEN** the drafter proposes a linear (depth=N branches=1) chain and the target's standard decode is run on the same prompt+chain
- **THEN** `target_forward_naive(tree)[deepest].argmax()` SHALL equal the non-speculative next-token argmax at the same position
<!-- test: unbacked -->

### Requirement: KV cache rollback on rejected speculative span

The integration code SHALL truncate the target's KV cache to
position `cache_len_pre_speculation + r` whenever the batched
`target_forward` is in use (phase 4c) and verification rejects at
tree node `r`. The rollback SHALL preserve the canonical
(non-speculative) cache state up to position
`cache_len_pre_speculation + r`.

For the naive sequential `target_forward` (phase 4b), no rollback is
required because each forward pass runs from a clean ancestor context
without speculatively writing to the canonical cache.

#### Scenario: batched rejection at depth 1 truncates cache cleanly

- **WHEN** a depth-3 speculative span is verified, the first draft is rejected, and target's cache was speculatively advanced to cache_len + 3
- **THEN** after `target.truncate_kv_cache(cache_len + 0)` the cache SHALL be at position `cache_len + 0` AND the next non-speculative `decode_token` call SHALL produce the same hidden state as if no speculation had occurred
<!-- test: unbacked -->

### Requirement: Phase-4 default-flip is gated by acceptance rate and ms/tok

The default value of the `LARQL_SPECULATIVE_DECODE` env var SHALL
remain `unset = off` until BOTH conditions are met on a fixed
256-prompt eval set with Gemma 3 4B Q4_K_M on RTX 4090:

1. Mean acceptance rate `α ≥ 0.6`
2. Mean decode latency `ms/tok ≤ 5.5`

If either condition fails, the default SHALL stay `off` and the
result SHALL be documented in
`openspec/changes/cuda-spec-phase4-integration/RETROSPECTIVE.md`.

#### Scenario: bench reports gate metrics and exit code reflects pass/fail

- **WHEN** `bench/decode_speculative.rs --gate` is run
- **THEN** the bench SHALL emit a JSON summary line containing `alpha`, `ms_per_token`, and `gate_passed: bool` AND exit 0 on pass, 1 on fail
<!-- test: unbacked -->
