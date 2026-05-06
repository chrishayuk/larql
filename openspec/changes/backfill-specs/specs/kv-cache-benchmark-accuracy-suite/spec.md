## ADDED Requirements

### Requirement: Top-1 token-match accuracy and prompt fixtures

The `kv_cache_benchmark::accuracy` module SHALL provide two reusable
prompt fixtures — `factual_prompts()` (≥ 20 entries) and
`diverse_prompts()` (≥ 10 entries) — each carrying both a prompt
string and its expected continuation, so Top-1 token-match accuracy
can be evaluated reproducibly. Each prompt SHALL be non-empty in
both fields. `AccuracyResult::token_match` SHALL record the strategy,
prompt category, prompt text, and a `top1_match` boolean and SHALL
leave `kl_divergence` / `js_divergence` as `NaN` (token-match runs
do not score full distributions). `AccuracyResult::needle` SHALL
populate `needle_found` and `needle_exact_match`. The
`format_accuracy_summary` formatter SHALL render per-strategy match
percentages so a regression is human-readable. When the `real-model`
feature is enabled, the live accuracy sweep SHALL print a
match-rate table over `factual_prompts()` and SHALL hard-fail if
Markov RS top-1 ever diverges from the Standard KV baseline.

#### Scenario: factual_prompts and diverse_prompts ship with valid entries
- **WHEN** `factual_prompts()` and `diverse_prompts()` are read
- **THEN** the first SHALL contain at least 20 entries and the second at least 10, with non-empty prompt and answer fields
<!-- test: kv_cache_benchmark::test_accuracy::test_accuracy_factual_prompts_exist -->
<!-- test: kv_cache_benchmark::test_accuracy::test_accuracy_diverse_prompts_exist -->

#### Scenario: AccuracyResult helpers cover token-match and needle paths
- **WHEN** `AccuracyResult::token_match` and `AccuracyResult::needle` are constructed
- **THEN** the token-match result SHALL leave `kl_divergence` / `js_divergence` `NaN`, set `top1_match` to the supplied boolean, and the needle helper SHALL set `needle_found` and `needle_exact_match` to the supplied booleans
<!-- test: kv_cache_benchmark::test_accuracy::test_accuracy_result_token_match -->
<!-- test: kv_cache_benchmark::test_accuracy::test_accuracy_result_needle -->

#### Scenario: format_accuracy_summary renders per-strategy match rates
- **WHEN** `format_accuracy_summary` is called over a mixed-strategy result set
- **THEN** the output SHALL contain `Standard KV`, `Markov RS`, `TurboQuant`, and `100.0%`
<!-- test: kv_cache_benchmark::test_accuracy::test_accuracy_summary_format -->

#### Scenario: Adversarial helpers expose entity-confusion and polysemy fixtures
- **WHEN** the entity-confusion and polysemy prompt fixtures are inspected
- **THEN** every entity-confusion prompt SHALL contain `"capital of"` with a non-empty expected token, and the polysemy fixtures SHALL exercise context-dependent meaning (`"deposit"` and `"river"`)
<!-- test: kv_cache_benchmark::test_accuracy::test_entity_confusion_prompts -->
<!-- test: kv_cache_benchmark::test_accuracy::test_polysemy_prompts -->

#### Scenario: Live top-1 sweep enforces Markov bit-perfect parity
- **WHEN** `run_all_strategies` is run over `factual_prompts()` against a real Gemma 3-4B
- **THEN** Markov RS top-1 SHALL equal the Standard KV baseline for every prompt
<!-- test: kv_cache_benchmark::test_real_model::test_accuracy_top1_factual_20 -->

#### Scenario: Markov RS bit-perfect spot check on canonical factual prompts
- **WHEN** the canonical `["The capital of France is", "Mozart was born in", "Water freezes at"]` prompts are evaluated
- **THEN** Markov RS top-1 SHALL match the baseline and `hidden_cosine` SHALL exceed `0.9999` on every prompt
<!-- test: kv_cache_benchmark::test_real_model::test_accuracy_markov_rs_bitperfect -->

### Requirement: KL/JS divergence and softmax helpers

The `accuracy` module SHALL expose `kl_divergence`, `js_divergence`,
and `softmax` helpers plus rank metrics (`top_k_overlap`,
`first_divergence`, `token_match_rate`, `reciprocal_rank`) so that
distribution-level comparisons across strategies use a shared
implementation. KL on identical distributions MUST be `~0` (within
`1e-10`). JS divergence MUST be symmetric within `1e-10` and
bounded by `ln 2`. Softmax MUST sum to `1.0` within `1e-6` and
preserve argmax. Top-k overlap MUST report `1.0` for identical
ordered lists, `0.0` for disjoint, and the correct fraction for
partial overlap. `first_divergence` MUST be `None` for equal
sequences and the index of the first mismatch otherwise.
`reciprocal_rank` MUST return `1.0` for a top-1 hit and `1/r` for
a hit at rank `r`, and `0.0` when the target is missing.

#### Scenario: KL divergence collapses on identical distributions
- **WHEN** `kl_divergence(p, p)` is computed
- **THEN** the result SHALL be within `1e-10` of zero, and on different distributions SHALL be a finite value in `(0, 10)`
<!-- test: kv_cache_benchmark::test_accuracy::test_kl_divergence_identical -->
<!-- test: kv_cache_benchmark::test_accuracy::test_kl_divergence_different -->

#### Scenario: JS divergence is symmetric and bounded
- **WHEN** `js_divergence(p, q)` and `js_divergence(q, p)` are compared, and JS is evaluated on disjoint distributions
- **THEN** the two values SHALL agree within `1e-10` and the bounded result SHALL be ≤ `0.7`
<!-- test: kv_cache_benchmark::test_accuracy::test_js_divergence_symmetric -->
<!-- test: kv_cache_benchmark::test_accuracy::test_js_divergence_bounded -->

#### Scenario: Softmax sums to one and preserves argmax
- **WHEN** `softmax([2.0, 1.0, 0.5, -1.0, 3.0])` is computed
- **THEN** the sum SHALL be within `1e-6` of `1.0`, and on `[1.0, 5.0, 2.0, 0.5]` the argmax SHALL be index `1`
<!-- test: kv_cache_benchmark::test_accuracy::test_softmax_sums_to_one -->
<!-- test: kv_cache_benchmark::test_accuracy::test_softmax_argmax_preserved -->

#### Scenario: Top-k overlap, divergence, and rank helpers are correct
- **WHEN** `top_k_overlap`, `first_divergence`, `token_match_rate`, and `reciprocal_rank` are exercised on identical, disjoint, and partial-overlap inputs
- **THEN** identical lists SHALL score `1.0` overlap and `1.0` token-match-rate, disjoint lists SHALL score `0.0` overlap, and partial overlap of `2/5` SHALL score `0.4`; equal sequences SHALL produce `first_divergence=None`, otherwise the first mismatch index; `reciprocal_rank` SHALL return `1.0` at rank 1, `1/3` at rank 3, and `0.0` for a missing target
<!-- test: kv_cache_benchmark::test_accuracy::test_top_k_overlap_identical -->
<!-- test: kv_cache_benchmark::test_accuracy::test_top_k_overlap_disjoint -->
<!-- test: kv_cache_benchmark::test_accuracy::test_top_k_overlap_partial -->
<!-- test: kv_cache_benchmark::test_accuracy::test_first_divergence_identical -->
<!-- test: kv_cache_benchmark::test_accuracy::test_first_divergence_at_position -->
<!-- test: kv_cache_benchmark::test_accuracy::test_token_match_rate_perfect -->
<!-- test: kv_cache_benchmark::test_accuracy::test_token_match_rate_partial -->
<!-- test: kv_cache_benchmark::test_accuracy::test_reciprocal_rank_first -->
<!-- test: kv_cache_benchmark::test_accuracy::test_reciprocal_rank_third -->
<!-- test: kv_cache_benchmark::test_accuracy::test_reciprocal_rank_missing -->

### Requirement: Needle-in-a-haystack fixtures and accuracy suite

The accuracy module SHALL expose `generate_haystack(target_tokens,
needle_pos, needle)` that returns a context guaranteed to contain
the needle at approximately the requested position, plus
`build_retention_conversation(turns)` that returns a fact-rich
multi-turn fixture with at least three query turns. Under the
`real-model` feature the `accuracy_suite` module SHALL provide a
canonical 100-prompt set spanning at least the categories
`{arithmetic, code, completion, conversational, factual, geographic,
reasoning, scientific}` with ≥ 10 prompts per category, a 20-prompt
quick subset of that set, the `paris_test()` calibration prompt
(referencing France with `Paris` as the expected continuation), an
ordered scaling needle suite, a multi-needle fixture of exactly 5
fact/answer/query triples, and a deterministic haystack builder.
The needle and accuracy formatters MUST emit human-readable tables
flagging context lengths and pass/fail per strategy. Live needle
sweeps SHALL exercise context lengths up to 4 K tokens and report
needle hits in the predicted top-10 (the published demonstration
that Markov-style bounded windows can recover where Standard KV
fails).

#### Scenario: Haystacks contain the needle near the requested position
- **WHEN** `generate_haystack(500, 200, "SECRET-CODE-12345")` and `generate_haystack(32_000, 5_000, "...AURORA-7749")` are evaluated
- **THEN** both contexts SHALL contain the needle, the returned needle SHALL match the input, and the short haystack SHALL exceed `200` chars while the long one exceeds `10_000` chars
<!-- test: kv_cache_benchmark::test_accuracy::test_haystack_generation_short -->
<!-- test: kv_cache_benchmark::test_accuracy::test_haystack_generation_long -->
<!-- test: kv_cache_benchmark::test_accuracy::test_haystack_needle_position -->

#### Scenario: Multi-turn retention fixture has facts and queries
- **WHEN** `build_retention_conversation(15)` and `build_retention_conversation(25)` are read
- **THEN** the returned conversations SHALL have the requested length, at least 3 query turns with `expected_fact = Some`, and at least 3 fact-establishing turns
<!-- test: kv_cache_benchmark::test_accuracy::test_retention_conversation_structure -->
<!-- test: kv_cache_benchmark::test_accuracy::test_retention_conversation_25_turns -->

#### Scenario: diverse_100 covers every category with quick_20 as a subset
- **WHEN** `prompts::diverse_100()` and `prompts::quick_20()` are evaluated
- **THEN** `diverse_100()` SHALL contain exactly 100 entries spanning the canonical 8 categories with ≥ 10 entries each, and every entry of `quick_20()` SHALL appear by `text` in `diverse_100()`
<!-- test: kv_cache_benchmark::test_accuracy_suite::with_model::test_diverse_100_has_100_prompts -->
<!-- test: kv_cache_benchmark::test_accuracy_suite::with_model::test_diverse_100_all_categories -->
<!-- test: kv_cache_benchmark::test_accuracy_suite::with_model::test_diverse_100_balanced_categories -->
<!-- test: kv_cache_benchmark::test_accuracy_suite::with_model::test_quick_20_is_subset -->

#### Scenario: Paris calibration prompt references France with Paris
- **WHEN** `prompts::paris_test()` is read
- **THEN** the prompt text SHALL contain `France` and `expected_contains` SHALL equal `Paris`
<!-- test: kv_cache_benchmark::test_accuracy_suite::with_model::test_paris_test_prompt -->

#### Scenario: Needle suite scales context length monotonically
- **WHEN** `needle::needle_tests()` is read
- **THEN** the suite SHALL contain at least 5 entries with strictly increasing `context_tokens`
<!-- test: kv_cache_benchmark::test_accuracy_suite::with_model::test_needle_tests_scaling -->

#### Scenario: Suite haystack builder embeds the needle and scales the body
- **WHEN** `needle::build_haystack(target, "NEEDLE")` is called for `target` in `[512, 4096, 32768]`
- **THEN** every result SHALL contain the needle and SHALL be longer than `target × 2` chars
<!-- test: kv_cache_benchmark::test_accuracy_suite::with_model::test_build_haystack_contains_needle -->
<!-- test: kv_cache_benchmark::test_accuracy_suite::with_model::test_build_haystack_length_reasonable -->

#### Scenario: needle_found is case-insensitive and rejects misses
- **WHEN** `needle_found(text, "AURORA")` is evaluated
- **THEN** matches with mixed and lower case SHALL return `true` and unrelated text SHALL return `false`
<!-- test: kv_cache_benchmark::test_accuracy_suite::with_model::test_needle_found_detection -->

#### Scenario: Multi-needle suite supplies five fact/answer/query triples
- **WHEN** `needle::multi_needle_tests()` is read
- **THEN** the suite SHALL contain exactly 5 triples whose `query` strings end with `?` and whose facts and answers are non-empty
<!-- test: kv_cache_benchmark::test_accuracy_suite::with_model::test_multi_needle_tests -->

#### Scenario: Needle and accuracy formatters render PASS/FAIL tables
- **WHEN** `format_needle_results` and `runner::format_accuracy_table` render canonical inputs
- **THEN** the needle table SHALL contain `PASS`, `FAIL`, `512 tokens`, and `32768 tokens`, and the accuracy table SHALL contain `Standard KV`, `Markov RS`, and `100.0%`
<!-- test: kv_cache_benchmark::test_accuracy_suite::with_model::test_format_needle_results -->
<!-- test: kv_cache_benchmark::test_accuracy_suite::with_model::test_format_accuracy_table -->

#### Scenario: Live needle sweep records hits in the top-10
- **WHEN** the canonical 512-token needle prompt is run through `run_all_strategies`
- **THEN** every strategy SHALL emit a top-1 token and a top-5 set without panicking
<!-- test: kv_cache_benchmark::test_real_model::test_needle_short_512 -->

### Requirement: Vindex KV comparison and Apollo factual accuracy

The `vindex_compare` module SHALL run the same forward pass against
two `VectorIndex` instances and emit a `PromptReport` (logit
cosine, argmax match, top-K Jaccard, forward / reverse / symmetric
KL, ref/cand top token id) plus an `AggregateReport` (n_prompts,
labels, config, mean argmax agreement, mean top-K Jaccard, mean
logit cosine, mean / p95 / max symmetric KL) so that storage-format
A/B's (FP4 ↔ Q4K, etc.) report a single fairness number.
`real_model::runner::run_all_strategies` SHALL surface a
graph-walk top-1 metric and the `accuracy_suite::runner` SHALL
expose a `StrategyAccuracy` schema covering top-1 match rate,
generation token-match rate, generation first-divergence index, and
needle pass rate; both SHALL feed the canonical accuracy table.
Apollo's accuracy sweep SHALL print uncompressed vs compressed
top-1 predictions and context-length ratios across the canonical
query set, so the ~20 000× compression claim is testable
side-by-side with first-token factual correctness.

#### Scenario: Apollo factual accuracy comparison runs both forward paths
- **WHEN** `query_greedy` and `query_greedy_compressed` run for every prompt in the canonical sweep
- **THEN** both invocations SHALL succeed and the trace SHALL expose a finite `top1_logit` for each
<!-- test: kv_cache_benchmark::test_apollo_accuracy::test_apollo_accuracy_sweep -->

#### Scenario: Graph Walk factual accuracy is reported through run_all_strategies
- **WHEN** `run_all_strategies` is run on the default factual prompt set
- **THEN** the per-prompt `top1_match` flag for the Graph Walk strategy SHALL be reported as a count and a percentage
<!-- test: kv_cache_benchmark::test_real_model::test_graph_walk_factual_accuracy -->

#### Scenario: Comparative Markov bit-perfect ratio across the prompt fleet
- **WHEN** `test_370k_memory_ratios` and `test_all_strategies_memory_ordering` are run side by side
- **THEN** Markov RS / Graph Walk SHALL come in below the canonical Standard / TurboQuant ratios while keeping bit-perfect parity claims intact
<!-- test: kv_cache_benchmark::test_comparative::test_370k_memory_ratios -->
<!-- test: kv_cache_benchmark::test_comparative::test_all_strategies_memory_ordering -->

### Requirement: Test suite coverage

The capability's behavior SHALL be exercised by every test in the test modules listed below. New tests added to those modules SHALL be considered part of this capability's contract by default.

#### Scenario: Workspace tests bound to this capability
- **WHEN** the trace tool resolves the wildcard module references for this capability
- **THEN** every backing test SHALL resolve to a real `#[test]` function in the workspace
<!-- test: kv_cache_benchmark::test_accuracy::**::* -->
<!-- test: kv_cache_benchmark::test_accuracy_suite::**::* -->
