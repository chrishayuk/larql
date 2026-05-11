## ADDED Requirements

### Requirement: Branching-tree spec decode amortises forward cost across multiple chains

The speculative decode path SHALL support tree-shaped drafts where
multiple candidate chains are verified in parallel against a single
target batched forward. Linear chains (the existing default) are the
degenerate case (`branches=1`).

Tree configuration:

1. `SpecConfig::branches` SHALL control the maximum number of
   parallel chains. Default `branches=1` preserves the existing
   linear-chain behaviour bit-exactly.
2. `LARQL_SPEC_BRANCHES=N` env var SHALL override the default for
   the duration of a `generate()` call.
3. `SpecConfig::tree_nodes()` SHALL continue to cap at 64 nodes per
   tree.

Drafter contract:

1. The `Drafter` trait SHALL gain a `propose_tree(h_target, depth,
   branches) -> DraftTree` method.
2. The default `propose_tree` SHALL fall back to `propose` to build
   a linear chain — preserves backwards compatibility for existing
   drafters.
3. `PromptLookupDrafter::propose_tree` SHALL find up to `branches`
   distinct n-gram matches in the lookback window (ranked
   rightmost-first) and build a `DraftTree` from their continuations,
   merging branches that share a common prefix.

Dispatch contract:

1. When `cfg.branches > 1` AND `propose_tree` returns a non-linear
   tree, the v3 dispatch SHALL route through a tree-aware spec
   scratch path.
2. When `propose_tree` returns a linear chain (branches=1 effective
   shape), the existing linear-chain spec scratch path SHALL run
   bit-exactly — branching is purely additive.
3. The tree spec scratch path SHALL fall through to the existing
   per-node `target_forward_via_speculative_decode_per_node` on any
   tree-kernel error (preserves conservative fallback).

#### Scenario: branches=1 preserves linear bit-exactness

- **WHEN** `LARQL_SPEC_BRANCHES=1` (or unset) is passed
- **THEN** the dispatcher SHALL pick the existing linear-chain spec
  scratch path AND the emitted token IDs SHALL be bit-identical to
  the pre-branching baseline on the same prompt + RNG seed
<!-- test: unbacked -->

#### Scenario: branches>1 yields more emits per spec iter on repetitive prompts

- **WHEN** PLD-tree at `branches=2 depth=2` is run on the
  translation-echo prompt
- **THEN** the bench output SHALL show ≥ 1.5× the average
  `emit/iter` count compared to `branches=1 depth=2` on the same
  prompt with the same RNG seed
<!-- test: unbacked -->

#### Scenario: tree-mask attention parity vs CPU reference

- **WHEN** the GPU tree-mask attention is run on 16 random tree
  shapes (depth 2-4, branches 2-4)
- **THEN** the per-node hidden state SHALL match the CPU
  `target_forward_with_hidden` reference with cosine ≥ 0.99
<!-- test: unbacked -->

#### Scenario: depth=2 branches=2 hits the D.3 perf-flip gate

- **WHEN** `larql bench output/gemma-3-4b-it-vindex --backends cuda
  --tokens 64` is run with `LARQL_SPEC_DEPTH=2 LARQL_SPEC_BRANCHES=2
  LARQL_DRAFTER=prompt_lookup LARQL_SPECULATIVE_DECODE=1` on the
  translation-echo prompt
- **THEN** the wall-clock SHALL be ≤ 11.7 ms/tok (= 1.6× plain
  decode floor of 7.53 ms/tok on RTX 4090)
<!-- test: unbacked -->

### Requirement: Multi-path tree verification picks the longest accepting chain

`verify_tree` SHALL implement multi-path verification when the
input tree has more than one root-to-leaf path. For each tree
node `n`, the verifier SHALL consume exactly one RNG draw
`u_n ∈ [0, 1)` (in BFS / tree-index order) and SHALL accept `n`
iff `u_n < p_target[n.id] / p_draft[n.id]` (Leviathan ratio). The
emitted span SHALL come from the root-to-leaf path with the
longest accepted-ancestor-prefix; ties SHALL be broken by lowest
leaf index (preserving the rightmost-match ordering produced by
PLD-tree).

Linear (single-path) trees SHALL reduce bit-exactly to
`verify_and_accept` — same RNG draws, same accept rule, same
residual sampling on rejection. This preserves parity with the
pre-branching baseline so existing single-path drafters keep
their emit distributions.

#### Scenario: branching-tree verifier rescues mid-chain rejection

- **WHEN** a 2-branch tree's first chain (e.g. PLD-tree's rightmost
  match) rejects mid-chain AND the second chain accepts fully
- **THEN** the emitted span SHALL come from the second chain (= more
  emits than the linear baseline on the same RNG seed)
<!-- test: speculative::verify::tests::verify_tree_branching_more_emits_than_linear_on_same_seed -->

#### Scenario: linear-chain verification preserves bit-parity

- **WHEN** `verify_tree` is invoked on a tree with exactly one
  root-to-leaf path (i.e. linear chain)
- **THEN** the returned `AcceptedSpan` SHALL be bit-identical to
  feeding the same drafts + per-position target distributions into
  `verify_and_accept` with the same seeded RNG
<!-- test: speculative::verify::tests::verify_tree_linear_matches_verify_and_accept -->
