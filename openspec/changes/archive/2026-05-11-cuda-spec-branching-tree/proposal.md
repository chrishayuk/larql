## Why

After the GPU softmax landed (PR #37), the spec decode path on
Gemma 3 4B Q4_K_M / RTX 4090 sits at **18.15 ms/tok**, vs plain
decode at **7.53 ms/tok** — spec is now **2.41× slower than plain
decode**. The D.3 perf-flip gate is **≤1.6× plain = ≤11.7 ms/tok**.

The remaining per-iter cost at depth=2 linear-chain α=0.855:

| Stage | Cost | % of iter |
|---|---|---|
| Forward (seq_len=2) | 12.5 ms | 53% |
| lm_head fused | 3.4 ms | 15% |
| verify_tree | 0.6 ms | 3% |
| Bonus decode_token | 5.6 ms | 24% |
| **Iter total** | **22 ms** | (3.7 emits/iter) |

The forward is already at near-theoretical for cuBLAS GEMM at M=2,
and the bonus decode is at the same speed as plain decode. **The
only remaining lever that doesn't require precision/sparsity
changes is to amortise the forward cost over more emits per iter
via branching**.

Branching trees (e.g. `depth=2 branches=2` → 7-node tree) verify
multiple alternative chains in parallel. The verify_tree algorithm
already supports arbitrary trees; the missing pieces are:

1. **A drafter that proposes multiple chains** (PLD finds multiple
   n-gram matches and emits each as a branch).
2. **A spec scratch + graph capture path that handles trees**
   (current `decode_tokens_speculative_seq_device` only accepts
   linear chains).
3. **A tree-aware attention kernel** — non-root branch positions
   must mask out sibling branches. The existing
   `fused_prefill_attn` is purely causal (each position attends to
   all earlier positions), which is wrong for non-trivial trees.

Estimated win at α≈0.85 per draft, branches=2, depth=2: emits per
iter rise from R+2 ≈ 3.7 to ~5.5-6.5 (probability of accepting at
least one chain ≥ 0.93). Iter cost rises modestly (~25-30% more
forward work for the 7-node tree vs 2-node chain), so wall-clock
improvement ~25-35% on prompts where PLD finds branches.

## What this change ships

A new spec change that lands in three sub-PRs:

1. **PLD-tree drafter** — extend `PromptLookupDrafter` to return
   multiple branches when multiple n-gram matches exist in the
   lookback window. Backwards-compatible (linear chain when only
   one match or `branches=1`).

2. **Tree-aware batched attention kernel** —
   `fused_prefill_attn_tree_mask` reads a per-node `ancestors[]`
   bitset and masks attention to only ancestor positions in the
   tree. Adds a new kernel variant; existing
   `fused_prefill_attn` stays for linear use.

3. **Spec scratch + dispatch branching path** — extend
   `decode_tokens_speculative_seq_device_scratch` and the v3
   dispatch to handle non-linear trees; capture one CUDA Graph per
   `(tree_shape, model_shape)` cache key.

## Capabilities

### Modified

- `inference-speculative-decoding` — adds 4 new scenarios covering
  PLD-tree drafter behaviour, tree-mask attention parity vs CPU
  reference, branching-tree dispatch correctness, and the perf
  delta gate vs the current linear path.

### Added

- `cuda-spec-tree-attention` — new capability for the tree-mask
  attention kernel and the per-node ancestor bitset contract.

## Impact

- Code change: ~400-600 LoC across `crates/larql-compute/src/cuda/attn.rs`
  (new kernel + wrapper), `crates/larql-compute/src/cuda/decode.rs`
  (tree variant of scratch path), `crates/larql-inference/src/speculative/
  {prompt_lookup,wiring}.rs` (drafter + dispatcher).
- Risk: medium-high. Tree attention is correctness-sensitive; subtle
  bugs in the ancestor mask would silently corrupt verify results.
  Land behind `LARQL_CUDA_SPEC_TREE=1` opt-in first.
- Test plan:
  - CPU reference `verify_tree` already supports trees — extend
    parity tests to compare GPU tree-mask attention vs CPU forward.
  - Add `test_speculative_branching_parity` that compares branching
    spec output to linear-chain spec output on 256 prompts (should
    differ in emit count but converge on completion).
  - Bench gate: `≤11.7 ms/tok` on the translation-echo prompt at
    depth=2 branches=2.

## Estimated effort

3-4 days of focused work:
- Day 1: PLD-tree drafter + linear-chain parity (small change).
- Day 2-3: Tree-mask attention kernel + parity vs CPU reference.
- Day 3-4: Spec scratch + dispatch tree path, CUDA Graph capture,
  bench validation.

## Carry-over from `cuda-spec-phase4b-complete`

The wins landed in this session that this change builds on:
- PR #34 PLD drafter (linear).
- PR #35 batched lm_head.
- PR #36 CUDA Graphs (linear-chain scratch + replay).
- PR #37 GPU softmax (fused GEMM+softmax).

Cumulative perf: 31.13 → 18.15 ms/tok (-42%) before branching.
Branching target: 18.15 → ~12 ms/tok (-34% additional, hitting
the D.3 perf-flip gate at depth=2 branches=2).
