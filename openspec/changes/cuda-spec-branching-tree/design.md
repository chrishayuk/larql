## Context

This change builds on the spec scratch + graph-capture infrastructure
landed in `cuda-spec-phase4b-complete` (PRs #34-#37). After GPU
softmax (#37), the remaining cost breakdown at depth=2 linear is:

- Forward (12.5 ms): dominant, at near-theoretical for cuBLAS at M=2.
- Bonus decode (5.6 ms): same speed as plain decode.
- lm_head + softmax (3.4 ms): bandwidth-limited on 1.3 GB f16 read.

The forward is already nearly optimal per-token; **the only remaining
lever that doesn't require precision/sparsity changes is amortising
forward cost over more emits per iter via branching**.

## Goal

Move from linear depth=2 (max emit = 3 = R+1+picked_id ≈ 3.7 at
α=0.85) to branching depth=2 branches=2 (max emit = 5 = R+1+picked_id
where R is the longest-accepted-branch length, average ~4.5 at
α=0.85). Iter cost rises ~25-30% due to larger tree; net wall-clock
improvement ~25-35% expected.

## Decision: tree shape and ancestor representation

**Pick ancestor-bitset representation**. Each tree node `n` gets a
`u64` bitset where bit `k` is set iff position `k` is an ancestor of
`n` (inclusive of `n` itself). Constraints:

- Tree size capped at 64 nodes (existing `SpecConfig::tree_nodes()`
  cap). One `u64` per node, total = `8 * tree_len` bytes per call.
- For a linear chain of length D, node `k` has ancestors
  `0..=k` → bitset `(1 << (k+1)) - 1`. Bit-identical to a causal mask.
- For a 7-node tree (depth=2 branches=2: root + 2 children + 4
  grandchildren), 7 distinct bitsets.

Why not a row-major mask matrix? `seq_len * seq_len * 1 bit` blows up
shared memory at large trees; bitset is more compact and exactly
matches the per-position lookup the attention loop already does.

**Alternative considered: per-depth offset table.** Rejected because
it requires the kernel to compute branch-id arithmetic which doesn't
play well with arbitrary tree shapes (e.g. unbalanced trees from PLD
where one branch is shorter than another).

## Decision: K/V cache layout

Two strategies considered:

| | Strategy A (tree-index-direct) | Strategy B (depth+branch) |
|---|---|---|
| K/V position | `base_pos + tree_index` | `base_pos + depth_offset[d] + branch_id` |
| Mask filtering | Yes (via ancestor bitset) | No (positions are unique) |
| Kernel changes | Add bitset arg | Add offset table |
| Cache slots used | `tree_len` (e.g. 7 for branches=2 depth=2) | `tree_len` |
| Cache cleanup | Truncate after verify (only keep R+1 accepted) | Same |

**Pick Strategy A.** Simpler kernel (just adds a bitset mask check);
the existing `kv_cache_write_seq_f32` kernel writes K/V at
`base_pos + sp` linearly, which matches tree-index-direct. The mask
is applied in the attention kernel's per-`j` loop.

## Decision: PLD-tree branch selection

When PLD's lookup finds multiple matches, the standard linear PLD
returns the rightmost. For tree-PLD we want the rightmost `branches`
matches. Edge cases:

1. **Fewer than `branches` matches**: degrades to linear chain of the
   actual match count (still benefits from any speedup).
2. **Matches share a common prefix**: merge into a shared subtree.
   E.g. if matches at position 100 and 250 both start with token
   `42`, the resulting tree has one node `42` at depth 1, then two
   different children at depth 2.
3. **Identical full chains**: deduplicate to a single chain. Two
   matches with identical continuations produce only one chain.

The merge logic for case 2 produces tighter trees that amortise more
work per K/V slot — important because the kernel cost scales with
`tree_len`, not with the number of accepted tokens.

## Decision: dispatch fallback

When the tree path isn't applicable (e.g. PLD returns a linear
chain, or `branches=1`), the v3 dispatch SHALL fall through to the
existing linear-chain `decode_tokens_speculative_seq_device_scratch`
path bit-exactly. The tree path is purely **additive** — never
slower than linear when only linear-shaped trees come in.

## Risks

1. **Tree attention mask correctness**: silent corruption if mask is
   wrong. Mitigation: parity test vs CPU reference on 16 random
   trees; bit-exact reduction when single-chain.

2. **PLD-tree match quality**: second-best matches may be lower
   quality, dragging average α down. Empirical: PLD usually finds 1-2
   high-quality matches in repetitive contexts; below that, branches
   degrade. Mitigation: bench at branches=1/2/4 across prompt types.

3. **CUDA Graph re-capture on tree-shape change**: if the drafter
   sometimes produces 2 branches and sometimes 1, the cache misses
   each iter. Mitigation: cache key includes tree shape; reuse hits
   when shape repeats. PLD-tree should produce consistent shapes on
   stable workloads.

4. **K/V cache pressure**: tree_len=7 vs 2 means 3.5× cache slots
   used per iter. With `DEFAULT_CUDA_KV_CACHE_MAX_SEQ` typically
   16384, this is fine (room for ~2300 iters). For long-context
   workloads beyond 4K tokens, monitor for OOM.

## Open questions

- **Does PLD-tree benefit RAG workloads more than translation-echo?**
  RAG has long repetitive passages with multiple plausible
  continuations. Empirical bench needed in T4.2.
- **Should the tree path co-exist with deferred-bonus?** Probably yes
  — they're orthogonal. Land branching first (bigger win), then
  layer deferred-bonus on top.
