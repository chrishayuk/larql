## ADDED Requirements

### Requirement: Tree-mask batched attention kernel for spec verify

A new CUDA kernel `fused_prefill_attn_tree_mask` SHALL implement
batched-prefill attention with per-position ancestor masking, enabling
spec decode to verify multiple parallel chains in a single forward
pass without sibling-branch attention leakage.

Kernel contract:

1. Inputs:
   - `q_seq`, `k_seq`, `v_seq` — `[tree_len, num_q_heads * head_dim]`
     (or `kv_dim` for K/V), f32 device-resident.
   - `k_cache`, `v_cache` — `[max_seq, num_kv_heads, head_dim]`, f16
     device-resident (the existing KV cache layout).
   - `q_norm`, `k_norm` — optional per-head normalisation weights.
   - `out_seq` — `[tree_len, num_q_heads * head_dim]` f32, written.
   - `base_pos_dev` — `&CudaSlice<i32>` (Phase B pattern from
     `cuda-spec-cuda-graph`), so the captured graph can be replayed
     at different cache positions.
   - `ancestors_dev` — `&CudaSlice<u64>`, one bitset per tree node.
     Bit `k` set iff position `k` is an ancestor of the node
     (inclusive of self). Linear-chain trees produce bitsets
     `(1 << (k+1)) - 1` (= identical to a causal mask).
   - `seq_len` (= `tree_len`), `opts: FusedDecodeAttentionOpts`.

2. Per-node attention loop SHALL mask scores at positions `j` such
   that `j > base_pos + seq_len - 1` (causal beyond tree) OR
   `(ancestors[sp] & (1 << j_in_tree)) == 0` (j is not an ancestor
   in the tree). All other `j` participate normally.

3. K/V writes from `kv_cache_write_seq_f32` SHALL write each tree
   node's K/V at position `base_pos + tree_index` linearly. The mask
   filters reads, not writes — sibling K/Vs coexist in the cache
   slab; attention just doesn't see them.

4. Linear-chain trees SHALL produce **bit-identical output** to the
   existing `fused_prefill_attn_pos_dev` kernel. The mask reduces to
   a causal mask when all ancestor bitsets are causal.

5. Tree size SHALL be capped at 64 nodes (matches
   `SpecConfig::tree_nodes()`); fits the bitset in one `u64` per
   node.

#### Scenario: linear-chain tree-mask is bit-identical to causal mask

- **WHEN** the tree-mask kernel is invoked with bitsets
  `ancestors[k] = (1 << (k+1)) - 1` for all `k`
- **THEN** the output SHALL be bit-identical (f32 equality) to
  `fused_prefill_attn_pos_dev` on the same `(q, k, v, cache,
  base_pos, seq_len)`
<!-- test: unbacked -->

#### Scenario: tree-mask attention matches CPU reference on random trees

- **WHEN** the tree-mask kernel is run on 16 random tree shapes
  (depth 2-4, branches 2-4) with synthetic Q/K/V tensors
- **THEN** the per-node output SHALL match the CPU
  `target_forward_with_hidden` reference (which already supports
  trees via `predict_q4k_full_vocab_probs` and `DraftTree::ancestors`)
  with cosine ≥ 0.99 and max-element f32 diff ≤ 1e-3
<!-- test: unbacked -->

#### Scenario: sibling branches don't leak attention

- **WHEN** two parallel chains share a root but diverge at depth 1
  AND each has 4 unique tokens
- **THEN** the attention score at depth-2 nodes of chain A SHALL NOT
  read K/V from chain B's depth-1 or depth-2 nodes (verified by
  setting chain B's K to all-ones and confirming chain A's output is
  unchanged from the chain-A-only reference)
<!-- test: unbacked -->

### Requirement: Ancestor bitset construction is consistent with DraftTree::ancestors

The host-side ancestor-bitset construction SHALL produce bitsets
consistent with `DraftTree::ancestors(node)`:

1. For each node `n`, the bitset's set bits SHALL exactly match
   `DraftTree::ancestors(n)` (which returns `[n, parent(n), ...,
   root]` indices).
2. The bitset SHALL be sized `tree_len` bits (fits in `u64` for
   `tree_len ≤ 64`).
3. For the root (tree_index=0), the bitset SHALL be `1u64` (only
   the root itself).

#### Scenario: bitset matches DraftTree::ancestors for branching=2 depth=2

- **WHEN** a 7-node tree (root + 2 children + 4 grandchildren) is
  built via `build_branching_tree(drafts, branches=2)`
- **THEN** each node's bitset SHALL satisfy
  `bitset.count_ones() == tree.ancestors(node).len()` AND
  `(bitset >> k) & 1 == 1 iff k in tree.ancestors(node)`
<!-- test: unbacked -->
