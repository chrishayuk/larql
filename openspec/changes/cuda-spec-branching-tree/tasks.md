## Phase T1 — PLD-tree drafter

- [ ] T1.1 Extend `PromptLookupDrafter::lookup_continuation` to
      return up to `branches` distinct matches, ranked by recency
      (rightmost first). Currently returns a single continuation.
- [ ] T1.2 Add `propose_tree(h_target, depth, branches)` method on
      `Drafter` trait (default impl wraps `propose` as a linear
      chain of `depth` nodes, branches=1).
- [ ] T1.3 PLD-tree impl: for each of `branches` matches, gather its
      continuation (up to `depth` tokens). Build a `DraftTree` with
      shared root + `branches` parallel chains. When matches share
      a common prefix, **merge** them into a shared subtree (so the
      same prefix tokens aren't re-decoded).
- [ ] T1.4 Bench: PLD-tree at branches=2 vs linear on translation-
      echo prompt. Verify α per branch ≈ α linear (no degradation
      from picking second-best match).
- [ ] T1.5 Unit tests: empty history, single-match (= linear),
      two-match-disjoint, two-match-shared-prefix, n-match-pruned.

**Validation gate**: PLD-tree at `branches=1` produces bit-identical
output to PLD linear. ≥ 5 unit tests covering tree-shape variants.

## Phase T2 — Tree-aware batched attention kernel

- [ ] T2.1 Design ancestor-bitset format: for each tree node `n`,
      a `u64` (capped at 64 nodes per tree) bitset of its ancestor
      indices (inclusive of self). Caller computes once per tree
      from `DraftTree::ancestors`.
- [ ] T2.2 New CUDA kernel `fused_prefill_attn_tree_mask`:
      identical to `fused_prefill_attn` except the per-position
      attention loop masks `j > base_pos + ancestors[sp]` AND
      `j` not in (base_pos's ancestor bitset of `sp`).
- [ ] T2.3 Wrapper `fused_prefill_attention_tree_device_into_pos_dev`
      mirrors the existing `_pos_dev` variant but adds an
      `ancestors_dev: &CudaSlice<u64>` arg (one u64 per node).
- [ ] T2.4 Modify `kv_cache_write_seq_f32` to also accept the
      ancestor bitset — node positions in cache need to be assigned
      so that branch siblings don't overwrite each other's K/V.
      Two strategies:
      - **Strategy A** (allocate-by-tree-index): K/V written at
        `base_pos + tree_index`. Tree indices are dense [0, tree_len).
        Attention's ancestor mask filters access.
      - **Strategy B** (allocate-by-depth-and-branch): K/V written
        at `base_pos + depth_offset[d] + branch_id`. Requires
        per-depth offset table.
      Strategy A is simpler — use it.
- [ ] T2.5 Parity test: CPU `target_forward_with_hidden` (which
      already supports trees via `predict_q4k_full_vocab_probs`)
      vs GPU tree-mask. Run on 16 random trees (varying
      depth/branches) with cosine ≥ 0.99 on per-node hidden.
- [ ] T2.6 Single-chain reduction: when `tree.is_linear_chain()`
      the tree-mask kernel should produce bit-identical output to
      the existing `fused_prefill_attn_pos_dev` kernel.

**Validation gate**: 16-seed parity vs CPU reference passes;
single-chain bit-exact reduction; existing spec parity test still
passes when tree path is opt-in.

## Phase T3 — Spec scratch + dispatch tree path

- [ ] T3.1 Extend `SpecDecodeScratch` shape key to include
      `tree_layout_hash` so scratch + graph cache distinguishes
      different tree shapes (e.g. depth=2 branches=2 vs
      depth=4 branches=1).
- [ ] T3.2 Add `decode_tokens_speculative_tree_seq_device` that
      processes a `DraftTree` (vs a linear chain). Internally
      converts tree → flattened `[seq_len, hidden]` with the
      ancestor bitset.
- [ ] T3.3 Wire `target_forward_via_speculative_decode_keep_cache_*`
      to dispatch to the tree path when `paths.len() > 1`. Falls
      back to existing per-node path on any error (preserves the
      conservative fallback).
- [ ] T3.4 v3 dispatch: read `LARQL_SPEC_BRANCHES` env var (default
      1 = linear). Plumb branches into `SpecConfig`.
- [ ] T3.5 Graph capture per tree shape: `spec_decode_graph` cache
      key becomes `(seq_len, branches, depth, model_shape)`.
- [ ] T3.6 Add tree-shape-aware `compute_full_vocab_probs_batched`
      — same behaviour, just larger `m` for the bigger trees.
      No new code; verify the existing path scales.

**Validation gate**: depth=2 branches=2 at α≈0.85 per draft yields
emit count ≥ 1.5× linear depth=2 emit count on translation-echo;
no parity regression on linear-chain test; bench shows ≤ 12 ms/tok
on the perf-flip prompt.

## Phase T4 — Bench + flip

- [ ] T4.1 Update `bench_speculative_cmd` (or `bench`) to print
      per-iter tree shape (linear vs branching, total node count).
- [ ] T4.2 Add `test_speculative_branching_parity` test that
      compares branching vs linear spec output on 256 synthetic
      prompts. Linear and branching may diverge mid-stream
      (different verify samples), but should converge on
      completion lengths and respect EOS.
- [ ] T4.3 Run full bench matrix:
      `branches ∈ {1, 2, 4}` × `depth ∈ {2, 3, 4}` on the
      translation-echo and RAG-style prompts. Pick the sweet
      spot and document it.
- [ ] T4.4 Default-flip: if the best (depth, branches) hits the
      D.3 gate (≤11.7 ms/tok), flip `LARQL_SPECULATIVE_DECODE=1`
      to default ON in the CLI.

**Validation gate**: D.3 (≤1.6× plain decode) hit on Gemma 3 4B
Q4_K_M / RTX 4090 with the chosen (depth, branches).

## Phase T5 — Documentation + cleanup

- [ ] T5.1 Update `openspec/specs/inference-speculative-decoding/spec.md`
      with the new branching contract.
- [ ] T5.2 Update `crates/larql-inference/src/speculative/mod.rs`
      module docstring covering the tree path.
- [ ] T5.3 Archive `cuda-spec-phase4b-complete` if the perf-flip
      gate hits.

## Out of scope

- Mixed-precision lm_head via `cublasGemmEx` — separate small
  follow-up (~1-2 ms, modest).
- Deferred bonus into next iter's spec batch — orthogonal
  optimization, ~2 ms saving. Can stack on top of branching.
- Branching beyond `branches=4` — diminishing returns on PLD's
  finite n-gram match count.
