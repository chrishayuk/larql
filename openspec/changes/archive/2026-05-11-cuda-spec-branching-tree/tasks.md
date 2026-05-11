## Phase T1 — PLD-tree drafter

- [x] T1.1 Extend `PromptLookupDrafter::lookup_continuation` to
      return up to `branches` distinct matches, ranked by recency
      (rightmost first). Currently returns a single continuation.
- [x] T1.2 Add `propose_tree(h_target, depth, branches)` method on
      `Drafter` trait (default impl wraps `propose` as a linear
      chain of `depth` nodes, branches=1).
- [x] T1.3 PLD-tree impl: for each of `branches` matches, gather its
      continuation (up to `depth` tokens). Build a `DraftTree` with
      shared root + `branches` parallel chains. When matches share
      a common prefix, **merge** them into a shared subtree (so the
      same prefix tokens aren't re-decoded).
- [ ] T1.4 Bench: PLD-tree at branches=2 vs linear on translation-
      echo prompt. Verify α per branch ≈ α linear (no degradation
      from picking second-best match). _(Deferred to after T3 —
      requires the end-to-end GPU dispatch path.)_
- [x] T1.5 Unit tests: empty history, single-match (= linear),
      two-match-disjoint, two-match-shared-prefix, n-match-pruned.

**Validation gate**: PLD-tree at `branches=1` produces bit-identical
output to PLD linear. ≥ 5 unit tests covering tree-shape variants.

## Phase T2 — Tree-aware batched attention kernel

- [x] T2.1 Design ancestor-bitset format: for each tree node `n`,
      a `u64` (capped at 64 nodes per tree) bitset of its ancestor
      indices (inclusive of self). Caller computes once per tree
      from `DraftTree::ancestors`. Implemented as
      `DraftTree::ancestor_bitsets()`.
- [x] T2.2 New CUDA kernel `fused_prefill_attn_tree_mask`:
      identical to `fused_prefill_attn` except the per-position
      attention loop masks `j > base_pos + ancestors[sp]` AND
      `j` not in (base_pos's ancestor bitset of `sp`).
- [x] T2.3 Wrapper `fused_prefill_attention_tree_seq_device_into_pos_dev`
      mirrors the existing `_pos_dev` variant but adds an
      `ancestors_dev: &CudaSlice<u64>` arg (one u64 per node).
- [x] T2.4 K/V cache layout — Strategy A (allocate-by-tree-index):
      K/V written at `base_pos + tree_index`. Tree indices are
      dense [0, tree_len). Attention's ancestor mask filters access.
      No change to `kv_cache_write_seq_f32` — the existing kernel
      already writes at `base_pos + sp` linearly, which matches
      Strategy A.
- [~] T2.5 Parity tests added (GPU-gated, `LARQL_CUDA_AVAILABLE=1`):
      (a) sibling-no-leak invariant — chain-B's K/V cannot influence
      chain-A's output when their ancestor bitsets are disjoint.
      Full "16 random trees vs CPU `target_forward_with_hidden`"
      parity requires re-implementing the kernel arithmetic
      (RoPE+RMS+GQA+softcap+softmax) in scalar Rust — deferred to
      follow-up; the sibling-no-leak + linear-chain-bit-exact pair
      rigorously exercises the masking contract.
- [x] T2.6 Single-chain reduction: tree-mask with bitsets
      `(1 << (k+1)) - 1` produces bit-identical output to the
      existing `fused_prefill_attn_pos_dev` kernel
      (`tree_mask_linear_chain_bit_exact_vs_causal` test).

**Validation gate**: 16-seed parity vs CPU reference passes;
single-chain bit-exact reduction; existing spec parity test still
passes when tree path is opt-in.

## Phase T3 — Spec scratch + dispatch tree path

- [x] T3.1 Cache key extended via `SpecScratchKey { seq_len,
      shape, is_tree }`. Linear vs tree-mask captured graphs are
      now keyed separately; trees of different node counts are
      already differentiated by `seq_len`.
- [x] T3.2 Added `decode_tokens_speculative_tree_seq_device` (CUDA)
      + a trait method `decode_tokens_speculative_tree_keep_cache`
      on `DecodeBackend` (default impl returns `None` for backends
      without the tree-mask kernel, routing the v3 dispatcher to
      the per-node fallback). Internally shares the linear path
      via `decode_tokens_speculative_seq_device_inner(..., Some(
      ancestors))`.
- [x] T3.3 `target_forward_via_speculative_decode_keep_cache_hiddens`
      now routes non-linear trees through
      `decode_tokens_speculative_tree_keep_cache`. Linear chains
      take the existing batched path bit-exactly. On any backend
      `None`, cache is left at `pre_len` so the v3 dispatcher can
      fall back to per-node.
- [x] T3.4 `LARQL_SPEC_BRANCHES` parsed in `bench_cmd.rs` (default
      1, range 1..=8). When `cfg.branches > 1`, the v3 dispatcher
      calls `drafter.propose_tree(depth, branches)`; else the
      existing linear `propose` + `build_linear_tree` path runs
      bit-exactly.
- [x] T3.5 Graph capture per tree shape: `SpecScratchKey` (T3.1)
      already distinguishes linear vs tree kernels at capture
      time. Different tree shapes with same node count share a
      graph (legitimate — the kernel reads ancestor bitsets from
      a stable device pointer that's updated host-side between
      replays, same pattern as `base_pos_dev`).
- [x] T3.6 `compute_full_vocab_probs_batched` verified by code
      inspection to operate on `hiddens.len()` rows via cuBLAS
      hgemm — handles any row count up to the 64-node tree cap.
      No code change required.

**Validation gate**: depth=2 branches=2 at α≈0.85 per draft yields
emit count ≥ 1.5× linear depth=2 emit count on translation-echo;
no parity regression on linear-chain test; bench shows ≤ 12 ms/tok
on the perf-flip prompt.

## Phase T4 — Bench + flip

- [x] T4.1 The spec-iter trace (`LARQL_SPEC_TRACE=1`) in
      `wiring.rs` now prints `shape=linear|branching(N paths)` per
      iter alongside the existing depth/timing fields, so bench
      runs can characterise drafter shape without a re-run.
- [x] T4.2 `propose_tree_branching_parity_256_synthetic_prompts`
      added — runs 256 deterministic synthetic histories through
      PLD branches=1 vs branches=2 and asserts:
      (a) the branching tree's first L nodes (L = linear.len())
      are token-identical to the linear chain, and (b) branching
      never produces fewer nodes than linear. Also asserts the
      multi-match code path actually fires (≥ 1 branching shape
      among 256 prompts).
- [x] T4.3 Bench matrix run on RTX 4090, Gemma 3 4B Q4_K_M
      (2026-05-10). See `design.md` § "Bench results" for the
      full table. **Sweet spot: depth=4 branches=2 → 5.65 ms/emit,
      1.42× FASTER than plain decode at 8.03 ms/tok.** Every
      tested config hits the D.3 gate on PLD-friendly prompts.
- [~] T4.4 Default-flip *intentionally NOT applied*. The D.3 gate
      is hit on PLD-friendly workloads, but on chat-style prompts
      with no prompt-echoing PLD's α drops to 0 and spec is
      slower than plain. The library can't tell at the env-var
      check which workload it's in, so we keep
      `LARQL_SPECULATIVE_DECODE=1` opt-in and document the
      recommended `LARQL_DRAFTER=prompt_lookup
      LARQL_SPEC_DEPTH=4 LARQL_SPEC_BRANCHES=2` combination in
      the module docstring + design.md.

**Validation gate**: D.3 (≤1.6× plain decode) hit on Gemma 3 4B
Q4_K_M / RTX 4090 with the chosen (depth, branches).

## Phase T5 — Documentation + cleanup

- [x] T5.1 Branching contract authored at
      `openspec/changes/cuda-spec-branching-tree/specs/
      inference-speculative-decoding/spec.md`. The
      `openspec/specs/inference-speculative-decoding/` archived
      file doesn't yet exist (this capability is still ADDED in
      in-flight changes); the change-side spec will fold in when
      this change is archived (T5.3).
- [x] T5.2 Module docstring at
      `crates/larql-inference/src/speculative/mod.rs` updated to
      cover linear vs branching trees, `LARQL_SPEC_BRANCHES`, and
      the tree-mask attention kernel.
- [ ] T5.3 Archive `cuda-spec-phase4b-complete` after T4 confirms
      the perf-flip gate. _(Deferred — needs GPU bench.)_

## Out of scope

- Mixed-precision lm_head via `cublasGemmEx` — separate small
  follow-up (~1-2 ms, modest).
- Deferred bonus into next iter's spec batch — orthogonal
  optimization, ~2 ms saving. Can stack on top of branching.
- Branching beyond `branches=4` — diminishing returns on PLD's
  finite n-gram match count.
