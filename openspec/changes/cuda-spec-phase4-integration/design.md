## Context

After PRs #1–#15, every kernel and CPU primitive for speculative
decoding is in main. The integration boundary (`maybe_speculative_step`)
and the off-the-shelf drafter (`SmallModelDrafter`) exist. The CLI
flag (`--draft-model`) loads the drafter. **What's missing**: the
actual call-site wiring at `gpu.rs:735` and the `target_forward`
closure that runs the target on a tree of N candidate tokens.

This document captures the integration design so the next session
implements rather than re-derives.

## Goals / Non-Goals

**Goals:**

- Define the call-site dispatch decision tree at `gpu.rs:735`.
- Specify how `target_forward` runs target on N tokens and emits
  per-node vocab probabilities.
- Specify KV cache advance + rollback semantics for accepted /
  rejected speculative spans.
- Specify the full-vocab probability path (not just top-k).
- Define the parity test methodology so phase 4b/c are gated by
  bit-equal token-ID equivalence to non-speculative.

**Non-Goals:**

- EAGLE draft head training or custom checkpoint formats.
- Multi-user concurrent serving (orthogonal to single-user latency).
- The actual implementation — that lives in phase 4b/c PRs.

## 1. The integration map

### 1.1 Where in the code

```text
crates/larql-cli/src/commands/primary/bench_cmd.rs
  └─ load drafter via --draft-model        (already in PR #15)
  └─ pass drafter into generate()          (NEW: phase 4b)

crates/larql-inference/src/layer_graph/generate/gpu.rs
  ├─ generate(...)                          (signature change: + Option<Box<dyn Drafter>>)
  └─ per-token loop @ ~line 735
       ├─ if speculative::enabled() && drafter.is_some():
       │    └─ maybe_speculative_step(drafter, cfg, h, cache_len, rng,
       │                              |tree| target_forward(tree))
       │    ├─ Some(tokens): emit each, advance cache by tokens.len()
       │    └─ None: fall through to legacy path (existing decode_token)
       └─ else: existing decode_token path (UNCHANGED)
```

### 1.2 The `generate()` signature change

```rust
pub fn generate(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    index: &VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: Range<usize>,
    drafter: Option<&mut SmallModelDrafter>,   // NEW
) -> GenerateResult
```

`drafter` is `Option<&mut SmallModelDrafter>` (not `Box<dyn Drafter>`)
to keep the type concrete — phase 4 only uses `SmallModelDrafter`.
Trait object is a future refinement if other Drafter impls land.

Existing callers pass `None`. The bench passes `Some(&mut drafter)`
when `--draft-model` is set.

## 2. The `target_forward` closure

### 2.1 Contract

```rust
fn target_forward(tree: &DraftTree) -> Vec<Vec<f32>>
```

Returns one vocab-sized probability vector per tree node, in node order.
The orchestrator's `verify_tree` consumes these to apply the rejection
rule.

### 2.2 Naive sequential implementation (phase 4b)

For each tree node `i` (BFS order):
1. Reconstruct the full ancestor sequence: `prompt + history + tree.ancestors(i)`
2. Run `predict_q4k(weights, tokenizer, &context, top_k=vocab, &index)` to get the target's distribution at the position **immediately after** the last token in the context
3. Convert the result to a `Vec<f32>` of length `vocab` (zero-fill missing entries)

**Performance**: O(tree_len × full_forward_pass). For depth=2 b=2
(5 nodes) at history=2000 tokens, this is **5× slower than baseline**.
Phase 4b is correctness-only; perf comes in phase 4c.

**Correctness gate**: `target_forward_naive(tree)[k]` SHALL equal the
target's standard decode probabilities at position `cache_len + k`
within fp32 ordering tolerance (1e-5 absolute per element).

### 2.3 Batched implementation (phase 4c)

Composes the 3 GPU kernels:

1. `cuda::q4k_batched::matvec_batched` for projections at `M_TILE = tree_len`
2. `cuda::attn_tree::tree_decode_attention` for attention with the tree mask
3. `cuda::sampling::verify_tree_gpu_parallel` (or a separate batched softmax) for vocab-sized probabilities

Per the empirical microbench (PR #11): M=8 takes 1.10× M=1 wall time.
So phase 4c's `target_forward` adds ~10% to one decode pass while
producing 5 per-node distributions — the architectural win.

**Correctness gate**: `target_forward_batched(tree)` SHALL produce the
same `AcceptedSpan` (token-ID equality) as `target_forward_naive(tree)`
across 64 fixed RNG seeds with the same draft tree input.

## 3. KV cache semantics

### 3.1 Advance on accept

After `maybe_speculative_step` returns `Some(tokens)`:
- Target's KV cache MUST be at position `cache_len + tokens.len()`
- Drafter's history MUST be at the same length

The naive `target_forward` re-runs from scratch, so the target's KV
cache is naturally rebuilt to the right state. **Caller must call
`drafter.accept(&tokens)` after a successful step.**

### 3.2 Rollback on reject

If verification rejects at tree node `r`:
- Tree positions `[r+1, tree_len)` were SPECULATIVELY APPLIED in
  the batched path (their K/V was written to the cache)
- The cache MUST be rolled back to position `cache_len + r`

For the **naive sequential** path (phase 4b): no rollback needed —
each forward runs from a clean ancestor context, never speculatively
writes to the canonical cache.

For the **batched** path (phase 4c): caller MUST track the
pre-speculative `cache_len` and truncate the target's cache after
rejection. The existing `DecodeBackend::truncate_kv_cache(len)` API
supports this.

### 3.3 Rotorquant interaction

Per `cuda-speculative-decoding/design.md` §5: the rotorquant compress
path needs a window-lag mode (`compress_with_window_lag`) so
speculative writes don't get permanently compressed before they're
known-accepted. **Phase 4c** depends on this; **phase 4b** doesn't
(no speculative cache writes in the naive path).

This proposal does NOT add the `compress_with_window_lag` API — that
lives in a separate `rotorquant-window-lag` change, prerequisite to
phase 4c.

## 4. The full-vocab probability path

### 4.1 Current state

`predict_q4k` returns `PredictResult { predictions: Vec<(String, f64)>, token_ids: Vec<u32> }`.
Length is `top_k` (1 in the bench, configurable). For verification
we need the **full vocab** distribution.

### 4.2 New API needed (phase 4b)

```rust
// crates/larql-inference/src/forward/predict/dense.rs
pub fn predict_full_vocab_probs(
    weights: &mut ModelWeights,
    tokenizer: &Tokenizer,
    token_ids: &[u32],
    index: &VectorIndex,
) -> Vec<f32>
```

Returns the softmax over the entire vocab (length = `weights.arch.vocab_size()`).
This is the existing `predict_q4k_hidden` + `lm_head` + `softmax`,
exposed without the top-k truncation.

**Cost**: identical to existing `predict_q4k` since softmax is O(vocab)
either way; we just don't sort + truncate.

### 4.3 Where to call it

Inside `target_forward_naive(tree)` for each tree node:

```rust
let context = [history.as_slice(), &tree.ancestor_tokens(node_idx)].concat();
let probs = predict_full_vocab_probs(weights, tokenizer, &context, index);
```

For batched (phase 4c): a single batched lm_head launch produces all
N distributions in one pass (vs N separate launches in naive).

## 5. Bonus token at the end of speculation

Per `verify_and_accept`'s all-accept branch, when every draft is
accepted we sample one bonus token from `p_target` at the deepest
accepted position. This bonus comes "for free" from the target's
forward pass that ran on the deepest tree node.

For naive: bonus is sampled from `target_forward_naive(tree)[deepest]`.
For batched: same — the kernel emits N+1 distributions naturally
(N from running on N tree positions, the +1 from the bonus position
predicted by the deepest forward).

## 6. Phasing

| Phase | Branch | Scope | Stop-ship gate |
|------:|--------|-------|----------------|
| 4a    | this PR | Design doc + spec | `openspec validate --strict` passes |
| 4b    | `feat/cuda-spec-naive-target-forward` | Naive sequential `target_forward` + `predict_full_vocab_probs` + dispatch wiring at `gpu.rs:735` | 256-prompt token-ID parity vs non-speculative on a fixed eval set |
| 4c    | `feat/cuda-spec-batched-target-forward` | Batched `target_forward` using the 3 GPU kernels + KV rollback + rotorquant lag | 256-prompt token-ID parity vs phase 4b's naive baseline; per-step latency ≤ 1.6× single-token decode |
| 4d    | `feat/cuda-spec-bench-and-eval` | `bench/decode_speculative.rs` + flip default if α ≥ 0.6 and ms/tok ≤ 5.5 | acceptance rate ≥ 0.6 on Gemma 3 4B Q4_K_M |

## 7. Test plan

### 7.1 Phase 4b

- **Unit**: `predict_full_vocab_probs` returns a normalized vec of length `vocab_size`; sum equals 1.0 within 1e-5
- **Unit**: `target_forward_naive(tree)[k][argmax]` equals the next-token argmax from non-speculative `predict_q4k` at the same position, for a fixed prompt + linear (depth=N branches=1) tree
- **Integration**: `generate()` with `Some(drafter)` and `LARQL_SPECULATIVE_DECODE=1` produces the same token IDs as `generate()` with `None` and the env unset, for 256 fixed prompts on Gemma 3 4B (the parity gate)
- **Integration**: `generate()` with the env unset + drafter passed produces bit-exactly the legacy path (no overhead, no diff)

### 7.2 Phase 4c

- **Unit**: `target_forward_batched(tree)` produces same `AcceptedSpan` as `target_forward_naive(tree)` across 64 fixed RNG seeds (the kernel-vs-CPU oracle equivalence we already have for verify_tree)
- **Integration**: 256-prompt parity vs phase 4b naive (which is parity-locked to non-speculative)
- **Perf**: per-step decode latency ≤ 1.6× single-token decode (empirical perf model says 1.10× — leaves headroom for KV management overhead)

### 7.3 Phase 4d

- **Bench**: `bench/decode_speculative.rs --measure α --report ms-per-tok` against `llama-cpp-turboquant`
- **Acceptance gate**: α ≥ 0.6 on the project eval set
- **Default-flip gate**: ms/tok ≤ 5.5 on Gemma 3 4B Q4_K_M, RTX 4090

## 8. Open questions

- **Sampling strategy** in `target_forward_naive`: temperature 1.0 (matches non-speculative greedy comparison)? Or pass through the bench's sampling config? **Decision**: temperature 1.0 for parity; bench separately exercises temperature paths in 4d.
- **Drafter sampling**: greedy (current) or temperature? Higher temperature → higher diversity → more rejections. **Decision**: greedy for 4b correctness; experiment in 4d.
- **Tree shape**: depth=2 branches=2 (5 nodes) per design.md, or depth=N branches=1 (linear) for phase 4b simplicity? **Decision**: linear for phase 4b (simplest correct), tree for phase 4c.
- **Full-vocab probability size**: Gemma 3 vocab = 256k → 1 MB per node × 5 nodes × 26 layers = ~130 MB per step. **Issue**: phase 4b's naive path with re-runs makes this fine; phase 4c's batched path needs to allocate this once not per-call.

## 9. Why not just …

- **… ship phase 4c directly without 4b?** Phase 4b's naive path is the parity oracle. Without it, 4c's batched correctness has no ground truth on real model output (only synthetic seeds from the existing `verify_tree` parity tests).
- **… use the existing `decode_token` repeatedly in `target_forward`?** Possible but less clean: `decode_token` advances the canonical KV cache, requiring rollback bookkeeping for the naive path too. `predict_q4k` from a clean context is conceptually simpler (re-runs from scratch, no cache state to preserve).
- **… train an EAGLE head instead of off-the-shelf?** Out of scope this proposal — covered in `cuda-speculative-decoding` proposal as a future option.
