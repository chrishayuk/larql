//! Naive sequential `target_forward` — phase 4b task B.2.
//!
//! Composes [`crate::predict_q4k_full_vocab_probs`] across the
//! tree's nodes to produce per-node target probability vectors.
//!
//! Each tree node's distribution comes from a full forward pass on
//! `history + ancestor_tokens(node)`. This is correctness-first —
//! O(tree_len × full_forward_pass) per call, which means a depth=2
//! branches=2 5-node tree at history=2000 is ~5× slower than
//! non-speculative.
//!
//! Phase 4c lands the batched implementation that composes the 3
//! GPU kernels (`q4k_batched`, `attn_tree`, `verify_tree_p`) for the
//! actual perf win. This naive path is the **parity oracle** for
//! the batched kernel.

use larql_compute::backend::DecodeBackend;
use larql_compute::FullPipelineLayer;
use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use super::tree::DraftTree;
use super::TokenId;

/// Attention dims passed to `target_forward_via_speculative_decode`.
/// Mirrors the parameter set that `decode_token` already takes.
#[derive(Clone, Copy, Debug)]
pub struct TargetForwardDims {
    pub hidden: usize,
    pub intermediate: usize,
    pub q_dim: usize,
    pub kv_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_base: f32,
}

/// **Phase 4c task C.2 — first slice.** Composes
/// `DecodeBackend::decode_tokens_speculative` (which advances + restores
/// the KV cache for each per-tree-node chain) with
/// [`crate::full_vocab_probs`] to produce per-node target probability
/// vectors.
///
/// Performance: O(tree_len × max_depth × decode_token). For depth=2
/// branches=1 (typical) on Gemma 3 4B / RTX 4090, that's ~3 ×
/// decode_token ≈ 22 ms per call (vs ~12 s for `target_forward_naive`).
/// **~500× speedup** over the from-scratch naive path.
///
/// Future C.2 follow-ups optimize further:
/// - Linear-chain optimization: process the full chain once, capture
///   per-position hiddens, avoid the per-node redundant chain walks
///   (drops cost from O(N×D) to O(N) for linear trees)
/// - Batched override on `CudaBackend::decode_tokens_speculative` that
///   composes `q4k_batched` + `tree_decode_attention` for true
///   single-pass batched compute (drops cost to O(1) batched call,
///   the architectural perf win)
///
/// Parity contract (locked, load-bearing): output bit-equal to
/// `target_forward_naive` within fp32 ordering tolerance (1e-5 per
/// element). The naive path is the parity oracle.
pub fn target_forward_via_speculative_decode(
    weights: &ModelWeights,
    history: &[TokenId],
    tree: &DraftTree,
    backend: &dyn DecodeBackend,
    layers: &[FullPipelineLayer<'_>],
    dims: TargetForwardDims,
) -> Option<Vec<Vec<f32>>> {
    target_forward_via_speculative_decode_with_probs(
        weights,
        history,
        tree,
        backend,
        layers,
        dims,
        |h| {
            // CPU fallback: apply_norm + dot_proj + softmax via
            // ndarray BLAS. ~50-100 ms per call on 4B (vocab=262144,
            // hidden=2560). Suitable for tests and callers without a
            // ComputeBackend in scope.
            let h_arr = match Array2::from_shape_vec((1, h.len()), h.to_vec()) {
                Ok(a) => a,
                Err(_) => return Vec::new(),
            };
            crate::forward::full_vocab_probs(weights, &h_arr, 1.0)
        },
    )
}

/// Variant of [`target_forward_via_speculative_decode`] that takes a
/// caller-supplied closure for the per-node lm_head + softmax step.
/// Lets the v3 spec dispatch route the lm_head GEMV through the GPU
/// (via `backend.q4k_matvec` on the index's lm_head bytes), avoiding
/// the ~50-100 ms CPU `dot_proj` per call that dominates spec wall-time
/// once the per-token decode and drafter slices are unblocked.
///
/// `compute_probs(h)` SHALL return the full softmax distribution
/// (length `vocab`) for the post-final-norm hidden state `h` (length
/// `dims.hidden`). Returning an empty Vec aborts the call (the helper
/// returns `None`).
pub fn target_forward_via_speculative_decode_with_probs<F>(
    weights: &ModelWeights,
    history: &[TokenId],
    tree: &DraftTree,
    backend: &dyn DecodeBackend,
    layers: &[FullPipelineLayer<'_>],
    dims: TargetForwardDims,
    mut compute_probs: F,
) -> Option<Vec<Vec<f32>>>
where
    F: FnMut(&[f32]) -> Vec<f32>,
{
    if !backend.has_kv_cache() {
        return None;
    }
    if backend.kv_cache_len() != history.len() {
        // Caller contract violation: cache must already cover the
        // canonical history before this call. We don't try to fix it.
        return None;
    }

    // Linear-chain optimization: when the tree has exactly one root-to-leaf
    // path (branches=1), walking the chain ONCE captures all per-node
    // hiddens. The chain order matches BFS order because each subsequent
    // node has the previous as parent, so `hiddens[k]` corresponds to
    // `tree.nodes()[k]`. This drops cost from O(N×D) to O(N) — for the
    // typical depth=2 b=1 default, that's 2 decode_token calls instead of 3.
    let paths = tree.root_to_leaf_paths();
    if paths.len() == 1 {
        let path = &paths[0];
        debug_assert_eq!(
            path.len(),
            tree.len(),
            "linear tree must have all nodes on the single path"
        );
        // Path order IS the chain order for a linear tree built via
        // build_linear_tree (root → child → ... → leaf, indices 0..N-1).
        // Verify defensively: each path[k] should equal k.
        for (k, &node_idx) in path.iter().enumerate() {
            if node_idx != k {
                // Tree is single-path but not built linearly — fall through
                // to the general per-node walk below for safety.
                return target_forward_via_speculative_decode_per_node(
                    weights,
                    history,
                    tree,
                    backend,
                    layers,
                    dims,
                    compute_probs,
                );
            }
        }
        let chain: Vec<TokenId> = path.iter().map(|&i| tree.nodes()[i].token.id).collect();
        let x_per_token: Vec<Vec<f32>> = chain
            .iter()
            .map(|&tok| {
                let h = crate::forward::embed_tokens_pub(weights, &[tok]);
                h.row(0).to_vec()
            })
            .collect();
        let hiddens = backend.decode_tokens_speculative(
            layers,
            &x_per_token,
            dims.hidden,
            dims.intermediate,
            dims.q_dim,
            dims.kv_dim,
            dims.num_q_heads,
            dims.num_kv_heads,
            dims.head_dim,
            dims.rope_base,
        )?;
        if hiddens.len() != tree.len() {
            return None;
        }
        let mut per_node = Vec::with_capacity(tree.len());
        for h in hiddens {
            let probs = compute_probs(&h);
            if probs.is_empty() {
                return None;
            }
            per_node.push(probs);
        }
        return Some(per_node);
    }

    // General tree case (branches > 1): per-node walks with rollback.
    target_forward_via_speculative_decode_per_node(
        weights,
        history,
        tree,
        backend,
        layers,
        dims,
        compute_probs,
    )
}

/// **Phase 4c skip-redundant-commit variant.** Same as
/// [`target_forward_via_speculative_decode_with_probs`] but uses the
/// keep-cache decode path: the backend's KV cache is left advanced
/// by `tree.len()` positions on success (one cache slot per drafted
/// token). The CALLER is responsible for truncating to the correct
/// position after `verify_tree` decides which prefix is accepted —
/// typically `truncate_kv_cache(pre_len + R)` then `decode_token(bonus)`
/// to fill the bonus position.
///
/// Linear-chain only: branching trees fall back to the existing
/// (truncate) `target_forward_via_speculative_decode_with_probs` path.
/// On any failure, cache state is RESTORED to pre-call length.
pub fn target_forward_via_speculative_decode_keep_cache_with_probs<F>(
    weights: &ModelWeights,
    history: &[TokenId],
    tree: &DraftTree,
    backend: &dyn DecodeBackend,
    layers: &[FullPipelineLayer<'_>],
    dims: TargetForwardDims,
    mut compute_probs: F,
) -> Option<Vec<Vec<f32>>>
where
    F: FnMut(&[f32]) -> Vec<f32>,
{
    if !backend.has_kv_cache() {
        return None;
    }
    let pre_len = backend.kv_cache_len();
    if pre_len != history.len() {
        return None;
    }

    let paths = tree.root_to_leaf_paths();
    if paths.len() != 1 {
        return target_forward_via_speculative_decode_with_probs(
            weights,
            history,
            tree,
            backend,
            layers,
            dims,
            compute_probs,
        );
    }
    let path = &paths[0];
    debug_assert_eq!(path.len(), tree.len());
    for (k, &node_idx) in path.iter().enumerate() {
        if node_idx != k {
            return target_forward_via_speculative_decode_with_probs(
                weights,
                history,
                tree,
                backend,
                layers,
                dims,
                compute_probs,
            );
        }
    }

    let chain: Vec<TokenId> = path.iter().map(|&i| tree.nodes()[i].token.id).collect();
    let x_per_token: Vec<Vec<f32>> = chain
        .iter()
        .map(|&tok| {
            let h = crate::forward::embed_tokens_pub(weights, &[tok]);
            h.row(0).to_vec()
        })
        .collect();
    let hiddens = backend.decode_tokens_speculative_keep_cache(
        layers,
        &x_per_token,
        dims.hidden,
        dims.intermediate,
        dims.q_dim,
        dims.kv_dim,
        dims.num_q_heads,
        dims.num_kv_heads,
        dims.head_dim,
        dims.rope_base,
    )?;
    if hiddens.len() != tree.len() {
        backend.truncate_kv_cache(pre_len);
        return None;
    }
    let mut per_node = Vec::with_capacity(tree.len());
    for h in hiddens {
        let probs = compute_probs(&h);
        if probs.is_empty() {
            backend.truncate_kv_cache(pre_len);
            return None;
        }
        per_node.push(probs);
    }
    Some(per_node)
}

/// Per-node fallback for non-linear trees. Walks each tree node's
/// ancestor chain through `decode_tokens_speculative` independently
/// (cache rolled back between calls). Cost: O(N × max_depth).
///
/// Used by `target_forward_via_speculative_decode` when the tree has
/// branching (`paths.len() > 1`). For linear trees (the typical
/// branches=1 default), the caller takes the O(N) optimized path
/// instead.
fn target_forward_via_speculative_decode_per_node<F>(
    weights: &ModelWeights,
    _history: &[TokenId],
    tree: &DraftTree,
    backend: &dyn DecodeBackend,
    layers: &[FullPipelineLayer<'_>],
    dims: TargetForwardDims,
    mut compute_probs: F,
) -> Option<Vec<Vec<f32>>>
where
    F: FnMut(&[f32]) -> Vec<f32>,
{
    let mut per_node = Vec::with_capacity(tree.len());
    for node_idx in 0..tree.len() {
        let chain: Vec<TokenId> = tree
            .ancestors(node_idx)
            .iter()
            .rev()
            .map(|&i| tree.nodes()[i].token.id)
            .collect();
        let x_per_token: Vec<Vec<f32>> = chain
            .iter()
            .map(|&tok| {
                let h = crate::forward::embed_tokens_pub(weights, &[tok]);
                h.row(0).to_vec()
            })
            .collect();
        let hiddens = backend.decode_tokens_speculative(
            layers,
            &x_per_token,
            dims.hidden,
            dims.intermediate,
            dims.q_dim,
            dims.kv_dim,
            dims.num_q_heads,
            dims.num_kv_heads,
            dims.head_dim,
            dims.rope_base,
        )?;
        if hiddens.len() != chain.len() {
            return None;
        }
        let last = hiddens.last()?;
        let probs = compute_probs(last);
        if probs.is_empty() {
            return None;
        }
        per_node.push(probs);
    }
    Some(per_node)
}

/// Compute per-node target probabilities using a caller-supplied
/// hidden-state callback. This is the **integration-friendly**
/// variant that doesn't require `&mut ModelWeights` in its
/// signature — the caller's closure handles whatever mutability
/// the underlying forward pass needs.
///
/// For phase 4b's gpu.rs:735 wiring, the closure can be implemented
/// using the existing `decode_token` machinery (which takes
/// `&self` on the trait object, not `&mut weights`), avoiding the
/// borrow conflict between `&layers` (borrows weights) and
/// `predict_q4k_hidden` (requires `&mut weights`).
///
/// `compute_hidden(tokens) -> Array2<f32>` SHALL return the
/// `[seq_len, hidden]` hidden-state output of the target's forward
/// pass on `tokens`. Caller is responsible for cache state — the
/// closure may use a temporary cache, the canonical cache plus
/// rollback, or any other strategy that yields the right hidden
/// state.
pub fn target_forward_with_hidden<F>(
    weights: &ModelWeights,
    history: &[TokenId],
    tree: &DraftTree,
    mut compute_hidden: F,
) -> Vec<Vec<f32>>
where
    F: FnMut(&[TokenId]) -> Array2<f32>,
{
    let mut per_node = Vec::with_capacity(tree.len());
    for node_idx in 0..tree.len() {
        let mut chain: Vec<TokenId> = tree
            .ancestors(node_idx)
            .iter()
            .rev()
            .map(|&i| tree.nodes()[i].token.id)
            .collect();
        let mut context: Vec<TokenId> = Vec::with_capacity(history.len() + chain.len());
        context.extend_from_slice(history);
        context.append(&mut chain);
        let h = compute_hidden(&context);
        let probs = crate::forward::predict::full_vocab_probs(weights, &h, 1.0);
        per_node.push(probs);
    }
    per_node
}

/// **Phase 4c skeleton** — batched target forward via composing the
/// 3 GPU kernels in main (`q4k_batched` + `attn_tree` + `verify_tree_p`).
///
/// Replaces the naive O(40x) re-runs of `target_forward_naive` with
/// a single batched forward pass at `q_tokens = tree.len()`, then a
/// batched lm_head + softmax per tree node.
///
/// This stub exists to lock the function signature for phase 4c. The
/// implementation lands in the phase 4c PR series — see
/// `openspec/changes/cuda-spec-phase4b-complete/tasks.md` task list
/// C.1–C.6 for the breakdown.
///
/// **Parity contract** (load-bearing): `target_forward_batched(...)`
/// SHALL produce the same `Vec<Vec<f32>>` output as
/// `target_forward_naive(...)` on the same `(history, tree)` inputs,
/// within fp32 ordering tolerance (1e-5 absolute per element). The
/// existing `with_hidden_callback_matches_naive` test demonstrates
/// the pattern at small scale; phase 4c task C.5 scales it to 64
/// fixed RNG seeds.
///
/// Currently returns `unimplemented!()`. Phase 4c implementation
/// composes the kernels via:
///   1. `cuda::q4k_batched::matvec_batched` for Q/K/V/wo/gate_up/down
///      projections at `M_TILE = tree.len()` (microbench: 7× speedup
///      at M=8 vs M=1×8)
///   2. `cuda::attn_tree::tree_decode_attention` for attention with
///      the per-q ancestor mask
///   3. lm_head + softmax over vocab for each of `tree.len()` hidden
///      states (new helper needed: `cuda::sampling::full_softmax_batched`
///      or sequential calls to `crate::forward::predict::full_vocab_probs`)
#[allow(unused_variables, unused_mut)]
pub fn target_forward_batched(
    weights: &mut ModelWeights,
    history: &[TokenId],
    tree: &DraftTree,
    index: &VectorIndex,
) -> Vec<Vec<f32>> {
    unimplemented!(
        "phase 4c — see openspec/changes/cuda-spec-phase4b-complete/tasks.md \
         tasks C.1-C.6 for the implementation breakdown. Until then, callers \
         use target_forward_naive (slow but correct) as the parity oracle."
    )
}

/// Run the target model's forward pass on the ancestor sequence of
/// every tree node and return per-node vocab probability vectors.
///
/// `history` is the accepted tokens so far (prompt + accepted span).
/// `tree` is the speculative draft tree. The returned vector has
/// length `tree.len()`; element `k` is the target's distribution
/// over the next token at position `history.len() + depth_of(k)`,
/// after consuming the ancestor chain `history + ancestors_of(k)`.
///
/// **Performance**: O(tree_len × full_forward_pass). Each call
/// re-runs the target from scratch on `history.len() + depth + 1`
/// tokens. Phase 4c batches this across tree positions; this naive
/// path is the parity oracle.
pub fn target_forward_naive(
    weights: &mut ModelWeights,
    history: &[TokenId],
    tree: &DraftTree,
    index: &VectorIndex,
) -> Vec<Vec<f32>> {
    let mut per_node = Vec::with_capacity(tree.len());
    for node_idx in 0..tree.len() {
        // Walk from node up to root, then reverse so chain is
        // [root, child, grandchild, ..., this_node]. ancestor[0] is
        // self, ancestor[last] is root — reverse gives root-first.
        let mut chain: Vec<TokenId> = tree
            .ancestors(node_idx)
            .iter()
            .rev()
            .map(|&i| tree.nodes()[i].token.id)
            .collect();
        // Build the full target context: history + ancestor chain.
        let mut context: Vec<TokenId> = Vec::with_capacity(history.len() + chain.len());
        context.extend_from_slice(history);
        context.append(&mut chain);
        let probs = crate::predict_q4k_full_vocab_probs(weights, &context, index);
        per_node.push(probs);
    }
    per_node
}

#[cfg(test)]
mod tests {
    use super::super::DraftToken;
    use super::*;
    use std::env;

    fn vindex_path_or_skip() -> Option<std::path::PathBuf> {
        env::var("LARQL_FULL_VOCAB_PROBS_VINDEX")
            .ok()
            .map(std::path::PathBuf::from)
    }

    fn load(path: &std::path::Path) -> (ModelWeights, tokenizers::Tokenizer, VectorIndex) {
        let mut callbacks = larql_vindex::SilentLoadCallbacks;
        let weights =
            larql_vindex::load_model_weights_q4k(path, &mut callbacks).expect("load weights");
        let tokenizer = crate::load_tokenizer(path).expect("load tokenizer");
        let index = crate::open_inference_vindex(path).expect("load vindex");
        (weights, tokenizer, index)
    }

    #[test]
    fn linear_chain_returns_n_distributions() {
        let Some(path) = vindex_path_or_skip() else {
            return;
        };
        let (mut weights, _tok, index) = load(&path);
        let history = vec![2u32, 100, 200];
        // Linear depth=2 chain: root + 1 child + 1 grandchild.
        let mut tree = DraftTree::from_root(DraftToken {
            id: 50,
            p_draft: 1.0,
        });
        let n1 = tree.add_child(
            0,
            DraftToken {
                id: 51,
                p_draft: 1.0,
            },
        );
        let _n2 = tree.add_child(
            n1,
            DraftToken {
                id: 52,
                p_draft: 1.0,
            },
        );

        let per_node = target_forward_naive(&mut weights, &history, &tree, &index);
        assert_eq!(per_node.len(), 3, "one distribution per tree node");
        for (i, probs) in per_node.iter().enumerate() {
            assert_eq!(
                probs.len(),
                weights.vocab_size,
                "node {i} probs must equal vocab_size"
            );
            let sum: f64 = probs.iter().map(|&p| p as f64).sum();
            assert!(
                (sum - 1.0).abs() < 1e-3,
                "node {i} probs must sum to 1.0, got {sum}"
            );
        }
    }

    #[test]
    fn root_distribution_matches_predict_q4k_full_vocab_probs_on_history_plus_root() {
        // The root node's distribution should match running the target
        // on `history + [root_id]` directly. This is the load-bearing
        // parity check — proves the ancestor walk reconstruction is
        // correct.
        let Some(path) = vindex_path_or_skip() else {
            return;
        };
        let (mut weights, _tok, index) = load(&path);
        let history = vec![2u32, 100, 200];
        let root_id = 50u32;
        let tree = DraftTree::from_root(DraftToken {
            id: root_id,
            p_draft: 1.0,
        });

        let via_naive = target_forward_naive(&mut weights, &history, &tree, &index);
        let via_direct = crate::predict_q4k_full_vocab_probs(
            &mut weights,
            &[history.as_slice(), &[root_id]].concat(),
            &index,
        );

        assert_eq!(via_naive.len(), 1);
        let via_naive_root = &via_naive[0];
        assert_eq!(via_naive_root.len(), via_direct.len());
        // fp32 round-trip through the same lm_head + softmax should
        // be bit-identical (same input tokens → same hidden →
        // same probs).
        for (i, (a, b)) in via_naive_root.iter().zip(&via_direct).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "vocab {i}: target_forward_naive {a} vs direct {b}"
            );
        }
    }

    #[test]
    fn with_hidden_callback_matches_naive() {
        // target_forward_with_hidden using a closure that calls
        // predict_q4k_hidden directly SHALL produce bit-identical
        // results to target_forward_naive (which calls
        // predict_q4k_full_vocab_probs internally).
        // Proves the callback variant is the parity-equivalent
        // integration-friendly API that next session uses for
        // gpu.rs:735 wiring.
        let Some(path) = vindex_path_or_skip() else {
            return;
        };
        let (mut weights, _tok, index) = load(&path);
        let history = vec![2u32, 100, 200];
        let mut tree = DraftTree::from_root(DraftToken {
            id: 50,
            p_draft: 1.0,
        });
        let _n1 = tree.add_child(
            0,
            DraftToken {
                id: 51,
                p_draft: 1.0,
            },
        );

        // Naive variant — uses predict_q4k_full_vocab_probs internally.
        let via_naive = target_forward_naive(&mut weights, &history, &tree, &index);

        // Callback variant — caller provides a closure that calls
        // predict_q4k_hidden directly. Same compute path, just
        // restructured so the function signature doesn't require
        // &mut weights.
        let mut weights_for_callback =
            larql_vindex::load_model_weights_q4k(&path, &mut larql_vindex::SilentLoadCallbacks)
                .unwrap();
        let via_callback = {
            let weights_immut: &ModelWeights = &weights_for_callback as &ModelWeights;
            target_forward_with_hidden(weights_immut, &history, &tree, |tokens| {
                // Hack: closure needs &mut weights, but we pass &weights_immut
                // outside. Use unsafe? No — restructure: the closure can take
                // a separately-loaded mutable copy.
                // For test simplicity, use a SECOND ModelWeights instance.
                let mut second = larql_vindex::load_model_weights_q4k(
                    &path,
                    &mut larql_vindex::SilentLoadCallbacks,
                )
                .unwrap();
                crate::vindex::predict_q4k_hidden(&mut second, tokens, &index, None)
            })
        };
        // Suppress unused warning.
        let _ = &mut weights_for_callback;

        assert_eq!(via_naive.len(), via_callback.len());
        for (k, (a, b)) in via_naive.iter().zip(&via_callback).enumerate() {
            assert_eq!(a.len(), b.len(), "node {k} length");
            for (i, (av, bv)) in a.iter().zip(b).enumerate() {
                assert!(
                    (av - bv).abs() < 1e-5,
                    "node {k} vocab {i}: naive {av} vs callback {bv}"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "phase 4c")]
    fn batched_stub_panics_until_phase4c() {
        // Phase 4c skeleton: target_forward_batched is a stub until
        // tasks C.1-C.6 land. This test asserts the stub fires (rather
        // than silently returning wrong output) and will start failing
        // when C.2 implements the real body — at which point the test
        // gets removed/replaced with the actual parity test.
        let Some(path) = vindex_path_or_skip() else {
            panic!("phase 4c stub test requires LARQL_FULL_VOCAB_PROBS_VINDEX env var");
        };
        let (mut weights, _tok, index) = load(&path);
        let history = vec![2u32];
        let tree = DraftTree::from_root(DraftToken {
            id: 50,
            p_draft: 1.0,
        });
        let _ = target_forward_batched(&mut weights, &history, &tree, &index);
    }

    #[test]
    fn via_speculative_decode_returns_none_without_kv_cache() {
        // CpuBackend doesn't implement DecodeBackend's KV-cache methods
        // (has_kv_cache returns false). The function MUST return None
        // rather than panicking — caller treats as "fall through to
        // non-speculative path".
        let Some(path) = vindex_path_or_skip() else {
            return;
        };
        let (weights, _tok, _index) = load(&path);
        let history = vec![2u32];
        let tree = DraftTree::from_root(DraftToken {
            id: 100,
            p_draft: 1.0,
        });
        struct NoKvBackend;
        impl larql_compute::backend::DecodeBackend for NoKvBackend {
            // All defaults — has_kv_cache returns false, so the call short-circuits.
        }
        let backend = NoKvBackend;
        let dims = TargetForwardDims {
            hidden: weights.hidden_size,
            intermediate: 1, // value irrelevant since we expect None
            q_dim: 1,
            kv_dim: 1,
            num_q_heads: 1,
            num_kv_heads: 1,
            head_dim: 1,
            rope_base: 10000.0,
        };
        let result =
            target_forward_via_speculative_decode(&weights, &history, &tree, &backend, &[], dims);
        assert!(
            result.is_none(),
            "must return None when backend lacks KV cache"
        );
    }

    #[test]
    fn empty_history_with_root_only_tree_works() {
        let Some(path) = vindex_path_or_skip() else {
            return;
        };
        let (mut weights, _tok, index) = load(&path);
        let history: Vec<u32> = vec![2]; // BOS only
        let tree = DraftTree::from_root(DraftToken {
            id: 100,
            p_draft: 1.0,
        });
        let per_node = target_forward_naive(&mut weights, &history, &tree, &index);
        assert_eq!(per_node.len(), 1);
        let sum: f64 = per_node[0].iter().map(|&p| p as f64).sum();
        assert!((sum - 1.0).abs() < 1e-3);
    }
}
