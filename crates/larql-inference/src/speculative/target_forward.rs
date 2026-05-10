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

use larql_models::ModelWeights;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use super::tree::DraftTree;
use super::TokenId;

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
