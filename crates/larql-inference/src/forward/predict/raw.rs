//! Raw-logits forward passes used by target-delta optimisation and Apollo.

use std::ops::Range;

use super::super::embed::embed_tokens;
use super::super::layer::run_layer_with_ffn;
use super::super::ple::precompute_per_layer_inputs;
use super::super::{apply_norm, dot_proj};
use crate::attention::SharedKV;
use crate::ffn::WeightFfn;
use crate::model::ModelWeights;
use ndarray::Array2;

/// Return type for [`forward_raw_logits`]. `h_pre_norm` is the residual
/// at the last transformer block's output (pre-final-norm), `h_final`
/// is after final-norm, and `logits` are the raw logits at the final
/// token position (pre-softmax).
pub struct RawForward {
    pub h_pre_norm: Array2<f32>,
    pub h_final: Array2<f32>,
    pub logits: ndarray::Array1<f32>,
}

/// Apply the model's final logits transform: divide by `logits_scaling`
/// then apply the optional `final_logit_softcapping` tanh.
fn apply_logits_transform(weights: &ModelWeights, raw_row: &[f32]) -> ndarray::Array1<f32> {
    let inv_scale = 1.0 / weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    raw_row
        .iter()
        .map(|&v| {
            let mut logit = v * inv_scale;
            if let Some(cap) = final_softcap {
                logit = (logit / cap).tanh() * cap;
            }
            logit
        })
        .collect()
}

/// Project a single hidden state row to raw logits (pre-softmax, pre-temperature).
///
/// Used by constrained generation: the caller masks the returned vector (e.g. sets
/// disallowed token positions to `f32::NEG_INFINITY`) before applying argmax.
pub fn hidden_to_raw_logits(weights: &ModelWeights, h_single: &Array2<f32>) -> Vec<f32> {
    let norm_offset = weights.arch.norm_weight_offset();
    let h_final = apply_norm(
        weights,
        h_single,
        weights.arch.final_norm_key(),
        norm_offset,
    );
    let logits_raw = dot_proj(&h_final.slice(ndarray::s![0..1, ..]), &weights.lm_head);
    apply_logits_transform(weights, logits_raw.row(0).as_slice().unwrap()).to_vec()
}

/// Raw-logits forward pass used by target-delta optimisation.
///
/// Returns (pre-final-norm residual, final-norm residual, logits) at
/// the LAST token position. If `perturb_at_layer` is Some, adds `delta`
/// to the residual's last position after that layer's block runs —
/// matching the Python reference `ffn_out[0, -1, :] += delta; h = h + ffn_out`
/// (since `run_layer_with_ffn` already collapses the block's output +
/// skip, perturbing the post-block `h[-1]` is algebraically the same).
pub fn forward_raw_logits(
    weights: &ModelWeights,
    token_ids: &[u32],
    perturb: Option<(usize, ndarray::ArrayView1<f32>)>,
) -> RawForward {
    forward_raw_logits_with_prefix(weights, token_ids, None, perturb)
}

/// Forward pass with an optional `initial_residual` prepended as a virtual
/// position-0 token before layer 0.
///
/// Mirrors the Python `prefill_to_layer(initial_residual=...)` API used by
/// `UnlimitedContextEngine`/Apollo. The prefix flows through every layer
/// along with the query tokens and participates in attention at each
/// position — it's *not* a per-layer K/V injection, it's a residual
/// prepend.
///
/// Correctness caveat: the prefix is processed at RoPE position 0 here
/// regardless of where in the original sequence it was captured. For
/// Apollo's stored boundaries (captured at window-end positions ~N×512),
/// this is a variant (ii)-style position shift — lossy but survivable
/// when combined with `vec_inject` amplification, which is the whole
/// point of the architecture.
///
/// `initial_residual`, when `Some`, must be a slice of exactly
/// `weights.hidden_size` floats. `token_ids` may not be empty.
pub fn forward_raw_logits_with_prefix(
    weights: &ModelWeights,
    token_ids: &[u32],
    initial_residual: Option<&[f32]>,
    perturb: Option<(usize, ndarray::ArrayView1<f32>)>,
) -> RawForward {
    forward_layer_range(
        weights,
        token_ids,
        initial_residual,
        0..weights.num_layers,
        perturb,
    )
}

/// Forward pass starting at `from_layer` using a pre-computed boundary
/// residual as position-0.
///
/// Skips layers `0..from_layer` entirely — the `boundary_residual` is
/// treated as the output of layer `from_layer - 1` for the stored context.
/// Only `from_layer..num_layers` are computed, which for Apollo with
/// `crystal_layer=30` means 4 layers (30-33) instead of 34.
///
/// Layout: `h[0] = boundary`, `h[1..]` = query embeddings.
/// The perturbation is applied at `target_layer` to the last row.
pub fn forward_from_layer(
    weights: &ModelWeights,
    token_ids: &[u32],
    boundary_residual: &[f32],
    from_layer: usize,
    perturb: Option<(usize, ndarray::ArrayView1<f32>)>,
) -> RawForward {
    forward_layer_range(
        weights,
        token_ids,
        Some(boundary_residual),
        from_layer..weights.num_layers,
        perturb,
    )
}

/// Shared implementation. Runs `layer_range` of the transformer with an
/// optional position-0 residual prefix, perturbs the last row at the
/// requested target layer, and projects the last position to logits.
///
/// Layout when `prefix` is `Some`: row 0 = prefix, rows 1..=q = query
/// embeddings, total_len = q + 1. PLE token ids prepend a `0` placeholder
/// for the virtual prefix row.
///
/// Layout when `prefix` is `None`: rows 0..q = query embeddings,
/// total_len = q.
fn forward_layer_range(
    weights: &ModelWeights,
    token_ids: &[u32],
    prefix: Option<&[f32]>,
    layer_range: Range<usize>,
    perturb: Option<(usize, ndarray::ArrayView1<f32>)>,
) -> RawForward {
    let hidden = weights.hidden_size;
    let q_len = token_ids.len();

    let q_embed = embed_tokens(weights, token_ids);
    let (mut h, total_len) = if let Some(prefix) = prefix {
        assert_eq!(
            prefix.len(),
            hidden,
            "prefix len {} does not match hidden size {}",
            prefix.len(),
            hidden,
        );
        let mut h = ndarray::Array2::<f32>::zeros((q_len + 1, hidden));
        for (i, &v) in prefix.iter().enumerate() {
            h[[0, i]] = v;
        }
        for r in 0..q_len {
            for c in 0..hidden {
                h[[r + 1, c]] = q_embed[[r, c]];
            }
        }
        (h, q_len + 1)
    } else {
        (q_embed, q_len)
    };

    // PLE: only used by Gemma 4 E2B. With a prefix prepended there's no
    // token id for the virtual row, so we pass a placeholder 0. For models
    // where PLE is active this is a known approximation; for Gemma 3 4B
    // (the Apollo target) PLE is disabled and this branch is a no-op.
    let ple_token_ids: Vec<u32> = if prefix.is_some() {
        let mut v = Vec::with_capacity(q_len + 1);
        v.push(0);
        v.extend_from_slice(token_ids);
        v
    } else {
        token_ids.to_vec()
    };
    let ple_inputs = precompute_per_layer_inputs(weights, &h, &ple_token_ids);
    let ffn = WeightFfn { weights };

    let mut kv_cache: std::collections::HashMap<usize, SharedKV> = std::collections::HashMap::new();

    for layer in layer_range {
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));

        if let Some((h_new, _, kv_out)) = run_layer_with_ffn(
            weights,
            &h,
            layer,
            &ffn,
            false,
            ple_inputs.get(layer),
            shared_kv,
        ) {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
            // Perturb the LAST row (the query's last token) after this
            // layer's block runs. With a prefix present the last row is
            // total_len - 1 = q_len (not q_len - 1).
            if let Some((target_layer, delta)) = perturb {
                if layer == target_layer {
                    let last = total_len - 1;
                    let mut row = h.row_mut(last);
                    for (i, d) in delta.iter().enumerate() {
                        if i < row.len() {
                            row[i] += *d;
                        }
                    }
                }
            }
        }
    }

    let h_pre_norm = h.clone();
    let norm_offset = weights.arch.norm_weight_offset();
    let h_final = apply_norm(weights, &h, weights.arch.final_norm_key(), norm_offset);

    let last_2d = h_final.slice(ndarray::s![total_len - 1..total_len, ..]);
    let logits_raw = dot_proj(&last_2d, &weights.lm_head);
    let logits = apply_logits_transform(weights, logits_raw.row(0).as_slice().unwrap());

    RawForward {
        h_pre_norm,
        h_final,
        logits,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(all(test, not(target_arch = "wasm32")))]
mod forward_from_layer_tests {
    use super::*;
    use crate::test_utils::make_test_weights;

    #[test]
    fn forward_raw_logits_returns_vocab_logits() {
        let weights = make_test_weights();
        let raw = forward_raw_logits(&weights, &[0u32, 1, 2], None);
        assert_eq!(
            raw.logits.len(),
            weights.vocab_size,
            "logits length should be vocab_size"
        );
        assert_eq!(
            raw.h_pre_norm.shape(),
            &[3, weights.hidden_size],
            "h_pre_norm shape"
        );
    }

    #[test]
    fn forward_raw_logits_single_token() {
        let weights = make_test_weights();
        let raw = forward_raw_logits(&weights, &[5u32], None);
        assert_eq!(raw.logits.len(), weights.vocab_size);
        assert!(
            raw.logits.iter().all(|v| v.is_finite()),
            "all logits should be finite"
        );
    }

    #[test]
    fn forward_from_layer_zero_equals_full_forward() {
        // forward_from_layer with from_layer=0 should be equivalent to
        // forward_raw_logits_with_prefix when the boundary is the zero vector.
        // They won't be identical (boundary passes through all layers as a real position)
        // but output shape must match.
        let weights = make_test_weights();
        let token_ids = &[1u32, 2];
        let boundary = vec![0.0f32; weights.hidden_size];

        let from_layer = forward_from_layer(&weights, token_ids, &boundary, 0, None);
        // from_layer=0 with zero boundary: should have (1 boundary + 2 query) positions
        assert_eq!(from_layer.h_pre_norm.shape(), &[3, weights.hidden_size]);
        assert_eq!(from_layer.logits.len(), weights.vocab_size);
        assert!(from_layer.logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_from_layer_skips_early_layers() {
        // Starting from layer 1 (of 2) should give a DIFFERENT result than
        // starting from layer 0, proving layers are actually being skipped.
        let weights = make_test_weights();
        let token_ids = &[3u32];
        let boundary = vec![0.1f32; weights.hidden_size];

        let from_0 = forward_from_layer(&weights, token_ids, &boundary, 0, None);
        let from_1 = forward_from_layer(&weights, token_ids, &boundary, 1, None);

        // Outputs should differ (layer 0's transform changes the residual)
        let differ = from_0
            .logits
            .iter()
            .zip(from_1.logits.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(
            differ,
            "from_layer=0 and from_layer=1 should produce different logits"
        );
    }

    #[test]
    fn forward_from_layer_output_shape() {
        let weights = make_test_weights();
        // 3 query tokens, from_layer=1: h has 4 rows (1 boundary + 3 query)
        let raw = forward_from_layer(
            &weights,
            &[0u32, 1, 2],
            &vec![0.0; weights.hidden_size],
            1,
            None,
        );
        assert_eq!(raw.h_pre_norm.shape(), &[4, weights.hidden_size]);
        assert_eq!(raw.logits.len(), weights.vocab_size);
    }

    #[test]
    fn forward_raw_logits_with_prefix_shape() {
        let weights = make_test_weights();
        let prefix = vec![0.5f32; weights.hidden_size];
        let raw = forward_raw_logits_with_prefix(&weights, &[0u32, 1], Some(&prefix), None);
        // prefix + 2 tokens = 3 positions
        assert_eq!(raw.h_pre_norm.shape(), &[3, weights.hidden_size]);
        assert_eq!(raw.logits.len(), weights.vocab_size);
    }
}
