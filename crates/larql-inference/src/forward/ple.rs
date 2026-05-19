//! Per-Layer Embeddings (PLE) — gated per-layer token embeddings.
//!
//! Gemma 4 E2B adds a per-layer embedding lookup to each layer's hidden state.
//! Two streams are combined: a model-level projection of the main embeddings,
//! and a per-layer token embedding lookup, scaled and gated.

use super::{apply_norm, dot_proj};
use crate::model::ModelWeights;
use ndarray::Array2;

/// Precompute per-layer input signals from token embeddings.
///
/// Combines two streams:
///   1. Model projection: main_embeds @ per_layer_model_projection.T * 1/sqrt(hidden)
///      → reshape to [seq, num_layers, ple_dim] → RMSNorm per layer
///   2. Per-layer token embed: embed_tokens_per_layer[token_ids] * sqrt(ple_dim)
///      → reshape to [seq, num_layers, ple_dim]
///      Combined: (stream1 + stream2) * 1/sqrt(2)
///
/// Returns a Vec of [seq, ple_dim] arrays, one per layer. Empty vec if PLE is not used.
pub fn precompute_per_layer_inputs(
    weights: &ModelWeights,
    main_embeds: &Array2<f32>,
    token_ids: &[u32],
) -> Vec<Array2<f32>> {
    let arch = &*weights.arch;
    if !arch.has_per_layer_embeddings() {
        return Vec::new();
    }

    let ple_dim = arch.per_layer_embed_dim();
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let hidden = weights.hidden_size;

    // Stream 1: model projection from main embeddings
    let model_proj_key = match arch.per_layer_model_projection_key() {
        Some(k) => k,
        None => return Vec::new(),
    };
    let w_model_proj = match weights.tensors.get(&model_proj_key) {
        Some(w) => w,
        None => return Vec::new(),
    };
    let projected = dot_proj(main_embeds, w_model_proj);
    let model_proj_scale = (hidden as f32).powf(-0.5);

    // Stream 2: per-layer token embeddings
    let ple_embed = arch
        .per_layer_embed_key()
        .and_then(|key| weights.tensors.get(&key));
    let embed_scale = (ple_dim as f32).sqrt();

    // Per-layer projection norm weight
    let proj_norm_w = arch
        .per_layer_projection_norm_key()
        .and_then(|key| weights.vectors.get(&key));
    let norm_offset = arch.norm_weight_offset();

    let norm_eps = arch.norm_eps();
    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    let mut per_layer_inputs = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let col_start = layer * ple_dim;
        let mut layer_input = Array2::<f32>::zeros((seq_len, ple_dim));

        for s in 0..seq_len {
            for d in 0..ple_dim {
                let val = projected[[s, col_start + d]] * model_proj_scale;
                layer_input[[s, d]] = val;
            }

            // Apply RMSNorm to stream 1 for this position
            if let Some(norm_w) = proj_norm_w {
                let mut sq_sum = 0.0f32;
                for d in 0..ple_dim {
                    sq_sum += layer_input[[s, d]] * layer_input[[s, d]];
                }
                let rms = (sq_sum / ple_dim as f32 + norm_eps).sqrt();
                let inv_rms = 1.0 / rms;
                for d in 0..ple_dim {
                    layer_input[[s, d]] *= inv_rms * (norm_offset + norm_w[d]);
                }
            }

            // Add stream 2: per-layer token embedding
            if let Some(embed) = ple_embed {
                let tok = token_ids[s] as usize;
                let row = embed.row(tok);
                for d in 0..ple_dim {
                    layer_input[[s, d]] += row[col_start + d] * embed_scale;
                }
            }

            // Scale combined by 1/sqrt(2)
            for d in 0..ple_dim {
                layer_input[[s, d]] *= inv_sqrt2;
            }
        }

        per_layer_inputs.push(layer_input);
    }

    per_layer_inputs
}

/// Apply Per-Layer Embeddings (PLE) to the hidden state after attention+FFN.
///
/// Runs at the end of each decoder layer:
///   gate = gelu_tanh(h @ input_gate.T)   → [seq, ple_dim]
///   gated = gate * per_layer_input        → [seq, ple_dim]
///   contribution = gated @ projection.T   → [seq, hidden]
///   normed = RMSNorm(contribution)
///   h = h + normed
pub(crate) fn apply_per_layer_embedding(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    per_layer_input: Option<&Array2<f32>>,
) -> Array2<f32> {
    let arch = &*weights.arch;
    let per_layer_input = match per_layer_input {
        Some(p) => p,
        None => return h.clone(),
    };

    let gate_key = match arch.per_layer_input_gate_key(layer) {
        Some(k) => k,
        None => return h.clone(),
    };
    let proj_key = match arch.per_layer_projection_key(layer) {
        Some(k) => k,
        None => return h.clone(),
    };
    let w_gate = match weights.tensors.get(&gate_key) {
        Some(w) => w,
        None => return h.clone(),
    };
    let w_proj = match weights.tensors.get(&proj_key) {
        Some(w) => w,
        None => return h.clone(),
    };

    // gate = h @ w_gate.T → [seq, ple_dim]
    let mut gate = dot_proj(h, w_gate);

    // Apply gelu_tanh activation to gate
    let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    for val in gate.iter_mut() {
        let x = *val;
        let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
        *val = 0.5 * x * (1.0 + inner.tanh());
    }

    // gated = gate * per_layer_input (element-wise)
    let gated = &gate * per_layer_input;

    // contribution = gated @ w_proj.T → [seq, hidden]
    let contribution = dot_proj(&gated, w_proj);

    // Apply post-PLE norm then residual add
    let norm_offset = arch.norm_weight_offset();
    let normed = match arch.post_per_layer_input_norm_key(layer) {
        Some(key) => apply_norm(weights, &contribution, &key, norm_offset),
        None => contribution,
    };

    h + &normed
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::test_utils::make_test_weights;
    use ndarray::Array2;

    fn input(seq: usize, hidden: usize) -> Array2<f32> {
        let data: Vec<f32> = (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.01).collect();
        Array2::from_shape_vec((seq, hidden), data).unwrap()
    }

    // ── precompute_per_layer_inputs ────────────────────────────────────────────

    #[test]
    fn precompute_returns_empty_when_arch_has_no_ple() {
        let weights = make_test_weights();
        // TinyModel arch does not have per_layer_embeddings → early return
        let embeds = input(3, weights.hidden_size);
        let token_ids = &[0u32, 1, 2];
        let result = precompute_per_layer_inputs(&weights, &embeds, token_ids);
        assert!(
            result.is_empty(),
            "non-PLE arch should return empty vec, got {} layers",
            result.len()
        );
    }

    #[test]
    fn precompute_returns_empty_when_projection_weight_missing() {
        // Even if arch claims PLE support, missing weight → empty return.
        // TinyModel arch doesn't enable PLE so this exercises the same early exit.
        let weights = make_test_weights();
        let embeds = Array2::zeros((1, weights.hidden_size));
        let result = precompute_per_layer_inputs(&weights, &embeds, &[0u32]);
        assert!(result.is_empty());
    }

    // ── apply_per_layer_embedding ─────────────────────────────────────────────

    #[test]
    fn apply_ple_none_input_returns_h_unchanged() {
        let weights = make_test_weights();
        let h = input(2, weights.hidden_size);
        let result = apply_per_layer_embedding(&weights, &h, 0, None);
        // None per_layer_input → h returned unchanged
        assert_eq!(result, h, "None per_layer_input should return h unchanged");
    }

    #[test]
    fn apply_ple_missing_gate_weight_returns_h_unchanged() {
        let weights = make_test_weights();
        let h = input(1, weights.hidden_size);
        // Provide a per_layer_input, but TinyModel has no per_layer gate tensors
        let dummy_input = Array2::zeros((1, 4));
        let result = apply_per_layer_embedding(&weights, &h, 0, Some(&dummy_input));
        // Gate key doesn't exist in TinyModel → returns h unchanged
        assert_eq!(result, h, "missing gate weight should return h unchanged");
    }

    #[test]
    fn apply_ple_output_shape_matches_input() {
        let weights = make_test_weights();
        let h = input(3, weights.hidden_size);
        let out = apply_per_layer_embedding(&weights, &h, 0, None);
        assert_eq!(out.shape(), h.shape());
    }

    // ── softmax (now in forward/ops) ──────────────────────────────────────────

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0f32, 2.0, 3.0, 0.5];
        let probs = crate::forward::softmax(&logits);
        let sum: f32 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax should sum to 1, got {sum}"
        );
    }

    #[test]
    fn softmax_preserves_argmax() {
        let logits = vec![0.1f32, 5.0, 0.2];
        let probs = crate::forward::softmax(&logits);
        let argmax = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(argmax, 1, "argmax should be preserved by softmax");
    }

    #[test]
    fn softmax_empty_input_returns_empty() {
        assert!(crate::forward::softmax(&[]).is_empty());
    }
}
