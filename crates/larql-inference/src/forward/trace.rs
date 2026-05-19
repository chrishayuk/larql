//! Tracing and calibration — capture residuals, activations, and attention weights.

use super::embed::embed_tokens;
use super::hooks::{LayerHook, NoopHook};
use super::layer::{
    apply_layer_scalar, run_attention, run_ffn, run_layer_with_capture_hooked, run_layer_with_ffn,
};
use super::ple::{apply_per_layer_embedding, precompute_per_layer_inputs};
use super::{LayerAttentionCapture, TraceResult};
use crate::ffn::{FfnBackend, WeightFfn};
use crate::model::ModelWeights;
use ndarray::Array2;

/// Per-layer residuals captured for speculation error analysis.
pub struct SpecCapture {
    /// Initial embedding (seq, hidden) before any transformer layers.
    pub h_0: Array2<f32>,
    /// Post-attention residual (last token only) at each layer — input to that layer's FFN.
    pub post_attn_last: Vec<Vec<f32>>,
    /// Post-full-layer residual (last token only) at each layer — output after FFN + PLE + scalar.
    pub post_layer_last: Vec<Vec<f32>>,
    /// Final hidden state (seq, hidden) after all layers, before final norm.
    pub h_final: Array2<f32>,
}

/// Single-pass capture for speculation error analysis.
///
/// Returns per-layer post-attention residuals (for true FFN delta) and
/// post-full-layer residuals (for logit-lens comparisons), plus the initial
/// embedding and final hidden state.
pub fn capture_spec_residuals(weights: &ModelWeights, token_ids: &[u32]) -> SpecCapture {
    let ffn = WeightFfn { weights };
    let h_0 = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h_0, token_ids);
    let seq_len = token_ids.len();
    let mut h = h_0.clone();

    let mut post_attn_last = Vec::with_capacity(weights.num_layers);
    let mut post_layer_last = Vec::with_capacity(weights.num_layers);

    for layer in 0..weights.num_layers {
        let h_post_attn = match run_attention(weights, &h, layer) {
            Some(pa) => pa,
            None => h.clone(),
        };
        post_attn_last.push(h_post_attn.row(seq_len - 1).to_vec());

        let (h_post_ffn, _) = run_ffn(weights, &h_post_attn, layer, &ffn, false);
        let mut h_new =
            apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
        apply_layer_scalar(weights, &mut h_new, layer);
        h = h_new;
        post_layer_last.push(h.row(seq_len - 1).to_vec());
    }

    SpecCapture {
        h_0,
        post_attn_last,
        post_layer_last,
        h_final: h,
    }
}

/// Run a forward pass through layers 0..=stop_layer and return the full
/// hidden state matrix (seq_len, hidden_size) at that layer.
pub fn forward_to_layer(
    weights: &ModelWeights,
    token_ids: &[u32],
    stop_layer: usize,
) -> Array2<f32> {
    let ffn = WeightFfn { weights };
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    for layer in 0..=stop_layer {
        h = match run_layer_with_ffn(weights, &h, layer, &ffn, false, ple_inputs.get(layer), None) {
            Some((h_new, _, _)) => h_new,
            None => continue,
        };
    }
    h
}

/// Run a forward pass and return last-token residuals at requested layers.
pub fn capture_residuals(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
) -> Vec<(usize, Vec<f32>)> {
    let trace = trace_forward(weights, token_ids, capture_layers, false, 0);
    trace.residuals
}

/// Capture decoy residuals at a single layer for a list of pre-tokenised
/// prompts. Returns one `Array1<f32>` per prompt (the last-token residual
/// at `layer`), in the same order as the input.
///
/// Used by INSERT's batch refine pass: the executor captures residuals
/// for canonical and template-matched decoy prompts, then feeds them
/// into the refine pass as suppression directions. One forward pass
/// per decoy. Cheap relative to the install itself.
pub fn capture_decoy_residuals(
    weights: &ModelWeights,
    token_ids_per_prompt: &[Vec<u32>],
    layer: usize,
) -> Vec<ndarray::Array1<f32>> {
    token_ids_per_prompt
        .iter()
        .map(|tokens| {
            let captured = capture_residuals(weights, tokens, &[layer]);
            // capture_residuals returns one (layer, vec) entry per
            // requested layer; we asked for exactly one.
            let (_, vec) = captured
                .into_iter()
                .next()
                .expect("capture_residuals must return one entry per requested layer");
            ndarray::Array1::from_vec(vec)
        })
        .collect()
}

/// Capture the **full** FFN activation matrix `(seq_len, ffn_dim)` at
/// a specific layer for one pre-tokenised prompt. Unlike
/// `capture_residuals` (which returns only the last token's residual
/// at the FFN entry point), this returns the per-token post-GEGLU
/// activation vectors — `k = silu(gate·x) * (up·x)` per position.
///
/// This is the key input for MEMIT-style closed-form weight edits:
/// ROME/MEMIT's covariance matrix `C = E_x[k(x) k(x)^T]` is built by
/// accumulating `K^T K / N` across many prompts, where each `K` is the
/// per-token activation matrix this function returns.
///
/// Requires the FFN backend to support activation capture. The
/// standard `WeightFfn` does; sparse backends may return zeros for
/// features they didn't score.
pub fn capture_ffn_activation_matrix(
    weights: &ModelWeights,
    token_ids: &[u32],
    layer: usize,
) -> Option<Array2<f32>> {
    use crate::ffn::WeightFfn;
    let ffn = WeightFfn { weights };

    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    for l in 0..=layer {
        // `run_layer_with_capture` returns the FFN activation matrix
        // (seq, ffn_dim) when `need_activation = true`, parallel to
        // `trace_forward_full`'s capture path but without the top-K
        // truncation that happens there.
        let need_activation = l == layer;
        let (h_new, activation, _, _) = crate::forward::layer::run_layer_with_capture(
            weights,
            &h,
            l,
            &ffn,
            need_activation,
            false,
            ple_inputs.get(l),
            None,
        )?;
        h = h_new;
        if l == layer {
            return activation;
        }
    }
    None
}

/// Accumulate the uncentered FFN activation covariance at a layer,
/// across many pre-tokenised prompts, in one pass. Returns a
/// `(ffn_dim, ffn_dim)` symmetric matrix approximately equal to
/// `E_x[k(x) k(x)^T]` where `k(x) = silu(gate·h) * (up·h)` at the
/// given layer.
///
/// Used at EXTRACT time by `COMPILE INTO VINDEX WITH MEMIT` to build
/// the covariance sidecar (`down_weights_covariance.bin`) that the
/// MEMIT weight-edit solve needs. Sampling a few thousand token
/// positions across a handful of diverse prompts is enough —
/// ~/chris-source/chris-experiments/compilation/15_v11_model/vindex_compile_rome_v11.py §20.3 shows
/// ~14K samples giving condition ~1e9, which is numerically stable.
///
/// Stable under accumulation: this is a true streaming implementation
/// (one Array2 of shape `(ffn_dim, ffn_dim)` + one counter), so the
/// memory footprint is fixed regardless of how many prompts you feed
/// in.
pub fn estimate_ffn_covariance(
    weights: &ModelWeights,
    token_ids_per_prompt: &[Vec<u32>],
    layer: usize,
) -> Option<(Array2<f32>, usize)> {
    // First pass: discover ffn_dim from the first successful capture.
    let first = token_ids_per_prompt
        .iter()
        .find_map(|tokens| capture_ffn_activation_matrix(weights, tokens, layer))?;
    let ffn_dim = first.shape()[1];

    // Accumulator — K^T K across all sampled token positions.
    // Float64 would be safer but Array2<f32> suffices at our scales
    // (we'll round to f32 when writing to disk anyway).
    let mut ktk = Array2::<f32>::zeros((ffn_dim, ffn_dim));
    let mut total_samples: usize = 0;

    // Re-process the first capture so we don't double-count it.
    // `K^T K` for a (seq, ffn_dim) matrix: each row's outer product
    // with itself, summed across rows.
    for row in first.rows() {
        for i in 0..ffn_dim {
            let vi = row[i];
            if vi == 0.0 {
                continue;
            }
            for j in 0..ffn_dim {
                ktk[[i, j]] += vi * row[j];
            }
        }
        total_samples += 1;
    }

    // Process the remaining prompts.
    let mut seen_first = false;
    for tokens in token_ids_per_prompt {
        if !seen_first {
            seen_first = true;
            continue;
        }
        let Some(k) = capture_ffn_activation_matrix(weights, tokens, layer) else {
            continue;
        };
        for row in k.rows() {
            for i in 0..ffn_dim {
                let vi = row[i];
                if vi == 0.0 {
                    continue;
                }
                for j in 0..ffn_dim {
                    ktk[[i, j]] += vi * row[j];
                }
            }
            total_samples += 1;
        }
    }

    if total_samples == 0 {
        return None;
    }

    // C = (K^T K) / N
    let scale = 1.0 / total_samples as f32;
    ktk.mapv_inplace(|v| v * scale);
    Some((ktk, total_samples))
}

/// Run a forward pass and capture both residuals and sparse activations.
pub fn trace_forward(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
    capture_activations: bool,
    activation_top_k: usize,
) -> TraceResult {
    let ffn = WeightFfn { weights };
    trace_forward_with_ffn(
        weights,
        token_ids,
        capture_layers,
        capture_activations,
        activation_top_k,
        &ffn,
    )
}

/// Run a forward pass with a custom FFN backend.
pub fn trace_forward_with_ffn(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
    capture_activations: bool,
    activation_top_k: usize,
    ffn: &dyn FfnBackend,
) -> TraceResult {
    trace_forward_full(
        weights,
        token_ids,
        capture_layers,
        capture_activations,
        activation_top_k,
        false,
        ffn,
    )
}

/// Run a forward pass capturing residuals, activations, and optionally attention weights.
///
/// Backwards-compatible wrapper around [`trace_forward_full_hooked`] using a
/// [`NoopHook`].
pub fn trace_forward_full(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
    capture_activations: bool,
    activation_top_k: usize,
    capture_attention: bool,
    ffn: &dyn FfnBackend,
) -> TraceResult {
    trace_forward_full_hooked(
        weights,
        token_ids,
        capture_layers,
        capture_activations,
        activation_top_k,
        capture_attention,
        ffn,
        &mut NoopHook,
    )
}

/// Hook-aware sibling of [`trace_forward_full`]. Fires the hook's callbacks
/// at every layer (not just `capture_layers`) — hooks decide for themselves
/// which layers they care about.
///
/// Use this for any inference-time intervention: pass a [`super::hooks::SteerHook`],
/// [`super::hooks::ZeroAblateHook`], a custom [`LayerHook`] impl, or a
/// [`super::hooks::CompositeHook`] combining several. The `TraceResult`
/// returned reflects the **post-intervention** residuals if the hook mutated
/// them.
#[allow(clippy::too_many_arguments)]
pub fn trace_forward_full_hooked(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
    capture_activations: bool,
    activation_top_k: usize,
    capture_attention: bool,
    ffn: &dyn FfnBackend,
    hook: &mut dyn LayerHook,
) -> TraceResult {
    let seq_len = token_ids.len();
    let max_layer = *capture_layers.iter().max().unwrap_or(&0);

    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut results = Vec::new();
    let mut activations: Vec<(usize, Vec<(usize, f32)>)> = Vec::new();
    let mut attention_captures: Vec<LayerAttentionCapture> = Vec::new();

    for layer in 0..=max_layer {
        let is_capture_layer = capture_layers.contains(&layer);
        let need_activation = capture_activations && is_capture_layer;
        let need_attention = capture_attention && is_capture_layer;

        let (h_new, activation, attn_weights, _) = match run_layer_with_capture_hooked(
            weights,
            &h,
            layer,
            ffn,
            need_activation,
            need_attention,
            ple_inputs.get(layer),
            None,
            hook,
        ) {
            Some(result) => result,
            None => continue,
        };
        h = h_new;

        if is_capture_layer {
            let last_row = h.row(seq_len - 1);
            results.push((layer, last_row.to_vec()));

            if let Some(act) = activation {
                let act_row = act.row(seq_len - 1);
                let mut indexed: Vec<(usize, f32)> = act_row.iter().copied().enumerate().collect();
                indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
                indexed.truncate(activation_top_k);
                activations.push((layer, indexed));
            }

            if let Some(weights) = attn_weights {
                attention_captures.push(LayerAttentionCapture { layer, weights });
            }
        }
    }

    TraceResult {
        residuals: results,
        activations,
        attention: attention_captures,
    }
}

/// Calibrate scalar gains from a forward pass: norm[L+1] / norm[L] at each layer.
pub fn calibrate_scalar_gains(weights: &ModelWeights, token_ids: &[u32]) -> Vec<f32> {
    let all_layers: Vec<usize> = (0..weights.num_layers).collect();
    let trace = trace_forward(weights, token_ids, &all_layers, false, 0);

    let mut gains = Vec::with_capacity(weights.num_layers);
    for i in 0..trace.residuals.len() {
        let norm_curr: f32 = trace.residuals[i]
            .1
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        if i + 1 < trace.residuals.len() {
            let norm_next: f32 = trace.residuals[i + 1]
                .1
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();
            gains.push(if norm_curr > 1e-12 {
                norm_next / norm_curr
            } else {
                1.0
            });
        } else {
            gains.push(1.0);
        }
    }
    gains
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::ModelWeights;
    use crate::test_utils::make_test_weights;
    use std::sync::OnceLock;

    fn shared_weights() -> &'static ModelWeights {
        static W: OnceLock<ModelWeights> = OnceLock::new();
        W.get_or_init(make_test_weights)
    }

    // ── capture_ffn_activation_matrix ─────────────────────────────────────────

    #[test]
    fn capture_ffn_activation_matrix_shape() {
        let weights = shared_weights();
        let result = capture_ffn_activation_matrix(weights, &[0u32, 1, 2], 0);
        let m = result.expect("should capture FFN activation at layer 0");
        assert_eq!(m.shape()[0], 3, "rows = seq_len");
        assert_eq!(m.shape()[1], weights.intermediate_size, "cols = ffn_dim");
        assert!(m.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn capture_ffn_activation_matrix_layer1() {
        let weights = shared_weights();
        let result = capture_ffn_activation_matrix(weights, &[0u32, 1], 1);
        let m = result.expect("should capture at layer 1");
        assert_eq!(m.shape(), &[2, weights.intermediate_size]);
    }

    #[test]
    fn capture_ffn_activation_matrix_single_token() {
        let weights = shared_weights();
        let result = capture_ffn_activation_matrix(weights, &[5u32], 0);
        let m = result.expect("single-token capture");
        assert_eq!(m.shape(), &[1, weights.intermediate_size]);
    }

    #[test]
    fn capture_ffn_activation_matrix_out_of_bounds_layer_returns_none() {
        let weights = shared_weights();
        // Layer 99 doesn't exist → should return None or fail gracefully
        let result = capture_ffn_activation_matrix(weights, &[0u32], 99);
        // Either None (layer out of range) or Some (shouldn't crash)
        if let Some(m) = result {
            assert!(m.iter().all(|v| v.is_finite()));
        }
    }

    // ── estimate_ffn_covariance ────────────────────────────────────────────────

    #[test]
    fn estimate_ffn_covariance_shape() {
        let weights = shared_weights();
        let prompts: Vec<Vec<u32>> = vec![vec![0u32, 1, 2], vec![3u32, 4], vec![5u32, 6, 7, 8]];
        let (cov, n_samples) =
            estimate_ffn_covariance(weights, &prompts, 0).expect("covariance should be computable");
        let ffn = weights.intermediate_size;
        assert_eq!(cov.shape(), &[ffn, ffn], "covariance is ffn_dim × ffn_dim");
        assert!(n_samples > 0, "should have accumulated samples");
        // Symmetric: C[i,j] ≈ C[j,i]
        for i in 0..ffn.min(4) {
            for j in 0..ffn.min(4) {
                assert!(
                    (cov[[i, j]] - cov[[j, i]]).abs() < 1e-4,
                    "covariance should be symmetric at [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn estimate_ffn_covariance_positive_semidefinite_diagonal() {
        let weights = shared_weights();
        let prompts = vec![vec![0u32, 1, 2, 3]];
        let (cov, _) = estimate_ffn_covariance(weights, &prompts, 0).unwrap();
        // Diagonal entries should be non-negative (x^T C x >= 0 for diagonal)
        for i in 0..cov.shape()[0] {
            assert!(
                cov[[i, i]] >= 0.0,
                "diagonal entry [{i},{i}] = {} should be >= 0",
                cov[[i, i]]
            );
        }
    }

    // ── capture_residuals ─────────────────────────────────────────────────────

    #[test]
    fn capture_residuals_count() {
        let weights = shared_weights();
        // capture_residuals(weights, token_ids, capture_layers) → Vec<(layer, residual_vec)>
        let residuals = capture_residuals(weights, &[0u32, 1, 2], &[0, 1]);
        assert!(!residuals.is_empty(), "residuals should be non-empty");
        for (layer, r) in &residuals {
            assert!(
                r.iter().all(|v| v.is_finite()),
                "layer {layer} residual has non-finite values"
            );
        }
    }

    #[test]
    fn capture_residuals_hidden_size() {
        let weights = shared_weights();
        let residuals = capture_residuals(weights, &[0u32], &[0]);
        for (_layer, r) in &residuals {
            assert_eq!(
                r.len() % weights.hidden_size,
                0,
                "residual len {} should be multiple of hidden_size {}",
                r.len(),
                weights.hidden_size
            );
        }
    }

    #[test]
    fn capture_residuals_returns_requested_layers() {
        let weights = shared_weights();
        let residuals = capture_residuals(weights, &[0u32, 1], &[0]);
        // Should return at least one entry for layer 0
        assert!(
            residuals.iter().any(|(l, _)| *l == 0),
            "should have layer 0 residual"
        );
    }

    // ── trace_forward_full_hooked ─────────────────────────────────────────────

    #[test]
    fn hooked_trace_with_noop_matches_baseline() {
        let weights = shared_weights();
        let ffn = WeightFfn { weights };
        let tokens = vec![0u32, 1, 2];
        let layers = vec![0, 1];

        let baseline = trace_forward_full(weights, &tokens, &layers, false, 0, false, &ffn);
        let hooked = trace_forward_full_hooked(
            weights,
            &tokens,
            &layers,
            false,
            0,
            false,
            &ffn,
            &mut crate::forward::NoopHook,
        );

        assert_eq!(baseline.residuals.len(), hooked.residuals.len());
        // BLAS on Windows reorders parallel reductions across successive
        // matmul calls (sometimes accompanied by `BLAS : Bad memory
        // unallocation!`), so two identical forward passes can drift in
        // the 1e-3 range. Linux/macOS BLAS stays well below 1e-6.
        const NOOP_HOOK_TOL: f32 = if cfg!(windows) { 1e-2 } else { 1e-6 };
        for ((bl, br), (hl, hr)) in baseline.residuals.iter().zip(hooked.residuals.iter()) {
            assert_eq!(bl, hl, "layer indices should match");
            for (b, h) in br.iter().zip(hr.iter()) {
                assert!(
                    (b - h).abs() < NOOP_HOOK_TOL,
                    "noop hook must not perturb residuals"
                );
            }
        }
    }

    #[test]
    fn hooked_trace_zero_ablate_propagates_through_remaining_layers() {
        let weights = shared_weights();
        let ffn = WeightFfn { weights };
        let tokens = vec![0u32, 1, 2];
        let layers: Vec<usize> = (0..weights.num_layers).collect();

        // Ablate layer 0 entirely; residuals at layers >0 must end up zero
        // since downstream layers see a zero residual entering them.
        let mut ablate = crate::forward::ZeroAblateHook::for_layers([0usize]);
        let result = trace_forward_full_hooked(
            weights,
            &tokens,
            &layers,
            false,
            0,
            false,
            &ffn,
            &mut ablate,
        );

        let layer0 = result
            .residuals
            .iter()
            .find(|(l, _)| *l == 0)
            .expect("layer 0 captured");
        assert!(
            layer0.1.iter().all(|v| *v == 0.0),
            "ZeroAblateHook should zero post-layer residual at layer 0"
        );
    }

    #[test]
    fn hooked_trace_record_captures_internal_state() {
        let weights = shared_weights();
        let ffn = WeightFfn { weights };
        let tokens = vec![0u32, 1];

        let mut record = crate::forward::RecordHook::for_layers([0usize, 1]);
        let _ = trace_forward_full_hooked(
            weights,
            &tokens,
            &[0, 1],
            false,
            0,
            false,
            &ffn,
            &mut record,
        );

        assert!(
            record.pre_layer.contains_key(&0) && record.pre_layer.contains_key(&1),
            "RecordHook should capture pre_layer at requested layers"
        );
        assert!(
            record.post_attention.contains_key(&0),
            "RecordHook should capture post_attention"
        );
        assert!(
            record.post_layer.contains_key(&1),
            "RecordHook should capture post_layer"
        );
        // Shape sanity: pre_layer at L1 should be (seq_len, hidden_size).
        let pre1 = record.pre_layer.get(&1).unwrap();
        assert_eq!(pre1.shape(), &[tokens.len(), weights.hidden_size]);
    }

    #[test]
    fn hooked_trace_fires_attention_weights_callback() {
        // on_attention_weights only fires when capture_attention=true on
        // a layer the trace was asked about.
        let weights = shared_weights();
        let ffn = WeightFfn { weights };
        let tokens = vec![0u32, 1, 2];

        let mut record = crate::forward::RecordHook::for_layers([0usize]);
        let _ = trace_forward_full_hooked(
            weights,
            &tokens,
            &[0],
            /*capture_activations=*/ false,
            0,
            /*capture_attention=*/ true,
            &ffn,
            &mut record,
        );

        let attn = record
            .attention_weights
            .get(&0)
            .expect("attention weights captured at layer 0");
        // Per-head: heads.len() = num_q_heads, each row has one entry per
        // attended position (last token attends to all 3 positions).
        let layer_num_q_heads = weights.arch.num_q_heads_for_layer(0);
        assert_eq!(
            attn.len(),
            layer_num_q_heads,
            "attention head count should equal num_q_heads"
        );
        for head in attn {
            assert_eq!(
                head.len(),
                tokens.len(),
                "each head row attends across all token positions"
            );
            assert!(head.iter().all(|v| v.is_finite()));
        }
    }

    // ── capture_spec_residuals ─────────────────────────────────────────

    #[test]
    fn capture_spec_residuals_returns_per_layer_last_token_dumps() {
        let weights = shared_weights();
        let tokens = vec![0u32, 1, 2];
        let spec = capture_spec_residuals(weights, &tokens);
        assert_eq!(spec.h_0.shape(), &[3, weights.hidden_size]);
        assert_eq!(spec.post_attn_last.len(), weights.num_layers);
        assert_eq!(spec.post_layer_last.len(), weights.num_layers);
        for v in &spec.post_attn_last {
            assert_eq!(v.len(), weights.hidden_size);
            assert!(v.iter().all(|x| x.is_finite()));
        }
        for v in &spec.post_layer_last {
            assert_eq!(v.len(), weights.hidden_size);
            assert!(v.iter().all(|x| x.is_finite()));
        }
        assert_eq!(spec.h_final.shape(), &[3, weights.hidden_size]);
    }

    #[test]
    fn capture_spec_residuals_single_token_works() {
        let weights = shared_weights();
        let spec = capture_spec_residuals(weights, &[5u32]);
        assert_eq!(spec.h_0.shape(), &[1, weights.hidden_size]);
        assert_eq!(spec.h_final.shape(), &[1, weights.hidden_size]);
        // Per-layer dumps still fire at seq_len=1.
        assert_eq!(spec.post_attn_last.len(), weights.num_layers);
    }

    // ── forward_to_layer ───────────────────────────────────────────────

    #[test]
    fn forward_to_layer_returns_full_seq_hidden() {
        let weights = shared_weights();
        let tokens = vec![1u32, 2, 3];
        let h = forward_to_layer(weights, &tokens, 0);
        assert_eq!(h.shape(), &[3, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_to_layer_progresses_through_layers() {
        // Stopping at layer 0 vs layer 1 should produce different residuals
        // unless the second layer is exactly an identity (it isn't with
        // random tinymodel weights).
        let weights = shared_weights();
        let tokens = vec![0u32, 1];
        let h0 = forward_to_layer(weights, &tokens, 0);
        let h1 = forward_to_layer(weights, &tokens, 1);
        let mut max_diff = 0.0f32;
        for (a, b) in h0.iter().zip(h1.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(
            max_diff > 1e-5,
            "layer 1 should mutate the residual, max_diff={max_diff}"
        );
    }

    // ── capture_decoy_residuals ────────────────────────────────────────

    #[test]
    fn capture_decoy_residuals_returns_one_array_per_prompt() {
        let weights = shared_weights();
        let prompts = vec![vec![0u32, 1], vec![2u32, 3, 4], vec![5u32]];
        let decoys = capture_decoy_residuals(weights, &prompts, 1);
        assert_eq!(decoys.len(), 3);
        for d in &decoys {
            assert_eq!(d.len(), weights.hidden_size);
            assert!(d.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn capture_decoy_residuals_empty_input_returns_empty() {
        let weights = shared_weights();
        let decoys = capture_decoy_residuals(weights, &[], 0);
        assert!(decoys.is_empty());
    }

    // ── estimate_ffn_covariance ────────────────────────────────────────

    #[test]
    fn estimate_ffn_covariance_returns_symmetric_psd_matrix() {
        let weights = shared_weights();
        let prompts = vec![vec![0u32, 1, 2], vec![3u32, 4, 5]];
        let (cov, samples) = estimate_ffn_covariance(weights, &prompts, 0)
            .expect("covariance must accumulate over multiple prompts");
        // Sum of seq_lens
        assert_eq!(samples, 6);
        let n = weights.intermediate_size;
        assert_eq!(cov.shape(), &[n, n]);
        // K^T K is symmetric — pin within float noise.
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (cov[[i, j]] - cov[[j, i]]).abs() < 1e-4,
                    "covariance not symmetric at ({i},{j}): {} vs {}",
                    cov[[i, j]],
                    cov[[j, i]]
                );
            }
        }
        // Diagonal must be non-negative (positive semidefinite).
        for i in 0..n {
            assert!(cov[[i, i]] >= 0.0, "diag[{i}] negative");
        }
    }

    #[test]
    fn estimate_ffn_covariance_single_prompt_works() {
        let weights = shared_weights();
        let prompts = vec![vec![0u32, 1, 2]];
        let (cov, samples) = estimate_ffn_covariance(weights, &prompts, 1)
            .expect("single prompt must still produce covariance");
        assert_eq!(samples, 3);
        assert_eq!(
            cov.shape(),
            &[weights.intermediate_size, weights.intermediate_size]
        );
    }

    #[test]
    fn estimate_ffn_covariance_empty_prompts_returns_none() {
        let weights = shared_weights();
        assert!(estimate_ffn_covariance(weights, &[], 0).is_none());
    }

    // ── trace_forward + trace_forward_with_ffn + trace_forward_full ────

    #[test]
    fn trace_forward_returns_residuals_at_requested_layers() {
        let weights = shared_weights();
        let tokens = vec![0u32, 1];
        let trace = trace_forward(weights, &tokens, &[0, 1], false, 0);
        assert_eq!(trace.residuals.len(), 2);
        assert_eq!(trace.residuals[0].0, 0);
        assert_eq!(trace.residuals[1].0, 1);
        assert!(trace.activations.is_empty());
    }

    #[test]
    fn trace_forward_with_activations_captures_topk_per_layer() {
        let weights = shared_weights();
        let tokens = vec![0u32, 1];
        let trace = trace_forward(weights, &tokens, &[0], true, 5);
        assert_eq!(trace.activations.len(), 1);
        let (layer, top) = &trace.activations[0];
        assert_eq!(*layer, 0);
        assert!(top.len() <= 5);
        // Activations sorted by magnitude desc.
        for w in top.windows(2) {
            assert!(w[0].1.abs() >= w[1].1.abs(), "top-K not sorted by |abs|");
        }
    }

    #[test]
    fn trace_forward_with_ffn_uses_supplied_backend() {
        let weights = shared_weights();
        let tokens = vec![0u32, 1];
        let ffn = WeightFfn { weights };
        let trace = trace_forward_with_ffn(weights, &tokens, &[0, 1], false, 0, &ffn);
        assert_eq!(trace.residuals.len(), 2);
    }

    #[test]
    fn trace_forward_full_with_attention_returns_attention_captures() {
        let weights = shared_weights();
        let ffn = WeightFfn { weights };
        let tokens = vec![0u32, 1];
        let trace = trace_forward_full(
            weights,
            &tokens,
            &[0],
            false,
            0,
            /*capture_attention=*/ true,
            &ffn,
        );
        assert_eq!(trace.attention.len(), 1);
        assert_eq!(trace.attention[0].layer, 0);
    }

    // ── calibrate_scalar_gains ─────────────────────────────────────────

    #[test]
    fn calibrate_scalar_gains_returns_one_per_layer() {
        let weights = shared_weights();
        let gains = calibrate_scalar_gains(weights, &[0u32, 1, 2]);
        assert_eq!(gains.len(), weights.num_layers);
        for g in &gains {
            assert!(g.is_finite(), "gain non-finite: {g}");
        }
    }

    #[test]
    fn calibrate_scalar_gains_last_layer_is_unity_fallback() {
        // Last layer has no successor → gain falls back to 1.0.
        let weights = shared_weights();
        let gains = calibrate_scalar_gains(weights, &[0u32]);
        assert_eq!(gains[gains.len() - 1], 1.0);
    }

    #[test]
    fn hooked_trace_fires_ffn_activation_callback() {
        // on_ffn_activation only fires when capture_activations=true on
        // a layer the trace was asked about.
        let weights = shared_weights();
        let ffn = WeightFfn { weights };
        let tokens = vec![0u32, 1];

        let mut record = crate::forward::RecordHook::for_layers([0usize]);
        let _ = trace_forward_full_hooked(
            weights,
            &tokens,
            &[0],
            /*capture_activations=*/ true,
            0,
            /*capture_attention=*/ false,
            &ffn,
            &mut record,
        );

        let act = record
            .ffn_activation
            .get(&0)
            .expect("FFN activation captured at layer 0");
        // Shape: (seq_len, ffn_dim).
        assert_eq!(act.shape(), &[tokens.len(), weights.intermediate_size]);
        assert!(act.iter().all(|v| v.is_finite()));
    }
}
