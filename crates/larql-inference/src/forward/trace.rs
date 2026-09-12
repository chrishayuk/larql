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
        let h_post_attn = match run_attention(larql_models::WeightsView::dense(weights), &h, layer)
        {
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

/// Attn-only forward with optional per-layer pre-W_O head zeroing.
///
/// Matches the `forward_attn_only` semantics used for the 2026-05-24
/// attn-vs-FFN decomposition: the FFN, PLE, and scalar tail are skipped at
/// every layer; the residual evolves only via the attention sublayer. At each
/// layer present in `head_zero_map`, the listed query-head slices are zeroed
/// before W_O via `run_attention_block_zero_pre_o_heads`. Other layers run
/// the standard attention block.
///
/// Returns a `HashMap<layer_index, (seq_len, hidden_size)>` for each layer in
/// `capture_layers`.
pub fn trace_forward_attn_only_with_head_zero(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
    head_zero_map: &std::collections::HashMap<usize, Vec<usize>>,
) -> std::collections::HashMap<usize, Array2<f32>> {
    let mut captures = std::collections::HashMap::new();
    if capture_layers.is_empty() {
        return captures;
    }
    let max_layer = *capture_layers.iter().max().unwrap_or(&0);
    let mut h = embed_tokens(weights, token_ids);
    let empty: Vec<usize> = Vec::new();

    for layer in 0..=max_layer {
        let heads_to_zero = head_zero_map.get(&layer).unwrap_or(&empty);
        let h_post_attn = if heads_to_zero.is_empty() {
            match run_attention(larql_models::WeightsView::dense(weights), &h, layer) {
                Some(p) => p,
                None => h.clone(),
            }
        } else {
            match crate::attention::run_attention_block_zero_pre_o_heads(
                larql_models::WeightsView::dense(weights),
                &h,
                layer,
                heads_to_zero,
                None,
            ) {
                Some((p, _)) => p,
                None => h.clone(),
            }
        };
        h = h_post_attn;
        if capture_layers.contains(&layer) {
            captures.insert(layer, h.clone());
        }
    }
    captures
}

/// Attn-only forward returning **pre-W_O per-head outputs** at each requested
/// layer. FFN, PLE, and scalar tail are skipped at every layer (matching the
/// `forward_attn_only` semantics). Used for Stage-2 per-head probing.
///
/// Returns a `HashMap<layer_index, (seq_len, num_q_heads * head_dim)>`. The
/// caller slices per-head: `pre_o[:, h * head_dim .. (h + 1) * head_dim]`.
pub fn trace_forward_attn_only_capture_pre_o(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
) -> std::collections::HashMap<usize, Array2<f32>> {
    let mut captures = std::collections::HashMap::new();
    if capture_layers.is_empty() {
        return captures;
    }
    let max_layer = *capture_layers.iter().max().unwrap_or(&0);
    let mut h = embed_tokens(weights, token_ids);

    for layer in 0..=max_layer {
        if capture_layers.contains(&layer) {
            // Capture both the post-attention residual (advances h) and pre_o.
            if let Some((h_post_attn, pre_o)) = crate::attention::run_attention_block_with_pre_o(
                larql_models::WeightsView::dense(weights),
                &h,
                layer,
            ) {
                captures.insert(layer, pre_o);
                h = h_post_attn;
            } else {
                // Skip layer with degenerate weights — leave h unchanged.
            }
        } else if let Some(h_post_attn) =
            run_attention(larql_models::WeightsView::dense(weights), &h, layer)
        {
            h = h_post_attn;
        }
    }
    captures
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
        h = match run_layer_with_ffn(
            larql_models::WeightsView::dense(weights),
            &h,
            layer,
            &ffn,
            false,
            ple_inputs.get(layer),
            None,
        ) {
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
            larql_models::WeightsView::dense(weights),
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
    //
    // Accumulate in f64. With ffn_dim in the thousands and activation
    // magnitudes ~1e3 (observed on Gemma 4 26B-A4B, ffn_dim = 2112), each
    // entry grows to ~1e6 * N; f32 (24-bit mantissa) loses the low-order
    // contributions and the accumulated Gram matrix stops being numerically
    // PSD, so the downstream Cholesky in memit.rs hits a negative pivot.
    // Converting to f64 *after* accumulation (memit.rs `cov_f64`) cannot
    // recover precision that was already lost here. We downcast to f32 once,
    // at the boundary, when returning.
    let mut ktk = Array2::<f64>::zeros((ffn_dim, ffn_dim));
    let mut total_samples: usize = 0;

    // Re-process the first capture so we don't double-count it.
    // `K^T K` for a (seq, ffn_dim) matrix: each row's outer product
    // with itself, summed across rows.
    for row in first.rows() {
        for i in 0..ffn_dim {
            let vi = row[i] as f64;
            if vi == 0.0 {
                continue;
            }
            for j in 0..ffn_dim {
                ktk[[i, j]] += vi * (row[j] as f64);
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
                let vi = row[i] as f64;
                if vi == 0.0 {
                    continue;
                }
                for j in 0..ffn_dim {
                    ktk[[i, j]] += vi * (row[j] as f64);
                }
            }
            total_samples += 1;
        }
    }

    if total_samples == 0 {
        return None;
    }

    // C = (K^T K) / N, then downcast to f32 once at the boundary.
    let scale = 1.0_f64 / total_samples as f64;
    ktk.mapv_inplace(|v| v * scale);
    let ktk_f32 = ktk.mapv(|v| v as f32);
    Some((ktk_f32, total_samples))
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
            larql_models::WeightsView::dense(weights),
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
mod tests;
