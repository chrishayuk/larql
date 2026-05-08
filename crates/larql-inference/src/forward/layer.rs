//! Layer dispatch — runs attention + FFN + PLE + layer_scalar for a single layer.
//!
//! Orchestrates the per-layer computation: attention (with optional KV sharing),
//! FFN, per-layer embeddings, and layer scalar multiplication.

use super::apply_norm;
use super::hooks::LayerHook;
use super::ple::apply_per_layer_embedding;
use crate::attention::{AttentionWeights, SharedKV};
use crate::ffn::FfnBackend;
use crate::model::ModelWeights;
use crate::residual::rms_norm;
use ndarray::Array2;

/// Public wrapper for run_attention — used by diagnostic/capture tooling.
pub fn run_attention_public(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
) -> Option<Array2<f32>> {
    run_attention(weights, h, layer)
}

/// Run attention for a single layer. Returns the post-attention residual.
pub(super) fn run_attention(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
) -> Option<Array2<f32>> {
    let (h_post_attn, _) = run_attention_inner(weights, h, layer, false, None)?;
    Some(h_post_attn)
}

/// Run attention with optional per-head weight capture and shared K/V.
pub(super) fn run_attention_inner(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Option<AttentionWeights>)> {
    let (h_post_attn, _attn_projected, attn_weights) =
        crate::attention::run_attention_block_shared(
            weights,
            h,
            layer,
            capture_attention,
            shared_kv,
        )?;
    Some((h_post_attn, attn_weights))
}

/// Run attention returning post-processed K/V for caching (KV sharing source layers).
pub(super) fn run_attention_with_kv_cache(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
) -> Option<(Array2<f32>, SharedKV)> {
    let (h_post_attn, _, _, k_rope, v_final) =
        crate::attention::run_attention_block_with_kv_out(weights, h, layer, false, None)?;
    Some((h_post_attn, (k_rope, v_final)))
}

/// Run FFN for a single layer using the given backend. Returns the post-FFN residual.
pub fn run_ffn(
    weights: &ModelWeights,
    h_post_attn: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    capture_activation: bool,
) -> (Array2<f32>, Option<Array2<f32>>) {
    let norm_offset = weights.arch.norm_weight_offset();
    let arch = &*weights.arch;

    // Layer-0 stage dumps (LARQL_CPU_STAGE_DUMP=<dir>) — matches the
    // Metal `LARQL_METAL_DUMP_LAYERS` convention. Lets us diff per-stage
    // intermediates between CPU and Metal for the first layer.
    let stage_dump_dir = if layer == 0 {
        std::env::var("LARQL_CPU_STAGE_DUMP").ok()
    } else {
        None
    };
    let dump_f32 = |name: &str, arr: &Array2<f32>| {
        if let Some(ref dir) = stage_dump_dir {
            let slice = arr.as_slice().unwrap_or(&[]);
            let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = std::fs::write(format!("{dir}/cpu_L0_{name}.f32"), &bytes);
        }
    };
    dump_f32("h_post_attn", h_post_attn);

    let pre_ffn_key = if arch.has_post_norms() {
        arch.pre_feedforward_layernorm_key(layer)
    } else {
        Some(arch.post_attention_layernorm_key(layer))
    };
    let h_ffn = match pre_ffn_key {
        Some(key) => apply_norm(weights, h_post_attn, &key, norm_offset),
        None => rms_norm(h_post_attn, None, norm_offset),
    };
    dump_f32("ffn_norm_out", &h_ffn);

    let (ffn_out, activation) = if capture_activation {
        let (out, act) = ffn.forward_with_activation(layer, &h_ffn);
        (out, Some(act))
    } else {
        (ffn.forward(layer, &h_ffn), None)
    };
    dump_f32("ffn_out_raw", &ffn_out);

    let res_mult = arch.residual_multiplier();
    let h_out = if arch.has_post_norms() {
        let normed = match arch.post_feedforward_layernorm_key(layer) {
            Some(key) => apply_norm(weights, &ffn_out, &key, norm_offset),
            None => rms_norm(&ffn_out, None, norm_offset),
        };
        if res_mult != 1.0 {
            h_post_attn + &(&normed * res_mult)
        } else {
            h_post_attn + &normed
        }
    } else if res_mult != 1.0 {
        h_post_attn + &(&ffn_out * res_mult)
    } else {
        h_post_attn + &ffn_out
    };

    (h_out, activation)
}

/// Apply per-layer scalar multiplier if present (e.g., Gemma 4 layer_scalar).
///
/// Skip when the scalar is 0.0 (absent / unloaded — multiplying would zero the
/// layer output, collapsing generation) or 1.0 (identity). Matches the Metal
/// `apply_whole_layer_scalar` in `metal/decode/moe_combine.rs:88-94` so the
/// CPU MoE path produces the same residual as the GPU path.
pub(crate) fn apply_layer_scalar(weights: &ModelWeights, h: &mut Array2<f32>, layer: usize) {
    if let Some(key) = weights.arch.layer_scalar_key(layer) {
        if let Some(scalars) = weights.vectors.get(&key) {
            if let Some(&scalar) = scalars.first() {
                if scalar != 0.0 && scalar != 1.0 {
                    *h *= scalar;
                }
            }
        }
    }
}

/// Run a single transformer layer with the given FFN backend.
///
/// Handles: attention → FFN → per-layer embedding → layer_scalar.
/// All four steps are needed for Gemma 4 correctness. Exposed `pub` so
/// alternate forward drivers (notably `vindex::predict_q4k`) get the same
/// sequence as `predict_with_temperature` without duplicating logic.
#[allow(clippy::type_complexity)]
pub fn run_layer_with_ffn(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    capture_activation: bool,
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Option<Array2<f32>>, Option<SharedKV>)> {
    let (h_post_attn, kv_out) = if shared_kv.is_some() {
        (
            run_attention_inner(weights, h, layer, false, shared_kv)?.0,
            None,
        )
    } else {
        let (h_pa, kv) = run_attention_with_kv_cache(weights, h, layer)?;
        (h_pa, Some(kv))
    };
    // Diagnostic: per-layer `h_post_attn` dump, paired with Metal's
    // `metal_layer_{LL}_h_post_attn.f32`. Lets the `residual_diff` tool
    // bisect any layer's drift into attention (compare h_post_attn) vs
    // FFN+PLE+scalar (compare h_out minus h_post_attn). Gated on the
    // same env var as the end-of-layer dump; no overhead when unset.
    if let Ok(dir) = std::env::var("LARQL_CPU_DUMP_LAYERS") {
        let slice = h_post_attn.as_slice().unwrap_or(&[]);
        let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
        let path = format!("{dir}/cpu_layer_{layer:02}_h_post_attn.f32");
        let _ = std::fs::write(&path, &bytes);
    }
    let (h_post_ffn, activation) = run_ffn(weights, &h_post_attn, layer, ffn, capture_activation);
    let mut h_out = apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_input);
    apply_layer_scalar(weights, &mut h_out, layer);
    Some((h_out, activation, kv_out))
}

/// Like [`run_layer_with_ffn`] but ALSO returns `h_post_attn` — the
/// post-attention residual *before* the FFN runs. Used by the
/// `attention-service-routes` HTTP/gRPC handlers, which expose the
/// per-layer attention output as part of their response (and are
/// happy to discard the FFN output, which the client routes through
/// the FFN service / dense path / expert shards separately).
///
/// The returned `h_out` is the same value `run_layer_with_ffn` would
/// have produced, so a chain of these calls equals a normal forward
/// pass on the host attention server.
#[allow(clippy::type_complexity)]
pub fn run_layer_with_ffn_capturing_h_post_attn(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    capture_activation: bool,
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
) -> Option<(
    Array2<f32>,
    Array2<f32>,
    Option<Array2<f32>>,
    Option<SharedKV>,
)> {
    let (h_post_attn, kv_out) = if shared_kv.is_some() {
        (
            run_attention_inner(weights, h, layer, false, shared_kv)?.0,
            None,
        )
    } else {
        let (h_pa, kv) = run_attention_with_kv_cache(weights, h, layer)?;
        (h_pa, Some(kv))
    };
    let (h_post_ffn, activation) = run_ffn(weights, &h_post_attn, layer, ffn, capture_activation);
    let mut h_out = apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_input);
    apply_layer_scalar(weights, &mut h_out, layer);
    Some((h_out, h_post_attn, activation, kv_out))
}

/// Backend-aware variant of [`run_layer_with_ffn_capturing_h_post_attn`].
///
/// Attention projections route through `backend` when provided. The rest of
/// the layer semantics stay identical: FFN backend, PLE, layer scalar, and
/// optional shared-KV handling are preserved.
#[allow(clippy::type_complexity)]
pub fn run_layer_with_ffn_capturing_h_post_attn_backend(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    capture_activation: bool,
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
    backend: Option<&dyn larql_compute::ComputeBackend>,
) -> Option<(
    Array2<f32>,
    Array2<f32>,
    Option<Array2<f32>>,
    Option<SharedKV>,
)> {
    let (h_post_attn, kv_out) = if let Some(shared_kv) = shared_kv {
        (
            run_attention_inner(weights, h, layer, false, Some(shared_kv))?.0,
            None,
        )
    } else {
        let (h_pa, k, v) =
            crate::attention::gpu::run_attention_with_kv_backend(weights, h, layer, backend)?;
        (h_pa, Some((k, v)))
    };
    let (h_post_ffn, activation) = run_ffn(weights, &h_post_attn, layer, ffn, capture_activation);
    let mut h_out = apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_input);
    apply_layer_scalar(weights, &mut h_out, layer);
    Some((h_out, h_post_attn, activation, kv_out))
}

/// Run a single transformer layer, optionally capturing attention weights.
///
/// Backwards-compatible wrapper: behaves identically to the pre-hook version
/// by passing a [`super::hooks::NoopHook`].
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub(super) fn run_layer_with_capture(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    capture_activation: bool,
    capture_attention: bool,
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
) -> Option<(
    Array2<f32>,
    Option<Array2<f32>>,
    Option<AttentionWeights>,
    Option<SharedKV>,
)> {
    run_layer_with_capture_hooked(
        weights,
        h,
        layer,
        ffn,
        capture_activation,
        capture_attention,
        ple_input,
        shared_kv,
        &mut super::hooks::NoopHook,
    )
}

/// Hook-aware sibling of [`run_layer_with_capture`]. Fires the [`LayerHook`]
/// callbacks at four points inside the layer: pre-layer, post-attention
/// (mut), attention-weights / FFN-activation if captured, post-layer (mut).
///
/// The two `&mut` callbacks (post-attention and post-layer) are what enable
/// activation patching, ablation, and steering.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn run_layer_with_capture_hooked(
    weights: &ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
    capture_activation: bool,
    capture_attention: bool,
    ple_input: Option<&Array2<f32>>,
    shared_kv: Option<&SharedKV>,
    hook: &mut dyn LayerHook,
) -> Option<(
    Array2<f32>,
    Option<Array2<f32>>,
    Option<AttentionWeights>,
    Option<SharedKV>,
)> {
    hook.on_pre_layer(layer, h);

    let (mut h_post_attn, attn_weights, kv_out) = if shared_kv.is_some() {
        let (h_post_attn, attn_weights) =
            run_attention_inner(weights, h, layer, capture_attention, shared_kv)?;
        (h_post_attn, attn_weights, None)
    } else {
        let (h_post_attn, _, attn_weights, k_rope, v_final) =
            crate::attention::run_attention_block_with_kv_out(
                weights,
                h,
                layer,
                capture_attention,
                None,
            )?;
        (h_post_attn, attn_weights, Some((k_rope, v_final)))
    };
    if let Some(ref w) = attn_weights {
        hook.on_attention_weights(layer, w);
    }
    hook.on_post_attention(layer, &mut h_post_attn);

    let (h_post_ffn, activation) = run_ffn(weights, &h_post_attn, layer, ffn, capture_activation);
    if let Some(ref act) = activation {
        hook.on_ffn_activation(layer, act);
    }

    let mut h_out = apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_input);
    apply_layer_scalar(weights, &mut h_out, layer);
    hook.on_post_layer(layer, &mut h_out);

    Some((h_out, activation, attn_weights, kv_out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::test_utils::make_test_weights;
    use crate::ffn::WeightFfn;
    use ndarray::Array2;

    fn h(rows: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (rows, hidden),
            (0..rows * hidden)
                .map(|i| (i as f32 + 1.0) * 0.02)
                .collect(),
        )
        .unwrap()
    }

    #[test]
    fn run_ffn_shape() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(3, weights.hidden_size);
        let (out, act) = run_ffn(&weights, &input, 0, &ffn, false);
        assert_eq!(out.shape(), &[3, weights.hidden_size]);
        assert!(act.is_none(), "capture_activation=false should return None");
    }

    #[test]
    fn run_ffn_captures_activation() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let (_, act) = run_ffn(&weights, &input, 0, &ffn, true);
        let a = act.expect("activation should be captured");
        assert_eq!(a.shape(), &[2, weights.intermediate_size]);
    }

    #[test]
    fn run_ffn_output_finite() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let (out, _) = run_ffn(&weights, &input, 0, &ffn, false);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn run_layer_with_ffn_shape() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(3, weights.hidden_size);
        let (h_out, _act, _kv) = run_layer_with_ffn(&weights, &input, 0, &ffn, false, None, None)
            .expect("run_layer_with_ffn failed");
        assert_eq!(h_out.shape(), &[3, weights.hidden_size]);
    }

    #[test]
    fn run_layer_with_ffn_all_layers() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        for layer in 0..weights.num_layers {
            assert!(
                run_layer_with_ffn(&weights, &input, layer, &ffn, false, None, None).is_some(),
                "layer {layer} failed"
            );
        }
    }
}
