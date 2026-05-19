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
use crate::residual::rms_norm_for_arch;
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

    // Layer-0 (or LARQL_STAGE_DUMP_LAYER) stage dumps — matches the Metal
    // `LARQL_METAL_DUMP_LAYERS` convention. Lets us diff per-stage
    // intermediates between CPU and Metal.
    let dump_cfg = super::dump_config::DumpConfig::get();
    let stage_dump_dir = dump_cfg.stage_dir(layer);
    let dump_f32 = |name: &str, arr: &Array2<f32>| {
        if let Some(dir) = stage_dump_dir {
            let slice = arr.as_slice().unwrap_or(&[]);
            let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = std::fs::write(super::dump_config::cpu_stage_path(dir, name), &bytes);
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
        None => rms_norm_for_arch(h_post_attn, None, norm_offset, &*weights.arch),
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
            None => rms_norm_for_arch(&ffn_out, None, norm_offset, &*weights.arch),
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
/// alternate forward drivers (notably `vindex::predict_kquant`) get the same
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
    if let Some(dir) = crate::forward::dump_config::DumpConfig::get().layer_dir() {
        let slice = h_post_attn.as_slice().unwrap_or(&[]);
        let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
        let path = crate::forward::dump_config::cpu_layer_h_post_attn_path(dir, layer);
        let _ = std::fs::write(&path, &bytes);
    }
    let (h_post_ffn, activation) = run_ffn(weights, &h_post_attn, layer, ffn, capture_activation);
    let mut h_out = apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_input);
    apply_layer_scalar(weights, &mut h_out, layer);
    Some((h_out, activation, kv_out))
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
    use crate::ffn::WeightFfn;
    use crate::test_utils::make_test_weights;
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

    #[test]
    fn run_attention_public_matches_inner() {
        let weights = make_test_weights();
        let input = h(3, weights.hidden_size);
        let pub_out = run_attention_public(&weights, &input, 0).unwrap();
        let inner_out = run_attention(&weights, &input, 0).unwrap();
        assert_eq!(pub_out.shape(), inner_out.shape());
        for (a, b) in pub_out.iter().zip(inner_out.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "public/inner attention differ: {a} vs {b}"
            );
        }
    }

    #[test]
    fn run_attention_inner_with_capture_attention_returns_weights() {
        let weights = make_test_weights();
        let input = h(3, weights.hidden_size);
        let (out, weights_opt) =
            run_attention_inner(&weights, &input, 0, /*capture=*/ true, None).unwrap();
        assert_eq!(out.shape(), &[3, weights.hidden_size]);
        let aw = weights_opt.expect("attention weights should be captured");
        // One distribution per Q-head, each with seq_len=3 entries (last position).
        assert_eq!(aw.heads.len(), weights.num_q_heads);
        for head in &aw.heads {
            assert_eq!(head.len(), 3);
        }
    }

    #[test]
    fn run_attention_with_kv_cache_returns_shared_kv() {
        let weights = make_test_weights();
        let input = h(2, weights.hidden_size);
        let (h_post_attn, (k, v)) =
            run_attention_with_kv_cache(&weights, &input, 0).expect("attn-with-kv must succeed");
        assert_eq!(h_post_attn.shape(), &[2, weights.hidden_size]);
        // K/V have shape (seq, num_kv_heads * head_dim).
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        assert_eq!(k.shape(), &[2, kv_dim]);
        assert_eq!(v.shape(), &[2, kv_dim]);
    }

    #[test]
    fn apply_layer_scalar_is_noop_when_key_absent() {
        // tinymodel arch returns None for layer_scalar_key — the function
        // must leave the input untouched.
        let weights = make_test_weights();
        let mut input = h(2, weights.hidden_size);
        let before = input.clone();
        apply_layer_scalar(&weights, &mut input, 0);
        assert_eq!(input, before);
    }

    #[test]
    fn run_layer_with_capture_returns_attention_weights_on_request() {
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let (h_out, _act, _attn, _kv) = run_layer_with_capture(
            &weights, &input, 0, &ffn, false, /*capture_attention=*/ true, None, None,
        )
        .expect("run_layer_with_capture must succeed");
        assert_eq!(h_out.shape(), &[2, weights.hidden_size]);
    }

    #[test]
    fn run_ffn_with_kv_share_skips_kv_output() {
        // shared_kv = Some ⇒ run_layer_with_ffn takes the inner-attention
        // path which doesn't return KV — exercise that branch.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        // Capture KV from layer 0 first, then re-feed at layer 1 as shared.
        let (_, shared) = run_attention_with_kv_cache(&weights, &input, 0).unwrap();
        let (h_out, _, kv_out) =
            run_layer_with_ffn(&weights, &input, 1, &ffn, false, None, Some(&shared))
                .expect("layer with shared KV must succeed");
        assert_eq!(h_out.shape(), &[2, weights.hidden_size]);
        assert!(kv_out.is_none(), "shared-KV path must not return new KV");
    }

    // ── Gemma3-arch (post-norms branch in run_ffn) ─────────────────────

    #[test]
    fn run_ffn_post_norms_arch_routes_through_post_norm_branch() {
        // Gemma3 has has_post_norms=true → run_ffn takes the
        // pre_feedforward_layernorm + post_feedforward_layernorm path.
        let weights = crate::test_utils::make_gemma3_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let (out, _) = run_ffn(&weights, &input, 0, &ffn, false);
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn run_layer_with_ffn_gemma3_arch_completes_full_layer() {
        // Full layer pass on Gemma3 — exercises post-norm + qk-norm in
        // attention AND post-norm in FFN simultaneously.
        let weights = crate::test_utils::make_gemma3_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let (h_out, _, kv) =
            run_layer_with_ffn(&weights, &input, 0, &ffn, false, None, None).unwrap();
        assert_eq!(h_out.shape(), &[2, weights.hidden_size]);
        assert!(h_out.iter().all(|v| v.is_finite()));
        assert!(kv.is_some());
    }

    // ── Starcoder2-arch (FFN biases) ───────────────────────────────────

    #[test]
    fn run_layer_with_ffn_starcoder2_arch_runs_bias_branches() {
        let weights = crate::test_utils::make_starcoder2_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let (h_out, _, _) =
            run_layer_with_ffn(&weights, &input, 0, &ffn, false, None, None).unwrap();
        assert_eq!(h_out.shape(), &[2, weights.hidden_size]);
        assert!(h_out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn run_layer_with_capture_hooked_shared_kv_branch() {
        // Hooked-capture variant of the shared-KV branch — exercises lines
        // around L247-250 in the `if shared_kv.is_some()` arm.
        let weights = make_test_weights();
        let ffn = WeightFfn { weights: &weights };
        let input = h(2, weights.hidden_size);
        let (_, shared) = run_attention_with_kv_cache(&weights, &input, 0).unwrap();
        let mut hook = crate::forward::hooks::NoopHook;
        let (h_out, _, _, kv_out) = run_layer_with_capture_hooked(
            &weights,
            &input,
            1,
            &ffn,
            false,
            false,
            None,
            Some(&shared),
            &mut hook,
        )
        .expect("hooked shared-KV path must succeed");
        assert_eq!(h_out.shape(), &[2, weights.hidden_size]);
        assert!(kv_out.is_none(), "shared-KV path must not return new KV");
    }
}
