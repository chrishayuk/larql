//! CPU attention block — full layer attention computation.
//!
//! norm → Q/K/V projection → bias → V-norm → QK-norm → RoPE → GQA → O projection → residual.
//! Supports KV sharing (reuse K/V from a source layer).

use super::gqa::{
    gqa_attention_with_all_weights, gqa_attention_with_weights, gqa_reduced_qk_all_weights,
};
use super::{AttentionAllWeights, AttentionWeights, SharedKV};
use ndarray::{s, Array2};

/// Run the full attention block. Returns (h_post_attn, attn_projected, optional_weights).
#[allow(clippy::too_many_arguments)]
pub fn run_attention_block(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
) -> Option<(Array2<f32>, Array2<f32>, Option<AttentionWeights>)> {
    run_attention_block_shared(weights, h, layer, capture_attention, None)
}

/// Run attention with optional shared K/V, returning K/V for caching.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn run_attention_block_with_kv_out(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
    shared_kv: Option<&SharedKV>,
) -> Option<(
    Array2<f32>,
    Array2<f32>,
    Option<AttentionWeights>,
    Array2<f32>,
    Array2<f32>,
)> {
    let (h_post, attn_proj, attn_w, k, v, _pre_o, _) = run_attention_block_core(
        weights,
        h,
        layer,
        capture_attention,
        shared_kv,
        None,
        None,
        None,
        None,
        false,
        None,
    )?;
    Some((h_post, attn_proj, attn_w, k, v))
}

/// Run attention with optional shared K/V (discards K/V output).
#[allow(clippy::too_many_arguments)]
pub fn run_attention_block_shared(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Array2<f32>, Option<AttentionWeights>)> {
    let (h_post, attn_proj, attn_w, _, _, _, _) = run_attention_block_core(
        weights,
        h,
        layer,
        capture_attention,
        shared_kv,
        None,
        None,
        None,
        None,
        false,
        None,
    )?;
    Some((h_post, attn_proj, attn_w))
}

/// Run attention, returning the pre-O-projection output per head.
/// Returns `(h_post_attn, pre_o)` where `pre_o` has shape `[seq, num_q * head_dim]`.
/// This is the equivalent of Python's `o_proj.register_forward_pre_hook`.
pub fn run_attention_block_with_pre_o(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
) -> Option<(Array2<f32>, Array2<f32>)> {
    let (h_post, _, _, _, _, pre_o, _) = run_attention_block_core(
        weights, h, layer, false, None, None, None, None, None, false, None,
    )?;
    Some((h_post, pre_o))
}

/// Run attention with optional shared K/V and return the pre-O-projection
/// output per query head.
///
/// This is the shared-KV-safe variant used by research/intervention adapters
/// that need to inspect a pre-W_O head before deciding how to replace it.
pub fn run_attention_block_shared_with_pre_o(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Array2<f32>)> {
    let (h_post, _, _, _, _, pre_o, _) = run_attention_block_core(
        weights, h, layer, false, shared_kv, None, None, None, None, false, None,
    )?;
    Some((h_post, pre_o))
}

/// Run attention with optional shared K/V and return both the pre-O output and
/// all per-query-position attention distributions.
///
/// This is a diagnostic surface for relation/address probes. It is separate
/// from normal attention capture because all-position weights are
/// O(heads * seq^2) memory.
pub fn run_attention_block_with_pre_o_and_all_attention_weights(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Array2<f32>, AttentionAllWeights)> {
    let (h_post, _, _, _, _, pre_o, all_weights) = run_attention_block_core(
        weights, h, layer, false, shared_kv, None, None, None, None, true, None,
    )?;
    Some((h_post, pre_o, all_weights?))
}

/// Run attention with optional shared K/V and return the pre-O output plus
/// all-position attention distributions computed from a reduced QK dot product.
///
/// The real attention output remains full-rank. Only the diagnostic attention
/// weights use `qk_rank`, so this can test reduced address computation without
/// changing the model forward path.
pub fn run_attention_block_with_pre_o_and_reduced_qk_attention_weights(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    shared_kv: Option<&SharedKV>,
    qk_rank: usize,
) -> Option<(Array2<f32>, Array2<f32>, AttentionAllWeights)> {
    let (h_post, _, _, _, _, pre_o, all_weights) = run_attention_block_core(
        weights,
        h,
        layer,
        false,
        shared_kv,
        None,
        None,
        None,
        None,
        false,
        Some(qk_rank),
    )?;
    Some((h_post, pre_o, all_weights?))
}

/// Run attention while zeroing selected pre-O-projection query heads before W_O.
///
/// Returns the post-attention residual and, when K/V were computed by this call,
/// the K/V pair for cross-layer sharing.
pub fn run_attention_block_zero_pre_o_heads(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    heads: &[usize],
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Option<SharedKV>)> {
    let (h_post, _, _, k_rope, v_final, _, _) = run_attention_block_core(
        weights,
        h,
        layer,
        false,
        shared_kv,
        Some(heads),
        None,
        None,
        None,
        false,
        None,
    )?;
    let kv_out = if shared_kv.is_none() {
        Some((k_rope, v_final))
    } else {
        None
    };
    Some((h_post, kv_out))
}

/// Run attention while replacing one pre-O-projection query head before W_O.
///
/// `replacement` must have shape `[seq_len, head_dim]`.
pub fn run_attention_block_replace_pre_o_head(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    head: usize,
    replacement: &Array2<f32>,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Option<SharedKV>)> {
    let (h_post, _, _, k_rope, v_final, _, _) = run_attention_block_core(
        weights,
        h,
        layer,
        false,
        shared_kv,
        None,
        Some((head, replacement)),
        None,
        None,
        false,
        None,
    )?;
    let kv_out = if shared_kv.is_none() {
        Some((k_rope, v_final))
    } else {
        None
    };
    Some((h_post, kv_out))
}

/// Run attention while explicitly subtracting selected query-head
/// contributions from the O-projected tensor before the attention residual path.
///
/// This is numerically equivalent to zeroing those pre-W_O heads, but it checks
/// the head-to-W_O block indexing independently.
pub fn run_attention_block_subtract_pre_o_heads(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    heads: &[usize],
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Option<SharedKV>)> {
    let (h_post, _, _, k_rope, v_final, _, _) = run_attention_block_core(
        weights,
        h,
        layer,
        false,
        shared_kv,
        None,
        None,
        Some(heads),
        None,
        false,
        None,
    )?;
    let kv_out = if shared_kv.is_none() {
        Some((k_rope, v_final))
    } else {
        None
    };
    Some((h_post, kv_out))
}

/// Run attention while replacing one query-head residual-space contribution
/// after W_O projection and before the attention residual path.
///
/// `replacement_delta` must have shape `[seq_len, hidden_size]` and represents
/// the residual-space contribution that should replace `W_O^head y_head`.
/// This is the Mode D validation surface: runtime lookup/add tables can bypass
/// W_O entirely while the rest of the layer remains unchanged.
pub fn run_attention_block_replace_head_residual_delta(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    head: usize,
    replacement_delta: &Array2<f32>,
    shared_kv: Option<&SharedKV>,
) -> Option<(Array2<f32>, Option<SharedKV>)> {
    let (h_post, _, _, k_rope, v_final, _, _) = run_attention_block_core(
        weights,
        h,
        layer,
        false,
        shared_kv,
        None,
        None,
        None,
        Some((head, replacement_delta)),
        false,
        None,
    )?;
    let kv_out = if shared_kv.is_none() {
        Some((k_rope, v_final))
    } else {
        None
    };
    Some((h_post, kv_out))
}

/// Core attention block implementation.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
fn run_attention_block_core(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
    shared_kv: Option<&SharedKV>,
    zero_pre_o_heads: Option<&[usize]>,
    replace_pre_o_head: Option<(usize, &Array2<f32>)>,
    subtract_pre_o_heads: Option<&[usize]>,
    replace_head_residual_delta: Option<(usize, &Array2<f32>)>,
    capture_all_attention: bool,
    reduced_qk_rank: Option<usize>,
) -> Option<(
    Array2<f32>,
    Array2<f32>,
    Option<AttentionWeights>,
    Array2<f32>,
    Array2<f32>,
    Array2<f32>,
    Option<AttentionAllWeights>,
)> {
    use crate::forward::{add_bias, dot_proj};
    use crate::residual::{rms_norm_heads, rms_norm_heads_no_weight};

    let arch = &*weights.arch;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_q = arch.num_q_heads_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let reps = num_q / num_kv;
    let scale = if arch.attention_multiplier() != 1.0 {
        arch.attention_multiplier() as f64
    } else {
        arch.attention_scale_for_layer(layer)
    };
    let seq_len = h.shape()[0];
    let norm_offset = arch.norm_weight_offset();

    // Per-layer stage dumps, paired with Metal via LARQL_CPU_STAGE_DUMP=<dir>.
    // Default is layer 0 (noise budget); set LARQL_STAGE_DUMP_LAYER=<N> to
    // capture a specific layer instead — Gemma 4 global layers (5, 11, …)
    // are useful for bisecting partial-RoPE / V-norm interactions.
    let dump_cfg = crate::forward::dump_config::DumpConfig::get();
    let stage_dump = dump_cfg.stage_dir(layer);
    let dump_f32 = |name: &str, arr: &Array2<f32>| {
        if let Some(dir) = stage_dump {
            let slice = arr.as_slice().unwrap_or(&[]);
            let bytes: Vec<u8> = slice.iter().flat_map(|v| v.to_le_bytes()).collect();
            let _ = std::fs::write(format!("{dir}/cpu_L0_{name}.f32"), &bytes);
        }
    };

    // Input norm
    let h_norm =
        crate::forward::apply_norm(weights, h, &arch.input_layernorm_key(layer), norm_offset);
    dump_f32("norm_out", &h_norm);

    // Q projection (always from current hidden state)
    let w_q = weights.tensors.get(&arch.attn_q_key(layer))?;
    let w_o = weights.tensors.get(&arch.attn_o_key(layer)).unwrap();
    let mut q_full = dot_proj(&h_norm, w_q);
    if let Some(bias) = arch
        .attn_q_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut q_full, bias);
    }
    dump_f32("q_out_raw", &q_full);

    // QK norm on Q
    let qk_offset = weights.arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 {
        qk_offset
    } else {
        norm_offset
    };
    let q_normed = match arch
        .attn_q_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&q_full, norm_w, num_q, head_dim, qk_norm_off),
        None => q_full,
    };
    dump_f32("q_out_after_qk_norm", &q_normed);

    // RoPE on Q
    let layer_rope_base = crate::forward_overrides::effective_rope_base_for_layer(arch, layer);
    let rotary_frac = arch.rotary_fraction_for_layer(layer);
    let pos_divisor =
        crate::forward_overrides::effective_rope_position_divisor_for_layer(arch, layer);
    let llama3 = crate::forward_overrides::effective_llama3_rope_scaling(arch);
    let q_rope = crate::attention::rope::apply_rope_partial_at_full(
        &q_normed,
        num_q,
        head_dim,
        layer_rope_base,
        rotary_frac,
        0,
        pos_divisor,
        llama3,
    );

    // K/V: either from shared cache or computed fresh
    let (k_rope, v_final) = if let Some((cached_k, cached_v)) = shared_kv {
        (cached_k.clone(), cached_v.clone())
    } else {
        let w_k = weights.tensors.get(&arch.attn_k_key(layer)).unwrap();

        let mut k_full = dot_proj(&h_norm, w_k);
        if let Some(bias) = arch
            .attn_k_bias_key(layer)
            .and_then(|k| weights.vectors.get(&k))
        {
            add_bias(&mut k_full, bias);
        }

        let k_normed = match arch
            .attn_k_norm_key(layer)
            .and_then(|k| weights.vectors.get(&k))
        {
            Some(norm_w) => rms_norm_heads(&k_full, norm_w, num_kv, head_dim, qk_norm_off),
            None => k_full.clone(),
        };

        // V projection. Always go through the stored W_v tensor when it
        // exists — including on `attention_k_eq_v` (Gemma 4 global) layers
        // where the bytes in W_v were derived from W_k at extraction time.
        // The reason: the vindex re-quantises V as Q6_K while K stays Q4_K
        // (see `format/weights/write.rs`: `is_v { quantize_q6_k } else {
        // quantize_q4_k }`), so `Q6_K_dequant(K_bytes)` is numerically
        // closer to the original bf16 weight than `Q4_K_dequant(K_bytes)`.
        // Metal's V projection uses the Q6_K path; the old CPU shortcut
        // (`v = k_full`) was ~0.25 off per element on Gemma 4 31B L5+,
        // which is what L5's attn_out drift was tracking.
        //
        // Fallback: when W_v is genuinely absent from the vindex (older
        // extracts with no v_proj tensor for `attention_k_eq_v` layers),
        // reuse `k_full` — matches pre-Q6K-V behaviour.
        let v_full = if let Some(w_v) = weights.tensors.get(&arch.attn_v_key(layer)) {
            let mut v = dot_proj(&h_norm, w_v);
            if let Some(bias) = arch
                .attn_v_bias_key(layer)
                .and_then(|k| weights.vectors.get(&k))
            {
                add_bias(&mut v, bias);
            }
            if arch.has_v_norm() {
                v = rms_norm_heads_no_weight(&v, num_kv, head_dim);
            }
            v
        } else if arch.has_v_norm() {
            rms_norm_heads_no_weight(&k_full, num_kv, head_dim)
        } else {
            k_full.clone()
        };

        let k_r = crate::attention::rope::apply_rope_partial_at_full(
            &k_normed,
            num_kv,
            head_dim,
            layer_rope_base,
            rotary_frac,
            0,
            pos_divisor,
            llama3,
        );
        (k_r, v_full)
    };

    dump_f32("q_out_after_rope", &q_rope);
    dump_f32("k_out_after_rope", &k_rope);
    dump_f32("v_out", &v_final);

    // GQA attention
    let softcap = arch.attn_logit_softcapping();
    let reduced_qk_weights = reduced_qk_rank.map(|rank| {
        gqa_reduced_qk_all_weights(
            &q_rope, &k_rope, num_q, head_dim, reps, scale, seq_len, softcap, rank,
        )
    });
    let (mut attn_out, attn_weights, full_all_attn_weights) = if capture_all_attention {
        let (out, all_weights) = gqa_attention_with_all_weights(
            &q_rope, &k_rope, &v_final, num_q, head_dim, reps, scale, seq_len, softcap,
        );
        (out, None, Some(all_weights))
    } else {
        let (out, weights) = gqa_attention_with_weights(
            &q_rope,
            &k_rope,
            &v_final,
            num_q,
            head_dim,
            reps,
            scale,
            seq_len,
            capture_attention,
            softcap,
        );
        (out, weights, None)
    };
    let all_attn_weights = reduced_qk_weights.or(full_all_attn_weights);
    if let Some(heads) = zero_pre_o_heads {
        for &head in heads {
            if head >= num_q {
                return None;
            }
            let start = head * head_dim;
            let end = start + head_dim;
            attn_out.slice_mut(s![.., start..end]).fill(0.0);
        }
    }
    if let Some((head, replacement)) = replace_pre_o_head {
        if head >= num_q || replacement.nrows() != seq_len || replacement.ncols() != head_dim {
            return None;
        }
        let start = head * head_dim;
        let end = start + head_dim;
        attn_out
            .slice_mut(s![.., start..end])
            .assign(&replacement.view());
    }
    dump_f32("attn_out", &attn_out);

    // O projection
    let mut attn_projected = dot_proj(&attn_out, w_o);
    if let Some(heads) = subtract_pre_o_heads {
        for &head in heads {
            if head >= num_q {
                return None;
            }
            let start = head * head_dim;
            let end = start + head_dim;
            let head_out = attn_out.slice(s![.., start..end]);
            let w_o_head = w_o.slice(s![.., start..end]);
            let contribution = dot_proj(&head_out, &w_o_head);
            attn_projected -= &contribution;
        }
    }
    if let Some((head, replacement_delta)) = replace_head_residual_delta {
        if head >= num_q
            || replacement_delta.nrows() != seq_len
            || replacement_delta.ncols() != weights.hidden_size
        {
            return None;
        }
        let start = head * head_dim;
        let end = start + head_dim;
        let head_out = attn_out.slice(s![.., start..end]);
        let w_o_head = w_o.slice(s![.., start..end]);
        let original_contribution = dot_proj(&head_out, &w_o_head);
        attn_projected -= &original_contribution;
        attn_projected += replacement_delta;
    }
    if let Some(bias) = arch
        .attn_o_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut attn_projected, bias);
    }
    dump_f32("o_out", &attn_projected);

    // Residual connection
    let res_mult = arch.residual_multiplier();
    let h_post_attn = if arch.has_post_norms() {
        let normed = crate::forward::apply_norm(
            weights,
            &attn_projected,
            &arch.post_attention_layernorm_key(layer),
            norm_offset,
        );
        if res_mult != 1.0 {
            h + &(&normed * res_mult)
        } else {
            h + &normed
        }
    } else if res_mult != 1.0 {
        h + &(&attn_projected * res_mult)
    } else {
        h + &attn_projected
    };

    Some((
        h_post_attn,
        attn_projected,
        attn_weights,
        k_rope,
        v_final,
        attn_out,
        all_attn_weights,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::make_test_weights;
    use ndarray::Array2;

    fn hidden(rows: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (rows, hidden),
            (0..rows * hidden)
                .map(|i| (i as f32 + 1.0) * 0.01)
                .collect(),
        )
        .unwrap()
    }

    // run_attention_block returns (h_post_attn, attn_proj, attn_weights)
    // — the second element is the projected attention output, not K/V.

    #[test]
    fn attention_block_output_shape() {
        let weights = make_test_weights();
        let h = hidden(3, weights.hidden_size);
        let (h_out, attn_proj, _) =
            run_attention_block(&weights, &h, 0, false).expect("run_attention_block failed");
        assert_eq!(h_out.shape(), &[3, weights.hidden_size]);
        assert_eq!(attn_proj.shape()[0], 3);
    }

    #[test]
    fn attention_block_output_finite() {
        let weights = make_test_weights();
        let h = hidden(2, weights.hidden_size);
        let (h_out, _, _) = run_attention_block(&weights, &h, 0, false).unwrap();
        assert!(h_out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn attention_block_single_token() {
        let weights = make_test_weights();
        let h = hidden(1, weights.hidden_size);
        let (h_out, attn_proj, _) = run_attention_block(&weights, &h, 0, false).unwrap();
        assert_eq!(h_out.shape(), &[1, weights.hidden_size]);
        assert_eq!(attn_proj.shape()[0], 1);
    }

    #[test]
    fn attention_block_all_layers() {
        let weights = make_test_weights();
        let h = hidden(2, weights.hidden_size);
        for layer in 0..weights.num_layers {
            assert!(
                run_attention_block(&weights, &h, layer, false).is_some(),
                "layer {layer} failed"
            );
        }
    }

    #[test]
    fn attention_block_with_kv_out_returns_kv() {
        let weights = make_test_weights();
        let h = hidden(3, weights.hidden_size);
        let result = run_attention_block_with_kv_out(&weights, &h, 0, false, None);
        // Returns (h_post, attn_proj, attn_w, k_rope, v_final) — 5 elements
        let (h_out, _attn_proj, _attn_w, k_rope, v_final) = result.unwrap();
        assert_eq!(h_out.shape(), &[3, weights.hidden_size]);
        assert_eq!(k_rope.shape()[0], 3);
        assert_eq!(v_final.shape()[0], 3);
    }

    #[test]
    fn attention_block_capture_returns_per_head_weights() {
        let weights = make_test_weights();
        let h = hidden(3, weights.hidden_size);
        let (_, _, attn_w) = run_attention_block(&weights, &h, 0, true).unwrap();
        let aw = attn_w.expect("capture=true must yield weights");
        assert_eq!(aw.heads.len(), weights.num_q_heads);
    }

    #[test]
    fn attention_block_with_pre_o_returns_per_head_pre_projection() {
        let weights = make_test_weights();
        let h = hidden(2, weights.hidden_size);
        let (h_post, pre_o) = run_attention_block_with_pre_o(&weights, &h, 0).unwrap();
        assert_eq!(h_post.shape(), &[2, weights.hidden_size]);
        // pre_o is `[seq, num_q * head_dim]`.
        assert_eq!(pre_o.shape(), &[2, weights.num_q_heads * weights.head_dim]);
    }

    #[test]
    fn attention_block_shared_with_pre_o_works_under_kv_share() {
        let weights = make_test_weights();
        let h = hidden(2, weights.hidden_size);
        let (_, shared) =
            crate::attention::run_attention_block_with_kv_out(&weights, &h, 0, false, None)
                .map(|(p, _, _, k, v)| (p, (k, v)))
                .unwrap();
        let (h_post, pre_o) =
            run_attention_block_shared_with_pre_o(&weights, &h, 1, Some(&shared)).unwrap();
        assert_eq!(h_post.shape(), &[2, weights.hidden_size]);
        assert_eq!(pre_o.shape()[1], weights.num_q_heads * weights.head_dim);
    }

    #[test]
    fn attention_block_with_all_attention_weights_returns_per_position_dist() {
        let weights = make_test_weights();
        let h = hidden(3, weights.hidden_size);
        let (_, _, all) =
            run_attention_block_with_pre_o_and_all_attention_weights(&weights, &h, 0, None)
                .unwrap();
        assert_eq!(all.heads.len(), weights.num_q_heads);
        for head in &all.heads {
            assert_eq!(head.len(), 3); // one distribution per Q position
        }
    }

    #[test]
    fn attention_block_with_reduced_qk_attention_weights_clamps_rank() {
        let weights = make_test_weights();
        let h = hidden(2, weights.hidden_size);
        let (_, _, all) = run_attention_block_with_pre_o_and_reduced_qk_attention_weights(
            &weights, &h, 0, None, /*qk_rank=*/ 4, // half of head_dim=8
        )
        .unwrap();
        assert_eq!(all.heads.len(), weights.num_q_heads);
    }

    // ── Intervention surfaces ──────────────────────────────────────────

    #[test]
    fn zero_pre_o_heads_changes_output_when_head_active() {
        let weights = make_test_weights();
        let h = hidden(2, weights.hidden_size);
        let baseline = run_attention_block(&weights, &h, 0, false).unwrap().0;
        let (zeroed, kv_out) =
            run_attention_block_zero_pre_o_heads(&weights, &h, 0, &[0], None).unwrap();
        assert_eq!(zeroed.shape(), baseline.shape());
        // KV is computed when shared_kv is None.
        assert!(kv_out.is_some());
        let mut max_diff = 0.0f32;
        for (a, b) in baseline.iter().zip(zeroed.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(max_diff > 1e-5, "zeroing head 0 should change output");
    }

    #[test]
    fn zero_pre_o_heads_under_shared_kv_omits_kv_output() {
        let weights = make_test_weights();
        let h = hidden(2, weights.hidden_size);
        let (_, shared) =
            crate::attention::run_attention_block_with_kv_out(&weights, &h, 0, false, None)
                .map(|(p, _, _, k, v)| (p, (k, v)))
                .unwrap();
        let (_, kv_out) =
            run_attention_block_zero_pre_o_heads(&weights, &h, 1, &[0], Some(&shared)).unwrap();
        assert!(kv_out.is_none(), "shared-KV path must not return KV");
    }

    #[test]
    fn replace_pre_o_head_substitutes_one_head() {
        let weights = make_test_weights();
        let h = hidden(2, weights.hidden_size);
        // Replacement is `[seq, head_dim]` of all-zeros — equivalent to
        // zeroing that head, so output must differ from baseline.
        let baseline = run_attention_block(&weights, &h, 0, false).unwrap().0;
        let zero_head = Array2::<f32>::zeros((2, weights.head_dim));
        let (replaced, _) =
            run_attention_block_replace_pre_o_head(&weights, &h, 0, 0, &zero_head, None).unwrap();
        let mut max_diff = 0.0f32;
        for (a, b) in baseline.iter().zip(replaced.iter()) {
            max_diff = max_diff.max((a - b).abs());
        }
        assert!(max_diff > 1e-5, "head replacement should change output");
    }

    #[test]
    fn subtract_pre_o_heads_matches_zero_pre_o_heads_numerically() {
        // Both paths zero the head's W_O contribution — output must match.
        let weights = make_test_weights();
        let h = hidden(2, weights.hidden_size);
        let (zeroed, _) =
            run_attention_block_zero_pre_o_heads(&weights, &h, 0, &[0], None).unwrap();
        let (subtracted, _) =
            run_attention_block_subtract_pre_o_heads(&weights, &h, 0, &[0], None).unwrap();
        for (a, b) in zeroed.iter().zip(subtracted.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "zero-pre-O and subtract-pre-O must match: {a} vs {b}"
            );
        }
    }

    #[test]
    fn replace_head_residual_delta_zero_replacement_matches_zero_head() {
        // A zero residual-space replacement should match the zero-pre-O path
        // up to W_O numerical noise (both eliminate the head contribution).
        let weights = make_test_weights();
        let h = hidden(2, weights.hidden_size);
        let zero_delta = Array2::<f32>::zeros((2, weights.hidden_size));
        let (with_delta, _) =
            run_attention_block_replace_head_residual_delta(&weights, &h, 0, 0, &zero_delta, None)
                .unwrap();
        let (zeroed, _) =
            run_attention_block_zero_pre_o_heads(&weights, &h, 0, &[0], None).unwrap();
        for (a, b) in with_delta.iter().zip(zeroed.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "zero residual delta should match zero-head: {a} vs {b}"
            );
        }
    }

    #[test]
    fn intervention_paths_return_none_for_missing_layer() {
        // Layer index past num_layers — every variant must return None
        // gracefully rather than panic.
        let weights = make_test_weights();
        let h = hidden(2, weights.hidden_size);
        let bogus_layer = weights.num_layers + 5;
        assert!(run_attention_block(&weights, &h, bogus_layer, false).is_none());
        assert!(run_attention_block_with_pre_o(&weights, &h, bogus_layer).is_none());
        assert!(
            run_attention_block_zero_pre_o_heads(&weights, &h, bogus_layer, &[0], None).is_none()
        );
    }

    // ── Gemma3-arch fixture (post-norms, QK norm, gelu_tanh) ───────────

    #[test]
    fn attention_block_with_qk_norm_keys_routes_through_qk_norm_branch() {
        // Gemma3 returns Some from attn_q_norm_key/attn_k_norm_key, hitting
        // the rms_norm_heads branch in run_attention_block_core that
        // tinymodel never exercises.
        let weights = crate::test_utils::make_gemma3_test_weights();
        let h = hidden(2, weights.hidden_size);
        let (h_post, _, _) = run_attention_block(&weights, &h, 0, false).unwrap();
        assert_eq!(h_post.shape(), &[2, weights.hidden_size]);
        assert!(h_post.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn attention_block_post_norms_arch_runs_through_post_norm_branch() {
        // Gemma3 has has_post_norms=true, so the post-attention path
        // takes a different branch in run_attention_block_core.
        let weights = crate::test_utils::make_gemma3_test_weights();
        let h = hidden(2, weights.hidden_size);
        let (h_post, _, _) = run_attention_block(&weights, &h, 1, false).unwrap();
        assert_eq!(h_post.shape(), &[2, weights.hidden_size]);
        assert!(h_post.iter().all(|v| v.is_finite()));
    }

    // ── Starcoder2-arch fixture (attention + FFN biases) ───────────────

    #[test]
    fn attention_block_with_q_k_v_o_biases_runs_add_bias_branches() {
        // Starcoder2 returns Some from every attn_*_bias_key, so every
        // `add_bias` call site fires.
        let weights = crate::test_utils::make_starcoder2_test_weights();
        let h = hidden(2, weights.hidden_size);
        let (h_post, _, _) = run_attention_block(&weights, &h, 0, false).unwrap();
        assert_eq!(h_post.shape(), &[2, weights.hidden_size]);
        assert!(h_post.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn attention_block_starcoder_with_kv_out_returns_finite_kv() {
        let weights = crate::test_utils::make_starcoder2_test_weights();
        let h = hidden(3, weights.hidden_size);
        let (_, _, _, k, v) =
            run_attention_block_with_kv_out(&weights, &h, 0, false, None).unwrap();
        assert_eq!(k.shape()[0], 3);
        assert_eq!(v.shape()[0], 3);
        assert!(k.iter().all(|x| x.is_finite()));
        assert!(v.iter().all(|x| x.is_finite()));
    }
}
