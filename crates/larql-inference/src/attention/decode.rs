//! Decode-step attention — GQA for a single new token against a
//! growing KV cache.
//!
//! Prefill does full O(seq²) attention and returns K/V per layer. Decode
//! runs one token at a time with O(cached_len) attention: Q for the new
//! token attends against [K_cache | K_new] and [V_cache | V_new], with
//! no causal mask needed (the new query is at the end and can see every
//! cached position + itself).
//!
//! The per-layer K/V cache type ([`larql_kv::KvCache`]) and the
//! generation loops that drive prefill→decode now live in `larql-kv`
//! (the canonical engine state shape) — see `larql-kv/src/cache.rs`
//! and `larql-kv/src/generation.rs`.

use ndarray::Array2;

use super::SharedKV;

/// GQA attention for a single decode step.
///
/// `q_new`: `[1, num_q * head_dim]` — Q for the new token only.
/// `k_full`: `[total_len, num_kv * head_dim]` — K_cache concatenated
/// with the new token's K_rope. Same for `v_full`.
///
/// Returns `[1, num_q * head_dim]` attention output for the new token.
/// No causal mask — the new token naturally sees everything, and the
/// cache only grew by 1 at the end.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_decode_step(
    q_new: &Array2<f32>,
    k_full: &Array2<f32>,
    v_full: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    softcap: Option<f32>,
) -> Array2<f32> {
    let total_len = k_full.shape()[0];
    let mut out = Array2::<f32>::zeros((1, num_q * head_dim));
    let scale_f32 = scale as f32;

    let mut scores = vec![0.0f32; total_len];
    for h in 0..num_q {
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        let q_row = q_new.slice(ndarray::s![0, q_off..q_off + head_dim]);
        let k_block = k_full.slice(ndarray::s![.., kv_off..kv_off + head_dim]);
        let raw: ndarray::Array1<f32> = k_block.dot(&q_row);
        for i in 0..total_len {
            let mut s = raw[i] * scale_f32;
            if let Some(cap) = softcap {
                s = (s / cap).tanh() * cap;
            }
            scores[i] = s;
        }
        // Softmax
        let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f64;
        for s in scores.iter_mut() {
            let e = ((*s - max_val) as f64).exp();
            *s = e as f32;
            sum += e;
        }
        let inv_sum = (1.0 / sum) as f32;
        for s in scores.iter_mut() {
            *s *= inv_sum;
        }
        // Weighted sum of V
        let v_block = v_full.slice(ndarray::s![.., kv_off..kv_off + head_dim]);
        let scores_view = ndarray::ArrayView1::from(&scores[..]);
        let weighted_v = v_block.t().dot(&scores_view);
        for d in 0..head_dim {
            out[[0, q_off + d]] = weighted_v[d];
        }
    }
    out
}

/// Run the attention block for one decode step using an incremental KV
/// cache. `h_new` is the `[1, hidden]` residual for the new token.
/// `kv_entry` is the layer's existing `(K_cache, V_cache)` or `None` on
/// first step. `abs_position` is the new token's absolute RoPE
/// position — the caller must pass its true position in the original
/// sequence, NOT the clipped cache length (those differ under a
/// sliding window). Returns the updated `(h_post_attn, new_kv)`.
///
/// CPU-only variant. For GPU projections use
/// [`run_attention_block_decode_step_backend`].
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn run_attention_block_decode_step(
    weights: &crate::model::ModelWeights,
    h_new: &Array2<f32>,
    layer: usize,
    kv_entry: Option<&SharedKV>,
    abs_position: usize,
) -> Option<(Array2<f32>, SharedKV)> {
    run_attention_block_decode_step_backend(weights, h_new, layer, kv_entry, abs_position, None)
}

/// Decode-step attention with optional GPU-accelerated projections
/// (Q/K/V/O matmuls route through `ComputeBackend::matmul_transb` when
/// `backend` is `Some`). GQA softmax + weighted-V stays on CPU —
/// that's O(cached_len × head_dim × num_q) per step and rarely the
/// bottleneck vs the hidden×hidden projection gemms.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn run_attention_block_decode_step_backend(
    weights: &crate::model::ModelWeights,
    h_new: &Array2<f32>,
    layer: usize,
    kv_entry: Option<&SharedKV>,
    abs_position: usize,
    backend: Option<&dyn larql_compute::ComputeBackend>,
) -> Option<(Array2<f32>, SharedKV)> {
    use crate::forward::add_bias;
    use crate::residual::{rms_norm_heads, rms_norm_heads_no_weight};
    use larql_compute::dot_proj_gpu;

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
    let norm_offset = arch.norm_weight_offset();
    let position = abs_position;

    let h_norm = crate::forward::apply_norm(
        weights,
        h_new,
        &arch.input_layernorm_key(layer),
        norm_offset,
    );

    let w_q = weights.tensors.get(&arch.attn_q_key(layer))?;
    let w_o = weights.tensors.get(&arch.attn_o_key(layer))?;
    let mut q_full = dot_proj_gpu(&h_norm, w_q, backend);
    if let Some(bias) = arch
        .attn_q_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut q_full, bias);
    }

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
        position,
        pos_divisor,
        llama3,
    );

    // New token's K, V — RoPE'd at `position`, then appended to cache.
    let w_k = weights.tensors.get(&arch.attn_k_key(layer))?;
    let v_from_k = !weights.tensors.contains_key(&arch.attn_v_key(layer));
    let w_v = if v_from_k {
        w_k
    } else {
        weights.tensors.get(&arch.attn_v_key(layer))?
    };

    let mut k_full_new = dot_proj_gpu(&h_norm, w_k, backend);
    let mut v_full_new = dot_proj_gpu(&h_norm, w_v, backend);
    if let Some(bias) = arch
        .attn_k_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut k_full_new, bias);
    }
    if let Some(bias) = arch
        .attn_v_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut v_full_new, bias);
    }
    if arch.has_v_norm() {
        v_full_new = rms_norm_heads_no_weight(&v_full_new, num_kv, head_dim);
    }
    let k_normed = match arch
        .attn_k_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&k_full_new, norm_w, num_kv, head_dim, qk_norm_off),
        None => k_full_new,
    };
    let k_new_rope = crate::attention::rope::apply_rope_partial_at_full(
        &k_normed,
        num_kv,
        head_dim,
        layer_rope_base,
        rotary_frac,
        position,
        pos_divisor,
        llama3,
    );

    // Concatenate cache + new along seq axis.
    let (k_concat, v_concat) = match kv_entry {
        Some((k_cached, v_cached)) => {
            let kv_dim = num_kv * head_dim;
            let total = k_cached.shape()[0] + 1;
            let mut k_out = Array2::<f32>::zeros((total, kv_dim));
            let mut v_out = Array2::<f32>::zeros((total, kv_dim));
            k_out
                .slice_mut(ndarray::s![..k_cached.shape()[0], ..])
                .assign(k_cached);
            v_out
                .slice_mut(ndarray::s![..v_cached.shape()[0], ..])
                .assign(v_cached);
            k_out
                .slice_mut(ndarray::s![k_cached.shape()[0].., ..])
                .assign(&k_new_rope);
            v_out
                .slice_mut(ndarray::s![v_cached.shape()[0].., ..])
                .assign(&v_full_new);
            (k_out, v_out)
        }
        None => (k_new_rope, v_full_new),
    };

    let softcap = arch.attn_logit_softcapping();
    let attn_out = gqa_attention_decode_step(
        &q_rope, &k_concat, &v_concat, num_q, head_dim, reps, scale, softcap,
    );

    let mut attn_projected = dot_proj_gpu(&attn_out, w_o, backend);
    if let Some(bias) = arch
        .attn_o_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut attn_projected, bias);
    }

    let res_mult = arch.residual_multiplier();
    let h_post_attn = if arch.has_post_norms() {
        let normed = crate::forward::apply_norm(
            weights,
            &attn_projected,
            &arch.post_attention_layernorm_key(layer),
            norm_offset,
        );
        if res_mult != 1.0 {
            h_new + &(&normed * res_mult)
        } else {
            h_new + &normed
        }
    } else if res_mult != 1.0 {
        h_new + &(&attn_projected * res_mult)
    } else {
        h_new + &attn_projected
    };

    Some((h_post_attn, (k_concat, v_concat)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::make_test_weights;
    use ndarray::Array2;

    #[test]
    fn decode_step_output_shape() {
        let weights = make_test_weights();
        let h = Array2::from_elem((1, weights.hidden_size), 0.1f32);
        let (h_out, (k, v)) =
            run_attention_block_decode_step(&weights, &h, 0, None, 0).expect("decode_step failed");
        assert_eq!(h_out.shape(), &[1, weights.hidden_size]);
        assert_eq!(k.shape()[0], 1, "K should have 1 new row");
        assert_eq!(v.shape()[0], 1, "V should have 1 new row");
    }

    #[test]
    fn decode_step_output_finite() {
        let weights = make_test_weights();
        let h = Array2::from_elem((1, weights.hidden_size), 0.5f32);
        let (h_out, _) =
            run_attention_block_decode_step(&weights, &h, 0, None, 0).expect("decode_step failed");
        assert!(h_out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn decode_step_kv_grows_with_prior() {
        let weights = make_test_weights();
        let h = Array2::from_elem((1, weights.hidden_size), 0.1f32);
        let (_, kv1) = run_attention_block_decode_step(&weights, &h, 0, None, 0).unwrap();
        assert_eq!(kv1.0.shape()[0], 1);
        let (_, kv2) = run_attention_block_decode_step(&weights, &h, 0, Some(&kv1), 1).unwrap();
        assert_eq!(kv2.0.shape()[0], 2, "K should grow by 1 per step");
    }

    #[test]
    fn decode_step_all_layers_succeed() {
        let weights = make_test_weights();
        let h = Array2::from_elem((1, weights.hidden_size), 0.3f32);
        for layer in 0..weights.num_layers {
            let result = run_attention_block_decode_step(&weights, &h, layer, None, 0);
            assert!(result.is_some(), "layer {layer} decode step failed");
        }
    }
}
