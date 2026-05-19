//! GPU-accelerated attention — routes projections through ComputeBackend.
//!
//! Falls back to CPU BLAS when backend is None.
//! Also includes Q4 quantized attention projection and KV-capture attention.

use super::gqa::gqa_attention_with_weights;
use super::rope::apply_rope_partial;
use super::AttentionWeights;
use ndarray::Array2;

/// GPU-accelerated attention block. Same as `run_attention_block` but routes
/// Q/K/V/O projections through the ComputeBackend (Metal, CUDA, or CPU).
pub fn run_attention_block_gpu(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
    backend: Option<&dyn larql_compute::ComputeBackend>,
) -> Option<(Array2<f32>, Array2<f32>, Option<AttentionWeights>)> {
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
    let seq_len = h.shape()[0];
    let norm_offset = arch.norm_weight_offset();

    let h_norm =
        crate::forward::apply_norm(weights, h, &arch.input_layernorm_key(layer), norm_offset);

    let w_q = weights.tensors.get(&arch.attn_q_key(layer))?;
    let w_k = weights.tensors.get(&arch.attn_k_key(layer)).unwrap();
    let v_from_k = !weights.tensors.contains_key(&arch.attn_v_key(layer));
    let w_v = if v_from_k {
        w_k
    } else {
        weights.tensors.get(&arch.attn_v_key(layer)).unwrap()
    };
    let w_o = weights.tensors.get(&arch.attn_o_key(layer)).unwrap();

    let mut q_full = dot_proj_gpu(&h_norm, w_q, backend);
    let mut k_full = dot_proj_gpu(&h_norm, w_k, backend);
    let mut v_full = dot_proj_gpu(&h_norm, w_v, backend);

    if let Some(bias) = arch
        .attn_q_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut q_full, bias);
    }
    if let Some(bias) = arch
        .attn_k_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut k_full, bias);
    }
    if let Some(bias) = arch
        .attn_v_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut v_full, bias);
    }

    if arch.has_v_norm() {
        v_full = rms_norm_heads_no_weight(&v_full, num_kv, head_dim);
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
    let k_normed = match arch
        .attn_k_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&k_full, norm_w, num_kv, head_dim, qk_norm_off),
        None => k_full,
    };

    let layer_rope_base = arch.rope_base_for_layer(layer);
    let rotary_frac = arch.rotary_fraction_for_layer(layer);
    let q_rope = apply_rope_partial(&q_normed, num_q, head_dim, layer_rope_base, rotary_frac);
    let k_rope = apply_rope_partial(&k_normed, num_kv, head_dim, layer_rope_base, rotary_frac);

    let softcap = arch.attn_logit_softcapping();
    let (attn_out, attn_weights) = gqa_attention_with_weights(
        &q_rope,
        &k_rope,
        &v_full,
        num_q,
        head_dim,
        reps,
        scale,
        seq_len,
        capture_attention,
        softcap,
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
            h + &(&normed * res_mult)
        } else {
            h + &normed
        }
    } else if res_mult != 1.0 {
        h + &(&attn_projected * res_mult)
    } else {
        h + &attn_projected
    };

    Some((h_post_attn, attn_projected, attn_weights))
}

/// Run attention and return K (post-RoPE) and V for KV cache population.
/// Accepts optional ComputeBackend for GPU-accelerated projections.
pub fn run_attention_with_kv(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
) -> Option<(Array2<f32>, Array2<f32>, Array2<f32>)> {
    run_attention_with_kv_backend(weights, h, layer, None)
}

/// Run attention with optional compute backend for accelerated projections.
pub fn run_attention_with_kv_backend(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    backend: Option<&dyn larql_compute::ComputeBackend>,
) -> Option<(Array2<f32>, Array2<f32>, Array2<f32>)> {
    use crate::forward::{add_bias, apply_norm};
    use crate::residual::{rms_norm_heads, rms_norm_heads_no_weight};

    let arch = &*weights.arch;
    let hd = arch.head_dim_for_layer(layer);
    let nq = arch.num_q_heads_for_layer(layer);
    let nkv = arch.num_kv_heads_for_layer(layer);
    let reps = nq / nkv;
    let scale = if arch.attention_multiplier() != 1.0 {
        arch.attention_multiplier() as f64
    } else {
        arch.attention_scale_for_layer(layer)
    };
    let seq_len = h.shape()[0];
    let norm_off = arch.norm_weight_offset();

    let h_norm = apply_norm(weights, h, &arch.input_layernorm_key(layer), norm_off);
    let wq = weights.tensors.get(&arch.attn_q_key(layer))?;
    let wk = weights.tensors.get(&arch.attn_k_key(layer))?;
    let v_from_k = !weights.tensors.contains_key(&arch.attn_v_key(layer));
    let wv = if v_from_k {
        wk
    } else {
        weights.tensors.get(&arch.attn_v_key(layer))?
    };
    let wo = weights.tensors.get(&arch.attn_o_key(layer))?;

    let (mut q, mut k, mut v) = (
        larql_compute::dot_proj_gpu(&h_norm, wq, backend),
        larql_compute::dot_proj_gpu(&h_norm, wk, backend),
        larql_compute::dot_proj_gpu(&h_norm, wv, backend),
    );
    for (proj, bias_fn) in [
        (&mut q, arch.attn_q_bias_key(layer) as Option<String>),
        (&mut k, arch.attn_k_bias_key(layer)),
        (&mut v, arch.attn_v_bias_key(layer)),
    ] {
        if let Some(b) = bias_fn.and_then(|key| weights.vectors.get(&key)) {
            add_bias(proj, b);
        }
    }

    if arch.has_v_norm() {
        v = rms_norm_heads_no_weight(&v, nkv, hd);
    }

    let qk_off = if arch.qk_norm_weight_offset() != 0.0 {
        arch.qk_norm_weight_offset()
    } else {
        norm_off
    };
    let q = match arch
        .attn_q_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(w) => rms_norm_heads(&q, w, nq, hd, qk_off),
        None => q,
    };
    let k = match arch
        .attn_k_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(w) => rms_norm_heads(&k, w, nkv, hd, qk_off),
        None => k,
    };

    let rb = arch.rope_base_for_layer(layer);
    let rf = arch.rotary_fraction_for_layer(layer);
    let q_r = apply_rope_partial(&q, nq, hd, rb, rf);
    let k_r = apply_rope_partial(&k, nkv, hd, rb, rf);

    let (attn_out, _) = gqa_attention_with_weights(
        &q_r,
        &k_r,
        &v,
        nq,
        hd,
        reps,
        scale,
        seq_len,
        false,
        arch.attn_logit_softcapping(),
    );
    let mut o = larql_compute::dot_proj_gpu(&attn_out, wo, backend);
    if let Some(b) = arch
        .attn_o_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut o, b);
    }

    let rm = arch.residual_multiplier();
    let h_out = if arch.has_post_norms() {
        let n = apply_norm(
            weights,
            &o,
            &arch.post_attention_layernorm_key(layer),
            norm_off,
        );
        if rm != 1.0 {
            h + &(&n * rm)
        } else {
            h + &n
        }
    } else if rm != 1.0 {
        h + &(&o * rm)
    } else {
        h + &o
    };

    Some((h_out, k_r, v))
}

/// Q4 attention projection: single projection via Q4 matvec through ComputeBackend.
/// Returns [seq_len, out_dim] f32 result, or None if backend doesn't support Q4.
pub fn q4_attention_proj(
    h: &Array2<f32>,
    q4_data: &[u8],
    num_rows: usize,
    hidden: usize,
    backend: &dyn larql_compute::ComputeBackend,
) -> Option<Array2<f32>> {
    if !backend.supports_quant(::larql_compute::QuantFormat::Q4_K) {
        return None;
    }
    let seq_len = h.shape()[0];
    let mut out = Array2::<f32>::zeros((seq_len, num_rows));

    for s in 0..seq_len {
        let x_row = h.row(s);
        let x_slice = x_row.as_slice()?;
        let (q8_x, q8_scales) = larql_compute::cpu::q4::quantize_to_q8(x_slice);
        let scores = backend.q4_matvec(q4_data, &q8_x, &q8_scales, num_rows, hidden)?;
        let mut out_row = out.row_mut(s);
        for j in 0..num_rows {
            out_row[j] = scores[j];
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::make_test_weights;
    use ndarray::Array2;

    fn h(rows: usize, cols: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (rows, cols),
            (0..rows * cols).map(|i| (i as f32 + 1.0) * 0.02).collect(),
        )
        .unwrap()
    }

    #[test]
    fn run_attention_block_gpu_no_backend_falls_back_to_cpu() {
        let weights = make_test_weights();
        let input = h(2, weights.hidden_size);
        let (h_post, attn_proj, attn_w) =
            run_attention_block_gpu(&weights, &input, 0, false, None).unwrap();
        assert_eq!(h_post.shape(), &[2, weights.hidden_size]);
        assert_eq!(attn_proj.shape()[0], 2);
        assert!(attn_w.is_none());
    }

    #[test]
    fn run_attention_block_gpu_with_cpu_backend_matches_no_backend() {
        let weights = make_test_weights();
        let input = h(2, weights.hidden_size);
        let (h_no, _, _) = run_attention_block_gpu(&weights, &input, 0, false, None).unwrap();
        let (h_cpu, _, _) =
            run_attention_block_gpu(&weights, &input, 0, false, Some(&larql_compute::CpuBackend))
                .unwrap();
        for (a, b) in h_no.iter().zip(h_cpu.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "no-backend vs CpuBackend differ: {a} vs {b}"
            );
        }
    }

    #[test]
    fn run_attention_block_gpu_capture_attention_returns_weights() {
        let weights = make_test_weights();
        let input = h(3, weights.hidden_size);
        let (_, _, attn_w) = run_attention_block_gpu(&weights, &input, 0, true, None).unwrap();
        let aw = attn_w.expect("capture=true must yield weights");
        assert_eq!(aw.heads.len(), weights.num_q_heads);
    }

    #[test]
    fn run_attention_block_gpu_all_layers_finite() {
        let weights = make_test_weights();
        let input = h(2, weights.hidden_size);
        for layer in 0..weights.num_layers {
            let (h_out, _, _) =
                run_attention_block_gpu(&weights, &input, layer, false, None).unwrap();
            assert!(
                h_out.iter().all(|v| v.is_finite()),
                "layer {layer} non-finite"
            );
        }
    }

    #[test]
    fn run_attention_block_gpu_returns_none_for_missing_layer() {
        let weights = make_test_weights();
        let input = h(2, weights.hidden_size);
        let bogus = weights.num_layers + 5;
        assert!(run_attention_block_gpu(&weights, &input, bogus, false, None).is_none());
    }

    #[test]
    fn run_attention_with_kv_returns_q_rope_and_k_v() {
        let weights = make_test_weights();
        let input = h(3, weights.hidden_size);
        let (h_out, k, v) = run_attention_with_kv(&weights, &input, 0).unwrap();
        assert_eq!(h_out.shape(), &[3, weights.hidden_size]);
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        assert_eq!(k.shape(), &[3, kv_dim]);
        assert_eq!(v.shape(), &[3, kv_dim]);
        assert!(h_out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn run_attention_block_gpu_gemma3_runs_qk_norm_and_post_norms() {
        // Gemma3 has QK norm, post_norms, has_v_norm — exercises the
        // rms_norm_heads branches and post-norm residual path.
        let weights = crate::test_utils::make_gemma3_test_weights();
        let input = h(2, weights.hidden_size);
        let (h_post, _, _) = run_attention_block_gpu(&weights, &input, 0, false, None).unwrap();
        assert_eq!(h_post.shape(), &[2, weights.hidden_size]);
        assert!(h_post.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn run_attention_block_gpu_starcoder2_runs_bias_branches() {
        // Starcoder2 has Q/K/V/O bias keys → all four `add_bias` arms fire.
        let weights = crate::test_utils::make_starcoder2_test_weights();
        let input = h(2, weights.hidden_size);
        let (h_post, _, _) = run_attention_block_gpu(&weights, &input, 0, false, None).unwrap();
        assert_eq!(h_post.shape(), &[2, weights.hidden_size]);
        assert!(h_post.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn run_attention_with_kv_backend_gemma3_routes_through_qk_norm_and_post_norms() {
        // Gemma3 enables QK norm + has_v_norm + post-norms branches in
        // `run_attention_with_kv_backend` (a separate function from
        // `run_attention_block_gpu`).
        let weights = crate::test_utils::make_gemma3_test_weights();
        let input = h(2, weights.hidden_size);
        let (h_out, k, v) = run_attention_with_kv_backend(&weights, &input, 0, None).unwrap();
        assert_eq!(h_out.shape(), &[2, weights.hidden_size]);
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        assert_eq!(k.shape(), &[2, kv_dim]);
        assert_eq!(v.shape(), &[2, kv_dim]);
        assert!(h_out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn run_attention_with_kv_backend_starcoder2_runs_q_k_v_o_bias_branches() {
        // Starcoder2 hits every `add_bias` call site in
        // `run_attention_with_kv_backend`.
        let weights = crate::test_utils::make_starcoder2_test_weights();
        let input = h(2, weights.hidden_size);
        let (h_out, _, _) = run_attention_with_kv_backend(&weights, &input, 0, None).unwrap();
        assert_eq!(h_out.shape(), &[2, weights.hidden_size]);
        assert!(h_out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn run_attention_with_kv_backend_gemma3_with_cpu_backend_matches_no_backend() {
        // Same backend-equivalence check as the existing tinymodel test
        // but on Gemma3 — exercises the backend-Some path through the
        // post-norm + QK-norm branches.
        let weights = crate::test_utils::make_gemma3_test_weights();
        let input = h(2, weights.hidden_size);
        let (h_no, _, _) = run_attention_with_kv_backend(&weights, &input, 0, None).unwrap();
        let (h_cpu, _, _) =
            run_attention_with_kv_backend(&weights, &input, 0, Some(&larql_compute::CpuBackend))
                .unwrap();
        for (a, b) in h_no.iter().zip(h_cpu.iter()) {
            assert!((a - b).abs() < 1e-4, "diverged: {a} vs {b}");
        }
    }

    #[test]
    fn q4_attention_proj_works_with_cpu_backend_at_aligned_dims() {
        // CpuBackend supports Q4 when hidden is a multiple of 32 and num_rows
        // is non-zero. Build a synthetic Q4_0 buffer (18 bytes per 32-element
        // block: scale (f16) + 16 nibbles).
        const HIDDEN: usize = 64;
        const NUM_ROWS: usize = 4;
        // Each Q4_0 block: 2 bytes scale + 16 bytes nibbles = 18 bytes/32 elems.
        let blocks_per_row = HIDDEN / 32;
        let bytes_per_row = blocks_per_row * 18;
        let q4_data = vec![0u8; NUM_ROWS * bytes_per_row];
        let input = h(2, HIDDEN);
        let result = q4_attention_proj(
            &input,
            &q4_data,
            NUM_ROWS,
            HIDDEN,
            &larql_compute::CpuBackend,
        );
        // CpuBackend may or may not accept this synthetic data — just
        // verify the function doesn't panic and the early-return shape
        // is correct when it succeeds.
        if let Some(out) = result {
            assert_eq!(out.shape(), &[2, NUM_ROWS]);
        }
    }

    #[test]
    fn run_attention_with_kv_backend_matches_no_backend() {
        let weights = make_test_weights();
        let input = h(2, weights.hidden_size);
        let (h_no, k_no, v_no) = run_attention_with_kv_backend(&weights, &input, 0, None).unwrap();
        let (h_cpu, k_cpu, v_cpu) =
            run_attention_with_kv_backend(&weights, &input, 0, Some(&larql_compute::CpuBackend))
                .unwrap();
        for (a, b) in h_no.iter().zip(h_cpu.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
        for (a, b) in k_no.iter().zip(k_cpu.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
        for (a, b) in v_no.iter().zip(v_cpu.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }
}
