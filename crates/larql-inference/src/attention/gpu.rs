//! GPU-accelerated attention — routes projections through ComputeBackend.
//!
//! Falls back to CPU BLAS when backend is None.
//! Also includes Q4 quantized attention projection and KV-capture attention.

use super::gqa::gqa_attention_with_weights;
use super::rope::apply_rope_partial;
use super::AttentionWeights;
#[cfg(all(feature = "cuda", target_os = "linux"))]
use larql_compute::{CudaBackend, CudaResidentF32Matrix};
use ndarray::Array2;

#[cfg(all(feature = "cuda", target_os = "linux"))]
/// CUDA-resident dense attention projections for one transformer layer.
///
/// This intentionally covers only Q/K/V/O dense attention weights. MoE expert
/// and vindex FFN tensors stay outside this cache so Kimi-scale experts remain
/// mmap-backed instead of being uploaded wholesale to VRAM.
pub struct CudaAttentionResidency {
    layer: usize,
    q: CudaResidentF32Matrix,
    k: CudaResidentF32Matrix,
    v: CudaResidentF32Matrix,
    o: CudaResidentF32Matrix,
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
impl CudaAttentionResidency {
    pub fn from_layer(
        weights: &crate::model::ModelWeights,
        cuda: &CudaBackend,
        layer: usize,
    ) -> Option<Self> {
        let arch = &*weights.arch;
        let w_q = weights.tensors.get(&arch.attn_q_key(layer))?;
        let w_k = weights.tensors.get(&arch.attn_k_key(layer))?;
        let v_from_k = !weights.tensors.contains_key(&arch.attn_v_key(layer));
        let w_v = if v_from_k {
            w_k
        } else {
            weights.tensors.get(&arch.attn_v_key(layer))?
        };
        let w_o = weights.tensors.get(&arch.attn_o_key(layer))?;
        Some(Self {
            layer,
            q: cuda.resident_f32_matrix(w_q.view())?,
            k: cuda.resident_f32_matrix(w_k.view())?,
            v: cuda.resident_f32_matrix(w_v.view())?,
            o: cuda.resident_f32_matrix(w_o.view())?,
        })
    }

    pub fn layer(&self) -> usize {
        self.layer
    }

    pub fn q_shape(&self) -> (usize, usize) {
        (self.q.rows(), self.q.cols())
    }

    pub fn k_shape(&self) -> (usize, usize) {
        (self.k.rows(), self.k.cols())
    }

    pub fn v_shape(&self) -> (usize, usize) {
        (self.v.rows(), self.v.cols())
    }

    pub fn o_shape(&self) -> (usize, usize) {
        (self.o.rows(), self.o.cols())
    }

    pub fn project_q(&self, cuda: &CudaBackend, h: &Array2<f32>) -> Option<Array2<f32>> {
        project_rows(cuda, &self.q, h)
    }

    pub fn project_k(&self, cuda: &CudaBackend, h: &Array2<f32>) -> Option<Array2<f32>> {
        project_rows(cuda, &self.k, h)
    }

    pub fn project_v(&self, cuda: &CudaBackend, h: &Array2<f32>) -> Option<Array2<f32>> {
        project_rows(cuda, &self.v, h)
    }

    pub fn project_o(&self, cuda: &CudaBackend, h: &Array2<f32>) -> Option<Array2<f32>> {
        project_rows(cuda, &self.o, h)
    }
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn project_rows(
    cuda: &CudaBackend,
    w: &CudaResidentF32Matrix,
    h: &Array2<f32>,
) -> Option<Array2<f32>> {
    if h.shape()[1] != w.cols() {
        return None;
    }
    let mut out = Array2::<f32>::zeros((h.shape()[0], w.rows()));
    for (row_idx, row) in h.rows().into_iter().enumerate() {
        let row_owned;
        let row_slice = match row.as_slice() {
            Some(slice) => slice,
            None => {
                row_owned = row.to_vec();
                row_owned.as_slice()
            }
        };
        let projected = w.gemv(cuda, row_slice)?;
        out.row_mut(row_idx)
            .as_slice_mut()
            .expect("standard output row")
            .copy_from_slice(&projected);
    }
    Some(out)
}

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

#[cfg(all(feature = "cuda", target_os = "linux"))]
/// Run attention using Q/K/V/O weights that have already been copied to CUDA.
///
/// This removes the per-projection weight upload from the CUDA attention path
/// while preserving the existing CPU/GQA/RoPE/norm orchestration. It is scoped
/// to dense attention weights; expert/vindex FFN weights are deliberately not
/// part of this residency object.
pub fn run_attention_block_cuda_resident(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
    cuda: &CudaBackend,
    resident: &CudaAttentionResidency,
) -> Option<(Array2<f32>, Array2<f32>, Option<AttentionWeights>)> {
    use crate::forward::add_bias;
    use crate::residual::{rms_norm_heads, rms_norm_heads_no_weight};

    if resident.layer() != layer {
        return None;
    }

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

    let mut q_full = resident.project_q(cuda, &h_norm)?;
    let mut k_full = resident.project_k(cuda, &h_norm)?;
    let mut v_full = resident.project_v(cuda, &h_norm)?;

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
        arch.attn_logit_softcapping(),
    );

    let mut attn_projected = resident.project_o(cuda, &attn_out)?;
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
    if !backend.has_q4() {
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
    use crate::engines::test_utils::make_test_weights;
    use ndarray::Array2;

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn cuda_attention_residency_projects_from_resident_qkvo_weights() {
        let Some(cuda) = larql_compute::CudaBackend::new() else {
            eprintln!("skipping CUDA attention residency test: CUDA unavailable");
            return;
        };
        let weights = make_test_weights();
        let h = Array2::from_shape_vec(
            (2, weights.hidden_size),
            (0..2 * weights.hidden_size)
                .map(|i| (i as f32 + 1.0) * 0.01)
                .collect(),
        )
        .unwrap();
        let h_norm = crate::forward::apply_norm(
            &weights,
            &h,
            &weights.arch.input_layernorm_key(0),
            weights.arch.norm_weight_offset(),
        );

        let resident = CudaAttentionResidency::from_layer(&weights, &cuda, 0)
            .expect("resident Q/K/V/O attention weights");
        assert_eq!(resident.layer(), 0);
        assert_eq!(
            resident.q_shape(),
            (weights.num_q_heads * weights.head_dim, weights.hidden_size)
        );
        assert_eq!(
            resident.k_shape(),
            (weights.num_kv_heads * weights.head_dim, weights.hidden_size)
        );
        assert_eq!(
            resident.v_shape(),
            (weights.num_kv_heads * weights.head_dim, weights.hidden_size)
        );
        assert_eq!(
            resident.o_shape(),
            (weights.hidden_size, weights.num_q_heads * weights.head_dim)
        );

        let q = resident
            .project_q(&cuda, &h_norm)
            .expect("resident q projection");
        let want_q = crate::forward::dot_proj(
            &h_norm,
            weights.tensors.get(&weights.arch.attn_q_key(0)).unwrap(),
        );
        assert_eq!(q.shape(), want_q.shape());
        for (got, want) in q.iter().zip(want_q.iter()) {
            assert!((got - want).abs() < 1e-3, "got={got} want={want}");
        }
    }
    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[test]
    fn cuda_resident_attention_block_matches_existing_cuda_attention_path() {
        let Some(cuda) = larql_compute::CudaBackend::new() else {
            eprintln!("skipping CUDA resident attention block test: CUDA unavailable");
            return;
        };
        let weights = make_test_weights();
        let h = Array2::from_shape_vec(
            (2, weights.hidden_size),
            (0..2 * weights.hidden_size)
                .map(|i| (i as f32 + 1.0) * 0.01)
                .collect(),
        )
        .unwrap();
        let resident = CudaAttentionResidency::from_layer(&weights, &cuda, 0)
            .expect("resident Q/K/V/O attention weights");

        let (got, got_proj, _) =
            run_attention_block_cuda_resident(&weights, &h, 0, false, &cuda, &resident)
                .expect("resident attention block");
        let (want, want_proj, _) = run_attention_block_gpu(&weights, &h, 0, false, Some(&cuda))
            .expect("existing cuda attention block");

        assert_eq!(got.shape(), want.shape());
        assert_eq!(got_proj.shape(), want_proj.shape());
        for (got, want) in got.iter().zip(want.iter()) {
            assert!((got - want).abs() < 1e-3, "got={got} want={want}");
        }
        for (got, want) in got_proj.iter().zip(want_proj.iter()) {
            assert!((got - want).abs() < 1e-3, "got={got} want={want}");
        }
    }
}
