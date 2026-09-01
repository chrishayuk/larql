use ndarray::Array2;

use crate::attention::SharedKV;

use super::dispatch::attn_int8_enabled;
use super::gqa_step::gqa_attention_decode_step;

/// Int8 decode-step projection: `[1, num_rows] = qw × x_q8k`. The activation
/// is pre-quantised ONCE by the caller (Q/K/V share `h_norm`'s Q8_K form).
/// The per-call kernels are single-threaded, so rows are rayon-chunked here
/// (same pattern as the Q4_K lm_head path). Returns `None` on formats other
/// than Q4_K/Q6_K or a non-256-multiple `in_dim` — caller falls back to the
/// f32-activation projection.
pub(super) fn q8k_direct_proj(
    qw: &crate::QuantWeight,
    x_q8k: &crate::cpu::ops::q4k_q8k_dot::Q8KActivation,
    num_rows: usize,
    in_dim: usize,
) -> Option<Array2<f32>> {
    use crate::cpu::ops::q4k_q8k_dot::{q4k_q8k_matvec_into, q6k_q8k_matvec_into};

    if !in_dim.is_multiple_of(256) {
        return None;
    }
    // Only the Q4_K / Q6_K k-quant layouts have a `q*k_q8k_matvec_into`
    // kernel below; gate on those two, but take the packed row stride from
    // the format helper instead of re-spelling `(in_dim/256)*144`/`*210`
    // (= `Q4_K_BLOCK_BYTES` / `Q6_K_BLOCK_BYTES` per 256-element block).
    let bytes_per_row = match qw.format() {
        crate::QuantFormat::Q4_K | crate::QuantFormat::Q6_K => {
            qw.format().packed_matrix_bytes(1, in_dim)?
        }
        _ => return None,
    };
    if qw.data.len() < num_rows * bytes_per_row {
        return None;
    }

    let mut out = vec![0.0f32; num_rows];
    const CHUNK_ROWS: usize = 32;
    crate::cpu::spin_pool::par_chunks_mut(&mut out, CHUNK_ROWS, |chunk_idx, chunk| {
        let row_start = chunk_idx * CHUNK_ROWS;
        let chunk_len = chunk.len().min(num_rows.saturating_sub(row_start));
        if chunk_len == 0 {
            return;
        }
        let w_chunk = &qw.data[row_start * bytes_per_row..(row_start + chunk_len) * bytes_per_row];
        let run = match qw.format() {
            crate::QuantFormat::Q4_K => {
                q4k_q8k_matvec_into(&mut chunk[..chunk_len], x_q8k, w_chunk, chunk_len, in_dim)
            }
            crate::QuantFormat::Q6_K => {
                q6k_q8k_matvec_into(&mut chunk[..chunk_len], x_q8k, w_chunk, chunk_len, in_dim)
            }
            _ => Ok(()),
        };
        // Every chunk is a sub-slice of the pre-flight-checked whole;
        // a refusal here is a defect in the chunking, not in the input.
        run.unwrap_or_else(|e| panic!("q8k_direct_proj: chunk {chunk_idx} refused: {e}"));
    });
    Array2::from_shape_vec((1, num_rows), out).ok()
}

/// Projection dispatch for the Q4K-direct attention step: int8 Q8_K route
/// when `LARQL_Q4K_ATTN_INT8=1` (quantising `x` lazily, at most once per
/// distinct input via the caller-held slot), else the f32-activation route.
/// A `None` from the int8 kernel (odd dims/format) falls back to f32-act
/// rather than aborting the layer.
pub(super) fn direct_proj(
    backend: &dyn crate::ComputeBackend,
    qw: &crate::QuantWeight,
    x: &Array2<f32>,
    x_q8k_slot: &mut Option<crate::cpu::ops::q4k_q8k_dot::Q8KActivation>,
    int8: bool,
    num_rows: usize,
    in_dim: usize,
) -> Option<Array2<f32>> {
    if int8 && in_dim.is_multiple_of(256) {
        if x_q8k_slot.is_none() {
            if let Some(x_slice) = x.as_slice() {
                x_q8k_slot.replace(crate::cpu::ops::q4k_q8k_dot::quantize_x_to_q8k(x_slice));
            }
        }
        if let Some(q8) = x_q8k_slot.as_ref() {
            if let Some(out) = q8k_direct_proj(qw, q8, num_rows, in_dim) {
                return Some(out);
            }
        }
    }
    q4k_direct_proj(backend, qw, x, num_rows, in_dim)
}

/// Single decode-step projection via Q4K/Q6K-direct matvec — no dequant.
///
/// `x` is `[1, in_dim]` (decode is one new token); `qw` carries the
/// `[num_rows, in_dim]` weight in its packed format (Q4_K / Q6_K). Dispatches
/// through `quant_matvec`, which routes Q4_K→`q4k_matvec`, Q6_K→`q6k_matvec`
/// (both take f32 input directly — no activation quant). Returns `[1, num_rows]`,
/// or `None` if the backend can't run that format (caller falls back to the f32
/// dequant path) or the input row isn't contiguous.
pub(super) fn q4k_direct_proj(
    backend: &dyn crate::ComputeBackend,
    qw: &crate::QuantWeight,
    x: &Array2<f32>,
    num_rows: usize,
    in_dim: usize,
) -> Option<Array2<f32>> {
    let x_slice = x.as_slice()?;
    let out = backend.quant_matvec(qw.format(), qw.data, x_slice, num_rows, in_dim)?;
    Array2::from_shape_vec((1, num_rows), out).ok()
}

/// Zero-pad a `[1, logical]` activation row out to `[1, stored]` so it can
/// feed a matvec running at a padded store's row width. The pad region of
/// every padded weight row dequantises to exactly zero, so the padded
/// products contribute nothing — this is the same exactness argument the
/// expert path pinned at hidden 288 (`stored_gate_up_cols`).
pub(super) fn zero_pad_row(x: &Array2<f32>, stored: usize) -> Array2<f32> {
    let logical = x.shape()[1];
    let mut out = Array2::<f32>::zeros((1, stored));
    out.slice_mut(ndarray::s![.., ..logical]).assign(x);
    out
}

/// Projection half of the Q4K-direct decode step: input norm, Q/K/V
/// projections (f32-act or int8 per `LARQL_Q4K_ATTN_INT8`), biases, QK/V
/// norms, RoPE. No KV-cache access at all — the caller appends
/// `k_new_rope`/`v_new` to its cache (in place, amortised O(1) on the
/// dispatch path) and then runs [`decode_step_attend_q4k_direct`] over views
/// of the full cache. Splitting here is what removes the per-layer-per-step
/// O(ctx) concat copy the monolithic form paid.
pub struct Q4kDecodeProj {
    pub q_rope: Array2<f32>,
    pub k_new_rope: Array2<f32>,
    pub v_new: Array2<f32>,
}

#[allow(clippy::too_many_arguments)]
pub fn decode_step_project_q4k_direct(
    weights: &larql_models::ModelWeights,
    h_new: &Array2<f32>,
    layer: usize,
    abs_position: usize,
    backend: &dyn crate::ComputeBackend,
    index: &dyn crate::KvIndex,
) -> Option<Q4kDecodeProj> {
    use crate::forward::add_bias;
    use crate::residual::{rms_norm_heads_no_weight, rms_norm_qk_for_arch};

    let arch = &*weights.arch;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_q = arch.num_q_heads_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let norm_offset = arch.norm_weight_offset();
    let position = abs_position;
    let hidden = weights.hidden_size;
    let q_dim = num_q * head_dim;
    let kv_dim = num_kv * head_dim;

    // Q4K-direct projection weights straight from the index. `None` → no Q4K
    // attn bytes for this layer; caller uses the f32 dequant path.
    let (wq, wk, wv, _wo) = crate::pipeline_layer::resolve_attn_weights(index, layer)?;

    let h_norm = crate::forward::apply_norm(
        weights,
        h_new,
        &arch.input_layernorm_key(layer),
        norm_offset,
    );

    // The store's own row width — writers pad rows to the quant block
    // (GPT-OSS: 2880 → 3072), and running the kernels at the logical
    // `hidden` truncates the superblock count and desynchronises every
    // row after the first. The three projections share one input, so an
    // inconsistent triple means these bytes aren't a padded row store —
    // fall back to the f32 dequant path rather than guess.
    let k_store = wq.stored_cols(q_dim, hidden);
    if k_store != wk.stored_cols(kv_dim, hidden) || k_store != wv.stored_cols(kv_dim, hidden) {
        return None;
    }
    let h_padded;
    let h_in: &Array2<f32> = if k_store == hidden {
        &h_norm
    } else {
        h_padded = zero_pad_row(&h_norm, k_store);
        &h_padded
    };

    // Int8 route (`LARQL_Q4K_ATTN_INT8=1`): Q/K/V share one Q8_K quantisation
    // of `h_norm` (filled lazily by the first projection).
    let int8 = attn_int8_enabled();
    let mut h_norm_q8k: Option<crate::cpu::ops::q4k_q8k_dot::Q8KActivation> = None;

    let mut q_full = direct_proj(backend, &wq, h_in, &mut h_norm_q8k, int8, q_dim, k_store)?;
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
        Some(norm_w) => rms_norm_qk_for_arch(&q_full, norm_w, num_q, head_dim, qk_norm_off, arch),
        None => q_full,
    };
    let layer_rope_base = crate::forward_overrides::effective_rope_base_for_layer(arch, layer);
    let rotary_frac = arch.rotary_fraction_for_layer(layer);
    let pos_divisor =
        crate::forward_overrides::effective_rope_position_divisor_for_layer(arch, layer);
    let rope_scaling = crate::forward_overrides::effective_rope_freq_scaling(arch);
    let q_rope = crate::attention::rope::apply_rope_partial_at_full(
        &q_normed,
        num_q,
        head_dim,
        layer_rope_base,
        rotary_frac,
        position,
        pos_divisor,
        rope_scaling,
    );

    let mut k_full_new = direct_proj(backend, &wk, h_in, &mut h_norm_q8k, int8, kv_dim, k_store)?;
    let mut v_full_new = direct_proj(backend, &wv, h_in, &mut h_norm_q8k, int8, kv_dim, k_store)?;
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
        Some(norm_w) => {
            rms_norm_qk_for_arch(&k_full_new, norm_w, num_kv, head_dim, qk_norm_off, arch)
        }
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
        rope_scaling,
    );

    Some(Q4kDecodeProj {
        q_rope,
        k_new_rope,
        v_new: v_full_new,
    })
}

/// Attend half of the Q4K-direct decode step: GQA over the FULL cache views
/// (which must already include this step's new K/V row), O projection,
/// post-attention norm + residual. Math is identical to the monolithic form;
/// only the cache representation (views vs owned concat) differs.
#[allow(clippy::too_many_arguments)]
pub fn decode_step_attend_q4k_direct(
    weights: &larql_models::ModelWeights,
    h_new: &Array2<f32>,
    layer: usize,
    q_rope: &Array2<f32>,
    k_all: ndarray::ArrayView2<f32>,
    v_all: ndarray::ArrayView2<f32>,
    backend: &dyn crate::ComputeBackend,
    index: &dyn crate::KvIndex,
) -> Option<Array2<f32>> {
    use crate::forward::add_bias;

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
    let hidden = weights.hidden_size;
    let q_dim = num_q * head_dim;

    let (_wq, _wk, _wv, wo) = crate::pipeline_layer::resolve_attn_weights(index, layer)?;
    let int8 = attn_int8_enabled();

    let softcap = arch.attn_logit_softcapping();
    let attn_out = gqa_attention_decode_step(
        q_rope,
        &k_all,
        &v_all,
        num_q,
        head_dim,
        reps,
        scale,
        softcap,
        crate::attention::sinks::resolve(
            arch.attn_sinks_key(layer),
            &weights.vectors,
            num_q,
            layer,
        ),
    );

    // Same stored-width contract as the projection half: the O rows may be
    // padded past `q_dim`, and the padded columns dequantise to zero.
    let o_k_store = wo.stored_cols(hidden, q_dim);
    let attn_padded;
    let attn_in: &Array2<f32> = if o_k_store == q_dim {
        &attn_out
    } else {
        attn_padded = zero_pad_row(&attn_out, o_k_store);
        &attn_padded
    };

    let mut attn_out_q8k: Option<crate::cpu::ops::q4k_q8k_dot::Q8KActivation> = None;
    let mut attn_projected = direct_proj(
        backend,
        &wo,
        attn_in,
        &mut attn_out_q8k,
        int8,
        hidden,
        o_k_store,
    )?;
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

    Some(h_post_attn)
}

/// Q4K-direct decode-step attention — reads the Q/K/V/O projection bytes
/// straight from the index (`resolve_attn_weights`) and runs them as
/// `quant_matvec` (Q4_K / Q6_K), skipping the up-front dequant-to-f32 of the
/// f32-BLAS path (`run_attention_block_decode_step_backend`).
///
/// LEGACY OWNED-CONCAT FORM: kept for callers that own their cache as
/// `SharedKV` tuples (larql-kv engine walk loops). It pays an O(ctx)
/// concat copy per call — the dispatch path (`CpuBackend::attention_step`)
/// instead uses the split project/append/attend flow above, which appends in
/// place. Outputs are identical either way.
///
/// Returns `None` (so the caller falls back to the f32 path) when the index has
/// no Q4K attention bytes for this layer, or the backend can't run a format.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn run_attention_block_decode_step_q4k_direct(
    weights: &larql_models::ModelWeights,
    h_new: &Array2<f32>,
    layer: usize,
    kv_entry: Option<&SharedKV>,
    abs_position: usize,
    backend: &dyn crate::ComputeBackend,
    index: &dyn crate::KvIndex,
) -> Option<(Array2<f32>, SharedKV)> {
    let arch = &*weights.arch;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let kv_dim = num_kv * head_dim;

    let proj = decode_step_project_q4k_direct(weights, h_new, layer, abs_position, backend, index)?;

    let (k_concat, v_concat) = match kv_entry {
        Some((k_cached, v_cached)) => {
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
                .assign(&proj.k_new_rope);
            v_out
                .slice_mut(ndarray::s![v_cached.shape()[0].., ..])
                .assign(&proj.v_new);
            (k_out, v_out)
        }
        None => (proj.k_new_rope, proj.v_new),
    };

    let h_post_attn = decode_step_attend_q4k_direct(
        weights,
        h_new,
        layer,
        &proj.q_rope,
        k_concat.view(),
        v_concat.view(),
        backend,
        index,
    )?;

    Some((h_post_attn, (k_concat, v_concat)))
}
