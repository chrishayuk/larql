//! Native attention and FFN decode steps against k-quant weights.
//!
//! Split out of `cached.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;
use crate::attention::{
    decode::gqa_attention_decode_step_windowed, rope::apply_rope_partial_at_full,
};
use crate::cpu::ops::q4k_q8k_dot::{quantize_x_to_q8k_into, Q8KActivation};
use crate::forward::embed_tokens_pub;
use crate::forward::layer::apply_layer_scalar;
use crate::forward::ple::{apply_per_layer_embedding, precompute_per_layer_inputs};
use crate::forward::{add_bias, apply_norm};
use crate::residual::{rms_norm_heads_no_weight, rms_norm_qk_for_arch};
use crate::ComputeBackend;
use larql_models::ModelWeights;
use ndarray::Array2;

/// One-row attention block using direct Q4_K/Q6_K matvec on the
/// quantised attention slices. Mirrors
/// [`crate::attention::decode::run_attention_block_decode_step_backend`]
/// but reads weights from `index.attn_kquant_layer_data(layer)` instead of
/// dequantised f32 in `weights.tensors`.
#[allow(clippy::too_many_arguments)]
/// Production-path attention decode step reading **quantised** weights
/// from the vindex (not f32 dequantised tensors). Same input/output
/// shape as
/// [`crate::attention::run_attention_block_decode_step_backend`], but
/// reads `index.attn_kquant_layer_data(layer)` directly and dispatches
/// the Q/K/V/O projections to the backend's native quantised matvec
/// (today Q4K / Q4_KF / Q6K via `q4k_matvec_q8_input`). Extending to
/// new quantised formats is internal to this function — the public
/// signature stays format-agnostic.
///
/// Used by `StandardEngine`'s coarse path and by research engines
/// (`MarkovResidual`, `WindowedCheckpoint`, `TurboQuant`) that want the
/// production decode kernel without inheriting the per-layer dispatch
/// trait's cached-K/V shape.
///
/// `h_new` must be a single-row residual (1 × hidden). Multi-row
/// prefill is handled by `predict_kquant_prefill` (separate shape; the
/// `q4k_` in that name is pre-existing debt — see ROADMAP U8/U9 for
/// the broader quant-agnostic rename of the kquant_forward module).
///
/// Returns `None` if the layer has no quantised attention data in the
/// index or if the backend's matvec for the format is unavailable.
pub fn attention_decode_step_native(
    weights: &ModelWeights,
    index: &dyn crate::KvIndex,
    // Kept on the helper signature for parity with the outer
    // `predict_kquant_decode_step_direct` API and any future asm dispatch
    // that wants runtime feature detection.
    _backend: &dyn ComputeBackend,
    h_new: &Array2<f32>,
    layer: usize,
    kv_entry: Option<&(Array2<f32>, Array2<f32>)>,
    abs_position: usize,
) -> Option<(Array2<f32>, (Array2<f32>, Array2<f32>))> {
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_q = arch.num_q_heads_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let reps = num_q / num_kv;
    let q_dim = num_q * head_dim;
    let kv_dim = num_kv * head_dim;
    let scale = if arch.attention_multiplier() != 1.0 {
        arch.attention_multiplier() as f64
    } else {
        arch.attention_scale_for_layer(layer)
    };
    let norm_offset = arch.norm_weight_offset();

    let h_norm = apply_norm(
        weights,
        h_new,
        &arch.input_layernorm_key(layer),
        norm_offset,
    );
    let h_norm_row: &[f32] = h_norm.row(0).to_slice().or_else(|| h_norm.as_slice())?;

    let attn = index.attn_kquant_layer_data(layer)?;
    let (q_bytes, q_fmt) = attn[0];
    let (k_bytes, k_fmt) = attn[1];
    let (v_bytes, v_fmt) = attn[2];
    let (o_bytes, o_fmt) = attn[3];

    // Q8_K-quantise `h_norm` once and reuse for Q / K / V projections.
    // sdot int8 dot is ~2-3× the f32 FMA throughput of the
    // `q4k_matvec_into` path; the quantisation step itself is O(hidden)
    // and amortises across the three projections (and O after attn).
    let mut h_norm_q8k = Q8KActivation::with_capacity(hidden);
    quantize_x_to_q8k_into(&mut h_norm_q8k, h_norm_row);

    let q_vec = matvec_q4k_or_q6k_q8k(q_bytes, q_fmt, &h_norm_q8k, q_dim, hidden)?;
    let mut q_full = vec_to_2d_row(q_vec);
    if let Some(bias) = arch
        .attn_q_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut q_full, bias);
    }

    let qk_offset = arch.qk_norm_weight_offset();
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
    // RoPE must match the staged path / prefill exactly: override-aware
    // base, the per-layer position divisor (Gemma 3 linear rope_scaling
    // applies ÷factor on GLOBAL layers only), and llama3 frequency
    // scaling. The unscaled `apply_rope_partial_at` here was the direct-
    // path divergence on gemma3-4b (global-layer K/Q rope'd at 8× the
    // position the prefill cache used).
    let layer_rope_base = crate::forward_overrides::effective_rope_base_for_layer(arch, layer);
    let rotary_frac = arch.rotary_fraction_for_layer(layer);
    let pos_divisor =
        crate::forward_overrides::effective_rope_position_divisor_for_layer(arch, layer);
    let rope_scaling = crate::forward_overrides::effective_rope_freq_scaling(arch);
    let q_rope = apply_rope_partial_at_full(
        &q_normed,
        num_q,
        head_dim,
        layer_rope_base,
        rotary_frac,
        abs_position,
        pos_divisor,
        rope_scaling,
    );

    let k_vec = matvec_q4k_or_q6k_q8k(k_bytes, k_fmt, &h_norm_q8k, kv_dim, hidden)?;
    let v_vec = matvec_q4k_or_q6k_q8k(v_bytes, v_fmt, &h_norm_q8k, kv_dim, hidden)?;
    let mut k_full_new = vec_to_2d_row(k_vec);
    let mut v_full_new = vec_to_2d_row(v_vec);
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
    let k_new_rope = apply_rope_partial_at_full(
        &k_normed,
        num_kv,
        head_dim,
        layer_rope_base,
        rotary_frac,
        abs_position,
        pos_divisor,
        rope_scaling,
    );

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
                .assign(&k_new_rope);
            v_out
                .slice_mut(ndarray::s![v_cached.shape()[0].., ..])
                .assign(&v_full_new);
            (k_out, v_out)
        }
        None => (k_new_rope, v_full_new),
    };

    let softcap = arch.attn_logit_softcapping();
    // Per-layer sliding window, same shared rule the Metal spec uses.
    // This is the CPU coarse Q4K decode — the path `standard` takes on
    // CpuBackend — so without it a Gemma-class model attends full history
    // here while Metal masks.
    let window = crate::forward_overrides::effective_attention_window_for_layer(arch, layer);
    let attn_out = gqa_attention_decode_step_windowed(
        &q_rope,
        &k_concat,
        &v_concat,
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
        window,
    );
    let attn_out_row: &[f32] = attn_out.row(0).to_slice().or_else(|| attn_out.as_slice())?;

    // Re-quantise the attention output for the O projection. Different
    // input from Q/K/V (attn_out vs h_norm), so we need a fresh Q8_K.
    let mut attn_out_q8k = Q8KActivation::with_capacity(q_dim);
    quantize_x_to_q8k_into(&mut attn_out_q8k, attn_out_row);
    let o_vec = matvec_q4k_or_q6k_q8k(o_bytes, o_fmt, &attn_out_q8k, hidden, q_dim)?;
    let mut attn_projected = vec_to_2d_row(o_vec);
    if let Some(bias) = arch
        .attn_o_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut attn_projected, bias);
    }

    let res_mult = arch.residual_multiplier();
    let h_post_attn = if arch.has_post_norms() {
        let normed = apply_norm(
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

/// One-row gated FFN block using direct native-quantised matvec on
/// the vindex's compact bytes (Q4K / Q6K today). Mirrors
/// [`crate::ffn::weight::dense_ffn_forward_backend`] but reads gate/up/
/// down from the vindex slices and avoids the f32 staging — same
/// production path that powers `larql run` / `larql bench --cpu` at
/// ~24 tok/s on Gemma 3 4B Q4K (M3 Max, 8 threads).
///
/// Returns `None` if the vindex layer lacks compact FFN bytes or the
/// architecture isn't supported by the direct-matvec path. Engines
/// that get `None` fall back to whichever `FfnBackend` they have.
///
/// `h_post_attn` must be a single-row residual (1 × hidden). Public
/// counterpart to [`attention_decode_step_native`] for the FFN side.
pub fn ffn_decode_step_native(
    weights: &ModelWeights,
    index: &dyn crate::KvIndex,
    backend: &dyn ComputeBackend,
    h_post_attn: &Array2<f32>,
    layer: usize,
) -> Option<Array2<f32>> {
    run_ffn_decode_step_q4k_direct(weights, index, backend, h_post_attn, layer)
}

/// One-row gated FFN block using direct Q4_K/Q6_K matvec. Mirrors
/// [`crate::ffn::weight::dense_ffn_forward_backend`] but reads gate/up/
/// down from the vindex slices and avoids the f32 staging.
pub(crate) fn run_ffn_decode_step_q4k_direct(
    weights: &ModelWeights,
    index: &dyn crate::KvIndex,
    _backend: &dyn ComputeBackend,
    h_post_attn: &Array2<f32>,
    layer: usize,
) -> Option<Array2<f32>> {
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let intermediate = index.num_features(layer);
    let norm_offset = arch.norm_weight_offset();

    // Pre-FFN norm: same selection logic as `run_ffn` — when the arch
    // uses post_norms, the pre-FFN key is `pre_feedforward_layernorm`;
    // otherwise it reuses `post_attention_layernorm` as the FFN input
    // norm. Falls back to weightless RMS when no key is set.
    let pre_ffn_key = if arch.has_post_norms() {
        arch.pre_feedforward_layernorm_key(layer)
    } else {
        Some(arch.post_attention_layernorm_key(layer))
    };
    let h_in = match pre_ffn_key {
        Some(key) => apply_norm(weights, h_post_attn, &key, norm_offset),
        None => crate::residual::rms_norm(h_post_attn, None, norm_offset),
    };
    let h_in_row: &[f32] = h_in.row(0).to_slice().or_else(|| h_in.as_slice())?;

    let ffn = index.interleaved_kquant_layer_data(layer)?;
    let (gate_bytes, gate_fmt) = ffn[0];
    let (up_bytes, up_fmt) = ffn[1];
    let (down_bytes, down_fmt) = ffn[2];

    // Only Gated FFNs reach this path today (it's what predict_kquant_hidden
    // currently dequantises). Non-gated archs route through the dequant
    // fallback via the per-layer gate at the caller.
    if arch.ffn_type() != larql_models::FfnType::Gated {
        return None;
    }

    // Q8_K-quantise `h_in` once and feed it to both gate and up via the
    // sdot-based fused matvec. This is the int8-dot Q4_K × Q8_K path
    // that closes the bandwidth gap to llama.cpp on M3 Max.
    let mut h_in_q8k = Q8KActivation::with_capacity(hidden);
    quantize_x_to_q8k_into(&mut h_in_q8k, h_in_row);

    // Two separate matvecs, each rayon-parallel inside
    // `matvec_q4k_or_q6k_q8k`. The "fused gate+up" variant in
    // `larql-compute` (`q4k_q8k_gate_up_into`) is single-threaded;
    // the input vector (10 KB) stays in L1 across two sequential
    // calls anyway, so we don't need explicit fusion to keep `x`
    // hot. Splitting lets both matvecs run row-parallel.
    let gate_vec = matvec_q4k_or_q6k_q8k(gate_bytes, gate_fmt, &h_in_q8k, intermediate, hidden)?;
    let up_vec = matvec_q4k_or_q6k_q8k(up_bytes, up_fmt, &h_in_q8k, intermediate, hidden)?;

    // Element-wise activation: activation(gate) * up. Rayon-chunked — the
    // per-element math (libm tanh/exp included) is unchanged, so the output
    // is bit-identical to the serial loop; the decode sample showed this
    // scalar pass serial on the main thread while the workers slept.
    let mut activated = vec![0.0f32; intermediate];
    {
        let gelu = arch.gate_up_is_gelu_tanh();
        let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
        let gate_ref = &gate_vec[..];
        let up_ref = &up_vec[..];
        crate::cpu::spin_pool::par_chunks_mut(&mut activated, 256, |ci, a_c| {
            let start = ci * 256;
            let g_c = &gate_ref[start..start + a_c.len()];
            let u_c = &up_ref[start..start + a_c.len()];
            if gelu {
                for ((a, &x), &u) in a_c.iter_mut().zip(g_c.iter()).zip(u_c.iter()) {
                    let inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
                    *a = 0.5 * x * (1.0 + inner.tanh()) * u;
                }
            } else {
                // SiLU = x * sigmoid(x). Same shape as dense_ffn_forward_backend.
                for ((a, &x), &u) in a_c.iter_mut().zip(g_c.iter()).zip(u_c.iter()) {
                    let sig = 1.0 / (1.0 + (-x).exp());
                    *a = x * sig * u;
                }
            }
        });
    }

    // down projection: out = activated @ W_down.T → [hidden].
    // Re-quantise the post-activation vector (`intermediate`-wide) for
    // the down matvec — different input from gate/up.
    //
    // The stored down row width may be PADDED up to a 256-multiple when
    // `intermediate` isn't one (e.g. the 26B-A4B hybrid-MoE dense slab:
    // intermediate 2112 stored as 2304-col Q6_K rows). Derive the stored
    // width from the byte length and zero-pad the activation to match —
    // pad columns multiply zero activations, so the result is exact.
    // (Twin of the same handling in larql-inference's cached.rs — keep in
    // lockstep, see the consolidation hazard in q4k-direct-attention.md.)
    let down_sb_bytes = match crate::QuantFormat::from_registry_tag(down_fmt)
        .filter(|f| f.route().q8k_matvec.is_some())
        .and_then(|f| f.packed_block_layout())
    {
        Some((_, block_bytes)) => block_bytes,
        None => return None,
    };
    let down_bytes_per_row = down_bytes.len() / hidden;
    if down_bytes_per_row == 0 || !down_bytes_per_row.is_multiple_of(down_sb_bytes) {
        return None;
    }
    let stored_cols =
        down_bytes_per_row / down_sb_bytes * larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
    if stored_cols < intermediate {
        return None;
    }
    let activated_padded: Vec<f32>;
    let act_slice: &[f32] = if stored_cols != intermediate {
        let mut p = vec![0.0f32; stored_cols];
        p[..intermediate].copy_from_slice(&activated);
        activated_padded = p;
        &activated_padded
    } else {
        &activated
    };
    let mut activated_q8k = Q8KActivation::with_capacity(stored_cols);
    quantize_x_to_q8k_into(&mut activated_q8k, act_slice);
    let down_vec =
        matvec_q4k_or_q6k_q8k(down_bytes, down_fmt, &activated_q8k, hidden, stored_cols)?;
    let mut out = vec_to_2d_row(down_vec);
    if let Some(bias) = arch
        .ffn_down_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut out, bias);
    }

    // Post-FFN residual + optional post-FFN layernorm. Same selection
    // logic as `run_ffn`: only fire when has_post_norms() AND the arch
    // exposes a post-FFN norm key.
    let res_mult = arch.residual_multiplier();
    let h_post_ffn = if arch.has_post_norms() {
        let normed = match arch.post_feedforward_layernorm_key(layer) {
            Some(key) => apply_norm(weights, &out, &key, norm_offset),
            None => crate::residual::rms_norm(&out, None, norm_offset),
        };
        if res_mult != 1.0 {
            h_post_attn + &(&normed * res_mult)
        } else {
            h_post_attn + &normed
        }
    } else if res_mult != 1.0 {
        h_post_attn + &(&out * res_mult)
    } else {
        h_post_attn + &out
    };

    Some(h_post_ffn)
}

/// Dequant-free decode step. Same shape contract as
/// [`predict_kquant_decode_step`] but routes every projection through
/// `backend.quant_matvec` instead of the per-layer
/// `insert_q4k_layer_tensors` → dense f32 staging dance. Returns `None`
/// if any layer has a format the direct-matvec path doesn't handle
/// (caller falls back to [`predict_kquant_decode_step`]).
pub fn predict_kquant_decode_step_direct(
    weights: &ModelWeights,
    token_id: u32,
    index: &dyn crate::KvIndex,
    backend: &dyn ComputeBackend,
    cache: &mut CpuKvCache,
    abs_position: usize,
) -> Option<Array2<f32>> {
    predict_kquant_decode_step_direct_with_state(
        weights,
        token_id,
        index,
        backend,
        cache,
        abs_position,
        None,
    )
}

/// Decode step with optional per-layer state capture (`Some(state)`
/// populates `h_in` / `k_new` / `v_new` per layer at near-zero cost
/// since this CPU path already walks the layers serially). Engines
/// that need per-layer state — `markov_residual` for residual storage,
/// `markov_residual_codec` ditto, `turbo_quant` for per-layer K/V
/// compression — call through here via `KvDispatch::
/// coarse_decode_step_with_state`. When `state` is `None` this is
/// bit-identical to [`predict_kquant_decode_step_direct`].
pub fn predict_kquant_decode_step_direct_with_state(
    weights: &ModelWeights,
    token_id: u32,
    index: &dyn crate::KvIndex,
    backend: &dyn ComputeBackend,
    cache: &mut CpuKvCache,
    abs_position: usize,
    mut state: Option<&mut crate::PerLayerDecodeState>,
) -> Option<Array2<f32>> {
    use ndarray::s;
    let num_layers = weights.num_layers;
    if cache.len() != num_layers {
        return None;
    }

    let mut h = embed_tokens_pub(weights, &[token_id]);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, &[token_id]);

    for layer in 0..num_layers {
        if let Some(s) = state.as_deref_mut() {
            s.h_in_per_layer
                .push(crate::state_handle::CpuStateHandle::boxed(h.clone()));
        }
        let kv_entry = cache[layer].as_ref();
        let (h_post_attn, new_kv) = attention_decode_step_native(
            weights,
            index,
            backend,
            &h,
            layer,
            kv_entry,
            abs_position,
        )?;
        if let Some(s) = state.as_deref_mut() {
            // new_kv is the full prior+new K/V; the new row is the
            // last row. Engines that cache per-layer K/V (markov_rs
            // hot_kv, turbo_quant compressed) consume this row.
            let n = new_kv.0.shape()[0];
            s.k_new_per_layer
                .push(crate::state_handle::CpuStateHandle::boxed(
                    new_kv.0.slice(s![n - 1..n, ..]).to_owned(),
                ));
            s.v_new_per_layer
                .push(crate::state_handle::CpuStateHandle::boxed(
                    new_kv.1.slice(s![n - 1..n, ..]).to_owned(),
                ));
        }
        cache[layer] = Some(new_kv);

        let h_post_ffn =
            run_ffn_decode_step_q4k_direct(weights, index, backend, &h_post_attn, layer)?;
        let mut h_out =
            apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
        apply_layer_scalar(weights, &mut h_out, layer);
        h = h_out;
    }

    Some(h)
}
