//! Core residual-stream compute: prefill, decode step, K/V recomputation.

use larql_compute::{dot_proj_gpu, ComputeBackend, QuantFormat};
use larql_vindex::VectorIndex;
use ndarray::{s, Array2};

use super::store::RsStore;
use crate::profiler::EngineProfiler;
use larql_inference::attention::SharedKV;
use larql_inference::attention::{
    apply_rope_partial_at, run_attention_block_decode_step_backend, run_attention_with_kv_backend,
};
use larql_inference::ffn::BackendFfn;
use larql_inference::forward::{add_bias, apply_norm, embed_tokens_pub, run_ffn};
use larql_inference::model::ModelWeights;
use larql_inference::residual::{rms_norm_heads, rms_norm_heads_no_weight};

pub struct RsPrefillResult {
    pub hidden: Array2<f32>,
    pub store: RsStore,
    pub memory_bytes: usize,
    pub window_tokens: usize,
}

pub fn rs_prefill(
    weights: &ModelWeights,
    token_ids: &[u32],
    max_window: Option<usize>,
    backend: &dyn ComputeBackend,
) -> RsPrefillResult {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens_pub(weights, token_ids);
    let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    let be = Some(backend);

    for layer in 0..num_layers {
        stored.push(h.clone());
        let (h_post_attn, _k, _v) = run_attention_with_kv_backend(weights, &h, layer, be)
            .expect("attention failed during MarkovRS prefill");
        let bffn = BackendFfn { weights, backend };
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &bffn, false);
        h = h_out;
    }

    let mut rs = RsStore {
        stored,
        cold_residuals: None,
        cold_kv: None,
        cold_abs_start: 0,
        next_position: seq_len,
        max_window,
    };

    let mut cold: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        rs.clip_layer(layer, &mut cold);
    }
    if cold.first().map_or(0, |c| c.shape()[0]) > 0 {
        let cold_kv: Vec<SharedKV> = (0..num_layers)
            .map(|layer| {
                recompute_kv(weights, &cold[layer], layer, 0, backend, None)
                    .expect("cold K/V pre-computation failed")
            })
            .collect();
        rs.cold_residuals = Some(cold);
        rs.cold_kv = Some(cold_kv);
        rs.cold_abs_start = 0;
    }

    let window_tokens = rs.window_tokens();
    let memory_bytes = rs.memory_bytes();
    RsPrefillResult {
        hidden: last_row(&h),
        store: rs,
        memory_bytes,
        window_tokens,
    }
}

pub fn rs_decode_step(
    weights: &ModelWeights,
    new_token_id: u32,
    rs: RsStore,
    backend: &dyn ComputeBackend,
) -> Option<(Array2<f32>, RsStore)> {
    rs_decode_step_inner(weights, new_token_id, rs, backend, None)
}

pub(crate) fn rs_decode_step_profiled(
    weights: &ModelWeights,
    new_token_id: u32,
    rs: RsStore,
    backend: &dyn ComputeBackend,
    profiler: &mut EngineProfiler,
) -> Option<(Array2<f32>, RsStore)> {
    rs_decode_step_inner(weights, new_token_id, rs, backend, Some(profiler))
}

fn rs_decode_step_inner(
    weights: &ModelWeights,
    new_token_id: u32,
    rs: RsStore,
    backend: &dyn ComputeBackend,
    mut profiler: Option<&mut EngineProfiler>,
) -> Option<(Array2<f32>, RsStore)> {
    use std::time::Instant;

    let num_layers = weights.num_layers;
    let abs_position = rs.next_position;
    let t_step = if profiler.is_some() {
        Some(Instant::now())
    } else {
        None
    };
    let mut h_new = embed_tokens_pub(weights, &[new_token_id]);
    let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    let mut recompute_cold_us = 0.0f64;
    let mut recompute_hot_us = 0.0f64;
    let mut attention_us = 0.0f64;
    let mut ffn_us = 0.0f64;

    for layer in 0..num_layers {
        let h_hot = &rs.stored[layer];
        let s_hot = h_hot.shape()[0];
        let hot_abs_start = abs_position.saturating_sub(s_hot);

        let (k_full, v_full) = if let Some(cold_kv) = &rs.cold_kv {
            let (k_cold, v_cold) = &cold_kv[layer];
            let t_hot = if profiler.is_some() {
                Some(Instant::now())
            } else {
                None
            };
            let (k_hot, v_hot) = recompute_kv(weights, h_hot, layer, hot_abs_start, backend, None)?;
            if let Some(t) = t_hot {
                recompute_hot_us += t.elapsed().as_secs_f64() * 1e6;
            }
            let c = k_cold.shape()[0];
            let kv_dim = k_cold.shape()[1];
            let mut k_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
            k_combined.slice_mut(s![..c, ..]).assign(k_cold);
            k_combined.slice_mut(s![c.., ..]).assign(&k_hot);
            let mut v_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
            v_combined.slice_mut(s![..c, ..]).assign(v_cold);
            v_combined.slice_mut(s![c.., ..]).assign(&v_hot);
            (k_combined, v_combined)
        } else {
            let (h_full, full_abs_start) = if let Some(cold) = &rs.cold_residuals {
                let h_cold = &cold[layer];
                let s_cold = h_cold.shape()[0];
                if s_cold > 0 {
                    let hidden = h_hot.shape()[1];
                    let mut combined = Array2::<f32>::zeros((s_cold + s_hot, hidden));
                    combined.slice_mut(s![..s_cold, ..]).assign(h_cold);
                    combined.slice_mut(s![s_cold.., ..]).assign(h_hot);
                    (combined, rs.cold_abs_start)
                } else {
                    (h_hot.clone(), hot_abs_start)
                }
            } else {
                (h_hot.clone(), hot_abs_start)
            };
            let t_cold = if profiler.is_some() {
                Some(Instant::now())
            } else {
                None
            };
            let (k, v) = recompute_kv(weights, &h_full, layer, full_abs_start, backend, None)?;
            if let Some(t) = t_cold {
                recompute_cold_us += t.elapsed().as_secs_f64() * 1e6;
            }
            (k, v)
        };

        new_stored.push(h_new.clone());

        let t_attn = if profiler.is_some() {
            Some(Instant::now())
        } else {
            None
        };
        let (h_post_attn, _new_kv) = run_attention_block_decode_step_backend(
            weights,
            &h_new,
            layer,
            Some(&(k_full, v_full)),
            abs_position,
            Some(backend),
        )?;
        if let Some(t) = t_attn {
            attention_us += t.elapsed().as_secs_f64() * 1e6;
        }

        let t_ffn = if profiler.is_some() {
            Some(Instant::now())
        } else {
            None
        };
        let bffn = BackendFfn { weights, backend };
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &bffn, false);
        if let Some(t) = t_ffn {
            ffn_us += t.elapsed().as_secs_f64() * 1e6;
        }
        h_new = h_out;
    }

    if let (Some(prof), Some(t_step)) = (profiler.as_mut(), t_step) {
        prof.recompute_cold.total_us += recompute_cold_us;
        prof.recompute_cold.count += 1;
        prof.recompute_hot.total_us += recompute_hot_us;
        prof.recompute_hot.count += 1;
        prof.attention.total_us += attention_us;
        prof.attention.count += 1;
        prof.ffn.total_us += ffn_us;
        prof.ffn.count += 1;
        prof.decode_total.record(t_step);
    }

    let mut updated_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for (stored, new_row) in rs.stored.iter().zip(new_stored.iter()) {
        let s_old = stored.shape()[0];
        let hidden_dim = stored.shape()[1];
        let mut combined = Array2::<f32>::zeros((s_old + 1, hidden_dim));
        combined.slice_mut(s![..s_old, ..]).assign(stored);
        combined.slice_mut(s![s_old.., ..]).assign(new_row);
        updated_stored.push(combined);
    }

    let mut updated_rs = RsStore {
        stored: updated_stored,
        cold_residuals: rs.cold_residuals,
        cold_kv: rs.cold_kv,
        cold_abs_start: rs.cold_abs_start,
        next_position: abs_position + 1,
        max_window: rs.max_window,
    };

    let mut overflow: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        updated_rs.clip_layer(layer, &mut overflow);
    }
    if overflow.first().map_or(0, |c| c.shape()[0]) > 0 {
        match updated_rs.cold_residuals.as_mut() {
            Some(cold) => {
                for layer in 0..num_layers {
                    let hidden = cold[layer].shape()[1];
                    let c_old = cold[layer].shape()[0];
                    let c_new = overflow[layer].shape()[0];
                    let mut merged = Array2::<f32>::zeros((c_old + c_new, hidden));
                    merged.slice_mut(s![..c_old, ..]).assign(&cold[layer]);
                    merged.slice_mut(s![c_old.., ..]).assign(&overflow[layer]);
                    cold[layer] = merged;
                }
            }
            None => {
                updated_rs.cold_residuals = Some(overflow);
            }
        }
        updated_rs.cold_kv = None;
    }

    Some((last_row(&h_new), updated_rs))
}

/// Recompute K/V from stored pre-layer residuals using `backend` for projection matmuls.
///
/// `index: Some(idx)` enables the Q4K-native fast path: per-row Q4K matvec
/// directly against the vindex's Q4K bytes, skipping the dequant-to-f32
/// step that's otherwise 8× the memory bandwidth. Quant-agnostic — the
/// backend's `quant_matvec` inspects the format byte and dispatches to
/// the right kernel (Q4K today; Q6K / future formats slot in
/// automatically). `None` keeps the f32 fallback for legacy callers.
pub fn recompute_kv(
    weights: &ModelWeights,
    h_stored: &Array2<f32>,
    layer: usize,
    abs_start: usize,
    backend: &dyn ComputeBackend,
    index: Option<&VectorIndex>,
) -> Option<(Array2<f32>, Array2<f32>)> {
    let arch = &*weights.arch;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let norm_offset = arch.norm_weight_offset();
    let qk_offset = arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 {
        qk_offset
    } else {
        norm_offset
    };

    let h_norm = apply_norm(
        weights,
        h_stored,
        &arch.input_layernorm_key(layer),
        norm_offset,
    );

    let kv_dim = num_kv * head_dim;
    let hidden = weights.hidden_size;
    let seq_len = h_norm.shape()[0];

    // Q4K-native path: per-row matvec on the vindex's raw Q4K bytes.
    // Saves the dequant-to-f32 cost (8× memory bandwidth) when the
    // backend supports Q4K matvec and the vindex has Q4K attn data.
    let q4k_path = index
        .and_then(|idx| idx.attn_kquant_layer_data(layer))
        .filter(|_| backend.supports_quant(::larql_compute::QuantFormat::Q4_K));

    let (mut k, mut v) = if let Some(attn_data) = q4k_path {
        // attn_data: [(Q, fmt), (K, fmt), (V, fmt), (O, fmt)]
        let (k_bytes, k_fmt) = attn_data[1];
        let (v_bytes, v_fmt) = attn_data[2];
        let k_format = parse_quant_format(k_fmt)?;
        let v_format = parse_quant_format(v_fmt)?;

        let mut k_out = Array2::<f32>::zeros((seq_len, kv_dim));
        let mut v_out = Array2::<f32>::zeros((seq_len, kv_dim));
        for row_idx in 0..seq_len {
            let x_row = h_norm.row(row_idx);
            let x_slice = x_row.as_slice()?;
            let k_row = backend.quant_matvec(k_format, k_bytes, x_slice, kv_dim, hidden)?;
            let v_row = backend.quant_matvec(v_format, v_bytes, x_slice, kv_dim, hidden)?;
            k_out
                .row_mut(row_idx)
                .iter_mut()
                .zip(k_row.iter())
                .for_each(|(o, &i)| *o = i);
            v_out
                .row_mut(row_idx)
                .iter_mut()
                .zip(v_row.iter())
                .for_each(|(o, &i)| *o = i);
        }
        (k_out, v_out)
    } else {
        // f32 fallback: read dequantised weights from `weights.tensors`.
        let w_k = weights.tensors.get(&arch.attn_k_key(layer))?;
        let v_from_k = !weights.tensors.contains_key(&arch.attn_v_key(layer));
        let w_v = if v_from_k {
            w_k
        } else {
            weights.tensors.get(&arch.attn_v_key(layer))?
        };
        let k = dot_proj_gpu(&h_norm, w_k, Some(backend));
        let v = dot_proj_gpu(&h_norm, w_v, Some(backend));
        (k, v)
    };

    if let Some(bias) = arch
        .attn_k_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut k, bias);
    }
    if let Some(bias) = arch
        .attn_v_bias_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        add_bias(&mut v, bias);
    }
    if arch.has_v_norm() {
        v = rms_norm_heads_no_weight(&v, num_kv, head_dim);
    }
    let k_normed = match arch
        .attn_k_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
    {
        Some(norm_w) => rms_norm_heads(&k, norm_w, num_kv, head_dim, qk_norm_off),
        None => k,
    };
    let k_rope = apply_rope_partial_at(
        &k_normed,
        num_kv,
        head_dim,
        arch.rope_base_for_layer(layer),
        arch.rotary_fraction_for_layer(layer),
        abs_start,
    );
    Some((k_rope, v))
}

fn parse_quant_format(fmt: &str) -> Option<QuantFormat> {
    match fmt {
        "Q4_K" => Some(QuantFormat::Q4_K),
        "Q4_KF" => Some(QuantFormat::Q4_KF),
        "Q6_K" => Some(QuantFormat::Q6_K),
        _ => None,
    }
}

/// Equivalent Standard KV memory in bytes for `seq_len` tokens (FP16).
pub fn kv_memory_bytes_for_seq(weights: &ModelWeights, seq_len: usize) -> usize {
    let arch = &*weights.arch;
    (0..weights.num_layers)
        .map(|l| {
            let kv_dim = arch.num_kv_heads_for_layer(l) * arch.head_dim_for_layer(l);
            seq_len * kv_dim * 2 * 2
        })
        .sum()
}

pub(super) fn last_row(h: &Array2<f32>) -> Array2<f32> {
    let last = h.shape()[0] - 1;
    h.slice(s![last..=last, ..]).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::CpuBackend;
    use larql_inference::test_utils::make_test_weights;

    // ── recompute_kv ──────────────────────────────────────────────────────────

    #[test]
    fn recompute_kv_returns_some_with_valid_weights() {
        let weights = make_test_weights();
        let h = Array2::from_elem((3, weights.hidden_size), 0.5f32);
        let result = recompute_kv(&weights, &h, 0, 0, &CpuBackend, None);
        assert!(
            result.is_some(),
            "recompute_kv should return Some with valid weights"
        );
    }

    #[test]
    fn recompute_kv_output_shape_correct() {
        let weights = make_test_weights();
        let seq_len = 4;
        let h = Array2::from_elem((seq_len, weights.hidden_size), 1.0f32);
        let (k, v) = recompute_kv(&weights, &h, 0, 0, &CpuBackend, None).unwrap();
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        assert_eq!(k.shape(), &[seq_len, kv_dim], "K shape mismatch");
        assert_eq!(v.shape(), &[seq_len, kv_dim], "V shape mismatch");
    }

    #[test]
    fn recompute_kv_output_is_finite() {
        let weights = make_test_weights();
        let h = Array2::from_elem((2, weights.hidden_size), 0.1f32);
        let (k, v) = recompute_kv(&weights, &h, 0, 0, &CpuBackend, None).unwrap();
        assert!(
            k.iter().all(|v| v.is_finite()),
            "K contains non-finite values"
        );
        assert!(
            v.iter().all(|v| v.is_finite()),
            "V contains non-finite values"
        );
    }

    #[test]
    fn recompute_kv_abs_start_shifts_rope() {
        let weights = make_test_weights();
        let h = Array2::from_elem((1, weights.hidden_size), 0.5f32);
        // Different abs_start should produce different RoPE-applied K
        let (k0, _) = recompute_kv(&weights, &h, 0, 0, &CpuBackend, None).unwrap();
        let (k5, _) = recompute_kv(&weights, &h, 0, 5, &CpuBackend, None).unwrap();
        let diff: f32 = k0.iter().zip(k5.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 0.0,
            "RoPE at different positions should produce different K"
        );
    }

    // ── rs_prefill ────────────────────────────────────────────────────────────

    #[test]
    fn rs_prefill_returns_correct_shape() {
        let weights = make_test_weights();
        let result = rs_prefill(&weights, &[0u32, 1, 2], None, &CpuBackend);
        assert_eq!(result.hidden.shape(), &[1, weights.hidden_size]);
        assert!(result.hidden.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rs_prefill_stores_all_layers() {
        let weights = make_test_weights();
        let result = rs_prefill(&weights, &[0u32], None, &CpuBackend);
        assert_eq!(result.store.stored.len(), weights.num_layers);
        assert_eq!(result.store.next_position, 1);
    }

    #[test]
    fn rs_prefill_with_window_clips_hot_store() {
        let weights = make_test_weights();
        let result = rs_prefill(&weights, &[0u32, 1, 2, 3, 4], Some(2), &CpuBackend);
        assert!(
            result.window_tokens <= 2,
            "window_tokens={} > 2",
            result.window_tokens
        );
    }

    // ── rs_decode_step ────────────────────────────────────────────────────────

    #[test]
    fn rs_decode_step_produces_finite_hidden() {
        let weights = make_test_weights();
        let prefill = rs_prefill(&weights, &[0u32], None, &CpuBackend);
        let (h, _) = rs_decode_step(&weights, 1, prefill.store, &CpuBackend).expect("decode step");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rs_decode_step_advances_position() {
        let weights = make_test_weights();
        let prefill = rs_prefill(&weights, &[0u32, 1], None, &CpuBackend);
        assert_eq!(prefill.store.next_position, 2);
        let (_, rs2) = rs_decode_step(&weights, 2, prefill.store, &CpuBackend).unwrap();
        assert_eq!(rs2.next_position, 3);
        let (_, rs3) = rs_decode_step(&weights, 3, rs2, &CpuBackend).unwrap();
        assert_eq!(rs3.next_position, 4);
    }

    #[test]
    fn rs_decode_step_with_cold_kv_branch_produces_finite_output() {
        // Windowed prefill with prompt longer than window forces cold_kv
        // population (compute.rs lines 60-68), then decode hits the
        // `Some(cold_kv)` branch (lines 128-147) instead of the
        // cold-residual recomputation path.
        let weights = make_test_weights();
        let prefill = rs_prefill(&weights, &[0u32, 1, 2, 3], Some(2), &CpuBackend);
        assert!(
            prefill.store.cold_kv.is_some(),
            "expected cold_kv to be set"
        );
        let (h, rs2) = rs_decode_step(&weights, 4, prefill.store, &CpuBackend)
            .expect("decode_step over cold_kv");
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
        // After overflow merges into cold_residuals, cold_kv is cleared
        // (compute.rs line 260) so a second decode exercises the
        // cold_residuals-only branch (lines 149-160).
        let (h2, _) =
            rs_decode_step(&weights, 5, rs2, &CpuBackend).expect("decode_step over cold_residuals");
        assert_eq!(h2.shape(), &[1, weights.hidden_size]);
        assert!(h2.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn kv_memory_bytes_for_seq_scales_linearly() {
        let weights = make_test_weights();
        let one = kv_memory_bytes_for_seq(&weights, 1);
        let ten = kv_memory_bytes_for_seq(&weights, 10);
        assert!(one > 0);
        assert_eq!(ten, one * 10, "kv memory must scale linearly with seq len");
    }

    // ── parse_quant_format pure helper (lines 384-391) ───────────────────

    #[test]
    fn parse_quant_format_recognises_q4k_q4kf_q6k() {
        assert!(matches!(
            parse_quant_format("Q4_K"),
            Some(QuantFormat::Q4_K)
        ));
        assert!(matches!(
            parse_quant_format("Q4_KF"),
            Some(QuantFormat::Q4_KF)
        ));
        assert!(matches!(
            parse_quant_format("Q6_K"),
            Some(QuantFormat::Q6_K)
        ));
    }

    #[test]
    fn parse_quant_format_unknown_returns_none() {
        assert!(parse_quant_format("Q8_0").is_none());
        assert!(parse_quant_format("F16").is_none());
        assert!(parse_quant_format("").is_none());
        assert!(parse_quant_format("Q4").is_none());
        assert!(parse_quant_format("nonsense").is_none());
    }

    // ── Profiler branches (lines 131, 137, 159, 164, 171, 178, 190, 195) ──
    //
    // Each timing branch fires only when `profiler.is_some()`. The existing
    // `with_profiling_enables_profiling_branch` test exercises one path;
    // these add coverage for the cold/hot/attn/ffn timing branches plus the
    // overflow-into-existing-cold-residuals merge path.

    #[test]
    fn profiled_decode_step_exercises_all_timing_branches() {
        use crate::profiler::EngineProfiler;
        let weights = make_test_weights();
        let prefill = rs_prefill(&weights, &[0u32, 1, 2, 3], Some(2), &CpuBackend);
        // Has cold_kv populated → exercises lines 130-147 (cold_kv branch
        // with profiler timing recompute_hot).
        assert!(prefill.store.cold_kv.is_some());
        let mut profiler = EngineProfiler::default();
        let result =
            rs_decode_step_profiled(&weights, 4, prefill.store, &CpuBackend, &mut profiler);
        assert!(result.is_some());
        // Profiler must record positive durations across all stages.
        assert!(profiler.recompute_hot.count > 0);
        assert!(profiler.attention.count > 0);
        assert!(profiler.ffn.count > 0);
        assert!(profiler.decode_total.count > 0);
    }

    #[test]
    fn profiled_decode_step_with_cold_residuals_only_path() {
        use crate::profiler::EngineProfiler;
        let weights = make_test_weights();
        // Two decodes from windowed prefill: first overflows + clears
        // cold_kv (compute.rs line 260); second hits the cold_residuals
        // branch (lines 149-160) under profiling.
        let prefill = rs_prefill(&weights, &[0u32, 1, 2, 3], Some(2), &CpuBackend);
        let (_, rs2) = rs_decode_step(&weights, 4, prefill.store, &CpuBackend).unwrap();
        assert!(
            rs2.cold_kv.is_none(),
            "cold_kv should be cleared after overflow"
        );
        let mut profiler = EngineProfiler::default();
        let result = rs_decode_step_profiled(&weights, 5, rs2, &CpuBackend, &mut profiler);
        assert!(result.is_some());
        // cold_residuals branch exercises recompute_cold counter (line 171).
        assert!(profiler.recompute_cold.count > 0);
    }

    #[test]
    fn decode_step_with_empty_cold_residuals_falls_through() {
        // Line 159: `(h_hot.clone(), hot_abs_start)` when cold tier exists
        // but s_cold == 0 (rare; happens if the engine ever clips out the
        // last cold row). Build the state by hand.
        use larql_inference::attention::SharedKV;
        use ndarray::Array2;
        let weights = make_test_weights();
        // Construct a store with cold_residuals = Some(vec![empty]) per
        // layer and cold_kv = None. The decode loop must take the "empty
        // cold" else branch (line 159).
        let num_layers = weights.num_layers;
        let hidden = weights.hidden_size;
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        let stored: Vec<Array2<f32>> = (0..num_layers)
            .map(|_| Array2::<f32>::zeros((1, hidden)))
            .collect();
        let cold_residuals: Vec<Array2<f32>> = (0..num_layers)
            .map(|_| Array2::<f32>::zeros((0, hidden)))
            .collect();
        let _ = (kv_dim, SharedKV::default()); // silence unused warnings if any
        let store = RsStore {
            stored,
            cold_residuals: Some(cold_residuals),
            cold_kv: None,
            cold_abs_start: 0,
            next_position: 1,
            max_window: None,
        };
        let result = rs_decode_step(&weights, 0, store, &CpuBackend);
        assert!(result.is_some());
        let (h, _) = result.unwrap();
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }
}
