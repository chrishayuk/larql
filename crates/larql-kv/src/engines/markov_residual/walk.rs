//! Q4K helpers — attention dequantisation re-export + WalkFfn-backed
//! forward paths.
//!
//! `ensure_attn_tensors_dequantised` moved to
//! [`larql_inference::vindex::dequant`] (2026-05-16) so the
//! `KvDispatch` trait impls in `larql-inference::kv_dispatch::*` can
//! call it without a `larql-kv → larql-inference → larql-kv` cycle.
//! Re-exported here to keep existing call sites compiling.

use larql_compute::ComputeBackend;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use super::compute::{last_row, recompute_kv, RsPrefillResult};
use super::store::RsStore;
use larql_inference::attention::run_attention_with_kv_backend;
use larql_inference::attention::SharedKV;
use larql_inference::forward::{embed_tokens_pub, run_ffn};
use larql_inference::model::ModelWeights;
use larql_inference::vindex::{WalkFfn, WalkFfnConfig};

/// Re-export — see [`larql_inference::vindex::dequant::ensure_attn_tensors_dequantised`].
pub use larql_inference::vindex::ensure_attn_tensors_dequantised;

/// Prefill using `WalkFfn` (Q4K FFN) instead of `BackendFfn` (f32 FFN).
pub(super) fn rs_prefill_walk(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_ids: &[u32],
    max_window: Option<usize>,
    backend: &dyn ComputeBackend,
) -> RsPrefillResult {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens_pub(weights, token_ids);
    let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    let be = Some(backend);

    // Hoist WalkFfn construction out of the per-layer loop. Previously
    // this rebuilt the WalkFfn 34 times per prefill (once per layer);
    // now once total. WalkFfn carries no per-layer state — it's the
    // gate-index + backend pair, both stable across the loop.
    let walk_ffn = WalkFfn::from_config(weights, index, WalkFfnConfig::dense(num_layers))
        .with_backend(backend);

    for layer in 0..num_layers {
        stored.push(h.clone());
        let (h_post_attn, _k, _v) = run_attention_with_kv_backend(weights, &h, layer, be)
            .expect("attention failed during MarkovRS Q4K prefill");
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
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
                recompute_kv(weights, &cold[layer], layer, 0, backend, Some(index))
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

/// Decode step using `WalkFfn` (Q4K FFN).
pub(super) fn rs_decode_step_walk(
    weights: &ModelWeights,
    index: &VectorIndex,
    new_token_id: u32,
    rs: RsStore,
    backend: &dyn ComputeBackend,
) -> Option<(Array2<f32>, RsStore)> {
    use ndarray::s;

    let instrument = std::env::var("LARQL_INSTRUMENT_MARKOV").is_ok();

    let num_layers = weights.num_layers;
    let abs_position = rs.next_position;
    let mut h_new = embed_tokens_pub(weights, &[new_token_id]);
    let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    // Hoist WalkFfn out of the per-layer loop — see note in
    // `rs_prefill_walk`. Was 34× construction per decode step.
    let walk_ffn = WalkFfn::from_config(weights, index, WalkFfnConfig::dense(num_layers))
        .with_backend(backend);

    // Per-stage timing accumulators (only when LARQL_INSTRUMENT_MARKOV is set).
    let mut t_recompute_kv = 0.0f64;
    let mut t_concat = 0.0f64;
    let mut t_attention = 0.0f64;
    let mut t_ffn = 0.0f64;
    let mut attn_helper_hits = 0usize;
    let mut attn_helper_misses = 0usize;
    let mut s_hot_first_layer = 0usize;

    for layer in 0..num_layers {
        let h_hot = &rs.stored[layer];
        let s_hot = h_hot.shape()[0];
        if layer == 0 {
            s_hot_first_layer = s_hot;
        }
        let hot_abs_start = abs_position.saturating_sub(s_hot);

        let t_kv_start = if instrument {
            Some(std::time::Instant::now())
        } else {
            None
        };

        let (k_full, v_full) = if let Some(cold_kv) = &rs.cold_kv {
            let (k_cold, v_cold) = &cold_kv[layer];
            let (k_hot, v_hot) =
                recompute_kv(weights, h_hot, layer, hot_abs_start, backend, Some(index))?;
            let kv_recompute_done = if instrument {
                Some(std::time::Instant::now())
            } else {
                None
            };
            if let (Some(start), Some(done)) = (t_kv_start, kv_recompute_done) {
                t_recompute_kv += done.duration_since(start).as_secs_f64() * 1000.0;
            }

            let t_concat_start = if instrument {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let c = k_cold.shape()[0];
            let kv_dim = k_cold.shape()[1];
            let mut k_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
            k_combined.slice_mut(s![..c, ..]).assign(k_cold);
            k_combined.slice_mut(s![c.., ..]).assign(&k_hot);
            let mut v_combined = Array2::<f32>::zeros((c + s_hot, kv_dim));
            v_combined.slice_mut(s![..c, ..]).assign(v_cold);
            v_combined.slice_mut(s![c.., ..]).assign(&v_hot);
            if let Some(start) = t_concat_start {
                t_concat += start.elapsed().as_secs_f64() * 1000.0;
            }
            (k_combined, v_combined)
        } else {
            let (h_full, full_abs_start) = match &rs.cold_residuals {
                Some(cold) if cold[layer].shape()[0] > 0 => {
                    let h_cold = &cold[layer];
                    let s_cold = h_cold.shape()[0];
                    let hidden = h_hot.shape()[1];
                    let mut combined = Array2::<f32>::zeros((s_cold + s_hot, hidden));
                    combined.slice_mut(s![..s_cold, ..]).assign(h_cold);
                    combined.slice_mut(s![s_cold.., ..]).assign(h_hot);
                    (combined, rs.cold_abs_start)
                }
                _ => (h_hot.clone(), hot_abs_start),
            };
            let pair = recompute_kv(
                weights,
                &h_full,
                layer,
                full_abs_start,
                backend,
                Some(index),
            )?;
            if let Some(start) = t_kv_start {
                t_recompute_kv += start.elapsed().as_secs_f64() * 1000.0;
            }
            pair
        };

        new_stored.push(h_new.clone());

        let t_attn_start = if instrument {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let kv_pair = (k_full, v_full);
        let native_result = larql_inference::vindex::attention_decode_step_native(
            weights,
            index,
            backend,
            &h_new,
            layer,
            Some(&kv_pair),
            abs_position,
        );
        if instrument {
            if native_result.is_some() {
                attn_helper_hits += 1;
            } else {
                attn_helper_misses += 1;
            }
        }
        let (h_post_attn, _new_kv) = native_result.or_else(|| {
            larql_inference::attention::run_attention_block_decode_step_backend(
                weights,
                &h_new,
                layer,
                Some(&kv_pair),
                abs_position,
                Some(backend),
            )
        })?;
        if let Some(start) = t_attn_start {
            t_attention += start.elapsed().as_secs_f64() * 1000.0;
        }

        let t_ffn_start = if instrument {
            Some(std::time::Instant::now())
        } else {
            None
        };
        // Try the production-path native-quantised FFN helper first —
        // direct Q4K/Q6K matvec on the vindex's compact gate/up/down
        // bytes. Falls back to WalkFfn (and then dense WeightFfn) when
        // the backend doesn't have native quant support or the layer
        // isn't direct-matvec-eligible.
        //
        // The fallback is the bottleneck: WalkFfn detects "zero features"
        // for dense Gemma layers and dispatches to dense WeightFfn,
        // which does f32 matmul on dequantised gate/up/down — ~70ms
        // per layer × 34 = 2.4s per token. Native Q4K FFN is ~0.7ms
        // per layer × 34 = ~25ms (matching the production CPU path).
        let h_out = larql_inference::vindex::ffn_decode_step_native(
            weights,
            index,
            backend,
            &h_post_attn,
            layer,
        )
        .unwrap_or_else(|| {
            let (h, _) = run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
            h
        });
        if let Some(start) = t_ffn_start {
            t_ffn += start.elapsed().as_secs_f64() * 1000.0;
        }
        h_new = h_out;
    }

    if instrument {
        let total = t_recompute_kv + t_concat + t_attention + t_ffn;
        eprintln!(
            "[markov-rs/decode] s_hot={s_hot_first_layer} recompute_kv={t_recompute_kv:.2}ms \
             concat={t_concat:.2}ms attention={t_attention:.2}ms ffn={t_ffn:.2}ms \
             total={total:.2}ms (attn_helper hits/miss={attn_helper_hits}/{attn_helper_misses})"
        );
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
