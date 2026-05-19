//! Q4K-walk paths for `MarkovResidualCodecEngine`.
//!
//! Mirrors `markov_residual/q4k.rs` with the cold tier routed through the
//! codec. Used when the engine is asked to run on a compact (Q4K-walk)
//! vindex — the dense `BackendFfn` path in [`super::compute`] cannot read
//! `--compact` FFN weights. This module delegates FFN to `WalkFfn`
//! (native Q4K matvec on the vindex's compact gate/up/down bytes) and
//! passes `Some(index)` to `recompute_kv` so the K/V projections also
//! take the Q4K-native path.

use larql_compute::ComputeBackend;
use larql_inference::attention::{
    run_attention_block_decode_step_backend, run_attention_with_kv_backend, SharedKV,
};
use larql_inference::forward::{embed_tokens_pub, run_ffn};
use larql_inference::model::ModelWeights;
use larql_inference::vindex::{WalkFfn, WalkFfnConfig};
use larql_vindex::VectorIndex;
use ndarray::{s, Array2};

use super::compute::RsPrefillResultCodec;
use crate::engines::markov_residual::recompute_kv;
use crate::engines::markov_residual_codec::codec::ColdResidualCodec;
use crate::engines::markov_residual_codec::store::{EncodedColdLayer, RsStoreCodec};

pub fn rs_prefill_codec_walk(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_ids: &[u32],
    max_window: Option<usize>,
    codec: ColdResidualCodec,
    backend: &dyn ComputeBackend,
) -> RsPrefillResultCodec {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens_pub(weights, token_ids);
    let mut stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    let be = Some(backend);

    let walk_ffn = WalkFfn::from_config(weights, index, WalkFfnConfig::dense(num_layers))
        .with_backend(backend);

    for layer in 0..num_layers {
        stored.push(h.clone());
        let (h_post_attn, _k, _v) = run_attention_with_kv_backend(weights, &h, layer, be)
            .expect("attention failed during MarkovResidualCodec Q4K prefill");
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
        h = h_out;
    }

    let hidden_size = weights.hidden_size;
    let mut rs = RsStoreCodec {
        stored,
        cold_encoded: None,
        cold_kv: None,
        cold_abs_start: 0,
        next_position: seq_len,
        max_window,
        codec,
    };

    let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        overflow_per_layer.push(rs.clip_layer_overflow(layer));
    }
    if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
        let mut encoded_layers: Vec<EncodedColdLayer> = Vec::with_capacity(num_layers);
        let mut cold_kv: Vec<SharedKV> = Vec::with_capacity(num_layers);
        for (layer, overflow) in overflow_per_layer.iter().enumerate() {
            let decoded_overflow = roundtrip(overflow, codec);
            let (k, v) = recompute_kv(weights, &decoded_overflow, layer, 0, backend, Some(index))
                .expect("cold K/V pre-computation failed");
            cold_kv.push((k, v));
            let mut enc = EncodedColdLayer::empty(hidden_size);
            enc.append(codec, overflow);
            encoded_layers.push(enc);
        }
        rs.cold_encoded = Some(encoded_layers);
        rs.cold_kv = Some(cold_kv);
        rs.cold_abs_start = 0;
    }

    RsPrefillResultCodec {
        hidden: last_row(&h),
        store: rs,
    }
}

pub fn rs_decode_step_codec_walk(
    weights: &ModelWeights,
    index: &VectorIndex,
    new_token_id: u32,
    rs: RsStoreCodec,
    backend: &dyn ComputeBackend,
) -> Option<(Array2<f32>, RsStoreCodec)> {
    let num_layers = weights.num_layers;
    let abs_position = rs.next_position;
    let mut h_new = embed_tokens_pub(weights, &[new_token_id]);
    let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    let walk_ffn = WalkFfn::from_config(weights, index, WalkFfnConfig::dense(num_layers))
        .with_backend(backend);

    for layer in 0..num_layers {
        let h_hot = &rs.stored[layer];
        let s_hot = h_hot.shape()[0];
        let hot_abs_start = abs_position.saturating_sub(s_hot);

        let (k_full, v_full) = if let Some(cold_kv) = &rs.cold_kv {
            let (k_cold, v_cold) = &cold_kv[layer];
            let (k_hot, v_hot) =
                recompute_kv(weights, h_hot, layer, hot_abs_start, backend, Some(index))?;
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
            let (h_full, full_abs_start) = match &rs.cold_encoded {
                Some(cold_layers) if cold_layers[layer].n_positions > 0 => {
                    let decoded = cold_layers[layer].decode(rs.codec);
                    let hidden = h_hot.shape()[1];
                    let mut combined = Array2::<f32>::zeros((decoded.shape()[0] + s_hot, hidden));
                    combined
                        .slice_mut(s![..decoded.shape()[0], ..])
                        .assign(&decoded);
                    combined
                        .slice_mut(s![decoded.shape()[0].., ..])
                        .assign(h_hot);
                    (combined, rs.cold_abs_start)
                }
                _ => (h_hot.clone(), hot_abs_start),
            };
            recompute_kv(
                weights,
                &h_full,
                layer,
                full_abs_start,
                backend,
                Some(index),
            )?
        };

        new_stored.push(h_new.clone());

        let kv_pair = (k_full, v_full);
        // Native Q4K attention helper, then dense fallback (same shape as
        // markov_residual::walk::rs_decode_step_walk).
        let native_result = larql_inference::vindex::attention_decode_step_native(
            weights,
            index,
            backend,
            &h_new,
            layer,
            Some(&kv_pair),
            abs_position,
        );
        let (h_post_attn, _new_kv) = native_result.or_else(|| {
            run_attention_block_decode_step_backend(
                weights,
                &h_new,
                layer,
                Some(&kv_pair),
                abs_position,
                Some(backend),
            )
        })?;

        // Native Q4K FFN, then WalkFfn fallback.
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
        h_new = h_out;
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

    let mut updated_rs = RsStoreCodec {
        stored: updated_stored,
        cold_encoded: rs.cold_encoded,
        cold_kv: rs.cold_kv,
        cold_abs_start: rs.cold_abs_start,
        next_position: abs_position + 1,
        max_window: rs.max_window,
        codec: rs.codec,
    };

    let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        overflow_per_layer.push(updated_rs.clip_layer_overflow(layer));
    }
    if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
        match updated_rs.cold_encoded.as_mut() {
            Some(layers) => {
                for (layer, overflow) in overflow_per_layer.iter().enumerate() {
                    layers[layer].append(updated_rs.codec, overflow);
                }
            }
            None => {
                let hidden = weights.hidden_size;
                let mut layers: Vec<EncodedColdLayer> = Vec::with_capacity(num_layers);
                for overflow in overflow_per_layer.iter() {
                    let mut enc = EncodedColdLayer::empty(hidden);
                    enc.append(updated_rs.codec, overflow);
                    layers.push(enc);
                }
                updated_rs.cold_encoded = Some(layers);
            }
        }
        updated_rs.cold_kv = None;
    }

    Some((last_row(&h_new), updated_rs))
}

fn roundtrip(block: &Array2<f32>, codec: ColdResidualCodec) -> Array2<f32> {
    if block.shape()[0] == 0 {
        return block.clone();
    }
    let mut tmp = EncodedColdLayer::empty(block.shape()[1]);
    tmp.append(codec, block);
    tmp.decode(codec)
}

fn last_row(h: &Array2<f32>) -> Array2<f32> {
    let last = h.shape()[0] - 1;
    h.slice(s![last..=last, ..]).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::CpuBackend;
    use larql_inference::test_utils::{make_test_vindex, make_test_weights};

    #[test]
    fn prefill_walk_returns_finite_hidden() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let result = rs_prefill_codec_walk(
            &weights,
            &index,
            &[0u32, 1, 2],
            None,
            ColdResidualCodec::Bf16,
            &CpuBackend,
        );
        assert_eq!(result.hidden.shape(), &[1, weights.hidden_size]);
        assert!(result.hidden.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn prefill_walk_with_overflow_populates_cold_tier() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let result = rs_prefill_codec_walk(
            &weights,
            &index,
            &[0u32, 1, 2, 3],
            Some(2),
            ColdResidualCodec::Bf16,
            &CpuBackend,
        );
        assert!(result.store.cold_encoded.is_some());
        assert!(result.store.cold_kv.is_some());
    }

    #[test]
    fn decode_walk_extends_position_and_returns_finite() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let prefill = rs_prefill_codec_walk(
            &weights,
            &index,
            &[0u32, 1],
            None,
            ColdResidualCodec::Bf16,
            &CpuBackend,
        );
        assert_eq!(prefill.store.next_position, 2);
        let (h, rs2) =
            rs_decode_step_codec_walk(&weights, &index, 2, prefill.store, &CpuBackend).unwrap();
        assert_eq!(rs2.next_position, 3);
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn decode_walk_with_cold_kv_path() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let prefill = rs_prefill_codec_walk(
            &weights,
            &index,
            &[0u32, 1, 2, 3],
            Some(2),
            ColdResidualCodec::Bf16,
            &CpuBackend,
        );
        assert!(prefill.store.cold_kv.is_some());
        let (h, _) =
            rs_decode_step_codec_walk(&weights, &index, 4, prefill.store, &CpuBackend).unwrap();
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn decode_walk_with_cold_encoded_after_eviction() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let prefill = rs_prefill_codec_walk(
            &weights,
            &index,
            &[0u32, 1, 2, 3],
            Some(2),
            ColdResidualCodec::Bf16,
            &CpuBackend,
        );
        let (_, rs2) =
            rs_decode_step_codec_walk(&weights, &index, 4, prefill.store, &CpuBackend).unwrap();
        // First decode clears cold_kv; second decode exercises the
        // cold_encoded path.
        let (h, _) = rs_decode_step_codec_walk(&weights, &index, 5, rs2, &CpuBackend).unwrap();
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
    }

    #[test]
    fn roundtrip_empty_block() {
        let empty: Array2<f32> = Array2::zeros((0, 8));
        let out = roundtrip(&empty, ColdResidualCodec::Bf16);
        assert_eq!(out.shape(), &[0, 8]);
    }
}
