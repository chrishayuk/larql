//! Core forward primitives for `MarkovResidualCodecEngine`.
//!
//! Mirrors `markov_residual::compute` with the cold tier swapped to a
//! codec-encoded representation. All forward compute (attention, FFN, K/V
//! recomputation) delegates to `larql_inference` / the production
//! `recompute_kv`. The differences are isolated to cold-tier read/write paths.

use larql_compute::ComputeBackend;
use larql_inference::attention::{
    run_attention_block_decode_step_backend, run_attention_with_kv_backend, SharedKV,
};
use larql_inference::ffn::BackendFfn;
use larql_inference::forward::{embed_tokens_pub, run_ffn};
use larql_inference::model::ModelWeights;
use ndarray::{s, Array2};

use crate::engines::markov_residual::recompute_kv;
use crate::engines::markov_residual_codec::codec::ColdResidualCodec;
use crate::engines::markov_residual_codec::store::{EncodedColdLayer, RsStoreCodec};

pub struct RsPrefillResultCodec {
    pub hidden: Array2<f32>,
    pub store: RsStoreCodec,
}

pub fn rs_prefill_codec(
    weights: &ModelWeights,
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

    for layer in 0..num_layers {
        stored.push(h.clone());
        let (h_post_attn, _k, _v) = run_attention_with_kv_backend(weights, &h, layer, be)
            .expect("attention failed during MarkovResidualCodec prefill");
        let bffn = BackendFfn { weights, backend };
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &bffn, false);
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

    // Clip overflow per layer; encode and pre-compute K/V for cold once.
    let mut overflow_per_layer: Vec<Array2<f32>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        overflow_per_layer.push(rs.clip_layer_overflow(layer));
    }
    if overflow_per_layer.first().map_or(0, |c| c.shape()[0]) > 0 {
        let mut encoded_layers: Vec<EncodedColdLayer> = Vec::with_capacity(num_layers);
        let mut cold_kv: Vec<SharedKV> = Vec::with_capacity(num_layers);
        for (layer, overflow) in overflow_per_layer.iter().enumerate() {
            let decoded_overflow = roundtrip(overflow, codec);
            let (k, v) = recompute_kv(weights, &decoded_overflow, layer, 0, backend, None)
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

pub fn rs_decode_step_codec(
    weights: &ModelWeights,
    new_token_id: u32,
    rs: RsStoreCodec,
    backend: &dyn ComputeBackend,
) -> Option<(Array2<f32>, RsStoreCodec)> {
    let num_layers = weights.num_layers;
    let abs_position = rs.next_position;
    let mut h_new = embed_tokens_pub(weights, &[new_token_id]);
    let mut new_stored: Vec<Array2<f32>> = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let h_hot = &rs.stored[layer];
        let s_hot = h_hot.shape()[0];
        let hot_abs_start = abs_position.saturating_sub(s_hot);

        let (k_full, v_full) = if let Some(cold_kv) = &rs.cold_kv {
            let (k_cold, v_cold) = &cold_kv[layer];
            let (k_hot, v_hot) = recompute_kv(weights, h_hot, layer, hot_abs_start, backend, None)?;
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
            let (h_full, full_abs_start) = if let Some(cold_layers) = &rs.cold_encoded {
                let enc = &cold_layers[layer];
                if enc.n_positions > 0 {
                    let decoded = enc.decode(rs.codec);
                    let hidden = h_hot.shape()[1];
                    let mut combined = Array2::<f32>::zeros((decoded.shape()[0] + s_hot, hidden));
                    combined
                        .slice_mut(s![..decoded.shape()[0], ..])
                        .assign(&decoded);
                    combined
                        .slice_mut(s![decoded.shape()[0].., ..])
                        .assign(h_hot);
                    (combined, rs.cold_abs_start)
                } else {
                    (h_hot.clone(), hot_abs_start)
                }
            } else {
                (h_hot.clone(), hot_abs_start)
            };
            let (k, v) = recompute_kv(weights, &h_full, layer, full_abs_start, backend, None)?;
            (k, v)
        };

        new_stored.push(h_new.clone());

        let (h_post_attn, _new_kv) = run_attention_block_decode_step_backend(
            weights,
            &h_new,
            layer,
            Some(&(k_full, v_full)),
            abs_position,
            Some(backend),
        )?;

        let bffn = BackendFfn { weights, backend };
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &bffn, false);
        h_new = h_out;
    }

    // Append the new row to each layer's hot tier.
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

    // Clip overflow into encoded cold tier; clear cold_kv to force recompute.
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

/// Apply the codec roundtrip to a block. Used during prefill cold setup so
/// that the cold K/V we precompute is consistent with what `decode` would
/// later produce.
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
    use larql_inference::test_utils::make_test_weights;

    #[test]
    fn prefill_returns_finite_hidden() {
        let weights = make_test_weights();
        let result = rs_prefill_codec(
            &weights,
            &[0u32, 1, 2],
            None,
            ColdResidualCodec::Bf16,
            &CpuBackend,
        );
        assert_eq!(result.hidden.shape(), &[1, weights.hidden_size]);
        assert!(result.hidden.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn prefill_no_window_does_not_create_cold_tier() {
        let weights = make_test_weights();
        let result = rs_prefill_codec(
            &weights,
            &[0u32, 1],
            None,
            ColdResidualCodec::Bf16,
            &CpuBackend,
        );
        assert!(result.store.cold_encoded.is_none());
        assert!(result.store.cold_kv.is_none());
    }

    #[test]
    fn prefill_with_overflow_creates_encoded_cold_tier() {
        let weights = make_test_weights();
        let result = rs_prefill_codec(
            &weights,
            &[0u32, 1, 2, 3],
            Some(2),
            ColdResidualCodec::Bf16,
            &CpuBackend,
        );
        assert!(result.store.cold_encoded.is_some());
        assert!(result.store.cold_kv.is_some());
        let layers = result.store.cold_encoded.as_ref().unwrap();
        assert_eq!(layers.len(), weights.num_layers);
        // 4 tokens, window=2 → 2 cold positions per layer.
        for l in layers {
            assert_eq!(l.n_positions, 2);
            assert_eq!(l.payload.len(), 2 * weights.hidden_size * 2);
        }
    }

    #[test]
    fn decode_step_extends_position() {
        let weights = make_test_weights();
        let prefill = rs_prefill_codec(
            &weights,
            &[0u32, 1],
            None,
            ColdResidualCodec::Bf16,
            &CpuBackend,
        );
        assert_eq!(prefill.store.next_position, 2);
        let (_, rs2) = rs_decode_step_codec(&weights, 2, prefill.store, &CpuBackend).unwrap();
        assert_eq!(rs2.next_position, 3);
    }

    #[test]
    fn decode_with_cold_kv_path_produces_finite_output() {
        let weights = make_test_weights();
        let prefill = rs_prefill_codec(
            &weights,
            &[0u32, 1, 2, 3],
            Some(2),
            ColdResidualCodec::Bf16,
            &CpuBackend,
        );
        assert!(prefill.store.cold_kv.is_some());
        let (h, _) = rs_decode_step_codec(&weights, 4, prefill.store, &CpuBackend).unwrap();
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn decode_with_cold_encoded_path_produces_finite_output() {
        // After enough decode steps, the post-eviction cold_kv-clear path is
        // exercised (we read from cold_encoded directly via decode).
        let weights = make_test_weights();
        let prefill = rs_prefill_codec(
            &weights,
            &[0u32, 1, 2, 3],
            Some(2),
            ColdResidualCodec::Bf16,
            &CpuBackend,
        );
        let (_, rs2) = rs_decode_step_codec(&weights, 4, prefill.store, &CpuBackend).unwrap();
        // Second decode: cold_kv was cleared by overflow at the first decode,
        // so this step exercises the cold_encoded recompute branch.
        let (h, _) = rs_decode_step_codec(&weights, 5, rs2, &CpuBackend).unwrap();
        assert_eq!(h.shape(), &[1, weights.hidden_size]);
        assert!(h.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn roundtrip_empty_block_short_circuits() {
        let empty: Array2<f32> = Array2::zeros((0, 8));
        let out = roundtrip(&empty, ColdResidualCodec::Bf16);
        assert_eq!(out.shape(), &[0, 8]);
    }

    #[test]
    fn roundtrip_preserves_within_bf16_precision() {
        let block =
            Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let out = roundtrip(&block, ColdResidualCodec::Bf16);
        for (orig, got) in block.iter().zip(out.iter()) {
            assert!((orig - got).abs() < 0.1);
        }
    }
}
