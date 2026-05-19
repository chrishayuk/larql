//! Multi-token extend with prior K,V checkpoint.
//!
//! Runs a CPU/GPU forward pass over new tokens, seeding each layer's attention
//! with an optional prior K,V cache (the window boundary checkpoint).

use larql_compute::ComputeBackend;
use larql_vindex::VectorIndex;
use ndarray::Array2;

use larql_inference::attention::{run_attention_block_decode_step_backend, SharedKV};
use larql_inference::ffn::BackendFfn;
use larql_inference::forward::{embed_tokens_pub, run_ffn};
use larql_inference::model::ModelWeights;
use larql_inference::vindex::{WalkFfn, WalkFfnConfig};

pub struct ExtendOutput {
    /// Hidden state at the last processed token, shape (1, hidden).
    pub last_hidden: Array2<f32>,
    /// Per-layer full K,V cache covering `[prior_tokens, new_tokens]`.
    pub kv_cache: Vec<SharedKV>,
    /// Per-layer last-row K,V ready to save as the next boundary checkpoint.
    pub new_checkpoint: Vec<SharedKV>,
}

/// Run the decoder forward over `token_ids` seeded with an optional prior K,V
/// checkpoint at each layer. Matmuls route through `backend`.
///
/// `abs_start` is the absolute position of the *first new token*.
pub fn rs_extend_from_checkpoint(
    weights: &ModelWeights,
    token_ids: &[u32],
    prior_kv: Vec<SharedKV>,
    abs_start: usize,
) -> Option<ExtendOutput> {
    rs_extend_from_checkpoint_backend(
        weights,
        token_ids,
        prior_kv,
        abs_start,
        &larql_compute::CpuBackend,
    )
}

/// Backend-dispatched variant of [`rs_extend_from_checkpoint`].
///
/// Takes `prior_kv` by value so the per-token extend loop can mutate it
/// in place. Cloning the prior K/V per step is O(window²) total over a
/// full window — a real overhead on growing caches.
pub fn rs_extend_from_checkpoint_backend(
    weights: &ModelWeights,
    token_ids: &[u32],
    prior_kv: Vec<SharedKV>,
    abs_start: usize,
    backend: &dyn ComputeBackend,
) -> Option<ExtendOutput> {
    let num_layers = weights.num_layers;

    if token_ids.is_empty() {
        return None;
    }
    if prior_kv.len() != num_layers {
        return None;
    }

    let mut kv_cache: Vec<SharedKV> = prior_kv;
    let mut last_hidden: Option<Array2<f32>> = None;

    for (i, &token_id) in token_ids.iter().enumerate() {
        let abs_position = abs_start + i;
        let mut h = embed_tokens_pub(weights, &[token_id]);

        for (layer, kv_slot) in kv_cache.iter_mut().enumerate() {
            let kv_entry: Option<&SharedKV> = if kv_slot.0.shape()[0] > 0 {
                Some(kv_slot)
            } else {
                None
            };

            let (h_post_attn, new_kv) = run_attention_block_decode_step_backend(
                weights,
                &h,
                layer,
                kv_entry,
                abs_position,
                Some(backend),
            )?;

            let bffn = BackendFfn { weights, backend };
            let (h_out, _) = run_ffn(weights, &h_post_attn, layer, &bffn, false);
            h = h_out;
            *kv_slot = new_kv;
        }

        last_hidden = Some(h);
    }

    let new_checkpoint: Vec<SharedKV> = kv_cache
        .iter()
        .map(|(k, v)| {
            let n = k.shape()[0];
            let last_k = k.slice(ndarray::s![n - 1..n, ..]).to_owned();
            let last_v = v.slice(ndarray::s![n - 1..n, ..]).to_owned();
            (last_k, last_v)
        })
        .collect();

    Some(ExtendOutput {
        last_hidden: last_hidden?,
        kv_cache,
        new_checkpoint,
    })
}

/// CPU Q4K variant of [`rs_extend_from_checkpoint_backend`].
///
/// Uses `WalkFfn` (reads Q4K bytes directly from `index`) for FFN instead of
/// `BackendFfn` (needs f32 tensors in `weights.tensors`). Attention projection
/// uses the dequantised f32 tensors already inserted by
/// `ensure_attn_tensors_dequantised`. Call that before this function.
pub fn rs_extend_from_checkpoint_q4k(
    weights: &ModelWeights,
    index: &VectorIndex,
    token_ids: &[u32],
    prior_kv: Vec<SharedKV>,
    abs_start: usize,
    backend: &dyn ComputeBackend,
) -> Option<ExtendOutput> {
    let num_layers = weights.num_layers;

    if token_ids.is_empty() {
        return None;
    }
    if prior_kv.len() != num_layers {
        return None;
    }

    let mut kv_cache: Vec<SharedKV> = prior_kv;
    let mut last_hidden: Option<Array2<f32>> = None;

    // Hoist WalkFfn out of both loops. Previously this rebuilt the
    // WalkFfn once per (token, layer) — N×34 times per extend call.
    // It's now once total. WalkFfn carries no per-(token,layer) state.
    let walk_ffn = WalkFfn::from_config(weights, index, WalkFfnConfig::dense(num_layers))
        .with_backend(backend);

    // Per-stage timing. Enabled by `LARQL_INSTRUMENT_UNLIMITED=1`.
    // Walks the same layers MarkovResidual's q4k path instruments — kept
    // shape-compatible so output comparisons line up.
    let instrument = std::env::var("LARQL_INSTRUMENT_UNLIMITED").is_ok();
    let mut t_embed = 0.0f64;
    let mut t_attention = 0.0f64;
    let mut t_ffn = 0.0f64;
    let mut t_attn_helper_misses = 0usize;
    let mut t_ffn_helper_misses = 0usize;

    for (i, &token_id) in token_ids.iter().enumerate() {
        let abs_position = abs_start + i;
        let t_embed_start = if instrument {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let mut h = embed_tokens_pub(weights, &[token_id]);
        if let Some(start) = t_embed_start {
            t_embed += start.elapsed().as_secs_f64() * 1000.0;
        }

        for (layer, kv_slot) in kv_cache.iter_mut().enumerate() {
            let kv_entry: Option<&SharedKV> = if kv_slot.0.shape()[0] > 0 {
                Some(kv_slot)
            } else {
                None
            };

            // Try production native-quantised attention helper first;
            // fall back to f32 path. Same pattern as MarkovResidual.
            let t_attn_start = if instrument {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let attn_native = larql_inference::vindex::attention_decode_step_native(
                weights,
                index,
                backend,
                &h,
                layer,
                kv_entry,
                abs_position,
            );
            if attn_native.is_none() && instrument {
                t_attn_helper_misses += 1;
            }
            let (h_post_attn, new_kv) = attn_native.or_else(|| {
                run_attention_block_decode_step_backend(
                    weights,
                    &h,
                    layer,
                    kv_entry,
                    abs_position,
                    Some(backend),
                )
            })?;
            if let Some(start) = t_attn_start {
                t_attention += start.elapsed().as_secs_f64() * 1000.0;
            }

            // Native-quantised FFN; falls back to WalkFfn (which falls
            // further to dense f32 if no sparse features). The native
            // path is ~100× faster on Gemma 3 4B Q4K — see
            // `bench/baselines/cpu/async-dispatch-2026-05-16.md`.
            let t_ffn_start = if instrument {
                Some(std::time::Instant::now())
            } else {
                None
            };
            let ffn_native = larql_inference::vindex::ffn_decode_step_native(
                weights,
                index,
                backend,
                &h_post_attn,
                layer,
            );
            if ffn_native.is_none() && instrument {
                t_ffn_helper_misses += 1;
            }
            let h_out = ffn_native.unwrap_or_else(|| {
                let (h, _) = run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
                h
            });
            if let Some(start) = t_ffn_start {
                t_ffn += start.elapsed().as_secs_f64() * 1000.0;
            }
            h = h_out;
            *kv_slot = new_kv;
        }

        last_hidden = Some(h);
    }

    if instrument {
        let total = t_embed + t_attention + t_ffn;
        eprintln!(
            "[unlimited-context/extend_q4k] tokens={} layers={num_layers} \
             embed={t_embed:.2}ms attention={t_attention:.2}ms ffn={t_ffn:.2}ms \
             total={total:.2}ms (attn_helper miss/ffn_helper miss={t_attn_helper_misses}/{t_ffn_helper_misses})",
            token_ids.len()
        );
    }

    let new_checkpoint: Vec<SharedKV> = kv_cache
        .iter()
        .map(|(k, v)| {
            let n = k.shape()[0];
            let last_k = k.slice(ndarray::s![n - 1..n, ..]).to_owned();
            let last_v = v.slice(ndarray::s![n - 1..n, ..]).to_owned();
            (last_k, last_v)
        })
        .collect();

    Some(ExtendOutput {
        last_hidden: last_hidden?,
        kv_cache,
        new_checkpoint,
    })
}

/// Build an empty (zero-row) K,V seed for use when no prior checkpoint exists.
pub fn empty_prior(weights: &ModelWeights) -> Vec<SharedKV> {
    let arch = &*weights.arch;
    (0..weights.num_layers)
        .map(|layer| {
            let kv_dim = arch.num_kv_heads_for_layer(layer) * arch.head_dim_for_layer(layer);
            (
                Array2::<f32>::zeros((0, kv_dim)),
                Array2::<f32>::zeros((0, kv_dim)),
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_inference::forward::hidden_to_raw_logits;
    use larql_inference::test_utils::make_test_weights;

    // ── empty_prior ───────────────────────────────────────────────────────────

    #[test]
    fn empty_prior_shape_per_layer() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        assert_eq!(prior.len(), weights.num_layers);
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        for (k, v) in &prior {
            assert_eq!(k.shape(), &[0, kv_dim]);
            assert_eq!(v.shape(), &[0, kv_dim]);
        }
    }

    // ── rs_extend_from_checkpoint ─────────────────────────────────────────────

    #[test]
    fn extend_empty_tokens_returns_none() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let result = rs_extend_from_checkpoint(&weights, &[], prior, 0);
        assert!(result.is_none(), "empty token_ids should return None");
    }

    #[test]
    fn extend_wrong_prior_len_returns_none() {
        let weights = make_test_weights();
        // prior has 0 layers but model has 2 — mismatch
        let result = rs_extend_from_checkpoint(&weights, &[0u32], Vec::new(), 0);
        assert!(result.is_none(), "prior length mismatch should return None");
    }

    #[test]
    fn extend_single_token_from_empty_prior() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let output = rs_extend_from_checkpoint(&weights, &[0u32], prior, 0)
            .expect("single token extend should succeed");
        assert_eq!(output.last_hidden.shape(), &[1, weights.hidden_size]);
        assert!(output.last_hidden.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn extend_kv_cache_grows_with_each_token() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let output =
            rs_extend_from_checkpoint(&weights, &[0u32, 1, 2], prior, 0).expect("3-token extend");
        // After 3 tokens from empty prior, K has 3 rows per layer
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        for (k, v) in &output.kv_cache {
            assert_eq!(k.shape(), &[3, kv_dim], "K should have 3 rows");
            assert_eq!(v.shape(), &[3, kv_dim], "V should have 3 rows");
        }
    }

    #[test]
    fn extend_checkpoint_is_last_row_of_kv_cache() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let output =
            rs_extend_from_checkpoint(&weights, &[0u32, 1], prior, 0).expect("2-token extend");
        // new_checkpoint should be the last row of each K/V
        for (layer, ((k_cache, v_cache), (k_ckpt, v_ckpt))) in output
            .kv_cache
            .iter()
            .zip(output.new_checkpoint.iter())
            .enumerate()
        {
            let n = k_cache.shape()[0];
            let last_k = k_cache.row(n - 1).to_vec();
            let ckpt_k = k_ckpt.row(0).to_vec();
            for (a, b) in last_k.iter().zip(ckpt_k.iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "layer {layer}: checkpoint K doesn't match last K cache row"
                );
            }
            let _ = (v_cache, v_ckpt); // symmetry — trust by shape
        }
    }

    #[test]
    fn extend_abs_start_shifts_rope() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let out0 = rs_extend_from_checkpoint(&weights, &[0u32], prior.clone(), 0).unwrap();
        let out5 = rs_extend_from_checkpoint(&weights, &[0u32], prior, 5).unwrap();
        // Different abs_start → different RoPE → different K
        let k0 = &out0.kv_cache[0].0;
        let k5 = &out5.kv_cache[0].0;
        let diff: f32 = k0.iter().zip(k5.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 0.0,
            "different abs_start should produce different K (RoPE)"
        );
    }

    #[test]
    fn extend_output_logits_are_finite() {
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let output = rs_extend_from_checkpoint(&weights, &[0u32], prior, 0).unwrap();
        let logits = hidden_to_raw_logits(&weights, &output.last_hidden);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn extend_seeded_from_checkpoint_matches_empty_start() {
        // Extending from a non-empty checkpoint should not panic and should be finite.
        let weights = make_test_weights();
        let prior = empty_prior(&weights);
        let first = rs_extend_from_checkpoint(&weights, &[0u32], prior, 0).unwrap();
        // Use the checkpoint from the first extend as the prior for the second
        let second = rs_extend_from_checkpoint(&weights, &[1u32], first.new_checkpoint.clone(), 1)
            .expect("extend from non-empty prior");
        assert_eq!(second.last_hidden.shape(), &[1, weights.hidden_size]);
        assert!(second.last_hidden.iter().all(|v| v.is_finite()));
    }

    // ── rs_extend_from_checkpoint_q4k (vindex-backed FFN path) ───────────────

    #[test]
    fn extend_q4k_empty_tokens_returns_none() {
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let prior = empty_prior(&weights);
        let out = rs_extend_from_checkpoint_q4k(&weights, &index, &[], prior, 0, &*backend);
        assert!(out.is_none(), "empty token_ids should return None");
    }

    #[test]
    fn extend_q4k_wrong_prior_len_returns_none() {
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let out =
            rs_extend_from_checkpoint_q4k(&weights, &index, &[0u32], Vec::new(), 0, &*backend);
        assert!(out.is_none(), "prior length mismatch should return None");
    }

    #[test]
    fn extend_q4k_grows_kv_cache_and_returns_finite() {
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let prior = empty_prior(&weights);
        let out =
            rs_extend_from_checkpoint_q4k(&weights, &index, &[0u32, 1, 2], prior, 0, &*backend)
                .expect("3-token Q4K extend");
        assert_eq!(out.last_hidden.shape(), &[1, weights.hidden_size]);
        assert!(out.last_hidden.iter().all(|v| v.is_finite()));
        // After 3 tokens from an empty prior, each layer's K/V has 3 rows.
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        for (k, v) in &out.kv_cache {
            assert_eq!(k.shape(), &[3, kv_dim]);
            assert_eq!(v.shape(), &[3, kv_dim]);
        }
        assert_eq!(out.new_checkpoint.len(), weights.num_layers);
    }

    #[test]
    fn extend_q4k_seeded_from_prior_matches_shape() {
        let weights = make_test_weights();
        let index = larql_inference::test_utils::make_test_vindex(&weights);
        let backend = larql_compute::cpu_backend();
        let first = rs_extend_from_checkpoint_q4k(
            &weights,
            &index,
            &[0u32, 1],
            empty_prior(&weights),
            0,
            &*backend,
        )
        .expect("first extend");
        let second = rs_extend_from_checkpoint_q4k(
            &weights,
            &index,
            &[2u32],
            first.kv_cache.clone(),
            2,
            &*backend,
        )
        .expect("extend over prior kv");
        assert_eq!(second.last_hidden.shape(), &[1, weights.hidden_size]);
        let kv_dim = weights.num_kv_heads * weights.head_dim;
        for (k, _) in &second.kv_cache {
            assert_eq!(k.shape(), &[3, kv_dim], "prior(2) + new(1) = 3 rows");
        }
    }
}
