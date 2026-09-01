//! Cached prefill and decode-step entry points.
//!
//! Split out of `cached.rs`; [`super`] composes the pieces.

#[allow(unused_imports)]
use super::*;
use crate::attention::{
    decode::run_attention_block_decode_step_backend, run_attention_with_kv_backend,
};
use crate::cpu::ops::q4k_q8k_dot::Q8KActivation;
use crate::forward::embed_tokens_pub;
use crate::forward::layer::apply_layer_scalar;
use crate::forward::ple::{apply_per_layer_embedding, precompute_per_layer_inputs};
use crate::forward::run_ffn;
use crate::kquant_forward::tensors::{
    insert_q4k_attn_tensors, insert_q4k_layer_tensors, remove_layer_tensors,
};
use larql_models::ModelWeights;
use ndarray::Array2;

/// Prefill: run the full prompt through every layer once, capturing
/// each layer's post-RoPE K and final V into the returned cache.
/// Returns the `[seq_len, hidden]` hidden state and the populated
/// cache. Caller takes the last row for lm_head.
pub fn predict_kquant_prefill(
    weights: &ModelWeights,
    token_ids: &[u32],
    index: &dyn crate::KvIndex,
) -> (Array2<f32>, CpuKvCache, CachedTimings) {
    predict_kquant_prefill_with_state(weights, token_ids, index, None)
}

/// Prefill with optional per-layer state capture (W1-GPU step 3
/// sibling of [`predict_kquant_decode_step_direct_with_state`]). When
/// `state` is `Some`, populates per-layer `h_in` ([seq_len, hidden]),
/// `k_new` ([seq_len, kv_dim]), `v_new` ([seq_len, kv_dim]) for every
/// position in the prompt — engines (markov_residual,
/// windowed_checkpoint, turbo_quant) use this to seed their state policy
/// from a single prefill pass without a follow-up CPU re-walk. When
/// `state` is `None`, bit-identical to [`predict_kquant_prefill`].
pub fn predict_kquant_prefill_with_state(
    weights: &ModelWeights,
    token_ids: &[u32],
    index: &dyn crate::KvIndex,
    mut state: Option<&mut crate::PerLayerDecodeState>,
) -> (Array2<f32>, CpuKvCache, CachedTimings) {
    let num_layers = weights.num_layers;
    let mut cache: CpuKvCache = vec![None; num_layers];
    let mut timings = CachedTimings::default();
    // Forward-local dequant scratch — per-forward derived state; `weights`
    // stays immutable. Readers resolve via with_scratch (scratch ∪ canonical).
    let mut scratch = larql_models::DequantScratch::new();

    let mut h = embed_tokens_pub(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    for layer in 0..num_layers {
        // q4k-direct prefill: project straight from the vindex's Q4_K/Q6_K bytes
        // (no per-layer f32 dequant). The FFN gate/up/down (`use_q4k_ffn`) and
        // attention Q/K/V/O (`use_q4k_attn`) are gated independently — each needs
        // its interleaved bytes present and the relevant dims 256-aligned (the
        // matmul contraction is `hidden` for Q/K/V/gate/up and `q_dim` for O).
        // The FFN weights are ~4× the attention weights, so the FFN path is the
        // bulk of the saving; the attn path closes the rest.
        const BLK: usize = larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
        let q_dim =
            weights.arch.num_q_heads_for_layer(layer) * weights.arch.head_dim_for_layer(layer);
        let use_q4k_ffn = weights.hidden_size.is_multiple_of(BLK)
            && index.interleaved_kquant_layer_data(layer).is_some();
        let use_q4k_attn = weights.hidden_size.is_multiple_of(BLK)
            && q_dim.is_multiple_of(BLK)
            && index.attn_kquant_layer_data(layer).is_some();

        let t0 = std::time::Instant::now();
        // Dequant only what the q4k-direct paths won't read straight from bytes.
        let inserted = match (use_q4k_attn, use_q4k_ffn) {
            (true, true) => Ok(Vec::new()),
            (false, true) => insert_q4k_attn_tensors(&mut scratch, weights, index, layer),
            _ => insert_q4k_layer_tensors(&mut scratch, weights, index, layer),
        }
        .unwrap_or_else(|err| panic!("{err}"));
        timings.dequant_ms += t0.elapsed().as_secs_f64() * 1000.0;

        // Snapshot pre-attention residual for this layer if engine wants it.
        if let Some(s) = state.as_deref_mut() {
            s.h_in_per_layer
                .push(crate::state_handle::CpuStateHandle::boxed(h.clone()));
        }

        // Attention with K/V capture. When `use_q4k_attn`, Q/K/V/O project
        // straight from the Q4_K/Q6_K bytes (empty scratch — norms still come
        // from canonical weights); else the dequantised f32 view path.
        let attn_index = if use_q4k_attn { Some(index) } else { None };
        let (h_post_attn, k_rope, v_final) = match run_attention_with_kv_backend(
            larql_models::WeightsView::with_scratch(weights, &scratch),
            &h,
            layer,
            None,
            attn_index,
        ) {
            Some(t) => t,
            None => {
                remove_layer_tensors(&mut scratch, inserted);
                return (h, cache, timings);
            }
        };

        if let Some(s) = state.as_deref_mut() {
            // Prefill K/V for THIS layer = full seq_len × kv_dim.
            s.k_new_per_layer
                .push(crate::state_handle::CpuStateHandle::boxed(k_rope.clone()));
            s.v_new_per_layer
                .push(crate::state_handle::CpuStateHandle::boxed(v_final.clone()));
        }

        let h_post_ffn = if use_q4k_ffn {
            let ffn = crate::ffn::Q4kMatmulFfn { weights, index };
            run_ffn(weights, &h_post_attn, layer, &ffn, false).0
        } else {
            let ffn = crate::ffn::ViewFfn {
                view: larql_models::WeightsView::with_scratch(weights, &scratch),
            };
            run_ffn(weights, &h_post_attn, layer, &ffn, false).0
        };
        let mut h_out =
            apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
        apply_layer_scalar(weights, &mut h_out, layer);

        remove_layer_tensors(&mut scratch, inserted);

        cache[layer] = Some((k_rope, v_final));
        h = h_out;
    }

    (h, cache, timings)
}

/// Decode step: run a single new token through every layer using the
/// prefill cache. Each layer's cache entry is appended to in place.
/// Returns the new `[1, hidden]` hidden state for lm_head.
///
/// `abs_position` is the absolute RoPE position of the new token —
/// `prompt_len + steps_already_decoded`. The caller maintains this
/// counter (typical: `prompt_len + step_index` starting at 0).
pub fn predict_kquant_decode_step(
    weights: &ModelWeights,
    token_id: u32,
    index: &dyn crate::KvIndex,
    cache: &mut CpuKvCache,
    abs_position: usize,
) -> Option<(Array2<f32>, CachedTimings)> {
    let num_layers = weights.num_layers;
    if cache.len() != num_layers {
        return None;
    }
    let mut timings = CachedTimings::default();
    let mut scratch = larql_models::DequantScratch::new();

    // 1-row embed + 1-row PLE for the new token.
    let mut h = embed_tokens_pub(weights, &[token_id]);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, &[token_id]);

    for layer in 0..num_layers {
        let t0 = std::time::Instant::now();
        let inserted = insert_q4k_layer_tensors(&mut scratch, weights, index, layer)
            .unwrap_or_else(|err| panic!("{err}"));
        timings.dequant_ms += t0.elapsed().as_secs_f64() * 1000.0;

        let kv_entry = cache[layer].as_ref();
        let (h_post_attn, new_kv) = match run_attention_block_decode_step_backend(
            larql_models::WeightsView::with_scratch(weights, &scratch),
            &h,
            layer,
            kv_entry,
            abs_position,
            None,
        ) {
            Some(t) => t,
            None => {
                remove_layer_tensors(&mut scratch, inserted);
                return None;
            }
        };
        cache[layer] = Some(new_kv);

        let ffn = crate::ffn::ViewFfn {
            view: larql_models::WeightsView::with_scratch(weights, &scratch),
        };
        let (h_post_ffn, _) = run_ffn(weights, &h_post_attn, layer, &ffn, false);
        let mut h_out =
            apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
        apply_layer_scalar(weights, &mut h_out, layer);

        remove_layer_tensors(&mut scratch, inserted);

        h = h_out;
    }

    Some((h, timings))
}

impl CachedTimings {
    /// Merge another timing block into self. Useful for accumulating
    /// per-step decode timings across a generation loop.
    pub fn add(&mut self, other: CachedTimings) {
        self.merge(other);
    }
}

// ── Phase 2: dequant-free decode step ───────────────────────────────────
//
// `predict_kquant_decode_step` (above) still pays the per-step Q4_K/Q6_K →
// f32 dequant cost via `insert_q4k_layer_tensors`. Profiling showed
// dequant is ~93% of CPU forward time even with the KV cache wired —
// gemm and attention are a small slice. This module routes Q/K/V/O and
// gate/up/down projections straight through `backend.quant_matvec`
// (CPU `q4k_matvec_into` / `q6k_matvec_into`), skipping the dequant
// staging entirely.

/// Format-aware Q*K × Q8_K matvec used by the production decode path.
/// Uses NEON `sdot` (Q4_K) or `vmlal_s8` (Q6_K) under the hood — ~2-3×
/// the f32-FMA throughput of `backend.quant_matvec`. Returns `None`
/// for any unsupported format (caller falls through to dequant).
pub(crate) fn matvec_q4k_or_q6k_q8k(
    bytes: &[u8],
    format: &str,
    x_q8k: &Q8KActivation,
    rows: usize,
    cols: usize,
) -> Option<Vec<f32>> {
    if rows == 0 || cols == 0 {
        return Some(vec![0.0f32; rows]);
    }
    const ELEMS_PER_BLOCK: usize = 256;
    if !cols.is_multiple_of(ELEMS_PER_BLOCK) {
        return None;
    }
    // Pre-flight length check only (the actual matvec recomputes this stride
    // internally). Gate on the kernel-backed formats via the `FormatRoute`
    // registry and take the packed row length from the format helper instead
    // of re-spelling `(cols/256)*144`.
    let bytes_per_row = match crate::QuantFormat::from_registry_tag(format) {
        Some(f) if f.route().q8k_matvec.is_some() => f.packed_matrix_bytes(1, cols)?,
        _ => return None,
    };
    if bytes.len() < rows * bytes_per_row {
        return None;
    }

    // `q4k_q8k_matvec_into` (larql-compute) is a single-threaded
    // per-row loop. Wrap it with `par_chunks_mut(CHUNK_ROWS)` here so
    // every Q4_K/Q6_K × Q8_K matvec on the decode path scales across
    // the 11 perf cores on M3 Max — matching the rayon strategy of
    // `q4k_matvec_into` in `q4_common.rs`. Without this, decode runs
    // single-threaded and the sdot path actually regresses vs the
    // (rayon-parallel) f32 path despite each row being faster.
    let mut out = vec![0.0f32; rows];
    crate::cpu::ops::q4k_q8k_dot::q4k_q8k_matvec_parallel(
        &mut out, x_q8k, bytes, rows, cols, format,
    )
    .unwrap_or_else(|e| panic!("matvec_q4k_or_q6k_q8k: refused after pre-flight checks: {e}"));
    Some(out)
}

/// True when every Q/K/V/O + gate/up/down slice for `layer` is in a
/// format the direct-matvec path knows how to handle. Used to gate
/// per-layer routing: the cached decode step prefers the direct
/// matvec when this returns true and falls back to the dequant path
/// otherwise (e.g. Q4_KF layers, padded down projections).
pub(crate) fn layer_supports_direct_matvec(index: &dyn crate::KvIndex, layer: usize) -> bool {
    // "Direct-matvec-capable" = the tag resolves to a format with a Q8K
    // matvec kernel in the `FormatRoute` registry (Q4_K/Q6_K today).
    let has_q8k_kernel = |tag: &str| {
        crate::QuantFormat::from_registry_tag(tag).is_some_and(|f| f.route().q8k_matvec.is_some())
    };
    let attn = match index.attn_kquant_layer_data(layer) {
        Some(a) => a,
        None => return false,
    };
    for (_, fmt) in attn.iter() {
        if !has_q8k_kernel(fmt) {
            return false;
        }
    }
    let ffn = match index.interleaved_kquant_layer_data(layer) {
        Some(f) => f,
        None => return false,
    };
    for (_, fmt) in ffn.iter() {
        if !has_q8k_kernel(fmt) {
            return false;
        }
    }
    // The down projection in the FFN is sometimes stored with a padded
    // intermediate dim (rounded up to a 256-multiple). `q4k_matvec_into`
    // rejects non-multiple `cols`, which would silently zero the
    // output — refuse the direct path so the dequant fallback runs.
    let intermediate = index.num_features(layer);
    intermediate.is_multiple_of(larql_models::quant::ggml::Q4_K_BLOCK_ELEMS)
}
