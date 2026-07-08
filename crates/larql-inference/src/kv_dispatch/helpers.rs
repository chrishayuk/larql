//! Reusable prefill + decode helpers that orchestrate the per-layer
//! loop via [`KvDispatch`] primitives.
//!
//! These are the engine-facing equivalents of
//! [`crate::forward::kv_prefill_run`] and
//! [`crate::forward::kv_decode_step_run`], rewritten to call
//! `backend.attention_prefill` / `backend.attention_step` per layer
//! instead of the direct `run_attention_*` functions.
//!
//! **Parity:** the helpers below produce bit-identical output to the
//! legacy `kv_prefill_run` / `kv_decode_step_run` when driven against
//! [`super::cpu::CpuKvHandle`] (verified in this file's
//! tests). Engines migrate from the legacy helpers to these helpers
//! in Step 3c of the ComputeBackend redesign.
//!
//! Hooks are not threaded through these helpers — the existing
//! hooked decode path
//! ([`crate::forward::generate_cached_hooked`]) keeps using the legacy
//! helpers because the trait surface doesn't carry `LayerHook`.
//! That's by design (`compute-backend-redesign.md` §4.2 non-goals).

use ndarray::Array2;

use super::{EngineBackend, KvHandle};
use crate::async_compute_backend::AsyncComputeBackend;
use crate::ffn::FfnBackend;
use crate::forward::layer::apply_layer_scalar;
use crate::forward::ple::{apply_per_layer_embedding, precompute_per_layer_inputs};
use crate::forward::{embed_tokens_pub, run_ffn};

/// Per-layer FFN dispatch for the KV-cached engine path, MoE-aware.
///
/// On hybrid-MoE architectures, try the backend's
/// [`FfnBackend::forward_moe_full_layer`] hook first — it returns the full
/// layer output (dense `h1` + experts `h2` + combine + outer-norm). A
/// remote-MoE backend ([`crate::ffn::RemoteMoeFfn`]) implements it to
/// dispatch experts to the shards, giving CPU `--moe-shards` a real KV
/// cache. When the hook declines (`None`) — or the model is dense — fall
/// back to the standard dense FFN, preserving prior behaviour exactly.
fn ffn_or_moe_layer(
    weights: larql_models::WeightsView,
    h_post_attn: &Array2<f32>,
    layer: usize,
    ffn: &dyn FfnBackend,
) -> Array2<f32> {
    if weights.arch.is_hybrid_moe() {
        if let Some(h_out) = ffn.forward_moe_full_layer(layer, h_post_attn) {
            return h_out;
        }
    }
    run_ffn(&weights, h_post_attn, layer, ffn, false).0
}

/// Prefill the K/V cache through every layer using `backend`'s
/// [`KvDispatch::attention_prefill`] intent. Returns the last row of
/// the post-FFN hidden state plus per-layer K/V handles.
///
/// `window` is passed through to the backend per layer — backends with
/// windowed-attention shader variants may use it; CPU backends ignore
/// it (the cache simply isn't clipped after prefill on this path —
/// callers that want a clipped prefill should call
/// [`KvDispatch::clip_kv`] per-layer after this returns).
pub fn kv_prefill_via_dispatch(
    backend: &dyn EngineBackend,
    weights: larql_models::WeightsView,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    window: Option<usize>,
    index: Option<&larql_vindex::VectorIndex>,
) -> Option<(Array2<f32>, Vec<KvHandle>)> {
    if prompt_ids.is_empty() {
        return None;
    }
    let h = embed_tokens_pub(&weights, prompt_ids);
    // Per-Layer Embedding inputs for Gemma-4 archs (empty Vec — and a
    // per-layer no-op — for everything else). Matches `kv_prefill_run`.
    let ple_inputs = precompute_per_layer_inputs(&weights, &h, prompt_ids);
    prefill_from_hidden_inner(backend, weights, ffn, &h, window, index, &ple_inputs)
}

/// Multi-modal-aware peer of [`kv_prefill_via_dispatch`]. Takes
/// pre-built initial hidden state (e.g. from
/// `larql_compute::forward::embed_plan` on an `EmbeddingPlan` that mixes
/// `Tokens` and `Precomputed` chunks) and drives the rest of prefill
/// unchanged.
///
/// The text-only call `kv_prefill_via_dispatch(prompt_ids)` and
/// `kv_prefill_from_hidden_via_dispatch(embed_tokens_pub(prompt_ids))`
/// produce bit-identical output by construction — the former is a
/// two-line wrapper around the latter. Pinned by tests at the bottom
/// of this module.
/// PLE caveat: this entry point has no token ids, so Gemma-4 Per-Layer
/// Embeddings cannot be computed here — it passes an empty PLE slab
/// (per-layer no-op). Gemma-4 callers must use the token-id entry
/// point; MM (Gemma-3-only today) is unaffected because Gemma 3 has no
/// PLE. `layer_scalar` IS still applied (weights-driven, no token ids
/// needed).
pub fn kv_prefill_from_hidden_via_dispatch(
    backend: &dyn EngineBackend,
    weights: larql_models::WeightsView,
    ffn: &dyn FfnBackend,
    initial_hidden: &Array2<f32>,
    window: Option<usize>,
    index: Option<&larql_vindex::VectorIndex>,
) -> Option<(Array2<f32>, Vec<KvHandle>)> {
    prefill_from_hidden_inner(backend, weights, ffn, initial_hidden, window, index, &[])
}

/// Shared prefill body. Runs, per layer: attention → FFN →
/// per-layer embedding → layer_scalar — all four steps are required
/// for Gemma-4 correctness (see `larql-compute::forward::layer`), and
/// the last two are no-ops on every other arch. Mirrors
/// `kv_prefill_run` in `larql-kv/src/generation.rs`.
fn prefill_from_hidden_inner(
    backend: &dyn EngineBackend,
    weights: larql_models::WeightsView,
    ffn: &dyn FfnBackend,
    initial_hidden: &Array2<f32>,
    window: Option<usize>,
    index: Option<&larql_vindex::VectorIndex>,
    ple_inputs: &[Array2<f32>],
) -> Option<(Array2<f32>, Vec<KvHandle>)> {
    if initial_hidden.nrows() == 0 {
        return None;
    }
    let num_layers = weights.num_layers;
    let mut handles: Vec<KvHandle> = Vec::with_capacity(num_layers);
    let mut h = initial_hidden.clone();

    for layer in 0..num_layers {
        let _t_attn = std::time::Instant::now();
        let (h_post_attn, mut handle) = backend.attention_prefill(
            weights,
            &h,
            layer,
            window,
            index.map(|v| v as &dyn larql_compute::KvIndex),
        )?;
        crate::decode_stages::record_attn(_t_attn.elapsed().as_nanos());
        if let Some(w) = window {
            backend.clip_kv(&mut handle, w);
        }
        handles.push(handle);

        let h_ffn = ffn_or_moe_layer(weights, &h_post_attn, layer, ffn);
        let mut h_out = apply_per_layer_embedding(&weights, &h_ffn, layer, ple_inputs.get(layer));
        apply_layer_scalar(&weights, &mut h_out, layer);
        h = h_out;
    }

    Some((last_row_as_2d(&h), handles))
}

/// Run one autoregressive decode step using `backend`'s
/// [`KvDispatch::attention_step`] intent per layer.
///
/// `handles` must contain one [`KvHandle`] per layer in `weights`. The
/// caller is responsible for tracking `abs_position` (the absolute
/// token index of the new token — usually `prompt_len + step_idx`).
///
/// `window` is forwarded to the backend's clip step per layer when
/// `Some`. Returns the post-FFN hidden state for the new token
/// (shape `[1, hidden]`).
#[allow(clippy::too_many_arguments)]
pub fn kv_decode_step_via_dispatch(
    backend: &dyn EngineBackend,
    weights: larql_models::WeightsView,
    ffn: &dyn FfnBackend,
    handles: &mut [KvHandle],
    token_id: u32,
    abs_position: usize,
    window: Option<usize>,
    index: Option<&larql_vindex::VectorIndex>,
) -> Option<Array2<f32>> {
    let num_layers = weights.num_layers;
    debug_assert_eq!(
        handles.len(),
        num_layers,
        "kv_decode_step_via_dispatch: handles.len() must equal weights.num_layers"
    );
    let h_new = embed_tokens_pub(&weights, &[token_id]);
    // PLE inputs are per-token — recompute for this single-token step.
    // Empty (and a no-op below) on non-Gemma-4 archs. Matches
    // `kv_decode_step_run`.
    let ple_inputs = precompute_per_layer_inputs(&weights, &h_new, &[token_id]);
    let mut h_step = h_new;

    for (layer, handle) in handles.iter_mut().enumerate().take(num_layers) {
        let _t_attn = std::time::Instant::now();
        let h_post_attn = backend.attention_step(
            weights,
            &h_step,
            handle,
            layer,
            abs_position,
            index.map(|v| v as &dyn larql_compute::KvIndex),
        )?;
        crate::decode_stages::record_attn(_t_attn.elapsed().as_nanos());
        if let Some(w) = window {
            backend.clip_kv(handle, w);
        }
        let h_ffn = ffn_or_moe_layer(weights, &h_post_attn, layer, ffn);
        let mut h_out = apply_per_layer_embedding(&weights, &h_ffn, layer, ple_inputs.get(layer));
        apply_layer_scalar(&weights, &mut h_out, layer);
        h_step = h_out;
    }

    Some(h_step)
}

// ── Async variants ──────────────────────────────────────────────────
//
// Mirror the sync helpers above but drive the per-layer loop through
// [`AsyncComputeBackend`]. Per `async-compute-backend.md` §11.5 v1: FFN
// stays on host, so the loop reads the post-attention `AttentionHandle`
// per layer before running FFN. The win at A4 (deferred dispatch) comes
// from K/V appends fusing into the *next* layer's attention command
// buffer — `read_hidden` only forces commit on the hidden, not on the
// cache write. v2 (Step A6+) adds `ffn_step_async` for full
// one-commit-per-decode-step shape.
//
// Engines opting in via `with_async_backend` route through these.

/// Async equivalent of [`kv_prefill_via_dispatch`].
///
/// Calls `backend.attention_prefill_async` per layer, reads the hidden
/// to drive FFN on host, then proceeds. Calls `backend.flush()` once at
/// the end so any deferred work clears before returning.
pub fn kv_prefill_via_dispatch_async(
    backend: &dyn AsyncComputeBackend,
    weights: larql_models::WeightsView,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    window: Option<usize>,
    index: Option<&larql_vindex::VectorIndex>,
) -> Option<(Array2<f32>, Vec<KvHandle>)> {
    if prompt_ids.is_empty() {
        return None;
    }
    let h = embed_tokens_pub(&weights, prompt_ids);
    // Gemma-4 PLE inputs — same recipe as the sync path.
    let ple_inputs = precompute_per_layer_inputs(&weights, &h, prompt_ids);
    prefill_from_hidden_inner_async(backend, weights, ffn, &h, window, index, &ple_inputs)
}

/// Async multi-modal-aware peer of [`kv_prefill_via_dispatch_async`].
/// Same shape as [`kv_prefill_from_hidden_via_dispatch`] but routes
/// per-layer attention through `AsyncComputeBackend` and reads each
/// hidden via `read_hidden` before FFN. Flushes once at the end.
///
/// Bit-identity contract: same as the sync peer. Pinned by the parity
/// test at the bottom of this module — sync vs async must agree on
/// CPU paths, MM vs text must agree when text is the input.
/// PLE caveat: same as the sync from-hidden peer — no token ids here,
/// so Gemma-4 PLE is skipped (empty slab, per-layer no-op);
/// `layer_scalar` is still applied.
pub fn kv_prefill_from_hidden_via_dispatch_async(
    backend: &dyn AsyncComputeBackend,
    weights: larql_models::WeightsView,
    ffn: &dyn FfnBackend,
    initial_hidden: &Array2<f32>,
    window: Option<usize>,
    index: Option<&larql_vindex::VectorIndex>,
) -> Option<(Array2<f32>, Vec<KvHandle>)> {
    prefill_from_hidden_inner_async(backend, weights, ffn, initial_hidden, window, index, &[])
}

/// Shared async prefill body — attention → FFN → per-layer embedding →
/// layer_scalar per layer, mirroring [`prefill_from_hidden_inner`].
fn prefill_from_hidden_inner_async(
    backend: &dyn AsyncComputeBackend,
    weights: larql_models::WeightsView,
    ffn: &dyn FfnBackend,
    initial_hidden: &Array2<f32>,
    window: Option<usize>,
    index: Option<&larql_vindex::VectorIndex>,
    ple_inputs: &[Array2<f32>],
) -> Option<(Array2<f32>, Vec<KvHandle>)> {
    if initial_hidden.nrows() == 0 {
        return None;
    }
    let num_layers = weights.num_layers;
    let mut handles: Vec<KvHandle> = Vec::with_capacity(num_layers);
    let mut h = initial_hidden.clone();

    for layer in 0..num_layers {
        let (h_post_attn_handle, mut handle) = backend.attention_prefill_async(
            weights,
            &h,
            layer,
            window,
            index.map(|v| v as &dyn larql_compute::KvIndex),
        );
        if let Some(w) = window {
            // Sync clip — backends with deferred dispatch must flush
            // before clip per spec §11.3.
            backend.clip_kv(&mut handle, w);
        }
        handles.push(handle);

        let h_post_attn = backend.read_hidden(h_post_attn_handle);
        let h_ffn = ffn_or_moe_layer(weights, &h_post_attn, layer, ffn);
        let mut h_out = apply_per_layer_embedding(&weights, &h_ffn, layer, ple_inputs.get(layer));
        apply_layer_scalar(&weights, &mut h_out, layer);
        h = h_out;
    }

    backend.flush().ok()?;
    Some((last_row_as_2d(&h), handles))
}

/// Async equivalent of [`kv_decode_step_via_dispatch`].
///
/// One decode step. Reads the per-layer hidden for FFN dispatch (v1
/// pattern). Flushes at the end of the step so the next call starts
/// from a quiescent backend.
#[allow(clippy::too_many_arguments)]
pub fn kv_decode_step_via_dispatch_async(
    backend: &dyn AsyncComputeBackend,
    weights: larql_models::WeightsView,
    ffn: &dyn FfnBackend,
    handles: &mut [KvHandle],
    token_id: u32,
    abs_position: usize,
    window: Option<usize>,
    index: Option<&larql_vindex::VectorIndex>,
) -> Option<Array2<f32>> {
    let num_layers = weights.num_layers;
    debug_assert_eq!(
        handles.len(),
        num_layers,
        "kv_decode_step_via_dispatch_async: handles.len() must equal weights.num_layers"
    );
    let h_new = embed_tokens_pub(&weights, &[token_id]);
    // Per-token Gemma-4 PLE inputs — same recipe as the sync decode step.
    let ple_inputs = precompute_per_layer_inputs(&weights, &h_new, &[token_id]);
    let mut h_step = h_new;

    for (layer, handle) in handles.iter_mut().enumerate().take(num_layers) {
        let h_post_attn_handle = backend.attention_step_async(
            weights,
            &h_step,
            handle,
            layer,
            abs_position,
            index.map(|v| v as &dyn larql_compute::KvIndex),
        );
        if let Some(w) = window {
            backend.clip_kv(handle, w);
        }
        let h_post_attn = backend.read_hidden(h_post_attn_handle);
        let h_ffn = ffn_or_moe_layer(weights, &h_post_attn, layer, ffn);
        let mut h_out = apply_per_layer_embedding(&weights, &h_ffn, layer, ple_inputs.get(layer));
        apply_layer_scalar(&weights, &mut h_out, layer);
        h_step = h_out;
    }

    backend.flush().ok()?;
    Some(h_step)
}

fn last_row_as_2d(h: &Array2<f32>) -> Array2<f32> {
    let seq_len = h.shape()[0];
    let hidden = h.shape()[1];
    let mut out = Array2::<f32>::zeros((1, hidden));
    out.row_mut(0).assign(&h.row(seq_len - 1));
    out
}

#[cfg(test)]
mod tests {
    //! Sync vs async dispatch parity, plus dispatch edge cases.
    //!
    //! Parity against the legacy `kv_prefill_run` / `kv_decode_step_run`
    //! reference lives in `larql-kv/tests/dispatch_parity.rs` — moved
    //! out of this module so it can import both crates without forcing
    //! a dev-dep cycle that compiles `larql-inference` twice.

    use super::super::KvDispatch;
    use super::*;
    use crate::ffn::WeightFfn;
    use crate::test_utils::make_test_weights;
    use larql_compute::CpuBackend;

    #[test]
    fn multi_step_decode_via_dispatch_keeps_handles_finite() {
        // Three decode steps in sequence — verifies the handle state
        // carries forward correctly across calls (same shape as
        // bit-parity test in larql-kv/tests/dispatch_parity.rs, but
        // self-contained: no legacy reference, just the dispatch path
        // and a finite-ness invariant).
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = WeightFfn { weights: &weights };
        let prompt = vec![0u32, 1];

        let (_, mut handles) = kv_prefill_via_dispatch(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &prompt,
            None,
            None,
        )
        .unwrap();

        for step in 0..3 {
            let token = (2 + step) as u32;
            let abs_position = prompt.len() + step;
            let h_trait = kv_decode_step_via_dispatch(
                &backend,
                larql_models::WeightsView::dense(&weights),
                &ffn,
                &mut handles,
                token,
                abs_position,
                None,
                None,
            )
            .expect("decode trait");
            assert!(
                h_trait.iter().all(|v| v.is_finite()),
                "step {step} produced non-finite hidden state"
            );
        }
    }

    #[test]
    fn prefill_empty_prompt_returns_none() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = WeightFfn { weights: &weights };
        let result = kv_prefill_via_dispatch(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &[],
            None,
            None,
        );
        assert!(result.is_none());
    }

    // ── Async helper parity ─────────────────────────────────────────

    #[test]
    fn prefill_async_matches_sync_dispatch() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = WeightFfn { weights: &weights };
        let prompt = vec![0u32, 1, 2, 3];

        let (h_sync, handles_sync) = kv_prefill_via_dispatch(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &prompt,
            None,
            None,
        )
        .unwrap();
        let (h_async, handles_async) = kv_prefill_via_dispatch_async(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &prompt,
            None,
            None,
        )
        .unwrap();

        assert_eq!(h_sync, h_async, "async prefill hidden must match sync");
        assert_eq!(handles_sync.len(), handles_async.len());
        for (i, (s, a)) in handles_sync.iter().zip(handles_async.iter()).enumerate() {
            let (k_s, v_s) = backend.read_kv_to_host(s).unwrap();
            let (k_a, v_a) = backend.read_kv_to_host(a).unwrap();
            assert_eq!(k_s, k_a, "K mismatch at layer {i}");
            assert_eq!(v_s, v_a, "V mismatch at layer {i}");
        }
    }

    #[test]
    fn prefill_async_windowed_matches_sync_dispatch() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = WeightFfn { weights: &weights };
        let prompt = vec![0u32, 1, 2, 3, 4];
        let window = Some(2);

        let (h_sync, _) = kv_prefill_via_dispatch(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &prompt,
            window,
            None,
        )
        .unwrap();
        let (h_async, _) = kv_prefill_via_dispatch_async(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &prompt,
            window,
            None,
        )
        .unwrap();

        assert_eq!(h_sync, h_async, "windowed async prefill must match sync");
    }

    #[test]
    fn decode_step_async_matches_sync_dispatch() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = WeightFfn { weights: &weights };
        let prompt = vec![0u32, 1, 2];

        let (_, mut handles_sync) = kv_prefill_via_dispatch(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &prompt,
            None,
            None,
        )
        .unwrap();
        let (_, mut handles_async) = kv_prefill_via_dispatch_async(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &prompt,
            None,
            None,
        )
        .unwrap();

        let next_token = 3u32;
        let abs_position = prompt.len();

        let h_sync = kv_decode_step_via_dispatch(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &mut handles_sync,
            next_token,
            abs_position,
            None,
            None,
        )
        .unwrap();
        let h_async = kv_decode_step_via_dispatch_async(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &mut handles_async,
            next_token,
            abs_position,
            None,
            None,
        )
        .unwrap();

        assert_eq!(h_sync, h_async, "async decode_step hidden must match sync");
    }

    #[test]
    fn multi_step_decode_async_matches_sync_dispatch() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = WeightFfn { weights: &weights };
        let prompt = vec![0u32, 1];

        let (_, mut handles_sync) = kv_prefill_via_dispatch(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &prompt,
            None,
            None,
        )
        .unwrap();
        let (_, mut handles_async) = kv_prefill_via_dispatch_async(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &prompt,
            None,
            None,
        )
        .unwrap();

        for step in 0..3 {
            let token = (2 + step) as u32;
            let abs_position = prompt.len() + step;
            let h_sync = kv_decode_step_via_dispatch(
                &backend,
                larql_models::WeightsView::dense(&weights),
                &ffn,
                &mut handles_sync,
                token,
                abs_position,
                None,
                None,
            )
            .unwrap();
            let h_async = kv_decode_step_via_dispatch_async(
                &backend,
                larql_models::WeightsView::dense(&weights),
                &ffn,
                &mut handles_async,
                token,
                abs_position,
                None,
                None,
            )
            .unwrap();
            assert_eq!(h_sync, h_async, "step {step} async vs sync must match");
        }
    }

    #[test]
    fn prefill_async_empty_prompt_returns_none() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = WeightFfn { weights: &weights };
        let result = kv_prefill_via_dispatch_async(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &[],
            None,
            None,
        );
        assert!(result.is_none());
    }

    // ─── Phase 1d.3a: embed-hoist bit-identity (sync + async) ───────────────
    //
    // These pin the refactor that landed `kv_prefill_from_hidden_via_dispatch`
    // as the new MM-aware entry point. The old `kv_prefill_via_dispatch` is
    // now a two-line wrapper that runs `embed_tokens_pub` then delegates.
    // Bit-identity of the wrapper-vs-direct path is the contract that makes
    // the engine seam (per ADR-0023) safe to land — and we have to verify
    // sync and async separately because they diverge on real (non-CPU)
    // backends, and on CPU the async path additionally calls flush() (a
    // no-op on CpuBackend but a real ordering primitive elsewhere).

    #[test]
    fn prefill_via_dispatch_bit_identical_to_from_hidden_sync() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = WeightFfn { weights: &weights };
        let tokens = vec![0u32, 1, 2, 3];

        let (h_text, handles_text) = kv_prefill_via_dispatch(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &tokens,
            None,
            None,
        )
        .unwrap();

        let initial_hidden = embed_tokens_pub(&weights, &tokens);
        let (h_hidden, handles_hidden) = kv_prefill_from_hidden_via_dispatch(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &initial_hidden,
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            h_text, h_hidden,
            "text and from-hidden paths must produce bit-identical last-row hidden"
        );
        assert_eq!(
            handles_text.len(),
            handles_hidden.len(),
            "handle count (= num_layers) must match across paths"
        );
    }

    #[test]
    fn prefill_via_dispatch_bit_identical_to_from_hidden_async() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = WeightFfn { weights: &weights };
        let tokens = vec![0u32, 1, 2, 3];

        let (h_text, handles_text) = kv_prefill_via_dispatch_async(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &tokens,
            None,
            None,
        )
        .unwrap();

        let initial_hidden = embed_tokens_pub(&weights, &tokens);
        let (h_hidden, handles_hidden) = kv_prefill_from_hidden_via_dispatch_async(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &initial_hidden,
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            h_text, h_hidden,
            "async text and from-hidden paths must produce bit-identical last-row hidden"
        );
        assert_eq!(handles_text.len(), handles_hidden.len());
    }

    #[test]
    fn prefill_from_hidden_returns_none_on_empty_input() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = WeightFfn { weights: &weights };
        let empty_hidden = Array2::<f32>::zeros((0, weights.hidden_size));
        let result = kv_prefill_from_hidden_via_dispatch(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &empty_hidden,
            None,
            None,
        );
        assert!(result.is_none(), "zero-row hidden should yield None");

        let result_async = kv_prefill_from_hidden_via_dispatch_async(
            &backend,
            larql_models::WeightsView::dense(&weights),
            &ffn,
            &empty_hidden,
            None,
            None,
        );
        assert!(
            result_async.is_none(),
            "async zero-row hidden should yield None"
        );
    }
}
