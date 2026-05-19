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
use crate::forward::{embed_tokens_pub, run_ffn};
use crate::model::ModelWeights;

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
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    window: Option<usize>,
    index: Option<&larql_vindex::VectorIndex>,
) -> Option<(Array2<f32>, Vec<KvHandle>)> {
    if prompt_ids.is_empty() {
        return None;
    }
    let num_layers = weights.num_layers;
    let mut handles: Vec<KvHandle> = Vec::with_capacity(num_layers);
    let mut h = embed_tokens_pub(weights, prompt_ids);

    for layer in 0..num_layers {
        let (h_post_attn, mut handle) =
            backend.attention_prefill(weights, &h, layer, window, index)?;
        if let Some(w) = window {
            backend.clip_kv(&mut handle, w);
        }
        handles.push(handle);

        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
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
    weights: &ModelWeights,
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
    let h_new = embed_tokens_pub(weights, &[token_id]);
    let mut h_step = h_new;

    for (layer, handle) in handles.iter_mut().enumerate().take(num_layers) {
        let h_post_attn =
            backend.attention_step(weights, &h_step, handle, layer, abs_position, index)?;
        if let Some(w) = window {
            backend.clip_kv(handle, w);
        }
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
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
    weights: &ModelWeights,
    ffn: &dyn FfnBackend,
    prompt_ids: &[u32],
    window: Option<usize>,
    index: Option<&larql_vindex::VectorIndex>,
) -> Option<(Array2<f32>, Vec<KvHandle>)> {
    if prompt_ids.is_empty() {
        return None;
    }
    let num_layers = weights.num_layers;
    let mut handles: Vec<KvHandle> = Vec::with_capacity(num_layers);
    let mut h = embed_tokens_pub(weights, prompt_ids);

    for layer in 0..num_layers {
        let (h_post_attn_handle, mut handle) =
            backend.attention_prefill_async(weights, &h, layer, window, index);
        if let Some(w) = window {
            // Sync clip — backends with deferred dispatch must flush
            // before clip per spec §11.3.
            backend.clip_kv(&mut handle, w);
        }
        handles.push(handle);

        let h_post_attn = backend.read_hidden(h_post_attn_handle);
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
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
    weights: &ModelWeights,
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
    let h_new = embed_tokens_pub(weights, &[token_id]);
    let mut h_step = h_new;

    for (layer, handle) in handles.iter_mut().enumerate().take(num_layers) {
        let h_post_attn_handle =
            backend.attention_step_async(weights, &h_step, handle, layer, abs_position, index);
        if let Some(w) = window {
            backend.clip_kv(handle, w);
        }
        let h_post_attn = backend.read_hidden(h_post_attn_handle);
        let (h_out, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
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

        let (_, mut handles) =
            kv_prefill_via_dispatch(&backend, &weights, &ffn, &prompt, None, None).unwrap();

        for step in 0..3 {
            let token = (2 + step) as u32;
            let abs_position = prompt.len() + step;
            let h_trait = kv_decode_step_via_dispatch(
                &backend,
                &weights,
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
        let result = kv_prefill_via_dispatch(&backend, &weights, &ffn, &[], None, None);
        assert!(result.is_none());
    }

    // ── Async helper parity ─────────────────────────────────────────

    #[test]
    fn prefill_async_matches_sync_dispatch() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = WeightFfn { weights: &weights };
        let prompt = vec![0u32, 1, 2, 3];

        let (h_sync, handles_sync) =
            kv_prefill_via_dispatch(&backend, &weights, &ffn, &prompt, None, None).unwrap();
        let (h_async, handles_async) =
            kv_prefill_via_dispatch_async(&backend, &weights, &ffn, &prompt, None, None).unwrap();

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

        let (h_sync, _) =
            kv_prefill_via_dispatch(&backend, &weights, &ffn, &prompt, window, None).unwrap();
        let (h_async, _) =
            kv_prefill_via_dispatch_async(&backend, &weights, &ffn, &prompt, window, None).unwrap();

        assert_eq!(h_sync, h_async, "windowed async prefill must match sync");
    }

    #[test]
    fn decode_step_async_matches_sync_dispatch() {
        let weights = make_test_weights();
        let backend = CpuBackend;
        let ffn = WeightFfn { weights: &weights };
        let prompt = vec![0u32, 1, 2];

        let (_, mut handles_sync) =
            kv_prefill_via_dispatch(&backend, &weights, &ffn, &prompt, None, None).unwrap();
        let (_, mut handles_async) =
            kv_prefill_via_dispatch_async(&backend, &weights, &ffn, &prompt, None, None).unwrap();

        let next_token = 3u32;
        let abs_position = prompt.len();

        let h_sync = kv_decode_step_via_dispatch(
            &backend,
            &weights,
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
            &weights,
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

        let (_, mut handles_sync) =
            kv_prefill_via_dispatch(&backend, &weights, &ffn, &prompt, None, None).unwrap();
        let (_, mut handles_async) =
            kv_prefill_via_dispatch_async(&backend, &weights, &ffn, &prompt, None, None).unwrap();

        for step in 0..3 {
            let token = (2 + step) as u32;
            let abs_position = prompt.len() + step;
            let h_sync = kv_decode_step_via_dispatch(
                &backend,
                &weights,
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
                &weights,
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
        let result = kv_prefill_via_dispatch_async(&backend, &weights, &ffn, &[], None, None);
        assert!(result.is_none());
    }
}
