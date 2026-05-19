//! Prediction — logits computation and all predict_* entry points.
//!
//! Submodules:
//! - `types`: Result structs and `LayerMode` enum
//! - `raw`: `RawForward`, `forward_raw_logits`, `forward_from_layer`, `hidden_to_raw_logits`
//! - `dense`: Dense weight forward passes and logit projection
//! - `ffn`: Custom FFN backend, router, and strategy forward passes

pub mod dense;
pub mod ffn;
pub mod raw;
pub mod types;

// ── Re-exports: preserve all `crate::forward::predict::*` paths ──

pub use types::{
    LayerAttentionCapture, LayerMode, PredictResult, PredictResultWithAttention,
    PredictResultWithResiduals, TraceResult,
};

pub use raw::{
    forward_from_layer, forward_raw_logits, forward_raw_logits_with_prefix, hidden_to_raw_logits,
    RawForward,
};

pub use dense::{
    logit_lens_top1, logits_to_predictions_pub, logits_to_predictions_q4_lm_head, predict,
    predict_from_hidden, predict_from_hidden_with_ffn, predict_with_ffn_trace,
    predict_with_temperature,
};

pub use ffn::{
    predict_with_ffn, predict_with_ffn_attention, predict_with_router, predict_with_strategy,
};

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::dense::cmp_desc_nan_last;

    #[test]
    fn topk_sort_nan_last_preserves_real_max() {
        // Logits with interleaved NaN must not displace the real maximum
        // from top-k. Earlier `partial_cmp().unwrap()` panicked on NaN;
        // the previous `unwrap_or(Equal)` patch stopped the panic but
        // let NaN sort anywhere — sometimes knocking the real max out.
        // `cmp_desc_nan_last` pushes NaN to the end so the top-k is
        // always correct among the real values.
        let probs: Vec<f32> = vec![0.1, 0.3, f32::NAN, 0.05, f32::NAN, 0.5, 0.2];
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        let k = 3;
        indexed.select_nth_unstable_by(k, cmp_desc_nan_last);
        indexed.truncate(k);
        indexed.sort_unstable_by(cmp_desc_nan_last);

        assert_eq!(indexed.len(), 3);
        let vals: Vec<f32> = indexed.iter().map(|(_, p)| *p).collect();
        assert!(
            vals.iter().all(|v| !v.is_nan()),
            "NaN leaked into top-3: {vals:?}"
        );
        // Real top-3 (descending) from the non-NaN set {0.1, 0.3, 0.05, 0.5, 0.2}
        // is [0.5, 0.3, 0.2].
        assert_eq!(vals, vec![0.5, 0.3, 0.2]);
    }

    #[test]
    fn topk_sort_all_nan_doesnt_panic() {
        // Degenerate case: every logit is NaN (catastrophic quant / NaN
        // cascade). The call must return *something* of the right length
        // rather than panicking — callers can decide how to treat a
        // NaN-only top-k.
        let probs: Vec<f32> = vec![f32::NAN; 10];
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        let k = 3;
        indexed.select_nth_unstable_by(k, cmp_desc_nan_last);
        indexed.truncate(k);
        indexed.sort_unstable_by(cmp_desc_nan_last);
        assert_eq!(indexed.len(), 3);
    }

    #[test]
    fn topk_sort_no_nan_is_plain_descending() {
        let probs: Vec<f32> = vec![0.1, 0.5, 0.3, 0.05, 0.7, 0.2];
        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(cmp_desc_nan_last);
        let vals: Vec<f32> = indexed.iter().map(|(_, p)| *p).collect();
        assert_eq!(vals, vec![0.7, 0.5, 0.3, 0.2, 0.1, 0.05]);
    }
}
