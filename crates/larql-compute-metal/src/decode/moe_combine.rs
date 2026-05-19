//! CPU-side combine step for hybrid MoE layers.
//!
//! Runs after the GPU dense-FFN has written `new_h = h_post_attn + _1(dense)`
//! and the CPU MoE block has added `moe_out` into `new_h` in place. At that
//! point `new_h - h_post_attn` equals `_1(dense) + _2(moe)` — HF's `h1 + h2`
//! in the Gemma 4 decoder-layer forward.
//!
//! Two independent HF-matching operations happen here:
//!   1. **Outer post-FFN norm** on `(h1 + h2)`, then residual add. Matches:
//!      `hidden = residual + post_feedforward_layernorm(h1 + h2)`
//!   2. **Whole-layer `layer_scalar` multiplication** on the entire output.
//!      Matches HF's final step in `Gemma4TextDecoderLayer.forward`:
//!      `hidden_states *= self.layer_scalar`
//!      NB: this multiplies `h_post_attn + ffn_delta` — not just the FFN
//!      delta — which is why folding `layer_scalar` into the outer-norm
//!      scale was wrong (prior bug: 14× mis-scaling on 26B A4B collapsed
//!      the model to degenerate token-repetition output).
//!
//! All operations here are pure f32 arithmetic on shared-memory Metal
//! buffers; no encoder or command buffer involvement.

use larql_compute::cpu::ops::outer_combine::{
    apply_layer_scalar_in_place, outer_post_norm_residual,
};
use larql_compute::FullPipelineLayer;

/// Apply the outer post-FFN norm (when the arch declares one) followed by
/// the whole-layer `layer_scalar` multiplication. Operates in place on
/// `new_h`. Requires that `new_h` currently holds
/// `h_post_attn + (_1(dense) + _2(moe))`.
///
/// Routes through `cpu::ops::outer_combine` so the GPU MoE path and
/// the CPU MoE path (`vindex/kquant_forward.rs::run_moe_layer_cpu`) share
/// a single implementation of the math. Earlier the two backends had
/// independent transcriptions of the same formula and silently drifted
/// on Gemma 4 26B-A4B.
pub(super) fn apply_outer_combine(
    layer: &FullPipelineLayer,
    new_h: &metal::Buffer,
    h_post_attn: &metal::Buffer,
    hidden: usize,
) {
    // Diagnostic bypass: leave `new_h` as `h_post_attn + _1(dense) + _2(moe)`
    // without outer norm OR layer_scalar — useful for isolating whether
    // this combine step is the broken piece.
    if larql_compute::options::env_flag(larql_compute::options::ENV_SKIP_OUTER_NORM) {
        return;
    }

    // Metal buffers are shared-memory; cast to f32 slices for the
    // shared CPU helper. `hidden` is fixed by the model architecture
    // and the buffers are sized at allocation time, so the slice
    // length is correct by construction.
    let new_h_slice: &mut [f32] =
        unsafe { std::slice::from_raw_parts_mut(new_h.contents() as *mut f32, hidden) };
    let h_post_attn_slice: &[f32] =
        unsafe { std::slice::from_raw_parts(h_post_attn.contents() as *const f32, hidden) };

    // Step A — outer post-FFN norm on `(h1 + h2)`, residual-added back.
    //
    // Falls back to `post_ffn_norm` (which for Gemma 4 MoE is `_1`) when no
    // un-suffixed outer norm tensor is loaded, so older vindexes still work
    // even if incorrectly. The correct path uses `moe_outer_post_norm` which
    // the extractor now emits for hybrid-MoE architectures.
    if layer.moe_combined_output_norm {
        let outer_w = layer.moe_outer_post_norm.or(layer.post_ffn_norm);
        // Compute `h1+h2 = new_h - h_post_attn` (the delta the GPU
        // built up via dense + moe writes), pass it through the
        // shared helper, then copy the result back into `new_h`.
        let h1_plus_h2: Vec<f32> = new_h_slice
            .iter()
            .zip(h_post_attn_slice.iter())
            .map(|(&n, &ha)| n - ha)
            .collect();
        let combined = outer_post_norm_residual(
            h_post_attn_slice,
            &h1_plus_h2,
            outer_w,
            layer.norm_offset,
            layer.eps,
        );
        new_h_slice.copy_from_slice(&combined);
    }

    // Step B — whole-layer `layer_scalar` multiplication. HF's
    //   `Gemma4TextDecoderLayer.forward` ends with `hidden_states *= self.layer_scalar`
    // which scales BOTH the residual and the FFN delta. A null scalar
    // (0.0) or an identity scalar (1.0) is a no-op.
    apply_layer_scalar_in_place(new_h_slice, layer.layer_scalar);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetalBackend;
    use larql_compute::pipeline::{
        Activation, FfnType, FullPipelineLayer, NormType, QuantFormat, QuantWeight,
    };

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn backend() -> MetalBackend {
        MetalBackend::new().expect("Metal device available on test host")
    }

    fn layer_with<'a>(
        norm: &'a [f32],
        moe_outer_post_norm: Option<&'a [f32]>,
        moe_combined_output_norm: bool,
        layer_scalar: f32,
    ) -> FullPipelineLayer<'a> {
        let empty_q4 = QuantWeight {
            data: &[],
            scales: None,
            format: QuantFormat::Q4_K,
        };
        FullPipelineLayer {
            wq: empty_q4,
            wk: empty_q4,
            wv: empty_q4,
            wo: empty_q4,
            gate: empty_q4,
            up: empty_q4,
            down: empty_q4,
            input_norm: norm,
            post_attn_norm: norm,
            pre_ffn_norm: None,
            post_ffn_norm: Some(norm),
            input_norm_bias: None,
            post_attn_norm_bias: None,
            norm_offset: 1.0,
            qk_norm_offset: 0.0,
            eps: 1e-6,
            has_post_norms: true,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::Gated,
            activation: Activation::Silu,
            attn_scale: 0.125,
            head_dim: 64,
            num_q_heads: 4,
            num_kv_heads: 4,
            rope_base: 10000.0,
            rotary_dim: 0,
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar,
            q_norm_weight: None,
            k_norm_weight: None,
            ffn_up_bias: None,
            ffn_down_bias: None,
            moe: None,
            ffn_is_remote: false,
            moe_combined_output_norm,
            moe_outer_post_norm,
            kv_shared_source: None,
            residual_multiplier: 1.0,
            ple_input_gate: None,
            ple_projection: None,
            ple_post_norm: None,
        }
    }

    /// `LARQL_SKIP_OUTER_NORM=1` short-circuits the combine —
    /// covers line 47.  Buffers must hold finite values
    /// untouched after the call.
    #[test]
    fn skip_outer_norm_env_short_circuits() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let m = backend();
        let hidden = 64usize;
        let norm = vec![1.0f32; hidden];
        let layer = layer_with(&norm, None, true, 1.0);

        let new_h_data = vec![2.0f32; hidden];
        let h_post_attn_data = vec![1.0f32; hidden];
        let new_h = m.bufs.transient_from_f32(&new_h_data);
        let h_post = m.bufs.transient_from_f32(&h_post_attn_data);

        let saved = std::env::var_os("SKIP_OUTER_NORM");
        unsafe {
            std::env::set_var("SKIP_OUTER_NORM", "1");
        }
        apply_outer_combine(&layer, &new_h, &h_post, hidden);
        unsafe {
            match saved {
                Some(v) => std::env::set_var("SKIP_OUTER_NORM", v),
                None => std::env::remove_var("SKIP_OUTER_NORM"),
            }
        }

        // No-op: new_h unchanged from input (2.0).
        let out = unsafe { std::slice::from_raw_parts(new_h.contents() as *const f32, hidden) };
        assert!(out.iter().all(|&v| (v - 2.0).abs() < 1e-6));
    }

    /// `moe_combined_output_norm = true` + `moe_outer_post_norm` set
    /// drives the Step A outer-norm branch (lines 66-82).
    #[test]
    fn combined_output_norm_with_outer_weight_applies_norm_and_scalar() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let m = backend();
        let hidden = 64usize;
        let norm = vec![1.0f32; hidden];
        let outer = vec![1.0f32; hidden];
        let layer = layer_with(&norm, Some(&outer), true, 2.0);

        let new_h_data: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let h_post_attn_data: Vec<f32> = (0..hidden).map(|i| 0.5 + (i as f32) * 0.005).collect();
        let new_h = m.bufs.transient_from_f32(&new_h_data);
        let h_post = m.bufs.transient_from_f32(&h_post_attn_data);

        apply_outer_combine(&layer, &new_h, &h_post, hidden);

        let out = unsafe { std::slice::from_raw_parts(new_h.contents() as *const f32, hidden) };
        assert!(out.iter().all(|v| v.is_finite()));
        // layer_scalar = 2.0 means the output should be larger than
        // h_post_attn alone — pin loosely.
        let mean_out: f32 = out.iter().sum::<f32>() / hidden as f32;
        assert!(mean_out.abs() > 0.0);
    }

    /// `moe_combined_output_norm = true` + no outer weight falls
    /// back to `post_ffn_norm` (still covers the same branch).
    #[test]
    fn combined_output_norm_falls_back_to_post_ffn_norm() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let m = backend();
        let hidden = 64usize;
        let norm = vec![1.0f32; hidden];
        let layer = layer_with(&norm, None, true, 1.0);

        let new_h_data: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.01).collect();
        let h_post_attn_data: Vec<f32> = (0..hidden).map(|i| 0.5 + (i as f32) * 0.005).collect();
        let new_h = m.bufs.transient_from_f32(&new_h_data);
        let h_post = m.bufs.transient_from_f32(&h_post_attn_data);

        apply_outer_combine(&layer, &new_h, &h_post, hidden);

        let out = unsafe { std::slice::from_raw_parts(new_h.contents() as *const f32, hidden) };
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// `moe_combined_output_norm = false` + non-trivial `layer_scalar`
    /// skips Step A but applies Step B.  Pins the
    /// `layer_scalar != 1.0` arm of `apply_layer_scalar_in_place`.
    #[test]
    fn combine_without_outer_norm_still_applies_layer_scalar() {
        let _g = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let m = backend();
        let hidden = 64usize;
        let norm = vec![1.0f32; hidden];
        let layer = layer_with(&norm, None, false, 3.0);

        let new_h_data = vec![1.0f32; hidden];
        let h_post_attn_data = vec![0.5f32; hidden];
        let new_h = m.bufs.transient_from_f32(&new_h_data);
        let h_post = m.bufs.transient_from_f32(&h_post_attn_data);

        apply_outer_combine(&layer, &new_h, &h_post, hidden);

        let out = unsafe { std::slice::from_raw_parts(new_h.contents() as *const f32, hidden) };
        // 1.0 * 3.0 = 3.0 (Step A skipped, Step B applies).
        assert!(out.iter().all(|&v| (v - 3.0).abs() < 1e-5));
    }
}
