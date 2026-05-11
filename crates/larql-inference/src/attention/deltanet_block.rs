//! Gated DeltaNet block forward (Phase C.4a of
//! `inference-qwen35-deltanet`).
//!
//! Integrates the C.1 building blocks (Conv1D-with-state, scalar
//! helpers, state cache) and the C.2 recurrence into a single-token
//! forward step for one Gated DeltaNet linear-attention layer.
//!
//! What's NOT included: the residual add, the post-attention
//! RMSNorm (`attn_post_norm`), and the FFN. Those live in the
//! per-layer router (Phase C.4b) which adds the residual + FFN
//! around this block.
//!
//! Math literal from `openspec/changes/inference-qwen35-deltanet/
//! design.md` §4:
//!
//! ```text
//! x_norm    = RMSNorm(x, attn_norm)
//! qkv_mixed = attn_qkv  @ x_norm           # [conv_dim]
//! z         = attn_gate @ x_norm           # [value_dim]
//! beta      = sigmoid(ssm_beta  @ x_norm)  # [n_v_heads]
//! alpha     = ssm_alpha @ x_norm           # [n_v_heads]
//! log_g     = ssm_a * softplus(alpha + ssm_dt)  # [n_v_heads]
//!
//! qkv_conv  = SiLU(Conv1D(qkv_mixed, ssm_conv1d, state.conv_state))
//! q_raw, k_raw, v_raw = split(qkv_conv, [key_dim, key_dim, value_dim])
//! q = L2Norm(reshape(q_raw, [head_v_dim, n_k_heads]))
//! k = L2Norm(reshape(k_raw, [head_v_dim, n_k_heads]))
//! v = reshape(v_raw, [head_v_dim, n_v_heads])
//!
//! o = delta_net_step(q, k, v, log_g, beta, state.recurrent_state)
//! o = RMSNorm(reshape(o, [value_dim]), ssm_norm) * SiLU(z)
//! y = ssm_out @ o                          # [hidden]
//! ```

use ndarray::{ArcArray2, Array1, Array2, Axis};

use super::deltanet_recurrence::delta_net_step;
use super::deltanet_state::{
    causal_conv1d_step, l2_normalize_per_head, sigmoid, softplus, DeltaNetLayerState,
};

/// One DeltaNet layer's weight tensors.
///
/// Stored as `ArcArray2<f32>` so cloning the struct (e.g. when
/// building a `Qwen35Weights` view) only bumps Arc refcounts — no
/// data copies. Shape convention: every projection has
/// `[out_features, in_features]` (matvec is `y = W @ x`).
///
/// 1-D tensors (norms, scalar per-head) are `Arc<[f32]>` for the
/// same cheap-clone reason; callers index with `.as_ref()` to get
/// `&[f32]`.
#[derive(Clone)]
pub struct DeltaNetLayerWeights {
    /// Pre-mixer RMSNorm weight `[hidden]`.
    pub attn_norm: std::sync::Arc<[f32]>,
    /// Fused QKV projection `[conv_dim, hidden]`.
    pub attn_qkv: ArcArray2<f32>,
    /// Z-gate projection `[value_dim, hidden]`.
    pub attn_gate: ArcArray2<f32>,
    /// Causal depthwise Conv1D weight `[d_conv, conv_dim]`.
    pub ssm_conv1d: ArcArray2<f32>,
    /// Per-head bias added to `alpha` before softplus `[n_v_heads]`.
    pub ssm_dt: std::sync::Arc<[f32]>,
    /// Per-head log-decay scalar `[n_v_heads]`.
    pub ssm_a: std::sync::Arc<[f32]>,
    /// Beta projection `[n_v_heads, hidden]`.
    pub ssm_beta: ArcArray2<f32>,
    /// Alpha projection `[n_v_heads, hidden]`.
    pub ssm_alpha: ArcArray2<f32>,
    /// Post-mixer per-head RMSNorm weight `[head_v_dim]`.
    pub ssm_norm: std::sync::Arc<[f32]>,
    /// Output projection `[hidden, value_dim]`.
    pub ssm_out: ArcArray2<f32>,
}

/// Per-layer shape constants. All architectures in scope (Qwen 3.6
/// dense + MoE) share the same DeltaNet shape, so this is a
/// per-model constant rather than a per-layer one — but the struct
/// stays per-block in case future variants differ.
#[derive(Clone, Copy, Debug)]
pub struct DeltaNetDims {
    /// Residual-stream width `[hidden]`. Qwen 3.6 27B: 5120.
    pub hidden: usize,
    /// Per-head dim (`ssm_state_size`). Qwen 3.6: 128.
    pub head_v_dim: usize,
    /// Number of V heads (`ssm_dt_rank`). Qwen 3.6: 48 / 32.
    pub n_v_heads: usize,
    /// Number of K heads (`ssm_group_count`). Qwen 3.6: 16.
    pub n_k_heads: usize,
    /// Conv1D window. Qwen 3.6: 4.
    pub d_conv: usize,
    /// RMSNorm epsilon. Qwen 3.6: 1e-6.
    pub eps: f32,
}

impl DeltaNetDims {
    /// `head_v_dim * n_k_heads`.
    #[inline]
    pub fn key_dim(&self) -> usize {
        self.head_v_dim * self.n_k_heads
    }

    /// `head_v_dim * n_v_heads`.
    #[inline]
    pub fn value_dim(&self) -> usize {
        self.head_v_dim * self.n_v_heads
    }

    /// `2 * key_dim + value_dim` (the Conv1D channel count).
    #[inline]
    pub fn conv_dim(&self) -> usize {
        2 * self.key_dim() + self.value_dim()
    }
}

/// One-token forward through the DeltaNet block. Mutates `state` in
/// place; returns the block's output, ready to be added back to the
/// residual stream by the caller.
pub fn deltanet_block_step(
    x: &Array1<f32>,
    weights: &DeltaNetLayerWeights,
    dims: &DeltaNetDims,
    state: &mut DeltaNetLayerState,
) -> Array1<f32> {
    debug_assert_eq!(x.len(), dims.hidden);
    debug_assert_eq!(weights.attn_norm.len(), dims.hidden);
    debug_assert_eq!(weights.attn_qkv.shape(), [dims.conv_dim(), dims.hidden]);
    debug_assert_eq!(weights.attn_gate.shape(), [dims.value_dim(), dims.hidden]);
    debug_assert_eq!(weights.ssm_conv1d.shape(), [dims.d_conv, dims.conv_dim()]);
    debug_assert_eq!(weights.ssm_dt.len(), dims.n_v_heads);
    debug_assert_eq!(weights.ssm_a.len(), dims.n_v_heads);
    debug_assert_eq!(weights.ssm_beta.shape(), [dims.n_v_heads, dims.hidden]);
    debug_assert_eq!(weights.ssm_alpha.shape(), [dims.n_v_heads, dims.hidden]);
    debug_assert_eq!(weights.ssm_norm.len(), dims.head_v_dim);
    debug_assert_eq!(weights.ssm_out.shape(), [dims.hidden, dims.value_dim()]);

    // 1. Pre-mixer RMSNorm.
    let x_norm = rms_norm_1d(x, &weights.attn_norm, dims.eps);

    // 2. Projections (matvec).
    let qkv_mixed = weights.attn_qkv.dot(&x_norm); // [conv_dim]
    let z = weights.attn_gate.dot(&x_norm); // [value_dim]
    let beta_raw = weights.ssm_beta.dot(&x_norm); // [n_v_heads]
    let alpha_raw = weights.ssm_alpha.dot(&x_norm); // [n_v_heads]

    // Per-head non-linearities.
    let beta: Array1<f32> = beta_raw.iter().map(|&v| sigmoid(v)).collect();
    let log_g: Array1<f32> = (0..dims.n_v_heads)
        .map(|h| weights.ssm_a[h] * softplus(alpha_raw[h] + weights.ssm_dt[h]))
        .collect();

    // 3. Causal Conv1D-with-state, then SiLU element-wise.
    let mut qkv_conv =
        causal_conv1d_step(weights.ssm_conv1d.view(), &mut state.conv_state, &qkv_mixed);
    for v in qkv_conv.iter_mut() {
        *v = silu(*v);
    }

    // 4. Split QKV into Q, K, V slabs.
    let key_dim = dims.key_dim();
    let value_dim = dims.value_dim();
    let q_raw = qkv_conv.slice(ndarray::s![..key_dim]).to_owned();
    let k_raw = qkv_conv.slice(ndarray::s![key_dim..2 * key_dim]).to_owned();
    let v_raw = qkv_conv.slice(ndarray::s![2 * key_dim..]).to_owned();

    // Reshape Q/K to [head_v_dim, n_k_heads], V to [head_v_dim, n_v_heads].
    let q = q_raw
        .into_shape_with_order((dims.head_v_dim, dims.n_k_heads))
        .expect("q reshape");
    let k = k_raw
        .into_shape_with_order((dims.head_v_dim, dims.n_k_heads))
        .expect("k reshape");
    let v = v_raw
        .into_shape_with_order((dims.head_v_dim, dims.n_v_heads))
        .expect("v reshape");

    // 5. L2-norm Q and K per head. V passes through.
    let q = l2_normalize_per_head(&q);
    let k = l2_normalize_per_head(&k);

    // 6. Delta-rule recurrence.
    let o = delta_net_step(&q, &k, &v, &log_g, &beta, &mut state.recurrent_state);
    // o shape: [head_v_dim, n_v_heads]. Flatten to [value_dim] in the
    // canonical row-major order so the post-mixer RMSNorm + Z-gate
    // see the layout `concat_heads(head_v_dim chunks)`.
    let o_flat: Array1<f32> = o.into_iter().collect();
    debug_assert_eq!(o_flat.len(), value_dim);

    // 7. Per-head RMSNorm by ssm_norm (weight is [head_v_dim], shared
    //    across heads). Reshape to [1, value_dim] for rms_norm_heads.
    let o_2d = o_flat
        .into_shape_with_order((1, value_dim))
        .expect("o reshape");
    let o_normed = crate::residual::rms_norm_heads(
        &o_2d,
        &weights.ssm_norm,
        dims.n_v_heads,
        dims.head_v_dim,
        0.0,
    );
    let mut o_flat = o_normed.index_axis_move(Axis(0), 0);

    // 8. Multiply by SiLU(Z) element-wise.
    for c in 0..value_dim {
        o_flat[c] *= silu(z[c]);
    }

    // 9. Output projection (matvec).
    weights.ssm_out.dot(&o_flat)
}

#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// RMSNorm on a single 1-D vector with weight broadcast across the
/// last axis. `(x / sqrt(mean(x²) + eps)) * weight`.
///
/// `pub(crate)` so the qwen35 full-attention block can reuse it for
/// its pre-norm step.
pub(crate) fn rms_norm_1d_pub(x: &Array1<f32>, weight: &[f32], eps: f32) -> Array1<f32> {
    rms_norm_1d(x, weight, eps)
}

fn rms_norm_1d(x: &Array1<f32>, weight: &[f32], eps: f32) -> Array1<f32> {
    debug_assert_eq!(x.len(), weight.len());
    let n = x.len();
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    Array1::from_iter(x.iter().zip(weight).map(|(&xv, &wv)| xv * inv * wv))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use std::sync::Arc;

    /// Convenience: build a `DeltaNetLayerWeights` from owned Vec /
    /// Array2 inputs, doing the Arc / ArcArray wrap inline. Reduces
    /// per-test boilerplate (10 fields × Arc-conversion).
    #[allow(clippy::too_many_arguments)]
    fn make_dn_weights(
        attn_norm: Vec<f32>,
        attn_qkv: Array2<f32>,
        attn_gate: Array2<f32>,
        ssm_conv1d: Array2<f32>,
        ssm_dt: Vec<f32>,
        ssm_a: Vec<f32>,
        ssm_beta: Array2<f32>,
        ssm_alpha: Array2<f32>,
        ssm_norm: Vec<f32>,
        ssm_out: Array2<f32>,
    ) -> DeltaNetLayerWeights {
        DeltaNetLayerWeights {
            attn_norm: Arc::from(attn_norm.as_slice()),
            attn_qkv: attn_qkv.into_shared(),
            attn_gate: attn_gate.into_shared(),
            ssm_conv1d: ssm_conv1d.into_shared(),
            ssm_dt: Arc::from(ssm_dt.as_slice()),
            ssm_a: Arc::from(ssm_a.as_slice()),
            ssm_beta: ssm_beta.into_shared(),
            ssm_alpha: ssm_alpha.into_shared(),
            ssm_norm: Arc::from(ssm_norm.as_slice()),
            ssm_out: ssm_out.into_shared(),
        }
    }

    /// Tiny end-to-end shape check: build a 1-head linear-attention
    /// layer with deterministic weights, feed it a known input, and
    /// verify the output is the right shape + the state was updated.
    /// Numerical correctness is the parity-vs-llama.cpp test's job
    /// (Phase C.5); here we just sanity-check that the pipeline
    /// composes correctly.
    fn make_tiny_dims() -> DeltaNetDims {
        DeltaNetDims {
            hidden: 4,
            head_v_dim: 2,
            n_v_heads: 1,
            n_k_heads: 1,
            d_conv: 2,
            eps: 1e-6,
        }
    }

    #[test]
    fn deltanet_block_step_output_has_hidden_shape() {
        let dims = make_tiny_dims();
        // conv_dim = 2*key_dim + value_dim = 2*2 + 2 = 6.
        let conv_dim = dims.conv_dim();
        let value_dim = dims.value_dim();
        let key_dim = dims.key_dim();

        // Small constant weights so we can predict shape + non-zero
        // output without overflowing.
        let weights = make_dn_weights(
            vec![1.0_f32; dims.hidden],
            Array2::from_elem((conv_dim, dims.hidden), 0.1_f32),
            Array2::from_elem((value_dim, dims.hidden), 0.1_f32),
            Array2::from_elem((dims.d_conv, conv_dim), 0.5_f32),
            vec![0.0_f32; dims.n_v_heads],
            vec![-1.0_f32; dims.n_v_heads], // moderate decay
            Array2::from_elem((dims.n_v_heads, dims.hidden), 0.1_f32),
            Array2::from_elem((dims.n_v_heads, dims.hidden), 0.1_f32),
            vec![1.0_f32; dims.head_v_dim],
            Array2::from_elem((dims.hidden, value_dim), 0.5_f32),
        );

        let mut state =
            DeltaNetLayerState::allocate(dims.d_conv, conv_dim, dims.head_v_dim, dims.n_v_heads);
        let x = Array1::from_elem(dims.hidden, 1.0_f32);

        let y = deltanet_block_step(&x, &weights, &dims, &mut state);
        assert_eq!(y.len(), dims.hidden);
        assert!(y.iter().all(|v| v.is_finite()));

        // Conv state should now hold the most recent qkv_mixed value
        // (since d_conv=2 → state_rows=1). For our constant inputs
        // qkv_mixed = attn_qkv @ x_norm; x_norm has unit RMS so its
        // norm scales x. We don't pin the value, just that it
        // moved off zero.
        let _ = key_dim;
        assert!(
            state.conv_state.iter().any(|v| v.abs() > 0.0),
            "conv_state should have absorbed the new token"
        );
        // Recurrent state likewise non-zero after step (rank-1 update
        // from k ⊗ d).
        assert!(
            state.recurrent_state.iter().any(|v| v.abs() > 0.0),
            "recurrent_state should reflect the rank-1 update"
        );
    }

    #[test]
    fn deltanet_block_step_zero_input_zero_output() {
        // x = 0 → x_norm = 0 (because RMS of zero vector with eps
        // gives x/sqrt(eps) * weight = 0). All projections = 0 →
        // conv input = 0, qkv_conv = SiLU(conv_out) where conv_out
        // depends on the (zero) state. First step: state was zero,
        // qkv_mixed = 0, so qkv_conv = 0. Recurrence inputs zero;
        // delta-rule rank-1 update is zero. Output projection of
        // zero = zero.
        let dims = make_tiny_dims();
        let conv_dim = dims.conv_dim();
        let value_dim = dims.value_dim();

        let weights = make_dn_weights(
            vec![1.0_f32; dims.hidden],
            Array2::zeros((conv_dim, dims.hidden)),
            Array2::zeros((value_dim, dims.hidden)),
            Array2::zeros((dims.d_conv, conv_dim)),
            vec![0.0_f32; dims.n_v_heads],
            vec![0.0_f32; dims.n_v_heads],
            Array2::zeros((dims.n_v_heads, dims.hidden)),
            Array2::zeros((dims.n_v_heads, dims.hidden)),
            vec![1.0_f32; dims.head_v_dim],
            Array2::zeros((dims.hidden, value_dim)),
        );

        let mut state =
            DeltaNetLayerState::allocate(dims.d_conv, conv_dim, dims.head_v_dim, dims.n_v_heads);
        let x = Array1::zeros(dims.hidden);

        let y = deltanet_block_step(&x, &weights, &dims, &mut state);
        for &v in y.iter() {
            assert!(v.abs() < 1e-5, "expected zero output, got {v}");
        }
    }

    #[test]
    fn deltanet_block_dims_helpers() {
        let dims = DeltaNetDims {
            hidden: 5120,
            head_v_dim: 128,
            n_v_heads: 48,
            n_k_heads: 16,
            d_conv: 4,
            eps: 1e-6,
        };
        // Qwen 3.6 27B sanity check.
        assert_eq!(dims.key_dim(), 128 * 16); // 2048
        assert_eq!(dims.value_dim(), 128 * 48); // 6144
        assert_eq!(dims.conv_dim(), 2 * 2048 + 6144); // 10240
    }

    #[test]
    fn rms_norm_1d_unit_weight_normalises() {
        let x = ndarray::array![3.0_f32, 4.0]; // RMS = sqrt((9+16)/2) = sqrt(12.5)
        let w = [1.0_f32, 1.0];
        let out = rms_norm_1d(&x, &w, 0.0);
        let rms = (12.5_f32).sqrt();
        assert!((out[0] - 3.0 / rms).abs() < 1e-5);
        assert!((out[1] - 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn rms_norm_1d_applies_weight() {
        let x = ndarray::array![3.0_f32, 4.0];
        let w = [2.0_f32, 0.5];
        let out = rms_norm_1d(&x, &w, 0.0);
        let rms = (12.5_f32).sqrt();
        assert!((out[0] - 2.0 * 3.0 / rms).abs() < 1e-5);
        assert!((out[1] - 0.5 * 4.0 / rms).abs() < 1e-5);
    }

    #[test]
    fn deltanet_block_step_two_calls_state_carries_forward() {
        // Run the block twice with the same input. State should
        // accumulate: the second call's recurrent state has more
        // mass than the first.
        let dims = make_tiny_dims();
        let conv_dim = dims.conv_dim();
        let value_dim = dims.value_dim();

        let weights = make_dn_weights(
            vec![1.0_f32; dims.hidden],
            Array2::from_elem((conv_dim, dims.hidden), 0.1_f32),
            Array2::from_elem((value_dim, dims.hidden), 0.1_f32),
            Array2::from_elem((dims.d_conv, conv_dim), 0.5_f32),
            vec![0.0_f32; dims.n_v_heads],
            vec![0.0_f32; dims.n_v_heads], // no decay; state accumulates
            Array2::from_elem((dims.n_v_heads, dims.hidden), 0.1_f32),
            Array2::from_elem((dims.n_v_heads, dims.hidden), 0.0_f32),
            vec![1.0_f32; dims.head_v_dim],
            Array2::from_elem((dims.hidden, value_dim), 0.5_f32),
        );

        let mut state =
            DeltaNetLayerState::allocate(dims.d_conv, conv_dim, dims.head_v_dim, dims.n_v_heads);
        let x = Array1::from_elem(dims.hidden, 1.0_f32);

        let _y1 = deltanet_block_step(&x, &weights, &dims, &mut state);
        let state_mass_after_1: f32 = state.recurrent_state.iter().map(|&v| v.abs()).sum();
        let _y2 = deltanet_block_step(&x, &weights, &dims, &mut state);
        let state_mass_after_2: f32 = state.recurrent_state.iter().map(|&v| v.abs()).sum();

        // Without decay, the state should retain at least as much
        // mass after step 2 as after step 1 (and typically more —
        // rank-1 updates compound when inputs are non-orthogonal).
        assert!(
            state_mass_after_2 >= state_mass_after_1 - 1e-5,
            "state mass shrank without decay: {state_mass_after_1} → {state_mass_after_2}"
        );
    }
}
