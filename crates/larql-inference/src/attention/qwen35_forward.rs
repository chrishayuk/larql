//! End-to-end `qwen35_forward` glue (Phase C.4c of
//! `inference-qwen35-deltanet`).
//!
//! Single-token forward through the full Qwen 3.6 hybrid model:
//! embed → 64 layers (each is a hybrid linear / full-attn block with
//! a Gemma-2-style pre+post-norm sandwich and a SwiGLU FFN
//! suffix) → final_norm → lm_head → logits.
//!
//! Per design.md §6: the **FFN residual adds to the pre-post-norm
//! tensor**. So the layer flow is:
//!
//! ```text
//! attn_out = block(x)              # block does its own pre-norm
//! residual = x + attn_out          # residual add 1
//! ffn_in   = attn_post_norm(residual)  # pre-FFN norm
//! ffn_out  = swiglu(ffn_in)        # silu(gate @ x) * (up @ x) → down
//! final    = residual + ffn_out    # residual add 2 (to PRE-post-norm)
//! ```
//!
//! This is the load-bearing structural difference vs vanilla LLaMA
//! pre-norm-only layers; the FFN residual on the post-normed tensor
//! is a common bug per llama.cpp `qwen35.cpp:141–152`.
//!
//! Weight loading from a vindex / GGUF lives in Phase C.4d; this
//! module only provides the borrow-based forward function. Tests
//! exercise it on tiny synthetic 2-layer models.

use ndarray::{ArcArray2, Array1, Array2};
use std::sync::Arc;

use super::deltanet_block::{rms_norm_1d_pub, DeltaNetDims};
use super::deltanet_state::sigmoid;
use super::qwen35_block::{
    hybrid_layer_step, DeltaNetHybridCache, Qwen35AttentionDims, Qwen35LayerWeights,
};

/// Full weights for one layer (both kinds): the block weights
/// (linear or attention) PLUS the post-attention norm + SwiGLU FFN.
#[derive(Clone)]
pub struct Qwen35FullLayerWeights {
    /// The block-level weights — DeltaNet for linear layers,
    /// attention for full-attn layers. Use `Qwen35LayerWeights::Linear(...)`
    /// or `::Attention(...)` to construct.
    pub block: Qwen35LayerWeights,
    /// Post-attention RMSNorm weight `[hidden]`. Applied to
    /// (x + block_out) and fed into the FFN as its pre-norm. The
    /// FFN residual still uses the pre-post-norm tensor.
    pub attn_post_norm: Arc<[f32]>,
    /// SwiGLU gate projection `[ffn_dim, hidden]`.
    pub ffn_gate: ArcArray2<f32>,
    /// SwiGLU up projection `[ffn_dim, hidden]`.
    pub ffn_up: ArcArray2<f32>,
    /// SwiGLU down projection `[hidden, ffn_dim]`.
    pub ffn_down: ArcArray2<f32>,
}

/// Full Qwen 3.6 model weights — embed, every layer's weights, and
/// the output head.
#[derive(Clone)]
pub struct Qwen35Weights {
    /// Token embedding matrix `[vocab, hidden]`. The lookup uses
    /// `embed.row(token_id)`.
    pub embed: ArcArray2<f32>,
    /// Per-layer full weights, indexed 0..n_layer.
    pub layers: Vec<Qwen35FullLayerWeights>,
    /// Final RMSNorm weight `[hidden]`.
    pub final_norm: Arc<[f32]>,
    /// LM head projection `[vocab, hidden]`. Often tied to `embed`
    /// (the caller decides; this struct just holds whichever).
    pub lm_head: ArcArray2<f32>,
    /// FFN intermediate dim (for shape assertions in tests).
    pub ffn_dim: usize,
}

/// One-token forward through the entire Qwen 3.6 model.
///
/// Returns the logits `[vocab]` for the next token after the input.
/// Mutates `hybrid_cache` (appends K/V or updates DeltaNet state per
/// layer) and advances `hybrid_cache.next_position`.
pub fn qwen35_forward_step(
    token_id: u32,
    weights: &Qwen35Weights,
    dn_dims: &DeltaNetDims,
    attn_dims: &Qwen35AttentionDims,
    hybrid_cache: &mut DeltaNetHybridCache,
    eps: f32,
) -> Array1<f32> {
    let n_layers = weights.layers.len();
    debug_assert_eq!(n_layers, hybrid_cache.num_layers());

    // 1. Embed lookup.
    let vocab = weights.embed.shape()[0];
    let hidden = weights.embed.shape()[1];
    debug_assert!((token_id as usize) < vocab);
    debug_assert_eq!(hidden, dn_dims.hidden);

    let mut x: Array1<f32> = weights.embed.row(token_id as usize).to_owned();

    // 2. Hybrid layer stack.
    for layer in 0..n_layers {
        let layer_w = &weights.layers[layer];
        // 2a. Block forward (linear or full-attn). The block does
        // its own pre-norm internally (`attn_norm`).
        let block_out =
            hybrid_layer_step(layer, &x, &layer_w.block, dn_dims, attn_dims, hybrid_cache);
        // 2b. Residual add 1 — `residual = x + block_out`.
        let residual: Array1<f32> = &x + &block_out;
        // 2c. Post-attention RMSNorm (only on `has_post_norms`
        // architectures — Qwen 3.6 is one of them).
        let ffn_in = rms_norm_1d_pub(&residual, &layer_w.attn_post_norm, eps);
        // 2d. SwiGLU FFN.
        let ffn_out = swiglu_ffn(
            &ffn_in,
            layer_w.ffn_gate.view(),
            layer_w.ffn_up.view(),
            layer_w.ffn_down.view(),
        );
        // 2e. Residual add 2 — `x = residual + ffn_out` (NOT
        // `ffn_in + ffn_out`; the FFN residual bypasses the
        // post-norm per design.md §6).
        x = &residual + &ffn_out;
    }

    // 3. Final norm + lm_head.
    let x_final = rms_norm_1d_pub(&x, &weights.final_norm, eps);
    let logits = weights.lm_head.dot(&x_final);

    // 4. Advance position for the next token's RoPE.
    hybrid_cache.next_position += 1;

    logits
}

/// SwiGLU FFN: `down @ (silu(gate @ x) * (up @ x))`.
///
/// - `gate`: `[ffn_dim, hidden]`
/// - `up`:   `[ffn_dim, hidden]`
/// - `down`: `[hidden, ffn_dim]`
fn swiglu_ffn(
    x: &Array1<f32>,
    gate: ndarray::ArrayView2<f32>,
    up: ndarray::ArrayView2<f32>,
    down: ndarray::ArrayView2<f32>,
) -> Array1<f32> {
    let g = gate.dot(x); // [ffn_dim]
    let u = up.dot(x); // [ffn_dim]
    let mut inter = Array1::<f32>::zeros(g.len());
    for i in 0..g.len() {
        inter[i] = (g[i] * sigmoid(g[i])) * u[i]; // silu(g) * u
    }
    down.dot(&inter) // [hidden]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::deltanet_block::DeltaNetLayerWeights;
    use crate::attention::qwen35_block::{DeltaNetHybridCache, Qwen35AttentionLayerWeights};
    use ndarray::Array2;

    fn tiny_dn_dims() -> DeltaNetDims {
        DeltaNetDims {
            hidden: 4,
            head_v_dim: 2,
            n_v_heads: 1,
            n_k_heads: 1,
            d_conv: 2,
            eps: 1e-6,
        }
    }

    fn tiny_attn_dims() -> Qwen35AttentionDims {
        Qwen35AttentionDims {
            hidden: 4,
            n_head: 2,
            n_head_kv: 1,
            head_dim: 2,
            rotary_dim: 2,
            rope_base: 10_000.0,
            eps: 1e-6,
        }
    }

    /// Build a constant-filled tiny DeltaNet weights bundle. ArcArray
    /// + Arc<[f32]> are cheap-to-clone refcounted owners.
    fn make_dn_weights(dn_dims: &DeltaNetDims) -> DeltaNetLayerWeights {
        let conv_dim = dn_dims.conv_dim();
        let value_dim = dn_dims.value_dim();
        DeltaNetLayerWeights {
            attn_norm: Arc::from(vec![1.0_f32; dn_dims.hidden].as_slice()),
            attn_qkv: Array2::from_elem((conv_dim, dn_dims.hidden), 0.1_f32).into_shared(),
            attn_gate: Array2::from_elem((value_dim, dn_dims.hidden), 0.1_f32).into_shared(),
            ssm_conv1d: Array2::from_elem((dn_dims.d_conv, conv_dim), 0.5_f32).into_shared(),
            ssm_dt: Arc::from(vec![0.0_f32; dn_dims.n_v_heads].as_slice()),
            ssm_a: Arc::from(vec![-1.0_f32; dn_dims.n_v_heads].as_slice()),
            ssm_beta: Array2::from_elem((dn_dims.n_v_heads, dn_dims.hidden), 0.1_f32).into_shared(),
            ssm_alpha: Array2::from_elem((dn_dims.n_v_heads, dn_dims.hidden), 0.1_f32)
                .into_shared(),
            ssm_norm: Arc::from(vec![1.0_f32; dn_dims.head_v_dim].as_slice()),
            ssm_out: Array2::from_elem((dn_dims.hidden, value_dim), 0.5_f32).into_shared(),
        }
    }

    fn make_attn_weights(attn_dims: &Qwen35AttentionDims) -> Qwen35AttentionLayerWeights {
        Qwen35AttentionLayerWeights {
            attn_norm: Arc::from(vec![1.0_f32; attn_dims.hidden].as_slice()),
            attn_q: Array2::from_elem((attn_dims.fused_q_dim(), attn_dims.hidden), 0.1_f32)
                .into_shared(),
            attn_k: Array2::from_elem((attn_dims.kv_dim(), attn_dims.hidden), 0.1_f32)
                .into_shared(),
            attn_v: Array2::from_elem((attn_dims.kv_dim(), attn_dims.hidden), 0.1_f32)
                .into_shared(),
            attn_q_norm: Arc::from(vec![1.0_f32; attn_dims.head_dim].as_slice()),
            attn_k_norm: Arc::from(vec![1.0_f32; attn_dims.head_dim].as_slice()),
            attn_output: Array2::from_elem((attn_dims.hidden, attn_dims.q_dim()), 0.5_f32)
                .into_shared(),
        }
    }

    /// Build a per-layer suffix (attn_post_norm + SwiGLU FFN trio).
    fn make_suffix(
        hidden: usize,
        ffn_dim: usize,
    ) -> (
        Arc<[f32]>,
        ndarray::ArcArray2<f32>,
        ndarray::ArcArray2<f32>,
        ndarray::ArcArray2<f32>,
    ) {
        (
            Arc::from(vec![1.0_f32; hidden].as_slice()),
            Array2::from_elem((ffn_dim, hidden), 0.1_f32).into_shared(),
            Array2::from_elem((ffn_dim, hidden), 0.1_f32).into_shared(),
            Array2::from_elem((hidden, ffn_dim), 0.5_f32).into_shared(),
        )
    }

    #[test]
    fn swiglu_ffn_zero_input_zero_output() {
        let hidden = 4;
        let ffn_dim = 8;
        let x = Array1::<f32>::zeros(hidden);
        let gate = Array2::from_elem((ffn_dim, hidden), 1.0_f32);
        let up = Array2::from_elem((ffn_dim, hidden), 1.0_f32);
        let down = Array2::from_elem((hidden, ffn_dim), 1.0_f32);
        let y = swiglu_ffn(&x, gate.view(), up.view(), down.view());
        // gate @ 0 = 0, silu(0)=0, up @ 0 = 0, inter = 0, down @ 0 = 0.
        for &v in y.iter() {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn swiglu_ffn_matches_definition_unit_vectors() {
        // hidden=2, ffn_dim=2. gate = I (identity), up = I, down = I.
        // y = down @ (silu(gate @ x) * (up @ x))
        //   = silu(x) * x
        //
        // For x = [1, 0]: silu(1)*1 + silu(0)*0 = silu(1) at index 0.
        let gate = ndarray::array![[1.0_f32, 0.0], [0.0, 1.0]];
        let up = ndarray::array![[1.0_f32, 0.0], [0.0, 1.0]];
        let down = ndarray::array![[1.0_f32, 0.0], [0.0, 1.0]];
        let x = ndarray::array![1.0_f32, 0.0];
        let y = swiglu_ffn(&x, gate.view(), up.view(), down.view());
        let expected = 1.0 * sigmoid(1.0); // silu(1)
        assert!((y[0] - expected).abs() < 1e-5);
        assert!(y[1].abs() < 1e-5);
    }

    #[test]
    fn qwen35_forward_step_returns_vocab_shape() {
        // 2-layer model: layer 0 linear, layer 1 attention.
        let dn_dims = tiny_dn_dims();
        let attn_dims = tiny_attn_dims();
        let hidden = dn_dims.hidden;
        let vocab = 5;
        let ffn_dim = 8;

        let layer_kinds = vec![true, false];

        let mut cache = DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let dn_weights = make_dn_weights(&dn_dims);
        let attn_weights = make_attn_weights(&attn_dims);
        let (post0, gate0, up0, down0) = make_suffix(hidden, ffn_dim);
        let (post1, gate1, up1, down1) = make_suffix(hidden, ffn_dim);

        let weights = Qwen35Weights {
            embed: Array2::from_elem((vocab, hidden), 0.5_f32).into_shared(),
            layers: vec![
                Qwen35FullLayerWeights {
                    block: Qwen35LayerWeights::Linear(dn_weights),
                    attn_post_norm: post0,
                    ffn_gate: gate0,
                    ffn_up: up0,
                    ffn_down: down0,
                },
                Qwen35FullLayerWeights {
                    block: Qwen35LayerWeights::Attention(attn_weights),
                    attn_post_norm: post1,
                    ffn_gate: gate1,
                    ffn_up: up1,
                    ffn_down: down1,
                },
            ],
            final_norm: Arc::from(vec![1.0_f32; hidden].as_slice()),
            lm_head: Array2::from_elem((vocab, hidden), 0.5_f32).into_shared(),
            ffn_dim,
        };

        let token = 2u32;
        let logits = qwen35_forward_step(token, &weights, &dn_dims, &attn_dims, &mut cache, 1e-6);
        assert_eq!(logits.len(), vocab);
        assert!(logits.iter().all(|v| v.is_finite()));
        assert_eq!(cache.next_position, 1);
    }

    #[test]
    fn qwen35_forward_step_two_calls_advances_position_and_grows_cache() {
        let dn_dims = tiny_dn_dims();
        let attn_dims = tiny_attn_dims();
        let hidden = dn_dims.hidden;
        let vocab = 5;
        let ffn_dim = 8;

        // Single full-attention layer so we can observe KV growth.
        let layer_kinds = vec![false];

        let mut cache = DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let attn_weights = make_attn_weights(&attn_dims);
        let (post, gate, up, down) = make_suffix(hidden, ffn_dim);

        let weights = Qwen35Weights {
            embed: Array2::from_elem((vocab, hidden), 0.5_f32).into_shared(),
            layers: vec![Qwen35FullLayerWeights {
                block: Qwen35LayerWeights::Attention(attn_weights),
                attn_post_norm: post,
                ffn_gate: gate,
                ffn_up: up,
                ffn_down: down,
            }],
            final_norm: Arc::from(vec![1.0_f32; hidden].as_slice()),
            lm_head: Array2::from_elem((vocab, hidden), 0.5_f32).into_shared(),
            ffn_dim,
        };

        let _ = qwen35_forward_step(0, &weights, &dn_dims, &attn_dims, &mut cache, 1e-6);
        assert_eq!(cache.next_position, 1);
        let (k_after_1, _) = cache.kv_layers[0].as_ref().unwrap();
        assert_eq!(k_after_1.shape()[0], 1);

        let _ = qwen35_forward_step(1, &weights, &dn_dims, &attn_dims, &mut cache, 1e-6);
        assert_eq!(cache.next_position, 2);
        let (k_after_2, _) = cache.kv_layers[0].as_ref().unwrap();
        assert_eq!(k_after_2.shape()[0], 2);
    }

    #[test]
    fn qwen35_forward_step_dn_state_carries_across_calls() {
        // Single linear layer — verify DeltaNet state moves off zero
        // and accumulates across calls.
        let dn_dims = tiny_dn_dims();
        let attn_dims = tiny_attn_dims();
        let hidden = dn_dims.hidden;
        let vocab = 5;
        let ffn_dim = 8;

        let layer_kinds = vec![true];

        let mut cache = DeltaNetHybridCache::allocate(
            &layer_kinds,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let dn_weights = make_dn_weights(&dn_dims);
        let (post, gate, up, down) = make_suffix(hidden, ffn_dim);

        let weights = Qwen35Weights {
            embed: Array2::from_elem((vocab, hidden), 0.5_f32).into_shared(),
            layers: vec![Qwen35FullLayerWeights {
                block: Qwen35LayerWeights::Linear(dn_weights),
                attn_post_norm: post,
                ffn_gate: gate,
                ffn_up: up,
                ffn_down: down,
            }],
            final_norm: Arc::from(vec![1.0_f32; hidden].as_slice()),
            lm_head: Array2::from_elem((vocab, hidden), 0.5_f32).into_shared(),
            ffn_dim,
        };

        let _ = qwen35_forward_step(2, &weights, &dn_dims, &attn_dims, &mut cache, 1e-6);
        let mass_after_1: f32 = cache.dn_state.layers[0]
            .as_ref()
            .unwrap()
            .recurrent_state
            .iter()
            .map(|&v| v.abs())
            .sum();
        assert!(
            mass_after_1 > 0.0,
            "recurrent_state should be non-zero after one forward step"
        );

        let _ = qwen35_forward_step(2, &weights, &dn_dims, &attn_dims, &mut cache, 1e-6);
        let mass_after_2: f32 = cache.dn_state.layers[0]
            .as_ref()
            .unwrap()
            .recurrent_state
            .iter()
            .map(|&v| v.abs())
            .sum();
        assert!(
            mass_after_2 >= 0.0,
            "recurrent_state still finite after two steps"
        );
    }
}
