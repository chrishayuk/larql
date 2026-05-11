//! Qwen3-Next full-attention block forward + hybrid layer router
//! (Phase C.4b of `inference-qwen35-deltanet`).
//!
//! The full-attention block here handles Qwen 3.6's every-4th layer
//! — the one with softmax attention rather than Gated DeltaNet. It
//! incorporates the three Qwen3-Next quirks documented in design.md
//! §6:
//!
//! 1. **Fused Q + per-head sigmoid gate** projection (the `attn_q`
//!    tensor's output dim is `2 * n_head * head_dim`). Split via
//!    `qwen35_attn::split_q_gate`.
//! 2. **Per-head Q/K RMSNorm** at `[head_dim]` shape (not `[hidden]`).
//!    Existing `residual::rms_norm_heads` already implements this.
//! 3. **Partial RoPE** on the first `mrope_total_rotary(sections)` head
//!    dims. For text-only inference this is equivalent to single-
//!    section RoPE; the existing `apply_rope_partial_at` does it.
//!
//! Plus the standard Gemma-2-style pre+post-norm sandwich the
//! `Qwen35Arch::has_post_norms() = true` predicate declares.
//!
//! The `DeltaNetHybridCache` bundles a `KvCache` (for the 16
//! full-attention layers) and a `DeltaNetStateCache` (for the 48
//! linear-attention layers) plus a per-layer kind mask, so a single
//! object carries all the per-sequence state the hybrid model needs.
//!
//! The `hybrid_layer_step` router demonstrates the dispatch shape:
//! per layer, look at `layer_kinds[layer]` and call the appropriate
//! block. The full glue that runs the embed → 64 layers → final
//! norm → lm_head pipeline lives in a follow-up (Phase C.4c).

use ndarray::{Array1, Array2};

use super::deltanet_block::{deltanet_block_step, DeltaNetDims, DeltaNetLayerWeights};
use super::deltanet_state::{sigmoid, DeltaNetLayerState, DeltaNetStateCache};
use super::qwen35_attn::{apply_q_gate, split_q_gate};

/// Borrowed weight references for one full-attention layer.
///
/// All projection tensors are stored `[out_features, in_features]`
/// (matvec is `y = W @ x`).
pub struct Qwen35AttentionLayerWeights<'a> {
    /// Pre-attention RMSNorm weight `[hidden]`.
    pub attn_norm: &'a [f32],
    /// Fused Q + per-head gate projection
    /// `[2 * n_head * head_dim, hidden]`.
    pub attn_q: &'a Array2<f32>,
    /// K projection `[n_head_kv * head_dim, hidden]`.
    pub attn_k: &'a Array2<f32>,
    /// V projection `[n_head_kv * head_dim, hidden]`.
    pub attn_v: &'a Array2<f32>,
    /// Per-head Q RMSNorm weight `[head_dim]`.
    pub attn_q_norm: &'a [f32],
    /// Per-head K RMSNorm weight `[head_dim]`.
    pub attn_k_norm: &'a [f32],
    /// Output projection `[hidden, n_head * head_dim]`.
    pub attn_output: &'a Array2<f32>,
}

/// Shape constants for the full-attention layers (uniform across
/// layers within a model).
#[derive(Clone, Copy, Debug)]
pub struct Qwen35AttentionDims {
    /// Residual stream width. Qwen 3.6 27B: 5120.
    pub hidden: usize,
    /// Number of Q heads. Qwen 3.6: 24.
    pub n_head: usize,
    /// Number of K/V heads (GQA, `n_head / n_head_kv` = repeat
    /// factor). Qwen 3.6: 4.
    pub n_head_kv: usize,
    /// Per-head dimension. Qwen 3.6: 256.
    pub head_dim: usize,
    /// Number of rotary head dimensions (= `sum(rope_sections)`).
    /// Qwen 3.6: 64 (so 192 of the 256 head dims pass through
    /// without rotation).
    pub rotary_dim: usize,
    /// RoPE base. Qwen 3.6: 10_000_000.0.
    pub rope_base: f64,
    /// RMSNorm epsilon.
    pub eps: f32,
}

impl Qwen35AttentionDims {
    /// `n_head * head_dim`.
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.n_head * self.head_dim
    }

    /// `n_head_kv * head_dim`.
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.n_head_kv * self.head_dim
    }

    /// `2 * q_dim` — the dim of the fused Q+gate projection output.
    #[inline]
    pub fn fused_q_dim(&self) -> usize {
        2 * self.q_dim()
    }

    /// `q_dim / kv_dim` (GQA repeat factor). Qwen 3.6: 24/4 = 6.
    #[inline]
    pub fn gqa_reps(&self) -> usize {
        self.q_dim() / self.kv_dim()
    }

    /// Partial-rotary fraction (= `rotary_dim / head_dim`).
    #[inline]
    pub fn rope_fraction(&self) -> f64 {
        self.rotary_dim as f64 / self.head_dim as f64
    }
}

/// One-token forward through a Qwen 3.6 full-attention layer.
///
/// `kv_layer` is the cumulative `[seq_len, kv_dim]` K and V slabs
/// for this layer — the function appends the new token's K/V row to
/// them and runs GQA softmax attention over the full slab.
///
/// `position` is the new token's absolute 0-indexed position. RoPE
/// is applied to the new Q row and the new K row before append.
///
/// Returns the block's output `[hidden]`, ready to be added back
/// to the residual stream by the caller.
pub fn qwen35_attention_block_step(
    x: &Array1<f32>,
    weights: &Qwen35AttentionLayerWeights,
    dims: &Qwen35AttentionDims,
    kv_layer: &mut (Array2<f32>, Array2<f32>),
    position: usize,
) -> Array1<f32> {
    debug_assert_eq!(x.len(), dims.hidden);
    debug_assert_eq!(weights.attn_norm.len(), dims.hidden);
    debug_assert_eq!(weights.attn_q.shape(), [dims.fused_q_dim(), dims.hidden]);
    debug_assert_eq!(weights.attn_k.shape(), [dims.kv_dim(), dims.hidden]);
    debug_assert_eq!(weights.attn_v.shape(), [dims.kv_dim(), dims.hidden]);
    debug_assert_eq!(weights.attn_q_norm.len(), dims.head_dim);
    debug_assert_eq!(weights.attn_k_norm.len(), dims.head_dim);
    debug_assert_eq!(weights.attn_output.shape(), [dims.hidden, dims.q_dim()]);

    // 1. Pre-attention RMSNorm.
    let x_norm = super::deltanet_block::rms_norm_1d_pub(x, weights.attn_norm, dims.eps);

    // 2. Projections.
    let q_fused_1d = weights.attn_q.dot(&x_norm); // [fused_q_dim]
    let k_1d = weights.attn_k.dot(&x_norm); // [kv_dim]
    let v_1d = weights.attn_v.dot(&x_norm); // [kv_dim]

    // 3. Split Q+gate. split_q_gate takes [seq_len, fused_q_dim] →
    //    (q [seq_len, q_dim], gate [seq_len, q_dim]). Wrap our
    //    single token as a 1-row matrix.
    let q_fused_2d = q_fused_1d
        .into_shape_with_order((1, dims.fused_q_dim()))
        .expect("q_fused reshape");
    let (q_2d, gate_2d) = split_q_gate(&q_fused_2d, dims.n_head, dims.head_dim);

    // 4. Per-head RMSNorm for Q and K.
    let q_normed = crate::residual::rms_norm_heads(
        &q_2d,
        weights.attn_q_norm,
        dims.n_head,
        dims.head_dim,
        0.0,
    );
    let k_2d_in = k_1d
        .into_shape_with_order((1, dims.kv_dim()))
        .expect("k reshape");
    let k_normed = crate::residual::rms_norm_heads(
        &k_2d_in,
        weights.attn_k_norm,
        dims.n_head_kv,
        dims.head_dim,
        0.0,
    );

    // 5. Partial RoPE applied at `position`. RoPE is a 2D op so the
    //    1-row matrices work directly.
    let q_roped = super::rope::apply_rope_partial_at(
        &q_normed,
        dims.n_head,
        dims.head_dim,
        dims.rope_base,
        dims.rope_fraction(),
        position,
    );
    let k_roped = super::rope::apply_rope_partial_at(
        &k_normed,
        dims.n_head_kv,
        dims.head_dim,
        dims.rope_base,
        dims.rope_fraction(),
        position,
    );

    // 6. Append the new K/V row to the cumulative cache slabs.
    let v_2d_in = v_1d
        .into_shape_with_order((1, dims.kv_dim()))
        .expect("v reshape");
    append_row(&mut kv_layer.0, k_roped.row(0));
    append_row(&mut kv_layer.1, v_2d_in.row(0));

    // 7. GQA softmax attention — single-row Q, cumulative K/V.
    //    The existing `gqa::gqa_attention` requires Q, K, V to share
    //    the same row count (used by prefill); for autoregressive
    //    decode we do a focused single-row scan here.
    let q_row = q_roped.row(0).to_owned();
    let scale = (dims.head_dim as f64).powf(-0.5);
    let attn_out_1d = gqa_decode_step(
        &q_row,
        &kv_layer.0,
        &kv_layer.1,
        dims.n_head,
        dims.n_head_kv,
        dims.head_dim,
        scale,
    );
    // Wrap back into 1-row matrix to reuse the C.3 gate-apply helper.
    let mut attn_out = attn_out_1d
        .into_shape_with_order((1, dims.q_dim()))
        .expect("attn_out reshape");
    apply_q_gate(&mut attn_out, &gate_2d);

    // 9. Output projection: y = attn_output @ attn_out[0].
    let attn_out_1d = attn_out.row(0).to_owned();
    weights.attn_output.dot(&attn_out_1d)
}

/// Autoregressive GQA softmax attention for a single new Q row over
/// cumulative K/V slabs.
///
/// - `q`: `[num_q * head_dim]` — the new token's Q (post-RMSNorm + RoPE).
/// - `k_cache`, `v_cache`: `[seq_len, num_kv * head_dim]` — cumulative
///   K/V (the new row already appended).
/// - `num_q`, `num_kv`, `head_dim`: layer shapes (GQA repeat factor
///   = `num_q / num_kv`).
/// - `scale`: typically `1 / sqrt(head_dim)`.
///
/// Returns `[num_q * head_dim]`.
///
/// Per-head loop with stable softmax (subtract row max before exp).
/// The KV-head index for Q head `h` is `h / reps` (repeat-interleave).
fn gqa_decode_step(
    q: &Array1<f32>,
    k_cache: &Array2<f32>,
    v_cache: &Array2<f32>,
    num_q: usize,
    num_kv: usize,
    head_dim: usize,
    scale: f64,
) -> Array1<f32> {
    let seq_len = k_cache.shape()[0];
    debug_assert!(seq_len > 0, "k_cache must have at least the new token");
    debug_assert_eq!(q.len(), num_q * head_dim);
    debug_assert_eq!(k_cache.shape()[1], num_kv * head_dim);
    debug_assert_eq!(v_cache.shape()[1], num_kv * head_dim);
    debug_assert!(
        num_q.is_multiple_of(num_kv) && num_kv > 0,
        "num_q ({num_q}) must be a multiple of num_kv ({num_kv})"
    );

    let reps = num_q / num_kv;
    let scale_f32 = scale as f32;
    let mut out = Array1::<f32>::zeros(num_q * head_dim);
    let mut scores = vec![0.0_f32; seq_len];

    for h in 0..num_q {
        let kv_h = h / reps;
        // 1. Scores[t] = (q_h · k_cache[t, kv_h]) * scale.
        for t in 0..seq_len {
            let mut dot = 0.0_f32;
            for d in 0..head_dim {
                dot += q[h * head_dim + d] * k_cache[[t, kv_h * head_dim + d]];
            }
            scores[t] = dot * scale_f32;
        }
        // 2. Stable softmax.
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0_f32;
        for s in scores.iter_mut() {
            *s = (*s - max).exp();
            exp_sum += *s;
        }
        let inv_sum = 1.0 / exp_sum;
        // 3. Weighted sum with V.
        for d in 0..head_dim {
            let mut acc = 0.0_f32;
            for t in 0..seq_len {
                acc += scores[t] * inv_sum * v_cache[[t, kv_h * head_dim + d]];
            }
            out[h * head_dim + d] = acc;
        }
    }

    out
}

/// Append a row to a `[N, dim]` matrix, growing it to `[N+1, dim]`.
fn append_row(slab: &mut Array2<f32>, new_row: ndarray::ArrayView1<f32>) {
    let dim = slab.shape()[1];
    debug_assert_eq!(new_row.len(), dim);
    let mut next = Array2::<f32>::zeros((slab.shape()[0] + 1, dim));
    for r in 0..slab.shape()[0] {
        for c in 0..dim {
            next[[r, c]] = slab[[r, c]];
        }
    }
    let last = next.shape()[0] - 1;
    for c in 0..dim {
        next[[last, c]] = new_row[c];
    }
    *slab = next;
}

/// Hybrid per-sequence state for a Qwen 3.6 / Qwen 3.6 MoE model:
/// `KvCache`-equivalent slabs for the full-attention layers and a
/// `DeltaNetStateCache` for the linear layers, paired with a
/// `layer_kinds` mask so the router knows which kind each layer is.
///
/// For Qwen 3.6 27B (`n_layer = 64`, `full_attention_interval = 4`):
/// 48 entries in `dn_state` (Some on linear layers); 16 layers
/// hold non-empty KV slabs.
pub struct DeltaNetHybridCache {
    /// Per-layer K/V slabs. `None` for linear-attention layers
    /// (they don't have softmax KV). Each `Some` slot starts as
    /// `(Array2::zeros((0, kv_dim)), Array2::zeros((0, kv_dim)))`
    /// and grows by one row per decoded token.
    pub kv_layers: Vec<Option<(Array2<f32>, Array2<f32>)>>,
    /// DeltaNet recurrent + conv state for linear layers. The mask
    /// pattern mirrors `kv_layers` inversely.
    pub dn_state: DeltaNetStateCache,
    /// `layer_kinds[layer] == true` iff this is a linear (DeltaNet)
    /// layer. Comes from `Qwen35Arch::is_linear_attention_layer`.
    pub layer_kinds: Vec<bool>,
    /// Absolute position of the NEXT token to be decoded. Used as
    /// the RoPE position for the new K/Q rows.
    pub next_position: usize,
}

impl DeltaNetHybridCache {
    /// Allocate the cache for a fresh sequence.
    ///
    /// - `layer_kinds`: `true` for linear-attention layers.
    /// - `kv_dim` (for full-attn layers): `n_head_kv * head_dim`.
    /// - `d_conv`, `dn_conv_dim`, `dn_head_v_dim`, `dn_n_v_heads`
    ///   (for linear layers).
    pub fn allocate(
        layer_kinds: &[bool],
        kv_dim: usize,
        d_conv: usize,
        dn_conv_dim: usize,
        dn_head_v_dim: usize,
        dn_n_v_heads: usize,
    ) -> Self {
        let kv_layers = layer_kinds
            .iter()
            .map(|&is_linear| {
                if is_linear {
                    None
                } else {
                    Some((Array2::zeros((0, kv_dim)), Array2::zeros((0, kv_dim))))
                }
            })
            .collect();
        let dn_state = DeltaNetStateCache::allocate(
            layer_kinds,
            d_conv,
            dn_conv_dim,
            dn_head_v_dim,
            dn_n_v_heads,
        );
        Self {
            kv_layers,
            dn_state,
            layer_kinds: layer_kinds.to_vec(),
            next_position: 0,
        }
    }

    pub fn num_layers(&self) -> usize {
        self.layer_kinds.len()
    }

    /// Reset all state to the empty initial condition. Use between
    /// sequences within the same process.
    pub fn reset(&mut self) {
        for slot in self.kv_layers.iter_mut().flatten() {
            slot.0 = Array2::zeros((0, slot.0.shape()[1]));
            slot.1 = Array2::zeros((0, slot.1.shape()[1]));
        }
        self.dn_state.reset();
        self.next_position = 0;
    }
}

/// Weight references for one Qwen 3.6 layer, tagged with the layer
/// kind. The router consumes this to know which block path to run.
pub enum Qwen35LayerWeights<'a> {
    /// Gated DeltaNet linear-attention layer (Qwen 3.6: 48 of 64).
    Linear(DeltaNetLayerWeights<'a>),
    /// Full softmax-attention layer (Qwen 3.6: 16 of 64).
    Attention(Qwen35AttentionLayerWeights<'a>),
}

/// Per-layer router: dispatches the layer kind to the right block
/// forward. Mutates the appropriate slot in `hybrid_cache`.
///
/// Returns the BLOCK output (NOT yet added to the residual; caller
/// adds it, applies the optional `attn_post_norm`, and runs FFN).
pub fn hybrid_layer_step(
    layer: usize,
    x: &Array1<f32>,
    weights: &Qwen35LayerWeights,
    dn_dims: &DeltaNetDims,
    attn_dims: &Qwen35AttentionDims,
    hybrid_cache: &mut DeltaNetHybridCache,
) -> Array1<f32> {
    debug_assert!(layer < hybrid_cache.num_layers());
    let is_linear = hybrid_cache.layer_kinds[layer];
    match (weights, is_linear) {
        (Qwen35LayerWeights::Linear(w), true) => {
            let state = hybrid_cache.dn_state.layers[layer]
                .as_mut()
                .expect("linear-layer state should be allocated");
            deltanet_block_step(x, w, dn_dims, state)
        }
        (Qwen35LayerWeights::Attention(w), false) => {
            let kv_layer = hybrid_cache.kv_layers[layer]
                .as_mut()
                .expect("full-attn KV slabs should be allocated");
            qwen35_attention_block_step(x, w, attn_dims, kv_layer, hybrid_cache.next_position)
        }
        _ => panic!(
            "layer {} kind mismatch: weights and layer_kinds disagree",
            layer
        ),
    }
}

#[inline]
fn _silu(x: f32) -> f32 {
    // Kept for parity with deltanet_block::silu; not currently
    // referenced (full-attn block uses the sigmoid'd gate from
    // split_q_gate directly). Reserved for FFN integration in
    // Phase C.4c.
    x * sigmoid(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::deltanet_state::{DeltaNetLayerState, DeltaNetStateCache};
    use ndarray::Array2;

    fn tiny_attn_dims() -> Qwen35AttentionDims {
        Qwen35AttentionDims {
            hidden: 4,
            n_head: 2,
            n_head_kv: 1, // GQA reps = 2
            head_dim: 2,
            rotary_dim: 2, // full rotation on the tiny head
            rope_base: 10_000.0,
            eps: 1e-6,
        }
    }

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

    #[test]
    fn attention_dims_helpers() {
        let d = tiny_attn_dims();
        assert_eq!(d.q_dim(), 4);
        assert_eq!(d.kv_dim(), 2);
        assert_eq!(d.fused_q_dim(), 8);
        assert_eq!(d.gqa_reps(), 2);
        assert!((d.rope_fraction() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn attention_dims_qwen35_27b_values() {
        let d = Qwen35AttentionDims {
            hidden: 5120,
            n_head: 24,
            n_head_kv: 4,
            head_dim: 256,
            rotary_dim: 64,
            rope_base: 10_000_000.0,
            eps: 1e-6,
        };
        assert_eq!(d.q_dim(), 24 * 256); // 6144
        assert_eq!(d.kv_dim(), 4 * 256); // 1024
        assert_eq!(d.fused_q_dim(), 2 * 6144); // 12288
        assert_eq!(d.gqa_reps(), 6); // 24 / 4
        assert!((d.rope_fraction() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn append_row_grows_slab() {
        let mut slab = Array2::<f32>::zeros((2, 3));
        slab[[0, 0]] = 1.0;
        slab[[1, 1]] = 2.0;
        let new_row = ndarray::arr1(&[10.0_f32, 20.0, 30.0]);
        super::append_row(&mut slab, new_row.view());
        assert_eq!(slab.shape(), &[3, 3]);
        // Old rows preserved.
        assert_eq!(slab[[0, 0]], 1.0);
        assert_eq!(slab[[1, 1]], 2.0);
        // New row at the end.
        assert_eq!(slab.row(2).to_vec(), vec![10.0, 20.0, 30.0]);
    }

    #[test]
    fn append_row_from_empty_slab() {
        let mut slab = Array2::<f32>::zeros((0, 3));
        let new_row = ndarray::arr1(&[1.0_f32, 2.0, 3.0]);
        super::append_row(&mut slab, new_row.view());
        assert_eq!(slab.shape(), &[1, 3]);
        assert_eq!(slab.row(0).to_vec(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn hybrid_cache_allocates_correct_per_layer_kind() {
        // Mimics Qwen 3.6 27B's 4-layer slice: linear, linear, linear, attn.
        let mask = vec![true, true, true, false];
        let cache = DeltaNetHybridCache::allocate(&mask, 4 /*kv_dim*/, 2, 4, 2, 1);
        assert_eq!(cache.num_layers(), 4);
        // Linear layers: KV slot = None, DN state slot = Some.
        for &i in &[0, 1, 2] {
            assert!(cache.kv_layers[i].is_none());
            assert!(cache.dn_state.layers[i].is_some());
        }
        // Full-attn layer: KV slot = Some (empty slab), DN state = None.
        assert!(cache.kv_layers[3].is_some());
        let (k0, v0) = cache.kv_layers[3].as_ref().unwrap();
        assert_eq!(k0.shape(), &[0, 4]);
        assert_eq!(v0.shape(), &[0, 4]);
        assert!(cache.dn_state.layers[3].is_none());
        assert_eq!(cache.next_position, 0);
    }

    #[test]
    fn hybrid_cache_reset_clears_state_and_position() {
        let mask = vec![true, false];
        let mut cache = DeltaNetHybridCache::allocate(&mask, 4, 2, 4, 2, 1);
        // Dirty the linear-layer state.
        if let Some(state) = cache.dn_state.layers[0].as_mut() {
            state.conv_state.fill(7.0);
            state.recurrent_state.fill(11.0);
        }
        // Grow the attn-layer KV slabs.
        if let Some((k, v)) = cache.kv_layers[1].as_mut() {
            let nr = ndarray::arr1(&[1.0_f32, 2.0, 3.0, 4.0]);
            super::append_row(k, nr.view());
            super::append_row(v, nr.view());
        }
        cache.next_position = 5;
        cache.reset();
        // Reset semantics: DN state zero, KV slabs back to [0, kv_dim], pos = 0.
        assert!(cache.dn_state.layers[0]
            .as_ref()
            .unwrap()
            .conv_state
            .iter()
            .all(|&v| v == 0.0));
        let (k, v) = cache.kv_layers[1].as_ref().unwrap();
        assert_eq!(k.shape(), &[0, 4]);
        assert_eq!(v.shape(), &[0, 4]);
        assert_eq!(cache.next_position, 0);
    }

    #[test]
    fn hybrid_layer_step_routes_to_deltanet_on_linear() {
        // Build tiny weights for both branches; layer 0 is linear,
        // so the router SHALL call deltanet_block_step.
        let dn_dims = tiny_dn_dims();
        let attn_dims = tiny_attn_dims();
        let mask = vec![true, false];
        let mut cache = DeltaNetHybridCache::allocate(
            &mask,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        // DeltaNet weights (constant-filled for shape correctness).
        let attn_norm = vec![1.0_f32; dn_dims.hidden];
        let attn_qkv = Array2::from_elem((dn_dims.conv_dim(), dn_dims.hidden), 0.1_f32);
        let attn_gate = Array2::from_elem((dn_dims.value_dim(), dn_dims.hidden), 0.1_f32);
        let ssm_conv1d = Array2::from_elem((dn_dims.d_conv, dn_dims.conv_dim()), 0.5_f32);
        let ssm_dt = vec![0.0_f32; dn_dims.n_v_heads];
        let ssm_a = vec![-1.0_f32; dn_dims.n_v_heads];
        let ssm_beta = Array2::from_elem((dn_dims.n_v_heads, dn_dims.hidden), 0.1_f32);
        let ssm_alpha = Array2::from_elem((dn_dims.n_v_heads, dn_dims.hidden), 0.1_f32);
        let ssm_norm = vec![1.0_f32; dn_dims.head_v_dim];
        let ssm_out = Array2::from_elem((dn_dims.hidden, dn_dims.value_dim()), 0.5_f32);

        let dn_weights = DeltaNetLayerWeights {
            attn_norm: &attn_norm,
            attn_qkv: &attn_qkv,
            attn_gate: &attn_gate,
            ssm_conv1d: &ssm_conv1d,
            ssm_dt: &ssm_dt,
            ssm_a: &ssm_a,
            ssm_beta: &ssm_beta,
            ssm_alpha: &ssm_alpha,
            ssm_norm: &ssm_norm,
            ssm_out: &ssm_out,
        };
        let layer_weights = Qwen35LayerWeights::Linear(dn_weights);

        let x = Array1::from_elem(dn_dims.hidden, 1.0_f32);
        let y = hybrid_layer_step(0, &x, &layer_weights, &dn_dims, &attn_dims, &mut cache);
        assert_eq!(y.len(), dn_dims.hidden);
        // Linear-layer state should be non-zero after one step.
        let state = cache.dn_state.layers[0].as_ref().unwrap();
        assert!(state.conv_state.iter().any(|&v| v.abs() > 0.0));
        // Full-attn KV slab still empty.
        let (k1, _v1) = cache.kv_layers[1].as_ref().unwrap();
        assert_eq!(k1.shape()[0], 0);
    }

    #[test]
    #[should_panic(expected = "kind mismatch")]
    fn hybrid_layer_step_panics_on_kind_mismatch() {
        // layer 0 mask says linear, but we pass Attention weights.
        let dn_dims = tiny_dn_dims();
        let attn_dims = tiny_attn_dims();
        let mask = vec![true]; // 1 layer, linear
        let mut cache = DeltaNetHybridCache::allocate(
            &mask,
            attn_dims.kv_dim(),
            dn_dims.d_conv,
            dn_dims.conv_dim(),
            dn_dims.head_v_dim,
            dn_dims.n_v_heads,
        );

        let attn_norm = vec![1.0_f32; attn_dims.hidden];
        let attn_q = Array2::zeros((attn_dims.fused_q_dim(), attn_dims.hidden));
        let attn_k = Array2::zeros((attn_dims.kv_dim(), attn_dims.hidden));
        let attn_v = Array2::zeros((attn_dims.kv_dim(), attn_dims.hidden));
        let attn_q_norm = vec![1.0_f32; attn_dims.head_dim];
        let attn_k_norm = vec![1.0_f32; attn_dims.head_dim];
        let attn_output = Array2::zeros((attn_dims.hidden, attn_dims.q_dim()));

        let attn_weights = Qwen35AttentionLayerWeights {
            attn_norm: &attn_norm,
            attn_q: &attn_q,
            attn_k: &attn_k,
            attn_v: &attn_v,
            attn_q_norm: &attn_q_norm,
            attn_k_norm: &attn_k_norm,
            attn_output: &attn_output,
        };
        let layer_weights = Qwen35LayerWeights::Attention(attn_weights);

        let x = Array1::from_elem(attn_dims.hidden, 1.0_f32);
        let _ = hybrid_layer_step(0, &x, &layer_weights, &dn_dims, &attn_dims, &mut cache);
    }

    /// Avoids the dead-code-warning silence on the reserved helper.
    #[test]
    fn silu_helper_is_reachable() {
        let _ = super::_silu(0.0);
        let _ = DeltaNetLayerState::allocate(2, 4, 2, 1);
    }
}
