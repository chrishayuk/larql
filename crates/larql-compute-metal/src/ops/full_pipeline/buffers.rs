//! Per-layer scratch buffer allocation for the full-pipeline dispatch.
//!
//! Pulled out of `dispatch_full_pipeline` so the orchestration body
//! reads as "for each layer, run the 11 stages" without 100 LOC of
//! buffer-sizing arithmetic in the way. Sizes mirror what the inner
//! loop needs at every position (per-layer Q/KV dims for Gemma 4's
//! sliding/global mix, hidden for everything else).

use metal::Buffer;

use crate::buffers::BufferCache;
use larql_models::quant::ggml::LEGACY_BLOCK_ELEMS;

/// Per-position byte-stride for the shared Q8 staging buffers.
///
/// `q8_bufs` and `q8s_bufs` are shared between two writers:
/// - the **Q8 attention-input path** writes `hidden` floats per position
///   (Q8 hidden bytes + per-block scales)
/// - the **O-projection input path** writes `layer_q_dim` floats per
///   position (Gemma 4 layers vary head_dim 256/512 between sliding /
///   global attention, so the per-layer q_dim isn't constant)
///
/// Both writers use offsets into the same backing buffer, so the row
/// stride must accommodate the larger of the two. Returns
/// `(q8_row_max, q8s_row_bytes)`:
/// - `q8_row_max` = max(`hidden`, max(layers[*].num_q_heads * layers[*].head_dim))
/// - `q8s_row_bytes` = `q8_row_max.div_ceil(32) * 4` — Q8 stores one f32
///   scale per 32-element block, padded to a whole block.
///
/// Pure arithmetic on `(num_q_heads, head_dim)` — exposed as a
/// standalone helper so it's unit-testable without a Metal backend.
pub(crate) fn q8_staging_size(
    layers: &[larql_compute::FullPipelineLayer<'_>],
    hidden: usize,
    q_dim_fallback: usize,
) -> (usize, usize) {
    let max_layer_q_dim = layers
        .iter()
        .map(|l| l.num_q_heads * l.head_dim)
        .max()
        .unwrap_or(q_dim_fallback);
    let q8_row_max = hidden.max(max_layer_q_dim);
    let q8s_row_bytes = q8_row_max.div_ceil(LEGACY_BLOCK_ELEMS) * 4;
    (q8_row_max, q8s_row_bytes)
}

/// Pre-allocated per-layer scratch + per-layer Q4 weight handles.
///
/// All vectors are `len() == num_layers` (or `+1` for `h_bufs` to
/// hold the input embedding plus each layer's output).
pub(super) struct LayerBuffers {
    // ── Q4 weight buffers (cached, mmap-backed) ──
    pub wq: Vec<Buffer>,
    pub wq_scale: Vec<Buffer>,
    pub wk: Vec<Buffer>,
    pub wk_scale: Vec<Buffer>,
    pub wv: Vec<Buffer>,
    pub wv_scale: Vec<Buffer>,
    pub wo: Vec<Buffer>,
    pub gate: Vec<Buffer>,
    pub up: Vec<Buffer>,
    pub down: Vec<Buffer>,
    // ── Norm weight buffers ──
    pub input_norm: Vec<Buffer>,
    pub post_attn_norm: Vec<Buffer>,
    pub pre_ffn_norm: Vec<Option<Buffer>>,
    pub post_ffn_norm: Vec<Option<Buffer>>,
    // ── Per-layer per-position scratch outputs ──
    pub h: Vec<Buffer>, // num_layers + 1: input + each layer's output
    pub norm_out: Vec<Buffer>,
    pub q_out: Vec<Buffer>,
    pub k_out: Vec<Buffer>,
    pub v_out: Vec<Buffer>,
    pub attn_out: Vec<Buffer>,
    pub o_out: Vec<Buffer>,
    pub h_post_attn: Vec<Buffer>,
    pub ffn_norm_out: Vec<Buffer>,
    pub gate_out: Vec<Buffer>,
    pub up_out: Vec<Buffer>,
    pub act_buf: Vec<Buffer>,
    pub down_out: Vec<Buffer>,
    pub q8: Vec<Buffer>,
    pub q8s: Vec<Buffer>,
    pub ffn_q8: Vec<Buffer>,
    pub ffn_q8s: Vec<Buffer>,
    // ── Geometry constants used to compute byte offsets in the inner loop ──
    pub q8_row_max: usize,
    pub q8s_row_bytes: usize,
}

impl LayerBuffers {
    /// Pre-cache weights + allocate scratch for every layer × every
    /// position. Sized for Gemma 4's mixed sliding/global geometry —
    /// each layer's intermediate buffer is sized from that layer's own
    /// `num_q_heads * head_dim`, not the function-level `q_dim`.
    pub fn allocate(
        bufs: &BufferCache,
        layers: &[larql_compute::FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        seq_len: usize,
        q_dim_fallback: usize,
    ) -> Self {
        let num_layers = layers.len();

        // Pre-cache attention weight buffers (stable across calls →
        // cache by slice identity skips per-token Metal-buffer alloc).
        let wq: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.wq.data)).collect();
        let wq_scale: Vec<_> = layers
            .iter()
            .map(|l| bufs.get_f32(l.wq.scales.unwrap_or(&[])))
            .collect();
        let wk: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.wk.data)).collect();
        let wk_scale: Vec<_> = layers
            .iter()
            .map(|l| bufs.get_f32(l.wk.scales.unwrap_or(&[])))
            .collect();
        let wv: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.wv.data)).collect();
        let wv_scale: Vec<_> = layers
            .iter()
            .map(|l| bufs.get_f32(l.wv.scales.unwrap_or(&[])))
            .collect();
        let wo: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.wo.data)).collect();
        let gate: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.gate.data)).collect();
        let up: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.up.data)).collect();
        let down: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.down.data)).collect();

        // Norm weight buffers — also stable.
        let input_norm: Vec<_> = layers.iter().map(|l| bufs.get_f32(l.input_norm)).collect();
        let post_attn_norm: Vec<_> = layers
            .iter()
            .map(|l| bufs.get_f32(l.post_attn_norm))
            .collect();
        let pre_ffn_norm: Vec<Option<_>> = layers
            .iter()
            .map(|l| l.pre_ffn_norm.map(|n| bufs.get_f32(n)))
            .collect();
        let post_ffn_norm: Vec<Option<_>> = layers
            .iter()
            .map(|l| l.post_ffn_norm.map(|n| bufs.get_f32(n)))
            .collect();

        // Q8 staging buffers shared between Q8 attention input and the
        // O-projection input — sized at `max(hidden, max_layer_q_dim)`
        // per position so both writers fit with offsets.
        let (q8_row_max, q8s_row_bytes) = q8_staging_size(layers, hidden, q_dim_fallback);

        let mut h = Vec::with_capacity(num_layers + 1);
        h.push(bufs.transient_from_f32(x));

        let mut norm_out = Vec::with_capacity(num_layers);
        let mut q_out = Vec::with_capacity(num_layers);
        let mut k_out = Vec::with_capacity(num_layers);
        let mut v_out = Vec::with_capacity(num_layers);
        let mut attn_out = Vec::with_capacity(num_layers);
        let mut o_out = Vec::with_capacity(num_layers);
        let mut h_post_attn = Vec::with_capacity(num_layers);
        let mut ffn_norm_out = Vec::with_capacity(num_layers);
        let mut gate_out = Vec::with_capacity(num_layers);
        let mut up_out = Vec::with_capacity(num_layers);
        let mut act_buf = Vec::with_capacity(num_layers);
        let mut down_out = Vec::with_capacity(num_layers);
        let mut q8 = Vec::with_capacity(num_layers);
        let mut q8s = Vec::with_capacity(num_layers);
        let mut ffn_q8 = Vec::with_capacity(num_layers);
        let mut ffn_q8s = Vec::with_capacity(num_layers);
        for layer in layers.iter() {
            let lq = layer.num_q_heads * layer.head_dim;
            let lkv = layer.num_kv_heads * layer.head_dim;
            norm_out.push(bufs.output((seq_len * hidden * 4) as u64));
            q_out.push(bufs.output((seq_len * lq * 4) as u64));
            k_out.push(bufs.output((seq_len * lkv * 4) as u64));
            v_out.push(bufs.output((seq_len * lkv * 4) as u64));
            attn_out.push(bufs.output((seq_len * lq * 4) as u64));
            o_out.push(bufs.output((seq_len * hidden * 4) as u64));
            h_post_attn.push(bufs.output((seq_len * hidden * 4) as u64));
            ffn_norm_out.push(bufs.output((seq_len * hidden * 4) as u64));
            gate_out.push(bufs.output((seq_len * inter * 4) as u64));
            up_out.push(bufs.output((seq_len * inter * 4) as u64));
            act_buf.push(bufs.output((seq_len * inter * 4) as u64));
            down_out.push(bufs.output((seq_len * hidden * 4) as u64));
            h.push(bufs.output((seq_len * hidden * 4) as u64));
            q8.push(bufs.output((seq_len * q8_row_max) as u64));
            q8s.push(bufs.output((seq_len * q8s_row_bytes) as u64));
            ffn_q8.push(bufs.output((seq_len * hidden) as u64));
            ffn_q8s.push(bufs.output((seq_len * hidden.div_ceil(LEGACY_BLOCK_ELEMS) * 4) as u64));
        }

        Self {
            wq,
            wq_scale,
            wk,
            wk_scale,
            wv,
            wv_scale,
            wo,
            gate,
            up,
            down,
            input_norm,
            post_attn_norm,
            pre_ffn_norm,
            post_ffn_norm,
            h,
            norm_out,
            q_out,
            k_out,
            v_out,
            attn_out,
            o_out,
            h_post_attn,
            ffn_norm_out,
            gate_out,
            up_out,
            act_buf,
            down_out,
            q8,
            q8s,
            ffn_q8,
            ffn_q8s,
            q8_row_max,
            q8s_row_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_compute::pipeline::*;
    use larql_models::quant::ggml::LEGACY_BLOCK_ELEMS;

    const HIDDEN_SMALL: usize = 1024;
    const HIDDEN_GEMMA3_4B: usize = 2560;
    const Q_DIM_SMALLER_THAN_HIDDEN: usize = 2048;
    const Q_DIM_LARGER_THAN_HIDDEN: usize = 4096;

    /// Minimal `FullPipelineLayer` for testing geometry math. All
    /// weight / norm slices borrow from the leaked statics so a test
    /// can stash multiple layers in one Vec without lifetime
    /// gymnastics. Q4 weights are sized for `K=32` * 18-byte blocks.
    fn synth_layer(
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> FullPipelineLayer<'static> {
        let q4 = Box::leak(vec![0u8; 32 * 18].into_boxed_slice());
        let norm = Box::leak(vec![1.0f32; 32].into_boxed_slice());
        let q4w = || QuantWeight {
            data: q4,
            scales: None,
            format: QuantFormat::Q4_K,
        };
        FullPipelineLayer {
            wq: q4w(),
            wk: q4w(),
            wv: q4w(),
            wo: q4w(),
            gate: q4w(),
            up: q4w(),
            down: q4w(),
            input_norm: norm,
            post_attn_norm: norm,
            pre_ffn_norm: None,
            post_ffn_norm: None,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            norm_offset: 1.0,
            qk_norm_offset: 1.0,
            eps: RMSNORM_EPSILON_DEFAULT,
            has_post_norms: false,
            norm_type: NormType::RmsNorm,
            ffn_type: FfnType::Gated,
            activation: Activation::Silu,
            attn_scale: 0.125,
            head_dim,
            num_q_heads,
            num_kv_heads,
            rope_base: ROPE_BASE_DEFAULT,
            rotary_dim: 0,
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar: 0.0,
            q_norm_weight: None,
            k_norm_weight: None,
            ffn_up_bias: None,
            ffn_down_bias: None,
            moe: None,
            ffn_is_remote: false,
            moe_combined_output_norm: false,
            moe_outer_post_norm: None,
            ple_input_gate: None,
            ple_projection: None,
            ple_post_norm: None,
            kv_shared_source: None,
            residual_multiplier: 1.0,
        }
    }

    /// Build a fresh Vec of N synth layers (FullPipelineLayer doesn't
    /// implement Clone, so the `vec![…; n]` form doesn't apply).
    fn synth_layers(
        n: usize,
        num_q: usize,
        num_kv: usize,
        hd: usize,
    ) -> Vec<FullPipelineLayer<'static>> {
        (0..n).map(|_| synth_layer(num_q, num_kv, hd)).collect()
    }

    /// Uniform-geometry case (Llama / Mistral / Gemma 3): every layer
    /// has the same num_q_heads and head_dim, so the Q8 staging row
    /// width is just `max(hidden, q_dim)`.
    #[test]
    fn q8_staging_uniform_geometry_picks_max_of_hidden_and_qdim() {
        // Gemma 3 4B: hidden=2560, q_dim = 8*256 = 2048 (q < hidden).
        let layers = synth_layers(4, 8, 4, 256);
        let (q8_row_max, q8s_row_bytes) =
            q8_staging_size(&layers, HIDDEN_GEMMA3_4B, Q_DIM_SMALLER_THAN_HIDDEN);
        assert_eq!(q8_row_max, HIDDEN_GEMMA3_4B); // hidden wins
        assert_eq!(q8s_row_bytes, HIDDEN_GEMMA3_4B / LEGACY_BLOCK_ELEMS * 4); // 80 blocks × 4 bytes = 320

        // Larger Q than hidden: q_dim wins.
        let layers = synth_layers(4, 16, 4, 256); // q_dim = 16*256 = 4096
        let (q8_row_max, q8s_row_bytes) =
            q8_staging_size(&layers, HIDDEN_GEMMA3_4B, Q_DIM_LARGER_THAN_HIDDEN);
        assert_eq!(q8_row_max, Q_DIM_LARGER_THAN_HIDDEN);
        assert_eq!(
            q8s_row_bytes,
            Q_DIM_LARGER_THAN_HIDDEN / LEGACY_BLOCK_ELEMS * 4
        ); // 512
    }

    /// Mixed sliding/global geometry (Gemma 4 31B): different layers
    /// have different head_dims (256 sliding / 512 global). The Q8
    /// staging buffer must size to the *largest* layer_q_dim across
    /// the model, not the first or fallback.
    #[test]
    fn q8_staging_mixed_geometry_picks_largest_layer_q_dim() {
        let layers = vec![
            // Sliding layer: head_dim=256, num_q_heads=14 → q_dim=3584
            synth_layer(14, 2, 256),
            // Global layer: head_dim=512, num_q_heads=14 → q_dim=7168
            synth_layer(14, 1, 512),
            // Another sliding layer.
            synth_layer(14, 2, 256),
        ];

        // Pass q_dim_fallback=3584 (the sliding layer's value) — the
        // helper must still pick the global layer's 7168.
        let (q8_row_max, _q8s_row_bytes) = q8_staging_size(&layers, 5376, 3584);
        assert_eq!(
            q8_row_max, 7168,
            "mixed geometry: must size to largest layer"
        );
    }

    /// Empty layer list: helper falls back to `q_dim_fallback`.
    /// Used as a defensive guard when the caller has no layers loaded.
    #[test]
    fn q8_staging_empty_layers_uses_fallback() {
        let layers: Vec<FullPipelineLayer<'static>> = vec![];
        let (q8_row_max, _) = q8_staging_size(&layers, HIDDEN_GEMMA3_4B, Q_DIM_SMALLER_THAN_HIDDEN);
        // hidden=2560 > fallback=2048, so hidden wins.
        assert_eq!(q8_row_max, HIDDEN_GEMMA3_4B);

        let (q8_row_max, _) = q8_staging_size(&layers, HIDDEN_SMALL, Q_DIM_LARGER_THAN_HIDDEN);
        assert_eq!(
            q8_row_max, Q_DIM_LARGER_THAN_HIDDEN,
            "fallback wins when fallback > hidden"
        );
    }

    /// `q8s_row_bytes` is always a multiple of 4 (one f32 per 32-elt
    /// block), and rounds *up* for non-multiple-of-32 row widths.
    #[test]
    fn q8s_row_bytes_rounds_up_to_full_block() {
        // q8_row_max = 32 → 1 block × 4 bytes = 4
        let layers = vec![synth_layer(1, 1, 32)];
        let (_, q8s) = q8_staging_size(&layers, 32, 32);
        assert_eq!(q8s, 4);

        // q8_row_max = 33 → 2 blocks × 4 = 8 (round up)
        let layers = vec![synth_layer(1, 1, 33)];
        let (_, q8s) = q8_staging_size(&layers, 33, 33);
        assert_eq!(q8s, 8);
    }
}
