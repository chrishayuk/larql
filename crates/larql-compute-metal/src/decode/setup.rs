//! Per-decode-token scratch and weight-buffer pre-allocation.
//!
//! [`DecodeScratch`] is built once at the top of
//! `decode_token_with_moe_split_fn` and threaded through the per-layer
//! loop. It owns:
//!
//! - Per-layer weight-buffer caches (`wq_bufs[l]`, `gate_bufs[l]`, …) so
//!   the second-and-onward decode token skips the per-slice
//!   `BufferCache::get_bytes` / `get_f32` rehydration cost.
//! - Per-stage scratch buffers (`q_out`, `ffn_norm_out`, …) reused across
//!   all layers within a single command-buffer encoder.
//! - Two ping-pong residual buffers (`h_a`, `h_b`) plus the layer-0
//!   embedding (`h_init`).
//! - Constants derived from `layers` (`max_q_dim`, `inter_padded`, `has_moe`).
//!
//! No behaviour change vs. the prior inline setup — pure code motion to
//! cut `decode/mod.rs` from one 1200-line method into a per-stage chain
//! that the profiler can reason about.
//!
//! Sized scratches:
//! - `q_out` / `attn_out_buf` use `max_q_dim` (per-layer max across the
//!   whole stack — Gemma 4 has heterogeneous q_dim per layer).
//! - `act_buf` is `inter_padded * 4` and **zero-initialised** so down_proj
//!   reads zero past `inter` (Q4_K/Q6_K super-blocks need 256-aligned rows).

use crate::buffers::BufferCache;
use larql_compute::FullPipelineLayer;
use larql_models::quant::ggml::LEGACY_BLOCK_ELEMS;
use larql_models::quant::ggml::Q4_K_BLOCK_ELEMS;
use metal::Buffer;

pub(super) struct DecodeScratch {
    // ── Per-layer weight buffer caches (length = num_layers) ──
    pub wq_bufs: Vec<Buffer>,
    pub wk_bufs: Vec<Buffer>,
    pub wv_bufs: Vec<Buffer>,
    pub wo_bufs: Vec<Buffer>,
    pub wq_scale_bufs: Vec<Buffer>,
    pub wk_scale_bufs: Vec<Buffer>,
    pub wv_scale_bufs: Vec<Buffer>,
    pub wo_scale_bufs: Vec<Buffer>,
    pub gate_bufs: Vec<Buffer>,
    pub up_bufs: Vec<Buffer>,
    pub down_bufs: Vec<Buffer>,
    pub input_norm_bufs: Vec<Buffer>,
    pub post_attn_norm_bufs: Vec<Buffer>,

    // ── Hidden-state ping-pong + layer-0 input ──
    pub h_init: Buffer,
    pub h_a: Buffer,
    pub h_b: Buffer,

    // ── Per-stage scratch (one buffer, reused every layer) ──
    pub q_out: Buffer,
    pub k_out: Buffer,
    pub v_out: Buffer,
    pub norm_f32_buf: Buffer,
    pub attn_out_buf: Buffer,
    pub o_out_buf: Buffer,
    pub h_post_attn: Buffer,
    pub ffn_norm_out: Buffer,
    pub ffn_q8: Buffer,
    pub ffn_q8s: Buffer,
    pub up_out: Buffer,
    /// Sized to `inter_padded` and zero-initialised so down_proj's matvec
    /// reads zero for any trailing padding columns. Only the first
    /// `inter` floats are written by GEGLU; the rest stay zero across
    /// all layers because nothing writes past `inter`.
    pub act_buf: Buffer,
    pub down_out: Buffer,
    pub gate_out_scratch: Buffer,
    pub normed_scratch: Buffer,
    pub o_q8_scratch: Buffer,
    pub o_q8s_scratch: Buffer,
    /// Currently dead but kept allocated so its lifetime matches the
    /// other scratches; removing it is a separate cleanup.
    pub scaled_scratch: Buffer,

    // ── Constants derived from `layers` ──
    pub inter_padded: usize,
    pub num_layers: usize,
    pub has_moe: bool,

    /// Clones of every buffer returned by `BufferCache::output` during
    /// construction.  Handed to a `ScratchGuard` in the decode function so
    /// all scratch buffers are returned to the pool after the decode step.
    pub scratch_clones: Vec<metal::Buffer>,
}

impl DecodeScratch {
    pub(super) fn new(
        bufs: &BufferCache,
        layers: &[FullPipelineLayer<'_>],
        x: &[f32],
        hidden: usize,
        inter: usize,
        q_dim: usize,
        kv_dim: usize,
    ) -> Self {
        let num_layers = layers.len();
        let inter_padded = inter.div_ceil(Q4_K_BLOCK_ELEMS) * Q4_K_BLOCK_ELEMS;

        // Scratch buffers are reused across all layers within the encoder.
        // When attention geometry varies layer to layer (Gemma 4 sliding=8192
        // vs global=16384 q_dim) we must size each scratch to the MAX across
        // layers; the outer scalar `q_dim` / `kv_dim` only reflect the first
        // layer's shape. Taking the per-layer max means a global layer's
        // 16384-wide Q output won't overflow a buffer sized for 8192.
        let max_q_dim = layers
            .iter()
            .map(|l| l.num_q_heads * l.head_dim)
            .max()
            .unwrap_or(q_dim);
        let max_kv_dim = layers
            .iter()
            .map(|l| l.num_kv_heads * l.head_dim)
            .max()
            .unwrap_or(kv_dim);

        let wq_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.wq.data)).collect();
        let wk_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.wk.data)).collect();
        let wv_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.wv.data)).collect();
        let wo_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.wo.data)).collect();
        // Stable across decode calls → cache by slice identity. Skips ~136
        // per-token Metal-buffer allocations for scales/norms on 34-layer
        // Gemma 3. `get_f32` hits the cache from the second decode onward.
        let wq_scale_bufs: Vec<_> = layers
            .iter()
            .map(|l| bufs.get_f32(l.wq.scales.unwrap_or(&[])))
            .collect();
        let wk_scale_bufs: Vec<_> = layers
            .iter()
            .map(|l| bufs.get_f32(l.wk.scales.unwrap_or(&[])))
            .collect();
        let wv_scale_bufs: Vec<_> = layers
            .iter()
            .map(|l| bufs.get_f32(l.wv.scales.unwrap_or(&[])))
            .collect();
        let wo_scale_bufs: Vec<_> = layers
            .iter()
            .map(|l| bufs.get_f32(l.wo.scales.unwrap_or(&[])))
            .collect();
        let gate_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.gate.data)).collect();
        let up_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.up.data)).collect();
        let down_bufs: Vec<_> = layers.iter().map(|l| bufs.get_bytes(l.down.data)).collect();
        let input_norm_bufs: Vec<_> = layers.iter().map(|l| bufs.get_f32(l.input_norm)).collect();
        let post_attn_norm_bufs: Vec<_> = layers
            .iter()
            .map(|l| bufs.get_f32(l.post_attn_norm))
            .collect();

        // Two h buffers for ping-pong: even layers write to h_a, odd to h_b.
        let h_init = bufs.transient_from_f32(x);
        let h_a = bufs.output((hidden * 4) as u64);
        let h_b = bufs.output((hidden * 4) as u64);

        // Pre-allocate scratch buffers reused across layers.
        // GPU processes layers sequentially within one cmd buffer, so
        // these buffers are never read and written simultaneously.
        let q_out = bufs.output((max_q_dim * 4) as u64);
        let k_out = bufs.output((max_kv_dim * 4) as u64);
        let v_out = bufs.output((max_kv_dim * 4) as u64);
        let norm_f32_buf = bufs.output((hidden * 4) as u64);
        let attn_out_buf = bufs.output((max_q_dim * 4) as u64);
        let o_out_buf = bufs.output((hidden * 4) as u64);
        let h_post_attn = bufs.output((hidden * 4) as u64);
        let ffn_norm_out = bufs.output((hidden * 4) as u64);
        let ffn_q8 = bufs.output(hidden as u64);
        let ffn_q8s = bufs.output((hidden / LEGACY_BLOCK_ELEMS * 4) as u64);
        let up_out = bufs.output((inter * 4) as u64);
        let act_buf = bufs.output((inter_padded * 4) as u64);
        {
            let ptr = act_buf.contents() as *mut f32;
            // SAFETY: `act_buf` is a freshly-allocated shared-storage
            // Metal buffer with `inter_padded * 4` bytes. We zero its
            // entire f32 capacity before any layer writes the live
            // `inter` columns; the trailing `inter_padded - inter`
            // columns stay zero for the remainder of the decode.
            unsafe { std::ptr::write_bytes(ptr, 0, inter_padded) };
        }
        let down_out = bufs.output((hidden * 4) as u64);
        let gate_out_scratch = bufs.output((inter * 4) as u64);
        let normed_scratch = bufs.output((hidden * 4) as u64);
        let o_q8_scratch = bufs.output(max_q_dim as u64);
        let o_q8s_scratch = bufs.output((max_q_dim / LEGACY_BLOCK_ELEMS * 4) as u64);
        let scaled_scratch = bufs.output((hidden * 4) as u64);

        let has_moe = layers.iter().any(|l| l.moe.is_some() || l.ffn_is_remote);

        // Collect clones of every output buffer so the decode function can
        // return them to the scratch pool after the GPU step completes.
        let scratch_clones = vec![
            h_a.clone(),
            h_b.clone(),
            q_out.clone(),
            k_out.clone(),
            v_out.clone(),
            norm_f32_buf.clone(),
            attn_out_buf.clone(),
            o_out_buf.clone(),
            h_post_attn.clone(),
            ffn_norm_out.clone(),
            ffn_q8.clone(),
            ffn_q8s.clone(),
            up_out.clone(),
            act_buf.clone(),
            down_out.clone(),
            gate_out_scratch.clone(),
            normed_scratch.clone(),
            o_q8_scratch.clone(),
            o_q8s_scratch.clone(),
            scaled_scratch.clone(),
        ];

        Self {
            wq_bufs,
            wk_bufs,
            wv_bufs,
            wo_bufs,
            wq_scale_bufs,
            wk_scale_bufs,
            wv_scale_bufs,
            wo_scale_bufs,
            gate_bufs,
            up_bufs,
            down_bufs,
            input_norm_bufs,
            post_attn_norm_bufs,
            h_init,
            h_a,
            h_b,
            q_out,
            k_out,
            v_out,
            norm_f32_buf,
            attn_out_buf,
            o_out_buf,
            h_post_attn,
            ffn_norm_out,
            ffn_q8,
            ffn_q8s,
            up_out,
            act_buf,
            down_out,
            gate_out_scratch,
            normed_scratch,
            o_q8_scratch,
            o_q8s_scratch,
            scaled_scratch,
            inter_padded,
            num_layers,
            has_moe,
            scratch_clones,
        }
    }
}
