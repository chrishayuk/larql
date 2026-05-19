//! Per-Layer Embeddings (PLE) — per-layer Metal dispatch.
//!
//! Gemma 4 E2B adds a gated per-layer embedding contribution to the
//! residual stream at the end of every decoder layer. The math
//! (mirroring [`crates/larql-inference/src/forward/ple.rs::apply_per_layer_embedding`]):
//!
//! ```text
//! gate    = h × W_input_gate.T              // [hidden] -> [ple_dim]   (f32_gemv)
//! gate    = gelu_tanh(gate) * per_layer_in  // [ple_dim]               (ple_gate_apply)
//! contrib = gate × W_projection.T           // [ple_dim] -> [hidden]   (f32_gemv)
//! h      += rms_norm(contrib) · w           // [hidden]                (post_ffn_norm_residual_add, reused)
//! ```
//!
//! Four dispatches per layer. The closing norm+residual reuses the
//! existing `post_ffn_norm_residual_add` shader by aliasing `h_post_attn`
//! and `new_h` to the same `h` buffer (per-thread read-modify-write at
//! the same index → no race on the single-TG kernel).
//!
//! `per_layer_input` for the active position is precomputed CPU-side
//! once per generation (Stream 1 model-projection × main embeds + Stream
//! 2 per-layer token embedding lookup, RMS-normed and × 1/sqrt(2)) and
//! uploaded into a Metal buffer as `[num_layers × ple_dim]` for decode
//! or `[num_layers × seq_len × ple_dim]` for prefill — see
//! `larql-inference/.../generate/gpu.rs`.

use crate::MetalBackend;
use larql_compute::FullPipelineLayer;
use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

/// Buffers the PLE block reads from / writes into. All buffers are f32.
pub(super) struct PleBufs<'a> {
    /// `[hidden]` residual stream — read-modify-write (`h += normed_contrib`).
    pub h: &'a Buffer,
    /// Buffer holding the precomputed per-layer-input table — typically a
    /// single contiguous `[num_layers × ple_dim]` (decode) or
    /// `[num_layers × seq_len × ple_dim]` (prefill) Metal buffer. The
    /// caller picks the row for this layer / position by setting
    /// `per_layer_input_offset` (in bytes).
    pub per_layer_input: &'a Buffer,
    /// Byte offset into `per_layer_input` for the active (layer, position)'s
    /// `[ple_dim]` row.
    pub per_layer_input_offset: u64,
    /// `[ple_dim]` scratch — populated by gate matvec, mutated by gate-apply.
    pub gate_scratch: &'a Buffer,
    /// `[hidden]` scratch — populated by projection matvec, consumed by closing norm+add.
    pub contrib_scratch: &'a Buffer,
}

impl MetalBackend {
    /// Encode the PLE block for one layer onto `enc`. No-op when the layer
    /// is not PLE-active (`layer.ple_spec()` returns `None`).
    ///
    /// The dispatch geometry mirrors:
    ///   - `f32_gemv` for the two matvecs (one TG per `rows_per_tg` rows).
    ///   - `ple_gate_apply` flat per-element (one thread per ple_dim).
    ///   - `post_ffn_norm_residual_add` single TG covering all of `hidden`.
    pub(super) fn encode_per_layer_embed(
        &self,
        enc: &ComputeCommandEncoderRef,
        layer: &FullPipelineLayer,
        bufs: PleBufs<'_>,
        hidden: usize,
        ple_dim: usize,
    ) {
        let Some(ple) = layer.ple_spec() else {
            return;
        };

        let gemv = &self.f32_gemv_pipeline;

        // ── Step 1: gate = h × W_input_gate.T → [ple_dim] ──
        let w_gate_buf = self.bufs.get_f32(ple.input_gate);
        let n1 = ple_dim as u32;
        let k1 = hidden as u32;
        let tgs1 = (ple_dim as u64).div_ceil(gemv.rows_per_tg);
        enc.set_compute_pipeline_state(&gemv.state);
        enc.set_buffer(0, Some(&w_gate_buf), 0);
        enc.set_buffer(1, Some(bufs.h), 0);
        enc.set_buffer(2, Some(bufs.gate_scratch), 0);
        enc.set_bytes(3, 4, &n1 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k1 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(tgs1, 1, 1),
            MTLSize::new(gemv.threads_per_tg, 1, 1),
        );

        // ── Step 2: gate = gelu_tanh(gate) * per_layer_input  (in-place) ──
        let n2 = ple_dim as u32;
        enc.set_compute_pipeline_state(&self.ffn.ple_gate_apply_pipeline);
        enc.set_buffer(0, Some(bufs.gate_scratch), 0); // gate_in
        enc.set_buffer(1, Some(bufs.per_layer_input), bufs.per_layer_input_offset); // per_layer_input row
        enc.set_buffer(2, Some(bufs.gate_scratch), 0); // gate_out (alias of gate_in)
        enc.set_bytes(3, 4, &n2 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_threads(
            MTLSize::new(ple_dim as u64, 1, 1),
            MTLSize::new((ple_dim as u64).min(256), 1, 1),
        );

        // ── Step 3: contrib = gate × W_projection.T → [hidden] ──
        let w_proj_buf = self.bufs.get_f32(ple.projection);
        let n3 = hidden as u32;
        let k3 = ple_dim as u32;
        let tgs3 = (hidden as u64).div_ceil(gemv.rows_per_tg);
        enc.set_compute_pipeline_state(&gemv.state);
        enc.set_buffer(0, Some(&w_proj_buf), 0);
        enc.set_buffer(1, Some(bufs.gate_scratch), 0);
        enc.set_buffer(2, Some(bufs.contrib_scratch), 0);
        enc.set_bytes(3, 4, &n3 as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &k3 as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(tgs3, 1, 1),
            MTLSize::new(gemv.threads_per_tg, 1, 1),
        );

        // ── Step 4: h += rms_norm(contrib) · w  (post_ffn_norm_residual_add, reused) ──
        // Aliases h_post_attn (buffer 1) and new_h (buffer 3) to the same
        // residual buffer. The shader is single-TG with each thread doing
        // `read h[i]` → `write h[i]` at one index per iteration, so the
        // aliasing has no race.
        let post_norm_buf = self.bufs.get_f32(ple.post_norm);
        let hidden_val = hidden as u32;
        let eps = layer.eps;
        let norm_offset = layer.norm_offset;
        enc.set_compute_pipeline_state(&self.norms.post_ffn_norm_residual_add_pipeline);
        enc.set_buffer(0, Some(bufs.contrib_scratch), 0); // down_out (== contrib)
        enc.set_buffer(1, Some(bufs.h), 0); // h_post_attn (== h)
        enc.set_buffer(2, Some(&post_norm_buf), 0); // w
        enc.set_buffer(3, Some(bufs.h), 0); // new_h (== h, alias)
        enc.set_bytes(4, 4, &hidden_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &norm_offset as *const f32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize::new(1, 1, 1),
            MTLSize::new(
                crate::kernels::DISPATCH_TG_MAX_THREADS.min(hidden as u64),
                1,
                1,
            ),
        );
    }
}
