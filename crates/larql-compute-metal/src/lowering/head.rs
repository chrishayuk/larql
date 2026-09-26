//! Lowering the final norm and output head (VINDEX3-G6c-2).
//!
//! ```text
//! h_final ─ final norm ─ lm_head matvec ─ multiplier ─ softcap ─ logits
//! ```
//!
//! Three judged facts live here that nothing else in the stack carries,
//! and each is a different value from its nearest neighbour:
//!
//! - **The final norm is a third norm configuration.** Muse-Glimmer's
//!   pre-block norms use eps 1e-5 with `weight_offset` 1.0, its post-block
//!   norms 1e-8 with 1.0, and its final norm 1e-5 with **0.0**. Carrying
//!   the branch norms' offset here is a silent centred-vs-uncentred bug.
//! - **The output multiplier** (0.196… for Glimmer). `None` = the op is
//!   absent, which is not the same claim as multiplying by one.
//! - **The final logit softcap** (20.0), applied *after* the multiplier.
//!   That order is semantic, unlike the query-scale/RoPE pair: tanh is
//!   nonlinear, so `softcap(m·x)` and `m·softcap(x)` are different
//!   functions.

use metal::Buffer;

use super::profile::{Stage, StageEncoders};

use super::{LoweredMatrix, MatmulRowsTarget, MatvecTarget};
use crate::MetalBackend;

/// What the head reads.
pub struct HeadWeights<'a> {
    pub projection: LoweredMatrix<'a>,
    /// Final norm weight (f32).
    pub norm_weight: &'a Buffer,
}

/// Device scratch: `hidden` then `vocab` floats.
pub struct HeadScratch<'a> {
    pub normed: &'a Buffer,
    pub raw_logits: &'a Buffer,
}

/// Geometry and judged semantics, straight off the plan.
pub struct HeadShape {
    pub hidden: usize,
    pub vocab: usize,
    pub norm_eps: f32,
    /// Glimmer's final norm is **uncentred** (0.0) where its branch norms
    /// are centred (1.0).
    pub norm_weight_offset: f32,
    /// `None` = the op is absent.
    pub multiplier: Option<f32>,
    /// `None` = the op is absent.
    pub softcap: Option<f32>,
}

impl MetalBackend {
    /// Encode final norm → head projection → multiplier → softcap.
    pub fn encode_head(
        &self,
        encs: &mut dyn StageEncoders,
        h_final: &Buffer,
        logits_out: &Buffer,
        w: &HeadWeights<'_>,
        s: &HeadScratch<'_>,
        shape: &HeadShape,
    ) {
        let enc = encs.stage(Stage::Head);
        crate::stages::input_norm::encode_f32(
            enc,
            &self.norms.rms_norm_pipeline,
            h_final,
            0,
            w.norm_weight,
            s.normed,
            0,
            shape.hidden,
            shape.norm_eps,
            shape.norm_weight_offset,
        );
        self.encode_matvec(
            enc,
            &w.projection,
            &MatvecTarget {
                x: s.normed,
                out: s.raw_logits,
                out_offset: 0,
                n: shape.vocab,
                k: shape.hidden,
            },
        );
        // Absent ops encode as 0.0, which the kernel reads as "skip" —
        // distinct from a multiplier of one or a cap of zero.
        let pipeline = &self.norms.head_scale_softcap_pipeline;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(s.raw_logits), 0);
        enc.set_buffer(1, Some(logits_out), 0);
        super::set_u32(enc, 2, shape.vocab as u32);
        super::set_f32(enc, 3, shape.multiplier.unwrap_or(0.0));
        super::set_f32(enc, 4, shape.softcap.unwrap_or(0.0));
        super::dispatch_linear(enc, pipeline, shape.vocab);
    }
}

impl MetalBackend {
    /// VERIFY-N: [`Self::encode_head`] over `rows` positions. `h_final`,
    /// `s.normed` are `[rows, hidden]`; `s.raw_logits` and `logits_out`
    /// are `[rows, vocab]`. The final norm runs per row, the projection
    /// as one multi-position matmul, multiplier/softcap elementwise over
    /// the block.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_head_rows(
        &self,
        encs: &mut dyn StageEncoders,
        h_final: &Buffer,
        logits_out: &Buffer,
        w: &HeadWeights<'_>,
        s: &HeadScratch<'_>,
        shape: &HeadShape,
        rows: usize,
    ) -> Result<(), String> {
        let enc = encs.stage(Stage::Head);
        self.encode_rms_norm_rows(
            enc,
            h_final,
            0,
            w.norm_weight,
            s.normed,
            0,
            shape.hidden,
            rows,
            shape.norm_eps,
            shape.norm_weight_offset,
        );
        self.encode_matmul_rows(
            enc,
            &w.projection,
            &MatmulRowsTarget {
                x: s.normed,
                x_offset: 0,
                out: s.raw_logits,
                out_offset: 0,
                n: shape.vocab,
                k: shape.hidden,
                rows,
            },
        )?;
        let pipeline = &self.norms.head_scale_softcap_pipeline;
        enc.set_compute_pipeline_state(pipeline);
        enc.set_buffer(0, Some(s.raw_logits), 0);
        enc.set_buffer(1, Some(logits_out), 0);
        super::set_u32(enc, 2, (rows * shape.vocab) as u32);
        super::set_f32(enc, 3, shape.multiplier.unwrap_or(0.0));
        super::set_f32(enc, 4, shape.softcap.unwrap_or(0.0));
        super::dispatch_linear(enc, pipeline, rows * shape.vocab);
        Ok(())
    }
}

/// Elements one `argmax_partial` threadgroup scans. 256 threads × 16
/// elements; a 262K vocabulary leaves 64 partials for the final pass.
pub const ARGMAX_BLOCK: usize = 4096;
/// Threads per argmax threadgroup (both passes).
pub const ARGMAX_THREADS: u64 = 256;

/// Device scratch for the two-pass argmax: block partials and the
/// single output index.
pub struct ArgmaxScratch<'a> {
    /// `argmax_partials(n)` f32s.
    pub partial_vals: &'a Buffer,
    /// `argmax_partials(n)` u32s.
    pub partial_idx: &'a Buffer,
    /// One u32: the index of the maximum (first on ties).
    pub out: &'a Buffer,
}

/// Number of block partials an argmax over `n` elements produces.
pub fn argmax_partials(n: usize) -> usize {
    n.div_ceil(ARGMAX_BLOCK).max(1)
}

impl MetalBackend {
    /// Encode `out[0] = argmax(x[0..n])`, first index on ties — the same
    /// contract as a host scan with strict `>`. Two dispatches.
    pub fn encode_argmax(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        x: &Buffer,
        n: usize,
        s: &ArgmaxScratch<'_>,
    ) {
        self.encode_argmax_at(enc, x, 0, n, s);
    }

    /// [`Self::encode_argmax`] over `x[x_offset..]` (a byte offset) — one
    /// position's logits inside a `[rows, vocab]` block.
    pub fn encode_argmax_at(
        &self,
        enc: &metal::ComputeCommandEncoderRef,
        x: &Buffer,
        x_offset: u64,
        n: usize,
        s: &ArgmaxScratch<'_>,
    ) {
        let blocks = argmax_partials(n);
        let p1 = &self.norms.argmax_partial_pipeline;
        enc.set_compute_pipeline_state(p1);
        enc.set_buffer(0, Some(x), x_offset);
        super::set_u32(enc, 1, n as u32);
        super::set_u32(enc, 2, ARGMAX_BLOCK as u32);
        enc.set_buffer(3, Some(s.partial_vals), 0);
        enc.set_buffer(4, Some(s.partial_idx), 0);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(blocks as u64, 1, 1),
            metal::MTLSize::new(ARGMAX_THREADS, 1, 1),
        );
        let p2 = &self.norms.argmax_final_pipeline;
        enc.set_compute_pipeline_state(p2);
        enc.set_buffer(0, Some(s.partial_vals), 0);
        enc.set_buffer(1, Some(s.partial_idx), 0);
        super::set_u32(enc, 2, blocks as u32);
        enc.set_buffer(3, Some(s.out), 0);
        enc.dispatch_thread_groups(
            metal::MTLSize::new(1, 1, 1),
            metal::MTLSize::new(ARGMAX_THREADS, 1, 1),
        );
    }
}
