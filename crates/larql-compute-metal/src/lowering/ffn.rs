//! Lowering a plan's gated FFN into one encoder (VINDEX3-G6b).
//!
//! The first fragment of a `LayerOpPlan` to be lowered, chosen because it
//! is self-contained — norm in, hidden-sized vector out, no KV cache and
//! no position policy — and because it is roughly three quarters of
//! Glimmer's weight bytes.
//!
//! Deliberately literal. Every operation the plan states gets its own
//! encoder action, in plan order:
//!
//! ```text
//! h ─ pre_ffn_norm ─┬─ gate proj ─┐
//!                   └─ up proj ───┴─ SiLU-GLU ─ down proj ─ post_ffn_norm ─ residual ─ h'
//! ```
//!
//! The post-FFN norm is present only under four-norm placement, and it
//! normalises the **branch output before** the residual add — not the
//! summed hidden state after it.
//!
//! No fusion yet. A fused activation+down kernel exists for Q4_K/Q6_K and
//! would be faster, but it would also make the correspondence between the
//! plan and the encoded work harder to check, and this rung's job is to
//! establish that correspondence. Fusion is a later, separately-judged
//! change.
//!
//! `stages::ffn::encode_gated` is not reused: it dispatches on
//! `larql_compute::QuantFormat`, which has no NVFP4 variant, so routing a
//! plan through it would mean either lying about the format or extending
//! the serving path's enum for a format it does not serve.

use metal::{Buffer, ComputeCommandEncoderRef};

use super::profile::{Stage, StageEncoders};

use super::{
    nvfp4_residual_fusion_enabled, nvfp4_segment, LoweredMatrix, MatvecOperands, MatvecTarget,
    PostNorm,
};
use crate::MetalBackend;

/// The weight streams one gated FFN reads, already resident.
pub struct FfnWeights<'a> {
    pub gate: LoweredMatrix<'a>,
    pub up: LoweredMatrix<'a>,
    pub down: LoweredMatrix<'a>,
    /// Pre-FFN norm weight (f32).
    pub norm_weight: &'a Buffer,
    /// The post-FFN norm, under four-norm placement. `None` = absent.
    pub post_norm: Option<PostNorm<'a>>,
}

/// Device scratch the lowered FFN needs. Caller-owned so the whole layer
/// (later, the whole token) can allocate once and reuse across layers
/// rather than churning the pool per operation.
pub struct FfnScratch<'a> {
    /// `hidden` floats — normalised input.
    pub normed: &'a Buffer,
    /// `intermediate` floats each.
    pub gate: &'a Buffer,
    pub up: &'a Buffer,
    pub act: &'a Buffer,
    /// `hidden` floats — down-projection output, before the residual.
    pub down: &'a Buffer,
}

/// Geometry and judged semantics, straight off the plan.
pub struct FfnShape {
    pub hidden: usize,
    pub intermediate: usize,
    /// From the plan's `NormOp`.
    pub norm_eps: f32,
    /// Centred-norm convention (`1 + w`); a plan fact, never assumed.
    pub norm_weight_offset: f32,
    /// The gate activation, from the plan's `FfnOp`: SiLU or tanh-GELU,
    /// each its own served kernel.
    pub activation: FfnActivation,
}

/// The gate/up COMBINE the lowering has a kernel for.
///
/// Named for the nonlinearity because for two of its three arms that is
/// all it is. The third is a whole combine — Kimi-K3's SiTU-GLU carries
/// its own parameters and does not read a nonlinearity beside it — which
/// is the same fold `larql_compute::MoeGateRule` performs, and for the
/// same reason: a layer must carry ONE combine rule rather than a policy
/// and an activation that can disagree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FfnActivation {
    Silu,
    GeluTanh,
    /// `beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta)`,
    /// the `situ_glu` kernel. `linear_beta: None` leaves the up branch
    /// untouched — a different function, not an infinite bound.
    SituGlu {
        beta: f32,
        linear_beta: Option<f32>,
    },
}

impl MetalBackend {
    /// Encode `h' = h + down(silu(gate(norm(h))) * up(norm(h)))`.
    ///
    /// `h_in` and `h_out` may be the same buffer only if the caller is
    /// certain no dispatch reads `h_in` after `h_out` is written; the
    /// residual reads both, so they must differ.
    pub fn encode_gated_ffn(
        &self,
        encs: &mut dyn StageEncoders,
        h_in: &Buffer,
        h_out: &Buffer,
        w: &FfnWeights<'_>,
        s: &FfnScratch<'_>,
        shape: &FfnShape,
    ) {
        let enc = encs.stage(Stage::DenseFfn);
        // A-5b rung 2a: under two-norm placement the residual add folds
        // into the down-projection write (bit-identical), one dispatch
        // fewer per layer. Four-norm placement keeps the branch norm.
        let fused_down = match w.post_norm.as_ref() {
            None if nvfp4_residual_fusion_enabled() => {
                nvfp4_segment(&w.down, h_out, 0, shape.hidden)
            }
            _ => None,
        };
        match fused_down {
            Some(seg) => {
                self.encode_gated_ffn_gate_up_act(enc, h_in, w, s, shape);
                // `_sliced` carries the segment's byte offsets. `down`
                // is a whole-buffer resident today, so both forms agree
                // — but the plain call silently DISCARDS the offsets
                // (MatvecOperands has no field for them), so the moment
                // dense FFN operands are co-located the way attention's
                // now are, this would bind at offset 0 and compute a
                // different matrix's rows with the residual added on
                // top: finite, plausible, wrong. Same defect that was
                // live on the o-proj path.
                self.encode_nvfp4_matvec_residual_sliced(
                    enc,
                    &MatvecOperands {
                        packed: seg.packed,
                        scales: seg.scales,
                        x: s.act,
                        out: h_out,
                        out_offset: 0,
                        n: shape.hidden,
                        k: shape.intermediate,
                    },
                    seg.tensor_scale,
                    h_in,
                    seg.packed_offset,
                    seg.scales_offset,
                );
            }
            None => {
                self.encode_gated_ffn_branch(enc, h_in, w, s, shape);
                // 5. post-FFN norm (four-norm placement only), then the
                //    residual. `b_scale` is 1.0: a residual multiplier is a
                //    judged plan fact and Glimmer's FFN residual has none,
                //    so passing anything else here would invent semantics.
                self.encode_branch_norm_then_residual(
                    enc,
                    h_in,
                    s.down,
                    h_out,
                    w.post_norm.as_ref(),
                    shape.hidden,
                );
            }
        }
    }

    /// The FFN branch alone — `s.down = down(act(gate(norm(h))) *
    /// up(norm(h)))`, no post-norm, no residual — so a hybrid layer can
    /// normalise and sum it with the expert branch before the layer's own
    /// post-FFN norm.
    pub fn encode_gated_ffn_branch(
        &self,
        enc: &ComputeCommandEncoderRef,
        h_in: &Buffer,
        w: &FfnWeights<'_>,
        s: &FfnScratch<'_>,
        shape: &FfnShape,
    ) {
        self.encode_gated_ffn_gate_up_act(enc, h_in, w, s, shape);
        // 4. down projection.
        self.encode_matvec(
            enc,
            &w.down,
            &MatvecTarget {
                x: s.act,
                out: s.down,
                out_offset: 0,
                n: shape.hidden,
                k: shape.intermediate,
            },
        );
    }

    /// The branch from an input that is ALREADY pre-normed into
    /// `s.normed` (a hybrid layer's multi-norm wrote it): gate/up,
    /// activation, down into `s.down`.
    pub fn encode_gated_ffn_branch_prenormed(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: &FfnWeights<'_>,
        s: &FfnScratch<'_>,
        shape: &FfnShape,
    ) {
        self.encode_gate_up_act_from_normed(enc, w, s, shape);
        self.encode_matvec(
            enc,
            &w.down,
            &MatvecTarget {
                x: s.act,
                out: s.down,
                out_offset: 0,
                n: shape.hidden,
                k: shape.intermediate,
            },
        );
    }

    /// Steps 1–3 of the branch: pre-norm, gate/up, activation into
    /// `s.act` — shared by the plain branch and the residual-fused down.
    fn encode_gated_ffn_gate_up_act(
        &self,
        enc: &ComputeCommandEncoderRef,
        h_in: &Buffer,
        w: &FfnWeights<'_>,
        s: &FfnScratch<'_>,
        shape: &FfnShape,
    ) {
        // 1. pre-FFN norm.
        crate::stages::input_norm::encode_f32(
            enc,
            &self.norms.rms_norm_pipeline,
            h_in,
            0,
            w.norm_weight,
            s.normed,
            0,
            shape.hidden,
            shape.norm_eps,
            shape.norm_weight_offset,
        );
        self.encode_gate_up_act_from_normed(enc, w, s, shape);
    }

    /// Steps 2–3: gate/up over `s.normed`, activation into `s.act`.
    fn encode_gate_up_act_from_normed(
        &self,
        enc: &ComputeCommandEncoderRef,
        w: &FfnWeights<'_>,
        s: &FfnScratch<'_>,
        shape: &FfnShape,
    ) {
        // 2. gate and up projections. Independent of each other, and the
        //    serial encoder still orders them after the norm that feeds
        //    them. A-5b: both NVFP4 → one dispatch.
        match (
            nvfp4_segment(&w.gate, s.gate, 0, shape.intermediate),
            nvfp4_segment(&w.up, s.up, 0, shape.intermediate),
        ) {
            (Some(g), Some(u)) => {
                self.encode_nvfp4_matvec_segments(enc, s.normed, shape.hidden, &[g, u])
            }
            _ => {
                self.encode_matvec(
                    enc,
                    &w.gate,
                    &MatvecTarget {
                        x: s.normed,
                        out: s.gate,
                        out_offset: 0,
                        n: shape.intermediate,
                        k: shape.hidden,
                    },
                );
                self.encode_matvec(
                    enc,
                    &w.up,
                    &MatvecTarget {
                        x: s.normed,
                        out: s.up,
                        out_offset: 0,
                        n: shape.intermediate,
                        k: shape.hidden,
                    },
                );
            }
        }
        // 3. the gated combine, by the plan's own rule.
        match shape.activation {
            FfnActivation::SituGlu { beta, linear_beta } => {
                crate::kernels::ffn::bind_situ_glu(
                    enc,
                    &self.ffn.situ_glu_pipeline,
                    (s.gate, 0),
                    (s.up, 0),
                    (s.act, 0),
                    shape.intermediate as u32,
                    beta,
                    linear_beta,
                    // The dense FFN's biases, when it has any, are already
                    // in `s.gate`/`s.up` — this lowering adds them through
                    // the `bias_add` kernel after each projection, not in
                    // the combine — so there is nothing staged for the
                    // combine to drop.
                    false,
                );
                super::dispatch_linear(enc, &self.ffn.situ_glu_pipeline, shape.intermediate);
            }
            FfnActivation::Silu | FfnActivation::GeluTanh => {
                let geglu = match shape.activation {
                    FfnActivation::GeluTanh => &self.ffn.geglu_gelu_tanh_pipeline,
                    _ => &self.ffn.geglu_pipeline,
                };
                encode_elementwise(enc, geglu, &[s.gate, s.up, s.act], shape.intermediate);
            }
        }
    }
}

/// Three-buffer elementwise kernel with a trailing `uint` length —
/// the shape `geglu_silu` and its siblings share.
fn encode_elementwise(
    enc: &ComputeCommandEncoderRef,
    pipeline: &metal::ComputePipelineState,
    bufs: &[&Buffer; 3],
    len: usize,
) {
    enc.set_compute_pipeline_state(pipeline);
    for (i, b) in bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(*b), 0);
    }
    let n = len as u32;
    enc.set_bytes(3, 4, &n as *const u32 as *const std::ffi::c_void);
    super::dispatch_linear(enc, pipeline, len);
}
