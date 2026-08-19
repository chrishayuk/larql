//! Lowering a whole decoder stack into one scheduling domain (G6c-1).
//!
//! G6b closed one layer. This composes N of them with the hidden state
//! and every layer's KV **resident on the device**, so the host is not in
//! the dependency chain at any point between the first upload and the
//! final readback.
//!
//! That is the entire point of the rung. The interpreter path commits and
//! waits 209 times per Glimmer token, and the measured cost of doing so
//! is queue starvation: ~215-271 us of empty queue before each dispatch,
//! flat in bytes, collapsing to ~57 us at queue depth 32. Here the queue
//! never drains, because nothing needs the host's answer.
//!
//! ## Two structural invariants
//!
//! ```text
//! no wait_until_completed() inside the layer loop
//! no readback / contents() inside the layer loop
//! ```
//!
//! Both are enforced by construction rather than by discipline:
//! [`MetalBackend::encode_stack`] takes an encoder it does not own and
//! returns nothing, so it *cannot* wait or read. The caller commits once,
//! after the whole stack is encoded.
//!
//! ## Per-layer policy is static
//!
//! Muse-Glimmer's 52 layers are 39 sliding(2048)+RoPE and 13 full+NoPE in
//! a 3:1 pattern. Every one of those differences is known before
//! execution begins — they come from the plan, not from anything the
//! stack computes — so encoding them costs no round trip. Note that in
//! *this* model span and position happen to be perfectly correlated
//! (sliding↔RoPE, full↔NoPE); they are independent fields and a caller
//! may combine them freely.
//!
//! ## Checkpoints
//!
//! Localising a divergence inside a 52-layer stream would normally mean
//! reintroducing the readbacks this rung removes. Instead a caller names
//! layers whose output should be *copied to its own device buffer*; all
//! of them are read after the single scheduling domain completes.

use metal::{Buffer, ComputeCommandEncoderRef};

use super::attention::{AttnScratch, AttnShape, AttnWeights};
use super::ffn::{FfnScratch, FfnShape, FfnWeights};
use crate::moe_descriptor::MoeExpertDescriptorTable;
use crate::moe_dispatch::MoeScratch;
use crate::MetalBackend;
use larql_compute::MoeLayerWeights;

/// A layer's routed FFN, as the served descriptor MoE path consumes it:
/// the router (a decoder-stack operand) and the expert bank (its own
/// object) resolved to registered regions, with the routing/gate
/// semantics carried on `moe` — the same `MoeLayerWeights` a served
/// `--routed-from` run builds, but assembled from a `RoutedFfnOp` rather
/// than a model family. Scratch and the descriptor table are per layer
/// because the whole stack encodes into one command buffer, so two
/// layers cannot share output buffers.
pub struct RoutedFfnLowering<'a> {
    pub moe: MoeLayerWeights<'a>,
    pub scratch: &'a MoeScratch,
    pub table: &'a MoeExpertDescriptorTable,
    /// Pre-experts norm epsilon (GPT-OSS: the pre-FFN norm's).
    pub eps: f32,
}

/// Gemma 4's hybrid FFN: a dense MLP and a routed expert block in one
/// layer, composed from the same encodes the dense and routed arms use —
/// transcribed from the plan's `HybridFfnOp` (itself HF's
/// `Gemma4TextDecoderLayer`):
///
/// ```text
/// d = post_dense_norm(dense(pre_ffn_norm(r)))
/// e = post_experts_norm(Σ w·expert(pre_experts_norm(r)))     router ← r
/// h' = (r + post_ffn_norm(d + e)) × layer_scale
/// ```
///
/// The router reads the RAW residual through a scale-less RMS norm times
/// `router.scale` times `hidden^-0.5` — folded here into one weighted
/// RMS-norm dispatch whose weight is `router_conditioning =
/// router.scale · hidden^-0.5` (weight offset 0), then projected,
/// softmaxed, top-k'd, renormalised and scaled per expert on the GPU
/// (`encode_moe_router_select` with `renormalize` and a per-expert scale
/// buffer). The experts read `pre_experts_norm(r)` through the descriptor
/// path with a ZERO residual, so the combine yields the bare expert sum.
pub struct HybridFfnLowering<'a> {
    pub dense: FfnWeights<'a>,
    pub dense_shape: FfnShape,
    pub routed: RoutedFfnLowering<'a>,
    /// `router.scale · hidden^-0.5`, `[hidden]`, as an RMS-norm weight.
    pub router_conditioning: &'a Buffer,
    /// `[experts]`, applied to the selected weights after renormalisation.
    pub per_expert_scale: &'a Buffer,
    /// The three branch norms' weights (`[hidden]` each) and their
    /// shared epsilon / offset.
    pub pre_experts_norm: &'a Buffer,
    pub post_dense_norm: &'a Buffer,
    pub post_experts_norm: &'a Buffer,
    pub branch_norm_eps: f32,
    pub branch_norm_weight_offset: f32,
    /// The layer's post-FFN norm (four-norm placement), applied to the
    /// summed branches before the residual add.
    pub post_ffn_norm: Option<super::PostNorm<'a>>,
    /// The whole layer output is multiplied by this after the residual
    /// add. `None` = no such op (not a multiply by one).
    pub layer_scale: Option<f32>,
}

/// A layer's FFN: dense, routed, or both. The stack encoder runs the
/// arm into the same hidden-state slot.
pub enum LayerFfnLowering<'a> {
    Dense {
        weights: FfnWeights<'a>,
        shape: FfnShape,
    },
    /// Boxed: a routed FFN carries a whole `MoeLayerWeights` (per-expert
    /// slice vectors), several times a dense op's size.
    Routed(Box<RoutedFfnLowering<'a>>),
    Hybrid(Box<HybridFfnLowering<'a>>),
}

/// One layer's complete lowering input.
pub struct LayerLowering<'a> {
    pub attn: AttnWeights<'a>,
    pub attn_shape: AttnShape,
    pub ffn: LayerFfnLowering<'a>,
    /// This layer's KV cache, `[T, num_kv, head_dim]`. Per layer, and
    /// resident for the whole stack — sharing one across layers would
    /// silently make every layer attend to the last layer's keys.
    pub k_cache: &'a Buffer,
    pub v_cache: &'a Buffer,
    /// This layer's rotary inverse-frequency table (`head_dim/2`
    /// floats), built for ITS position policy and head width — Gemma 4's
    /// sliding and full layers differ in both. A NoPE layer binds any live
    /// buffer (the table is not read).
    pub inv_freq: &'a Buffer,
}

/// Scratch reused by every layer. Allocated once for the stack, not per
/// layer: 52 layers of per-layer allocation is 52 pool round trips of
/// pure overhead, and the buffers are dead the moment the layer ends.
pub struct StackScratch<'a> {
    /// Two `hidden`-sized buffers the hidden state alternates between.
    /// Ping-pong rather than in-place because the residual add reads the
    /// layer input while writing the layer output.
    pub h_a: &'a Buffer,
    pub h_b: &'a Buffer,
    /// Attention intermediates, all `hidden` or `q_rows` sized.
    pub attn_normed: &'a Buffer,
    pub q: &'a Buffer,
    pub gate: &'a Buffer,
    pub concat: &'a Buffer,
    pub gated: &'a Buffer,
    pub attn_out: &'a Buffer,
    pub attn_post: &'a Buffer,
    /// FFN intermediates.
    pub ffn_normed: &'a Buffer,
    pub ffn_gate: &'a Buffer,
    pub ffn_up: &'a Buffer,
    pub ffn_act: &'a Buffer,
    pub ffn_down: &'a Buffer,
    pub ffn_post: &'a Buffer,
    /// Hybrid-FFN intermediates (`hidden` each). `None` when the stack has
    /// no hybrid layer — a dense or routed stack allocates nothing for
    /// them, and a hybrid layer in a stack without them is a caller bug
    /// the encoder refuses loudly.
    pub hybrid: Option<HybridScratch<'a>>,
}

/// The hybrid layer's own intermediates.
pub struct HybridScratch<'a> {
    /// The dense branch after its post-norm.
    pub dense_out: &'a Buffer,
    /// The conditioned router input.
    pub router_in: &'a Buffer,
    /// The bare weighted expert sum (combine with a zero residual).
    pub expert_sum: &'a Buffer,
    /// The expert branch after its post-norm.
    pub experts_out: &'a Buffer,
    /// `d + e`.
    pub branch_sum: &'a Buffer,
    /// A `hidden`-sized buffer of zeros, the combine's residual input.
    pub zero: &'a Buffer,
}

/// A layer whose output should be captured, and where to put it.
pub struct Checkpoint<'a> {
    /// Capture the hidden state *after* this layer index completes.
    pub after_layer: usize,
    /// A `hidden`-sized device buffer the caller reads after the command
    /// buffer completes.
    pub into: &'a Buffer,
}

impl MetalBackend {
    /// Encode `layers` back to back into `enc`, hidden state resident
    /// throughout.
    ///
    /// Returns which of the two ping-pong buffers holds the final hidden
    /// state — the caller cannot know without counting layers, and
    /// guessing is a silent off-by-one that returns a whole layer's stale
    /// output.
    ///
    /// Encodes only. No commit, no wait, no readback: the caller owns the
    /// scheduling domain, which is what keeps the queue full.
    pub fn encode_stack<'a>(
        &self,
        enc: &ComputeCommandEncoderRef,
        h_in: &'a Buffer,
        layers: &[LayerLowering<'_>],
        s: &StackScratch<'a>,
        checkpoints: &[Checkpoint<'_>],
    ) -> &'a Buffer {
        let mut src = h_in;
        for (index, layer) in layers.iter().enumerate() {
            // Alternate destinations so no dispatch writes a buffer an
            // earlier dispatch in the same layer still reads.
            let mid = if std::ptr::eq(src, s.h_a) {
                s.h_b
            } else {
                s.h_a
            };
            let dst = if std::ptr::eq(mid, s.h_a) {
                s.h_b
            } else {
                s.h_a
            };

            let ascratch = AttnScratch {
                normed: s.attn_normed,
                q: s.q,
                k_cache: layer.k_cache,
                v_cache: layer.v_cache,
                gate: s.gate,
                concat: s.concat,
                gated: s.gated,
                attn_out: s.attn_out,
                inv_freq: layer.inv_freq,
            };
            self.encode_attention(enc, src, mid, &layer.attn, &ascratch, &layer.attn_shape);

            let hidden = match &layer.ffn {
                LayerFfnLowering::Dense { weights, shape } => {
                    let fscratch = FfnScratch {
                        normed: s.ffn_normed,
                        gate: s.ffn_gate,
                        up: s.ffn_up,
                        act: s.ffn_act,
                        down: s.ffn_down,
                    };
                    self.encode_gated_ffn(enc, mid, dst, weights, &fscratch, shape);
                    shape.hidden
                }
                // The routed FFN reads the post-attention residual (`mid`)
                // and writes `dst = mid + Σ w·expert` — the same slot the
                // dense FFN fills, so the stack schedule is unchanged. The
                // pre-experts norm rides inside the routed encode.
                LayerFfnLowering::Routed(r) => {
                    self.encode_moe_layer_gpu_route(
                        enc, &r.moe, r.scratch, r.table, mid, dst, r.eps,
                    );
                    hidden_of(&r.moe)
                }
                LayerFfnLowering::Hybrid(h) => {
                    let hs = s
                        .hybrid
                        .as_ref()
                        .expect("a hybrid layer needs the stack's hybrid scratch");
                    let fscratch = FfnScratch {
                        normed: s.ffn_normed,
                        gate: s.ffn_gate,
                        up: s.ffn_up,
                        act: s.ffn_act,
                        down: s.ffn_down,
                    };
                    self.encode_hybrid_ffn(enc, mid, dst, h, &fscratch, hs);
                    h.dense_shape.hidden
                }
            };

            for cp in checkpoints.iter().filter(|c| c.after_layer == index) {
                // A copy, not a readback: the value lands in a device
                // buffer the caller reads *after* the stream completes,
                // so localisation costs no round trip.
                self.encode_residual_add(enc, dst, dst, cp.into, hidden, 0.0);
            }
            src = dst;
        }
        src
    }
}

impl StackScratch<'_> {
    /// Buffers the attention half needs, so a caller allocating scratch
    /// cannot silently under-size one and read past it.
    pub const ATTENTION_BUFFERS: usize = 7;
    /// Buffers the FFN half needs.
    pub const FFN_BUFFERS: usize = 6;
    /// Buffers a hybrid layer needs on top of the FFN half.
    pub const HYBRID_BUFFERS: usize = 6;
}

impl MetalBackend {
    /// Encode one hybrid FFN layer from `mid` (the post-attention
    /// residual) into `dst` — see [`HybridFfnLowering`] for the program.
    fn encode_hybrid_ffn(
        &self,
        enc: &ComputeCommandEncoderRef,
        mid: &Buffer,
        dst: &Buffer,
        h: &HybridFfnLowering<'_>,
        fscratch: &FfnScratch<'_>,
        hs: &HybridScratch<'_>,
    ) {
        let hidden = h.dense_shape.hidden;
        // Dense branch: d = post_dense_norm(dense(pre_ffn_norm(r))).
        self.encode_gated_ffn_branch(enc, mid, &h.dense, fscratch, &h.dense_shape);
        crate::stages::input_norm::encode_f32(
            enc,
            &self.norms.rms_norm_pipeline,
            fscratch.down,
            0,
            h.post_dense_norm,
            hs.dense_out,
            0,
            hidden,
            h.branch_norm_eps,
            h.branch_norm_weight_offset,
        );
        // Expert input: pre_experts_norm(r), into the MoE scratch's
        // staging buffer (its zero tail serves a padded row width exactly
        // as the served path's does).
        let x_experts = h.routed.scratch.x_buf.clone();
        crate::stages::input_norm::encode_f32(
            enc,
            &self.norms.rms_norm_pipeline,
            mid,
            0,
            h.pre_experts_norm,
            &x_experts,
            0,
            hidden,
            h.branch_norm_eps,
            h.branch_norm_weight_offset,
        );
        // Router input: rms_no_weight(r) · scale · hidden^-0.5, one
        // weighted norm with the folded weight (offset 0).
        crate::stages::input_norm::encode_f32(
            enc,
            &self.norms.rms_norm_pipeline,
            mid,
            0,
            h.router_conditioning,
            hs.router_in,
            0,
            hidden,
            h.branch_norm_eps,
            0.0,
        );
        let moe = &h.routed.moe;
        let w_buf = self.bufs.get_f32(moe.router_proj);
        let logits =
            self.encode_moe_router_logits(enc, &w_buf, hs.router_in, None, moe.num_experts, hidden);
        let (ids, weights) = self.encode_moe_router_select(
            enc,
            &logits,
            Some(h.per_expert_scale),
            moe.num_experts,
            moe.top_k,
            true,
        );
        // Experts over pre_experts_norm(r), combined onto a ZERO residual:
        // the bare weighted sum lands in `expert_sum`.
        self.encode_experts_and_combine_descriptor_x_buf(
            enc,
            &x_experts,
            moe,
            h.routed.scratch,
            h.routed.table,
            &ids,
            &weights,
            hs.zero,
            hs.expert_sum,
        );
        crate::stages::input_norm::encode_f32(
            enc,
            &self.norms.rms_norm_pipeline,
            hs.expert_sum,
            0,
            h.post_experts_norm,
            hs.experts_out,
            0,
            hidden,
            h.branch_norm_eps,
            h.branch_norm_weight_offset,
        );
        // d + e, then the layer's post-FFN norm and the residual add.
        self.encode_residual_add(
            enc,
            hs.dense_out,
            hs.experts_out,
            hs.branch_sum,
            hidden,
            1.0,
        );
        self.encode_branch_norm_then_residual(
            enc,
            mid,
            hs.branch_sum,
            dst,
            h.post_ffn_norm.as_ref(),
            hidden,
        );
        if let Some(scale) = h.layer_scale {
            self.encode_scale_vector(enc, dst, hidden, scale);
        }
    }
}

/// Hidden width a routed layer writes — the router projection's input
/// width (`[num_experts, hidden]`).
fn hidden_of(moe: &MoeLayerWeights<'_>) -> usize {
    moe.router_proj.len() / moe.num_experts.max(1)
}
