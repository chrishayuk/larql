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

/// A layer's FFN: dense or routed. The stack encoder runs one or the
/// other into the same hidden-state slot.
pub enum LayerFfnLowering<'a> {
    Dense {
        weights: FfnWeights<'a>,
        shape: FfnShape,
    },
    /// Boxed: a routed FFN carries a whole `MoeLayerWeights` (per-expert
    /// slice vectors), several times a dense op's size.
    Routed(Box<RoutedFfnLowering<'a>>),
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
    /// RoPE inverse frequencies, shared by every rotary layer.
    pub inv_freq: &'a Buffer,
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
                inv_freq: s.inv_freq,
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
}

/// Hidden width a routed layer writes — the router projection's input
/// width (`[num_experts, hidden]`).
fn hidden_of(moe: &MoeLayerWeights<'_>) -> usize {
    moe.router_proj.len() / moe.num_experts.max(1)
}
