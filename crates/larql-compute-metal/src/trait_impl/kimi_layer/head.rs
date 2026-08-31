//! The model head on device: final RMS norm, then the vocabulary
//! projection.
//!
//! It is encoded into the SAME command buffer as the decoder-layer
//! chain, reading the last layer's output buffer directly. That is the
//! whole point: the token already costs exactly one GPU epoch, and a
//! head submitted separately would make it two for a projection that is
//! pure streaming work.
//!
//! The projection is one grouped dispatch of a single slot — the same
//! kernel, offset table and arithmetic as `o_proj`. A `vocab x hidden`
//! bf16 matrix is not structurally different from any other projection
//! in this crate; it is only much larger, which is a grid-size fact
//! rather than a kernel one.

use metal::{Buffer, ComputeCommandEncoderRef};

use super::super::bf16_grouped::{encode_grouped, GroupedBinding, GroupedShape};
use super::super::grouped_experts::{ExpertOffset, GroupedError, InputLayout};
use super::KimiLayerCall;
use crate::MetalBackend;

/// The head's weights. `weight` is `vocab x hidden` row-major, in
/// `encoding` — the same grouped-kernel family every projection in
/// this crate reads, so the dispatch selects its pipeline by encoding
/// exactly as the expert and KDA paths do.
#[derive(Clone, Copy)]
pub struct KimiHead<'a> {
    pub norm_weight: &'a [f32],
    pub norm_eps: f32,
    pub weight: &'a [u8],
    pub vocab: usize,
    pub encoding: super::ExpertEncoding,
}

/// Device scratch for one head evaluation.
pub(crate) struct HeadScratch {
    pub(crate) normed: Buffer,
    pub(crate) logits: Buffer,
}

impl MetalBackend {
    /// The layer chain with the head appended INSIDE the same command
    /// buffer, returning logits instead of the last hidden state.
    ///
    /// The hidden state never crosses to the host: the head reads the
    /// last layer's output buffer, and the single readback is the
    /// vocabulary vector the caller actually wanted.
    pub fn kimi_decoder_layers_with_head(
        &self,
        layers: &[KimiLayerCall<'_>],
        head: &KimiHead<'_>,
        x: &[f32],
        trace: Option<&mut super::ExecutionTrace>,
    ) -> Result<(Vec<f32>, f64), GroupedError> {
        let (scratch, kda, head_scratch, gpu_ms) =
            self.encode_layer_chain(layers, Some(head), x)?;
        super::collect_routes(layers, &scratch, trace);
        let s = head_scratch.expect("a head was requested, so the chain encoded one");
        let logits = crate::buffers::read_buffer_f32(&s.logits, head.vocab);
        self.bufs().recycle(s.normed);
        self.bufs().recycle(s.logits);
        self.recycle_chain(scratch, kda);
        Ok((logits, gpu_ms))
    }

    /// The head ALONE, in its own command buffer.
    ///
    /// Gates and calibration: production runs the head inside the layer
    /// chain's command buffer, where it costs no submission. This exists
    /// so a disagreement can be attributed to the head rather than to
    /// the twenty-six layers in front of it.
    pub fn kimi_head(
        &self,
        head: &KimiHead<'_>,
        x: &[f32],
    ) -> Result<(Vec<f32>, f64), GroupedError> {
        Self::validate_head(head, x.len())?;
        let s = self.kimi_head_scratch(x.len(), head.vocab);
        let buf_x = self.bufs().transient_from_f32(x);
        let cmd = self.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let held = self.encode_kimi_head(enc, head, &buf_x, &s, x.len());
        enc.end_encoding();
        cmd.commit();
        let _ = crate::cb_status::wait_checked(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/kimi_layer/head.rs:kimi_head",
        );
        let gpu_ms = crate::decode::gpu_timing::gpu_elapsed_ms(cmd);
        drop(held);
        let logits = crate::buffers::read_buffer_f32(&s.logits, head.vocab);
        self.bufs().recycle(s.normed);
        self.bufs().recycle(s.logits);
        Ok((logits, gpu_ms))
    }

    /// Shapes checked before anything is encoded, for the same reason
    /// the layer half validates up front: an encoder dropped without
    /// `end_encoding` aborts the process, so a refusal discovered
    /// mid-encode would not be recoverable.
    pub(crate) fn validate_head(head: &KimiHead<'_>, hidden: usize) -> Result<(), GroupedError> {
        if head.vocab == 0 || head.norm_weight.len() != hidden {
            return Err(GroupedError::HeadShapeMismatch {
                vocab: head.vocab,
                hidden,
                have_bytes: head.weight.len(),
            });
        }
        // Bytes at the ENCODING's own stride — the bf16 arithmetic
        // would over-demand on a quantised head and under-demand the
        // other way, exactly the failure the KDA validator refuses.
        let want = head
            .encoding
            .matrix_bytes(head.vocab, hidden)
            .ok_or(GroupedError::KNotSuperblockAligned { k: hidden })?;
        if head.weight.len() != want {
            return Err(GroupedError::HeadShapeMismatch {
                vocab: head.vocab,
                hidden,
                have_bytes: head.weight.len(),
            });
        }
        Ok(())
    }

    pub(crate) fn kimi_head_scratch(&self, hidden: usize, vocab: usize) -> HeadScratch {
        HeadScratch {
            normed: self.bufs().output((hidden * 4) as u64),
            logits: self.bufs().output((vocab * 4) as u64),
        }
    }

    /// Encode norm + projection against `input`, the last layer's own
    /// output buffer. Returns the buffers that must outlive the wait.
    pub(crate) fn encode_kimi_head(
        &self,
        enc: &ComputeCommandEncoderRef,
        head: &KimiHead<'_>,
        input: &Buffer,
        s: &HeadScratch,
        hidden: usize,
    ) -> Vec<Buffer> {
        let norm_w = self.bufs().get_f32(head.norm_weight);
        let (w, w_offset) = self.bufs().weights(head.weight);
        let offsets = self.stable_offset_table(&HEAD_SLOT);
        self.encode_rms_norm(enc, input, &norm_w, &s.normed, hidden, head.norm_eps);
        encode_grouped(
            enc,
            self.grouped_handle_for(head.encoding),
            GroupedBinding {
                w: &w,
                w_offset,
                offsets: &offsets,
                x: &s.normed,
                out: &s.logits,
            },
            1,
            GroupedShape {
                n: head.vocab,
                k: hidden,
                layout: InputLayout::Shared,
            },
        );
        vec![norm_w, w, offsets]
    }
}

/// The head is always one slot at offset zero — there is nothing to
/// select. Named rather than built inline so it shares
/// `stable_offset_table`'s cache with the other constant tables.
pub(crate) static HEAD_SLOT: [ExpertOffset; 1] = [ExpertOffset(0)];
