//! One whole MoE expert FFN in ONE command buffer.
//!
//! Rung 3 of the Kimi Metal ladder, and the only lever left that is
//! worth as much as the kernel itself.
//!
//! **The measurement that earned it.** Rung 2 put every selected
//! expert's projection in one grouped dispatch and took the kernel to
//! 288-372 GB/s — 80-100% of this machine's memory roofline, with a row
//! tiling sweep from 1152 to 9216 threadgroups showing the kernel is not
//! waiting on occupancy. What it left behind was submission: ~0.30 ms of
//! wall per stage against ~0.14 ms of GPU-busy, so **more than half of
//! every stage's wall time is the cost of getting the work onto the
//! GPU**, and a three-stage FFN pays it three times.
//!
//! **The hypothesis this path tests:** collapsing three submissions into
//! one should be a large block-level win with the arithmetic unchanged.
//! GPU-busy time should stay put; wall should fall by roughly two
//! submissions' worth.
//!
//! ```text
//! one command buffer, one encoder
//!     grouped gate GEMV      -> [slots, inter]
//!     grouped up   GEMV      -> [slots, inter]
//!     geglu_silu             -> h = silu(gate) * up
//!     grouped down GEMV      -> [slots, hidden]
//! commit once, wait once, read the output once
//! ```
//!
//! **Gate and up are deliberately NOT fused.** They read the same
//! activation and are consumed together, so fusing them is the obvious
//! next lowering — which is exactly why it has to wait: doing it here
//! would mix "fewer submissions" with "fewer dispatches, one input read,
//! less intermediate traffic" and neither mechanism would be
//! attributable.
//!
//! **Why the activation had to move to the GPU.** It is the one part of
//! the block that was not already a Metal kernel, and a CPU activation
//! would force a commit-and-wait in the middle to read gate and up back
//! — which is the cost this rung exists to remove. It uses the crate's
//! existing `geglu_silu`, whose `(g / (1 + exp(-g))) * up` is the same
//! expression the CPU path evaluates, so this is a relocation rather
//! than a reformulation. Whether `exp` agrees to the last bit between
//! Metal and the host's libm is measured, not assumed — see the
//! whole-block parity gate.
//!
//! **Ordering is the encoder's, not ours.** Dispatches sharing one
//! compute encoder run in issue order with implicit barriers between
//! them, which is the same property that made rung 2's nine batched
//! per-expert dispatches serialise. Here that serialisation is the
//! dependency the block needs: down must not start before the
//! activation, and the activation must not start before both GEMVs.

use metal::{Buffer, ComputeCommandEncoderRef};

use super::bf16_grouped::{encode_grouped, validate_grouped, GroupedBinding, GroupedShape};
use super::grouped_experts::{ExpertOffset, GroupedError, InputLayout};
use crate::kernels::KernelHandle;
use crate::MetalBackend;

/// Threads per threadgroup for the element-wise activation dispatch,
/// matching `stages::ffn`'s own GEGLU dispatch so the two cannot drift.
const ACTIVATION_THREADS_PER_TG: u64 = 256;

/// How the gate and up halves of the block are lowered.
///
/// A knob rather than a decision because it is the instrument for rung
/// 4: the three arms share everything else — one command buffer, one
/// encoder, the same banks, the same down dispatch — so the difference
/// between them is fusion and nothing else. Rung 3's control arm stays
/// permanently available as [`Self::Separate`], which is what keeps the
/// comparison about traffic rather than about command-buffer structure.
///
/// All arms produce **bit-identical** output: each accumulator walks its
/// row identically in every one, and the activation is the same
/// expression whether evaluated in register or through a buffer.
///
/// **Measured: the fused arms are 8-9% SLOWER on GPU-busy** at Kimi's
/// shapes, at both tilings, with intermediates instrumented at 0.173% of
/// bytes moved — so there was no traffic mechanism to exploit. See
/// [`crate::shaders::bf16_grouped_gate_up`] for the table and the
/// attribution. [`Self::Separate`] is the default and should stay there;
/// the fused arms are kept as the evidence and so the question can be
/// re-asked at K3's very different expert shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockLowering {
    /// gate, up, activation, down — four dispatches. Rung 3's shape, and
    /// the control.
    Separate,
    /// fused gate+up, activation, down — three dispatches. One traversal
    /// of the input instead of two.
    FusedGateUp(FusedTiling),
    /// fused gate+up+activation, down — two dispatches. Also drops the
    /// standalone activation and two intermediate streams.
    FusedGateUpAct(FusedTiling),
}

/// Rows per threadgroup for a fused dispatch.
///
/// A fused kernel does in one threadgroup what two unfused ones did
/// between them, so fusing at the same tiling also HALVES the launch —
/// two changes at once. `Rows4` restores the threadgroup count, which is
/// what separates a launch effect from a register-pressure one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedTiling {
    /// Matches the unfused kernel's tiling; half its threadgroup count.
    Rows8,
    /// Half the rows per threadgroup, so the same threadgroup count the
    /// unfused gate and up issue between them.
    Rows4,
}

impl FusedTiling {
    fn index(self) -> usize {
        match self {
            Self::Rows8 => 0,
            Self::Rows4 => 1,
        }
    }
}

/// One projection's weights for every selected slot, and where each
/// slot's payload starts within them.
#[derive(Clone, Copy)]
pub struct ExpertBankRef<'a> {
    /// Every selected expert's payload for this projection, one buffer.
    pub weights: &'a [u8],
    /// Byte offset of each slot's `[n, k]` matrix. Identity lives here,
    /// never in a row's position.
    pub offsets: &'a [ExpertOffset],
}

/// The three projections of one MoE expert FFN, in the checkpoint's own
/// gate/up/down naming — never alphabetic order.
#[derive(Clone, Copy)]
pub struct MoeFfnBanks<'a> {
    /// `w1`, `[inter, hidden]`.
    pub gate: ExpertBankRef<'a>,
    /// `w3`, `[inter, hidden]`.
    pub up: ExpertBankRef<'a>,
    /// `w2`, `[hidden, inter]`.
    pub down: ExpertBankRef<'a>,
    pub hidden: usize,
    pub inter: usize,
}

impl MoeFfnBanks<'_> {
    /// Slot count, refusing banks that disagree about it.
    ///
    /// Three tables of different lengths would not fault — the shorter
    /// projections would simply compute fewer slots and the block would
    /// return a mix of this token's experts and whatever the pooled
    /// output buffer last held.
    fn slots(&self) -> Result<usize, GroupedError> {
        let n = self.gate.offsets.len();
        if n == 0 {
            return Err(GroupedError::NoExpertsSelected);
        }
        for bank in [&self.up, &self.down] {
            if bank.offsets.len() != n {
                return Err(GroupedError::SlotCountMismatch {
                    expected: n,
                    found: bank.offsets.len(),
                });
            }
        }
        Ok(n)
    }
}

impl MetalBackend {
    /// The standalone `silu(gate) * up` dispatch — the crate's existing
    /// `geglu_silu`, which the fused-activation lowering replaces.
    pub(crate) fn encode_geglu_silu(
        &self,
        enc: &ComputeCommandEncoderRef,
        gate: &Buffer,
        up: &Buffer,
        out: &Buffer,
        elems: u32,
    ) {
        enc.set_compute_pipeline_state(&self.ffn.geglu_pipeline);
        enc.set_buffer(0, Some(gate), 0);
        enc.set_buffer(1, Some(up), 0);
        enc.set_buffer(2, Some(out), 0);
        enc.set_bytes(3, 4, &elems as *const u32 as *const std::ffi::c_void);
        enc.dispatch_threads(
            metal::MTLSize::new(elems as u64, 1, 1),
            metal::MTLSize::new(ACTIVATION_THREADS_PER_TG, 1, 1),
        );
    }

    /// `down(silu(gate(x)) * up(x))` for every selected expert, as ONE
    /// submission.
    ///
    /// Returns `[slots, hidden]` row-major unweighted per-expert
    /// outputs. Unweighted because combining with routing weights is
    /// `slots x hidden` floats against `3 x slots x hidden x inter`
    /// weight bytes read — free by comparison, and leaving it outside
    /// keeps the block verifiable against a per-stage path.
    pub fn bf16_moe_ffn_block(
        &self,
        banks: MoeFfnBanks<'_>,
        x: &[f32],
    ) -> Result<(Vec<f32>, f64), GroupedError> {
        self.bf16_moe_ffn_block_lowered(banks, x, BlockLowering::Separate)
    }

    /// The same block at a chosen lowering. See [`BlockLowering`].
    pub fn bf16_moe_ffn_block_lowered(
        &self,
        banks: MoeFfnBanks<'_>,
        x: &[f32],
        lowering: BlockLowering,
    ) -> Result<(Vec<f32>, f64), GroupedError> {
        let (mut outs, gpu_ms) =
            self.bf16_moe_ffn_blocks(&[MoeBlockCall { banks, x }], lowering)?;
        Ok((outs.remove(0), gpu_ms))
    }

    /// **Several MoE blocks in ONE command buffer.**
    ///
    /// Rung 5a's instrument. Rung 3 collapsed a block's three
    /// submissions into one and took 0.38 ms of wall off it; what
    /// remained was ~40% of block wall in host time — one submission
    /// floor plus readback, paid once per block. This asks what is left
    /// when that is paid once per N blocks instead.
    ///
    /// **This is a ceiling, not a production shape.** Every block's
    /// input and selected experts have to be known before the command
    /// buffer is built. A real decoder cannot do that yet: the next
    /// layer's hidden state comes out of KDA/MLA/residual/norm work that
    /// still runs on the host, and its routing decision depends on that
    /// state. So this measures how much performance is waiting for the
    /// operators that would let execution stay on-device across layers —
    /// it does not deliver it.
    ///
    /// Scratch (gate, up, activation) is shared across every block: the
    /// encoder runs dispatches in order with barriers between them, so
    /// block `i+1`'s gate write cannot race block `i`'s down read. That
    /// safety is asserted, not assumed — the multi-block outputs must
    /// equal the same blocks run one per command buffer.
    pub fn bf16_moe_ffn_blocks(
        &self,
        blocks: &[MoeBlockCall<'_>],
        lowering: BlockLowering,
    ) -> Result<(Vec<Vec<f32>>, f64), GroupedError> {
        if blocks.is_empty() {
            return Err(GroupedError::NoExpertsSelected);
        }

        // Validate every block before encoding anything: an encoder
        // dropped without `end_encoding` aborts the process, so a
        // refusal discovered halfway through would not be an error the
        // caller could handle.
        let mut plans = Vec::with_capacity(blocks.len());
        for call in blocks {
            plans.push(call.validate()?);
        }

        let inter = blocks[0].banks.inter;
        let max_slots = plans.iter().map(|p| p.slots).max().expect("non-empty");
        // One scratch set for the whole batch, sized for the widest
        // block. Three distinct pops so gate, up and the activation
        // cannot alias even though they are the same size.
        let scratch_bytes = (max_slots * inter * 4) as u64;
        let buf_gate = self.bufs.output(scratch_bytes);
        let buf_up = self.bufs.output(scratch_bytes);
        let buf_h = self.bufs.output(scratch_bytes);

        let handle = self.default_grouped_handle();
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        let mut outs = Vec::with_capacity(blocks.len());
        // Weight and offset buffers must outlive the wait; holding the
        // cache clones here guarantees it independent of encoder
        // retention semantics.
        let mut held = Vec::with_capacity(blocks.len());

        for (call, plan) in blocks.iter().zip(&plans) {
            let b = call.banks;
            let (w_gate, o_gate) = self.bufs.weights(b.gate.weights);
            let (w_up, o_up) = self.bufs.weights(b.up.weights);
            let (w_down, o_down) = self.bufs.weights(b.down.weights);
            let off_gate = self.offset_table(b.gate.offsets);
            let off_up = self.offset_table(b.up.offsets);
            let off_down = self.offset_table(b.down.offsets);
            let buf_x = self.bufs.transient_from_f32(&call.x[..b.hidden]);
            let buf_out = self.bufs.output((plan.slots * b.hidden * 4) as u64);
            let elems = (plan.slots * b.inter) as u32;

            match lowering {
                BlockLowering::Separate => {
                    encode_grouped(
                        enc,
                        handle,
                        GroupedBinding {
                            w: &w_gate,
                            w_offset: o_gate,
                            offsets: &off_gate,
                            x: &buf_x,
                            out: &buf_gate,
                        },
                        plan.slots,
                        plan.projection,
                    );
                    encode_grouped(
                        enc,
                        handle,
                        GroupedBinding {
                            w: &w_up,
                            w_offset: o_up,
                            offsets: &off_up,
                            x: &buf_x,
                            out: &buf_up,
                        },
                        plan.slots,
                        plan.projection,
                    );
                    self.encode_geglu_silu(enc, &buf_gate, &buf_up, &buf_h, elems);
                }
                BlockLowering::FusedGateUp(tiling) => {
                    encode_gate_up(
                        enc,
                        &self.bf16_gate_up_variants[tiling.index()],
                        GateUpBinding {
                            w_gate: &w_gate,
                            off_gate: &off_gate,
                            w_up: &w_up,
                            off_up: &off_up,
                            x: &buf_x,
                            out_a: &buf_gate,
                            out_b: &buf_up,
                        },
                        plan.slots,
                        plan.projection,
                    );
                    self.encode_geglu_silu(enc, &buf_gate, &buf_up, &buf_h, elems);
                }
                BlockLowering::FusedGateUpAct(tiling) => {
                    // `out_b` is bound but never written — the kernel
                    // keeps the binding so both variants share one
                    // argument layout. Bound to a real buffer rather
                    // than left unset because an unbound argument is
                    // undefined behaviour even for a kernel that does
                    // not touch it.
                    encode_gate_up(
                        enc,
                        &self.bf16_gate_up_silu_variants[tiling.index()],
                        GateUpBinding {
                            w_gate: &w_gate,
                            off_gate: &off_gate,
                            w_up: &w_up,
                            off_up: &off_up,
                            x: &buf_x,
                            out_a: &buf_h,
                            out_b: &buf_up,
                        },
                        plan.slots,
                        plan.projection,
                    );
                }
            }
            encode_grouped(
                enc,
                handle,
                GroupedBinding {
                    w: &w_down,
                    w_offset: o_down,
                    offsets: &off_down,
                    x: &buf_h,
                    out: &buf_out,
                },
                plan.slots,
                plan.down,
            );
            outs.push((buf_out, plan.slots * b.hidden));
            held.push((w_gate, w_up, w_down, off_gate, off_up, off_down, buf_x));
        }
        enc.end_encoding();
        cmd.commit();
        crate::cb_status::wait_checked(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/bf16_moe_block/mod.rs:blocks",
        )
        .map_err(|detail| GroupedError::CommandBufferFailed {
            site: "crates/larql-compute-metal/src/trait_impl/bf16_moe_block/mod.rs:blocks",
            detail,
        })?;

        let gpu_ms = crate::decode::gpu_timing::gpu_elapsed_ms(cmd);
        let results: Vec<Vec<f32>> = outs
            .iter()
            .map(|(buf, len)| crate::buffers::read_buffer_f32(buf, *len))
            .collect();
        for (buf, _) in outs {
            self.bufs.recycle(buf);
        }
        for b in [buf_gate, buf_up, buf_h] {
            self.bufs.recycle(b);
        }
        drop(held);
        Ok((results, gpu_ms))
    }
}

/// One block's work: which experts, and the activation they consume.
#[derive(Clone, Copy)]
pub struct MoeBlockCall<'a> {
    pub banks: MoeFfnBanks<'a>,
    pub x: &'a [f32],
}

/// One validated block, so the encode loop cannot re-derive a shape
/// differently from the one that was checked.
struct BlockPlan {
    slots: usize,
    projection: GroupedShape,
    down: GroupedShape,
}

impl MoeBlockCall<'_> {
    fn validate(&self) -> Result<BlockPlan, GroupedError> {
        let slots = self.banks.slots()?;
        let (hidden, inter) = (self.banks.hidden, self.banks.inter);
        let projection = GroupedShape {
            n: inter,
            k: hidden,
            layout: InputLayout::Shared,
        };
        let down = GroupedShape {
            n: hidden,
            k: inter,
            layout: InputLayout::PerSlot,
        };
        validate_grouped(
            self.banks.gate.weights,
            self.banks.gate.offsets,
            self.x,
            projection,
        )?;
        validate_grouped(
            self.banks.up.weights,
            self.banks.up.offsets,
            self.x,
            projection,
        )?;
        let staged = vec![0.0f32; slots * inter];
        validate_grouped(
            self.banks.down.weights,
            self.banks.down.offsets,
            &staged,
            down,
        )?;
        Ok(BlockPlan {
            slots,
            projection,
            down,
        })
    }
}

/// The seven device buffers a fused gate+up dispatch binds.
pub(crate) struct GateUpBinding<'a> {
    pub w_gate: &'a Buffer,
    pub off_gate: &'a Buffer,
    pub w_up: &'a Buffer,
    pub off_up: &'a Buffer,
    pub x: &'a Buffer,
    /// `gate` for the plain fused kernel, `h` for the activated one.
    pub out_a: &'a Buffer,
    /// `up` for the plain fused kernel; bound but unwritten for the
    /// activated one.
    pub out_b: &'a Buffer,
}

/// Encode one fused gate+up dispatch into an existing encoder.
///
/// Same 2-D `(row_tiles, slots)` grid as the unfused grouped kernel and
/// the same tiling, so a fused-vs-separate comparison differs in fusion
/// alone. Geometry is read from the bound handle, never hardcoded.
pub(crate) fn encode_gate_up(
    enc: &ComputeCommandEncoderRef,
    handle: &KernelHandle,
    b: GateUpBinding<'_>,
    slots: usize,
    shape: GroupedShape,
) {
    let n_u32 = shape.n as u32;
    let k_u32 = shape.k as u32;
    let row_tiles = (shape.n as u64).div_ceil(handle.rows_per_tg);

    enc.set_compute_pipeline_state(&handle.state);
    enc.set_buffer(0, Some(b.w_gate), 0);
    enc.set_buffer(1, Some(b.off_gate), 0);
    enc.set_buffer(2, Some(b.w_up), 0);
    enc.set_buffer(3, Some(b.off_up), 0);
    enc.set_buffer(4, Some(b.x), 0);
    enc.set_buffer(5, Some(b.out_a), 0);
    enc.set_buffer(6, Some(b.out_b), 0);
    enc.set_bytes(7, 4, &n_u32 as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &k_u32 as *const u32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(row_tiles, slots as u64, 1),
        metal::MTLSize::new(handle.threads_per_tg, 1, 1),
    );
}

#[cfg(test)]
mod tests;
