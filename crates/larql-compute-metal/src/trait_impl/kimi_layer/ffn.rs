//! Which feed-forward half a layer runs.
//!
//! The same move R6c made for attention: Kimi's stack is one dense
//! layer followed by twenty-six routed ones, and that is a *parameter*
//! of the decoder layer rather than a second layer type. Everything
//! around it — both norms, the attention, both residuals — is written
//! once and does not know which it got.
//!
//! A dense MLP is the routed path with the router deleted: one slot,
//! offset zero, combine weight 1.0. It reuses the grouped kernel, the
//! GEGLU and `moe_combine` unchanged, so the arithmetic on the dense
//! layer is the arithmetic the twenty-six routed layers already prove.

use metal::{Buffer, ComputeCommandEncoderRef};

use super::super::bf16_grouped::{
    encode_grouped, encode_grouped_windowed, GroupedBinding, GroupedShape, SlotWindow,
};
use super::super::grouped_experts::{ExpertOffset, GroupedError, InputLayout};
use super::{bytemuck_u32, EncodedRegion, KimiMoeWeights, LayerScratch};
use crate::MetalBackend;

/// Bytes per f32 in the activation planes the slot windows index.
const F32_BYTES: usize = 4;

/// A plain gated MLP: `down(silu(gate(x)) * up(x))`.
#[derive(Clone, Copy)]
pub struct KimiDenseFfn<'a> {
    /// `[inter, hidden]` bf16.
    pub gate: &'a [u8],
    /// `[inter, hidden]` bf16.
    pub up: &'a [u8],
    /// `[hidden, inter]` bf16.
    pub down: &'a [u8],
    pub inter: usize,
}

/// The feed-forward half of a layer.
///
/// `Moe` is much larger than `Dense` — three projections each carrying
/// a routed region, addressing and an optional shared region — but the
/// whole spec is a borrowed, `Copy` descriptor rebuilt per layer call
/// on the decode hot path, so boxing the variant would trade a few
/// stack bytes for a per-layer heap allocation every token.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy)]
pub enum FfnSpec<'a> {
    Moe(KimiMoeWeights<'a>),
    Dense(KimiDenseFfn<'a>),
}

impl<'a> FfnSpec<'a> {
    /// Intermediate width. The dense layer's is its own — Kimi's
    /// `dense_intermediate_size` is four times the routed experts'.
    pub fn inter(&self) -> usize {
        match self {
            Self::Moe(m) => m.inter,
            Self::Dense(d) => d.inter,
        }
    }

    /// How many expert slots the FFN evaluates: `top_k` routed plus the
    /// shared branch when the layer carries one, or exactly one for a
    /// dense MLP.
    ///
    /// Read from `gate`; validation refuses a layer whose projections
    /// disagree about the branch's existence before this number is used
    /// for anything.
    pub fn slots(&self) -> usize {
        match self {
            Self::Moe(m) => m.top_k + usize::from(m.gate.shared.is_some()),
            Self::Dense(_) => 1,
        }
    }

    /// Router width. Zero for a dense layer, which has no router at all
    /// — not a router over one expert.
    pub fn experts(&self) -> usize {
        match self {
            Self::Moe(m) => m.router_bias.len(),
            Self::Dense(_) => 0,
        }
    }

    pub fn top_k(&self) -> usize {
        match self {
            Self::Moe(m) => m.top_k,
            Self::Dense(_) => 0,
        }
    }

    /// The MoE weights, when this is a routed layer.
    pub fn moe(&self) -> Option<&KimiMoeWeights<'a>> {
        match self {
            Self::Moe(m) => Some(m),
            Self::Dense(_) => None,
        }
    }

    /// Bank sizes checked host-side, since the kernel cannot.
    pub(crate) fn validate_dense(d: &KimiDenseFfn<'_>, hidden: usize) -> Result<(), GroupedError> {
        if d.inter == 0 {
            return Err(GroupedError::NoExpertsSelected);
        }
        let per = d.inter * hidden * 2;
        for (which, bank) in [(0usize, d.gate), (1, d.up), (2, d.down)] {
            if bank.len() != per {
                return Err(GroupedError::OffsetOutOfRange {
                    slot: which,
                    offset: 0,
                    need: per,
                    have: bank.len(),
                });
            }
        }
        Ok(())
    }
}

/// The dense FFN is always one slot at offset zero, and combines with
/// weight one — the shared branch's own rule, which `moe_combine`
/// already applies unscaled.
pub(crate) static DENSE_SLOT: [ExpertOffset; 1] = [ExpertOffset(0)];
pub(crate) static DENSE_COMBINE_WEIGHT: [f32; 1] = [1.0];

impl MetalBackend {
    /// The routed half: router, grouped experts, GEGLU, combine.
    ///
    /// Returns the buffers it bound, which the caller holds until the
    /// wait.
    pub(crate) fn encode_moe_ffn(
        &self,
        enc: &ComputeCommandEncoderRef,
        m: &KimiMoeWeights<'_>,
        s: &LayerScratch,
        hidden: usize,
    ) -> Vec<Buffer> {
        let f32b = |v: &[f32]| self.bufs().get_f32(v);
        let (rw, router_bias) = (f32b(m.router_weight), f32b(m.router_bias));
        let (experts, inter) = (m.router_bias.len(), m.inter);
        let has_shared = m.gate.shared.is_some();
        let slots = m.top_k + usize::from(has_shared);
        // `uncached_bytes`, NOT `get_bytes`, and the difference is not a
        // style choice.
        //
        // The residency map looks cacheable — it is a property of the
        // layer, not of the route. But the cache keys on `(ptr, len)`,
        // and a caller that builds a table into a temporary `Vec` gets
        // whatever a previous `Vec` at that address uploaded. That is
        // exactly what happened: a control that swapped one layer's
        // offsets, ran, dropped the table, then swapped ANOTHER layer's
        // into a fresh `Vec` at the same address, silently ran the
        // second layer against the FIRST layer's residency and was
        // refused for routing outside a bank it did hold.
        //
        // Caching it was measured at ~0.7 ms a token out of 84 — noise
        // against a hazard that produces a plausible refusal, or worse a
        // plausible answer, from another layer's map.
        // Resolved into their registered regions, so the buffer the
        // encoder binds IS the one the residency set holds.
        let wts = |v: &[u8]| self.bufs().weights(v);
        let (bank_gate, bank_up, bank_down) = (
            wts(m.gate.routed.bytes),
            wts(m.up.routed.bytes),
            wts(m.down.routed.bytes),
        );

        // Router: logits, then the decision. It writes WHICH expert and
        // nothing about where any bytes are.
        self.encode_f32_gemv_into(enc, &rw, &s.post_normed, &s.logits, experts, hidden);
        self.encode_router_select(enc, m, &router_bias, s, experts);

        // Then each projection resolves its OWN address for that logical
        // expert. Three dispatches of `top_k` threads: the price of
        // having no shared coordinate anywhere in the model. The shared
        // branch is absent here by construction — it is not routed, so
        // it owns no entry in any address table.
        let mut held = vec![rw, router_bias];
        for (bank, offsets) in [
            (&m.gate, &s.gate_offsets),
            (&m.up, &s.up_offsets),
            (&m.down, &s.down_offsets),
        ] {
            held.push(self.encode_expert_addresses(enc, bank, s, offsets, experts, m.top_k));
        }

        let projection = GroupedShape {
            n: inter,
            k: hidden,
            layout: InputLayout::Shared,
        };
        for (bank, spec, offsets, out) in [
            (&bank_gate, &m.gate, &s.gate_offsets, &s.gate_out),
            (&bank_up, &m.up, &s.up_offsets, &s.up_out),
        ] {
            encode_grouped(
                enc,
                // Each projection picks its OWN kernel from its OWN
                // declared encoding. Nothing here consults "the bank's"
                // format, because there is no such thing.
                self.grouped_handle_for(spec.routed.encoding),
                GroupedBinding {
                    w: &bank.0,
                    w_offset: bank.1,
                    offsets,
                    x: &s.post_normed,
                    out,
                },
                m.top_k,
                projection,
            );
        }
        // The shared branch: its own region, its own single-slot
        // dispatch, landing in slot `top_k` of the same activation
        // planes. Sharing the routed dispatch would require the bytes to
        // be co-located with the routed bank, and the real container
        // disproves that assumption — co-dispatch is a layout
        // optimisation an artifact may earn, never an invariant.
        let shared_table = has_shared.then(|| self.stable_offset_table(&DENSE_SLOT));
        if let Some(table) = &shared_table {
            let out_row = (m.top_k * inter * F32_BYTES) as u64;
            for (region, out) in [(&m.gate.shared, &s.gate_out), (&m.up.shared, &s.up_out)] {
                let region = region.as_ref().expect("validated consistent");
                held.push(self.encode_shared_projection(
                    enc,
                    region,
                    table,
                    &s.post_normed,
                    out,
                    projection,
                    SlotWindow {
                        x_bytes: 0,
                        out_bytes: out_row,
                    },
                ));
            }
        }
        self.encode_geglu_silu(enc, &s.gate_out, &s.up_out, &s.h, (slots * inter) as u32);
        let down_shape = GroupedShape {
            n: hidden,
            k: inter,
            layout: InputLayout::PerSlot,
        };
        encode_grouped(
            enc,
            self.grouped_handle_for(m.down.routed.encoding),
            GroupedBinding {
                w: &bank_down.0,
                w_offset: bank_down.1,
                offsets: &s.down_offsets,
                x: &s.h,
                out: &s.expert_out,
            },
            m.top_k,
            down_shape,
        );
        if let Some(table) = &shared_table {
            let region = m.down.shared.as_ref().expect("validated consistent");
            held.push(self.encode_shared_projection(
                enc,
                region,
                table,
                &s.h,
                &s.expert_out,
                down_shape,
                SlotWindow {
                    x_bytes: (m.top_k * inter * F32_BYTES) as u64,
                    out_bytes: (m.top_k * hidden * F32_BYTES) as u64,
                },
            ));
        }
        self.encode_moe_combine(enc, s, &s.weights, hidden, slots);
        held.extend([bank_gate.0, bank_up.0, bank_down.0]);
        held.extend(shared_table);
        held
    }

    /// One projection's shared-branch dispatch: one slot, weights from
    /// the branch's OWN region under its own encoding, output windowed
    /// into slot `top_k` of the caller's plane.
    ///
    /// Returns the bank buffer it bound, which the caller holds until
    /// the wait.
    #[allow(clippy::too_many_arguments)]
    fn encode_shared_projection(
        &self,
        enc: &ComputeCommandEncoderRef,
        region: &EncodedRegion<'_>,
        offsets: &Buffer,
        x: &Buffer,
        out: &Buffer,
        shape: GroupedShape,
        window: SlotWindow,
    ) -> Buffer {
        let bank = self.bufs().weights(region.bytes);
        encode_grouped_windowed(
            enc,
            self.grouped_handle_for(region.encoding),
            GroupedBinding {
                w: &bank.0,
                w_offset: bank.1,
                offsets,
                x,
                out,
            },
            1,
            shape,
            window,
        );
        bank.0
    }

    /// One grouped dispatch under a NAMED encoding, end to end.
    ///
    /// The encoding-aware sibling of `bf16_grouped_experts`: bounds are
    /// checked against the encoding's OWN per-expert stride (a Q8_0 bank
    /// is smaller than the BF16 arithmetic it stands in for, so the BF16
    /// validator would demand bytes that rightly do not exist), and the
    /// handle comes from `grouped_handle_for`, so this cannot pair bytes
    /// with another encoding's kernel. This is the direct gate for a
    /// grouped kernel's decode: quantise a bank, dispatch, compare with
    /// the CPU dequantised reference.
    pub fn grouped_experts_encoded(
        &self,
        encoding: super::ExpertEncoding,
        weights: &[u8],
        offsets: &[ExpertOffset],
        x: &[f32],
        shape: GroupedShape,
    ) -> Result<Vec<f32>, GroupedError> {
        if offsets.is_empty() {
            return Err(GroupedError::NoExpertsSelected);
        }
        let per_expert = encoding
            .matrix_bytes(shape.n, shape.k)
            .ok_or(GroupedError::KNotSuperblockAligned { k: shape.k })?;
        for (slot, off) in offsets.iter().enumerate() {
            let need = off.0 as usize + per_expert;
            if need > weights.len() {
                return Err(GroupedError::OffsetOutOfRange {
                    slot,
                    offset: off.0,
                    need,
                    have: weights.len(),
                });
            }
        }
        let x_needed = match shape.layout {
            InputLayout::Shared => shape.k,
            InputLayout::PerSlot => shape.k * offsets.len(),
        };
        if x.len() < x_needed {
            return Err(GroupedError::OffsetOutOfRange {
                slot: 0,
                offset: 0,
                need: x_needed,
                have: x.len(),
            });
        }

        let (buf_w, w_offset) = self.bufs().weights(weights);
        let buf_o = self.offset_table(offsets);
        let buf_x = self.bufs().transient_from_f32(&x[..x_needed]);
        let buf_out = self.bufs().output((offsets.len() * shape.n * 4) as u64);

        let cmd = self.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        encode_grouped(
            enc,
            self.grouped_handle_for(encoding),
            GroupedBinding {
                w: &buf_w,
                w_offset,
                offsets: &buf_o,
                x: &buf_x,
                out: &buf_out,
            },
            offsets.len(),
            shape,
        );
        enc.end_encoding();
        cmd.commit();
        let _ = crate::cb_status::wait_checked(
            cmd,
            "crates/larql-compute-metal/src/trait_impl/kimi_layer/ffn.rs:encoded",
        );
        Ok(crate::buffers::read_buffer_f32(
            &buf_out,
            offsets.len() * shape.n,
        ))
    }

    /// The grouped kernel that reads this encoding.
    ///
    /// All three share one binding ABI, so this is a handle swap and
    /// never a different lowering.
    fn grouped_handle_for(&self, encoding: super::ExpertEncoding) -> &crate::kernels::KernelHandle {
        match encoding {
            super::ExpertEncoding::Bf16 => self.default_grouped_handle(),
            super::ExpertEncoding::Q80 => &self.quant.q8_0_grouped_experts_pipeline,
            super::ExpertEncoding::Q6K => &self.quant.q6k_grouped_experts_pipeline,
            super::ExpertEncoding::Q4K => &self.quant.q4k_grouped_experts_pipeline,
        }
    }

    /// Resolve every selected logical expert to a byte address IN ONE
    /// PROJECTION's routed bank.
    ///
    /// Routed slots only: the shared branch is not routed, so it has no
    /// logical id to resolve and no entry here — its region is bound
    /// directly by its own dispatch.
    ///
    /// Returns the offset table it bound, which the caller holds until
    /// the wait. A bank that addresses by identity tabulates nothing,
    /// but Metal still needs a bound buffer, so a one-element
    /// placeholder stands in and is never read.
    fn encode_expert_addresses(
        &self,
        enc: &ComputeCommandEncoderRef,
        bank: &super::ProjectionBank<'_>,
        s: &LayerScratch,
        offsets: &Buffer,
        experts: usize,
        top_k: usize,
    ) -> Buffer {
        let table = match bank.addressing {
            super::ExpertAddressing::Table(t) => self.bufs().uncached_bytes(bytemuck_u32(t)),
            super::ExpertAddressing::Identity { .. } => self.bufs().uncached_bytes(&[0u8; 4]),
        };
        let (k, stride) = (top_k as u32, bank.addressing.identity_stride());
        let e = experts as u32;
        enc.set_compute_pipeline_state(&self.kimi.expert_addresses);
        enc.set_buffer(0, Some(&s.chosen), 0);
        enc.set_buffer(1, Some(&table), 0);
        enc.set_buffer(2, Some(offsets), 0);
        enc.set_buffer(3, Some(&s.refusals), 0);
        enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &stride as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &e as *const u32 as *const std::ffi::c_void);
        crate::lowering::dispatch_linear(enc, &self.kimi.expert_addresses, top_k);
        table
    }

    /// The dense half: the SAME grouped kernel, GEGLU and combine, with
    /// one slot at offset zero and a constant combine weight.
    ///
    /// No router runs — a dense layer does not have one, and giving it a
    /// one-expert router would spend a dispatch to rediscover a constant.
    pub(crate) fn encode_dense_ffn(
        &self,
        enc: &ComputeCommandEncoderRef,
        d: &KimiDenseFfn<'_>,
        s: &LayerScratch,
        hidden: usize,
    ) -> Vec<Buffer> {
        let wts = |v: &[u8]| self.bufs().weights(v);
        let (gate, up, down) = (wts(d.gate), wts(d.up), wts(d.down));
        let offsets = self.stable_offset_table(&DENSE_SLOT);
        let combine = self.bufs().get_f32(&DENSE_COMBINE_WEIGHT);
        let projection = GroupedShape {
            n: d.inter,
            k: hidden,
            layout: InputLayout::Shared,
        };
        for (bank, out) in [(&gate, &s.gate_out), (&up, &s.up_out)] {
            encode_grouped(
                enc,
                self.default_grouped_handle(),
                GroupedBinding {
                    w: &bank.0,
                    w_offset: bank.1,
                    offsets: &offsets,
                    x: &s.post_normed,
                    out,
                },
                1,
                projection,
            );
        }
        self.encode_geglu_silu(enc, &s.gate_out, &s.up_out, &s.h, d.inter as u32);
        encode_grouped(
            enc,
            self.default_grouped_handle(),
            GroupedBinding {
                w: &down.0,
                w_offset: down.1,
                offsets: &offsets,
                x: &s.h,
                out: &s.expert_out,
            },
            1,
            GroupedShape {
                n: hidden,
                k: d.inter,
                layout: InputLayout::PerSlot,
            },
        );
        self.encode_moe_combine(enc, s, &combine, hidden, 1);
        vec![gate.0, up.0, down.0, offsets, combine]
    }
}
