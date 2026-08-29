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

use super::super::bf16_grouped::{encode_grouped, GroupedBinding, GroupedShape};
use super::super::grouped_experts::{ExpertOffset, GroupedError, InputLayout};
use super::{bytemuck_u32, KimiMoeWeights, LayerScratch};
use crate::MetalBackend;

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
    /// shared branch, or exactly one for a dense MLP.
    pub fn slots(&self) -> usize {
        match self {
            Self::Moe(m) => m.top_k + 1,
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
        let (experts, slots, inter) = (m.router_bias.len(), m.top_k + 1, m.inter);
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
        let (bank_gate, bank_up, bank_down) =
            (wts(m.gate.bytes), wts(m.up.bytes), wts(m.down.bytes));

        // Router: logits, then the decision. It writes WHICH expert and
        // nothing about where any bytes are.
        self.encode_f32_gemv_into(enc, &rw, &s.post_normed, &s.logits, experts, hidden);
        self.encode_router_select(enc, m, &router_bias, s, experts);

        // Then each projection resolves its OWN address for that logical
        // expert. Three dispatches of `top_k + 1` threads: the price of
        // having no shared coordinate anywhere in the model.
        let mut tables = Vec::with_capacity(3);
        for (bank, offsets) in [
            (&m.gate, &s.gate_offsets),
            (&m.up, &s.up_offsets),
            (&m.down, &s.down_offsets),
        ] {
            tables.push(self.encode_expert_addresses(enc, bank, s, offsets, experts, m.top_k));
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
                self.grouped_handle_for(spec.encoding),
                GroupedBinding {
                    w: &bank.0,
                    w_offset: bank.1,
                    offsets,
                    x: &s.post_normed,
                    out,
                },
                slots,
                projection,
            );
        }
        self.encode_geglu_silu(enc, &s.gate_out, &s.up_out, &s.h, (slots * inter) as u32);
        encode_grouped(
            enc,
            self.grouped_handle_for(m.down.encoding),
            GroupedBinding {
                w: &bank_down.0,
                w_offset: bank_down.1,
                offsets: &s.down_offsets,
                x: &s.h,
                out: &s.expert_out,
            },
            slots,
            GroupedShape {
                n: hidden,
                k: inter,
                layout: InputLayout::PerSlot,
            },
        );
        self.encode_moe_combine(enc, s, &s.weights, hidden, slots);
        let mut held = vec![rw, router_bias, bank_gate.0, bank_up.0, bank_down.0];
        held.extend(tables);
        held
    }

    /// The grouped kernel that reads this encoding.
    ///
    /// All three share one binding ABI, so this is a handle swap and
    /// never a different lowering.
    fn grouped_handle_for(&self, encoding: super::ExpertEncoding) -> &crate::kernels::KernelHandle {
        match encoding {
            super::ExpertEncoding::Bf16 => self.default_grouped_handle(),
            super::ExpertEncoding::Q6K => &self.quant.q6k_grouped_experts_pipeline,
            super::ExpertEncoding::Q4K => &self.quant.q4k_grouped_experts_pipeline,
        }
    }

    /// Resolve every selected logical expert to a byte address IN ONE
    /// PROJECTION's bank.
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
        let (shared, e) = (bank.shared_offset, experts as u32);
        enc.set_compute_pipeline_state(&self.kimi.expert_addresses);
        enc.set_buffer(0, Some(&s.chosen), 0);
        enc.set_buffer(1, Some(&table), 0);
        enc.set_buffer(2, Some(offsets), 0);
        enc.set_buffer(3, Some(&s.refusals), 0);
        enc.set_bytes(4, 4, &k as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &stride as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(6, 4, &shared as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(7, 4, &e as *const u32 as *const std::ffi::c_void);
        crate::lowering::dispatch_linear(enc, &self.kimi.expert_addresses, top_k + 1);
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
