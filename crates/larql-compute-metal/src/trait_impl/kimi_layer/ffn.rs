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
        // A full bank tabulates nothing — the kernel multiplies by a
        // stride instead — but Metal still needs a bound buffer, so a
        // one-element placeholder stands in. It is never read.
        let offsets_table = match m.addressing {
            super::ExpertAddressing::Table(t) => self.bufs().uncached_bytes(bytemuck_u32(t)),
            super::ExpertAddressing::Identity { .. } => self.bufs().uncached_bytes(&[0u8; 4]),
        };
        // Resolved into their registered regions, so the buffer the
        // encoder binds IS the one the residency set holds.
        let wts = |v: &[u8]| self.bufs().weights(v);
        let (bank_gate, bank_up, bank_down) = (wts(m.gate), wts(m.up), wts(m.down));

        // Router: logits, then the whole decision in one dispatch that
        // writes the offset table the expert kernel will read.
        self.encode_f32_gemv_into(enc, &rw, &s.post_normed, &s.logits, experts, hidden);
        self.encode_router_select(enc, m, &router_bias, &offsets_table, s, experts);

        let projection = GroupedShape {
            n: inter,
            k: hidden,
            layout: InputLayout::Shared,
        };
        for (bank, out) in [(&bank_gate, &s.gate_out), (&bank_up, &s.up_out)] {
            encode_grouped(
                enc,
                self.default_grouped_handle(),
                GroupedBinding {
                    w: &bank.0,
                    w_offset: bank.1,
                    offsets: &s.offsets,
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
            self.default_grouped_handle(),
            GroupedBinding {
                w: &bank_down.0,
                w_offset: bank_down.1,
                offsets: &s.offsets,
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
        vec![
            rw,
            router_bias,
            offsets_table,
            bank_gate.0,
            bank_up.0,
            bank_down.0,
        ]
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
