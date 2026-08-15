//! BW10 byte accounting for one routed MoE expert block.
//!
//! ONE authority, consumed by both the legacy CPU-routed encode and the
//! descriptor-driven GPU-route encode. That is not tidiness: the S2
//! calibration arm runs those two paths against the same weights, so if
//! each counted its own bytes a disagreement would be indistinguishable
//! from a real byte delta. Sharing the computation makes "both arms move
//! identical bytes" a checkable property of the instrument.
//!
//! # Semantic versus physical, precisely
//!
//! The stored row width is block-padded by the writer: GPT-OSS's 2880
//! hidden becomes 3072 stored columns under Q6_K, while native MXFP4
//! stores 2880 unpadded. `row_bytes` is derived from the PADDED width, so
//!
//! ```text
//! semantic_gate_up = physical_gate_up × hidden / weight_cols
//! semantic_down    = physical_down    × inter  / inter_padded
//! ```
//!
//! Representation choice (Q6_K versus MXFP4) is therefore a change in
//! PHYSICAL bytes and a legitimate saving; padding is amplification. The
//! two must not be conflated, or MXFP4 would read as "less amplified"
//! when what it actually did was store fewer bytes per weight.

use larql_compute::exec_policy::{self, ExecutionStrategy};
use larql_compute::movement_ledger::{OperandMovement, Tier};
use larql_models::quant::mxfp4::MXFP4_GROUP_ELEMS;

use crate::moe_dispatch::MoeScratch;

/// Bytes of e8m0 exponent per group, for split-scale MXFP4 banks. Q6_K
/// and inline-scale formats carry their scales inside the block, so
/// `bytes_per_block` already accounts for them and this term is unused.
const E8M0_BYTES_PER_GROUP: usize = 1;

/// The shape terms one expert block's byte accounting needs. Extracted
/// from `MoeScratch` so the arithmetic is a pure function and can be
/// tested without a Metal device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ExpertLayerShape {
    /// Expert slots actually dispatched this layer.
    pub n_slots: usize,
    pub inter: usize,
    pub inter_padded: usize,
    pub hidden: usize,
    /// Stored gate/up row width in elements — block-padded.
    pub weight_cols: usize,
    pub row_bytes: usize,
    pub down_row_bytes: usize,
    /// Scales live in their own streams (native MXFP4) rather than inside
    /// the quant block.
    pub split_scales: bool,
}

impl ExpertLayerShape {
    pub(crate) fn from_scratch(scratch: &MoeScratch, n_slots: usize, split_scales: bool) -> Self {
        Self {
            n_slots,
            inter: scratch.inter,
            inter_padded: scratch.inter_padded,
            hidden: scratch.hidden,
            weight_cols: scratch.weight_cols,
            row_bytes: scratch.row_bytes,
            down_row_bytes: scratch.down_row_bytes,
            split_scales,
        }
    }

    /// Physical payload bytes for one expert's fused gate+up slice.
    fn gate_up_payload(&self) -> usize {
        2 * self.inter * self.row_bytes
    }

    /// Physical payload bytes for one expert's down slice.
    fn down_payload(&self) -> usize {
        self.hidden * self.down_row_bytes
    }

    /// Split e8m0 stream bytes for one expert, zero under inline scales.
    fn scale_bytes(&self) -> usize {
        if !self.split_scales {
            return 0;
        }
        let gate_up_groups = 2 * self.inter * self.weight_cols / MXFP4_GROUP_ELEMS;
        let down_groups = self.hidden * self.inter_padded / MXFP4_GROUP_ELEMS;
        (gate_up_groups + down_groups) * E8M0_BYTES_PER_GROUP
    }

    /// Ratio of semantic to stored extent on the gate/up K axis.
    fn gate_up_semantic_num_den(&self) -> (usize, usize) {
        (self.hidden, self.weight_cols.max(1))
    }

    /// Ratio of semantic to stored extent on the down K axis.
    fn down_semantic_num_den(&self) -> (usize, usize) {
        (self.inter, self.inter_padded.max(1))
    }

    /// This layer's expert-weight movement for one token.
    ///
    /// Every physical byte bound is streamed by the grouped kernel — the
    /// dispatch walks whole rows — so this is a fully-consumed read. A
    /// future sub-expert or gathered arm must switch to
    /// `partially_consumed` and supply a defensible useful count; that is
    /// exactly the distinction BW-B exists to measure.
    pub(crate) fn movement(&self) -> OperandMovement {
        let (gu_num, gu_den) = self.gate_up_semantic_num_den();
        let (dn_num, dn_den) = self.down_semantic_num_den();

        let gu_payload = self.gate_up_payload();
        let dn_payload = self.down_payload();

        // Scale streams are padded on the same axes as the payload they
        // describe, so they inherit the same semantic ratio. Split the
        // stream by axis rather than pro-rating the total.
        let (gu_scale, dn_scale) = if self.split_scales {
            let gu = 2 * self.inter * self.weight_cols / MXFP4_GROUP_ELEMS * E8M0_BYTES_PER_GROUP;
            let dn = self.hidden * self.inter_padded / MXFP4_GROUP_ELEMS * E8M0_BYTES_PER_GROUP;
            (gu, dn)
        } else {
            (0, 0)
        };
        debug_assert_eq!(gu_scale + dn_scale, self.scale_bytes());

        let gu_physical = gu_payload + gu_scale;
        let dn_physical = dn_payload + dn_scale;
        let physical_per_expert = gu_physical + dn_physical;
        let semantic_per_expert = gu_physical * gu_num / gu_den + dn_physical * dn_num / dn_den;

        OperandMovement::fully_consumed(
            (semantic_per_expert * self.n_slots) as u64,
            (physical_per_expert * self.n_slots) as u64,
            // Expert banks are mmap-registered regions read by the GPU
            // out of unified memory. A cold-estate arm that faults them
            // from storage must record NVMe separately — see the I/O
            // sampler, which attributes fault traffic on its own.
            Tier::Dram,
        )
    }
}

/// Decide how this layer's routed expert group is physically satisfied,
/// and record that decision against the ledger.
///
/// This replaced a bare `record_expert_layer(shape)` when the execution
/// seam landed, and the replacement is deliberate rather than additive:
/// both encode arms already had to call the byte authority here, so
/// routing the decision through the SAME call makes it impossible for a
/// backend to skip an expert group without the ledger hearing about it,
/// or to record avoided bytes for work it actually ran. The two counters
/// move together or not at all.
///
/// The caller MUST honour the returned strategy — that is the one half
/// this function cannot enforce. `test_exec_policy_expert_skip` holds
/// both Metal arms to it.
#[must_use = "the caller must honour the returned strategy, or the ledger lies"]
pub(crate) fn resolve_expert_layer(shape: &ExpertLayerShape, layer: usize) -> ExecutionStrategy {
    exec_policy::resolve_expert_group(layer, shape.n_slots, shape.movement())
}

#[cfg(test)]
#[path = "tests/experts.rs"]
mod tests;
