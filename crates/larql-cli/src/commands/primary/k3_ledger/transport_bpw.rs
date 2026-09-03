//! What a conditional bit-rate is worth, on each arm, in tok/s.
//!
//! Pure. A conditional entropy is a statistic; this module turns it into the
//! only two numbers that decide whether to build anything, and applies R4 to
//! both before the codec exists.
//!
//! ## The two arms answer differently, and must never be merged
//!
//! **Resident arm** — every weight in DRAM, bandwidth-bound. Routed experts are
//! 25.83 GB/token against dense 29.86, so they are the *smaller* half. Zero the
//! routed bytes entirely and the ceiling is `total/dense` ≈ **1.87×**. No expert
//! transport codec, however good, can exceed that. It is the R4 zero-out bound
//! and it is computed here rather than quoted.
//!
//! **External arm** — dense resident, experts crossing a storage link. Here the
//! codec attacks the whole traffic term, so the *ratio* is unbounded even though
//! the absolute rate is dire. The gain is exactly linear in effective bpw and is
//! therefore **invariant to the expert-cache hit rate** — which is why the ratio
//! column travels between cache designs and the tok/s column does not.
//!
//! ## Scale bits are already nearly gone
//!
//! The naive scale term is 0.25 bpw (one e8m0 byte per 32 weights), but
//! `AdaptiveSuperblockDelta` already reaches **0.0673 bpw** with no cross-expert
//! trick at all. So the honest denominator is the 4.06731 exact floor, not 4.25,
//! and the entire headroom available to conditional *scale* coding is 0.0673 bpw
//! — third-order. [`effective_bpw`] takes the min so a conditional scale result
//! can never be credited with savings the existing format already banked.

use serde::Serialize;

use super::frontier::ServingPremises;
use super::geometry::K3Geometry;
use super::serving_format::{self, Container};

/// Nibbles per packed byte — how expert bytes become expert weights.
pub const WEIGHTS_PER_PACKED_BYTE: u64 = 2;

/// Bits naming a mode when a codec offers more than one. Charged per expert,
/// not per weight, so it amortises to nothing; carried so the record shows it
/// was priced rather than ignored.
pub const CODEC_MODE_BITS: f64 = 1.0;

/// All-in bits per weight the checkpoint spends today.
pub fn native_bpw() -> f64 {
    Container::Mxfp4.all_in_bits()
}

/// The cheapest exact format already available without any cross-expert result.
/// Every compression claim is scored against this, not against the native rate.
pub fn exact_floor_bpw() -> f64 {
    serving_format::exact_floor().all_in_bits()
}

/// Scale bits per weight the existing exact floor already achieves.
pub fn banked_scale_bpw() -> f64 {
    serving_format::exact_floor().scale.bits_per_weight()
}

/// Bits to name a coding parent, amortised over the expert it codes.
pub fn index_bpw(n_experts: usize, weights_per_expert: u64) -> f64 {
    if n_experts <= 1 || weights_per_expert == 0 {
        return 0.0;
    }
    ((n_experts as f64).log2() + CODEC_MODE_BITS) / weights_per_expert as f64
}

/// Compose a transport rate. `scale_bits` is clamped at the banked floor: a
/// conditional scale measurement may improve on the raw e8m0 stream, but it
/// cannot be credited for savings `AdaptiveSuperblockDelta` already delivers.
pub fn effective_bpw(payload_bits: f64, scale_bits: f64, index_bits: f64) -> f64 {
    payload_bits.max(0.0) + scale_bits.min(banked_scale_bpw()).max(0.0) + index_bits.max(0.0)
}

/// One row of the kill table.
#[derive(Debug, Clone, Serialize)]
pub struct TransportRow {
    pub payload_bits: f64,
    pub effective_bpw: f64,
    pub compression_vs_native: f64,
    pub compression_vs_exact_floor: f64,
    /// Resident arm: all weights in DRAM, bandwidth-bound.
    pub resident_tok_s: f64,
    pub resident_gain: f64,
    /// External arm: dense resident, experts over the link. Hit-rate-invariant
    /// in `external_gain`; `external_tok_s` assumes every routed expert misses.
    pub external_tok_s: f64,
    pub external_gain: f64,
}

/// Arm-independent context, computed once from the checkpoint.
#[derive(Debug, Clone, Serialize)]
pub struct TransportContext {
    pub dense_bytes: f64,
    pub routed_bytes: f64,
    pub weights_per_expert: u64,
    pub native_bpw: f64,
    pub exact_floor_bpw: f64,
    /// R4: what the resident arm reaches when routed bytes go to zero.
    pub resident_r4_ceiling_tok_s: f64,
    /// R4 as a ratio — the hard cap on every expert-side lever.
    pub expert_side_ceiling: f64,
    pub link_gb_s: f64,
    pub baseline_resident_tok_s: f64,
    pub baseline_external_tok_s: f64,
}

pub fn context(geom: &K3Geometry, p: &ServingPremises, link_gb_s: f64) -> TransportContext {
    let dense = geom.dense_bytes(p.dense_all_in_bits(geom));
    let routed = geom.routed_bytes_per_position();
    let link = link_gb_s * 1e9;
    TransportContext {
        dense_bytes: dense,
        routed_bytes: routed,
        weights_per_expert: geom.expert_bytes * WEIGHTS_PER_PACKED_BYTE,
        native_bpw: native_bpw(),
        exact_floor_bpw: exact_floor_bpw(),
        resident_r4_ceiling_tok_s: safe_div(p.effective_bw(), dense),
        expert_side_ceiling: safe_div(dense + routed, dense),
        link_gb_s,
        baseline_resident_tok_s: safe_div(p.effective_bw(), dense + routed),
        baseline_external_tok_s: safe_div(link, routed),
    }
}

fn safe_div(num: f64, den: f64) -> f64 {
    if den <= 0.0 {
        0.0
    } else {
        num / den
    }
}

/// Price one payload rate on both arms.
pub fn row(
    ctx: &TransportContext,
    p: &ServingPremises,
    payload_bits: f64,
    scale_bits: f64,
) -> TransportRow {
    let bpw = effective_bpw(
        payload_bits,
        scale_bits,
        index_bpw(1, ctx.weights_per_expert),
    );
    row_at_bpw(ctx, p, payload_bits, bpw)
}

/// Price one payload rate that additionally names a per-expert coding parent.
pub fn row_with_parent(
    ctx: &TransportContext,
    p: &ServingPremises,
    payload_bits: f64,
    scale_bits: f64,
    n_experts: usize,
) -> TransportRow {
    let bpw = effective_bpw(
        payload_bits,
        scale_bits,
        index_bpw(n_experts, ctx.weights_per_expert),
    );
    row_at_bpw(ctx, p, payload_bits, bpw)
}

fn row_at_bpw(
    ctx: &TransportContext,
    p: &ServingPremises,
    payload_bits: f64,
    bpw: f64,
) -> TransportRow {
    let shrink = bpw / ctx.native_bpw;
    let routed = ctx.routed_bytes * shrink;
    let resident = safe_div(p.effective_bw(), ctx.dense_bytes + routed);
    let external = safe_div(ctx.link_gb_s * 1e9, routed);
    TransportRow {
        payload_bits,
        effective_bpw: bpw,
        compression_vs_native: safe_div(ctx.native_bpw, bpw),
        compression_vs_exact_floor: safe_div(ctx.exact_floor_bpw, bpw),
        resident_tok_s: resident,
        resident_gain: safe_div(resident, ctx.baseline_resident_tok_s),
        external_tok_s: external,
        external_gain: safe_div(external, ctx.baseline_external_tok_s),
    }
}

/// The kill table: what each conditional bit-rate would actually be worth.
pub fn kill_table(
    ctx: &TransportContext,
    p: &ServingPremises,
    payload_grid: &[f64],
    scale_bits: f64,
) -> Vec<TransportRow> {
    payload_grid
        .iter()
        .map(|&b| row(ctx, p, b, scale_bits))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::super::geometry::DENSE_4_25;
    use super::*;
    use crate::commands::primary::k3_ledger::geometry::k3_reference;

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "expected ~{b}, got {a}");
    }

    fn ctx() -> (TransportContext, ServingPremises) {
        let p = ServingPremises {
            dense: DENSE_4_25,
            ..Default::default()
        };
        (context(&k3_reference(), &p, 3.5), p)
    }

    #[test]
    fn the_banked_denominators_are_the_exact_floor_not_the_native_rate() {
        approx(native_bpw(), 4.25, 1e-9);
        approx(exact_floor_bpw(), 4.06730625, 1e-6);
        approx(banked_scale_bpw(), 0.06730625, 1e-6);
    }

    #[test]
    fn scale_bits_are_clamped_at_what_the_existing_format_already_banks() {
        // A conditional scale result claiming 0.20 bpw is not an improvement:
        // AdaptiveSuperblockDelta already reaches 0.0673 with no conditioning.
        approx(
            effective_bpw(2.0, 0.20, 0.0),
            2.0 + banked_scale_bpw(),
            1e-9,
        );
        // But a genuinely better one is credited.
        approx(effective_bpw(2.0, 0.01, 0.0), 2.01, 1e-9);
    }

    #[test]
    fn naming_a_parent_among_896_costs_essentially_nothing() {
        let (c, _) = ctx();
        let bits = index_bpw(896, c.weights_per_expert);
        assert!(bits < 1e-5, "index overhead {bits} should be negligible");
        assert!(bits > 0.0, "but it must still be priced, not assumed zero");
    }

    #[test]
    fn a_single_implicit_prototype_costs_nothing_at_all() {
        approx(index_bpw(1, 1_000_000), 0.0, 1e-12);
        approx(index_bpw(896, 0), 0.0, 1e-12);
    }

    #[test]
    fn r4_caps_every_expert_side_lever_below_two_x() {
        let (c, _) = ctx();
        assert!(
            c.expert_side_ceiling < 2.0 && c.expert_side_ceiling > 1.8,
            "R4 ceiling {} should be ~1.87",
            c.expert_side_ceiling
        );
        approx(c.baseline_resident_tok_s, 6.6, 0.2);
    }

    #[test]
    fn even_a_free_codec_cannot_beat_the_r4_ceiling_on_the_resident_arm() {
        let (c, p) = ctx();
        let free = row(&c, &p, 0.0, 0.0);
        assert!(
            free.resident_tok_s <= c.resident_r4_ceiling_tok_s + 1e-9,
            "{} exceeded the zero-out ceiling {}",
            free.resident_tok_s,
            c.resident_r4_ceiling_tok_s
        );
        assert!(free.resident_gain < 2.0);
    }

    #[test]
    fn the_external_arm_baseline_is_link_bound_and_dire() {
        let (c, _) = ctx();
        // 25.83 GB/token over a 3.5 GB/s link. The 147.6x gap against 20 tok/s
        // is exactly this figure: 20 / 0.1355 = 147.6.
        approx(c.baseline_external_tok_s, 0.1355, 0.005);
        approx(20.0 / c.baseline_external_tok_s, 147.6, 3.0);
    }

    #[test]
    fn external_gain_is_linear_in_bpw_and_resident_gain_is_not() {
        let (c, p) = ctx();
        // Payload chosen so payload + scale lands exactly on half the native
        // rate; the scale term is charged, so it must be handed in too.
        let half = row(
            &c,
            &p,
            native_bpw() / 2.0 - banked_scale_bpw(),
            banked_scale_bpw(),
        );
        approx(half.effective_bpw, native_bpw() / 2.0, 1e-9);
        // The external arm sees the full ratio; the resident arm is diluted by
        // the dense half it cannot touch. Halving expert bytes buys 2.00x over
        // the link and 1.30x in DRAM — quoting one for the other overstates the
        // resident lever by 54%.
        approx(half.external_gain, 2.0, 0.01);
        approx(half.resident_gain, 1.30, 0.01);
        assert!(half.resident_gain < c.expert_side_ceiling);
    }

    #[test]
    fn a_three_bit_conditional_result_is_worth_almost_nothing() {
        let (c, p) = ctx();
        let r = row(&c, &p, 3.0, 0.0);
        assert!(r.compression_vs_exact_floor < 1.4);
        assert!(
            r.resident_gain < 1.2,
            "3 bits given a parent is not a codec: {}",
            r.resident_gain
        );
    }

    #[test]
    fn a_one_bit_conditional_result_is_a_real_but_bounded_lever() {
        let (c, p) = ctx();
        let r = row(&c, &p, 1.0, 0.0);
        assert!(r.compression_vs_native > 3.5, "{}", r.compression_vs_native);
        assert!(r.external_gain > 3.5);
        // ...and still under 2x where the dense half is present.
        assert!(r.resident_gain < c.expert_side_ceiling);
    }

    #[test]
    fn the_kill_table_is_monotone_in_payload_bits() {
        let (c, p) = ctx();
        let rows = kill_table(&c, &p, &[0.75, 1.0, 1.5, 2.0, 2.5, 3.0], 0.0);
        assert_eq!(rows.len(), 6);
        for w in rows.windows(2) {
            assert!(w[0].effective_bpw < w[1].effective_bpw);
            assert!(w[0].resident_tok_s > w[1].resident_tok_s);
            assert!(w[0].compression_vs_native > w[1].compression_vs_native);
        }
    }

    #[test]
    fn naming_a_parent_does_not_measurably_move_the_economics() {
        let (c, p) = ctx();
        let plain = row(&c, &p, 1.5, 0.0);
        let graph = row_with_parent(&c, &p, 1.5, 0.0, 896);
        approx(plain.resident_tok_s, graph.resident_tok_s, 1e-3);
    }

    #[test]
    fn a_zero_bandwidth_premise_does_not_divide_by_zero() {
        let geom = k3_reference();
        let p = ServingPremises {
            bandwidth_gb_s: 0.0,
            ..Default::default()
        };
        let c = context(&geom, &p, 0.0);
        let r = row(&c, &p, 1.0, 0.0);
        assert_eq!(r.resident_tok_s, 0.0);
        assert_eq!(r.external_tok_s, 0.0);
        assert_eq!(r.external_gain, 0.0);
    }
}
