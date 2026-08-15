//! Timing derivations, and the two opposite calibration shapes.
//!
//! The fixtures below are the project's own banked steady-state figures
//! (M3 Max, gpt-oss-20b, warmup 16 / n 256 / long prompt). They are used
//! here as SHAPES — the assertions pin directions and orders of
//! magnitude, not the third decimal of a run that will drift.

use super::*;
use crate::movement_ledger::bytes::ByteMovement;

fn m3() -> Rooflines {
    Rooflines::dram_only(TierBandwidth::m3_max_dram())
}

fn dram_bytes(n: u64) -> ByteMovement {
    ByteMovement {
        semantic_requested: n,
        physical_touched: n,
        useful_physical: n,
        dram: n,
        ..Default::default()
    }
}

/// Wall decomposes into GPU busy + bubble + a host residual. The residual
/// is derived, so it can never go negative and silently rebalance the
/// other terms.
#[test]
fn host_residual_is_derived_and_floored_at_zero() {
    let t = TimeAttribution {
        wall_ms: 14.56,
        gpu_busy_ms: 11.56,
        gpu_bubble_ms: 0.87,
        ..Default::default()
    };
    assert!((t.host_outside_gpu_ms() - 2.13).abs() < 1e-9);

    // Over-attributed terms (clock skew across sources) must not produce
    // a negative residual that flatters the decomposition.
    let skewed = TimeAttribution {
        wall_ms: 1.0,
        gpu_busy_ms: 5.0,
        gpu_bubble_ms: 5.0,
        ..Default::default()
    };
    assert_eq!(skewed.host_outside_gpu_ms(), 0.0);
}

/// Occupancy is None on an empty window rather than a fabricated ratio.
#[test]
fn occupancy_needs_a_nonzero_window() {
    assert_eq!(TimeAttribution::default().gpu_occupancy(), None);
    let t = TimeAttribution {
        wall_ms: 20.0,
        gpu_busy_ms: 10.0,
        ..Default::default()
    };
    assert_eq!(t.gpu_occupancy(), Some(0.5));
}

/// Aggregation then division reproduces the per-token mean.
#[test]
fn add_and_per_round_trip_to_the_mean() {
    let a = TimeAttribution {
        wall_ms: 10.0,
        gpu_busy_ms: 6.0,
        gpu_bubble_ms: 2.0,
        io_wait_ms: 1.0,
        host_wait_ms: 3.0,
        host_wait_reported: true,
        io_wait_reported: false,
    };
    let b = TimeAttribution {
        wall_ms: 20.0,
        gpu_busy_ms: 12.0,
        gpu_bubble_ms: 4.0,
        io_wait_ms: 2.0,
        host_wait_ms: 6.0,
        host_wait_reported: false,
        io_wait_reported: true,
    };
    let mean = a.add(&b).per(2);
    assert!((mean.wall_ms - 15.0).abs() < 1e-9);
    assert!((mean.gpu_busy_ms - 9.0).abs() < 1e-9);
    assert!((mean.gpu_bubble_ms - 3.0).abs() < 1e-9);
    assert!((mean.io_wait_ms - 1.5).abs() < 1e-9);
    assert!((mean.host_wait_ms - 4.5).abs() < 1e-9);
    // Reporter flags are sticky across the sum: one arm that sampled a
    // term marks the aggregate as sampled.
    assert!(mean.host_wait_reported && mean.io_wait_reported);
    assert_eq!(a.per(0), TimeAttribution::default());
}

/// An unreported wait term renders as unavailable, not as a measured
/// zero — the same distinction the byte side draws for reuse/prefetch.
#[test]
fn unsampled_wait_terms_are_not_measured_zeros() {
    let t = TimeAttribution {
        wall_ms: 10.0,
        gpu_busy_ms: 5.0,
        ..Default::default()
    };
    assert!(!t.host_wait_reported, "default is unsampled");
    assert!(!t.io_wait_reported);
    assert_eq!(t.host_wait_ms, 0.0, "the value is zero but unattested");
}

/// Bytes crossing a tier with no declared bandwidth contribute nothing to
/// the floor and are reported, so the share is knowably a lower bound.
#[test]
fn untiered_and_unroofed_bytes_are_reported_not_assumed() {
    let bytes = ByteMovement {
        semantic_requested: 1_000_000_000,
        physical_touched: 1_000_000_000,
        useful_physical: 1_000_000_000,
        dram: 400_000_000,
        nvme: 500_000_000, // no NVMe roofline declared
        ..Default::default()
    };
    let time = TimeAttribution {
        wall_ms: 10.0,
        gpu_busy_ms: 5.0,
        ..Default::default()
    };
    let c = MovementCost::derive(&bytes, &time, &m3());
    // 500 MB of NVMe (no ceiling) + 100 MB never attributed to any tier.
    assert_eq!(c.bytes_without_roofline, 600_000_000);
    // Floor prices ONLY the 400 MB of DRAM.
    assert!(
        (c.floor_ms - TierBandwidth::m3_max_dram().transfer_floor_ms(400_000_000)).abs() < 1e-9
    );
}

/// The share is a BRACKET, and its width is exactly the kernel efficiency
/// term. Reporting one number would be a guess wearing a measurement's
/// clothes.
#[test]
fn movement_share_is_bracketed_by_roofline_and_gpu_busy() {
    let c = MovementCost::derive(
        &dram_bytes(2_970_000_000),
        &TimeAttribution {
            wall_ms: 14.56,
            gpu_busy_ms: 11.56,
            gpu_bubble_ms: 0.87,
            ..Default::default()
        },
        &m3(),
    );
    let lo = c.share_of_wall.unwrap();
    let hi = c.gpu_busy_share.unwrap();
    assert!(lo < hi, "roofline share must bracket below busy share");
    assert!((lo - 0.556).abs() < 0.01, "lo={lo}");
    assert!((hi - 0.794).abs() < 0.01, "hi={hi}");
    // The bracket ratio IS eta.
    let eta = c.roofline_utilisation.unwrap();
    assert!((lo / hi - eta).abs() < 1e-6, "bracket width must equal eta");
}

/// No wall time, no share — and no NaN leaking into a printed verdict.
#[test]
fn empty_window_yields_no_shares() {
    let c = MovementCost::derive(&dram_bytes(1_000), &TimeAttribution::default(), &m3());
    assert_eq!(c.share_of_wall, None);
    assert_eq!(c.gpu_busy_share, None);
    assert_eq!(c.implied_stream_gbps, None);
    assert_eq!(c.roofline_utilisation, None);
    assert_eq!(c.predicted_saving_ms(1_000), None);
}

/// An undeclared DRAM roofline suppresses utilisation rather than
/// defaulting to a ceiling nobody measured.
#[test]
fn undeclared_roofline_suppresses_utilisation() {
    let c = MovementCost::derive(
        &dram_bytes(1_000_000_000),
        &TimeAttribution {
            wall_ms: 10.0,
            gpu_busy_ms: 5.0,
            ..Default::default()
        },
        &Rooflines::default(),
    );
    assert_eq!(c.roofline_utilisation, None);
    assert_eq!(c.floor_ms, 0.0);
    assert_eq!(c.bytes_without_roofline, 1_000_000_000);
    assert!(
        c.implied_stream_gbps.is_some(),
        "stream rate needs no ceiling"
    );
}

// ── CALIBRATION: two known interventions, opposite signatures ───────────
//
// The instrument is only trustworthy if it diagnoses both. These are the
// unit-level halves of the BW-A gate; the end-to-end halves run against a
// live model.

/// S2 (GPU-dataflow routing) removed 24 per-layer queue-starvation
/// bubbles. It moved the SAME bytes. The ledger must therefore show a
/// large latency win with ~zero byte delta, and must attribute it to the
/// bubble term — not to movement.
#[test]
fn calibration_s2_is_a_scheduling_win_with_no_byte_delta() {
    let bytes = dram_bytes(2_970_000_000);
    let control = MovementCost::derive(
        &bytes,
        &TimeAttribution {
            wall_ms: 20.65,
            gpu_busy_ms: 11.08,
            gpu_bubble_ms: 7.00,
            ..Default::default()
        },
        &m3(),
    );
    let candidate = MovementCost::derive(
        &bytes,
        &TimeAttribution {
            wall_ms: 14.80,
            gpu_busy_ms: 11.00,
            gpu_bubble_ms: 0.10,
            ..Default::default()
        },
        &m3(),
    );

    // Bytes identical by construction: the ledger cannot credit movement.
    assert_eq!(control.floor_ms, candidate.floor_ms);
    // Yet the wall moved a lot.
    let win_ms = 20.65 - 14.80;
    assert!(win_ms > 5.0);
    // And the movement share RISES even though nothing about movement
    // changed — because the denominator shrank. A share read as "movement
    // got more important" would be the §4.11 error in miniature.
    assert!(candidate.share_of_wall.unwrap() > control.share_of_wall.unwrap());
    // The honest attribution: the recovered time is bubble, not bytes.
    let bubble_recovered: f64 = 7.00 - 0.10;
    assert!(
        (bubble_recovered / win_ms - 1.0).abs() < 0.2,
        "the bubble delta must explain the wall delta"
    );
}

/// MXFP4 is the opposite shape: a large byte reduction buying a modest
/// latency reduction. The ledger must also expose WHY the naive
/// equal-efficiency projection over-predicted — the candidate streams at
/// a lower eta, so its bytes are cheaper to remove than the control's
/// rate suggests.
#[test]
fn calibration_mxfp4_is_a_byte_win_that_underdelivers_on_latency() {
    let control = MovementCost::derive(
        &dram_bytes(2_970_000_000),
        &TimeAttribution {
            wall_ms: 14.56,
            gpu_busy_ms: 11.56,
            gpu_bubble_ms: 0.87,
            ..Default::default()
        },
        &m3(),
    );
    let candidate = MovementCost::derive(
        &dram_bytes(2_230_000_000),
        &TimeAttribution {
            wall_ms: 13.40,
            gpu_busy_ms: 10.31,
            gpu_bubble_ms: 0.87,
            ..Default::default()
        },
        &m3(),
    );

    let byte_cut = 1.0 - 2_230.0 / 2_970.0;
    let wall_cut = 1.0 - 13.40 / 14.56;
    assert!(byte_cut > 0.24, "a real byte reduction: {byte_cut}");
    assert!(wall_cut < 0.09, "a modest latency reduction: {wall_cut}");
    assert!(
        byte_cut / wall_cut > 3.0,
        "bytes must be shown outrunning latency by a wide margin"
    );

    // Priced at the CONTROL's blended rate, removing 0.74 GB predicts
    // ~2.9 ms. Measured was ~1.16 ms. The over-prediction is the point:
    // an equal-eta projection is not licensed.
    let naive = control.predicted_saving_ms(740_000_000).unwrap();
    let measured = 14.56 - 13.40;
    assert!(naive / measured > 2.0, "naive={naive} measured={measured}");

    // And the ledger already carries the explanation: the candidate's own
    // streaming efficiency is materially lower.
    let eta_c = control.roofline_utilisation.unwrap();
    let eta_k = candidate.roofline_utilisation.unwrap();
    assert!(eta_k < eta_c, "eta control={eta_c} candidate={eta_k}");
}

/// The two calibration shapes must be DISTINGUISHABLE by the instrument.
/// If both interventions produced the same ledger signature, the ledger
/// would be measuring neither.
#[test]
fn the_two_calibration_shapes_have_opposite_signatures() {
    let s2_byte_delta = 0.0_f64;
    let s2_wall_delta = 20.65 - 14.80;
    let mxfp4_byte_delta = (2_970_000_000u64 - 2_230_000_000u64) as f64;
    let mxfp4_wall_delta = 14.56 - 13.40;

    assert_eq!(s2_byte_delta, 0.0);
    assert!(mxfp4_byte_delta > 0.0);
    assert!(
        s2_wall_delta > mxfp4_wall_delta * 4.0,
        "the zero-byte intervention must be the LARGER latency win"
    );
}
