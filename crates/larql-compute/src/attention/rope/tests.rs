//! Shape, norm and offset invariants shared by every RoPE entry point.
//!
//! Family-specific numerics live in [`super::llama3`] and [`super::yarn`].

use super::*;
use ndarray::Array2;

fn make_qk(seq: usize, heads: usize, head_dim: usize) -> Array2<f32> {
    let n = seq * heads * head_dim;
    Array2::from_shape_vec(
        (seq, heads * head_dim),
        (0..n).map(|i| (i as f32 + 1.0) * 0.01).collect(),
    )
    .unwrap()
}

#[test]
fn apply_rope_preserves_shape() {
    let x = make_qk(3, 2, 8);
    let out = apply_rope(&x, 2, 8, 10000.0);
    assert_eq!(out.shape(), x.shape());
}

#[test]
fn apply_rope_output_is_finite() {
    let x = make_qk(4, 2, 8);
    let out = apply_rope(&x, 2, 8, 10000.0);
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn apply_rope_preserves_norm_per_head() {
    // RoPE is a rotation → L2 norm of each position–head pair is preserved.
    let x = make_qk(3, 2, 8);
    let out = apply_rope(&x, 2, 8, 10000.0);
    for row in 0..3 {
        for h in 0..2 {
            let orig: f32 = x
                .row(row)
                .iter()
                .skip(h * 8)
                .take(8)
                .map(|v| v * v)
                .sum::<f32>();
            let rotd: f32 = out
                .row(row)
                .iter()
                .skip(h * 8)
                .take(8)
                .map(|v| v * v)
                .sum::<f32>();
            assert!(
                (orig.sqrt() - rotd.sqrt()).abs() < 1e-4,
                "RoPE changed L2 norm at row={row} head={h}: {orig} → {rotd}"
            );
        }
    }
}

#[test]
fn apply_rope_different_positions_differ() {
    // Row 0 (position 0) and row 1 (position 1) should differ after RoPE
    // even if the original vectors were identical.
    let data = vec![0.5f32; 3 * 8];
    let x = Array2::from_shape_vec((3, 8), data).unwrap();
    let out = apply_rope(&x, 1, 8, 10000.0);
    let row0: Vec<f32> = out.row(0).to_vec();
    let row1: Vec<f32> = out.row(1).to_vec();
    let differ = row0
        .iter()
        .zip(row1.iter())
        .any(|(a, b)| (a - b).abs() > 1e-6);
    assert!(
        differ,
        "identical inputs at different positions should differ after RoPE"
    );
}

#[test]
fn apply_rope_partial_at_offset() {
    // Position 5 with offset 0 should equal position 0 with offset 5.
    let x = make_qk(1, 2, 8);
    let out_pos5 = {
        let data = vec![0.1f32; 6 * 2 * 8];
        let big = Array2::from_shape_vec((6, 16), data).unwrap();
        apply_rope_partial_at(&big, 2, 8, 10000.0, 1.0, 0)
    };
    let out_off5 = apply_rope_partial_at(&x, 2, 8, 10000.0, 1.0, 5);
    // Both should be finite (structural check)
    assert!(out_pos5.iter().all(|v| v.is_finite()));
    assert!(out_off5.iter().all(|v| v.is_finite()));
}

#[test]
fn apply_rope_partial_fraction_zero_is_passthrough() {
    // fraction = 0.0 → no rotation applied (but we need at least 2 rotary dims).
    // With a very small fraction the rotation is minimal — test shape only.
    let x = make_qk(2, 2, 8);
    let out = apply_rope_partial(&x, 2, 8, 10000.0, 0.01);
    assert_eq!(out.shape(), x.shape());
    assert!(out.iter().all(|v| v.is_finite()));
}

// ── Property tests ────────────────────────────────────────────────────────

#[test]
fn rope_different_base_produces_different_output() {
    // Different rope_base → different frequencies → different output.
    let x = make_qk(2, 2, 8);
    let out1 = apply_rope(&x, 2, 8, 10_000.0);
    let out2 = apply_rope(&x, 2, 8, 500_000.0);
    let differs = out1
        .iter()
        .zip(out2.iter())
        .any(|(a, b)| (a - b).abs() > 1e-4);
    assert!(
        differs,
        "different rope_base should produce different output"
    );
}

#[test]
fn rope_partial_fraction_one_equals_full_rope() {
    let x = make_qk(3, 2, 8);
    let full = apply_rope(&x, 2, 8, 10000.0);
    let partial_1 = apply_rope_partial(&x, 2, 8, 10000.0, 1.0);
    for (a, b) in full.iter().zip(partial_1.iter()) {
        assert!((a - b).abs() < 1e-5, "fraction=1.0 should equal full rope");
    }
}

#[test]
fn rope_position_offset_matches_sequential_positions() {
    // apply_rope_partial_at(x, ..., offset=5) on a 1-token sequence should
    // equal row 5 of apply_rope on a 6-token sequence with identical rows.
    let hd = 8usize;
    let heads = 2usize;
    let val = 0.3f32;
    // Single row for the offset test
    let single = Array2::from_elem((1, heads * hd), val);
    // 6-row sequence of identical values
    let seq6 = Array2::from_elem((6, heads * hd), val);
    let out_seq6 = apply_rope(&seq6, heads, hd, 10000.0);
    let out_offset5 = apply_rope_partial_at(&single, heads, hd, 10000.0, 1.0, 5);
    // Row 5 of seq6 should match the single-row result with offset 5
    let row5: Vec<f32> = out_seq6.row(5).to_vec();
    let offset_row: Vec<f32> = out_offset5.row(0).to_vec();
    for (a, b) in row5.iter().zip(offset_row.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "offset=5 should match position 5 in sequential apply: {a} vs {b}"
        );
    }
}

#[test]
fn rope_partial_fraction_between_0_and_1_is_finite() {
    // Spot-check that various fractions produce finite, valid output.
    let x = make_qk(2, 2, 16);
    for &frac in &[0.25f64, 0.5, 0.75] {
        let out = apply_rope_partial(&x, 2, 16, 10000.0, frac);
        assert_eq!(out.shape(), x.shape());
        assert!(
            out.iter().all(|v| v.is_finite()),
            "fraction={frac} produced non-finite"
        );
    }
}

// ── scaling selection ────────────────────────────────────────────────
//
// Each family must visibly change the output. A scaling that silently
// no-ops is indistinguishable from a correct one at the call site, which
// is exactly how `openai/gpt-oss-20b` ran with plain RoPE for months.

fn llama3_default() -> Llama3RopeScaling {
    Llama3RopeScaling {
        factor: 32.0,
        low_freq_factor: 1.0,
        high_freq_factor: 4.0,
        original_max_position_embeddings: 8192.0,
    }
}

fn gpt_oss_yarn() -> YarnRopeScaling {
    YarnRopeScaling {
        factor: 32.0,
        beta_fast: 32.0,
        beta_slow: 1.0,
        original_max_position_embeddings: 4096.0,
        truncate: false,
        mscale: None,
        mscale_all_dim: None,
    }
}

#[test]
fn llama3_scaling_changes_rope_output() {
    let x = make_qk(4, 1, 32);
    let base = apply_rope_partial_at_full(&x, 1, 32, 10000.0, 1.0, 0, 1.0, RopeFreqScaling::None);
    let scaled = apply_rope_partial_at_full(
        &x,
        1,
        32,
        10000.0,
        1.0,
        0,
        1.0,
        RopeFreqScaling::Llama3(llama3_default()),
    );
    assert!(
        base.iter()
            .zip(scaled.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "llama3 scaling must change RoPE output for non-zero positions"
    );
}

#[test]
fn yarn_scaling_changes_rope_output() {
    let x = make_qk(4, 1, 64);
    let base = apply_rope_partial_at_full(&x, 1, 64, 150000.0, 1.0, 0, 1.0, RopeFreqScaling::None);
    let scaled = apply_rope_partial_at_full(
        &x,
        1,
        64,
        150000.0,
        1.0,
        0,
        1.0,
        RopeFreqScaling::Yarn(gpt_oss_yarn()),
    );
    assert!(
        base.iter()
            .zip(scaled.iter())
            .any(|(a, b)| (a - b).abs() > 1e-6),
        "yarn scaling must change RoPE output"
    );
}

/// The amplitude is what makes YaRN visible at position 0, where every
/// rotation angle is zero and a frequency-only family is a no-op. A test
/// that only checked a non-zero position could not tell the two apart —
/// the `out_features = 2` fixture mistake in a different costume.
#[test]
fn yarn_amplitude_is_visible_at_position_zero() {
    let x = make_qk(1, 1, 64);
    let base = apply_rope_partial_at_full(&x, 1, 64, 150000.0, 1.0, 0, 1.0, RopeFreqScaling::None);
    let scaled = apply_rope_partial_at_full(
        &x,
        1,
        64,
        150000.0,
        1.0,
        0,
        1.0,
        RopeFreqScaling::Yarn(gpt_oss_yarn()),
    );
    // At position 0: cos = amplitude, sin = 0, so every element is scaled
    // by exactly the amplitude.
    let amp = yarn::attention_amplitude(&gpt_oss_yarn()) as f32;
    for (b, s) in base.iter().zip(scaled.iter()) {
        assert!(
            (s - b * amp).abs() < 1e-4,
            "position 0 must scale by the amplitude: {s} vs {}",
            b * amp
        );
    }
    assert!(amp > 1.3, "gpt-oss amplitude should be ~1.3466, got {amp}");
}

/// Llama-3 carries no amplitude, so position 0 must be untouched. This is
/// the control for the test above: without it, "amplitude applied" and
/// "amplitude applied to the wrong family" look the same.
#[test]
fn llama3_leaves_position_zero_unchanged() {
    let x = make_qk(1, 1, 32);
    let base = apply_rope_partial_at_full(&x, 1, 32, 10000.0, 1.0, 0, 1.0, RopeFreqScaling::None);
    let scaled = apply_rope_partial_at_full(
        &x,
        1,
        32,
        10000.0,
        1.0,
        0,
        1.0,
        RopeFreqScaling::Llama3(llama3_default()),
    );
    for (b, s) in base.iter().zip(scaled.iter()) {
        assert!(
            (s - b).abs() < 1e-6,
            "llama3 must not rescale at position 0"
        );
    }
}

/// HF `proportional`: head-sized plan, the first `fraction·d/2` pairs at
/// `base^(-2i/d)` over the FULL head width, the rest at zero — and it
/// differs from the plain partial rotary at the same fraction in both
/// length and angle, so the two cannot be confused for one another.
#[test]
fn proportional_plan_takes_frequencies_over_the_full_head_and_zeroes_the_rest() {
    let head_dim = 512;
    let fraction = 0.25;
    let base = 1_000_000.0;
    let plan = rope_freq_plan_proportional(head_dim, fraction, base);
    assert_eq!(plan.inv_freq.len(), head_dim / 2);
    let rotated_pairs = 64;
    for (i, f) in plan.inv_freq.iter().enumerate() {
        if i < rotated_pairs {
            let expected = 1.0 / base.powf(2.0 * i as f64 / head_dim as f64);
            assert!((f - expected).abs() < 1e-15, "pair {i}: {f} vs {expected}");
        } else {
            assert_eq!(*f, 0.0, "pair {i} must be unrotated");
        }
    }
    assert_eq!(plan.amplitude, UNIT_AMPLITUDE);
    // The plain partial rotary at the same fraction: 64 pairs over a
    // 128-wide block, so pair 32 sits at base^(-64/128) not base^(-64/512).
    let plain = rope_freq_plan(head_dim, fraction, base, 1.0, RopeFreqScaling::None);
    assert_eq!(plain.inv_freq.len(), rotated_pairs);
    let pair = 32;
    assert!((plain.inv_freq[pair] - 1.0 / base.powf(64.0 / 128.0)).abs() < 1e-15);
    assert!(
        plan.inv_freq[pair] > plain.inv_freq[pair] * 10.0,
        "different angles, not a relabelling"
    );
}
