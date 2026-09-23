//! Metrics against closed-form answers. W3 (a foreign numpy reference over
//! real logits) is in `plan_tests.rs`.

use super::*;

/// Agreement for quantities computed in f64 from exact inputs.
const EXACT: f64 = 1e-12;

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= EXACT
}

#[test]
fn identical_rows_have_zero_kl_and_full_agreement() {
    let row = [1.0f32, 2.0, 3.0, 0.5, -1.0, 0.0, 2.5];
    let s = position_metrics(&row, &row, Some(2)).unwrap();
    assert_eq!(s.kl, 0.0);
    assert!(s.top1_agree);
    assert_eq!(s.top5_overlap, TOP_K_OVERLAP);
    assert_eq!(s.delta_nll, Some(0.0));
}

/// Reference p = (1/4, 3/4), candidate q = (1/2, 1/2), next token 0.
#[test]
fn a_two_token_distribution_matches_its_closed_form() {
    let reference = [0.0f32, 3.0f32.ln()];
    let candidate = [0.0f32, 0.0];
    let s = position_metrics(&reference, &candidate, Some(0)).unwrap();
    let kl = 0.25 * (0.25f64 / 0.5).ln() + 0.75 * (0.75f64 / 0.5).ln();
    assert!((s.kl - kl).abs() < 1e-7, "{} vs {kl}", s.kl);
    // NLL(candidate) - NLL(reference) = -ln(1/2) + ln(1/4) = -ln 2.
    assert!((s.delta_nll.unwrap() + 2f64.ln()).abs() < 1e-7);
    assert!((s.reference_margin - 0.5).abs() < 1e-7);
    let entropy = -(0.25 * 0.25f64.ln() + 0.75 * 0.75f64.ln());
    assert!((s.reference_entropy - entropy).abs() < 1e-7);
    // The candidate's tie resolves to the lowest id, 0; the reference's
    // argmax is 1.
    assert!(!s.top1_agree);
    // A two-token vocabulary has two top tokens, and both are shared.
    assert_eq!(s.top5_overlap, 2);
}

#[test]
fn kl_is_invariant_to_a_constant_shift_of_either_row() {
    let reference = [0.3f32, -1.2, 2.2, 0.0];
    let candidate = [0.1f32, -1.0, 2.0, 0.4];
    let shifted: Vec<f32> = candidate.iter().map(|v| v + 40.0).collect();
    let a = position_metrics(&reference, &candidate, None).unwrap();
    let b = position_metrics(&reference, &shifted, None).unwrap();
    assert!((a.kl - b.kl).abs() < 1e-9);
    assert_eq!(a.delta_nll, None);
}

#[test]
fn mismatched_empty_or_non_finite_rows_are_refused() {
    assert_eq!(
        position_metrics(&[0.0, 1.0], &[0.0], None),
        Err(MetricError::VocabularyMismatch {
            reference: 2,
            candidate: 1
        })
    );
    assert_eq!(position_metrics(&[], &[], None), Err(MetricError::Empty));
    assert_eq!(
        position_metrics(&[0.0, f32::NAN], &[0.0, 1.0], None),
        Err(MetricError::NonFinite { arm: "reference" })
    );
    assert_eq!(
        position_metrics(&[0.0, 1.0], &[f32::INFINITY, 1.0], None),
        Err(MetricError::NonFinite { arm: "candidate" })
    );
}

fn metric(sample: usize, category: &str, kl: f64, agree: bool, margin: f64) -> PositionMetrics {
    PositionMetrics {
        sample,
        position: 0,
        category: category.to_string(),
        kl,
        top1_agree: agree,
        top5_overlap: 4,
        delta_nll: if agree { Some(0.5) } else { None },
        reference_margin: margin,
        reference_entropy: 1.0,
    }
}

#[test]
fn aggregates_use_nearest_rank_percentiles() {
    let positions: Vec<_> = (1..=100)
        .map(|i| metric(0, "prose", f64::from(i), i % 4 != 0, 0.2))
        .collect();
    let a = aggregate(&positions).unwrap();
    assert_eq!(a.positions, 100);
    assert!(close(a.kl_mean, 50.5));
    assert_eq!(a.kl_p50, 50.0);
    assert_eq!(a.kl_p99, 99.0);
    assert_eq!(a.kl_max, 100.0);
    assert!(close(a.top1_agreement, 0.75));
    assert!(close(a.top5_overlap_mean, 4.0));
    // Only the agreeing positions carried a next-token ΔNLL of 0.5.
    assert!(close(a.delta_nll_mean.unwrap(), 0.5));
    assert_eq!(aggregate(&[]), None);
}

#[test]
fn summaries_split_by_category_and_margin_and_omit_empty_groups() {
    let positions = vec![
        metric(0, "code", 0.1, true, 0.05),
        metric(1, "prose", 0.2, true, 0.3),
        metric(1, "prose", 0.3, false, 1.0),
        metric(2, "code", 0.4, true, 0.3),
    ];
    let s = summarise(&positions).unwrap();
    assert_eq!(s.all.positions, 4);
    let categories: Vec<&str> = s.by_category.iter().map(|(c, _)| c.as_str()).collect();
    assert_eq!(categories, ["code", "prose"], "first-seen order");
    assert_eq!(s.by_category[0].1.positions, 2);
    // One position per band, and a margin of exactly 1.0 lands in the last.
    let bands: Vec<usize> = s.by_margin_band.iter().map(|(_, a)| a.positions).collect();
    assert_eq!(bands, [1, 2, 1]);
    let only_confident = vec![metric(0, "code", 0.1, true, 0.9)];
    let s = summarise(&only_confident).unwrap();
    assert_eq!(s.by_margin_band.len(), 1);
    assert_eq!(s.by_margin_band[0].0, MARGIN_BANDS[2]);
    assert_eq!(summarise(&[]), None);
}
