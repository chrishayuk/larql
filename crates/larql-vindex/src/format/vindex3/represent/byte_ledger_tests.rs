//! `tests` for [`super`].

use super::*;

fn scope(scope: &str, family: &str, baseline: u64, candidate: u64) -> ScopeBytes {
    ScopeBytes {
        scope: scope.into(),
        family: family.into(),
        baseline_bytes: baseline,
        candidate_bytes: candidate,
    }
}

/// **The measured Kimi-Linear-48B-A3B ledger**, BF16 baseline against
/// the four-family Q8_0 map: 27 layers with layer 0 dense, 256 experts
/// of 3 x 2304 x 1024 per MoE layer with 8 routed per token, MLA at
/// layers {3,7,11,15,19,23,26} and KDA at the other 20, vocabulary
/// 163,840. Byte figures from the probe's own reports.
pub(in crate::format::vindex3::represent) fn kimi_four_family() -> ByteLedger {
    ByteLedger {
        model: "Kimi-Linear-48B-A3B-Instruct".into(),
        baseline_representation: "BF16".into(),
        candidate_representation: "experts L20-26 + KDA{20,21,22,24,25} + MLA{23,26} + head, Q8_0"
            .into(),
        scopes: vec![
            scope(
                "routed experts L0-19",
                "routed experts",
                2_151_677_952,
                2_151_677_952,
            ),
            scope(
                "routed experts L20-26",
                "routed experts",
                792_723_456,
                421_134_336,
            ),
            scope(
                "KDA projections x15",
                "KDA projections",
                1_132_462_080,
                1_132_462_080,
            ),
            scope(
                "KDA{20,21,22,24,25}",
                "KDA projections",
                377_487_360,
                200_540_160,
            ),
            scope(
                "MLA projections x5",
                "MLA projections",
                291_143_680,
                291_143_680,
            ),
            scope("MLA{23,26}", "MLA projections", 116_457_472, 61_868_032),
            scope("output head", "output head", 754_974_720, 401_080_320),
            scope(
                "shared experts x26",
                "shared experts",
                368_050_176,
                368_050_176,
            ),
        ],
    }
}

#[test]
fn the_checkpoint_is_experts_and_the_token_is_not() {
    let l = kimi_four_family();
    assert_eq!(l.baseline_bytes_per_token(), 5_984_976_896);
    let shares: Vec<(&str, f64)> = l
        .baseline_by_family()
        .into_iter()
        .map(|(f, b)| (f, 100.0 * b as f64 / l.baseline_bytes_per_token() as f64))
        .collect();
    // 97% of the CHECKPOINT is experts; 49% of the TOKEN is. That gap
    // is the whole argument for a whole-decoder map over an
    // expert-only one, so it is pinned here rather than in prose.
    let expected = [
        ("routed experts", 49.2),
        ("KDA projections", 25.2),
        ("output head", 12.6),
        ("MLA projections", 6.8),
        ("shared experts", 6.1),
    ];
    assert_eq!(shares.len(), expected.len());
    for ((got_f, got_pct), (want_f, want_pct)) in shares.iter().zip(expected) {
        assert_eq!(*got_f, want_f);
        assert!(
            (got_pct - want_pct).abs() < 0.05,
            "{want_f}: {got_pct:.1}% vs {want_pct}%"
        );
    }
}

#[test]
fn the_four_family_map_removes_the_measured_fraction() {
    let l = kimi_four_family();
    assert_eq!(l.candidate_bytes_per_token(), 5_027_956_736);
    assert_eq!(l.bytes_removed(), 957_020_160);
    assert!((l.fraction_removed() - 0.15990).abs() < 1e-5);
}

#[test]
fn breadth_is_counted_in_families_and_in_scopes() {
    let l = kimi_four_family();
    assert_eq!(
        l.families_changed().into_iter().collect::<Vec<_>>(),
        [
            "KDA projections",
            "MLA projections",
            "output head",
            "routed experts"
        ],
        "shared experts were refused on economics, so the map has four families"
    );
    assert_eq!(l.scopes_changed(), 4, "four scope entries moved");
}

#[test]
fn unchanged_scopes_still_count_toward_the_baseline() {
    // A ledger listing only the CHANGED scopes could not say what
    // fraction of the whole was removed, which is the one number a
    // throughput prediction is a function of.
    let full = kimi_four_family();
    let changed_only = ByteLedger {
        scopes: full
            .scopes
            .iter()
            .filter(|s| s.changed())
            .cloned()
            .collect(),
        ..full.clone()
    };
    assert_eq!(changed_only.bytes_removed(), full.bytes_removed());
    assert!(
        changed_only.fraction_removed() > 2.0 * full.fraction_removed(),
        "dropping the unchanged scopes inflates the fraction — {:.3} vs {:.3}",
        changed_only.fraction_removed(),
        full.fraction_removed()
    );
}

#[test]
fn an_empty_ledger_reports_zero_rather_than_a_nan() {
    let l = ByteLedger {
        model: "nothing".into(),
        baseline_representation: "BF16".into(),
        candidate_representation: "BF16".into(),
        scopes: Vec::new(),
    };
    assert_eq!(l.baseline_bytes_per_token(), 0);
    assert_eq!(l.fraction_removed(), 0.0);
    assert!(l.fraction_removed().is_finite());
    assert_eq!(l.scopes_changed(), 0);
}

#[test]
fn a_representation_that_grew_removes_nothing_rather_than_underflowing() {
    // Not hypothetical: a codec with per-block scales can exceed BF16
    // on a narrow tensor.
    let l = ByteLedger {
        model: "m".into(),
        baseline_representation: "BF16".into(),
        candidate_representation: "wider".into(),
        scopes: vec![scope("s", "f", 1_000, 1_400)],
    };
    assert_eq!(l.bytes_removed(), 0);
    assert_eq!(l.fraction_removed(), 0.0);
    assert_eq!(l.scopes[0].removed(), 0);
    assert!(l.scopes[0].changed(), "it did change — it just grew");
    assert_eq!(l.scopes_changed(), 1);
}

#[test]
fn a_map_that_changes_nothing_has_no_breadth() {
    let mut l = kimi_four_family();
    for s in &mut l.scopes {
        s.candidate_bytes = s.baseline_bytes;
    }
    assert_eq!(l.bytes_removed(), 0);
    assert_eq!(l.fraction_removed(), 0.0);
    assert_eq!(l.scopes_changed(), 0);
    assert!(l.families_changed().is_empty());
}
