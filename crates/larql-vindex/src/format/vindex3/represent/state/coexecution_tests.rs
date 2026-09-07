//! **The falsifier: batching must be invisible to identity and evidence.**
//!
//! These assert that co-execution changes nothing observable — NOT that
//! it is faster. Speed is what the ledger reports; soundness is what
//! makes reporting it legitimate.

use super::super::super::measurement::EvidenceScale;
use super::super::fixtures;
use super::*;

fn keys(scale: EvidenceScale) -> Vec<MeasurementKey> {
    [
        fixtures::p(),
        fixtures::t1(),
        fixtures::s2(),
        fixtures::s1(),
    ]
    .iter()
    .map(|s| fixtures::key_for(s, scale))
    .collect()
}

/// **THE property.** Assembling a batch leaves every member's key
/// bit-identical to the key it would have had alone, in the same order.
#[test]
fn batching_does_not_change_any_members_identity() {
    let alone = keys(EvidenceScale::Authority);
    let batch = ExecutionBatch::assemble(alone.clone()).expect("co-executable");
    assert_eq!(batch.members(), alone.as_slice());
    assert_eq!(batch.len(), 4);
    // and each digest individually, so a reordering could not pass
    for (b, a) in batch.members().iter().zip(&alone) {
        assert_eq!(b.as_str(), a.as_str());
        assert_eq!(b.state(), a.state());
        assert_eq!(b.bank(), a.bank());
        assert_eq!(b.scale(), a.scale());
        assert_eq!(b.instrument(), a.instrument());
    }
}

/// A batch spanning two evidence SCALES is refused: one run cannot be
/// both a diagnostic and an authority reading, and the ladder is built
/// on their being different experiments.
#[test]
fn requests_at_different_scales_are_not_co_executable() {
    let two = vec![
        fixtures::key_for(&fixtures::p(), EvidenceScale::Authority),
        fixtures::key_for(&fixtures::t1(), EvidenceScale::Diagnostic),
    ];
    assert_eq!(ExecutionBatch::assemble(two), Err(BatchRefusal::MixedScale));
}

/// The same state twice would be recorded twice under one key, which the
/// registry cannot tell from a repeated measurement.
#[test]
fn a_repeated_state_is_refused() {
    let p = fixtures::key_for(&fixtures::p(), EvidenceScale::Authority);
    assert_eq!(
        ExecutionBatch::assemble(vec![p.clone(), p]),
        Err(BatchRefusal::DuplicateState)
    );
}

/// An empty batch is refused rather than succeeding vacuously — a batch
/// that "completes" with no members reads downstream as a measurement.
#[test]
fn an_empty_batch_is_refused() {
    assert_eq!(ExecutionBatch::assemble(vec![]), Err(BatchRefusal::Empty));
    for r in [
        BatchRefusal::Empty,
        BatchRefusal::MixedBank,
        BatchRefusal::MixedScale,
        BatchRefusal::MixedInstrument,
        BatchRefusal::DuplicateState,
    ] {
        assert!(!r.to_string().is_empty(), "every refusal must say why");
    }
}

/// A single request is a legal batch of one. The abstraction must not
/// require a population to be usable, or the executor grows a second
/// path for the ordinary case.
#[test]
fn one_request_is_a_legal_batch() {
    let one = vec![fixtures::key_for(&fixtures::p(), EvidenceScale::Authority)];
    let b = ExecutionBatch::assemble(one.clone()).expect("batch of one");
    assert_eq!(b.members(), one.as_slice());
}

/// The ledger reports what was shared, and is undefined rather than
/// misleading when there is nothing to divide by.
#[test]
fn the_sharing_ledger_reports_amplification_against_one_member() {
    // The GLM shape: 8 arms, 53 experts materialised where 8 isolated
    // traversals would have touched 8 x 53-ish.
    let l = SharingLedger {
        members: 8,
        units_if_isolated: 8 * 49,
        units_materialised: 53,
    };
    let amp = l.amplification_vs_one().expect("defined");
    assert!((amp - 53.0 / 49.0).abs() < 1e-12, "{amp}");
    assert!(amp > 1.0 && amp < 1.2, "expected ~1.08x, got {amp}");
    let saved = l.saving().expect("defined");
    assert!(
        saved > 0.85,
        "8 arms for 1.08x should save >85%, got {saved}"
    );
    assert!(l.describe().contains("8 members"));

    // Shared nothing: amplification equals the member count.
    let none = SharingLedger {
        members: 4,
        units_if_isolated: 40,
        units_materialised: 40,
    };
    assert!((none.amplification_vs_one().expect("defined") - 4.0).abs() < 1e-12);
    assert_eq!(none.saving(), Some(0.0));

    // Nothing materialised: no ratio is reported rather than 0 or NaN.
    let empty = SharingLedger {
        members: 0,
        units_if_isolated: 0,
        units_materialised: 0,
    };
    assert_eq!(empty.amplification_vs_one(), None);
    assert_eq!(empty.saving(), None);
    assert!(empty.describe().contains("nothing materialised"));
}
