//! `StagingReport`'s two derived answers.
//!
//! Both exist so a caller never has to reconstruct them, and both have a
//! wrong answer that looks right: a total that omits metadata understates
//! the transfer fourfold on GLM-5.3-Flash, and a disagreement reported as
//! `0` is indistinguishable from agreement.

use crate::error::VindexError;
use crate::format::vindex3::artifact::StagingReport;

fn report(payload: Result<u64, VindexError>, declared: Option<u64>) -> StagingReport {
    StagingReport {
        header_bytes: 10_680_000,
        metadata_bytes: 28_710_000,
        shards: 62,
        payload_bytes: payload,
        declared_total: declared,
    }
}

#[test]
fn the_staged_total_counts_metadata_as_well_as_headers() {
    // GLM-5.3-Flash's real split. Quoting headers alone would report
    // 10.68 MB for a pass that actually moved 39.39 MB.
    let staged = report(Ok(328_330_000_000), None).staged_bytes();
    assert_eq!(staged, 39_390_000);
}

#[test]
fn an_index_that_agrees_with_its_headers_reports_no_disagreement() {
    assert_eq!(
        report(Ok(7_319_475_200), Some(7_319_475_200)).index_disagreement(),
        None,
        "agreement is not a difference of zero — there is nothing to say"
    );
}

#[test]
fn a_tied_weight_index_reports_the_difference_it_understates_by() {
    // granite-4.2-3b: the index counts the tied member once, the file
    // serialises it twice. Exactly one 513,802,240-byte member apart.
    assert_eq!(
        report(Ok(7_319_475_200), Some(6_805_672_960)).index_disagreement(),
        Some(513_802_240)
    );
}

#[test]
fn nothing_to_compare_against_is_not_a_disagreement() {
    // Three different reasons there is nothing to report, and none of
    // them is "they differ by zero".
    assert_eq!(report(Ok(1_000), None).index_disagreement(), None);
    assert_eq!(
        report(Err(VindexError::Parse("census failed".into())), Some(1_000)).index_disagreement(),
        None,
        "a failed census cannot be compared against anything"
    );
}
