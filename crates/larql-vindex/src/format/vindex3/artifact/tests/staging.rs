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

// ── the JSON two front doors both print ──────────────────────────────

use super::super::staging::staging_report_json as render;

#[test]
fn the_rendered_figures_separate_what_was_read_from_what_it_stands_in_for() {
    let v = render(
        "GLM-5.3-Flash",
        Some("abc123"),
        &report(Ok(328_330_000_000), None),
    );
    assert_eq!(v["artifact"], "GLM-5.3-Flash");
    assert_eq!(v["commit"], "abc123");
    assert_eq!(v["shards"], 62);
    // Headers alone would understate the pass fourfold, so `staged` is
    // the total and `headers` stays visible beside it.
    assert_eq!(v["staged"], "39.39 MB");
    assert_eq!(v["headers"], "10.68 MB");
    assert_eq!(v["metadata"], "28.71 MB");
    assert_eq!(v["stands_in_for"], "328.33 GB (305.78 GiB)");
}

#[test]
fn a_local_artifact_has_no_commit_to_report() {
    let v = render("checkpoint", None, &report(Ok(1_000), None));
    assert!(v["commit"].is_null(), "{v}");
}

/// The index figure appears only when it contradicts the headers —
/// printing it when the two agree would read as a units bug rather than
/// as a fact about the checkpoint.
#[test]
fn an_agreeing_index_is_not_restated() {
    let agree = render(
        "granite",
        None,
        &report(Ok(7_319_475_200), Some(7_319_475_200)),
    );
    assert!(agree["index_declares"].is_null(), "{agree}");

    let disagree = render(
        "granite",
        None,
        &report(Ok(7_319_475_200), Some(6_805_672_960)),
    );
    assert!(!disagree["index_declares"].is_null(), "{disagree}");
}

/// A census that failed carries no payload figure, and the rendering
/// says nothing rather than inventing a zero.
#[test]
fn a_failed_census_reports_no_payload_and_no_index_claim() {
    let v = render(
        "broken",
        None,
        &report(Err(VindexError::Parse("no headers".into())), Some(123)),
    );
    assert!(v["stands_in_for"].is_null(), "{v}");
    assert!(v["index_declares"].is_null(), "{v}");
}

/// The wrapper's only other answer: a local checkpoint staged nothing,
/// so there is nothing to report — not an empty report, which would
/// read as "staged, and it cost zero".
#[test]
fn a_local_artifact_reports_no_staging_at_all() {
    let dir = tempfile::tempdir().unwrap();
    crate::format::vindex3::fixtures::miniature_glimmer(dir.path());
    let resolved = crate::format::vindex3::artifact::resolve(dir.path()).unwrap();
    assert!(
        resolved.staging_json().is_none(),
        "a local artifact has no staging pass to describe"
    );
}
