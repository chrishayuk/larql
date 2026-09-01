//! One ingest, two callers.
//!
//! `vindex encode` and `larql vindex3 encode` both come through
//! [`encode_from_specs`], so what it does IS what a container is. The
//! gates here are the properties a second orchestration would have been
//! free to get differently: the capability snapshot, the transfer ledger,
//! and whether a local encode claims to have fetched anything.

use std::path::Path;

use serial_test::serial;

use crate::format::huggingface::range::test_support::{MockRepo, RangeBehaviour, MOCK_REPO};
use crate::format::vindex3::artifact::{encode_from_specs, resolve_all};
use crate::format::vindex3::fixtures::miniature_glimmer;

/// Encode the fixture from a local directory.
fn encode_local() -> (
    tempfile::TempDir,
    crate::format::vindex3::artifact::IngestOutcome,
) {
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_glimmer(checkpoint.path());
    let container = tempfile::tempdir().unwrap();
    let resolved = resolve_all(&[checkpoint.path().to_path_buf()]).unwrap();
    let outcome = encode_from_specs(resolved, container.path(), None).unwrap();
    (container, outcome)
}

#[test]
fn a_local_encode_writes_a_container_and_claims_no_transfer() {
    let (container, outcome) = encode_local();

    assert!(outcome.representations > 0, "something was encoded");
    assert!(outcome.total_payload_bytes > 0);
    assert_eq!(outcome.container, container.path());
    assert!(
        outcome.transfers.is_empty(),
        "nothing crossed a network, so the ledger must say nothing — \
         an empty transfer entry reading 0 bytes would be a different claim"
    );
}

#[test]
fn the_capability_snapshot_travels_with_the_container() {
    // A container without the tokenizer binds with token-id capability
    // only. That is not obviously wrong at encode time — it just answers
    // differently later — which is exactly why it belongs to the one
    // shared ingest rather than to whichever CLI happened to call it.
    //
    // The fixture ships no tokenizer of its own, so one is written here:
    // the claim under test is that ingest CARRIES what the checkpoint
    // has, not that this particular fixture has it.
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_glimmer(checkpoint.path());
    std::fs::write(checkpoint.path().join("tokenizer.json"), b"{\"model\":{}}").unwrap();

    let container = tempfile::tempdir().unwrap();
    let resolved = resolve_all(&[checkpoint.path().to_path_buf()]).unwrap();
    let outcome = encode_from_specs(resolved, container.path(), None).unwrap();

    assert!(
        outcome
            .capabilities
            .iter()
            .any(|name| name == "tokenizer.json"),
        "a checkpoint carrying a tokenizer must have it snapshotted: {:?}",
        outcome.capabilities
    );
    for name in &outcome.capabilities {
        assert!(
            container.path().join(name).exists(),
            "`{name}` was reported as copied but is not in the container"
        );
    }
}

#[test]
fn a_checkpoint_with_no_capability_files_reports_none() {
    // The other half, and the honest one: absence is reported as absence
    // rather than as a copy that silently did nothing.
    let (_container, outcome) = encode_local();
    assert!(
        outcome.capabilities.is_empty(),
        "the bare fixture carries no tokenizer: {:?}",
        outcome.capabilities
    );
}

#[test]
#[serial]
fn a_repo_encode_reports_what_it_actually_fetched() {
    // The ratio IS the claim. Reporting it from the source's own counter
    // rather than from the plan is what makes "the checkpoint was never
    // downloaded" checkable instead of asserted.
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_glimmer(checkpoint.path());
    let _repo = MockRepo::serve(checkpoint.path(), RangeBehaviour::Honour);

    let container = tempfile::tempdir().unwrap();
    let resolved = resolve_all(&[Path::new(&format!("hf://{MOCK_REPO}")).to_path_buf()]).unwrap();
    let outcome = encode_from_specs(resolved, container.path(), None).unwrap();

    assert_eq!(outcome.transfers.len(), 1, "one repo, one ledger entry");
    let transfer = &outcome.transfers[0];
    assert!(transfer.tensors > 0, "tensors crossed the wire");
    assert_eq!(
        transfer.fetched, transfer.declared,
        "this fixture's plan binds every tensor, so the two totals agree"
    );
    assert!((transfer.fraction() - 1.0).abs() < f64::EPSILON);
}

#[test]
fn a_transfer_that_declared_nothing_reports_no_fraction() {
    // Guarding the divisor rather than the caller: a ratio of NaN would
    // render as `NaN%` in a line whose whole job is to be believable.
    use crate::format::vindex3::artifact::RemoteTransfer;

    let empty = RemoteTransfer {
        name: "nothing".to_string(),
        fetched: 0,
        declared: 0,
        tensors: 0,
    };
    assert_eq!(empty.fraction(), 0.0);
    assert!(empty.fraction().is_finite());
}

#[test]
#[serial]
fn both_spellings_of_one_checkpoint_encode_the_same_container() {
    // The single-authority claim, made checkable. A local encode and a
    // repo encode of the SAME bytes must agree on the payload, or the two
    // CLIs' entry points describe different models.
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_glimmer(checkpoint.path());

    let local_container = tempfile::tempdir().unwrap();
    let local = encode_from_specs(
        resolve_all(&[checkpoint.path().to_path_buf()]).unwrap(),
        local_container.path(),
        None,
    )
    .unwrap();

    let _repo = MockRepo::serve(checkpoint.path(), RangeBehaviour::Honour);
    let remote_container = tempfile::tempdir().unwrap();
    let remote = encode_from_specs(
        resolve_all(&[Path::new(&format!("hf://{MOCK_REPO}")).to_path_buf()]).unwrap(),
        remote_container.path(),
        None,
    )
    .unwrap();

    assert_eq!(local.representations, remote.representations);
    assert_eq!(
        local.total_payload_bytes, remote.total_payload_bytes,
        "the same checkpoint, read two ways, is the same payload"
    );
    assert_eq!(local.capabilities, remote.capabilities);
}
