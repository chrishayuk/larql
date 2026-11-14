//! What a failed COMPOSE install leaves in the session, measured by forcing the failure.
//!
//! **These tests record a known defect, not a desired behaviour.** A failed install is expected to
//! leave the session exactly as it found it, and today it does not: the install writes the decoy
//! cache, the fact's raw residual and the overlay slot, and rewrites every other slot at the layer,
//! before the steps that can fail. Only a fault before the first write changes nothing.
//!
//! Fixing that is a separate change, with its own review. When it lands, the tables below must
//! become empty and `a_failed_install_changes_nothing` must pass for every point; until then they
//! state the truth, so that nothing about the experiment depends on an assumption.

use larql_lql::install_fault::InstallFaultPoint as P;

use crate::support::*;

const PROMPTS: [&str; 3] = ["The a of is", "The c of is", "a b c"];

const POINTS: [P; 8] = [
    P::BeforeInstall,
    P::AfterCachesCommitted,
    P::AfterSlotStaged,
    P::AfterRefine,
    P::DuringBalance { iteration: 0 },
    P::DuringBalance { iteration: 1 },
    P::AfterBalance,
    P::BeforeRecord,
];

/// The components a failed install changes today, per fault point, when no earlier install exists
/// at the layer (so the decoy cache is new).
const CHANGED_FIRST: [&[&str]; 8] = [
    &[],
    &[
        "session.decoy_residual_cache",
        "session.raw_install_residuals",
    ],
    &[
        "overlay.counts",
        "overlay.vectors",
        "session.decoy_residual_cache",
        "session.raw_install_residuals",
    ],
    &[
        "overlay.counts",
        "overlay.vectors",
        "session.decoy_residual_cache",
        "session.raw_install_residuals",
    ],
    &[
        "overlay.counts",
        "overlay.vectors",
        "session.decoy_residual_cache",
        "session.raw_install_residuals",
    ],
    &[
        "overlay.counts",
        "overlay.vectors",
        "session.decoy_residual_cache",
        "session.raw_install_residuals",
    ],
    &[
        "overlay.counts",
        "overlay.vectors",
        "session.decoy_residual_cache",
        "session.raw_install_residuals",
    ],
    &[
        "overlay.counts",
        "overlay.vectors",
        "session.decoy_residual_cache",
        "session.installed_edges",
        "session.raw_install_residuals",
    ],
];

/// The same, when an earlier install already cached the layer's decoys.
const CHANGED_SECOND: [&[&str]; 8] = [
    &[],
    &["session.raw_install_residuals"],
    &[
        "overlay.counts",
        "overlay.vectors",
        "session.raw_install_residuals",
    ],
    &[
        "overlay.counts",
        "overlay.vectors",
        "session.raw_install_residuals",
    ],
    &[
        "overlay.counts",
        "overlay.vectors",
        "session.raw_install_residuals",
    ],
    &[
        "overlay.counts",
        "overlay.vectors",
        "session.raw_install_residuals",
    ],
    &[
        "overlay.counts",
        "overlay.vectors",
        "session.raw_install_residuals",
    ],
    &[
        "overlay.counts",
        "overlay.vectors",
        "session.installed_edges",
        "session.raw_install_residuals",
    ],
];

fn predictions(session: &mut larql_lql::Session) -> Vec<Vec<String>> {
    PROMPTS.iter().map(|p| infer(session, p)).collect()
}

/// A session in a patch recording, optionally with one earlier install, an armed fault, and the
/// result of the install that hits it.
struct Failed {
    changed: Vec<&'static str>,
    predictions_changed: bool,
    error: String,
}

fn fail_at(fixture: &Fixture, point: P, earlier_install: bool) -> Failed {
    let mut session = fixture.session();
    run(
        &mut session,
        &format!("BEGIN PATCH \"{}\";", q(&fixture.path("recording.vlp"))),
    );
    if earlier_install {
        run(&mut session, INSERT);
    }
    let before = session.state_fingerprint();
    let predictions_before = predictions(&mut session);
    session.set_install_fault(Some(point));
    let error = try_run(
        &mut session,
        if earlier_install {
            INSERT_SECOND
        } else {
            INSERT
        },
    )
    .unwrap_err();
    Failed {
        changed: session.state_fingerprint().changed_since(&before),
        predictions_changed: predictions_before != predictions(&mut session),
        error,
    }
}

#[test]
fn a_fault_before_the_install_writes_anything_changes_nothing() {
    let fixture = Fixture::new();
    for earlier in [false, true] {
        let failed = fail_at(&fixture, P::BeforeInstall, earlier);
        assert!(
            failed.error.contains("injected fault at before-install"),
            "{}",
            failed.error
        );
        assert!(failed.changed.is_empty(), "{:?}", failed.changed);
        assert!(!failed.predictions_changed);
    }
}

/// KNOWN DEFECT: see the module comment.
#[test]
fn known_defect_a_failed_install_leaves_the_session_changed_after_its_first_write() {
    let fixture = Fixture::new();
    for (earlier, table) in [(false, CHANGED_FIRST), (true, CHANGED_SECOND)] {
        for (point, expected) in POINTS.iter().zip(table) {
            let failed = fail_at(&fixture, *point, earlier);
            assert!(
                failed.error.contains(&format!("injected fault at {point}")),
                "{}",
                failed.error
            );
            assert_eq!(
                failed.changed,
                expected.to_vec(),
                "earlier install: {earlier}, fault at {point}"
            );
        }
    }
}

/// The requirement the defect violates. It fails today, so it is ignored; the fix makes it pass and
/// removes the attribute.
#[test]
#[ignore = "known defect: a failed COMPOSE install is not atomic; see the module comment"]
fn a_failed_install_changes_nothing() {
    let fixture = Fixture::new();
    for earlier in [false, true] {
        for point in POINTS {
            let failed = fail_at(&fixture, point, earlier);
            assert!(
                failed.changed.is_empty(),
                "earlier install: {earlier}, fault at {point}: {:?}",
                failed.changed
            );
            assert!(
                !failed.predictions_changed,
                "earlier install: {earlier}, fault at {point}"
            );
        }
    }
}

#[test]
fn known_defect_a_failed_install_can_change_predictions() {
    let fixture = Fixture::new();
    // With no earlier install, the half-written slot is already live in the forward pass.
    for point in [
        P::AfterSlotStaged,
        P::AfterRefine,
        P::DuringBalance { iteration: 0 },
        P::AfterBalance,
        P::BeforeRecord,
    ] {
        let failed = fail_at(&fixture, point, false);
        assert!(
            failed.predictions_changed,
            "fault at {point} left predictions unchanged"
        );
    }
    for point in [P::BeforeInstall, P::AfterCachesCommitted] {
        let failed = fail_at(&fixture, point, false);
        assert!(!failed.predictions_changed, "fault at {point}");
    }
}

/// KNOWN DEFECT: after a fault just before the patch operation is recorded, the slot is live in
/// the overlay and registered for later checks, yet the active recording has no operation for it.
/// `SAVE PATCH` would write a patch without an edit the session is still running.
#[test]
fn known_defect_the_overlay_can_hold_an_edit_the_saved_patch_lacks() {
    let fixture = Fixture::new();
    let mut session = fixture.session();
    let patch = fixture.path("recording.vlp");
    run(&mut session, &format!("BEGIN PATCH \"{}\";", q(&patch)));
    let before = session.state_fingerprint();
    session.set_install_fault(Some(P::BeforeRecord));
    try_run(&mut session, INSERT).unwrap_err();
    let changed = session.state_fingerprint().changed_since(&before);
    assert!(changed.contains(&"overlay.vectors") && changed.contains(&"session.installed_edges"));
    assert!(!changed.contains(&"session.patch_recording"));

    run(&mut session, "SAVE PATCH;");
    let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(&patch).unwrap()).unwrap();
    assert_eq!(
        saved["operations"].as_array().unwrap().len(),
        0,
        "the saved patch has no operation"
    );
}
