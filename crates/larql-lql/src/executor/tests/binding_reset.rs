//! Gate B, B1: a bind refuses over unsaved operations, and
//! `reset_binding_state` gives every field the previous binding
//! built its fresh value, and keeps the session's own configuration.

use super::*;

use crate::executor::{InstalledEdge, PatchRecording};

/// A session whose every binding-derived field holds something.
fn dirty() -> Session {
    let mut session = Session::new();
    // An empty anonymous recording: a bind is refused before the reset over
    // one that holds operations.
    session.patch_recording = Some(PatchRecording {
        path: String::new(),
        operations: Vec::new(),
    });
    session.auto_patch = true;
    session
        .decoy_residual_cache
        .insert(0, vec![larql_vindex::ndarray::Array1::from(vec![1.0f32])]);
    session
        .raw_install_residuals
        .insert((0, 1), larql_vindex::ndarray::Array1::from(vec![2.0f32]));
    session.installed_edges.push(InstalledEdge {
        layer: 0,
        feature: 1,
        canonical_prompt: "The b of a is".into(),
        target: "[5]".into(),
        target_id: 5,
    });
    *session.relation_resolver.borrow_mut() = Some((std::path::PathBuf::from("old.vindex"), None));
    session.epoch = 3;
    session.mutations_since_minor = 2;
    session.mutations_since_major = 1;
    session
}

#[test]
fn the_reset_leaves_the_fingerprint_of_a_new_session() {
    let mut session = dirty();
    let fresh = Session::new().state_fingerprint();
    assert!(!session.state_fingerprint().changed_since(&fresh).is_empty());
    session.reset_binding_state();
    assert_eq!(session.state_fingerprint(), fresh);
}

#[test]
fn a_rebind_is_refused_only_over_unsaved_operations() {
    let mut session = Session::new();
    assert!(session.refuse_rebind_over_unsaved_recording("USE").is_ok());
    session.patch_recording = Some(PatchRecording {
        path: "named.vlp".into(),
        operations: Vec::new(),
    });
    assert!(session.refuse_rebind_over_unsaved_recording("USE").is_ok());
    session.patch_recording = Some(PatchRecording {
        path: String::new(),
        operations: vec![larql_vindex::PatchOp::Delete {
            layer: 0,
            feature: 0,
            reason: None,
        }],
    });
    let err = session
        .refuse_rebind_over_unsaved_recording("EXTRACT")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("EXTRACT refused: the anonymous recording holds 1"),
        "{err}"
    );
}

#[test]
fn the_reset_keeps_the_sessions_own_configuration() {
    let mut session = dirty();
    session.set_verifier("verifier-path");
    session.contract_violations = 4;
    session.set_representation_source(RepresentationSource::Auto);
    session.reset_binding_state();
    assert_eq!(
        session.verifier.as_deref(),
        Some(std::path::Path::new("verifier-path"))
    );
    assert_eq!(session.contract_violations, 4);
    assert_eq!(session.representation_source(), RepresentationSource::Auto);
}
