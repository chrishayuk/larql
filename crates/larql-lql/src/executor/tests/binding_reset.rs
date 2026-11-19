//! Gate B, B1: `reset_binding_state` gives every field the previous binding
//! built its fresh value, and keeps the session's own configuration.

use super::*;

use crate::executor::{InstalledEdge, PatchRecording};

/// A session whose every binding-derived field holds something.
fn dirty() -> Session {
    let mut session = Session::new();
    session.patch_recording = Some(PatchRecording {
        path: String::new(),
        operations: vec![larql_vindex::PatchOp::Delete {
            layer: 0,
            feature: 0,
            reason: None,
        }],
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
    let line = session.reset_binding_state();
    assert_eq!(session.state_fingerprint(), fresh);
    let line = line.expect("an open recording is named");
    assert!(line.contains("the anonymous recording"), "{line}");
    assert!(line.contains("1 unsaved operation(s)"), "{line}");
}

#[test]
fn the_reset_keeps_the_sessions_own_configuration() {
    let mut session = dirty();
    session.set_verifier("verifier-path");
    session.contract_violations = 4;
    session.set_representation_source(RepresentationSource::Auto);
    assert!(session.reset_binding_state().is_some());
    assert_eq!(
        session.verifier.as_deref(),
        Some(std::path::Path::new("verifier-path"))
    );
    assert_eq!(session.contract_violations, 4);
    assert_eq!(session.representation_source(), RepresentationSource::Auto);
    assert!(Session::new().reset_binding_state().is_none());
}
