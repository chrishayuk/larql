//! Gate B, B6 (`docs/gate-b-b6.md` r2, R1 and D4): two independent sessions
//! that reach the same state write byte-identical output trees, for every
//! COMPILE target and for COMPACT INTO VINDEX.
//!
//! The comparison is over the whole tree: the same relative paths, the same
//! file kinds, and the same bytes in every file. Nothing is masked; the
//! permitted-nondeterminism list is empty (D4).
//!
//! `COMPILE INTO MODEL` is not covered: on `741ecdab` it fails for every
//! input before it writes (its weight writer expects an `index.json` in the
//! fresh output directory), so there is no output to compare. By owner
//! ruling a successful export is out of B6's scope; its failure must change
//! nothing, which `into_model.rs` checks.

use crate::support::*;

#[test]
fn vindex2_compile_into_vindex_is_byte_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let source = v2_vindex(dir.path());
    assert_two_sessions_agree(
        &source,
        &[V2_INSERT],
        "COMPILE CURRENT INTO VINDEX",
        dir.path(),
    );
}

#[test]
fn vindex2_compile_into_vindex_of_an_unedited_session_is_byte_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let source = v2_vindex(dir.path());
    assert_two_sessions_agree(&source, &[], "COMPILE CURRENT INTO VINDEX", dir.path());
}

#[test]
fn vindex3_compile_into_vindex_is_byte_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let source = v3_container(dir.path());
    assert_two_sessions_agree(
        &source,
        &[V3_INSERT],
        "COMPILE CURRENT INTO VINDEX",
        dir.path(),
    );
}

#[test]
fn vindex3_compact_into_vindex_is_byte_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let source = v3_container(dir.path());
    assert_two_sessions_agree(&source, &[], "COMPACT INTO VINDEX", dir.path());
}

#[test]
fn vindex2_compile_of_knn_facts_at_several_layers_is_byte_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let source = v2_vindex(dir.path());
    assert_sessions_agree(&source, V2_KNN, "COMPILE CURRENT INTO VINDEX", dir.path());
}

#[test]
fn vindex3_compile_of_knn_facts_at_several_layers_is_byte_deterministic() {
    let dir = tempfile::tempdir().unwrap();
    let source = v3_container(dir.path());
    assert_sessions_agree(&source, V3_KNN, "COMPILE CURRENT INTO VINDEX", dir.path());
}
