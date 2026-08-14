//! The exec verb's resume plumbing: the plane scan that decides where a
//! run restarts, the plane reader that rebuilds the interpreter's state,
//! and the sidecar match that refuses to splice two different runs.

use std::path::Path;

use crate::commands::primary::shannon_trace::dump::{plane_name, MANIFEST_NAME};

use super::super::exec::{
    last_complete_plane, prepare_resume, read_plane, ResumeSidecar, RESUME_NAME,
};

/// Small, deliberately awkward geometry — nothing divides anything.
const SEQ: usize = 3;
const HIDDEN: usize = 5;
const TOTAL_LAYERS: usize = 4;

/// Write plane `index` holding `value` at every element; `truncate`
/// drops the final byte, modelling a run killed mid-write.
fn write_fixture_plane(dir: &Path, index: usize, value: f32, truncate: bool) {
    let mut bytes: Vec<u8> = (0..SEQ * HIDDEN)
        .flat_map(|_| value.to_le_bytes())
        .collect();
    if truncate {
        bytes.pop();
    }
    std::fs::write(dir.join(plane_name(index)), bytes).unwrap();
}

fn sidecar() -> ResumeSidecar {
    ResumeSidecar {
        engine: "vindex3-test".to_string(),
        container: "container".to_string(),
        component: "target".to_string(),
        token_ids: vec![1, 2, 3],
    }
}

#[test]
fn the_plane_scan_stops_before_a_truncated_file() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture_plane(dir.path(), 0, 0.0, false);
    write_fixture_plane(dir.path(), 1, 1.0, false);
    write_fixture_plane(dir.path(), 2, 2.0, true);
    write_fixture_plane(dir.path(), 3, 3.0, false);
    // Plane 2 is cut off, so the run resumes from plane 1 — and plane 3,
    // though intact, is unreachable because the state feeding it is not
    // trustworthy.
    assert_eq!(
        last_complete_plane(dir.path(), SEQ, HIDDEN, TOTAL_LAYERS),
        Some(1)
    );
}

#[test]
fn the_plane_scan_reports_nothing_for_an_empty_directory() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(
        last_complete_plane(dir.path(), SEQ, HIDDEN, TOTAL_LAYERS),
        None
    );
}

#[test]
fn a_plane_round_trips_through_the_reader() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture_plane(dir.path(), 1, 2.5, false);
    let rows = read_plane(&dir.path().join(plane_name(1)), SEQ, HIDDEN).unwrap();
    assert_eq!(rows.len(), SEQ);
    assert!(rows.iter().all(|r| r.len() == HIDDEN));
    assert!(rows.iter().flatten().all(|&v| v == 2.5));
}

#[test]
fn the_reader_refuses_a_wrong_sized_plane() {
    let dir = tempfile::tempdir().unwrap();
    write_fixture_plane(dir.path(), 1, 1.0, true);
    let err = read_plane(&dir.path().join(plane_name(1)), SEQ, HIDDEN).unwrap_err();
    assert!(err.to_string().contains("expected"), "{err}");
}

#[test]
fn resume_refuses_a_mismatched_sidecar() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join(RESUME_NAME),
        serde_json::to_string(&sidecar()).unwrap(),
    )
    .unwrap();
    let mut other = sidecar();
    other.token_ids.push(4);
    let err = prepare_resume(dir.path(), &other, SEQ, HIDDEN, TOTAL_LAYERS).unwrap_err();
    assert!(err.to_string().contains("refusing to splice"), "{err}");
}

#[test]
fn resume_refuses_a_completed_dump() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(MANIFEST_NAME), "{}").unwrap();
    std::fs::write(
        dir.path().join(RESUME_NAME),
        serde_json::to_string(&sidecar()).unwrap(),
    )
    .unwrap();
    let err = prepare_resume(dir.path(), &sidecar(), SEQ, HIDDEN, TOTAL_LAYERS).unwrap_err();
    assert!(err.to_string().contains("already complete"), "{err}");
}

#[test]
fn resume_refuses_when_no_run_was_started() {
    let dir = tempfile::tempdir().unwrap();
    let err = prepare_resume(dir.path(), &sidecar(), SEQ, HIDDEN, TOTAL_LAYERS).unwrap_err();
    assert!(err.to_string().contains("no resume record"), "{err}");
}

#[test]
fn resume_state_is_the_last_complete_plane() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join(RESUME_NAME),
        serde_json::to_string(&sidecar()).unwrap(),
    )
    .unwrap();
    write_fixture_plane(dir.path(), 0, 0.5, false);
    write_fixture_plane(dir.path(), 1, 1.5, false);
    let point = prepare_resume(dir.path(), &sidecar(), SEQ, HIDDEN, TOTAL_LAYERS)
        .unwrap()
        .expect("two complete planes must yield a resume point");
    assert_eq!(point.next_layer, 1);
    assert!(point.hidden.iter().flatten().all(|&v| v == 1.5));
}
