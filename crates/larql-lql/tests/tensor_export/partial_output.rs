//! An export that fails while it is writing leaves no partial output behind.

use larql_vindex::format::vindex3::encode::segment::read_segment_header;

use crate::support::*;

/// Truncate the canonical segment so the matrix's bytes run out part-way: the header still reads, the read of the
/// tensor body fails after the `.npy` has been created.
fn truncate_inside_matrix(fx: &Fixture) {
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(fx.root.join("index.json")).unwrap())
            .unwrap();
    let segment = fx.root.join(
        index["representations"][CANONICAL]["segment"]
            .as_str()
            .unwrap(),
    );
    let (header, payload_start) = read_segment_header(&segment).unwrap();
    let t = header.tensors.iter().find(|t| t.name == MATRIX).unwrap();
    let cut = payload_start + t.offset + (t.len / 2);
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&segment)
        .unwrap();
    file.set_len(cut).unwrap();
}

#[test]
fn a_read_that_fails_part_way_removes_the_partial_npy_and_writes_no_sidecar() {
    let fx = Fixture::bf16();
    let mut s = fx.session();
    truncate_inside_matrix(&fx);
    let out = fx.path("partial.npy");
    let err = try_run(&mut s, &export(MATRIX, "", &out, "")).unwrap_err();
    assert!(err.contains("read the stored tensor"), "{err}");
    assert!(!out.exists(), "the partial .npy must be removed");
    assert!(
        !fx.path("partial.npy.json").exists(),
        "no sidecar for a failed export"
    );
}

#[test]
fn a_failed_export_leaves_the_name_free_for_a_later_one() {
    let fx = Fixture::bf16();
    let mut s = fx.session();
    truncate_inside_matrix(&fx);
    let out = fx.path("again.npy");
    assert!(try_run(&mut s, &export(MATRIX, "", &out, "")).is_err());
    // The refusal to overwrite must not fire on what a failed export would have left: nothing is left.
    let err = try_run(&mut s, &export(MATRIX, "", &out, "")).unwrap_err();
    assert!(!err.contains("refusing to write"), "{err}");
}
