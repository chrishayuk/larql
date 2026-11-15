//! What `EXPORT WEIGHT` and `SHOW TENSORS` refuse, and that a refusal writes nothing.

use larql_inference::test_utils::write_synthetic_model_dir;
use larql_lql::Session;

use crate::support::*;

#[test]
fn a_packed_quantised_tensor_is_refused() {
    let fx = Fixture::bf16();
    let mut s = fx.session();
    let out = fx.path("q8.npy");
    let err = try_run(
        &mut s,
        &export(&format!("{DERIVED}::{MATRIX}"), "", &out, ""),
    )
    .unwrap_err();
    assert!(err.contains("EXPORT WEIGHT refused"), "{err}");
    assert!(err.contains("Q8_0"), "{err}");
    assert!(!out.exists());
}

#[test]
fn unknown_names_are_refused() {
    let fx = Fixture::bf16();
    let mut s = fx.session();
    for name in ["9.mlp.down_proj.weight", "target.nothing@BF16::weight"] {
        let err = try_run(&mut s, &export(name, "", &fx.path("x.npy"), "")).unwrap_err();
        assert!(err.contains("SHOW TENSORS"), "{name}: {err}");
    }
}

#[test]
fn a_bad_selection_is_refused_and_writes_nothing() {
    let fx = Fixture::bf16();
    let mut s = fx.session();
    let out = fx.path("bad.npy");
    for selection in [
        " ROWS (64)",
        " COLUMNS (256)",
        " ROWS (1, 1)",
        " COLUMNS (3, 4, 3)",
    ] {
        let err = try_run(&mut s, &export(MATRIX, selection, &out, "")).unwrap_err();
        assert!(err.contains("EXPORT WEIGHT refused"), "{selection}: {err}");
        assert!(!out.exists(), "{selection}");
    }
    let err = try_run(&mut s, &export(GAIN, " COLUMNS (0)", &out, "")).unwrap_err();
    assert!(err.contains("no columns"), "{err}");
}

#[test]
fn an_existing_output_or_sidecar_is_never_overwritten() {
    let fx = Fixture::bf16();
    let mut s = fx.session();
    let out = fx.path("taken.npy");
    std::fs::write(&out, b"keep").unwrap();
    let err = try_run(&mut s, &export(MATRIX, "", &out, "")).unwrap_err();
    assert!(err.contains("refusing to write"), "{err}");
    assert_eq!(std::fs::read(&out).unwrap(), b"keep");

    let other = fx.path("free.npy");
    let side = fx.path("free.npy.json");
    std::fs::write(&side, b"keep").unwrap();
    let err = try_run(&mut s, &export(MATRIX, "", &other, "")).unwrap_err();
    assert!(err.contains("refusing to write"), "{err}");
    assert!(!other.exists());
    assert_eq!(std::fs::read(&side).unwrap(), b"keep");

    // A second export to the same file is refused: the first stands.
    let once = fx.path("once.npy");
    run(&mut s, &export(GAIN, "", &once, ""));
    let first = std::fs::read(&once).unwrap();
    assert!(try_run(&mut s, &export(MATRIX, "", &once, "")).is_err());
    assert_eq!(std::fs::read(&once).unwrap(), first);
}

#[test]
fn an_output_that_is_not_npy_is_refused() {
    let fx = Fixture::bf16();
    let mut s = fx.session();
    for name in ["down.bin", "down", ".npy"] {
        let out = fx.path(name);
        let err = try_run(&mut s, &export(MATRIX, "", &out, "")).unwrap_err();
        assert!(err.contains(".npy"), "{name}: {err}");
        assert!(!out.exists(), "{name}");
    }
}

#[test]
fn a_missing_output_directory_leaves_nothing() {
    let fx = Fixture::bf16();
    let mut s = fx.session();
    let out = fx.path("missing/dir/down.npy");
    assert!(try_run(&mut s, &export(MATRIX, "", &out, "")).is_err());
    assert!(!out.exists());
}

#[test]
fn both_statements_need_a_vindex3_container() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_model_dir(dir.path()).unwrap();
    let mut s = Session::new();
    run(&mut s, &format!("USE \"{}\";", q(dir.path())));
    let out = dir.path().join("w.npy");
    for stmt in ["SHOW TENSORS;".to_string(), export(MATRIX, "", &out, "")] {
        let err = try_run(&mut s, &stmt).unwrap_err();
        assert!(err.contains("not a VINDEX3 container"), "{stmt}: {err}");
    }
    assert!(!out.exists());
    let mut unbound = Session::new();
    for stmt in ["SHOW TENSORS;".to_string(), export(MATRIX, "", &out, "")] {
        let err = try_run(&mut unbound, &stmt).unwrap_err();
        assert!(err.contains("No backend loaded"), "{stmt}: {err}");
    }
}
