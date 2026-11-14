//! `larql compile --vindex`: a patch holding an op the mode cannot install is refused whole, through the real
//! binary. It must not compile the inserts, drop the rest and report success.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A committed `.vlp` holding structured stored-weight ops (the verifier's conformance patch).
const WEIGHT_OP_PATCH: &str = "../../verifier/conformance/structured_patch/patch.vlp";
/// A base that does not exist: the refusal must come before the model is loaded.
const ABSENT_BASE: &str = "no-such-base-model";

fn compile(patch: &Path, out: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_larql"))
        .args(["compile", "--base", ABSENT_BASE, "--vindex"])
        .arg(patch)
        .arg("--output")
        .arg(out)
        .output()
        .expect("runs larql")
}

fn said(o: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&o.stdout),
        String::from_utf8_lossy(&o.stderr)
    )
}

fn refused_whole(o: &Output, out: &Path, kind: &str) {
    let text = said(o);
    assert!(!o.status.success(), "{text}");
    assert!(text.contains("cannot compile the other ops"), "{text}");
    assert!(text.contains(kind), "{text}");
    assert!(text.contains("nothing was written"), "{text}");
    assert!(!out.exists(), "a refused compile writes nothing");
}

#[test]
fn a_patch_of_stored_weight_ops_is_refused_not_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("compiled");
    let patch = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(WEIGHT_OP_PATCH);
    let o = compile(&patch, &out);
    refused_whole(&o, &out, "structured_weight");
    assert!(
        said(&o).contains("COMPILE CURRENT INTO VINDEX"),
        "{}",
        said(&o)
    );
}

#[test]
fn an_insert_beside_an_op_the_mode_cannot_install_is_refused_too() {
    let dir = tempfile::tempdir().unwrap();
    let patch = dir.path().join("mixed.vlp");
    std::fs::write(
        &patch,
        r#"{"version":1,"base_model":"m","created_at":"","operations":[
            {"op":"insert","layer":0,"feature":1,"entity":"e","relation":"r","target":"t"},
            {"op":"delete","layer":0,"feature":2}
        ]}"#,
    )
    .unwrap();
    let out = dir.path().join("compiled");
    let o = compile(&patch, &out);
    refused_whole(&o, &out, "1 delete");
}
