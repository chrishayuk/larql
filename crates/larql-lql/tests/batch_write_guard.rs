//! A batch writes nothing after a failure (D16 of WSP-001): a `COMPILE` that
//! follows a refused `APPLY PATCH` must not leave a container that looks like
//! the edited one.

use larql_lql::{parse, run_batch_outcome, EXIT_ERROR, EXIT_OK};
use larql_vindex::format::vindex3::fixtures::bf16_container_with_q8_pack;

const TENSOR: &str = "1.mlp.down_proj.weight";

fn q(p: &std::path::Path) -> String {
    p.display().to_string().replace('\\', "\\\\")
}

#[test]
fn a_compile_after_a_refused_apply_patch_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let container = bf16_container_with_q8_pack(dir.path());
    let bad = dir.path().join("corrupt.vlp");
    std::fs::write(&bad, "{ not a patch").unwrap();
    let out = dir.path().join("edited.vindex3");
    let o = run_batch_outcome(&format!(
        "USE \"{}\"; APPLY PATCH \"{}\"; COMPILE CURRENT INTO VINDEX \"{}\";",
        q(&container),
        q(&bad),
        q(&out)
    ));
    assert!(
        !out.exists(),
        "no container after a failed APPLY PATCH: {:?}",
        o.lines
    );
    assert_eq!(o.errors, 2, "{:?}", o.lines);
    assert_eq!(o.exit_code(), EXIT_ERROR);
    let refusal = o.lines.last().unwrap();
    assert!(refusal.starts_with("Error: refused `COMPILE`"), "{refusal}");
    assert!(refusal.contains("1 earlier statement(s)"), "{refusal}");
}

#[test]
fn every_write_after_a_failure_is_refused_and_counted() {
    let dir = tempfile::tempdir().unwrap();
    let container = bf16_container_with_q8_pack(dir.path());
    let (vlp, out) = (dir.path().join("p.vlp"), dir.path().join("o.vindex3"));
    let o = run_batch_outcome(&format!(
        "USE \"{c}\"; BEGIN PATCH \"{p}\"; \
         UPDATE WEIGHTS SET DELTA = 0.0625 WHERE TENSOR = \"{TENSOR}\" AND ROW = 1 AND COL = 1; \
         NOT A STATEMENT; SAVE PATCH; COMPILE CURRENT INTO VINDEX \"{o}\";",
        c = q(&container),
        p = q(&vlp),
        o = q(&out)
    ));
    assert!(!vlp.exists() && !out.exists(), "{:?}", o.lines);
    assert_eq!(
        o.errors, 3,
        "the parse failure and both refused writes: {:?}",
        o.lines
    );
}

#[test]
fn statements_that_write_nothing_still_run_after_a_failure() {
    let dir = tempfile::tempdir().unwrap();
    let container = bf16_container_with_q8_pack(dir.path());
    let o = run_batch_outcome(&format!(
        "NOT A STATEMENT; USE \"{}\"; SHOW PATCHES;",
        q(&container)
    ));
    assert_eq!(o.errors, 1, "{:?}", o.lines);
    assert!(
        o.lines.iter().any(|l| l.starts_with("Using:")),
        "{:?}",
        o.lines
    );
}

#[test]
fn with_no_failure_the_same_compile_writes_its_container() {
    let dir = tempfile::tempdir().unwrap();
    let container = bf16_container_with_q8_pack(dir.path());
    let out = dir.path().join("edited.vindex3");
    let o = run_batch_outcome(&format!(
        "USE \"{}\"; UPDATE WEIGHTS SET DELTA = 0.0625 WHERE TENSOR = \"{TENSOR}\" AND ROW = 1 AND COL = 1; \
         COMPILE CURRENT INTO VINDEX \"{}\";",
        q(&container),
        q(&out)
    ));
    assert_eq!(o.exit_code(), EXIT_OK, "{:?}", o.lines);
    assert!(out.join("weight_patches.json").exists());
}

#[test]
fn which_statements_write_an_artifact() {
    let writes = |s: &str| {
        parse(s)
            .unwrap_or_else(|e| panic!("{s}: {e}"))
            .writes_artifact()
    };
    for s in [
        "COMPILE CURRENT INTO VINDEX \"o\";",
        "SAVE PATCH;",
        "COMPACT INTO VINDEX \"o\";",
        "DIFF \"a\" \"b\" INTO PATCH \"p.vlp\";",
        "TRACE \"x\" SAVE \"t.trace\";",
        "EXTRACT MODEL \"m\" INTO \"o.vindex\";",
        "STATS |> SAVE PATCH;",
    ] {
        assert!(writes(s), "{s} writes");
    }
    for s in [
        "STATS;",
        "SHOW PATCHES;",
        "APPLY PATCH \"p.vlp\";",
        "BEGIN PATCH \"p.vlp\";",
        "DIFF \"a\" \"b\";",
        "TRACE \"x\";",
        "STATS |> SHOW MODELS;",
        "UPDATE WEIGHTS SET DELTA = 0.5 WHERE TENSOR = \"t\" AND ROW = 0 AND COL = 0;",
    ] {
        assert!(!writes(s), "{s} writes nothing");
    }
}
