//! A chained compile: editing the weights of a container that is itself the output of a weight compile (Gate B,
//! B7 r2, row S-6, ruling D-3 = c). B7 does not verify that path. These tests record what the engine does with it,
//! so the acceptance record can say whether the capability exists outside the verified boundary. They
//! characterise; they do not endorse.
//!
//! Observed at `3406037d`: **no chained edit reaches a container**, but the refusal is not made in one place:
//!
//! | Entry point on a weight-patched container | When it is written | `COMPILE` |
//! |---|---|---|
//! | `UPDATE WEIGHTS` | refused (chained) | carries the container unchanged |
//! | `REMOVE DIRECTION`, `ADD LOW RANK` | **staged**, reported not executable | refused (chained), nothing written |
//! | `APPLY PATCH` of a patch bound to the base | refused (bound to another artifact) | carries the container unchanged |
//! | `BEGIN PATCH` … `ADD LOW RANK` … `SAVE PATCH` | **saved**, bound to the patched container | — |
//! | `APPLY PATCH` of that patch | **applied**, pending | refused (chained), nothing written |
//!
//! The structured rows are a finding (B7 r2 §5.4), kept as evidence and not changed here: the structured ops are
//! admitted when written and refused only at `COMPILE`, where `UPDATE WEIGHTS` is refused when written, and a
//! `.vlp` bound to a patched container can be recorded although no compile can ever take it. FACT-WSP-1 must
//! refuse this path itself (S-6), not rely on `COMPILE`.

use std::path::Path;

use larql_vindex::format::vindex3::weight_patch::read_bf16_tensor;

use crate::support::*;

const PROFILE: &str = "f64-seq-rne1/v1";
const F64_ONE: u64 = 0x3ff0_0000_0000_0000;
/// Columns the first compile does not edit, so no refusal below is a footprint conflict.
const OTHER_COLS: [usize; 2] = [50, 51];
/// What every refusal of a chained edit says.
const CHAINED_REFUSAL: &str = "already carries a weight-patch ledger";

fn compile(s: &mut larql_lql::Session, out: &Path) -> Vec<String> {
    run(s, &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(out)))
}

fn cols(columns: &[usize]) -> String {
    columns
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// A rank-one f64 operand file over `columns`: `a` is one at row 0, `b` is one everywhere.
fn low_rank_operands(fx: &Fixture, rows: usize, columns: &[usize]) -> String {
    let hex = |v: Vec<u64>| v.iter().map(|b| format!("{b:016x}")).collect::<Vec<_>>();
    let a = (0..rows)
        .map(|r| if r == 0 { F64_ONE } else { 0 })
        .collect();
    let doc = serde_json::json!({ "rank": 1, "a": hex(a), "b": hex(vec![F64_ONE; columns.len()]) });
    let path = fx.path("a_b.json");
    std::fs::write(&path, doc.to_string()).unwrap();
    q(path)
}

/// A direction file `u` that is one at row 0.
fn direction(fx: &Fixture, rows: usize) -> String {
    let bits: Vec<String> = (0..rows)
        .map(|r| format!("{:016x}", if r == 0 { F64_ONE } else { 0 }))
        .collect();
    let path = fx.path("u.json");
    std::fs::write(&path, serde_json::json!({ "bits": bits }).to_string()).unwrap();
    q(path)
}

/// The four ways an edit is written on `first`, each on columns the first compile left alone.
fn attempts(fx: &Fixture, first: &Path) -> [String; 4] {
    let ([rows, _], _) = read_bf16_tensor(first, TENSOR).unwrap();
    let patch = save_patch(fx, "other.vlp", &OTHER_COLS, DELTA);
    [
        update(&OTHER_COLS, DELTA),
        format!(
            "REMOVE DIRECTION \"{}\" FROM \"{TENSOR}\" COLUMNS ({}) SCALE 1.0 NUMERICS \"{PROFILE}\";",
            direction(fx, rows),
            cols(&OTHER_COLS)
        ),
        format!(
            "ADD LOW RANK \"{}\" TO \"{TENSOR}\" COLUMNS ({}) NUMERICS \"{PROFILE}\";",
            low_rank_operands(fx, rows, &OTHER_COLS),
            cols(&OTHER_COLS)
        ),
        format!("APPLY PATCH \"{}\";", q(&patch)),
    ]
}

/// A weight-compiled container, from the standard delta edit.
fn first_compile(fx: &Fixture) -> std::path::PathBuf {
    let first = fx.path("first.vindex3");
    let mut s = session_on(&fx.container);
    run(&mut s, &update(&COLS, DELTA));
    compile(&mut s, &first);
    first
}

/// After whatever `s` holds, `COMPILE` either refuses as chained and writes nothing, or (`carries`) writes a
/// container identical in ledger and weights to `first`.
fn assert_compile(
    s: &mut larql_lql::Session,
    fx: &Fixture,
    first: &Path,
    carries: bool,
    what: &str,
) {
    let out = fx.path("out.vindex3");
    let _ = std::fs::remove_dir_all(&out);
    let compiled = try_run(s, &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(&out)));
    if carries {
        compiled.unwrap_or_else(|e| panic!("{what}: compiles: {e}"));
        assert_eq!(
            ledger(&out),
            ledger(first),
            "{what}: the ledger is carried, not rewritten"
        );
        assert_eq!(
            tensor_bytes(&out, CANONICAL, TENSOR),
            tensor_bytes(first, CANONICAL, TENSOR),
            "{what}: the weights are carried"
        );
    } else {
        let refused = compiled.expect_err(what);
        assert!(
            refused.contains(CHAINED_REFUSAL),
            "{what}: refused as chained: {refused}"
        );
        assert!(!out.exists(), "{what}: nothing is written");
    }
}

#[test]
fn no_chained_edit_reaches_a_container_through_any_entry_point() {
    let fx = Fixture::new();
    let first = first_compile(&fx);
    let [update_weights, remove_direction, add_low_rank, apply_base_patch] = attempts(&fx, &first);

    let mut s = session_on(&first);
    let refused = try_run(&mut s, &update_weights).expect_err("UPDATE WEIGHTS");
    assert!(
        refused.contains(CHAINED_REFUSAL),
        "UPDATE WEIGHTS: {refused}"
    );
    assert_compile(&mut s, &fx, &first, true, "after a refused UPDATE WEIGHTS");

    for structured in [&remove_direction, &add_low_rank] {
        let mut s = session_on(&first);
        let staged = run(&mut s, structured);
        assert!(
            staged
                .iter()
                .any(|l| l.contains("NOT executable") && l.contains(CHAINED_REFUSAL)),
            "{structured}: staged, and reported as not executable: {staged:?}"
        );
        assert_compile(&mut s, &fx, &first, false, structured);
    }

    let mut s = session_on(&first);
    let refused = try_run(&mut s, &apply_base_patch).expect_err("APPLY PATCH");
    assert!(
        refused.contains("written against another artifact"),
        "APPLY PATCH: {refused}"
    );
    assert_compile(&mut s, &fx, &first, true, "after a refused APPLY PATCH");
}

#[test]
fn a_patch_recorded_on_a_weight_patched_container_saves_and_applies_but_never_compiles() {
    let fx = Fixture::new();
    let first = first_compile(&fx);
    let [_, _, add_low_rank, _] = attempts(&fx, &first);

    let own = fx.path("own.vlp");
    let mut s = session_on(&first);
    run(&mut s, &format!("BEGIN PATCH \"{}\";", q(&own)));
    run(&mut s, &add_low_rank);
    run(&mut s, "SAVE PATCH;");
    assert!(own.is_file(), "the patch is saved");

    let mut s = session_on(&first);
    run(&mut s, &format!("APPLY PATCH \"{}\";", q(&own)));
    assert_compile(&mut s, &fx, &first, false, "the applied patch");
}
