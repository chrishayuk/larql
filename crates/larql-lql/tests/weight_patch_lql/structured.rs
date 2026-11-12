//! `REMOVE DIRECTION`: a structured weight edit through LQL, from an operand file to the compiled bytes,
//! with no f32 delta anywhere on the path.

use larql_vindex::format::vindex3::weight_patch::bf16::patched_bits;
use larql_vindex::format::vindex3::weight_patch::read_bf16_tensor;
use larql_vindex::format::vindex3::weight_patch::structured::{
    run as run_op, BaseTensor, OpKind, OperandBits, ScalarBits, StructuredOp, TensorRole,
};

use crate::support::*;

const PROFILE: &str = "f64-seq-rne1/v1";
const COLUMN: usize = 33;
const F64_ONE: u64 = 0x3ff0_0000_0000_0000;
/// `1 + 2^-30`, as the decimal LQL parses it to exactly these f64 bits.
const SCALE_NEAR_ONE: &str = "1.000000000931322574615478515625";
const SCALE_NEAR_ONE_BITS: u64 = 0x3ff0_0000_0040_0000;
const DELTA_NEIGHBOURS: i64 = 8;

fn base(fx: &Fixture) -> ([usize; 2], Vec<u16>) {
    read_bf16_tensor(&fx.container, TENSOR).unwrap()
}

fn row_with_nonzero(base: &[u16], cols: usize) -> usize {
    (0..base.len() / cols)
        .find(|&r| base[r * cols + COLUMN] & 0x7fff != 0)
        .unwrap()
}

/// Write an operand file: `e_row` as f64 bits.
fn operand_file(fx: &Fixture, name: &str, rows: usize, row: usize, origin: Option<&str>) -> String {
    let bits: Vec<String> = (0..rows)
        .map(|r| format!("{:016x}", if r == row { F64_ONE } else { 0 }))
        .collect();
    let mut doc = serde_json::json!({ "bits": bits });
    if let Some(o) = origin {
        doc["origin"] = o.into();
    }
    let path = fx.path(name);
    std::fs::write(&path, doc.to_string()).unwrap();
    q(path)
}

fn remove(operand: &str, columns: &str, scale: &str) -> String {
    format!(
        "REMOVE DIRECTION \"{operand}\" FROM \"{TENSOR}\" COLUMNS ({columns}) SCALE {scale} NUMERICS \"{PROFILE}\";"
    )
}

fn compile(s: &mut larql_lql::Session, out: &std::path::Path) -> Vec<String> {
    run(s, &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(out)))
}

fn element(container: &std::path::Path, row: usize, col: usize) -> u16 {
    let ([_, cols], bits) = read_bf16_tensor(container, TENSOR).unwrap();
    bits[row * cols + col]
}

fn reachable_by_a_delta(old: u16, new: u16) -> bool {
    let f = |b: u16| f32::from_bits((b as u32) << 16);
    let centre = (f(new) - f(old)).to_bits() as i64;
    (-DELTA_NEIGHBOURS..=DELTA_NEIGHBOURS).any(|s| {
        let d = f32::from_bits((centre + s) as u32);
        d.is_finite() && patched_bits(old, d) == Some(new)
    })
}

#[test]
fn a_near_cancelling_removal_compiles_to_the_exact_result_no_delta_can_reach() {
    let fx = Fixture::new();
    let ([rows, cols], b) = base(&fx);
    let row = row_with_nonzero(&b, cols);
    let operand = operand_file(&fx, "u.json", rows, row, None);

    let mut s = session_on(&fx.container);
    let staged = run(
        &mut s,
        &remove(&operand, &COLUMN.to_string(), SCALE_NEAR_ONE),
    );
    assert!(
        staged[0].starts_with("Staged: remove direction"),
        "{staged:?}"
    );
    assert!(
        staged[0].contains("1 element(s) would change"),
        "{staged:?}"
    );
    assert!(staged
        .iter()
        .any(|l| l.contains("no f32 delta is involved")));

    let out = fx.path("out.vindex3");
    let compiled = compile(&mut s, &out);
    assert!(
        compiled.iter().any(|l| l.contains("weight-patches/v3")),
        "{compiled:?}"
    );

    // What the frozen interpreter says, computed here from the same operands.
    let want = run_op(
        &StructuredOp {
            numerics: Some(PROFILE.into()),
            shape: [rows, cols],
            kind: OpKind::RemoveDirection {
                u: OperandBits::F64(
                    (0..rows)
                        .map(|r| if r == row { F64_ONE } else { 0 })
                        .collect(),
                ),
                alpha: ScalarBits::F64(SCALE_NEAR_ONE_BITS),
                columns: vec![COLUMN],
            },
            fixups: vec![],
        },
        &BaseTensor {
            shape: [rows, cols],
            bits: &b,
            role: TensorRole::Ordinary,
        },
    )
    .unwrap();
    let got = element(&out, row, COLUMN);
    assert_eq!(got, want[row * cols + COLUMN]);
    assert_ne!(got & 0x7fff, 0, "a tiny nonzero");
    assert!(
        !reachable_by_a_delta(b[row * cols + COLUMN], got),
        "an UPDATE WEIGHTS delta could have written this, so the case proves nothing"
    );
    assert!(is_coherent(&out));
    assert_eq!(ledger(&out)["schema"], "weight-patches/v3");
    assert_eq!(
        ledger(&out)["targets"][0]["edits"][0]["encoding"],
        "result_bits"
    );
}

#[test]
fn a_saved_patch_applied_in_a_fresh_session_compiles_the_same_bytes() {
    let fx = Fixture::new();
    let ([rows, cols], b) = base(&fx);
    let row = row_with_nonzero(&b, cols);
    let operand = operand_file(&fx, "u.json", rows, row, Some("feature:rep@d41d8c"));
    let vlp = fx.path("s.vlp");

    let mut direct = session_on(&fx.container);
    run(
        &mut direct,
        &remove(&operand, &COLUMN.to_string(), SCALE_NEAR_ONE),
    );
    let direct_out = fx.path("direct.vindex3");
    compile(&mut direct, &direct_out);

    let mut writer = session_on(&fx.container);
    run(&mut writer, &format!("BEGIN PATCH \"{}\";", q(&vlp)));
    run(
        &mut writer,
        &remove(&operand, &COLUMN.to_string(), SCALE_NEAR_ONE),
    );
    run(&mut writer, "SAVE PATCH;");
    let text = std::fs::read_to_string(&vlp).unwrap();
    assert!(text.contains("\"structured_weight\""), "{text}");
    assert!(
        text.contains("feature:rep@d41d8c"),
        "the operand's origin is carried"
    );

    let mut fresh = session_on(&fx.container);
    let applied = run(&mut fresh, &format!("APPLY PATCH \"{}\";", q(&vlp)));
    assert!(
        applied[0].contains("1 structured weight op(s)"),
        "{applied:?}"
    );
    let patched_out = fx.path("patched.vindex3");
    compile(&mut fresh, &patched_out);

    let (a, c) = (digests(&direct_out), digests(&patched_out));
    for id in [CANONICAL, DERIVED] {
        assert_eq!(a[id], c[id], "{id}: the same bytes direct or by patch");
    }

    run(&mut fresh, &format!("REMOVE PATCH \"{}\";", q(&vlp)));
    let reverted = fx.path("reverted.vindex3");
    let error = try_run(
        &mut fresh,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(&reverted)),
    );
    assert!(
        error.is_err() || digests(&reverted) == digests(&fx.container),
        "with the patch removed nothing is left to compile, or the original comes back"
    );
}

#[test]
fn a_second_removal_on_a_shared_column_is_refused_and_a_disjoint_one_is_not() {
    let fx = Fixture::new();
    let ([rows, cols], b) = base(&fx);
    let row = row_with_nonzero(&b, cols);
    let operand = operand_file(&fx, "u.json", rows, row, None);
    let mut s = session_on(&fx.container);
    run(&mut s, &remove(&operand, "33, 34", SCALE_NEAR_ONE));
    let error = try_run(&mut s, &remove(&operand, "34", SCALE_NEAR_ONE)).unwrap_err();
    assert!(
        error.contains("column 34") || error.contains("column"),
        "{error}"
    );
    assert!(error.contains("REMOVE DIRECTION refused"), "{error}");
    run(&mut s, &remove(&operand, "40", SCALE_NEAR_ONE));
    let out = fx.path("two.vindex3");
    compile(&mut s, &out);
    assert!(is_coherent(&out));
}

#[test]
fn a_delta_edit_and_a_structured_edit_of_one_tensor_are_refused_either_way_round() {
    let fx = Fixture::new();
    let ([rows, cols], b) = base(&fx);
    let row = row_with_nonzero(&b, cols);
    let operand = operand_file(&fx, "u.json", rows, row, None);

    let mut first_delta = session_on(&fx.container);
    run(&mut first_delta, &update(&[40], DELTA));
    let error = try_run(&mut first_delta, &remove(&operand, "33", SCALE_NEAR_ONE)).unwrap_err();
    assert!(error.contains("one form"), "{error}");

    let mut first_structured = session_on(&fx.container);
    run(
        &mut first_structured,
        &remove(&operand, "33", SCALE_NEAR_ONE),
    );
    let error = try_run(&mut first_structured, &update(&[40], DELTA)).unwrap_err();
    assert!(error.contains("one form"), "{error}");
}

#[test]
fn a_removal_that_changes_nothing_is_refused_when_written_and_stages_nothing() {
    let fx = Fixture::new();
    let ([rows, _], _) = base(&fx);
    let operand = operand_file(&fx, "u.json", rows, 0, None);
    let mut s = session_on(&fx.container);
    let error = try_run(&mut s, &remove(&operand, "33", "0.0")).unwrap_err();
    assert!(error.contains("REMOVE DIRECTION refused"), "{error}");
    let out = fx.path("none.vindex3");
    assert!(
        try_run(
            &mut s,
            &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(&out))
        )
        .is_err()
            || !out.join("weight_patches.json").exists()
    );
}

#[test]
fn bad_inputs_are_refused_by_name_before_anything_is_staged() {
    let fx = Fixture::new();
    let ([rows, _], _) = base(&fx);
    let operand = operand_file(&fx, "u.json", rows, 0, None);
    let mut s = session_on(&fx.container);

    let unknown = format!(
        "REMOVE DIRECTION \"{operand}\" FROM \"{TENSOR}\" COLUMNS (33) SCALE 1.0 NUMERICS \"f64-seq-rne9/v1\";"
    );
    assert!(try_run(&mut s, &unknown).unwrap_err().contains("numerics"));

    let f32_profile = format!(
        "REMOVE DIRECTION \"{operand}\" FROM \"{TENSOR}\" COLUMNS (33) SCALE 1.0 NUMERICS \"f32-seq-rne1/v1\";"
    );
    assert!(try_run(&mut s, &f32_profile)
        .unwrap_err()
        .contains("f64-working"));

    let missing = remove(&q(fx.path("absent.json")), "33", "1.0");
    assert!(try_run(&mut s, &missing)
        .unwrap_err()
        .contains("cannot read operand file"));

    let bad = fx.path("bad.json");
    std::fs::write(&bad, r#"{"bits":["3ff0"]}"#).unwrap();
    assert!(try_run(&mut s, &remove(&q(&bad), "33", "1.0"))
        .unwrap_err()
        .contains("16 lowercase hex digits"));

    let extra = fx.path("extra.json");
    std::fs::write(&extra, r#"{"bits":[],"unknown":1}"#).unwrap();
    assert!(try_run(&mut s, &remove(&q(&extra), "33", "1.0")).is_err());

    let out_of_range = remove(&operand, "999999", "1.0");
    assert!(try_run(&mut s, &out_of_range).is_err());

    // Nothing above staged anything: a compile has nothing to write.
    let out = fx.path("none.vindex3");
    let compiled = try_run(
        &mut s,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(&out)),
    );
    assert!(compiled.is_err() || !out.join("weight_patches.json").exists());
}

#[test]
fn an_edit_staged_against_another_container_is_refused_on_apply() {
    let fx = Fixture::new();
    let ([rows, cols], b) = base(&fx);
    let row = row_with_nonzero(&b, cols);
    let operand = operand_file(&fx, "u.json", rows, row, None);
    let vlp = fx.path("s.vlp");
    let mut writer = session_on(&fx.container);
    run(&mut writer, &format!("BEGIN PATCH \"{}\";", q(&vlp)));
    run(
        &mut writer,
        &remove(&operand, &COLUMN.to_string(), SCALE_NEAR_ONE),
    );
    run(&mut writer, "SAVE PATCH;");

    let text = std::fs::read_to_string(&vlp).unwrap();
    let mut doc: serde_json::Value = serde_json::from_str(&text).unwrap();
    doc["operations"][0]["source"]["payload_sha256"] = "0".repeat(64).into();
    let altered = fx.path("altered.vlp");
    std::fs::write(&altered, doc.to_string()).unwrap();
    let mut fresh = session_on(&fx.container);
    let error = try_run(&mut fresh, &format!("APPLY PATCH \"{}\";", q(&altered))).unwrap_err();
    assert!(
        error.contains("digest") || error.contains("another artifact"),
        "{error}"
    );
}

#[test]
fn the_scale_literal_reaches_the_op_as_exactly_the_f64_it_names() {
    let parsed = larql_lql::parse(&format!(
        "REMOVE DIRECTION \"u.json\" FROM \"{TENSOR}\" COLUMNS (1) SCALE {SCALE_NEAR_ONE} NUMERICS \"{PROFILE}\";"
    ))
    .unwrap();
    let larql_lql::Statement::RemoveDirection { scale, .. } = parsed else {
        panic!("{parsed:?}");
    };
    assert_eq!(scale.to_bits(), SCALE_NEAR_ONE_BITS);
}

#[test]
fn an_integer_scale_is_a_scale() {
    for text in ["1", "0", "2"] {
        let parsed = larql_lql::parse(&format!(
            "REMOVE DIRECTION \"u.json\" FROM \"{TENSOR}\" COLUMNS (1) SCALE {text} NUMERICS \"{PROFILE}\";"
        ))
        .unwrap();
        let larql_lql::Statement::RemoveDirection { scale, .. } = parsed else {
            panic!("{parsed:?}");
        };
        assert_eq!(scale, text.parse::<f64>().unwrap());
    }
}

#[test]
fn the_statement_parses_with_a_signed_scale_and_a_column_list() {
    let parsed = larql_lql::parse(&format!(
        "REMOVE DIRECTION \"u.json\" FROM \"{TENSOR}\" COLUMNS (1, 4, 9) SCALE -0.5 NUMERICS \"{PROFILE}\";"
    ))
    .unwrap();
    match &parsed {
        larql_lql::Statement::RemoveDirection {
            columns,
            scale,
            numerics,
            tensor,
            operand,
        } => {
            assert_eq!(columns, &[1usize, 4, 9]);
            assert_eq!(*scale, -0.5);
            assert_eq!(numerics, PROFILE);
            assert_eq!(tensor, TENSOR);
            assert_eq!(operand, "u.json");
        }
        other => panic!("{other:?}"),
    }
    assert!(larql_lql::parse(
        "REMOVE DIRECTION \"u\" FROM \"t\" COLUMNS () SCALE 1 NUMERICS \"p\";"
    )
    .is_err());
    assert!(
        larql_lql::parse("REMOVE PATCH \"p.vlp\";").is_ok(),
        "REMOVE PATCH is unchanged"
    );
}

#[test]
fn an_integer_scale_too_large_to_be_exact_is_refused_not_rounded() {
    let text = format!(
        "REMOVE DIRECTION \"u.json\" FROM \"{TENSOR}\" COLUMNS (1) SCALE 9007199254740993 NUMERICS \"{PROFILE}\";"
    );
    assert!(larql_lql::parse(&text)
        .unwrap_err()
        .to_string()
        .contains("exactly"));
}
