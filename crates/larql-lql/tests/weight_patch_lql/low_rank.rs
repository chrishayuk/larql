//! `ADD LOW RANK`: a structured `add_low_rank_columns` (VLP numerics v2) through LQL, from an operand file to
//! the compiled bytes, recorded in a `.vlp` and held beside `REMOVE DIRECTION` by the column-footprint rule.

use larql_vindex::format::vindex3::weight_patch::read_bf16_tensor;
use larql_vindex::format::vindex3::weight_patch::structured::{
    run as run_op, BaseTensor, OpKind, OperandBits, StructuredOp, TensorRole,
};

use crate::support::*;

const PROFILE: &str = "f64-seq-rne1/v1";
const F32_PROFILE: &str = "f32-seq-rne1/v1";
const F64_EIGHTH: u64 = 0x3fc0_0000_0000_0000;
const F64_ONE: u64 = 0x3ff0_0000_0000_0000;
const F32_ONE: u32 = 0x3f80_0000;
/// The edited columns, as LQL lists them and as the operand file's `b` is laid out.
const COLUMNS: [usize; 3] = [8, 9, 33];
/// The row whose `a` factor is zero: no element of it may change.
const SILENT_ROW: usize = 2;
const ORIGIN: &str = "alpha=1/8 u=ones-but-row-2 v=ones";

fn columns_text(columns: &[usize]) -> String {
    columns
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

fn a_bits(rows: usize) -> Vec<u64> {
    (0..rows)
        .map(|r| if r == SILENT_ROW { 0 } else { F64_EIGHTH })
        .collect()
}

/// Write a rank-one operand file over `columns` in the f64 profile.
fn operand_file(fx: &Fixture, name: &str, columns: &[usize], origin: Option<&str>) -> String {
    let ([rows, _], _) = read_bf16_tensor(&fx.container, TENSOR).unwrap();
    let hex = |v: &[u64]| v.iter().map(|b| format!("{b:016x}")).collect::<Vec<_>>();
    let mut doc = serde_json::json!({
        "rank": 1,
        "a": hex(&a_bits(rows)),
        "b": hex(&vec![F64_ONE; columns.len()]),
    });
    if let Some(o) = origin {
        doc["origin"] = o.into();
    }
    let path = fx.path(name);
    std::fs::write(&path, doc.to_string()).unwrap();
    q(path)
}

fn add(operands: &str, columns: &[usize], numerics: &str) -> String {
    format!(
        "ADD LOW RANK \"{operands}\" TO \"{TENSOR}\" COLUMNS ({}) NUMERICS \"{numerics}\";",
        columns_text(columns)
    )
}

fn remove(operand: &str, columns: &str) -> String {
    format!(
        "REMOVE DIRECTION \"{operand}\" FROM \"{TENSOR}\" COLUMNS ({columns}) SCALE 1.0 NUMERICS \"{PROFILE}\";"
    )
}

fn direction_file(fx: &Fixture, rows: usize) -> String {
    let bits: Vec<String> = (0..rows)
        .map(|r| format!("{:016x}", if r == 0 { F64_ONE } else { 0 }))
        .collect();
    let path = fx.path("u.json");
    std::fs::write(&path, serde_json::json!({ "bits": bits }).to_string()).unwrap();
    q(path)
}

fn compile(s: &mut larql_lql::Session, out: &std::path::Path) -> Vec<String> {
    run(s, &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(out)))
}

#[test]
fn the_statement_parses_into_its_operand_file_tensor_columns_and_profile() {
    let parsed = larql_lql::parse(
        "add low rank \"ab.json\" to \"t\" columns (1, 2, 40) numerics \"f64-seq-rne1/v1\";",
    )
    .unwrap();
    let larql_lql::Statement::AddLowRank {
        operands,
        tensor,
        columns,
        numerics,
    } = parsed
    else {
        panic!("parses as ADD LOW RANK, got {parsed:?}");
    };
    assert_eq!(
        (
            operands.as_str(),
            tensor.as_str(),
            columns,
            numerics.as_str()
        ),
        ("ab.json", "t", vec![1, 2, 40], "f64-seq-rne1/v1")
    );
    assert!(
        larql_lql::parse("ADD LOW RANK \"ab.json\" TO \"t\" COLUMNS () NUMERICS \"p\";").is_err()
    );
    assert!(larql_lql::parse("ADD LOW \"ab.json\" TO \"t\" COLUMNS (1) NUMERICS \"p\";").is_err());
}

#[test]
fn an_added_update_compiles_to_exactly_what_the_frozen_interpreter_gives() {
    let fx = Fixture::new();
    let ([rows, cols], base) = read_bf16_tensor(&fx.container, TENSOR).unwrap();
    let operands = operand_file(&fx, "ab.json", &COLUMNS, None);

    let mut s = session_on(&fx.container);
    let staged = run(&mut s, &add(&operands, &COLUMNS, PROFILE));
    assert!(
        staged[0].starts_with("Staged: add rank-1 update"),
        "{staged:?}"
    );

    let out = fx.path("out.vindex3");
    let compiled = compile(&mut s, &out);
    assert!(
        compiled.iter().any(|l| l.contains("weight-patches/v4")),
        "{compiled:?}"
    );
    assert!(is_coherent(&out));

    let want = run_op(
        &StructuredOp {
            numerics: Some(PROFILE.into()),
            shape: [rows, cols],
            kind: OpKind::AddLowRankColumns {
                rank: 1,
                a: OperandBits::F64(a_bits(rows)),
                b: OperandBits::F64(vec![F64_ONE; COLUMNS.len()]),
                columns: COLUMNS.to_vec(),
            },
            fixups: vec![],
        },
        &BaseTensor {
            shape: [rows, cols],
            bits: &base,
            role: TensorRole::Ordinary,
        },
    )
    .unwrap();
    let ([_, _], got) = read_bf16_tensor(&out, TENSOR).unwrap();
    assert_eq!(got, want);
    for c in 0..cols {
        assert_eq!(
            got[SILENT_ROW * cols + c],
            base[SILENT_ROW * cols + c],
            "the row with a zero a-factor is untouched"
        );
    }
    let changed = (0..rows * cols).filter(|&i| got[i] != base[i]).count();
    assert!(
        staged[0].contains(&format!("{changed} element(s) would change")),
        "{staged:?}"
    );
    // Recorded by its operation, not its elements.
    let target = &ledger(&out)["targets"][0];
    assert!(target.get("edits").is_none(), "{target}");
    assert_eq!(target["operations"][0]["op"], "low_rank_columns_weight");
    assert_eq!(target["changed_elements"], changed);
}

#[test]
fn a_saved_patch_carries_its_own_tag_and_origin_and_compiles_the_same_bytes_in_a_fresh_session() {
    let fx = Fixture::new();
    let operands = operand_file(&fx, "ab.json", &COLUMNS, Some(ORIGIN));
    let vlp = fx.path("lr.vlp");

    let mut direct = session_on(&fx.container);
    run(&mut direct, &add(&operands, &COLUMNS, PROFILE));
    let direct_out = fx.path("direct.vindex3");
    compile(&mut direct, &direct_out);

    let mut writer = session_on(&fx.container);
    run(&mut writer, &format!("BEGIN PATCH \"{}\";", q(&vlp)));
    run(&mut writer, &add(&operands, &COLUMNS, PROFILE));
    run(&mut writer, "SAVE PATCH;");
    let text = std::fs::read_to_string(&vlp).unwrap();
    assert!(text.contains("\"low_rank_columns_weight\""), "{text}");
    assert!(text.contains(ORIGIN), "the operands' origin is carried");

    let mut fresh = session_on(&fx.container);
    let applied = run(&mut fresh, &format!("APPLY PATCH \"{}\";", q(&vlp)));
    assert!(
        applied[0].contains("1 structured weight op(s)"),
        "{applied:?}"
    );
    let patched_out = fx.path("patched.vindex3");
    compile(&mut fresh, &patched_out);
    let (a, b) = (digests(&direct_out), digests(&patched_out));
    for id in [CANONICAL, DERIVED] {
        assert_eq!(a[id], b[id], "{id}: the same bytes direct or by patch");
    }
}

#[test]
fn it_shares_no_column_with_a_removal_and_compiles_beside_a_disjoint_one() {
    let fx = Fixture::new();
    let ([rows, _], _) = read_bf16_tensor(&fx.container, TENSOR).unwrap();
    let operands = operand_file(&fx, "ab.json", &COLUMNS, None);
    let direction = direction_file(&fx, rows);

    let mut s = session_on(&fx.container);
    run(&mut s, &add(&operands, &COLUMNS, PROFILE));
    let error = try_run(&mut s, &remove(&direction, "9")).unwrap_err();
    assert!(error.contains("column"), "{error}");

    let mut other_way = session_on(&fx.container);
    run(&mut other_way, &remove(&direction, "9"));
    let error = try_run(&mut other_way, &add(&operands, &COLUMNS, PROFILE)).unwrap_err();
    assert!(error.contains("ADD LOW RANK refused"), "{error}");

    run(&mut s, &remove(&direction, "40"));
    let out = fx.path("both.vindex3");
    compile(&mut s, &out);
    assert!(is_coherent(&out));
}

#[test]
fn a_delta_edit_and_a_low_rank_edit_of_one_tensor_are_refused_either_way_round() {
    let fx = Fixture::new();
    let operands = operand_file(&fx, "ab.json", &COLUMNS, None);

    let mut first_delta = session_on(&fx.container);
    run(&mut first_delta, &update(&[40], DELTA));
    let error = try_run(&mut first_delta, &add(&operands, &COLUMNS, PROFILE)).unwrap_err();
    assert!(error.contains("one form"), "{error}");

    let mut first_low_rank = session_on(&fx.container);
    run(&mut first_low_rank, &add(&operands, &COLUMNS, PROFILE));
    let error = try_run(&mut first_low_rank, &update(&[40], DELTA)).unwrap_err();
    assert!(error.contains("one form"), "{error}");
}

#[test]
fn the_f32_profile_takes_eight_digit_operands() {
    let fx = Fixture::new();
    let ([rows, _], _) = read_bf16_tensor(&fx.container, TENSOR).unwrap();
    let path = fx.path("f32.json");
    let hex = |n: usize| vec![format!("{F32_ONE:08x}"); n];
    std::fs::write(
        &path,
        serde_json::json!({ "rank": 1, "a": hex(rows), "b": hex(1) }).to_string(),
    )
    .unwrap();
    let mut s = session_on(&fx.container);
    let staged = run(&mut s, &add(&q(&path), &[33], F32_PROFILE));
    assert!(staged[0].contains(F32_PROFILE), "{staged:?}");
}

#[test]
fn bad_inputs_are_refused_by_name_before_anything_is_staged() {
    let fx = Fixture::new();
    let operands = operand_file(&fx, "ab.json", &COLUMNS, None);
    let mut s = session_on(&fx.container);

    assert!(
        try_run(&mut s, &add(&operands, &COLUMNS, "f64-seq-rne9/v1"))
            .unwrap_err()
            .contains("numerics")
    );
    assert!(try_run(&mut s, &add(&operands, &COLUMNS, F32_PROFILE))
        .unwrap_err()
        .contains("8 lowercase hex digits"));
    assert!(
        try_run(&mut s, &add(&q(fx.path("absent.json")), &COLUMNS, PROFILE))
            .unwrap_err()
            .contains("cannot read operand file")
    );

    let extra = fx.path("extra.json");
    std::fs::write(
        &extra,
        r#"{"rank":1,"a":[],"b":[],"alpha":"3ff0000000000000"}"#,
    )
    .unwrap();
    assert!(try_run(&mut s, &add(&q(&extra), &COLUMNS, PROFILE)).is_err());

    // b laid out over every column (v1's layout) is a shape refusal, not a silent reinterpretation.
    let wrong_layout = operand_file(&fx, "wide.json", &[0, 1, 2, 3, 4, 5, 6, 7], None);
    let error = try_run(&mut s, &add(&wrong_layout, &COLUMNS, PROFILE)).unwrap_err();
    assert!(error.contains("shape_invalid"), "{error}");

    // Nothing above staged anything: a compile has nothing to write.
    let out = fx.path("none.vindex3");
    let compiled = try_run(
        &mut s,
        &format!("COMPILE CURRENT INTO VINDEX \"{}\";", q(&out)),
    );
    assert!(compiled.is_err() || !out.join("weight_patches.json").exists());
}

fn uniform_file(fx: &Fixture, name: &str, a: u64, b: u64, columns: usize) -> String {
    let ([rows, _], _) = read_bf16_tensor(&fx.container, TENSOR).unwrap();
    let path = fx.path(name);
    std::fs::write(
        &path,
        serde_json::json!({
            "rank": 1,
            "a": vec![format!("{a:016x}"); rows],
            "b": vec![format!("{b:016x}"); columns],
        })
        .to_string(),
    )
    .unwrap();
    q(path)
}

/// SG-EDIT-1's sham: a zero-valued op (every term has a zero factor) stages, and compiles to a container whose
/// tensor bytes are the base's, recorded as an operation target that changed nothing.
#[test]
fn a_zero_valued_update_stages_as_a_sham_and_compiles_to_the_bases_bytes() {
    let fx = Fixture::new();
    let sham = uniform_file(&fx, "zero.json", 0, F64_ONE, COLUMNS.len());
    let mut s = session_on(&fx.container);
    let staged = run(&mut s, &add(&sham, &COLUMNS, PROFILE));
    assert!(
        staged[0].contains("0 element(s) would change"),
        "{staged:?}"
    );
    assert!(
        staged.iter().any(|l| l.contains("declared sham")),
        "{staged:?}"
    );

    let out = fx.path("sham.vindex3");
    compile(&mut s, &out);
    assert!(is_coherent(&out));
    for representation in [CANONICAL, DERIVED] {
        assert_eq!(
            tensor_bytes(&out, representation, TENSOR),
            tensor_bytes(&fx.container, representation, TENSOR),
            "{representation}"
        );
    }
    let target = &ledger(&out)["targets"][0];
    assert_eq!(target["changed_elements"], 0);
    assert_eq!(target["base_tensor_sha256"], target["result_tensor_sha256"]);
}

/// A nonzero op whose every result rounds back to the stored bits is not a sham: it is refused as changing nothing.
#[test]
fn a_nonzero_update_that_rounds_away_entirely_is_refused() {
    const F64_TWO_POW_MINUS_60: u64 = 0x3c30_0000_0000_0000;
    let fx = Fixture::new();
    let tiny = uniform_file(&fx, "tiny.json", F64_TWO_POW_MINUS_60, F64_ONE, 1);
    let mut s = session_on(&fx.container);
    let error = try_run(&mut s, &add(&tiny, &[33], PROFILE)).unwrap_err();
    assert!(error.contains("ADD LOW RANK refused"), "{error}");
}
