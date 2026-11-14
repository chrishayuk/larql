//! A structured `remove_direction` op as a `.vlp` instruction: declared by its operands, resolved by the
//! frozen interpreter into exact BF16 results, compiled into a container that records results, not deltas.

use larql_vindex::format::vindex3::weight_patch::bf16::patched_bits;
use larql_vindex::format::vindex3::weight_patch::structured::{
    run, BaseTensor, OpKind, OperandBits, ScalarBits, StructuredOp, TensorRole,
};
use larql_vindex::format::vindex3::weight_patch::{
    apply_declared_patches, check_ledger, parse_ledger, read_bf16_tensor, DeclaredPatch,
    EditEncoding, StructuredWeightOp, WeightPatchError, WEIGHT_PATCHES_SCHEMA,
    WEIGHT_PATCHES_SCHEMA_V3,
};

use crate::fixture::*;

const PROFILE: &str = "f64-seq-rne1/v1";
const COLUMN: usize = 33;
/// `1 + 2^-30` as f64 bits: removing a unit row vector with this alpha nearly cancels the element, leaving a
/// tiny nonzero far below the f32 spacing of the element it came from.
const ALPHA_NEAR_ONE: u64 = 0x3ff0_0000_4000_0000;
const F64_ONE: u64 = 0x3ff0_0000_0000_0000;
/// How many f32 neighbours of the true difference a delta search tries on each side.
const DELTA_NEIGHBOURS: i64 = 8;
const LEDGER: &str = "weight_patches.json";

/// A row of the fixture tensor whose element in `COLUMN` is finite and nonzero.
fn row_with_nonzero(base: &[u16], cols: usize) -> usize {
    (0..base.len() / cols)
        .find(|&r| base[r * cols + COLUMN] & 0x7fff != 0)
        .expect("the fixture tensor has a nonzero element in the column")
}

/// `u = e_row`: one at `row`, zero elsewhere.
fn unit_row(rows: usize, row: usize) -> OperandBits {
    OperandBits::F64(
        (0..rows)
            .map(|r| if r == row { F64_ONE } else { 0 })
            .collect(),
    )
}

fn near_cancelling_op(src: &Source) -> (StructuredWeightOp, usize) {
    let ([rows, cols], base) = read_bf16_tensor(&src.container, TENSOR).unwrap();
    let row = row_with_nonzero(&base, cols);
    let op = StructuredWeightOp::bind(
        &src.container,
        TENSOR,
        PROFILE,
        &unit_row(rows, row),
        ScalarBits::F64(ALPHA_NEAR_ONE),
        vec![COLUMN],
    )
    .unwrap();
    (op, row)
}

/// Whether some f32 delta near `new - old` makes `RNE_bf16(old + delta)` equal `new`.
fn some_delta_reaches(old: u16, new: u16) -> bool {
    let to_f32 = |bits: u16| f32::from_bits((bits as u32) << 16);
    let centre = (to_f32(new) - to_f32(old)).to_bits() as i64;
    (-DELTA_NEIGHBOURS..=DELTA_NEIGHBOURS).any(|step| {
        let d = f32::from_bits((centre + step) as u32);
        d.is_finite() && patched_bits(old, d) == Some(new)
    })
}

#[test]
fn a_recorded_op_is_intact_and_bound_to_the_exact_source() {
    let src = source_with_pack();
    let (op, _) = near_cancelling_op(&src);
    op.check_intact().unwrap();
    op.check_binding(&src.container).unwrap();
    assert_eq!(op.source.representation, CANONICAL);
    assert_eq!(op.digest, op.compute_digest());
    assert!(
        op.operand.iter().all(|h| h.len() == 16),
        "f64 bits, fixed width"
    );
}

#[test]
fn resolving_gives_exactly_what_the_frozen_interpreter_gives() {
    let src = source_with_pack();
    let (op, row) = near_cancelling_op(&src);
    let resolved = op.resolve(&src.container).unwrap();

    let ([rows, cols], base) = read_bf16_tensor(&src.container, TENSOR).unwrap();
    let direct = run(
        &StructuredOp {
            numerics: Some(PROFILE.into()),
            shape: [rows, cols],
            kind: OpKind::RemoveDirection {
                u: unit_row(rows, row),
                alpha: ScalarBits::F64(ALPHA_NEAR_ONE),
                columns: vec![COLUMN],
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
    let changed: Vec<(usize, usize, u16)> = (0..rows * cols)
        .filter(|&i| base[i] != direct[i])
        .map(|i| (i / cols, i % cols, direct[i]))
        .collect();
    let got: Vec<(usize, usize, u16)> = resolved
        .edits
        .iter()
        .map(|e| (e.row, e.col, e.new_bits))
        .collect();
    assert_eq!(got, changed);
    assert_eq!(
        got.len(),
        1,
        "a unit row vector changes only its own row of the column"
    );
    assert_eq!((got[0].0, got[0].1), (row, COLUMN));
}

#[test]
fn the_result_is_one_no_f32_delta_can_reach() {
    let src = source_with_pack();
    let (op, row) = near_cancelling_op(&src);
    let edit = op.resolve(&src.container).unwrap().edits[0];
    let (_, base) = read_bf16_tensor(&src.container, TENSOR).unwrap();
    let old = base[row * TENSOR_COLS + COLUMN];
    assert_ne!(
        edit.new_bits & 0x7fff,
        0,
        "a tiny nonzero, not a cancellation to zero"
    );
    assert!(
        !some_delta_reaches(old, edit.new_bits),
        "{old:#06x} -> {:#06x} is expressible as a delta, so this does not show the gap",
        edit.new_bits
    );
}

#[test]
fn compiling_records_results_and_the_container_holds_them() {
    let src = source_with_pack();
    let (op, row) = near_cancelling_op(&src);
    let patch = op.resolve(&src.container).unwrap();
    let want = patch.edits[0].new_bits;
    let out = src.out("structured.vindex3");
    let report = apply_declared_patches(
        &src.container,
        &[DeclaredPatch::Results(patch)],
        std::slice::from_ref(&op.digest),
        &out,
    )
    .unwrap();

    let bytes = tensor_bytes(&out, CANONICAL, TENSOR);
    let at = (row * TENSOR_COLS + COLUMN) * 2;
    assert_eq!(u16::from_le_bytes([bytes[at], bytes[at + 1]]), want);

    assert_eq!(report.ledger.schema, WEIGHT_PATCHES_SCHEMA_V3);
    assert_eq!(report.ledger.origins, vec![op.digest.clone()]);
    let edit = &report.ledger.targets[0].edits[0];
    assert_eq!(edit.encoding, EditEncoding::ResultBits);
    assert_eq!((edit.delta_bits, edit.new_bits), (0, want));

    let source_bytes = tensor_bytes(&src.container, CANONICAL, TENSOR);
    let untouched: usize = bytes
        .iter()
        .zip(&source_bytes)
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        untouched <= 2,
        "exactly one BF16 element differs ({untouched} bytes)"
    );

    let index = index_of(&out);
    assert!(
        check_ledger(&out, &index).is_empty(),
        "the ledger describes the container"
    );
}

#[test]
fn a_ledger_of_results_under_the_old_schema_is_a_defect() {
    let src = source_with_pack();
    let (op, _) = near_cancelling_op(&src);
    let out = src.out("structured.vindex3");
    apply_declared_patches(
        &src.container,
        &[DeclaredPatch::Results(op.resolve(&src.container).unwrap())],
        &[],
        &out,
    )
    .unwrap();
    let path = out.join(LEDGER);
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(
        &path,
        text.replace(WEIGHT_PATCHES_SCHEMA_V3, WEIGHT_PATCHES_SCHEMA),
    )
    .unwrap();
    let defects = check_ledger(&out, &index_of(&out));
    assert!(
        defects
            .iter()
            .any(|d| d.contains("result-bits edits under schema")),
        "{defects:?}"
    );
    assert!(parse_ledger(&std::fs::read_to_string(&path).unwrap()).is_ok());
}

#[test]
fn a_delta_only_patch_still_writes_the_old_schema() {
    let src = source_with_pack();
    let out = src.out("delta.vindex3");
    let report = larql_vindex::format::vindex3::weight_patch::apply_weight_patch(
        &src.container,
        &larql_vindex::format::vindex3::weight_patch::WeightPatch {
            tensor: TENSOR.into(),
            edits: vec![larql_vindex::format::vindex3::weight_patch::WeightEdit {
                row: 5,
                col: 33,
                delta: 0.0625,
            }],
        },
        &out,
    )
    .unwrap();
    assert_eq!(report.ledger.schema, WEIGHT_PATCHES_SCHEMA);
    assert!(report.ledger.targets[0]
        .edits
        .iter()
        .all(|e| e.encoding == EditEncoding::Delta));
    let text = std::fs::read_to_string(out.join(LEDGER)).unwrap();
    assert!(
        !text.contains("result_bits"),
        "the encoding field is absent for a delta"
    );
}

#[test]
fn a_changed_field_is_refused_by_the_digest_or_the_binding() {
    let src = source_with_pack();
    let (op, _) = near_cancelling_op(&src);

    let mut damaged = op.clone();
    damaged.columns = vec![COLUMN + 1];
    assert!(matches!(
        damaged.check_intact(),
        Err(WeightPatchError::PatchOp(_))
    ));

    let mut elsewhere = op.clone();
    elsewhere.source.payload_sha256 = "0".repeat(64);
    elsewhere.digest = elsewhere.compute_digest();
    elsewhere.check_intact().unwrap();
    assert!(matches!(
        elsewhere.resolve(&src.container),
        Err(WeightPatchError::WrongSource { .. })
    ));
}

#[test]
fn operands_of_the_wrong_width_for_the_profile_are_refused() {
    let src = source_with_pack();
    let ([rows, _], _) = read_bf16_tensor(&src.container, TENSOR).unwrap();
    let f32_operand = OperandBits::F32(vec![0x3f80_0000; rows]);
    let op = StructuredWeightOp::bind(
        &src.container,
        TENSOR,
        PROFILE,
        &f32_operand,
        ScalarBits::F32(0x3f80_0000),
        vec![COLUMN],
    )
    .unwrap();
    let error = op.check_intact().unwrap_err().to_string();
    assert!(error.contains("16 lowercase hex digits"), "{error}");
}

#[test]
fn an_unknown_profile_is_refused_by_name() {
    let src = source_with_pack();
    let ([rows, _], _) = read_bf16_tensor(&src.container, TENSOR).unwrap();
    let op = StructuredWeightOp::bind(
        &src.container,
        TENSOR,
        "f64-seq-rne9/v1",
        &unit_row(rows, 0),
        ScalarBits::F64(F64_ONE),
        vec![COLUMN],
    )
    .unwrap();
    assert!(op
        .resolve(&src.container)
        .unwrap_err()
        .to_string()
        .contains("numerics"));
}

#[test]
fn an_op_that_changes_nothing_edits_no_elements() {
    let src = source_with_pack();
    let ([rows, _], _) = read_bf16_tensor(&src.container, TENSOR).unwrap();
    let op = StructuredWeightOp::bind(
        &src.container,
        TENSOR,
        PROFILE,
        &unit_row(rows, 0),
        ScalarBits::F64(0),
        vec![COLUMN],
    )
    .unwrap();
    assert!(matches!(
        op.resolve(&src.container),
        Err(WeightPatchError::Empty)
    ));
}

#[test]
fn the_interpreters_refusals_surface_with_their_codes() {
    let src = source_with_pack();
    let ([rows, cols], _) = read_bf16_tensor(&src.container, TENSOR).unwrap();
    let op = StructuredWeightOp::bind(
        &src.container,
        TENSOR,
        PROFILE,
        &unit_row(rows, 0),
        ScalarBits::F64(F64_ONE),
        vec![cols],
    )
    .unwrap();
    let error = op.resolve(&src.container).unwrap_err().to_string();
    assert!(error.contains("structured op"), "{error}");
}

#[test]
fn a_result_that_is_not_finite_is_refused() {
    use larql_vindex::format::vindex3::weight_patch::{ResultEdit, ResultPatch};
    let src = source_with_pack();
    let error = apply_declared_patches(
        &src.container,
        &[DeclaredPatch::Results(ResultPatch {
            tensor: TENSOR.into(),
            edits: vec![ResultEdit {
                row: 0,
                col: 0,
                new_bits: 0x7f80,
            }],
        })],
        &[],
        &src.out("bad.vindex3"),
    )
    .unwrap_err();
    assert!(
        matches!(error, WeightPatchError::NonFiniteResult { .. }),
        "{error}"
    );
    assert!(!src.out("bad.vindex3").exists(), "nothing is left behind");
}
