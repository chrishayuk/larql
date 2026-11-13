//! `add_low_rank_columns` (VLP numerics v2) as a `.vlp` instruction: declared by its operands, resolved by the
//! frozen interpreter into exact BF16 results, compiled into a container that records results, and held beside
//! `remove_direction` ops by the same column-footprint rule.

use larql_vindex::format::vindex3::knowledge::KnowledgeOverlay;
use larql_vindex::format::vindex3::weight_patch::op_target::StructuredOpRecord;
use larql_vindex::format::vindex3::weight_patch::structured::{
    run, BaseTensor, OpKind, OperandBits, ScalarBits, StructuredOp, TensorRole,
};
use larql_vindex::format::vindex3::weight_patch::{
    apply_declared_patches, check_ledger, declare_patches, read_bf16_tensor, record_op,
    DeclaredPatch, LowRankColumnsOp, StructuredWeightOp, WeightEdit, WeightOpRef, WeightPatchError,
    WEIGHT_PATCHES_SCHEMA_V4,
};
use larql_vindex::PatchOp;

use crate::fixture::*;
use crate::structured_role::{container_with_addressable_embedding, UNIQUE_NAME};

const PROFILE: &str = "f64-seq-rne1/v1";
const F32_PROFILE: &str = "f32-seq-rne1/v1";
const F64_ONE: u64 = 0x3ff0_0000_0000_0000;
const F64_EIGHTH: u64 = 0x3fc0_0000_0000_0000;
const F64_NEG_ZERO: u64 = 0x8000_0000_0000_0000;
/// The columns of the head-slice-shaped edit the tests make: a contiguous block and a stray column.
const COLUMNS: [usize; 4] = [8, 9, 10, 33];
/// The row whose `a` factor is zero: none of its elements may change.
const SILENT_ROW: usize = 3;
/// The position in `COLUMNS` whose `b` factor is zero: its whole column may not change.
const SILENT_POSITION: usize = 1;

/// A rank-one op over `COLUMNS`: `a` is `1/8` except a zero at `SILENT_ROW`, `b` is one except a `-0` at
/// `SILENT_POSITION`.
fn rank_one_bits(rows: usize) -> (OperandBits, OperandBits) {
    let a = (0..rows)
        .map(|r| if r == SILENT_ROW { 0 } else { F64_EIGHTH })
        .collect();
    let b = (0..COLUMNS.len())
        .map(|p| {
            if p == SILENT_POSITION {
                F64_NEG_ZERO
            } else {
                F64_ONE
            }
        })
        .collect();
    (OperandBits::F64(a), OperandBits::F64(b))
}

fn rank_one(src: &Source, columns: &[usize]) -> LowRankColumnsOp {
    let ([rows, _], _) = read_bf16_tensor(&src.container, TENSOR).unwrap();
    let (a, _) = rank_one_bits(rows);
    let b = OperandBits::F64(vec![F64_ONE; columns.len()]);
    LowRankColumnsOp::bind(&src.container, TENSOR, PROFILE, 1, &a, &b, columns.to_vec()).unwrap()
}

fn the_op(src: &Source) -> LowRankColumnsOp {
    let ([rows, _], _) = read_bf16_tensor(&src.container, TENSOR).unwrap();
    let (a, b) = rank_one_bits(rows);
    LowRankColumnsOp::bind(&src.container, TENSOR, PROFILE, 1, &a, &b, COLUMNS.to_vec()).unwrap()
}

fn remove_direction(src: &Source, columns: &[usize]) -> StructuredWeightOp {
    let ([rows, _], _) = read_bf16_tensor(&src.container, TENSOR).unwrap();
    let mut u = vec![0u64; rows];
    u[0] = F64_ONE;
    StructuredWeightOp::bind(
        &src.container,
        TENSOR,
        PROFILE,
        &OperandBits::F64(u),
        ScalarBits::F64(F64_ONE),
        columns.to_vec(),
    )
    .unwrap()
}

#[test]
fn a_bound_op_is_intact_and_bound_to_the_exact_source() {
    let src = source_with_pack();
    let op = the_op(&src);
    op.check_intact().unwrap();
    op.check_binding(&src.container).unwrap();
    assert_eq!(op.source.representation, CANONICAL);
    assert_eq!(op.digest, op.compute_digest());
    assert_eq!(op.a.len(), TENSOR_ROWS);
    assert_eq!(op.b.len(), COLUMNS.len(), "b is compact: rank × |J|");
    assert!(op.a.iter().chain(&op.b).all(|h| h.len() == 16));
}

#[test]
fn resolving_gives_exactly_what_the_frozen_interpreter_gives_and_only_written_elements_change() {
    let src = source_with_pack();
    let op = the_op(&src);
    let resolved = op.resolve(&src.container).unwrap();

    let ([rows, cols], base) = read_bf16_tensor(&src.container, TENSOR).unwrap();
    let (a, b) = rank_one_bits(rows);
    let direct = run(
        &StructuredOp {
            numerics: Some(PROFILE.into()),
            shape: [rows, cols],
            kind: OpKind::AddLowRankColumns {
                rank: 1,
                a,
                b,
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
    assert!(!got.is_empty());
    for &(r, c, _) in &got {
        assert!(COLUMNS.contains(&c), "column {c} is outside J");
        assert_ne!(r, SILENT_ROW, "a row whose a-factor is zero is untouched");
        assert_ne!(
            c, COLUMNS[SILENT_POSITION],
            "a column whose b-factor is -0 is untouched"
        );
    }
}

#[test]
fn compiling_records_the_operation_and_leaves_every_element_outside_the_written_set_byte_identical()
{
    let src = source_with_pack();
    let op = the_op(&src);
    let (declared, origins) =
        declare_patches(&src.container, &[WeightOpRef::LowRankColumns(&op)]).unwrap();
    let out = src.out("low-rank.vindex3");
    let report = apply_declared_patches(&src.container, &declared, &origins, &out).unwrap();
    assert_eq!(report.ledger.schema, WEIGHT_PATCHES_SCHEMA_V4);
    assert_eq!(report.ledger.origins, vec![op.digest.clone()]);
    let target = &report.ledger.targets[0];
    assert!(
        target.edits.is_empty(),
        "an operation target lists no elements"
    );
    assert_eq!(target.operations.len(), 1);
    assert_eq!(
        target.operations[0].op,
        StructuredOpRecord::LowRankColumns(op.clone())
    );
    assert!(check_ledger(&out, &index_of(&out)).is_empty());

    let written: std::collections::HashSet<(usize, usize)> = op
        .resolve(&src.container)
        .unwrap()
        .edits
        .iter()
        .map(|e| (e.row, e.col))
        .collect();
    let before = tensor_bytes(&src.container, CANONICAL, TENSOR);
    let after = tensor_bytes(&out, CANONICAL, TENSOR);
    for i in 0..TENSOR_ROWS * TENSOR_COLS {
        let (r, c) = (i / TENSOR_COLS, i % TENSOR_COLS);
        let same = before[2 * i..2 * i + 2] == after[2 * i..2 * i + 2];
        assert_eq!(same, !written.contains(&(r, c)), "element ({r}, {c})");
    }
}

#[test]
fn a_zero_operand_changes_nothing_and_is_refused_as_empty() {
    let src = source_with_pack();
    let ([rows, _], _) = read_bf16_tensor(&src.container, TENSOR).unwrap();
    let op = LowRankColumnsOp::bind(
        &src.container,
        TENSOR,
        PROFILE,
        1,
        &OperandBits::F64(vec![0; rows]),
        &OperandBits::F64(vec![F64_ONE; COLUMNS.len()]),
        COLUMNS.to_vec(),
    )
    .unwrap();
    assert!(matches!(
        op.resolve(&src.container),
        Err(WeightPatchError::Empty)
    ));
}

#[test]
fn a_changed_field_is_refused_by_the_digest_or_the_binding() {
    let src = source_with_pack();
    let op = the_op(&src);

    for damage in [
        |o: &mut LowRankColumnsOp| o.columns[0] += 1,
        |o: &mut LowRankColumnsOp| o.a.swap(0, SILENT_ROW),
        |o: &mut LowRankColumnsOp| std::mem::swap(&mut o.a, &mut o.b),
        |o: &mut LowRankColumnsOp| o.rank = 2,
        |o: &mut LowRankColumnsOp| o.origin = Some("elsewhere".into()),
    ] {
        let mut damaged = op.clone();
        damage(&mut damaged);
        assert!(matches!(
            damaged.check_intact(),
            Err(WeightPatchError::PatchOp(_))
        ));
    }

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
fn origin_is_sealed_into_the_digest() {
    let src = source_with_pack();
    let op = the_op(&src).with_origin("alpha=0.125 u=e_k v=ones");
    op.check_intact().unwrap();
    assert_ne!(op.digest, the_op(&src).digest);
}

#[test]
fn operands_of_the_wrong_width_for_the_profile_are_refused() {
    let src = source_with_pack();
    let ([rows, _], _) = read_bf16_tensor(&src.container, TENSOR).unwrap();
    let op = LowRankColumnsOp::bind(
        &src.container,
        TENSOR,
        F32_PROFILE,
        1,
        &OperandBits::F64(vec![F64_ONE; rows]),
        &OperandBits::F64(vec![F64_ONE; COLUMNS.len()]),
        COLUMNS.to_vec(),
    )
    .unwrap();
    let error = op.check_intact().unwrap_err().to_string();
    assert!(error.contains("8 lowercase hex digits"), "{error}");
}

#[test]
fn the_interpreters_refusals_surface_with_their_codes() {
    let src = source_with_pack();
    let ([rows, cols], _) = read_bf16_tensor(&src.container, TENSOR).unwrap();
    let cases: [(Vec<usize>, usize, usize, &str); 3] = [
        (vec![cols], 1, 1, "columns_invalid"),
        (vec![2, 1], 1, 2, "columns_invalid"),
        (vec![1], 1, 2, "shape_invalid"),
    ];
    for (columns, rank, b_len, code) in cases {
        let op = LowRankColumnsOp::bind(
            &src.container,
            TENSOR,
            PROFILE,
            rank,
            &OperandBits::F64(vec![F64_ONE; rows * rank]),
            &OperandBits::F64(vec![F64_ONE; b_len]),
            columns,
        )
        .unwrap();
        let error = op.resolve(&src.container).unwrap_err().to_string();
        assert!(error.contains(code), "{code}: {error}");
    }
}

#[test]
fn the_token_embedding_is_refused_by_what_the_container_says_it_is() {
    let src = source_with_pack();
    let container = container_with_addressable_embedding(&src);
    let ([rows, _], _) = read_bf16_tensor(&container, UNIQUE_NAME).unwrap();
    let op = LowRankColumnsOp::bind(
        &container,
        UNIQUE_NAME,
        PROFILE,
        1,
        &OperandBits::F64(vec![F64_ONE; rows]),
        &OperandBits::F64(vec![F64_ONE]),
        vec![3],
    )
    .unwrap();
    match op.resolve(&container) {
        Err(WeightPatchError::Structured(why)) => {
            assert!(why.contains("tensor_refused"), "{why}")
        }
        other => panic!("the embedding must be refused, got {other:?}"),
    }
}

mod footprint {
    use super::*;

    #[test]
    fn a_low_rank_op_and_a_remove_direction_op_compile_together_on_disjoint_columns() {
        let src = source_with_pack();
        let lr = rank_one(&src, &[8, 9]);
        let rd = remove_direction(&src, &[40]);
        let ops = [
            WeightOpRef::LowRankColumns(&lr),
            WeightOpRef::Structured(&rd),
        ];
        let (declared, origins) = declare_patches(&src.container, &ops).unwrap();
        assert_eq!(declared.len(), 1, "one tensor, one operation target");
        let DeclaredPatch::Operations(joined) = &declared[0] else {
            panic!("a tensor with a low-rank op compiles as an operation target");
        };
        assert_eq!(joined.operations.len(), 2, "both ops are its operations");
        for own in [
            lr.resolve(&src.container).unwrap(),
            rd.resolve(&src.container).unwrap(),
        ] {
            assert!(own
                .edits
                .iter()
                .all(|e| joined.result[e.row * TENSOR_COLS + e.col] == e.new_bits));
        }
        assert_eq!(
            joined.changed_elements,
            joined
                .operations
                .iter()
                .map(|o| o.changed_elements)
                .sum::<usize>()
        );
        let out = src.out("both.vindex3");
        apply_declared_patches(&src.container, &declared, &origins, &out).unwrap();
        assert!(check_ledger(&out, &index_of(&out)).is_empty());
    }

    #[test]
    fn a_shared_column_is_refused_whichever_kind_comes_first() {
        let src = source_with_pack();
        let lr = rank_one(&src, &[8, 9]);
        let rd = remove_direction(&src, &[9]);
        let lr2 = rank_one(&src, &[9, 12]);
        for ops in [
            [
                WeightOpRef::LowRankColumns(&lr),
                WeightOpRef::Structured(&rd),
            ],
            [
                WeightOpRef::Structured(&rd),
                WeightOpRef::LowRankColumns(&lr),
            ],
            [
                WeightOpRef::LowRankColumns(&lr),
                WeightOpRef::LowRankColumns(&lr2),
            ],
        ] {
            assert!(matches!(
                declare_patches(&src.container, &ops),
                Err(WeightPatchError::ColumnConflict { col: 9, .. })
            ));
        }
    }

    #[test]
    fn a_delta_op_and_a_low_rank_op_on_one_tensor_are_refused_either_way_round() {
        let src = source_with_pack();
        let lr = rank_one(&src, &[8]);
        let d = record_op(
            &src.container,
            TENSOR,
            &[WeightEdit {
                row: 5,
                col: 40,
                delta: 0.0625,
            }],
        )
        .unwrap();
        for ops in [
            [WeightOpRef::LowRankColumns(&lr), WeightOpRef::Delta(&d)],
            [WeightOpRef::Delta(&d), WeightOpRef::LowRankColumns(&lr)],
        ] {
            assert!(matches!(
                declare_patches(&src.container, &ops),
                Err(WeightPatchError::MixedForms { .. })
            ));
        }
    }

    #[test]
    fn the_session_overlay_holds_it_by_the_same_rule_and_counts_it_as_structured() {
        let src = source_with_pack();
        let mut overlay = KnowledgeOverlay::new();
        overlay
            .stage_low_rank_columns_op(rank_one(&src, &[8, 9]))
            .unwrap();
        overlay
            .stage_structured_op(remove_direction(&src, &[40]))
            .unwrap();
        assert_eq!(overlay.structured_op_count(), 2);
        assert_eq!(overlay.weight_edit_count(), 0);
        assert!(matches!(
            overlay.stage_structured_op(remove_direction(&src, &[9])),
            Err(WeightPatchError::ColumnConflict { col: 9, .. })
        ));

        let mut tampered = rank_one(&src, &[50]);
        tampered.b[0] = format!("{F64_EIGHTH:016x}");
        assert!(overlay.stage_low_rank_columns_op(tampered).is_err());
    }
}

mod vlp_file {
    use super::*;

    #[test]
    fn it_round_trips_under_its_own_tag_and_reads_back_as_a_weight_op() {
        let src = source_with_pack();
        let op = PatchOp::LowRankColumnsWeight(the_op(&src));
        let text = serde_json::to_string(&op).unwrap();
        assert!(text.contains(r#""op":"low_rank_columns_weight""#), "{text}");
        let back: PatchOp = serde_json::from_str(&text).unwrap();
        let Some(WeightOpRef::LowRankColumns(read)) = back.weight_op() else {
            panic!("reads back as the low-rank op");
        };
        assert_eq!(read, &the_op(&src));
        read.check_intact().unwrap();
    }

    #[test]
    fn a_negative_column_is_refused_while_the_patch_is_parsed() {
        let src = source_with_pack();
        let text = serde_json::to_string(&PatchOp::LowRankColumnsWeight(the_op(&src)))
            .unwrap()
            .replace(r#""columns":[8,"#, r#""columns":[-1,"#);
        assert!(text.contains("-1"));
        assert!(serde_json::from_str::<PatchOp>(&text).is_err());
    }
}
