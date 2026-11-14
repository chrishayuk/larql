//! `weight-patches/v4` operation targets: a tensor compiled from structured ops is recorded by its ops, not its
//! elements (docs/verification/weight-patches-v4.md). The compact record must produce exactly the bytes the
//! element form produces, stay consistent under `check_ledger`, and stay bounded.

use larql_vindex::format::vindex3::weight_patch::op_target::{
    check_cost, operand_sha256, tensor_sha256, StructuredCost, StructuredOpRecord,
};
use larql_vindex::format::vindex3::weight_patch::structured::OperandBits;
use larql_vindex::format::vindex3::weight_patch::{
    apply_declared_patches, check_ledger, declare_patches, parse_ledger, read_bf16_tensor,
    record_op, tensor_shape, DeclaredPatch, LowRankColumnsOp, WeightEdit, WeightOpRef,
    WeightPatchError, WeightPatchLedger, MAX_STRUCTURED_OPERAND_ELEMENTS, MAX_STRUCTURED_RANK,
    MAX_STRUCTURED_TENSOR_ELEMENTS, MAX_STRUCTURED_WORK, WEIGHT_PATCHES_SCHEMA,
    WEIGHT_PATCHES_SCHEMA_V3, WEIGHT_PATCHES_SCHEMA_V4,
};

use crate::fixture::*;

const PROFILE: &str = "f64-seq-rne1/v1";
const F64_EIGHTH: u64 = 0x3fc0_0000_0000_0000;
const F64_ONE: u64 = 0x3ff0_0000_0000_0000;
const COLUMNS: [usize; 3] = [8, 9, 33];
/// Another tensor of the fixture, in the same canonical representation.
const OTHER_TENSOR: &str = "0.mlp.down_proj.weight";
const LEDGER: &str = "weight_patches.json";

/// One change to a compiled ledger.
type Tamper = Box<dyn Fn(&mut WeightPatchLedger)>;

fn rank_one(src: &Source, columns: &[usize]) -> LowRankColumnsOp {
    let ([rows, _], _) = read_bf16_tensor(&src.container, TENSOR).unwrap();
    LowRankColumnsOp::bind(
        &src.container,
        TENSOR,
        PROFILE,
        1,
        &OperandBits::F64(vec![F64_EIGHTH; rows]),
        &OperandBits::F64(vec![F64_ONE; columns.len()]),
        columns.to_vec(),
    )
    .unwrap()
}

fn compile(src: &Source, ops: &[WeightOpRef<'_>], name: &str) -> std::path::PathBuf {
    let (declared, origins) = declare_patches(&src.container, ops).unwrap();
    let out = src.out(name);
    apply_declared_patches(&src.container, &declared, &origins, &out).unwrap();
    out
}

fn read_ledger(out: &std::path::Path) -> WeightPatchLedger {
    parse_ledger(&std::fs::read_to_string(out.join(LEDGER)).unwrap()).unwrap()
}

/// Rewrite the compiled ledger and return `check_ledger`'s defects.
fn defects_after(
    out: &std::path::Path,
    change: impl FnOnce(&mut WeightPatchLedger),
) -> Vec<String> {
    let mut ledger = read_ledger(out);
    change(&mut ledger);
    std::fs::write(out.join(LEDGER), serde_json::to_string(&ledger).unwrap()).unwrap();
    check_ledger(out, &index_of(out))
}

fn copy_tree(from: &std::path::Path, to: &std::path::Path) {
    std::fs::create_dir_all(to).unwrap();
    for e in std::fs::read_dir(from).unwrap() {
        let e = e.unwrap();
        let target = to.join(e.file_name());
        if e.file_type().unwrap().is_dir() {
            copy_tree(&e.path(), &target);
        } else {
            std::fs::copy(e.path(), target).unwrap();
        }
    }
}

mod equivalence {
    use super::*;

    /// The compact record and the element record of the same op compile to the same bytes, in the canonical
    /// representation and in every re-derived one.
    #[test]
    fn the_compact_and_the_element_form_of_one_op_compile_identical_bytes() {
        let src = source_with_pack();
        let op = rank_one(&src, &COLUMNS);

        let compact = compile(&src, &[WeightOpRef::LowRankColumns(&op)], "compact.vindex3");
        assert_eq!(read_ledger(&compact).schema, WEIGHT_PATCHES_SCHEMA_V4);

        let elements = src.out("elements.vindex3");
        let report = apply_declared_patches(
            &src.container,
            &[DeclaredPatch::Results(op.resolve(&src.container).unwrap())],
            std::slice::from_ref(&op.digest),
            &elements,
        )
        .unwrap();
        assert_eq!(report.ledger.schema, WEIGHT_PATCHES_SCHEMA_V3);

        for representation in [CANONICAL, DERIVED] {
            for tensor in [TENSOR, OTHER_TENSOR] {
                assert_eq!(
                    tensor_bytes(&compact, representation, tensor),
                    tensor_bytes(&elements, representation, tensor),
                    "{representation} / {tensor}"
                );
            }
            let (a, b) = (index_of(&compact), index_of(&elements));
            assert_eq!(
                a.representations[representation].payload_sha256,
                b.representations[representation].payload_sha256
            );
        }
        assert_eq!(read_ledger(&compact).result, report.ledger.result);
    }

    #[test]
    fn the_record_names_the_base_and_result_digests_and_the_changed_count() {
        let src = source_with_pack();
        let op = rank_one(&src, &COLUMNS);
        let out = compile(&src, &[WeightOpRef::LowRankColumns(&op)], "out.vindex3");
        let ledger = read_ledger(&out);
        let target = &ledger.targets[0];

        let (_, base) = read_bf16_tensor(&src.container, TENSOR).unwrap();
        let (_, result) = read_bf16_tensor(&out, TENSOR).unwrap();
        assert_eq!(
            target.base_tensor_sha256.as_deref(),
            Some(tensor_sha256(&base).as_str())
        );
        assert_eq!(
            target.result_tensor_sha256.as_deref(),
            Some(tensor_sha256(&result).as_str())
        );
        let changed = base.iter().zip(&result).filter(|(a, b)| a != b).count();
        assert_eq!(target.changed_elements, Some(changed));
        assert_eq!(target.operations[0].changed_elements, changed);

        let ([rows, _], _) = read_bf16_tensor(&src.container, TENSOR).unwrap();
        let digests = &target.operations[0].operands_sha256;
        assert_eq!(
            digests["a"],
            operand_sha256(&OperandBits::F64(vec![F64_EIGHTH; rows]))
        );
        assert_eq!(
            digests["b"],
            operand_sha256(&OperandBits::F64(vec![F64_ONE; COLUMNS.len()]))
        );
        assert!(check_ledger(&out, &index_of(&out)).is_empty());
    }

    /// An element target beside an operation target keeps its element form under `v4`, and the schemas a compile
    /// without any low-rank op writes are unchanged.
    #[test]
    fn element_targets_keep_their_form_and_only_a_low_rank_op_makes_v4() {
        let src = source_with_pack();
        let lr = rank_one(&src, &COLUMNS);
        let delta = record_op(
            &src.container,
            OTHER_TENSOR,
            &[WeightEdit {
                row: 1,
                col: 2,
                delta: 0.0625,
            }],
        )
        .unwrap();
        let out = compile(
            &src,
            &[WeightOpRef::LowRankColumns(&lr), WeightOpRef::Delta(&delta)],
            "mixed.vindex3",
        );
        let ledger = read_ledger(&out);
        assert_eq!(ledger.schema, WEIGHT_PATCHES_SCHEMA_V4);
        let by_name = |n: &str| ledger.targets.iter().find(|t| t.tensor == n).unwrap();
        assert!(!by_name(TENSOR).operations.is_empty());
        assert_eq!(by_name(OTHER_TENSOR).edits.len(), 1);
        assert!(by_name(OTHER_TENSOR).operations.is_empty());
        assert!(check_ledger(&out, &index_of(&out)).is_empty());

        let text = std::fs::read_to_string(out.join(LEDGER)).unwrap();
        let raw: serde_json::Value = serde_json::from_str(&text).unwrap();
        let element = raw["targets"]
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["tensor"] == OTHER_TENSOR)
            .unwrap();
        for absent in [
            "operations",
            "base_tensor_sha256",
            "result_tensor_sha256",
            "changed_elements",
        ] {
            assert!(
                element.get(absent).is_none(),
                "{absent} on an element target"
            );
        }

        let delta_only = compile(&src, &[WeightOpRef::Delta(&delta)], "delta.vindex3");
        assert_eq!(read_ledger(&delta_only).schema, WEIGHT_PATCHES_SCHEMA);
    }
}

mod consistency {
    use super::*;

    fn compiled(src: &Source) -> std::path::PathBuf {
        let op = rank_one(src, &COLUMNS);
        compile(src, &[WeightOpRef::LowRankColumns(&op)], "out.vindex3")
    }

    fn has(defects: &[String], needle: &str) -> bool {
        defects.iter().any(|d| d.contains(needle))
    }

    #[test]
    fn a_tampered_result_digest_operand_digest_or_count_is_a_defect() {
        let src = source_with_pack();
        let out = compiled(&src);
        let d = defects_after(&out, |l| {
            l.targets[0].result_tensor_sha256 = Some("0".repeat(64));
        });
        assert!(has(&d, "result_tensor_sha256"), "{d:?}");

        let out = src.out("b.vindex3");
        copy_tree(&compiled_copy_source(&src), &out);
        let d = defects_after(&out, |l| {
            l.targets[0].operations[0]
                .operands_sha256
                .insert("a".into(), "0".repeat(64));
        });
        assert!(has(&d, "operands_sha256"), "{d:?}");

        let out = src.out("c.vindex3");
        copy_tree(&compiled_copy_source(&src), &out);
        let d = defects_after(&out, |l| {
            l.targets[0].changed_elements = l.targets[0].changed_elements.map(|n| n + 1);
        });
        assert!(has(&d, "add up"), "{d:?}");
    }

    /// A pristine compiled container to copy, so each tamper starts from the same bytes.
    fn compiled_copy_source(src: &Source) -> std::path::PathBuf {
        let pristine = src.out("pristine.vindex3");
        if !pristine.exists() {
            let op = rank_one(src, &COLUMNS);
            let (declared, origins) =
                declare_patches(&src.container, &[WeightOpRef::LowRankColumns(&op)]).unwrap();
            apply_declared_patches(&src.container, &declared, &origins, &pristine).unwrap();
        }
        pristine
    }

    #[test]
    fn an_altered_op_a_foreign_binding_or_overlapping_columns_is_a_defect() {
        let src = source_with_pack();
        let out = compiled(&src);
        let d = defects_after(&out, |l| {
            if let StructuredOpRecord::LowRankColumns(op) = &mut l.targets[0].operations[0].op {
                op.columns[0] += 1;
            }
        });
        assert!(has(&d, "digest"), "{d:?}");

        let out = src.out("b.vindex3");
        copy_tree(&compiled_copy_source(&src), &out);
        let d = defects_after(&out, |l| {
            if let StructuredOpRecord::LowRankColumns(op) = &mut l.targets[0].operations[0].op {
                op.source.index_sha256 = "0".repeat(64);
                op.digest = op.compute_digest();
            }
        });
        assert!(has(&d, "bound to another source"), "{d:?}");

        let out = src.out("c.vindex3");
        copy_tree(&compiled_copy_source(&src), &out);
        let d = defects_after(&out, |l| {
            let twin = l.targets[0].operations[0].clone();
            l.targets[0].operations.push(twin);
        });
        assert!(has(&d, "which another operation also writes"), "{d:?}");
    }

    #[test]
    fn the_schema_must_match_whether_operation_targets_are_present() {
        let src = source_with_pack();
        let out = compiled(&src);
        let d = defects_after(&out, |l| l.schema = WEIGHT_PATCHES_SCHEMA_V3.into());
        assert!(has(&d, "operation targets"), "{d:?}");
    }

    #[test]
    fn a_missing_digest_or_an_element_list_beside_operations_is_a_defect() {
        let src = source_with_pack();
        let out = compiled(&src);
        let d = defects_after(&out, |l| l.targets[0].base_tensor_sha256 = None);
        assert!(has(&d, "lacks its base or result digest"), "{d:?}");

        let out = src.out("b.vindex3");
        copy_tree(&compiled_copy_source(&src), &out);
        let elements = parse_ledger(
            &std::fs::read_to_string(
                compile(
                    &src,
                    &[WeightOpRef::Delta(
                        &record_op(
                            &src.container,
                            TENSOR,
                            &[WeightEdit {
                                row: 0,
                                col: 0,
                                delta: 0.0625,
                            }],
                        )
                        .unwrap(),
                    )],
                    "delta.vindex3",
                )
                .join(LEDGER),
            )
            .unwrap(),
        )
        .unwrap()
        .targets[0]
            .edits
            .clone();
        let d = defects_after(&out, |l| l.targets[0].edits = elements);
        assert!(has(&d, "both element edits and operations"), "{d:?}");
    }

    /// Each check of an operation target on its own: a fresh copy of the pristine container per tamper.
    #[test]
    fn every_other_field_of_an_operation_target_is_checked() {
        let src = source_with_pack();
        let pristine = compiled_copy_source(&src);
        let reseal = |l: &mut WeightPatchLedger, f: &dyn Fn(&mut LowRankColumnsOp)| {
            if let StructuredOpRecord::LowRankColumns(op) = &mut l.targets[0].operations[0].op {
                f(op);
                op.digest = op.compute_digest();
            }
        };
        let cases: Vec<(&str, Tamper)> = vec![
            (
                "not a BF16 matrix",
                Box::new(|l: &mut WeightPatchLedger| l.targets[0].dtype = "F32".into()),
            ),
            (
                "dtype or shape differs",
                Box::new(|l: &mut WeightPatchLedger| l.targets[0].shape[0] += 1),
            ),
            (
                "records no change",
                Box::new(|l: &mut WeightPatchLedger| {
                    l.targets[0].base_tensor_sha256 = l.targets[0].result_tensor_sha256.clone()
                }),
            ),
            (
                "records no changed element",
                Box::new(|l: &mut WeightPatchLedger| {
                    l.targets[0].changed_elements = Some(0);
                    l.targets[0].operations[0].changed_elements = 0;
                }),
            ),
            (
                "names tensor",
                Box::new(move |l: &mut WeightPatchLedger| {
                    reseal(l, &|op| op.tensor = OTHER_TENSOR.into())
                }),
            ),
            (
                "-column tensor",
                Box::new(move |l: &mut WeightPatchLedger| {
                    reseal(l, &|op| *op.columns.last_mut().unwrap() = TENSOR_COLS + 1)
                }),
            ),
            (
                "rank",
                Box::new(move |l: &mut WeightPatchLedger| {
                    reseal(l, &|op| op.rank = MAX_STRUCTURED_RANK + 1)
                }),
            ),
        ];
        for (i, (needle, tamper)) in cases.into_iter().enumerate() {
            let out = src.out(&format!("t{i}.vindex3"));
            copy_tree(&pristine, &out);
            let d = defects_after(&out, |l| tamper(l));
            assert!(has(&d, needle), "{needle}: {d:?}");
        }
    }

    #[test]
    fn changed_tensor_bytes_no_longer_hash_to_the_record() {
        let src = source_with_pack();
        let out = compiled(&src);
        let segment = out.join(&index_of(&out).representations[CANONICAL].segment);
        let (header, start) =
            larql_vindex::format::vindex3::encode::segment::read_segment_header(&segment).unwrap();
        let t = header.tensors.iter().find(|t| t.name == TENSOR).unwrap();
        let mut bytes = std::fs::read(&segment).unwrap();
        // The low byte of the first element of the edited tensor.
        bytes[(start + t.offset) as usize] ^= 1;
        std::fs::write(&segment, bytes).unwrap();
        let d = check_ledger(&out, &index_of(&out));
        assert!(
            d.iter()
                .any(|x| x.contains("does not hash") || x.contains("differ")),
            "{d:?}"
        );
    }
}

mod bounds {
    use super::*;

    fn over(cost: StructuredCost) -> &'static str {
        match check_cost(&cost) {
            Err(WeightPatchError::OverBound { what, .. }) => what,
            other => panic!("expected a bound refusal, got {other:?}"),
        }
    }

    #[test]
    fn each_bound_refuses_one_past_its_limit_and_admits_its_limit() {
        let at_limit = StructuredCost {
            rank: MAX_STRUCTURED_RANK,
            operand_elements: MAX_STRUCTURED_OPERAND_ELEMENTS,
            work: MAX_STRUCTURED_WORK,
            tensor_elements: MAX_STRUCTURED_TENSOR_ELEMENTS as u64,
        };
        check_cost(&at_limit).unwrap();
        assert_eq!(
            over(StructuredCost {
                rank: MAX_STRUCTURED_RANK + 1,
                ..at_limit
            }),
            "rank"
        );
        assert_eq!(
            over(StructuredCost {
                operand_elements: MAX_STRUCTURED_OPERAND_ELEMENTS + 1,
                ..at_limit
            }),
            "operand elements"
        );
        assert!(over(StructuredCost {
            work: MAX_STRUCTURED_WORK + 1,
            ..at_limit
        })
        .contains("work"));
        assert_eq!(
            over(StructuredCost {
                tensor_elements: MAX_STRUCTURED_TENSOR_ELEMENTS as u64 + 1,
                ..at_limit
            }),
            "tensor elements"
        );
    }

    #[test]
    fn an_op_over_the_rank_bound_is_refused_before_any_arithmetic() {
        let src = source_with_pack();
        let [rows, _] = tensor_shape(&src.container, TENSOR).unwrap();
        let rank = MAX_STRUCTURED_RANK + 1;
        let op = LowRankColumnsOp::bind(
            &src.container,
            TENSOR,
            PROFILE,
            rank,
            &OperandBits::F64(vec![F64_EIGHTH; rows * rank]),
            &OperandBits::F64(vec![F64_ONE; rank]),
            vec![3],
        )
        .unwrap();
        assert!(matches!(
            op.check_intact(),
            Err(WeightPatchError::OverBound { what: "rank", .. })
        ));
        assert!(matches!(
            declare_patches(&src.container, &[WeightOpRef::LowRankColumns(&op)]),
            Err(WeightPatchError::OverBound { what: "rank", .. })
        ));
    }

    #[test]
    fn the_cost_of_an_op_is_its_rank_operands_and_multiply_adds() {
        let src = source_with_pack();
        let op = rank_one(&src, &COLUMNS);
        let cost = StructuredOpRecord::LowRankColumns(op).cost();
        assert_eq!(cost.rank, 1);
        assert_eq!(cost.operand_elements, TENSOR_ROWS + COLUMNS.len());
        assert_eq!(cost.work, (TENSOR_ROWS * COLUMNS.len()) as u64);
    }
}

mod apply {
    use super::*;

    fn operation_patch(
        src: &Source,
    ) -> larql_vindex::format::vindex3::weight_patch::op_target::OperationPatch {
        rank_one(src, &COLUMNS)
            .resolve_tensor(&src.container)
            .unwrap()
    }

    #[test]
    fn an_operation_patch_resolved_against_another_base_is_refused() {
        let src = source_with_pack();
        let mut p = operation_patch(&src);
        p.base_tensor_sha256 = "0".repeat(64);
        let out = src.out("stale.vindex3");
        let e = apply_declared_patches(&src.container, &[DeclaredPatch::Operations(p)], &[], &out)
            .unwrap_err();
        assert!(e.to_string().contains("differs from the base"), "{e}");
        assert!(!out.exists());
    }

    #[test]
    fn a_malformed_operation_patch_is_refused_before_anything_is_written() {
        let src = source_with_pack();
        let mut empty = operation_patch(&src);
        empty.changed_elements = 0;
        let mut short = operation_patch(&src);
        short.result.pop();
        let mut infinite = operation_patch(&src);
        infinite.result[5] = 0x7f80;
        let mut misshapen = operation_patch(&src);
        misshapen.shape = [misshapen.shape[1], misshapen.shape[0]];
        for (p, name) in [
            (empty, "empty"),
            (short, "short"),
            (infinite, "infinite"),
            (misshapen, "misshapen"),
        ] {
            let out = src.out(name);
            assert!(
                apply_declared_patches(&src.container, &[DeclaredPatch::Operations(p)], &[], &out)
                    .is_err(),
                "{name}"
            );
            assert!(!out.exists(), "{name}");
        }
    }

    #[test]
    fn the_report_counts_the_operation_targets_changed_elements() {
        let src = source_with_pack();
        let p = operation_patch(&src);
        let changed = p.changed_elements;
        let out = src.out("out.vindex3");
        let report =
            apply_declared_patches(&src.container, &[DeclaredPatch::Operations(p)], &[], &out)
                .unwrap();
        assert_eq!(report.elements, changed);
        assert_eq!(report.tensors, vec![TENSOR.to_string()]);
    }
}

mod sham {
    use super::*;

    fn zero_valued(src: &Source, a: Vec<u64>, b: Vec<u64>, rank: usize) -> LowRankColumnsOp {
        LowRankColumnsOp::bind(
            &src.container,
            TENSOR,
            PROFILE,
            rank,
            &OperandBits::F64(a),
            &OperandBits::F64(b),
            COLUMNS.to_vec(),
        )
        .unwrap()
    }

    const F64_NEG_ZERO: u64 = 1 << 63;

    #[test]
    fn a_zero_valued_op_compiles_to_the_bases_bytes_and_a_consistent_ledger() {
        let src = source_with_pack();
        let [rows, _] = tensor_shape(&src.container, TENSOR).unwrap();
        // a = -0 everywhere: zero by operands, sign notwithstanding.
        let op = zero_valued(
            &src,
            vec![F64_NEG_ZERO; rows],
            vec![F64_ONE; COLUMNS.len()],
            1,
        );
        assert!(StructuredOpRecord::LowRankColumns(op.clone())
            .is_zero_valued()
            .unwrap());
        let resolved = op.resolve_tensor(&src.container).unwrap();
        assert!(resolved.is_sham().unwrap());
        assert_eq!(resolved.changed_elements, 0);
        assert!(
            matches!(op.resolve(&src.container), Err(WeightPatchError::Empty)),
            "the element form cannot carry a sham"
        );

        let out = compile(&src, &[WeightOpRef::LowRankColumns(&op)], "sham.vindex3");
        for representation in [CANONICAL, DERIVED] {
            for tensor in [TENSOR, OTHER_TENSOR] {
                assert_eq!(
                    tensor_bytes(&out, representation, tensor),
                    tensor_bytes(&src.container, representation, tensor),
                    "{representation} / {tensor}"
                );
            }
        }
        let ledger = read_ledger(&out);
        assert_eq!(ledger.schema, WEIGHT_PATCHES_SCHEMA_V4);
        assert_eq!(ledger.targets[0].changed_elements, Some(0));
        assert_eq!(
            ledger.targets[0].base_tensor_sha256,
            ledger.targets[0].result_tensor_sha256
        );
        assert!(check_ledger(&out, &index_of(&out)).is_empty());

        // A sham that claims a change, or a non-sham that claims none, is a defect.
        let d = defects_after(&out, |l| {
            l.targets[0].changed_elements = Some(1);
            l.targets[0].operations[0].changed_elements = 1;
        });
        assert!(d.iter().any(|x| x.contains("zero-valued")), "{d:?}");
    }

    /// Zero-valued is decided per term: K = 2 with term 0 zero in `a` and term 1 zero in `b` is zero-valued; a term
    /// with both factors nonzero anywhere is not.
    #[test]
    fn zero_valued_is_decided_term_by_term() {
        let src = source_with_pack();
        let [rows, _] = tensor_shape(&src.container, TENSOR).unwrap();
        let n = COLUMNS.len();
        // a is rows x 2 row-major: column 0 zero, column 1 one. b is 2 x n: row 0 one, row 1 zero.
        let a: Vec<u64> = (0..rows * 2)
            .map(|i| if i % 2 == 0 { 0 } else { F64_ONE })
            .collect();
        let b: Vec<u64> = (0..2 * n)
            .map(|i| if i < n { F64_ONE } else { 0 })
            .collect();
        let crossed = zero_valued(&src, a.clone(), b, 2);
        assert!(StructuredOpRecord::LowRankColumns(crossed)
            .is_zero_valued()
            .unwrap());

        let mut b_live: Vec<u64> = (0..2 * n)
            .map(|i| if i < n { F64_ONE } else { 0 })
            .collect();
        b_live[n] = F64_ONE; // term 1 now has a nonzero factor on both sides at column 0
        let live = zero_valued(&src, a, b_live, 2);
        assert!(!StructuredOpRecord::LowRankColumns(live)
            .is_zero_valued()
            .unwrap());
    }

    #[test]
    fn a_removal_is_never_a_sham() {
        let src = source_with_pack();
        let [rows, _] = tensor_shape(&src.container, TENSOR).unwrap();
        let rd = larql_vindex::format::vindex3::weight_patch::StructuredWeightOp::bind(
            &src.container,
            TENSOR,
            PROFILE,
            &OperandBits::F64(vec![0; rows]),
            larql_vindex::format::vindex3::weight_patch::structured::ScalarBits::F64(F64_ONE),
            vec![40],
        )
        .unwrap();
        assert!(!StructuredOpRecord::RemoveDirection(rd)
            .is_zero_valued()
            .unwrap());
    }
}
