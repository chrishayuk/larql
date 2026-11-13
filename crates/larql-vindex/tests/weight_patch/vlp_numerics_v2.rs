//! The structured-op interpreter against the frozen VLP numerics v2 conformance suite
//! (`conformance/vlp-numerics/v2/`): `add_low_rank_columns`. The vectors were made by a separate exact-rational
//! Python generator and reproduced 91/91 by a blind evaluator written from the specification alone; this
//! implementation shares no code with either.
//!
//! **What passing establishes:** every output byte and every refusal code of all 91 cases, and every recorded
//! wrong-semantics result under the matching mutation. **What it does not:** that a compiled container is
//! correct. That is for the container-level tests and the independent verifier.

use larql_vindex::format::vindex3::weight_patch::structured::{
    run, run_with, BaseTensor, Mutation, OpKind, OperandBits, Profile, RefusalCode, StructuredOp,
    TensorRole,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::vlp_numerics::{case, ours, outcome};

const VECTORS: &str = include_str!("../../../../conformance/vlp-numerics/v2/vectors.json");
const FREEZE: &str = include_str!("../../../../conformance/vlp-numerics/v2/FREEZE.json");
/// The case count of the frozen v2 suite.
const V2_CASES: usize = 91;
/// The op every v2 case declares.
const V2_KIND: &str = "add_low_rank_columns";

/// Each v2 mutation and the contract's name for the wrong semantics it injects (v2 SPEC §6).
const MUTATIONS: [(Mutation, &str); 3] = [
    (Mutation::ZeroPaddedFull, "zero_padded_full"),
    (Mutation::ZeroUpdateNotKept, "zero_update_not_kept"),
    (Mutation::LowRankFromZero, "low_rank_from_zero"),
];

/// The case whose column is negative. `usize` columns cannot hold it, so it never reaches the interpreter: the
/// `.vlp` reader refuses it while parsing. [`a_negative_column_is_refused_before_the_interpreter`] pins its
/// expected code; every other test runs the remaining cases.
const NEGATIVE_COLUMN_CASE: &str = "ref-column-negative";

fn all_cases() -> Vec<Value> {
    let doc: Value = serde_json::from_str(VECTORS).expect("vectors.json parses");
    doc["cases"].as_array().unwrap().clone()
}

/// Every case the interpreter's types can express.
fn cases() -> Vec<Value> {
    all_cases()
        .into_iter()
        .filter(|c| c["id"] != NEGATIVE_COLUMN_CASE)
        .collect()
}

#[test]
fn a_negative_column_is_refused_before_the_interpreter() {
    let all = all_cases();
    let negative: Vec<&Value> = all
        .iter()
        .filter(|c| {
            c["op"]["columns"]
                .as_array()
                .unwrap()
                .iter()
                .any(|x| x.as_u64().is_none())
        })
        .collect();
    assert_eq!(
        negative.len(),
        1,
        "exactly one case is not a usize column list"
    );
    assert_eq!(negative[0]["id"], NEGATIVE_COLUMN_CASE);
    assert_eq!(
        outcome(&negative[0]["expect"]),
        Err(RefusalCode::ColumnsInvalid.as_str().to_string())
    );
}

#[test]
fn the_suite_is_the_frozen_v2() {
    let freeze: Value = serde_json::from_str(FREEZE).unwrap();
    let want = freeze["sha256"]["vectors.json"].as_str().unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(VECTORS.as_bytes())),
        want,
        "vectors.json differs from the v2 freeze"
    );
    let cases = all_cases();
    assert_eq!(cases.len(), V2_CASES);
    assert!(cases
        .iter()
        .all(|c| c["op"]["kind"].as_str() == Some(V2_KIND)));
}

#[test]
fn every_case_matches_the_contract() {
    let mut failures = Vec::new();
    for c in cases() {
        let k = case(&c);
        let got = ours(&k, Mutation::None);
        let want = outcome(&c["expect"]);
        if got != want {
            failures.push(format!("{}: got {got:?}, want {want:?}", k.id));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} cases disagree:\n{}",
        failures.len(),
        V2_CASES - 1,
        failures.join("\n")
    );
}

/// Each v2 mutation reproduces exactly the recorded wrong result wherever the vectors record one, and the
/// contract's own result everywhere else (the generator records every variant that changes a result).
#[test]
fn every_mutation_reproduces_the_recorded_wrong_semantics() {
    let mut failures = Vec::new();
    let mut recorded = 0;
    for c in cases() {
        let k = case(&c);
        for (m, name) in MUTATIONS {
            let want = match c["under_wrong_semantics"].get(name) {
                Some(v) => {
                    recorded += 1;
                    outcome(v)
                }
                None => outcome(&c["expect"]),
            };
            let got = ours(&k, m);
            if got != want {
                failures.push(format!("{} under {name}: got {got:?}, want {want:?}", k.id));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{} disagreements:\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(recorded > 0);
}

/// Every case built to discriminate a semantics is caught by that semantics' mutation.
#[test]
fn every_targeted_mutation_changes_its_case() {
    for c in cases() {
        let k = case(&c);
        for d in c["discriminates"].as_array().unwrap() {
            let (m, _) = MUTATIONS
                .iter()
                .find(|(_, n)| *n == d.as_str().unwrap())
                .expect("a v2 variant");
            assert_ne!(
                ours(&k, *m),
                outcome(&c["expect"]),
                "{}: {d} is not caught",
                k.id
            );
        }
    }
}

/// The identity law the op exists for (§4): with `a = 0`, every element of rows × J, a `-0` included, comes back
/// bit for bit, and elements outside J, a NaN included, are never read.
#[test]
fn a_zero_operand_is_a_byte_identical_no_op() {
    let base = [0x8000u16, 0x7fc0, 0x3f80, 0x8000];
    let op = StructuredOp {
        numerics: Some(Profile::F64SeqRne1.id().to_string()),
        shape: [2, 2],
        kind: OpKind::AddLowRankColumns {
            rank: 1,
            a: OperandBits::F64(vec![0.0f64.to_bits(), (-0.0f64).to_bits()]),
            b: OperandBits::F64(vec![1.0f64.to_bits()]),
            columns: vec![0],
        },
        fixups: vec![],
    };
    let b = BaseTensor {
        shape: [2, 2],
        bits: &base,
        role: TensorRole::Ordinary,
    };
    assert_eq!(run(&op, &b).unwrap(), base.to_vec());
}

/// An operand of the wrong width is refused as the profile's mismatch, after the length checks.
#[test]
fn an_operand_of_the_wrong_width_is_numerics_invalid() {
    let base = [0x3f80u16];
    let op = StructuredOp {
        numerics: Some(Profile::F32SeqRne1.id().to_string()),
        shape: [1, 1],
        kind: OpKind::AddLowRankColumns {
            rank: 1,
            a: OperandBits::F64(vec![1.0f64.to_bits()]),
            b: OperandBits::F64(vec![1.0f64.to_bits()]),
            columns: vec![0],
        },
        fixups: vec![],
    };
    let b = BaseTensor {
        shape: [1, 1],
        bits: &base,
        role: TensorRole::Ordinary,
    };
    let e = run_with(&op, &b, Mutation::None).unwrap_err();
    assert_eq!(e.refusal_code(), Some(RefusalCode::NumericsInvalid));
}
