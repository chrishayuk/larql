//! The structured-op interpreter against the frozen VLP numerics v1 conformance suite
//! (`conformance/vlp-numerics/v1/`). The vectors were made by a separate, exact-rational Python formulation and
//! reproduced by an independent reviewer's evaluator; this implementation shares no code with either.
//!
//! **What passing establishes:** that this interpreter conforms to the tested numerical contract — every
//! output byte and every refusal code of all 77 cases, and every recorded wrong-semantics result under the
//! matching mutation. **What it does not:** that a real model patch is correct. That is for the container-level
//! tests and the independent G3 verifier.

use larql_vindex::format::vindex3::weight_patch::structured::{
    check_float_environment, run, run_with, BaseTensor, Canaries, Fixup, Mutation, OpKind,
    OperandBits, Profile, RefusalCode, ScalarBits, StructuredOp, TensorRole,
};
use serde_json::Value;
use sha2::{Digest, Sha256};

const VECTORS: &str = include_str!("../../../../conformance/vlp-numerics/v1/vectors.json");
const FREEZE: &str = include_str!("../../../../conformance/vlp-numerics/v1/FREEZE.json");
/// The case count of the frozen v1 suite.
const V1_CASES: usize = 77;

/// Each mutation and the contract's name for the wrong semantics it injects (SPEC §5).
const MUTATIONS: [(Mutation, &str); 11] = [
    (Mutation::FmaInReduction, "fma_in_reduction"),
    (Mutation::ReversedReduction, "reversed_reduction"),
    (Mutation::PairwiseReduction, "pairwise_reduction"),
    (Mutation::AlphaTimesUc, "alpha_times_uc"),
    (Mutation::F32Accumulation, "f32_accumulation"),
    (Mutation::SingleRounding, "single_rounding"),
    (Mutation::DoubleRounding, "double_rounding"),
    (Mutation::FtzDaz, "ftz_daz"),
    (Mutation::Bf16OutputFlush, "bf16_output_flush"),
    (Mutation::LowRankFromZero, "low_rank_from_zero"),
    (Mutation::ZeroKeepsSign, "zero_keeps_sign"),
];

fn doc() -> Value {
    serde_json::from_str(VECTORS).expect("vectors.json parses")
}

fn hex16(v: &Value) -> u16 {
    u16::from_str_radix(v.as_str().unwrap(), 16).unwrap()
}

fn bits_list(v: &Value) -> Vec<u16> {
    v.as_array().unwrap().iter().map(hex16).collect()
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

/// Operand bits at the width the case's profile declares: 8 hex digits for the f32 profile, else 16.
fn operands(v: &Value, f32_width: bool) -> OperandBits {
    let hexes = v.as_array().unwrap().iter().map(|x| x.as_str().unwrap());
    if f32_width {
        OperandBits::F32(hexes.map(|h| u32::from_str_radix(h, 16).unwrap()).collect())
    } else {
        OperandBits::F64(hexes.map(|h| u64::from_str_radix(h, 16).unwrap()).collect())
    }
}

pub(crate) struct Case {
    pub(crate) id: String,
    op: StructuredOp,
    base_shape: [usize; 2],
    base_bits: Vec<u16>,
    role: TensorRole,
}

pub(crate) fn case(c: &Value) -> Case {
    let op = &c["op"];
    // An absent profile and a JSON `null` one are both "no profile".
    let numerics = op
        .get("numerics")
        .and_then(Value::as_str)
        .map(str::to_string);
    let f32_width = numerics.as_deref() == Some(Profile::F32SeqRne1.id());
    let shape = usizes(&op["shape"]);
    let kind = match op["kind"].as_str().unwrap() {
        "remove_direction" => {
            let alpha = op["alpha"].as_str().unwrap();
            OpKind::RemoveDirection {
                u: operands(&op["u"], f32_width),
                alpha: if f32_width {
                    ScalarBits::F32(u32::from_str_radix(alpha, 16).unwrap())
                } else {
                    ScalarBits::F64(u64::from_str_radix(alpha, 16).unwrap())
                },
                columns: usizes(&op["columns"]),
            }
        }
        "add_low_rank" => OpKind::AddLowRank {
            rank: op["rank"].as_u64().unwrap() as usize,
            a: operands(&op["a"], f32_width),
            b: operands(&op["b"], f32_width),
        },
        "add_low_rank_columns" => OpKind::AddLowRankColumns {
            rank: op["rank"].as_u64().unwrap() as usize,
            a: operands(&op["a"], f32_width),
            b: operands(&op["b"], f32_width),
            columns: usizes(&op["columns"]),
        },
        "zero_columns" => OpKind::ZeroColumns {
            columns: usizes(&op["columns"]),
        },
        other => panic!("unknown op kind {other}"),
    };
    let fixups = op["fixups"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| Fixup {
            row: f[0].as_u64().unwrap() as usize,
            col: f[1].as_u64().unwrap() as usize,
            bits: hex16(&f[2]),
        })
        .collect();
    let bs = usizes(&c["base"]["shape"]);
    Case {
        id: c["id"].as_str().unwrap().to_string(),
        op: StructuredOp {
            numerics,
            shape: [shape[0], shape[1]],
            kind,
            fixups,
        },
        base_shape: [bs[0], bs[1]],
        base_bits: bits_list(&c["base"]["bf16"]),
        role: if c.get("tensor_role").and_then(Value::as_str) == Some("embed_tokens") {
            TensorRole::TokenEmbedding
        } else {
            TensorRole::Ordinary
        },
    }
}

/// An outcome in the vectors' terms: `Ok(bits)` or `Err(code)`.
pub(crate) fn outcome(v: &Value) -> Result<Vec<u16>, String> {
    match v.get("refuse") {
        Some(code) => Err(code.as_str().unwrap().to_string()),
        None => Ok(bits_list(&v["bf16"])),
    }
}

pub(crate) fn ours(c: &Case, m: Mutation) -> Result<Vec<u16>, String> {
    let base = BaseTensor {
        shape: c.base_shape,
        bits: &c.base_bits,
        role: c.role,
    };
    run_with(&c.op, &base, m).map_err(|e| {
        e.refusal_code()
            .expect("an op refusal, not the environment")
            .as_str()
            .to_string()
    })
}

#[test]
fn the_suite_is_the_frozen_v1() {
    let freeze: Value = serde_json::from_str(FREEZE).unwrap();
    let want = freeze["sha256"]["vectors.json"].as_str().unwrap();
    assert_eq!(
        format!("{:x}", Sha256::digest(VECTORS.as_bytes())),
        want,
        "vectors.json differs from the v1 freeze"
    );
    assert_eq!(doc()["cases"].as_array().unwrap().len(), V1_CASES);
}

#[test]
fn the_float_environment_is_ieee() {
    check_float_environment().expect("this platform must run the contract with gradual underflow");
}

#[test]
fn every_case_matches_the_contract() {
    let mut failures = Vec::new();
    for c in doc()["cases"].as_array().unwrap() {
        let k = case(c);
        let got = ours(&k, Mutation::None);
        if got != outcome(&c["expect"]) {
            failures.push(format!(
                "{}: got {got:?}, want {:?}",
                k.id,
                outcome(&c["expect"])
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {V1_CASES} cases disagree:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn run_is_run_with_no_mutation() {
    for c in doc()["cases"].as_array().unwrap() {
        let k = case(c);
        let base = BaseTensor {
            shape: k.base_shape,
            bits: &k.base_bits,
            role: k.role,
        };
        assert_eq!(
            run(&k.op, &base),
            run_with(&k.op, &base, Mutation::None),
            "{}",
            k.id
        );
    }
}

/// Each mutation reproduces exactly the recorded wrong result wherever the vectors record one, and the
/// contract's own result everywhere else (the generator records every variant that changes a result).
#[test]
fn every_mutation_reproduces_the_recorded_wrong_semantics() {
    let mut failures = Vec::new();
    let mut recorded = 0;
    for c in doc()["cases"].as_array().unwrap() {
        let k = case(c);
        let wrong = &c["under_wrong_semantics"];
        for (m, name) in MUTATIONS {
            let want = match wrong.get(name) {
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
    for c in doc()["cases"].as_array().unwrap() {
        let k = case(c);
        for d in c["discriminates"].as_array().unwrap() {
            let (m, _) = MUTATIONS
                .iter()
                .find(|(_, n)| *n == d.as_str().unwrap())
                .expect("a known variant");
            assert_ne!(
                ours(&k, *m),
                outcome(&c["expect"]),
                "{}: {d} is not caught",
                k.id
            );
        }
    }
}

#[test]
fn every_refusal_code_is_exercised_and_spelled_as_the_contract() {
    let doc = doc();
    let listed: Vec<&str> = doc["refusals"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let mut ours: Vec<&str> = RefusalCode::ALL.iter().map(|c| c.as_str()).collect();
    let mut theirs = listed.clone();
    ours.sort();
    theirs.sort();
    assert_eq!(ours, theirs);
    for code in listed {
        assert!(doc["cases"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["expect"].get("refuse").and_then(Value::as_str) == Some(code)));
    }
}

#[test]
fn profile_ids_are_the_contract_ids() {
    let doc = doc();
    let mut ids: Vec<&str> = doc["profiles"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    let mut ours: Vec<&str> = Profile::ALL.iter().map(|p| p.id()).collect();
    ids.sort();
    ours.sort();
    assert_eq!(ids, ours);
}

/// Not a vector case: an API-level mismatch between the profile and the operand width is refused as an invalid
/// profile, never reinterpreted.
#[test]
fn operands_at_the_wrong_width_are_refused() {
    let base = [0x3f80u16];
    let b = BaseTensor {
        shape: [1, 1],
        bits: &base,
        role: TensorRole::Ordinary,
    };
    for (numerics, u, alpha) in [
        (
            Profile::F64SeqRne1,
            OperandBits::F32(vec![0x3f80_0000]),
            ScalarBits::F32(0x3f80_0000),
        ),
        (
            Profile::F32SeqRne1,
            OperandBits::F64(vec![1.0f64.to_bits()]),
            ScalarBits::F64(1.0f64.to_bits()),
        ),
    ] {
        let op = StructuredOp {
            numerics: Some(numerics.id().to_string()),
            shape: [1, 1],
            kind: OpKind::RemoveDirection {
                u,
                alpha,
                columns: vec![0],
            },
            fixups: vec![],
        };
        let r = run(&op, &b).unwrap_err();
        assert_eq!(r.refusal_code(), Some(RefusalCode::NumericsInvalid));
        assert!(r.to_string().contains("numerics_invalid"), "{r}");
    }
    let op = StructuredOp {
        numerics: Some(Profile::F64SeqRne1.id().to_string()),
        shape: [1, 1],
        kind: OpKind::RemoveDirection {
            u: OperandBits::F64(vec![1.0f64.to_bits()]),
            alpha: ScalarBits::F32(0x3f80_0000),
            columns: vec![0],
        },
        fixups: vec![],
    };
    assert_eq!(
        run(&op, &b).unwrap_err().refusal_code(),
        Some(RefusalCode::NumericsInvalid)
    );
}

#[test]
fn a_shape_with_a_zero_extent_is_refused() {
    let op = StructuredOp {
        numerics: None,
        shape: [0, 1],
        kind: OpKind::ZeroColumns { columns: vec![0] },
        fixups: vec![],
    };
    let b = BaseTensor {
        shape: [0, 1],
        bits: &[],
        role: TensorRole::Ordinary,
    };
    assert_eq!(
        run(&op, &b).unwrap_err().refusal_code(),
        Some(RefusalCode::ShapeInvalid)
    );
}

#[test]
fn operand_bits_report_their_length() {
    assert!(OperandBits::F32(vec![]).is_empty());
    assert_eq!(OperandBits::F64(vec![0, 0]).len(), 2);
}

/// Each canary, falsified one at a time, is named; the honest measurement passes.
#[test]
fn each_float_environment_canary_is_checked() {
    let honest = || Canaries::measure();
    assert!(honest().verdict().is_ok());
    let mut c = honest();
    c.f64_half_min = 0;
    assert!(c
        .verdict()
        .unwrap_err()
        .to_string()
        .contains("f64 MIN_POSITIVE"));
    let mut c = honest();
    c.f64_sub_times_one = 0;
    assert!(c
        .verdict()
        .unwrap_err()
        .to_string()
        .contains("f64 subnormal"));
    let mut c = honest();
    c.f32_half_min = 0;
    assert!(c
        .verdict()
        .unwrap_err()
        .to_string()
        .contains("f32 MIN_POSITIVE"));
    let mut c = honest();
    c.f32_sub_times_one = 0;
    assert!(c
        .verdict()
        .unwrap_err()
        .to_string()
        .contains("f32 subnormal"));
}

fn f64s(xs: &[f64]) -> OperandBits {
    OperandBits::F64(xs.iter().map(|x| x.to_bits()).collect())
}

fn low_rank(
    shape: [usize; 2],
    rank: usize,
    a: &[f64],
    b: &[f64],
    fixups: Vec<Fixup>,
) -> StructuredOp {
    StructuredOp {
        numerics: Some(Profile::F64SeqRne1.id().to_string()),
        shape,
        kind: OpKind::AddLowRank {
            rank,
            a: f64s(a),
            b: f64s(b),
        },
        fixups,
    }
}

fn code(op: &StructuredOp, base: &[u16], shape: [usize; 2]) -> Option<RefusalCode> {
    let b = BaseTensor {
        shape,
        bits: base,
        role: TensorRole::Ordinary,
    };
    run(op, &b).err().and_then(|e| e.refusal_code())
}

/// Declared sizes whose products overflow `usize` are refused, never wrapped or panicked on.
#[test]
fn declared_sizes_that_overflow_are_refused() {
    let huge = usize::MAX / 2 + 1;
    let zc = StructuredOp {
        numerics: None,
        shape: [huge, 2],
        kind: OpKind::ZeroColumns { columns: vec![0] },
        fixups: vec![],
    };
    assert_eq!(code(&zc, &[], [huge, 2]), Some(RefusalCode::ShapeInvalid));
    let lr = low_rank([1, 1], huge, &[], &[], vec![]);
    assert_eq!(
        code(&lr, &[0x3f80], [1, 1]),
        Some(RefusalCode::ShapeInvalid)
    );
}

/// Two defects at once: the earlier phase of SPEC §4 decides, and within the compute phase the first failing
/// operation in iteration order does.
#[test]
fn with_two_defects_the_spec_order_decides() {
    let big = 2.0f64.powi(119);
    // unsorted fixups (parse) before a column beyond the tensor (header)
    let rd = StructuredOp {
        numerics: Some(Profile::F64SeqRne1.id().to_string()),
        shape: [1, 2],
        kind: OpKind::RemoveDirection {
            u: f64s(&[1.0]),
            alpha: ScalarBits::F64(1.0f64.to_bits()),
            columns: vec![5],
        },
        fixups: vec![
            Fixup {
                row: 0,
                col: 1,
                bits: 0x3f80,
            },
            Fixup {
                row: 0,
                col: 0,
                bits: 0x3f80,
            },
        ],
    };
    assert_eq!(
        code(&rd, &[0x3f80, 0x3f80], [1, 2]),
        Some(RefusalCode::FixupsInvalid)
    );
    // a column beyond the tensor (header) before a NaN base element (tensor bytes)
    let mut rd2 = rd.clone();
    rd2.fixups.clear();
    assert_eq!(
        code(&rd2, &[0x7fc0, 0x3f80], [1, 2]),
        Some(RefusalCode::ColumnsInvalid)
    );
    // element 0's chain overflow comes before element 1's intermediate overflow ...
    let a = [2.0f64.powi(600)];
    let first_overflow = low_rank(
        [1, 2],
        1,
        &a,
        &[big / 2.0f64.powi(600), 2.0f64.powi(600)],
        vec![],
    );
    assert_eq!(
        code(&first_overflow, &[0x7f7f, 0x0000], [1, 2]),
        Some(RefusalCode::Overflow)
    );
    // ... and the other way round
    let first_intermediate = low_rank(
        [1, 2],
        1,
        &a,
        &[2.0f64.powi(600), big / 2.0f64.powi(600)],
        vec![],
    );
    assert_eq!(
        code(&first_intermediate, &[0x0000, 0x7f7f], [1, 2]),
        Some(RefusalCode::NonFiniteIntermediate)
    );
    // the compute phase (overflow) comes before the fixups phase (a no-op fixup)
    let both = low_rank(
        [1, 2],
        1,
        &[1.0],
        &[big, 0.0],
        vec![Fixup {
            row: 0,
            col: 1,
            bits: 0x3f80,
        }],
    );
    assert_eq!(
        code(&both, &[0x7f7f, 0x3f80], [1, 2]),
        Some(RefusalCode::Overflow)
    );
}

/// The f32-accumulation mutation's exact-product rounding at f32's overflow tie: an f64 product that rounds
/// onto the tie, from an exact product just below it, is f32::MAX, not inf.
#[test]
fn f32_accumulation_rounds_below_the_f32_overflow_tie_to_max() {
    let op = StructuredOp {
        numerics: Some(Profile::F64SeqRne1.id().to_string()),
        shape: [1, 1],
        kind: OpKind::RemoveDirection {
            u: OperandBits::F64(vec![0x47ea_f286_af28_6bca]),
            alpha: ScalarBits::F64(0.0f64.to_bits()),
            columns: vec![0],
        },
        fixups: vec![],
    };
    let base = [0x3f98u16];
    let b = BaseTensor {
        shape: [1, 1],
        bits: &base,
        role: TensorRole::Ordinary,
    };
    let out = run_with(&op, &b, Mutation::F32Accumulation).expect("f32::MAX is finite");
    // alpha = 0, so w - 0*c = w: the base element comes back unchanged
    assert_eq!(out, vec![0x3f98]);
}

/// The canary, falsified by the hardware: on aarch64 setting FPCR.FZ makes `run` refuse with the environment
/// error before computing anything; restoring it makes the same op run.
#[cfg(target_arch = "aarch64")]
#[test]
fn run_refuses_under_hardware_flush_to_zero() {
    use larql_vindex::format::vindex3::weight_patch::structured::StructuredError;
    /// FPCR.FZ: flush-to-zero for single and double precision.
    const FPCR_FZ: u64 = 1 << 24;
    let op = low_rank([1, 1], 1, &[1.0], &[1.0], vec![]);
    let base = [0x3f80u16];
    let b = BaseTensor {
        shape: [1, 1],
        bits: &base,
        role: TensorRole::Ordinary,
    };
    let saved: u64;
    unsafe { std::arch::asm!("mrs {}, fpcr", out(reg) saved) };
    unsafe { std::arch::asm!("msr fpcr, {}", in(reg) saved | FPCR_FZ) };
    let under_ftz = run(&op, &b);
    unsafe { std::arch::asm!("msr fpcr, {}", in(reg) saved) };
    assert!(
        matches!(under_ftz, Err(StructuredError::Environment(_))),
        "{under_ftz:?}"
    );
    assert_eq!(run(&op, &b).unwrap(), vec![0x4000]);
}

/// The environment error is not an op refusal: it carries no contract code, and says what failed.
#[test]
fn an_environment_error_has_no_refusal_code() {
    use larql_vindex::format::vindex3::weight_patch::structured::{
        FloatEnvironmentError, StructuredError,
    };
    let e = StructuredError::Environment(FloatEnvironmentError(
        "an f64 subnormal is read as zero (DAZ)",
    ));
    assert_eq!(e.refusal_code(), None);
    assert!(e.to_string().contains("DAZ"), "{e}");
}
