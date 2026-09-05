//! The direct K-quant plan: observed off the bytes, run by the codec's
//! kernel, sliced at the codec's stride, and switchable to the widening
//! path without a rebuild.
//!
//! These are the executor-level half of PARETO-1's v3 gate. The
//! kernel-level half (`represent/kquant_direct_tests.rs`) settles that
//! each codec's kernel agrees with its decoder on FOREIGN bytes; this
//! settles that the plan executor reaches that kernel — and only that
//! kernel — from a `WeightRows::KQuant` slab, and that the arm switch
//! selects what it says.

use super::super::kernels::ScalarF32;
use super::super::physical::{KQuantExecution, PhysicalProjectionPlan, KQUANT_EXEC_WIDEN};
use super::super::projector::{DenseProjector, WeightRows};
use crate::format::vindex3::opplan::exec::backend::{MatrixClass, WeightFormat};
use crate::format::vindex3::opplan::exec::production::select_cpu;
use crate::format::vindex3::opplan::exec::realization::{
    RealizationForm, RealizationId, RepresentationFacts, SelectionReason,
};
use crate::format::vindex3::opplan::planned::{Operation, PlannedOperand};
use crate::format::vindex3::opplan::OperandRef;
use crate::format::vindex3::represent::codec::{RepresentationExtent, RequiredAccess};
use crate::format::vindex3::represent::kquant::{KQuant, COMPILABLE, Q4_K, Q6_K, Q8_0};

/// Rows and a width every codec's block divides: 512 is 16 Q8_0 blocks
/// and 2 super-blocks, so each row has more than one block and a scale
/// read at the wrong offset would be caught.
const ROWS: usize = 3;
const IN_DIM: usize = 512;

/// Bound on accumulation-order disagreement in units of the row's own
/// magnitude scale `sum|w x|` — the quantity such error is bounded by.
/// The elementwise relative metric is blind to cancellation and is
/// printed for the record only; see `exec/tests/kquant_projection.rs`.
const ACCUMULATION_ORDER_NORMALISED: f64 = 5e-6;

/// Weights whose magnitude varies by row and within a row, so blocks get
/// different scales and rows cannot be confused for each other.
fn weights() -> Vec<f32> {
    (0..ROWS * IN_DIM)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.03 * (1.0 + (i / IN_DIM) as f32))
        .collect()
}

fn activation() -> Vec<f32> {
    (0..IN_DIM).map(|i| ((i % 13) as f32 - 6.0) / 9.0).collect()
}

/// The authority: decode the same stored bytes, then the literal scalar
/// transcription — element order, no library reassociation.
fn decode_then_scalar(codec: KQuant, blocks: &[u8], x: &[f32], rows: usize) -> Vec<f32> {
    let decoded = codec.decode(blocks, rows * IN_DIM, "t").expect("decode");
    let mut out = vec![0.0f32; rows];
    ScalarF32.project_rows(WeightRows::F32(&decoded), x, &mut out);
    out
}

fn worst_relative(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let denom = (x.abs().max(y.abs()) as f64).max(1e-6);
            ((*x as f64) - (*y as f64)).abs() / denom
        })
        .fold(0.0f64, f64::max)
}

/// Worst per-row difference over that row's `sum|w x|`.
fn worst_normalised(a: &[f32], b: &[f32], w: &[f32], x: &[f32]) -> f64 {
    let k = x.len();
    a.iter()
        .zip(b)
        .enumerate()
        .map(|(r, (p, q))| {
            let scale: f64 = w[r * k..(r + 1) * k]
                .iter()
                .zip(x)
                .map(|(wi, xi)| (wi * xi).abs() as f64)
                .sum();
            ((*p as f64) - (*q as f64)).abs() / scale.max(1e-12)
        })
        .fold(0.0f64, f64::max)
}

fn stored(codec: KQuant) -> Vec<u8> {
    codec.encode(&weights(), "t").expect("encode")
}

/// The bytes decide the plan, the plan names the format, and the plan's
/// kernel computes what decoding the same bytes computes.
///
/// Q8_0 is bit-for-bit: its kernel folds the f16 scale per element, which
/// is the decoder's own association. The super-block codecs accumulate
/// in a different order and are held to the layer-1 bound instead.
#[test]
fn the_observation_lands_on_the_direct_plan_for_every_codec() {
    let x = activation();
    for codec in COMPILABLE {
        let blocks = stored(codec);
        let rows = WeightRows::KQuant {
            blocks: &blocks,
            codec,
        };
        let plan = PhysicalProjectionPlan::for_resident(rows, IN_DIM);
        assert_eq!(plan, PhysicalProjectionPlan::FusedKQuant, "{}", codec.name);
        assert_eq!(plan.format(), WeightFormat::KQuant, "{}", codec.name);
        assert_eq!(rows.rows(IN_DIM), ROWS, "{}", codec.name);
        assert_eq!(rows.bytes(), blocks.len(), "{}", codec.name);

        let mut direct = vec![0.0f32; ROWS];
        plan.kernel().project_rows(rows, &x, &mut direct);
        let decoded = codec.decode(&blocks, ROWS * IN_DIM, "t").expect("decode");
        let want = decode_then_scalar(codec, &blocks, &x, ROWS);
        let norm = worst_normalised(&direct, &want, &decoded, &x);
        println!(
            "  layer-2 plan  {:<5} [{ROWS},{IN_DIM}]  normalised {norm:.2e}  (elementwise {:.2e})",
            codec.name,
            worst_relative(&direct, &want)
        );
        if codec == Q8_0 {
            assert_eq!(
                direct, want,
                "Q8_0 must be bit-for-bit with decode-then-multiply"
            );
        } else {
            assert!(
                norm < ACCUMULATION_ORDER_NORMALISED,
                "{}: {norm:e} of the row scale is more than accumulation order",
                codec.name
            );
        }
    }
}

/// A sub-slab is cut at the codec's row stride and computes exactly the
/// rows it names — the same rows of the full projection, bit for bit,
/// because a row's arithmetic does not depend on which slab it is in.
#[test]
fn slicing_rows_keeps_each_row_under_its_own_blocks() {
    let x = activation();
    for codec in COMPILABLE {
        let blocks = stored(codec);
        let all = WeightRows::KQuant {
            blocks: &blocks,
            codec,
        };
        let per_row = codec.row_bytes(IN_DIM).expect("512 is on every grid");
        let tail = all.slice_rows(IN_DIM, 1, 2);
        assert_eq!(tail.rows(IN_DIM), 2, "{}", codec.name);
        assert_eq!(tail.bytes(), 2 * per_row, "{}", codec.name);
        assert_eq!(
            tail.primary_addr(),
            all.primary_addr() + per_row,
            "{}: the sub-slab must start one row in",
            codec.name
        );

        let mut whole = vec![0.0f32; ROWS];
        let mut part = vec![0.0f32; 2];
        let kernel = PhysicalProjectionPlan::for_resident(all, IN_DIM).kernel();
        kernel.project_rows(all, &x, &mut whole);
        kernel.project_rows(tail, &x, &mut part);
        assert_eq!(
            part,
            whole[1..],
            "{}: a slab must compute its own rows",
            codec.name
        );
    }
}

/// A width off the block grid describes no rows, and the executor's row
/// count says so rather than rounding.
#[test]
fn a_width_off_the_block_grid_has_no_rows() {
    for codec in [Q6_K, Q4_K] {
        let blocks = stored(codec);
        let rows = WeightRows::KQuant {
            blocks: &blocks,
            codec,
        };
        // 512 - 32 is on Q8_0's grid but not on a super-block's.
        assert_eq!(rows.rows(IN_DIM - 32), 0, "{}", codec.name);
        assert_eq!(codec.row_bytes(IN_DIM - 32), None, "{}", codec.name);
        assert_eq!(codec.row_bytes(0), None, "{}", codec.name);
    }
    assert_eq!(Q8_0.row_bytes(IN_DIM - 32), Some(15 * Q8_0.bytes_per_block));
}

/// Only the exact word widens. Every other value — including a
/// plausible misspelling — is the default, and the run's own plan report
/// is what says which arm ran.
#[test]
fn the_env_value_selects_the_arm_exactly() {
    assert_eq!(
        KQuantExecution::from_env_value(None),
        KQuantExecution::Direct
    );
    assert_eq!(
        KQuantExecution::from_env_value(Some(KQUANT_EXEC_WIDEN)),
        KQuantExecution::Widen
    );
    assert_eq!(
        KQuantExecution::from_env_value(Some("  widen\n")),
        KQuantExecution::Widen,
        "surrounding whitespace is not a different word"
    );
    for not_the_word in ["direct", "", "WIDEN", "wideen", "widen=1"] {
        assert_eq!(
            KQuantExecution::from_env_value(Some(not_the_word)),
            KQuantExecution::Direct,
            "{not_the_word:?} must not select the widening arm"
        );
    }
    assert_eq!(KQuantExecution::default(), KQuantExecution::Direct);
}

/// A planned projection large enough to stream, so the size policy would
/// have had an opinion if it were consulted.
fn planned(operation: Operation) -> PlannedOperand {
    PlannedOperand {
        operand: OperandRef {
            object: "target.decoder_stack".into(),
            tensor: "0.mlp.up_proj.weight".into(),
            dtype: String::new(),
            shape: vec![17408, 5120],
        },
        operation,
        access: operation.access(),
        extent: RepresentationExtent::TERMINAL,
        layer: Some(0),
        logical_elements: 17408 * 5120,
    }
}

fn facts(label: &str) -> RepresentationFacts {
    RepresentationFacts::resolve(label)
}

/// The production selector binds a stored K-quant in place under the
/// direct arm and widens it under the other — and answers exactly as the
/// boolean ladder did before it, for everything that is not a stored
/// K-quant. The candidates come from the codec's declarations: a K-quant
/// label offers Direct(FusedKQuant) because the codec declares it, and an
/// F16 label offers no direct realization at all.
#[test]
fn the_policy_answers_the_pack_under_direct_and_f32_under_widen() {
    for operation in [
        Operation::Project(MatrixClass::AttentionProjection),
        Operation::Project(MatrixClass::FfnProjection),
        Operation::OutputHead,
    ] {
        for label in ["Q4_K", "Q6_K", "Q8_0"] {
            let direct =
                select_cpu(&planned(operation), &facts(label), KQuantExecution::Direct).unwrap();
            assert_eq!(
                direct.realization,
                RealizationId::cpu(RealizationForm::Direct(PhysicalProjectionPlan::FusedKQuant)),
                "{operation:?} {label} direct"
            );
            assert_eq!(direct.realization.format(), WeightFormat::KQuant);
            assert_eq!(direct.reason, SelectionReason::DirectDeclared);
            assert!(direct.candidates.contains(&direct.realization));
            // The widened arm is the v2 authority path: decode, then BLAS.
            let widened =
                select_cpu(&planned(operation), &facts(label), KQuantExecution::Widen).unwrap();
            assert_eq!(
                widened.realization,
                RealizationId::cpu(RealizationForm::Decode(PhysicalProjectionPlan::BlasF32)),
                "{operation:?} {label} widen"
            );
            assert_eq!(widened.realization.format(), WeightFormat::F32);
            assert_eq!(widened.reason, SelectionReason::ArmPrefersDecode);
            // Both arms saw the same candidates; the arm only orders them.
            assert_eq!(direct.candidates, widened.candidates);
        }
        // Not a stored K-quant: the arm is irrelevant and the answer is
        // decode, because F16 declares no direct realization.
        for arm in [KQuantExecution::Direct, KQuantExecution::Widen] {
            let plain = select_cpu(&planned(operation), &facts("F16"), arm).unwrap();
            assert_eq!(
                plain.realization.format(),
                WeightFormat::F32,
                "{operation:?} {arm:?}"
            );
            assert_eq!(plain.reason, SelectionReason::NoDirectRealization);
        }
    }
    // The bank is sliced at load; a K-quant bank provides the row access
    // that slicing needs, and no pack survives to bind.
    let bank = PlannedOperand {
        operation: Operation::ExpertBankSlice,
        access: RequiredAccess::RowRandom,
        ..planned(Operation::ExpertBankSlice)
    };
    let sliced = select_cpu(&bank, &facts("Q4_K"), KQuantExecution::Direct).unwrap();
    assert_eq!(
        sliced.realization,
        RealizationId::cpu(RealizationForm::SliceStored {
            convert: WeightFormat::F32
        })
    );
    // A stored NVFP4 pack keeps precedence over the arm: one label names
    // one codec, so the two claims cannot meet on one operand any more,
    // and the ladder's order is what says NVFP4 wins where it is declared.
    let pack = select_cpu(
        &planned(Operation::Project(MatrixClass::FfnProjection)),
        &facts("NVFP4"),
        KQuantExecution::Direct,
    )
    .unwrap();
    assert_eq!(pack.realization.format(), WeightFormat::Nvfp4);
}
