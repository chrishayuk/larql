//! **Layer 2 of PARETO-1's v3 gate: a real projection through the plan
//! executor, direct against widened, on the SAME stored bytes.**
//!
//! Layer 1 (`represent/kquant_direct_tests.rs`) settles that each codec's
//! kernel agrees with its decoder on foreign bytes. What it cannot see
//! is everything between a container and a kernel: which format the
//! policy declares, whether the loader binds the stored blocks or a
//! derivative, whether the slice is cut at the right stride, whether the
//! executor observes `FusedKQuant` or quietly runs `BlasF32`, LARQL Q8,
//! or something it manufactured. A 34-vs-18-byte stride bug of exactly
//! that kind has already produced garbage in this workspace, so this
//! layer is load-bearing and not a formality.
//!
//! ```text
//! compiled pack  -> OperandStore -> load_weight(KQuant) -> FusedKQuant     the candidate
//! the SAME pack  -> OperandStore -> load_weight(F32)    -> BlasF32         rung A's authority
//! ```
//!
//! Both arms open the same container and read the same tensor. The
//! encoder does not re-enter the causal graph — it wrote the bytes both
//! arms read, and Amendment 2 removed it from the comparison.
//!
//! The mutations at the end are valued as highly as the agreement: a
//! gate that cannot return the other answer is not a gate.

use crate::format::filenames::INDEX_JSON;
use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::index::Vindex3Index;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::{WeightFormat, WeightSlice};
use crate::format::vindex3::opplan::exec::cpu::kernels::ScalarF32;
use crate::format::vindex3::opplan::exec::cpu::physical::project_matrix;
use crate::format::vindex3::opplan::exec::cpu::projector::{DenseProjector, WeightRows};
use crate::format::vindex3::opplan::exec::cpu::{ledger, PhysicalProjectionPlan};
use crate::format::vindex3::opplan::exec::operands::{
    OperandSource, OperandStore, RepresentationSource,
};
use crate::format::vindex3::opplan::exec::weights::{load_weight, LoadedWeight};
use crate::format::vindex3::opplan::OperandRef;
use crate::format::vindex3::represent::kquant::{self, KQuant, COMPILABLE, Q4_K, Q6_K, Q8_0};
use crate::format::vindex3::represent::{compile_representation, policy, RepresentSpec};
use serial_test::serial;

/// Bound on accumulation-order disagreement, in units of each row's own
/// magnitude scale — see [`worst_normalised`].
///
/// An f32 dot over `k` terms differs between two summation orders by at
/// most about `k * eps * sum|w x|` and typically `sqrt(k) * eps`: for
/// k = 5120 and eps = 6e-8 that is 4.3e-6 worst case and ~4e-7 typical.
/// Measured on this fixture the three arms sit at ~1e-7 (direct vs
/// scalar, BLAS vs scalar, direct vs BLAS), so this leaves an order of
/// magnitude over the typical and still refuses the worst case a wrong
/// scale or a wrong stride produces, which is O(1).
const ACCUMULATION_ORDER_NORMALISED: f64 = 5e-6;

fn spec(encoding: &str) -> RepresentSpec {
    RepresentSpec {
        encoding: encoding.to_string(),
        objects: Vec::new(),
        roles: policy::RolePolicy::default(),
        deployment: false,
        protect: policy::Protections::default(),
    }
}

/// The dense fixture, encoded, then compiled to `codec`. Returns the
/// source and the compiled container.
fn compiled(tmp: &tempfile::TempDir, codec: KQuant) -> (std::path::PathBuf, std::path::PathBuf) {
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let src = tmp.path().join("src.vindex3");
    let out = tmp.path().join(format!("{}.vindex3", codec.name));
    encode_fixture_container(dense_f32_model, &checkpoint, &src, "target");
    compile_representation(&src, &out, &spec(codec.name))
        .unwrap_or_else(|e| panic!("{} compiles the fixture: {e}", codec.name));
    (src, out)
}

/// The first two-dimensional tensor stored as `codec` in the compiled
/// container, as an operand reference.
fn a_stored_matrix(out: &std::path::Path, codec: KQuant) -> OperandRef {
    let index: Vindex3Index =
        serde_json::from_str(&std::fs::read_to_string(out.join(INDEX_JSON)).unwrap()).unwrap();
    let entry = index
        .representations
        .values()
        .find(|e| e.encoding == codec.name)
        .unwrap_or_else(|| panic!("a {} representation was registered", codec.name));
    let (header, _) = read_segment_header(&out.join(&entry.segment)).unwrap();
    let t = header
        .tensors
        .iter()
        .find(|t| t.dtype == codec.name && t.shape.len() == 2)
        .unwrap_or_else(|| panic!("the pack holds a two-dimensional {} tensor", codec.name));
    OperandRef {
        object: entry.object.clone(),
        tensor: t.name.clone(),
        dtype: t.dtype.clone(),
        shape: t.shape.clone(),
    }
}

fn open(out: &std::path::Path, codec: KQuant) -> OperandStore {
    let inspection = inspect_container(out, false).unwrap();
    OperandStore::open_for(
        out,
        &inspection,
        Some(codec.name),
        RepresentationSource::Stored,
    )
    .expect("the compiled pack binds")
}

fn activation(k: usize) -> Vec<f32> {
    (0..k).map(|i| ((i % 23) as f32 - 11.0) / 13.0).collect()
}

/// Elementwise worst relative difference — printed for the record, NOT
/// the gate. It is the metric layer 1 used, and it is blind to
/// cancellation: an output near zero has a large relative error for a
/// tiny absolute one, which is a property of the output and not of
/// either arm. The 64-row fixture produced exactly such a row.
fn worst_relative(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let denom = (x.abs().max(y.abs()) as f64).max(1e-6);
            ((*x as f64) - (*y as f64)).abs() / denom
        })
        .fold(0.0f64, f64::max)
}

/// Worst per-row difference in units of that row's magnitude scale,
/// `sum_i |w_ri * x_i|` — the quantity accumulation-order error is
/// actually bounded by. The gate.
fn worst_normalised(a: &[f32], b: &[f32], w: &[f32], x: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
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

/// One stored matrix per codec, through normal VINDEX loading, both ways.
///
/// Asserted, in order: the direct arm binds the stored blocks and the
/// widened arm binds an f32 image; neither manufactures anything; the
/// executor OBSERVES `FusedKQuant` for the one and `BlasF32` for the
/// other; the real projection path runs the direct plan (the process
/// ledger moves); and the two projections agree to accumulation order —
/// bit-for-bit against the scalar transcription for Q8_0.
// Serial: the widened arm goes through `StagedF32::stage`, which bumps
// the process-global staging counters a serial staged-test asserts an
// exact delta on. A concurrent map here corrupts that delta — it did,
// on the Linux CI runner, where this and the counter test overlapped.
#[test]
#[serial]
fn a_stored_matrix_projects_the_same_direct_and_widened() {
    for codec in COMPILABLE {
        let tmp = tempfile::tempdir().unwrap();
        let (_src, out) = compiled(&tmp, codec);
        let op = a_stored_matrix(&out, codec);
        let store = open(&out, codec);
        let src: OperandSource<'_> = (&store).into();

        let direct = load_weight(src, &op, WeightFormat::KQuant).expect("binds the pack");
        let widened = load_weight(src, &op, WeightFormat::F32).expect("widens the pack");
        assert!(
            matches!(direct, LoadedWeight::KQuant { codec: c, .. } if c == codec),
            "{}: the direct arm must hold the stored blocks under their own codec",
            codec.name
        );
        assert!(
            widened.is_widened_f32(),
            "{}: the authority arm is f32",
            codec.name
        );
        assert_eq!(
            store.runtime_quantised(),
            0,
            "{}: neither arm may manufacture a representation",
            codec.name
        );

        let (out_dim, in_dim) = (op.shape[0], op.shape[1]);
        let x = activation(in_dim);
        let direct_rows = direct.slice().rows(out_dim, in_dim).unwrap();
        let widened_rows = widened.slice().rows(out_dim, in_dim).unwrap();
        assert_eq!(
            PhysicalProjectionPlan::for_resident(direct_rows, in_dim),
            PhysicalProjectionPlan::FusedKQuant,
            "{}: the executor must observe the direct plan, not BLAS, LARQL Q8 or NVFP4",
            codec.name
        );
        assert_eq!(
            PhysicalProjectionPlan::for_resident(widened_rows, in_dim),
            PhysicalProjectionPlan::BlasF32,
            "{}: the authority arm is rung A's BLAS f32 path",
            codec.name
        );

        // Through the REAL projection path, and proven to have run the
        // direct plan by the process ledger rather than by inference.
        let before = ledger().get(PhysicalProjectionPlan::FusedKQuant).calls;
        let y_direct = project_matrix(&direct.slice(), &x, out_dim, in_dim).unwrap();
        let y_widened = project_matrix(&widened.slice(), &x, out_dim, in_dim).unwrap();
        assert!(
            ledger().get(PhysicalProjectionPlan::FusedKQuant).calls > before,
            "{}: the executor did not record a FusedKQuant call",
            codec.name
        );

        // Three arms over the SAME widened image: the candidate, the
        // authority (BLAS), and the literal scalar transcription. Printing
        // all three pairs separates the candidate's disagreement from the
        // authority's own reassociation — BLAS is not the scalar loop
        // either, and a reader of one number could not tell whose
        // arithmetic it was measuring.
        let image = widened.slice().as_f32().unwrap();
        let mut scalar = vec![0.0f32; out_dim];
        ScalarF32.project_rows(widened_rows, &x, &mut scalar);
        let d_vs_b = worst_normalised(&y_direct, &y_widened, image, &x);
        let d_vs_s = worst_normalised(&y_direct, &scalar, image, &x);
        let b_vs_s = worst_normalised(&y_widened, &scalar, image, &x);
        println!(
            "  layer-2  {:<5} {} [{out_dim},{in_dim}]  normalised: direct-vs-BLAS {d_vs_b:.2e}  \
             direct-vs-scalar {d_vs_s:.2e}  BLAS-vs-scalar {b_vs_s:.2e}   \
             (elementwise direct-vs-BLAS {:.2e})",
            codec.name,
            op.tensor,
            worst_relative(&y_direct, &y_widened)
        );
        assert!(
            d_vs_b < ACCUMULATION_ORDER_NORMALISED,
            "{}: direct and the BLAS authority disagree by {d_vs_b:e} of the row scale — more \
             than accumulation order",
            codec.name
        );
        assert!(
            d_vs_s < ACCUMULATION_ORDER_NORMALISED,
            "{}: direct and the scalar transcription disagree by {d_vs_s:e} of the row scale",
            codec.name
        );

        // Against the literal scalar transcription of the SAME widened
        // image, Q8_0 is exact: the kernel folds the f16 scale per element
        // in the decoder's own association.
        if codec == Q8_0 {
            assert_eq!(
                y_direct, scalar,
                "Q8_0 direct must be bit-for-bit with the scalar decode-then-multiply"
            );
        }
    }
}

/// The comparison can fail. Perturbing one stored byte in the pack must
/// move the direct projection — otherwise the agreement above could be
/// two arms ignoring the same bytes.
#[test]
fn the_direct_projection_reads_the_stored_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    let (_src, out) = compiled(&tmp, Q8_0);
    let op = a_stored_matrix(&out, Q8_0);
    let store = open(&out, Q8_0);
    let LoadedWeight::KQuant { blocks, codec } =
        load_weight((&store).into(), &op, WeightFormat::KQuant).unwrap()
    else {
        panic!("the direct arm binds blocks");
    };
    let (out_dim, in_dim) = (op.shape[0], op.shape[1]);
    let x = activation(in_dim);
    let clean = project_matrix(
        &WeightSlice::KQuant {
            blocks: &blocks,
            codec,
        },
        &x,
        out_dim,
        in_dim,
    )
    .unwrap();
    let mut dirty = blocks.clone();
    // A code byte in the middle of the stream, past any block header.
    let victim = dirty.len() / 2;
    dirty[victim] ^= 0x10;
    let perturbed = project_matrix(
        &WeightSlice::KQuant {
            blocks: &dirty,
            codec,
        },
        &x,
        out_dim,
        in_dim,
    )
    .unwrap();
    assert_ne!(
        clean, perturbed,
        "one stored byte changed and the projection did not"
    );
}

/// The direct format binds ONLY a stored K-quant. Asked for over the
/// source container's float tensor it refuses by name rather than
/// encoding one — a manufactured pack would be the artifact's
/// derivative, not the artifact.
#[test]
fn the_direct_format_refuses_a_tensor_that_is_not_a_stored_kquant() {
    let tmp = tempfile::tempdir().unwrap();
    let (src, _out) = compiled(&tmp, Q8_0);
    let inspection = inspect_container(&src, false).unwrap();
    let store = OperandStore::open(&src, &inspection).unwrap();
    // Any two-dimensional tensor of the source: take the compiled one's
    // name, which the source holds at its own precision.
    let op = {
        let compiled_op = a_stored_matrix(&_out, Q8_0);
        let dtype = store
            .stored_dtype(&compiled_op)
            .expect("the source holds the same tensor")
            .to_string();
        OperandRef {
            dtype,
            ..compiled_op
        }
    };
    assert!(
        kquant::lookup(&op.dtype).is_none(),
        "the source is not a K-quant"
    );
    let err = load_weight((&store).into(), &op, WeightFormat::KQuant)
        .expect_err("a float tensor is not a stored pack")
        .to_string();
    assert!(err.contains("not a K-quant"), "{err}");
    assert!(
        err.contains(&op.dtype),
        "the refusal names the stored dtype: {err}"
    );
    assert_eq!(
        store.runtime_quantised(),
        0,
        "the refusal must not have encoded anything"
    );
}

/// A shape the stored stream does not describe is refused at load, by
/// name, before any row could be read at the wrong stride.
#[test]
fn the_loader_refuses_a_geometry_the_stream_does_not_describe() {
    let tmp = tempfile::tempdir().unwrap();
    let (_src, out) = compiled(&tmp, Q8_0);
    let op = a_stored_matrix(&out, Q8_0);
    let store = open(&out, Q8_0);
    let src: OperandSource<'_> = (&store).into();
    assert!(
        load_weight(src, &op, WeightFormat::KQuant).is_ok(),
        "the control case"
    );

    // Twice the width: the stream is half what that shape needs.
    let wide = OperandRef {
        shape: vec![op.shape[0], op.shape[1] * 2],
        ..op.clone()
    };
    let err = load_weight(src, &wide, WeightFormat::KQuant)
        .expect_err("half the bytes the shape needs")
        .to_string();
    assert!(err.contains("do not describe shape"), "{err}");
    assert!(err.contains("wrong stride"), "{err}");

    // Half the rows: the stream is twice what that shape needs. The
    // longer direction is the one a prefix cut would have accepted.
    let short = OperandRef {
        shape: vec![op.shape[0] / 2, op.shape[1]],
        ..op.clone()
    };
    let err = load_weight(src, &short, WeightFormat::KQuant)
        .expect_err("twice the bytes the shape needs")
        .to_string();
    assert!(err.contains("do not describe shape"), "{err}");

    // A width off the block grid has no plan at all.
    let ragged = OperandRef {
        shape: vec![op.shape[0], op.shape[1] - 1],
        ..op.clone()
    };
    let err = load_weight(src, &ragged, WeightFormat::KQuant)
        .expect_err("a ragged row has no block layout")
        .to_string();
    assert!(err.contains("whole number"), "{err}");
}

/// Bytes of one codec labelled as another are refused at the slice, in
/// BOTH directions. Q6_K's 210-byte blocks under Q4_K's 144 would pass a
/// prefix cut — the stream is longer than Q4_K wants — and be walked at
/// the wrong stride; the exact-length rule is what closes that.
#[test]
fn bytes_under_another_codec_are_refused_not_reinterpreted() {
    let tmp = tempfile::tempdir().unwrap();
    let (_src, out) = compiled(&tmp, Q6_K);
    let op = a_stored_matrix(&out, Q6_K);
    let store = open(&out, Q6_K);
    let LoadedWeight::KQuant { blocks, codec } =
        load_weight((&store).into(), &op, WeightFormat::KQuant).unwrap()
    else {
        panic!("the direct arm binds blocks");
    };
    assert_eq!(codec, Q6_K);
    let (out_dim, in_dim) = (op.shape[0], op.shape[1]);

    let as_q6 = WeightSlice::KQuant {
        blocks: &blocks,
        codec: Q6_K,
    };
    assert!(as_q6.rows(out_dim, in_dim).is_ok(), "the control case");

    // Q6_K bytes, read as Q4_K: LONGER than the shape needs.
    let as_q4 = WeightSlice::KQuant {
        blocks: &blocks,
        codec: Q4_K,
    };
    let err = as_q4
        .rows(out_dim, in_dim)
        .expect_err("210-byte blocks are not 144-byte blocks")
        .to_string();
    assert!(err.contains("more than"), "{err}");
    assert!(
        err.contains("Q4_K"),
        "the refusal names the codec that was asked: {err}"
    );

    // Q6_K bytes, read as Q8_0: SHORTER than the shape needs (272 bytes a
    // row against 210), the direction a prefix cut also catches.
    let as_q8 = WeightSlice::KQuant {
        blocks: &blocks,
        codec: Q8_0,
    };
    let err = as_q8
        .rows(out_dim, in_dim)
        .expect_err("34-byte blocks over 256 elements need more than Q6_K's 210")
        .to_string();
    assert!(err.contains("resident"), "{err}");

    // The right codec at the wrong stride: twice the width.
    let err = as_q6
        .rows(out_dim, in_dim * 2)
        .expect_err("the stream holds half of that")
        .to_string();
    assert!(err.contains("resident"), "{err}");
    // A width off the grid.
    let err = as_q6
        .rows(out_dim, in_dim - 1)
        .expect_err("a ragged width is not on the block grid")
        .to_string();
    assert!(err.contains("whole number"), "{err}");

    // And the row view, once admitted, is cut at Q6_K's stride and not
    // at any other's.
    let WeightRows::KQuant { blocks: cut, .. } = as_q6.rows(out_dim, in_dim).unwrap() else {
        panic!("a K-quant slice yields K-quant rows");
    };
    assert_eq!(cut.len(), out_dim * Q6_K.row_bytes(in_dim).unwrap());
}

/// Rung 3c: one stored Q6_K pack, two realizations. The STORED footprint
/// is the container's recorded length under both — a property of the
/// representation instance, not of how it runs — while the resident,
/// staging and working-set accounting differ, and each reconciles with
/// the object the loader actually binds.
#[test]
#[serial]
fn a_stored_pack_has_one_stored_footprint_and_two_realization_costs() {
    use crate::format::vindex3::opplan::exec::accounting::{
        expectations, reconcile, stored_footprint, BlockGeometry, Bound,
    };
    use crate::format::vindex3::opplan::exec::backend::MatrixClass;
    use crate::format::vindex3::opplan::exec::cpu::physical::KQuantExecution;
    use crate::format::vindex3::opplan::exec::production::select_cpu;
    use crate::format::vindex3::opplan::exec::realization::{
        RealizationForm, RealizationRecord, RepresentationFacts,
    };
    use crate::format::vindex3::opplan::planned::{Operation, PlannedOperand};
    use crate::format::vindex3::represent::codec::RepresentationExtent;

    let tmp = tempfile::tempdir().unwrap();
    let (_src, out) = compiled(&tmp, Q6_K);
    let op = a_stored_matrix(&out, Q6_K);
    let store = open(&out, Q6_K);
    let operation = Operation::Project(MatrixClass::FfnProjection);
    let planned = PlannedOperand {
        operand: op.clone(),
        operation,
        access: operation.access(),
        extent: RepresentationExtent::TERMINAL,
        layer: Some(0),
        declared_representation: None,
        logical_elements: op.shape.iter().product(),
    };
    let facts = RepresentationFacts::resolve(Q6_K.name);
    let mut costs = Vec::new();
    for (arm, format) in [
        (KQuantExecution::Direct, WeightFormat::KQuant),
        (KQuantExecution::Widen, WeightFormat::F32),
    ] {
        let selection = select_cpu(&planned, &facts, arm).unwrap();
        assert_eq!(selection.realization.format(), format);
        let record = RealizationRecord {
            planned: planned.clone(),
            representation: Q6_K.name.to_string(),
            provider: facts.registered.as_ref().map(|r| r.identity.clone()),
            selection,
        };
        let expected = expectations(
            std::slice::from_ref(&record),
            |o| store.stored_len(o),
            BlockGeometry::executor(),
        );
        let loaded = load_weight((&store).into(), &op, format).unwrap();
        let observed = vec![Bound::one(&op, &loaded)
            .observed(operation, Some(0))
            .unwrap()];
        reconcile(&expected, &observed).unwrap_or_else(|e| panic!("{arm:?}: {e}"));
        costs.push((
            arm,
            stored_footprint(&expected).bytes,
            expected[0].declared_resident,
            expected[0].staging,
            expected[0].working_set(),
        ));
    }
    let (direct, decode) = (&costs[0], &costs[1]);
    assert_eq!(
        direct.1, decode.1,
        "one stored footprint: {direct:?} vs {decode:?}"
    );
    assert_eq!(direct.1, store.stored_len(&op).unwrap());
    assert!(
        direct.2 < decode.2,
        "direct holds the pack; decode holds an f32 image"
    );
    assert_eq!(direct.3, 0, "direct stages nothing");
    assert_eq!(decode.3, decode.2, "decode stages the image it then holds");
    assert!(direct.4 < decode.4);
    assert!(matches!(
        select_cpu(&planned, &facts, KQuantExecution::Direct)
            .unwrap()
            .realization
            .form,
        RealizationForm::Direct(PhysicalProjectionPlan::FusedKQuant)
    ));
}
