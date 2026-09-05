//! Layer 2 of PARETO-1's v3 gate on a REAL anchor: the largest and the
//! smallest stored K-quant matrices of the first compiled object, loaded
//! through normal VINDEX loading, direct against widened.
//!
//! Ignored unless `LARQL_KQUANT_ANCHOR` names a compiled container. The
//! committed fixture test beside this one (`kquant_projection.rs`) runs
//! the same claims on a 64-wide model in CI; this one runs them on
//! Qwen3.8-27B's `17408 x 5120` and `48 x 5120` shapes, where the row
//! stride is thousands of blocks and a stride bug has room to show.
//!
//! Numbers are PRINTED, not only thresholded, because layer 2 of the gate
//! is a record in the campaign file and "passed" hides whether the
//! agreement was 1e-7 or 4e-6.

use crate::format::filenames::INDEX_JSON;
use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::index::Vindex3Index;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::WeightFormat;
use crate::format::vindex3::opplan::exec::cpu::kernels::ScalarF32;
use crate::format::vindex3::opplan::exec::cpu::physical::project_matrix;
use crate::format::vindex3::opplan::exec::cpu::projector::DenseProjector;
use crate::format::vindex3::opplan::exec::cpu::{ledger, PhysicalProjectionPlan};
use crate::format::vindex3::opplan::exec::operands::{
    OperandSource, OperandStore, RepresentationSource,
};
use crate::format::vindex3::opplan::exec::weights::{load_weight, LoadedWeight};
use crate::format::vindex3::opplan::OperandRef;
use crate::format::vindex3::represent::kquant;

/// Names the compiled container to test against.
const ANCHOR_ENV: &str = "LARQL_KQUANT_ANCHOR";

/// Bound on accumulation-order disagreement in units of each row's own
/// magnitude scale `sum|w x|` — the same gate as the fixture test, for
/// the same reason: the elementwise relative metric is blind to
/// cancellation, and a 5120-wide real row cancels more often than a
/// fixture's. k = 5120 puts the typical `sqrt(k) * eps` at ~4e-7 and the
/// worst case `k * eps` at 3e-4 relative to the row scale; the arms sit
/// at ~1e-7 and this bound refuses an O(1) stride or scale error.
const ACCUMULATION_ORDER_NORMALISED: f64 = 5e-6;

fn activation(k: usize) -> Vec<f32> {
    (0..k).map(|i| ((i % 23) as f32 - 11.0) / 13.0).collect()
}

/// Printed for the record; blind to cancellation, so not the gate.
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

/// Worst per-row difference over that row's `sum|w x|`. The gate.
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

/// The largest and smallest two-dimensional K-quant tensors of the first
/// compiled object, with the codec the container declares.
fn extremes(root: &std::path::Path) -> (kquant::KQuant, Vec<OperandRef>) {
    let index: Vindex3Index =
        serde_json::from_str(&std::fs::read_to_string(root.join(INDEX_JSON)).unwrap()).unwrap();
    let map = index
        .precision_map
        .as_ref()
        .expect("an anchor declares its precision map");
    let codec = kquant::lookup(&map.encoding)
        .unwrap_or_else(|| panic!("`{}` is not a K-quant this executor runs", map.encoding));
    let entry = index
        .representations
        .values()
        .find(|e| e.encoding == codec.name)
        .expect("a compiled representation");
    let (header, _) = read_segment_header(&root.join(&entry.segment)).unwrap();
    let mut matrices: Vec<&_> = header
        .tensors
        .iter()
        .filter(|t| t.dtype == codec.name && t.shape.len() == 2)
        .collect();
    assert!(
        !matrices.is_empty(),
        "the pack holds two-dimensional {} tensors",
        codec.name
    );
    matrices.sort_by_key(|t| t.shape.iter().product::<usize>());
    let pick = |t: &crate::format::vindex3::encode::segment::SegmentTensor| OperandRef {
        object: entry.object.clone(),
        tensor: t.name.clone(),
        dtype: t.dtype.clone(),
        shape: t.shape.clone(),
    };
    let mut ops = vec![pick(matrices[0])];
    if matrices.len() > 1 {
        ops.push(pick(matrices[matrices.len() - 1]));
    }
    (codec, ops)
}

#[test]
#[ignore = "needs LARQL_KQUANT_ANCHOR naming a compiled K-quant container"]
fn a_real_anchor_projects_the_same_direct_and_widened() {
    let Some(root) = std::env::var_os(ANCHOR_ENV).map(std::path::PathBuf::from) else {
        panic!("set {ANCHOR_ENV} to a compiled container");
    };
    let (codec, ops) = extremes(&root);
    let inspection = inspect_container(&root, false).unwrap();
    let store = OperandStore::open_for(
        &root,
        &inspection,
        Some(codec.name),
        RepresentationSource::Stored,
    )
    .expect("the anchor binds under `stored`");
    let src: OperandSource<'_> = (&store).into();

    for op in &ops {
        let started = std::time::Instant::now();
        let direct = load_weight(src, op, WeightFormat::KQuant).expect("binds the pack");
        let load_direct = started.elapsed();
        let started = std::time::Instant::now();
        let widened = load_weight(src, op, WeightFormat::F32).expect("widens the pack");
        let load_widened = started.elapsed();
        assert!(matches!(direct, LoadedWeight::KQuant { codec: c, .. } if c == codec));
        assert!(widened.is_widened_f32());
        assert_eq!(store.runtime_quantised(), 0, "nothing may be manufactured");

        let (out_dim, in_dim) = (op.shape[0], op.shape[1]);
        let x = activation(in_dim);
        assert_eq!(
            PhysicalProjectionPlan::for_resident(
                direct.slice().rows(out_dim, in_dim).unwrap(),
                in_dim
            ),
            PhysicalProjectionPlan::FusedKQuant
        );
        assert_eq!(
            PhysicalProjectionPlan::for_resident(
                widened.slice().rows(out_dim, in_dim).unwrap(),
                in_dim
            ),
            PhysicalProjectionPlan::BlasF32
        );

        let before = ledger().get(PhysicalProjectionPlan::FusedKQuant).calls;
        // Each arm twice: the first direct call pays the kernel pool's
        // start-up, and a "22 ms for 261 KB" first reading is that cost,
        // not the kernel's. The second call is reported.
        let y_direct = project_matrix(&direct.slice(), &x, out_dim, in_dim).unwrap();
        let started = std::time::Instant::now();
        let again = project_matrix(&direct.slice(), &x, out_dim, in_dim).unwrap();
        let t_direct = started.elapsed();
        assert_eq!(y_direct, again, "the direct arm must be deterministic");
        let y_widened = project_matrix(&widened.slice(), &x, out_dim, in_dim).unwrap();
        let started = std::time::Instant::now();
        let again = project_matrix(&widened.slice(), &x, out_dim, in_dim).unwrap();
        let t_widened = started.elapsed();
        assert_eq!(y_widened, again, "the BLAS arm must be deterministic");
        assert!(ledger().get(PhysicalProjectionPlan::FusedKQuant).calls > before);

        // The literal scalar transcription over the same widened image,
        // so the authority's own reassociation is visible beside the
        // candidate's.
        let image = widened.slice().as_f32().unwrap();
        let mut scalar = vec![0.0f32; out_dim];
        ScalarF32.project_rows(
            widened.slice().rows(out_dim, in_dim).unwrap(),
            &x,
            &mut scalar,
        );
        let d_vs_b = worst_normalised(&y_direct, &y_widened, image, &x);
        let d_vs_s = worst_normalised(&y_direct, &scalar, image, &x);
        let b_vs_s = worst_normalised(&y_widened, &scalar, image, &x);
        println!(
            "  layer-2 real  {:<5} {:<36} [{out_dim},{in_dim}]  normalised: direct-vs-BLAS \
             {d_vs_b:.2e}  direct-vs-scalar {d_vs_s:.2e}  BLAS-vs-scalar {b_vs_s:.2e}  \
             (elementwise direct-vs-BLAS {:.2e})  resident {} B direct / {} B widened  \
             load {:.2}s / {:.2}s  project(warm) {:.2} ms / {:.2} ms",
            codec.name,
            op.tensor,
            worst_relative(&y_direct, &y_widened),
            direct.resident_bytes(),
            widened.resident_bytes(),
            load_direct.as_secs_f64(),
            load_widened.as_secs_f64(),
            t_direct.as_secs_f64() * 1e3,
            t_widened.as_secs_f64() * 1e3,
        );
        assert!(
            d_vs_b < ACCUMULATION_ORDER_NORMALISED,
            "{} {}: direct and the BLAS authority disagree by {d_vs_b:e} of the row scale",
            codec.name,
            op.tensor
        );
        assert!(
            d_vs_s < ACCUMULATION_ORDER_NORMALISED,
            "{} {}: direct and the scalar transcription disagree by {d_vs_s:e} of the row scale",
            codec.name,
            op.tensor
        );
        if codec == kquant::Q8_0 {
            assert_eq!(
                y_direct, scalar,
                "Q8_0 direct must be bit-for-bit with the scalar decode-then-multiply on the \
                 real anchor"
            );
        }
    }
}
