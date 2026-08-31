//! **The CPU's NVFP4 projection arm is the same program as the
//! decoder** — asserted here rather than inherited from a comment.
//!
//! `FusedNvfp4` exists because a representation with no execution path
//! cannot be measured: before it, every backend that requested NVFP4 was
//! a device backend, so a model whose token mixer has no device kernel
//! could have a pack compiled, verified, and then unrunnable anywhere.
//! Its own doc comment claims the fused kernel is bit-exact against
//! decode-then-multiply. That claim is the thing worth testing, and it
//! is exactly the kind that drifts silently: the arm is reachable only
//! from a compiled NVFP4 pack, so a build where nothing exercises it
//! keeps compiling, keeps passing, and keeps being wrong.
//!
//! The oracle here is deliberately NOT the fused kernel's own internals:
//! it is `larql_models::quant::nvfp4::dequantize_into` — the decoder the
//! format's fidelity claims are written against — followed by an
//! ordinary dot product.

use larql_models::quant::nvfp4;

use crate::format::vindex3::opplan::exec::cpu::kernels::FusedNvfp4;
use crate::format::vindex3::opplan::exec::cpu::projector::{DenseProjector, WeightRows};

const ROWS: usize = 5;
/// Two 16-element groups per row, so the per-group scales do real work
/// and a kernel that read one scale per ROW would be caught.
const K: usize = 32;

/// Values whose magnitude varies per group — a flat matrix would let a
/// wrong scale index pass unnoticed.
fn weights() -> Vec<f32> {
    (0..ROWS * K)
        .map(|i| {
            let group = (i % K) / 16;
            let base = ((i % 13) as f32 - 6.0) * 0.0625;
            base * if group == 0 { 1.0 } else { 8.0 }
        })
        .collect()
}

fn activation() -> Vec<f32> {
    (0..K).map(|i| ((i % 7) as f32 - 3.0) * 0.25).collect()
}

/// The arm and the decoder compute the same function.
///
/// Not bit-identity: the fused kernel decodes into registers and
/// accumulates as it goes, while the oracle materialises every weight
/// first, so the two differ by float summation order alone. The bound is
/// therefore tight relative to the magnitudes involved — tight enough
/// that a wrong scale, a swapped nibble, or a group read at the wrong
/// stride could not hide under it.
#[test]
fn the_fused_nvfp4_arm_matches_decode_then_multiply() {
    let original = weights();
    let matrix = nvfp4::quantize(&original, ROWS, K).expect("a 16-multiple width is legal");
    let x = activation();

    let mut fused = vec![0.0f32; ROWS];
    FusedNvfp4.project_rows(
        WeightRows::Nvfp4 {
            packed: &matrix.packed,
            scales: &matrix.scales,
            tensor_scale: matrix.tensor_scale,
        },
        &x,
        &mut fused,
    );

    // The oracle: decode the pack the way the format defines, then
    // multiply.
    let mut decoded = vec![0.0f32; ROWS * K];
    nvfp4::dequantize_into(&matrix, ROWS, K, &mut decoded).expect("decode");
    let oracle: Vec<f32> = decoded
        .chunks_exact(K)
        .map(|row| row.iter().zip(&x).map(|(w, v)| w * v).sum())
        .collect();

    let scale = oracle.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1.0);
    for (r, (got, want)) in fused.iter().zip(&oracle).enumerate() {
        assert!(
            (got - want).abs() <= 1e-5 * scale,
            "row {r}: fused {got} against decode-then-multiply {want}"
        );
    }

    // And the arm is doing work: an all-zero answer would satisfy a
    // tolerance test against a near-zero oracle.
    assert!(
        oracle.iter().any(|v| v.abs() > 1e-3),
        "the fixture must produce a non-trivial product"
    );
}

/// The kernel refuses a representation it cannot read, by name.
///
/// The pairing of format to kernel is made once, by
/// `PhysicalProjectionPlan`, and observed rather than re-derived — so a
/// mismatch here is that invariant broken, not a runtime condition to
/// absorb. It panics on purpose; this pins that it panics *saying which
/// kernel and which weights*.
#[test]
#[should_panic(expected = "nvfp4")]
fn the_fused_nvfp4_arm_refuses_other_representations() {
    let w = vec![1.0f32; 8];
    let x = vec![1.0f32; 4];
    let mut out = vec![0.0f32; 2];
    FusedNvfp4.project_rows(WeightRows::F32(&w), &x, &mut out);
}

/// Its threading declaration is part of the contract: the kernel needs
/// EXTERNAL row-splitting to reach its throughput, and a caller that
/// read it as self-threading would run it single-shot.
#[test]
fn the_fused_nvfp4_arm_declares_external_parallelism() {
    use crate::format::vindex3::opplan::exec::cpu::projector::CpuParallelism;
    assert_eq!(FusedNvfp4.parallelism(), CpuParallelism::ExternalPool);
}
