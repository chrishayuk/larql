//! `MatMul::submission_clock` on a real device.
//!
//! The clock splits a device call's wait into queue latency and GPU
//! execution, so the V3 Metal baseline can say which of the two a
//! per-call cost is. It asserts no latency number, since those vary with
//! the machine. What it pins:
//!
//! | arm | claim |
//! |---|---|
//! | 1 | one gemv adds exactly one submission. The count is observed at commit, not assumed per call |
//! | 2 | the GPU span is positive (real timestamps were read, not zeros) and no larger than commit to completion, of which it is a sub-interval |
//! | 3 | the counters belong to the backend instance: a second backend's clock does not move |
//! | 4 | a multi-matrix gemv is one submission, however many matrices it carries |

#![cfg(target_os = "macos")]

use larql_compute::backend::matmul::MatMul;
use larql_compute_metal::MetalBackend;
use larql_models::quant::nvfp4::{self, NVFP4_GROUP_ELEMS};

/// Rows of the test matrix, large enough to be a real dispatch.
const ROWS: usize = 64;
/// Groups per row.
const GROUPS_PER_ROW: usize = 16;
const K: usize = NVFP4_GROUP_ELEMS * GROUPS_PER_ROW;
/// Matrices in the multi-gemv arm.
const MULTI: usize = 3;

fn matrix() -> nvfp4::Nvfp4Matrix {
    let values: Vec<f32> = (0..ROWS * K)
        .map(|i| ((i % 97) as f32 / 97.0) - 0.5)
        .collect();
    nvfp4::quantize(&values, ROWS, K).expect("quantise")
}

fn input() -> Vec<f32> {
    (0..K).map(|i| (i % 11) as f32 * 0.01).collect()
}

#[test]
fn one_gemv_is_one_timed_submission_on_its_own_backend() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let Some(other) = MetalBackend::new() else {
        return;
    };
    let m = matrix();
    let x = input();

    let before = gpu.submission_clock().expect("the Metal backend measures");
    let other_before = other
        .submission_clock()
        .expect("the Metal backend measures");
    gpu.nvfp4_gemv(&m.packed, &m.scales, m.tensor_scale, &x, ROWS, K)
        .expect("nvfp4 gemv");
    let after = gpu.submission_clock().unwrap();

    assert_eq!(after.submissions - before.submissions, 1);
    let gpu_ns = after.gpu_nanos - before.gpu_nanos;
    let done_ns = after.commit_to_done_nanos - before.commit_to_done_nanos;
    assert!(gpu_ns > 0, "no GPU span was read");
    assert!(
        gpu_ns <= done_ns,
        "GPU span {gpu_ns} ns exceeds commit to completion {done_ns} ns"
    );
    assert_eq!(other.submission_clock().unwrap(), other_before);
}

#[test]
fn a_multi_matrix_gemv_is_one_submission() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device; skipping");
        return;
    };
    let m = matrix();
    let x = input();
    let operands: Vec<_> = (0..MULTI)
        .map(|_| {
            (
                m.packed.as_slice(),
                m.scales.as_slice(),
                m.tensor_scale,
                ROWS,
                K,
            )
        })
        .collect();

    let before = gpu.submission_clock().unwrap();
    let out = gpu
        .nvfp4_gemv_multi(&operands, &x)
        .expect("nvfp4 gemv multi");
    let after = gpu.submission_clock().unwrap();

    assert_eq!(out.len(), MULTI);
    assert_eq!(after.submissions - before.submissions, 1);
}
