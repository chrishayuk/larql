//! The Q4_0 CPU kernels refuse operands whose lengths do not match their
//! dimensions.
//!
//! `q4_vecmat::dispatch` and `q4_matvec::dispatch_q8` are safe functions
//! that hand pointers and dimensions to a C kernel. The vecmat had no
//! length check at all, and the matvec's were `debug_assert`s, so in a
//! release build a slice shorter than the dimensions claimed was read past
//! its end. The checks now run in every build.
//!
//! The "too long" cases are the ones that are safe to run against the old
//! code (nothing is read out of bounds), so they are the reproducers; the
//! "too short" cases are what the checks exist to stop.

use larql_compute::cpu::ops::q4_common::{quantize_q4_0, quantize_to_q8};
use larql_compute::cpu::ops::{q4_matvec, q4_vecmat};

const HIDDEN: usize = 64;
const ROWS: usize = 4;
/// Bytes of one row of `HIDDEN` values: 18 per 32-element block.
const ROW_BYTES: usize = HIDDEN / 32 * 18;

fn matrix() -> Vec<u8> {
    let values: Vec<f32> = (0..ROWS * HIDDEN)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let q4 = quantize_q4_0(&values);
    assert_eq!(q4.len(), ROWS * ROW_BYTES, "the fixture is a whole matrix");
    q4
}

fn activation() -> Vec<f32> {
    vec![1.0; ROWS]
}

/// The panic message of `f`, or `None` if it returned.
fn panic_message(f: impl FnOnce() + std::panic::UnwindSafe) -> Option<String> {
    let payload = std::panic::catch_unwind(f).err()?;
    Some(
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default(),
    )
}

// ── vecmat ──────────────────────────────────────────────────────────────

#[test]
fn vecmat_with_matching_lengths_computes() {
    let out = q4_vecmat::dispatch(&activation(), &matrix(), ROWS, HIDDEN);

    assert_eq!(out.len(), HIDDEN);
    assert!(out.iter().any(|v| *v != 0.0));
}

#[test]
fn vecmat_refuses_an_activation_longer_than_the_row_count() {
    let mut act = activation();
    act.push(1.0);

    let message = panic_message(move || {
        q4_vecmat::dispatch(&act, &matrix(), ROWS, HIDDEN);
    })
    .expect("a mismatched activation must be refused");

    assert!(message.contains("activation"), "{message}");
}

#[test]
fn vecmat_refuses_matrix_bytes_that_are_not_the_stated_shape() {
    let mut q4 = matrix();
    q4.extend_from_slice(&[0u8; ROW_BYTES]);

    let message = panic_message(move || {
        q4_vecmat::dispatch(&activation(), &q4, ROWS, HIDDEN);
    })
    .expect("a mismatched matrix must be refused");

    assert!(message.contains("q4_data"), "{message}");
}

#[test]
fn vecmat_refuses_a_hidden_size_that_is_not_whole_blocks() {
    // 48 is not a multiple of 32. The matrix is longer than the kernel
    // would read for it, so the old code read nothing out of bounds.
    const RAGGED_HIDDEN: usize = 48;

    let message = panic_message(|| {
        q4_vecmat::dispatch(&activation(), &matrix(), ROWS, RAGGED_HIDDEN);
    })
    .expect("a ragged hidden size must be refused");

    assert!(message.contains("multiple of 32"), "{message}");
}

#[test]
fn vecmat_refuses_operands_shorter_than_the_stated_shape() {
    let short_activation = panic_message(|| {
        q4_vecmat::dispatch(&activation()[..ROWS - 1], &matrix(), ROWS, HIDDEN);
    });
    let short_matrix = panic_message(|| {
        let q4 = matrix();
        q4_vecmat::dispatch(&activation(), &q4[..q4.len() - ROW_BYTES], ROWS, HIDDEN);
    });

    assert!(short_activation.is_some_and(|m| m.contains("activation")));
    assert!(short_matrix.is_some_and(|m| m.contains("q4_data")));
}

// ── matvec ──────────────────────────────────────────────────────────────

#[test]
fn matvec_with_matching_lengths_computes() {
    let x = vec![0.5f32; HIDDEN];

    let scores = q4_matvec::dispatch(&matrix(), &x, ROWS, HIDDEN);

    assert_eq!(scores.len(), ROWS);
}

#[test]
fn matvec_refuses_operands_that_are_not_the_stated_shape() {
    let (q8_x, q8_scales) = quantize_to_q8(&vec![0.5f32; HIDDEN]);
    let q4 = matrix();

    let short_matrix = {
        let (q4, q8_x, q8_scales) = (q4.clone(), q8_x.clone(), q8_scales.clone());
        panic_message(move || {
            q4_matvec::dispatch_q8(&q4[..q4.len() - ROW_BYTES], &q8_x, &q8_scales, ROWS, HIDDEN);
        })
    };
    let short_input = {
        let (q4, q8_x, q8_scales) = (q4.clone(), q8_x.clone(), q8_scales.clone());
        panic_message(move || {
            q4_matvec::dispatch_q8(&q4, &q8_x[..HIDDEN - 1], &q8_scales, ROWS, HIDDEN);
        })
    };
    let short_scales = panic_message(move || {
        q4_matvec::dispatch_q8(&q4, &q8_x, &q8_scales[..1], ROWS, HIDDEN);
    });

    assert!(short_matrix.is_some_and(|m| m.contains("q4_data")));
    assert!(short_input.is_some_and(|m| m.contains("q8_x")));
    assert!(short_scales.is_some_and(|m| m.contains("q8_scales")));
}
