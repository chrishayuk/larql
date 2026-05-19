//! Shared helpers for the per-kernel test files in this directory.
//!
//! Each top-level `.rs` file under `tests/` is its own test binary in
//! Cargo's model, so they can't share state at the module level. The
//! standard idiom is `#[path = "common/mod.rs"] mod common;` in each
//! test file, which inlines this module's contents into that binary.
//! Helpers are `#[allow(dead_code)]` because no single binary uses
//! every utility.

#![allow(dead_code)]

/// Build a `MetalBackend`. Panics with a clear message if Metal isn't
/// available — these tests are gated on `--features metal`, but the
/// host still has to expose a Metal device.
pub fn get_metal() -> larql_compute_metal::MetalBackend {
    larql_compute_metal::MetalBackend::new().expect(
        "Metal device required for these tests (rerun with --features metal on Apple Silicon)",
    )
}

/// Largest absolute element-wise diff between two equal-length slices.
/// The fold-style implementation matches the existing
/// `test_metal_shaders.rs` helper so error messages stay consistent.
pub fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Cosine similarity in `f64` accumulation. Returns `0.0` when either
/// vector is all-zero, matching the convention used elsewhere in the
/// project's diff tooling.
pub fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f64;
    let mut an = 0.0f64;
    let mut bn = 0.0f64;
    for i in 0..a.len() {
        let x = a[i] as f64;
        let y = b[i] as f64;
        dot += x * y;
        an += x * x;
        bn += y * y;
    }
    if an > 0.0 && bn > 0.0 {
        (dot / (an.sqrt() * bn.sqrt())) as f32
    } else {
        0.0
    }
}
