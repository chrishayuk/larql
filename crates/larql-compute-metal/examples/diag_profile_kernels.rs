//! Per-kernel Metal GPU bandwidth profiler — entry point.
//!
//! Logic lives in `src/metal/diag/kernel_profile.rs`. This is a thin
//! wrapper so the profiler can be invoked as a standalone binary.
//!
//! Usage:
//!   cargo run --release --features metal -p larql-compute --example diag_profile_kernels
//!
//! Output: GB/s per kernel in isolation AND batched (34× / cmd buffer),
//! bottleneck classification (compute-bound vs bandwidth-bound), and the
//! projected ms/tok contribution for each kernel.
//!
//! See PERFORMANCE.md for the reference numbers (2026-04-26, M3 Max).

extern crate blas_src;

#[cfg(not(target_os = "macos"))]
fn main() {
    eprintln!("This example requires macOS and --features metal");
}

#[cfg(target_os = "macos")]
fn main() {
    let _results = larql_compute_metal::diag::kernel_profile::profile_all(
        34, // n_layers
        5,  // warmup iterations
        50, // measurement iterations
    );
}
