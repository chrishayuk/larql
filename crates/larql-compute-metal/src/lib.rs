//! # larql-compute-metal
//!
//! Metal GPU backend for [`larql_compute`].  Compiles to an empty crate
//! on non-macOS hosts — the entire implementation lives under
//! `#[cfg(target_os = "macos")]`.  Apple-silicon binaries depend on
//! both `larql-compute` (for the [`ComputeBackend`] trait and the CPU
//! fallback) and this crate (for the [`MetalBackend`] impl).
//!
//! ## Crate layout
//!
//! - [`backend`] — [`MetalBackend`] struct, construction,
//!   calibration, and the per-thread accessors (KV-cache, PLE input
//!   table).
//! - [`options`] — [`BackendOptions`] / [`DecodeFlags`] (env-derived
//!   feature toggles snapshotted at construction).
//! - [`kernels`] — per-domain pipeline registries plus the common
//!   [`KernelHandle`] + [`TiledKernel`] pattern.
//! - [`buffers`] — GPU buffer cache, scratch pool, read helpers.
//! - `shaders` — per-shader MSL sources + dispatch geometry.
//! - `ops` — per-operation dispatch wrappers (`q4_matvec`,
//!   `q4_vecmat`, …).
//! - `stages` — per-pipeline-stage encoders (`encode_qkv`,
//!   `encode_attn`, `encode_ffn`, …).
//! - `decode` — autoregressive decode loop assembly.
//! - `diag` — kernel-bench / decode-stage timing diagnostics.
//! - `trait_impl` — `ComputeBackend` sub-trait impls for
//!   `MetalBackend`.
//!
//! ## ABI
//!
//! Constructors return `Option<MetalBackend>` — `None` when the host
//! has no Metal device.  Callers compose fallback chains explicitly:
//!
//! ```rust,no_run
//! # #[cfg(target_os = "macos")] {
//! use larql_compute::{cpu_backend, ComputeBackend};
//!
//! let backend: Box<dyn ComputeBackend> =
//!     larql_compute_metal::metal_backend()
//!         .map(|m| Box::new(m) as Box<dyn ComputeBackend>)
//!         .unwrap_or_else(|| cpu_backend());
//! # }
//! ```

#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

// ───── Module tree (macOS only) ─────
//
// The entire implementation is gated on `target_os = "macos"` so the
// crate compiles to an empty lib on Linux / Windows.  Each `mod` is
// individually cfg-gated rather than nested inside a single `mod
// platform` block so error messages point at the actual file paths.

#[cfg(target_os = "macos")]
pub mod backend;
#[cfg(target_os = "macos")]
pub mod buffers;
#[cfg(target_os = "macos")]
pub mod calibration;
#[cfg(target_os = "macos")]
pub mod decode;
#[cfg(target_os = "macos")]
pub mod diag;
#[cfg(target_os = "macos")]
pub mod kernels;
#[cfg(target_os = "macos")]
pub mod ops;
#[cfg(target_os = "macos")]
pub mod options;
#[cfg(target_os = "macos")]
pub mod shaders;
#[cfg(target_os = "macos")]
pub mod stages;
#[cfg(target_os = "macos")]
pub mod trait_impl;

#[cfg(target_os = "macos")]
mod decode_hybrid;
#[cfg(target_os = "macos")]
mod direct_ops;
#[cfg(target_os = "macos")]
mod f32_ops;
#[cfg(target_os = "macos")]
mod moe_dispatch;
#[cfg(target_os = "macos")]
mod pipeline;

// ───── Curated public surface ─────

#[cfg(target_os = "macos")]
pub use backend::{MetalBackend, PleInputBuffer};
#[cfg(target_os = "macos")]
pub use buffers::{read_buffer_f32, try_read_buffer_f32, BufferCache, ScratchGuard};
#[cfg(target_os = "macos")]
pub use decode::profile::{take_last_split_timings, ProfileTimings};
#[cfg(target_os = "macos")]
pub use kernels::{AttentionKernels, FfnKernels, KernelHandle, NormKernels, QuantKernels};
#[cfg(target_os = "macos")]
pub use moe_dispatch::MoeScratch;
#[cfg(target_os = "macos")]
pub use options::{BackendOptions, DecodeFlags};

/// Re-export of the `metal-rs` `Buffer` type so downstream crates
/// (`larql-server`'s expert-route cache, primarily) can hold cached
/// `(gate_up, down)` Metal buffer pairs without taking a direct
/// dependency on the `metal` crate.
#[cfg(target_os = "macos")]
pub use ::metal::Buffer as MetalBuffer;

/// Build a Metal backend with default options derived from the
/// process environment.  Returns `None` when no Metal device is
/// available (no Apple GPU, or the requested kernel pipelines failed
/// to compile).  Historical env-driven defaults
/// (`LARQL_Q4K_MATVEC_8SG`, `LARQL_Q6K_8SG`, `LARQL_FUSED_*`, …)
/// continue to work through [`BackendOptions::from_env`].
///
/// Sugar for `MetalBackend::new()`.  Provided so caller code reads
/// uniformly across backends — a future `larql-compute-vulkan` exposes
/// `vulkan_backend()` with the same shape.
#[cfg(target_os = "macos")]
pub fn metal_backend() -> Option<MetalBackend> {
    MetalBackend::new()
}

/// Build a Metal backend with explicit options.  Useful for embedding
/// LARQL in a host that owns its own configuration surface, or for
/// tests that want a reproducible backend independent of process env.
/// Returns `None` when no Metal device is available.
#[cfg(target_os = "macos")]
pub fn metal_backend_with_options(options: BackendOptions) -> Option<MetalBackend> {
    MetalBackend::with_options(options)
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    #[test]
    fn metal_backend_returns_some_on_apple_silicon_or_none_off_host() {
        let _ = metal_backend();
    }

    #[test]
    fn metal_backend_with_options_threads_through_to_backend_constructor() {
        let _ = metal_backend_with_options(BackendOptions::default());
    }
}
