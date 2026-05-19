//! Per-layer residual capture + comparison for backend parity testing.
//!
//! ## Why a module
//!
//! Earlier diagnostics drove backend dumps via env vars
//! (`LARQL_CPU_DUMP_LAYERS`, `LARQL_METAL_DUMP_LAYERS`,
//! `LARQL_DECODE_DUMP_LAYERS`, `LARQL_STAGE_DUMP_LAYER`, `LARQL_DUMP_L0`),
//! each writing slightly different file formats into ad-hoc temp dirs.
//! That worked for one-off bisects but couldn't be threaded into proper
//! tests without each test re-implementing the same temp-dir + file-read
//! plumbing. This module owns that boilerplate, returns typed
//! [`ResidualCapture`] structs in memory, and exposes a single comparison
//! entry point ([`compare_captures`]).
//!
//! ## Three captures, one comparison
//!
//! Each capture corresponds to a real forward path the production code
//! takes. Tests can compare any pair to assert backend parity.
//!
//! - [`ResidualCapture::cpu_prefill`] — `predict_kquant_hidden` per-layer
//!   output. Reference path.
//! - [`ResidualCapture::metal_prefill`] — `prefill_kquant` per-layer output.
//!   Should match CPU prefill bit-exactly modulo float noise.
//! - [`ResidualCapture::metal_decode`] — `prefill_kquant` followed by
//!   `decode_token`, capturing the decode call's per-layer output.
//!   Should match a CPU prefill of the same total sequence length at
//!   the new position.
//!
//! All three return `Vec<f32>` per layer (length `seq_len * hidden` for
//! prefill captures; length `hidden` for decode captures).
//!
//! ## Usage
//!
//! ```ignore
//! use larql_inference::residual_diff::{ResidualCapture, compare_captures, ParityThreshold};
//!
//! let cpu = ResidualCapture::cpu_prefill(&mut weights, &ids, &index)?;
//! let metal = ResidualCapture::metal_prefill(&mut weights, &ids, &index, &be)?;
//! let report = compare_captures(&cpu, &metal, ParityThreshold::tight());
//! report.assert_clean()?;  // panics with first-bad-layer detail
//! ```
//!
//! ## Internals
//!
//! Capture is implemented over the existing env-var-driven dump hooks
//! in `vindex/kquant_forward.rs`, `metal/ops/full_pipeline.rs`, and
//! `metal/decode/mod.rs`. We allocate a private `tempfile::TempDir`,
//! set the env vars on the current process for the duration of one
//! forward, then read the resulting `.f32` blobs back into a `Vec<f32>`
//! per layer. The TempDir guard releases the disk on drop.
//!
//! Any future direct-callback hook (avoiding the fs round-trip) can
//! replace [`capture::run_with_dump_dir`] without touching the public
//! surface.

mod capture;
mod compare;
mod stages;

pub use capture::ResidualCapture;
pub use compare::{compare_captures, LayerStat, ParityReport, ParityThreshold};
pub use stages::{compare_stages, StageCapture, StagePair, StageReport};
