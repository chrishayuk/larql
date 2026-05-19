//! `MarkovResidualCodecEngine` — `MarkovResidualEngine` with a codec layer
//! on the cold residual tier.
//!
//! Specification: [`crates/larql-inference/docs/specs/markov-residual-codec-engine.md`].
//!
//! The hot tier is unchanged (`f32`). The cold tier is encoded through a
//! caller-selected [`codec::ColdResidualCodec`] — `Bf16` is the only v0.1
//! default with a robust contract per §2.1 + §4.7 of the spec.
//!
//! Per the spec, this engine is **not** bit-identical to
//! `MarkovResidualEngine`: each codec carries a per-architecture KL bound.
//! For `Bf16` the bound is the `f32` → `bf16` → `f32` roundtrip on cold
//! residuals, which is small but non-zero.

pub mod codec;
pub mod compute;
pub mod engine;
pub mod store;
pub mod walk;

pub use codec::ColdResidualCodec;
pub use engine::MarkovResidualCodecEngine;
pub use store::{EncodedColdLayer, RsStoreCodec};
