//! `BoundaryPerLayerEngine` — `MarkovResidualEngine` with a per-layer codec
//! policy on the cold tier.
//!
//! Specification: [`crates/larql-inference/docs/specs/boundary-per-layer-engine.md`].
//!
//! v0.1 ships the policy + calibration-store infrastructure but restricts
//! per-layer codec choice to [`ColdResidualCodec::Bf16`]. Future versions
//! will add `Int8Clip3Sigma` and other codecs to the per-layer mixing
//! surface once their per-layer calibration sweeps land — see the spec's
//! Phase 3 calibration plan.

pub mod calibration;
pub mod engine;
pub mod policy;
pub mod store;

pub use calibration::{
    BoundaryCalibrationRecord, BoundaryCalibrationStore, CalibrationError, InMemoryCalibrationStore,
};
pub use engine::BoundaryPerLayerEngine;
pub use policy::{BoundaryLayerPolicy, PolicyError};
pub use store::{PerLayerEncodedColdLayer, RsStorePerLayer};
