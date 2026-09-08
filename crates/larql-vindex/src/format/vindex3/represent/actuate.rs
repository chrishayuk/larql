//! **From an experiment the optimiser selected to a request something
//! can perform.**
//!
//! Stage 4/4b closed the read path: a stored record derives what should
//! be measured next, from its own facts, with nothing injected. Stage 5a
//! made a measurement run INSTRUCTED rather than configured: every
//! control on a run became a field, resolved by name and refused when
//! unknown. Between the two there was no join, and the join was being
//! made by a human reading a digest and setting six environment
//! variables.
//!
//! ```text
//! SearchSnapshot
//!      ↓  derived, no arguments
//! next_experiment()  →  Selection / MeasurementKey
//!      ↓  THIS MODULE
//! MeasurementRequest      what, under which protocol, judged how
//!      ↓  by procedure name, never by device
//! ExperimentExecutor      + ArtifactLocator: where those things are
//!      ↓
//! representation build / bind / instrument
//! ```
//!
//! # The frozen scope
//!
//! > **It turns an experiment the optimiser has ALREADY SELECTED into a
//! > request an executor can perform. It does not choose which
//! > experiments should exist, it does not perform one, and it does not
//! > write to the scientific record.**
//!
//! and the falsifier that matters:
//!
//! > **If preparing an experiment can change what is measured — the
//! > state, the corpus, the scale, the instrument or the gate — the
//! > bridge is wrong.** The bridge translates an authorisation; it must
//! > not be able to author one.
//!
//! Every other property here is convenience. This one is why the layer
//! may exist at all: an actuation path that can quietly re-aim an
//! experiment turns the whole 1a-1c identity model into decoration,
//! because the key would stop naming what was run.
//!
//! # Provider-neutral, on purpose
//!
//! Nothing in this module names a device, a backend, a kernel, a
//! lowering target, a store id or a container path. An executor is
//! selected by the PROCEDURE the record declares, resolved by name and
//! refused when this build does not implement it — the same discipline
//! as `layout_admission`, `compiled_bytes` and `gate_by_id`.
//!
//! The sequencing reason is worth stating, because it is easy to lose:
//! the representation plane is already pluggable and the lowering plane
//! is not yet equally open. An actuation bridge that learned `Cpu`,
//! `Metal` or a kernel enum would finish the autonomous optimiser by
//! cementing exactly the lowering authority the codec work is
//! dismantling. A new codec, extent, residency policy or lowering
//! provider should enter the action space by declaring itself, and be
//! measured under the existing evidence contract, without the optimiser
//! learning its name.
//!
//! # What this module deliberately does not do
//!
//! - **It does not build the overlay.** A request says which map must be
//!   presented and which physical state it must resolve to; producing
//!   those bytes belongs to a provider. A locator with nothing built
//!   refuses by name rather than silently building one, because a build
//!   is precisely where a representation could come to differ from the
//!   one the key names.
//! - **It does not ingest.** Executing yields what was observed and what
//!   the run verified. Whether that constitutes the measurement its key
//!   names, and how it is sealed into `SearchFacts`, is stage 6 —
//!   answered carefully there rather than cheaply here by a call to
//!   `MeasurementRegistry::record`.
//! - **It does not loop.** Stage 7.
//! - **It does not batch.** `ExecutionBatch` already owns co-execution
//!   and takes keys; batch width must not become a field of a request,
//!   because that would put scheduling identity back into the
//!   measurement.
//! - **It does not decide that a failed run means a failed candidate.**
//!   `MeasurementRefusal` already separates *nothing was measured* from
//!   *the run proved nothing*, and neither is a verdict on the
//!   representation.

pub mod prepare;
pub mod request;

pub use prepare::{PreparedExperiment, Ready};
pub use request::{MeasurementRequest, RequestRefusal};

#[cfg(test)]
mod tests;
