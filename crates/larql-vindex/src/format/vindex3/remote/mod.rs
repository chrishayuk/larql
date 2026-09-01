//! **A VINDEX3 container whose authority is remote and whose bytes are
//! hydrated selectively.**
//!
//! Rung 1 streamed a canonical *checkpoint* into a container. This
//! streams a *container*, and the difference is what gets to be partial:
//!
//! ```text
//! rung 1   HF BF16 → range read → encode → complete local VINDEX3
//! rung 2a  remote VINDEX3 → hydrate the execution set → run
//! ```
//!
//! # Two phases, because planning reads headers
//!
//! [`plan_component_ops`](super::opplan::plan_component_ops) resolves each
//! object's tensor table from its segment header, so the execution set
//! cannot be known until every header is in hand. Hydration is therefore
//! the same shape as rung 1, one level up:
//!
//! ```text
//! index.json + system_graph.json      tiny
//!            ↓
//! every object's segment HEADER       tiny — a tensor table, not payload
//!            ↓
//! inspect → plan → required_objects   no payload read
//!            ↓
//! payload of the required objects     the only large transfer
//! ```
//!
//! # Headers and payloads live in different directories, deliberately
//!
//! A header-only segment stub is a file that *exists* — and
//! [`OperandStore`](super::opplan::exec::operands::OperandStore) treats an
//! existing segment as resident. Staging stubs beside the hydrated
//! payloads would make every object look resident while most held no
//! bytes, which is precisely the silent-wrong-answer class this whole
//! line exists to avoid. So the planning view and the execution view are
//! separate roots, and only the execution root is ever opened for
//! operands.
//!
//! # Deny by default, then seal
//!
//! Nothing may be fetched until it has been explicitly allowed, and after
//! [`RemoteContainer::seal`] nothing may be fetched at all. The seal is a
//! hard error at the point of violation rather than a counter checked
//! afterwards: a counter reading zero cannot distinguish "nothing was
//! fetched" from "nothing was asked", and the invariant being defended is
//! that `PREPARE` and `RUN` have no remote dependency whatsoever.
//!
//! ```text
//! DESCRIBE   authority complete
//! PLAN       execution known
//! HYDRATE    residency may change      ← reads permitted
//! SEAL       residency mutation closes
//! PREPARE    required bytes must already be here
//! RUN        no remote dependency
//! ```
//!
//! A future faulting contract must opt out of this lifecycle explicitly,
//! not weaken it.

mod hydrate;

pub use hydrate::{HydrationReport, NetworkPhase, RemoteContainer};

#[cfg(test)]
mod tests;
