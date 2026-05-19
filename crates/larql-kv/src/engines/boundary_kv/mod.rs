//! BoundaryKvEngine — `Standard` in-session, `larql-boundary` frames at chunk boundaries.
//!
//! Specification: [`crates/larql-inference/docs/specs/boundary-kv-engine.md`].
//!
//! The engine behaves as `StandardEngine` for prefill and decode; the additive
//! capability is per-chunk emission of [`larql_boundary::BoundaryFrame`] objects
//! into a [`BoundaryArchive`]. These frames are a transport / save-restore
//! format — they are not consulted during in-session decode, and the in-session
//! correctness contract is unchanged from `Standard` (bit-identical at the same
//! quantisation tier).
//!
//! See the spec for the contract and `BOUNDARY_REF_PROTOCOL.md` for the wire
//! format.

pub mod archive;
pub mod engine;
pub mod identity;

pub use archive::{ArchiveError, BoundaryArchive, InMemoryArchive};
pub use engine::{BoundaryKvEngine, BoundaryKvEngineConfig};
pub use identity::BoundaryModelIdentity;
