//! Artifact resolution, re-exported from the library.
//!
//! The rules for what an artifact argument MEANS — revision pinning, the
//! tied-weight payload census, the name a container records — live in
//! `larql_vindex::format::vindex3::artifact` because TWO binaries ingest
//! models. A copy here would be a second authority on a model's identity,
//! free to drift from the one `vindex` uses.
//!
//! This module is the seam, not the implementation.

pub use larql_vindex::format::vindex3::artifact::*;
