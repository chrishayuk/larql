//! Bringing a model in: what an artifact argument means, and what
//! ingesting one costs.
//!
//! Three spellings, one result:
//!
//! ```text
//! ./checkpoint            a directory of config.json + *.safetensors
//! ./inventory.json        a saved inventory
//! hf://org/name[@rev]     a repo — headers staged, payloads left there
//! ```
//!
//! The third is the one worth explaining. Admission — inventory, plan,
//! capability closure — reads safetensors *headers* and never a payload
//! byte, so a repo can be admitted from a few MB of staged headers. Only
//! if the plan is admissible does anything ask for a tensor, and then it
//! asks for that tensor's byte range and nothing else. A 328 GB
//! checkpoint is never on this disk, in whole or in part.
//!
//! # Why this lives in the library
//!
//! Two binaries ingest models — `vindex` and `larql vindex3` — and the
//! rules for what an artifact argument MEANS are not presentation. A copy
//! in each would be two authorities on revision pinning, on the
//! tied-weight payload census, and on the name a container records, free
//! to disagree about a model's identity or to write different containers
//! from the same input.
//!
//! # Nothing here prints
//!
//! Staging is slow enough to want progress, but a library that writes to
//! stderr decides for every caller. [`ResolvedArtifact::staging`] and
//! [`IngestOutcome`] return the figures; the callers render them. So the
//! CLI keeps its voice and a future server keeps its silence.

mod ingest;
mod resolve;
mod size;
mod staging;

pub use ingest::{encode_from_specs, IngestOutcome, RemoteTransfer};
pub use resolve::{
    is_remote_spec, resolve, resolve_all, resolve_pinned_commit, resolve_pinned_commits,
    ArtifactPayloads, ResolvedArtifact, INVENTORY_EXT,
};
pub use size::size;
pub use staging::StagingReport;

#[cfg(test)]
mod tests;
