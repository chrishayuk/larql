//! Architecture registry — declarative metadata for every `model_type`
//! [`super::detect_from_json`] recognises.
//!
//! This is descriptive data alongside `detect_from_json`'s dispatch,
//! not a replacement for it — rewriting that `match` to be table-driven
//! would be a much bigger, riskier change for comparatively little
//! extra safety. Instead, [`tests::registry_entries_are_honored_by_detect_from_json`]
//! exercises every pattern this module declares against the real
//! dispatch, so a registry entry that stops matching (e.g. someone
//! reorders or narrows a `match` arm) fails CI. That test does **not**
//! catch the reverse — a new `match` arm added without a matching
//! registry entry — so a registry entry is still something to remember
//! when adding an architecture, the same as adding the `mod` in
//! `architectures/mod.rs`.
//!
//! The consumer this exists for: Vindex Factory's PR-time architecture
//! gate (docs/vindex-factory.md §15.2) — "does the pinned larql release
//! understand this `model_type`, and if so what does it support" —
//! answerable without loading a model.
//!
//! One file per concept: [`pattern`] (how a string is matched),
//! [`attention`] (attention-family vocabulary), [`entry`] (one row's
//! shape), [`table`] (the data itself).

mod attention;
mod entry;
mod pattern;
mod table;

pub use attention::AttentionKind;
pub use entry::{ArchitectureEntry, ComponentRole};
pub use pattern::ModelTypeMatch;
pub use table::{ARCHITECTURE_REGISTRY, LLAMA_FAMILY};

/// Look up the registry entry for a `model_type` string, using the same
/// first-match-wins order `detect_from_json` uses. `None` means
/// `detect_from_json` would fall back to `GenericArch` — i.e. this
/// `model_type` isn't supported yet.
pub fn find_architecture(model_type: &str) -> Option<&'static ArchitectureEntry> {
    ARCHITECTURE_REGISTRY.iter().find(|e| e.matches(model_type))
}

#[cfg(test)]
mod tests;
