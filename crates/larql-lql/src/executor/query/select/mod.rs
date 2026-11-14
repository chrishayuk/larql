//! `SELECT * FROM {EDGES, FEATURES, ENTITIES}` and `NEAREST TO`.
//!
//! Each verb lives in its own file so the per-file budget stays
//! reviewable and so per-verb tests against the synthetic vindex
//! fixture stay scoped.
//!
//! Defaults and table widths shared across verbs live as `pub(super)`
//! constants in `format.rs`, with `// why` comments explaining how
//! each was chosen.

pub(crate) mod edges;
mod entities;
mod features;
mod format;
mod nearest;
