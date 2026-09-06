//! The VINDEX3 container — writing one, and opening one.
//!
//! # Why this module had to exist before "VINDEX3 works" could be said
//!
//! The VINDEX3 execution runtime ([`crate::runtime`]) was validated to
//! bit-identity against the shipped path, repeatedly and on a real 26B model.
//! But every one of those runs sourced its operands from a **VINDEX2** file:
//! `index.json.version` was 2 at every step. Nothing could write a VINDEX3
//! container, so `ContainerGeneration::V3` appeared in exactly one test,
//! constructed from a hand-written JSON string.
//!
//! That made the honest claim narrower than it sounded:
//!
//! ```text
//! proven      the VINDEX3 runtime, fed VINDEX2 operands, matches production
//! not proven  a VINDEX3 container can be written, opened, bound and executed
//! ```
//!
//! This module closes the second line for the control fixture. The middle of
//! the stack — LYRW v2 tables, the programme manifest, capability and
//! authority, the bound executor — was already built; what was missing was the
//! two ends, and an end that nothing exercises is an end nobody has debugged.
//!
//! # Layout
//!
//! ```text
//! <root>/
//! ├── index.json           schema 3 — sole root authority (§12)
//! ├── moe_manifest.json    programme description (§8)
//! └── routed/layer_000.lyrw  LYRW v2 bank (§6)
//! ```
//!
//! # What is deliberately not here
//!
//! No transcoding. A container assembler that also converted formats would be
//! the silent conversion §9.1 forbids, buried one layer below where anyone
//! would look for it. Regions are placed exactly as their producer wrote them.

pub mod artifact;
pub mod auxiliary_references;
pub mod build;
pub mod compact;
pub mod compile;
pub mod diff;
pub mod encode;
/// Miniature whole-model checkpoint writers, public so sibling crates'
/// integration tests can encode a real container without duplicating
/// the frozen fixture geometry the executor's parity gates certify.
pub mod fixtures;
pub mod fixtures_kimi;
pub mod gguf;
pub mod graph;
pub mod import;
pub mod index;
pub mod inspect;
/// The V3 query surface: browse-facing semantic roles bound to the
/// executable plan's operands (V3-LQL-3A).
pub mod knowledge;
pub mod opplan;
pub mod plan;
pub mod profile;
pub mod read;
pub mod remote;
pub mod represent;
/// Conformance fixture A, public so integration tests and future gate arms can
/// build a real container without duplicating its frozen dimensions.
pub mod test_support;
pub mod variants;
pub mod verify;
pub mod verify_system;
pub mod write;

pub use build::ContainerBuilder;
pub use import::{import_one_layer, write_segment_file, ExpertScaleStreams, MoeLayerSource};
pub use index::{Vindex3Index, PROFILE_EXACT};
pub use profile::{Profile, ProfileSelectionError, ResolvedProfile};
pub use read::Vindex3Container;
pub use variants::{RegionSetVariants, StoredVariant, VariantCatalogue, VariantDefect};
pub use verify::{validate_downloaded_container, ContainerDefect};
pub use write::{segment_path, write_container, ContainerSpec, SegmentSource, MOE_MANIFEST_JSON};
