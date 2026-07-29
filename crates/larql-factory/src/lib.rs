//! Vindex Factory driver — recipe schema, `build_id` canonicaliser, and
//! structural validator. See `docs/vindex-factory.md` §3.1: this crate
//! is the single implementation both the GitHub Action and the rig
//! worker call (as `larql recipe validate` / `larql recipe build-id`),
//! so there's nothing to keep in sync between them.
//!
//! Scope: §4's recipe schema, §5's `build_id`, the structural half of
//! §6.1's PR-check gate, §6.1 step 4's size/cost estimator, §9's card
//! generator, and §7's PREFLIGHT→RELEASE build driver (`run_build`) —
//! MIRROR and REGISTER are external-harness concerns, and VERIFY here
//! is checksum integrity only; see `build`'s module doc for why.

#![deny(missing_docs)]

mod build;
mod build_id;
mod capabilities;
mod card;
mod constants;
mod estimate;
mod hex;
mod recipe;
mod validate;

pub use build::{
    run as run_build, BuildRecord, BuildStatus, CommandOutput, CommandRunner, OutputRecord, Stage,
    SubprocessRunner,
};
pub use build_id::build_id;
pub use capabilities::{
    manifest as capabilities_manifest, ArchitectureCapability, CapabilityManifest,
};
pub use card::{
    render as render_card, revision_tag, CardInputs, LogitMatchResult, ReconstructionResult,
    SliceSummary, VerificationReport,
};
pub use estimate::{
    estimate as estimate_size, ExecutorClass, HttpError as EstimateError, ModelDims,
    OutputEstimate, SizeEstimate,
};
pub use recipe::{
    Budget, BudgetRequires, Extractor, HubPublish, LogitMatch, Metadata, MirrorPublish, OutputSpec,
    Publish, Recipe, Reconstruction, Source, Spec, Verify, API_VERSION, KIND,
};
pub use validate::{validate, RecipeError};
