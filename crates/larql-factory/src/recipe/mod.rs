//! Recipe types — the Rust model of `docs/vindex-factory.md` §4's YAML
//! schema (`apiVersion: vindex.chukai.io/v1`, `kind: VindexBuild`).
//!
//! Field shape mirrors the spec's example recipe exactly; this is the
//! v0.1 schema, not the K3-scale evolution in §15.3/§15.4 (per-output
//! `destination`/`carve`, pinned `routing_stats`) — those land when a
//! recipe actually needs them.
//!
//! One file per nested group (`source`, `extractor`, `output`,
//! `verify`, `publish`, `budget`), matching the YAML's own nesting so a
//! new field lands in one obvious place.

mod budget;
mod extractor;
mod output;
mod publish;
mod source;
mod verify;

pub use budget::{Budget, BudgetRequires};
pub use extractor::Extractor;
pub use output::OutputSpec;
pub use publish::{HubPublish, MirrorPublish, Publish};
pub use source::Source;
pub use verify::{LogitMatch, Reconstruction, Verify};

use serde::{Deserialize, Serialize};

/// `apiVersion` this crate understands. A recipe declaring anything else
/// fails validation — see [`crate::RecipeError::UnsupportedApiVersion`].
pub const API_VERSION: &str = "vindex.chukai.io/v1";

/// `kind` this crate understands.
pub const KIND: &str = "VindexBuild";

/// Root of a recipe file (one per published vindex family).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Recipe {
    /// Schema version tag. Must equal [`API_VERSION`].
    #[serde(rename = "apiVersion")]
    pub api_version: String,

    /// Document kind. Must equal [`KIND`].
    pub kind: String,

    /// Identity — excluded from `build_id` (§5): renaming a recipe
    /// doesn't change the bytes it produces.
    pub metadata: Metadata,

    /// The build itself.
    pub spec: Spec,
}

/// Identity fields. Not part of `build_id`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Metadata {
    /// Short slug, e.g. `gemma-3-4b-it`. Used to derive Hub repo names
    /// via `publish.hub.repo_template`.
    pub name: String,

    /// Human-readable description.
    #[serde(default)]
    pub description: Option<String>,
}

/// The build specification. `source` + `extractor` + `outputs` determine
/// the produced bytes and feed `build_id`; `verify` + `publish` +
/// `budget` don't (§5).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Spec {
    /// Pinned upstream.
    pub source: Source,
    /// Pinned extractor invocation.
    pub extractor: Extractor,
    /// What to slice and where it goes.
    pub outputs: Vec<OutputSpec>,
    /// Verification thresholds. Required — see §4.1.
    pub verify: Verify,
    /// Publication targets.
    pub publish: Publish,
    /// Executor selection and cost/resource ceilings.
    pub budget: Budget,
}

impl Recipe {
    /// Parse a recipe from YAML source.
    pub fn from_yaml(src: &str) -> Result<Self, serde_saphyr::Error> {
        serde_saphyr::from_str(src)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_support::SAMPLE_RECIPE_YAML as SAMPLE;

    #[test]
    fn parses_the_sample_recipe() {
        let recipe = Recipe::from_yaml(SAMPLE).unwrap();
        assert_eq!(recipe.api_version, API_VERSION);
        assert_eq!(recipe.kind, KIND);
        assert_eq!(recipe.metadata.name, "gemma-3-4b-it");
        assert_eq!(recipe.spec.outputs.len(), 6);
    }

    #[test]
    fn rejects_malformed_yaml() {
        assert!(Recipe::from_yaml("not: [valid, yaml: shape").is_err());
    }
}
