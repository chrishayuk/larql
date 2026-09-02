//! `larql capabilities` — the capability manifest from
//! docs/vindex-factory.md §15.2: "does the pinned `larql` release
//! understand this `model_type`, and if so what does it support",
//! answerable without loading a model.
//!
//! Built from `larql-models`' real architecture registry
//! (`larql_models::detect::ARCHITECTURE_REGISTRY`), not a hand-typed
//! list — see that module's docs for why a second, disconnected list
//! here would have been exactly the drift risk this feature exists to
//! prevent.
//!
//! Scope today: this crate only *produces* the manifest. Actually
//! gating a recipe's `source.hf_repo@revision` against it needs the
//! target's `config.json`, which means HF API access — that's
//! `larql recipe estimate`'s territory (network I/O), not this
//! structural/local command.

mod types;

pub use types::{ArchitectureCapability, CapabilityManifest, ModelTypePattern};

use larql_models::detect::ARCHITECTURE_REGISTRY;

/// Build the capability manifest for the running `larql` binary.
pub fn manifest() -> CapabilityManifest {
    CapabilityManifest {
        larql_version: env!("CARGO_PKG_VERSION").to_string(),
        architectures: ARCHITECTURE_REGISTRY
            .iter()
            .map(|entry| ArchitectureCapability {
                model_type: entry.model_type.to_string(),
                matches: entry.patterns.iter().map(Into::into).collect(),
                attention_kind: entry.attention_kind,
                quant_formats: entry.quant_formats.to_vec(),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_covers_every_registry_entry() {
        let m = manifest();
        assert_eq!(m.architectures.len(), ARCHITECTURE_REGISTRY.len());
        assert!(!m.larql_version.is_empty());
    }

    /// The regression this guards: the manifest exists to answer "does
    /// this build understand this `model_type`", and a checkpoint does
    /// not declare the registry's representative label — Gemma 3 text
    /// declares `gemma3_text`, Qwen 3 declares `qwen3`. A manifest
    /// carrying labels alone answers "no" for both, and the Factory gate
    /// built on it rejects recipes this build serves.
    #[test]
    fn manifest_carries_the_patterns_a_declared_model_type_is_matched_by() {
        let m = manifest();
        let matches = |declared: &str| {
            m.architectures.iter().any(|a| {
                a.matches.iter().any(|p| match p {
                    ModelTypePattern::Exact(s) => declared == s,
                    ModelTypePattern::Prefix(s) => declared.starts_with(s.as_str()),
                })
            })
        };
        // Spellings real checkpoints declare, none of which equal a label.
        for declared in ["gemma3_text", "gemma4_text", "qwen3", "granitemoehybrid"] {
            assert!(
                matches(declared),
                "{declared} must resolve through the manifest"
            );
            assert!(
                larql_models::detect::find_architecture(declared).is_some(),
                "{declared} must also resolve through the registry itself"
            );
        }
        // And the manifest must not start answering yes to everything.
        assert!(!matches("definitely-not-an-architecture"));
    }

    #[test]
    fn manifest_includes_a_known_architecture() {
        let m = manifest();
        assert!(m.architectures.iter().any(|a| a.model_type == "gemma3"));
    }
}
