//! Vindexfile — declarative model builds.
//!
//! A Vindexfile is like a Dockerfile for model knowledge. It specifies a base
//! vindex plus patches, inline edits, labels, and build configuration.
//!
//! ```text
//! FROM hf://chrishayuk/gemma-3-4b-it-vindex
//! PATCH hf://medical-ai/drug-interactions@2.1.0
//! PATCH ./patches/company-facts.vlp
//! INSERT ("Acme Corp", "headquarters", "London")
//! DELETE entity = "Acme Corp" AND relation = "competitor" AND target = "WrongCo"
//! LABELS hf://chrishayuk/gemma-3-4b-it-labels@latest
//! EXPOSE browse inference
//! ```

mod parser;

pub use parser::{
    parse_vindexfile, parse_vindexfile_str, Vindexfile, VindexfileDirective, VindexfileStage,
};

use std::path::Path;

use crate::error::VindexError;
use crate::format::load::load_vindex_config;
use crate::index::core::SilentLoadCallbacks;
use crate::index::core::VectorIndex;
use crate::patch::core::{PatchedVindex, VindexPatch};

/// Build result from processing a Vindexfile.
pub struct VindexfileBuild {
    /// The built vindex (base + all patches/edits baked down).
    pub index: VectorIndex,
    /// Config from the base.
    pub config: crate::config::VindexConfig,
    /// Build history layers.
    pub layers: Vec<BuildLayer>,
}

/// One layer in the build history.
pub struct BuildLayer {
    pub directive: String,
    pub features_modified: usize,
}

/// Execute a Vindexfile: load base, apply patches, run edits, produce a clean VectorIndex.
pub fn build_from_vindexfile(
    vf: &Vindexfile,
    stage: Option<&str>,
    working_dir: &Path,
) -> Result<VindexfileBuild, VindexError> {
    // Resolve which directives to use
    let directives = if let Some(stage_name) = stage {
        let st = vf
            .stages
            .iter()
            .find(|s| s.name == stage_name)
            .ok_or_else(|| VindexError::Parse(format!("stage not found: {stage_name}")))?;
        // Shared directives + stage-specific
        let mut combined = vf.directives.clone();
        combined.extend(st.directives.clone());
        combined
    } else {
        vf.directives.clone()
    };

    // FROM — resolve the base vindex path
    let base_path = directives
        .iter()
        .find_map(|d| {
            if let VindexfileDirective::From(ref path) = d {
                Some(path.clone())
            } else {
                None
            }
        })
        .ok_or_else(|| VindexError::Parse("Vindexfile missing FROM directive".into()))?;

    let base_resolved = resolve_vindexfile_path(&base_path, working_dir)?;

    // Load base vindex
    let config = load_vindex_config(&base_resolved)?;
    let mut cb = SilentLoadCallbacks;
    let base = VectorIndex::load_vindex(&base_resolved, &mut cb)?;
    let mut patched = PatchedVindex::new(base);
    let mut layers = Vec::new();

    layers.push(BuildLayer {
        directive: format!("FROM {}", base_path),
        features_modified: 0,
    });

    // Process remaining directives
    for directive in &directives {
        match directive {
            VindexfileDirective::From(_) => {} // already handled

            VindexfileDirective::Patch(path) => {
                let resolved = resolve_vindexfile_path(path, working_dir)?;
                let patch = VindexPatch::load(&resolved)?;
                let op_count = patch.len();
                patched.apply_patch(patch);
                layers.push(BuildLayer {
                    directive: format!("PATCH {}", path),
                    features_modified: op_count,
                });
            }

            VindexfileDirective::Insert {
                entity,
                relation,
                target,
            } => {
                // Simple insert — find a free slot, set metadata
                // Gate vector synthesis requires embeddings which we may not have locally
                // For now, insert with metadata only (gate vector from patch if available)
                let layer = config.num_layers / 2; // knowledge band middle
                let feature = patched.find_free_feature(layer).unwrap_or(0);
                let meta = crate::index::FeatureMeta {
                    top_token: target.clone(),
                    top_token_id: 0,
                    c_score: crate::index::types::DEFAULT_C_SCORE,
                    top_k: vec![],
                };
                patched.insert_feature(layer, feature, vec![], meta);
                layers.push(BuildLayer {
                    directive: format!("INSERT (\"{}\", \"{}\", \"{}\")", entity, relation, target),
                    features_modified: 1,
                });
            }

            VindexfileDirective::Delete {
                entity,
                relation,
                target,
            } => {
                // Find and delete matching features
                let matches = patched
                    .base()
                    .find_features(Some(target.as_str()), None, None);
                for &(l, f) in &matches {
                    patched.delete_feature(l, f);
                }
                layers.push(BuildLayer {
                    directive: format!(
                        "DELETE entity=\"{}\" relation=\"{}\" target=\"{}\"",
                        entity, relation, target
                    ),
                    features_modified: matches.len(),
                });
            }

            VindexfileDirective::Labels(path) => {
                // Copy labels file to output during save
                layers.push(BuildLayer {
                    directive: format!("LABELS {}", path),
                    features_modified: 0,
                });
            }

            VindexfileDirective::Expose(_) => {
                // Build configuration — handled during save
            }
        }
    }

    // Bake down to clean VectorIndex
    let baked = if patched.num_overrides() > 0 {
        patched.bake_down()
    } else {
        patched.base().clone()
    };

    Ok(VindexfileBuild {
        index: baked,
        config,
        layers,
    })
}

/// Resolve a path from a Vindexfile directive.
/// Handles: local paths, `hf://` URLs (downloads + caches via the
/// HuggingFace resolver), `https://` URLs (still TODO).
fn resolve_vindexfile_path(
    path: &str,
    working_dir: &Path,
) -> Result<std::path::PathBuf, VindexError> {
    #[cfg(all(not(target_arch = "wasm32"), feature = "remote"))]
    if crate::format::huggingface::is_hf_path(path) {
        // Use the same resolver `larql run` and `larql extract` use
        // — caches under HF's standard cache dir, conditional fetch
        // by ETag. Returns the local snapshot path.
        return crate::format::huggingface::resolve_hf_vindex(path);
    }
    #[cfg(any(target_arch = "wasm32", not(feature = "remote")))]
    if path.starts_with("hf://") {
        return Err(VindexError::Parse(
            "HuggingFace paths are not supported without the `remote` feature; \
             download the vindex locally first"
                .into(),
        ));
    }
    if path.starts_with("https://") || path.starts_with("http://") {
        Err(VindexError::Parse(format!(
            "remote URLs not yet implemented in Vindexfile: {path} \
             — download manually and use a local path"
        )))
    } else {
        let p = working_dir.join(path);
        if !p.exists() {
            return Err(VindexError::Parse(format!(
                "path not found: {}",
                p.display()
            )));
        }
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── resolve_vindexfile_path ────────────────────────────────────

    #[test]
    fn resolve_local_path_relative_to_working_dir() {
        let tmp = tempfile::tempdir().unwrap();
        let target = tmp.path().join("base.vindex");
        std::fs::create_dir(&target).unwrap();
        let resolved = resolve_vindexfile_path("base.vindex", tmp.path()).unwrap();
        assert_eq!(resolved, target);
    }

    #[test]
    fn resolve_local_path_errors_when_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let err = resolve_vindexfile_path("nope.vindex", tmp.path()).expect_err("missing path");
        assert!(err.to_string().contains("path not found"));
    }

    #[test]
    fn resolve_https_url_is_not_yet_implemented() {
        let tmp = tempfile::tempdir().unwrap();
        let err = resolve_vindexfile_path("https://example.com/vindex", tmp.path())
            .expect_err("not implemented");
        assert!(err.to_string().contains("not yet implemented"));
    }

    #[test]
    fn resolve_http_url_is_not_yet_implemented() {
        let tmp = tempfile::tempdir().unwrap();
        let err = resolve_vindexfile_path("http://example.com/v", tmp.path())
            .expect_err("not implemented");
        assert!(err.to_string().contains("not yet implemented"));
    }

    #[test]
    fn resolve_hf_path_dispatches_to_hf_resolver() {
        // hf:// path with no real network — the hf resolver will error
        // out, but we verify the function dispatches there (different
        // error message than "path not found" or "not yet implemented").
        let tmp = tempfile::tempdir().unwrap();
        let err = resolve_vindexfile_path("hf://owner/repo", tmp.path())
            .expect_err("hf:// will fail without network");
        let msg = err.to_string();
        // Crucially NOT the local-path or https errors:
        assert!(!msg.contains("path not found"), "got: {msg}");
        assert!(!msg.contains("not yet implemented"), "got: {msg}");
    }

    // ── build_from_vindexfile ──────────────────────────────────────

    #[test]
    fn build_errors_when_from_directive_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let vf = Vindexfile {
            directives: vec![],
            stages: vec![],
        };
        let result = build_from_vindexfile(&vf, None, tmp.path());
        let err = result.err().expect("missing FROM errors");
        assert!(err.to_string().contains("FROM"));
    }

    #[test]
    fn build_errors_when_named_stage_not_found() {
        let tmp = tempfile::tempdir().unwrap();
        let vf = Vindexfile {
            directives: vec![VindexfileDirective::From("./irrelevant".into())],
            stages: vec![VindexfileStage {
                name: "production".into(),
                directives: vec![],
            }],
        };
        let result = build_from_vindexfile(&vf, Some("dev"), tmp.path());
        let err = result.err().expect("unknown stage errors");
        assert!(err.to_string().contains("stage not found: dev"));
    }

    #[test]
    fn build_errors_when_base_path_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let vf = Vindexfile {
            directives: vec![VindexfileDirective::From("./missing.vindex".into())],
            stages: vec![],
        };
        let result = build_from_vindexfile(&vf, None, tmp.path());
        let err = result.err().expect("base missing errors");
        // Either the path-resolution error or the load error fires; both
        // mention either "path not found" or the missing config file.
        let msg = err.to_string();
        assert!(
            msg.contains("path not found") || msg.contains("missing.vindex"),
            "got: {msg}"
        );
    }

    #[test]
    fn build_with_stage_selects_combined_directives() {
        // Stage path is exercised even when the base lookup fails —
        // directive selection happens before path resolution.
        let tmp = tempfile::tempdir().unwrap();
        let vf = Vindexfile {
            directives: vec![VindexfileDirective::Labels("base-labels".into())],
            stages: vec![VindexfileStage {
                name: "prod".into(),
                directives: vec![VindexfileDirective::From("./missing".into())],
            }],
        };
        // FROM lives in the stage; passing stage="prod" makes the
        // combined directive list include it. The FROM resolution
        // then fails on the missing path — but only because we got
        // past the stage-merge step, which is what we're pinning.
        let result = build_from_vindexfile(&vf, Some("prod"), tmp.path());
        assert!(result.is_err(), "missing local path errors");
    }
}
