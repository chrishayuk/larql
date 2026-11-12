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
//!
//! `DELETE` removes a fact that an earlier `INSERT` or `PATCH` in the same
//! Vindexfile installed, matched on entity, relation and target exactly. A
//! feature of the base vindex carries only its target token, so it cannot
//! be deleted by fact, and a `DELETE` that matches nothing fails the build.

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

/// INSERT placement policy: metadata-only inserts land in the middle of
/// the layer stack — the knowledge band where relational features
/// concentrate (`num_layers / INSERT_LAYER_BAND_DIVISOR`).
const INSERT_LAYER_BAND_DIVISOR: usize = 2;

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
    // The facts this build has installed and not since removed.
    let mut installed = InstalledFacts::default();

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
                let operations = patch.operations.clone();
                // Fallible apply: a corrupt embedded vector fails the
                // build instead of silently skipping the patch.
                patched.try_apply_patch(patch)?;
                installed.record_patch(&operations);
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
                // Simple insert — find a free slot, set metadata.
                // Gate vector synthesis requires embeddings which we may
                // not have locally, so this is a metadata-only insert:
                // the empty gate vec below means "no gate override"
                // (`insert_feature` stores nothing in `overrides_gate`;
                // the slot is claimed via `overrides_meta`).
                let layer = config.num_layers / INSERT_LAYER_BAND_DIVISOR;
                // No free slot is an error — the old `unwrap_or(0)`
                // silently overwrote feature 0's metadata.
                let feature = patched.find_free_feature(layer).ok_or_else(|| {
                    VindexError::Parse(format!(
                        "INSERT (\"{entity}\", \"{relation}\", \"{target}\"): \
                         no free feature slot at layer {layer}"
                    ))
                })?;
                let meta = crate::index::FeatureMeta {
                    top_token: target.clone(),
                    top_token_id: 0,
                    c_score: crate::index::types::DEFAULT_C_SCORE,
                    top_k: vec![],
                };
                patched.insert_feature(layer, feature, vec![], meta);
                installed.install(
                    Fact::new(entity, Some(relation.as_str()), target),
                    (layer, feature),
                );
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
                // A DELETE names a fact, and a fact has an identity only
                // where this build installed it: a feature record of the
                // base carries a target token and no entity or relation.
                // Matching the base on the target alone deleted every
                // feature whose top token merely contained it.
                let slots = installed.remove(&Fact::new(entity, Some(relation.as_str()), target));
                if slots.is_empty() {
                    return Err(VindexError::Parse(format!(
                        "DELETE (\"{entity}\", \"{relation}\", \"{target}\"): no INSERT or PATCH in \
                         this Vindexfile installed that fact. A feature of the base vindex has no \
                         entity or relation to match, so it cannot be deleted by fact; delete it \
                         by slot with a PATCH"
                    )));
                }
                for &(l, f) in &slots {
                    patched.delete_feature(l, f);
                }
                let matches = slots;
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

/// A fact as a directive names it. A patch insert may carry no relation.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Fact {
    entity: String,
    relation: Option<String>,
    target: String,
}

impl Fact {
    fn new(entity: &str, relation: Option<&str>, target: &str) -> Self {
        Self {
            entity: entity.to_string(),
            relation: relation.map(str::to_string),
            target: target.to_string(),
        }
    }
}

/// The facts a build has installed, each with the slot it lives in.
///
/// This is the only place a fact's entity and relation are known, so it is
/// what a `DELETE` directive is matched against.
#[derive(Debug, Default)]
struct InstalledFacts {
    facts: Vec<(Fact, (usize, usize))>,
}

impl InstalledFacts {
    /// Record `fact` at `slot`, replacing whatever the build had there.
    fn install(&mut self, fact: Fact, slot: (usize, usize)) {
        self.vacate(slot);
        self.facts.push((fact, slot));
    }

    /// Forget whatever fact the build had installed at `slot`.
    fn vacate(&mut self, slot: (usize, usize)) {
        self.facts.retain(|(_, at)| *at != slot);
    }

    /// Follow a patch's operations in order: an insert installs its fact,
    /// and a delete or an update of a slot ends the fact that was there (an
    /// update rewrites the record, which may no longer be that fact).
    fn record_patch(&mut self, operations: &[crate::patch::core::PatchOp]) {
        use crate::patch::core::PatchOp;
        for op in operations {
            match op {
                PatchOp::Insert {
                    layer,
                    feature,
                    relation,
                    entity,
                    target,
                    ..
                } => self.install(
                    Fact::new(entity, relation.as_deref(), target),
                    (*layer, *feature),
                ),
                PatchOp::Delete { layer, feature, .. } | PatchOp::Update { layer, feature, .. } => {
                    self.vacate((*layer, *feature))
                }
                PatchOp::InsertKnn { .. } | PatchOp::DeleteKnn { .. } | PatchOp::WeightDelta(_) => {
                }
            }
        }
    }

    /// Remove every installed instance of `fact` and return its slots.
    fn remove(&mut self, fact: &Fact) -> Vec<(usize, usize)> {
        let slots: Vec<(usize, usize)> = self
            .facts
            .iter()
            .filter(|(installed, _)| installed == fact)
            .map(|(_, slot)| *slot)
            .collect();
        self.facts.retain(|(installed, _)| installed != fact);
        slots
    }
}

/// Resolve a path from a Vindexfile directive.
/// Handles: local paths, `hf://` URLs (downloads + caches via the
/// HuggingFace resolver), `https://` URLs (still TODO).
fn resolve_vindexfile_path(
    path: &str,
    working_dir: &Path,
) -> Result<std::path::PathBuf, VindexError> {
    if crate::format::huggingface::is_hf_path(path) {
        // Use the same resolver `larql run` and `larql extract` use
        // — caches under HF's standard cache dir, conditional fetch
        // by ETag. Returns the local snapshot path.
        crate::format::huggingface::resolve_hf_vindex(path)
    } else if path.starts_with("https://") || path.starts_with("http://") {
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

    /// Save a tiny synthetic base through the real save path, so a build
    /// can actually load it. Everything above stops at an error before the
    /// base loads, which left the whole directive loop untested.
    fn save_synthetic_base(dir: &Path, layers: usize, features: usize, hidden: usize) {
        std::fs::create_dir_all(dir).unwrap();
        let gate_vectors: Vec<Option<ndarray::Array2<f32>>> = (0..layers)
            .map(|l| {
                Some(ndarray::Array2::from_shape_fn(
                    (features, hidden),
                    |(f, h)| (l * features * hidden + f * hidden + h) as f32 * 0.01 - 0.5,
                ))
            })
            .collect();
        let down_meta: Vec<Option<Vec<Option<crate::index::FeatureMeta>>>> = (0..layers)
            .map(|_| {
                Some(
                    (0..features)
                        .map(|i| {
                            Some(crate::index::FeatureMeta {
                                top_token: format!("tok{i}"),
                                top_token_id: i as u32,
                                c_score: 0.5,
                                top_k: vec![],
                            })
                        })
                        .collect(),
                )
            })
            .collect();
        let index = VectorIndex::new(gate_vectors, down_meta, layers, hidden);
        let layer_infos = index.save_gate_vectors(dir).unwrap();
        index.save_down_meta(dir).unwrap();
        // Minimal tokenizer so load_vindex's tokenizer read succeeds.
        std::fs::write(
            dir.join("tokenizer.json"),
            r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#,
        )
        .unwrap();
        let config = crate::config::VindexConfig {
            version: 2,
            model: "vindexfile-test".into(),
            family: "synthetic".into(),
            num_layers: layers,
            hidden_size: hidden,
            intermediate_size: features,
            vocab_size: 16,
            embed_scale: 1.0,
            layers: layer_infos,
            down_top_k: 1,
            ..Default::default()
        };
        VectorIndex::save_config(&config, dir).unwrap();
    }

    /// The directive loop end-to-end over a real base: INSERT claims a free
    /// slot (an override, so the result bakes down), DELETE removes the
    /// fact an earlier directive installed, LABELS and EXPOSE record/no-op, and
    /// the build history lists one layer per effective directive.
    #[test]
    fn build_executes_directives_over_a_real_base() {
        let tmp = tempfile::tempdir().unwrap();
        save_synthetic_base(&tmp.path().join("base.vindex"), 2, 4, 8);

        let vf = Vindexfile {
            directives: vec![
                VindexfileDirective::From("./base.vindex".into()),
                VindexfileDirective::Insert {
                    entity: "Acme".into(),
                    relation: "hq".into(),
                    target: "London".into(),
                },
                VindexfileDirective::Delete {
                    entity: "Acme".into(),
                    relation: "hq".into(),
                    target: "London".into(),
                },
                VindexfileDirective::Labels("labels.json".into()),
                VindexfileDirective::Expose(vec!["browse".into()]),
            ],
            stages: vec![],
        };
        let build = build_from_vindexfile(&vf, None, tmp.path()).expect("build succeeds");

        // FROM + INSERT + DELETE + LABELS recorded; EXPOSE is save-time only.
        let recorded: Vec<&str> = build
            .layers
            .iter()
            .map(|l| l.directive.split_whitespace().next().unwrap())
            .collect();
        assert_eq!(recorded, ["FROM", "INSERT", "DELETE", "LABELS"]);
        assert_eq!(
            build.layers[1].features_modified, 1,
            "INSERT claims one slot"
        );
        // DELETE names the fact the INSERT above installed, and removes
        // exactly that slot (`tests/vindexfile_delete.rs` holds the rest).
        assert_eq!(
            build.layers[2].features_modified, 1,
            "DELETE removes the one inserted fact"
        );
        assert_eq!(build.config.num_layers, 2, "config comes from the base");
    }

    /// INSERT with no free slot at the target layer is a hard error.
    /// Regression: `find_free_feature(layer).unwrap_or(0)` used to
    /// silently overwrite feature 0's metadata.
    #[test]
    fn insert_with_no_free_slot_is_error() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("base.vindex");
        // Layer 1 (num_layers / INSERT_LAYER_BAND_DIVISOR) has no
        // features at all — `find_free_feature` must return None there.
        std::fs::create_dir_all(&dir).unwrap();
        let gate_vectors: Vec<Option<ndarray::Array2<f32>>> =
            vec![Some(ndarray::Array2::from_elem((4, 8), 0.25)), None];
        let down_meta: Vec<Option<Vec<Option<crate::index::FeatureMeta>>>> = vec![None, None];
        let index = VectorIndex::new(gate_vectors, down_meta, 2, 8);
        let layer_infos = index.save_gate_vectors(&dir).unwrap();
        index.save_down_meta(&dir).unwrap();
        std::fs::write(
            dir.join("tokenizer.json"),
            r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#,
        )
        .unwrap();
        let config = crate::config::VindexConfig {
            version: 2,
            model: "vindexfile-test".into(),
            family: "synthetic".into(),
            num_layers: 2,
            hidden_size: 8,
            intermediate_size: 4,
            vocab_size: 16,
            embed_scale: 1.0,
            layers: layer_infos,
            down_top_k: 1,
            ..Default::default()
        };
        VectorIndex::save_config(&config, &dir).unwrap();

        let vf = Vindexfile {
            directives: vec![
                VindexfileDirective::From("./base.vindex".into()),
                VindexfileDirective::Insert {
                    entity: "Acme".into(),
                    relation: "hq".into(),
                    target: "London".into(),
                },
            ],
            stages: vec![],
        };
        let err = build_from_vindexfile(&vf, None, tmp.path())
            .err()
            .expect("INSERT with no free slot must error");
        let msg = err.to_string();
        assert!(msg.contains("no free feature slot"), "{msg}");
        assert!(msg.contains("layer 1"), "{msg}");
    }

    /// Without any override-producing directive the build skips bake-down
    /// and returns a clone of the base — the other arm of the final branch.
    #[test]
    fn build_without_overrides_returns_the_base() {
        let tmp = tempfile::tempdir().unwrap();
        save_synthetic_base(&tmp.path().join("base.vindex"), 2, 4, 8);

        let vf = Vindexfile {
            directives: vec![VindexfileDirective::From("./base.vindex".into())],
            stages: vec![],
        };
        let build = build_from_vindexfile(&vf, None, tmp.path()).expect("build succeeds");
        assert_eq!(build.layers.len(), 1, "only FROM in the history");
        assert_eq!(build.layers[0].features_modified, 0);
    }
}
