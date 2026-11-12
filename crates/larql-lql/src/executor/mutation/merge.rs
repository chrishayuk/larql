//! `MERGE source [INTO target]` — merge another vindex's features into
//! the current session overlay under a conflict strategy.
//!
//! The source is a V2 vindex (the portable feature-table artifact);
//! the target is whichever binding the session holds — V2's
//! `PatchedVindex` or V3's `KnowledgeOverlay`. The conflict decision
//! is shared, so one MERGE means the same thing on either backend.

use std::path::PathBuf;

use crate::ast::ConflictStrategy;
use crate::error::LqlError;
use crate::executor::{Backend, Session};

/// What a MERGE source has to match before any of it is written.
struct MergeTarget<'a> {
    /// The model the bound index was built from.
    model: &'a str,
    num_layers: usize,
    /// Feature slots at a layer of the bound index.
    features_at: &'a dyn Fn(usize) -> usize,
    /// The session's current (overlay-merged) record at a slot.
    current: &'a dyn Fn(usize, usize) -> Option<larql_vindex::FeatureMeta>,
}

/// The records a MERGE writes, and how many it leaves as they are.
struct MergePlan {
    writes: Vec<(usize, usize, larql_vindex::FeatureMeta)>,
    skipped: usize,
}

/// Decide the whole MERGE without touching the session.
///
/// A source record means something only in the model it was read from and
/// at a slot the target has, so both are checked for every record before
/// one is written: a refused MERGE changes nothing.
fn plan_merge(
    source_model: &str,
    source: &larql_vindex::VectorIndex,
    target: &MergeTarget<'_>,
    strategy: &ConflictStrategy,
) -> Result<MergePlan, LqlError> {
    // Both names must be present, as APPLY PATCH requires: a source that
    // declares no model cannot be shown to belong to this one.
    if source_model.is_empty() || target.model.is_empty() {
        return Err(LqlError::Execution(format!(
            "MERGE refused: the source declares the model `{source_model}` and this is `{}`; \
             without both names the source cannot be shown to belong here. Nothing was merged.",
            target.model
        )));
    }
    if source_model != target.model {
        return Err(LqlError::Execution(format!(
            "MERGE refused: the source was built from `{source_model}`, and this is `{}`. \
             Nothing was merged.",
            target.model
        )));
    }
    // Records are read layer by layer through the gate table. A source
    // without one (a quantised or FP4-only vindex) would read as holding
    // no records, and the merge would answer "0 merged" over all of them.
    if source.loaded_layers().is_empty() {
        return Err(LqlError::Execution(
            "MERGE refused: the source has no gate vectors, so its records cannot be read. \
             Nothing was merged."
                .into(),
        ));
    }
    let mut plan = MergePlan {
        writes: Vec::new(),
        skipped: 0,
    };
    for layer in source.loaded_layers() {
        for feature in 0..source.num_features(layer) {
            let Some(source_meta) = source.feature_meta(layer, feature) else {
                continue;
            };
            let features = if layer < target.num_layers {
                (target.features_at)(layer)
            } else {
                0
            };
            if feature >= features {
                return Err(LqlError::Execution(format!(
                    "MERGE refused: the source has a record at L{layer}/F{feature}, which is \
                     out of bounds here ({} layers; layer {layer} has {features} features). \
                     Nothing was merged.",
                    target.num_layers
                )));
            }
            let existing = (target.current)(layer, feature);
            if merge_should_write(existing.as_ref(), &source_meta, strategy) {
                plan.writes.push((layer, feature, source_meta));
            } else {
                plan.skipped += 1;
            }
        }
    }
    Ok(plan)
}

/// Whether two paths name the same directory.
fn same_directory(a: &std::path::Path, b: &std::path::Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

impl Session {
    pub(crate) fn exec_merge(
        &mut self,
        source: &str,
        target: Option<&str>,
        conflict: Option<ConflictStrategy>,
    ) -> Result<Vec<String>, LqlError> {
        let source_path = PathBuf::from(source);
        if !source_path.exists() {
            return Err(LqlError::Execution(format!(
                "source vindex not found: {}",
                source_path.display()
            )));
        }

        let bound_path = match &self.backend {
            Backend::Vindex { path, .. } => path.clone(),
            Backend::Vindex3 { path, .. } => path.clone(),
            _ => return Err(LqlError::NoBackend),
        };
        // A MERGE writes to the session overlay of the bound index and to
        // nothing else. `INTO` may name that index; any other path would be
        // reported as merged into while nothing was written to it.
        if let Some(t) = target {
            let named = PathBuf::from(t);
            if !named.exists() {
                return Err(LqlError::Execution(format!(
                    "target vindex not found: {}",
                    named.display()
                )));
            }
            if !same_directory(&named, &bound_path) {
                return Err(LqlError::Execution(format!(
                    "MERGE INTO {} refused: a MERGE writes to the overlay of the bound vindex \
                     ({}), and to no other. USE the target first. Nothing was merged.",
                    named.display(),
                    bound_path.display()
                )));
            }
        }

        let strategy = conflict.unwrap_or(ConflictStrategy::KeepSource);

        // Load source
        let source_config = larql_vindex::load_vindex_config(&source_path)
            .map_err(|e| LqlError::exec("failed to load source config", e))?;
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let source_index = larql_vindex::VectorIndex::load_vindex(&source_path, &mut cb)
            .map_err(|e| LqlError::exec("failed to load source", e))?;

        // Plan against the bound index with a readonly borrow. The source is
        // read through the heap+mmap-aware accessors, so this works on
        // production `load_vindex` output (where down_meta is mmapped).
        let plan = match &self.backend {
            Backend::Vindex {
                patched, config, ..
            } => plan_merge(
                &source_config.model,
                &source_index,
                &MergeTarget {
                    model: &config.model,
                    num_layers: patched.base().num_layers,
                    features_at: &|layer| patched.num_features(layer),
                    current: &|layer, feature| patched.feature_meta(layer, feature),
                },
                &strategy,
            )?,
            Backend::Vindex3 {
                knowledge,
                overlay,
                runtime,
                ..
            } => {
                let view = knowledge.as_ref().ok_or_else(|| {
                    LqlError::Execution(
                        "MERGE needs the browse view and this container carries no \
                         tokenizer.json — feature annotations cannot be decoded"
                            .into(),
                    )
                })?;
                plan_merge(
                    &source_config.model,
                    &source_index,
                    &MergeTarget {
                        model: runtime.model_name(),
                        num_layers: runtime.plan().layers.len(),
                        features_at: &|layer| view.num_features(layer),
                        current: &|layer, feature| {
                            overlay.resolve_feature_meta(
                                layer,
                                feature,
                                view.feature_meta(layer, feature),
                            )
                        },
                    },
                    &strategy,
                )?
            }
            _ => return Err(LqlError::NoBackend),
        };

        let mut out = Vec::new();
        if !plan.writes.is_empty() {
            // Only now, with every record checked and something to record,
            // does the statement touch session state. Each write is recorded
            // as the UPDATE it is, so SAVE PATCH and COMPILE see the merge.
            out.extend(self.ensure_patch_session());
            self.advance_epoch();
            match &mut self.backend {
                Backend::Vindex { patched, .. } => {
                    for (layer, feature, meta) in &plan.writes {
                        patched.update_feature_meta(*layer, *feature, meta.clone());
                    }
                }
                Backend::Vindex3 { overlay, .. } => {
                    for (layer, feature, meta) in &plan.writes {
                        overlay.update_feature_meta(*layer, *feature, meta.clone());
                    }
                }
                _ => unreachable!("every other backend was refused above"),
            }
            if let Some(ref mut recording) = self.patch_recording {
                for (layer, feature, meta) in &plan.writes {
                    recording.operations.push(larql_vindex::PatchOp::Update {
                        layer: *layer,
                        feature: *feature,
                        gate_vector_b64: None,
                        up_vector_b64: None,
                        down_vector_b64: None,
                        down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                            top_token: meta.top_token.clone(),
                            top_token_id: meta.top_token_id,
                            c_score: meta.c_score,
                        }),
                    });
                }
            }
        }

        out.push(format!(
            "Merged {} → {} (patch overlay)",
            source_path.display(),
            bound_path.display()
        ));
        out.push(format!(
            "  {} features merged, {} skipped (strategy: {:?})",
            plan.writes.len(),
            plan.skipped,
            strategy
        ));
        Ok(out)
    }
}

/// The conflict decision, shared by both backends: absent target slots
/// always take the source; present ones follow the strategy.
fn merge_should_write(
    existing: Option<&larql_vindex::FeatureMeta>,
    source: &larql_vindex::FeatureMeta,
    strategy: &ConflictStrategy,
) -> bool {
    match (existing, strategy) {
        (None, _) => true,
        (Some(_), ConflictStrategy::KeepSource) => true,
        (Some(_), ConflictStrategy::KeepTarget) => false,
        (Some(existing), ConflictStrategy::HighestConfidence) => source.c_score > existing.c_score,
    }
}
