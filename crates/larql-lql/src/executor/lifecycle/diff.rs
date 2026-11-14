//! `DIFF a b [INTO PATCH p]` — two-way vindex diff with optional
//! extraction as a `.vlp` patch file.

use std::path::PathBuf;

use crate::ast::VindexRef;
use crate::error::LqlError;
use crate::executor::{Backend, Session};

impl Session {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn exec_diff(
        &self,
        a: &VindexRef,
        b: &VindexRef,
        layer_filter: Option<u32>,
        _relation: Option<&str>,
        limit: Option<u32>,
        into_patch: Option<&str>,
        physical: bool,
    ) -> Result<Vec<String>, LqlError> {
        // The one V2/V3 decision for DIFF: a side is V3 when its path
        // carries the V3 generation marker, or when CURRENT is bound
        // to a V3 container. Mixed-generation diffs are a later rung.
        let is_v3 = |r: &VindexRef| -> bool {
            match r {
                VindexRef::Current => matches!(self.backend, Backend::Vindex3 { .. }),
                VindexRef::Path(p) => matches!(
                    larql_vindex::format::generation::detect_generation(std::path::Path::new(p)),
                    Ok(larql_vindex::format::generation::ContainerGeneration::V3)
                ),
            }
        };
        match (is_v3(a), is_v3(b)) {
            (true, true) => {
                return self.exec_diff_v3(
                    a,
                    b,
                    layer_filter,
                    _relation,
                    limit,
                    into_patch,
                    physical,
                )
            }
            (false, false) => {}
            _ => {
                return Err(LqlError::Execution(
                    "DIFF across generations (one VINDEX2 side, one VINDEX3 side) is a \
                     later rung — realise both models in one generation first"
                        .into(),
                ))
            }
        }
        if physical {
            return Err(LqlError::Execution(
                "DIFF … PHYSICAL is a VINDEX3 report (segment-level); VINDEX2 sides use \
                 the default logical diff"
                    .into(),
            ));
        }
        let (path_a, model_a, index_a) = self.diff_side(a)?;
        let (path_b, model_b, index_b) = self.diff_side(b)?;

        let limit = limit.unwrap_or(20) as usize;

        let mut out = Vec::new();
        out.push(format!(
            "Diff: {} vs {}",
            path_a.display(),
            path_b.display()
        ));
        out.push(format!(
            "{:<8} {:<8} {:<20} {:<20} {:>10}",
            "Layer", "Feature", "A (token)", "B (token)", "Status"
        ));
        out.push("-".repeat(70));

        // Every layer either side has: a layer only B holds is all "added".
        let mut layers_a = index_a.loaded_layers();
        layers_a.extend(index_b.loaded_layers());
        layers_a.sort_unstable();
        layers_a.dedup();
        let mut diff_count = 0;

        for layer in &layers_a {
            if let Some(l) = layer_filter {
                if *layer != l as usize {
                    continue;
                }
            }
            if diff_count >= limit {
                break;
            }

            // `feature_meta` covers both heap (mutation overrides) and
            // mmap (the production read path). The earlier `down_meta_at`
            // version was heap-only, which made DIFF a no-op on the
            // standard load_vindex output where down_meta lives in the
            // mmap'd `down_meta.bin`.
            let len_a = index_a.num_features(*layer);
            let len_b = index_b.num_features(*layer);
            let max_features = len_a.max(len_b);

            for feat in 0..max_features {
                if diff_count >= limit {
                    break;
                }

                let meta_a = index_a.feature_meta(*layer, feat);
                let meta_b = index_b.feature_meta(*layer, feat);

                let status = match (&meta_a, &meta_b) {
                    (Some(a), Some(b)) => {
                        if a.top_token != b.top_token || (a.c_score - b.c_score).abs() > 0.01 {
                            "modified"
                        } else {
                            continue;
                        }
                    }
                    (Some(_), None) => "removed",
                    (None, Some(_)) => "added",
                    (None, None) => continue,
                };

                let tok_a = meta_a.as_ref().map(|m| m.top_token.as_str()).unwrap_or("-");
                let tok_b = meta_b.as_ref().map(|m| m.top_token.as_str()).unwrap_or("-");

                out.push(format!(
                    "L{:<7} F{:<7} {:<20} {:<20} {:>10}",
                    layer, feat, tok_a, tok_b, status
                ));
                diff_count += 1;
            }
        }

        if diff_count == 0 {
            out.push("  (no differences found)".into());
        } else {
            out.push(format!(
                "\n{} differences shown (limit {})",
                diff_count, limit
            ));
        }

        // If INTO PATCH specified, extract diff as a .vlp file
        if let Some(patch_path) = into_patch {
            // The patch turns A into B, so A's model is the one it applies
            // to. APPLY PATCH refuses a patch whose model is not the bound
            // one, so a patch across two models, or from an index that
            // names none, could never be applied as what it claims to be.
            if model_a.is_empty() || model_a != model_b {
                return Err(LqlError::Execution(format!(
                    "DIFF INTO PATCH refused: {} was built from `{model_a}` and {} from \
                     `{model_b}`. A patch is applied to the one model it names, so both \
                     sides must be that model. No patch was written.",
                    path_a.display(),
                    path_b.display()
                )));
            }
            let mut operations = Vec::new();

            // Re-scan without limit for the full diff
            for layer in &layers_a {
                if let Some(l) = layer_filter {
                    if *layer != l as usize {
                        continue;
                    }
                }
                let len_a = index_a.num_features(*layer);
                let len_b = index_b.num_features(*layer);

                for feat in 0..len_a.max(len_b) {
                    let ma = index_a.feature_meta(*layer, feat);
                    let mb = index_b.feature_meta(*layer, feat);

                    match (&ma, &mb) {
                        (Some(a_meta), Some(b))
                            if a_meta.top_token != b.top_token
                                || (a_meta.c_score - b.c_score).abs() > 0.01 =>
                        {
                            operations.push(larql_vindex::PatchOp::Update {
                                layer: *layer,
                                feature: feat,
                                gate_vector_b64: None,
                                up_vector_b64: None,
                                down_vector_b64: None,
                                down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                                    top_token: b.top_token.clone(),
                                    top_token_id: b.top_token_id,
                                    c_score: b.c_score,
                                }),
                            });
                        }
                        (Some(_), None) => {
                            operations.push(larql_vindex::PatchOp::Delete {
                                layer: *layer,
                                feature: feat,
                                reason: Some("removed in target".into()),
                            });
                        }
                        (None, Some(b)) => {
                            operations.push(larql_vindex::PatchOp::Insert {
                                layer: *layer,
                                feature: feat,
                                relation: None,
                                entity: String::new(),
                                target: b.top_token.clone(),
                                confidence: Some(b.c_score),
                                gate_vector_b64: None,
                                up_vector_b64: None,
                                down_vector_b64: None,
                                down_meta: Some(larql_vindex::patch::core::PatchDownMeta {
                                    top_token: b.top_token.clone(),
                                    top_token_id: b.top_token_id,
                                    c_score: b.c_score,
                                }),
                            });
                        }
                        _ => {}
                    }
                }
            }

            let patch = larql_vindex::VindexPatch {
                version: 1,
                base_model: model_a,
                base_checksum: None,
                created_at: String::new(),
                description: Some(format!(
                    "Diff: {} vs {}",
                    path_a.display(),
                    path_b.display()
                )),
                author: None,
                tags: vec![],
                operations,
            };

            let (ins, upd, del) = patch.counts();
            patch
                .save(std::path::Path::new(patch_path))
                .map_err(|e| LqlError::exec("failed to save patch", e))?;
            out.push(format!(
                "Extracted: {} ({} ops: {} inserts, {} updates, {} deletes)",
                patch_path,
                patch.len(),
                ins,
                upd,
                del,
            ));
        }

        Ok(out)
    }

    /// One side of a VINDEX2 DIFF: where it is, the model it was built
    /// from, and its feature records.
    ///
    /// `CURRENT` is the session: the bound vindex with its patch overlay,
    /// which is what the user has in front of them. Loading the bound path
    /// from disk instead showed the base files, so `DIFF "base" CURRENT`
    /// after any number of edits reported no differences.
    fn diff_side(&self, side: &VindexRef) -> Result<(PathBuf, String, DiffSide<'_>), LqlError> {
        if let (
            VindexRef::Current,
            Backend::Vindex {
                path,
                config,
                patched,
                ..
            },
        ) = (side, &self.backend)
        {
            return Ok((
                path.clone(),
                config.model.clone(),
                DiffSide::Session(patched),
            ));
        }
        let path = self.resolve_vindex_ref(side)?;
        let config = larql_vindex::load_vindex_config(&path)
            .map_err(|e| LqlError::exec(format!("failed to load {}", path.display()), e))?;
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let index = larql_vindex::VectorIndex::load_vindex(&path, &mut cb)
            .map_err(|e| LqlError::exec(format!("failed to load {}", path.display()), e))?;
        Ok((path, config.model, DiffSide::Loaded(Box::new(index))))
    }

    /// Resolve a VindexRef to a concrete path.
    fn resolve_vindex_ref(&self, vref: &VindexRef) -> Result<PathBuf, LqlError> {
        match vref {
            VindexRef::Current => match &self.backend {
                Backend::Vindex { path, .. } => Ok(path.clone()),
                Backend::Weight { model_id, .. } => Err(LqlError::Execution(
                    crate::executor::extract_hint::extract_first(
                        "CURRENT refers to a live model, not a vindex",
                        model_id,
                    ),
                )),
                _ => Err(LqlError::NoBackend),
            },
            VindexRef::Path(p) => {
                let path = PathBuf::from(p);
                if !path.exists() {
                    return Err(LqlError::Execution(format!(
                        "vindex not found: {}",
                        path.display()
                    )));
                }
                Ok(path)
            }
        }
    }
}

/// The feature records of one side of a VINDEX2 DIFF.
enum DiffSide<'a> {
    /// A vindex loaded from disk.
    Loaded(Box<larql_vindex::VectorIndex>),
    /// The session's bound vindex, read through its patch overlay.
    Session(&'a larql_vindex::PatchedVindex),
}

impl DiffSide<'_> {
    fn loaded_layers(&self) -> Vec<usize> {
        match self {
            Self::Loaded(index) => index.loaded_layers(),
            Self::Session(patched) => patched.loaded_layers(),
        }
    }

    fn num_features(&self, layer: usize) -> usize {
        match self {
            Self::Loaded(index) => index.num_features(layer),
            Self::Session(patched) => patched.num_features(layer),
        }
    }

    fn feature_meta(&self, layer: usize, feature: usize) -> Option<larql_vindex::FeatureMeta> {
        match self {
            Self::Loaded(index) => index.feature_meta(layer, feature),
            Self::Session(patched) => patched.feature_meta(layer, feature),
        }
    }
}
