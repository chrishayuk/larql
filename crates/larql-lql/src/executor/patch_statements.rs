//! Patch statements: `BEGIN PATCH`, `SAVE PATCH`, `APPLY PATCH`, `SHOW PATCHES`, `REMOVE PATCH`.
//!
//! A patch applies through the session overlay and never touches the bound files. On a VINDEX3 binding the
//! stored-weight ops a patch carries are executed by `INFER` from the moment it is applied until it is removed
//! (`executor::vindex3::live`); they are written into a container only by `COMPILE CURRENT INTO VINDEX`.

use std::path::PathBuf;

use crate::error::LqlError;
use crate::executor::{Backend, PatchRecording, Session};

/// What `SAVE PATCH` wrote.
#[derive(Debug, Clone)]
pub struct SavedPatch {
    pub path: PathBuf,
    pub operations: usize,
    pub inserts: usize,
    pub updates: usize,
    pub deletes: usize,
    /// Stored-weight edits: a delta's elements, one per structured op.
    pub weight_edits: usize,
}

/// What `APPLY PATCH` applied.
#[derive(Debug, Clone)]
pub struct AppliedPatch {
    pub path: String,
    /// The SHA-256 of the file's bytes, as read and parsed.
    pub sha256: String,
    pub operations: usize,
    pub inserts: usize,
    pub updates: usize,
    pub deletes: usize,
    /// Elements of the `weight_delta` ops it carries.
    pub weight_edits: usize,
    /// Structured weight ops it carries.
    pub structured_ops: usize,
    /// The live-weights report the statement prints after applying.
    live: Vec<String>,
}

/// What `REMOVE PATCH` removed.
#[derive(Debug, Clone)]
pub struct RemovedPatch {
    /// Its 1-based place in the applied list, as `SHOW PATCHES` numbered it.
    pub number: usize,
    /// Whether it carried stored-weight ops, so removing it changed what executes.
    pub carried_weight_ops: bool,
    live: Vec<String>,
}

/// One applied patch, as `SHOW PATCHES` lists it.
#[derive(Debug, Clone)]
pub struct PatchSummary {
    /// The name `REMOVE PATCH` takes: the patch's description, or the path it was applied from.
    pub name: Option<String>,
    /// The file it was applied from, and the SHA-256 of its bytes (a VINDEX3 binding keeps both).
    pub path: Option<String>,
    pub sha256: Option<String>,
    pub operations: usize,
    pub inserts: usize,
    pub updates: usize,
    pub deletes: usize,
}

/// What `SHOW PATCHES` reports.
#[derive(Debug, Clone, Default)]
pub struct PatchesReport {
    pub applied: Vec<PatchSummary>,
    /// Entries the session overlay holds outside any applied patch.
    pub session_overrides: usize,
    /// Stored-weight ops in effect (applied or staged), and the elements of their deltas.
    pub weight_ops: usize,
    pub weight_elements: usize,
    /// The open recording: its path (empty for the anonymous one) and its pending operations.
    pub recording: Option<(String, usize)>,
}

impl Session {
    /// The typed operation behind `BEGIN PATCH`: open a recording that `SAVE PATCH` writes to `path`.
    pub fn begin_patch(&mut self, path: &str) -> Result<(), LqlError> {
        // An empty path is how an anonymous recording is marked. Opening a
        // named session with one would make a recording SAVE PATCH refuses and
        // BEGIN PATCH cannot adopt.
        if path.is_empty() {
            return Err(LqlError::Execution(
                "BEGIN PATCH needs a file name, e.g. BEGIN PATCH \"edits.vlp\".".into(),
            ));
        }
        if self.patch_recording.is_some() && !self.auto_patch {
            return Err(LqlError::Execution(
                "patch session already active. Run SAVE PATCH or discard first.".into(),
            ));
        }
        // If there was an auto-patch, upgrade it to a named one
        self.patch_recording = Some(PatchRecording {
            path: path.to_string(),
            operations: if self.auto_patch {
                // Keep existing operations from auto-patch
                self.patch_recording
                    .take()
                    .map(|r| r.operations)
                    .unwrap_or_default()
            } else {
                Vec::new()
            },
        });
        self.auto_patch = false;
        Ok(())
    }

    pub(super) fn exec_begin_patch(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
        self.begin_patch(path)?;
        Ok(vec![format!("Patch session started: {path}")])
    }

    /// The typed operation behind `SAVE PATCH`: write the open recording, which leaves the session only once
    /// the file is written.
    pub fn save_patch(&mut self) -> Result<SavedPatch, LqlError> {
        // The recording is the only copy of the session's edits in a form that
        // can be saved. It leaves the session only once the file is written:
        // every refusal and every failed write below keeps it.
        let recording = self.patch_recording.take().ok_or_else(|| {
            LqlError::Execution("no active patch session. Run BEGIN PATCH first.".into())
        })?;
        if recording.path.is_empty() {
            let pending = recording.operations.len();
            self.patch_recording = Some(recording);
            return Err(LqlError::Execution(format!(
                "anonymous patch session with {pending} pending operation(s): nothing was saved \
                 and the operations are kept. Name the patch with BEGIN PATCH \"filename.vlp\", \
                 then run SAVE PATCH."
            )));
        }

        let model_name = match &self.backend {
            Backend::Vindex { config, .. } => config.model.clone(),
            Backend::Vindex3 { runtime, .. } => runtime.model_name().to_string(),
            Backend::Weight { model_id, .. } => model_id.clone(),
            _ => "unknown".into(),
        };

        // A patch that carries stored-weight edits is version 2: the op is
        // additive, and a reader that does not know it refuses the file.
        let weight_edits: usize = recording
            .operations
            .iter()
            .map(|op| match op {
                larql_vindex::PatchOp::WeightDelta(w) => w.edits.len(),
                larql_vindex::PatchOp::StructuredWeight(_)
                | larql_vindex::PatchOp::LowRankColumnsWeight(_) => 1,
                _ => 0,
            })
            .sum();
        let patch = larql_vindex::VindexPatch {
            version: if weight_edits > 0 { 2 } else { 1 },
            base_model: model_name,
            base_checksum: None,
            created_at: String::new(),
            description: None,
            author: None,
            tags: vec![],
            operations: recording.operations,
        };

        let (ins, upd, del) = patch.counts();
        let path = PathBuf::from(&recording.path);
        if let Err(e) = patch.save(&path) {
            let pending = patch.operations.len();
            self.patch_recording = Some(PatchRecording {
                path: recording.path,
                operations: patch.operations,
            });
            return Err(LqlError::exec(
                format!(
                    "failed to save patch to {}; the {pending} pending operation(s) are kept, \
                     so SAVE PATCH can be run again",
                    path.display()
                ),
                e,
            ));
        }

        self.auto_patch = false;
        Ok(SavedPatch {
            path,
            operations: patch.operations.len(),
            inserts: ins,
            updates: upd,
            deletes: del,
            weight_edits,
        })
    }

    pub(super) fn exec_save_patch(&mut self) -> Result<Vec<String>, LqlError> {
        let saved = self.save_patch()?;
        let weights = if saved.weight_edits > 0 {
            format!(
                ", {} weight edit(s) bound to the source container",
                saved.weight_edits
            )
        } else {
            String::new()
        };
        Ok(vec![format!(
            "Saved: {} ({} inserts, {} updates, {} deletes{weights})",
            saved.path.display(),
            saved.inserts,
            saved.updates,
            saved.deletes,
        )])
    }

    /// The typed operation behind `APPLY PATCH`: check the `.vlp` at `path` whole, then apply it through the
    /// session overlay. A refused patch leaves the session as it was.
    pub fn apply_patch(&mut self, path: &str) -> Result<AppliedPatch, LqlError> {
        let patch_path = PathBuf::from(path);
        if !patch_path.exists() {
            return Err(LqlError::Execution(format!("patch not found: {path}")));
        }

        // Read once: the bytes parsed are the bytes digested, so a capture can name exactly what was applied.
        let bytes =
            std::fs::read(&patch_path).map_err(|e| LqlError::exec("failed to load patch", e))?;
        let file_sha256 = {
            use sha2::{Digest, Sha256};
            format!("{:x}", Sha256::digest(&bytes))
        };
        let mut patch: larql_vindex::VindexPatch = serde_json::from_slice(&bytes)
            .map_err(|e| LqlError::exec("failed to load patch", e))?;

        let (ins, upd, del) = patch.counts();
        let total = patch.len();

        // REMOVE PATCH addresses applied patches by description; a
        // patch saved without one is addressed by the path it was
        // applied from — on every backend, or removal could never
        // find it.
        if patch.description.is_none() {
            patch.description = Some(path.to_string());
        }

        // Stored-weight ops are bound to the exact container they were written
        // against. Check that before anything is applied.
        use larql_vindex::format::vindex3::weight_patch::WeightOpRef;
        let weight_ops: Vec<WeightOpRef<'_>> = patch
            .operations
            .iter()
            .filter_map(larql_vindex::PatchOp::weight_op)
            .collect();
        let weight_edits: usize = weight_ops
            .iter()
            .map(|w| match w {
                WeightOpRef::Delta(d) => d.edits.len(),
                WeightOpRef::Structured(_) | WeightOpRef::LowRankColumns(_) => 0,
            })
            .sum();
        let structured = weight_ops
            .iter()
            .filter(|w| w.structured_columns().is_some())
            .count();
        if !weight_ops.is_empty() {
            match &self.backend {
                Backend::Vindex3 {
                    path: container, ..
                } => {
                    for op in &weight_ops {
                        op.check_intact()
                            .and_then(|()| op.check_binding(container))
                            .map_err(|e| {
                                LqlError::Execution(format!("failed to apply patch: {e}"))
                            })?;
                    }
                }
                Backend::Vindex { .. } => return Err(LqlError::Execution(
                    "failed to apply patch: it edits the stored weights of a VINDEX3 container, \
                         and this session is bound to a VINDEX2 vindex"
                        .into(),
                )),
                _ => return Err(LqlError::NoBackend),
            }
        }

        // Apply through the session overlay (base files untouched). The patch
        // is checked whole first, on both backends: the model it was written
        // against, every slot it names and the width of every vector it
        // carries. A refused patch leaves the overlay as it was.
        let refused = |e| LqlError::exec("failed to apply patch", e);
        match &mut self.backend {
            Backend::Vindex {
                patched, config, ..
            } => {
                patch.check_base_model(&config.model).map_err(refused)?;
                patched.try_apply_patch(patch).map_err(refused)?;
            }
            Backend::Vindex3 {
                overlay,
                runtime,
                knowledge,
                patch_files,
                ..
            } => {
                patch
                    .check_base_model(runtime.model_name())
                    .map_err(refused)?;
                // The V3 overlay holds no base, so the container's shape is
                // supplied here. Without a browse view there is no feature
                // space to name a slot in, and a slot op is out of bounds.
                let plan = runtime.plan();
                patch
                    .check_geometry(&larql_vindex::PatchGeometry {
                        hidden_size: plan
                            .embedding
                            .as_ref()
                            .and_then(|e| e.table.shape.get(1).copied()),
                        num_layers: plan.layers.len(),
                        features_at: &|layer| {
                            knowledge
                                .as_ref()
                                .map_or(0, |view| view.num_features(layer))
                        },
                    })
                    .map_err(refused)?;
                overlay.try_apply_patch(patch).map_err(refused)?;
                patch_files.push(crate::executor::vindex3::AppliedPatchFile {
                    path: path.to_string(),
                    sha256: file_sha256.clone(),
                });
            }
            _ => return Err(LqlError::NoBackend),
        }
        // Applying any patch may change what the stored-weight ops in effect execute as.
        let live = self.refresh_live_weights();
        Ok(AppliedPatch {
            path: path.to_string(),
            sha256: file_sha256,
            operations: total,
            inserts: ins,
            updates: upd,
            deletes: del,
            weight_edits,
            structured_ops: structured,
            live,
        })
    }

    pub(super) fn exec_apply_patch(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
        let applied = self.apply_patch(path)?;
        let AppliedPatch {
            operations: total,
            inserts: ins,
            updates: upd,
            deletes: del,
            weight_edits,
            structured_ops: structured,
            ..
        } = applied;
        let mut weights = String::new();
        if weight_edits > 0 {
            weights.push_str(&format!(
                "; {weight_edits} stored-weight edit(s), bound to this container and pending COMPILE"
            ));
        }
        if structured > 0 {
            weights.push_str(&format!(
                "; {structured} structured weight op(s), bound to this container and pending COMPILE"
            ));
        }
        let mut out = vec![format!(
            "Applied: {path} ({total} operations: {ins} inserts, {upd} updates, {del} deletes{weights})"
        )];
        out.extend(applied.live);
        Ok(out)
    }

    /// The typed operation behind `SHOW PATCHES`: the applied patches, the session overlay's own entries, the
    /// stored-weight ops in effect and the open recording.
    pub fn patches(&self) -> Result<PatchesReport, LqlError> {
        let (patches, session_overrides, files) = match &self.backend {
            Backend::Vindex { patched, .. } => (&patched.patches, patched.num_overrides(), &[][..]),
            Backend::Vindex3 {
                overlay,
                patch_files,
                ..
            } => (&overlay.patches, overlay.num_overrides(), &patch_files[..]),
            _ => {
                self.require_patched()?;
                unreachable!("require_patched errors on every other backend");
            }
        };
        let applied = patches
            .iter()
            .enumerate()
            .map(|(i, patch)| {
                let (inserts, updates, deletes) = patch.counts();
                PatchSummary {
                    name: patch.description.clone(),
                    path: files.get(i).map(|f| f.path.clone()),
                    sha256: files.get(i).map(|f| f.sha256.clone()),
                    operations: patch.len(),
                    inserts,
                    updates,
                    deletes,
                }
            })
            .collect();
        let (weight_ops, weight_elements) = match &self.backend {
            Backend::Vindex3 { overlay, .. } => {
                (overlay.weight_ops().len(), overlay.weight_edit_count())
            }
            _ => (0, 0),
        };
        Ok(PatchesReport {
            applied,
            session_overrides,
            weight_ops,
            weight_elements,
            recording: self
                .patch_recording
                .as_ref()
                .map(|r| (r.path.clone(), r.operations.len())),
        })
    }

    pub(super) fn exec_show_patches(&self) -> Result<Vec<String>, LqlError> {
        // The applied-patch list plus the session-overlay entry count,
        // whichever overlay the binding holds: V2's slot overrides or
        // V3's KNN entries.
        let report = self.patches()?;
        let session_overrides = report.session_overrides;
        let mut out = Vec::new();

        if report.applied.is_empty() && session_overrides == 0 {
            out.push("  (no patches applied)".into());
        } else {
            for (i, patch) in report.applied.iter().enumerate() {
                let name = patch.name.as_deref().unwrap_or("(unnamed)");
                out.push(format!(
                    "  {}. {:<40} {} ops ({} ins, {} upd, {} del)",
                    i + 1,
                    name,
                    patch.operations,
                    patch.inserts,
                    patch.updates,
                    patch.deletes,
                ));
            }
            if session_overrides > 0 && report.applied.is_empty() {
                out.push(format!(
                    "  (anonymous session: {session_overrides} overrides)"
                ));
            }
            let file_total: usize = report.applied.iter().map(|p| p.operations).sum();
            let overlay_total = session_overrides;
            if file_total > 0 || overlay_total > 0 {
                out.push(format!(
                    "  Total: {} from files, {} in session",
                    file_total, overlay_total
                ));
            }
        }

        if let Backend::Vindex3 { overlay, .. } = &self.backend {
            if report.weight_ops > 0 {
                out.push(format!(
                    "  Stored-weight edits: {} element(s) in {} op(s), pending COMPILE",
                    report.weight_elements, report.weight_ops
                ));
                out.extend(crate::executor::vindex3::live_weight_lines(overlay));
            }
        }

        if let Some((path, pending)) = &report.recording {
            let label = if path.is_empty() { "(anonymous)" } else { path };
            out.push(format!("  Recording: {} ({} ops pending)", label, pending));
        }

        Ok(out)
    }

    /// The typed operation behind `REMOVE PATCH`: remove the applied patch `path` names.
    pub fn remove_patch(&mut self, path: &str) -> Result<RemovedPatch, LqlError> {
        let by_name = |p: &larql_vindex::VindexPatch| p.description.as_deref() == Some(path);
        // Whether the removed patch carried stored-weight ops: its removal is then a change to what executes.
        let mut carried_weight_ops = false;
        let pos = match &mut self.backend {
            Backend::Vindex { patched, .. } => match patched.patches.iter().position(by_name) {
                Some(i) => {
                    patched.remove_patch(i);
                    Some(i)
                }
                None => None,
            },
            Backend::Vindex3 {
                overlay,
                patch_files,
                ..
            } => match overlay.patches.iter().position(by_name) {
                Some(i) => {
                    if i < patch_files.len() {
                        patch_files.remove(i);
                    }
                    carried_weight_ops = overlay.patches[i]
                        .operations
                        .iter()
                        .any(|op| op.weight_op().is_some());
                    overlay.remove_patch(i);
                    Some(i)
                }
                None => None,
            },
            _ => return Err(LqlError::NoBackend),
        };
        let Some(i) = pos else {
            return Err(LqlError::Execution(format!("patch not found: {path}")));
        };
        let live = self.refresh_live_weights();
        Ok(RemovedPatch {
            number: i + 1,
            carried_weight_ops,
            live,
        })
    }

    pub(super) fn exec_remove_patch(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
        let removed = self.remove_patch(path)?;
        let mut out = vec![format!("Removed patch #{}", removed.number)];
        if removed.carried_weight_ops && removed.live.is_empty() {
            out.push(crate::executor::vindex3::nothing_in_effect_line());
        }
        out.extend(removed.live);
        Ok(out)
    }
}
