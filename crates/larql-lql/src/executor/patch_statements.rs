//! Patch statements: `BEGIN PATCH`, `SAVE PATCH`, `APPLY PATCH`, `SHOW PATCHES`, `REMOVE PATCH`.
//!
//! A patch applies through the session overlay and never touches the bound files. On a VINDEX3 binding the
//! stored-weight ops a patch carries are executed by `INFER` from the moment it is applied until it is removed
//! (`executor::vindex3::live`); they are written into a container only by `COMPILE CURRENT INTO VINDEX`.

use std::path::PathBuf;

use crate::error::LqlError;
use crate::executor::{Backend, PatchRecording, Session};

impl Session {
    pub(super) fn exec_begin_patch(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
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
        Ok(vec![format!("Patch session started: {path}")])
    }

    pub(super) fn exec_save_patch(&mut self) -> Result<Vec<String>, LqlError> {
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

        let weights = if weight_edits > 0 {
            format!(", {weight_edits} weight edit(s) bound to the source container")
        } else {
            String::new()
        };
        Ok(vec![format!(
            "Saved: {} ({} inserts, {} updates, {} deletes{weights})",
            path.display(),
            ins,
            upd,
            del,
        )])
    }

    pub(super) fn exec_apply_patch(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
        let patch_path = PathBuf::from(path);
        if !patch_path.exists() {
            return Err(LqlError::Execution(format!("patch not found: {path}")));
        }

        let mut patch = larql_vindex::VindexPatch::load(&patch_path)
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
            }
            _ => return Err(LqlError::NoBackend),
        }

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
        // Applying any patch may change what the stored-weight ops in effect execute as.
        out.extend(self.refresh_live_weights());
        Ok(out)
    }

    pub(super) fn exec_show_patches(&self) -> Result<Vec<String>, LqlError> {
        // The applied-patch list plus the session-overlay entry count,
        // whichever overlay the binding holds: V2's slot overrides or
        // V3's KNN entries.
        let (patches, session_overrides) = match &self.backend {
            Backend::Vindex { patched, .. } => (&patched.patches, patched.num_overrides()),
            Backend::Vindex3 { overlay, .. } => (&overlay.patches, overlay.num_overrides()),
            _ => {
                self.require_patched()?;
                unreachable!("require_patched errors on every other backend");
            }
        };
        let mut out = Vec::new();

        if patches.is_empty() && session_overrides == 0 {
            out.push("  (no patches applied)".into());
        } else {
            for (i, patch) in patches.iter().enumerate() {
                let (ins, upd, del) = patch.counts();
                let name = patch.description.as_deref().unwrap_or("(unnamed)");
                out.push(format!(
                    "  {}. {:<40} {} ops ({} ins, {} upd, {} del)",
                    i + 1,
                    name,
                    patch.len(),
                    ins,
                    upd,
                    del,
                ));
            }
            if session_overrides > 0 && patches.is_empty() {
                out.push(format!(
                    "  (anonymous session: {session_overrides} overrides)"
                ));
            }
            let file_total: usize = patches.iter().map(|p| p.len()).sum();
            let overlay_total = session_overrides;
            if file_total > 0 || overlay_total > 0 {
                out.push(format!(
                    "  Total: {} from files, {} in session",
                    file_total, overlay_total
                ));
            }
        }

        if let Backend::Vindex3 { overlay, .. } = &self.backend {
            let ops = overlay.weight_ops();
            if !ops.is_empty() {
                out.push(format!(
                    "  Stored-weight edits: {} element(s) in {} op(s), pending COMPILE",
                    overlay.weight_edit_count(),
                    ops.len()
                ));
                out.extend(crate::executor::vindex3::live_weight_lines(overlay));
            }
        }

        if let Some(ref recording) = self.patch_recording {
            let label = if recording.path.is_empty() {
                "(anonymous)"
            } else {
                &recording.path
            };
            out.push(format!(
                "  Recording: {} ({} ops pending)",
                label,
                recording.operations.len()
            ));
        }

        Ok(out)
    }

    pub(super) fn exec_remove_patch(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
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
            Backend::Vindex3 { overlay, .. } => match overlay.patches.iter().position(by_name) {
                Some(i) => {
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
        let mut out = vec![format!("Removed patch #{}", i + 1)];
        let live = self.refresh_live_weights();
        if carried_weight_ops && live.is_empty() {
            out.push(crate::executor::vindex3::nothing_in_effect_line());
        }
        out.extend(live);
        Ok(out)
    }
}
