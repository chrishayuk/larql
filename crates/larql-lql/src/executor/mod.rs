//! LQL Executor — dispatches parsed AST statements to backend operations
//!
//! The base vindex is always readonly. All mutations go through a patch overlay.
//! INSERT/DELETE/UPDATE auto-start an anonymous patch session if none is active.

mod backend;
mod compact;
mod extract_hint;
mod helpers;
mod introspection;
mod knowledge;
mod lifecycle;
mod memit_persist;
pub mod model_resolver;
mod mutation;
pub(crate) mod query;
mod relation_resolver;
mod remote;
mod trace;
pub(crate) mod tuning;
mod verify;
mod vindex3;

#[cfg(test)]
mod tests;

use std::path::PathBuf;

use crate::ast::*;
use crate::capability::CapabilityProfile;
use crate::error::LqlError;

pub mod balance_trace;
pub mod install_fault;

pub(crate) use backend::{Backend, InstalledEdge, PatchRecording};

/// Session state for the REPL / batch executor.
pub struct Session {
    pub(crate) backend: Backend,
    /// The capability gate this session executes under. `Full` unless
    /// the embedding surface tightened it — see `crate::capability`.
    profile: CapabilityProfile,
    /// Active patch recording session (between BEGIN PATCH and SAVE PATCH).
    /// If None and a mutation happens, an anonymous patch is auto-started.
    pub(crate) patch_recording: Option<PatchRecording>,
    /// Whether the current patch was auto-started (anonymous).
    pub(crate) auto_patch: bool,
    /// Architecture A: per-layer cached decoy residuals. Kept for backward compat.
    #[allow(dead_code)]
    pub(crate) decoy_residual_cache:
        std::collections::HashMap<usize, Vec<larql_vindex::ndarray::Array1<f32>>>,
    /// Raw captured residuals per installed slot, indexed by
    /// `(layer, feature)`. Each entry is the unscaled residual the
    /// model produced for the install prompt at that layer's FFN
    /// entry (last token), captured on the clean base index before
    /// any installs were applied. Used by INSERT's online refine
    /// pass to rebuild the full constellation from raw inputs each
    /// time a new slot lands — if we refined against the currently
    /// stored (already-refined) peers, compound drift across
    /// iterations would leave the latest insert dominating (see
    /// `refine_demo` 10-fact run where every prompt returned the
    /// last-installed target before this cache existed).
    #[allow(dead_code)]
    pub(crate) raw_install_residuals:
        std::collections::HashMap<(usize, usize), larql_vindex::ndarray::Array1<f32>>,
    /// Per-install fact metadata. Enables cross-fact balance: when a
    /// new INSERT's local balance converges, we replay every prior
    /// install's canonical prompt through INFER and scale the NEW
    /// install's down_col further if any prior fact regressed below
    /// the retrieval floor. Without this, a single install at N=5+
    /// can grow a gate that fires on template-matched siblings,
    /// hijacking their prior install's target (observed as "H" on
    /// every template query after Hyrule→Hateno install).
    #[allow(dead_code)]
    pub(crate) installed_edges: Vec<InstalledEdge>,
    /// FR3 synonym-robust relation resolver, built lazily on first synonym
    /// SELECT and cached per vindex path (rebuilt if the active vindex changes).
    /// `RefCell` for interior mutability under `&self` query methods; the inner
    /// `Option` caches a failed/empty build so it isn't retried every query.
    pub(crate) relation_resolver:
        std::cell::RefCell<Option<(PathBuf, Option<relation_resolver::RelationResolver>)>>,
    /// LSM epoch counter — advances on every mutation (INSERT/DELETE/UPDATE).
    pub(crate) epoch: u64,
    /// Mutations since last minor compaction (L0 → L1).
    pub(crate) mutations_since_minor: usize,
    /// Mutations since last major compaction (L1 → L2).
    pub(crate) mutations_since_major: usize,
    /// The external verifier for `VERIFY REPRESENTATION`, if set explicitly.
    pub(crate) verifier: Option<std::path::PathBuf>,
    /// `VERIFY REPRESENTATION` statements in this session that reported `violated`.
    pub(crate) contract_violations: usize,
    /// Observer of COMPOSE balance passes, if one is set. See [`crate::balance_trace`].
    pub(crate) balance_trace: crate::executor::balance_trace::ObserverSlot,
    /// Fault armed for the next COMPOSE install, if any. See [`crate::install_fault`].
    pub(crate) install_fault: crate::executor::install_fault::FaultSlot,
    /// Resolves a `USE` literal that is not a path. See [`model_resolver`].
    pub(crate) model_resolver: Option<model_resolver::ModelResolver>,
}

impl Default for Session {
    fn default() -> Self {
        Self::new()
    }
}

impl Session {
    pub fn new() -> Self {
        Self {
            backend: Backend::None,
            profile: CapabilityProfile::default(),
            patch_recording: None,
            auto_patch: false,
            decoy_residual_cache: std::collections::HashMap::new(),
            raw_install_residuals: std::collections::HashMap::new(),
            installed_edges: Vec::new(),
            relation_resolver: std::cell::RefCell::new(None),
            epoch: 0,
            mutations_since_minor: 0,
            mutations_since_major: 0,
            verifier: None,
            contract_violations: 0,
            balance_trace: crate::executor::balance_trace::ObserverSlot::default(),
            install_fault: crate::executor::install_fault::FaultSlot::default(),
            model_resolver: None,
        }
    }

    /// Sets (or, with `None`, removes) the hook `USE` asks about a literal
    /// that is not an existing path. See [`model_resolver`].
    pub fn set_model_resolver(&mut self, resolver: Option<model_resolver::ModelResolver>) {
        self.model_resolver = resolver;
    }

    /// Sets (or, with `None`, removes) the observer of COMPOSE balance passes. It only observes:
    /// installs are bit-identical with and without it. See [`crate::balance_trace`].
    pub fn set_balance_observer(
        &mut self,
        observer: Option<Box<dyn crate::executor::balance_trace::BalanceObserver>>,
    ) {
        self.balance_trace.set(observer);
    }

    /// How many times a balance observer panicked and was detached in this session.
    pub fn balance_observer_failures(&self) -> usize {
        self.balance_trace.failures()
    }

    /// Ensure a patch session is active. If not, auto-start an anonymous one.
    /// Returns messages about auto-patch start (empty if already active).
    pub(crate) fn ensure_patch_session(&mut self) -> Vec<String> {
        if self.patch_recording.is_some() {
            return vec![];
        }
        self.patch_recording = Some(PatchRecording {
            path: String::new(), // anonymous
            operations: Vec::new(),
        });
        self.auto_patch = true;
        vec!["Auto-patch started (use SAVE PATCH \"file.vlp\" to persist, or edits are lost on exit)".into()]
    }

    /// The verifier `VERIFY REPRESENTATION` runs, ahead of `LARQL_VERIFIER` and
    /// discovery. An empty path turns discovery off, so the statement refuses.
    pub fn set_verifier(&mut self, verifier: impl Into<std::path::PathBuf>) {
        self.verifier = Some(verifier.into());
    }

    /// How many `VERIFY REPRESENTATION` statements in this session reported
    /// `violated`. They return normally, so this is the way to tell.
    pub fn contract_violations(&self) -> usize {
        self.contract_violations
    }

    /// Constrain every subsequent `execute` to `profile`. Judged at
    /// the head of `execute`, so it also governs the remote transport
    /// and both legs of a pipe.
    pub fn set_profile(&mut self, profile: CapabilityProfile) {
        self.profile = profile;
    }

    /// The capability profile this session executes under.
    pub fn profile(&self) -> CapabilityProfile {
        self.profile
    }

    pub fn execute(&mut self, stmt: &Statement) -> Result<Vec<String>, LqlError> {
        // The capability gate: after parsing, before any execution —
        // including the remote fork, so no transport bypasses it. A
        // pipe is judged whole here; a refused leg refuses the
        // statement before the other leg runs.
        self.profile.check(stmt)?;

        // Remote backend: forward supported queries via HTTP.
        if self.is_remote() {
            return self.execute_remote(stmt);
        }

        match stmt {
            Statement::Pipe { left, right } => {
                let mut out = self.execute(left)?;
                out.extend(self.execute(right)?);
                Ok(out)
            }
            Statement::Use { target } => self.exec_use(target),
            Statement::Stats { vindex } => self.exec_stats(vindex.as_deref()),
            Statement::VerifyOperator {
                operator,
                contract,
                witness,
                container,
            } => self.exec_verify_operator(operator, contract, witness, container),
            Statement::VerifyRepresentation {
                container,
                contract,
                source,
            } => self.exec_verify_representation(container, contract, source),
            Statement::Walk {
                prompt,
                top,
                layers,
                mode,
                compare,
            } => self.exec_walk(prompt, *top, layers.as_ref(), *mode, *compare),
            Statement::Describe {
                entity,
                band,
                layer,
                relations_only,
                mode,
            } => self.exec_describe(entity, *band, *layer, *relations_only, *mode),
            Statement::Select {
                source,
                fields,
                conditions,
                nearest,
                order,
                limit,
            } => match source {
                SelectSource::Edges => {
                    self.exec_select(fields, conditions, nearest.as_ref(), order.as_ref(), *limit)
                }
                SelectSource::Features => self.exec_select_features(conditions, *limit),
                SelectSource::Entities => self.exec_select_entities(conditions, *limit),
            },
            Statement::Explain {
                prompt,
                mode,
                layers,
                band,
                verbose,
                top,
                relations_only,
                with_attention,
            } => match mode {
                ExplainMode::Walk => self.exec_explain(prompt, layers.as_ref(), *verbose),
                ExplainMode::Infer => {
                    self.exec_infer_trace(prompt, *top, *band, *relations_only, *with_attention)
                }
            },
            Statement::ShowRelations {
                layer,
                with_examples,
                mode,
            } => self.exec_show_relations(*layer, *with_examples, *mode),
            Statement::ShowLayers { range } => self.exec_show_layers(range.as_ref()),
            Statement::ShowFeatures {
                layer,
                conditions,
                limit,
            } => self.exec_show_features(*layer, conditions, *limit),
            Statement::ShowEntities { layer, limit } => self.exec_show_entities(*layer, *limit),
            Statement::ShowModels => self.exec_show_models(),
            Statement::ShowComponents => self.exec_show_components(),
            Statement::ShowRepresentations { object } => {
                self.exec_show_representations(object.as_deref())
            }
            Statement::ShowProvenance { object } => self.exec_show_provenance(object.as_deref()),
            Statement::ShowAuthority => self.exec_show_authority(),
            Statement::ShowCompactStatus => self.exec_show_compact_status(),
            Statement::CompactInto { output } => self.exec_compact_into(output),
            Statement::CompactMinor => self.exec_compact_minor(),
            Statement::CompactMajor { full, lambda } => self.exec_compact_major(*full, *lambda),
            Statement::Extract {
                model,
                output,
                components,
                layers,
                extract_level,
                format,
            } => self.exec_extract(
                model,
                output,
                components.as_deref(),
                layers.as_ref(),
                *extract_level,
                *format,
            ),
            Statement::Compile {
                vindex,
                output,
                format,
                target,
                on_conflict,
            } => self.exec_compile(vindex, output, *format, *target, *on_conflict),
            Statement::Diff {
                a,
                b,
                layer,
                relation,
                limit,
                into_patch,
                physical,
            } => self.exec_diff(
                a,
                b,
                *layer,
                relation.as_deref(),
                *limit,
                into_patch.as_deref(),
                *physical,
            ),
            Statement::Insert {
                entity,
                relation,
                target,
                layer,
                confidence,
                alpha,
                mode,
            } => {
                let mut out = self.ensure_patch_session();
                out.extend(self.exec_insert(
                    entity,
                    relation,
                    target,
                    *layer,
                    *confidence,
                    *alpha,
                    *mode,
                )?);
                self.advance_epoch();
                Ok(out)
            }
            Statement::Infer {
                prompt,
                top,
                compare,
                route,
                generate,
            } => self.exec_infer(prompt, *top, *compare, route.as_ref(), *generate),
            Statement::Delete { conditions } => {
                let mut out = self.ensure_patch_session();
                out.extend(self.exec_delete(conditions)?);
                self.advance_epoch();
                Ok(out)
            }
            Statement::Update { set, conditions } => {
                let mut out = self.ensure_patch_session();
                out.extend(self.exec_update(set, conditions)?);
                self.advance_epoch();
                Ok(out)
            }
            // A direct weight edit stages without starting an anonymous patch:
            // it is recorded only when a named BEGIN PATCH recording is open.
            Statement::UpdateWeights {
                tensor,
                row,
                cols,
                delta,
            } => self.exec_update_weights(tensor, *row, cols, *delta),
            Statement::Merge {
                source,
                target,
                conflict,
            } => self.exec_merge(source, target.as_deref(), *conflict),
            Statement::Rebalance {
                max_iters,
                floor,
                ceiling,
            } => self.exec_rebalance(*max_iters, *floor, *ceiling),

            // ── Patch commands ──
            Statement::BeginPatch { path } => self.exec_begin_patch(path),
            Statement::SavePatch => self.exec_save_patch(),
            Statement::ApplyPatch { path } => self.exec_apply_patch(path),
            Statement::ShowPatches => self.exec_show_patches(),
            Statement::RemovePatch { path } => self.exec_remove_patch(path),
            // ── Trace commands ──
            Statement::Trace {
                prompt,
                answer,
                decompose,
                layers,
                positions,
                save,
            } => self.exec_trace(
                prompt,
                answer.as_deref(),
                *decompose,
                layers.as_ref(),
                *positions,
                save.as_deref(),
            ),
        }
    }

    /// Execute a statement against a remote backend.
    fn execute_remote(&mut self, stmt: &Statement) -> Result<Vec<String>, LqlError> {
        match stmt {
            Statement::Use { target } => self.exec_use(target),
            Statement::Describe { entity, band, mode, .. } => {
                self.remote_describe(entity, *band, *mode)
            }
            Statement::Walk { prompt, top, layers, .. } => {
                self.remote_walk(prompt, *top, layers.as_ref())
            }
            Statement::Infer { prompt, top, compare, route: _, generate } => {
                if generate.is_some() {
                    return Err(LqlError::Execution(
                        "INFER GENERATE is not supported on a remote backend yet".into(),
                    ));
                }
                self.remote_infer(prompt, *top, *compare)
            }
            Statement::Stats { .. } => self.remote_stats(),
            Statement::ShowRelations { mode, with_examples, .. } => self.remote_show_relations(*mode, *with_examples),
            Statement::Insert { entity, relation, target, layer, confidence, alpha: _, mode: _ } => {
                // Remote backend doesn't forward ALPHA or MODE — the
                // HTTP protocol doesn't have a schema for them yet.
                // Local backend honours both via `exec_insert`.
                self.remote_insert(entity, relation, target, *layer, *confidence)
            }
            Statement::Delete { conditions } => self.remote_delete(conditions),
            Statement::Update { set, conditions } => self.remote_update(set, conditions),
            Statement::Select { source: _, fields: _, conditions, nearest: _, order: _, limit } => {
                self.remote_select(conditions, *limit)
            }
            Statement::Explain { prompt, mode, layers, band, verbose: _, top, relations_only, with_attention } => {
                match mode {
                    ExplainMode::Infer => self.remote_explain_infer(prompt, *top, *band, *relations_only, *with_attention),
                    ExplainMode::Walk => self.remote_walk(prompt, *top, layers.as_ref()),
                }
            }
            Statement::ApplyPatch { path } => self.remote_apply_local_patch(path),
            Statement::ShowPatches => self.remote_show_patches(),
            Statement::RemovePatch { path } => self.remote_remove_local_patch(path),
            Statement::Pipe { left, right } => {
                let mut out = self.execute(left)?;
                out.extend(self.execute(right)?);
                Ok(out)
            }
            _ => Err(LqlError::Execution(
                "this statement is not supported on a remote backend. \
                 Supported: DESCRIBE, WALK, INFER, EXPLAIN INFER, EXPLAIN WALK, SELECT, STATS, \
                 SHOW RELATIONS, INSERT, DELETE, UPDATE, APPLY PATCH, SHOW PATCHES, REMOVE PATCH, USE. \
                 TRACE requires a local vindex (USE \"path.vindex\")."
                    .into(),
            )),
        }
    }

    // ── Patch execution ──

    fn exec_begin_patch(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
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

    fn exec_save_patch(&mut self) -> Result<Vec<String>, LqlError> {
        let recording = self.patch_recording.take().ok_or_else(|| {
            LqlError::Execution("no active patch session. Run BEGIN PATCH first.".into())
        })?;

        if recording.path.is_empty() {
            return Err(LqlError::Execution(
                "anonymous patch session — use SAVE PATCH \"filename.vlp\" or BEGIN PATCH \"filename.vlp\" first.".into(),
            ));
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
        patch
            .save(&path)
            .map_err(|e| LqlError::exec("failed to save patch", e))?;

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

    fn exec_apply_patch(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
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
        let weight_ops: Vec<&larql_vindex::format::vindex3::weight_patch::WeightDeltaOp> = patch
            .operations
            .iter()
            .filter_map(|op| match op {
                larql_vindex::PatchOp::WeightDelta(w) => Some(w),
                _ => None,
            })
            .collect();
        let weight_edits: usize = weight_ops.iter().map(|w| w.edits.len()).sum();
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

        // Apply through the session overlay (base files untouched).
        match &mut self.backend {
            Backend::Vindex { patched, .. } => {
                patched.apply_patch(patch);
            }
            // The V3 overlay is all-or-nothing and *reports* an
            // inapplicable patch (corrupt vector, or feature-slot ops
            // awaiting the compose rung) instead of silently dropping
            // it.
            Backend::Vindex3 { overlay, .. } => {
                overlay
                    .try_apply_patch(patch)
                    .map_err(|e| LqlError::exec("failed to apply patch", e))?;
            }
            _ => return Err(LqlError::NoBackend),
        }

        let weights = if weight_edits > 0 {
            format!("; {weight_edits} stored-weight edit(s), bound to this container and pending COMPILE")
        } else {
            String::new()
        };
        Ok(vec![format!(
            "Applied: {path} ({total} operations: {ins} inserts, {upd} updates, {del} deletes{weights})"
        )])
    }

    fn exec_show_patches(&self) -> Result<Vec<String>, LqlError> {
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

    fn exec_remove_patch(&mut self, path: &str) -> Result<Vec<String>, LqlError> {
        let by_name = |p: &larql_vindex::VindexPatch| p.description.as_deref() == Some(path);
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
                    overlay.remove_patch(i);
                    Some(i)
                }
                None => None,
            },
            _ => return Err(LqlError::NoBackend),
        };
        match pos {
            Some(i) => Ok(vec![format!("Removed patch #{}", i + 1)]),
            None => Err(LqlError::Execution(format!("patch not found: {path}"))),
        }
    }

    /// Bump the LSM epoch + minor/major mutation counters. Called after
    /// every INSERT/DELETE/UPDATE.
    pub(crate) fn advance_epoch(&mut self) {
        self.epoch += 1;
        self.mutations_since_minor += 1;
        self.mutations_since_major += 1;
    }
}
