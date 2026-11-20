//! LQL Executor — dispatches parsed AST statements to backend operations
//!
//! The base vindex is always readonly. All mutations go through a patch overlay.
//! INSERT/DELETE/UPDATE auto-start an anonymous patch session if none is active.

mod backend;
mod compact;
mod extract_hint;
mod facts;
mod helpers;
mod introspection;
mod knowledge;
mod lifecycle;
mod memit_persist;
pub mod model_resolver;
mod mutation;
mod patch_statements;
mod predicate;
pub(crate) mod query;
mod relation_resolver;
mod remote;
mod remove_refusal;
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

/// The typed operations behind the VINDEX3 introspection and weight-edit statements, and what they return.
///
/// Each is a [`Session`] method the statement's executor calls and then formats, so another interface (the
/// Python binding) gets the same values the statement prints or writes without building LQL text or reading
/// console output: [`Session::tensors`] (`SHOW TENSORS`), [`Session::read_weight`] (`EXPORT WEIGHT`),
/// [`Session::capture`] (`CAPTURE`), [`Session::tokenize`] and [`Session::decode`] (the encoders `INFER` and
/// `CAPTURE` use), [`Session::predict`] and [`Session::generate`] (`INFER`), [`Session::stage_low_rank`]
/// (`ADD LOW RANK`), [`Session::stage_weight_delta`] (`UPDATE WEIGHTS`), and [`Session::begin_patch`],
/// [`Session::save_patch`], [`Session::apply_patch`], [`Session::remove_patch`] and [`Session::patches`]
/// (`BEGIN`/`SAVE`/`APPLY`/`REMOVE PATCH`, `SHOW PATCHES`). They run under the session's binding but not
/// through [`Session::execute`], so the capability profile, which judges statements, does not apply to them.
pub mod operations {
    pub use super::patch_statements::{
        AppliedPatch, PatchSummary, PatchesReport, RemovedPatch, SavedPatch,
    };
    pub use super::vindex3::{
        npy_file_bytes, operand_bits_from_values, operand_origin, CaptureGeometry, CaptureSpec,
        Captured, DiscardedWeightEdits, EditedElement, Generated, NpyDescr, OperandValues,
        Prediction, Predictions, StagedLowRank, StagedWeightDelta, WeightRead,
    };
    pub use crate::ast::{CaptureContinuation, CaptureLayers, CapturePositions, CaptureSite};
    pub use larql_vindex::format::vindex3::stored_tensor::{BareName, TensorListing};
    pub use larql_vindex::format::vindex3::weight_patch::structured::OperandBits;
    pub use larql_vindex::format::vindex3::weight_patch::LowRankColumnsOp;
}

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
    /// Whether a VINDEX3 container this session binds may have a representation manufactured for it at load.
    /// See [`Self::set_representation_source`].
    pub(crate) representation_source: RepresentationSource,
}

/// Where a VINDEX3 execution representation may come from, as the session's binding policy.
pub use facts::{AssertedFact, FactDurability, FactRealization, FactStatus};
pub use larql_vindex::format::vindex3::opplan::exec::operands::RepresentationSource;

/// What a session binds a VINDEX3 container under unless told otherwise: the stored weights, with nothing
/// manufactured at load. The same default as `vindex3 exec --backend production` and the CPU server, so that
/// `INFER` in a session computes with the weights the session's edits and `VERIFY` statements are about.
pub const DEFAULT_REPRESENTATION_SOURCE: RepresentationSource = RepresentationSource::Stored;

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
            representation_source: DEFAULT_REPRESENTATION_SOURCE,
        }
    }

    /// Refuses a statement that would bind something new while the open
    /// recording holds unsaved operations (Gate B, B1). Binding again resets
    /// the recording with the rest of the session, so going ahead would lose
    /// work the user has not authorised losing. Checked before the statement
    /// does anything, so a refusal leaves the binding and the recording as
    /// they were. A recording with no operations holds nothing to lose.
    pub(crate) fn refuse_rebind_over_unsaved_recording(&self, verb: &str) -> Result<(), LqlError> {
        let Some(recording) = &self.patch_recording else {
            return Ok(());
        };
        let pending = recording.operations.len();
        if pending == 0 {
            return Ok(());
        }
        let (held_by, keep) = if recording.path.is_empty() {
            (
                "the anonymous recording".to_string(),
                "name it with BEGIN PATCH \"file.vlp\" and run SAVE PATCH",
            )
        } else {
            (
                format!("the recording for {}", recording.path),
                "run SAVE PATCH",
            )
        };
        Err(LqlError::Execution(format!(
            "{verb} refused: {held_by} holds {pending} unsaved operation(s), and binding again \
             would discard them. Nothing was changed. To keep them, {keep}, then run {verb} again."
        )))
    }

    /// After a successful bind (`USE`, or EXTRACT's auto-bind): give every
    /// piece of state that derives from the previous binding its fresh value,
    /// so the session is as one created and bound there (Gate B, B1). Run
    /// only once the new backend is in place: a bind that fails leaves the
    /// previous session as it was. The statement has already refused if the
    /// open recording held anything
    /// ([`Self::refuse_rebind_over_unsaved_recording`]), so the recording
    /// this drops is empty.
    ///
    /// The destructuring is exhaustive, so a field added to `Session` does
    /// not compile until it is classified here.
    pub(crate) fn reset_binding_state(&mut self) {
        debug_assert!(
            self.patch_recording
                .as_ref()
                .is_none_or(|r| r.operations.is_empty()),
            "a bind reached the reset with unsaved operations"
        );
        let Session {
            // Kept: the new binding itself, and the session's own
            // configuration and counts, which no binding derives.
            backend: _,
            profile: _,
            verifier: _,
            contract_violations: _,
            balance_trace: _,
            install_fault: _,
            model_resolver: _,
            representation_source: _,
            // Reset: everything the previous binding's statements built.
            patch_recording,
            auto_patch,
            decoy_residual_cache,
            raw_install_residuals,
            installed_edges,
            relation_resolver,
            epoch,
            mutations_since_minor,
            mutations_since_major,
        } = Session::new();
        self.patch_recording = patch_recording;
        self.auto_patch = auto_patch;
        self.decoy_residual_cache = decoy_residual_cache;
        self.raw_install_residuals = raw_install_residuals;
        self.installed_edges = installed_edges;
        self.relation_resolver = relation_resolver;
        self.epoch = epoch;
        self.mutations_since_minor = mutations_since_minor;
        self.mutations_since_major = mutations_since_major;
    }

    /// Choose where a VINDEX3 container bound by a LATER `USE` may take its execution representation from.
    ///
    /// The default, [`RepresentationSource::Stored`], executes the stored weights: the loader manufactures
    /// nothing. [`RepresentationSource::Auto`] lets the CPU loader quantise a matrix whose stored image does
    /// not fit the host's cache, which is faster on large models and changes the arithmetic, and so the
    /// logits. It is a performance choice the embedding program makes explicitly. A container already bound
    /// keeps the source it was bound under.
    pub fn set_representation_source(&mut self, source: RepresentationSource) {
        self.representation_source = source;
    }

    /// The tensors of the bound VINDEX3 container that were quantised at load so far, by name: the operands
    /// this session does not execute from their stored bytes. `None` without a VINDEX3 binding. Empty under
    /// [`RepresentationSource::Stored`]. Operands load when first needed, so ask after the statement whose
    /// execution is in question.
    pub fn runtime_quantised_tensors(&self) -> Option<Vec<String>> {
        match &self.backend {
            Backend::Vindex3 { runtime, .. } => Some(runtime.store().runtime_quantised_tensors()),
            _ => None,
        }
    }

    /// The source a later `USE` will bind a VINDEX3 container under.
    pub fn representation_source(&self) -> RepresentationSource {
        self.representation_source
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
        vec!["Auto-patch started (run BEGIN PATCH \"file.vlp\" then SAVE PATCH to persist, or edits are lost on exit)".into()]
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
            Statement::VerifyPatch {
                patch,
                contract,
                base,
                container,
            } => self.exec_verify_patch(patch, contract, base, container),
            Statement::VerifyLedger {
                container,
                contract,
                base,
            } => self.exec_verify_ledger(container, contract, base),
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
            Statement::ShowTensors { filter } => self.exec_show_tensors(filter.as_deref()),
            Statement::ExportWeight {
                tensor,
                rows,
                columns,
                output,
                bits,
            } => {
                self.exec_export_weight(tensor, rows.as_deref(), columns.as_deref(), output, *bits)
            }
            Statement::Capture {
                site,
                layers,
                prompt,
                continuation,
                positions,
                output,
            } => self.exec_capture(&crate::executor::vindex3::CaptureRequest {
                spec: crate::executor::vindex3::CaptureSpec {
                    site: *site,
                    layers: layers.as_ref(),
                    prompt,
                    continuation: continuation.as_ref(),
                    positions,
                },
                output,
            }),
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
                // FACT-ID-1: a live key, an id collision or a multi-token
                // KNN target refuses before the session is touched.
                let version = self.check_fact_insert(
                    entity,
                    relation,
                    target,
                    matches!(mode, crate::ast::InsertMode::Knn),
                )?;
                // The recording must be open for the install to record into,
                // but a refused or failed INSERT must not leave one it
                // opened (Gate B, B5).
                let opened = self.patch_recording.is_none();
                let mut out = self.ensure_patch_session();
                match self.exec_insert(entity, relation, target, *layer, *confidence, *alpha, *mode)
                {
                    Ok(lines) => out.extend(lines),
                    Err(e) => {
                        if opened {
                            self.patch_recording = None;
                            self.auto_patch = false;
                        }
                        return Err(e);
                    }
                }
                out.push(format!(
                    "  fact {} (v{version})",
                    larql_vindex::FactKey::new(entity, relation).fact_id()
                ));
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
            // DELETE and UPDATE open the auto-patch themselves, once the
            // WHERE clause is validated and has matched something: a
            // refused or no-op statement leaves the session as it was.
            // (they also advance the epoch themselves, only on a change).
            Statement::Delete { conditions } => self.exec_delete(conditions),
            Statement::Update { set, conditions } => self.exec_update(set, conditions),
            // A direct weight edit stages without starting an anonymous patch:
            // it is recorded only when a named BEGIN PATCH recording is open.
            Statement::UpdateWeights {
                tensor,
                row,
                cols,
                delta,
            } => self.exec_update_weights(tensor, *row, cols, *delta),
            Statement::RemoveDirection {
                operand,
                tensor,
                columns,
                scale,
                numerics,
            } => self.exec_remove_direction(operand, tensor, columns, *scale, numerics),
            Statement::AddLowRank {
                operands,
                tensor,
                columns,
                numerics,
            } => self.exec_add_low_rank(operands, tensor, columns, numerics),
            Statement::DiscardWeightEdits => self.exec_discard_weight_edits(),
            Statement::Merge {
                source,
                target,
                conflict,
            } => self.exec_merge(source, target.as_deref(), *conflict),
            Statement::Rebalance {
                max_iters,
                floor,
                ceiling,
            } => {
                // REBALANCE rescales the down vectors of earlier installs,
                // which may sit in saved patches: record each change as a
                // re-fit so the recording replays to this state (Gate B, B3).
                let prior = self.prior_install_vectors();
                let rebalanced = self.exec_rebalance(*max_iters, *floor, *ceiling)?;
                let refits = self.refits_since(&prior);
                if refits.is_empty() {
                    return Ok(rebalanced);
                }
                let mut out = self.ensure_patch_session();
                out.extend(rebalanced);
                for op in refits {
                    self.record_knowledge_op(op);
                }
                Ok(out)
            }

            // ── Patch commands ──
            Statement::BeginPatch { path } => self.exec_begin_patch(path),
            Statement::SavePatch => self.exec_save_patch(),
            Statement::ApplyPatch { path } => self.exec_apply_patch(path),
            Statement::ShowPatches => self.exec_show_patches(),
            Statement::RemovePatch { path } => self.exec_remove_patch(path),
            Statement::Help { topic } => crate::help::help_lines(topic.as_deref()),
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
            Statement::ApplyPatch { path } => self.remote_apply_patch(path),
            Statement::ShowPatches => self.remote_show_patches(),
            Statement::RemovePatch { path } => self.remote_remove_patch(path),
            Statement::Help { topic } => crate::help::help_lines(topic.as_deref()),
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

    /// Bump the LSM epoch + minor/major mutation counters. Called after
    /// every INSERT/DELETE/UPDATE.
    pub(crate) fn advance_epoch(&mut self) {
        self.epoch += 1;
        self.mutations_since_minor += 1;
        self.mutations_since_major += 1;
    }
}
