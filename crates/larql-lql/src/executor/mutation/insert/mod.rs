//! `INSERT INTO EDGES` — Compose (FFN overlay) + Knn (retrieval override)
//! paths.
//!
//! Compose mode runs a five-phase pipeline, each phase in its own file:
//!
//! 1. `plan_install` (plan.rs) — resolve install layer, compute target
//!    embedding.
//! 2. `capture_install_residuals` (capture.rs) — canonical-prompt
//!    forward pass + decoy capture.
//! 3. `install_slots` (compose.rs) — per-layer gate / up / down
//!    synthesis + cliff-breaker refine pass.
//! 4. `balance_installed` (balance.rs) — greedy down_col scaling into
//!    the probability band.
//! 5. `cross_fact_regression_check` (balance.rs) — shrink this install
//!    if it hijacks prior installs.
//!
//! This file is just the orchestrator that wires them together +
//! produces the user-facing output summary.

mod balance;
mod capture;
mod compose;
mod compose_v3;
mod knn;

pub(crate) use compose_v3::probe_target_prob;
mod knn_v3;
mod plan;

use crate::ast::InsertMode;
use crate::error::LqlError;
use crate::executor::tuning::DEFAULT_INSERT_ALPHA_MUL;
use crate::executor::Session;

/// Default `c_score` (confidence) for an INSERT without an explicit
/// `CONFIDENCE` clause. 0.9 lands above the retrieval floor for both
/// KNN and compose-mode installs.
const DEFAULT_INSERT_CONFIDENCE: f32 = 0.9;

impl Session {
    // Arg count mirrors the `Statement::Insert` AST variant 1:1 — each
    // parameter is a distinct AST field destructured by the dispatcher
    // in `executor::execute`. Bundling them into a struct would just
    // push the destructuring onto the caller.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn exec_insert(
        &mut self,
        entity: &str,
        relation: &str,
        target: &str,
        layer_hint: Option<u32>,
        confidence: Option<f32>,
        alpha_override: Option<f32>,
        mode: InsertMode,
    ) -> Result<Vec<String>, LqlError> {
        match mode {
            InsertMode::Knn => {
                return self.exec_insert_knn(entity, relation, target, layer_hint, confidence);
            }
            InsertMode::Compose => { /* fallthrough */ }
        }

        // A compose install re-fits the vectors of earlier installs (the
        // batch refine and the balance passes). Snapshot them first so each
        // one it changes is recorded as a re-fit, and the recording replays
        // to exactly this state (Gate B, B3).
        let prior = self.prior_install_vectors();
        // The V3 compose install: vectors land in the knowledge
        // overlay and reach execution through the operand-source seam.
        let out = if matches!(self.backend, crate::executor::Backend::Vindex3 { .. }) {
            self.exec_insert_compose_v3(
                entity,
                relation,
                target,
                layer_hint,
                confidence,
                alpha_override,
            )
        } else {
            self.exec_insert_compose_v2(
                entity,
                relation,
                target,
                layer_hint,
                confidence,
                alpha_override,
            )
        }?;
        self.record_refits(&prior);
        Ok(out)
    }

    /// The VINDEX2 compose install.
    fn exec_insert_compose_v2(
        &mut self,
        entity: &str,
        relation: &str,
        target: &str,
        layer_hint: Option<u32>,
        confidence: Option<f32>,
        alpha_override: Option<f32>,
    ) -> Result<Vec<String>, LqlError> {
        // ALPHA is the dimensionless multiplier on the layer's median
        // down-vector norm — the actual down vector written into the
        // overlay is `target_embed_unit * d_ref * alpha_mul`. The
        // default lives in `executor::tuning` (DEFAULT_INSERT_ALPHA_MUL).
        let alpha_mul = alpha_override.unwrap_or(DEFAULT_INSERT_ALPHA_MUL);
        let c_score = confidence.unwrap_or(DEFAULT_INSERT_CONFIDENCE);

        // ── Phase 1: plan ──
        let plan = self.plan_install(target, layer_hint)?;

        // ── Phase 1b: capture canonical + decoy residuals ──
        let captured = self.capture_install_residuals(entity, relation, &plan)?;

        // Commit decoys to the session cache now that Phase 1's
        // immutable borrow of `self` has ended. Phase 2's refine pass
        // reads from the cache.
        for (layer, decoys) in captured.pending_decoys {
            self.decoy_residual_cache.insert(layer, decoys);
        }

        // ── Phase 2: install slots ──
        let mut installed = self.install_slots(
            &plan,
            &captured.per_layer,
            alpha_mul,
            c_score,
            entity,
            relation,
            target,
        )?;

        if installed.is_empty() {
            return Err(LqlError::Execution(
                "no free feature slots in target layers".into(),
            ));
        }

        // Commit the new raw residuals to the session cache. Future
        // INSERTs read from `self.raw_install_residuals` to rebuild
        // the full per-layer constellation each time (see the
        // batch-refine block in compose.rs).
        for slot in &installed {
            if let Some(residual) = &slot.raw_residual {
                self.raw_install_residuals
                    .insert((slot.layer, slot.feature), residual.clone());
            }
        }

        // ── Phase 3: balance + cross-fact regression check ──
        if plan.use_constellation {
            self.balance_installed(&installed, entity, relation, target)?;
            self.cross_fact_regression_check(&installed)?;
            self.refresh_installed_patch_ops_from_overlay(&mut installed)?;

            // Register THIS fact for future cross-balance passes.
            let rel_words = relation.replace(['-', '_'], " ");
            let canonical_prompt = format!("The {rel_words} of {entity} is");
            for slot in &installed {
                self.installed_edges.push(crate::executor::InstalledEdge {
                    layer: slot.layer,
                    feature: slot.feature,
                    canonical_prompt: canonical_prompt.clone(),
                    target: target.to_string(),
                    target_id: plan.target_id,
                });
            }
        }

        // ── Phase 4: record patch ops + build output summary ──
        for slot in &installed {
            self.record_knowledge_op(slot.patch_op.clone());
        }

        Ok(format_insert_summary(
            &installed,
            &plan,
            entity,
            relation,
            target,
            layer_hint,
            alpha_override,
            alpha_mul,
        ))
    }
}

impl Session {
    fn refresh_installed_patch_ops_from_overlay(
        &self,
        installed: &mut [compose::InstalledSlot],
    ) -> Result<(), LqlError> {
        if installed.is_empty() {
            return Ok(());
        }

        let (_, _, patched) = self.require_vindex()?;
        for slot in installed {
            if let larql_vindex::PatchOp::Insert {
                gate_vector_b64,
                up_vector_b64,
                down_vector_b64,
                ..
            } = &mut slot.patch_op
            {
                if let Some(gate) = patched.overrides_gate_at(slot.layer, slot.feature) {
                    *gate_vector_b64 = Some(larql_vindex::patch::core::encode_gate_vector(gate));
                }
                if let Some(up) = patched.up_override_at(slot.layer, slot.feature) {
                    *up_vector_b64 = Some(larql_vindex::patch::core::encode_gate_vector(up));
                }
                if let Some(down) = patched.down_override_at(slot.layer, slot.feature) {
                    *down_vector_b64 = Some(larql_vindex::patch::core::encode_gate_vector(down));
                }
            }
        }

        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
fn format_insert_summary(
    installed: &[compose::InstalledSlot],
    plan: &plan::InstallPlan,
    entity: &str,
    relation: &str,
    target: &str,
    layer_hint: Option<u32>,
    alpha_override: Option<f32>,
    alpha_mul: f32,
) -> Vec<String> {
    let mut out = Vec::new();
    let center_note = match layer_hint {
        Some(l) => format!(", centered on L{l}"),
        None => String::new(),
    };
    let inserted_count = installed.len();
    let first_layer = installed.first().map(|s| s.layer);
    let last_layer = installed.last().map(|s| s.layer);
    let layer_span = match (first_layer, last_layer) {
        (Some(lo), Some(hi)) if lo == hi => format!("L{lo}"),
        (Some(lo), Some(hi)) => format!("L{lo}-L{hi} ({} layers)", inserted_count),
        _ => String::from("(no layers)"),
    };
    out.push(format!(
        "Inserted: {} —[{}]→ {} at {}{}",
        entity, relation, target, layer_span, center_note,
    ));
    if plan.use_constellation {
        let alpha_note = if alpha_override.is_some() {
            format!(", alpha_mul={alpha_mul:.3}")
        } else {
            String::new()
        };
        out.push(format!(
            "  mode: constellation (trace-guided gate + up + down{alpha_note}, gate_scale=30, install_compiled_slot, balanced)"
        ));
    } else {
        out.push("  mode: embedding (no model weights — gate only, no down override)".into());
    }
    out
}
