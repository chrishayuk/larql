//! `UPDATE EDGES SET ... WHERE ...` — rewrite feature metadata via the
//! session overlay (V2's `PatchedVindex`, V3's `KnowledgeOverlay`).
//!
//! One logical operation, written once: candidates resolve through the
//! knowledge seam, the current (overlay-merged) metadata is read back
//! through it, and the same `PatchOp::Update` is recorded — so a saved
//! patch replays identically on either backend. An UPDATE on a
//! tombstoned slot resurrects it (the V2 contract).

use crate::ast::{Assignment, Condition, Value};
use crate::error::LqlError;
use crate::executor::{Backend, Session};

/// Apply the SET assignments onto one feature's current metadata.
fn apply_assignments(meta: &mut larql_vindex::FeatureMeta, set: &[Assignment]) {
    for assignment in set {
        match assignment.field.as_str() {
            "target" | "top_token" => {
                if let Value::String(ref s) = assignment.value {
                    meta.top_token = s.clone();
                }
            }
            "confidence" | "c_score" => {
                if let Value::Number(n) = assignment.value {
                    meta.c_score = n as f32;
                } else if let Value::Integer(n) = assignment.value {
                    meta.c_score = n as f32;
                }
            }
            _ => {}
        }
    }
}

impl Session {
    pub(crate) fn exec_update(
        &mut self,
        set: &[Assignment],
        conditions: &[Condition],
    ) -> Result<Vec<String>, LqlError> {
        // Validate the clause, resolve its targets and build the new
        // metas with a readonly borrow, then apply.
        let matches = self.resolve_mutation_targets(conditions)?;
        let update_ops: Vec<(usize, usize, larql_vindex::FeatureMeta)> = {
            let ctx = self.browse()?;
            let mut ops = Vec::new();
            for (layer, feature) in matches {
                if let Some(mut meta) = ctx.source.feature_meta(layer, feature) {
                    apply_assignments(&mut meta, set);
                    ops.push((layer, feature, meta));
                }
            }
            ops
        };

        if update_ops.is_empty() {
            return Ok(vec!["  (no matching features found)".into()]);
        }

        // Only now — validated, resolved, and with something to record —
        // does the statement touch session state.
        let mut out = self.ensure_patch_session();

        match &mut self.backend {
            Backend::Vindex { patched, .. } => {
                for (layer, feature, meta) in &update_ops {
                    patched.update_feature_meta(*layer, *feature, meta.clone());
                }
            }
            Backend::Vindex3 { overlay, .. } => {
                for (layer, feature, meta) in &update_ops {
                    overlay.update_feature_meta(*layer, *feature, meta.clone());
                }
            }
            _ => unreachable!("browse() refused every other backend above"),
        }

        // Record to patch session
        for (layer, feature, meta) in &update_ops {
            if let Some(ref mut recording) = self.patch_recording {
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

        out.push(format!(
            "Updated {} features (patch overlay)",
            update_ops.len()
        ));
        Ok(out)
    }
}
