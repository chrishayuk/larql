//! `UPDATE EDGES SET ... WHERE ...` — rewrite feature metadata via the
//! session overlay (V2's `PatchedVindex`, V3's `KnowledgeOverlay`).
//!
//! One logical operation, written once: candidates resolve through the
//! knowledge seam, the current (overlay-merged) metadata is read back
//! through it, and the same `PatchOp::Update` is recorded — so a saved
//! patch replays identically on either backend. A tombstoned slot has
//! no current metadata, so no clause addresses it and UPDATE leaves it
//! deleted.

use crate::ast::{Assignment, Condition};
use crate::error::LqlError;
use crate::executor::{Backend, Session};

use super::update_set::UpdateSet;

impl Session {
    pub(crate) fn exec_update(
        &mut self,
        set: &[Assignment],
        conditions: &[Condition],
    ) -> Result<Vec<String>, LqlError> {
        // Validate both clauses, resolve the targets and build the new
        // metas with a readonly borrow, then apply. The SET side is checked
        // first and whether or not any feature matches: a statement that
        // cannot be applied is refused, not answered "no matching features".
        let set = UpdateSet::parse(set)?;
        let matches = self.resolve_mutation_targets(conditions)?;
        let update_ops: Vec<(usize, usize, larql_vindex::FeatureMeta)> = {
            let ctx = self.browse()?;
            let mut ops = Vec::new();
            for (layer, feature) in matches {
                if let Some(mut meta) = ctx.source.feature_meta(layer, feature) {
                    set.apply(&mut meta);
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
        self.advance_epoch();

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
