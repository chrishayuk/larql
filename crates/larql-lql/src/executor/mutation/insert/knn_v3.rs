//! The VINDEX3 arm of `INSERT … MODE KNN` (V3-LQL-3B).
//!
//! Same logical operation as the V2 arm in `knn.rs` — capture the
//! canonical prompt's residual at the install layer, store it as an
//! entity-keyed retrieval entry — with the capture supplied by the V3
//! runtime's own execution: `execute_streaming` runs the one canonical
//! traversal and this arm subscribes to its per-layer taps. No
//! `ModelWeights` is loaded and no V2 file is read; the key comes from
//! the same arithmetic INFER runs, which is what makes same-prompt
//! retrieval exact.
//!
//! The stored entry lands in the session's [`KnowledgeOverlay`] and is
//! immediately visible to DESCRIBE (L0 section) and to INFER's
//! post-logits override — the same observability contract as V2's
//! `PatchedVindex`.
//!
//! [`KnowledgeOverlay`]: larql_vindex::format::vindex3::knowledge::KnowledgeOverlay

use super::knn::{knn_canonical_prompt, DEFAULT_KNN_CONFIDENCE};
use crate::error::LqlError;
use crate::executor::{Backend, Session};

impl Session {
    pub(crate) fn exec_insert_knn_v3(
        &mut self,
        entity: &str,
        relation: &str,
        target: &str,
        layer_hint: Option<u32>,
        confidence: Option<f32>,
    ) -> Result<Vec<String>, LqlError> {
        let bos = self.v3_bos_token();
        let (install_layer, key, target_id);
        {
            let Backend::Vindex3 {
                runtime,
                tokenizer,
                overlay,
                ..
            } = &self.backend
            else {
                unreachable!("caller matched the backend");
            };
            let tokenizer = tokenizer.as_ref().ok_or_else(|| {
                LqlError::Execution(
                    "INSERT needs a tokenizer (the canonical prompt and the target must \
                     tokenize) and this container carries no tokenizer.json"
                        .into(),
                )
            })?;

            // ── Phase 1: install layer from the plan ──
            // V2 defaults to `knowledge.hi − 1`; the V3 plan declares
            // no band semantics yet (all bands span the stack), so the
            // same formula resolves to the penultimate layer.
            let num_layers = runtime.plan().layers.len();
            install_layer = match layer_hint {
                Some(l) => (l as usize).min(num_layers.saturating_sub(1)),
                None => num_layers.saturating_sub(2),
            };

            // The V2 contract: the stored target id is the first token
            // of the space-prefixed target surface.
            let spaced_target = format!(" {target}");
            let target_encoding = tokenizer
                .encode(spaced_target.as_str(), false)
                .map_err(|e| LqlError::exec("tokenize error", e))?;
            target_id = target_encoding.get_ids().first().copied().unwrap_or(0);

            // ── Phase 2: capture the residual via the plan's taps,
            // over the same effective program INFER runs ──
            let prompt = knn_canonical_prompt(entity, relation);
            let overrides = crate::executor::vindex3::execution_overrides(runtime, overlay)?;
            key = crate::executor::vindex3::capture_layer_residual(
                runtime,
                tokenizer,
                prompt.as_str(),
                install_layer,
                overrides.as_ref(),
                bos,
            )?;
        }

        // ── Phase 3: the shared logical store-and-record ──
        let c_score = confidence.unwrap_or(DEFAULT_KNN_CONFIDENCE);
        self.knn_finalize(
            install_layer,
            key,
            target_id,
            entity,
            relation,
            target,
            c_score,
            "KNN — residual capture (VINDEX3 plan taps, retrieval-override)",
        )
    }
}
