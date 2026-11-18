//! `INSERT INTO EDGES ... MODE KNN` — Architecture B retrieval override.
//!
//! Captures the model's residual at the install layer for the canonical
//! prompt and stores it as a KNN key alongside the target token. INFER
//! checks the KnnStore at `cos > 0.75` and overrides the model's
//! prediction when a match fires.
//!
//! Scales freely (N facts store as N independent entries; no cross-fact
//! interference). Doesn't participate in the forward pass — the fact
//! isn't woven into the FFN features, it's a lookup-table entry that
//! intercepts the output. For chaining, multi-hop, or "the FFN is the
//! graph" integration, use `InsertMode::Compose` instead.
//!
//! Validated at 25K edges, 87 edges/s, 100% same-prompt retrieval.

use crate::error::LqlError;
use crate::executor::{Backend, Session};

/// Default `c_score` for a KNN insert without an explicit CONFIDENCE
/// clause — retrieval entries are exact, so full confidence.
pub(crate) const DEFAULT_KNN_CONFIDENCE: f32 = 1.0;

/// The canonical prompt whose residual becomes the retrieval key —
/// shared verbatim by every backend so the same INSERT stores the same
/// logical fact everywhere.
pub(crate) fn knn_canonical_prompt(entity: &str, relation: &str) -> String {
    let rel_words = relation.replace(['-', '_'], " ");
    format!("The {rel_words} of {entity} is")
}

impl Session {
    /// Phase 3 of the KNN insert — the shared logical operation:
    /// store the captured key in the bound backend's KnnStore, record
    /// the patch op, and report. Both the V2 and V3 arms end here.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn knn_finalize(
        &mut self,
        install_layer: usize,
        key: Vec<f32>,
        target_id: u32,
        entity: &str,
        relation: &str,
        target: &str,
        c_score: f32,
        mode_note: &str,
    ) -> Result<Vec<String>, LqlError> {
        let key_b64 = larql_vindex::patch::core::encode_gate_vector(&key);
        let total = {
            let store = match &mut self.backend {
                Backend::Vindex { patched, .. } => &mut patched.knn_store,
                Backend::Vindex3 { overlay, .. } => &mut overlay.knn_store,
                _ => return Err(LqlError::NoBackend),
            };
            store.add(
                install_layer,
                key,
                target_id,
                target.to_string(),
                entity.to_string(),
                relation.to_string(),
                c_score,
            );
            store.len()
        };

        let patch_op = larql_vindex::PatchOp::InsertKnn {
            layer: install_layer,
            entity: entity.to_string(),
            relation: relation.to_string(),
            target: target.to_string(),
            target_id,
            confidence: Some(c_score),
            key_vector_b64: key_b64,
        };
        self.record_knowledge_op(patch_op);

        Ok(vec![
            format!(
                "Inserted: {} —[{}]→ {} at L{} (KNN store)",
                entity, relation, target, install_layer,
            ),
            format!("  mode: {mode_note}"),
            format!("  KNN store: {total} entries total"),
        ])
    }

    pub(crate) fn exec_insert_knn(
        &mut self,
        entity: &str,
        relation: &str,
        target: &str,
        layer_hint: Option<u32>,
        confidence: Option<f32>,
    ) -> Result<Vec<String>, LqlError> {
        if matches!(self.backend, Backend::Vindex3 { .. }) {
            return self.exec_insert_knn_v3(entity, relation, target, layer_hint, confidence);
        }
        // ── Phase 1: Read config, determine install layer ──
        let (install_layer, has_weights);
        {
            let (_path, config, _patched) = self.require_vindex()?;
            let bands = config
                .layer_bands
                .clone()
                .or_else(|| larql_vindex::LayerBands::for_family(&config.family, config.num_layers))
                .unwrap_or(larql_vindex::LayerBands {
                    syntax: (0, config.num_layers.saturating_sub(1)),
                    knowledge: (0, config.num_layers.saturating_sub(1)),
                    output: (0, config.num_layers.saturating_sub(1)),
                });
            install_layer = if let Some(l) = layer_hint {
                (l as usize).min(config.num_layers.saturating_sub(1))
            } else {
                bands
                    .knowledge
                    .1
                    .saturating_sub(1)
                    .min(config.num_layers.saturating_sub(1))
            };
            has_weights = config.has_model_weights;
        }

        // ── Phase 2: Capture residual via forward pass ──
        let residual_key: Vec<f32>;
        let target_id: u32;
        if has_weights {
            let (path, config, patched) = self.require_vindex()?;
            let mut cb = larql_vindex::SilentLoadCallbacks;
            let tokenizer = larql_vindex::load_vindex_tokenizer(path)
                .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

            let spaced_target = format!(" {target}");
            let target_encoding = tokenizer
                .encode(spaced_target.as_str(), false)
                .map_err(|e| LqlError::exec("tokenize error", e))?;
            target_id = target_encoding.get_ids().first().copied().unwrap_or(0);

            let prompt = knn_canonical_prompt(entity, relation);
            let token_ids =
                crate::executor::query::encode_vindex_prompt(config, &tokenizer, prompt.as_str())?;

            // `InferenceWeights::load` branches on `config.quant` — callers
            // do not need to know the on-disk format.
            let mut iw = larql_inference::InferenceWeights::load(path, config, &mut cb)
                .map_err(|e| LqlError::exec("failed to load model weights", e))?;
            // Install only needs the residuals (knn_store=None → no override
            // fires); the route mode is irrelevant here.
            let residuals = iw
                .infer_patched(
                    &tokenizer,
                    patched,
                    None,
                    &token_ids,
                    1,
                    &larql_inference::KnnRouteMode::Legacy,
                )
                .residuals;

            residual_key = residuals
                .into_iter()
                .find(|(l, _)| *l == install_layer)
                .map(|(_, r)| r)
                .ok_or_else(|| {
                    LqlError::Execution(format!("no residual captured at layer {install_layer}"))
                })?;
        } else {
            let (path, _config, _patched) = self.require_vindex()?;
            let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
                .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
            let tokenizer = larql_vindex::load_vindex_tokenizer(path)
                .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;
            let hidden = embed.shape()[1];
            let spaced_target = format!(" {target}");
            let target_encoding = tokenizer
                .encode(spaced_target.as_str(), false)
                .map_err(|e| LqlError::exec("tokenize error", e))?;
            target_id = target_encoding.get_ids().first().copied().unwrap_or(0);

            residual_key = crate::executor::helpers::entity_query_vec(
                &tokenizer,
                &embed,
                embed_scale,
                entity,
            )?
            .map(|a| a.to_vec())
            .unwrap_or_else(|| vec![0.0f32; hidden]);
        }

        // ── Phase 3: the shared logical store-and-record ──
        let c_score = confidence.unwrap_or(DEFAULT_KNN_CONFIDENCE);
        let mode_note = if has_weights {
            "KNN — residual capture (Architecture B, retrieval-override)"
        } else {
            "KNN — embedding key (no model weights)"
        };
        self.knn_finalize(
            install_layer,
            residual_key,
            target_id,
            entity,
            relation,
            target,
            c_score,
            mode_note,
        )
    }
}
