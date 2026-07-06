//! `INFER` — full forward pass with attention. Requires model weights.

use crate::error::LqlError;
use crate::executor::helpers::format_knn_override_summary;
use crate::executor::{Backend, Session};

impl Session {
    pub(crate) fn exec_infer(
        &mut self,
        prompt: &str,
        top: Option<u32>,
        compare: bool,
        route: Option<&crate::ast::InferRoute>,
    ) -> Result<Vec<String>, LqlError> {
        let top_k = top.unwrap_or(5) as usize;
        // Resolve the KnnStore router: an explicit `ROUTE VERIFY [FALLBACK]
        // [TOPK n]` clause wins; otherwise inherit the `LARQL_KNN_*` env default
        // (so unclaused INFER stays byte-identical to the Python binding —
        // ADR 0001).
        let route_mode = match route {
            Some(r) => {
                let k = r
                    .topk
                    .map(|k| k as usize)
                    .unwrap_or(larql_inference::forward::KNN_VERIFY_TOPK);
                let thr = larql_inference::forward::KNN_COSINE_THRESHOLD;
                if r.fallback {
                    larql_inference::KnnRouteMode::TwoTier { k, threshold: thr }
                } else {
                    larql_inference::KnnRouteMode::Verified { k, threshold: thr }
                }
            }
            None => larql_inference::KnnRouteMode::from_env(),
        };

        // Retrieval-augmented early exit (FR): opt-in via `ROUTE VERIFY … EXIT`
        // or `LARQL_KNN_EARLY_EXIT`. Verified-only — the FR2 fallback needs the
        // full forward to stay parity-identical, so `FALLBACK` disables it. Only
        // takes effect below when `route_mode` actually resolves to `Verified`.
        let exit_requested = match route {
            Some(r) => r.exit && !r.fallback,
            None => std::env::var_os("LARQL_KNN_EARLY_EXIT").is_some(),
        };

        // Weight backend: dense inference (no vindex needed)
        if let Backend::Weight {
            weights, tokenizer, ..
        } = &self.backend
        {
            let token_ids = super::encode_dense_prompt(weights, tokenizer, prompt)?;

            let start = std::time::Instant::now();
            let result = larql_inference::predict(weights, tokenizer, &token_ids, top_k);
            let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

            let mut out = Vec::new();
            out.push("Predictions (dense — no vindex):".into());
            for (i, (tok, prob)) in result.predictions.iter().enumerate() {
                out.push(format!("  {:2}. {:20} ({:.2}%)", i + 1, tok, prob * 100.0));
            }
            out.push(format!("  {:.0}ms", elapsed_ms));
            if !compare {
                out.push(String::new());
                out.push(
                    "Tip: EXTRACT into a vindex for walk FFN (sparse, faster, editable).".into(),
                );
            }
            return Ok(out);
        }

        // Vindex backend: walk FFN with optional dense comparison
        let (path, config, patched) = self.require_vindex()?;

        if !config.has_model_weights {
            return Err(LqlError::Execution(format!(
                "INFER requires model weights. This vindex was built without --include-weights.\n\
                 Rebuild: EXTRACT MODEL \"{}\" INTO \"{}\" WITH INFERENCE",
                config.model,
                path.display(),
            )));
        }

        let mut cb = larql_vindex::SilentLoadCallbacks;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

        let token_ids = super::encode_vindex_prompt(config, &tokenizer, prompt)?;

        // Shared INFER pipeline — walk FFN (unlimited features) plus KnnStore
        // side-channel override. Same code path as `PyVindex::infer`; see ADR
        // 0001 (docs/adr/0001-python-lql-infer-parity.md).
        let mut iw = larql_inference::InferenceWeights::load(path, config, &mut cb)
            .map_err(|e| LqlError::exec("failed to load model weights", e))?;
        // Early-exit applies only to the verified router (parity-safe). Any
        // other resolved mode (Legacy / TwoTier) runs the full forward.
        let (infer, exited) = match (exit_requested, &route_mode) {
            (true, larql_inference::KnnRouteMode::Verified { k, threshold }) => iw
                .infer_patched_early_exit(
                    &tokenizer,
                    patched,
                    Some(&patched.knn_store),
                    &token_ids,
                    top_k,
                    *k,
                    *threshold,
                ),
            _ => (
                iw.infer_patched(
                    &tokenizer,
                    patched,
                    Some(&patched.knn_store),
                    &token_ids,
                    top_k,
                    &route_mode,
                ),
                false,
            ),
        };

        let trace_layers = larql_inference::walk_trace_from_residuals(&infer.residuals, patched);

        let mut out = Vec::new();
        out.push("Predictions (walk FFN):".into());
        if let Some(ovr) = &infer.knn_override {
            out.push(format!(
                "   1. {:20} (100.00%, {})",
                ovr.token,
                format_knn_override_summary(ovr, infer.model_top1.as_ref()),
            ));
            for (i, (tok, prob)) in infer.predictions.iter().skip(1).enumerate() {
                out.push(format!("  {:2}. {:20} ({:.2}%)", i + 2, tok, prob * 100.0));
            }
        } else {
            for (i, (tok, prob)) in infer.predictions.iter().enumerate() {
                out.push(format!("  {:2}. {:20} ({:.2}%)", i + 1, tok, prob * 100.0));
            }
        }
        out.push(format!("  {:.0}ms", infer.walk_ms));
        if infer.knn_override.is_some() {
            if exited {
                out.push(
                    "  note: early-exit — verified hit fired at the resolved layer; the stored \
                     target was emitted and the remaining layers + lm_head were skipped."
                        .into(),
                );
            } else {
                out.push(
                    "  note: KNN override is a post-logits retrieval sidecar, not an FFN/residual edit."
                        .into(),
                );
            }
        }

        out.push(String::new());
        out.push("Inference trace (features that fired with attention):".into());
        let classifier = self.relation_classifier();
        for (layer, hits) in &trace_layers {
            if hits.is_empty() {
                continue;
            }
            for hit in hits.iter().take(3) {
                let label = classifier
                    .and_then(|rc| rc.label_for_feature(*layer, hit.feature))
                    .unwrap_or("");
                let label_str = if label.is_empty() {
                    String::new()
                } else {
                    format!("{:<14}", label)
                };
                let top_token = hit.meta.top_token.trim();
                let down_top: String = hit
                    .meta
                    .top_k
                    .iter()
                    .take(3)
                    .map(|t| t.token.clone())
                    .collect::<Vec<_>>()
                    .join(", ");
                out.push(format!(
                    "  L{:2}: {} F{:<5} gate={:+.1}  → {:15} [{}]",
                    layer, label_str, hit.feature, hit.gate_score, top_token, down_top,
                ));
            }
        }

        if compare {
            let start = std::time::Instant::now();
            let dense = iw.predict_dense(&tokenizer, &token_ids, top_k);
            let dense_ms = start.elapsed().as_secs_f64() * 1000.0;

            out.push(String::new());
            out.push("Predictions (dense):".into());
            for (i, (tok, prob)) in dense.predictions.iter().enumerate() {
                out.push(format!("  {:2}. {:20} ({:.2}%)", i + 1, tok, prob * 100.0));
            }
            out.push(format!("  {:.0}ms", dense_ms));
        }

        Ok(out)
    }
}
