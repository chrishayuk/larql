//! The native-ternary arm of `POST /v1/infer`, for BitNet 1.58
//! `--keep-quant` containers.
//!
//! Skips dense weight loading entirely (~5 GB saved on a 2 B BitNet) and
//! runs `predict_bitnet` against the pre-loaded `BitnetModel`. Walk mode
//! is supported via residual capture + KNN-store override (no sparse FFN
//! — see `larql_inference::ternary::infer_bitnet_walk`).

use crate::band_utils::{INFER_MODE_COMPARE, INFER_MODE_DENSE, INFER_MODE_WALK};
use crate::error::ServerError;
use crate::state::{elapsed_ms, AppState, LoadedModel};

use super::{format_knn_override, format_predictions, infer_mode_flags, round_ms, InferRequest};

/// `mode` value the ternary dense path reports, so a client can tell the
/// native-ternary forward from the dense f32 one.
pub(crate) const BITNET_MODE_LABEL: &str = "bitnet";

/// Resolve the mode a BitNet request runs in.
///
/// A `--dense-only` container has no gate vectors and so no KNN store to
/// walk. When the client left `mode` out, the default (walk) cannot run,
/// so the request is served dense — the answer it would get from any
/// container that can only do dense. When the client *asked* for `walk`
/// or `compare`, serving dense instead would answer a different question
/// (and a `compare` response would come back half-empty), so it is
/// refused with the reason.
pub(crate) fn resolve_bitnet_mode(
    requested: Option<&str>,
    dense_only: bool,
) -> Result<&str, String> {
    match requested {
        None if dense_only => Ok(INFER_MODE_DENSE),
        None => Ok(INFER_MODE_WALK),
        Some(mode) if dense_only && (mode == INFER_MODE_WALK || mode == INFER_MODE_COMPARE) => {
            Err(format!(
                "mode \"{mode}\" needs gate vectors, but this BitNet container was built \
                 --dense-only and carries none; omit `mode` or send \"{INFER_MODE_DENSE}\""
            ))
        }
        Some(mode) => Ok(mode),
    }
}

pub(super) fn run_bitnet_infer(
    state: &AppState,
    model: &LoadedModel,
    req: &InferRequest,
    session_id: Option<&str>,
) -> Result<serde_json::Value, ServerError> {
    let mode = resolve_bitnet_mode(req.mode.as_deref(), model.is_dense_only())
        .map_err(ServerError::BadRequest)?;
    let bitnet_guard = model
        .get_or_load_bitnet()
        .map_err(ServerError::InferenceUnavailable)?;
    let bitnet: &larql_inference::ternary::BitnetModel = &bitnet_guard;

    let encoding = model
        .tokenizer
        .encode(req.prompt.as_str(), true)
        .map_err(|e| ServerError::Internal(format!("tokenize error: {e}")))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    if token_ids.is_empty() {
        return Err(ServerError::BadRequest("empty prompt".into()));
    }

    let start = std::time::Instant::now();
    let (is_compare, use_walk, use_dense) = infer_mode_flags(mode);
    let mut result = serde_json::Map::new();
    result.insert("prompt".into(), serde_json::json!(req.prompt));

    if use_walk {
        // Same view as the dense path (see `view_blocking`).
        let walk_pred = {
            let patched = state.sessions.view_blocking(model, session_id);
            larql_inference::ternary::infer_bitnet_walk(
                bitnet,
                &model.tokenizer,
                Some(&patched.knn_store),
                &token_ids,
                req.top,
            )
        };

        let predictions = format_predictions(&walk_pred.predictions);
        if let Some(ovr) = &walk_pred.knn_override {
            result.insert(
                "knn_override".into(),
                format_knn_override(ovr, walk_pred.model_top1.as_ref()),
            );
        }
        if is_compare {
            result.insert(INFER_MODE_WALK.into(), serde_json::json!(predictions));
            result.insert(
                "walk_ms".into(),
                serde_json::json!(round_ms(walk_pred.walk_ms)),
            );
        } else {
            result.insert("predictions".into(), serde_json::json!(predictions));
            result.insert("mode".into(), serde_json::json!(INFER_MODE_WALK));
        }
    }

    if use_dense {
        let dense_start = std::time::Instant::now();
        let pred =
            larql_inference::ternary::predict_bitnet(bitnet, &model.tokenizer, &token_ids, req.top);
        let dense_ms = elapsed_ms(dense_start);

        let pred_pairs: Vec<(String, f64)> =
            pred.into_iter().map(|p| (p.token, p.probability)).collect();
        let predictions = format_predictions(&pred_pairs);
        if is_compare {
            result.insert(INFER_MODE_DENSE.into(), serde_json::json!(predictions));
            result.insert("dense_ms".into(), serde_json::json!(round_ms(dense_ms)));
        } else {
            result.insert("predictions".into(), serde_json::json!(predictions));
            result.insert("mode".into(), serde_json::json!(BITNET_MODE_LABEL));
        }
    }

    result.insert("latency_ms".into(), serde_json::json!(elapsed_ms(start)));
    Ok(serde_json::Value::Object(result))
}
