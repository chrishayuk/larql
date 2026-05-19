//! Pure-function validators for walk-ffn requests. Split off so the
//! HTTP handler and the gRPC fan-out can share the same correctness
//! checks without dragging in the codec or core FFN computation.

use crate::error::ServerError;
use crate::state::LoadedModel;

use super::types::WalkFfnRequest;

/// Returns the layer indices to scan based on the request shape:
/// `layers` array takes precedence over the scalar `layer`; absence
/// of both is a 400.
pub(crate) fn collect_scan_layers(req: &WalkFfnRequest) -> Result<Vec<usize>, ServerError> {
    if let Some(ref layers) = req.layers {
        Ok(layers.clone())
    } else if let Some(layer) = req.layer {
        Ok(vec![layer])
    } else {
        Err(ServerError::BadRequest(
            "must provide 'layer' or 'layers'".into(),
        ))
    }
}

/// Validates that the request's `residual` length matches the
/// expected `seq_len * hidden_size` (full-output) or `hidden_size`
/// (features-only) — rejecting with a 400 on mismatch. Also rejects
/// `seq_len == 0` in full-output mode.
pub(crate) fn validate_residual(req: &WalkFfnRequest, hidden: usize) -> Result<(), ServerError> {
    let expected_len = if req.full_output {
        req.seq_len
            .checked_mul(hidden)
            .ok_or_else(|| ServerError::BadRequest("seq_len * hidden overflow".into()))?
    } else {
        hidden
    };
    if req.residual.len() != expected_len {
        return Err(ServerError::BadRequest(format!(
            "residual has {} elements, expected {expected_len} (seq_len={} * hidden_size={hidden})",
            req.residual.len(),
            if req.full_output { req.seq_len } else { 1 },
        )));
    }
    if req.full_output && req.seq_len == 0 {
        return Err(ServerError::BadRequest("seq_len must be >= 1".into()));
    }
    Ok(())
}

/// Validates that every layer in `scan_layers` is owned by this
/// shard — sharded deployments split layer ownership and a 400 here
/// guides the router to the right replica.
pub(crate) fn validate_owned(
    model: &LoadedModel,
    scan_layers: &[usize],
) -> Result<(), ServerError> {
    let patched = model.patched.blocking_read();
    let base = patched.base();
    for &layer in scan_layers {
        if !base.is_layer_owned(layer) {
            let range_desc = match base.owned_layer_range() {
                Some((s, e)) => format!("{s}–{}", e - 1),
                None => "all".into(),
            };
            return Err(ServerError::BadRequest(format!(
                "layer {layer} not served by this shard (owned: {range_desc})"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_req(
        layer: Option<usize>,
        layers: Option<Vec<usize>>,
        residual: Vec<f32>,
        seq_len: usize,
        full_output: bool,
    ) -> WalkFfnRequest {
        WalkFfnRequest {
            layer,
            layers,
            residual,
            seq_len,
            top_k: 8,
            full_output,
            moe_layer: false,
        }
    }

    #[test]
    fn collect_scan_layers_prefers_array() {
        let req = make_req(Some(0), Some(vec![1, 2, 3]), vec![0.0; 4], 1, false);
        assert_eq!(collect_scan_layers(&req).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn collect_scan_layers_falls_back_to_scalar() {
        let req = make_req(Some(7), None, vec![0.0; 4], 1, false);
        assert_eq!(collect_scan_layers(&req).unwrap(), vec![7]);
    }

    #[test]
    fn collect_scan_layers_neither_field_errors() {
        let req = make_req(None, None, vec![0.0; 4], 1, false);
        assert!(collect_scan_layers(&req).is_err());
    }

    #[test]
    fn validate_residual_features_only_must_match_hidden() {
        let req = make_req(Some(0), None, vec![0.0; 4], 1, false);
        assert!(validate_residual(&req, 4).is_ok());
        // hidden=8 but residual has 4 elements — must reject.
        let bad = make_req(Some(0), None, vec![0.0; 4], 1, false);
        assert!(validate_residual(&bad, 8).is_err());
    }

    #[test]
    fn validate_residual_full_output_uses_seq_len_x_hidden() {
        let req = make_req(Some(0), None, vec![0.0; 16], 2, true);
        assert!(validate_residual(&req, 8).is_ok());
        let bad = make_req(Some(0), None, vec![0.0; 16], 2, true);
        assert!(
            validate_residual(&bad, 7).is_err(),
            "16 floats with hidden=7 seq_len=2 (expected 14) must reject"
        );
    }

    #[test]
    fn validate_residual_full_output_seq_len_zero_errors() {
        let req = make_req(Some(0), None, Vec::new(), 0, true);
        assert!(validate_residual(&req, 4).is_err());
    }

    #[test]
    fn validate_residual_seq_len_overflow_errors() {
        let req = make_req(Some(0), None, vec![0.0; 4], usize::MAX, true);
        assert!(validate_residual(&req, usize::MAX).is_err());
    }
}
