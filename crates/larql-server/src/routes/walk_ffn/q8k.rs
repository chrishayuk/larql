//! `POST /v1/walk-ffn-q8k` — Q8K-prenormed dense-FFN batch endpoint.
//!
//! The client has already applied the FFN input norm and quantised
//! the activation to Q8_K. The server decodes each entry, runs the
//! NEON/AVX2 Q4K×Q8K gate+up kernel (or the Metal backend when
//! available), and returns the FFN delta per layer as f32.
//!
//! Returns 404 if the vindex doesn't have interleaved Q4K data
//! (ffn-only servers without Q4K weights can't serve this endpoint).
//!
//! Coverage caveat: this handler requires the model to have
//! `interleaved_kquant_mmap_ref().is_some()` — i.e. an actual
//! Q4K-quantised vindex on disk. The synthetic f32 fixture doesn't
//! satisfy this; the handler is excluded from per-file coverage gating
//! until a Q4K-quantised test fixture lands.

use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::Response;
use larql_vindex::QuantizedFfnAccess;

/// Content-type for the Q8K dense-FFN batch protocol.
pub(crate) const Q8K_BATCH_CT: &str = "application/x-larql-ffn-q8k-batch";

#[utoipa::path(
    post,
    path = "/v1/walk-ffn-q8k",
    tag = "expert",
    request_body(
        content_type = "application/x-larql-ffn-q8k-batch",
        description = "Q8K-prenormed dense-FFN batch: client has applied FFN input norm + Q8 quantisation. \
                       404 if the vindex lacks interleaved Q4K data.",
    ),
    responses(
        (status = 200, content_type = "application/x-larql-ffn-q8k-batch",
         description = "Per-layer FFN delta as f32", body = Vec<u8>),
        (status = 400, body = crate::error::ErrorBody),
        (status = 404, body = crate::error::ErrorBody),
    ),
)]
pub async fn handle_walk_ffn_q8k(
    State(state): State<std::sync::Arc<crate::state::AppState>>,
    request: axum::extract::Request,
) -> Result<Response, crate::error::ServerError> {
    state.bump_requests();

    let body = axum::body::to_bytes(request.into_body(), 64 * 1024 * 1024)
        .await
        .map_err(|e| crate::error::ServerError::BadRequest(format!("read body: {e}")))?;

    let result = tokio::task::spawn_blocking(move || {
        use larql_inference::ffn::remote::{decode_q8k_batch_request, encode_q8k_batch_response};
        use larql_inference::vindex::kquant_ffn_forward_layer_q8k;

        let model = state
            .model(None)
            .ok_or_else(|| crate::error::ServerError::NotFound("no model loaded".into()))?;

        // Require interleaved Q4K to serve this endpoint.
        let has_q4k = {
            let patched = model.patched.blocking_read();
            patched.base().interleaved_kquant_mmap_ref().is_some()
        };
        if !has_q4k {
            return Err(crate::error::ServerError::NotFound(
                "this server does not have interleaved Q4K data — \
                 /v1/walk-ffn-q8k not available"
                    .into(),
            ));
        }

        let entries =
            decode_q8k_batch_request(&body).map_err(crate::error::ServerError::BadRequest)?;

        let patched = model.patched.blocking_read();
        let start = std::time::Instant::now();

        // ── Metal GPU dispatch path ───────────────────────────────────────
        #[cfg(all(feature = "metal-experts", target_os = "macos"))]
        {
            let backend_opt = model
                .metal_backend
                .get_or_init(larql_compute_metal::MetalBackend::new);
            if let Some(backend) = backend_opt.as_ref() {
                // Lazily build per-layer [gate, up, down] Metal buffers from
                // the interleaved Q4K mmap (zero-copy for page-aligned mmap data).
                let layer_bufs = model.metal_ffn_layer_bufs.get_or_init(|| {
                    (0..model.config.num_layers)
                        .filter_map(|l| {
                            let data = patched.base().interleaved_kquant_layer_data(l)?;
                            let gate_buf = backend.bufs().get_bytes(data[0].0);
                            let up_buf = backend.bufs().get_bytes(data[1].0);
                            let down_buf = backend.bufs().get_bytes(data[2].0);
                            Some([gate_buf, up_buf, down_buf])
                        })
                        .collect::<Vec<_>>()
                });

                if layer_bufs.len() == model.config.num_layers {
                    let hidden = model.config.hidden_size;
                    let inter = model.config.intermediate_size;
                    let block = larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
                    let inter_padded = inter.div_ceil(block) * block;

                    let mut response_entries: Vec<(usize, Vec<f32>)> =
                        Vec::with_capacity(entries.len());
                    for entry in &entries {
                        let layer = entry.layer_idx;
                        if layer >= model.config.num_layers {
                            return Err(crate::error::ServerError::BadRequest(format!(
                                "layer {layer} out of range (num_layers = {})",
                                model.config.num_layers
                            )));
                        }
                        if !patched.base().is_layer_owned(layer) {
                            let range_desc = match patched.base().owned_layer_range() {
                                Some((s, e)) => format!("{s}–{}", e - 1),
                                None => "all".into(),
                            };
                            return Err(crate::error::ServerError::BadRequest(format!(
                                "layer {layer} not served by this shard (owned: {range_desc})"
                            )));
                        }

                        let bufs = &layer_bufs[layer];
                        // Decode Q8K → f32: h_norm[b*256 + i] = d[b] * qs[b*256 + i]
                        let n_blocks = entry.q8k.d.len();
                        let mut h_norm = vec![0.0f32; hidden];
                        for b in 0..n_blocks {
                            let d = entry.q8k.d[b];
                            let base = b * 256;
                            for i in 0..256 {
                                h_norm[base + i] = d * (entry.q8k.qs[base + i] as f32);
                            }
                        }

                        let out = backend.run_dense_ffn_q4k(
                            &h_norm,
                            &bufs[0], // gate
                            &bufs[1], // up
                            &bufs[2], // down
                            hidden,
                            inter,
                            inter_padded,
                        );
                        response_entries.push((layer, out));
                    }

                    let _latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    let ref_entries: Vec<(usize, &[f32])> = response_entries
                        .iter()
                        .map(|(l, v)| (*l, v.as_slice()))
                        .collect();
                    let resp_bytes = encode_q8k_batch_response(&ref_entries);
                    if model.release_mmap_after_request {
                        patched.base().release_mmap_pages();
                    }
                    return Ok::<_, crate::error::ServerError>(resp_bytes);
                }
            }
        }

        // ── CPU fallback (NEON Q4K×Q8K) ──────────────────────────────────
        let weights = model
            .get_or_load_weights()
            .map_err(crate::error::ServerError::InferenceUnavailable)?;

        let arch = &*weights.arch;

        use rayon::prelude::*;
        let response_entries: Result<Vec<(usize, Vec<f32>)>, crate::error::ServerError> = entries
            .par_iter()
            .map(|entry| {
                let layer = entry.layer_idx;
                if layer >= model.config.num_layers {
                    return Err(crate::error::ServerError::BadRequest(format!(
                        "layer {layer} out of range (num_layers = {})",
                        model.config.num_layers
                    )));
                }
                if !patched.base().is_layer_owned(layer) {
                    let range_desc = match patched.base().owned_layer_range() {
                        Some((s, e)) => format!("{s}–{}", e - 1),
                        None => "all".into(),
                    };
                    return Err(crate::error::ServerError::BadRequest(format!(
                        "layer {layer} not served by this shard (owned: {range_desc})"
                    )));
                }
                let out = kquant_ffn_forward_layer_q8k(arch, patched.base(), layer, &entry.q8k);
                Ok((layer, out.into_raw_vec_and_offset().0))
            })
            .collect();
        let response_entries = response_entries?;

        let _latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        let ref_entries: Vec<(usize, &[f32])> = response_entries
            .iter()
            .map(|(l, v)| (*l, v.as_slice()))
            .collect();
        let resp_bytes = encode_q8k_batch_response(&ref_entries);

        if model.release_mmap_after_request {
            patched.base().release_mmap_pages();
        }

        Ok::<_, crate::error::ServerError>(resp_bytes)
    })
    .await
    .map_err(|e| crate::error::ServerError::Internal(e.to_string()))??;

    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, Q8K_BATCH_CT)
        .body(axum::body::Body::from(result))
        .unwrap())
}
