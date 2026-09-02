//! `POST /v1/walk-ffn-q8k` — Q8K-prenormed dense-FFN batch endpoint.
//!
//! The client has already applied the FFN input norm and quantised
//! the activation to Q8_K. The server decodes each entry, runs the
//! Q4K×Q8K gate+up kernel (NEON/hand-asm on aarch64, scalar on x86_64
//! today — see `larql_compute::cpu::ops::q4k_q8k_dot::kernel_class_summary`,
//! logged once at server startup) or the Metal backend when available,
//! and returns the FFN delta per layer as f32.
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
use larql_compute::cpu::ops::q4k_q8k_dot::Q8KActivation;
use larql_vindex::QuantizedFfnAccess;

/// Dequantise a Q8_K-quantised activation back to f32:
/// `h[b*256 + i] = d[b] * qs[b*256 + i]`. Shared by the Metal per-entry
/// dispatch and the CPU batched-GEMM path below.
fn q8k_activation_to_f32(q8k: &Q8KActivation, hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; hidden];
    for (b, &d) in q8k.d.iter().enumerate() {
        let base = b * 256;
        for i in 0..256 {
            out[base + i] = d * (q8k.qs[base + i] as f32);
        }
    }
    out
}

/// Content-type for the Q8K dense-FFN batch protocol — single-sourced in
/// the shared walk-ffn codec (`larql_inference::ffn::remote`), same pattern
/// as `crate::http::BINARY_FFN_CONTENT_TYPE`.
pub(crate) use larql_inference::ffn::remote::Q8K_BATCH_CT;

#[utoipa::path(
    post,
    path = "/v1/walk-ffn-q8k",
    tag = "expert",
    request_body(
        content_type = Q8K_BATCH_CT,
        description = "Q8K-prenormed dense-FFN batch: client has applied FFN input norm + Q8 quantisation. \
                       404 if the vindex lacks interleaved Q4K data.",
    ),
    responses(
        (status = 200, content_type = Q8K_BATCH_CT,
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
    // Drain/heartbeat visibility — same guard as `handle_walk_ffn`
    // (ROADMAP hardening item 13: this endpoint was invisible to GT6
    // drain and req_per_sec before).
    let _rif_guard = super::types::track_model_request(&state);

    // Opt-in timing extension (DEC-1A two-scoreboard schema): when the
    // client sends `x-larql-timing: 1`, append the 8-byte serve_us trailer
    // to the response. Without the header the response bytes are
    // byte-identical to the pre-extension wire.
    let timing = larql_inference::ffn::remote::timing_requested(
        request
            .headers()
            .get(larql_inference::ffn::remote::TIMING_HEADER)
            .and_then(|v| v.to_str().ok()),
    );

    let body = axum::body::to_bytes(request.into_body(), crate::http::REQUEST_BODY_LIMIT_BYTES)
        .await
        .map_err(|e| crate::error::ServerError::BadRequest(format!("read body: {e}")))?;

    // serve_us clock: starts once the request body is fully received —
    // upload time belongs to the client's transmit term, not serve.
    let t_serve = std::time::Instant::now();

    let result = tokio::task::spawn_blocking(move || {
        use larql_inference::ffn::remote::{
            append_timing_trailer, decode_q8k_batch_request, encode_q8k_batch_response,
        };
        use larql_inference::vindex::{kquant_ffn_forward_layer, kquant_ffn_forward_layer_q8k};

        let model = state.v2_or_unsupported(None)?;

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

        // Shape validation before any kernel touches the activations: Q8K
        // activations quantise in 256-element super-blocks, so the model's
        // hidden size must be block-aligned (production clients only pick
        // the Q8K wire when it is) and every entry must carry exactly
        // hidden/256 blocks. Without this gate a shape-mismatched frame
        // panics inside the FFN kernels (slice out of range) → 500.
        {
            let hidden = model.config.hidden_size;
            let block = larql_inference::ffn::Q4K_Q8K_SUPERBLOCK_ELEMS;
            if hidden % block != 0 {
                return Err(crate::error::ServerError::BadRequest(format!(
                    "hidden_size {hidden} is not a multiple of {block} — \
                     Q8K wire not supported for this model; use /v1/walk-ffn"
                )));
            }
            let expected_blocks = hidden / block;
            for (i, entry) in entries.iter().enumerate() {
                let n_blocks = entry.q8k.d.len();
                if n_blocks != expected_blocks {
                    return Err(crate::error::ServerError::BadRequest(format!(
                        "entry {i} (layer {}): {n_blocks} Q8K blocks, expected \
                         {expected_blocks} for hidden_size {hidden}",
                        entry.layer_idx
                    )));
                }
            }
        }

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
                        let h_norm = q8k_activation_to_f32(&entry.q8k, hidden);

                        let t_layer = std::time::Instant::now();
                        let out = backend.run_dense_ffn_q4k(
                            &h_norm,
                            &bufs[0], // gate
                            &bufs[1], // up
                            &bufs[2], // down
                            hidden,
                            inter,
                            inter_padded,
                        );
                        model
                            .layer_latency_tracker
                            .record(layer as u32, t_layer.elapsed().as_secs_f32() * 1000.0);
                        response_entries.push((layer, out));
                    }

                    let _latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                    let ref_entries: Vec<(usize, &[f32])> = response_entries
                        .iter()
                        .map(|(l, v)| (*l, v.as_slice()))
                        .collect();
                    let mut resp_bytes = encode_q8k_batch_response(&ref_entries);
                    if timing {
                        append_timing_trailer(
                            &mut resp_bytes,
                            t_serve.elapsed().as_secs_f32() * 1e6,
                        );
                    }
                    if model.release_mmap_after_request {
                        patched.base().release_mmap_pages();
                    }
                    return Ok::<_, crate::error::ServerError>(resp_bytes);
                }
            }
        }

        // ── CPU fallback (Q4K×Q8K) ────────────────────────────────────────
        let weights = model
            .get_or_load_weights()
            .map_err(crate::error::ServerError::InferenceUnavailable)?;

        let arch = &*weights.arch;
        let hidden = model.config.hidden_size;

        // Validate every entry's layer bounds/ownership up front, before
        // any compute — matches the previous per-entry checks.
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
        }

        // Group entries by layer: a batch-B request from the replay/predispatch
        // client sends B entries that share one layer, and firing B independent
        // single-row matvecs re-streams that layer's full Q4K gate/up/down
        // bytes B times (the DEC-0/DEC-1 batch curve becomes ~linear in B by
        // construction — see docs/audits/dec-readiness-review-2026-07-22.md
        // §1a). Same-layer groups of >1 dequantise to f32 and run ONE batched
        // GEMM through `kquant_ffn_forward_layer` (amortising weights across
        // all rows, same as the f32/f16/i8 arm); a singleton group keeps the
        // existing single-row Q4K×Q8K kernel (no batching to fix, and it
        // avoids dequantising gate/up for the common single-token decode
        // case). Different layers still parallelise across groups.
        let mut by_layer: std::collections::BTreeMap<usize, Vec<usize>> =
            std::collections::BTreeMap::new();
        for (i, entry) in entries.iter().enumerate() {
            by_layer.entry(entry.layer_idx).or_default().push(i);
        }

        use rayon::prelude::*;
        let mut outputs: Vec<Option<Vec<f32>>> = vec![None; entries.len()];
        let group_results: Vec<(Vec<usize>, Vec<Vec<f32>>)> = by_layer
            .into_par_iter()
            .map(|(layer, idxs)| {
                let t_layer = std::time::Instant::now();
                let result = if idxs.len() == 1 {
                    let out = kquant_ffn_forward_layer_q8k(
                        arch,
                        patched.base(),
                        layer,
                        &entries[idxs[0]].q8k,
                    );
                    (idxs, vec![out.into_raw_vec_and_offset().0])
                } else {
                    let mut flat = Vec::with_capacity(idxs.len() * hidden);
                    for &i in &idxs {
                        flat.extend_from_slice(&q8k_activation_to_f32(&entries[i].q8k, hidden));
                    }
                    let x =
                        larql_vindex::ndarray::Array2::from_shape_vec((idxs.len(), hidden), flat)
                            .expect("q8k batch shape");
                    let out = kquant_ffn_forward_layer(arch, patched.base(), layer, &x);
                    let rows = out.rows().into_iter().map(|r| r.to_vec()).collect();
                    (idxs, rows)
                };
                model
                    .layer_latency_tracker
                    .record(layer as u32, t_layer.elapsed().as_secs_f32() * 1000.0);
                result
            })
            .collect();
        for (idxs, rows) in group_results {
            for (i, row) in idxs.into_iter().zip(rows) {
                outputs[i] = Some(row);
            }
        }

        let response_entries: Vec<(usize, Vec<f32>)> = entries
            .iter()
            .zip(outputs)
            .map(|(entry, out)| {
                (
                    entry.layer_idx,
                    out.expect("every entry index is assigned exactly once by its layer group"),
                )
            })
            .collect();

        let _latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        let ref_entries: Vec<(usize, &[f32])> = response_entries
            .iter()
            .map(|(l, v)| (*l, v.as_slice()))
            .collect();
        let mut resp_bytes = encode_q8k_batch_response(&ref_entries);
        if timing {
            append_timing_trailer(&mut resp_bytes, t_serve.elapsed().as_secs_f32() * 1e6);
        }

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
