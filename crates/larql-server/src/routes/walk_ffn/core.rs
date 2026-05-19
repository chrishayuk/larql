//! Core architecture-correct FFN forward pass.
//!
//! [`run_full_output_core`] runs the layer-by-layer FFN (with the L2
//! cache, lazy weight load, Q4K / non-Q4K branching) and the MoE
//! full-layer path (dense FFN + remote expert dispatch + combine +
//! outer norm + optional Gemma-4 layer-scalar). Returns a typed
//! [`FfnOutput`] that both the JSON and binary encoders consume.
//!
//! Coverage caveat: the `moe_layer: true` branch (~L37-220 of this
//! file) requires `model.moe_remote` to be set, which means an actual
//! remote-MoE shard backend — not exercised by the current synthetic
//! fixture. This file is excluded from per-file coverage gating in
//! `coverage-policy.json` until a MoE-shard test fixture lands.

use larql_vindex::PatchOverrides;

use crate::error::ServerError;
use crate::state::LoadedModel;

use super::types::{FfnEntry, FfnOutput, WalkFfnRequest};

/// Architecture-correct FFN forward pass for one or more layers.
/// Returns a typed [`FfnOutput`] used by both JSON and binary encoders.
pub(crate) fn run_full_output_core(
    model: &LoadedModel,
    req: &WalkFfnRequest,
    scan_layers: &[usize],
    start: std::time::Instant,
) -> Result<FfnOutput, ServerError> {
    use larql_inference::ffn::FfnBackend;
    use larql_vindex::ndarray::Array2;

    // MoE full-layer path: server does dense-FFN + remote expert dispatch + combine.
    if req.moe_layer {
        if !req.full_output {
            return Err(ServerError::BadRequest(
                "moe_layer=true requires full_output=true".into(),
            ));
        }
        let moe_remote = model.moe_remote.as_ref().ok_or_else(|| {
            ServerError::BadRequest(
                "moe_layer=true but server has no --moe-shards configured".into(),
            )
        })?;

        let hidden = model.config.hidden_size;
        let seq_len = req.seq_len;
        let x = Array2::from_shape_vec((seq_len, hidden), req.residual.clone())
            .map_err(|e| ServerError::Internal(format!("reshape residual: {e}")))?;

        let weights_guard = model
            .get_or_load_weights()
            .map_err(ServerError::InferenceUnavailable)?;
        let weights: &larql_inference::ModelWeights = &weights_guard;
        let arch = &*weights.arch;
        let patched = model.patched.blocking_read();
        let norm_offset = arch.norm_weight_offset();
        let eps = arch.norm_eps();

        let mut entries = Vec::with_capacity(scan_layers.len());
        for &layer in scan_layers {
            if layer >= model.config.num_layers {
                return Err(ServerError::BadRequest(format!(
                    "layer {layer} out of range (num_layers = {})",
                    model.config.num_layers
                )));
            }

            // Dense FFN via Q4K proxy (reads mmap, no tensor insertion needed).
            struct Q4kProxy<'a> {
                arch: &'a dyn larql_models::ModelArchitecture,
                index: &'a larql_vindex::VectorIndex,
            }
            impl larql_inference::ffn::FfnBackend for Q4kProxy<'_> {
                fn forward(
                    &self,
                    layer: usize,
                    x: &larql_vindex::ndarray::Array2<f32>,
                ) -> larql_vindex::ndarray::Array2<f32> {
                    larql_inference::vindex::kquant_ffn_forward_layer(
                        self.arch, self.index, layer, x,
                    )
                }
                fn forward_with_activation(
                    &self,
                    layer: usize,
                    x: &larql_vindex::ndarray::Array2<f32>,
                ) -> (
                    larql_vindex::ndarray::Array2<f32>,
                    larql_vindex::ndarray::Array2<f32>,
                ) {
                    let o = self.forward(layer, x);
                    (o.clone(), o)
                }
                fn name(&self) -> &str {
                    "q4k-proxy"
                }
            }
            let proxy = Q4kProxy {
                arch,
                index: patched.base(),
            };

            // Run the full FFN forward which returns h_post_ffn (residual already added).
            // We need only the delta: h1 = h_post_ffn - x.
            let (h_post_ffn_dense, _) =
                larql_inference::forward::run_ffn(weights, &x, layer, &proxy, false);
            let h1 = &h_post_ffn_dense - &x;

            // Build router weights from model vectors.
            fn get_vec(
                vectors: &std::collections::HashMap<String, Vec<f32>>,
                k: Option<String>,
            ) -> &[f32] {
                k.and_then(|k| vectors.get(&k))
                    .map(|v| v.as_slice())
                    .unwrap_or(&[])
            }

            let router_proj_key = arch.moe_router_key(layer).ok_or_else(|| {
                ServerError::BadRequest(format!("layer {layer}: no MoE router weights"))
            })?;
            let router_proj = weights
                .vectors
                .get(&router_proj_key)
                .ok_or_else(|| {
                    ServerError::BadRequest(format!("layer {layer}: router_proj not in vectors"))
                })?
                .as_slice();

            let router = larql_inference::ffn::MoeRouterWeights {
                router_proj,
                router_scale: get_vec(&weights.vectors, arch.moe_router_scale_key(layer)),
                router_per_expert_scale: get_vec(
                    &weights.vectors,
                    arch.moe_router_per_expert_scale_key(layer),
                ),
                router_norm: get_vec(&weights.vectors, arch.moe_router_norm_key(layer)),
                router_norm_parameter_free: arch.moe_router_norm_parameter_free(),
                router_input_scalar: arch.moe_router_input_scalar().unwrap_or(1.0),
                pre_experts_norm: get_vec(&weights.vectors, arch.moe_pre_experts_norm_key(layer)),
                post_experts_norm: get_vec(&weights.vectors, arch.moe_post_experts_norm_key(layer)),
                num_experts: arch.num_experts(),
                top_k: arch.num_experts_per_token(),
            };

            // Remote expert dispatch — returns the expert-block contribution
            // (same shape as x).
            let h2 = moe_remote
                .forward_moe_seq(layer, &x, &router, norm_offset, eps)
                .map_err(|e| ServerError::Internal(format!("moe dispatch L{layer}: {e}")))?;

            // Combine: h1 (dense delta) + h2 (expert delta).
            let combined = &h1 + &h2;

            // Outer post-norm + residual combine:
            //   out[pos][i] = x[pos][i] + norm(combined[pos])[i]
            // where norm(c)[i] = c[i] / rms(c) * (outer_w[i] + norm_offset)
            // If no outer norm weight, combined is added directly.
            let outer_w_vec: Option<&Vec<f32>> = if arch.moe_has_combined_output_norm() {
                arch.moe_post_outer_norm_key(layer)
                    .or_else(|| arch.post_feedforward_layernorm_key(layer))
                    .and_then(|k| weights.vectors.get(&k))
            } else {
                None
            };

            let mut out_buf = Array2::<f32>::zeros((seq_len, hidden));
            for pos in 0..seq_len {
                let x_row = x.row(pos);
                let c_row = combined.row(pos);
                let c_slice = c_row.as_slice().expect("contiguous");
                let out_row = if let Some(outer_w) = outer_w_vec {
                    let rms =
                        (c_slice.iter().map(|v| v * v).sum::<f32>() / hidden as f32 + eps).sqrt();
                    x_row
                        .iter()
                        .zip(c_slice.iter())
                        .zip(outer_w.iter())
                        .map(|((&xi, &ci), &wi)| xi + ci / rms * (wi + norm_offset))
                        .collect::<Vec<f32>>()
                } else {
                    x_row
                        .iter()
                        .zip(c_slice.iter())
                        .map(|(&xi, &ci)| xi + ci)
                        .collect::<Vec<f32>>()
                };
                for (dst, src) in out_buf.row_mut(pos).iter_mut().zip(out_row.iter()) {
                    *dst = *src;
                }
            }

            // Layer scalar (Gemma 4 feature — multiply output by a per-layer scalar).
            if let Some(key) = arch.layer_scalar_key(layer) {
                if let Some(scalars) = weights.vectors.get(&key) {
                    if let Some(&s) = scalars.first() {
                        if s != 0.0 && s != 1.0 {
                            out_buf *= s;
                        }
                    }
                }
            }

            entries.push(FfnEntry {
                layer,
                output: out_buf.into_raw_vec_and_offset().0,
            });
        }

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        return Ok(FfnOutput {
            entries,
            seq_len,
            latency_ms,
        });
    }

    let weights = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;

    let patched = model.patched.blocking_read();
    let is_q4k = model.config.quant == larql_vindex::QuantFormat::Q4K;
    let walk_ffn = if is_q4k {
        None
    } else {
        Some(larql_inference::vindex::WalkFfn::new_unlimited(
            &weights, &*patched,
        ))
    };

    let hidden = model.config.hidden_size;
    let seq_len = req.seq_len;
    let x = Array2::from_shape_vec((seq_len, hidden), req.residual.clone())
        .map_err(|e| ServerError::Internal(format!("reshape residual: {e}")))?;

    let use_l2_cache = seq_len == 1;

    let mut entries = Vec::with_capacity(scan_layers.len());
    for &layer in scan_layers {
        if layer >= model.config.num_layers {
            return Err(ServerError::BadRequest(format!(
                "layer {layer} out of range (num_layers = {})",
                model.config.num_layers
            )));
        }

        let l2_key = if use_l2_cache
            && !(*patched).has_overrides_at(layer)
            && req.top_k > 0
            && patched.gate_vectors_at(layer).is_some()
        {
            let x_1d = x.row(0).to_owned();
            let hits = patched.gate_knn(layer, &x_1d, req.top_k);
            let feat_ids: Vec<usize> = hits.iter().map(|(f, _)| *f).collect();
            let key = crate::ffn_l2_cache::FfnL2Cache::key(&feat_ids);
            if let Some(cached) = model.ffn_l2_cache.get(layer, key) {
                entries.push(FfnEntry {
                    layer,
                    output: (*cached).clone(),
                });
                continue;
            }
            Some(key)
        } else {
            None
        };

        let layer_t0 = std::time::Instant::now();
        let out = if let Some(ref wf) = walk_ffn {
            wf.forward(layer, &x)
        } else {
            larql_inference::vindex::kquant_ffn_forward_layer(
                &*weights.arch,
                patched.base(),
                layer,
                &x,
            )
        };
        let layer_ms = layer_t0.elapsed().as_secs_f32() * 1000.0;
        model.layer_latency_tracker.record(layer as u32, layer_ms);

        let output: Vec<f32> = out.into_iter().collect();
        debug_assert_eq!(output.len(), seq_len * hidden);

        if let Some(key) = l2_key {
            model.ffn_l2_cache.insert(layer, key, output.clone());
        }

        entries.push(FfnEntry { layer, output });
    }

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(FfnOutput {
        entries,
        seq_len,
        latency_ms,
    })
}
