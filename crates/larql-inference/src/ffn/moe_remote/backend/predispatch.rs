//! Pre-dispatched remote-MoE forward.

use super::super::error::RemoteMoeError;
use super::super::metrics;
use super::super::multi_layer_wire::{MultiLayerResult, MultiLayerTask, MultiLayerTaskQ8K};
use super::super::router::{rms_norm, MoeRouterWeights};
use larql_compute::cpu::ops::moe::quantize_x_to_q8k;

#[allow(unused_imports)]
use super::*;

impl RemoteMoeBackend {
    /// Pre-dispatch: route ALL layers at once, fire ONE batch call per shard
    /// (parallel), return h2 per layer.
    ///
    /// # Why faster than streaming
    ///
    /// `forward_moe` / `forward_moe_stream` make N sequential round-trips (one
    /// per layer). `forward_moe_predispatch` collapses them into ONE call per
    /// shard regardless of layer count.  The trade-off: each layer's expert
    /// input is computed from `h_post_attn` captured WITHOUT prior layers'
    /// expert contributions (pass-1 approximation), so the returned h2 values
    /// are slightly wrong for layers > 0.  In practice the error is small
    /// enough that the model still produces the correct top-1 token.
    ///
    /// # Usage
    ///
    /// 1. Run Metal with `moe_fn = |l, h| { capture[l] = h.to_vec(); zeros }`.
    /// 2. Call `forward_moe_predispatch(&captures, routers, ...)` — ONE async call.
    /// 3. Run Metal again with `moe_fn = |l, _h| { h2_per_layer[l].clone() }`.
    pub fn forward_moe_predispatch(
        &self,
        // h_post_attn captured per layer in the SKIP_MOE pass
        h_per_layer: &[Vec<f32>],
        // router weights for each layer (same length as h_per_layer)
        routers: &[MoeRouterWeights<'_>],
        norm_offset: f32,
        eps: f32,
    ) -> Result<Vec<Vec<f32>>, RemoteMoeError> {
        let num_layers = h_per_layer.len().min(routers.len());
        if num_layers == 0 {
            return Ok(vec![]);
        }
        let hidden = h_per_layer[0].len();
        let t0 = std::time::Instant::now();

        // Route each layer locally, build one dispatch task per (layer, shard).
        // One task = one call_layer_batch request to the server's
        // /v1/experts/layer-batch endpoint (efficient Q8_K path, weighted sum
        // returned).  This replaces the old call_batch path which hit
        // /v1/expert/batch (legacy per-item f32 path, ~7× slower per expert).
        struct LayerTask {
            layer: usize,
            shard_idx: usize,
            expert_ids: Vec<u32>,
            expert_weights: Vec<f32>,
        }

        let mut tasks: Vec<LayerTask> = Vec::with_capacity(num_layers);
        // h_norm per layer — captured during routing (first return value of route()).
        // Already computed, zero extra cost.  Used to build Q8K-prenormed wire tasks
        // that cut upload 4× vs sending the raw f32 residual.
        let mut h_norm_per_layer: Vec<Option<larql_compute::Q8KActivation>> =
            (0..num_layers).map(|_| None).collect();
        {
            let shards = self.shards.read().unwrap();
            let num_shards = shards.len();
            let all_http = !shards.is_empty() && shards.iter().all(|s| !s.is_grpc());
            for l in 0..num_layers {
                let (h_norm, expert_indices, expert_weights) =
                    routers[l].route(&h_per_layer[l], norm_offset, eps);
                if expert_indices.is_empty() {
                    continue;
                }
                // Capture Q8K-quantised h_norm for the multi-layer fast path.
                if all_http && h_norm.len() % 256 == 0 {
                    h_norm_per_layer[l] = Some(quantize_x_to_q8k(&h_norm));
                }
                let mut shard_ids: Vec<Vec<u32>> = vec![Vec::new(); num_shards];
                let mut shard_wts: Vec<Vec<f32>> = vec![Vec::new(); num_shards];
                for (&eid, &w) in expert_indices.iter().zip(expert_weights.iter()) {
                    // An expert no shard owns is an expert this layer cannot
                    // run — refused, as on the per-layer and streaming paths,
                    // rather than skipped.
                    let si = shards
                        .iter()
                        .position(|s| s.owns_unit(l, eid))
                        .ok_or(RemoteMoeError::NoShard { expert_id: eid })?;
                    shard_ids[si].push(eid as u32);
                    shard_wts[si].push(w);
                }
                for si in 0..num_shards {
                    if !shard_ids[si].is_empty() {
                        tasks.push(LayerTask {
                            layer: l,
                            shard_idx: si,
                            expert_ids: std::mem::take(&mut shard_ids[si]),
                            expert_weights: std::mem::take(&mut shard_wts[si]),
                        });
                    }
                }
            }
        } // shards lock released
        let t_route = t0.elapsed().as_secs_f64() * 1000.0;

        // ── Fast path: one multi-layer request per shard ────────────────────────
        //
        // When all shards are HTTP/UDS, collapse the 30 per-layer calls into
        // one request per shard.  The server processes layers sequentially so
        // rayon runs at full utilisation (no oversubscription), cutting server
        // compute from ~180 ms to ~30 ms and network from 30 × RTT to 1 × RTT.
        {
            let shards_guard = self.shards.read().unwrap();
            // Use `is_grpc()` helper to avoid naming the private UdsState type.
            let all_http = !shards_guard.is_empty() && shards_guard.iter().all(|s| !s.is_grpc());
            drop(shards_guard);

            if all_http {
                // Group tasks by shard — use Q8K if all h_norms were captured,
                // otherwise fall back to f32 residual.
                // Q8K wire: 4× smaller upload (client pre-quantises h_norm).
                // Disable with LARQL_DISABLE_Q8K_WIRE=1 for debugging.
                let q8k_enabled = super::super::runtime::RemoteMoeRuntime::get().q8k_enabled();
                let use_q8k = q8k_enabled
                    && h_norm_per_layer.iter().enumerate().all(|(l, q)| {
                        let has_task = tasks.iter().any(|t| t.layer == l);
                        !has_task || q.is_some()
                    });
                let shards_guard = self.shards.read().unwrap();
                let num_shards = shards_guard.len();
                let shard_results: Vec<(usize, Result<Vec<MultiLayerResult>, RemoteMoeError>)> =
                    if use_q8k {
                        let mut per_shard: Vec<Vec<MultiLayerTaskQ8K>> =
                            (0..num_shards).map(|_| Vec::new()).collect();
                        for task in &tasks {
                            if let Some(q8k) = &h_norm_per_layer[task.layer] {
                                per_shard[task.shard_idx].push(MultiLayerTaskQ8K {
                                    layer: task.layer,
                                    hidden,
                                    qs: q8k.qs.clone(),
                                    d: q8k.d.clone(),
                                    sums: q8k.sums.clone(),
                                    expert_ids: task.expert_ids.clone(),
                                    weights: task.expert_weights.clone(),
                                });
                            }
                        }
                        per_shard
                            .par_iter()
                            .enumerate()
                            .filter(|(si, t)| {
                                if t.is_empty() {
                                    metrics::record_skip(&shards_guard[*si].config.url);
                                    false
                                } else {
                                    true
                                }
                            })
                            .map(|(si, t)| (si, shards_guard[si].call_multi_layer_batch_q8k(t)))
                            .collect()
                    } else {
                        let mut per_shard: Vec<Vec<MultiLayerTask>> =
                            (0..num_shards).map(|_| Vec::new()).collect();
                        for task in &tasks {
                            per_shard[task.shard_idx].push(MultiLayerTask {
                                layer: task.layer,
                                residual: h_per_layer[task.layer].clone(),
                                expert_ids: task.expert_ids.clone(),
                                weights: task.expert_weights.clone(),
                            });
                        }
                        per_shard
                            .par_iter()
                            .enumerate()
                            .filter(|(si, t)| {
                                if t.is_empty() {
                                    metrics::record_skip(&shards_guard[*si].config.url);
                                    false
                                } else {
                                    true
                                }
                            })
                            .map(|(si, t)| (si, shards_guard[si].call_multi_layer_batch(t)))
                            .collect()
                    };
                let shard_urls: Vec<String> =
                    shards_guard.iter().map(|s| s.config.url.clone()).collect();
                drop(shards_guard);

                let t_dispatch = t0.elapsed().as_secs_f64() * 1000.0;
                let mut h2_per_layer: Vec<Vec<f32>> = vec![vec![0.0f32; hidden]; num_layers];
                for (si, result) in shard_results {
                    // A failed shard fails the dispatch: its partial sums are
                    // experts every layer it served would otherwise silently
                    // lose.
                    let shard_err = |e: RemoteMoeError| {
                        let expert_ids: Vec<u32> = tasks
                            .iter()
                            .filter(|t| t.shard_idx == si)
                            .flat_map(|t| t.expert_ids.iter().copied())
                            .collect();
                        let layers: Vec<usize> = tasks
                            .iter()
                            .filter(|t| t.shard_idx == si)
                            .map(|t| t.layer)
                            .collect();
                        e.at_shard(&shard_urls[si], &layers, &expert_ids)
                    };
                    for r in result.map_err(shard_err)? {
                        accumulate_partial(&mut h2_per_layer, r.layer, &r.h2, hidden)
                            .map_err(shard_err)?;
                    }
                }
                let t_accum = t0.elapsed().as_secs_f64() * 1000.0;
                if super::super::runtime::RemoteMoeRuntime::get().verbose {
                    eprintln!(
                        "[predispatch/multi] route={:.1}ms dispatch={:.1}ms accum={:.1}ms  shards={} wire={}",
                        t_route,
                        t_dispatch - t_route,
                        t_accum - t_dispatch,
                        num_shards,
                        if use_q8k { "q8k" } else { "f32" },
                    );
                }
                // Post-experts norm (caller expects it applied).
                for (l, h2) in h2_per_layer.iter_mut().enumerate() {
                    if !routers[l].post_experts_norm.is_empty() {
                        *h2 = rms_norm(h2, routers[l].post_experts_norm, eps, norm_offset);
                    }
                }
                return Ok(h2_per_layer);
            }
        }

        // ── Fallback: 30 parallel per-layer calls (gRPC shards) ─────────────────
        let shards = self.shards.read().unwrap();
        let task_results: Vec<(usize, Result<Vec<f32>, RemoteMoeError>)> = tasks
            .par_iter()
            .map(|task| {
                let shard = &shards[task.shard_idx];
                let result = shard
                    .call_layer_batch(
                        task.layer,
                        &h_per_layer[task.layer],
                        &task.expert_ids,
                        &task.expert_weights,
                    )
                    .map_err(|e| e.at_shard(&shard.config.url, &[task.layer], &task.expert_ids));
                (task.layer, result)
            })
            .collect();
        drop(shards);
        let t_dispatch = t0.elapsed().as_secs_f64() * 1000.0;

        // Accumulate per-layer partial sums.
        let mut h2_per_layer: Vec<Vec<f32>> = vec![vec![0.0f32; hidden]; num_layers];
        for (layer, result) in task_results {
            accumulate_partial(&mut h2_per_layer, layer, &result?, hidden)?;
        }

        let t_accum = t0.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[predispatch] route={:.1}ms dispatch={:.1}ms accum={:.1}ms  tasks={}",
            t_route,
            t_dispatch - t_route,
            t_accum - t_dispatch,
            tasks.len(),
        );

        // Apply post-experts norm per layer.
        for (l, h2) in h2_per_layer.iter_mut().enumerate() {
            if !routers[l].post_experts_norm.is_empty() {
                *h2 = rms_norm(h2, routers[l].post_experts_norm, eps, norm_offset);
            }
        }

        Ok(h2_per_layer)
    }
}

/// Add one shard's partial sum for `layer` into `h2_per_layer`.
///
/// A partial of the wrong width, or for a layer that was not dispatched, is a
/// malformed response — refused, because dropping it would leave that shard's
/// experts out of the layer without a trace.
pub(in crate::ffn::moe_remote) fn accumulate_partial(
    h2_per_layer: &mut [Vec<f32>],
    layer: usize,
    partial: &[f32],
    hidden: usize,
) -> Result<(), RemoteMoeError> {
    let num_layers = h2_per_layer.len();
    let acc = h2_per_layer.get_mut(layer).ok_or_else(|| {
        RemoteMoeError::BadResponse(format!(
            "partial sum for layer {layer}, but only {num_layers} layer(s) were dispatched"
        ))
    })?;
    if partial.len() != hidden {
        return Err(RemoteMoeError::BadResponse(format!(
            "layer {layer} partial sum has {} floats, expected {hidden}",
            partial.len()
        )));
    }
    for (a, &v) in acc.iter_mut().zip(partial) {
        *a += v;
    }
    Ok(())
}
