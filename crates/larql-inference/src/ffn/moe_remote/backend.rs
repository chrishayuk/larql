use std::sync::{Arc, RwLock};

use rayon::prelude::*;

use super::config::ShardConfig;
use super::error::RemoteMoeError;
use super::metrics;
use super::router::{rms_norm, MoeRouterWeights};
use super::shard::{Shard, ShardTransport};
use super::wire::{ExpertCallItem, ExpertResultItem};

pub(super) mod predispatch;
mod stream;

/// Per-shard call list element: (position, expert_id, residual).
type ShardCallItem = (usize, usize, Vec<f32>);
/// Output of `forward_layer_moe`: (output rows, optional per-expert (logit, weight)).
type LayerMoeResult = (Vec<f32>, Vec<(f32, f32)>);

// ── RemoteMoeBackend ───────────────────────────────────────────────────────

/// Remote MoE expert backend. Thread-safe — all methods take `&self`.
///
/// The shard map is stored behind an `RwLock` so `reshard()` can replace it
/// without interrupting in-flight `forward_moe` calls on other threads.
pub struct RemoteMoeBackend {
    pub(super) shards: Arc<RwLock<Vec<Shard>>>,
}

impl RemoteMoeBackend {
    /// Build with no shards and no health check. Tests only — the backend
    /// will return errors on any actual dispatch attempt.
    #[cfg(test)]
    pub fn new_disconnected() -> Self {
        Self {
            shards: Arc::new(RwLock::new(vec![])),
        }
    }

    /// Build from a shard list. Performs a health check on each shard.
    pub fn connect(configs: Vec<ShardConfig>) -> Result<Self, RemoteMoeError> {
        let shards: Result<Vec<Shard>, _> = configs.into_iter().map(Shard::connect).collect();
        Ok(Self {
            shards: Arc::new(RwLock::new(shards?)),
        })
    }

    /// Replace the shard map live (no model reload, no inference interruption).
    ///
    /// Reconnects to new shards, then atomically swaps the map.
    /// In-flight requests against old shards complete normally.
    pub fn reshard(&self, configs: Vec<ShardConfig>) -> Result<(), RemoteMoeError> {
        let new_shards: Result<Vec<Shard>, _> = configs.into_iter().map(Shard::connect).collect();
        *self.shards.write().unwrap() = new_shards?;
        Ok(())
    }

    /// Returns true if all shards use gRPC transport (`grpc://` URLs).
    /// When true, `open_streams` is available and `forward_moe_stream` can be used.
    pub fn has_grpc_shards(&self) -> bool {
        let shards = self.shards.read().unwrap();
        !shards.is_empty()
            && shards
                .iter()
                .all(|s| matches!(s.transport, ShardTransport::Grpc(_)))
    }

    /// Latency-stats probe: test-call each shard with a zero-length batch and
    /// return `(url, rtt_ms)` per shard. Non-fatal — returns partial results.
    pub fn probe_latency(&self) -> Vec<(String, f64)> {
        let shards = self.shards.read().unwrap();
        shards
            .par_iter()
            .map(|shard| {
                let t = std::time::Instant::now();
                let _ = shard.call_batch(&[]);
                let rtt_ms = t.elapsed().as_secs_f64() * 1000.0;
                (shard.config.url.clone(), rtt_ms)
            })
            .collect()
    }

    /// Run one MoE layer forward pass with experts dispatched remotely.
    ///
    /// Steps:
    ///   1. Router runs locally on `h` using `router`.
    ///   2. Selected experts are grouped by owning shard.
    ///   3. One `POST /v1/expert/batch` per shard (parallel).
    ///   4. Weighted outputs are summed; post-experts norm applied.
    ///
    /// Returns the expert-block contribution (same shape as `h`).
    pub fn forward_moe(
        &self,
        layer: usize,
        h: &[f32],
        router: &MoeRouterWeights<'_>,
        norm_offset: f32,
        eps: f32,
    ) -> Result<Vec<f32>, RemoteMoeError> {
        let hidden = h.len();
        if hidden == 0 || router.num_experts == 0 || router.top_k == 0 {
            return Ok(vec![0.0f32; hidden]);
        }

        // 1. Route locally.
        let (_h_norm, expert_indices, expert_weights) = router.route(h, norm_offset, eps);

        // 2. Build per-shard (expert_id, weight) lists.  The new
        //    layer-batch wire format ships ONE residual per shard plus K
        //    (expert_id, weight) pairs — saves the K-1 redundant residual
        //    copies that the legacy `call_batch` path forced.
        let shards = self.shards.read().unwrap();
        let mut shard_calls: Vec<(usize, Vec<u32>, Vec<f32>)> = (0..shards.len())
            .map(|i| (i, Vec::new(), Vec::new()))
            .collect();

        for (&expert_id, &weight) in expert_indices.iter().zip(expert_weights.iter()) {
            let shard_idx = shards
                .iter()
                .position(|s| s.owns_unit(layer, expert_id))
                .ok_or(RemoteMoeError::NoShard { expert_id })?;
            shard_calls[shard_idx].1.push(expert_id as u32);
            shard_calls[shard_idx].2.push(weight);
        }

        // 3. Parallel dispatch — one layer-batch call per shard that has
        //    work.  Each shard returns its own router-weighted partial sum;
        //    the client just sums shard partials (no per-expert weighting
        //    needed because the server already applied the weights).
        let shard_timing = metrics::shard_timing_enabled();
        let layer_start = std::time::Instant::now();
        let non_empty: Vec<(usize, &Vec<u32>, &Vec<f32>)> = shard_calls
            .iter()
            .filter(|(_, ids, _)| !ids.is_empty())
            .map(|(si, ids, ws)| (*si, ids, ws))
            .collect();
        if metrics::enabled() {
            for (si, ids, _) in &shard_calls {
                if ids.is_empty() {
                    metrics::record_skip(&shards[*si].config.url);
                }
            }
        }

        let results_per_shard: Vec<Result<Vec<f32>, RemoteMoeError>> = non_empty
            .par_iter()
            .map(|(si, ids, ws)| {
                let shard_url = &shards[*si].config.url;
                let issue_us = layer_start.elapsed().as_secs_f64() * 1e6;
                if shard_timing {
                    eprintln!(
                        "[moe-shard-timing] transport=http layer={layer} shard={shard_url} K={} issue_us={issue_us:.0}",
                        ids.len(),
                    );
                }
                let call_start = std::time::Instant::now();
                let result = shards[*si]
                    .call_layer_batch(layer, h, ids, ws)
                    .and_then(|out| {
                        if out.len() == hidden {
                            Ok(out)
                        } else {
                            Err(RemoteMoeError::BadResponse(format!(
                                "shard returned {} floats, expected {hidden}",
                                out.len()
                            )))
                        }
                    })
                    .map_err(|e| e.at_shard(shard_url, &[layer], ids));
                if shard_timing {
                    let done_us = layer_start.elapsed().as_secs_f64() * 1e6;
                    let wall_us = call_start.elapsed().as_secs_f64() * 1e6;
                    eprintln!(
                        "[moe-shard-timing] transport=http layer={layer} shard={shard_url} K={} done_us={done_us:.0} wall_us={wall_us:.0}",
                        ids.len(),
                    );
                }
                result
            })
            .collect();

        // 4. Sum shard partials into the layer's combined expert output.
        let mut out = vec![0.0f32; hidden];
        for result in results_per_shard {
            // Any shard failing fails the layer: a missing partial sum is
            // missing experts, and summing the rest would hide that.
            let shard_out = result?;
            for (acc, &v) in out.iter_mut().zip(shard_out.iter()) {
                *acc += v;
            }
        }

        // 5. Post-experts norm.
        Ok(rms_norm(&out, router.post_experts_norm, eps, norm_offset))
    }

    /// Batch MoE forward for a full sequence of positions in one shot.
    ///
    /// Runs the router on every row of `h`, then issues **one** HTTP batch
    /// call per shard per layer (instead of one call per position). For a
    /// prefill of N positions this reduces dispatch from `N × shards` calls
    /// to `shards` calls — 18× fewer round trips for an 18-token context.
    ///
    /// Results are stitched back into an `[N, hidden]` output array by
    /// sequential index: the server returns items in request order, so we
    /// can match result[i] → request[i] without a position tag in the
    /// wire format.
    pub fn forward_moe_seq(
        &self,
        layer: usize,
        h: &ndarray::Array2<f32>,
        router: &MoeRouterWeights<'_>,
        norm_offset: f32,
        eps: f32,
    ) -> Result<ndarray::Array2<f32>, RemoteMoeError> {
        let seq_len = h.nrows();
        let hidden = h.ncols();
        if hidden == 0 || router.num_experts == 0 || router.top_k == 0 {
            return Ok(ndarray::Array2::zeros((seq_len, hidden)));
        }

        // 1. Route every position locally.
        // routing[pos] = (expert_indices, expert_weights)
        let mut routing: Vec<(Vec<usize>, Vec<f32>)> = Vec::with_capacity(seq_len);
        for pos in 0..seq_len {
            let row: Vec<f32> = h.row(pos).to_vec();
            let (_, idx, wts) = router.route(&row, norm_offset, eps);
            routing.push((idx, wts));
        }

        // 2. Build per-shard call lists preserving (pos, local_idx) so we
        //    can reconstruct the output ordering.
        //    shard_items[si] = Vec<(pos, expert_id, residual)>
        let shards = self.shards.read().unwrap();
        let mut shard_items: Vec<Vec<ShardCallItem>> =
            (0..shards.len()).map(|_| Vec::new()).collect();

        for (pos, route) in routing.iter().enumerate().take(seq_len) {
            let row: Vec<f32> = h.row(pos).to_vec();
            for &expert_id in &route.0 {
                let si = shards
                    .iter()
                    .position(|s| s.owns_unit(layer, expert_id))
                    .ok_or(RemoteMoeError::NoShard { expert_id })?;
                shard_items[si].push((pos, expert_id, row.clone()));
            }
        }

        // 3. One batch call per shard that has work (parallel).
        let non_empty: Vec<(usize, &Vec<ShardCallItem>)> = shard_items
            .iter()
            .enumerate()
            .filter(|(_, items)| !items.is_empty())
            .collect();

        let dispatch_results: Vec<(usize, Result<Vec<ExpertResultItem>, RemoteMoeError>)> =
            non_empty
                .par_iter()
                .map(|(si, items)| {
                    let calls: Vec<ExpertCallItem> = items
                        .iter()
                        .map(|(_, expert_id, residual)| ExpertCallItem {
                            layer,
                            expert_id: *expert_id,
                            residual: residual.clone(),
                        })
                        .collect();
                    let result = shards[*si].call_batch(&calls).map_err(|e| {
                        let expert_ids: Vec<u32> =
                            items.iter().map(|(_, eid, _)| *eid as u32).collect();
                        e.at_shard(&shards[*si].config.url, &[layer], &expert_ids)
                    });
                    (*si, result)
                })
                .collect();

        // 4. Reassemble: for each shard, result[i] corresponds to
        //    shard_items[si][i].  Accumulate weighted sums per position.
        let mut out = ndarray::Array2::<f32>::zeros((seq_len, hidden));

        for (si, result) in dispatch_results {
            let items = &shard_items[si];
            let results = result?;
            if results.len() != items.len() {
                return Err(RemoteMoeError::BadResponse(format!(
                    "shard returned {} results for {} requests at layer {layer}",
                    results.len(),
                    items.len()
                )));
            }
            for ((pos, expert_id, _), item) in items.iter().zip(results.iter()) {
                if item.output.len() != hidden {
                    return Err(RemoteMoeError::BadResponse(format!(
                        "expert {expert_id} at pos {pos} returned {} floats, expected {hidden}",
                        item.output.len()
                    )));
                }
                // Find the weight for this expert at this position.
                let weight = routing[*pos]
                    .0
                    .iter()
                    .zip(routing[*pos].1.iter())
                    .find(|(&eid, _)| eid == *expert_id)
                    .map(|(_, &w)| w)
                    .unwrap_or(0.0);

                let mut row = out.row_mut(*pos);
                for (acc, &val) in row.iter_mut().zip(item.output.iter()) {
                    *acc += weight * val;
                }
            }
        }

        // 5. Post-experts norm per position.
        if !router.post_experts_norm.is_empty() {
            for pos in 0..seq_len {
                let row_vec: Vec<f32> = out.row(pos).to_vec();
                let normed = rms_norm(&row_vec, router.post_experts_norm, eps, norm_offset);
                for (dst, src) in out.row_mut(pos).iter_mut().zip(normed.iter()) {
                    *dst = *src;
                }
            }
        }

        Ok(out)
    }
}

/// The remote route as a [`MoeExpertBackend`].
///
/// Builds its own router from `weights`, which is the one line that moved in
/// from `moe_ffn_block_cpu` when the seam became a trait. A layer with no
/// router returns zeros, matching what the block loop did when
/// `build_moe_router_weights` returned `None` — the behaviour is unchanged,
/// it simply now lives with the route that needs it.
impl crate::ffn::moe_backend::MoeExpertBackend for RemoteMoeBackend {
    fn forward_moe_seq(
        &self,
        weights: &larql_models::ModelWeights,
        layer: usize,
        h: &ndarray::Array2<f32>,
        norm_offset: f32,
        eps: f32,
    ) -> Result<ndarray::Array2<f32>, crate::ffn::moe_backend::MoeBackendError> {
        let arch = &*weights.arch;
        let Some(router) = crate::vindex::build_moe_router_weights(weights, arch, layer) else {
            return Ok(ndarray::Array2::zeros((h.nrows(), h.ncols())));
        };
        Ok(RemoteMoeBackend::forward_moe_seq(
            self,
            layer,
            h,
            &router,
            norm_offset,
            eps,
        )?)
    }

    fn name(&self) -> &'static str {
        "remote"
    }
}
