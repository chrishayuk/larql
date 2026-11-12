//! Streaming remote-MoE dispatch.

use super::super::error::RemoteMoeError;
use super::super::metrics;
use super::super::router::{rms_norm, MoeRouterWeights};
use super::super::stream::{InflightMoe, ShardStream};

#[allow(unused_imports)]
use super::*;

impl RemoteMoeBackend {
    /// Open one gRPC streaming channel per shard for a decode step.
    ///
    /// Returns a `Vec<ShardStream>`, one per shard in the internal shard map.
    /// Each stream stays open until dropped; the caller sends one
    /// `ExpertLayerInput` per MoE layer and receives one `ExpertLayerOutput`.
    ///
    /// Use in `generate_with_remote_moe`:
    ///   ```ignore
    ///   let mut streams = backend.open_streams()?;
    ///   // inside moe_fn for each layer:
    ///   let h2 = backend.forward_moe_stream(layer, h_post_attn, &router, &mut streams, norm_offset, eps)?;
    ///   // streams are dropped (and gRPC streams closed) at end of decode step.
    ///   ```
    pub fn open_streams(&self) -> Result<Vec<ShardStream>, RemoteMoeError> {
        let shards = self.shards.read().unwrap();
        shards.iter().map(|shard| shard.open_stream()).collect()
    }

    /// Run one MoE layer via the already-open per-shard streams.
    ///
    /// Eliminates the per-call connection overhead of `forward_moe` — the
    /// gRPC streams stay alive for the entire decode step (30 layers) so
    /// each layer only pays the cost of sending/receiving one proto frame
    /// over an existing HTTP/2 connection (~0.5ms vs ~12ms per layer).
    pub fn forward_moe_stream(
        &self,
        layer: usize,
        h: &[f32],
        router: &MoeRouterWeights<'_>,
        streams: &mut [ShardStream],
        norm_offset: f32,
        eps: f32,
    ) -> Result<Vec<f32>, RemoteMoeError> {
        let inflight = self.forward_moe_stream_fire(layer, h, router, streams, norm_offset, eps)?;
        self.forward_moe_stream_collect(streams, inflight)
    }

    /// Fire half of `forward_moe_stream`: route locally, push one input per
    /// shard onto its async dispatch task, and return immediately.
    ///
    /// Pair with [`Self::forward_moe_stream_collect`] to retrieve the result.
    /// The [`InflightMoe`] handle carries the post-norm context so the caller
    /// does not need to keep the [`MoeRouterWeights`] borrow alive across the
    /// fire/collect boundary.
    ///
    /// Used by the GPU/MoE overlap path: the metal decode loop fires the MoE
    /// call as soon as `h_post_attn` is ready, encodes dense FFN on a fresh
    /// command buffer, and then collects — letting GPU dense FFN run in
    /// parallel with the remote round trip.
    pub fn forward_moe_stream_fire(
        &self,
        layer: usize,
        h: &[f32],
        router: &MoeRouterWeights<'_>,
        streams: &[ShardStream],
        norm_offset: f32,
        eps: f32,
    ) -> Result<InflightMoe, RemoteMoeError> {
        let hidden = h.len();
        if hidden == 0 || router.num_experts == 0 || router.top_k == 0 || streams.is_empty() {
            return Ok(InflightMoe {
                layer,
                hidden,
                active_stream_indices: Vec::new(),
                post_experts_norm: Vec::new(),
                norm_offset,
                eps,
            });
        }

        // 1. Route locally.
        let (_h_norm, expert_indices, expert_weights) = router.route(h, norm_offset, eps);

        // 2. Encode residual bytes once. The client applies post-experts norm
        // after collecting all shard outputs, so the gRPC request must not
        // carry that hidden-sized tensor per shard/layer.
        let residual_bytes: Vec<u8> = h.iter().flat_map(|v| v.to_le_bytes()).collect();

        // 3. Distribute expert_ids/weights across shards.
        let shards_guard = self.shards.read().unwrap();
        let num_shards = shards_guard.len();
        let shard_urls: Vec<String> = shards_guard.iter().map(|s| s.config.url.clone()).collect();
        let mut shard_eids: Vec<Vec<u32>> = vec![Vec::new(); num_shards];
        let mut shard_ewts: Vec<Vec<f32>> = vec![Vec::new(); num_shards];
        for (&eid, &w) in expert_indices.iter().zip(expert_weights.iter()) {
            let si = shards_guard
                .iter()
                .position(|s| s.owns_unit(layer, eid))
                .ok_or(RemoteMoeError::NoShard { expert_id: eid })?;
            shard_eids[si].push(eid as u32);
            shard_ewts[si].push(w);
        }
        drop(shards_guard);
        let active_stream_indices: Vec<usize> = shard_eids
            .iter()
            .enumerate()
            .filter_map(|(si, ids)| (!ids.is_empty()).then_some(si))
            .collect();
        if metrics::enabled() {
            for (si, url) in shard_urls.iter().enumerate() {
                if shard_eids[si].is_empty() {
                    metrics::record_skip(url);
                }
            }
        }
        if active_stream_indices.is_empty() {
            return Ok(InflightMoe {
                layer,
                hidden,
                active_stream_indices,
                post_experts_norm: router.post_experts_norm.to_vec(),
                norm_offset,
                eps,
            });
        }
        if active_stream_indices.iter().any(|&si| si >= streams.len()) {
            return Err(RemoteMoeError::BadResponse(format!(
                "stream map has {} streams for {num_shards} shards",
                streams.len()
            )));
        }

        // 4. Fire one input per stream in parallel.
        //
        // Each fire is `tokio::sync::mpsc::UnboundedSender::send` (non-blocking
        // channel push, ~1µs) plus building the `ExpertLayerInput` struct,
        // which clones `residual_bytes` (~hidden × 4 = 11 KB) per shard.
        // Rayon's thread pool is already initialised across the inference path
        // and amortises scheduling to single-µs overhead per task, so parallel
        // fire wins even at N=2 and scales linearly with shard count.
        //
        // Single-shard fast path skips the rayon overhead — same shape as
        // the parallel-collect path.
        let shard_timing = metrics::shard_timing_enabled();
        let layer_start = std::time::Instant::now();
        if active_stream_indices.len() == 1 {
            let si = active_stream_indices[0];
            let input = larql_router_protocol::ExpertLayerInput {
                layer: layer as u32,
                expert_ids: shard_eids[si].clone(),
                expert_weights: shard_ewts[si].clone(),
                residual: residual_bytes.clone(),
                post_experts_norm: Vec::new(),
                norm_offset,
                eps,
            };
            if shard_timing {
                let issue_us = layer_start.elapsed().as_secs_f64() * 1e6;
                eprintln!(
                    "[moe-shard-timing] transport=grpc layer={layer} shard={} K={} fire_us={issue_us:.0}",
                    shard_urls[si],
                    shard_eids[si].len(),
                );
            }
            streams[si].fire(input)?;
        } else {
            let residual_ref: &[u8] = &residual_bytes;
            active_stream_indices
                .par_iter()
                .try_for_each(|&si| -> Result<(), RemoteMoeError> {
                    let issue_us = layer_start.elapsed().as_secs_f64() * 1e6;
                    if shard_timing {
                        eprintln!(
                            "[moe-shard-timing] transport=grpc layer={layer} shard={} K={} fire_us={issue_us:.0}",
                            shard_urls[si],
                            shard_eids[si].len(),
                        );
                    }
                    let input = larql_router_protocol::ExpertLayerInput {
                        layer: layer as u32,
                        expert_ids: shard_eids[si].clone(),
                        expert_weights: shard_ewts[si].clone(),
                        residual: residual_ref.to_vec(),
                        post_experts_norm: Vec::new(),
                        norm_offset,
                        eps,
                    };
                    streams[si].fire(input)
                })?;
        }

        Ok(InflightMoe {
            layer,
            hidden,
            active_stream_indices,
            post_experts_norm: router.post_experts_norm.to_vec(),
            norm_offset,
            eps,
        })
    }

    /// Collect half of `forward_moe_stream`: condvar-wait one partial weighted
    /// sum per shard, accumulate, and apply the post-experts RMS norm.
    ///
    /// Each shard returns the raw weighted sum of its own experts (without
    /// post-norm) so the caller can sum across shards and norm the combined
    /// output once — `rms_norm(a) + rms_norm(b) ≠ rms_norm(a + b)`.
    pub fn forward_moe_stream_collect(
        &self,
        streams: &[ShardStream],
        inflight: InflightMoe,
    ) -> Result<Vec<f32>, RemoteMoeError> {
        self.forward_moe_stream_collect_with_timing(streams, inflight)
            .map(|(h2, _)| h2)
    }

    /// Same as [`Self::forward_moe_stream_collect`] but also returns
    /// per-shard `(wall_collect_ms, server_compute_ms)` for diagnostics.
    /// The `wall_collect_ms` is the wall-clock time the caller waited
    /// for that shard's response (network + server compute + decode);
    /// `server_compute_ms` is what the server reported (when timing is
    /// enabled there).  `network_ms ≈ wall_collect_ms − server_compute_ms`.
    pub fn forward_moe_stream_collect_with_timing(
        &self,
        streams: &[ShardStream],
        inflight: InflightMoe,
    ) -> Result<LayerMoeResult, RemoteMoeError> {
        let InflightMoe {
            layer,
            hidden,
            active_stream_indices,
            post_experts_norm,
            norm_offset,
            eps,
        } = inflight;
        let n_streams = active_stream_indices.len();

        if hidden == 0 || n_streams == 0 {
            return Ok((vec![0.0f32; hidden], Vec::new()));
        }

        // Parallel collect across shards: spawn one OS thread per stream and
        // join them all. Each thread blocks on its shard's `result_rx` condvar
        // independently, so the per-layer collect wall time is `max(per_shard)`
        // not `sum(per_shard)`. The win scales linearly with shard count and
        // is the load-bearing primitive for multi-shard remote topologies
        // (Kimi K2.6 / DeepSeek V4 class deployments) — see roadmap F-COLLECT.
        //
        // Single-shard runs hit the `n_streams == 1` shortcut to skip the
        // thread::scope overhead (~50µs/layer) — measurable on a single-shard
        // colocated bench where parallel and sequential are equivalent anyway.
        let shard_timing = metrics::shard_timing_enabled();
        let collect_start = std::time::Instant::now();
        type CollectResult = (usize, f32, Result<(Vec<f32>, f32), RemoteMoeError>);
        let results: Vec<CollectResult> = if n_streams == 1 {
            let si = active_stream_indices[0];
            let t0 = std::time::Instant::now();
            let res = streams[si].collect_with_timing();
            let wall_ms = t0.elapsed().as_secs_f32() * 1000.0;
            vec![(si, wall_ms, res)]
        } else {
            std::thread::scope(|s| {
                let handles: Vec<_> = streams
                    .iter()
                    .enumerate()
                    .filter_map(|(si, stream)| {
                        active_stream_indices.contains(&si).then_some((si, stream))
                    })
                    .map(|(si, stream)| {
                        s.spawn(move || -> CollectResult {
                            let t0 = std::time::Instant::now();
                            let res = stream.collect_with_timing();
                            let wall_ms = t0.elapsed().as_secs_f32() * 1000.0;
                            (si, wall_ms, res)
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("collect thread panicked"))
                    .collect()
            })
        };

        let mut out = vec![0.0f32; hidden];
        let mut per_shard: Vec<(f32, f32)> = Vec::with_capacity(n_streams);
        for (si, wall_ms, res) in results {
            let (partial, server_compute_ms) = res?;
            if shard_timing {
                let done_us = collect_start.elapsed().as_secs_f64() * 1e6;
                eprintln!(
                    "[moe-shard-timing] transport=grpc layer={layer} shard_index={si} collect_done_us={done_us:.0} wall_us={:.0} server_compute_us={:.0}",
                    wall_ms as f64 * 1000.0,
                    server_compute_ms as f64 * 1000.0,
                );
            }
            per_shard.push((wall_ms, server_compute_ms));
            // A partial of the wrong width is that shard's experts missing,
            // not a partial to skip.
            if partial.len() != hidden {
                return Err(RemoteMoeError::BadResponse(format!(
                    "shard {si} returned {} floats for layer {layer}, expected {hidden}",
                    partial.len()
                )));
            }
            for (acc, v) in out.iter_mut().zip(partial.iter()) {
                *acc += v;
            }
        }

        let normed = rms_norm(&out, &post_experts_norm, eps, norm_offset);
        Ok((normed, per_shard))
    }
}
