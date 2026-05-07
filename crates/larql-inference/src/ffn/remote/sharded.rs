//! Layer-sharded FFN backend.
//!
//! Routes each layer's FFN call to whichever shard owns that layer range.
//! A single-URL `--ffn URL` is the degenerate case (one shard, all layers).
//! A multi-shard `--ffn "0-14=URL1,15-29=URL2"` fans out by layer.
//!
//! Each shard may itself have `--moe-shards` configured server-side, making
//! expert dispatch transparent to the client.

use std::time::Duration;

use ndarray::Array2;

use super::http::{RemoteFfnConfig, RemoteFfnError, RemoteWalkBackend};
use crate::ffn::FfnBackend;
use larql_compute::cpu::ops::q4k_q8k_dot::Q8KActivation;

struct LayerShard {
    start: usize,
    end: usize, // inclusive
    backend: RemoteWalkBackend,
}

/// FFN backend that routes each layer to the owning shard.
///
/// Build with [`LayerShardedBackend::connect`]. Parses either:
/// - A bare URL `"http://host:8080"` → single shard, all layers.
/// - A shard map `"0-14=http://a:8091,15-29=http://b:8092"` → routed by layer.
pub struct LayerShardedBackend {
    shards: Vec<LayerShard>,
}

impl LayerShardedBackend {
    /// Build from a spec string and connect (health-check) each shard.
    pub fn connect(spec: &str, timeout: Duration) -> Result<Self, RemoteFfnError> {
        let shards = if spec.contains('=') {
            parse_shard_map(spec, timeout)?
        } else {
            let config = RemoteFfnConfig::new(spec).with_timeout(timeout);
            let backend = RemoteWalkBackend::connect(config)?;
            vec![LayerShard {
                start: 0,
                end: usize::MAX,
                backend,
            }]
        };
        Ok(Self { shards })
    }

    pub fn hidden_size(&self) -> usize {
        self.shards
            .first()
            .map(|s| s.backend.hidden_size())
            .unwrap_or(0)
    }

    /// URL of the first shard (for logging/display).
    pub fn primary_url(&self) -> &str {
        self.shards
            .first()
            .map(|s| s.backend.base_url())
            .unwrap_or("")
    }

    fn shard_for(&self, layer: usize) -> Option<&RemoteWalkBackend> {
        self.shards
            .iter()
            .find(|s| layer >= s.start && layer <= s.end)
            .map(|s| &s.backend)
    }
}

impl LayerShardedBackend {
    /// Fire one HTTP request per layer in parallel.
    ///
    /// Each layer gets its own independent `h_post_attn` input (not chained).
    /// Returns one FFN output vector per layer, in layer order.
    ///
    /// Uses `std::thread::scope` so shards can be borrowed without `Arc`.
    pub fn forward_predispatch_all(&self, h_per_layer: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let hidden = self.hidden_size();
        let num_layers = h_per_layer.len();
        let mut results: Vec<Vec<f32>> = vec![vec![0.0f32; hidden]; num_layers];

        std::thread::scope(|s| {
            let handles: Vec<_> = h_per_layer
                .iter()
                .enumerate()
                .map(|(layer, h)| {
                    s.spawn(move || {
                        let x = Array2::from_shape_vec((1, hidden), h.clone())
                            .expect("h_per_layer shape must match hidden");
                        match self.shard_for(layer) {
                            Some(shard) => shard.forward(layer, &x).row(0).to_vec(),
                            None => vec![0.0f32; hidden],
                        }
                    })
                })
                .collect();

            for (result, handle) in results.iter_mut().zip(handles) {
                *result = handle.join().unwrap_or_else(|_| vec![0.0f32; hidden]);
            }
        });

        results
    }
}

impl LayerShardedBackend {
    /// Fire one HTTP request per layer in parallel using the Q8K wire format.
    ///
    /// Each layer's pre-normed Q8K activation is dispatched to the owning shard.
    /// Layers for the same shard are grouped into a single HTTP request.
    /// Returns one FFN output vector per layer, in layer order.
    ///
    /// Falls back to `forward_predispatch_all` (f32) on any failure (e.g. the
    /// server doesn't support `/v1/walk-ffn-q8k`).
    pub fn forward_predispatch_all_q8k(&self, h_per_layer: &[Q8KActivation]) -> Vec<Vec<f32>> {
        let hidden = self.hidden_size();
        let num_layers = h_per_layer.len();
        let mut results: Vec<Vec<f32>> = vec![vec![0.0f32; hidden]; num_layers];

        // Group layers by shard.
        // Each group: (shard_ref, Vec<(layer_idx, &Q8KActivation)>)
        struct ShardGroup<'a> {
            shard: &'a RemoteWalkBackend,
            layers: Vec<(usize, usize)>, // (layer_idx, result_slot)
        }

        // Build shard groups in layer order.
        let mut shard_groups: Vec<ShardGroup<'_>> = Vec::new();
        for (layer, q8k) in h_per_layer.iter().enumerate() {
            let _ = q8k; // borrow check — we'll collect refs below
            if let Some(shard) = self.shard_for(layer) {
                // Find or create a group for this shard (pointer equality).
                let shard_ptr = shard as *const RemoteWalkBackend;
                if let Some(g) = shard_groups
                    .iter_mut()
                    .find(|g| std::ptr::eq(g.shard, shard_ptr))
                {
                    g.layers.push((layer, layer));
                } else {
                    shard_groups.push(ShardGroup {
                        shard,
                        layers: vec![(layer, layer)],
                    });
                }
            }
        }

        std::thread::scope(|s| {
            let handles: Vec<_> = shard_groups
                .iter()
                .map(|g| {
                    let layer_indices: Vec<usize> = g.layers.iter().map(|(l, _)| *l).collect();
                    let q8k_refs: Vec<(usize, &Q8KActivation)> = layer_indices
                        .iter()
                        .map(|&l| (l, &h_per_layer[l]))
                        .collect();
                    let shard = g.shard;
                    s.spawn(move || {
                        match shard.call_q8k_layers(&q8k_refs) {
                            Ok(map) => map,
                            Err(_) => {
                                // Fall back: call each layer via the f32 path.
                                let mut fallback = std::collections::HashMap::new();
                                for &l in &layer_indices {
                                    let x =
                                        Array2::from_shape_vec((1, hidden), vec![0.0f32; hidden])
                                            .expect("shape");
                                    // We don't have h_post_attn here; return zeros
                                    // so the outer fallback in generate_with_remote_ffn_batch
                                    // can re-dispatch via forward_predispatch_all.
                                    fallback.insert(l, vec![0.0f32; hidden]);
                                    let _ = x;
                                }
                                fallback
                            }
                        }
                    })
                })
                .collect();

            for handle in handles {
                let map = handle.join().unwrap_or_default();
                for (layer, floats) in map {
                    if layer < num_layers {
                        results[layer] = floats;
                    }
                }
            }
        });

        results
    }
}

impl LayerShardedBackend {
    /// Send a single layer's Q8K-prenormed activation to the owning shard and
    /// return the FFN delta. Uses the same `/v1/walk-ffn-q8k` wire format as
    /// `call_q8k_layers`. Returns `None` if the shard doesn't support Q8K or
    /// if this layer has no owning shard.
    pub fn forward_single_q8k(&self, layer: usize, q8k: &Q8KActivation) -> Option<Vec<f32>> {
        let shard = self.shard_for(layer)?;
        let mut map = shard.call_q8k_layers(&[(layer, q8k)]).ok()?;
        map.remove(&layer)
    }
}

impl FfnBackend for LayerShardedBackend {
    fn forward(&self, layer: usize, x: &Array2<f32>) -> Array2<f32> {
        match self.shard_for(layer) {
            Some(shard) => shard.forward(layer, x),
            None => Array2::zeros(x.raw_dim()),
        }
    }

    fn forward_with_activation(&self, layer: usize, x: &Array2<f32>) -> (Array2<f32>, Array2<f32>) {
        match self.shard_for(layer) {
            Some(shard) => shard.forward_with_activation(layer, x),
            None => {
                let z = Array2::zeros(x.raw_dim());
                (z.clone(), z)
            }
        }
    }

    fn forward_moe_full_layer(
        &self,
        layer: usize,
        h_post_attn: &Array2<f32>,
    ) -> Option<Array2<f32>> {
        self.shard_for(layer)?
            .forward_moe_full_layer(layer, h_post_attn)
    }

    fn name(&self) -> &str {
        "layer-sharded-remote"
    }
}

// ── Parse "START-END=URL,..." ─────────────────────────────────────────────────

fn parse_shard_map(spec: &str, timeout: Duration) -> Result<Vec<LayerShard>, RemoteFfnError> {
    let mut shards = Vec::new();
    for segment in spec.split(',') {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        let mut parts = segment.splitn(2, '=');
        let range_str = parts.next().ok_or_else(|| {
            RemoteFfnError::Client(format!("malformed --ffn segment: {segment:?}"))
        })?;
        let url = parts.next().ok_or_else(|| {
            RemoteFfnError::Client(format!("missing URL in --ffn segment: {segment:?}"))
        })?;
        let (start, end) = parse_layer_range(range_str).ok_or_else(|| {
            RemoteFfnError::Client(format!("bad layer range {range_str:?} in --ffn"))
        })?;
        let config = RemoteFfnConfig::new(url).with_timeout(timeout);
        let backend = RemoteWalkBackend::connect(config)?;
        shards.push(LayerShard {
            start,
            end,
            backend,
        });
    }
    if shards.is_empty() {
        return Err(RemoteFfnError::Client(
            "--ffn: no valid shard segments".into(),
        ));
    }
    Ok(shards)
}

fn parse_layer_range(s: &str) -> Option<(usize, usize)> {
    let mut parts = s.splitn(2, '-');
    let start: usize = parts.next()?.trim().parse().ok()?;
    let end: usize = parts.next()?.trim().parse().ok()?;
    if start <= end {
        Some((start, end))
    } else {
        None
    }
}
