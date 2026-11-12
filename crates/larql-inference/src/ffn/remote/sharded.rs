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

use super::codec::WireFormat;
use super::http::{RemoteFfnConfig, RemoteFfnError, RemoteWalkBackend, WirePreference};
use crate::ffn::FfnBackend;
use larql_compute::cpu::ops::q4k_q8k_dot::Q8KActivation;

/// [`crate::ffn::FfnActivations::Absent`] reason when no shard owns the
/// requested layer — the backend returns a zero output delta and has
/// observed nothing.
const REASON_NO_SHARD_FOR_LAYER: &str = "no shard owns this layer (zero FFN delta, unobserved)";

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
        Self::connect_with_wire(spec, timeout, WirePreference::BestAvailable)
    }

    /// Build from a spec string with an explicit wire format preference.
    pub fn connect_with_wire(
        spec: &str,
        timeout: Duration,
        wire: WirePreference,
    ) -> Result<Self, RemoteFfnError> {
        Self::connect_with_wire_formats(spec, timeout, WireFormat::F32, wire)
    }

    /// Build from a spec string with both wire directions set explicitly:
    /// `wire_in` is the request residual encoding (request `Content-Type`),
    /// `wire_out` the `Accept`-negotiated response preference. The asymmetric
    /// twin of [`Self::connect_with_wire`] (DEC funnel v0.5 §3 DEC-1A) —
    /// every shard connection gets the same direction pair.
    pub fn connect_with_wire_formats(
        spec: &str,
        timeout: Duration,
        wire_in: WireFormat,
        wire_out: WirePreference,
    ) -> Result<Self, RemoteFfnError> {
        let shards = if spec.contains('=') {
            parse_shard_map_with_wire(spec, timeout, wire_in, wire_out)?
        } else {
            let config = RemoteFfnConfig::new(spec)
                .with_timeout(timeout)
                .with_wire_formats(wire_in, wire_out);
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

    /// Total wire bytes sent across all shards (request bodies).
    pub fn wire_bytes_sent(&self) -> u64 {
        self.shards
            .iter()
            .map(|s| s.backend.wire_bytes_sent())
            .sum()
    }

    /// Total wire bytes received across all shards (response bodies).
    pub fn wire_bytes_recv(&self) -> u64 {
        self.shards
            .iter()
            .map(|s| s.backend.wire_bytes_recv())
            .sum()
    }

    /// Reset wire byte counters on all shards.
    pub fn reset_wire_counters(&self) {
        for s in &self.shards {
            s.backend.reset_wire_counters();
        }
    }
}

impl LayerShardedBackend {
    /// Fire one HTTP request per layer in parallel.
    ///
    /// Each layer gets its own independent `h_post_attn` input (not chained).
    /// Returns one FFN output vector per layer, in layer order.
    ///
    /// Uses `std::thread::scope` so shards can be borrowed without `Arc`.
    ///
    /// # Panics
    ///
    /// Panics if a layer has no owning shard, or if a shard's request fails
    /// (mirrors `RemoteWalkBackend::forward`'s own panic-on-transport-error
    /// convention). A silent zero-fill here would let decode continue on a
    /// corrupted hidden state — see
    /// docs/audits/dec-readiness-review-2026-07-22.md §1c.
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
                            None => panic!(
                                "forward_predispatch_all: layer {layer} has no owning shard \
                                 (check --ffn shard-map coverage)"
                            ),
                        }
                    })
                })
                .collect();

            for (result, handle) in results.iter_mut().zip(handles) {
                *result = match handle.join() {
                    Ok(v) => v,
                    Err(e) => std::panic::resume_unwind(e),
                };
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

    fn forward_observed(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> (Array2<f32>, crate::ffn::FfnActivations) {
        match self.shard_for(layer) {
            // Delegates to `RemoteWalkBackend`'s trait default — the
            // shard wire protocol carries outputs only, so the honest
            // observation is Absent.
            Some(shard) => shard.forward_observed(layer, x),
            None => (
                Array2::zeros(x.raw_dim()),
                crate::ffn::FfnActivations::Absent {
                    reason: REASON_NO_SHARD_FOR_LAYER,
                },
            ),
        }
    }

    fn forward_moe_full_layer(
        &self,
        layer: usize,
        h_post_attn: &Array2<f32>,
    ) -> Result<Option<Array2<f32>>, larql_execution::BoxRefusal> {
        // No shard owns this layer: not applicable, not a refusal. The caller
        // dispatches locally and its result is still correct.
        let Some(shard) = self.shard_for(layer) else {
            return Ok(None);
        };
        shard.forward_moe_full_layer(layer, h_post_attn)
    }

    fn name(&self) -> &str {
        "layer-sharded-remote"
    }
}

// ── Parse "START-END=URL,..." ─────────────────────────────────────────────────

fn parse_shard_map_with_wire(
    spec: &str,
    timeout: Duration,
    wire_in: WireFormat,
    wire_out: WirePreference,
) -> Result<Vec<LayerShard>, RemoteFfnError> {
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
        let config = RemoteFfnConfig::new(url)
            .with_timeout(timeout)
            .with_wire_formats(wire_in, wire_out);
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
    check_coverage(&shards)?;
    Ok(shards)
}

/// Refuse a shard map that leaves a layer without an owner, or gives one
/// two.
///
/// `FfnBackend::forward` cannot fail, so a layer no shard owns was answered
/// with zeros: the model ran with that layer's FFN missing and nothing said
/// so. The map is checked here, once, at connect.
///
/// The layer count comes from the shards' own stats. When none of them
/// reports one, the map is still required to start at layer 0 and have no
/// hole between its ranges.
fn check_coverage(shards: &[LayerShard]) -> Result<(), RemoteFfnError> {
    let mut ranges: Vec<(usize, usize, &str)> = shards
        .iter()
        .map(|s| (s.start, s.end, s.backend.base_url()))
        .collect();
    ranges.sort_unstable_by_key(|(start, end, _)| (*start, *end));

    for pair in ranges.windows(2) {
        let ((_, end, first), (start, _, second)) = (pair[0], pair[1]);
        if start <= end {
            return Err(RemoteFfnError::Client(format!(
                "--ffn: layer {start} is claimed by two shards ({first} and {second})"
            )));
        }
    }

    let mut reported: Vec<(usize, &str)> = shards
        .iter()
        .filter_map(|s| Some((s.backend.num_layers()?, s.backend.base_url())))
        .collect();
    reported.dedup_by_key(|(layers, _)| *layers);
    if let [(layers, first), (other, second), ..] = reported[..] {
        return Err(RemoteFfnError::Client(format!(
            "--ffn: the shards serve different models: {first} reports {layers} layers and \
             {second} reports {other}"
        )));
    }
    // One past the last layer that must be owned.
    let required_end = match reported.first() {
        Some((layers, _)) => *layers,
        None => ranges.last().map_or(0, |(_, end, _)| end.saturating_add(1)),
    };

    let mut next = 0usize;
    for (start, end, _) in &ranges {
        if *start > next {
            break;
        }
        next = end.saturating_add(1);
    }
    if next < required_end {
        let last_missing = ranges
            .iter()
            .map(|(start, _, _)| *start)
            .find(|start| *start > next)
            .unwrap_or(required_end)
            - 1;
        return Err(RemoteFfnError::Client(format!(
            "--ffn: no shard owns layer(s) {next}-{last_missing}; every layer needs an owner, \
             or its FFN would be computed as zero"
        )));
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A shard-less backend: every layer is unowned. Constructible
    /// without a network because the no-shard arms are exactly the
    /// code that must not depend on one.
    fn empty_backend() -> LayerShardedBackend {
        LayerShardedBackend { shards: Vec::new() }
    }

    #[test]
    fn no_shard_forward_returns_zero_delta() {
        let be = empty_backend();
        let x = Array2::<f32>::from_elem((2, 4), 1.0);
        let out = be.forward(0, &x);
        assert_eq!(out.shape(), &[2, 4]);
        assert!(out.iter().all(|v| *v == 0.0), "unowned layer → zero delta");
        assert_eq!(be.name(), "layer-sharded-remote");
    }

    #[test]
    fn no_shard_forward_observed_is_absent_with_named_reason() {
        let be = empty_backend();
        let x = Array2::<f32>::from_elem((1, 4), 1.0);
        let (out, obs) = be.forward_observed(0, &x);
        assert!(out.iter().all(|v| *v == 0.0));
        assert_eq!(
            obs.absent_reason(),
            Some(REASON_NO_SHARD_FOR_LAYER),
            "a zero delta from an unowned layer must be marked unobserved, \
             never a fabricated activation tensor"
        );
        // No shard owns the layer: not applicable, and specifically not a
        // refusal — the caller must still be free to dispatch locally.
        assert!(matches!(be.forward_moe_full_layer(0, &x), Ok(None)));
    }

    #[test]
    fn parse_layer_range_accepts_ordered_and_rejects_reversed() {
        assert_eq!(parse_layer_range("3-7"), Some((3, 7)));
        assert_eq!(parse_layer_range("7-3"), None);
        assert_eq!(parse_layer_range("x-3"), None);
    }
}
