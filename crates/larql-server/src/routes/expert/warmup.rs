//! Boot-time warmup helpers for MoE shards.
//!
//! Both functions are no-ops when `LARQL_NO_WARMUP=1` (useful in low-RSS
//! dev setups). Run inside `spawn_blocking` from the main entrypoint.

use crate::env_flags;
use crate::state::LoadedModel;

/// Eager warmup of the per-(layer, expert) HNSW unit cache for **walk** /
/// interpretability KNN queries.  Iterates every `(layer, expert)` this
/// shard owns and pre-builds an HNSW index over that expert's gate slice
/// (`moe_intermediate_size` vectors per unit, vs `num_experts ×
/// moe_intermediate_size` for the layer-level index).
///
/// Independent of the Metal expert cache: this is for the gate-KNN code
/// path (`gate_knn_expert`), not the MoE forward pass.  Skipped when
/// `LARQL_NO_WARMUP=1`.  Requires `--hnsw` to actually be useful at query
/// time, but the cache is populated regardless so flipping the toggle on
/// later doesn't pay a build burst.
///
/// Returns `(units_built, num_layers, experts_per_shard)` so the caller
/// can log a one-line summary.  All builds happen in parallel via rayon.
pub fn warmup_hnsw_unit_cache(model: &LoadedModel) -> Result<(usize, usize, usize), String> {
    if env_flags::no_warmup() {
        return Ok((0, 0, 0));
    }
    let weights = model.get_or_load_weights()?;
    let arch = &*weights.arch;
    if !arch.is_hybrid_moe() {
        return Ok((0, 0, 0));
    }
    let num_layers = model.config.num_layers;
    let num_experts = arch.num_experts();
    let moe_inter = arch.moe_intermediate_size();
    if num_layers == 0 || moe_inter == 0 {
        return Ok((0, 0, 0));
    }
    // Resolve the (layer, expert_id) ownership set for this shard.
    // Priority: `--units` manifest (`unit_filter`) → `--experts START-END`
    // (`expert_filter`, layer-uniform) → all experts on every layer.
    let owned_units: Vec<(usize, usize)> = if let Some(units) = model.unit_filter.as_ref() {
        let mut v: Vec<(usize, usize)> = units.iter().copied().collect();
        v.sort_unstable();
        v
    } else {
        let (start, end_excl) = model.expert_filter.unwrap_or((0, num_experts));
        (0..num_layers)
            .flat_map(|l| (start..end_excl).map(move |e| (l, e)))
            .collect()
    };
    let n_experts_owned = if let Some(units) = model.unit_filter.as_ref() {
        units
            .iter()
            .map(|(_, e)| *e)
            .collect::<std::collections::HashSet<_>>()
            .len()
    } else {
        let (start, end_excl) = model.expert_filter.unwrap_or((0, num_experts));
        end_excl.saturating_sub(start)
    };

    // Build the (layer, feat_start, feat_end) triples for every owned unit.
    // feat_start_for_expert_e = e * moe_intermediate_size — same layout the
    // gate_knn_expert callers use.
    let mut units: Vec<(usize, usize, usize)> = Vec::with_capacity(owned_units.len());
    for (layer, eid) in owned_units {
        let fs = eid * moe_inter;
        let fe = (eid + 1) * moe_inter;
        units.push((layer, fs, fe));
    }

    // We need a `&VectorIndex` to call `warmup_hnsw_units`.  The patched
    // overlay's `blocking_read` exposes that synchronously — fine here
    // because this runs inside a `spawn_blocking` job during startup.
    let patched = model.patched.blocking_read();
    let n_built = patched.base().warmup_hnsw_units(&units);
    drop(patched);
    Ok((n_built, num_layers, n_experts_owned))
}

/// Eager warmup of the Metal expert buffer cache.
///
/// Iterates every `(layer, expert_id)` owned by this shard and calls
/// `cached_buffer_for_bytes` on the expert's gate_up + down mmap regions,
/// populating `BufferCache` so that subsequent RPC calls hit instantly
/// instead of paying the first-touch ~10–28ms Metal-buffer allocation.
///
/// Returns the number of (gate_up, down) buffer pairs staged.
///
/// Skipped when `LARQL_NO_WARMUP=1` (useful in low-RSS dev setups; warmup
/// allocates ~10MB × experts_owned × num_layers of Metal-resident memory).
#[cfg(all(feature = "metal-experts", target_os = "macos"))]
pub fn warmup_metal_expert_cache(model: &LoadedModel) -> Result<usize, String> {
    use larql_compute_metal::MetalBackend;

    if env_flags::no_warmup() {
        return Ok(0);
    }

    let weights = model.get_or_load_weights()?;
    let arch = &*weights.arch;
    if !arch.is_hybrid_moe() || !weights.has_per_layer_ffn() {
        return Ok(0);
    }

    let backend_slot = model.metal_backend.get_or_init(MetalBackend::new);
    let Some(backend) = backend_slot.as_ref() else {
        return Ok(0);
    };

    let num_layers = model.config.num_layers;
    let num_experts = arch.num_experts();

    // Same ownership-resolution pattern as warmup_hnsw_unit_cache:
    // unit_filter > expert_filter > all.  See that function for rationale.
    let owned_units: Vec<(usize, usize)> = if let Some(units) = model.unit_filter.as_ref() {
        let mut v: Vec<(usize, usize)> = units.iter().copied().collect();
        v.sort_unstable();
        v
    } else {
        let (start, end_excl) = model.expert_filter.unwrap_or((0, num_experts));
        (0..num_layers)
            .flat_map(|l| (start..end_excl).map(move |e| (l, e)))
            .collect()
    };

    let mut staged = 0usize;
    for (layer, eid) in owned_units {
        if let Some((gu, dn)) = weights.get_layer_entry_bytes(layer, eid) {
            // Each call returns a cached Buffer; first call pays the
            // mmap → Metal allocation/copy, subsequent calls are O(1)
            // hash lookups.  We discard the returned Buffer here — the
            // cache holds it for the server's lifetime.
            let _ = backend.cached_buffer_for_bytes(gu);
            let _ = backend.cached_buffer_for_bytes(dn);
            staged += 1;
        }
    }
    Ok(staged)
}
