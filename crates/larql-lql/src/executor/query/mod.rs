//! Query executor: WALK, INFER, SELECT, DESCRIBE, EXPLAIN.
//!
//! Each verb lives in its own file. Shared helpers (layer-band
//! resolution) live here because both DESCRIBE and EXPLAIN INFER
//! consume them.

mod describe;
mod explain;
mod infer;
mod infer_trace;
mod select;
mod walk;

/// Resolve the layer-band boundaries from the vindex config, with a
/// family-based default and a final whole-range fallback.
pub(super) fn manifest_candidate_token_ids(
    prompt_token_ids: &[u32],
    vocab_size: usize,
    limit: usize,
) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::new();
    if vocab_size == 0 {
        return ids;
    }

    let cap = limit.max(prompt_token_ids.len()).min(vocab_size);
    for &id in prompt_token_ids {
        if (id as usize) < vocab_size && !ids.contains(&id) {
            ids.push(id);
            if ids.len() >= cap {
                return sorted_unique(ids);
            }
        }
    }
    if limit == 0 {
        return sorted_unique(ids);
    }

    let prefix_target = (limit / 2).max(1).min(vocab_size);
    for id in 0..prefix_target as u32 {
        if !ids.contains(&id) {
            ids.push(id);
            if ids.len() >= limit.min(vocab_size) {
                return sorted_unique(ids);
            }
        }
    }

    let target = limit.min(vocab_size);
    let remaining = target.saturating_sub(ids.len());
    if remaining > 0 {
        let stride = (vocab_size / remaining.max(1)).max(1);
        let mut id = prefix_target;
        while ids.len() < target && id < vocab_size {
            let candidate = id as u32;
            if !ids.contains(&candidate) {
                ids.push(candidate);
            }
            id = id.saturating_add(stride);
        }
    }
    sorted_unique(ids)
}

fn sorted_unique(mut ids: Vec<u32>) -> Vec<u32> {
    ids.sort_unstable();
    ids.dedup();
    ids
}

pub(super) fn resolve_bands(config: &larql_vindex::VindexConfig) -> larql_vindex::LayerBands {
    let last = config.num_layers.saturating_sub(1);
    config
        .layer_bands
        .clone()
        .or_else(|| larql_vindex::LayerBands::for_family(&config.family, config.num_layers))
        .unwrap_or(larql_vindex::LayerBands {
            syntax: (0, last),
            knowledge: (0, last),
            output: (0, last),
        })
}

#[cfg(test)]
mod tests {
    use super::manifest_candidate_token_ids;

    #[test]
    fn manifest_candidates_mix_prefix_prompt_and_vocab_stride() {
        let ids = manifest_candidate_token_ids(&[900, 3], 1_000, 10);

        assert!(ids.contains(&900), "prompt token should be retained");
        assert!(
            ids.iter().any(|&id| id < 5),
            "low-id prefix should be sampled"
        );
        assert!(
            ids.iter().any(|&id| id > 500),
            "large-vocab sampling should include high-id candidates"
        );
        assert!(ids.len() <= 10);
    }
}
