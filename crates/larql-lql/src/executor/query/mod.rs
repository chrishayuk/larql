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

pub(super) fn prefer_readable_feature_meta(
    mut meta: larql_vindex::FeatureMeta,
) -> larql_vindex::FeatureMeta {
    let readable: Vec<larql_models::TopKEntry> = meta
        .top_k
        .iter()
        .filter(|entry| crate::executor::helpers::is_readable_token(&entry.token))
        .cloned()
        .collect();
    if !readable.is_empty() {
        meta.top_k = readable;
        if let Some(first) = meta.top_k.first() {
            meta.top_token = first.token.clone();
            meta.top_token_id = first.token_id;
        }
    }
    meta
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
    use super::{manifest_candidate_token_ids, prefer_readable_feature_meta};
    use larql_models::TopKEntry;
    use larql_vindex::FeatureMeta;

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

    #[test]
    fn prefer_readable_feature_meta_promotes_readable_token() {
        let meta = FeatureMeta {
            top_token: "�".into(),
            top_token_id: 1,
            c_score: 0.1,
            top_k: vec![
                TopKEntry {
                    token: "�".into(),
                    token_id: 1,
                    logit: 3.0,
                },
                TopKEntry {
                    token: "Paris".into(),
                    token_id: 2,
                    logit: 2.0,
                },
                TopKEntry {
                    token: "France".into(),
                    token_id: 3,
                    logit: 1.0,
                },
            ],
        };

        let meta = prefer_readable_feature_meta(meta);

        assert_eq!(meta.top_token, "Paris");
        assert_eq!(meta.top_token_id, 2);
        assert_eq!(meta.top_k[0].token, "Paris");
    }
}
