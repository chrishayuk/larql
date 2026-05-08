//! `WALK` — pure vindex feature scan, no attention.

use crate::ast::{Range, WalkMode};
use crate::error::LqlError;
use crate::executor::Session;
use std::collections::HashSet;

impl Session {
    pub(crate) fn exec_walk(
        &self,
        prompt: &str,
        top: Option<u32>,
        layers: Option<&Range>,
        mode: Option<WalkMode>,
        compare: bool,
    ) -> Result<Vec<String>, LqlError> {
        let (path, config, patched) = self.require_vindex()?;
        let top_k = top.unwrap_or(10) as usize;

        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| LqlError::exec("tokenize error", e))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        if token_ids.is_empty() {
            return Err(LqlError::Execution("empty prompt".into()));
        }

        let last_tok = *token_ids.last().unwrap();
        let token_str = tokenizer
            .decode(&[last_tok], true)
            .unwrap_or_else(|_| format!("T{last_tok}"));

        let (embed_rows, embed_scale) = larql_vindex::load_vindex_embedding_rows(path, &[last_tok])
            .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
        let embed_row = embed_rows.row(0);
        let query: larql_vindex::ndarray::Array1<f32> = embed_row.mapv(|v| v * embed_scale);

        let all_layers = patched.loaded_layers();
        let walk_layers: Vec<usize> = if let Some(range) = layers {
            (range.start as usize..=range.end as usize)
                .filter(|l| all_layers.contains(l))
                .collect()
        } else {
            all_layers
        };

        let start = std::time::Instant::now();
        let trace = patched.walk(&query, &walk_layers, top_k);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let mode_str = match mode {
            Some(WalkMode::Pure) => "pure (sparse KNN only)",
            Some(WalkMode::Dense) => "dense (full matmul)",
            Some(WalkMode::Hybrid) | None => "hybrid (default)",
        };

        let mut out = Vec::new();
        out.push(format!(
            "Feature scan for {:?} (token {:?}, {} layers, mode={})",
            prompt,
            token_str.trim(),
            walk_layers.len(),
            mode_str,
        ));
        out.push(String::new());

        let show_per_layer = if compare { 5 } else { 3 };
        let semantic_token_limit = std::env::var("LARQL_GGUF_MANIFEST_DOWN_META_TOKENS")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(256)
            .min(config.vocab_size);
        let semantic_hit_limit = std::env::var("LARQL_GGUF_MANIFEST_DOWN_META_HITS")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(12);
        let mut semantic_candidates: Vec<u32> = token_ids.clone();
        semantic_candidates.extend(0..semantic_token_limit as u32);
        semantic_candidates.sort_unstable();
        semantic_candidates.dedup();
        let semantic_order: HashSet<(usize, usize)> =
            walk_resolution_order(&trace, show_per_layer, semantic_hit_limit)
                .into_iter()
                .collect();
        for (layer_pos, (layer, hits)) in trace.layers.iter().enumerate() {
            if hits.is_empty() {
                continue;
            }
            for (hit_pos, hit) in hits.iter().take(show_per_layer).enumerate() {
                let mut meta = hit.meta.clone();
                if semantic_order.contains(&(layer_pos, hit_pos)) && !semantic_candidates.is_empty()
                {
                    if let Ok(resolved) = larql_vindex::load_vindex_gguf_feature_meta(
                        path,
                        *layer,
                        hit.feature,
                        &semantic_candidates,
                        3,
                    ) {
                        meta = resolved;
                        for entry in &mut meta.top_k {
                            if let Ok(decoded) = tokenizer.decode(&[entry.token_id], true) {
                                let decoded = decoded.trim().to_string();
                                if !decoded.is_empty() {
                                    entry.token = decoded;
                                }
                            }
                        }
                        meta = super::prefer_readable_feature_meta(meta);
                        if !crate::executor::helpers::is_readable_token(&meta.top_token) {
                            meta = hit.meta.clone();
                        }
                    }
                }
                let down_top: String = meta
                    .top_k
                    .iter()
                    .take(3)
                    .map(|t| t.token.clone())
                    .collect::<Vec<_>>()
                    .join(", ");
                out.push(format!(
                    "  L{:2}: F{:<5} gate={:+.1}  top={:15}  down=[{}]",
                    layer,
                    hit.feature,
                    hit.gate_score,
                    format!("{:?}", meta.top_token),
                    down_top,
                ));
            }
        }

        out.push(format!("\n{:.1}ms", elapsed_ms));
        if compare {
            out.push(String::new());
            out.push(
                "Note: COMPARE shows more features per layer. For inference use INFER.".into(),
            );
        } else {
            out.push(String::new());
            out.push("Note: pure vindex scan (no attention). For inference use INFER.".into());
        }

        Ok(out)
    }
}

fn walk_resolution_order(
    trace: &larql_vindex::WalkTrace,
    show_per_layer: usize,
    limit: usize,
) -> Vec<(usize, usize)> {
    let mut selected = Vec::new();
    for rank in 0..show_per_layer {
        let mut rank_hits: Vec<(usize, usize, f32)> = trace
            .layers
            .iter()
            .enumerate()
            .filter_map(|(layer_pos, (_layer, hits))| {
                hits.get(rank)
                    .map(|hit| (layer_pos, rank, hit.gate_score.abs()))
            })
            .collect();
        rank_hits.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        for (layer_pos, hit_pos, _score) in rank_hits {
            if selected.len() >= limit {
                return selected;
            }
            selected.push((layer_pos, hit_pos));
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::walk_resolution_order;
    use larql_models::TopKEntry;
    use larql_vindex::{FeatureMeta, WalkHit, WalkTrace};

    fn hit(layer: usize, feature: usize, gate_score: f32) -> WalkHit {
        WalkHit {
            layer,
            feature,
            gate_score,
            meta: FeatureMeta {
                top_token: format!("T{feature}"),
                top_token_id: feature as u32,
                c_score: 0.0,
                top_k: vec![TopKEntry {
                    token: format!("T{feature}"),
                    token_id: feature as u32,
                    logit: gate_score,
                }],
            },
        }
    }

    #[test]
    fn walk_resolution_order_spreads_manifest_label_budget_across_layers() {
        let trace = WalkTrace {
            layers: vec![
                (1, vec![hit(1, 10, 0.9), hit(1, 11, 0.8), hit(1, 12, 0.7)]),
                (2, vec![hit(2, 20, 0.6), hit(2, 21, 0.5), hit(2, 22, 0.4)]),
                (3, vec![hit(3, 30, 0.95), hit(3, 31, 0.3), hit(3, 32, 0.2)]),
            ],
        };

        let order = walk_resolution_order(&trace, 3, 4);

        assert_eq!(order, vec![(2, 0), (0, 0), (1, 0), (0, 1)]);
    }
}
