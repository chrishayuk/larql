//! Introspection executor: SHOW RELATIONS, SHOW LAYERS, SHOW FEATURES, SHOW MODELS, SHOW COMPACT STATUS.

use std::collections::HashMap;

use super::helpers::{dir_size, format_bytes, format_number, is_content_token};
use super::Session;
use crate::ast::*;
use crate::error::LqlError;
use larql_vindex::format::filenames::INDEX_JSON;

impl Session {
    pub(crate) fn exec_show_compact_status(&self) -> Result<Vec<String>, LqlError> {
        let (_path, _config, patched) = self.require_vindex()?;
        let l0_entries = patched.knn_store.len();
        let l1_edges = patched.num_overrides();
        let l1_layers: std::collections::HashSet<usize> = patched
            .overrides_gate_iter()
            .map(|(layer, _, _)| layer)
            .collect();
        let n_layers = patched.num_layers();
        let features_per_layer = if n_layers > 0 {
            patched.num_features(0)
        } else {
            0
        };
        let hidden_dim = patched.hidden_size();
        let memit_supported = hidden_dim >= 1024;

        let mut out = Vec::new();
        out.push(format!("Storage engine status (epoch {}):", self.epoch));
        out.push(format!(
            "  L0 (WAL/KNN):    {} entries (0 tombstones)",
            l0_entries,
        ));
        out.push(format!(
            "  L1 (arch-A):     {} edges across {} layers",
            l1_edges,
            l1_layers.len(),
        ));
        if memit_supported {
            out.push("  L2 (MEMIT):      0 facts across 0 cycles".to_string());
        } else {
            out.push(format!(
                "  L2 (MEMIT):      not available (hidden_dim={} < 1024)",
                hidden_dim,
            ));
        }
        out.push(format!(
            "  Base model:      {} layers × {} features",
            n_layers, features_per_layer,
        ));
        Ok(out)
    }
}

impl Session {
    pub(crate) fn exec_show_relations(
        &self,
        layer_filter: Option<u32>,
        with_examples: bool,
        mode: DescribeMode,
    ) -> Result<Vec<String>, LqlError> {
        let (_path, _config, patched) = self.require_vindex()?;

        let all_layers = patched.loaded_layers();
        let scan_layers: Vec<usize> = if let Some(l) = layer_filter {
            vec![l as usize]
        } else {
            all_layers
                .iter()
                .copied()
                .filter(|l| *l >= 14 && *l <= 27)
                .collect()
        };

        // ── Probe-confirmed relations (skip for Raw mode) ──
        let classifier = self.relation_classifier();
        let mut probe_relations: HashMap<String, usize> = HashMap::new();
        if mode != DescribeMode::Raw {
            if let Some(rc) = classifier {
                for &layer in &scan_layers {
                    let num_features = patched.num_features(layer);
                    for feat in 0..num_features {
                        if let Some(label) = rc.label_for_feature(layer, feat) {
                            *probe_relations.entry(label.to_string()).or_insert(0) += 1;
                        }
                    }
                }
            }
        }

        // ── Raw token relations (skip for Brief mode unless no probes) ──
        struct TokenInfo {
            count: usize,
            max_score: f32,
            min_layer: usize,
            max_layer: usize,
            original: String,
            examples: Vec<String>,
        }

        let show_raw = mode == DescribeMode::Raw
            || mode == DescribeMode::Verbose
            || probe_relations.is_empty();

        let mut tokens: HashMap<String, TokenInfo> = HashMap::new();
        if show_raw {
            for &layer in &scan_layers {
                if let Some(metas) = patched.down_meta_at(layer) {
                    for meta in metas.iter().flatten() {
                        let tok = meta.top_token.trim();
                        if !is_content_token(tok) {
                            continue;
                        }
                        if meta.c_score < 0.2 {
                            continue;
                        }
                        let key = tok.to_lowercase();
                        let examples: Vec<String> = meta
                            .top_k
                            .iter()
                            .filter(|t| t.token.trim() != tok && is_content_token(t.token.trim()))
                            .take(3)
                            .map(|t| t.token.trim().to_string())
                            .collect();
                        let entry = tokens.entry(key).or_insert(TokenInfo {
                            count: 0,
                            max_score: 0.0,
                            min_layer: layer,
                            max_layer: layer,
                            original: tok.to_string(),
                            examples,
                        });
                        entry.count += 1;
                        if meta.c_score > entry.max_score {
                            entry.max_score = meta.c_score;
                        }
                        if layer < entry.min_layer {
                            entry.min_layer = layer;
                        }
                        if layer > entry.max_layer {
                            entry.max_layer = layer;
                        }
                    }
                }
            }
        }

        let mut out = Vec::new();
        let layer_label = if let Some(l) = layer_filter {
            format!("L{}", l)
        } else {
            "L14-27".into()
        };

        // ── Probe-confirmed section ──
        if !probe_relations.is_empty() {
            let total_labels: usize = probe_relations.values().sum();
            out.push(format!(
                "Probe-confirmed relations ({} labels):",
                total_labels
            ));
            out.push(format!("{:<25} {:>8}", "Relation", "Features"));
            out.push("-".repeat(35));

            let mut probe_sorted: Vec<(&String, &usize)> = probe_relations.iter().collect();
            probe_sorted.sort_by(|a, b| b.1.cmp(a.1));

            let limit = if mode == DescribeMode::Brief {
                30
            } else {
                probe_sorted.len()
            };
            for (name, count) in probe_sorted.into_iter().take(limit) {
                out.push(format!("{:<25} {:>8}", name, count));
            }
        }

        // ── Raw token section ──
        if show_raw && !tokens.is_empty() {
            if !probe_relations.is_empty() {
                out.push(String::new());
            }

            let mut sorted: Vec<(&str, &TokenInfo)> = tokens
                .values()
                .map(|info| (info.original.as_str(), info))
                .collect();
            sorted.sort_by_key(|x| std::cmp::Reverse(x.1.count));

            let limit = if mode == DescribeMode::Verbose {
                50
            } else {
                30
            };
            sorted.truncate(limit);

            out.push(format!("Top output tokens ({}):", layer_label));
            out.push(format!(
                "{:<25} {:>8} {:>8} {:>10}",
                "Token", "Count", "Score", "Layers"
            ));
            out.push("-".repeat(55));

            for (tok, info) in &sorted {
                let examples_str = if with_examples && !info.examples.is_empty() {
                    format!("  e.g. {}", info.examples.join(", "))
                } else {
                    String::new()
                };
                out.push(format!(
                    "{:<25} {:>8} {:>8.2} {:>5}-{}{}",
                    tok, info.count, info.max_score, info.min_layer, info.max_layer, examples_str,
                ));
            }
        }

        if out.is_empty() || (probe_relations.is_empty() && tokens.is_empty()) {
            out.push("  (no relations found)".into());
        }

        Ok(out)
    }

    pub(crate) fn exec_show_layers(&self, range: Option<&Range>) -> Result<Vec<String>, LqlError> {
        let (_path, _config, patched) = self.require_vindex()?;

        let all_layers = patched.loaded_layers();
        let show_layers: Vec<usize> = if let Some(r) = range {
            (r.start as usize..=r.end as usize)
                .filter(|l| all_layers.contains(l))
                .collect()
        } else {
            all_layers
        };

        let mut out = Vec::new();
        out.push(format!(
            "{:<8} {:>10} {:>10} {:>15}",
            "Layer", "Features", "With Meta", "Top Token"
        ));
        out.push("-".repeat(48));

        for layer in &show_layers {
            let gate_count = patched
                .gate_vectors_at(*layer)
                .map(|m| m.shape()[0])
                .unwrap_or(0);
            let (meta_count, top_tok) = if let Some(metas) = patched.down_meta_at(*layer) {
                let count = metas.iter().filter(|m| m.is_some()).count();
                let mut freq: HashMap<&str, usize> = HashMap::new();
                for m in metas.iter().flatten() {
                    *freq.entry(&m.top_token).or_default() += 1;
                }
                let top = freq
                    .into_iter()
                    .max_by_key(|(_, c)| *c)
                    .map(|(t, _)| t.to_string())
                    .unwrap_or_default();
                (count, top)
            } else {
                (0, String::new())
            };

            out.push(format!(
                "L{:<7} {:>10} {:>10} {:>15}",
                layer,
                format_number(gate_count),
                format_number(meta_count),
                top_tok
            ));
        }

        Ok(out)
    }

    pub(crate) fn exec_show_features(
        &self,
        layer: u32,
        conditions: &[Condition],
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let (_path, config, patched) = self.require_vindex()?;
        // Default to num_layers — a manageable screenful that matches
        // the model's depth. Use LIMIT for more or fewer.
        let limit = limit.unwrap_or(config.num_layers as u32) as usize;

        // Extract filters from WHERE conditions
        let token_filter = conditions
            .iter()
            .find(|c| c.field == "relation" || c.field == "token")
            .and_then(|c| {
                if let Value::String(ref s) = c.value {
                    Some(s.as_str())
                } else {
                    None
                }
            });
        let min_score = conditions
            .iter()
            .find(|c| c.field == "confidence" || c.field == "c_score")
            .and_then(|c| match &c.value {
                Value::Number(n) => Some(*n as f32),
                Value::Integer(n) => Some(*n as f32),
                _ => None,
            });

        let nf = patched.num_features(layer as usize);
        if nf == 0 {
            return Err(LqlError::Execution(format!("no features at layer {layer}")));
        }

        let mut out = Vec::new();
        out.push(format!(
            "{:<8} {:<20} {:>10} {:>30}",
            "Feature", "Top Token", "Score", "Down outputs"
        ));
        out.push("-".repeat(72));

        let mut count = 0;
        for feat_idx in 0..nf {
            if count >= limit {
                break;
            }
            if let Some(meta) = patched.feature_meta(layer as usize, feat_idx) {
                // Apply WHERE filters
                if let Some(tf) = token_filter {
                    if !meta.top_token.to_lowercase().contains(&tf.to_lowercase()) {
                        continue;
                    }
                }
                if let Some(ms) = min_score {
                    if meta.c_score < ms {
                        continue;
                    }
                }

                let down_tokens: String = meta
                    .top_k
                    .iter()
                    .take(5)
                    .map(|t| t.token.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");

                out.push(format!(
                    "F{:<7} {:<20} {:>10.4} {:>30}",
                    feat_idx, meta.top_token, meta.c_score, down_tokens
                ));
                count += 1;
            }
        }

        Ok(out)
    }

    pub(crate) fn exec_show_entities(
        &self,
        layer_filter: Option<u32>,
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let (_path, config, patched) = self.require_vindex()?;
        let limit = limit.unwrap_or(50) as usize;

        let scan_layers: Vec<usize> = if let Some(l) = layer_filter {
            vec![l as usize]
        } else {
            (0..config.num_layers).collect()
        };

        // Collect distinct top_tokens across all scanned features.
        let mut entity_counts: std::collections::HashMap<String, (usize, f32)> =
            std::collections::HashMap::new();

        for layer in &scan_layers {
            let nf = patched.num_features(*layer);
            for feat in 0..nf {
                if let Some(meta) = patched.feature_meta(*layer, feat) {
                    let tok = meta.top_token.trim().to_string();
                    // Filter to named entities: starts with uppercase
                    // ASCII, 3+ chars, all alphabetic. This skips
                    // function words (the, of, in), subword fragments,
                    // and non-Latin scripts that are typically noise
                    // from polysemantic features.
                    if tok.len() < 3 {
                        continue;
                    }
                    let first = tok.chars().next().unwrap_or(' ');
                    if !first.is_ascii_uppercase() {
                        continue;
                    }
                    if !tok.chars().all(|c| c.is_alphabetic()) {
                        continue;
                    }
                    let entry = entity_counts.entry(tok).or_insert((0, 0.0));
                    entry.0 += 1;
                    if meta.c_score > entry.1 {
                        entry.1 = meta.c_score;
                    }
                }
            }
        }

        // Sort by feature count descending.
        let mut entities: Vec<(String, usize, f32)> = entity_counts
            .into_iter()
            .map(|(tok, (count, max_score))| (tok, count, max_score))
            .collect();
        entities.sort_by_key(|x| std::cmp::Reverse(x.1));
        entities.truncate(limit);

        let mut out = Vec::new();
        let layer_note = if let Some(l) = layer_filter {
            format!(" at layer {l}")
        } else {
            format!(" across {} layers", scan_layers.len())
        };
        out.push(format!(
            "Distinct entities{layer_note} ({} total, showing top {limit}):",
            entities.len().max(limit)
        ));
        out.push(format!(
            "{:<24} {:>10} {:>10}",
            "Entity", "Features", "Max Score"
        ));
        out.push("-".repeat(48));

        for (tok, count, max_score) in &entities {
            out.push(format!("{:<24} {:>10} {:>10.4}", tok, count, max_score));
        }

        if entities.is_empty() {
            out.push("  (no entities found)".into());
        }

        Ok(out)
    }

    pub(crate) fn exec_show_models(&self) -> Result<Vec<String>, LqlError> {
        let mut out = Vec::new();
        out.push(format!(
            "{:<35} {:>10} {:>8} {:>12}",
            "Model", "Size", "Layers", "Status"
        ));
        out.push("-".repeat(70));

        let cwd = std::env::current_dir().unwrap_or_default();
        if let Ok(entries) = std::fs::read_dir(&cwd) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let index_json = path.join(INDEX_JSON);
                    if index_json.exists() {
                        if let Ok(config) = larql_vindex::load_vindex_config(&path) {
                            let size = dir_size(&path);
                            out.push(format!(
                                "{:<35} {:>10} {:>8} {:>12}",
                                path.file_name().unwrap_or_default().to_string_lossy(),
                                format_bytes(size),
                                config.num_layers,
                                "ready",
                            ));
                        }
                    }
                }
            }
        }

        if out.len() == 2 {
            out.push("  (no vindexes found in current directory)".into());
        }

        Ok(out)
    }
}
