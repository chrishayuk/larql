//! `SELECT * FROM {EDGES, FEATURES, ENTITIES}` + `NEAREST TO` KNN.

use crate::ast::{CompareOp, Condition, Field, NearestClause, OrderBy, Value};
use crate::error::LqlError;
use crate::executor::Session;

impl Session {
    pub(crate) fn exec_select(
        &self,
        _fields: &[Field],
        conditions: &[Condition],
        nearest: Option<&NearestClause>,
        order: Option<&OrderBy>,
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let (path, _config, patched) = self.require_vindex()?;

        // Handle NEAREST TO clause — KNN lookup
        if let Some(nc) = nearest {
            return self.exec_select_nearest(patched, path, nc, limit);
        }

        let all_layers = patched.loaded_layers();
        // Default limit: num_layers when filtering by feature (user
        // expects to see the feature across all layers), otherwise 20.
        let feature_filter_present = conditions.iter().any(|c| c.field == "feature");
        let default_limit = if feature_filter_present {
            patched.num_layers()
        } else {
            20
        };
        let limit = limit.unwrap_or(default_limit as u32) as usize;

        let entity_filter = conditions
            .iter()
            .find(|c| c.field == "entity")
            .and_then(|c| {
                if let Value::String(ref s) = c.value {
                    Some(s.as_str())
                } else {
                    None
                }
            });
        let relation_filter = conditions
            .iter()
            .find(|c| c.field == "relation")
            .and_then(|c| {
                if let Value::String(ref s) = c.value {
                    Some(s.as_str())
                } else {
                    None
                }
            });
        let layer_filter = conditions
            .iter()
            .find(|c| c.field == "layer")
            .and_then(|c| {
                if let Value::Integer(n) = c.value {
                    Some(n as usize)
                } else {
                    None
                }
            });
        let feature_filter = conditions
            .iter()
            .find(|c| c.field == "feature")
            .and_then(|c| {
                if let Value::Integer(n) = c.value {
                    Some(n as usize)
                } else {
                    None
                }
            });
        let score_filter = conditions
            .iter()
            .find(|c| c.field == "score" || c.field == "confidence")
            .and_then(|c| {
                let val = match &c.value {
                    Value::Number(n) => Some(*n as f32),
                    Value::Integer(n) => Some(*n as f32),
                    _ => None,
                };
                val.map(|v| (c.op.clone(), v))
            });

        struct Row {
            layer: usize,
            feature: usize,
            top_token: String,
            also: String,
            relation: String,
            c_score: f32,
        }

        let mut rows: Vec<Row> = Vec::new();
        let classifier = self.relation_classifier();

        let scan_layers: Vec<usize> = if let Some(l) = layer_filter {
            vec![l]
        } else {
            all_layers.clone()
        };

        // When entity + relation are both specified, use walk-based lookup:
        // embed the entity, walk all layers, find features that fire,
        // then filter by relation label. This finds "capital features that
        // activate for France" rather than "capital features whose top token
        // contains France".
        if let (Some(entity), Some(rel)) = (entity_filter, relation_filter) {
            let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
                .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
            let tokenizer = larql_vindex::load_vindex_tokenizer(path)
                .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

            let encoding = tokenizer
                .encode(entity, false)
                .map_err(|e| LqlError::exec("tokenize error", e))?;
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            if !token_ids.is_empty() {
                let hidden = embed.shape()[1];
                let query = if token_ids.len() == 1 {
                    embed.row(token_ids[0] as usize).mapv(|v| v * embed_scale)
                } else {
                    let mut avg = larql_vindex::ndarray::Array1::<f32>::zeros(hidden);
                    for &tok in &token_ids {
                        avg += &embed.row(tok as usize).mapv(|v| v * embed_scale);
                    }
                    avg /= token_ids.len() as f32;
                    avg
                };

                // Use a large top_k because the raw embedding query
                // has low cosine with deep-layer gate directions (the
                // residual stream has been transformed by N layers of
                // attention+FFN). We need to scan widely to find the
                // relation-labeled features that fire on this entity.
                let trace = patched.walk(&query, &scan_layers, 500);

                for (layer_idx, hits) in &trace.layers {
                    for hit in hits {
                        if let Some(feature_f) = feature_filter {
                            if hit.feature != feature_f {
                                continue;
                            }
                        }
                        let rel_label = classifier
                            .and_then(|rc| rc.label_for_feature(*layer_idx, hit.feature))
                            .unwrap_or("")
                            .to_string();
                        if rel_label.is_empty() {
                            continue;
                        }
                        let rel_norm = rel.to_lowercase();
                        let label_norm = rel_label.to_lowercase();
                        if !label_norm.contains(&rel_norm) && !rel_norm.contains(&label_norm) {
                            continue;
                        }
                        let also = hit
                            .meta
                            .top_k
                            .iter()
                            .skip(1)
                            .take(3)
                            .map(|e| e.token.clone())
                            .collect::<Vec<_>>()
                            .join(", ");
                        rows.push(Row {
                            layer: *layer_idx,
                            feature: hit.feature,
                            top_token: hit.meta.top_token.clone(),
                            also,
                            relation: rel_label,
                            c_score: hit.gate_score,
                        });
                    }
                }
            }
        } else {
            // Standard scan: iterate features via feature_meta() which
            // handles both heap and mmap modes. Earlier versions used
            // down_meta_at() which only reads heap-side metadata and
            // returned empty results on mmap-mode vindexes.
            for layer in &scan_layers {
                let nf = patched.num_features(*layer);
                for feat_idx in 0..nf {
                    if let Some(feature_f) = feature_filter {
                        if feat_idx != feature_f {
                            continue;
                        }
                    }
                    if let Some(meta) = patched.feature_meta(*layer, feat_idx) {
                        if let Some(ent) = entity_filter {
                            if !meta.top_token.to_lowercase().contains(&ent.to_lowercase()) {
                                continue;
                            }
                        }
                        let rel_label = classifier
                            .and_then(|rc| rc.label_for_feature(*layer, feat_idx))
                            .unwrap_or("")
                            .to_string();
                        if let Some(rel) = relation_filter {
                            if rel_label.is_empty() {
                                continue;
                            }
                            let rel_norm = rel.to_lowercase();
                            let label_norm = rel_label.to_lowercase();
                            if !label_norm.contains(&rel_norm) && !rel_norm.contains(&label_norm) {
                                continue;
                            }
                        }
                        let also = meta
                            .top_k
                            .iter()
                            .skip(1)
                            .take(3)
                            .map(|e| e.token.clone())
                            .collect::<Vec<_>>()
                            .join(", ");
                        rows.push(Row {
                            layer: *layer,
                            feature: feat_idx,
                            top_token: meta.top_token.clone(),
                            also,
                            relation: rel_label,
                            c_score: meta.c_score,
                        });
                    }
                }
            }
        }

        if let Some(ord) = order {
            match ord.field.as_str() {
                "confidence" | "c_score" => {
                    rows.sort_by(|a, b| {
                        let cmp = a
                            .c_score
                            .partial_cmp(&b.c_score)
                            .unwrap_or(std::cmp::Ordering::Equal);
                        if ord.descending {
                            cmp.reverse()
                        } else {
                            cmp
                        }
                    });
                }
                "layer" => {
                    rows.sort_by(|a, b| {
                        let cmp = a.layer.cmp(&b.layer);
                        if ord.descending {
                            cmp.reverse()
                        } else {
                            cmp
                        }
                    });
                }
                _ => {}
            }
        }

        // Apply score filter (WHERE score > N / score < N).
        if let Some((ref op, threshold)) = score_filter {
            rows.retain(|r| match op {
                CompareOp::Gt => r.c_score > threshold,
                CompareOp::Lt => r.c_score < threshold,
                CompareOp::Gte => r.c_score >= threshold,
                CompareOp::Lte => r.c_score <= threshold,
                CompareOp::Eq => (r.c_score - threshold).abs() < 0.001,
                _ => true,
            });
        }

        rows.truncate(limit);

        let show_relation =
            relation_filter.is_some() || rows.iter().any(|r| !r.relation.is_empty());
        let show_also = rows.iter().any(|r| !r.also.is_empty());

        let mut out = Vec::new();
        if show_relation {
            if show_also {
                out.push(format!(
                    "{:<8} {:<8} {:<16} {:<28} {:<14} {:>8}",
                    "Layer", "Feature", "Token", "Also", "Relation", "Score"
                ));
                out.push("-".repeat(86));
            } else {
                out.push(format!(
                    "{:<8} {:<8} {:<20} {:<20} {:>10}",
                    "Layer", "Feature", "Token", "Relation", "Score"
                ));
                out.push("-".repeat(70));
            }
        } else if show_also {
            out.push(format!(
                "{:<8} {:<8} {:<16} {:<28} {:>8}",
                "Layer", "Feature", "Token", "Also", "Score"
            ));
            out.push("-".repeat(72));
        } else {
            out.push(format!(
                "{:<8} {:<8} {:<20} {:>10}",
                "Layer", "Feature", "Token", "Score"
            ));
            out.push("-".repeat(50));
        }

        for row in &rows {
            let also_display = if row.also.is_empty() {
                String::new()
            } else {
                format!("[{}]", row.also)
            };
            if show_relation {
                if show_also {
                    out.push(format!(
                        "L{:<7} F{:<7} {:16} {:28} {:14} {:>8.4}",
                        row.layer,
                        row.feature,
                        row.top_token,
                        also_display,
                        row.relation,
                        row.c_score
                    ));
                } else {
                    out.push(format!(
                        "L{:<7} F{:<7} {:20} {:20} {:>10.4}",
                        row.layer, row.feature, row.top_token, row.relation, row.c_score
                    ));
                }
            } else if show_also {
                out.push(format!(
                    "L{:<7} F{:<7} {:16} {:28} {:>8.4}",
                    row.layer, row.feature, row.top_token, also_display, row.c_score
                ));
            } else {
                out.push(format!(
                    "L{:<7} F{:<7} {:20} {:>10.4}",
                    row.layer, row.feature, row.top_token, row.c_score
                ));
            }
        }

        if rows.is_empty() {
            out.push("  (no matching edges)".into());
        }

        Ok(out)
    }

    /// SELECT NEAREST TO — KNN lookup at a specific layer.
    fn exec_select_nearest(
        &self,
        index: &larql_vindex::PatchedVindex,
        path: &std::path::Path,
        nc: &NearestClause,
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let limit = limit.unwrap_or(20) as usize;

        let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
            .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

        let encoding = tokenizer
            .encode(nc.entity.as_str(), false)
            .map_err(|e| LqlError::exec("tokenize error", e))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        if token_ids.is_empty() {
            return Ok(vec!["  (entity not found)".into()]);
        }

        // Build query from entity embedding
        let hidden = embed.shape()[1];
        let query = if token_ids.len() == 1 {
            embed.row(token_ids[0] as usize).mapv(|v| v * embed_scale)
        } else {
            let mut avg = larql_vindex::ndarray::Array1::<f32>::zeros(hidden);
            for &tok in &token_ids {
                avg += &embed.row(tok as usize).mapv(|v| v * embed_scale);
            }
            avg /= token_ids.len() as f32;
            avg
        };

        // KNN at the specified layer
        let hits = index.gate_knn(nc.layer as usize, &query, limit);

        let classifier = self.relation_classifier();

        let mut out = Vec::new();
        out.push(format!(
            "{:<8} {:<8} {:<16} {:<28} {:<14} {:>8}",
            "Layer", "Feature", "Token", "Also", "Relation", "Score"
        ));
        out.push("-".repeat(86));

        for (feat, score) in &hits {
            let meta = index.feature_meta(nc.layer as usize, *feat);
            let tok = meta
                .as_ref()
                .map(|m| m.top_token.clone())
                .unwrap_or_else(|| "-".into());
            let also = meta
                .as_ref()
                .map(|m| {
                    let items: Vec<_> = m
                        .top_k
                        .iter()
                        .skip(1)
                        .take(3)
                        .map(|e| e.token.clone())
                        .collect();
                    if items.is_empty() {
                        String::new()
                    } else {
                        format!("[{}]", items.join(", "))
                    }
                })
                .unwrap_or_default();
            let rel = classifier
                .and_then(|rc| rc.label_for_feature(nc.layer as usize, *feat))
                .unwrap_or("");
            out.push(format!(
                "L{:<7} F{:<7} {:16} {:28} {:14} {:>8.4}",
                nc.layer, feat, tok, also, rel, score
            ));
        }

        if hits.is_empty() {
            out.push("  (no matching features)".into());
        }

        Ok(out)
    }

    // ── SELECT * FROM FEATURES ──

    pub(crate) fn exec_select_features(
        &self,
        conditions: &[Condition],
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let (_path, config, patched) = self.require_vindex()?;
        let classifier = self.relation_classifier();

        let layer_filter = conditions
            .iter()
            .find(|c| c.field == "layer")
            .and_then(|c| {
                if let Value::Integer(n) = c.value {
                    Some(n as usize)
                } else {
                    None
                }
            });
        let feature_filter = conditions
            .iter()
            .find(|c| c.field == "feature")
            .and_then(|c| {
                if let Value::Integer(n) = c.value {
                    Some(n as usize)
                } else {
                    None
                }
            });
        let token_filter = conditions
            .iter()
            .find(|c| c.field == "token" || c.field == "entity")
            .and_then(|c| {
                if let Value::String(ref s) = c.value {
                    Some(s.as_str())
                } else {
                    None
                }
            });

        let default_limit = if feature_filter.is_some() {
            config.num_layers
        } else if layer_filter.is_some() {
            config.intermediate_size
        } else {
            34
        };
        let limit = limit.unwrap_or(default_limit as u32) as usize;

        let scan_layers: Vec<usize> = if let Some(l) = layer_filter {
            vec![l]
        } else {
            (0..config.num_layers).collect()
        };

        let mut out = Vec::new();
        out.push(format!(
            "{:<8} {:<8} {:<16} {:<28} {:<14} {:>8}",
            "Layer", "Feature", "Token", "Also", "Relation", "Score"
        ));
        out.push("-".repeat(86));

        let mut count = 0;
        for layer in &scan_layers {
            let nf = patched.num_features(*layer);
            for feat in 0..nf {
                if count >= limit {
                    break;
                }
                if let Some(ff) = feature_filter {
                    if feat != ff {
                        continue;
                    }
                }
                if let Some(meta) = patched.feature_meta(*layer, feat) {
                    if let Some(tf) = token_filter {
                        if meta.top_token.to_lowercase() != tf.to_lowercase() {
                            continue;
                        }
                    }
                    let also: String = meta
                        .top_k
                        .iter()
                        .skip(1)
                        .take(3)
                        .map(|e| e.token.clone())
                        .collect::<Vec<_>>()
                        .join(", ");
                    let also_display = if also.is_empty() {
                        String::new()
                    } else {
                        format!("[{}]", also)
                    };
                    let rel = classifier
                        .and_then(|rc| rc.label_for_feature(*layer, feat))
                        .unwrap_or("");
                    out.push(format!(
                        "L{:<7} F{:<7} {:16} {:28} {:14} {:>8.4}",
                        layer, feat, meta.top_token, also_display, rel, meta.c_score
                    ));
                    count += 1;
                }
            }
            if count >= limit {
                break;
            }
        }

        if count == 0 {
            out.push("  (no matching features)".into());
        }

        Ok(out)
    }

    // ── SELECT * FROM ENTITIES ──

    pub(crate) fn exec_select_entities(
        &self,
        conditions: &[Condition],
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let (_path, config, patched) = self.require_vindex()?;

        let layer_filter = conditions
            .iter()
            .find(|c| c.field == "layer")
            .and_then(|c| {
                if let Value::Integer(n) = c.value {
                    Some(n as usize)
                } else {
                    None
                }
            });
        let entity_filter = conditions
            .iter()
            .find(|c| c.field == "entity" || c.field == "token")
            .and_then(|c| {
                if let Value::String(ref s) = c.value {
                    Some(s.as_str())
                } else {
                    None
                }
            });
        let limit = limit.unwrap_or(50) as usize;

        let scan_layers: Vec<usize> = if let Some(l) = layer_filter {
            vec![l]
        } else {
            (0..config.num_layers).collect()
        };

        // Common English stop words to filter out — these are capitalized
        // at sentence starts but aren't named entities.
        const STOP_WORDS: &[&str] = &[
            "The", "For", "And", "But", "Not", "This", "That", "With", "From", "Into", "Will",
            "Can", "One", "All", "Any", "Has", "Had", "Was", "Are", "Were", "Been", "His", "Her",
            "Its", "Our", "Who", "How", "Why", "When", "What", "Where", "Which", "Each", "Both",
            "Some", "Most", "Many", "Much", "More", "Such", "Than", "Then", "Also", "Just", "Now",
            "May", "Per", "Pre", "Pro", "Con", "Dis", "Via", "Yet", "Nor", "Should", "Would",
            "Could", "Did", "Does", "Too", "Very", "Instead", "Mon", "Three", "Four", "Five",
            "Six", "Seven", "Eight", "Nine", "Ten", "First", "Second", "Third", "Fourth", "Fifth",
            "Sixth", "Forty", "Fifty", "Only", "Over", "Under", "After", "Before", "About",
            "Above", "Below", "Between", "Through",
        ];

        // Collect distinct entity-like tokens.
        let mut entity_counts: std::collections::HashMap<String, (usize, f32)> =
            std::collections::HashMap::new();

        for layer in &scan_layers {
            let nf = patched.num_features(*layer);
            for feat in 0..nf {
                if let Some(meta) = patched.feature_meta(*layer, feat) {
                    let tok = meta.top_token.trim().to_string();
                    // Named entities: uppercase start, 3+ chars, all alphabetic.
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
                    if STOP_WORDS.contains(&tok.as_str()) {
                        continue;
                    }
                    // Entity name filter (WHERE entity = "X").
                    if let Some(ef) = entity_filter {
                        if !tok.to_lowercase().contains(&ef.to_lowercase()) {
                            continue;
                        }
                    }
                    let entry = entity_counts.entry(tok).or_insert((0, 0.0));
                    entry.0 += 1;
                    if meta.c_score > entry.1 {
                        entry.1 = meta.c_score;
                    }
                }
            }
        }

        let mut entities: Vec<(String, usize, f32)> = entity_counts
            .into_iter()
            .map(|(tok, (count, max_score))| (tok, count, max_score))
            .collect();
        entities.sort_by_key(|x| std::cmp::Reverse(x.1));
        entities.truncate(limit);

        let mut out = Vec::new();
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
}
