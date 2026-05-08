//! `DESCRIBE <entity>` — walk-based edge scan, MoE-aware.

use std::collections::HashMap;

use crate::ast::{DescribeMode, LayerBand};
use crate::error::LqlError;
use crate::executor::helpers::is_content_token;
use crate::executor::{Backend, Session};

use super::resolve_bands;

impl Session {
    pub(crate) fn exec_describe(
        &self,
        entity: &str,
        band: Option<LayerBand>,
        layer: Option<u32>,
        relations_only: bool,
        mode: DescribeMode,
    ) -> Result<Vec<String>, LqlError> {
        let verbose = mode != DescribeMode::Brief;

        // MoE router-based DESCRIBE if available
        if let Some(router_result) = self.try_moe_describe(entity, band, layer, verbose)? {
            return Ok(router_result);
        }

        // ── Phase 1: load embeddings + tokenizer, build query vector ──
        let (path, config, patched) = self.require_vindex()?;
        let query = describe_build_query(entity, path)?;

        if query.is_none() {
            return Ok(vec![format!("{entity}\n  (not found)")]);
        }
        let (query, token_ids) = query.unwrap();

        // ── Phase 2: pick scan layers from band/layer filter ──
        let bands = resolve_bands(config);
        let scan_layers = describe_scan_layers(&bands, &patched.loaded_layers(), band, layer);

        // ── Phase 3: walk + collect edges ──
        let trace = patched.walk(&query, &scan_layers, 20);
        let trace = describe_resolve_gguf_labels(path, config.vocab_size, &token_ids, trace);
        let manifest_browse = path.join("gguf_down_meta_manifest.json").is_file();
        let mut edges = describe_collect_edges(&trace, entity, manifest_browse);

        // ── Phase 3b: append KNN store entries for this entity ──
        let knn_hits = patched.knn_store.entries_for_entity(entity);
        for (knn_layer, entry) in knn_hits {
            edges.push(DescribeEdge {
                gate: entry.confidence * 10.0, // scale to match gate score range
                layers: vec![knn_layer],
                count: 1,
                original: entry.target_token.clone(),
                also: vec![format!("[knn:{}]", entry.relation)],
                best_layer: knn_layer,
                best_feature: 0,
            });
        }

        // ── Phase 4: format ──
        let mut out = vec![entity.to_string()];
        if edges.is_empty() {
            out.push("  (no edges found)".into());
            return Ok(out);
        }

        // Signal strength indicator: helps users interpret noisy results
        // for abstract/functional tokens vs clean entity-level knowledge.
        let max_gate = edges.iter().map(|e| e.gate).fold(0.0_f32, f32::max);
        let edge_count = edges.len();
        let signal = if max_gate >= 20.0 {
            "clean"
        } else if max_gate >= 10.0 {
            "moderate"
        } else {
            "diffuse"
        };
        out.push(format!(
            "  signal: {} ({} edges, max gate {:.1})",
            signal, edge_count, max_gate,
        ));

        let formatted =
            describe_format_and_split(&edges, self.relation_classifier(), relations_only, &bands);

        let max_edges = if mode == DescribeMode::Brief { 10 } else { 30 };

        if !formatted.syntax.is_empty() {
            out.push(format!(
                "  Syntax (L{}-{}):",
                bands.syntax.0, bands.syntax.1
            ));
            for edge in formatted.syntax.iter().take(max_edges) {
                out.push(format_describe_edge(edge, mode));
            }
        }
        if !formatted.knowledge.is_empty() {
            out.push(format!(
                "  Edges (L{}-{}):",
                bands.knowledge.0, bands.knowledge.1
            ));
            for edge in formatted.knowledge.iter().take(max_edges) {
                out.push(format_describe_edge(edge, mode));
            }
        }
        if !formatted.output_band.is_empty() {
            out.push(format!(
                "  Output (L{}-{}):",
                bands.output.0, bands.output.1
            ));
            let cap = if mode == DescribeMode::Brief {
                5
            } else {
                max_edges
            };
            for edge in formatted.output_band.iter().take(cap) {
                out.push(format_describe_edge(edge, mode));
            }
        }

        Ok(out)
    }

    // ── MoE Router-guided DESCRIBE ──

    /// For MoE models: use the router to select experts, then gate KNN within
    /// only the selected experts' features. Same output format as dense DESCRIBE.
    /// Returns None if no router (dense model — falls through to standard gate KNN).
    fn try_moe_describe(
        &self,
        entity: &str,
        _band: Option<LayerBand>,
        _layer: Option<u32>,
        verbose: bool,
    ) -> Result<Option<Vec<String>>, LqlError> {
        let router = match &self.backend {
            Backend::Vindex {
                router: Some(r),
                config,
                ..
            } => {
                if config
                    .model_config
                    .as_ref()
                    .and_then(|mc| mc.moe.as_ref())
                    .is_none()
                {
                    return Ok(None);
                }
                r
            }
            _ => return Ok(None),
        };

        let (path, config, _) = self.require_vindex()?;

        let (embed, embed_scale) = larql_vindex::load_vindex_embeddings(path)
            .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

        let encoding = tokenizer
            .encode(entity, false)
            .map_err(|e| LqlError::exec("tokenize error", e))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        if token_ids.is_empty() {
            return Ok(Some(vec![format!("{entity}\n  (not found)")]));
        }

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

        let last = config.num_layers.saturating_sub(1);
        let bands = config
            .layer_bands
            .clone()
            .or_else(|| larql_vindex::LayerBands::for_family(&config.family, config.num_layers))
            .unwrap_or(larql_vindex::LayerBands {
                syntax: (0, last),
                knowledge: (0, last),
                output: (0, last),
            });

        let start = std::time::Instant::now();

        // ── Per-layer expert routing ──
        let mut out = vec![entity.to_string()];

        // Aggregate: which experts are most active across the knowledge band?
        let knowledge_range = bands.knowledge.0..=bands.knowledge.1;
        let expert_summary = router.route_all_layers(&query, knowledge_range.clone());

        // Show per-layer routing in verbose mode
        if verbose {
            out.push(format!(
                "  Routing (L{}-{}):",
                bands.knowledge.0, bands.knowledge.1
            ));
            for l in knowledge_range.clone() {
                if let Some(result) = router.route(l, &query) {
                    let experts_str: String = result
                        .experts
                        .iter()
                        .enumerate()
                        .map(|(i, e)| format!("E{} ({:.0}%)", e, result.probs[i] * 100.0))
                        .collect::<Vec<_>>()
                        .join(", ");
                    out.push(format!("    L{:2}: {}", l, experts_str));
                }
            }
            out.push(String::new());
        }

        // ── Expert summary ──
        let layers_total = bands.knowledge.1 - bands.knowledge.0 + 1;
        out.push(format!(
            "  Experts (L{}-{}):",
            bands.knowledge.0, bands.knowledge.1
        ));
        let max_experts = if verbose { 15 } else { 6 };
        for (eid, count, avg_prob) in expert_summary.iter().take(max_experts) {
            out.push(format!(
                "    E{:<4} {}/{} layers  ({:.0}% avg)",
                eid,
                count,
                layers_total,
                avg_prob * 100.0,
            ));
        }

        // ── Co-routed entities: what else routes to the same experts? ──
        let top_experts: Vec<usize> = expert_summary.iter().take(3).map(|(e, _, _)| *e).collect();

        if !top_experts.is_empty() {
            out.push(String::new());
            out.push("  Similar (shares experts):".into());

            let mid_layer = (bands.knowledge.0 + bands.knowledge.1) / 2;

            // Sample vocab and find entities that route to the same experts
            let sample_step = (embed.shape()[0] / 2000).max(1);
            let mut corouted_all: HashMap<usize, Vec<(String, f32)>> = HashMap::new();

            for tid in (0..embed.shape()[0]).step_by(sample_step) {
                let tok_emb = embed.row(tid).mapv(|v| v * embed_scale);
                if let Some(result) = router.route(mid_layer, &tok_emb) {
                    for (i, &eid) in result.experts.iter().enumerate() {
                        if top_experts.contains(&eid) {
                            let tok_str = tokenizer
                                .decode(&[tid as u32], true)
                                .unwrap_or_default()
                                .trim()
                                .to_string();
                            if is_content_token(&tok_str)
                                && tok_str.len() > 1
                                && tok_str.to_lowercase() != entity.to_lowercase()
                            {
                                corouted_all
                                    .entry(eid)
                                    .or_default()
                                    .push((tok_str, result.probs[i]));
                            }
                        }
                    }
                }
            }

            for &eid in &top_experts {
                if let Some(tokens) = corouted_all.get_mut(&eid) {
                    tokens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    tokens.dedup_by(|a, b| a.0.to_lowercase() == b.0.to_lowercase());
                    let display: String = tokens
                        .iter()
                        .take(10)
                        .map(|(t, _)| t.as_str())
                        .collect::<Vec<_>>()
                        .join(", ");
                    out.push(format!("    E{}: {}", eid, display));
                }
            }
        }

        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
        out.push(format!("\n  {:.0}ms", elapsed_ms));

        Ok(Some(out))
    }
}

// ── DESCRIBE helpers ────────────────────────────────────────────────────
//
// `exec_describe` is a five-phase pipeline (load query → resolve bands →
// walk → collect edges → format). The helpers below split each phase out
// of the main function so the orchestration reads top-down.

type DescribeQuery = (larql_vindex::ndarray::Array1<f32>, Vec<u32>);

/// Tokenise `entity` and build a query vector by averaging its token
/// embeddings (single tokens get their embed row directly). Returns
/// `Ok(None)` when the entity tokenises to nothing — the caller emits
/// the "(not found)" line.
fn describe_build_query(
    entity: &str,
    path: &std::path::Path,
) -> Result<Option<DescribeQuery>, LqlError> {
    let tokenizer = larql_vindex::load_vindex_tokenizer(path)
        .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;

    let encoding = tokenizer
        .encode(entity, false)
        .map_err(|e| LqlError::exec("tokenize error", e))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    if token_ids.is_empty() {
        return Ok(None);
    }

    let (embed, embed_scale) = larql_vindex::load_vindex_embedding_rows(path, &token_ids)
        .map_err(|e| LqlError::exec("failed to load embeddings", e))?;
    let hidden = embed.shape()[1];
    let query = if token_ids.len() == 1 {
        embed.row(0).mapv(|v| v * embed_scale)
    } else {
        let mut avg = larql_vindex::ndarray::Array1::<f32>::zeros(hidden);
        for row_idx in 0..token_ids.len() {
            let row = embed.row(row_idx);
            avg += &row.mapv(|v| v * embed_scale);
        }
        avg /= token_ids.len() as f32;
        avg
    };
    Ok(Some((query, token_ids)))
}

/// Filter `all_layers` down to those covered by the requested band /
/// explicit layer.
fn describe_scan_layers(
    bands: &larql_vindex::LayerBands,
    all_layers: &[usize],
    band: Option<LayerBand>,
    layer: Option<u32>,
) -> Vec<usize> {
    if let Some(l) = layer {
        return vec![l as usize];
    }
    match band {
        Some(LayerBand::Syntax) => all_layers
            .iter()
            .copied()
            .filter(|l| *l >= bands.syntax.0 && *l <= bands.syntax.1)
            .collect(),
        Some(LayerBand::Knowledge) => all_layers
            .iter()
            .copied()
            .filter(|l| *l >= bands.knowledge.0 && *l <= bands.knowledge.1)
            .collect(),
        Some(LayerBand::Output) => all_layers
            .iter()
            .copied()
            .filter(|l| *l >= bands.output.0 && *l <= bands.output.1)
            .collect(),
        Some(LayerBand::All) | None => all_layers.to_vec(),
    }
}

/// Per-target accumulator for the walk-collected edges.
struct DescribeEdge {
    gate: f32,
    layers: Vec<usize>,
    count: usize,
    original: String,
    also: Vec<String>,
    best_layer: usize,
    best_feature: usize,
}

/// A formatted edge ready to be rendered into the output buffer. Built
/// from a `DescribeEdge` by `describe_format_and_split` after label
/// resolution and the RELATIONS ONLY filter.
struct FormattedEdge {
    /// Probe label, raw cluster label, or empty when no label is known.
    label: String,
    is_probe: bool,
    is_cluster: bool,
    target: String,
    gate: f32,
    primary_layer: usize,
    layers: Vec<usize>,
    count: usize,
    also: Vec<String>,
}

/// The three formatted-edge buckets returned by
/// `describe_format_and_split`, one per layer band.
struct DescribeBands {
    syntax: Vec<FormattedEdge>,
    knowledge: Vec<FormattedEdge>,
    output_band: Vec<FormattedEdge>,
}

fn describe_resolve_gguf_labels(
    path: &std::path::Path,
    vocab_size: usize,
    prompt_token_ids: &[u32],
    mut trace: larql_vindex::WalkTrace,
) -> larql_vindex::WalkTrace {
    let semantic_token_limit = std::env::var("LARQL_GGUF_MANIFEST_DOWN_META_TOKENS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(256)
        .min(vocab_size);
    let semantic_hit_limit = std::env::var("LARQL_GGUF_MANIFEST_DOWN_META_HITS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(12);
    if semantic_token_limit == 0 || semantic_hit_limit == 0 {
        return trace;
    }
    let Ok(tokenizer) = larql_vindex::load_vindex_tokenizer(path) else {
        return trace;
    };
    let decoded_vocab: Vec<(u32, String)> = tokenizer
        .get_vocab(true)
        .into_values()
        .filter(|&id| (id as usize) < vocab_size)
        .filter_map(|id| {
            tokenizer
                .decode(&[id], true)
                .ok()
                .map(|decoded| (id, decoded.trim().to_string()))
        })
        .collect();
    let candidates = if decoded_vocab.is_empty() {
        super::manifest_candidate_token_ids(prompt_token_ids, vocab_size, semantic_token_limit)
    } else {
        super::manifest_candidate_token_ids_with_labels(
            prompt_token_ids,
            vocab_size,
            semantic_token_limit,
            &decoded_vocab,
        )
    };
    let mut resolved = 0usize;
    for (layer_pos, hit_pos) in describe_resolution_order(&trace, semantic_hit_limit) {
        let layer = trace.layers[layer_pos].0;
        let hit = &mut trace.layers[layer_pos].1[hit_pos];
        if let Ok(mut meta) =
            larql_vindex::load_vindex_gguf_feature_meta(path, layer, hit.feature, &candidates, 4)
        {
            for entry in &mut meta.top_k {
                if let Ok(decoded) = tokenizer.decode(&[entry.token_id], true) {
                    let decoded = decoded.trim().to_string();
                    if !decoded.is_empty() {
                        entry.token = decoded;
                    }
                }
            }
            meta = super::prefer_readable_feature_meta(meta);
            hit.meta = meta;
            resolved += 1;
            if resolved >= semantic_hit_limit {
                break;
            }
        }
    }
    trace
}

fn describe_resolution_order(trace: &larql_vindex::WalkTrace, limit: usize) -> Vec<(usize, usize)> {
    let mut order: Vec<(usize, usize, f32)> = trace
        .layers
        .iter()
        .enumerate()
        .flat_map(|(layer_pos, (_layer, hits))| {
            hits.iter()
                .enumerate()
                .map(move |(hit_pos, hit)| (layer_pos, hit_pos, hit.gate_score.abs()))
        })
        .collect();
    order.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    order
        .into_iter()
        .take(limit)
        .map(|(layer_pos, hit_pos, _)| (layer_pos, hit_pos))
        .collect()
}

/// Walk the trace, deduplicate by lowercased target token, and apply
/// content / coherence filters. The output is sorted descending by gate.
fn describe_collect_edges(
    trace: &larql_vindex::WalkTrace,
    entity: &str,
    manifest_browse: bool,
) -> Vec<DescribeEdge> {
    let entity_lower = entity.to_lowercase();
    let gate_threshold = if manifest_browse { 0.0_f32 } else { 5.0_f32 };
    let mut edges: HashMap<String, DescribeEdge> = HashMap::new();

    for (layer_idx, hits) in &trace.layers {
        for hit in hits {
            if hit.gate_score < gate_threshold {
                continue;
            }
            let tok = &hit.meta.top_token;
            if manifest_browse {
                if !super::is_manifest_label_token(tok) {
                    continue;
                }
            } else if !is_content_token(tok) {
                continue;
            }
            if tok.to_lowercase() == entity_lower {
                continue;
            }

            let also_readable: Vec<String> = hit
                .meta
                .top_k
                .iter()
                .filter(|t| {
                    t.token.to_lowercase() != tok.to_lowercase()
                        && t.token.to_lowercase() != entity_lower
                        && crate::executor::helpers::is_readable_token(&t.token)
                        && t.logit > 0.0
                })
                .take(5)
                .map(|t| t.token.clone())
                .collect();

            let also: Vec<String> = also_readable
                .iter()
                .filter(|t| is_content_token(t))
                .take(3)
                .cloned()
                .collect();

            // Coherence filter: skip weak dense-vindex edges with no content secondaries.
            // Manifest-backed Kimi/DeepSeek2 browse scans use bounded gate/down-meta
            // projections whose scores are intentionally much smaller, so keep
            // readable selected hits instead of requiring dense-vindex magnitudes.
            if !manifest_browse
                && also.is_empty()
                && !also_readable.is_empty()
                && hit.gate_score < 20.0
            {
                continue;
            }

            let key = tok.to_lowercase();
            let entry = edges.entry(key).or_insert_with(|| DescribeEdge {
                gate: 0.0,
                layers: Vec::new(),
                count: 0,
                original: tok.to_string(),
                also,
                best_layer: *layer_idx,
                best_feature: hit.feature,
            });

            if hit.gate_score > entry.gate {
                entry.gate = hit.gate_score;
                entry.best_layer = *layer_idx;
                entry.best_feature = hit.feature;
            }
            if !entry.layers.contains(layer_idx) {
                entry.layers.push(*layer_idx);
            }
            entry.count += 1;
        }
    }

    let mut ranked: Vec<DescribeEdge> = edges.into_values().collect();
    ranked.sort_by(|a, b| {
        b.gate
            .partial_cmp(&a.gate)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    ranked
}

/// Resolve relation labels from the optional `RelationClassifier`, apply
/// the RELATIONS ONLY filter, and split the resulting `FormattedEdge`s
/// into syntax / knowledge / output buckets according to which band the
/// edge's primary layer falls in.
fn describe_format_and_split(
    edges: &[DescribeEdge],
    classifier: Option<&crate::relations::RelationClassifier>,
    relations_only: bool,
    bands: &larql_vindex::LayerBands,
) -> DescribeBands {
    let formatted: Vec<FormattedEdge> = edges
        .iter()
        .map(|info| {
            let (label, is_probe, is_cluster) = if let Some(rc) = classifier {
                if let Some(lbl) = rc.label_for_feature(info.best_layer, info.best_feature) {
                    let probe = rc.is_probe_label(info.best_layer, info.best_feature);
                    (lbl.to_string(), probe, !probe)
                } else {
                    (String::new(), false, false)
                }
            } else {
                (String::new(), false, false)
            };
            FormattedEdge {
                label,
                is_probe,
                is_cluster,
                target: info.original.clone(),
                gate: info.gate,
                primary_layer: info.best_layer,
                layers: info.layers.clone(),
                count: info.count,
                also: info.also.clone(),
            }
        })
        .filter(|e| !relations_only || e.is_probe || e.is_cluster)
        .collect();

    let mut out = DescribeBands {
        syntax: Vec::new(),
        knowledge: Vec::new(),
        output_band: Vec::new(),
    };
    for edge in formatted {
        let primary = edge.primary_layer;
        if primary >= bands.syntax.0 && primary <= bands.syntax.1 {
            out.syntax.push(edge);
        } else if primary >= bands.knowledge.0 && primary <= bands.knowledge.1 {
            out.knowledge.push(edge);
        } else if primary >= bands.output.0 && primary <= bands.output.1 {
            out.output_band.push(edge);
        } else {
            // Layer outside any band — fall back to knowledge.
            out.knowledge.push(edge);
        }
    }
    out
}

/// Render a single `FormattedEdge` into a single line of DESCRIBE output.
/// The three modes share the same shape:
///
///   - **Verbose** (default): `[relation]    → target  gate  L20-L27  Nx  also: ...`
///   - **Brief**: compact `relation    → target  gate  L26`, no also-tokens
///   - **Raw**: no labels, otherwise like Verbose
fn format_describe_edge(edge: &FormattedEdge, mode: DescribeMode) -> String {
    match mode {
        DescribeMode::Verbose => {
            let bracket_label = if edge.label.is_empty() {
                format!("{:<14}", "[—]")
            } else {
                let tag = format!("[{}]", edge.label);
                format!("{:<14}", tag)
            };
            let (min_l, max_l) = layer_range(&edge.layers);
            let layer_str = if min_l == max_l {
                format!("L{min_l}")
            } else {
                format!("L{min_l}-{max_l}")
            };
            let also = format_also(&edge.also);
            format!(
                "    {} → {:20} {:>7.1}  {:<8} {}x{}",
                bracket_label, edge.target, edge.gate, layer_str, edge.count, also,
            )
        }
        DescribeMode::Brief => {
            let label = if edge.is_probe {
                format!("{:<12}", edge.label)
            } else {
                format!("{:<12}", "")
            };
            format!(
                "    {} → {:20} {:>7.1}  L{:<3}",
                label, edge.target, edge.gate, edge.primary_layer,
            )
        }
        DescribeMode::Raw => {
            let (min_l, max_l) = layer_range(&edge.layers);
            let layer_str = if min_l == max_l {
                format!("L{min_l}")
            } else {
                format!("L{min_l}-{max_l}")
            };
            let also = format_also(&edge.also);
            format!(
                "                 → {:20} {:>7.1}  {:<8} {}x{}",
                edge.target, edge.gate, layer_str, edge.count, also,
            )
        }
    }
}

fn layer_range(layers: &[usize]) -> (usize, usize) {
    let min_l = *layers.iter().min().unwrap_or(&0);
    let max_l = *layers.iter().max().unwrap_or(&0);
    (min_l, max_l)
}

fn format_also(also: &[String]) -> String {
    if also.is_empty() {
        String::new()
    } else {
        format!("  also: {}", also.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::{describe_collect_edges, describe_resolution_order};
    use larql_models::TopKEntry;
    use larql_vindex::{FeatureMeta, WalkHit, WalkTrace};

    #[test]
    fn describe_resolves_highest_gate_hits_across_layers_first() {
        let trace = WalkTrace {
            layers: vec![
                (
                    1,
                    vec![
                        walk_hit(1, 10, 0.1, "early"),
                        walk_hit(1, 11, 0.2, "early2"),
                    ],
                ),
                (
                    40,
                    vec![
                        walk_hit(40, 99, 9.0, "late"),
                        walk_hit(40, 100, 8.0, "late2"),
                    ],
                ),
            ],
        };

        let order = describe_resolution_order(&trace, 2);
        assert_eq!(order, vec![(1, 0), (1, 1)]);
    }

    fn walk_hit(layer: usize, feature: usize, gate_score: f32, token: &str) -> WalkHit {
        WalkHit {
            layer,
            feature,
            gate_score,
            meta: FeatureMeta {
                top_token: token.to_string(),
                top_token_id: 0,
                c_score: gate_score,
                top_k: vec![TopKEntry {
                    token: token.to_string(),
                    token_id: 0,
                    logit: gate_score,
                }],
            },
        }
    }

    #[test]
    fn manifest_browse_describe_keeps_low_score_readable_hits() {
        let trace = WalkTrace {
            layers: vec![(
                1,
                vec![WalkHit {
                    layer: 1,
                    feature: 7,
                    gate_score: 0.12,
                    meta: FeatureMeta {
                        top_token: "Paris".to_string(),
                        top_token_id: 42,
                        c_score: 0.12,
                        top_k: vec![
                            TopKEntry {
                                token: "Paris".to_string(),
                                token_id: 42,
                                logit: 0.12,
                            },
                            TopKEntry {
                                token: "capital".to_string(),
                                token_id: 43,
                                logit: 0.08,
                            },
                        ],
                    },
                }],
            )],
        };

        let dense_edges = describe_collect_edges(&trace, "France", false);
        assert!(dense_edges.is_empty());

        let manifest_edges = describe_collect_edges(&trace, "France", true);
        assert_eq!(manifest_edges.len(), 1);
        assert_eq!(manifest_edges[0].original, "Paris");
        assert_eq!(manifest_edges[0].also, vec!["capital".to_string()]);
        assert_eq!(manifest_edges[0].best_layer, 1);
        assert_eq!(manifest_edges[0].best_feature, 7);
    }
}
