//! Read-side remote forwarders: DESCRIBE, WALK, INFER, EXPLAIN INFER,
//! STATS, SHOW RELATIONS, SELECT.

use crate::ast::{LayerBand, Range};
use crate::error::LqlError;
use crate::executor::predicate::{EdgePredicate, PredicateScope, RelationOp, ScoreOp};
use crate::executor::{Backend, Session};

use super::{
    ENDPOINT_DESCRIBE, ENDPOINT_EXPLAIN_INFER, ENDPOINT_INFER, ENDPOINT_RELATIONS, ENDPOINT_SELECT,
    ENDPOINT_STATS, ENDPOINT_WALK,
};

/// Default `top` for `WALK` when the user doesn't specify one.
const REMOTE_WALK_DEFAULT_TOP: u32 = 10;
/// Default `TOP N` for INFER and EXPLAIN INFER.
const REMOTE_INFER_DEFAULT_TOP: u32 = 5;
/// Default per-layer features for EXPLAIN INFER.
const REMOTE_EXPLAIN_PER_LAYER_DEFAULT: u32 = 3;
/// Default `LIMIT` for remote SELECT.
const REMOTE_SELECT_DEFAULT_LIMIT: u32 = 20;

fn band_str(band: Option<LayerBand>, default_when_none: &'static str) -> &'static str {
    match band {
        Some(LayerBand::Syntax) => "syntax",
        Some(LayerBand::Knowledge) => "knowledge",
        Some(LayerBand::Output) => "output",
        Some(LayerBand::All) => "all",
        None => default_when_none,
    }
}

fn band_label(band: Option<LayerBand>) -> &'static str {
    match band {
        Some(LayerBand::Syntax) => " (syntax)",
        Some(LayerBand::Knowledge) => " (knowledge)",
        Some(LayerBand::Output) => " (output)",
        _ => "",
    }
}

impl Session {
    pub(crate) fn remote_describe(
        &self,
        entity: &str,
        band: Option<LayerBand>,
        mode: crate::ast::DescribeMode,
    ) -> Result<Vec<String>, LqlError> {
        let verbose = mode == crate::ast::DescribeMode::Verbose;
        let show_also = matches!(
            mode,
            crate::ast::DescribeMode::Verbose | crate::ast::DescribeMode::Raw
        );

        let body = self.remote_get_json(
            ENDPOINT_DESCRIBE,
            &[
                ("entity", entity),
                ("band", band_str(band, "knowledge")),
                ("verbose", if verbose { "true" } else { "false" }),
            ],
        )?;

        let mut out = vec![entity.to_string()];

        if let Some(edges) = body["edges"].as_array() {
            if edges.is_empty() {
                out.push("  (no edges found)".into());
            } else {
                for edge in edges {
                    let target = edge["target"].as_str().unwrap_or("?");
                    let gate = edge["gate_score"].as_f64().unwrap_or(0.0);
                    let layer = edge["layer"].as_u64().unwrap_or(0);
                    let relation = edge["relation"].as_str().unwrap_or("");
                    let source = edge["source"].as_str().unwrap_or("");

                    let show_labels = mode != crate::ast::DescribeMode::Raw;
                    let label = if show_labels && !relation.is_empty() {
                        format!("{:<12}", relation)
                    } else {
                        format!("{:<12}", "")
                    };

                    let tag = if show_labels && source == "probe" {
                        "  (probe)"
                    } else {
                        ""
                    };

                    let also_str = if show_also {
                        edge["also"]
                            .as_array()
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            })
                            .filter(|s| !s.is_empty())
                            .map(|s| format!("  also: {s}"))
                            .unwrap_or_default()
                    } else {
                        String::new()
                    };

                    out.push(format!(
                        "    {} → {:20} {:>7.1}  L{:<3}{}{}",
                        label, target, gate, layer, tag, also_str,
                    ));
                }
            }
        }

        if let Some(ms) = body["latency_ms"].as_f64() {
            out.push(format!("\n{:.1}ms (remote)", ms));
        }

        // Overlay local patch edges.
        if let Backend::Remote { local_patches, .. } = &self.backend {
            if !local_patches.is_empty() {
                let entity_lower = entity.to_lowercase();
                let mut local_edges = Vec::new();
                for patch in local_patches {
                    for op in &patch.operations {
                        if let larql_vindex::PatchOp::Insert {
                            entity: ent,
                            target,
                            relation,
                            layer,
                            confidence,
                            ..
                        } = op
                        {
                            if ent.to_lowercase() == entity_lower {
                                local_edges.push((
                                    relation.as_deref().unwrap_or(""),
                                    target.as_str(),
                                    *layer,
                                    confidence.unwrap_or(0.9),
                                ));
                            }
                        }
                    }
                }
                if !local_edges.is_empty() {
                    out.push("  Local patch edges:".into());
                    for (relation, target, layer, conf) in &local_edges {
                        let label = if relation.is_empty() {
                            format!("{:<12}", "")
                        } else {
                            format!("{:<12}", relation)
                        };
                        out.push(format!(
                            "    {} → {:20} {:>7.2}  L{:<3}  (local)",
                            label, target, conf, layer,
                        ));
                    }
                }
            }
        }

        Ok(out)
    }

    pub(crate) fn remote_walk(
        &self,
        prompt: &str,
        top: Option<u32>,
        layers: Option<&Range>,
    ) -> Result<Vec<String>, LqlError> {
        let top_k = top.unwrap_or(REMOTE_WALK_DEFAULT_TOP).to_string();
        let layers_str = layers.map(|r| format!("{}-{}", r.start, r.end));

        let mut params: Vec<(&str, &str)> = vec![("prompt", prompt), ("top", top_k.as_str())];
        if let Some(ref s) = layers_str {
            params.push(("layers", s.as_str()));
        }

        let body = self.remote_get_json(ENDPOINT_WALK, &params)?;

        let mut out = Vec::new();
        out.push(format!("Feature scan for {:?}", prompt));
        out.push(String::new());

        if let Some(hits) = body["hits"].as_array() {
            for hit in hits {
                let layer = hit["layer"].as_u64().unwrap_or(0);
                let feature = hit["feature"].as_u64().unwrap_or(0);
                let gate = hit["gate_score"].as_f64().unwrap_or(0.0);
                let target = hit["target"].as_str().unwrap_or("?");

                out.push(format!(
                    "  L{:2}: F{:<5} gate={:+.1}  top={:?}",
                    layer, feature, gate, target,
                ));
            }
        }

        if let Some(ms) = body["latency_ms"].as_f64() {
            out.push(format!("\n{:.1}ms (remote)", ms));
        }

        Ok(out)
    }

    pub(crate) fn remote_infer(
        &self,
        prompt: &str,
        top: Option<u32>,
        compare: bool,
    ) -> Result<Vec<String>, LqlError> {
        let mode = if compare { "compare" } else { "walk" };
        let request = serde_json::json!({
            "prompt": prompt,
            "top": top.unwrap_or(REMOTE_INFER_DEFAULT_TOP),
            "mode": mode,
        });

        let result = self.remote_post_json(ENDPOINT_INFER, &request, true)?;

        let mut out = Vec::new();

        if compare {
            for mode in &["walk", "dense"] {
                if let Some(preds) = result[mode].as_array() {
                    out.push(format!("Predictions ({mode}):"));
                    for (i, p) in preds.iter().enumerate() {
                        let tok = p["token"].as_str().unwrap_or("?");
                        let prob = p["probability"].as_f64().unwrap_or(0.0);
                        out.push(format!("  {:2}. {:20} ({:.2}%)", i + 1, tok, prob * 100.0));
                    }
                    if let Some(ms) = result[format!("{mode}_ms")].as_f64() {
                        out.push(format!("  {:.0}ms", ms));
                    }
                    out.push(String::new());
                }
            }
        } else if let Some(preds) = result["predictions"].as_array() {
            out.push("Predictions (walk FFN):".into());
            for (i, p) in preds.iter().enumerate() {
                let tok = p["token"].as_str().unwrap_or("?");
                let prob = p["probability"].as_f64().unwrap_or(0.0);
                out.push(format!("  {:2}. {:20} ({:.2}%)", i + 1, tok, prob * 100.0));
            }
        }

        if let Some(override_obj) = result["knn_override"].as_object() {
            out.push(remote_knn_override_line(override_obj));
            out.push(
                "note: KNN override is a post-logits retrieval sidecar, not an FFN/residual edit."
                    .into(),
            );
        }

        if let Some(ms) = result["latency_ms"].as_f64() {
            out.push(format!("{:.0}ms (remote)", ms));
        }

        Ok(out)
    }

    pub(crate) fn remote_explain_infer(
        &self,
        prompt: &str,
        top: Option<u32>,
        band: Option<LayerBand>,
        relations_only: bool,
        with_attention: bool,
    ) -> Result<Vec<String>, LqlError> {
        let per_layer = top.unwrap_or(REMOTE_EXPLAIN_PER_LAYER_DEFAULT);

        let request = serde_json::json!({
            "prompt": prompt,
            "top": top.unwrap_or(REMOTE_INFER_DEFAULT_TOP),
            "per_layer": per_layer,
            "band": band_str(band, "all"),
            "relations_only": relations_only,
            "with_attention": with_attention,
        });

        let result = self.remote_post_json(ENDPOINT_EXPLAIN_INFER, &request, false)?;

        let mut out = Vec::new();
        out.push(format!(
            "Inference trace for {:?}{}:",
            prompt,
            band_label(band)
        ));

        if let Some(override_obj) = result["knn_override"].as_object() {
            let tok = override_obj
                .get("token")
                .and_then(|v| v.as_str())
                .unwrap_or("?");
            out.push(format!(
                "Prediction: {} ({})",
                tok,
                remote_knn_override_summary(override_obj)
            ));
            out.push(
                "Pending retrieval override: not part of the residual/FFN trace until materialized."
                    .into(),
            );
        } else if let Some(preds) = result["predictions"].as_array() {
            if let Some(first) = preds.first() {
                let tok = first["token"].as_str().unwrap_or("?");
                let prob = first["probability"].as_f64().unwrap_or(0.0);
                out.push(format!("Prediction: {} ({:.2}%)", tok, prob * 100.0));
            }
        }
        out.push(String::new());

        if let Some(layers) = result["trace"].as_array() {
            for layer_obj in layers {
                let layer = layer_obj["layer"].as_u64().unwrap_or(0);
                let features = layer_obj["features"].as_array();

                if with_attention {
                    let feat = features.and_then(|f| f.first());
                    let feature_str = if let Some(feat) = feat {
                        let relation = feat["relation"]
                            .as_str()
                            .or_else(|| feat["relation"].as_null().map(|_| ""))
                            .unwrap_or("");
                        if relations_only && relation.is_empty() {
                            None
                        } else {
                            let gate = feat["gate_score"].as_f64().unwrap_or(0.0);
                            let top_token = feat["top_token"].as_str().unwrap_or("?");
                            let name = if !relation.is_empty() {
                                relation
                            } else {
                                top_token
                            };
                            Some(format!("{:<14} {:+.1}", name, gate))
                        }
                    } else {
                        None
                    };
                    let empty = format!("{:19}", "");
                    let feature_part = feature_str.as_deref().unwrap_or(&empty);

                    let attn_part = layer_obj
                        .get("attention")
                        .and_then(|a| a.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|v| {
                            let tok = v["token"].as_str()?;
                            let w = v["weight"].as_f64()?;
                            Some(format!("{}({:.0}%)", tok, w * 100.0))
                        })
                        .unwrap_or_default();

                    let lens_part = layer_obj
                        .get("lens")
                        .and_then(|l| {
                            let tok = l["token"].as_str()?;
                            let prob = l["probability"].as_f64()?;
                            Some(format!("{} ({:.1}%)", tok, prob * 100.0))
                        })
                        .unwrap_or_default();

                    if feature_str.is_some() || !lens_part.is_empty() {
                        out.push(format!(
                            "  L{:2}  {:<19}  {:<16} → {}",
                            layer, feature_part, attn_part, lens_part,
                        ));
                    }
                } else if let Some(features) = features {
                    for feat in features {
                        let feature = feat["feature"].as_u64().unwrap_or(0);
                        let gate = feat["gate_score"].as_f64().unwrap_or(0.0);
                        let relation = feat["relation"]
                            .as_str()
                            .or_else(|| feat["relation"].as_null().map(|_| ""))
                            .unwrap_or("");
                        if relations_only && relation.is_empty() {
                            continue;
                        }
                        let label_str = if relation.is_empty() {
                            format!("{:14}", "")
                        } else {
                            format!("{:<14}", relation)
                        };
                        let top_token = feat["top_token"].as_str().unwrap_or("?");
                        let top_tokens: String = feat["top_tokens"]
                            .as_array()
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|v| v.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            })
                            .unwrap_or_default();
                        out.push(format!(
                            "  L{:2}: {} F{:<5} gate={:+.1}  → {:15} [{}]",
                            layer, label_str, feature, gate, top_token, top_tokens,
                        ));
                    }
                }
            }
        }

        if let Some(ms) = result["latency_ms"].as_f64() {
            out.push(format!("\n{:.0}ms (remote)", ms));
        }

        Ok(out)
    }

    pub(crate) fn remote_stats(&self) -> Result<Vec<String>, LqlError> {
        let body = self.remote_get_json(ENDPOINT_STATS, &[])?;
        let url = match &self.backend {
            Backend::Remote { url, .. } => url.as_str(),
            _ => "?",
        };

        let mut out = Vec::new();
        out.push(format!("Model: {}", body["model"].as_str().unwrap_or("?")));
        out.push(format!(
            "Family: {}",
            body["family"].as_str().unwrap_or("?")
        ));
        out.push(format!("Layers: {}", body["layers"].as_u64().unwrap_or(0)));
        out.push(format!(
            "Features: {}",
            body["features"].as_u64().unwrap_or(0)
        ));
        out.push(format!(
            "Hidden: {}",
            body["hidden_size"].as_u64().unwrap_or(0)
        ));
        out.push(format!("Dtype: {}", body["dtype"].as_str().unwrap_or("?")));
        out.push(format!(
            "Extract level: {}",
            body["extract_level"].as_str().unwrap_or("?")
        ));

        if let Some(bands) = body.get("layer_bands") {
            if let (Some(s), Some(k), Some(o)) = (
                bands.get("syntax"),
                bands.get("knowledge"),
                bands.get("output"),
            ) {
                out.push(format!(
                    "Bands: syntax {}-{}, knowledge {}-{}, output {}-{}",
                    s[0], s[1], k[0], k[1], o[0], o[1]
                ));
            }
        }

        if let Some(loaded) = body.get("loaded") {
            out.push(format!(
                "Loaded: browse={}, inference={}",
                loaded["browse"].as_bool().unwrap_or(false),
                loaded["inference"].as_bool().unwrap_or(false),
            ));
        }

        out.push(format!("Remote: {url}"));

        Ok(out)
    }

    pub(crate) fn remote_show_relations(
        &self,
        mode: crate::ast::DescribeMode,
        with_examples: bool,
    ) -> Result<Vec<String>, LqlError> {
        use crate::ast::DescribeMode;
        let body = self.remote_get_json(ENDPOINT_RELATIONS, &[])?;

        let mut out = Vec::new();

        if mode != DescribeMode::Raw {
            if let Some(probes) = body["probe_relations"].as_array() {
                if !probes.is_empty() {
                    let probe_count = body["probe_count"].as_u64().unwrap_or(0);
                    out.push(format!("Probe-confirmed relations ({probe_count} labels):"));
                    out.push(format!("{:<25} {:>8}", "Relation", "Features"));
                    out.push("-".repeat(35));
                    for rel in probes {
                        let name = rel["name"].as_str().unwrap_or("?");
                        let count = rel["count"].as_u64().unwrap_or(0);
                        out.push(format!("{:<25} {:>8}", name, count));
                    }
                    out.push(String::new());
                }
            }
        }

        let show_raw = mode == DescribeMode::Raw || mode == DescribeMode::Verbose || out.is_empty();

        if show_raw {
            if let Some(rels) = body["relations"].as_array() {
                if !rels.is_empty() {
                    out.push("Top output tokens:".to_string());
                    out.push(format!(
                        "{:<25} {:>8} {:>8} {:>10}",
                        "Token", "Count", "Score", "Layers"
                    ));
                    out.push("-".repeat(55));
                    for rel in rels {
                        let name = rel["name"].as_str().unwrap_or("?");
                        let count = rel["count"].as_u64().unwrap_or(0);
                        let score = rel["max_score"].as_f64().unwrap_or(0.0);
                        let min_l = rel["min_layer"].as_u64().unwrap_or(0);
                        let max_l = rel["max_layer"].as_u64().unwrap_or(0);
                        let examples_str = if with_examples {
                            if let Some(arr) = rel["examples"].as_array() {
                                let ex: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
                                if ex.is_empty() {
                                    String::new()
                                } else {
                                    format!("  e.g. {}", ex.join(", "))
                                }
                            } else {
                                String::new()
                            }
                        } else {
                            String::new()
                        };
                        out.push(format!(
                            "{:<25} {:>8} {:>8.2} {:>5}-{}{}",
                            name, count, score, min_l, max_l, examples_str,
                        ));
                    }
                }
            }
        }

        Ok(out)
    }

    pub(crate) fn remote_select(
        &self,
        conditions: &[crate::ast::Condition],
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let mut body = serde_json::Map::new();
        body.insert(
            "limit".into(),
            serde_json::json!(limit.unwrap_or(REMOTE_SELECT_DEFAULT_LIMIT)),
        );

        // The same grammar as a local SELECT, validated before anything
        // is sent. The server's query takes `entity`, `relation`, `layer`
        // and a confidence floor; a condition it cannot express is
        // refused here rather than sent as something weaker.
        let predicate = EdgePredicate::parse(conditions, PredicateScope::EDGES)?;
        let not_remote = |what: &str| {
            LqlError::Execution(format!(
                "unsupported WHERE clause: a remote SELECT does not evaluate {what}; \
                 supported remotely: entity = <string>, relation = <string>, \
                 layer = <integer>, score >= <number>"
            ))
        };
        if predicate.feature.is_some() {
            return Err(not_remote("`feature`"));
        }
        if let Some(entity) = predicate.entity {
            body.insert("entity".into(), serde_json::json!(entity));
        }
        if let Some(relation) = predicate.relation {
            if relation.op != RelationOp::Eq {
                return Err(not_remote("`relation` with != or LIKE"));
            }
            body.insert("relation".into(), serde_json::json!(relation.wanted));
        }
        if let Some(layer) = predicate.layer {
            body.insert("layer".into(), serde_json::json!(layer));
        }
        if let Some(score) = predicate.score {
            if score.op != ScoreOp::Gte {
                return Err(not_remote("`score` with an operator other than >="));
            }
            body.insert("min_confidence".into(), serde_json::json!(score.threshold));
        }

        let result =
            self.remote_post_json(ENDPOINT_SELECT, &serde_json::Value::Object(body), false)?;

        let mut out = Vec::new();

        if let Some(edges) = result["edges"].as_array() {
            if edges.is_empty() {
                out.push("  (no matching edges)".into());
            } else {
                out.push(format!(
                    "  {:<20} {:<15} {:>6}  {:<6} {}",
                    "Target", "Relation", "Score", "Layer", "Feature"
                ));
                out.push(format!("  {}", "-".repeat(65)));
                for edge in edges {
                    let layer = edge["layer"].as_u64().unwrap_or(0);
                    let feature = edge["feature"].as_u64().unwrap_or(0);
                    let target = edge["target"].as_str().unwrap_or("?");
                    let score = edge["c_score"].as_f64().unwrap_or(0.0);
                    let relation = edge["relation"].as_str().unwrap_or("");
                    out.push(format!(
                        "  {:<20} {:<15} {:>6.3}  L{:<5} F{}",
                        target, relation, score, layer, feature
                    ));
                }
            }
        }

        if let Some(total) = result["total"].as_u64() {
            out.push(format!("\n{} total", total));
        }

        Ok(out)
    }
}

// ── KNN-override formatters ──────────────────────────────────────

pub(super) fn remote_knn_override_line(
    override_obj: &serde_json::Map<String, serde_json::Value>,
) -> String {
    let tok = override_obj
        .get("token")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    format!(
        "KNN override: {} ({})",
        tok,
        remote_knn_override_summary(override_obj)
    )
}

pub(super) fn remote_knn_override_summary(
    override_obj: &serde_json::Map<String, serde_json::Value>,
) -> String {
    let cosine = override_obj
        .get("cosine")
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0);
    let layer = override_obj
        .get("layer")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let mut summary = format!("source=knn_override/post_logits, cos={cosine:.2}, L{layer}");
    if let Some(model_top1) = override_obj.get("model_top1").and_then(|v| v.as_object()) {
        let tok = model_top1
            .get("token")
            .and_then(|v| v.as_str())
            .unwrap_or("?");
        let prob = model_top1
            .get("probability")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        summary.push_str(&format!(", model_top1={} ({:.2}%)", tok, prob * 100.0));
    }
    summary
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_obj(pairs: &[(&str, serde_json::Value)]) -> serde_json::Map<String, serde_json::Value> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn band_str_maps_each_variant() {
        assert_eq!(band_str(Some(LayerBand::Syntax), "x"), "syntax");
        assert_eq!(band_str(Some(LayerBand::Knowledge), "x"), "knowledge");
        assert_eq!(band_str(Some(LayerBand::Output), "x"), "output");
        assert_eq!(band_str(Some(LayerBand::All), "x"), "all");
        assert_eq!(band_str(None, "fallback"), "fallback");
    }

    #[test]
    fn band_label_brackets_known_bands_and_skips_all_or_none() {
        assert_eq!(band_label(Some(LayerBand::Syntax)), " (syntax)");
        assert_eq!(band_label(Some(LayerBand::Knowledge)), " (knowledge)");
        assert_eq!(band_label(Some(LayerBand::Output)), " (output)");
        assert_eq!(band_label(Some(LayerBand::All)), "");
        assert_eq!(band_label(None), "");
    }

    #[test]
    fn knn_override_summary_baseline_no_top1() {
        let obj = mk_obj(&[
            ("cosine", serde_json::json!(0.987)),
            ("layer", serde_json::json!(26)),
        ]);
        let s = remote_knn_override_summary(&obj);
        assert!(s.contains("source=knn_override/post_logits"));
        assert!(s.contains("cos=0.99"));
        assert!(s.contains("L26"));
        assert!(!s.contains("model_top1"));
    }

    #[test]
    fn knn_override_summary_appends_model_top1_when_present() {
        let obj = mk_obj(&[
            ("cosine", serde_json::json!(0.50)),
            ("layer", serde_json::json!(10)),
            (
                "model_top1",
                serde_json::json!({"token": "London", "probability": 0.42}),
            ),
        ]);
        let s = remote_knn_override_summary(&obj);
        assert!(s.contains("model_top1=London (42.00%)"));
    }

    #[test]
    fn knn_override_line_combines_token_and_summary() {
        let obj = mk_obj(&[
            ("token", serde_json::json!("Paris")),
            ("cosine", serde_json::json!(0.95)),
            ("layer", serde_json::json!(26)),
        ]);
        let s = remote_knn_override_line(&obj);
        assert!(s.starts_with("KNN override: Paris"));
        assert!(s.contains("cos=0.95"));
    }

    #[test]
    fn knn_override_line_handles_missing_token() {
        let obj = mk_obj(&[
            ("cosine", serde_json::json!(0.0)),
            ("layer", serde_json::json!(0)),
        ]);
        let s = remote_knn_override_line(&obj);
        assert!(s.starts_with("KNN override: ?"));
    }
}
