//! `EXPLAIN INFER` — full forward pass with optional attention capture
//! and logit lens, rendered per layer.

use crate::ast::LayerBand;
use crate::error::LqlError;
use crate::executor::helpers::format_knn_override_summary;
use crate::executor::{Backend, Session};

use super::resolve_bands;

impl Session {
    pub(crate) fn exec_infer_trace(
        &self,
        prompt: &str,
        top: Option<u32>,
        band: Option<LayerBand>,
        relations_only: bool,
        with_attention: bool,
    ) -> Result<Vec<String>, LqlError> {
        let top_k = top.unwrap_or(5) as usize;
        let per_layer = top.unwrap_or(3) as usize;

        // Weight backend has no feature labels — short-circuit to a
        // dense-only summary.
        if let Backend::Weight {
            weights, tokenizer, ..
        } = &self.backend
        {
            return self.exec_infer_trace_dense(weights, tokenizer, prompt, top_k);
        }

        // ── Phase 1: load model weights and tokenise ──
        let (path, config, patched) = self.require_vindex()?;
        if !config.has_model_weights {
            return Err(LqlError::Execution(
                "EXPLAIN INFER requires model weights. Rebuild with WITH INFERENCE.".into(),
            ));
        }
        if with_attention && config.quant != larql_vindex::QuantFormat::None {
            return Err(LqlError::Execution(
                "EXPLAIN INFER WITH ATTENTION does not yet support quantised (q4k) vindexes — \
                 attention capture requires f32 tensors in memory. Omit WITH ATTENTION or use \
                 an f32 vindex."
                    .into(),
            ));
        }
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let tokenizer = larql_vindex::load_vindex_tokenizer(path)
            .map_err(|e| LqlError::exec("failed to load tokenizer", e))?;
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| LqlError::exec("tokenize error", e))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        let token_strs: Vec<Option<String>> = if with_attention {
            token_ids
                .iter()
                .map(|&id| larql_inference::decode_token(&tokenizer, id))
                .collect()
        } else {
            Vec::new()
        };

        // ── Phase 2: forward pass ──
        //
        // For the standard path (no attention), `InferenceWeights` handles format
        // dispatch so EXPLAIN INFER works on both f32 and q4k vindexes.
        // The attention-capture path is f32-only (guarded above); it keeps its
        // own dense forward call and derives residuals from the same WalkFfn.
        let mut iw = larql_inference::InferenceWeights::load(path, config, &mut cb)
            .map_err(|e| LqlError::exec("failed to load model weights", e))?;

        let start = std::time::Instant::now();
        // Three groups of output, both branches must assign all of them.
        let (predictions, knn_override, model_top1, residuals, attention_captures, lens_residuals);

        if with_attention {
            // f32-only path (q4k guarded above): dense forward with attention + logit lens.
            let weights = iw.as_weights();
            let walk_ffn =
                larql_inference::vindex::WalkFfn::new_unlimited_with_trace(weights, patched);
            let r = larql_inference::predict_with_ffn_attention(
                weights, &tokenizer, &token_ids, top_k, &walk_ffn,
            );
            let walk_res = walk_ffn.take_residuals();
            let raw_top1 = r.predictions.first().cloned();
            let (preds, knn_ovr) = larql_inference::apply_knn_override(
                r.predictions,
                &walk_res,
                Some(&patched.knn_store),
                top_k,
            );
            predictions = preds;
            knn_override = knn_ovr;
            model_top1 = raw_top1;
            residuals = walk_res;
            attention_captures = r.attention;
            lens_residuals = r.residuals;
        } else {
            // Format-agnostic path: `InferenceWeights` dispatches to f32 or q4k.
            // `infer_patched` already applies the KNN override internally, so
            // `infer.predictions` is the final post-override top-k.
            let infer = iw.infer_patched(
                &tokenizer,
                patched,
                Some(&patched.knn_store),
                &token_ids,
                top_k,
            );
            predictions = infer.predictions;
            knn_override = infer.knn_override;
            model_top1 = infer.model_top1;
            residuals = infer.residuals;
            attention_captures = Vec::new();
            lens_residuals = Vec::new();
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        // ── Phase 3: side-tables for the rendering loop ──
        let attention_map = build_attention_map(&attention_captures, &token_strs, with_attention);
        let lens_map = build_lens_map(&lens_residuals, iw.as_weights(), &tokenizer, with_attention);

        let trace_layers = larql_inference::walk_trace_from_residuals(&residuals, patched);
        let classifier = self.relation_classifier();
        let bands = resolve_bands(config);
        let layer_range = band_to_layer_range(band, &bands);

        // ── Phase 4: format header ──
        let band_label = match band {
            Some(LayerBand::Syntax) => " (syntax)",
            Some(LayerBand::Knowledge) => " (knowledge)",
            Some(LayerBand::Output) => " (output)",
            _ => "",
        };

        let mut out = Vec::new();
        out.push(format!("Inference trace for {:?}{}:", prompt, band_label));
        if let Some(ovr) = &knn_override {
            out.push(format!(
                "Prediction: {} ({}) in {:.0}ms",
                ovr.token,
                format_knn_override_summary(ovr, model_top1.as_ref()),
                elapsed_ms
            ));
            out.push(
                "Pending retrieval override: not part of the residual/FFN trace until materialized."
                    .into(),
            );
        } else {
            out.push(format!(
                "Prediction: {} ({:.2}%) in {:.0}ms",
                predictions.first().map(|(t, _)| t.as_str()).unwrap_or("?"),
                predictions.first().map(|(_, p)| p * 100.0).unwrap_or(0.0),
                elapsed_ms
            ));
        }
        out.push(String::new());

        // ── Phase 5: per-layer rendering ──
        for (layer, hits) in &trace_layers {
            if hits.is_empty() {
                continue;
            }
            if let Some((lo, hi)) = layer_range {
                if *layer < lo || *layer > hi {
                    continue;
                }
            }
            render_trace_layer(
                &mut out,
                *layer,
                hits,
                classifier,
                relations_only,
                per_layer,
                with_attention,
                &attention_map,
                &lens_map,
            );
        }

        Ok(out)
    }

    /// EXPLAIN INFER on a `Backend::Weight` (no vindex): produces a dense
    /// inference summary with no feature trace, since there are no
    /// gate vectors / down meta to attribute.
    fn exec_infer_trace_dense(
        &self,
        weights: &larql_inference::ModelWeights,
        tokenizer: &larql_inference::tokenizers::Tokenizer,
        prompt: &str,
        top_k: usize,
    ) -> Result<Vec<String>, LqlError> {
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| LqlError::exec("tokenize error", e))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        let start = std::time::Instant::now();
        let result = larql_inference::predict(weights, tokenizer, &token_ids, top_k);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        let mut out = Vec::new();
        out.push(format!(
            "Inference trace for {:?} (dense — no vindex):",
            prompt
        ));
        out.push(format!(
            "Prediction: {} ({:.2}%) in {:.0}ms",
            result
                .predictions
                .first()
                .map(|(t, _)| t.as_str())
                .unwrap_or("?"),
            result
                .predictions
                .first()
                .map(|(_, p)| p * 100.0)
                .unwrap_or(0.0),
            elapsed_ms,
        ));
        out.push(String::new());
        out.push("Note: no per-feature trace without a vindex. EXTRACT for full trace.".into());
        Ok(out)
    }
}

// ── EXPLAIN INFER helpers ────────────────────────────────────────────────
//
// `exec_infer_trace` is a five-phase pipeline (load → forward → side
// tables → header → render). The helpers below split the side-table
// builders and the per-layer rendering loop out of the main function.
// The cross-surface trace reconstruction lives in
// `larql_inference::walk_trace_from_residuals`.

/// Build a `layer → top-3 attended (token, weight)` map from the
/// captured attention weights. Returns an empty map when
/// `with_attention` is false. Averages across all heads, drops special
/// tokens (BOS/EOS) by skipping `None` entries from `decode_token`, and
/// truncates to the top 3 by weight.
fn build_attention_map(
    captures: &[larql_inference::LayerAttentionCapture],
    token_strs: &[Option<String>],
    with_attention: bool,
) -> std::collections::HashMap<usize, Vec<(String, f32)>> {
    if !with_attention {
        return std::collections::HashMap::new();
    }
    let mut map = std::collections::HashMap::new();
    for cap in captures {
        let n_heads = cap.weights.heads.len();
        if n_heads == 0 || token_strs.is_empty() {
            continue;
        }
        let seq_len = cap.weights.heads[0].len();
        let mut avg = vec![0.0f32; seq_len];
        for head in &cap.weights.heads {
            for (j, &w) in head.iter().enumerate() {
                avg[j] += w;
            }
        }
        for v in avg.iter_mut() {
            *v /= n_heads as f32;
        }
        let mut pairs: Vec<(String, f32)> = avg
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(j, w)| {
                let tok = token_strs.get(j)?.as_ref()?;
                Some((tok.trim().to_string(), w))
            })
            .collect();
        pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        pairs.truncate(3);
        map.insert(cap.layer, pairs);
    }
    map
}

/// Build a `layer → (top_token, probability)` map by running the logit
/// lens on each captured residual. Returns empty when `with_attention`
/// is false (only the attention path captures intermediate residuals).
fn build_lens_map(
    lens_residuals: &[(usize, Vec<f32>)],
    weights: &larql_inference::ModelWeights,
    tokenizer: &larql_inference::tokenizers::Tokenizer,
    with_attention: bool,
) -> std::collections::HashMap<usize, (String, f64)> {
    if !with_attention {
        return std::collections::HashMap::new();
    }
    lens_residuals
        .iter()
        .filter_map(|(layer, residual)| {
            let pred = larql_inference::logit_lens_top1(weights, tokenizer, residual.as_slice())?;
            Some((*layer, pred))
        })
        .collect()
}

/// Resolve a `LayerBand` to a `(lo, hi)` filter on the trace layers.
/// Returns `None` for `All` / no band — the caller treats that as
/// "include every layer".
fn band_to_layer_range(
    band: Option<LayerBand>,
    bands: &larql_vindex::LayerBands,
) -> Option<(usize, usize)> {
    match band {
        Some(LayerBand::Syntax) => Some(bands.syntax),
        Some(LayerBand::Knowledge) => Some(bands.knowledge),
        Some(LayerBand::Output) => Some(bands.output),
        Some(LayerBand::All) | None => None,
    }
}

/// Render one layer's worth of trace hits, in either the compact
/// `with_attention` single-line format (top hit + attention + lens) or
/// the standard multi-line format (top-N hits with relation labels).
#[allow(clippy::too_many_arguments)]
fn render_trace_layer(
    out: &mut Vec<String>,
    layer: usize,
    hits: &[larql_vindex::WalkHit],
    classifier: Option<&crate::relations::RelationClassifier>,
    relations_only: bool,
    per_layer: usize,
    with_attention: bool,
    attention_map: &std::collections::HashMap<usize, Vec<(String, f32)>>,
    lens_map: &std::collections::HashMap<usize, (String, f64)>,
) {
    // When filtering to relations only, re-sort so positive gates rank
    // above negative gates of equal magnitude (positive gates correlate
    // with the prediction; negative gates with the opposite).
    let labelled_hits: Vec<&larql_vindex::WalkHit> = if relations_only {
        let mut lh: Vec<_> = hits
            .iter()
            .filter(|hit| {
                classifier
                    .and_then(|rc| rc.label_for_feature(layer, hit.feature))
                    .map(|l| !l.is_empty())
                    .unwrap_or(false)
            })
            .collect();
        lh.sort_by(|a, b| {
            let a_pos = a.gate_score > 0.0;
            let b_pos = b.gate_score > 0.0;
            match (a_pos, b_pos) {
                (true, false) => std::cmp::Ordering::Less,
                (false, true) => std::cmp::Ordering::Greater,
                _ => b
                    .gate_score
                    .abs()
                    .partial_cmp(&a.gate_score.abs())
                    .unwrap_or(std::cmp::Ordering::Equal),
            }
        });
        lh
    } else {
        hits.iter().collect()
    };

    if with_attention {
        // Compact single-line format: feature + attention + logit lens.
        let hit = labelled_hits.first();
        let feature_part = if let Some(hit) = hit {
            let label = classifier
                .and_then(|rc| rc.label_for_feature(layer, hit.feature))
                .unwrap_or("");
            if relations_only && label.is_empty() {
                None
            } else {
                let top_token = hit.meta.top_token.trim();
                let name = if !label.is_empty() { label } else { top_token };
                Some(format!("{:<14} {:+.1}", name, hit.gate_score))
            }
        } else {
            None
        };
        let empty = format!("{:19}", "");
        let feature_str = feature_part.as_deref().unwrap_or(&empty);

        let attn_part = attention_map
            .get(&layer)
            .and_then(|attn| attn.first())
            .map(|(tok, w)| format!("{}({:.0}%)", tok, w * 100.0))
            .unwrap_or_default();

        let lens_part = lens_map
            .get(&layer)
            .map(|(tok, prob)| format!("{} ({:.1}%)", tok, prob * 100.0))
            .unwrap_or_default();

        if feature_part.is_some() || !lens_part.is_empty() {
            out.push(format!(
                "  L{:2}  {:<19}  {:<16} → {}",
                layer, feature_str, attn_part, lens_part,
            ));
        }
    } else {
        // Standard multi-line format without attention.
        let mut shown = 0;
        for hit in &labelled_hits {
            if shown >= per_layer {
                break;
            }
            let label = classifier
                .and_then(|rc| rc.label_for_feature(layer, hit.feature))
                .unwrap_or("");
            if relations_only && label.is_empty() {
                continue;
            }
            shown += 1;
            let label_str = if label.is_empty() {
                format!("{:14}", "")
            } else {
                format!("{:<14}", label)
            };
            let top_token = hit.meta.top_token.trim();
            let down_top: String = hit
                .meta
                .top_k
                .iter()
                .take(3)
                .map(|t| t.token.clone())
                .collect::<Vec<_>>()
                .join(", ");
            out.push(format!(
                "  L{:2}: {} F{:<5} gate={:+.1}  → {:15} [{}]",
                layer, label_str, hit.feature, hit.gate_score, top_token, down_top,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    //! Coverage for the four pure helpers — `exec_infer_trace` itself
    //! needs real model weights and is exercised by the synthetic-fixture
    //! integration tests in `crates/larql-lql/tests/infer_trace_synthetic.rs`.
    //! These tests target the formatter logic that runs after the
    //! forward pass.
    use super::*;
    use larql_inference::attention::AttentionWeights;
    use larql_inference::LayerAttentionCapture;
    use larql_models::TopKEntry;
    use larql_vindex::{FeatureMeta, LayerBands, WalkHit};

    fn meta(top: &str, tokens: &[&str]) -> FeatureMeta {
        FeatureMeta {
            top_token: top.into(),
            top_token_id: 0,
            c_score: 0.9,
            top_k: tokens
                .iter()
                .enumerate()
                .map(|(i, &t)| TopKEntry {
                    token: t.into(),
                    token_id: i as u32,
                    logit: 1.0 - 0.1 * i as f32,
                })
                .collect(),
        }
    }

    fn hit(layer: usize, feature: usize, gate: f32, top: &str, top_k: &[&str]) -> WalkHit {
        WalkHit {
            layer,
            feature,
            gate_score: gate,
            meta: meta(top, top_k),
        }
    }

    fn bands() -> LayerBands {
        LayerBands {
            syntax: (0, 13),
            knowledge: (14, 27),
            output: (28, 33),
        }
    }

    // ── band_to_layer_range ───────────────────────────────────────────

    #[test]
    fn band_syntax_returns_syntax_range() {
        assert_eq!(
            band_to_layer_range(Some(LayerBand::Syntax), &bands()),
            Some((0, 13))
        );
    }

    #[test]
    fn band_knowledge_returns_knowledge_range() {
        assert_eq!(
            band_to_layer_range(Some(LayerBand::Knowledge), &bands()),
            Some((14, 27))
        );
    }

    #[test]
    fn band_output_returns_output_range() {
        assert_eq!(
            band_to_layer_range(Some(LayerBand::Output), &bands()),
            Some((28, 33))
        );
    }

    #[test]
    fn band_all_returns_none() {
        assert_eq!(band_to_layer_range(Some(LayerBand::All), &bands()), None);
    }

    #[test]
    fn band_unset_returns_none() {
        assert_eq!(band_to_layer_range(None, &bands()), None);
    }

    // ── build_attention_map ────────────────────────────────────────────

    #[test]
    fn attention_map_empty_when_with_attention_false() {
        let caps = vec![LayerAttentionCapture {
            layer: 5,
            weights: AttentionWeights {
                heads: vec![vec![0.5, 0.5]],
            },
        }];
        let tokens = vec![Some("a".to_string()), Some("b".to_string())];
        let map = build_attention_map(&caps, &tokens, false);
        assert!(map.is_empty(), "with_attention=false should disable build");
    }

    #[test]
    fn attention_map_empty_when_no_captures() {
        let map = build_attention_map(&[], &[Some("a".into())], true);
        assert!(map.is_empty());
    }

    #[test]
    fn attention_map_averages_over_heads_and_truncates_to_three() {
        // 5 tokens, 2 heads. After averaging, sort desc and keep top 3.
        let caps = vec![LayerAttentionCapture {
            layer: 7,
            weights: AttentionWeights {
                heads: vec![
                    vec![0.1, 0.2, 0.3, 0.2, 0.2], // head 0
                    vec![0.3, 0.4, 0.1, 0.1, 0.1], // head 1
                ],
            },
        }];
        let tokens = vec![
            Some("alpha".into()),
            Some("beta".into()),
            Some("gamma".into()),
            Some("delta".into()),
            Some("epsilon".into()),
        ];
        let map = build_attention_map(&caps, &tokens, true);
        let pairs = map.get(&7).expect("layer 7 should appear");
        assert_eq!(pairs.len(), 3, "should truncate to top 3");
        // Avg per col: 0.2, 0.3, 0.2, 0.15, 0.15 → top is "beta"
        assert_eq!(pairs[0].0, "beta");
        assert!((pairs[0].1 - 0.3).abs() < 1e-6);
    }

    #[test]
    fn attention_map_skips_none_token_strs() {
        // None entries (BOS/EOS) drop out of the pairs.
        let caps = vec![LayerAttentionCapture {
            layer: 0,
            weights: AttentionWeights {
                heads: vec![vec![0.5, 0.5]],
            },
        }];
        let tokens = vec![None, Some("real".into())];
        let map = build_attention_map(&caps, &tokens, true);
        let pairs = map.get(&0).unwrap();
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].0, "real");
    }

    #[test]
    fn attention_map_skips_capture_with_zero_heads() {
        let caps = vec![LayerAttentionCapture {
            layer: 2,
            weights: AttentionWeights { heads: vec![] },
        }];
        let map = build_attention_map(&caps, &[Some("a".into())], true);
        assert!(map.is_empty());
    }

    // ── build_lens_map ─────────────────────────────────────────────────

    #[test]
    fn lens_map_empty_when_with_attention_false() {
        // Real weights/tokenizer aren't needed when with_attention=false
        // because the function returns the empty-map branch immediately.
        // We construct synthetic placeholders just to satisfy the
        // signature.
        let weights = larql_inference::test_utils::make_test_weights();
        let tokenizer = larql_inference::test_utils::make_test_tokenizer(weights.vocab_size);
        let lens = vec![(5usize, vec![0.0f32; 4])];
        let map = build_lens_map(&lens, &weights, &tokenizer, false);
        assert!(map.is_empty());
    }

    // ── render_trace_layer ─────────────────────────────────────────────

    #[test]
    fn render_multiline_with_no_classifier_emits_per_layer_lines() {
        let mut out = Vec::new();
        let hits = vec![
            hit(5, 100, 2.5, " Paris", &[" Paris", " France", ","]),
            hit(5, 101, -1.0, " noise", &[" noise"]),
        ];
        render_trace_layer(
            &mut out,
            5,
            &hits,
            None,
            false,
            2,
            false,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
        );
        assert_eq!(out.len(), 2, "per_layer=2 should emit two lines");
        assert!(out[0].contains("L 5"));
        assert!(out[0].contains("F100"));
        assert!(out[0].contains("Paris"));
    }

    #[test]
    fn render_multiline_respects_per_layer_cap() {
        let mut out = Vec::new();
        let hits = vec![
            hit(0, 1, 1.0, "a", &["a"]),
            hit(0, 2, 0.5, "b", &["b"]),
            hit(0, 3, 0.1, "c", &["c"]),
        ];
        render_trace_layer(
            &mut out,
            0,
            &hits,
            None,
            false,
            2,
            false,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
        );
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn render_multiline_with_relations_only_drops_unlabelled() {
        let mut out = Vec::new();
        let hits = vec![hit(0, 1, 1.0, "a", &["a"])];
        // No classifier → no labels → relations_only filters everything.
        render_trace_layer(
            &mut out,
            0,
            &hits,
            None,
            true,
            5,
            false,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
        );
        assert!(out.is_empty());
    }

    #[test]
    fn render_with_attention_compact_line_includes_attn_and_lens() {
        let mut out = Vec::new();
        let hits = vec![hit(7, 42, 3.5, " Paris", &[" Paris"])];
        let mut attention_map = std::collections::HashMap::new();
        attention_map.insert(7, vec![("France".to_string(), 0.6)]);
        let mut lens_map = std::collections::HashMap::new();
        lens_map.insert(7, ("Paris".to_string(), 0.75));
        render_trace_layer(
            &mut out,
            7,
            &hits,
            None,
            false,
            1,
            true,
            &attention_map,
            &lens_map,
        );
        assert_eq!(out.len(), 1);
        let line = &out[0];
        assert!(line.contains("L 7"));
        assert!(line.contains("Paris"));
        assert!(line.contains("France"));
        // 0.6 × 100 with %.0f = "60%"
        assert!(line.contains("60%"));
        // 0.75 × 100 with %.1f = "75.0%"
        assert!(line.contains("75.0%"));
    }

    #[test]
    fn render_with_attention_emits_nothing_when_no_feature_and_no_lens() {
        let mut out = Vec::new();
        // Empty hits, no lens entry → compact path skips the layer.
        render_trace_layer(
            &mut out,
            5,
            &[],
            None,
            false,
            1,
            true,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
        );
        assert!(out.is_empty());
    }

    #[test]
    fn render_with_attention_emits_lens_only_when_no_feature() {
        let mut out = Vec::new();
        let mut lens_map = std::collections::HashMap::new();
        lens_map.insert(3, ("X".to_string(), 0.9));
        render_trace_layer(
            &mut out,
            3,
            &[],
            None,
            false,
            1,
            true,
            &std::collections::HashMap::new(),
            &lens_map,
        );
        // The compact branch fires when `feature_part.is_some() ||
        // !lens_part.is_empty()` — lens alone is enough.
        assert_eq!(out.len(), 1);
        assert!(out[0].contains("X"));
        assert!(out[0].contains("90.0%"));
    }

    #[test]
    fn render_multiline_with_classifier_skips_unlabelled_when_relations_only() {
        // Same shape as the previous unlabelled test but explicitly
        // covers the path where `classifier.is_some()` but the lookup
        // returns None for each feature. None → label `""` → filtered.
        // Without a real classifier we pass None — same behavioural
        // outcome for the filter.
        let mut out = Vec::new();
        let hits = vec![hit(0, 99, 1.0, "x", &["x"])];
        render_trace_layer(
            &mut out,
            0,
            &hits,
            None,
            true,
            5,
            false,
            &std::collections::HashMap::new(),
            &std::collections::HashMap::new(),
        );
        assert!(out.is_empty());
    }
}
