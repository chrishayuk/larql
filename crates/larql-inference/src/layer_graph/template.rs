use ndarray::Array2;

use super::{LayerGraph, LayerOutput};
use crate::ffn::FfnBackend;
use crate::model::ModelWeights;

// ── Template detection ──

/// Known template patterns for routing.
#[derive(Clone, Debug)]
pub struct TemplatePattern {
    pub name: String,
    /// Token prefix that identifies this template (before the entity slot).
    pub prefix_tokens: Vec<u32>,
    /// Layer range for cached regime.
    pub cached_layers: std::ops::RangeInclusive<usize>,
}

/// Detect which template a token sequence matches, if any.
/// Matches by longest prefix overlap.
pub fn detect_template(token_ids: &[u32], templates: &[TemplatePattern]) -> Option<usize> {
    let mut best = None;
    let mut best_len = 0;

    for (i, tmpl) in templates.iter().enumerate() {
        let prefix = &tmpl.prefix_tokens;
        if prefix.len() > token_ids.len() {
            continue;
        }
        // Check if tokens start with this prefix (skipping BOS if present)
        let offset = if token_ids.len() > prefix.len() && token_ids[0] != prefix[0] {
            1
        } else {
            0
        };
        if offset + prefix.len() > token_ids.len() {
            continue;
        }
        let matches = prefix.iter().zip(&token_ids[offset..]).all(|(a, b)| a == b);
        if matches && prefix.len() > best_len {
            best = Some(i);
            best_len = prefix.len();
        }
    }
    best
}

// ── Template-guided walk: score only features in the template's universe ──

/// Per-template per-layer feature universe: the set of features that ever
/// fire for this template across diverse entities.
///
/// Built by running forward passes for a template with many entities,
/// capturing which features activate at each layer, and taking the union.
pub struct TemplateUniverse {
    pub name: String,
    /// layer → sorted vec of feature indices that fire for this template.
    pub features: std::collections::HashMap<usize, Vec<usize>>,
}

impl TemplateUniverse {
    /// Build by running dense forward passes for a template with multiple entities.
    /// `template`: format string with `{}` for entity slot.
    /// `entities`: list of entities to test.
    /// `activation_threshold`: minimum |activation| to count a feature as firing.
    pub fn build(
        weights: &ModelWeights,
        tokenizer: &tokenizers::Tokenizer,
        name: &str,
        template: &str,
        entities: &[&str],
        ffn: &dyn FfnBackend,
        activation_threshold: f32,
    ) -> Self {
        let all_layers: Vec<usize> = (0..weights.num_layers).collect();
        let mut layer_features: std::collections::HashMap<usize, std::collections::HashSet<usize>> =
            std::collections::HashMap::new();

        for entity in entities {
            let prompt = template.replace("{}", entity);
            let encoding = match tokenizer.encode(prompt.as_str(), true) {
                Ok(e) => e,
                Err(_) => continue,
            };
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            let trace = crate::forward::trace_forward_full(
                weights,
                &token_ids,
                &all_layers,
                true,
                500,
                false,
                ffn,
            );

            for (layer, acts) in &trace.activations {
                let set = layer_features.entry(*layer).or_default();
                for (feat, act) in acts {
                    if act.abs() > activation_threshold {
                        set.insert(*feat);
                    }
                }
            }
        }

        let features = layer_features
            .into_iter()
            .map(|(layer, set)| {
                let mut v: Vec<usize> = set.into_iter().collect();
                v.sort_unstable();
                (layer, v)
            })
            .collect();

        Self {
            name: name.to_string(),
            features,
        }
    }

    /// Get the feature universe for a layer.
    pub fn get(&self, layer: usize) -> Option<&[usize]> {
        self.features.get(&layer).map(|v| v.as_slice())
    }

    /// Total features across all layers.
    pub fn total_features(&self) -> usize {
        self.features.values().map(|v| v.len()).sum()
    }

    /// Print a summary.
    pub fn summary(&self) {
        let mut layers: Vec<usize> = self.features.keys().copied().collect();
        layers.sort();
        for &layer in &layers {
            let n = self.features[&layer].len();
            if n > 0 {
                print!("L{layer}:{n} ");
            }
        }
        println!();
    }
}

/// Guided walk layer graph: dense attention + walk FFN restricted to
/// the template's per-layer feature universe.
///
/// Instead of scoring all 10,240 features, scores only the ~100-400
/// that the template ever activates. Per-feature dot products + accumulations.
pub struct GuidedWalkLayerGraph<'a> {
    pub weights: &'a ModelWeights,
    pub universe: &'a TemplateUniverse,
    pub index: &'a dyn larql_vindex::GateIndex,
}

impl<'a> LayerGraph for GuidedWalkLayerGraph<'a> {
    fn forward_layer(
        &self,
        weights: &ModelWeights,
        h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput> {
        // Attention: dense matmul
        let (h_post_attn, _attn_proj, _) =
            crate::attention::run_attention_block(weights, h, layer, false)?;

        // FFN: guided walk — score only template universe features
        let residual = guided_walk_ffn(weights, &h_post_attn, layer, self.universe, self.index);

        Some(LayerOutput {
            residual,
            activation: None,
            attention: None,
        })
    }

    fn name(&self) -> &str {
        "guided-walk"
    }
}

/// Guided walk FFN: pre-FFN norm → gate scores for universe → GEGLU → accumulate.
///
/// Gate: scores all features (one gate_scores_batch call), but only processes
/// the template universe features for up/down. The gate call is the same cost
/// as dense, but up/down computation drops from 10,240 to ~100-400 features.
/// Up/down: per-feature dot products and scaled adds (no matmul).
fn guided_walk_ffn(
    weights: &ModelWeights,
    h_post_attn: &Array2<f32>,
    layer: usize,
    universe: &TemplateUniverse,
    index: &dyn larql_vindex::GateIndex,
) -> Array2<f32> {
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let hidden = h_post_attn.shape()[1];
    let seq_len = h_post_attn.shape()[0];

    // Pre-FFN norm
    let pre_ffn_key = if arch.has_post_norms() {
        arch.pre_feedforward_layernorm_key(layer)
    } else {
        Some(arch.post_attention_layernorm_key(layer))
    };
    let h_ffn = match pre_ffn_key {
        Some(key) => crate::forward::apply_norm(weights, h_post_attn, &key, norm_offset),
        None => crate::residual::rms_norm(h_post_attn, None, norm_offset),
    };

    // Get template universe for this layer
    let features = match universe.get(layer) {
        Some(f) if !f.is_empty() => f,
        _ => return h_post_attn.clone(),
    };

    let up_view = match index.up_layer_matrix(layer) {
        Some(v) => v,
        None => return h_post_attn.clone(),
    };
    let down_view = match index.down_layer_matrix(layer) {
        Some(v) => v,
        None => return h_post_attn.clone(),
    };

    let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
    let use_gelu = matches!(
        arch.activation(),
        larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
    );

    // Gate scores: one batch call, then index into universe features only.
    // This is still a matmul for gate, but up/down are per-feature only.
    let gate_scores = match index.gate_scores_batch(layer, &h_ffn) {
        Some(gs) => gs,
        None => return h_post_attn.clone(),
    };

    let mut ffn_out = Array2::<f32>::zeros((seq_len, hidden));

    for s in 0..seq_len {
        let x_row = h_ffn.row(s);
        let mut out_row = ffn_out.row_mut(s);

        for &feat in features {
            let gate_score = gate_scores[[s, feat]];

            let act = if is_gated {
                let up_score = up_view.row(feat).dot(&x_row);
                let activated_gate = if use_gelu {
                    crate::ffn::gelu_tanh(gate_score)
                } else {
                    gate_score * crate::ffn::sigmoid(gate_score)
                };
                activated_gate * up_score
            } else {
                let v = gate_score;
                if use_gelu {
                    crate::ffn::gelu_tanh(v)
                } else {
                    v * crate::ffn::sigmoid(v)
                }
            };

            if act.abs() > 1e-10 {
                let down_row = down_view.row(feat);
                out_row.scaled_add(act, &down_row);
            }
        }
    }

    // Post-FFN norm + residual
    let res_mult = arch.residual_multiplier();
    if arch.has_post_norms() {
        let normed = match arch.post_feedforward_layernorm_key(layer) {
            Some(key) => crate::forward::apply_norm(weights, &ffn_out, &key, norm_offset),
            None => crate::residual::rms_norm(&ffn_out, None, norm_offset),
        };
        if res_mult != 1.0 {
            h_post_attn + &(&normed * res_mult)
        } else {
            h_post_attn + &normed
        }
    } else if res_mult != 1.0 {
        h_post_attn + &(&ffn_out * res_mult)
    } else {
        h_post_attn + &ffn_out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffn::WeightFfn;
    use crate::test_utils::{make_test_tokenizer, make_test_vindex, make_test_weights};
    use larql_models::ModelWeights;
    use ndarray::Array2;
    use std::sync::OnceLock;

    fn weights() -> &'static ModelWeights {
        static W: OnceLock<ModelWeights> = OnceLock::new();
        W.get_or_init(make_test_weights)
    }

    fn input(seq: usize, hidden: usize) -> Array2<f32> {
        let data: Vec<f32> = (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.01).collect();
        Array2::from_shape_vec((seq, hidden), data).unwrap()
    }

    // ── detect_template ───────────────────────────────────────────────────────

    #[test]
    fn detect_no_templates_returns_none() {
        assert!(detect_template(&[1, 2, 3], &[]).is_none());
    }

    #[test]
    fn detect_no_match_returns_none() {
        let t = TemplatePattern {
            name: "t".into(),
            prefix_tokens: vec![10, 11, 12],
            cached_layers: 0..=5,
        };
        assert!(detect_template(&[1, 2, 3], &[t]).is_none());
    }

    #[test]
    fn detect_exact_prefix_match() {
        let t = TemplatePattern {
            name: "t".into(),
            prefix_tokens: vec![1, 2, 3],
            cached_layers: 0..=5,
        };
        assert_eq!(detect_template(&[1, 2, 3, 99], &[t]), Some(0));
    }

    #[test]
    fn detect_longest_prefix_wins() {
        let short = TemplatePattern {
            name: "short".into(),
            prefix_tokens: vec![1, 2],
            cached_layers: 0..=5,
        };
        let long = TemplatePattern {
            name: "long".into(),
            prefix_tokens: vec![1, 2, 3],
            cached_layers: 0..=5,
        };
        // long prefix (index 1) should win
        assert_eq!(detect_template(&[1, 2, 3, 99], &[short, long]), Some(1));
    }

    #[test]
    fn detect_bos_offset_allows_bos_at_token0() {
        // prefix_tokens = [5, 6]; token_ids = [1 (BOS), 5, 6, 99]
        // With BOS offset: skip token 0, check tokens [1..] = [5, 6, 99] → matches at offset 1
        let t = TemplatePattern {
            name: "t".into(),
            prefix_tokens: vec![5, 6],
            cached_layers: 0..=5,
        };
        assert_eq!(detect_template(&[1, 5, 6, 99], &[t]), Some(0));
    }

    #[test]
    fn detect_prefix_too_long_for_input_returns_none() {
        let t = TemplatePattern {
            name: "t".into(),
            prefix_tokens: vec![1, 2, 3, 4, 5],
            cached_layers: 0..=5,
        };
        assert!(detect_template(&[1, 2], &[t]).is_none());
    }

    // ── TemplateUniverse ──────────────────────────────────────────────────────

    #[test]
    fn universe_build_empty_entities_is_empty() {
        // Empty entity list → no tokenizations, no trace_forward_full calls.
        // Tests the build scaffolding without triggering the Whitespace
        // pre-tokenizer issue: that tokenizer strips brackets from "[N]"
        // words → OOV → UNK (ID 32, out-of-range for 32-vocab test weights).
        let w = weights();
        let tokenizer = make_test_tokenizer(w.vocab_size);
        let ffn = WeightFfn { weights: w };
        let universe =
            TemplateUniverse::build(w, &tokenizer, "test-template", "[0] {}", &[], &ffn, 0.01);
        assert_eq!(universe.name, "test-template");
        assert_eq!(universe.total_features(), 0);
    }

    #[test]
    fn universe_get_missing_layer_returns_none() {
        let universe = TemplateUniverse {
            name: "empty".into(),
            features: std::collections::HashMap::new(),
        };
        assert!(universe.get(0).is_none());
    }

    #[test]
    fn universe_get_populated_layer_returns_features() {
        let mut features = std::collections::HashMap::new();
        features.insert(3usize, vec![0usize, 5, 12]);
        let universe = TemplateUniverse {
            name: "t".into(),
            features,
        };
        assert_eq!(universe.get(3), Some([0usize, 5, 12].as_slice()));
        assert!(universe.get(0).is_none());
    }

    #[test]
    fn universe_total_features_sums_layers() {
        let mut features = std::collections::HashMap::new();
        features.insert(0, vec![1, 2, 3]);
        features.insert(1, vec![4, 5]);
        let universe = TemplateUniverse {
            name: "t".into(),
            features,
        };
        assert_eq!(universe.total_features(), 5);
    }

    // ── GuidedWalkLayerGraph ──────────────────────────────────────────────────

    #[test]
    fn guided_walk_empty_universe_returns_correct_shape() {
        let w = weights();
        let idx = make_test_vindex(w);
        let universe = TemplateUniverse {
            name: "empty".into(),
            features: std::collections::HashMap::new(),
        };
        let g = GuidedWalkLayerGraph {
            weights: w,
            universe: &universe,
            index: &idx,
        };
        let h = input(1, w.hidden_size);
        let out = g.forward_layer(w, &h, 0);
        assert!(out.is_some());
        assert_eq!(out.unwrap().residual.shape(), &[1, w.hidden_size]);
    }

    #[test]
    fn guided_walk_name() {
        let w = weights();
        let idx = make_test_vindex(w);
        let universe = TemplateUniverse {
            name: "t".into(),
            features: std::collections::HashMap::new(),
        };
        let g = GuidedWalkLayerGraph {
            weights: w,
            universe: &universe,
            index: &idx,
        };
        assert_eq!(g.name(), "guided-walk");
    }

    #[test]
    fn guided_walk_with_per_layer_features_drives_walk_ffn_loop() {
        // Hand-construct a universe with a few features per layer so the
        // `guided_walk_ffn` per-feature loop actually iterates.
        let w = weights();
        let idx = make_test_vindex(w);
        let mut features = std::collections::HashMap::new();
        // intermediate_size is 32 in the synthetic fixture.
        for layer in 0..w.num_layers {
            features.insert(layer, vec![0usize, 5, 10, 17, 31]);
        }
        let universe = TemplateUniverse {
            name: "non-empty".into(),
            features,
        };
        let g = GuidedWalkLayerGraph {
            weights: w,
            universe: &universe,
            index: &idx,
        };
        let h = input(2, w.hidden_size);
        for layer in 0..w.num_layers {
            let out = g.forward_layer(w, &h, layer).expect("layer should run");
            assert_eq!(out.residual.shape(), &[2, w.hidden_size]);
            assert!(
                out.residual.iter().all(|v| v.is_finite()),
                "layer {layer} should produce finite output"
            );
        }
    }

    #[test]
    fn universe_summary_runs_without_panicking() {
        // summary() prints to stdout — we just verify it doesn't panic on
        // populated and empty universes. Captures both branches (n > 0 and
        // n == 0).
        let mut features = std::collections::HashMap::new();
        features.insert(0, vec![1, 2, 3]);
        features.insert(1, vec![]);
        let universe = TemplateUniverse {
            name: "test".into(),
            features,
        };
        universe.summary();

        let empty = TemplateUniverse {
            name: "empty".into(),
            features: std::collections::HashMap::new(),
        };
        empty.summary();
    }

    #[test]
    fn detect_template_picks_first_among_equal_length_prefixes() {
        // Two templates of equal length both match → first index wins
        // (the longest-prefix-wins tiebreak only fires when lengths differ).
        let a = TemplatePattern {
            name: "a".into(),
            prefix_tokens: vec![1, 2],
            cached_layers: 0..=5,
        };
        let b = TemplatePattern {
            name: "b".into(),
            prefix_tokens: vec![1, 2],
            cached_layers: 0..=5,
        };
        assert_eq!(detect_template(&[1, 2, 99], &[a, b]), Some(0));
    }

    #[test]
    fn guided_walk_all_layers_finite() {
        let w = weights();
        let idx = make_test_vindex(w);
        let universe = TemplateUniverse {
            name: "t".into(),
            features: std::collections::HashMap::new(),
        };
        let g = GuidedWalkLayerGraph {
            weights: w,
            universe: &universe,
            index: &idx,
        };
        let h = input(2, w.hidden_size);
        for layer in 0..w.num_layers {
            let out = g.forward_layer(w, &h, layer).expect("layer {layer}");
            assert!(out.residual.iter().all(|v| v.is_finite()), "layer {layer}");
        }
    }

    /// `TemplateUniverse::build` entity loop body — lines 80-113.
    ///
    /// `universe_build_empty_entities_is_empty` covers the no-entity
    /// scaffolding; this test drives the body by passing real entities
    /// through a pre-tokenizer-free tokenizer (built from the same JSON
    /// the on-disk fixture uses) so `tokenizer.encode("[0] [1]")` lands
    /// in-vocab and `trace_forward_full` completes without UNK panics.
    #[test]
    fn universe_build_with_real_entities_drives_loop_body() {
        let w = weights();
        let ffn = WeightFfn { weights: w };
        // pre_tokenizer = null tokenizer so "[N]" stays a single token.
        let tok_json = crate::test_utils::synthetic_tokenizer_json(w.vocab_size);
        let tokenizer = tokenizers::Tokenizer::from_bytes(tok_json.as_bytes()).expect("tokenizer");
        let universe = TemplateUniverse::build(
            w,
            &tokenizer,
            "real-entities",
            "[5] {}",
            &["[0]", "[1]", "[2]"],
            &ffn,
            0.0, // floor=0 ⇒ insert every fired feature, exercising both branches.
        );
        assert_eq!(universe.name, "real-entities");
        // Body of the loop should have inserted at least *some* features
        // somewhere across the 2-layer fixture — the inner
        // `set.insert(*feat)` branch is what we want covered.
        assert!(
            universe.total_features() > 0,
            "build should populate feature set across layers"
        );
    }

    /// With both interleaved-f32 AND feature-major payloads attached to
    /// the vindex, `up_layer_matrix` / `down_layer_matrix` /
    /// `gate_scores_batch` all return Some, so `guided_walk_ffn` runs the
    /// inner per-feature loop (lines 240-269) instead of bailing at the
    /// up/down early-return guards.
    ///
    /// `InterleavedF32TestFixtures` alone only populates `interleaved_*`
    /// storage, which serves `interleaved_up` / `interleaved_down` —
    /// **not** `up_layer_matrix` (that one reads `up_features`).
    /// So we also attach the feature-major payload before constructing
    /// the GuidedWalkLayerGraph.
    #[test]
    fn guided_walk_with_interleaved_f32_runs_inner_accumulation_loop() {
        use crate::test_utils::{
            attach_feature_major_f32_to_test_vindex, attach_interleaved_f32_to_test_vindex,
            make_test_vindex, make_test_weights,
        };
        let w = make_test_weights();
        let mut index = make_test_vindex(&w);
        attach_interleaved_f32_to_test_vindex(&w, &mut index);
        attach_feature_major_f32_to_test_vindex(&w, &mut index);

        // Supply a non-empty feature universe so the inner loop iterates.
        let mut features = std::collections::HashMap::new();
        for layer in 0..w.num_layers {
            // intermediate_size is small in the synthetic fixture, pick a few.
            features.insert(layer, vec![0usize, 3, 7]);
        }
        let universe = TemplateUniverse {
            name: "interleaved-test".into(),
            features,
        };
        let g = GuidedWalkLayerGraph {
            weights: &w,
            universe: &universe,
            index: &index,
        };
        let h = input(2, w.hidden_size);
        for layer in 0..w.num_layers {
            let out = g.forward_layer(&w, &h, layer).expect("layer must run");
            assert_eq!(out.residual.shape(), &[2, w.hidden_size]);
            assert!(
                out.residual.iter().all(|v| v.is_finite()),
                "layer {layer} guided walk residual must be finite"
            );
        }
    }
}
