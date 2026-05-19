//! Entity probing — run tokens through the model to confirm which features fire.
//!
//! For each entity in the reference data, embed it, run gate KNN at each
//! knowledge layer, and record which features activate. This gives confirmed
//! (entity, feature) mappings — ground truth for pair-based relation labeling.

use std::collections::HashMap;

use ndarray::Array1;

use crate::VectorIndex;

/// Result of probing: maps (layer, feature) → list of entity names that activate it.
pub struct ProbeResult {
    /// (layer, feature) → entity names
    pub feature_entities: HashMap<(usize, usize), Vec<String>>,
    /// Total entities probed
    pub num_entities: usize,
    /// Total (entity, feature) activations found
    pub num_activations: usize,
}

/// Probe a set of entities against a VectorIndex to find which features they activate.
///
/// For each entity:
/// 1. Look up its embedding (may be multi-token → averaged)
/// 2. Run gate KNN at each knowledge layer
/// 3. Record the top-K features that fire
///
/// Returns a map from (layer, feature) → entity names.
pub fn probe_entities(
    entities: &[String],
    index: &VectorIndex,
    embed: &ndarray::Array2<f32>,
    embed_scale: f32,
    tokenizer: &tokenizers::Tokenizer,
    layers: &[usize],
    top_k: usize,
) -> ProbeResult {
    let hidden = embed.shape()[1];
    let vocab_size = embed.shape()[0];
    let mut feature_entities: HashMap<(usize, usize), Vec<String>> = HashMap::new();
    let mut num_activations = 0;

    let total = entities.len();
    for (ei, entity) in entities.iter().enumerate() {
        if ei % 1000 == 0 && ei > 0 {
            eprint!(
                "\r    Probed {}/{} entities ({} activations)...",
                ei, total, num_activations
            );
        }
        // Encode entity → token IDs → averaged embedding
        let encoding = match tokenizer.encode(entity.as_str(), false) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let ids = encoding.get_ids();
        if ids.is_empty() {
            continue;
        }

        let mut avg = Array1::<f32>::zeros(hidden);
        let mut n = 0;
        for &id in ids {
            if id > 2 && (id as usize) < vocab_size {
                avg += &embed.row(id as usize).mapv(|v| v * embed_scale);
                n += 1;
            }
        }
        if n == 0 {
            continue;
        }
        avg /= n as f32;

        // Run gate KNN at each layer — only keep strong activations
        for &layer in layers {
            let hits = index.gate_knn(layer, &avg, top_k);
            // Only keep the top 3 strongest activations per layer (most specific)
            for (feature, score) in hits.into_iter().take(3) {
                if score > 5.0 {
                    feature_entities
                        .entry((layer, feature))
                        .or_default()
                        .push(entity.clone());
                    num_activations += 1;
                }
            }
        }
    }

    if total > 1000 {
        eprintln!(
            "\r    Probed {}/{} entities ({} activations)    ",
            total, total, num_activations
        );
    }

    ProbeResult {
        feature_entities,
        num_entities: entities.len(),
        num_activations,
    }
}

/// Extract unique entity names from the reference triple files.
pub fn extract_probe_entities(triples_path: &std::path::Path) -> Vec<String> {
    let mut entities = std::collections::HashSet::new();

    if let Ok(text) = std::fs::read_to_string(triples_path) {
        if let Ok(data) = serde_json::from_str::<serde_json::Value>(&text) {
            if let Some(obj) = data.as_object() {
                for (_rel, val) in obj {
                    if let Some(pairs) = val.get("pairs").and_then(|v| v.as_array()) {
                        for pair in pairs {
                            if let Some(arr) = pair.as_array() {
                                // Add both subject and object as probe entities
                                for item in arr {
                                    if let Some(s) = item.as_str() {
                                        let s = s.trim();
                                        // Only probe short, clean entity names
                                        if s.len() >= 2
                                            && s.len() <= 30
                                            && !s.contains('(')
                                            && !s.contains("http")
                                        {
                                            entities.insert(s.to_string());
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut sorted: Vec<String> = entities.into_iter().collect();
    sorted.sort();
    sorted
}

/// Build confirmed (input_entity, output_token) pairs for each feature
/// by combining probe results with down_meta output tokens.
///
/// Only includes features that fire SELECTIVELY — if a feature fires for
/// too many entities, it's a category feature, not a specific fact.
/// Max 20 entities per feature keeps only the specific ones.
pub fn build_confirmed_pairs(
    probe: &ProbeResult,
    index: &VectorIndex,
) -> Vec<(String, String, usize, usize)> {
    // (entity, target, layer, feature)
    let mut pairs = Vec::new();
    let mut skipped_broad = 0;

    for (&(layer, feature), entities) in &probe.feature_entities {
        // Skip broad features that fire for many entities — they're category features
        if entities.len() > 20 {
            skipped_broad += 1;
            continue;
        }

        if let Some(meta) = index.feature_meta(layer, feature) {
            let target = &meta.top_token;
            if target.len() >= 2 {
                for entity in entities {
                    pairs.push((entity.clone(), target.clone(), layer, feature));
                }
            }
        }
    }

    if skipped_broad > 0 {
        eprintln!(
            "  Skipped {} broad features (>20 entities each)",
            skipped_broad
        );
    }

    pairs
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn extract_entities_from_json() {
        let dir = std::env::temp_dir().join("probe_test");
        std::fs::create_dir_all(&dir).ok();
        let path = dir.join("test_triples.json");
        std::fs::write(
            &path,
            r#"{
            "capital": {"pairs": [["France", "Paris"], ["Germany", "Berlin"]]},
            "language": {"pairs": [["France", "French"]]}
        }"#,
        )
        .unwrap();

        let entities = extract_probe_entities(&path);
        assert!(entities.contains(&"France".to_string()));
        assert!(entities.contains(&"Paris".to_string()));
        assert!(entities.contains(&"Berlin".to_string()));
        assert!(entities.contains(&"French".to_string()));
        assert!(entities.contains(&"Germany".to_string()));

        std::fs::remove_file(&path).ok();
    }

    use larql_models::TopKEntry;

    fn empty_tokenizer() -> tokenizers::Tokenizer {
        use tokenizers::models::wordlevel::WordLevel;
        use tokenizers::TokenizerBuilder;
        let model = WordLevel::builder()
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        TokenizerBuilder::<
            tokenizers::models::wordlevel::WordLevel,
            tokenizers::normalizers::NormalizerWrapper,
            tokenizers::pre_tokenizers::PreTokenizerWrapper,
            tokenizers::processors::PostProcessorWrapper,
            tokenizers::decoders::DecoderWrapper,
        >::default()
        .with_model(model)
        .build()
        .unwrap()
        .into()
    }

    fn meta(token: &str, score: f32) -> crate::FeatureMeta {
        crate::FeatureMeta {
            top_token: token.to_string(),
            top_token_id: 1,
            c_score: score,
            top_k: vec![TopKEntry {
                token: token.to_string(),
                token_id: 1,
                logit: score,
            }],
        }
    }

    fn vindex_with_meta(
        num_layers: usize,
        hidden: usize,
        per_layer_meta: &[Vec<Option<crate::FeatureMeta>>],
    ) -> VectorIndex {
        let mut v = VectorIndex::empty(num_layers, hidden);
        for (layer, metas) in per_layer_meta.iter().enumerate() {
            if layer < num_layers {
                v.metadata.down_meta[layer] = Some(metas.clone());
            }
        }
        v
    }

    // ── extract_probe_entities ────────────────────────────────────

    #[test]
    fn extract_entities_returns_empty_for_missing_file() {
        let path = std::path::Path::new("/tmp/does-not-exist-probe-test.json");
        let entities = extract_probe_entities(path);
        assert!(entities.is_empty());
    }

    #[test]
    fn extract_entities_returns_empty_for_invalid_json() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bad.json");
        std::fs::write(&path, "not json {").unwrap();
        assert!(extract_probe_entities(&path).is_empty());
    }

    #[test]
    fn extract_entities_filters_long_names_and_parens_and_urls() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("filtered.json");
        std::fs::write(
            &path,
            r#"{
                "rel": {"pairs": [
                    ["a", "ok"],
                    ["this is way too long an entity name to keep around xxxx", "y"],
                    ["with(parens)", "z"],
                    ["http://urls", "u"],
                    [" Trim ", "t"]
                ]}
            }"#,
        )
        .unwrap();

        let entities = extract_probe_entities(&path);
        assert!(entities.contains(&"ok".to_string()));
        assert!(entities.contains(&"Trim".to_string()), "trims whitespace");
        // Filters: length cap, parens, urls.
        assert!(!entities
            .iter()
            .any(|e| e.starts_with("this is way too long")));
        assert!(!entities.iter().any(|e| e.contains('(')));
        assert!(!entities.iter().any(|e| e.contains("http")));
        // Single-char entities filtered (length >= 2 only).
        assert!(!entities.iter().any(|e| e.len() < 2));
    }

    #[test]
    fn extract_entities_returns_sorted_unique_set() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("dupes.json");
        std::fs::write(
            &path,
            r#"{
                "r1": {"pairs": [["zb", "ab"], ["zb", "yc"]]},
                "r2": {"pairs": [["ab", "zb"]]}
            }"#,
        )
        .unwrap();

        let entities = extract_probe_entities(&path);
        // Three unique strings, ascending.
        assert_eq!(
            entities,
            vec!["ab".to_string(), "yc".to_string(), "zb".to_string()]
        );
    }

    // ── build_confirmed_pairs ────────────────────────────────────

    #[test]
    fn build_confirmed_pairs_empty_probe_returns_empty() {
        let v = VectorIndex::empty(2, 4);
        let probe = ProbeResult {
            feature_entities: HashMap::new(),
            num_entities: 0,
            num_activations: 0,
        };
        let pairs = build_confirmed_pairs(&probe, &v);
        assert!(pairs.is_empty());
    }

    #[test]
    fn build_confirmed_pairs_emits_one_per_entity_per_feature() {
        let v = vindex_with_meta(
            2,
            4,
            &[
                vec![Some(meta("Paris", 0.9))],        // layer 0, feat 0
                vec![None, Some(meta("French", 0.8))], // layer 1, feat 1
            ],
        );
        let mut probe = ProbeResult {
            feature_entities: HashMap::new(),
            num_entities: 2,
            num_activations: 0,
        };
        probe
            .feature_entities
            .insert((0, 0), vec!["France".into(), "Italy".into()]);
        probe.feature_entities.insert((1, 1), vec!["France".into()]);

        let mut pairs = build_confirmed_pairs(&probe, &v);
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                ("France".to_string(), "French".to_string(), 1, 1),
                ("France".to_string(), "Paris".to_string(), 0, 0),
                ("Italy".to_string(), "Paris".to_string(), 0, 0),
            ]
        );
    }

    #[test]
    fn build_confirmed_pairs_skips_broad_features() {
        // 21 entities → broader-than-20 → skipped.
        let v = vindex_with_meta(1, 4, &[vec![Some(meta("X", 0.5))]]);
        let mut probe = ProbeResult {
            feature_entities: HashMap::new(),
            num_entities: 21,
            num_activations: 0,
        };
        let many: Vec<String> = (0..21).map(|i| format!("e{i}")).collect();
        probe.feature_entities.insert((0, 0), many);

        let pairs = build_confirmed_pairs(&probe, &v);
        assert!(pairs.is_empty(), "broad features get filtered");
    }

    #[test]
    fn build_confirmed_pairs_skips_features_without_meta() {
        // VectorIndex returns None for feature_meta(0, 0) because no
        // down_meta is populated.
        let v = VectorIndex::empty(1, 4);
        let mut probe = ProbeResult {
            feature_entities: HashMap::new(),
            num_entities: 1,
            num_activations: 0,
        };
        probe.feature_entities.insert((0, 0), vec!["alpha".into()]);

        let pairs = build_confirmed_pairs(&probe, &v);
        assert!(pairs.is_empty(), "missing meta → skip");
    }

    #[test]
    fn build_confirmed_pairs_skips_single_char_top_token() {
        // top_token of length 1 is filtered (target.len() < 2).
        let v = vindex_with_meta(1, 4, &[vec![Some(meta("a", 0.5))]]);
        let mut probe = ProbeResult {
            feature_entities: HashMap::new(),
            num_entities: 1,
            num_activations: 0,
        };
        probe.feature_entities.insert((0, 0), vec!["alpha".into()]);
        let pairs = build_confirmed_pairs(&probe, &v);
        assert!(pairs.is_empty());
    }

    // ── probe_entities ────────────────────────────────────────────

    #[test]
    fn probe_entities_returns_empty_result_on_empty_input() {
        let v = VectorIndex::empty(2, 4);
        let embed = ndarray::Array2::<f32>::zeros((16, 4));
        let tok = empty_tokenizer();
        let result = probe_entities(&[], &v, &embed, 1.0, &tok, &[0, 1], 5);
        assert!(result.feature_entities.is_empty());
        assert_eq!(result.num_entities, 0);
        assert_eq!(result.num_activations, 0);
    }

    #[test]
    fn probe_entities_skips_unencodable_entities() {
        // The trivial WordLevel tokenizer with an empty vocab returns
        // [UNK] (id 0) for every word — id 0 fails the `id > 2` filter,
        // so n=0 for every entity and the inner loop short-circuits.
        // Result: zero activations even with non-empty input.
        let v = VectorIndex::empty(2, 4);
        let embed = ndarray::Array2::<f32>::ones((16, 4));
        let tok = empty_tokenizer();
        let result = probe_entities(
            &["alpha".into(), "beta".into()],
            &v,
            &embed,
            1.0,
            &tok,
            &[0, 1],
            5,
        );
        assert_eq!(result.num_entities, 2);
        assert_eq!(result.num_activations, 0);
        assert!(result.feature_entities.is_empty());
    }
}
