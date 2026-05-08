//! Pure unit tests for `larql_server::band_utils`.
//!
//! No HTTP server is needed — all tests call the functions directly.

use larql_server::band_utils::{
    filter_layers_by_band, get_layer_bands, BAND_ALL, BAND_KNOWLEDGE, BAND_OUTPUT, BAND_SYNTAX,
    INFER_MODE_COMPARE, INFER_MODE_DENSE, INFER_MODE_WALK, INSERT_MODE_CONSTELLATION,
    INSERT_MODE_EMBEDDING,
};
use larql_server::ffn_l2_cache::FfnL2Cache;
use larql_server::state::LoadedModel;
use larql_vindex::ndarray::Array2;
use larql_vindex::{
    ExtractLevel, LayerBands, PatchedVindex, QuantFormat, VectorIndex, VindexConfig,
    VindexLayerInfo,
};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

// ══════════════════════════════════════════════════════════════
// BAND CONSTANTS
// ══════════════════════════════════════════════════════════════

#[test]
fn band_constants_correct_values() {
    assert_eq!(BAND_ALL, "all");
    assert_eq!(BAND_KNOWLEDGE, "knowledge");
    assert_eq!(BAND_OUTPUT, "output");
    assert_eq!(BAND_SYNTAX, "syntax");
}

#[test]
fn mode_constants_correct_values() {
    assert_eq!(INFER_MODE_WALK, "walk");
    assert_eq!(INFER_MODE_DENSE, "dense");
    assert_eq!(INFER_MODE_COMPARE, "compare");
}

#[test]
fn insert_mode_constants_correct_values() {
    assert_eq!(INSERT_MODE_CONSTELLATION, "constellation");
    assert_eq!(INSERT_MODE_EMBEDDING, "embedding");
}

// ══════════════════════════════════════════════════════════════
// filter_layers_by_band
// ══════════════════════════════════════════════════════════════

fn sample_bands() -> LayerBands {
    LayerBands {
        syntax: (0, 1),
        knowledge: (2, 3),
        output: (4, 4),
    }
}

fn all_layers() -> Vec<usize> {
    vec![0, 1, 2, 3, 4]
}

#[test]
fn filter_syntax_returns_syntax_layers() {
    let bands = sample_bands();
    let result = filter_layers_by_band(all_layers(), BAND_SYNTAX, &bands);
    assert_eq!(result, vec![0, 1]);
}

#[test]
fn filter_knowledge_returns_knowledge_layers() {
    let bands = sample_bands();
    let result = filter_layers_by_band(all_layers(), BAND_KNOWLEDGE, &bands);
    assert_eq!(result, vec![2, 3]);
}

#[test]
fn filter_output_returns_output_layers() {
    let bands = sample_bands();
    let result = filter_layers_by_band(all_layers(), BAND_OUTPUT, &bands);
    assert_eq!(result, vec![4]);
}

#[test]
fn filter_all_returns_all_layers() {
    let bands = sample_bands();
    let result = filter_layers_by_band(all_layers(), BAND_ALL, &bands);
    assert_eq!(result, vec![0, 1, 2, 3, 4]);
}

#[test]
fn filter_unknown_band_returns_all_layers() {
    let bands = sample_bands();
    let result = filter_layers_by_band(all_layers(), "other", &bands);
    assert_eq!(result, vec![0, 1, 2, 3, 4]);
}

#[test]
fn filter_empty_input_returns_empty() {
    let bands = sample_bands();
    let result = filter_layers_by_band(vec![], BAND_SYNTAX, &bands);
    assert!(result.is_empty());
}

#[test]
fn filter_no_match_in_band_returns_empty() {
    let bands = sample_bands(); // syntax=(0,1)
    let result = filter_layers_by_band(vec![5, 6, 7], BAND_SYNTAX, &bands);
    assert!(result.is_empty());
}

// ══════════════════════════════════════════════════════════════
// get_layer_bands
// ══════════════════════════════════════════════════════════════

fn make_minimal_model(layer_bands: Option<LayerBands>) -> Arc<LoadedModel> {
    let hidden = 4;
    let gate = Array2::<f32>::zeros((2, hidden));
    let index = VectorIndex::new(vec![Some(gate)], vec![None], 1, hidden);
    let patched = PatchedVindex::new(index);
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    let tokenizer = larql_vindex::tokenizers::Tokenizer::from_bytes(tok_json).unwrap();
    Arc::new(LoadedModel {
        id: "band-test".into(),
        path: PathBuf::from("/nonexistent"),
        config: VindexConfig {
            version: 2,
            model: "test/model".to_string(),
            family: "test".to_string(),
            source: None,
            checksums: None,
            num_layers: 5,
            hidden_size: hidden,
            intermediate_size: 8,
            vocab_size: 4,
            embed_scale: 1.0,
            extract_level: ExtractLevel::Browse,
            dtype: larql_vindex::StorageDtype::default(),
            quant: QuantFormat::None,
            layer_bands,
            layers: vec![VindexLayerInfo {
                layer: 0,
                num_features: 2,
                offset: 0,
                length: 32,
                num_experts: None,
                num_features_per_expert: None,
            }],
            down_top_k: 2,
            has_model_weights: false,
            model_config: None,
            fp4: None,
            ffn_layout: None,
        },
        patched: tokio::sync::RwLock::new(patched),
        embeddings: Array2::<f32>::zeros((4, hidden)),
        embed_scale: 1.0,
        tokenizer,
        infer_disabled: true,
        ffn_only: false,
        embed_only: false,
        embed_store: None,
        release_mmap_after_request: false,
        weights: std::sync::OnceLock::new(),
        probe_labels: HashMap::new(),
        ffn_l2_cache: FfnL2Cache::new(1),
        expert_filter: None,
        unit_filter: None,
        moe_remote: None,
        #[cfg(feature = "metal-experts")]
        metal_backend: std::sync::OnceLock::new(),
        #[cfg(feature = "metal-experts")]
        moe_scratches: std::sync::Mutex::new(std::collections::HashMap::new()),
        #[cfg(feature = "metal-experts")]
        metal_ffn_layer_bufs: std::sync::OnceLock::new(),
    })
}

#[test]
fn get_layer_bands_uses_config_bands_when_present() {
    let explicit_bands = LayerBands {
        syntax: (0, 1),
        knowledge: (2, 3),
        output: (4, 4),
    };
    let model = make_minimal_model(Some(explicit_bands.clone()));
    let bands = get_layer_bands(&model);
    assert_eq!(bands.syntax, explicit_bands.syntax);
    assert_eq!(bands.knowledge, explicit_bands.knowledge);
    assert_eq!(bands.output, explicit_bands.output);
}

#[test]
fn get_layer_bands_falls_back_when_none() {
    // When layer_bands is None and family is "test" (no known mapping),
    // falls back to the flat-all-layers default: syntax=(0,last), etc.
    let model = make_minimal_model(None);
    let bands = get_layer_bands(&model);
    // The flat fallback sets all bands to (0, num_layers-1) = (0, 4).
    let last = model.config.num_layers.saturating_sub(1);
    assert_eq!(bands.syntax.0, 0);
    assert_eq!(bands.syntax.1, last);
}

#[test]
fn filter_knowledge_with_zero_width_band() {
    // Edge case: knowledge band covers only layer 2 (start == end).
    let bands = LayerBands {
        syntax: (0, 0),
        knowledge: (2, 2),
        output: (3, 3),
    };
    let all = vec![0, 1, 2, 3, 4];
    let result = filter_layers_by_band(all, BAND_KNOWLEDGE, &bands);
    assert_eq!(result, vec![2]);
}
