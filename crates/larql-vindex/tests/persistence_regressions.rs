//! Persistence regressions for vindex load/save invariants.
//!
//! These tests are synthetic and architecture-agnostic: they exercise storage
//! contracts without downloading models or assuming a model family.

use larql_vindex::format::filenames::{
    DOWN_META_BIN, EMBEDDINGS_BIN, GATE_VECTORS_BIN, TOKENIZER_JSON, WEIGHT_MANIFEST_JSON,
};
use larql_vindex::{FeatureMeta, VectorIndex, VindexConfig, VindexLayerInfo};
use ndarray::{Array1, Array2};
use tempfile::tempdir;

const NUM_LAYERS: usize = 2;
const HIDDEN_SIZE: usize = 4;
const FEATURES_PER_LAYER: usize = 3;
const INTERMEDIATE_SIZE: usize = 3;
const VOCAB_SIZE: usize = 100;
const DOWN_TOP_K: usize = 1;
const MODEL_NAME: &str = "test";
const MODEL_FAMILY: &str = "test";
const MINIMAL_TOKENIZER_JSON: &str =
    r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
const LEGACY_DOWN_META_JSONL: &str = "down_meta.jsonl";

const GATE_STRONG: f32 = 1.0;
const GATE_HALF: f32 = 0.5;
const GATE_NEGATIVE: f32 = -1.0;

const TOKEN_PARIS: &str = "Paris";
const META_PARIS_ID: u32 = 100;
const META_PARIS_SCORE: f32 = 0.95;
const TOKEN_FRENCH: &str = "French";
const META_FRENCH_ID: u32 = 101;
const META_FRENCH_SCORE: f32 = 0.88;
const TOKEN_EUROPE: &str = "Europe";
const META_EUROPE_ID: u32 = 102;
const META_EUROPE_SCORE: f32 = 0.75;
const TOKEN_BERLIN: &str = "Berlin";
const META_BERLIN_ID: u32 = 200;
const META_BERLIN_SCORE: f32 = 0.90;
const TOKEN_SPAIN: &str = "Spain";
const META_SPAIN_ID: u32 = 202;
const META_SPAIN_SCORE: f32 = 0.70;
const LEGACY_JSONL_TOKEN: &str = "Lisbon";
const LEGACY_JSONL_TOKEN_ID: u32 = 321;
const LEGACY_JSONL_SCORE: f32 = 0.77;

const F32_BYTES: usize = std::mem::size_of::<f32>();
const F16_BYTES: usize = 2;
const DOWN_META_MAGIC_LITERAL: &[u8; 4] = b"DMET";
const DOWN_META_FORMAT_VERSION: u32 = 1;
const DOWN_META_TRUNCATED_FEATURES: u32 = 3;

const MOE_EXPERTS: usize = 2;
const MOE_FEATURES_PER_EXPERT: usize = 2;

const COMPACT_NUM_LAYERS: usize = 2;
const COMPACT_HIDDEN_SIZE: usize = 8;
const COMPACT_INTERMEDIATE_SIZE: usize = 4;
const COMPACT_VOCAB_SIZE: usize = 16;
const COMPACT_MODEL_TYPE: &str = "llama";
const COMPACT_MODEL_NAME: &str = "test/compact";
const COMPACT_ERROR_FRAGMENT: &str = "compact FFN vindexes";
const COMPACT_ROPE_BASE: f64 = 10000.0;
const EMPTY_WEIGHT_MANIFEST_JSON: &str = "[]";

fn make_top_k(token: &str, id: u32, logit: f32) -> larql_models::TopKEntry {
    larql_models::TopKEntry {
        token: token.to_string(),
        token_id: id,
        logit,
    }
}

fn make_meta(token: &str, id: u32, score: f32) -> FeatureMeta {
    FeatureMeta {
        top_token: token.to_string(),
        top_token_id: id,
        c_score: score,
        top_k: vec![make_top_k(token, id, score)],
    }
}

fn test_index() -> VectorIndex {
    let mut gate0 = Array2::<f32>::zeros((FEATURES_PER_LAYER, HIDDEN_SIZE));
    gate0[[0, 0]] = GATE_STRONG;
    gate0[[1, 1]] = GATE_STRONG;
    gate0[[2, 2]] = GATE_STRONG;

    let mut gate1 = Array2::<f32>::zeros((FEATURES_PER_LAYER, HIDDEN_SIZE));
    gate1[[0, 3]] = GATE_STRONG;
    gate1[[1, 0]] = GATE_HALF;
    gate1[[1, 1]] = GATE_HALF;
    gate1[[2, 2]] = GATE_NEGATIVE;

    let meta0 = vec![
        Some(make_meta(TOKEN_PARIS, META_PARIS_ID, META_PARIS_SCORE)),
        Some(make_meta(TOKEN_FRENCH, META_FRENCH_ID, META_FRENCH_SCORE)),
        Some(make_meta(TOKEN_EUROPE, META_EUROPE_ID, META_EUROPE_SCORE)),
    ];
    let meta1 = vec![
        Some(make_meta(TOKEN_BERLIN, META_BERLIN_ID, META_BERLIN_SCORE)),
        None,
        Some(make_meta(TOKEN_SPAIN, META_SPAIN_ID, META_SPAIN_SCORE)),
    ];

    VectorIndex::new(
        vec![Some(gate0), Some(gate1)],
        vec![Some(meta0), Some(meta1)],
        NUM_LAYERS,
        HIDDEN_SIZE,
    )
}

fn write_minimal_tokenizer(dir: &std::path::Path) {
    std::fs::write(dir.join(TOKENIZER_JSON), MINIMAL_TOKENIZER_JSON).unwrap();
}

fn test_vindex_config(dtype: larql_vindex::StorageDtype) -> VindexConfig {
    VindexConfig {
        version: 2,
        model: MODEL_NAME.into(),
        family: MODEL_FAMILY.into(),
        num_layers: NUM_LAYERS,
        hidden_size: HIDDEN_SIZE,
        intermediate_size: INTERMEDIATE_SIZE,
        vocab_size: VOCAB_SIZE,
        embed_scale: 1.0,
        layers: vec![],
        down_top_k: DOWN_TOP_K,
        has_model_weights: false,
        source: None,
        checksums: None,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        model_config: None,
        fp4: None,
        ffn_layout: None,
    }
}

fn gate_file_bytes(dtype_bytes: usize) -> u64 {
    (NUM_LAYERS * FEATURES_PER_LAYER * HIDDEN_SIZE * dtype_bytes) as u64
}

fn dim0_query() -> Array1<f32> {
    let mut query = vec![0.0; HIDDEN_SIZE];
    query[0] = GATE_STRONG;
    Array1::from_vec(query)
}

fn layer_info(layer: usize, offset: u64, length: u64, moe_fields: bool) -> VindexLayerInfo {
    VindexLayerInfo {
        layer,
        num_features: FEATURES_PER_LAYER,
        offset,
        length,
        num_experts: moe_fields.then_some(MOE_EXPERTS),
        num_features_per_expert: moe_fields.then_some(MOE_FEATURES_PER_EXPERT),
    }
}

#[test]
fn save_vindex_preserves_f16_gate_dtype() {
    let idx = test_index();
    let dir = tempdir().unwrap();
    write_minimal_tokenizer(dir.path());

    let mut config = test_vindex_config(larql_vindex::StorageDtype::F16);
    idx.save_vindex(dir.path(), &mut config).unwrap();

    let gate_size = std::fs::metadata(dir.path().join(GATE_VECTORS_BIN))
        .unwrap()
        .len();
    assert_eq!(gate_size, gate_file_bytes(F16_BYTES));
    assert_eq!(config.dtype, larql_vindex::StorageDtype::F16);

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let loaded = VectorIndex::load_vindex(dir.path(), &mut cb).unwrap();
    let query = dim0_query();
    let hits = loaded.gate_knn(0, &query, DOWN_TOP_K);
    assert_eq!(hits[0].0, 0);
}

#[test]
fn save_vindex_preserves_mmap_down_meta() {
    let idx = test_index();
    let src = tempdir().unwrap();
    let dst = tempdir().unwrap();
    write_minimal_tokenizer(src.path());
    write_minimal_tokenizer(dst.path());

    let mut config = test_vindex_config(larql_vindex::StorageDtype::F32);
    idx.save_vindex(src.path(), &mut config).unwrap();

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let loaded = VectorIndex::load_vindex(src.path(), &mut cb).unwrap();
    assert!(loaded.metadata.down_meta_mmap.is_some());
    assert_eq!(loaded.total_down_meta(), NUM_LAYERS * FEATURES_PER_LAYER);

    let mut out_config = config.clone();
    loaded.save_vindex(dst.path(), &mut out_config).unwrap();

    let reloaded = VectorIndex::load_vindex(dst.path(), &mut cb).unwrap();
    assert_eq!(reloaded.total_down_meta(), NUM_LAYERS * FEATURES_PER_LAYER);
    assert_eq!(
        reloaded.feature_meta(0, 0).unwrap().top_token_id,
        META_PARIS_ID
    );
    assert!(reloaded.feature_meta(1, 1).is_none());
}

#[test]
fn save_vindex_preserves_layer_moe_metadata() {
    let idx = test_index();
    let dir = tempdir().unwrap();
    write_minimal_tokenizer(dir.path());

    let layer_len = (FEATURES_PER_LAYER * HIDDEN_SIZE * F32_BYTES) as u64;
    let mut config = test_vindex_config(larql_vindex::StorageDtype::F32);
    config.layers = vec![
        layer_info(0, 0, layer_len, true),
        layer_info(1, layer_len, layer_len, true),
    ];

    idx.save_vindex(dir.path(), &mut config).unwrap();
    assert_eq!(config.layers[0].num_experts, Some(MOE_EXPERTS));
    assert_eq!(
        config.layers[0].num_features_per_expert,
        Some(MOE_FEATURES_PER_EXPERT)
    );
    assert_eq!(config.layers[1].num_experts, Some(MOE_EXPERTS));
}

#[test]
fn load_vindex_falls_back_to_legacy_down_meta_jsonl() {
    let idx = test_index();
    let dir = tempdir().unwrap();
    write_minimal_tokenizer(dir.path());

    let layer_infos = idx.save_gate_vectors(dir.path()).unwrap();
    let mut config = test_vindex_config(larql_vindex::StorageDtype::F32);
    config.layers = layer_infos;
    VectorIndex::save_config(&config, dir.path()).unwrap();

    let legacy_record = serde_json::json!({
        "layer": 0,
        "feature": 1,
        "top_token": LEGACY_JSONL_TOKEN,
        "top_token_id": LEGACY_JSONL_TOKEN_ID,
        "c_score": LEGACY_JSONL_SCORE,
        "top_k": [{
            "token": LEGACY_JSONL_TOKEN,
            "token_id": LEGACY_JSONL_TOKEN_ID,
            "logit": LEGACY_JSONL_SCORE,
        }],
    });
    std::fs::write(
        dir.path().join(LEGACY_DOWN_META_JSONL),
        serde_json::to_string(&legacy_record).unwrap(),
    )
    .unwrap();

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let loaded = VectorIndex::load_vindex(dir.path(), &mut cb).unwrap();
    let meta = loaded.feature_meta(0, 1).unwrap();
    assert_eq!(meta.top_token, LEGACY_JSONL_TOKEN);
    assert_eq!(meta.top_token_id, LEGACY_JSONL_TOKEN_ID);
}

#[test]
fn load_vindex_rejects_truncated_down_meta_bin() {
    let idx = test_index();
    let dir = tempdir().unwrap();
    write_minimal_tokenizer(dir.path());

    let layer_infos = idx.save_gate_vectors(dir.path()).unwrap();
    let mut config = test_vindex_config(larql_vindex::StorageDtype::F32);
    config.layers = layer_infos;
    VectorIndex::save_config(&config, dir.path()).unwrap();

    let mut bytes = Vec::new();
    bytes.extend_from_slice(DOWN_META_MAGIC_LITERAL);
    bytes.extend_from_slice(&DOWN_META_FORMAT_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(NUM_LAYERS as u32).to_le_bytes());
    bytes.extend_from_slice(&(DOWN_TOP_K as u32).to_le_bytes());
    bytes.extend_from_slice(&DOWN_META_TRUNCATED_FEATURES.to_le_bytes());
    std::fs::write(dir.path().join(DOWN_META_BIN), bytes).unwrap();

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let err = match VectorIndex::load_vindex(dir.path(), &mut cb) {
        Ok(_) => panic!("truncated down_meta.bin should fail"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains("truncated down_meta.bin"),
        "unexpected error: {err}"
    );
}

fn compact_model_config() -> larql_vindex::VindexModelConfig {
    larql_vindex::VindexModelConfig {
        model_type: COMPACT_MODEL_TYPE.into(),
        head_dim: COMPACT_HIDDEN_SIZE,
        num_q_heads: 1,
        num_kv_heads: 1,
        rope_base: COMPACT_ROPE_BASE,
        sliding_window: None,
        moe: None,
        global_head_dim: None,
        num_global_kv_heads: None,
        partial_rotary_factor: None,
        sliding_window_pattern: None,
        layer_types: None,
        attention_k_eq_v: false,
        num_kv_shared_layers: None,
        per_layer_embed_dim: None,
        rope_local_base: None,
        query_pre_attn_scalar: None,
        final_logit_softcapping: None,
        attention_multiplier: None,
        residual_multiplier: None,
        logits_scaling: None,
        norm_eps: None,
    }
}

#[test]
fn load_model_weights_rejects_compact_missing_dense_ffn() {
    let dir = tempdir().unwrap();
    let mut config = VindexConfig {
        version: 2,
        model: COMPACT_MODEL_NAME.into(),
        family: COMPACT_MODEL_TYPE.into(),
        source: None,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        checksums: None,
        num_layers: COMPACT_NUM_LAYERS,
        hidden_size: COMPACT_HIDDEN_SIZE,
        intermediate_size: COMPACT_INTERMEDIATE_SIZE,
        vocab_size: COMPACT_VOCAB_SIZE,
        embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::All,
        has_model_weights: true,
        layer_bands: None,
        layers: vec![],
        down_top_k: DOWN_TOP_K,
        model_config: Some(compact_model_config()),
        fp4: None,
        ffn_layout: None,
    };

    let gate = vec![0.0f32; COMPACT_NUM_LAYERS * COMPACT_INTERMEDIATE_SIZE * COMPACT_HIDDEN_SIZE];
    let gate_bytes =
        larql_vindex::config::dtype::encode_floats(&gate, larql_vindex::StorageDtype::F32);
    std::fs::write(dir.path().join(GATE_VECTORS_BIN), gate_bytes).unwrap();

    let compact_layer_len = (COMPACT_INTERMEDIATE_SIZE * COMPACT_HIDDEN_SIZE * F32_BYTES) as u64;
    config.layers = vec![
        VindexLayerInfo {
            layer: 0,
            num_features: COMPACT_INTERMEDIATE_SIZE,
            offset: 0,
            length: compact_layer_len,
            num_experts: None,
            num_features_per_expert: None,
        },
        VindexLayerInfo {
            layer: 1,
            num_features: COMPACT_INTERMEDIATE_SIZE,
            offset: compact_layer_len,
            length: compact_layer_len,
            num_experts: None,
            num_features_per_expert: None,
        },
    ];
    VectorIndex::save_config(&config, dir.path()).unwrap();

    let embed = vec![0.0f32; COMPACT_VOCAB_SIZE * COMPACT_HIDDEN_SIZE];
    let embed_bytes =
        larql_vindex::config::dtype::encode_floats(&embed, larql_vindex::StorageDtype::F32);
    std::fs::write(dir.path().join(EMBEDDINGS_BIN), embed_bytes).unwrap();
    std::fs::write(
        dir.path().join(WEIGHT_MANIFEST_JSON),
        EMPTY_WEIGHT_MANIFEST_JSON,
    )
    .unwrap();

    let mut cb = larql_vindex::SilentLoadCallbacks;
    let err = match larql_vindex::load_model_weights(dir.path(), &mut cb) {
        Ok(_) => panic!("compact vindex without dense FFN tensors should fail"),
        Err(err) => err,
    };
    assert!(
        err.to_string().contains(COMPACT_ERROR_FRAGMENT),
        "unexpected error: {err}"
    );
}
