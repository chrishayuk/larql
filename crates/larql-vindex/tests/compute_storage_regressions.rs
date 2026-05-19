//! Focused coverage for compute/storage modules with synthetic fixtures.

use std::io::Write;

use larql_compute::{ComputeBackend, DecodeBackend, MatMul, QuantMatVec};
use larql_vindex::format::filenames::{layer_weights_filename, ROUTER_WEIGHTS_BIN};
use larql_vindex::format::weights::write_layers::{
    bf16_bytes_to_f32, pad_cols_to_256, parse_layer_weights_header, quantize_dense_entry,
    quantize_moe_entries, write_layer_weights, LayerEntry, LayerWeightFormat,
};
use larql_vindex::format::{down_meta, filenames::DOWN_META_BIN};
use larql_vindex::{
    ExtractLevel, FeatureMeta, MoeConfig, QuantFormat, RouterIndex, StorageDtype, VectorIndex,
    VindexConfig, VindexModelConfig,
};
use ndarray::{array, Array2};
use tempfile::tempdir;

const NUM_LAYERS: usize = 2;
const HIDDEN_SIZE: usize = 3;
const NUM_EXPERTS: usize = 3;
const ROUTER_TOP_K: usize = 2;
const VOCAB_SIZE: usize = 16;
const INTERMEDIATE_SIZE: usize = 4;
const DOWN_TOP_K: usize = 1;
const ROPE_BASE: f64 = 10000.0;

const LAYER_INDEX: usize = 7;
const LAYER_INTERMEDIATE: usize = 3;
const LAYER_HIDDEN: usize = 2;
const F32_BYTES: usize = std::mem::size_of::<f32>();
const U32_BYTES: usize = std::mem::size_of::<u32>();
const U64_BYTES: usize = std::mem::size_of::<u64>();
const LAYER_HEADER_FIELDS: usize = 6;
const LAYER_OFFSET_FIELDS: usize = 4;
const LAYER_HEADER_BYTES: usize = LAYER_HEADER_FIELDS * U32_BYTES;
const LAYER_OFFSET_BYTES: usize = LAYER_OFFSET_FIELDS * U64_BYTES;
const LAYER_MAGIC: u32 = u32::from_le_bytes(*b"LYRW");
const UNSUPPORTED_VERSION: u32 = 99;

const GATE_FEATURES: usize = 2;
const GATE_SEQ_LEN: usize = 2;
const TEST_GATE_BIN: &str = "gate.bin";
const TEST_GATE_F16_BIN: &str = "gate_f16.bin";
const TEST_GATE_F16_BACKEND_BIN: &str = "gate_f16_backend.bin";
const DOWN_META_TOP_K: usize = 2;
const DOWN_META_MAGIC: u32 = 0x444D4554;
const DOWN_META_VERSION: u32 = 1;
const TOKEN_ALPHA_ID: u32 = 11;
const TOKEN_BETA_ID: u32 = 12;
const TOKEN_ALPHA_SCORE: f32 = 0.75;
const TOKEN_BETA_SCORE: f32 = 0.5;
const MINIMAL_TOKENIZER_JSON: &str =
    r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
const FAKE_BACKEND_NAME: &str = "fake-gemv";

struct FakeGemvBackend;

impl MatMul for FakeGemvBackend {
    fn matmul(
        &self,
        a: ndarray::ArrayView2<f32>,
        b: ndarray::ArrayView2<f32>,
    ) -> ndarray::Array2<f32> {
        a.dot(&b)
    }

    fn matmul_transb(
        &self,
        a: ndarray::ArrayView2<f32>,
        b: ndarray::ArrayView2<f32>,
    ) -> ndarray::Array2<f32> {
        a.dot(&b.t())
    }

    fn f32_gemv_force(&self, w: ndarray::ArrayView2<f32>, x: &[f32]) -> Option<Vec<f32>> {
        Some(
            w.rows()
                .into_iter()
                .map(|row| row.iter().zip(x).map(|(lhs, rhs)| lhs * rhs).sum())
                .collect(),
        )
    }

    fn f16_gemv_force(&self, w_f16: &[u8], x: &[f32], n: usize, k: usize) -> Option<Vec<f32>> {
        let decoded = larql_models::quant::half::decode_f16(w_f16);
        Some(
            decoded
                .chunks_exact(k)
                .take(n)
                .map(|row| row.iter().zip(x).map(|(lhs, rhs)| lhs * rhs).sum())
                .collect(),
        )
    }
}

impl QuantMatVec for FakeGemvBackend {}
impl DecodeBackend for FakeGemvBackend {}

impl ComputeBackend for FakeGemvBackend {
    fn name(&self) -> &str {
        FAKE_BACKEND_NAME
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn repeat_bf16(bytes: [u8; 2], count: usize) -> Vec<u8> {
    (0..count).flat_map(|_| bytes).collect()
}

fn error_string<T>(result: Result<T, larql_vindex::VindexError>) -> String {
    match result {
        Ok(_) => panic!("expected error"),
        Err(err) => err.to_string(),
    }
}

fn tokenizer() -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_bytes(MINIMAL_TOKENIZER_JSON.as_bytes()).unwrap()
}

fn feature_meta(token_id: u32, score: f32) -> FeatureMeta {
    FeatureMeta {
        top_token: format!("T{token_id}"),
        top_token_id: token_id,
        c_score: score,
        top_k: vec![larql_models::TopKEntry {
            token: format!("T{token_id}"),
            token_id,
            logit: score,
        }],
    }
}

fn test_config(model_config: Option<VindexModelConfig>) -> VindexConfig {
    VindexConfig {
        version: 2,
        model: "synthetic".into(),
        family: "synthetic".into(),
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
        extract_level: ExtractLevel::Browse,
        dtype: StorageDtype::F32,
        quant: QuantFormat::None,
        layer_bands: None,
        model_config,
        fp4: None,
        ffn_layout: None,
    }
}

fn moe_model_config() -> VindexModelConfig {
    VindexModelConfig {
        model_type: "synthetic_moe".into(),
        head_dim: HIDDEN_SIZE,
        num_q_heads: 1,
        num_kv_heads: 1,
        rope_base: ROPE_BASE,
        sliding_window: None,
        moe: Some(MoeConfig {
            num_experts: NUM_EXPERTS,
            top_k: ROUTER_TOP_K,
            shared_expert: false,
            router_type: "top_k_softmax".into(),
            moe_intermediate_size: Some(INTERMEDIATE_SIZE),
            hybrid: false,
        }),
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

fn write_router_weights(dir: &std::path::Path, values: &[f32]) {
    std::fs::write(dir.join(ROUTER_WEIGHTS_BIN), f32_bytes(values)).unwrap();
}

#[test]
fn router_load_requires_moe_config_and_complete_file() {
    let dir = tempdir().unwrap();
    assert!(RouterIndex::load(dir.path(), &test_config(Some(moe_model_config()))).is_none());

    write_router_weights(dir.path(), &[1.0, 2.0, 3.0]);
    assert!(RouterIndex::load(dir.path(), &test_config(None)).is_none());
    assert!(RouterIndex::load(dir.path(), &test_config(Some(moe_model_config()))).is_none());
}

#[test]
fn router_loads_and_routes_top_k_experts() {
    let dir = tempdir().unwrap();
    let per_layer = NUM_EXPERTS * HIDDEN_SIZE + NUM_EXPERTS;
    let mut values = vec![0.0; NUM_LAYERS * per_layer];

    values[0] = 1.0;
    values[HIDDEN_SIZE + 1] = 2.0;
    values[2 * HIDDEN_SIZE + 2] = 3.0;
    values[NUM_EXPERTS * HIDDEN_SIZE] = 0.25;
    values[NUM_EXPERTS * HIDDEN_SIZE + 1] = -0.25;

    let second_layer = per_layer;
    values[second_layer] = -1.0;
    values[second_layer + HIDDEN_SIZE + 1] = 4.0;
    values[second_layer + 2 * HIDDEN_SIZE + 2] = 1.0;

    write_router_weights(dir.path(), &values);
    let router = RouterIndex::load(dir.path(), &test_config(Some(moe_model_config()))).unwrap();

    assert_eq!(router.weights.len(), NUM_LAYERS);
    assert_eq!(router.biases.len(), NUM_LAYERS);

    let routed = router.route(0, &array![0.0, 1.0, 1.0]).unwrap();
    assert_eq!(routed.experts, vec![2, 1]);
    assert!(routed.scores[0] > routed.scores[1]);
    assert!((routed.probs.iter().sum::<f32>() - 1.0).abs() < f32::EPSILON);
    assert!(router.route(NUM_LAYERS, &array![1.0, 0.0, 0.0]).is_none());
}

#[test]
fn router_all_layers_counts_and_probability_average() {
    let router = RouterIndex {
        weights: vec![
            array![[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            array![[0.0, 0.0, 2.0], [0.0, 3.0, 0.0], [1.0, 0.0, 0.0]],
        ],
        biases: vec![array![0.0, 0.0, 0.0], array![0.0, 0.0, 0.0]],
        num_experts: NUM_EXPERTS,
        top_k: ROUTER_TOP_K,
    };

    let summary = router.route_all_layers(&array![0.0, 1.0, 1.0], 0..=1);
    assert_eq!(summary[0].0, 1);
    assert_eq!(summary[0].1, NUM_LAYERS);
    assert!((summary[0].2 - 0.5).abs() < f32::EPSILON);
}

fn heap_gate_index() -> VectorIndex {
    VectorIndex::new(
        vec![Some(array![[1.0, 0.0, 2.0], [0.0, -1.0, 1.0]])],
        vec![None],
        1,
        HIDDEN_SIZE,
    )
}

#[test]
fn gate_scores_batch_handles_empty_missing_and_heap_fallback() {
    let idx = heap_gate_index();
    assert!(idx
        .gate_scores_batch(0, &Array2::zeros((0, HIDDEN_SIZE)))
        .is_none());
    assert!(idx
        .gate_scores_batch(NUM_LAYERS, &Array2::zeros((1, HIDDEN_SIZE)))
        .is_none());

    let scores = idx
        .gate_scores_batch(0, &array![[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]])
        .unwrap();
    assert_eq!(scores.shape(), &[GATE_SEQ_LEN, GATE_FEATURES]);
    assert_eq!(scores[[0, 0]], 7.0);
    assert_eq!(scores[[0, 1]], 1.0);
    assert_eq!(scores[[1, 0]], 0.0);
    assert_eq!(scores[[1, 1]], -1.0);
}

#[test]
fn gate_scores_batch_uses_f32_mmap_fast_path() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(TEST_GATE_BIN);
    let gate_values = [1.0, 0.0, 2.0, 0.0, -1.0, 1.0];
    {
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&f32_bytes(&gate_values)).unwrap();
        file.flush().unwrap();
    }
    let file = std::fs::File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let idx = VectorIndex::new_mmap(
        mmap,
        vec![larql_vindex::index::types::GateLayerSlice {
            float_offset: 0,
            num_features: GATE_FEATURES,
        }],
        StorageDtype::F32,
        None,
        1,
        HIDDEN_SIZE,
    );

    let scores = idx.gate_scores_batch(0, &array![[1.0, 2.0, 3.0]]).unwrap();
    assert_eq!(scores.shape(), &[1, GATE_FEATURES]);
    assert_eq!(scores[[0, 0]], 7.0);
    assert_eq!(scores[[0, 1]], 1.0);
}

#[test]
fn gate_scores_batch_uses_f16_mmap_decode_cache() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(TEST_GATE_F16_BIN);
    let gate_values = [1.0, 0.0, 2.0, 0.0, -1.0, 1.0];
    std::fs::write(&path, larql_models::quant::half::encode_f16(&gate_values)).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let idx = VectorIndex::new_mmap(
        mmap,
        vec![larql_vindex::index::types::GateLayerSlice {
            float_offset: 0,
            num_features: GATE_FEATURES,
        }],
        StorageDtype::F16,
        None,
        1,
        HIDDEN_SIZE,
    );

    let scores = idx.gate_scores_batch(0, &array![[1.0, 2.0, 3.0]]).unwrap();
    assert_eq!(scores[[0, 0]], 7.0);
    assert_eq!(scores[[0, 1]], 1.0);
    assert!(idx.gate.f16_decode_cache.lock().unwrap()[0].is_some());
}

#[test]
fn gate_scores_batch_backend_uses_single_row_backend_fast_paths() {
    let mut warmed = heap_gate_index();
    // Test poke: the gate KNN warmed-cache fast path needs both the
    // warmed cache to be populated AND the storage to advertise a
    // matching layer slice. After step 6 the slice lives on
    // `MmapStorage`, so we install it via the setter with a
    // throwaway zero-byte mmap (only the slice meta is consulted on
    // the warmed-cache path, the bytes are never read).
    let throwaway = memmap2::MmapOptions::new()
        .len(GATE_FEATURES * 3 * 4)
        .map_anon()
        .unwrap()
        .make_read_only()
        .unwrap();
    std::sync::Arc::make_mut(&mut warmed.storage).set_gate_vectors(
        std::sync::Arc::new(throwaway),
        larql_vindex::StorageDtype::F32,
        vec![larql_vindex::index::types::GateLayerSlice {
            float_offset: 0,
            num_features: GATE_FEATURES,
        }],
    );
    warmed.gate.warmed_gates.write().unwrap()[0] = Some(vec![1.0, 0.0, 2.0, 0.0, -1.0, 1.0]);

    let backend = FakeGemvBackend;
    let warmed_scores = warmed
        .gate_scores_batch_backend(0, &array![[1.0, 2.0, 3.0]], Some(&backend))
        .unwrap();
    assert_eq!(warmed_scores[[0, 0]], 7.0);
    assert_eq!(warmed_scores[[0, 1]], 1.0);

    let dir = tempdir().unwrap();
    let path = dir.path().join(TEST_GATE_F16_BACKEND_BIN);
    let gate_values = [1.0, 0.0, 2.0, 0.0, -1.0, 1.0];
    std::fs::write(&path, larql_models::quant::half::encode_f16(&gate_values)).unwrap();
    let file = std::fs::File::open(&path).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let f16_idx = VectorIndex::new_mmap(
        mmap,
        vec![larql_vindex::index::types::GateLayerSlice {
            float_offset: 0,
            num_features: GATE_FEATURES,
        }],
        StorageDtype::F16,
        None,
        1,
        HIDDEN_SIZE,
    );

    let f16_scores = f16_idx
        .gate_scores_batch_backend(0, &array![[1.0, 2.0, 3.0]], Some(&backend))
        .unwrap();
    assert_eq!(f16_scores[[0, 0]], 7.0);
    assert_eq!(f16_scores[[0, 1]], 1.0);
}

#[test]
fn layer_weight_writer_round_trips_header_offsets_and_data() {
    let dir = tempdir().unwrap();
    let entries = vec![
        LayerEntry {
            gate_up: vec![1, 2, 3],
            down: vec![4, 5],
        },
        LayerEntry {
            gate_up: vec![6],
            down: vec![7, 8, 9, 10],
        },
    ];

    write_layer_weights(
        dir.path(),
        LAYER_INDEX,
        LayerWeightFormat::F32,
        &entries,
        LAYER_INTERMEDIATE,
        LAYER_HIDDEN,
    )
    .unwrap();

    let bytes = std::fs::read(dir.path().join(layer_weights_filename(LAYER_INDEX))).unwrap();
    let (format, num_entries, inter, hidden, offsets) = parse_layer_weights_header(&bytes).unwrap();

    assert_eq!(format, LayerWeightFormat::F32);
    assert_eq!(num_entries, entries.len());
    assert_eq!(inter, LAYER_INTERMEDIATE);
    assert_eq!(hidden, LAYER_HIDDEN);
    let expected_first_gate_offset = LAYER_HEADER_BYTES + entries.len() * LAYER_OFFSET_BYTES;
    let expected_first_down_offset = expected_first_gate_offset + entries[0].gate_up.len();
    assert_eq!(
        offsets[0],
        (
            expected_first_gate_offset,
            entries[0].gate_up.len(),
            expected_first_down_offset,
            entries[0].down.len()
        )
    );
    assert_eq!(&bytes[offsets[1].0..offsets[1].0 + offsets[1].1], &[6]);
    assert_eq!(
        &bytes[offsets[1].2..offsets[1].2 + offsets[1].3],
        &[7, 8, 9, 10]
    );
}

#[test]
fn layer_weight_parser_rejects_malformed_headers() {
    assert!(parse_layer_weights_header(&[]).is_none());

    let mut bad_magic = vec![0u8; LAYER_HEADER_BYTES];
    assert!(parse_layer_weights_header(&bad_magic).is_none());

    bad_magic[0..U32_BYTES].copy_from_slice(&LAYER_MAGIC.to_le_bytes());
    bad_magic[2 * U32_BYTES..3 * U32_BYTES].copy_from_slice(&UNSUPPORTED_VERSION.to_le_bytes());
    assert!(parse_layer_weights_header(&bad_magic).is_none());

    let mut short_table = bad_magic;
    short_table[2 * U32_BYTES..3 * U32_BYTES]
        .copy_from_slice(&(LayerWeightFormat::F32 as u32).to_le_bytes());
    short_table[3 * U32_BYTES..4 * U32_BYTES].copy_from_slice(&1u32.to_le_bytes());
    assert!(parse_layer_weights_header(&short_table).is_none());
}

#[test]
fn layer_weight_quant_helpers_cover_dense_and_moe_shapes() {
    let bf16_one = 0x3F80u16.to_le_bytes();
    let bf16_two = 0x4000u16.to_le_bytes();
    assert_eq!(
        bf16_bytes_to_f32(&[bf16_one[0], bf16_one[1], bf16_two[0], bf16_two[1]]),
        vec![1.0, 2.0]
    );

    let (padded, padded_cols) = pad_cols_to_256(&[1.0, 2.0, 3.0, 4.0], 2, 2);
    assert_eq!(padded_cols, larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS);
    assert_eq!(padded[0], 1.0);
    assert_eq!(padded[1], 2.0);
    assert_eq!(padded[padded_cols], 3.0);

    let dense = quantize_dense_entry(
        &[1.0, 2.0, 3.0, 4.0],
        &[5.0, 6.0, 7.0, 8.0],
        &[9.0, 10.0, 11.0, 12.0],
        2,
        2,
        LayerWeightFormat::F32,
    )
    .unwrap();
    assert_eq!(dense.gate_up.len(), 8 * F32_BYTES);
    assert_eq!(
        dense.down.len(),
        2 * larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS * F32_BYTES
    );

    let gate_up_bf16 = repeat_bf16(bf16_one, NUM_EXPERTS * 2 * LAYER_HIDDEN);
    let down_bf16 = repeat_bf16(bf16_two, NUM_EXPERTS * LAYER_HIDDEN);
    let moe = quantize_moe_entries(
        &gate_up_bf16,
        &down_bf16,
        NUM_EXPERTS,
        1,
        LAYER_HIDDEN,
        LayerWeightFormat::F32,
    )
    .unwrap();
    assert_eq!(moe.len(), NUM_EXPERTS);
    assert_eq!(moe[0].gate_up.len(), 2 * LAYER_HIDDEN * F32_BYTES);
}

#[test]
fn layer_weight_quant_helpers_reject_unimplemented_formats() {
    let err = match quantize_dense_entry(
        &[1.0, 2.0],
        &[3.0, 4.0],
        &[5.0, 6.0],
        1,
        2,
        LayerWeightFormat::FP4,
    ) {
        Ok(_) => panic!("FP4 per-layer quantization should be rejected until implemented"),
        Err(err) => err.to_string(),
    };
    assert!(err.contains("FP4"), "{err}");
    assert!(err.contains("does not implement"), "{err}");
}

#[test]
fn down_meta_binary_read_write_and_mmap_round_trip() {
    let dir = tempdir().unwrap();
    let meta = vec![
        Some(vec![
            Some(feature_meta(TOKEN_ALPHA_ID, TOKEN_ALPHA_SCORE)),
            None,
        ]),
        None,
        Some(vec![Some(feature_meta(TOKEN_BETA_ID, TOKEN_BETA_SCORE))]),
    ];

    let count = down_meta::write_binary(dir.path(), &meta, DOWN_META_TOP_K).unwrap();
    assert_eq!(count, 2);
    assert!(down_meta::has_binary(dir.path()));

    let (loaded, loaded_count) = down_meta::read_binary(dir.path(), &tokenizer()).unwrap();
    assert_eq!(loaded_count, 2);
    assert_eq!(
        loaded[0].as_ref().unwrap()[0]
            .as_ref()
            .unwrap()
            .top_token_id,
        TOKEN_ALPHA_ID
    );
    assert!(loaded[0].as_ref().unwrap()[1].is_none());
    assert!(loaded[1].is_none());

    let mmap = down_meta::mmap_binary(dir.path(), std::sync::Arc::new(tokenizer())).unwrap();
    assert_eq!(
        mmap.feature_meta(0, 0).unwrap().top_token_id,
        TOKEN_ALPHA_ID
    );
    assert!(mmap.feature_meta(0, 1).is_none());
    assert_eq!(mmap.feature_meta(2, 0).unwrap().top_token_id, TOKEN_BETA_ID);
    assert!(mmap.feature_meta(3, 0).is_none());
}

#[test]
fn down_meta_binary_rejects_bad_headers() {
    let dir = tempdir().unwrap();
    let path = dir.path().join(DOWN_META_BIN);

    std::fs::write(&path, [0u8; 8]).unwrap();
    let too_small = error_string(down_meta::mmap_binary(
        dir.path(),
        std::sync::Arc::new(tokenizer()),
    ));
    assert!(too_small.contains("too small"));

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&DOWN_META_VERSION.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&DOWN_META_TOP_K.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    assert!(
        error_string(down_meta::read_binary(dir.path(), &tokenizer()))
            .contains("invalid down_meta.bin magic")
    );

    bytes[0..U32_BYTES].copy_from_slice(&DOWN_META_MAGIC.to_le_bytes());
    bytes[U32_BYTES..2 * U32_BYTES].copy_from_slice(&UNSUPPORTED_VERSION.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    assert!(error_string(down_meta::mmap_binary(
        dir.path(),
        std::sync::Arc::new(tokenizer())
    ))
    .contains("unsupported down_meta.bin version"));
}
