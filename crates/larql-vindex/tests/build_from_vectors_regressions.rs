//! Regression coverage for building vindexes from pre-extracted NDJSON vectors.

#![cfg(not(target_arch = "wasm32"))]
use std::io::Write;
use std::path::Path;

use larql_vindex::extract::stage_labels::{
    STAGE_DOWN_META, STAGE_EMBEDDINGS, STAGE_GATE_VECTORS, STAGE_TOKENIZER,
};
use larql_vindex::format::filenames::{
    DOWN_META_JSONL, EMBEDDINGS_BIN, GATE_VECTORS_BIN, INDEX_JSON, TOKENIZER_JSON,
};
use larql_vindex::{build_vindex_from_vectors, IndexBuildCallbacks, VindexConfig};
use tempfile::tempdir;

const HIDDEN_SIZE: usize = 2;
const F32_BYTES: usize = std::mem::size_of::<f32>();
const GEMMA_MODEL: &str = "gemma-test";
const LLAMA_MODEL: &str = "llama-test";
const AMBIGUOUS_MODEL: &str = "gemma-llama-brand-name";
const MINIMAL_TOKENIZER_JSON: &str =
    r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;

#[derive(Default)]
struct RecordingCallbacks {
    stages: Vec<String>,
    done: Vec<String>,
}

impl IndexBuildCallbacks for RecordingCallbacks {
    fn on_stage(&mut self, stage: &str) {
        self.stages.push(stage.to_string());
    }

    fn on_stage_done(&mut self, stage: &str, _elapsed_ms: f64) {
        self.done.push(stage.to_string());
    }
}

fn write_lines(path: &Path, lines: &[String]) {
    let mut file = std::fs::File::create(path).unwrap();
    for line in lines {
        writeln!(file, "{line}").unwrap();
    }
}

fn vector_line(layer: usize, feature: usize, values: &[f32]) -> String {
    serde_json::json!({
        "layer": layer,
        "feature": feature,
        "vector": values,
    })
    .to_string()
}

fn embedding_line(feature: usize, values: &[f32]) -> String {
    serde_json::json!({
        "feature": feature,
        "vector": values,
    })
    .to_string()
}

fn down_line(layer: usize, feature: usize, with_top_k: bool) -> String {
    let mut value = serde_json::json!({
        "layer": layer,
        "feature": feature,
        "top_token": "Paris",
        "top_token_id": 123,
        "c_score": 0.75,
    });
    if with_top_k {
        value["top_k"] = serde_json::json!([
            {"token": "Paris", "token_id": 123, "logit": 0.75},
            {"token": "France", "token_id": 456, "logit": 0.5}
        ]);
    }
    value.to_string()
}

fn make_vectors_dir(root: &Path, model: &str) -> std::path::PathBuf {
    make_vectors_dir_with_header(root, model, serde_json::json!({}))
}

fn make_vectors_dir_with_header(
    root: &Path,
    model: &str,
    header_fields: serde_json::Value,
) -> std::path::PathBuf {
    let vectors = root.join("vectors");
    std::fs::create_dir_all(&vectors).unwrap();
    let mut header = serde_json::json!({
        "_header": true,
        "model": model,
        "dimension": HIDDEN_SIZE,
    });
    if let Some(extra) = header_fields.as_object() {
        let header_obj = header.as_object_mut().unwrap();
        for (key, value) in extra {
            header_obj.insert(key.clone(), value.clone());
        }
    }
    let header = header.to_string();
    write_lines(
        &vectors.join("ffn_gate.vectors.jsonl"),
        &[
            header.clone(),
            vector_line(1, 0, &[3.0, 4.0]),
            vector_line(0, 1, &[1.0, 2.0]),
        ],
    );
    write_lines(
        &vectors.join("embeddings.vectors.jsonl"),
        &[
            header,
            embedding_line(2, &[0.2, 0.3]),
            embedding_line(0, &[0.0, 0.1]),
        ],
    );
    write_lines(
        &vectors.join("ffn_down.vectors.jsonl"),
        &[down_line(0, 1, true), down_line(1, 0, false)],
    );
    vectors
}

fn read_f32_file(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap()
        .chunks_exact(F32_BYTES)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

#[test]
fn build_from_vectors_writes_sorted_binary_layout_and_config() {
    let root = tempdir().unwrap();
    std::fs::write(root.path().join(TOKENIZER_JSON), MINIMAL_TOKENIZER_JSON).unwrap();
    let vectors = make_vectors_dir_with_header(
        root.path(),
        GEMMA_MODEL,
        serde_json::json!({
            "model_config": {
                "model_type": "gemma3",
                "hidden_size": HIDDEN_SIZE,
                "num_hidden_layers": 2,
                "intermediate_size": 2,
                "head_dim": HIDDEN_SIZE,
                "num_attention_heads": 1,
                "num_key_value_heads": 1
            }
        }),
    );
    let output = root.path().join("out");
    let mut callbacks = RecordingCallbacks::default();

    build_vindex_from_vectors(&vectors, &output, &mut callbacks).unwrap();

    assert_eq!(
        callbacks.stages,
        vec![
            STAGE_GATE_VECTORS,
            STAGE_EMBEDDINGS,
            STAGE_DOWN_META,
            STAGE_TOKENIZER
        ]
    );
    assert_eq!(callbacks.stages, callbacks.done);

    let config: VindexConfig =
        serde_json::from_slice(&std::fs::read(output.join(INDEX_JSON)).unwrap()).unwrap();
    assert_eq!(config.model, GEMMA_MODEL);
    assert_eq!(config.family, "gemma3");
    assert_eq!(config.model_config.as_ref().unwrap().model_type, "gemma3");
    assert_eq!(config.num_layers, 2);
    assert_eq!(config.hidden_size, HIDDEN_SIZE);
    assert_eq!(config.intermediate_size, 2);
    assert_eq!(config.vocab_size, 3);
    assert_eq!(config.down_top_k, 2);
    assert_eq!(config.embed_scale, (HIDDEN_SIZE as f32).sqrt());
    assert_eq!(config.layers[0].num_features, 2);
    assert_eq!(config.layers[1].num_features, 1);

    assert_eq!(
        read_f32_file(&output.join(GATE_VECTORS_BIN)),
        vec![0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        read_f32_file(&output.join(EMBEDDINGS_BIN)),
        vec![0.0, 0.1, 0.0, 0.0, 0.2, 0.3]
    );
    assert!(output.join(TOKENIZER_JSON).exists());

    let down_jsonl = std::fs::read_to_string(output.join(DOWN_META_JSONL)).unwrap();
    let records: Vec<serde_json::Value> = down_jsonl
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0]["t"], "Paris");
    assert_eq!(records[0]["i"], 123);
    assert_eq!(records[0]["k"].as_array().unwrap().len(), 2);
    assert!(records[1]["k"].as_array().unwrap().is_empty());
}

#[test]
fn build_from_vectors_handles_absent_tokenizer_and_llama_family() {
    let root = tempdir().unwrap();
    let vectors = make_vectors_dir_with_header(
        root.path(),
        LLAMA_MODEL,
        serde_json::json!({
            "family": "llama",
            "embed_scale": 1.0
        }),
    );
    let output = root.path().join("out");
    let mut callbacks = RecordingCallbacks::default();

    build_vindex_from_vectors(&vectors, &output, &mut callbacks).unwrap();

    let config: VindexConfig =
        serde_json::from_slice(&std::fs::read(output.join(INDEX_JSON)).unwrap()).unwrap();
    assert_eq!(config.family, "llama");
    assert_eq!(config.embed_scale, 1.0);
    assert!(!output.join(TOKENIZER_JSON).exists());
    assert!(!callbacks
        .stages
        .iter()
        .any(|stage| stage == STAGE_TOKENIZER));
}

#[test]
fn build_from_vectors_does_not_infer_family_from_model_name() {
    let root = tempdir().unwrap();
    let vectors = make_vectors_dir(root.path(), AMBIGUOUS_MODEL);
    let output = root.path().join("out");
    let mut callbacks = RecordingCallbacks::default();

    build_vindex_from_vectors(&vectors, &output, &mut callbacks).unwrap();

    let config: VindexConfig =
        serde_json::from_slice(&std::fs::read(output.join(INDEX_JSON)).unwrap()).unwrap();
    assert_eq!(config.model, AMBIGUOUS_MODEL);
    assert_eq!(config.family, "unknown");
    assert_eq!(config.embed_scale, 1.0);
    assert!(config.model_config.is_none());
    assert!(config.layer_bands.is_none());
}

#[test]
fn build_from_vectors_reports_missing_or_empty_gate_file() {
    let root = tempdir().unwrap();
    let vectors = root.path().join("vectors");
    let output = root.path().join("out");
    std::fs::create_dir_all(&vectors).unwrap();

    let mut callbacks = RecordingCallbacks::default();
    let missing = build_vindex_from_vectors(&vectors, &output, &mut callbacks)
        .unwrap_err()
        .to_string();
    assert!(missing.contains("ffn_gate.vectors.jsonl not found"));

    std::fs::write(vectors.join("ffn_gate.vectors.jsonl"), "").unwrap();
    let empty = build_vindex_from_vectors(&vectors, &output, &mut callbacks)
        .unwrap_err()
        .to_string();
    assert!(empty.contains("empty gate file"));
}

fn write_gate_file(root: &Path, gate_lines: &[String]) -> std::path::PathBuf {
    let vectors = root.join("vectors");
    std::fs::create_dir_all(&vectors).unwrap();
    let header = serde_json::json!({
        "_header": true,
        "model": GEMMA_MODEL,
        "dimension": HIDDEN_SIZE,
    })
    .to_string();
    let mut all = vec![header];
    all.extend_from_slice(gate_lines);
    write_lines(&vectors.join("ffn_gate.vectors.jsonl"), &all);
    // build_vindex_from_vectors only opens embeddings + down after gate parses,
    // so empty stubs are sufficient for these failure-path tests.
    std::fs::write(vectors.join("embeddings.vectors.jsonl"), "").unwrap();
    std::fs::write(vectors.join("ffn_down.vectors.jsonl"), "").unwrap();
    vectors
}

#[test]
fn build_from_vectors_rejects_gate_record_missing_layer() {
    let root = tempdir().unwrap();
    let vectors = write_gate_file(
        root.path(),
        &[serde_json::json!({"feature": 0, "vector": [1.0, 2.0]}).to_string()],
    );
    let mut callbacks = RecordingCallbacks::default();
    let err = build_vindex_from_vectors(&vectors, &root.path().join("out"), &mut callbacks)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("missing or non-integer 'layer' field"),
        "{err}"
    );
}

#[test]
fn build_from_vectors_rejects_gate_record_with_non_array_vector() {
    let root = tempdir().unwrap();
    let vectors = write_gate_file(
        root.path(),
        &[serde_json::json!({"layer": 0, "feature": 0, "vector": "oops"}).to_string()],
    );
    let mut callbacks = RecordingCallbacks::default();
    let err = build_vindex_from_vectors(&vectors, &root.path().join("out"), &mut callbacks)
        .unwrap_err()
        .to_string();
    assert!(err.contains("missing or non-array 'vector' field"), "{err}");
}

#[test]
fn build_from_vectors_rejects_gate_record_with_non_float_vector_element() {
    let root = tempdir().unwrap();
    let vectors = write_gate_file(
        root.path(),
        &[serde_json::json!({"layer": 0, "feature": 0, "vector": [1.0, "nope"]}).to_string()],
    );
    let mut callbacks = RecordingCallbacks::default();
    let err = build_vindex_from_vectors(&vectors, &root.path().join("out"), &mut callbacks)
        .unwrap_err()
        .to_string();
    assert!(err.contains("non-float element in 'vector' array"), "{err}");
}
