//! Architecture capability contract tests for weight writers.

use larql_vindex::format::filenames::{
    ATTN_WEIGHTS_BIN, ATTN_WEIGHTS_KQUANT_BIN, INDEX_JSON, INTERLEAVED_KQUANT_BIN,
    LEGACY_ATTN_WEIGHTS_Q4K_BIN, LEGACY_INTERLEAVED_Q4K_BIN, WEIGHT_MANIFEST_JSON,
};
use larql_vindex::format::weights::{
    write_model_weights, write_model_weights_kquant, WeightSource,
};
use larql_vindex::IndexBuildCallbacks;
use tempfile::tempdir;

const HIDDEN_SIZE: usize = 4096;
const INTERMEDIATE_SIZE: usize = 12288;
const NUM_LAYERS: usize = 4;
const NUM_HEADS: usize = 32;
const NUM_KV_HEADS: usize = 32;
const HEAD_DIM: usize = 128;
const KV_LORA_RANK: usize = 512;
const Q_LORA_RANK: usize = 1536;

#[derive(Default)]
struct SilentCallbacks;

impl IndexBuildCallbacks for SilentCallbacks {}

struct EmptyWeightSource {
    arch: Box<dyn larql_models::ModelArchitecture>,
}

impl EmptyWeightSource {
    fn deepseek_mla() -> Self {
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "deepseek_v2",
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": NUM_HEADS,
            "num_key_value_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "kv_lora_rank": KV_LORA_RANK,
            "q_lora_rank": Q_LORA_RANK
        }));
        Self { arch }
    }
}

impl WeightSource for EmptyWeightSource {
    fn get_tensor(&self, _key: &str) -> Option<(Vec<f32>, usize, usize)> {
        None
    }

    fn get_vector(&self, _key: &str) -> Option<Vec<f32>> {
        None
    }

    fn arch(&self) -> &dyn larql_models::ModelArchitecture {
        &*self.arch
    }

    fn num_layers(&self) -> usize {
        self.arch.config().num_layers
    }

    fn lm_head(&self) -> Option<(Vec<f32>, usize, usize)> {
        None
    }

    fn vector_names(&self) -> Vec<String> {
        Vec::new()
    }

    fn get_packed_bf16(&self, _key: &str) -> Option<Vec<u8>> {
        None
    }
}

fn write_minimal_index_json(dir: &std::path::Path) {
    std::fs::write(
        dir.join(INDEX_JSON),
        serde_json::json!({
            "version": 1,
            "model": "deepseek-mla",
            "family": "deepseek",
            "num_layers": NUM_LAYERS,
            "hidden_size": HIDDEN_SIZE,
            "intermediate_size": INTERMEDIATE_SIZE,
            "vocab_size": 0,
            "embed_scale": 1.0,
            "layers": [],
            "down_top_k": 0
        })
        .to_string(),
    )
    .unwrap();
}

#[test]
fn standard_weight_writers_reject_mla_before_emitting_weight_files() {
    let dir = tempdir().unwrap();
    write_minimal_index_json(dir.path());

    let source = EmptyWeightSource::deepseek_mla();
    let mut callbacks = SilentCallbacks;

    let q4k_err = write_model_weights_kquant(&source, dir.path(), &mut callbacks)
        .expect_err("k-quant writer must reject MLA layouts before writing files")
        .to_string();
    assert!(q4k_err.contains("MLA"), "{q4k_err}");
    // Neither the new kquant-named files nor the legacy q4k-named files
    // may exist when the writer rejected the architecture up-front.
    assert!(!dir.path().join(ATTN_WEIGHTS_KQUANT_BIN).exists());
    assert!(!dir.path().join(LEGACY_ATTN_WEIGHTS_Q4K_BIN).exists());
    assert!(!dir.path().join(INTERLEAVED_KQUANT_BIN).exists());
    assert!(!dir.path().join(LEGACY_INTERLEAVED_Q4K_BIN).exists());
    assert!(!dir.path().join(WEIGHT_MANIFEST_JSON).exists());

    let f32_err = write_model_weights(&source, dir.path(), &mut callbacks)
        .expect_err("f32 writer must reject MLA layouts before writing files")
        .to_string();
    assert!(f32_err.contains("MLA"), "{f32_err}");
    assert!(!dir.path().join(ATTN_WEIGHTS_BIN).exists());
    assert!(!dir.path().join(WEIGHT_MANIFEST_JSON).exists());
}
