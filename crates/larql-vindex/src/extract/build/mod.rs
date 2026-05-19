//! Build a .vindex from model weights — the extraction/clustering pipeline.
//!
//! Single entry point: `build_vindex` (full pipeline from weights). For
//! mid-run resume, see the streaming pipeline's checkpoint mechanism in
//! `extract::streaming` — the older `build_vindex_resume` path was
//! removed 2026-05-09 (it read the legacy `down_meta.jsonl` format that
//! nothing produces any more).
//!
//! `build_vindex` is structured around a `BuildContext` that holds the
//! shared inputs + accumulator state across the stages:
//!   1. `write_gate_vectors`            — gate matrices per layer (handles MoE)
//!   2. `write_embeddings`              — embedding table
//!   3. `write_down_meta_and_clusters`  — per-feature top-k tokens + collect
//!                                        offset directions for clustering
//!   4. `run_clustering`                — k-means + label clusters
//!   5. `write_tokenizer`
//!   6. `write_index_json`              — config + provenance + checksums
//!
//! Stage 3 lives in [`down_meta`], stage 6 in [`index_json`].
//! Discrete helpers live in `super::build_helpers`.

mod down_meta;
mod index_json;

use crate::extract::stage_labels::*;
use std::io::BufWriter;
use std::path::Path;

use larql_models::{FfnType, ModelWeights};

use crate::config::dtype::{write_floats, StorageDtype};
use crate::config::VindexLayerInfo;
use crate::error::VindexError;
use crate::format::filenames::*;

use super::build_helpers::{run_clustering_pipeline, ClusterData};

pub use crate::extract::callbacks::IndexBuildCallbacks;

pub(super) fn knowledge_layer_range(family: &str, num_layers: usize) -> Option<(usize, usize)> {
    crate::LayerBands::for_family(family, num_layers).map(|bands| {
        let start = bands.knowledge.0.min(num_layers);
        let end = bands.knowledge.1.saturating_add(1).min(num_layers);
        (start, end)
    })
}

// ═══════════════════════════════════════════════════════════════════════
// BuildContext — shared state across pipeline stages
// ═══════════════════════════════════════════════════════════════════════

/// Holds the inputs + accumulators for the build pipeline. Each stage
/// method on `BuildContext` reads inputs and mutates the accumulators
/// (`layer_infos`, `cluster_*`); the derived constants are set in `new`.
pub(super) struct BuildContext<'a> {
    // Inputs
    pub(super) weights: &'a ModelWeights,
    pub(super) tokenizer: &'a tokenizers::Tokenizer,
    pub(super) output_dir: &'a Path,
    pub(super) callbacks: &'a mut dyn IndexBuildCallbacks,
    pub(super) dtype: StorageDtype,
    pub(super) down_top_k: usize,

    // Derived constants
    pub(super) num_layers: usize,
    pub(super) hidden_size: usize,
    pub(super) intermediate_size: usize,
    pub(super) vocab_size: usize,
    pub(super) embed_scale: f32,
    pub(super) is_moe: bool,
    pub(super) n_experts: usize,

    // Stage 1 → Stage 6 (consumed by `write_index_json`)
    pub(super) layer_infos: Vec<VindexLayerInfo>,

    // Stage 3 collects → Stage 4 drains (`run_clustering`).
    pub(super) cluster_directions: Vec<f32>,
    pub(super) cluster_features: Vec<(usize, usize)>,
    pub(super) cluster_top_tokens: Vec<String>,
    pub(super) cluster_input_tokens: Vec<String>,
    pub(super) cluster_output_tokens: Vec<String>,
}

impl<'a> BuildContext<'a> {
    fn new(
        weights: &'a ModelWeights,
        tokenizer: &'a tokenizers::Tokenizer,
        output_dir: &'a Path,
        callbacks: &'a mut dyn IndexBuildCallbacks,
        dtype: StorageDtype,
        down_top_k: usize,
    ) -> Self {
        Self {
            num_layers: weights.num_layers,
            hidden_size: weights.hidden_size,
            intermediate_size: weights.intermediate_size,
            vocab_size: weights.vocab_size,
            embed_scale: weights.arch.embed_scale(),
            is_moe: weights.arch.is_moe(),
            n_experts: weights.arch.num_experts(),
            weights,
            tokenizer,
            output_dir,
            callbacks,
            dtype,
            down_top_k,
            layer_infos: Vec::new(),
            cluster_directions: Vec::new(),
            cluster_features: Vec::new(),
            cluster_top_tokens: Vec::new(),
            cluster_input_tokens: Vec::new(),
            cluster_output_tokens: Vec::new(),
        }
    }

    /// Stage 1 — write `gate_vectors.bin` (one matrix per layer; MoE
    /// concatenates each expert's matrix). Populates `layer_infos`.
    fn write_gate_vectors(&mut self) -> Result<(), VindexError> {
        self.callbacks.on_stage(STAGE_GATE_VECTORS);
        let gate_path = self.output_dir.join(GATE_VECTORS_BIN);
        let mut gate_file = BufWriter::new(std::fs::File::create(&gate_path)?);
        let mut offset: u64 = 0;

        for layer in 0..self.num_layers {
            self.callbacks
                .on_layer_start(COMP_GATE, layer, self.num_layers);
            let start = std::time::Instant::now();

            if self.is_moe && self.n_experts > 0 {
                // MoE: write each expert's gate matrix contiguously
                let mut total_features = 0usize;
                let mut layer_bytes = 0u64;
                let mut features_per_expert = 0usize;

                for expert in 0..self.n_experts {
                    let gate_key = match self.weights.arch.expert_ffn_gate_key(layer, expert) {
                        Some(k) => k,
                        None => continue,
                    };
                    let w_gate = match self.weights.tensors.get(&gate_key) {
                        Some(w) => w,
                        None => continue,
                    };
                    features_per_expert = w_gate.shape()[0];
                    total_features += features_per_expert;
                    let data = w_gate.as_slice().unwrap();
                    layer_bytes += write_floats(&mut gate_file, data, self.dtype)?;
                }

                // Also include shared expert if present
                if let Some(shared_key) = self.weights.arch.shared_expert_gate_key(layer) {
                    if let Some(w_gate) = self.weights.tensors.get(&shared_key) {
                        let n = w_gate.shape()[0];
                        total_features += n;
                        let data = w_gate.as_slice().unwrap();
                        layer_bytes += write_floats(&mut gate_file, data, self.dtype)?;
                    }
                }

                if total_features > 0 {
                    self.layer_infos.push(VindexLayerInfo {
                        layer,
                        num_features: total_features,
                        offset,
                        length: layer_bytes,
                        num_experts: Some(self.n_experts),
                        num_features_per_expert: Some(features_per_expert),
                    });
                    offset += layer_bytes;
                }
            } else {
                // Dense: single feature-input-direction matrix per layer.
                // Gated FFN routes through `ffn_gate`; non-gated FFN (GPT-2,
                // StarCoder2) reuses `ffn_up` for the same role.
                let gate_key = match self.weights.arch.ffn_type() {
                    FfnType::Gated => self.weights.arch.ffn_gate_key(layer),
                    FfnType::Standard => self.weights.arch.ffn_up_key(layer),
                };
                let w_gate = match self.weights.tensors.get(&gate_key) {
                    Some(w) => w,
                    None => continue,
                };
                let num_features = w_gate.shape()[0];
                let data = w_gate.as_slice().unwrap();
                let length = write_floats(&mut gate_file, data, self.dtype)?;
                self.layer_infos.push(VindexLayerInfo {
                    layer,
                    num_features,
                    offset,
                    length,
                    num_experts: None,
                    num_features_per_expert: None,
                });
                offset += length;
            }

            self.callbacks
                .on_layer_done(COMP_GATE, layer, start.elapsed().as_secs_f64() * 1000.0);
        }
        self.callbacks.on_stage_done(STAGE_GATE_VECTORS, 0.0);
        Ok(())
    }

    /// Stage 2 — write `embeddings.bin`.
    fn write_embeddings(&mut self) -> Result<(), VindexError> {
        self.callbacks.on_stage(STAGE_EMBEDDINGS);
        let embed_path = self.output_dir.join(EMBEDDINGS_BIN);
        let embed_data = self.weights.embed.as_slice().unwrap();
        let embed_bytes = crate::config::dtype::encode_floats(embed_data, self.dtype);
        std::fs::write(&embed_path, &embed_bytes)?;
        self.callbacks.on_stage_done(STAGE_EMBEDDINGS, 0.0);
        Ok(())
    }

    /// Stage 4 — k-means + label the collected cluster directions.
    /// Drains the `cluster_*` accumulators.
    fn run_clustering(&mut self) -> Result<(), VindexError> {
        run_clustering_pipeline(
            ClusterData {
                directions: std::mem::take(&mut self.cluster_directions),
                features: std::mem::take(&mut self.cluster_features),
                top_tokens: std::mem::take(&mut self.cluster_top_tokens),
                input_tokens: std::mem::take(&mut self.cluster_input_tokens),
                output_tokens: std::mem::take(&mut self.cluster_output_tokens),
            },
            self.hidden_size,
            self.weights,
            self.tokenizer,
            self.output_dir,
            self.callbacks,
        )
    }

    /// Stage 5 — copy the tokenizer JSON.
    fn write_tokenizer(&mut self) -> Result<(), VindexError> {
        self.callbacks.on_stage(STAGE_TOKENIZER);
        let tokenizer_json = self
            .tokenizer
            .to_string(true)
            .map_err(|e| VindexError::Parse(format!("tokenizer serialize: {e}")))?;
        std::fs::write(self.output_dir.join(TOKENIZER_JSON), tokenizer_json)?;
        self.callbacks.on_stage_done(STAGE_TOKENIZER, 0.0);
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Entry points
// ═══════════════════════════════════════════════════════════════════════

/// Build a .vindex from model weights and write it to disk.
///
/// Reads gate vectors and down projections directly from safetensors,
/// projects down vectors to vocabulary for top-k token metadata,
/// writes everything to a self-contained directory.
#[allow(clippy::too_many_arguments)]
pub fn build_vindex(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    model_name: &str,
    output_dir: &Path,
    down_top_k: usize,
    extract_level: crate::ExtractLevel,
    dtype: StorageDtype,
    callbacks: &mut dyn IndexBuildCallbacks,
) -> Result<(), VindexError> {
    // Refuse the extract before any output is created when the requested
    // tier needs attention tensors that the writer cannot represent
    // (e.g. MLA on the standard Q/K/V/O manifests). A late failure
    // inside the writer leaves a half-written vindex on disk.
    crate::format::weights::ensure_extract_level_supported(&*weights.arch, extract_level)?;

    std::fs::create_dir_all(output_dir)?;
    let mut ctx = BuildContext::new(weights, tokenizer, output_dir, callbacks, dtype, down_top_k);
    ctx.write_gate_vectors()?;
    ctx.write_embeddings()?;
    ctx.write_down_meta_and_clusters()?;
    ctx.run_clustering()?;
    ctx.write_tokenizer()?;
    ctx.write_index_json(model_name, extract_level)?;
    Ok(())
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use ndarray::ArcArray2;
    use std::collections::HashMap;
    use tempfile::TempDir;

    use super::{build_vindex, knowledge_layer_range};
    use crate::{
        ExtractLevel, SilentBuildCallbacks, SilentLoadCallbacks, StorageDtype, VectorIndex,
    };

    // ── synthetic model fixture ──────────────────────────────────────────

    const NUM_LAYERS: usize = 2;
    const HIDDEN: usize = 8;
    const INTERMEDIATE: usize = 4;
    const VOCAB: usize = 16;

    #[test]
    fn knowledge_layer_range_uses_model_band_policy() {
        assert_eq!(knowledge_layer_range("llama", 32), Some((13, 26)));
        assert_eq!(knowledge_layer_range("gemma3", 34), Some((14, 28)));
        assert_eq!(knowledge_layer_range("tiny", 4), None);
    }

    fn make_weights() -> larql_models::ModelWeights {
        let mut tensors: HashMap<String, ArcArray2<f32>> = HashMap::new();
        let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

        for layer in 0..NUM_LAYERS {
            let mut gate = ndarray::Array2::<f32>::zeros((INTERMEDIATE, HIDDEN));
            for i in 0..INTERMEDIATE {
                gate[[i, i % HIDDEN]] = 1.0;
            }
            tensors.insert(
                format!("layers.{layer}.mlp.gate_proj.weight"),
                gate.into_shared(),
            );

            let mut up = ndarray::Array2::<f32>::zeros((INTERMEDIATE, HIDDEN));
            for i in 0..INTERMEDIATE {
                up[[i, (i + 1) % HIDDEN]] = 0.5;
            }
            tensors.insert(
                format!("layers.{layer}.mlp.up_proj.weight"),
                up.into_shared(),
            );

            let mut down = ndarray::Array2::<f32>::zeros((HIDDEN, INTERMEDIATE));
            for i in 0..INTERMEDIATE {
                down[[i % HIDDEN, i]] = 0.3;
            }
            tensors.insert(
                format!("layers.{layer}.mlp.down_proj.weight"),
                down.into_shared(),
            );

            for suffix in &["q_proj", "k_proj", "v_proj", "o_proj"] {
                let mut a = ndarray::Array2::<f32>::zeros((HIDDEN, HIDDEN));
                for i in 0..HIDDEN {
                    a[[i, i]] = 1.0;
                }
                tensors.insert(
                    format!("layers.{layer}.self_attn.{suffix}.weight"),
                    a.into_shared(),
                );
            }
            vectors.insert(
                format!("layers.{layer}.input_layernorm.weight"),
                vec![1.0; HIDDEN],
            );
            vectors.insert(
                format!("layers.{layer}.post_attention_layernorm.weight"),
                vec![1.0; HIDDEN],
            );
        }
        vectors.insert("norm.weight".into(), vec![1.0; HIDDEN]);

        let mut embed = ndarray::Array2::<f32>::zeros((VOCAB, HIDDEN));
        for i in 0..VOCAB {
            embed[[i, i % HIDDEN]] = 1.0;
        }
        let embed = embed.into_shared();
        let lm_head = embed.clone();

        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": HIDDEN,
            "num_hidden_layers": NUM_LAYERS,
            "intermediate_size": INTERMEDIATE,
            "head_dim": HIDDEN,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "rope_theta": 10000.0,
            "vocab_size": VOCAB,
        }));
        larql_models::ModelWeights {
            tensors,
            vectors,
            raw_bytes: HashMap::new(),
            skipped_tensors: Vec::new(),
            packed_mmaps: HashMap::new(),
            packed_byte_ranges: HashMap::new(),
            embed,
            lm_head,
            position_embed: None,
            num_layers: NUM_LAYERS,
            hidden_size: HIDDEN,
            intermediate_size: INTERMEDIATE,
            vocab_size: VOCAB,
            head_dim: HIDDEN,
            num_q_heads: 1,
            num_kv_heads: 1,
            rope_base: 10000.0,
            arch,
        }
    }

    const TOK_JSON: &str =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;

    fn tokenizer() -> tokenizers::Tokenizer {
        tokenizers::Tokenizer::from_bytes(TOK_JSON).unwrap()
    }

    fn run_build(dir: &std::path::Path, level: ExtractLevel, dtype: StorageDtype) {
        let weights = make_weights();
        let tok = tokenizer();
        let mut cb = SilentBuildCallbacks;
        build_vindex(&weights, &tok, "test/unit", dir, 3, level, dtype, &mut cb).unwrap();
    }

    // ── build output file inventory ──────────────────────────────────────

    #[test]
    fn build_browse_writes_required_files() {
        let dir = TempDir::new().unwrap();
        run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
        assert!(
            dir.path().join("gate_vectors.bin").exists(),
            "gate_vectors.bin missing"
        );
        assert!(
            dir.path().join("embeddings.bin").exists(),
            "embeddings.bin missing"
        );
        assert!(
            dir.path().join("down_meta.bin").exists(),
            "down_meta.bin missing"
        );
        assert!(dir.path().join("index.json").exists(), "index.json missing");
        assert!(
            dir.path().join("tokenizer.json").exists(),
            "tokenizer.json missing"
        );
    }

    #[test]
    fn build_browse_does_not_write_weight_files() {
        let dir = TempDir::new().unwrap();
        run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
        // Browse level: no model weights
        assert!(!dir.path().join("attn_weights.bin").exists());
        assert!(!dir.path().join("up_weights.bin").exists());
        assert!(!dir.path().join("down_weights.bin").exists());
    }

    #[test]
    fn build_all_writes_weight_files() {
        let dir = TempDir::new().unwrap();
        run_build(dir.path(), ExtractLevel::All, StorageDtype::F32);
        assert!(
            dir.path().join("attn_weights.bin").exists(),
            "attn_weights.bin missing"
        );
        assert!(
            dir.path().join("up_weights.bin").exists(),
            "up_weights.bin missing"
        );
        assert!(
            dir.path().join("down_weights.bin").exists(),
            "down_weights.bin missing"
        );
    }

    // ── index.json content ───────────────────────────────────────────────

    #[test]
    fn build_index_json_has_correct_shape() {
        let dir = TempDir::new().unwrap();
        run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
        let cfg = crate::format::load::load_vindex_config(dir.path()).unwrap();
        assert_eq!(cfg.num_layers, NUM_LAYERS);
        assert_eq!(cfg.hidden_size, HIDDEN);
        assert_eq!(cfg.intermediate_size, INTERMEDIATE);
        assert_eq!(cfg.vocab_size, VOCAB);
        assert_eq!(cfg.model, "test/unit");
        assert_eq!(cfg.version, 2);
    }

    #[test]
    fn build_browse_has_model_weights_false() {
        let dir = TempDir::new().unwrap();
        run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
        let cfg = crate::format::load::load_vindex_config(dir.path()).unwrap();
        assert!(!cfg.has_model_weights);
    }

    #[test]
    fn build_all_has_model_weights_true() {
        let dir = TempDir::new().unwrap();
        run_build(dir.path(), ExtractLevel::All, StorageDtype::F32);
        let cfg = crate::format::load::load_vindex_config(dir.path()).unwrap();
        assert!(cfg.has_model_weights);
    }

    #[test]
    fn build_records_source_provenance() {
        let dir = TempDir::new().unwrap();
        run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
        let cfg = crate::format::load::load_vindex_config(dir.path()).unwrap();
        let src = cfg.source.unwrap();
        assert_eq!(src.huggingface_repo.as_deref(), Some("test/unit"));
        assert!(!src.larql_version.is_empty());
    }

    #[test]
    fn build_records_checksums() {
        let dir = TempDir::new().unwrap();
        run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
        let cfg = crate::format::load::load_vindex_config(dir.path()).unwrap();
        let checksums = cfg.checksums.unwrap();
        assert!(
            checksums.contains_key("gate_vectors.bin"),
            "gate_vectors.bin not in checksums"
        );
    }

    #[test]
    fn build_layer_infos_match_num_layers() {
        let dir = TempDir::new().unwrap();
        run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
        let cfg = crate::format::load::load_vindex_config(dir.path()).unwrap();
        assert_eq!(cfg.layers.len(), NUM_LAYERS);
        for (i, info) in cfg.layers.iter().enumerate() {
            assert_eq!(info.layer, i, "layer index mismatch at position {i}");
            assert_eq!(
                info.num_features, INTERMEDIATE,
                "wrong feature count at layer {i}"
            );
        }
    }

    // ── gate_vectors.bin content ─────────────────────────────────────────

    #[test]
    fn build_gate_vectors_bin_size_matches_config() {
        let dir = TempDir::new().unwrap();
        run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
        let cfg = crate::format::load::load_vindex_config(dir.path()).unwrap();
        let expected: u64 = cfg.layers.iter().map(|l| l.length).sum();
        let actual = std::fs::metadata(dir.path().join("gate_vectors.bin"))
            .unwrap()
            .len();
        assert_eq!(actual, expected, "gate_vectors.bin size mismatch");
    }

    // ── round-trip: build then load ──────────────────────────────────────

    #[test]
    fn build_then_load_vindex_succeeds() {
        let dir = TempDir::new().unwrap();
        run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
        let mut cb = SilentLoadCallbacks;
        let index = VectorIndex::load_vindex(dir.path(), &mut cb).unwrap();
        assert_eq!(index.num_layers, NUM_LAYERS);
        assert_eq!(index.hidden_size, HIDDEN);
        assert_eq!(index.total_gate_vectors(), NUM_LAYERS * INTERMEDIATE);
    }

    #[test]
    fn build_then_load_gate_knn_returns_results() {
        let dir = TempDir::new().unwrap();
        run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
        let mut cb = SilentLoadCallbacks;
        let index = VectorIndex::load_vindex(dir.path(), &mut cb).unwrap();
        let query = ndarray::Array1::from_vec(vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let hits = index.gate_knn(0, &query, 2);
        assert!(!hits.is_empty(), "gate_knn returned no results after build");
    }

    #[test]
    fn build_f16_dtype_round_trips() {
        let dir = TempDir::new().unwrap();
        run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F16);
        let cfg = crate::format::load::load_vindex_config(dir.path()).unwrap();
        assert_eq!(cfg.dtype, StorageDtype::F16);
        let mut cb = SilentLoadCallbacks;
        let index = VectorIndex::load_vindex(dir.path(), &mut cb).unwrap();
        assert_eq!(index.num_layers, NUM_LAYERS);
    }

    #[test]
    fn build_idempotent_on_existing_dir() {
        let dir = TempDir::new().unwrap();
        // First build
        run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
        // Second build into same directory should overwrite cleanly
        run_build(dir.path(), ExtractLevel::Browse, StorageDtype::F32);
        let cfg = crate::format::load::load_vindex_config(dir.path()).unwrap();
        assert_eq!(cfg.num_layers, NUM_LAYERS);
    }

    // ── architecture capability gate ─────────────────────────────────────

    fn make_mla_weights() -> larql_models::ModelWeights {
        let arch = larql_models::detect_from_json(&serde_json::json!({
            "model_type": "deepseek_v2",
            "hidden_size": HIDDEN,
            "intermediate_size": INTERMEDIATE,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": HIDDEN,
            "kv_lora_rank": 4,
            "q_lora_rank": 4,
            "rope_theta": 10000.0,
            "vocab_size": VOCAB,
        }));
        assert!(arch.uses_mla(), "fixture must produce an MLA architecture");

        let mut embed = ndarray::Array2::<f32>::zeros((VOCAB, HIDDEN));
        for i in 0..VOCAB {
            embed[[i, i % HIDDEN]] = 1.0;
        }
        let embed = embed.into_shared();
        let lm_head = embed.clone();

        larql_models::ModelWeights {
            tensors: HashMap::new(),
            vectors: HashMap::new(),
            raw_bytes: HashMap::new(),
            skipped_tensors: Vec::new(),
            packed_mmaps: HashMap::new(),
            packed_byte_ranges: HashMap::new(),
            embed,
            lm_head,
            position_embed: None,
            num_layers: NUM_LAYERS,
            hidden_size: HIDDEN,
            intermediate_size: INTERMEDIATE,
            vocab_size: VOCAB,
            head_dim: HIDDEN,
            num_q_heads: 1,
            num_kv_heads: 1,
            rope_base: 10000.0,
            arch,
        }
    }

    #[test]
    fn build_browse_passes_for_mla_arch() {
        // Browse-level extracts don't need attention; MLA must succeed.
        let dir = TempDir::new().unwrap();
        let weights = make_mla_weights();
        let tok = tokenizer();
        let mut cb = SilentBuildCallbacks;
        build_vindex(
            &weights,
            &tok,
            "test/mla",
            dir.path(),
            3,
            ExtractLevel::Browse,
            StorageDtype::F32,
            &mut cb,
        )
        .expect("Browse-level MLA extract should succeed (no attention written)");
        // Sanity: gate_vectors / embeddings still got written.
        assert!(dir.path().join("gate_vectors.bin").exists());
        assert!(dir.path().join("embeddings.bin").exists());
    }

    #[test]
    fn build_inference_rejects_mla_before_writing() {
        // The capability gate must fire before any output file is created
        // so a failed extract leaves no half-populated vindex on disk.
        let dir = TempDir::new().unwrap();
        let weights = make_mla_weights();
        let tok = tokenizer();
        let mut cb = SilentBuildCallbacks;
        let err = build_vindex(
            &weights,
            &tok,
            "test/mla",
            dir.path(),
            3,
            ExtractLevel::Inference,
            StorageDtype::F32,
            &mut cb,
        )
        .expect_err("Inference-level MLA extract must be rejected up front");
        let msg = err.to_string();
        assert!(
            msg.contains("MLA"),
            "error should mention MLA capability gap: {msg}"
        );

        let written: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        assert!(
            written.is_empty(),
            "MLA rejection must happen before any file is written; \
             found leftovers: {written:?}"
        );
    }
}
