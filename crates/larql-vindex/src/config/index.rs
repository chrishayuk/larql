//! Top-level vindex on-disk shape — `index.json` + per-layer info
//! + per-record `down_meta.bin` shape.
//!
//! Carved out of the monolithic `config/types.rs` in the 2026-04-25
//! round-2 cleanup. Aggregates types from sibling modules
//! (`quantization`, `compliance`, `model`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::compliance::LayerBands;
use super::model::VindexModelConfig;
use super::quantization::{Fp4Config, QuantFormat};

#[derive(Clone, Default, Serialize, Deserialize)]
pub struct VindexConfig {
    /// Format version.
    pub version: u32,
    /// Original model name (e.g., "google/gemma-3-4b-it").
    pub model: String,
    /// Model family (e.g., "gemma3", "llama").
    pub family: String,
    /// Provenance: which model checkpoint this vindex was built from.
    #[serde(default)]
    pub source: Option<VindexSource>,
    /// SHA256 checksums of each binary file for integrity verification.
    #[serde(default)]
    pub checksums: Option<BTreeMap<String, String>>,
    /// Number of layers.
    pub num_layers: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Intermediate (FFN) size.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Embedding scale factor.
    pub embed_scale: f32,
    /// What level of weights are included.
    #[serde(default)]
    pub extract_level: ExtractLevel,
    /// Storage precision (f32 or f16).
    #[serde(default)]
    pub dtype: crate::config::dtype::StorageDtype,
    /// Quantisation format of the model weights written alongside this
    /// vindex. `None` means float storage controlled by `dtype`;
    /// `Q4K` means Q4_K/Q6_K blocks in `attn_weights_q4k.bin` +
    /// `interleaved_kquant.bin`. Loaders dispatch on this field so they
    /// don't have to sniff filenames.
    #[serde(default)]
    pub quant: QuantFormat,
    /// Model-specific layer band boundaries for DESCRIBE and label matching.
    #[serde(default)]
    pub layer_bands: Option<LayerBands>,
    /// Per-layer info for gate_vectors.bin layout.
    pub layers: Vec<VindexLayerInfo>,
    /// Top-K tokens stored per feature in down metadata.
    pub down_top_k: usize,
    /// Whether model_weights.bin is present (legacy, use extract_level).
    #[serde(default)]
    pub has_model_weights: bool,
    /// Model config for architecture reconstruction.
    #[serde(default)]
    pub model_config: Option<VindexModelConfig>,
    /// Optional FP4/FP8 block-storage manifest. Set when one or more FFN
    /// projections are stored in the block-quantised format described
    /// in `docs/specs/vindex-format-spec.md` §5.10 and
    /// `docs/specs/fp4-format-spec.md`.
    /// Absent or null → legacy f16/f32 projection files are
    /// authoritative and loaders use the legacy codepath.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fp4: Option<Fp4Config>,

    /// FFN weight storage layout (§5.12). When `PerLayer`, FFN weights
    /// live in `layers/layer_{L:02}.weights` — one file per layer, format
    /// declared in each file's header. Works for both dense
    /// (num_entries=1) and MoE (num_entries=num_experts). Absent → legacy
    /// flat-file layout (`interleaved_kquant.bin` / `experts_packed.bin`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ffn_layout: Option<FfnLayout>,

    /// BitNet 1.58 native ternary layout, when the vindex was built
    /// from an I2_S GGUF with `--keep-quant`.  When `Some`, the
    /// `bitnet/` subdirectory holds the I2_S-packed BitLinear
    /// weights (one `.i2s` file per logical tensor) and the
    /// `bitnet/scales.f32` concatenation of per-channel scales.
    /// Loaders dispatch on this field to construct
    /// `BitLinearWeight` containers without ever materialising f16
    /// or f32 weight tensors at runtime.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bitnet_layout: Option<BitnetLayout>,
}

/// Layout descriptor for a `--keep-quant` BitNet vindex.
///
/// Captures everything the loader needs to mmap the `bitnet/` files
/// and reconstruct typed `BitLinearWeight`s without re-reading the
/// source GGUF.  Per-tensor metadata is keyed by the GGUF tensor
/// name (e.g. `blk.0.attn_q.weight`); the corresponding bytes live
/// at `<bitnet/>/<tensor_name>.i2s`, the scales at
/// `<bitnet/scales.f32>` starting at `scale_offset` for `rows`
/// f32 entries.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BitnetLayout {
    /// Per-tensor entries.  Order is meaningful (it determines the
    /// scale offset packing) but loaders look up by name.
    pub tensors: Vec<BitnetTensorEntry>,
    /// Total number of f32 entries in `bitnet/scales.f32`.  Used to
    /// validate the file size at load time.
    pub total_scale_count: usize,
    /// RMSnorm epsilon for the model (used by `BitNetFfn` and
    /// attention sub-norms).  Read from `*.attention.layer_norm_rms
    /// _epsilon` GGUF metadata at convert time so the loader does
    /// not re-parse the source.
    #[serde(default = "default_rms_eps")]
    pub rms_eps: f32,
    /// Dimension of one attention head.  Read from
    /// `*.rope.dimension_count` (= `head_dim` for BitNet b1.58).
    /// Loaders use this + `n_q_heads` to decompose Q/K/V projections
    /// into per-head subvectors.
    #[serde(default)]
    pub head_dim: usize,
    /// Number of query heads. From `*.attention.head_count`.
    #[serde(default)]
    pub n_q_heads: usize,
    /// Number of key/value heads (GQA).  From `*.attention.head_count_kv`.
    #[serde(default)]
    pub n_kv_heads: usize,
    /// RoPE theta (base).  From `*.rope.freq_base` (or its f32 cousin).
    #[serde(default = "default_rope_base")]
    pub rope_base: f64,
}

fn default_rms_eps() -> f32 {
    1e-5
}

fn default_rope_base() -> f64 {
    10000.0
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BitnetTensorEntry {
    /// GGUF tensor name (e.g. `blk.0.ffn_down.weight`).
    pub name: String,
    /// Output dimension (number of rows in the matvec sense).
    pub rows: usize,
    /// Input dimension (must be a multiple of 4 for the I2_S
    /// packing).
    pub cols: usize,
    /// Byte offset into `bitnet/scales.f32` where this tensor's
    /// per-row scale vector starts.  Length is `rows` f32s.
    pub scale_offset: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FfnLayout {
    PerLayer,
}

/// Provenance: which model checkpoint this vindex was built from.
///
/// The pre-v1 nullables (`huggingface_repo`, `huggingface_revision`,
/// `safetensors_sha256`) stay optional on the in-process struct so
/// existing manifests deserialise unchanged. The v1 provenance
/// hardening (`base_model_sha`, `extractor_sha`,
/// `base_safetensors_sha256` as a per-shard map) lives in additional
/// optional fields populated by new extracts and by the
/// `backfill-provenance` step (TODO). Translation to the v1 spec
/// (`format::spec::translate_source`) requires all three new fields
/// present.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VindexSource {
    #[serde(default)]
    pub huggingface_repo: Option<String>,
    #[serde(default)]
    pub huggingface_revision: Option<String>,
    /// Legacy single-shard digest. Pre-v1 vindexes that captured a
    /// single safetensors file used this. Superseded by
    /// `base_safetensors_sha256` for multi-shard models.
    #[serde(default)]
    pub safetensors_sha256: Option<String>,
    /// ISO 8601 timestamp of extraction.
    pub extracted_at: String,
    /// Version of larql used for extraction.
    pub larql_version: String,

    // ── v1 provenance fields (optional on disk, required by spec) ──
    /// Upstream git commit SHA at extract time. Pins the exact bytes
    /// the vindex was built from. Required for v1 publish.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_model_sha: Option<String>,
    /// Git SHA of the larql repo at extract time. Combined with
    /// `larql_version` this lets a validator reproduce the extraction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extractor_sha: Option<String>,
    /// Per-shard SHA256 of every safetensors file in the upstream
    /// repo, keyed by filename. Catches upstream force-pushes that
    /// mutate bytes under a stable commit hash.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_safetensors_sha256: Option<std::collections::BTreeMap<String, String>>,
}

/// What components are included in the vindex. Strictly increasing —
/// each tier is a superset of the previous.
///
/// | Tier        | Adds                                   | Enables                                |
/// |-------------|----------------------------------------|----------------------------------------|
/// | `browse`    | gate, embed, down_meta, tokenizer      | WALK / DESCRIBE / SELECT               |
/// | `attention` | + attention + norms                    | client-side of `run --ffn URL` (Act 2) |
/// | `inference` | + FFN up/down                          | full local forward pass (INFER)        |
/// | `all`       | + lm_head + any COMPILE extras         | COMPILE                                |
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[derive(Default)]
pub enum ExtractLevel {
    /// Gate + embed + down_meta + tokenizer. Enables WALK, DESCRIBE,
    /// SELECT. No forward pass possible.
    #[default]
    Browse,
    /// + attention + norms. Enables the client-side half of
    /// `larql run --ffn URL` (Act 2 of the Gemma 4 MoE demo). Cannot
    /// run a forward pass alone — FFN must live somewhere else.
    Attention,
    /// + FFN up/down weights. Enables full local INFER.
    Inference,
    /// + lm_head (when not tied to embed) + anything else future
    /// COMPILE passes need. Enables COMPILE.
    All,
}

impl ExtractLevel {
    /// Whether this tier includes attention weights + norms.
    /// True for Attention, Inference, All.
    pub fn writes_attn(self) -> bool {
        self >= Self::Attention
    }

    /// Whether this tier includes FFN up/down weight files (the full
    /// compute weights, not just the gate used by KNN).
    /// True for Inference, All.
    pub fn writes_ffn(self) -> bool {
        self >= Self::Inference
    }

    /// Whether this tier writes lm_head. When the model ties
    /// embeddings (embed_tokens shares weights with lm_head), the
    /// writer may still skip it — this is the intent flag.
    /// True for Inference, All.
    pub fn writes_lm_head(self) -> bool {
        self >= Self::Inference
    }
}

impl std::fmt::Display for ExtractLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Browse => write!(f, "browse"),
            Self::Attention => write!(f, "attention"),
            Self::Inference => write!(f, "inference"),
            Self::All => write!(f, "all"),
        }
    }
}

impl VindexConfig {
    /// Estimate the resident heap size of `ModelWeights` after
    /// `load_model_weights_with_opts` completes against this vindex.
    ///
    /// The estimate is intentionally conservative — it assumes the
    /// loader materialises every weight tensor at the configured
    /// `dtype` (f16 = 2 B/elem, f32 = 4 B/elem) plus a uniform
    /// 12 % overhead for `Vec<>` headers, padding, and the per-layer
    /// dequant scratch buffers used during forward.  Used by
    /// `larql-server`'s startup pre-flight check (BUG-infer-deadlock
    /// §5.5) to refuse to start when the cgroup is sized below the
    /// load.
    ///
    /// The estimate is *not* exact — it does not (yet) model the
    /// per-channel scale tensors used by ternary BitNet weights, the
    /// extra working buffers needed by a dense forward, or the
    /// kernel-page-cache contribution from mmap files.  Tolerance is
    /// roughly ±10–15 % vs measured RSS-after-load on the
    /// vindexes in production today.
    pub fn estimate_resident_bytes(&self) -> u64 {
        if !self.has_inference_weights() {
            // Browse-only vindex — the in-process structures are
            // gate vectors + tokenizer + tiny overhead.
            return self.browse_only_resident_bytes();
        }
        // Quantized vindexes (Q4_K/Q6_K/…, ~4.5 bits/elem) do NOT
        // load their weights at the dtype width this estimator
        // assumes — sizing them with `bytes_per_float` over-counts
        // 3–4× and would make memcheck REFUSE a quantized model that
        // actually fits.  Since quantized models are the main reason
        // to care about RSS, that defeats the feature.  Rather than
        // model every quant format's exact resident layout here
        // (fragile), return 0 so the startup pre-flight skips
        // quantized vindexes.  The caller treats a 0 estimate as
        // "skip" (see bootstrap memcheck).  Dense f16/f32 vindexes —
        // the case this estimator was validated against — still get a
        // real estimate below.
        if self.quant != crate::QuantFormat::None {
            return 0;
        }
        let elem = crate::config::dtype::bytes_per_float(self.dtype) as u64;
        let layers = self.num_layers as u64;
        let hidden = self.hidden_size as u64;
        let inter = self.intermediate_size as u64;
        let vocab = self.vocab_size as u64;

        // embed: vocab * hidden * elem
        let embed = vocab.saturating_mul(hidden).saturating_mul(elem);
        // lm_head: same shape as embed (or zero if tied; we don't
        // track tying explicitly, so assume present).
        let lm_head = embed;
        // Per-layer attn: q + k + v + o, each hidden * hidden.
        let attn_per_layer = 4u64
            .saturating_mul(hidden)
            .saturating_mul(hidden)
            .saturating_mul(elem);
        // Per-layer FFN: gate + up + down, each hidden * inter.
        let ffn_per_layer = 3u64
            .saturating_mul(hidden)
            .saturating_mul(inter)
            .saturating_mul(elem);
        // Per-layer norms (input_norm, post_attn_norm), 2 * hidden * f32.
        let norm_per_layer = 2u64.saturating_mul(hidden).saturating_mul(4);

        let per_layer = attn_per_layer
            .saturating_add(ffn_per_layer)
            .saturating_add(norm_per_layer);
        let total = embed
            .saturating_add(lm_head)
            .saturating_add(per_layer.saturating_mul(layers));

        // 12 % overhead for Vec<> headers, padding, dequant buffers.
        total.saturating_add(total / 8)
    }

    /// Whether this vindex has inference-level weights to load.
    /// True for `Inference` / `All` extract levels OR when the legacy
    /// `has_model_weights` flag is set.
    pub fn has_inference_weights(&self) -> bool {
        self.has_model_weights
            || self.extract_level == ExtractLevel::Inference
            || self.extract_level == ExtractLevel::All
    }

    /// Resident-size estimate for a browse-only vindex — just the
    /// gate matrices + embeddings + tokenizer.  Sized as the f32
    /// expansion of the gate vectors (worst case under warmup).
    fn browse_only_resident_bytes(&self) -> u64 {
        let hidden = self.hidden_size as u64;
        let vocab = self.vocab_size as u64;
        // Gate vectors: sum(num_features) * hidden * f32.
        let gate: u64 = self
            .layers
            .iter()
            .map(|l| (l.num_features as u64) * hidden * 4)
            .sum();
        // Embeddings: vocab * hidden * f32 (warmed).
        let embed = vocab.saturating_mul(hidden).saturating_mul(4);
        gate.saturating_add(embed).saturating_add(64 * 1024 * 1024)
        // ~64 MiB for tokenizer + assorted overhead.
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VindexLayerInfo {
    pub layer: usize,
    pub num_features: usize,
    /// Byte offset into gate_vectors.bin.
    pub offset: u64,
    /// Byte length of this layer's gate data.
    pub length: u64,
    /// Number of experts at this layer (None or absent for dense models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_experts: Option<usize>,
    /// Features per expert (None or absent for dense models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_features_per_expert: Option<usize>,
}

/// Down metadata entry in the NDJSON file (compact, no vectors).
#[derive(Serialize, Deserialize)]
pub struct DownMetaRecord {
    #[serde(rename = "l")]
    pub layer: usize,
    #[serde(rename = "f")]
    pub feature: usize,
    #[serde(rename = "t")]
    pub top_token: String,
    #[serde(rename = "i")]
    pub top_token_id: u32,
    #[serde(rename = "c")]
    pub c_score: f32,
    #[serde(rename = "k")]
    pub top_k: Vec<DownMetaTopK>,
}

#[derive(Serialize, Deserialize)]
pub struct DownMetaTopK {
    #[serde(rename = "t")]
    pub token: String,
    #[serde(rename = "i")]
    pub token_id: u32,
    #[serde(rename = "s")]
    pub logit: f32,
}

#[cfg(test)]
mod fp4_schema_tests {
    use super::*;
    // Bring sibling-module types into scope — Fp4Config / Precision /
    // ProjectionFormat / Projections live in `config::quantization`,
    // and the FP4 filename constants live in `format::filenames`.
    use super::super::quantization::{Fp4Config, Precision};
    use crate::format::filenames::{DOWN_FEATURES_FP8_BIN, GATE_VECTORS_FP4_BIN};

    #[test]
    fn option_b_default_shape() {
        let cfg = Fp4Config::option_b_default();
        assert_eq!(cfg.fp4_format_version, 1);
        assert_eq!(cfg.block_elements, 256);
        assert_eq!(cfg.sub_block_elements, 32);
        assert_eq!(cfg.sub_block_scale_dtype, "fp8_e4m3");
        assert_eq!(cfg.block_scale_dtype, "fp8_e4m3");
        assert_eq!(cfg.value_encoding, "fp4_e2m1_mxfp4_nibble_order");
        assert!(matches!(cfg.projections.gate.precision, Precision::Fp4));
        assert!(matches!(cfg.projections.up.precision, Precision::Fp4));
        assert!(matches!(cfg.projections.down.precision, Precision::Fp8));
        assert_eq!(cfg.projections.gate.file, GATE_VECTORS_FP4_BIN);
        assert_eq!(cfg.projections.down.file, DOWN_FEATURES_FP8_BIN);
        assert_eq!(cfg.compliance_gate.threshold_ratio, 16.0);
        assert_eq!(cfg.compliance_gate.min_compliant_fraction, 0.99);
        assert!(matches!(
            cfg.compliance_gate.fallback_precision,
            Precision::Fp8
        ));
        assert_eq!(cfg.compliance_report, "fp4_compliance.json");
    }

    #[test]
    fn fp4_config_serde_round_trip() {
        let cfg = Fp4Config::option_b_default();
        let json = serde_json::to_string(&cfg).unwrap();
        let back: Fp4Config = serde_json::from_str(&json).unwrap();
        assert_eq!(back.fp4_format_version, cfg.fp4_format_version);
        assert_eq!(back.block_elements, cfg.block_elements);
        assert_eq!(back.projections.gate.file, cfg.projections.gate.file);
    }

    #[test]
    fn precision_json_is_snake_case() {
        let cfg = Fp4Config::option_b_default();
        let json = serde_json::to_string(&cfg).unwrap();
        // The JSON surface must use the stable tags the format spec pins.
        assert!(json.contains("\"fp4\""));
        assert!(json.contains("\"fp8\""));
        assert!(!json.contains("\"Fp4\""), "camel/title case leaked: {json}");
    }

    #[test]
    fn vindex_config_without_fp4_serialises_without_key() {
        // Verify the `skip_serializing_if = "Option::is_none"` path so a
        // legacy vindex's index.json is byte-stable after a round trip.
        let cfg = VindexConfig {
            version: 2,
            model: "x".into(),
            family: "gemma3".into(),
            source: None,
            checksums: None,
            num_layers: 1,
            hidden_size: 256,
            intermediate_size: 1024,
            vocab_size: 32,
            embed_scale: 1.0,
            extract_level: ExtractLevel::default(),
            dtype: Default::default(),
            quant: QuantFormat::None,
            layer_bands: None,
            layers: vec![],
            down_top_k: 10,
            has_model_weights: false,
            model_config: None,
            fp4: None,
            ffn_layout: None,
            bitnet_layout: None,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(
            !json.contains("\"fp4\""),
            "legacy config leaked fp4 field: {json}"
        );

        // And still deserialises when the key is absent (default).
        let parsed: VindexConfig = serde_json::from_str(&json).unwrap();
        assert!(parsed.fp4.is_none());
    }

    #[test]
    fn ffn_layout_round_trips_as_snake_case_enum() {
        let parsed: VindexConfig =
            serde_json::from_str(r#"{"version":2,"model":"x","family":"gemma3","num_layers":0,"hidden_size":0,"intermediate_size":0,"vocab_size":0,"embed_scale":1.0,"layers":[],"down_top_k":0,"ffn_layout":"per_layer"}"#)
                .unwrap();
        assert_eq!(parsed.ffn_layout, Some(FfnLayout::PerLayer));
        let json = serde_json::to_string(&parsed).unwrap();
        assert!(json.contains("\"ffn_layout\":\"per_layer\""));
    }

    #[test]
    fn extract_level_display_covers_all_variants() {
        assert_eq!(ExtractLevel::Browse.to_string(), "browse");
        assert_eq!(ExtractLevel::Attention.to_string(), "attention");
        assert_eq!(ExtractLevel::Inference.to_string(), "inference");
        assert_eq!(ExtractLevel::All.to_string(), "all");
    }

    #[test]
    fn extract_level_writes_attn_matches_strict_ordering() {
        assert!(!ExtractLevel::Browse.writes_attn());
        assert!(ExtractLevel::Attention.writes_attn());
        assert!(ExtractLevel::Inference.writes_attn());
        assert!(ExtractLevel::All.writes_attn());
    }

    #[test]
    fn extract_level_writes_ffn_only_at_inference_and_above() {
        assert!(!ExtractLevel::Browse.writes_ffn());
        assert!(!ExtractLevel::Attention.writes_ffn());
        assert!(ExtractLevel::Inference.writes_ffn());
        assert!(ExtractLevel::All.writes_ffn());
    }

    #[test]
    fn extract_level_writes_lm_head_only_at_inference_and_above() {
        assert!(!ExtractLevel::Browse.writes_lm_head());
        assert!(!ExtractLevel::Attention.writes_lm_head());
        assert!(ExtractLevel::Inference.writes_lm_head());
        assert!(ExtractLevel::All.writes_lm_head());
    }

    #[test]
    fn vindex_source_v1_provenance_fields_round_trip() {
        let mut digests = std::collections::BTreeMap::new();
        digests.insert("model-00001-of-00002.safetensors".into(), "a".repeat(64));
        digests.insert("model-00002-of-00002.safetensors".into(), "b".repeat(64));
        let src = VindexSource {
            huggingface_repo: Some("google/gemma-3-4b-it".into()),
            huggingface_revision: Some("main".into()),
            safetensors_sha256: None,
            extracted_at: "2026-05-17T12:00:00Z".into(),
            larql_version: "0.2.0".into(),
            base_model_sha: Some("1adbacd6b6dee75c".into()),
            extractor_sha: Some("9f3a2c".into()),
            base_safetensors_sha256: Some(digests),
        };
        let json = serde_json::to_string(&src).unwrap();
        assert!(json.contains("\"base_model_sha\":\"1adbacd6b6dee75c\""));
        assert!(json.contains("\"extractor_sha\":\"9f3a2c\""));
        assert!(json.contains("\"base_safetensors_sha256\""));

        let back: VindexSource = serde_json::from_str(&json).unwrap();
        assert_eq!(back.base_model_sha.as_deref(), Some("1adbacd6b6dee75c"));
        assert_eq!(back.extractor_sha.as_deref(), Some("9f3a2c"));
        assert_eq!(
            back.base_safetensors_sha256.as_ref().map(|m| m.len()),
            Some(2)
        );
    }

    #[test]
    fn vindex_source_v1_fields_omitted_when_none() {
        let src = VindexSource {
            huggingface_repo: None,
            huggingface_revision: None,
            safetensors_sha256: None,
            extracted_at: "2026-05-17T12:00:00Z".into(),
            larql_version: "0.2.0".into(),
            base_model_sha: None,
            extractor_sha: None,
            base_safetensors_sha256: None,
        };
        let json = serde_json::to_string(&src).unwrap();
        assert!(
            !json.contains("base_model_sha"),
            "None v1 fields must be omitted (skip_serializing_if): {json}"
        );
        assert!(!json.contains("extractor_sha"), "{json}");
        assert!(!json.contains("base_safetensors_sha256"), "{json}");
    }

    #[test]
    fn pre_v1_source_json_deserialises_with_v1_fields_as_none() {
        // Pre-v1 manifests on disk don't have base_model_sha /
        // extractor_sha / base_safetensors_sha256. The struct must
        // deserialise cleanly with them as None.
        let pre_v1 = r#"{
            "huggingface_repo": "google/gemma-3-4b-it",
            "huggingface_revision": null,
            "safetensors_sha256": null,
            "extracted_at": "2026-05-17T12:00:00Z",
            "larql_version": "0.1.0"
        }"#;
        let src: VindexSource = serde_json::from_str(pre_v1).unwrap();
        assert!(src.base_model_sha.is_none());
        assert!(src.extractor_sha.is_none());
        assert!(src.base_safetensors_sha256.is_none());
    }

    #[test]
    fn vindex_config_with_fp4_round_trips() {
        let cfg = VindexConfig {
            version: 2,
            model: "x".into(),
            family: "gemma3".into(),
            source: None,
            checksums: None,
            num_layers: 1,
            hidden_size: 256,
            intermediate_size: 1024,
            vocab_size: 32,
            embed_scale: 1.0,
            extract_level: ExtractLevel::default(),
            dtype: Default::default(),
            quant: QuantFormat::None,
            layer_bands: None,
            layers: vec![],
            down_top_k: 10,
            has_model_weights: false,
            model_config: None,
            fp4: Some(Fp4Config::option_b_default()),
            ffn_layout: None,
            bitnet_layout: None,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        assert!(json.contains("\"fp4\""));
        let parsed: VindexConfig = serde_json::from_str(&json).unwrap();
        let fp4 = parsed.fp4.expect("round trip kept fp4");
        assert!(matches!(fp4.projections.down.precision, Precision::Fp8));
    }
}

#[cfg(test)]
mod resident_size_tests {
    use super::*;
    use crate::config::dtype::StorageDtype;

    fn cfg(extract: ExtractLevel, dtype: StorageDtype, num_layers: usize) -> VindexConfig {
        VindexConfig {
            num_layers,
            hidden_size: 2560,
            intermediate_size: 6912,
            vocab_size: 128_256,
            extract_level: extract,
            dtype,
            layers: (0..num_layers)
                .map(|i| VindexLayerInfo {
                    layer: i,
                    num_features: 6912,
                    offset: 0,
                    length: 0,
                    num_experts: None,
                    num_features_per_expert: None,
                })
                .collect(),
            ..VindexConfig::default()
        }
    }

    /// Bug-report scenario: BitNet b1.58 2 B 4 T at 30 layers,
    /// hidden 2560, intermediate 6912, vocab 128 256, dtype f16,
    /// extract_level Inference.  Estimator should land in the
    /// 5–6 GB ballpark to match measured RSS-after-load.
    #[test]
    fn estimate_for_bitnet_2b_inference_lands_in_expected_range() {
        let c = cfg(ExtractLevel::Inference, StorageDtype::F16, 30);
        let est = c.estimate_resident_bytes();
        // Production triage observed ~5.0 GB heap at peak.
        // Estimator allows for f16 storage + 12 % overhead;
        // accept anywhere in [4 GB, 8 GB].
        let gb = (est as f64) / (1024.0 * 1024.0 * 1024.0);
        assert!((4.0..=8.0).contains(&gb), "got {gb} GB");
    }

    /// Browse-only vindex (no inference weights) reports a smaller
    /// resident estimate than the inference path — the latter
    /// includes the full attention + FFN per-layer tensors which
    /// dominate at scale.
    #[test]
    fn estimate_for_browse_level_is_smaller_than_inference() {
        let browse = cfg(ExtractLevel::Browse, StorageDtype::F16, 30);
        let infer = cfg(ExtractLevel::Inference, StorageDtype::F16, 30);
        let b = browse.estimate_resident_bytes();
        let i = infer.estimate_resident_bytes();
        assert!(b < i, "browse {b} bytes vs inference {i} bytes");
    }

    /// f32-storage doubles the inference estimate vs f16 — sanity
    /// check that `bytes_per_float` is plumbed in correctly.
    #[test]
    fn estimate_doubles_for_f32_vs_f16() {
        let f16 = cfg(ExtractLevel::Inference, StorageDtype::F16, 30);
        let f32 = cfg(ExtractLevel::Inference, StorageDtype::F32, 30);
        let r16 = f16.estimate_resident_bytes();
        let r32 = f32.estimate_resident_bytes();
        // Norms (per-layer ~10 KiB) and the 12 % overhead constant
        // make the ratio a bit under 2x; accept the [1.7, 2.1] band.
        let ratio = (r32 as f64) / (r16 as f64);
        assert!(
            (1.7..=2.1).contains(&ratio),
            "ratio {ratio} (r16={r16}, r32={r32})"
        );
    }

    /// Quantized vindexes (Q4_K etc.) must NOT be sized at the dtype
    /// width — that over-counts 3–4× and would make memcheck refuse
    /// a model that fits.  estimate_resident_bytes returns 0 ("skip
    /// the pre-flight") for them.
    #[test]
    fn estimate_skips_quantized() {
        let mut q4k = cfg(ExtractLevel::Inference, StorageDtype::F16, 30);
        q4k.quant = crate::QuantFormat::Q4K;
        assert_eq!(
            q4k.estimate_resident_bytes(),
            0,
            "Q4_K vindex must skip memcheck (returns 0)"
        );

        // A plain dense f16 vindex still gets a real (non-zero)
        // estimate — the case this estimator was validated against.
        let dense = cfg(ExtractLevel::Inference, StorageDtype::F16, 30);
        assert!(
            dense.estimate_resident_bytes() > 0,
            "dense f16 vindex must still be estimated"
        );
    }

    /// has_inference_weights honours both the legacy
    /// has_model_weights flag and the modern extract_level field.
    #[test]
    fn has_inference_weights_handles_legacy_and_modern_flags() {
        let mut browse = cfg(ExtractLevel::Browse, StorageDtype::F16, 1);
        assert!(!browse.has_inference_weights());
        browse.has_model_weights = true; // legacy flag
        assert!(browse.has_inference_weights());

        let infer = cfg(ExtractLevel::Inference, StorageDtype::F16, 1);
        assert!(infer.has_inference_weights());
    }
}
