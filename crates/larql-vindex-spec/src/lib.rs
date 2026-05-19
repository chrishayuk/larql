//! Public contract for the vindex on-disk format.
//!
//! This crate is the *spec* — Rust types that model the v1 manifest's
//! structural and provenance contract, plus the validator threshold
//! matrix. It has zero larql-* deps so the writer (`larql-vindex`),
//! the loader, and external validators (CLI, Space) can all depend on
//! it without dragging in the wider workspace.
//!
//! ## Scope
//!
//! The spec models what the **validator** cares about:
//! - `vindex_spec_version` compatibility check.
//! - Provenance hardening: `source` is required and includes
//!   `base_model_sha` + `base_safetensors_sha256` + `extractor_sha`.
//! - `checksums` covers every `.bin` file the manifest references.
//! - Structural fields: dims, `extract_level`, `dtype`, `quant`,
//!   `layers` (with single-file or sharded slots), `down_top_k`.
//!
//! Loader-domain fields (`model_config`, `fp4`, `ffn_layout`,
//! `layer_bands`) are passed through via `serde(flatten)` into
//! [`VindexManifest::extra`]. They round-trip cleanly but the spec
//! doesn't validate their internal shape — that's the loader's job
//! and evolves under the on-disk `version` field, not
//! `vindex_spec_version`.
//!
//! ## Versioning
//!
//! Bumping [`VINDEX_SPEC_VERSION`] is a breaking change for every
//! published vindex. Tooling pins this crate via Cargo; the spec
//! version in the manifest is the integer compatibility tag, not the
//! evolution channel.

#![deny(missing_docs)]

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub mod thresholds;

/// Current spec version. Manifests with a different value are rejected
/// by [`VindexManifest::validate_self_consistency`].
pub const VINDEX_SPEC_VERSION: u32 = 1;

/// Per-shard size cap. Any `.bin` larger than this must split into
/// `<base>-NNNNN-of-NNNNN.bin` (zero-padded, 1-indexed).
///
/// 20 GiB — chosen to keep individual LFS uploads resumable and to
/// parallelise on typical home-uplink connections.
pub const MAX_SHARD_BYTES: u64 = 20 * 1024 * 1024 * 1024;

// ─── Top-level manifest ──────────────────────────────────────────────

/// Root of the v1 manifest (lives at `index.json` in the vindex root).
///
/// Fields with no `#[serde(skip_serializing_if)]` or `#[serde(default)]`
/// are REQUIRED on disk. Pre-v1 `index.json` had several nullables
/// here (`source`, `checksums`, `model_config`); v1 hardens them.
///
/// Loader-specific fields not modelled here (e.g. `model_config`,
/// `fp4`, `ffn_layout`, `layer_bands`) land in [`Self::extra`] via
/// `serde(flatten)`. They survive round-trips unchanged.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VindexManifest {
    /// Public spec version. Must equal [`VINDEX_SPEC_VERSION`].
    pub vindex_spec_version: u32,

    /// On-disk file-format version. Independent of the spec version —
    /// loader-domain fields evolve under this.
    pub version: u32,

    /// Upstream model identifier, e.g. `google/gemma-3-4b-it`.
    pub model: String,

    /// Architecture family for loader dispatch (`gemma3`, `llama`,
    /// `mistral`, `granite`, ...).
    pub family: String,

    /// Upstream provenance — where this vindex was extracted from.
    pub source: Source,

    /// SHA256 of every `.bin` the manifest references. The validator
    /// walks this map first; any file mentioned by `layers` whose
    /// digest doesn't match (or isn't listed) fails validation.
    pub checksums: BTreeMap<String, String>,

    /// Number of transformer layers in the upstream model.
    pub num_layers: u32,

    /// Hidden dimension.
    pub hidden_size: u32,

    /// FFN intermediate dimension.
    pub intermediate_size: u32,

    /// Vocabulary size.
    pub vocab_size: u32,

    /// Embedding scaling factor used by the loader.
    pub embed_scale: f32,

    /// What components are present, ordered Browse < Attention <
    /// Inference < All. Each tier is a superset of the previous.
    pub extract_level: ExtractLevel,

    /// Storage precision for float-stored tensors (gate vectors,
    /// embeddings, norms, ...). Independent of `quant`, which only
    /// covers the FFN weight blocks.
    pub dtype: StorageDtype,

    /// Quant scheme for the FFN weight files. `None` = float storage
    /// (controlled by `dtype`); `Q4K` (alias `Kquant`) = Q4_K/Q6_K
    /// blocks in `interleaved_kquant.bin` / `attn_weights_kquant.bin`
    /// (writers emit these new names; readers also accept the legacy
    /// `interleaved_q4k.bin` / `attn_weights_q4k.bin`).
    pub quant: QuantFormat,

    /// Per-layer offset table. Each entry uses either single-file or
    /// sharded form — never both, never neither.
    pub layers: Vec<LayerEntry>,

    /// K used for the top-K gate-feature lookup at runtime.
    pub down_top_k: u32,

    /// True when full weight tensors (not just gate vectors) are
    /// present. Derived from `extract_level` but kept explicit because
    /// some legacy vindexes set it independently.
    pub has_model_weights: bool,

    /// Loader-domain fields that survive round-trip but aren't
    /// validated by the spec (model_config, fp4, ffn_layout,
    /// layer_bands, etc.).
    #[serde(flatten)]
    pub extra: serde_json::Map<String, serde_json::Value>,
}

/// Upstream provenance. All fields REQUIRED in v1 — the pre-v1
/// nullables on `huggingface_revision` and `safetensors_sha256` are
/// retired here.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Source {
    /// Canonical HF repo of the upstream model.
    pub huggingface_repo: String,

    /// Branch or tag pulled at extract time (usually `"main"`).
    pub huggingface_revision: String,

    /// Git commit SHA of the upstream repo at extract time. The
    /// validator pulls exactly these bytes when reconstructing.
    pub base_model_sha: String,

    /// SHA256 of every safetensors shard in the upstream repo, keyed
    /// by filename. Catches upstream force-pushes that mutate bytes
    /// under a stable commit hash.
    pub base_safetensors_sha256: BTreeMap<String, String>,

    /// ISO 8601 timestamp of extraction.
    pub extracted_at: String,

    /// `larql` crate version that produced the vindex.
    pub larql_version: String,

    /// Git SHA of the `larql` repo at extract time. Combined with
    /// `larql_version` this lets a validator reproduce the extraction
    /// bit-for-bit (modulo float-reduction non-determinism).
    pub extractor_sha: String,
}

/// Strictly increasing extraction tier. Mirrors larql-vindex's
/// `config::index::ExtractLevel` — same lowercase serde tags so
/// existing manifests round-trip.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExtractLevel {
    /// Gate + embed + down_meta + tokenizer. Enables WALK, DESCRIBE,
    /// SELECT. No forward pass.
    Browse,
    /// `+` attention + norms. Enables the client side of remote-FFN
    /// inference.
    Attention,
    /// `+` FFN up/down weights. Enables full local INFER.
    Inference,
    /// `+` lm_head + COMPILE extras. Enables COMPILE.
    All,
}

/// Storage precision for float tensors. Mirrors larql-vindex's
/// `config::dtype::StorageDtype`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StorageDtype {
    /// IEEE 754 binary32.
    F32,
    /// IEEE 754 binary16.
    F16,
}

/// Quant scheme for FFN weight files. Mirrors larql-vindex's
/// `config::quantization::QuantFormat`. The v1 schema's `quant` enum
/// accepts `"none"`, `"q4k"`, and `"kquant"` — the latter two both map
/// to [`QuantFormat::Kquant`] (Q4_K / Q6_K family). Writers continue to
/// emit `"q4k"` so v1 vindexes published by old binaries keep their
/// wire format; a v2 schema (separate change) will flip the canonical
/// tag to `"kquant"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QuantFormat {
    /// Float storage controlled by [`StorageDtype`].
    None,
    /// Q4_K / Q6_K blocks in `interleaved_kquant.bin` /
    /// `attn_weights_kquant.bin` (or legacy `*_q4k` filenames). Accepts
    /// both `"q4k"` (default Serialize) and `"kquant"` on Deserialize.
    #[serde(alias = "kquant")]
    Q4K,
}

/// One layer's slot in a (possibly sharded) weight file.
///
/// Single-file form: `file` + `offset` + `length` populated, `shards`
/// absent. Sharded form: `shards` populated, the other three absent.
/// Exactly one form is present.
///
/// MoE-only fields (`num_experts`, `num_features_per_expert`) match
/// the existing `VindexLayerInfo` shape so MoE vindexes
/// (e.g. gemma-4-26B-A4B) round-trip.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayerEntry {
    /// Layer index, 0-based.
    pub layer: u32,

    /// Number of features at this layer (e.g. intermediate dim for FFN).
    pub num_features: u32,

    /// Source filename — single-file form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,

    /// Byte offset into the file — single-file form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub offset: Option<u64>,

    /// Length in bytes — single-file form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub length: Option<u64>,

    /// Sharded form. Each entry covers a contiguous range; the
    /// validator concatenates them in order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shards: Option<Vec<ShardSlot>>,

    /// Number of experts at this layer. None or absent for dense models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_experts: Option<u32>,

    /// Features per expert at this layer. None or absent for dense
    /// models.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub num_features_per_expert: Option<u32>,
}

/// One shard of a sharded layer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ShardSlot {
    /// Source filename. Convention: `<base>-NNNNN-of-NNNNN.bin`.
    pub file: String,
    /// Byte offset within this shard file.
    pub offset: u64,
    /// Length within this shard.
    pub length: u64,
}

// ─── Errors ──────────────────────────────────────────────────────────

/// Errors raised by [`VindexManifest::validate_self_consistency`].
#[derive(Debug, thiserror::Error)]
pub enum SpecError {
    /// The manifest declares a spec version this crate doesn't
    /// understand.
    #[error("manifest declares vindex_spec_version={got}, this crate supports {supported}")]
    UnsupportedSpecVersion {
        /// Version the manifest declares.
        got: u32,
        /// Version this crate supports.
        supported: u32,
    },

    /// A layer entry has both single-file and sharded slots, or
    /// neither.
    #[error(
        "layer {layer} must specify exactly one of single-file (file+offset+length) or sharded form; got single_file={has_single}, sharded={has_shards}"
    )]
    LayerSlotInvalid {
        /// Layer index.
        layer: u32,
        /// True if any of file/offset/length is present.
        has_single: bool,
        /// True if `shards` is present.
        has_shards: bool,
    },

    /// A layer references a file not present in `checksums`.
    #[error("layer {layer} references file '{file}' that is missing from `checksums`")]
    LayerFileUnchecksummed {
        /// Layer index.
        layer: u32,
        /// File the layer references.
        file: String,
    },
}

impl VindexManifest {
    /// Structural self-consistency check — no I/O. Catches wrong spec
    /// version, layer-slot ambiguity, and layer files missing from the
    /// `checksums` map. Full validation (sha256s on disk,
    /// reconstruction against upstream) lives in `larql verify`.
    pub fn validate_self_consistency(&self) -> Result<(), SpecError> {
        if self.vindex_spec_version != VINDEX_SPEC_VERSION {
            return Err(SpecError::UnsupportedSpecVersion {
                got: self.vindex_spec_version,
                supported: VINDEX_SPEC_VERSION,
            });
        }

        for layer in &self.layers {
            let has_single =
                layer.file.is_some() || layer.offset.is_some() || layer.length.is_some();
            let has_shards = layer.shards.is_some();
            if has_single == has_shards {
                return Err(SpecError::LayerSlotInvalid {
                    layer: layer.layer,
                    has_single,
                    has_shards,
                });
            }

            // Every file mentioned must appear in `checksums`.
            if let Some(ref f) = layer.file {
                if !self.checksums.contains_key(f) {
                    return Err(SpecError::LayerFileUnchecksummed {
                        layer: layer.layer,
                        file: f.clone(),
                    });
                }
            }
            if let Some(ref shards) = layer.shards {
                for s in shards {
                    if !self.checksums.contains_key(&s.file) {
                        return Err(SpecError::LayerFileUnchecksummed {
                            layer: layer.layer,
                            file: s.file.clone(),
                        });
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> VindexManifest {
        let mut checksums = BTreeMap::new();
        checksums.insert("interleaved_kquant.bin".into(), "a".repeat(64));
        VindexManifest {
            vindex_spec_version: VINDEX_SPEC_VERSION,
            version: 2,
            model: "google/gemma-3-4b-it".into(),
            family: "gemma3".into(),
            source: Source {
                huggingface_repo: "google/gemma-3-4b-it".into(),
                huggingface_revision: "main".into(),
                base_model_sha: "1adbacd6b6dee75c".into(),
                base_safetensors_sha256: BTreeMap::new(),
                extracted_at: "2026-05-17T12:00:00Z".into(),
                larql_version: "0.2.0".into(),
                extractor_sha: "9f3a2c".into(),
            },
            checksums,
            num_layers: 34,
            hidden_size: 2560,
            intermediate_size: 10240,
            vocab_size: 262208,
            embed_scale: 50.596_443,
            extract_level: ExtractLevel::Inference,
            dtype: StorageDtype::F16,
            quant: QuantFormat::Q4K,
            layers: vec![LayerEntry {
                layer: 0,
                num_features: 10240,
                file: Some("interleaved_kquant.bin".into()),
                offset: Some(0),
                length: Some(52_428_800),
                shards: None,
                num_experts: None,
                num_features_per_expert: None,
            }],
            down_top_k: 10,
            has_model_weights: true,
            extra: serde_json::Map::new(),
        }
    }

    #[test]
    fn sample_manifest_self_consistent() {
        sample_manifest().validate_self_consistency().unwrap();
    }

    #[test]
    fn rejects_mismatched_spec_version() {
        let mut m = sample_manifest();
        m.vindex_spec_version = 999;
        let err = m.validate_self_consistency().unwrap_err();
        assert!(matches!(err, SpecError::UnsupportedSpecVersion { .. }));
    }

    #[test]
    fn rejects_layer_with_both_slot_forms() {
        let mut m = sample_manifest();
        m.layers[0].shards = Some(vec![ShardSlot {
            file: "x-00001-of-00001.bin".into(),
            offset: 0,
            length: 1,
        }]);
        let err = m.validate_self_consistency().unwrap_err();
        assert!(matches!(
            err,
            SpecError::LayerSlotInvalid {
                has_single: true,
                has_shards: true,
                ..
            }
        ));
    }

    #[test]
    fn rejects_layer_with_neither_slot_form() {
        let mut m = sample_manifest();
        m.layers[0].file = None;
        m.layers[0].offset = None;
        m.layers[0].length = None;
        let err = m.validate_self_consistency().unwrap_err();
        assert!(matches!(
            err,
            SpecError::LayerSlotInvalid {
                has_single: false,
                has_shards: false,
                ..
            }
        ));
    }

    #[test]
    fn rejects_layer_file_not_in_checksums() {
        let mut m = sample_manifest();
        m.layers[0].file = Some("missing.bin".into());
        let err = m.validate_self_consistency().unwrap_err();
        match err {
            SpecError::LayerFileUnchecksummed { file, .. } => assert_eq!(file, "missing.bin"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn roundtrips_through_json() {
        let m = sample_manifest();
        let json = serde_json::to_string(&m).unwrap();
        let back: VindexManifest = serde_json::from_str(&json).unwrap();
        back.validate_self_consistency().unwrap();
        assert_eq!(back.model, m.model);
        assert_eq!(back.extract_level, m.extract_level);
        assert_eq!(back.dtype, m.dtype);
        assert_eq!(back.quant, m.quant);
    }

    #[test]
    fn extra_loader_fields_round_trip_via_flatten() {
        let mut m = sample_manifest();
        m.extra.insert(
            "model_config".into(),
            serde_json::json!({ "model_type": "gemma3", "head_dim": 256 }),
        );
        m.extra.insert(
            "fp4".into(),
            serde_json::json!({ "fp4_format_version": 1, "block_elements": 256 }),
        );

        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains("\"model_config\""));
        assert!(json.contains("\"fp4\""));

        let back: VindexManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.extra.get("model_config"), m.extra.get("model_config"));
        assert_eq!(back.extra.get("fp4"), m.extra.get("fp4"));
    }

    #[test]
    fn moe_layer_fields_round_trip() {
        let mut m = sample_manifest();
        m.layers[0].num_experts = Some(8);
        m.layers[0].num_features_per_expert = Some(1280);
        let json = serde_json::to_string(&m).unwrap();
        let back: VindexManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.layers[0].num_experts, Some(8));
        assert_eq!(back.layers[0].num_features_per_expert, Some(1280));
    }

    #[test]
    fn quant_format_serialises_lowercase() {
        let m = sample_manifest();
        let json = serde_json::to_string(&m).unwrap();
        // v1 wire-format contract: writers emit `"q4k"`. Readers also
        // accept `"kquant"` via the deserialize alias — see the next
        // test. Flip this assertion only when v2 lands.
        assert!(json.contains("\"quant\":\"q4k\""));
    }

    #[test]
    fn quant_format_deserialize_accepts_kquant_alias() {
        // Forward-compat: a v1 reader must accept manifests written
        // under the future canonical tag `"kquant"`.
        let json = "{\"quant\":\"kquant\"}";
        #[derive(Deserialize)]
        struct OnlyQuant {
            quant: QuantFormat,
        }
        let q: OnlyQuant = serde_json::from_str(json).unwrap();
        assert_eq!(q.quant, QuantFormat::Q4K);
    }

    #[test]
    fn extract_level_serialises_lowercase() {
        let m = sample_manifest();
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains("\"extract_level\":\"inference\""));
    }

    #[test]
    fn extract_level_ordering_strict() {
        assert!(ExtractLevel::Browse < ExtractLevel::Attention);
        assert!(ExtractLevel::Attention < ExtractLevel::Inference);
        assert!(ExtractLevel::Inference < ExtractLevel::All);
    }
}
