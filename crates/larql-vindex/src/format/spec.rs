//! Translation between the in-process `VindexConfig` and the public
//! v1 manifest type from `larql-vindex-spec`.
//!
//! The writer continues to emit `VindexConfig` to disk for back-compat.
//! Code paths that need a v1-compliant manifest (the publish flow,
//! `larql verify`, future export tools) go through
//! [`TryFrom<&VindexConfig> for VindexManifest`].
//!
//! The translation can fail when the in-process config is missing
//! provenance fields v1 requires (`base_model_sha`, `extractor_sha`,
//! the per-shard `base_safetensors_sha256` map). Those didn't exist
//! pre-v1 and the extractor isn't capturing them yet — see
//! [`SpecTranslationError`] for the surfaced cases. The expected
//! recovery is a `larql vindex backfill-provenance` step (TODO) that
//! fetches the upstream commit hash + safetensors digests and rewrites
//! `index.json` in place.
//!
//! Loader-domain fields (`model_config`, `fp4`, `ffn_layout`,
//! `layer_bands`) are passed into [`VindexManifest::extra`] via
//! `serde_json::Value` so they round-trip without the spec crate
//! needing to know their shape.

use larql_vindex_spec::{
    ExtractLevel as SpecExtractLevel, LayerEntry as SpecLayerEntry, QuantFormat as SpecQuantFormat,
    Source as SpecSource, StorageDtype as SpecStorageDtype, VindexManifest, VINDEX_SPEC_VERSION,
};

use crate::config::dtype::StorageDtype;
use crate::config::index::{ExtractLevel, VindexConfig, VindexLayerInfo, VindexSource};
use crate::config::quantization::QuantFormat;

/// Errors returned by [`VindexManifest::try_from`] when `VindexConfig`
/// is missing fields v1 requires.
#[derive(Debug, thiserror::Error)]
pub enum SpecTranslationError {
    /// `VindexConfig.source` was `None`. Pre-v1 vindexes built before
    /// the provenance struct was always populated.
    #[error("VindexConfig.source is None; cannot emit a v1 manifest without provenance")]
    MissingSource,

    /// `VindexConfig.source.huggingface_repo` was `None`.
    #[error("source.huggingface_repo is missing; v1 requires it")]
    MissingHuggingfaceRepo,

    /// `VindexConfig.source.huggingface_revision` was `None`.
    #[error("source.huggingface_revision is missing; v1 requires it. Re-extract or backfill.")]
    MissingHuggingfaceRevision,

    /// `VindexConfig.source` had no `base_model_sha` field at all
    /// (pre-v1 source struct never carried it).
    #[error(
        "source.base_model_sha is missing; v1 requires it. Run `larql vindex backfill-provenance` (TODO) to fetch the upstream commit hash."
    )]
    MissingBaseModelSha,

    /// `VindexConfig.source` had no `extractor_sha` field at all.
    #[error(
        "source.extractor_sha is missing; v1 requires it. Re-extract with a build that records the larql repo SHA."
    )]
    MissingExtractorSha,

    /// `VindexConfig.source` had no `base_safetensors_sha256` map
    /// (pre-v1 had a nullable single hex string instead).
    #[error(
        "source.base_safetensors_sha256 is missing or empty; v1 requires a per-shard digest map. Run backfill or re-extract."
    )]
    MissingSafetensorsDigests,

    /// `VindexConfig.checksums` was `None` or empty.
    #[error("checksums is missing or empty; v1 requires SHA256 for every .bin file")]
    MissingChecksums,
}

impl TryFrom<&VindexConfig> for VindexManifest {
    type Error = SpecTranslationError;

    fn try_from(cfg: &VindexConfig) -> Result<Self, Self::Error> {
        let src = cfg
            .source
            .as_ref()
            .ok_or(SpecTranslationError::MissingSource)?;
        let spec_source = translate_source(src)?;

        let checksums_in = cfg
            .checksums
            .as_ref()
            .ok_or(SpecTranslationError::MissingChecksums)?;
        if checksums_in.is_empty() {
            return Err(SpecTranslationError::MissingChecksums);
        }
        let checksums = checksums_in
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let layers = cfg.layers.iter().map(translate_layer).collect();
        let extra = collect_loader_extras(cfg);

        Ok(VindexManifest {
            vindex_spec_version: VINDEX_SPEC_VERSION,
            version: cfg.version,
            model: cfg.model.clone(),
            family: cfg.family.clone(),
            source: spec_source,
            checksums,
            num_layers: cfg.num_layers as u32,
            hidden_size: cfg.hidden_size as u32,
            intermediate_size: cfg.intermediate_size as u32,
            vocab_size: cfg.vocab_size as u32,
            embed_scale: cfg.embed_scale,
            extract_level: translate_extract_level(cfg.extract_level),
            dtype: translate_dtype(cfg.dtype),
            quant: translate_quant(cfg.quant),
            layers,
            down_top_k: cfg.down_top_k as u32,
            has_model_weights: cfg.has_model_weights,
            extra,
        })
    }
}

fn translate_source(src: &VindexSource) -> Result<SpecSource, SpecTranslationError> {
    let huggingface_repo = src
        .huggingface_repo
        .clone()
        .ok_or(SpecTranslationError::MissingHuggingfaceRepo)?;
    let huggingface_revision = src
        .huggingface_revision
        .clone()
        .ok_or(SpecTranslationError::MissingHuggingfaceRevision)?;
    let base_model_sha = src
        .base_model_sha
        .clone()
        .ok_or(SpecTranslationError::MissingBaseModelSha)?;
    let extractor_sha = src
        .extractor_sha
        .clone()
        .ok_or(SpecTranslationError::MissingExtractorSha)?;
    let base_safetensors_sha256 = src
        .base_safetensors_sha256
        .clone()
        .ok_or(SpecTranslationError::MissingSafetensorsDigests)?;
    if base_safetensors_sha256.is_empty() {
        return Err(SpecTranslationError::MissingSafetensorsDigests);
    }

    Ok(SpecSource {
        huggingface_repo,
        huggingface_revision,
        base_model_sha,
        base_safetensors_sha256,
        extracted_at: src.extracted_at.clone(),
        larql_version: src.larql_version.clone(),
        extractor_sha,
    })
}

fn translate_extract_level(lvl: ExtractLevel) -> SpecExtractLevel {
    match lvl {
        ExtractLevel::Browse => SpecExtractLevel::Browse,
        ExtractLevel::Attention => SpecExtractLevel::Attention,
        ExtractLevel::Inference => SpecExtractLevel::Inference,
        ExtractLevel::All => SpecExtractLevel::All,
    }
}

fn translate_dtype(dt: StorageDtype) -> SpecStorageDtype {
    match dt {
        StorageDtype::F32 => SpecStorageDtype::F32,
        StorageDtype::F16 => SpecStorageDtype::F16,
    }
}

fn translate_quant(q: QuantFormat) -> SpecQuantFormat {
    match q {
        QuantFormat::None => SpecQuantFormat::None,
        QuantFormat::Q4K => SpecQuantFormat::Q4K,
    }
}

fn translate_layer(info: &VindexLayerInfo) -> SpecLayerEntry {
    SpecLayerEntry {
        layer: info.layer as u32,
        num_features: info.num_features as u32,
        file: None,
        offset: Some(info.offset),
        length: Some(info.length),
        shards: None,
        num_experts: info.num_experts.map(|n| n as u32),
        num_features_per_expert: info.num_features_per_expert.map(|n| n as u32),
    }
}

/// Pack the loader-domain fields (`model_config`, `fp4`, `ffn_layout`,
/// `layer_bands`) into the spec's `extra` map. Each round-trips as
/// `serde_json::Value`; the spec doesn't validate their shape.
fn collect_loader_extras(cfg: &VindexConfig) -> serde_json::Map<String, serde_json::Value> {
    let mut extra = serde_json::Map::new();

    if let Some(ref bands) = cfg.layer_bands {
        if let Ok(v) = serde_json::to_value(bands) {
            extra.insert("layer_bands".into(), v);
        }
    }
    if let Some(ref mc) = cfg.model_config {
        if let Ok(v) = serde_json::to_value(mc) {
            extra.insert("model_config".into(), v);
        }
    }
    if let Some(ref fp4) = cfg.fp4 {
        if let Ok(v) = serde_json::to_value(fp4) {
            extra.insert("fp4".into(), v);
        }
    }
    if let Some(ref layout) = cfg.ffn_layout {
        if let Ok(v) = serde_json::to_value(layout) {
            extra.insert("ffn_layout".into(), v);
        }
    }

    extra
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::quantization::QuantFormat;
    use std::collections::HashMap;

    fn minimal_cfg() -> VindexConfig {
        VindexConfig {
            version: 2,
            model: "google/gemma-3-4b-it".into(),
            family: "gemma3".into(),
            source: None,
            checksums: None,
            num_layers: 1,
            hidden_size: 256,
            intermediate_size: 1024,
            vocab_size: 32,
            embed_scale: 1.0,
            extract_level: ExtractLevel::Inference,
            dtype: StorageDtype::F16,
            quant: QuantFormat::Q4K,
            layer_bands: None,
            layers: vec![],
            down_top_k: 10,
            has_model_weights: true,
            model_config: None,
            fp4: None,
            ffn_layout: None,
        }
    }

    #[test]
    fn errors_when_source_missing() {
        let cfg = minimal_cfg();
        let err = VindexManifest::try_from(&cfg).unwrap_err();
        assert!(matches!(err, SpecTranslationError::MissingSource));
    }

    fn pre_v1_source() -> VindexSource {
        VindexSource {
            huggingface_repo: Some("google/gemma-3-4b-it".into()),
            huggingface_revision: Some("main".into()),
            safetensors_sha256: None,
            extracted_at: "2026-05-17T12:00:00Z".into(),
            larql_version: "0.2.0".into(),
            base_model_sha: None,
            extractor_sha: None,
            base_safetensors_sha256: None,
        }
    }

    fn v1_source() -> VindexSource {
        let mut digests = std::collections::BTreeMap::new();
        digests.insert("model-00001-of-00002.safetensors".into(), "a".repeat(64));
        digests.insert("model-00002-of-00002.safetensors".into(), "b".repeat(64));
        VindexSource {
            base_model_sha: Some("1adbacd6b6dee75c".into()),
            extractor_sha: Some("9f3a2c".into()),
            base_safetensors_sha256: Some(digests),
            ..pre_v1_source()
        }
    }

    fn cfg_with_checksums(source: VindexSource) -> VindexConfig {
        let mut cfg = minimal_cfg();
        cfg.source = Some(source);
        let mut checks = HashMap::new();
        checks.insert("gate_vectors.bin".into(), "c".repeat(64));
        cfg.checksums = Some(checks);
        cfg
    }

    #[test]
    fn errors_when_checksums_missing() {
        let mut cfg = minimal_cfg();
        cfg.source = Some(v1_source());
        // checksums None — must surface as MissingChecksums even when
        // provenance is otherwise complete.
        let err = VindexManifest::try_from(&cfg).unwrap_err();
        assert!(matches!(err, SpecTranslationError::MissingChecksums));
    }

    #[test]
    fn errors_on_missing_base_model_sha() {
        let cfg = cfg_with_checksums(pre_v1_source());
        let err = VindexManifest::try_from(&cfg).unwrap_err();
        assert!(matches!(err, SpecTranslationError::MissingBaseModelSha));
    }

    #[test]
    fn errors_on_missing_extractor_sha() {
        let mut src = pre_v1_source();
        src.base_model_sha = Some("abc".into());
        let cfg = cfg_with_checksums(src);
        let err = VindexManifest::try_from(&cfg).unwrap_err();
        assert!(matches!(err, SpecTranslationError::MissingExtractorSha));
    }

    #[test]
    fn errors_on_missing_safetensors_digests() {
        let mut src = pre_v1_source();
        src.base_model_sha = Some("abc".into());
        src.extractor_sha = Some("def".into());
        let cfg = cfg_with_checksums(src);
        let err = VindexManifest::try_from(&cfg).unwrap_err();
        assert!(matches!(
            err,
            SpecTranslationError::MissingSafetensorsDigests
        ));
    }

    #[test]
    fn errors_on_empty_safetensors_digests() {
        let mut src = pre_v1_source();
        src.base_model_sha = Some("abc".into());
        src.extractor_sha = Some("def".into());
        src.base_safetensors_sha256 = Some(std::collections::BTreeMap::new());
        let cfg = cfg_with_checksums(src);
        let err = VindexManifest::try_from(&cfg).unwrap_err();
        assert!(matches!(
            err,
            SpecTranslationError::MissingSafetensorsDigests
        ));
    }

    #[test]
    fn succeeds_with_full_v1_provenance() {
        let cfg = cfg_with_checksums(v1_source());
        let manifest = VindexManifest::try_from(&cfg).expect("complete provenance must translate");
        assert_eq!(
            manifest.vindex_spec_version,
            larql_vindex_spec::VINDEX_SPEC_VERSION
        );
        assert_eq!(manifest.model, "google/gemma-3-4b-it");
        assert_eq!(manifest.source.huggingface_repo, "google/gemma-3-4b-it");
        assert_eq!(manifest.source.huggingface_revision, "main");
        assert_eq!(manifest.source.base_model_sha, "1adbacd6b6dee75c");
        assert_eq!(manifest.source.extractor_sha, "9f3a2c");
        assert_eq!(manifest.source.base_safetensors_sha256.len(), 2);
        manifest
            .validate_self_consistency()
            .expect("translated manifest must self-validate");
    }

    #[test]
    fn translated_manifest_round_trips_through_json() {
        let cfg = cfg_with_checksums(v1_source());
        let manifest = VindexManifest::try_from(&cfg).unwrap();
        let json = serde_json::to_string(&manifest).unwrap();
        let back: VindexManifest = serde_json::from_str(&json).unwrap();
        back.validate_self_consistency().unwrap();
        assert_eq!(back.source.base_model_sha, "1adbacd6b6dee75c");
    }

    #[test]
    fn extract_level_round_trips() {
        for (cfg, spec) in [
            (ExtractLevel::Browse, SpecExtractLevel::Browse),
            (ExtractLevel::Attention, SpecExtractLevel::Attention),
            (ExtractLevel::Inference, SpecExtractLevel::Inference),
            (ExtractLevel::All, SpecExtractLevel::All),
        ] {
            assert_eq!(translate_extract_level(cfg), spec);
        }
    }

    #[test]
    fn quant_format_round_trips() {
        assert_eq!(translate_quant(QuantFormat::None), SpecQuantFormat::None);
        assert_eq!(translate_quant(QuantFormat::Q4K), SpecQuantFormat::Q4K);
    }

    #[test]
    fn dtype_round_trips() {
        assert_eq!(translate_dtype(StorageDtype::F32), SpecStorageDtype::F32);
        assert_eq!(translate_dtype(StorageDtype::F16), SpecStorageDtype::F16);
    }

    #[test]
    fn layer_translation_preserves_moe_fields() {
        let layer = VindexLayerInfo {
            layer: 0,
            num_features: 10240,
            offset: 0,
            length: 52_428_800,
            num_experts: Some(8),
            num_features_per_expert: Some(1280),
        };
        let spec_layer = translate_layer(&layer);
        assert_eq!(spec_layer.layer, 0);
        assert_eq!(spec_layer.num_features, 10240);
        assert_eq!(spec_layer.offset, Some(0));
        assert_eq!(spec_layer.length, Some(52_428_800));
        assert_eq!(spec_layer.num_experts, Some(8));
        assert_eq!(spec_layer.num_features_per_expert, Some(1280));
        assert!(spec_layer.shards.is_none());
    }
}
