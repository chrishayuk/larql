//! Quantisation surface — per-tensor format tags, precision tier,
//! projection-format manifest, and the FP4/FP8 (exp 26) config.
//!
//! Carved out of the monolithic `config/types.rs` in the 2026-04-25
//! round-2 cleanup. `Fp4Config` carries a `ComplianceGate` (defined
//! in the sibling `compliance` module).

use larql_models::quant::ggml::K_QUANT_BLOCK_ELEMS;
use serde::{Deserialize, Serialize};

use crate::format::filenames::{DOWN_FEATURES_FP8_BIN, GATE_VECTORS_FP4_BIN, UP_FEATURES_FP4_BIN};

use super::compliance::ComplianceGate;

/// Per-vindex quant scheme. Mirrors the `quant` enum in the v1 wire
/// schema (`crates/larql-vindex-spec/schema/vindex-v1.schema.json`):
/// readers accept `"none"`, `"q4k"`, and `"kquant"`; writers continue
/// to emit `"q4k"` for v1 vindexes so existing tooling keeps round-
/// tripping. A future v2 schema flips the canonical tag to `"kquant"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum QuantFormat {
    #[default]
    None,
    /// Q4_K / Q6_K family. Serialises as `"q4k"`. Accepts the alias
    /// `"kquant"` on Deserialize so vindexes written under the new
    /// canonical tag still load.
    #[serde(alias = "kquant")]
    Q4K,
}

impl std::fmt::Display for QuantFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Q4K => write!(f, "q4k"),
        }
    }
}

/// Per-projection storage precision tag for FP4 vindexes.
///
/// Legal values for `Fp4Config.projections.{gate,up,down}.precision`.
/// Readers MUST dispatch on this tag and MUST NOT sniff filenames.
/// Unrecognised values should produce an explicit error rather than
/// silently downgrade — future tags (e.g. `fp6`, `nf4`) will require
/// a code-path addition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    /// FP4 E2M1 values + FP8 E4M3 sub-block scales + FP8 E4M3 block scale.
    Fp4,
    /// FP8 E4M3 values + FP8 E4M3 block scale. No sub-block scales.
    Fp8,
    /// Legacy IEEE half-precision. Uses the non-suffixed filename.
    F16,
    /// Legacy f32. Uses the non-suffixed filename.
    F32,
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fp4 => write!(f, "fp4"),
            Self::Fp8 => write!(f, "fp8"),
            Self::F16 => write!(f, "f16"),
            Self::F32 => write!(f, "f32"),
        }
    }
}

/// One projection's storage descriptor in the FP4 manifest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectionFormat {
    pub precision: Precision,
    /// Filename relative to the vindex directory. Readers open this
    /// file directly. Must be the legacy name (e.g. `gate_vectors.bin`)
    /// when `precision` is `f16`/`f32`, and the suffixed name (e.g.
    /// `gate_vectors_fp4.bin`) when `precision` is `fp4`/`fp8`.
    pub file: String,
}

/// The three FFN projection tags covered by FP4 storage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Projections {
    pub gate: ProjectionFormat,
    pub up: ProjectionFormat,
    pub down: ProjectionFormat,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fp4Config {
    pub fp4_format_version: u32,
    /// Elements per FP4/FP8 block. v1 pins this at 256 (the largest
    /// size that divides every model family LARQL currently ships).
    pub block_elements: u32,
    /// Elements per sub-block. v1 pins this at 32 (matches OCP MXFP4).
    pub sub_block_elements: u32,
    /// Scale dtype for the 8 per-sub-block scales inside each FP4 block.
    /// v1: `"fp8_e4m3"`.
    pub sub_block_scale_dtype: String,
    /// Scale dtype for the per-block scale (both FP4 and FP8 blocks).
    /// v1: `"fp8_e4m3"`.
    pub block_scale_dtype: String,
    /// Encoding identifier for the FP4 4-bit values themselves.
    /// v1: `"fp4_e2m1_mxfp4_nibble_order"`.
    pub value_encoding: String,
    /// Per-projection precision + filename.
    pub projections: Projections,
    /// Compliance policy applied by the extractor.
    pub compliance_gate: ComplianceGate,
    /// Filename of the compliance sidecar (relative to vindex dir).
    /// v1 default: `"fp4_compliance.json"`.
    pub compliance_report: String,
}

impl Fp4Config {
    /// The v1 default: 256-element blocks, 32-element sub-blocks,
    /// FP4 E2M1 values with FP8 E4M3 two-level scales, MXFP4 nibble order.
    /// `projections` is filled by the caller.
    pub fn v1_defaults(projections: Projections) -> Self {
        Self {
            fp4_format_version: 1,
            block_elements: K_QUANT_BLOCK_ELEMS as u32,
            sub_block_elements: 32,
            sub_block_scale_dtype: "fp8_e4m3".into(),
            block_scale_dtype: "fp8_e4m3".into(),
            value_encoding: "fp4_e2m1_mxfp4_nibble_order".into(),
            projections,
            compliance_gate: ComplianceGate {
                threshold_ratio: 16.0,
                min_compliant_fraction: 0.99,
                fallback_precision: Precision::Fp8,
            },
            compliance_report: "fp4_compliance.json".into(),
        }
    }

    /// Option B default: FP4 gate + FP4 up + FP8 down.
    pub fn option_b_default() -> Self {
        Self::v1_defaults(Projections {
            gate: ProjectionFormat {
                precision: Precision::Fp4,
                file: GATE_VECTORS_FP4_BIN.into(),
            },
            up: ProjectionFormat {
                precision: Precision::Fp4,
                file: UP_FEATURES_FP4_BIN.into(),
            },
            down: ProjectionFormat {
                precision: Precision::Fp8,
                file: DOWN_FEATURES_FP8_BIN.into(),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_format_default_is_none() {
        assert_eq!(QuantFormat::default(), QuantFormat::None);
    }

    #[test]
    fn quant_format_display() {
        assert_eq!(QuantFormat::None.to_string(), "none");
        assert_eq!(QuantFormat::Q4K.to_string(), "q4k");
    }

    #[test]
    fn quant_format_serde_round_trip() {
        let j = serde_json::to_string(&QuantFormat::Q4K).unwrap();
        let back: QuantFormat = serde_json::from_str(&j).unwrap();
        assert_eq!(back, QuantFormat::Q4K);
    }

    #[test]
    fn quant_format_serializes_as_q4k_not_kquant() {
        // v1 wire-format contract: writers MUST emit `"q4k"` for the
        // k-quant variant so vindexes published by post-rename binaries
        // still load in pre-rename readers that don't know `"kquant"`.
        // Flip this assertion only when the spec bumps to v2.
        assert_eq!(serde_json::to_string(&QuantFormat::Q4K).unwrap(), "\"q4k\"");
    }

    #[test]
    fn quant_format_deserialize_accepts_kquant_alias() {
        // Forward-compat: a vindex written under the future v2 canonical
        // tag `"kquant"` loads in v1 readers as the same `Q4K` variant.
        let back: QuantFormat = serde_json::from_str("\"kquant\"").unwrap();
        assert_eq!(back, QuantFormat::Q4K);
    }

    #[test]
    fn precision_display_all_variants() {
        assert_eq!(Precision::Fp4.to_string(), "fp4");
        assert_eq!(Precision::Fp8.to_string(), "fp8");
        assert_eq!(Precision::F16.to_string(), "f16");
        assert_eq!(Precision::F32.to_string(), "f32");
    }

    #[test]
    fn precision_serde_snake_case() {
        let j = serde_json::to_string(&Precision::Fp4).unwrap();
        assert_eq!(j, "\"fp4\"");
        let back: Precision = serde_json::from_str(&j).unwrap();
        assert_eq!(back, Precision::Fp4);
    }

    #[test]
    fn fp4_config_v1_defaults_block_geometry() {
        let cfg = Fp4Config::v1_defaults(Fp4Config::option_b_default().projections);
        assert_eq!(cfg.fp4_format_version, 1);
        assert_eq!(cfg.block_elements, 256);
        assert_eq!(cfg.sub_block_elements, 32);
        assert_eq!(cfg.sub_block_scale_dtype, "fp8_e4m3");
        assert_eq!(cfg.block_scale_dtype, "fp8_e4m3");
        assert_eq!(cfg.value_encoding, "fp4_e2m1_mxfp4_nibble_order");
    }

    #[test]
    fn fp4_config_option_b_projection_precisions() {
        let cfg = Fp4Config::option_b_default();
        assert_eq!(cfg.projections.gate.precision, Precision::Fp4);
        assert_eq!(cfg.projections.up.precision, Precision::Fp4);
        assert_eq!(cfg.projections.down.precision, Precision::Fp8);
    }

    #[test]
    fn fp4_config_compliance_gate_defaults() {
        let cfg = Fp4Config::option_b_default();
        assert_eq!(cfg.compliance_gate.fallback_precision, Precision::Fp8);
        assert!(cfg.compliance_gate.min_compliant_fraction > 0.0);
    }

    #[test]
    fn fp4_config_compliance_report_filename() {
        let cfg = Fp4Config::option_b_default();
        assert_eq!(cfg.compliance_report, "fp4_compliance.json");
    }
}
