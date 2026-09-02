//! Wire shape of `larql capabilities`'s output — the manifest §15.2
//! describes: "does this larql release understand `model_type` X, and
//! if so what does it support."

use larql_models::detect::{AttentionKind, ModelTypeMatch};
use larql_vindex_spec::QuantFormat;
use serde::Serialize;

/// Every architecture a specific `larql` release recognises, plus what
/// each one supports. This is what a `chuk-vindex-recipes` PR check
/// resolves a recipe's target `model_type` against (once that check is
/// wired up — this crate only produces the manifest today, see the
/// `capabilities` module docs).
#[derive(Debug, Serialize)]
pub struct CapabilityManifest {
    /// The `larql` release that produced this manifest
    /// (`CARGO_PKG_VERSION` of the binary that ran `larql capabilities`).
    pub larql_version: String,
    /// One entry per recognised architecture family.
    pub architectures: Vec<ArchitectureCapability>,
}

/// What one recognised architecture family supports.
#[derive(Debug, Serialize)]
pub struct ArchitectureCapability {
    /// Representative `model_type` label (see
    /// [`larql_models::detect::ArchitectureEntry::model_type`] for the
    /// caveat on what this means for Qwen/Granite).
    pub model_type: String,
    /// How a checkpoint's declared `model_type` is matched against this
    /// entry.
    ///
    /// Without this the manifest cannot answer the question it exists to
    /// answer. `model_type` above is a *representative label*, and half
    /// the registry matches by prefix: a consumer comparing a
    /// checkpoint's declared `gemma3_text`, `qwen3`, `granitemoehybrid`
    /// or `deepseek_v2` against the labels alone concludes "unsupported"
    /// for four families this build fully recognises. The gate this
    /// manifest feeds (docs/vindex-factory.md §15.2) would then reject
    /// recipes that work.
    pub matches: Vec<ModelTypePattern>,
    /// Attention mechanism this family uses.
    pub attention_kind: AttentionKind,
    /// Quant formats the extractor supports for this family today.
    pub quant_formats: Vec<QuantFormat>,
}

/// One way a declared `model_type` may match an architecture entry.
///
/// The wire form of [`ModelTypeMatch`], kept as its own type so the
/// registry does not have to take a serde dependency to be reportable.
#[derive(Debug, Serialize)]
#[serde(tag = "kind", content = "value", rename_all = "lowercase")]
pub enum ModelTypePattern {
    /// The declared `model_type` must equal this string.
    Exact(String),
    /// The declared `model_type` must start with this string.
    Prefix(String),
}

impl From<&ModelTypeMatch> for ModelTypePattern {
    fn from(m: &ModelTypeMatch) -> Self {
        match m {
            ModelTypeMatch::Exact(s) => Self::Exact((*s).to_string()),
            ModelTypeMatch::Prefix(p) => Self::Prefix((*p).to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_serialises_to_the_expected_shape() {
        let manifest = CapabilityManifest {
            larql_version: "0.1.0".into(),
            architectures: vec![ArchitectureCapability {
                model_type: "gemma3".into(),
                matches: vec![ModelTypePattern::Prefix("gemma3".into())],
                attention_kind: AttentionKind::Standard,
                quant_formats: vec![QuantFormat::None, QuantFormat::Q4K],
            }],
        };
        let json = serde_json::to_string(&manifest).unwrap();
        assert!(json.contains("\"larql_version\":\"0.1.0\""));
        assert!(json.contains("\"model_type\":\"gemma3\""));
        assert!(json.contains("\"attention_kind\":\"standard\""));
        assert!(json.contains("\"quant_formats\":[\"none\",\"q4k\"]"));
        assert!(json.contains("\"kind\":\"prefix\""));
        assert!(json.contains("\"value\":\"gemma3\""));
    }
}
