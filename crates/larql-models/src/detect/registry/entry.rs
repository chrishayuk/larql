//! One recognised architecture family: how it's matched, and what it
//! supports.

use larql_vindex_spec::QuantFormat;

use super::attention::AttentionKind;
use super::pattern::ModelTypeMatch;

/// Quant formats a standard (non-MLA) architecture's extractor path
/// supports today.
pub(super) const STANDARD_QUANT_FORMATS: &[QuantFormat] = &[QuantFormat::None, QuantFormat::Q4K];

/// Quant formats an MLA architecture's extractor path supports today —
/// the Q4K weight writer hard-rejects MLA
/// (`larql-vindex`'s `write_model_weights_kquant_with_opts`).
pub(super) const MLA_QUANT_FORMATS: &[QuantFormat] = &[QuantFormat::None];

/// Quant formats a pure-SSM architecture's extractor path supports today:
/// unquantised only — no writer has ever quantised an SSM operand estate,
/// and claiming a format here would assert an untested path.
pub(super) const SSM_QUANT_FORMATS: &[QuantFormat] = &[QuantFormat::None];

/// One recognised architecture family: how its `model_type` is matched,
/// and what it supports.
#[derive(Clone, Copy, Debug)]
pub struct ArchitectureEntry {
    /// Representative `model_type` label for reporting. Not necessarily
    /// what [`crate::config::ModelArchitecture::family`] returns for
    /// every config this entry matches — a few architectures (Qwen,
    /// Granite) echo the config's own `model_type` back from `family()`
    /// rather than normalising to one label, since their families cover
    /// several distinct upstream `model_type` strings.
    pub model_type: &'static str,
    /// Pattern(s) this entry matches. The first
    /// [`super::ARCHITECTURE_REGISTRY`] entry whose pattern(s) match
    /// wins — same first-match-wins order as `detect_from_json`'s
    /// `match` arms, which the table mirrors.
    pub patterns: &'static [ModelTypeMatch],
    /// Attention mechanism this family uses.
    pub attention_kind: AttentionKind,
    /// Quant formats the extractor supports for this family today.
    pub quant_formats: &'static [QuantFormat],
}

impl ArchitectureEntry {
    pub(super) fn matches(&self, model_type: &str) -> bool {
        self.patterns.iter().any(|p| p.matches(model_type))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> ArchitectureEntry {
        ArchitectureEntry {
            model_type: "gemma3",
            patterns: &[ModelTypeMatch::Prefix("gemma3")],
            attention_kind: AttentionKind::Standard,
            quant_formats: STANDARD_QUANT_FORMATS,
        }
    }

    #[test]
    fn matches_delegates_to_its_patterns() {
        let entry = sample();
        assert!(entry.matches("gemma3_text"));
        assert!(!entry.matches("gemma2"));
    }

    #[test]
    fn quant_format_consts_are_distinct() {
        assert_ne!(STANDARD_QUANT_FORMATS, MLA_QUANT_FORMATS);
        assert!(STANDARD_QUANT_FORMATS.contains(&QuantFormat::Q4K));
        assert!(!MLA_QUANT_FORMATS.contains(&QuantFormat::Q4K));
    }
}
