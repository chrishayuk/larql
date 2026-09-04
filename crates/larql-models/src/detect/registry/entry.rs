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
    /// Architectures this family declares as occupying its component
    /// slots — **lineage, never substitutability**.
    ///
    /// A container checkpoint may name one identity at the top and a
    /// different one on a sub-component, and those are not competing
    /// claims about the same level of abstraction. Kimi K3 declares
    /// `kimi_k3` on the container and `kimi_linear` on its text
    /// component; both are true, and reading either alone would decide
    /// which model the runtime serves by accident of traversal.
    ///
    /// # What a declaration here does NOT mean
    ///
    /// ```text
    /// KimiK3 declares text = KimiLinear      lineage           YES
    /// KimiK3 executes as KimiLinear          substitutability  NO
    /// KimiLinear declares KimiK3             symmetry          NO
    /// KimiK3 inherits KimiLinear's semantics capability        NO
    /// ```
    ///
    /// The relation is DIRECTIONAL and confers nothing. It is spelled as
    /// a declaration rather than as a compatibility predicate on purpose:
    /// a name like `is_compatible_with` invites symmetry, symmetry
    /// invites substitution, and a later caller expecting execution
    /// equivalence would get it silently from a relation that only ever
    /// meant "who occupies this slot".
    pub components: &'static [(ComponentRole, &'static str)],
}

/// A slot a container architecture can declare an occupant for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentRole {
    /// The text decoder.
    Text,
}

impl ArchitectureEntry {
    /// The architecture this family declares for `role`, if any.
    ///
    /// Deliberately not `is_compatible_with`: see [`Self::components`].
    pub fn declares_component(&self, role: ComponentRole) -> Option<&'static str> {
        self.components
            .iter()
            .find(|(r, _)| *r == role)
            .map(|(_, model_type)| *model_type)
    }

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
            components: &[],
        }
    }

    #[test]
    fn matches_delegates_to_its_patterns() {
        let entry = sample();
        assert!(entry.matches("gemma3_text"));
        assert!(!entry.matches("gemma2"));
    }

    /// A container that declares its text occupant, as Kimi K3 does.
    fn container() -> ArchitectureEntry {
        ArchitectureEntry {
            model_type: "kimi_k3",
            patterns: &[ModelTypeMatch::Exact("kimi_k3")],
            attention_kind: AttentionKind::Standard,
            quant_formats: STANDARD_QUANT_FORMATS,
            components: &[(ComponentRole::Text, "kimi_linear")],
        }
    }

    /// The declaration is readable, and reads as itself.
    #[test]
    fn a_declared_component_is_returned_for_its_role() {
        assert_eq!(
            container().declares_component(ComponentRole::Text),
            Some("kimi_linear")
        );
    }

    /// An entry that declares nothing answers `None` — the absence is a
    /// real answer, never an empty string or the entry's own type.
    #[test]
    fn an_entry_declaring_nothing_answers_none() {
        assert_eq!(sample().declares_component(ComponentRole::Text), None);
    }

    /// **Directional, and it must be.** The container names its occupant;
    /// the occupant never names the container. Reading this relation
    /// backwards is how lineage would turn into substitutability.
    #[test]
    fn the_relation_is_not_symmetric() {
        let occupant = ArchitectureEntry {
            model_type: "kimi_linear",
            ..container()
        };
        let occupant = ArchitectureEntry {
            components: &[],
            ..occupant
        };
        assert_eq!(
            container().declares_component(ComponentRole::Text),
            Some("kimi_linear")
        );
        assert_eq!(occupant.declares_component(ComponentRole::Text), None);
    }

    #[test]
    fn quant_format_consts_are_distinct() {
        assert_ne!(STANDARD_QUANT_FORMATS, MLA_QUANT_FORMATS);
        assert!(STANDARD_QUANT_FORMATS.contains(&QuantFormat::Q4K));
        assert!(!MLA_QUANT_FORMATS.contains(&QuantFormat::Q4K));
    }
}
