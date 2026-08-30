//! Attention mechanism family, for capability reporting.

use serde::Serialize;

/// Attention mechanism family. Not consumed by detection itself — each
/// [`crate::config::ModelArchitecture`] impl already knows its own
/// attention shape via [`crate::config::ModelArchitecture::uses_mla`].
/// This exists purely for capability reporting: a downstream consumer
/// deciding whether it can serve/quantise a model needs to know MLA
/// changes the KV-cache and quant story before it loads anything.
/// `Serialize` (lowercase tags, matching `larql-vindex-spec`'s enums)
/// so a capability manifest can emit it directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AttentionKind {
    /// Standard multi-head / grouped-query attention.
    Standard,
    /// Multi-head Latent Attention (compressed KV).
    Mla,
    /// A recurrence in place of attention (pure SSM): no KV cache exists
    /// at all, which changes the serving story more than MLA's
    /// compression does.
    Recurrent,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variants_are_distinct() {
        assert_ne!(AttentionKind::Standard, AttentionKind::Mla);
    }

    #[test]
    fn serialises_lowercase() {
        assert_eq!(
            serde_json::to_string(&AttentionKind::Standard).unwrap(),
            "\"standard\""
        );
        assert_eq!(
            serde_json::to_string(&AttentionKind::Mla).unwrap(),
            "\"mla\""
        );
    }
}
