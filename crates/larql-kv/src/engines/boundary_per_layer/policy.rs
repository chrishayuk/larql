//! `BoundaryLayerPolicy` — per-layer codec assignment.
//!
//! A policy is a `Vec<ColdResidualCodec>` of length `num_layers`. It carries
//! its model fingerprint so a calibration record can be looked up against
//! it without ambiguity (see [`super::calibration`]).
//!
//! v0.1 enforces that every entry is [`ColdResidualCodec::Bf16`]. Future
//! versions will lift the restriction once additional codecs gain per-layer
//! calibration support.

use crate::engines::markov_residual_codec::codec::ColdResidualCodec;

/// Errors a [`BoundaryLayerPolicy`] may surface during construction.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PolicyError {
    #[error("policy must specify a codec for every layer (have {got}, need {expected})")]
    LayerCountMismatch { got: usize, expected: usize },
    #[error("layer {layer} uses unsupported codec {codec:?}; v0.1 supports only Bf16")]
    UnsupportedCodec {
        layer: usize,
        codec: ColdResidualCodec,
    },
}

/// Per-layer codec policy keyed to a model fingerprint.
#[derive(Debug, Clone)]
pub struct BoundaryLayerPolicy {
    pub model_revision: String,
    pub entries: Vec<ColdResidualCodec>,
}

impl BoundaryLayerPolicy {
    /// Build a policy with explicit per-layer entries. Validates length and
    /// codec support; v0.1 supports only `Bf16` per layer.
    pub fn new(
        model_revision: impl Into<String>,
        num_layers: usize,
        entries: Vec<ColdResidualCodec>,
    ) -> Result<Self, PolicyError> {
        if entries.len() != num_layers {
            return Err(PolicyError::LayerCountMismatch {
                got: entries.len(),
                expected: num_layers,
            });
        }
        for (layer, codec) in entries.iter().enumerate() {
            if !v0_1_supported(*codec) {
                return Err(PolicyError::UnsupportedCodec {
                    layer,
                    codec: *codec,
                });
            }
        }
        Ok(Self {
            model_revision: model_revision.into(),
            entries,
        })
    }

    /// Convenience: build a policy that assigns `Bf16` to every layer.
    pub fn bf16_uniform(model_revision: impl Into<String>, num_layers: usize) -> Self {
        // Bf16-everywhere is always v0.1-supported; this never returns Err.
        Self::new(
            model_revision,
            num_layers,
            vec![ColdResidualCodec::Bf16; num_layers],
        )
        .expect("bf16_uniform must succeed")
    }

    pub fn num_layers(&self) -> usize {
        self.entries.len()
    }

    pub fn codec_for(&self, layer: usize) -> ColdResidualCodec {
        self.entries[layer]
    }

    /// Stable identity for calibration lookup. v0.1 uses the model revision
    /// plus a digest of the entry sequence; the exact format does not need
    /// to be cryptographically strong because the store also checks
    /// `model_revision` equality.
    pub fn fingerprint(&self) -> String {
        let mut digest = self.model_revision.clone();
        digest.push('|');
        for (i, c) in self.entries.iter().enumerate() {
            digest.push_str(&format!("{i}={}", c.label()));
            if i + 1 < self.entries.len() {
                digest.push(',');
            }
        }
        digest
    }
}

fn v0_1_supported(codec: ColdResidualCodec) -> bool {
    matches!(codec, ColdResidualCodec::Bf16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_uniform_returns_correct_layer_count() {
        let p = BoundaryLayerPolicy::bf16_uniform("rev-1", 4);
        assert_eq!(p.num_layers(), 4);
        for l in 0..4 {
            assert_eq!(p.codec_for(l), ColdResidualCodec::Bf16);
        }
    }

    #[test]
    fn explicit_entries_validate_length() {
        let err = BoundaryLayerPolicy::new("rev", 3, vec![ColdResidualCodec::Bf16; 2]).unwrap_err();
        assert_eq!(
            err,
            PolicyError::LayerCountMismatch {
                got: 2,
                expected: 3,
            }
        );
    }

    #[test]
    fn fingerprint_is_stable_across_calls() {
        let p1 = BoundaryLayerPolicy::bf16_uniform("rev-1", 2);
        let p2 = BoundaryLayerPolicy::bf16_uniform("rev-1", 2);
        assert_eq!(p1.fingerprint(), p2.fingerprint());
    }

    #[test]
    fn fingerprint_differs_on_revision_change() {
        let p1 = BoundaryLayerPolicy::bf16_uniform("rev-1", 2);
        let p2 = BoundaryLayerPolicy::bf16_uniform("rev-2", 2);
        assert_ne!(p1.fingerprint(), p2.fingerprint());
    }

    #[test]
    fn fingerprint_differs_on_layer_count_change() {
        let p1 = BoundaryLayerPolicy::bf16_uniform("rev", 2);
        let p2 = BoundaryLayerPolicy::bf16_uniform("rev", 3);
        assert_ne!(p1.fingerprint(), p2.fingerprint());
    }

    #[test]
    fn policy_error_display_is_informative() {
        let err = PolicyError::LayerCountMismatch {
            got: 1,
            expected: 4,
        };
        let s = err.to_string();
        assert!(s.contains("1"));
        assert!(s.contains("4"));
    }
}
