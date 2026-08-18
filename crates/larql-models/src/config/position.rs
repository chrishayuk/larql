//! Per-layer positional-encoding policy.
//!
//! Absence of positional rotation is an **intentional execution property**,
//! not a parameter value. Muse-Glimmer's global layers carry no position
//! encoding at all ("RoPE, local layers only" per the release card), and the
//! checkpoint spells that as `layer_rope_theta[i] == 0` — a sentinel that is
//! only meaningful at the parse boundary. Internally a zero theta must never
//! circulate: `1/0^(i/d)` is degenerate, and a resolver that stores `0.0`
//! where it means "none" has re-invented the magic value this type exists to
//! remove.
//!
//! The sentinel is honoured exactly once, in
//! [`PositionPolicy::from_declared_theta`]; everything downstream matches on
//! the variant.

use super::rope::YarnRopeScaling;
use serde::{Deserialize, Serialize};

/// The HF `layer_rope_theta` sentinel for "no positional encoding on this
/// layer". Consumed at the parse boundary only.
const NOPE_THETA_SENTINEL: f64 = 0.0;

/// How a layer encodes position.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PositionPolicy {
    /// Rotary position embedding at the given base frequency.
    Rope { theta: f64 },
    /// Rotary position embedding at `theta`, with YaRN scaling: a
    /// per-dimension blend of extrapolated and interpolated frequencies
    /// **and** an amplitude on `cos`/`sin` that rescales every logit at
    /// every position (`YarnRopeScaling::attention_amplitude`). Carried as
    /// its own variant because a consumer that only knows `Rope { theta }`
    /// would serve the model at the wrong attention temperature everywhere
    /// — the fact this variant exists to keep from being dropped at the
    /// container boundary (VINDEX3 A-9.0).
    Yarn {
        theta: f64,
        scaling: YarnRopeScaling,
    },
    /// Rotary on the first `rotary_fraction` of each head only, the rest of
    /// the head unrotated. `basis` says which width the inverse frequencies
    /// are taken over: the rotary width (the plain partial rotary of
    /// Phi/GPT-NeoX — HF `default` + `partial_rotary_factor`) or the full
    /// head width (HF `proportional`, Gemma 4's full-attention layers over
    /// `global_head_dim`). The two rotate the same dims at DIFFERENT angles
    /// — `base^(2i/128)` vs `base^(2i/512)` on Gemma 4 — so the basis is
    /// part of the policy, not a detail a consumer may pick.
    PartialRope {
        theta: f64,
        rotary_fraction: f64,
        basis: RotaryFrequencyBasis,
    },
    /// No positional encoding — the layer attends position-agnostically.
    None,
}

/// The width the inverse-frequency series of a partial rotary is taken
/// over. See [`PositionPolicy::PartialRope`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RotaryFrequencyBasis {
    /// `base^(2i / rotary_width)` — the plain partial rotary.
    RotaryWidth,
    /// `base^(2i / head_dim)` — HF `proportional`.
    HeadWidth,
}

impl PositionPolicy {
    /// Interpret one declared per-layer theta, honouring the upstream
    /// zero-as-NoPE sentinel at this boundary and nowhere else.
    pub fn from_declared_theta(theta: f64) -> Self {
        if theta == NOPE_THETA_SENTINEL {
            Self::None
        } else {
            Self::Rope { theta }
        }
    }

    /// Interpret a declared per-layer theta under a checkpoint-wide YaRN
    /// block: the NoPE sentinel still means none; a rotating layer carries
    /// the scaling.
    pub fn from_declared_theta_with_yarn(theta: f64, scaling: Option<YarnRopeScaling>) -> Self {
        match (Self::from_declared_theta(theta), scaling) {
            (Self::Rope { theta }, Some(scaling)) => Self::Yarn { theta, scaling },
            (policy, _) => policy,
        }
    }

    /// The rope base when the policy is rotary (scaled or not); `None` for a
    /// NoPE layer. Callers that need a theta must handle absence — there is
    /// no default.
    pub fn rope_theta(self) -> Option<f64> {
        match self {
            Self::Rope { theta } | Self::Yarn { theta, .. } | Self::PartialRope { theta, .. } => {
                Some(theta)
            }
            Self::None => None,
        }
    }

    /// The rotated fraction of each head when the policy is a partial
    /// rotary; `None` for a full rotary, YaRN and NoPE — "no partial rotary
    /// declared", which is not the same claim as a fraction of 1.0.
    pub fn rotary_fraction(self) -> Option<f64> {
        match self {
            Self::PartialRope {
                rotary_fraction, ..
            } => Some(rotary_fraction),
            Self::Rope { .. } | Self::Yarn { .. } | Self::None => None,
        }
    }

    /// The HF `rope_type` spelling this policy answers to when it is not
    /// the default class: `yarn`, or `proportional` for a head-width-basis
    /// partial rotary. `None` for the default class (plain rotary, plain
    /// partial rotary) and for NoPE.
    pub fn declared_rope_type(self) -> Option<&'static str> {
        match self {
            Self::Yarn { .. } => Some(super::rope_types::ROPE_TYPE_YARN),
            Self::PartialRope {
                basis: RotaryFrequencyBasis::HeadWidth,
                ..
            } => Some(super::rope_types::ROPE_TYPE_PROPORTIONAL),
            Self::PartialRope {
                basis: RotaryFrequencyBasis::RotaryWidth,
                ..
            }
            | Self::Rope { .. }
            | Self::None => None,
        }
    }

    /// The YaRN block when the policy is scaled rotary; `None` for plain
    /// rotary and NoPE alike.
    pub fn yarn(self) -> Option<YarnRopeScaling> {
        match self {
            Self::Yarn { scaling, .. } => Some(scaling),
            Self::Rope { .. } | Self::PartialRope { .. } | Self::None => None,
        }
    }

    /// Whether the layer rotates at all (plain or scaled).
    pub fn is_rotary(self) -> bool {
        !matches!(self, Self::None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_the_nope_sentinel_at_the_boundary() {
        assert_eq!(
            PositionPolicy::from_declared_theta(0.0),
            PositionPolicy::None
        );
        assert_eq!(
            PositionPolicy::from_declared_theta(500000.0),
            PositionPolicy::Rope { theta: 500000.0 }
        );
    }

    #[test]
    fn nope_layers_have_no_theta_to_offer() {
        assert_eq!(PositionPolicy::None.rope_theta(), None);
        assert_eq!(
            PositionPolicy::Rope { theta: 10000.0 }.rope_theta(),
            Some(10000.0)
        );
    }

    #[test]
    fn serialises_tagged() {
        assert_eq!(
            serde_json::to_string(&PositionPolicy::None).unwrap(),
            "{\"kind\":\"none\"}"
        );
        assert_eq!(
            serde_json::to_string(&PositionPolicy::Rope { theta: 500000.0 }).unwrap(),
            "{\"kind\":\"rope\",\"theta\":500000.0}"
        );
    }

    #[test]
    fn round_trips() {
        for policy in [
            PositionPolicy::None,
            PositionPolicy::Rope { theta: 1e6 },
            PositionPolicy::Yarn {
                theta: 150000.0,
                scaling: gpt_oss_yarn(),
            },
        ] {
            let json = serde_json::to_string(&policy).unwrap();
            let back: PositionPolicy = serde_json::from_str(&json).unwrap();
            assert_eq!(back, policy);
        }
    }

    fn gpt_oss_yarn() -> YarnRopeScaling {
        YarnRopeScaling {
            factor: 32.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_max_position_embeddings: 4096.0,
            truncate: false,
            mscale: None,
            mscale_all_dim: None,
        }
    }

    #[test]
    fn a_yarn_block_attaches_only_to_a_rotating_layer() {
        let scaling = gpt_oss_yarn();
        assert_eq!(
            PositionPolicy::from_declared_theta_with_yarn(150000.0, Some(scaling)),
            PositionPolicy::Yarn {
                theta: 150000.0,
                scaling
            }
        );
        // The NoPE sentinel wins over a checkpoint-wide YaRN block.
        assert_eq!(
            PositionPolicy::from_declared_theta_with_yarn(0.0, Some(scaling)),
            PositionPolicy::None
        );
        // No block: plain rotary, exactly as `from_declared_theta`.
        assert_eq!(
            PositionPolicy::from_declared_theta_with_yarn(150000.0, None),
            PositionPolicy::Rope { theta: 150000.0 }
        );
    }

    #[test]
    fn scaled_rotary_still_offers_its_theta_and_only_it_offers_yarn() {
        let scaling = gpt_oss_yarn();
        let yarn = PositionPolicy::Yarn {
            theta: 150000.0,
            scaling,
        };
        assert_eq!(yarn.rope_theta(), Some(150000.0));
        assert_eq!(yarn.yarn(), Some(scaling));
        assert_eq!(PositionPolicy::Rope { theta: 1e4 }.yarn(), None);
        assert_eq!(PositionPolicy::None.yarn(), None);
    }

    fn gemma4_partial() -> PositionPolicy {
        PositionPolicy::PartialRope {
            theta: 1_000_000.0,
            rotary_fraction: 0.25,
            basis: RotaryFrequencyBasis::HeadWidth,
        }
    }

    /// A partial rotary offers its theta and its fraction, answers to
    /// HF's `proportional` spelling only on the head-width basis, and
    /// carries no YaRN block; the plain-partial basis is the default class.
    #[test]
    fn a_partial_rotary_answers_for_its_fraction_and_class() {
        let proportional = gemma4_partial();
        assert_eq!(proportional.rope_theta(), Some(1_000_000.0));
        assert_eq!(proportional.rotary_fraction(), Some(0.25));
        assert_eq!(proportional.declared_rope_type(), Some("proportional"));
        assert_eq!(proportional.yarn(), None);
        assert!(proportional.is_rotary());
        let plain_partial = PositionPolicy::PartialRope {
            theta: 10_000.0,
            rotary_fraction: 0.5,
            basis: RotaryFrequencyBasis::RotaryWidth,
        };
        assert_eq!(plain_partial.declared_rope_type(), None);
        assert_eq!(plain_partial.rotary_fraction(), Some(0.5));
        // Full rotary, YaRN and NoPE declare no fraction; YaRN answers `yarn`.
        assert_eq!(PositionPolicy::Rope { theta: 1e4 }.rotary_fraction(), None);
        assert_eq!(PositionPolicy::None.rotary_fraction(), None);
        assert_eq!(
            PositionPolicy::Rope { theta: 1e4 }.declared_rope_type(),
            None
        );
        assert_eq!(PositionPolicy::None.declared_rope_type(), None);
        let yarn = PositionPolicy::Yarn {
            theta: 150000.0,
            scaling: gpt_oss_yarn(),
        };
        assert_eq!(yarn.declared_rope_type(), Some("yarn"));
        assert_eq!(yarn.rotary_fraction(), None);
    }

    /// The partial rotary round-trips with its basis tagged.
    #[test]
    fn a_partial_rotary_round_trips_with_its_basis() {
        let json = serde_json::to_string(&gemma4_partial()).unwrap();
        assert!(json.contains("\"kind\":\"partial_rope\""), "{json}");
        assert!(json.contains("\"basis\":\"head_width\""), "{json}");
        let back: PositionPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(back, gemma4_partial());
    }

    #[test]
    fn rotary_means_plain_or_scaled_but_not_nope() {
        assert!(PositionPolicy::Rope { theta: 1e4 }.is_rotary());
        assert!(PositionPolicy::Yarn {
            theta: 1e4,
            scaling: gpt_oss_yarn()
        }
        .is_rotary());
        assert!(!PositionPolicy::None.is_rotary());
    }
}
