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
    /// Multi-axis rotary ("M-RoPE", Qwen-VL family): the rotary frequency
    /// slots are divided between three position axes — time, height,
    /// width — so one head can carry a token's position in a 3-D grid.
    ///
    /// Its own variant rather than [`Self::PartialRope`] plus metadata
    /// held elsewhere, for the reason [`Self::Yarn`] is its own variant:
    /// these four facts *jointly* define the positional operator. Split
    /// them and it becomes possible for the graph to say "partial rotary"
    /// while silently dropping the multi-axis transformation — which is
    /// precisely the state this variant was introduced to end, where
    /// Qwen3.8 resolved to `Rope { theta }` and would have rotated all
    /// 256 head dims instead of 64.
    ///
    /// **On the text path this operator is exactly degenerate.** When
    /// `t == h == w` — every text-only position — the axis assignment
    /// selects between identical values, so the result is bit-identical
    /// to a plain partial rotary (measured against HF: max abs difference
    /// `0.0`). No text-only test can falsify [`Self::section`] or
    /// [`Self::interleaved`]; both are carried and lowered on the
    /// strength of the declaration and the image path's eventual
    /// execution, never on text parity.
    MRope {
        theta: f64,
        /// Fraction of each head that rotates: `0.25` on Qwen3.8, which
        /// is 64 dims of a 256-dim head.
        rotary_fraction: f64,
        basis: RotaryFrequencyBasis,
        /// Frequency slots per axis, in `(t, h, w)` order.
        ///
        /// Sums to the FREQUENCY count — `rotary_dim / 2` — and **not**
        /// to `rotary_dim`. `[11, 11, 10]` on Qwen3.8: 32 frequencies
        /// over a 64-dim rotary block. Reading the sum as `rotary_dim`
        /// closes only if the head width is taken as 128, which is the
        /// *Gated DeltaNet* head width and a different operator.
        ///
        /// An array, not a list: HF expands `inv_freq` to `(3, …)` and
        /// iterates H and W explicitly, so three axes is the operator's
        /// arity rather than a configurable length.
        section: [usize; 3],
        /// Whether the axes interleave across the frequency slots
        /// (`THWTHW…`, HF's `apply_interleaved_mrope`) or occupy
        /// contiguous blocks (`TTT…HHH…WWW…`).
        interleaved: bool,
    },
    /// A **relative** position scheme: position enters through a learned
    /// relative term of width `d_rel` over a bounded `extent`, not through
    /// any rotation of Q/K.
    ///
    /// Its own variant for the reason [`Self::Yarn`] and [`Self::MRope`]
    /// are: a consumer that only knows rotary would otherwise be handed a
    /// `theta` this checkpoint never declared and rotate a model that does
    /// not rotate. Inkling-Small is the case — `d_rel: 16`,
    /// `rel_extent: 1024`, and no rope key anywhere — and before this
    /// variant it resolved to `Rope { theta }` at the parser's default
    /// base, on all 42 layers.
    ///
    /// **Declared, not executable.** No backend consumes this yet; the
    /// variant exists so the graph can say what the checkpoint said
    /// instead of inventing a rotation, and so execution refuses
    /// explicitly rather than running the wrong operator.
    Relative { d_rel: usize, extent: usize },
    /// No positional encoding — the layer attends position-agnostically.
    None,
}

/// Axis index (0 = t, 1 = h, 2 = w) each frequency slot draws its
/// position from, under an M-RoPE `section`/`interleaved` policy.
///
/// Transcribed from HF `apply_interleaved_mrope`, which starts from the
/// T-axis frequencies and overwrites `slice(1, section[1] * 3, 3)` with H
/// and `slice(2, section[2] * 3, 3)` with W. Expressing it as a per-slot
/// axis table rather than a tensor permutation is what lets the executor
/// run the real assignment on 1-D positions: the lookup happens, and it
/// happens to select equal values.
///
/// `n_freqs` is `rotary_dim / 2`. Slots beyond what `section` accounts
/// for stay on the T axis, matching HF's "just overwrite the first
/// dimension" construction.
pub fn mrope_axis_table(section: [usize; 3], interleaved: bool, n_freqs: usize) -> Vec<u8> {
    let mut axes = vec![0u8; n_freqs];
    if interleaved {
        for (axis, offset) in [(1usize, 1usize), (2, 2)] {
            let mut slot = offset;
            while slot < section[axis] * 3 && slot < n_freqs {
                axes[slot] = axis as u8;
                slot += 3;
            }
        }
    } else {
        let mut slot = section[0].min(n_freqs);
        for (axis, count) in [(1u8, section[1]), (2, section[2])] {
            for _ in 0..count {
                if slot >= n_freqs {
                    break;
                }
                axes[slot] = axis;
                slot += 1;
            }
        }
    }
    axes
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
            Self::Rope { theta }
            | Self::Yarn { theta, .. }
            | Self::PartialRope { theta, .. }
            | Self::MRope { theta, .. } => Some(theta),
            // A relative scheme has no base: it does not rotate.
            Self::Relative { .. } | Self::None => None,
        }
    }

    /// The rotated fraction of each head when the policy is a partial
    /// rotary; `None` for a full rotary, YaRN and NoPE — "no partial rotary
    /// declared", which is not the same claim as a fraction of 1.0.
    pub fn rotary_fraction(self) -> Option<f64> {
        match self {
            Self::PartialRope {
                rotary_fraction, ..
            }
            | Self::MRope {
                rotary_fraction, ..
            } => Some(rotary_fraction),
            Self::Rope { .. } | Self::Yarn { .. } | Self::Relative { .. } | Self::None => None,
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
            Self::MRope {
                basis: RotaryFrequencyBasis::HeadWidth,
                ..
            } => Some(super::rope_types::ROPE_TYPE_PROPORTIONAL),
            // M-RoPE's own spelling lives in `mrope_section`, not in
            // `rope_type`: Qwen3.8 declares `rope_type: "default"` and
            // carries the multi-axis facts beside it.
            Self::PartialRope {
                basis: RotaryFrequencyBasis::RotaryWidth,
                ..
            }
            | Self::MRope {
                basis: RotaryFrequencyBasis::RotaryWidth,
                ..
            }
            | Self::Rope { .. }
            | Self::Relative { .. }
            | Self::None => None,
        }
    }

    /// The YaRN block when the policy is scaled rotary; `None` for plain
    /// rotary and NoPE alike.
    pub fn yarn(self) -> Option<YarnRopeScaling> {
        match self {
            Self::Yarn { scaling, .. } => Some(scaling),
            Self::Rope { .. }
            | Self::PartialRope { .. }
            | Self::MRope { .. }
            | Self::Relative { .. }
            | Self::None => None,
        }
    }

    /// The multi-axis sectioning when the policy is M-RoPE; `None`
    /// otherwise — "this layer declares no axis split", never an implied
    /// single-axis default.
    pub fn mrope(self) -> Option<([usize; 3], bool)> {
        match self {
            Self::MRope {
                section,
                interleaved,
                ..
            } => Some((section, interleaved)),
            Self::Rope { .. }
            | Self::Yarn { .. }
            | Self::PartialRope { .. }
            | Self::Relative { .. }
            | Self::None => None,
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
