//! The KDA block's declared geometry, and the spelling used for a layer
//! whose declared interleave could not be read.
//!
//! The interleave itself lives in [`super::interleave`] — one abstraction
//! over every spelling — rather than here, where it was specific to
//! `linear_attn_config`.

use serde::{Deserialize, Serialize};

/// `layer_types` spelling for a layer whose interleave was **declared but
/// could not be resolved**.
///
/// Deliberately outside every recognised vocabulary, so it resolves to no
/// span, fails `matches_declaration`, and blocks. The alternative — leaving
/// the per-layer declaration absent — would let the caller's default answer
/// for a topology the checkpoint actually stated and this build failed to
/// read, which is the silent degradation this module exists to prevent.
pub const LAYER_TYPE_UNRESOLVED_INTERLEAVE: &str = "declared-interleave(unresolved)";

/// The KDA block's declared geometry — the second half of what
/// `linear_attn_config` states, beside the interleave.
///
/// Separate from
/// [`LinearAttentionTopology`](crate::inventory::LinearAttentionTopology),
/// which reads Qwen3.8's `linear_*` spellings and describes Gated DeltaNet.
/// The two operators are not variants of one another: Gated DeltaNet
/// carries a *fused* q|k|v projection, one conv, full-rank gates and a
/// per-**head** `dt_bias`, while KDA carries split projections, three
/// convs, low-rank f/g gates and a per-**channel** `dt_bias`. Keeping the
/// geometries apart is what stops one being read as the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KdaGeometry {
    /// `num_heads` — Hv. 64 on GLM-5.3-Flash, 32 on Kimi Linear.
    pub num_heads: usize,
    /// `head_dim` — Dk = Dv. 128 on both observed checkpoints, but read,
    /// never assumed.
    pub head_dim: usize,
    /// `short_conv_kernel_size` — width of the depthwise causal conv each
    /// of q, k and v passes through independently.
    pub conv_kernel: usize,
}

impl KdaGeometry {
    /// Read the geometry from a `linear_attn_config` section.
    ///
    /// All three or none: a partial declaration is refused rather than
    /// completed with defaults, the same contract
    /// `LinearAttentionTopology::from_config` holds for Gated DeltaNet.
    pub fn read(section: &serde_json::Value) -> Option<Self> {
        let field = |key: &str| section[key].as_u64().map(|v| v as usize).filter(|v| *v > 0);
        Some(Self {
            num_heads: field("num_heads")?,
            head_dim: field("head_dim")?,
            conv_kernel: field("short_conv_kernel_size")?,
        })
    }

    /// Width of the value/gate side, `Hv·Dv` — the row count of each of
    /// the three projections, of each conv, and the length of `dt_bias`.
    ///
    /// This is the number that separates KDA from Gated DeltaNet at a
    /// glance: DeltaNet's `dt_bias` is `[Hv]`, KDA's is `[Hv·Dv]`.
    pub fn value_width(self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Elements in one layer's recurrent state: a `Dk × Dv` matrix per
    /// head.
    pub fn state_elements(self) -> usize {
        self.num_heads * self.head_dim * self.head_dim
    }
}

/// Which decay-gate KDA's recurrence actually computes.
///
/// **Two checkpoints declare `linear_attn_config.gate_lower_bound: -5.0`
/// and do different things with it.** The value alone therefore cannot
/// select the form; the family that owns the reference implementation
/// must state it. That is why this is an enum resolved by
/// [`ModelArchitecture::kda_gate_form`](crate::config::ModelArchitecture::kda_gate_form)
/// and not a bool derived from the config.
///
/// Measured on the real GLM-5.3-Flash layer 0 against the pinned
/// `transformers` reference (`scripts/glm_gate_control.py`): swapping
/// [`Self::Softplus`] in for the shipped [`Self::ClampedSigmoid`] moves
/// the layer output by a relative **2.50e-2** at 8 positions from a zero
/// state, and moves the decay gate itself from a mean of `-0.906` to
/// `-2.528` — a 2.8× faster decay per step, compounding with context
/// length. Every shape closes either way.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "form", rename_all = "snake_case")]
pub enum KdaGateForm {
    /// `g = -exp(A_log) · softplus(f_b(f_a(x)) + dt_bias)`.
    ///
    /// Kimi Linear. Its `modeling_kimi.py` calls
    /// `fused_kda_gate(g, A_log, head_dim, g_bias=dt_bias)`, which selects
    /// the softplus branch, and reads `gate_lower_bound` nowhere — neither
    /// the modeling file nor `configuration_kimi.py` mentions it. The
    /// declared bound is provenance on this family.
    ///
    /// Unbounded below, so the state can decay arbitrarily fast.
    Softplus,
    /// `g = lower_bound · sigmoid(exp(A_log) · (f_b(f_a(x)) + dt_bias))`.
    ///
    /// GLM-5.3-Flash. `Glm5NextTextForgetGate.forward` takes this branch
    /// whenever `config.linear_lower_bound is not None`, and
    /// `Glm5NextTextConfig.__init__` fills that from
    /// `linear_attn_config.gate_lower_bound` (defaulting it to `-5.0` when
    /// `safe_gate` is set and the bound is null). The bound is an
    /// execution input on this family.
    ///
    /// Bounded: `g ∈ [lower_bound, 0]`, so `exp(g) ≥ exp(lower_bound)` —
    /// the state cannot decay faster than that floor.
    ClampedSigmoid {
        /// The declared bound, carried verbatim. Negative on every
        /// observed checkpoint; the sign is the checkpoint's to state.
        lower_bound: f32,
    },
}

impl KdaGateForm {
    /// The bound this form actually applies, if it applies one.
    ///
    /// [`Self::Softplus`] answers `None` even when the checkpoint declared
    /// a bound — the point of the distinction is that the declaration and
    /// the computation can disagree.
    pub fn applied_lower_bound(self) -> Option<f32> {
        match self {
            Self::Softplus => None,
            Self::ClampedSigmoid { lower_bound } => Some(lower_bound),
        }
    }

    /// GLM-5.3-Flash's own selection rule, transcribed from
    /// `Glm5NextTextConfig.__init__`:
    ///
    /// ```text
    /// self.linear_lower_bound = linear_attn_dict.get("gate_lower_bound", self.linear_lower_bound)
    /// if linear_attn_dict.get("safe_gate", True) and self.linear_lower_bound is None:
    ///     self.linear_lower_bound = -5.0
    /// ```
    ///
    /// so a null bound with `safe_gate` still clamps, and only an explicit
    /// `safe_gate: false` with no bound reaches the softplus branch.
    ///
    /// Kept here rather than in the GLM architecture because it is a
    /// property of the *section*, and a second family adopting the same
    /// spelling should get the same reading rather than its own copy.
    pub fn from_glm5_section(section: &serde_json::Value) -> Self {
        let declared = section["gate_lower_bound"].as_f64().map(|v| v as f32);
        let safe_gate = section["safe_gate"].as_bool().unwrap_or(true);
        match (declared, safe_gate) {
            (Some(lower_bound), _) => Self::ClampedSigmoid { lower_bound },
            (None, true) => Self::ClampedSigmoid {
                lower_bound: GLM5_DEFAULT_GATE_LOWER_BOUND,
            },
            (None, false) => Self::Softplus,
        }
    }
}

/// `Glm5NextTextConfig.linear_lower_bound`'s default, applied when the
/// checkpoint declares `safe_gate` but no bound.
pub const GLM5_DEFAULT_GATE_LOWER_BOUND: f32 = -5.0;

#[cfg(test)]
mod tests {
    use super::*;

    fn section(v: serde_json::Value) -> serde_json::Value {
        v
    }

    #[test]
    fn geometry_is_read_whole_or_not_at_all() {
        let full = section(serde_json::json!({
            "num_heads": 64, "head_dim": 128, "short_conv_kernel_size": 4
        }));
        let g = KdaGeometry::read(&full).expect("a complete declaration reads");
        assert_eq!(g.num_heads, 64);
        assert_eq!(g.value_width(), 8192);
        assert_eq!(g.state_elements(), 64 * 128 * 128);

        // A partial declaration is refused rather than completed.
        for missing in ["num_heads", "head_dim", "short_conv_kernel_size"] {
            let mut partial = full.clone();
            partial.as_object_mut().expect("object").remove(missing);
            assert!(
                KdaGeometry::read(&partial).is_none(),
                "a declaration missing `{missing}` must not be completed with a default"
            );
        }
        // Zero is not a width.
        let zeroed = section(serde_json::json!({
            "num_heads": 0, "head_dim": 128, "short_conv_kernel_size": 4
        }));
        assert!(KdaGeometry::read(&zeroed).is_none());
    }

    /// The distinction the enum exists for: a declared bound the family
    /// APPLIES against one it merely carries.
    #[test]
    fn only_the_clamped_form_reports_an_applied_bound() {
        assert_eq!(KdaGateForm::Softplus.applied_lower_bound(), None);
        assert_eq!(
            KdaGateForm::ClampedSigmoid { lower_bound: -5.0 }.applied_lower_bound(),
            Some(-5.0)
        );
    }

    /// `Glm5NextTextConfig.__init__`'s own rule, in all three branches.
    #[test]
    fn the_glm5_selection_rule_covers_its_three_branches() {
        // A declared bound clamps, whatever `safe_gate` says — the
        // reference reads the bound first and only defaults a null one.
        for safe in [
            serde_json::Value::Null,
            serde_json::json!(true),
            serde_json::json!(false),
        ] {
            let mut s = serde_json::json!({ "gate_lower_bound": -3.5 });
            if !safe.is_null() {
                s["safe_gate"] = safe.clone();
            }
            assert_eq!(
                KdaGateForm::from_glm5_section(&s),
                KdaGateForm::ClampedSigmoid { lower_bound: -3.5 },
                "a declared bound is applied regardless of safe_gate={safe}"
            );
        }

        // Absent bound, `safe_gate` absent or true → the class default.
        for s in [
            serde_json::json!({}),
            serde_json::json!({ "safe_gate": true }),
        ] {
            assert_eq!(
                KdaGateForm::from_glm5_section(&s),
                KdaGateForm::ClampedSigmoid {
                    lower_bound: GLM5_DEFAULT_GATE_LOWER_BOUND
                },
            );
        }

        // Only an explicit `safe_gate: false` with no bound reaches
        // softplus.
        assert_eq!(
            KdaGateForm::from_glm5_section(&serde_json::json!({ "safe_gate": false })),
            KdaGateForm::Softplus
        );
    }

    /// The two forms must not compare equal — the whole point is that a
    /// consumer can tell them apart.
    #[test]
    fn the_forms_are_distinguishable_and_round_trip_through_serde() {
        let soft = KdaGateForm::Softplus;
        let clamped = KdaGateForm::ClampedSigmoid { lower_bound: -5.0 };
        assert_ne!(soft, clamped);
        assert_ne!(clamped, KdaGateForm::ClampedSigmoid { lower_bound: -4.0 });

        for form in [soft, clamped] {
            let json = serde_json::to_value(form).expect("serialises");
            let back: KdaGateForm = serde_json::from_value(json.clone()).expect("round-trips");
            assert_eq!(back, form, "{json}");
        }
        // The tag is what a container reader dispatches on.
        assert_eq!(
            serde_json::to_value(KdaGateForm::Softplus).expect("ser")["form"],
            serde_json::json!("softplus")
        );
    }

    #[test]
    fn the_unresolved_interleave_spelling_is_outside_every_vocabulary() {
        for known in ["full_attention", "linear_attention", "sliding_attention"] {
            assert_ne!(LAYER_TYPE_UNRESOLVED_INTERLEAVE, known);
        }
    }
}
