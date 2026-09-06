//! FFN activation functions and the gated/standard FFN shape.

use serde::{Deserialize, Serialize};

use super::ExpertGatePolicy;

/// The HF `hidden_act` spelling that names a gate POLICY rather than a
/// bare nonlinearity: Kimi-K3's SiTU-GLU, registered by the checkpoint's
/// own module as `ACT2FN["situ"] = SituAndMul`.
///
/// One constant, read by [`ActivationDeclaration::judge`] and by
/// [`hf_combine_name`], so the name a config declares and the name a plan
/// resolves cannot be spelled by two different string literals.
pub const SITU_NAME: &str = "situ";

/// Activation function used in the FFN.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Activation {
    /// SiLU / Swish (Gemma, Llama)
    Silu,
    /// GELU (GPT-2, BERT)
    Gelu,
    /// GELU with tanh approximation
    GeluTanh,
    /// ReLU
    Relu,
}

/// HF `hidden_act` / `hidden_activation` spellings, one row per variant.
/// The single definition of the name↔variant mapping — parsers and
/// inventory classification both read this table.
const HF_ACTIVATION_NAMES: &[(&str, Activation)] = &[
    ("silu", Activation::Silu),
    ("swish", Activation::Silu),
    ("gelu", Activation::Gelu),
    ("gelu_new", Activation::GeluTanh),
    ("gelu_pytorch_tanh", Activation::GeluTanh),
    ("relu", Activation::Relu),
];

impl Activation {
    /// Map an HF activation name to a variant. `None` for a spelling this
    /// build has never judged — callers must not guess a default for an
    /// unrecognised name.
    pub fn from_hf_name(name: &str) -> Option<Self> {
        HF_ACTIVATION_NAMES
            .iter()
            .find(|(hf_name, _)| name.eq_ignore_ascii_case(hf_name))
            .map(|&(_, activation)| activation)
    }

    /// Which of the two implemented gate/up FFN kernel families this
    /// activation dispatches to on the CPU walk / kquant paths:
    /// `true` = gelu-tanh, `false` = SiLU.
    ///
    /// This is the ONE definition of that mapping — the 2026-07-30
    /// vindex/walk-FFN review (§4) found it copy-pasted across eight
    /// walk backends, where a new `Activation` variant would silently
    /// land in the SiLU arm. The match is deliberately exhaustive (no
    /// wildcard): adding a variant fails compilation here instead.
    ///
    /// - [`Activation::Gelu`] (exact GELU) is served by the tanh
    ///   approximation — a deliberate, documented approximation on
    ///   these paths (no exact-GELU kernel exists; no in-tree
    ///   architecture currently returns `Gelu`).
    /// - [`Activation::Relu`] has NO gate/up kernel; it panics loudly
    ///   rather than silently computing SiLU numerics. No in-tree
    ///   architecture returns `Relu`.
    pub fn uses_gelu_tanh_gate_up(self) -> bool {
        match self {
            Activation::GeluTanh | Activation::Gelu => true,
            Activation::Silu => false,
            Activation::Relu => panic!(
                "Activation::Relu has no gate/up FFN kernel on the walk/kquant paths \
                 (only gelu-tanh and SiLU are implemented)"
            ),
        }
    }
}

/// HF spellings that name the FFN SHAPE together with its nonlinearity —
/// Falcon's `activation: "swiglu"` is "gated, SiLU on the gate" in one
/// word. One row per gated variant; the single definition, beside
/// [`HF_ACTIVATION_NAMES`], so a probe and a parser cannot disagree about
/// what `geglu` means.
const HF_GLU_NAMES: &[(&str, Activation)] = &[
    ("swiglu", Activation::Silu),
    ("geglu", Activation::Gelu),
    ("reglu", Activation::Relu),
];

impl Activation {
    /// The canonical HF spelling of this variant — the first row of
    /// [`HF_ACTIVATION_NAMES`] that names it. Every variant has one;
    /// `activation_names_round_trip` pins that.
    pub fn hf_name(self) -> Option<&'static str> {
        HF_ACTIVATION_NAMES
            .iter()
            .find(|&&(_, activation)| activation == self)
            .map(|&(name, _)| name)
    }

    /// The gated-FFN spelling of this nonlinearity, if HF has one.
    pub fn hf_glu_name(self) -> Option<&'static str> {
        HF_GLU_NAMES
            .iter()
            .find(|&&(_, activation)| activation == self)
            .map(|&(name, _)| name)
    }
}

/// The FFN shape an HF `activation` spelling names: a GLU name is the
/// gated shape with that nonlinearity on the gate, a plain nonlinearity
/// name is the ungated shape. `None` for a spelling this build has never
/// judged — callers must not guess.
pub fn ffn_shape_from_hf_name(name: &str) -> Option<(FfnType, Activation)> {
    HF_GLU_NAMES
        .iter()
        .find(|(glu, _)| name.eq_ignore_ascii_case(glu))
        .map(|&(_, activation)| (FfnType::Gated, activation))
        .or_else(|| {
            Activation::from_hf_name(name).map(|activation| (FfnType::Standard, activation))
        })
}

/// The HF spelling of an FFN shape — the inverse of
/// [`ffn_shape_from_hf_name`], from the same two tables. A gated
/// nonlinearity HF has no GLU word for (gelu-tanh) is spelled
/// `gated-<name>` so the answer is still distinguishable from the ungated
/// shape.
pub fn ffn_shape_hf_name(ffn_type: FfnType, activation: Activation) -> Option<String> {
    let name = activation.hf_name()?;
    Some(match ffn_type {
        FfnType::Gated => activation
            .hf_glu_name()
            .map_or_else(|| format!("gated-{name}"), str::to_string),
        FfnType::Standard => name.to_string(),
    })
}

/// What a checkpoint's `hidden_act` / `hidden_activation` declaration
/// actually says — the distinction the FFN's execution path turns on.
///
/// Before this existed, [`ModelArchitecture::activation`] read
/// `.and_then(Activation::from_hf_name).unwrap_or(Activation::Silu)`,
/// which collapses two different states into one branch: *the config is
/// silent* and *the config declared something this build has never
/// judged*. Two checkpoints in the conformance estate sit in the second
/// state — Kimi-K3's `situ` and BitNet's `relu2` — and both were told
/// they compute SiLU. That is the failure this enum exists to make
/// unrepresentable:
///
/// ```text
/// checkpoint silent                    -> default permitted
/// checkpoint declares a known thing    -> execute that thing
/// checkpoint declares an unknown thing -> refuse, or report unknown
/// checkpoint declares an unknown thing -> silently execute a default   NEVER
/// ```
///
/// [`ModelArchitecture::activation`]: super::ModelArchitecture::activation
#[derive(Debug, Clone, PartialEq)]
pub enum ActivationDeclaration {
    /// No `hidden_act` / `hidden_activation` key. The family default
    /// answers, and that is a checked default rather than an assumed one:
    /// the checkpoint asked no question.
    Absent,
    /// A name this build has judged, and the variant it names.
    Nonlinearity(Activation),
    /// A name that names a gate POLICY — the whole combine, not just the
    /// nonlinearity on the gate branch. Carries the declared spelling, so
    /// a caller reports the checkpoint's own word.
    NamesGatePolicy(&'static str),
    /// A name this build has never judged, carried verbatim. Execution
    /// paths must refuse on this; the planner already reports it (the
    /// leaf grades `mismatched` because the probe and the declaration
    /// disagree).
    Unjudged(String),
}

impl ActivationDeclaration {
    /// Judge one `hidden_act` declaration. The single place the four
    /// states are told apart.
    pub fn judge(hidden_act: Option<&str>) -> Self {
        match hidden_act {
            None => Self::Absent,
            Some(name) if name.eq_ignore_ascii_case(SITU_NAME) => Self::NamesGatePolicy(SITU_NAME),
            Some(name) => match Activation::from_hf_name(name) {
                Some(activation) => Self::Nonlinearity(activation),
                None => Self::Unjudged(name.to_string()),
            },
        }
    }

    /// The declared name, for a refusal message. `None` when the config
    /// was silent — there is no name to quote.
    pub fn declared_name(&self) -> Option<&str> {
        match self {
            Self::Absent => None,
            Self::Nonlinearity(activation) => activation.hf_name(),
            Self::NamesGatePolicy(name) => Some(name),
            Self::Unjudged(name) => Some(name),
        }
    }
}

/// The HF spelling of the combine a component actually computes: the gate
/// policy's own name when the policy is not plain gating, the
/// nonlinearity's name otherwise.
///
/// This is what makes a correct SiTU model *report* as correct. The
/// plan's activation probe compares a declared `hidden_act` against what
/// the built surface carries; comparing against the surface's
/// [`Activation`] alone would answer `silu` to a declared `situ` even
/// with the policy right, and the leaf would read `mismatched` forever.
///
/// [`ExpertGatePolicy::ClampedGlu`] returns `None` deliberately: no HF
/// word names it (GPT-OSS declares `silu` beside a `swiglu_limit`), so
/// there is nothing to answer with and the caller keeps its existing
/// behaviour. [`ExpertGatePolicy::ClampedGated`] answers `None` for the
/// same reason and lands on the right answer for a second reason:
/// GLM-5.3-Flash also declares `hidden_act: "silu"` beside its own
/// `swiglu_limit`, and its combine IS silu-based, so falling through to
/// the activation's own name is what makes the leaf agree.
pub fn hf_combine_name(policy: ExpertGatePolicy, activation: Activation) -> Option<String> {
    match policy {
        ExpertGatePolicy::SituGlu { .. } => Some(SITU_NAME.to_string()),
        ExpertGatePolicy::ClampedGlu { .. } | ExpertGatePolicy::ClampedGated { .. } => None,
        ExpertGatePolicy::Gated => activation.hf_name().map(str::to_string),
    }
}

/// Whether the FFN uses a gated architecture.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FfnType {
    /// Gated: SiLU(x @ gate.T) * (x @ up.T) @ down.T (Gemma, Llama)
    Gated,
    /// Standard: activation(x @ up.T) @ down.T (GPT-2)
    Standard,
}
