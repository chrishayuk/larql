//! Which tensors a representation applies to, by **role**.
//!
//! The obvious policy — "it is a 2-D matrix, so quantise it" — is not a
//! policy, it is a shape test. It happens to produce a good number on a
//! dense model and it silently 4-bits the embedding table and the output
//! head, which are among the places you would least want to be first.
//!
//! Eligibility is therefore semantic: a tensor's role decides, and the
//! default is conservative at every role where 4-bit is known to be
//! delicate. Explicit opt-in makes it more aggressive; nothing makes it
//! more aggressive by accident.
//!
//! ```text
//! decoder linear weights   REPRESENT   attention q/k/v/o, mlp gate/up/down
//! expert weights           REPRESENT   the prize at MoE scale
//!
//! embedding                PRESERVE
//! output head              PRESERVE
//! vision / audio / drafter PRESERVE    a whole component, not a tensor
//! norms                    PRESERVE
//! router / gate            PRESERVE    tiny, and routing errors compound
//! small vectors, biases    PRESERVE
//! anything unrecognised    PRESERVE    fail safe, never fail small
//! ```
//!
//! The last line is the one that matters most. A tensor this classifier
//! cannot name is a tensor nobody has reasoned about, and quantising it
//! because it happened to be 2-D is how a policy acquires behaviour its
//! author never chose.

use serde::{Deserialize, Serialize};
use std::fmt;

/// What a tensor does, as far as representation eligibility is concerned.
///
/// The serialised form is exactly what [`Role::name`] returns, because
/// `Serialize`/`Deserialize` are written in terms of `name`/`parse`
/// rather than derived.
///
/// Deriving `rename_all = "kebab-case"` would work today and is wrong in
/// principle: it makes the VARIANT NAME a second, independent definition
/// of the wire form. The two agree now, so a rename of either would keep
/// compiling and silently split the vocabulary — a precision map on disk
/// still deserialising while `--include-role` no longer accepted the
/// same word. One definition, pinned by `compat_tests`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    /// Attention and dense-FFN projections — the bulk of a dense model.
    DecoderLinear,
    /// The bulk matmuls of a recurrent token mixer: Gated DeltaNet's
    /// fused `in_proj_qkv`, its `in_proj_z` and `out_proj`; KDA's split
    /// q/k/v and out; Mamba2's fused `in_proj` and `out_proj`.
    ///
    /// The same shape of operation as [`Self::DecoderLinear`] at the same
    /// scale, and compiled by the same default — its own role because a
    /// profile must be able to say something different about a
    /// recurrence than about softmax attention without editing a
    /// substring list, and because a report that cannot name them cannot
    /// distinguish "protected" from "unrecognised".
    RecurrenceProjection,
    /// A recurrent mixer's *control* path: the narrow projections that
    /// emit per-head decay and write strength (Gated DeltaNet
    /// `in_proj_a`/`in_proj_b`, KDA's gate factorisations and `b_proj`).
    ///
    /// **Not protected by the default representation policy**, and the
    /// reason is a cost/benefit judgement rather than a null result.
    ///
    /// Q-BANK-1 compared an otherwise identical Qwen3.8 NVFP4 programme
    /// with these projections preserved at BF16 against one compiling
    /// them, matched on commit, reference bank and backend, differing by
    /// this role alone. Protection reduced top-1 flips from 106 to 98
    /// across 1,740 measured positions — a real effect, McNemar exact
    /// two-sided p = 0.0215 — while changing mean KL by 0.000389
    /// bits/token (1.6% of the mean) and costing 33,914,496 bytes,
    /// 0.0112 bits/weight. The metrics disagreed on direction: KL p95
    /// and mean ΔNLL were better *without* protection.
    ///
    /// The benefit concentrated in low-margin positions, where the BF16
    /// reference was already least decisive; the high-margin tertile had
    /// zero flips under either programme. So the measured effect did not
    /// justify a default precision exception.
    ///
    /// The architecture treating recurrent *state* specially —
    /// Qwen3.8 declares `linear_attention.state_dtype: float32` against
    /// otherwise-bf16 weights — does not imply that the projections
    /// controlling that state inherit special representation policy.
    /// Precision exceptions require behavioural evidence sufficient to
    /// justify their cost.
    ///
    /// Still its own role, because naming the operand is what made the
    /// question askable: `--protect` is `--include-role`'s inverse, and
    /// a profile with different evidence — a longer context, a different
    /// recurrence — can hold these back deliberately rather than by an
    /// accident of spelling.
    RecurrenceControl,
    /// Routed-expert weights — the bulk of an MoE model.
    ExpertWeight,
    /// Token embedding table.
    Embedding,
    /// Output / LM head.
    OutputHead,
    /// Any normalisation weight.
    Norm,
    /// Router or expert-gate weights.
    Router,
    /// 1-D vectors and biases.
    SmallVector,
    /// A weight belonging to a component that is not the primary text
    /// model — a vision or audio perception tower, a speculative drafter.
    ///
    /// Its tensors are named exactly like a decoder's (`attn.q_proj`,
    /// `mlp.fc1`), so a name-based classifier reads them as ordinary
    /// decoder linear work and quantises a perception encoder that the
    /// wider ecosystem hard-protects. The component's declared role is the
    /// signal that separates them; the tensor's name cannot.
    AuxiliaryComponent,
    /// Recognised as nothing in particular.
    Unknown,
}

impl Role {
    /// Every role, for CLI parsing and exhaustive reporting.
    pub const ALL: &'static [Role] = &[
        Role::DecoderLinear,
        Role::RecurrenceProjection,
        Role::RecurrenceControl,
        Role::ExpertWeight,
        Role::Embedding,
        Role::OutputHead,
        Role::Norm,
        Role::Router,
        Role::SmallVector,
        Role::AuxiliaryComponent,
        Role::Unknown,
    ];

    /// Lower-kebab name, used by `--include-role` and in reports.
    pub fn name(self) -> &'static str {
        match self {
            Role::DecoderLinear => "decoder-linear",
            Role::RecurrenceProjection => "recurrence-projection",
            Role::RecurrenceControl => "recurrence-control",
            Role::ExpertWeight => "expert-weight",
            Role::Embedding => "embedding",
            Role::OutputHead => "output-head",
            Role::Norm => "norm",
            Role::Router => "router",
            Role::SmallVector => "small-vector",
            Role::AuxiliaryComponent => "auxiliary-component",
            Role::Unknown => "unknown",
        }
    }

    pub fn parse(s: &str) -> Option<Role> {
        Role::ALL.iter().copied().find(|r| r.name() == s)
    }

    /// Whether the conservative default compiles this role.
    pub fn in_default_policy(self) -> bool {
        matches!(
            self,
            Role::DecoderLinear
                | Role::RecurrenceProjection
                | Role::RecurrenceControl
                | Role::ExpertWeight
        )
    }
}

impl Serialize for Role {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

impl<'de> Deserialize<'de> for Role {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = <&str as Deserialize>::deserialize(d)?;
        Role::parse(s).ok_or_else(|| {
            serde::de::Error::custom(format!(
                "`{s}` is not a role; a precision map naming it cannot be resolved"
            ))
        })
    }
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Classify one tensor from the object that holds it and its own name.
///
/// Both signals are needed. The object says what kind of thing this is —
/// `target.embedding` holds an embedding whatever its tensor is called —
/// and the tensor name discriminates within a decoder stack, where a norm
/// and a projection sit side by side under the same object.
pub fn classify(object: &str, tensor: &str, shape: &[usize]) -> Role {
    classify_in(true, object, tensor, shape)
}

/// Classify with the component's declared role in hand.
///
/// `primary_text` is the discriminator a tensor name cannot supply. A
/// perception tower's weights are called `attn.q_proj.weight` and
/// `mlp.fc1.weight` — identical to a decoder's — so classifying on names
/// alone quantises the vision encoder along with the language model. Muse
/// Glimmer is the case that showed it: 806 tensors, 3.45 GB, every one of
/// them reading as ordinary decoder linear work.
pub fn classify_in(primary_text: bool, object: &str, tensor: &str, shape: &[usize]) -> Role {
    // A tensor that is not a matrix cannot carry a block-quantised
    // representation regardless of what it means, so this is settled first.
    if shape.len() != 2 {
        return Role::SmallVector;
    }
    // A whole auxiliary component is out of scope before any tensor in it
    // is examined — the decision belongs to the component, not the tensor.
    if !primary_text {
        return Role::AuxiliaryComponent;
    }

    let obj = object.to_ascii_lowercase();
    let name = tensor.to_ascii_lowercase();

    // Object-level roles: the object *is* the thing.
    if obj.contains("embedding") || obj.contains("embed_tokens") {
        return Role::Embedding;
    }
    if obj.contains("output_head") || obj.contains("lm_head") {
        return Role::OutputHead;
    }
    if obj.contains("final_norm") {
        return Role::Norm;
    }

    // Tensor-level roles within a stack or a bank.
    if name.contains("norm") {
        return Role::Norm;
    }
    // `router` and a bare `gate` select experts; `gate_proj` is the GLU
    // gate half of a dense FFN and is ordinary decoder linear work. The
    // two are one token apart and mean entirely different things.
    if name.contains("router") || name.contains("gate.weight") || name.ends_with(".gate") {
        return Role::Router;
    }
    if name.ends_with("bias") {
        return Role::SmallVector;
    }

    if obj.contains("expert") {
        return Role::ExpertWeight;
    }

    let is_projection = name.contains("_proj.")
        || name.ends_with("_proj")
        || name.contains("self_attn.")
        || name.contains("attention.")
        || name.contains("mlp.");
    if is_projection {
        return Role::DecoderLinear;
    }

    Role::Unknown
}

/// Which roles a compilation compiles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RolePolicy {
    included: Vec<Role>,
}

impl Default for RolePolicy {
    fn default() -> Self {
        Self {
            included: Role::ALL
                .iter()
                .copied()
                .filter(|r| r.in_default_policy())
                .collect(),
        }
    }
}

impl RolePolicy {
    /// Add a role the default leaves preserved. The escape hatch for a
    /// profile that has decided, deliberately, to be more aggressive.
    pub fn including(mut self, role: Role) -> Self {
        if !self.included.contains(&role) {
            self.included.push(role);
            self.included.sort();
        }
        self
    }

    pub fn compiles(&self, role: Role) -> bool {
        self.included.contains(&role)
    }

    pub fn roles(&self) -> &[Role] {
        &self.included
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: &[usize] = &[2560, 2560];

    #[test]
    fn decoder_projections_are_compiled_by_default() {
        let p = RolePolicy::default();
        for t in [
            "0.self_attn.q_proj.weight",
            "7.self_attn.k_proj.weight",
            "7.self_attn.v_proj.weight",
            "7.self_attn.o_proj.weight",
            "39.mlp.gate_proj.weight",
            "39.mlp.up_proj.weight",
            "39.mlp.down_proj.weight",
        ] {
            let role = classify("target.decoder_stack", t, M);
            assert_eq!(role, Role::DecoderLinear, "{t}");
            assert!(p.compiles(role), "{t}");
        }
    }

    #[test]
    fn the_embedding_and_head_are_preserved_by_default() {
        let p = RolePolicy::default();
        // The whole point of the change: both are 2-D matrices, and both
        // are places 4-bit is known to be delicate.
        assert_eq!(classify("target.embedding", "weight", M), Role::Embedding);
        assert_eq!(
            classify("target.output_head", "weight", M),
            Role::OutputHead
        );
        assert!(!p.compiles(Role::Embedding));
        assert!(!p.compiles(Role::OutputHead));
    }

    #[test]
    fn expert_weights_are_compiled_but_their_router_is_not() {
        let p = RolePolicy::default();
        assert_eq!(
            classify("target.expert_bank", "3.experts.gate_up_proj", M),
            Role::ExpertWeight
        );
        assert!(p.compiles(Role::ExpertWeight));

        // Routing errors select the wrong expert entirely, which is not a
        // small numerical perturbation.
        assert_eq!(
            classify("target.expert_bank", "3.router.weight", M),
            Role::Router
        );
        assert!(!p.compiles(Role::Router));
    }

    #[test]
    fn a_glu_gate_half_is_not_a_router() {
        // `mlp.gate_proj` and `router`/`gate` are one token apart and mean
        // different things; conflating them would preserve half of every
        // dense FFN or quantise every routing decision.
        assert_eq!(
            classify("target.decoder_stack", "5.mlp.gate_proj.weight", M),
            Role::DecoderLinear
        );
        assert_eq!(
            classify("target.decoder_stack", "5.mlp.gate.weight", M),
            Role::Router
        );
    }

    #[test]
    fn norms_and_vectors_are_never_compiled() {
        let p = RolePolicy::default();
        assert_eq!(
            classify("target.decoder_stack", "0.input_layernorm.weight", &[2560]),
            Role::SmallVector
        );
        assert_eq!(
            classify(
                "target.decoder_stack",
                "0.post_attention_layernorm.weight",
                M
            ),
            Role::Norm
        );
        assert_eq!(classify("target.final_norm", "weight", M), Role::Norm);
        assert_eq!(
            classify("target.decoder_stack", "0.self_attn.q_proj.bias", M),
            Role::SmallVector
        );
        for r in [Role::Norm, Role::SmallVector] {
            assert!(!p.compiles(r));
        }
    }

    #[test]
    fn an_unrecognised_matrix_is_preserved_not_compiled() {
        // Fail safe: a tensor nobody has reasoned about must not acquire a
        // lossy representation because it happened to be 2-D.
        let role = classify("target.something_new", "mystery.weight", M);
        assert_eq!(role, Role::Unknown);
        assert!(!RolePolicy::default().compiles(role));
    }

    #[test]
    fn a_role_can_be_opted_in_explicitly() {
        let p = RolePolicy::default().including(Role::Embedding);
        assert!(p.compiles(Role::Embedding));
        assert!(p.compiles(Role::DecoderLinear));
        // Opting one role in must not opt others in with it.
        assert!(!p.compiles(Role::OutputHead));
        assert!(!p.compiles(Role::Router));
    }

    #[test]
    fn a_perception_tower_is_not_decoder_work() {
        // Muse Glimmer's vision tower carries `layers.attn.q_proj.weight`
        // and `layers.mlp.fc1.weight` — byte-for-byte the naming a text
        // decoder uses. Only the component's role tells them apart.
        for t in [
            "layers.attn.q_proj.weight",
            "layers.attn.v_proj.weight",
            "layers.mlp.fc1.weight",
            "vision_projection.weight",
        ] {
            assert_eq!(
                classify_in(false, "vision.perception_tower", t, M),
                Role::AuxiliaryComponent,
                "{t}"
            );
        }
        // The decoder-shaped names among them ARE decoder work inside the
        // text model — which is precisely why the name cannot decide.
        for t in [
            "layers.attn.q_proj.weight",
            "layers.attn.v_proj.weight",
            "layers.mlp.fc1.weight",
        ] {
            assert_eq!(
                classify_in(true, "target.decoder_stack", t, M),
                Role::DecoderLinear,
                "{t}"
            );
        }
        assert!(!RolePolicy::default().compiles(Role::AuxiliaryComponent));
    }

    #[test]
    fn role_names_round_trip() {
        for r in Role::ALL {
            assert_eq!(Role::parse(r.name()), Some(*r), "{}", r.name());
        }
        assert_eq!(Role::parse("not-a-role"), None);
    }
}

/// Finer-grained protection than a role: individual projections, and
/// ranges of layer depth.
///
/// Role eligibility answers "is this the kind of weight the encoding
/// applies to". This answers "and should *this one* be spent", which is a
/// different question and the one a precision map is made of. Q-BANK-1
/// showed why it is needed: Granite at uniform NVFP4 moves the argmax on
/// one position in six, and most of those flips are not near-ties — so the
/// interesting work is finding which bytes cause them.
///
/// Empty means protect nothing extra, which is R0.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Protections {
    rules: Vec<ProtectRule>,
}

/// One protection. Stated conditions are ANDed; unstated ones match
/// anything.
///
/// The union/intersection distinction matters and Q-BANK-1 forced it.
/// Granite's damage is FFN-heavy *and* late-layer-heavy, and neither
/// coarse split is both cheap and effective — protecting all FFN costs
/// 3.4 GiB, protecting all late layers costs 1.1 GiB and leaves the tail
/// at p99 1.12. Their *intersection* is the candidate worth testing, and
/// a rule list of separate projections and ranges can only express the
/// union.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProtectRule {
    pub projection: Option<String>,
    pub layers: Option<(u32, u32)>,
}

impl ProtectRule {
    fn matches(&self, tensor: &str) -> bool {
        if let Some(p) = &self.projection {
            if projection_of(tensor) != Some(p.as_str()) {
                return false;
            }
        }
        if let Some((lo, hi)) = self.layers {
            match layer_of(tensor) {
                Some(l) if l >= lo && l <= hi => {}
                _ => return false,
            }
        }
        // A rule stating nothing matches everything, which is how "compile
        // nothing" is written.
        true
    }

    fn describe(&self) -> String {
        match (&self.projection, self.layers) {
            (Some(p), Some((lo, hi))) => format!("{p}@{lo}-{hi}"),
            (Some(p), None) => p.clone(),
            (None, Some((lo, hi))) => format!("layers {lo}-{hi}"),
            (None, None) => "*".into(),
        }
    }
}

impl Protections {
    /// Protect every tensor whose projection matches, at any depth.
    pub fn projection(mut self, name: impl Into<String>) -> Self {
        self.rules.push(ProtectRule {
            projection: Some(name.into()),
            layers: None,
        });
        self
    }

    /// Protect an inclusive range of layer depths, any projection.
    pub fn layers(mut self, lo: u32, hi: u32) -> Self {
        self.rules.push(ProtectRule {
            projection: None,
            layers: Some((lo, hi)),
        });
        self
    }

    /// Protect one projection *within* a depth range — the intersection.
    pub fn projection_in(mut self, name: impl Into<String>, lo: u32, hi: u32) -> Self {
        self.rules.push(ProtectRule {
            projection: Some(name.into()),
            layers: Some((lo, hi)),
        });
        self
    }

    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Whether this tensor is held at source precision despite its role
    /// being eligible. Rules are a union; each rule is a conjunction.
    pub fn protects(&self, tensor: &str) -> bool {
        self.rules.iter().any(|r| r.matches(tensor))
    }

    /// Express these protections as precision-map exceptions.
    ///
    /// Protections are how a *compilation* is asked for; exceptions are how
    /// the resulting *program* is stated. Keeping the conversion here means
    /// the map a container carries is derived from the same object the
    /// compiler acted on, rather than reconstructed alongside it.
    pub fn as_exceptions(&self) -> Vec<super::map::Exception> {
        self.rules
            .iter()
            .map(|r| super::map::Exception {
                projection: r.projection.clone(),
                layers: r.layers,
                encoding: None,
            })
            .collect()
    }

    pub fn describe(&self) -> String {
        if self.rules.is_empty() {
            return "none".into();
        }
        self.rules
            .iter()
            .map(ProtectRule::describe)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// `0.self_attn.q_proj.weight` -> `q_proj`.
pub fn projection_of(tensor: &str) -> Option<&str> {
    let parts: Vec<&str> = tensor.split('.').collect();
    (parts.len() >= 2).then(|| parts[parts.len() - 2])
}

/// Leading layer index of an object-relative tensor name, when it has one.
///
/// Object-relative names start at the layer (`0.self_attn...`) because the
/// object *is* the stack; a tensor without a leading index belongs to no
/// particular depth and no depth range can protect it.
pub fn layer_of(tensor: &str) -> Option<u32> {
    tensor.split('.').next()?.parse().ok()
}

#[cfg(test)]
mod protection_tests {
    use super::*;

    #[test]
    fn a_projection_is_protected_at_every_depth() {
        let p = Protections::default().projection("v_proj");
        assert!(p.protects("0.self_attn.v_proj.weight"));
        assert!(p.protects("39.self_attn.v_proj.weight"));
        assert!(!p.protects("0.self_attn.q_proj.weight"));
        assert!(!p.protects("0.mlp.down_proj.weight"));
    }

    #[test]
    fn a_depth_range_is_inclusive_at_both_ends() {
        let p = Protections::default().layers(0, 3);
        for l in 0..=3 {
            assert!(p.protects(&format!("{l}.mlp.up_proj.weight")), "layer {l}");
        }
        assert!(!p.protects("4.mlp.up_proj.weight"));
    }

    #[test]
    fn protections_compose_as_a_union() {
        // Two independent reasons to hold a tensor back; either suffices.
        let p = Protections::default()
            .projection("down_proj")
            .layers(30, 39);
        assert!(p.protects("5.mlp.down_proj.weight"), "by projection");
        assert!(p.protects("31.self_attn.q_proj.weight"), "by depth");
        assert!(!p.protects("5.self_attn.q_proj.weight"), "neither");
    }

    #[test]
    fn a_tensor_with_no_leading_depth_is_not_caught_by_a_range() {
        // The embedding's tensor is just `weight`; a layer range says
        // nothing about it, and must not silently claim it.
        let p = Protections::default().layers(0, 100);
        assert!(!p.protects("weight"));
        assert_eq!(layer_of("weight"), None);
        assert_eq!(projection_of("0.self_attn.q_proj.weight"), Some("q_proj"));
    }

    #[test]
    fn an_intersection_protects_only_where_both_hold() {
        // The R2 mechanism. A union of "gate_proj" and "layers 30-39"
        // would protect gate at every depth AND every projection late;
        // the intersection protects late gate only, which is a quarter of
        // the bytes and the actual hypothesis under test.
        let p = Protections::default().projection_in("gate_proj", 30, 39);
        assert!(
            p.protects("35.mlp.gate_proj.weight"),
            "both conditions hold"
        );
        assert!(
            !p.protects("5.mlp.gate_proj.weight"),
            "right projection, wrong depth"
        );
        assert!(
            !p.protects("35.mlp.up_proj.weight"),
            "right depth, wrong projection"
        );
        assert_eq!(p.describe(), "gate_proj@30-39");
    }

    #[test]
    fn a_union_and_an_intersection_are_different_policies() {
        let union = Protections::default()
            .projection("gate_proj")
            .layers(30, 39);
        let inter = Protections::default().projection_in("gate_proj", 30, 39);
        // The union catches both of these; the intersection catches neither.
        assert!(union.protects("5.mlp.gate_proj.weight"));
        assert!(union.protects("35.mlp.up_proj.weight"));
        assert!(!inter.protects("5.mlp.gate_proj.weight"));
        assert!(!inter.protects("35.mlp.up_proj.weight"));
    }

    #[test]
    fn empty_protections_are_r0() {
        let p = Protections::default();
        assert!(p.is_empty());
        assert!(!p.protects("0.self_attn.v_proj.weight"));
        assert_eq!(p.describe(), "none");
    }
}
