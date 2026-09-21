//! **K3-LATENTMOE-1 — the latent routed branch, declared and addressed.**
//!
//! Kimi-K3's routed experts do not run at the model's hidden width. The
//! block input is projected DOWN to `routed_expert_hidden_size`, the
//! experts run there, their weighted aggregate is normalised, and
//! `routed_expert_up_proj` returns it to the residual stream. The router
//! and the shared experts stay outside that bottleneck.
//!
//! This module holds the declaration to the estate from BOTH sides, the
//! way `k3_q_lora_closure` holds `q_lora_rank` to the query operands:
//!
//! ```text
//! declared, wrapper shipped      -> closes, and the op carries the form
//! declared, an operand absent    -> MissingOperand naming THAT role
//! not declared, wrapper shipped  -> refused BY NAME, never re-read as latent
//! declared width 0               -> the form is selected, the width refused
//! norm flag with no width        -> inert: no norm built, and none required
//! ```
//!
//! # The arm that decides whether this rung means anything
//!
//! [`the_expert_bank_is_sized_from_the_declaration_not_the_component_width`]
//! is the one to read first. A latent width can be made to LOOK carried
//! by giving it a home on the surface: the blocker count moves, the
//! container round-trips, and the expert bank is still sized from the
//! component's `hidden`. The plan then reports a fact the operand plane
//! contradicts — hollow carriage — and the only thing that catches it is
//! an estate whose bank is stored at the DECLARED width and refuses at
//! the other one.
//!
//! So the bank here is shipped at the latent width, and the negative arm
//! ships the same bank at `hidden` and requires a refusal. `hidden`,
//! `latent` and the expert intermediate width are pairwise distinct, and
//! `hidden / 2` is none of them — K3 itself satisfies `hidden / 2 ==
//! latent`, so a fixture that inherited that coincidence would pass
//! against a build that derived the width instead of reading it.
//!
//! Attention is plain full-attention softmax for the reason
//! `kimi_moe_closure` gives: this rung is the routed branch, and pulling
//! KDA or MLA geometry in would test another rung's machinery under this
//! one's name.

use crate::format::vindex3::encode::encode_graph;
use crate::format::vindex3::graph::OperandRole;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::{
    plan_component_ops, ClosureDefect, ExpertBank, LayerFfn, OpPlanOutcome,
};
use crate::format::vindex3::plan::plan_system;
use crate::format::vindex3::plan::tests_support::custom_artifact;

const HIDDEN: usize = 32;
/// The bottleneck. Deliberately not `HIDDEN / 2` (16) and not
/// [`MOE_INTER`] — see the module docs.
const LATENT: usize = 20;
const Q_HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const EXPERTS: usize = 3;
const TOP_K: usize = 2;
const MOE_INTER: usize = 12;
const SHARED_EXPERTS: usize = 1;
const LAYERS: usize = 2;
const VOCAB: usize = 64;
/// The layer's norm epsilon — and, for `routed_expert_norm`, the value it
/// actually runs at. The neighbouring MLA low-rank norms in this family
/// run at `KimiRMSNorm`'s class default `1e-6` instead, which is why the
/// op carries this one rather than an executor reaching for whichever
/// epsilon is nearest.
const NORM_EPS: f64 = 1e-5;
/// The epsilon two earlier K3 rungs found on low-rank norms, and the one
/// this branch does NOT use.
const CLASS_DEFAULT_EPS: f64 = 1e-6;

/// What a variant ships for the latent wrapper.
#[derive(Clone, Copy, PartialEq)]
struct Wrapper {
    down: bool,
    norm: bool,
    up: bool,
}

impl Wrapper {
    const ALL: Self = Self {
        down: true,
        norm: true,
        up: true,
    };
    const NONE: Self = Self {
        down: false,
        norm: false,
        up: false,
    };
}

/// What a variant DECLARES about the branch.
#[derive(Clone, Copy)]
struct Declared {
    /// `None` = the leaf is absent, which is the uniform form.
    width: Option<usize>,
    /// `None` = `latent_moe_use_norm` absent; the reference reads it with
    /// a `getattr(..., False)`, so absent and `false` build the same
    /// program and differ only in what the plan reports as declared.
    use_norm: Option<bool>,
}

impl Declared {
    const LATENT_WITH_NORM: Self = Self {
        width: Some(LATENT),
        use_norm: Some(true),
    };
    const UNIFORM: Self = Self {
        width: None,
        use_norm: None,
    };

    /// The width the EXPERTS run at under this declaration — what their
    /// bank must be stored at for the estate to agree with the config.
    fn expert_width(self) -> usize {
        self.width.unwrap_or(HIDDEN)
    }
}

fn config(declared: Declared) -> serde_json::Value {
    let mut cfg = serde_json::json!({
        "architectures": ["KimiLinearForCausalLM"],
        "model_type": "kimi_linear",
        "hidden_size": HIDDEN,
        "intermediate_size": HIDDEN * 4,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": Q_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "vocab_size": VOCAB,
        "rope_theta": 10000.0,
        "rms_norm_eps": NORM_EPS,
        "linear_attn_config": {
            "kda_layers": [],
            "full_attn_layers": (1..=LAYERS).collect::<Vec<_>>()
        },
        "first_k_dense_replace": 0,
        "num_experts": EXPERTS,
        "num_experts_per_token": TOP_K,
        "num_shared_experts": SHARED_EXPERTS,
        "moe_intermediate_size": MOE_INTER,
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": true,
    });
    let obj = cfg.as_object_mut().unwrap();
    if let Some(width) = declared.width {
        obj.insert("routed_expert_hidden_size".to_string(), width.into());
    }
    if let Some(flag) = declared.use_norm {
        obj.insert("latent_moe_use_norm".to_string(), flag.into());
    }
    cfg
}

/// One routed layer's tensors, with the expert bank stored at
/// `expert_width` and the wrapper shipped as `wrapper` says.
fn layer_tensors(
    layer: usize,
    expert_width: usize,
    wrapper: Wrapper,
    latent: usize,
) -> Vec<(String, Vec<usize>)> {
    let prefix = format!("model.layers.{layer}.");
    let mut tensors = vec![
        (
            format!("{prefix}self_attn.q_proj.weight"),
            vec![Q_HEADS * HEAD_DIM, HIDDEN],
        ),
        (
            format!("{prefix}self_attn.k_proj.weight"),
            vec![KV_HEADS * HEAD_DIM, HIDDEN],
        ),
        (
            format!("{prefix}self_attn.v_proj.weight"),
            vec![KV_HEADS * HEAD_DIM, HIDDEN],
        ),
        (
            format!("{prefix}self_attn.o_proj.weight"),
            vec![HIDDEN, Q_HEADS * HEAD_DIM],
        ),
        (format!("{prefix}input_layernorm.weight"), vec![HIDDEN]),
        (
            format!("{prefix}post_attention_layernorm.weight"),
            vec![HIDDEN],
        ),
        // The router reads the UN-projected block input, so its `k` is
        // `HIDDEN` in every arm here — including the latent ones. This is
        // the placement fact the oracle could not build a mutant for: at
        // a `[EXPERTS, HIDDEN]` matrix, routing on the bottleneck fails
        // on shape rather than computing a different model.
        (
            format!("{prefix}block_sparse_moe.gate.weight"),
            vec![EXPERTS, HIDDEN],
        ),
        (
            format!("{prefix}block_sparse_moe.gate.e_score_correction_bias"),
            vec![EXPERTS],
        ),
        // The shared branch likewise: `HIDDEN` in, `HIDDEN` out, summed
        // after the up-projection.
        (
            format!("{prefix}block_sparse_moe.shared_experts.gate_proj.weight"),
            vec![MOE_INTER * SHARED_EXPERTS, HIDDEN],
        ),
        (
            format!("{prefix}block_sparse_moe.shared_experts.up_proj.weight"),
            vec![MOE_INTER * SHARED_EXPERTS, HIDDEN],
        ),
        (
            format!("{prefix}block_sparse_moe.shared_experts.down_proj.weight"),
            vec![HIDDEN, MOE_INTER * SHARED_EXPERTS],
        ),
    ];
    if wrapper.down {
        tensors.push((
            format!("{prefix}block_sparse_moe.routed_expert_down_proj.weight"),
            vec![latent, HIDDEN],
        ));
    }
    if wrapper.norm {
        tensors.push((
            format!("{prefix}block_sparse_moe.routed_expert_norm.weight"),
            vec![latent],
        ));
    }
    if wrapper.up {
        tensors.push((
            format!("{prefix}block_sparse_moe.routed_expert_up_proj.weight"),
            vec![HIDDEN, latent],
        ));
    }
    // The bank: `w1`/`w3` take the experts' input width, `w2` returns it.
    for expert in 0..EXPERTS {
        tensors.push((
            format!("{prefix}block_sparse_moe.experts.{expert}.w1.weight"),
            vec![MOE_INTER, expert_width],
        ));
        tensors.push((
            format!("{prefix}block_sparse_moe.experts.{expert}.w3.weight"),
            vec![MOE_INTER, expert_width],
        ));
        tensors.push((
            format!("{prefix}block_sparse_moe.experts.{expert}.w2.weight"),
            vec![expert_width, MOE_INTER],
        ));
    }
    tensors
}

/// Plan a component that declares `declared` and ships `wrapper`, with
/// its expert bank stored at `expert_width`.
fn plan_with(declared: Declared, wrapper: Wrapper, expert_width: usize) -> OpPlanOutcome {
    let mut tensors = vec![
        ("model.embed_tokens.weight".to_string(), vec![VOCAB, HIDDEN]),
        ("model.norm.weight".to_string(), vec![HIDDEN]),
        ("lm_head.weight".to_string(), vec![VOCAB, HIDDEN]),
    ];
    for layer in 0..LAYERS {
        tensors.extend(layer_tensors(
            layer,
            expert_width,
            wrapper,
            declared.width.unwrap_or(LATENT),
        ));
    }
    let borrowed: Vec<(&str, &[usize])> = tensors
        .iter()
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let inventory = custom_artifact(dir.path(), &config(declared), &borrowed);
    let named = vec![("k3-latent".to_string(), inventory)];
    let out = tempfile::tempdir().unwrap();
    // The GRAPH seam, as `k3_q_lora_closure` uses and for the same
    // reason: the production writer refuses to keep an estate whose plan
    // does not admit, and several arms here exist to see what OPERAND
    // closure says below that refusal — including one whose config the
    // plan stage is right to block.
    let system = plan_system(&named);
    encode_graph(&system.graph, &named, out.path()).unwrap();
    let inspection = inspect_container(out.path(), false).unwrap();
    plan_component_ops(&inspection, out.path(), "target").unwrap()
}

/// The agreeing estate for a declaration: bank at the declared width,
/// wrapper shipped iff the form carries one.
fn plan_agreeing(declared: Declared) -> OpPlanOutcome {
    let wrapper = match declared.width {
        Some(_) => Wrapper {
            down: true,
            norm: declared.use_norm.unwrap_or(false),
            up: true,
        },
        None => Wrapper::NONE,
    };
    plan_with(declared, wrapper, declared.expert_width())
}

fn missing_roles(outcome: &OpPlanOutcome) -> Vec<String> {
    outcome
        .defects
        .iter()
        .filter_map(|d| match d {
            ClosureDefect::MissingOperand { role, .. } => Some(format!("{role:?}")),
            _ => None,
        })
        .collect()
}

fn implied_absent(outcome: &OpPlanOutcome) -> Vec<(String, String)> {
    outcome
        .defects
        .iter()
        .filter_map(|d| match d {
            ClosureDefect::OperandImpliesAbsentOp {
                tensor,
                required_primitive,
                ..
            } => Some((tensor.clone(), required_primitive.clone())),
            _ => None,
        })
        .collect()
}

fn routed(outcome: &OpPlanOutcome) -> &crate::format::vindex3::opplan::RoutedFfnOp {
    let plan = outcome.plan.as_ref().expect("the op plan");
    match &plan.layers[0].ffn {
        Some(LayerFfn::Routed(op)) => op,
        _ => panic!("layer 0 planned a non-routed FFN"),
    }
}

/// **The fixture must be able to see what it is asked about.** Three
/// pairwise-distinct widths, and a latent that is not the derivation a
/// reader would reach for.
///
/// First, because every arm below is decoration if `LATENT` happens to
/// equal a number the build could have produced by accident.
#[test]
fn the_geometry_can_distinguish_the_derivations_it_must_refuse() {
    assert_ne!(LATENT, HIDDEN);
    assert_ne!(LATENT, HIDDEN / 2, "K3's own coincidence must not be here");
    assert_ne!(LATENT, MOE_INTER);
    assert_ne!(HIDDEN / 2, MOE_INTER);
    assert_ne!(NORM_EPS, CLASS_DEFAULT_EPS);
}

/// **The positive arm.** Declared and shipped: the estate closes, and the
/// op carries the whole form — the width, both projections, and the norm
/// with its own epsilon.
#[test]
fn a_declared_latent_branch_with_the_wrapper_shipped_closes() {
    let outcome = plan_agreeing(Declared::LATENT_WITH_NORM);
    assert!(
        outcome.closed(),
        "the agreeing estate must close: {:?}",
        outcome.defects
    );
    let op = routed(&outcome);
    let latent = op.latent.as_ref().expect("the op carries the wrapper");
    assert_eq!(latent.width, LATENT);
    assert!(latent
        .down
        .tensor
        .ends_with("routed_expert_down_proj.weight"));
    assert!(latent.up.tensor.ends_with("routed_expert_up_proj.weight"));
    let norm = latent.norm.as_ref().expect("the flag declares one");
    assert!(norm.weight.tensor.ends_with("routed_expert_norm.weight"));
    assert_eq!(
        norm.eps, NORM_EPS as f32 as f64,
        "the routed-expert norm runs at the LAYER's epsilon, not the class default \
         two earlier K3 rungs found on the low-rank norms"
    );
    // The f32 round-trip is deliberate and is not this rung's to change:
    // `ModelArchitecture::norm_eps` narrows to f32, and EVERY norm in the
    // component reaches its epsilon through it. Reading the config's f64
    // here instead would give the routed norm a second, better derivation
    // of "the layer's epsilon" than the layer's other norms have — one
    // fact, two authorities, differing in the eighth digit. The claim
    // this rung makes is about WHICH epsilon, and that claim is a factor
    // of ten wide.
    assert_ne!(norm.eps, CLASS_DEFAULT_EPS);
    assert!(
        (norm.eps / NORM_EPS - 1.0).abs() < 1e-6,
        "the widened layer epsilon must still BE the layer's, got {}",
        norm.eps
    );
}

/// **The baseline.** The same fixture with neither leaf declared closes
/// exactly as it did before this rung, and plans no wrapper.
///
/// Without this the arms above could pass on a build that made every
/// routed FFN latent.
#[test]
fn an_undeclared_branch_closes_unchanged_and_plans_no_wrapper() {
    let outcome = plan_agreeing(Declared::UNIFORM);
    assert!(outcome.closed(), "{:?}", outcome.defects);
    let op = routed(&outcome);
    assert!(op.latent.is_none());
    let ExpertBank::PerExpert { gate, .. } = &op.bank else {
        panic!("this fixture ships a per-expert bank");
    };
    assert_eq!(gate.len(), EXPERTS);
}

/// **Hollow carriage, refused.** The declaration is what the bank is
/// sized from — so the same bank that closes at the declared latent width
/// is refused at the component's `hidden`.
///
/// Both arms are needed. The positive one alone would pass on a build
/// that ignored the declaration and sized from `hidden`, because this
/// fixture's bank would then be the wrong one in both. The negative one
/// alone would pass on a build that refused everything.
#[test]
fn the_expert_bank_is_sized_from_the_declaration_not_the_component_width() {
    let at_latent = plan_with(Declared::LATENT_WITH_NORM, Wrapper::ALL, LATENT);
    assert!(
        at_latent.closed(),
        "a bank stored at the DECLARED width must close: {:?}",
        at_latent.defects
    );

    let at_hidden = plan_with(Declared::LATENT_WITH_NORM, Wrapper::ALL, HIDDEN);
    let mismatches: Vec<_> = at_hidden
        .defects
        .iter()
        .filter_map(|d| match d {
            ClosureDefect::GeometryMismatch {
                tensor,
                expected,
                actual,
            } => Some((tensor.clone(), expected.clone(), actual.clone())),
            _ => None,
        })
        .collect();
    assert!(
        !mismatches.is_empty(),
        "a bank stored at the component's hidden width under a latent declaration must be \
         refused; defects were {:?}",
        at_hidden.defects
    );
    // And refused against the DECLARED width — not against `hidden`,
    // which is what a build that never read the declaration would expect.
    for (tensor, expected, actual) in &mismatches {
        assert!(
            expected.contains(&LATENT),
            "{tensor}: expected {expected:?} names no latent width — the contract was built \
             from something other than the declaration"
        );
        assert!(
            actual.contains(&HIDDEN),
            "{tensor}: actual {actual:?} is not the hidden-width bank this arm ships"
        );
    }
}

/// **Declared, and each wrapper operand absent in turn**: the role is
/// named, one at a time, so no refusal is standing in for another.
#[test]
fn each_absent_wrapper_operand_names_its_own_role() {
    for (missing, role) in [
        (
            Wrapper {
                down: false,
                norm: true,
                up: true,
            },
            "MoeLatentDownProj",
        ),
        (
            Wrapper {
                down: true,
                norm: false,
                up: true,
            },
            "MoeLatentNorm",
        ),
        (
            Wrapper {
                down: true,
                norm: true,
                up: false,
            },
            "MoeLatentUpProj",
        ),
    ] {
        let outcome = plan_with(Declared::LATENT_WITH_NORM, missing, LATENT);
        let roles = missing_roles(&outcome);
        assert!(
            roles.iter().any(|r| r == role),
            "expected a MissingOperand for {role}, got {roles:?} from {:?}",
            outcome.defects
        );
    }
}

/// **The other direction, and the one that matters most.** A checkpoint
/// shipping the wrapper WITHOUT the declaration is refused by name — the
/// tensors never select the form.
///
/// The rule is not symmetry for its own sake: the expert bank's stored
/// width IS the quantity the declaration decides, so a build willing to
/// infer the form from the operands would be reading the answer off the
/// thing it is meant to be judging, and would execute a checkpoint with a
/// stray wrapper tensor as a different model.
#[test]
fn the_wrapper_without_the_declaration_is_refused_by_name() {
    let outcome = plan_with(Declared::UNIFORM, Wrapper::ALL, HIDDEN);
    let implied = implied_absent(&outcome);
    for spelling in [
        "routed_expert_down_proj.weight",
        "routed_expert_norm.weight",
        "routed_expert_up_proj.weight",
    ] {
        assert!(
            implied.iter().any(|(tensor, _)| tensor.ends_with(spelling)),
            "{spelling} shipped without a declaration must be refused by name, got {implied:?}"
        );
    }
    assert!(
        outcome.plan.as_ref().is_none_or(
            |p| matches!(&p.layers[0].ffn, Some(LayerFfn::Routed(op)) if op.latent.is_none())
        ),
        "a wrapper tensor must never be re-read as a latent declaration"
    );
}

/// **The norm is refused on its own.** A `routed_expert_norm` shipped
/// under a declared branch whose flag is false is a different refusal
/// from one shipped with no branch at all, and the message says which.
#[test]
fn a_norm_shipped_against_a_false_flag_is_refused_by_name() {
    let declared = Declared {
        width: Some(LATENT),
        use_norm: Some(false),
    };
    let outcome = plan_with(declared, Wrapper::ALL, LATENT);
    let implied = implied_absent(&outcome);
    let (_, reason) = implied
        .iter()
        .find(|(tensor, _)| tensor.ends_with("routed_expert_norm.weight"))
        .unwrap_or_else(|| panic!("the norm must be refused by name, got {implied:?}"));
    assert!(
        reason.contains("latent_moe_use_norm"),
        "the refusal must name the flag that switched the norm off, got {reason:?}"
    );
    // The rest of the wrapper is untouched: down and up are still
    // required and still close.
    assert!(
        missing_roles(&outcome).is_empty(),
        "only the norm is at issue here: {:?}",
        outcome.defects
    );
}

/// **The flag without the width is inert**, exactly as the reference
/// nests it: `if self.use_latent_moe:` encloses `if
/// self.latent_moe_use_norm:`, so a config setting the flag and no width
/// builds NO norm — not a norm on the un-projected residual.
///
/// The plan says so by planning no wrapper at all and requiring none, and
/// a `routed_expert_norm` shipped beside that flag is still refused.
#[test]
fn the_norm_flag_without_a_width_builds_nothing() {
    let declared = Declared {
        width: None,
        use_norm: Some(true),
    };
    let outcome = plan_with(declared, Wrapper::NONE, HIDDEN);
    assert!(
        outcome.closed(),
        "an inert flag must not require operands: {:?}",
        outcome.defects
    );
    assert!(routed(&outcome).latent.is_none());
    // What the PLAN stage says about the flag is a separate judgment and
    // deliberately not softened here: the schema holds no value for it,
    // so it grades unrepresented and blocks. That is the honest reading —
    // a checkpoint declaring a norm for a branch it never declared is
    // worth a human look, and the corpus has no such row.

    let with_norm = plan_with(
        declared,
        Wrapper {
            down: false,
            norm: true,
            up: false,
        },
        HIDDEN,
    );
    let implied = implied_absent(&with_norm);
    assert!(
        implied
            .iter()
            .any(|(tensor, _)| tensor.ends_with("routed_expert_norm.weight")),
        "a norm shipped under an inert flag must still be refused, got {implied:?}"
    );
}

/// **Zero is a declared width, not an absent one.** The reference selects
/// the form with `is not None`, so `routed_expert_hidden_size: 0` selects
/// a bottleneck of no width — refused naming the DECLARATION, never
/// quietly demoted to the uniform form.
#[test]
fn a_zero_width_selects_the_form_and_is_refused_by_name() {
    let declared = Declared {
        width: Some(0),
        use_norm: Some(true),
    };
    let outcome = plan_with(declared, Wrapper::ALL, HIDDEN);
    let named: Vec<&String> = outcome
        .defects
        .iter()
        .filter_map(|d| match d {
            ClosureDefect::FfnWidthDeclaration { detail, .. } => Some(detail),
            _ => None,
        })
        .collect();
    assert!(
        named
            .iter()
            .any(|detail| detail.contains("routed_expert_hidden_size")),
        "a zero latent width must be refused naming the declaration, got {:?}",
        outcome.defects
    );
}

/// The three wrapper spellings classify to the three wrapper roles, and
/// to nothing else.
///
/// A cheap arm, and the one that would catch the wrapper being filed
/// under the expert bank: these are DENSE per-layer operands — one of
/// each per routed layer whatever the expert count — and a build that put
/// them in the bank would multiply them by 896 on the real checkpoint.
#[test]
fn the_wrapper_spellings_classify_to_the_wrapper_roles() {
    use crate::format::vindex3::graph::roles::classify_stack_tensor;
    const LAYER: usize = 3;
    for (spelling, want) in [
        ("routed_expert_down_proj", OperandRole::MoeLatentDownProj),
        ("routed_expert_norm", OperandRole::MoeLatentNorm),
        ("routed_expert_up_proj", OperandRole::MoeLatentUpProj),
    ] {
        // Relative to the decoder stack, which is how the classifier is
        // asked and how the closure defects above name their tensors.
        let tensor = format!("{LAYER}.block_sparse_moe.{spelling}.weight");
        assert_eq!(
            classify_stack_tensor(&tensor),
            Some((LAYER, want)),
            "{tensor}"
        );
    }
}
