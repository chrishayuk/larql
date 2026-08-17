//! How far a declared fact travels — the VINDEX3-boundary authority gate.
//!
//! The inventory answers *did a parser read this key?*
//! ([`KeyStatus::Consumed`](larql_models::inventory::KeyStatus::Consumed)).
//! The plan used to treat that answer as *can VINDEX3 represent this
//! fact?* — a different question about a different object, and the gap
//! between them is silent by construction: a fact the parser reads into
//! `ModelConfig` and VINDEX3 then drops looks fully covered from the
//! plan's side.
//!
//! GPT-OSS is the witness. It declares `rope_scaling = {rope_type:
//! "yarn", factor: 32}` for a 131k context. Every one of those leaves
//! classifies `consumed` — the parser genuinely reads them. But
//! [`PositionPolicy`] expresses `Rope { theta } | None` and nothing
//! else, and no other field under `format/vindex3/` carries a scaling
//! block, so the model would plan, encode and execute as **plain rope at
//! θ=150000**, with the plan reporting no defect at all. (VINDEX1/2 do
//! carry it, as raw JSON — so this is a regression the older path does
//! not have.)
//!
//! ```text
//! config.json fact
//!    ↓  parsed        larql-models' parser stored it in ModelConfig
//!    ↓  represented   the VINDEX3 system graph persists it
//!    ↓  lowered       it reaches the generic op plan as an op parameter
//!    ↓  executed      an executor reads that op parameter
//! ```
//!
//! Each execution-semantic key needs a [`CarriageRule`] declaring which
//! of those stages it reaches. Rules claiming [`Carriage::Represented`]
//! or deeper carry a **probe** that reads the value back off the built
//! graph, so the claim is checked against the schema rather than
//! trusted; a probe that disagrees with the declaration blocks. Rules
//! that honestly stop at [`Carriage::Parsed`] must say why, and are
//! reported rather than hidden. A key with **no rule at all** blocks —
//! that is the state this module exists to abolish.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::super::graph::Component;

/// How far a declared fact travels from `config.json` into execution.
///
/// Ordered: a deeper stage implies every shallower one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Carriage {
    /// A registered parser read the key into `ModelConfig`. This is what
    /// the inventory's `consumed` status means, and on its own it is not
    /// evidence of anything downstream.
    Parsed,
    /// The VINDEX3 system graph persists it: a container round-trips the
    /// fact, so encoding does not lose it.
    Represented,
    /// It reaches the generic op plan as an op parameter, so a backend
    /// receives it rather than re-deriving it.
    Lowered,
    /// An executor reads that op parameter on the path under test.
    Executed,
}

impl Carriage {
    /// The stage name as the report prints it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Parsed => "parsed",
            Self::Represented => "represented",
            Self::Lowered => "lowered",
            Self::Executed => "executed",
        }
    }
}

/// What VINDEX3 claims about one execution-semantic config leaf, and the
/// means of checking the claim.
pub struct CarriageRule {
    /// Flattened config leaf name this rule governs (`rope_type`), matched
    /// after the container path — `text_config.rope_parameters.rope_type`
    /// and `rope_scaling.rope_type` share one rule, because they are the
    /// same fact under two spellings.
    pub leaf: &'static str,
    /// The deepest stage VINDEX3 carries this fact to.
    pub reaches: Carriage,
    /// Where in the schema it lands (or why it stops), printed in the
    /// finding so a reader never has to grep for the answer.
    pub site: &'static str,
    /// Reads the carried value back off the built component. `None` when
    /// the component cannot answer (no surface, no attention table); the
    /// gate then reports carriage without a value comparison rather than
    /// inventing a disagreement.
    ///
    /// Required for [`Carriage::Represented`] and deeper, and unused for
    /// [`Carriage::Parsed`] — a rule that stops at the parser has nothing
    /// to read back.
    pub probe: Option<fn(&Component) -> Option<Value>>,
}

/// The rules. Every leaf classified
/// [`ExecutionSemantic`](super::report::SemanticClass::ExecutionSemantic)
/// must appear here or block.
///
/// Adding a key here is a claim about the VINDEX3 schema, not about the
/// parser — which is the whole point of the module.
pub const CARRIAGE_RULES: &[CarriageRule] = &[
    // ── Position ────────────────────────────────────────────────────
    CarriageRule {
        leaf: "rope_theta",
        reaches: Carriage::Lowered,
        site: "Component.attention[].position (PositionPolicy::Rope) → AttentionOp.position",
        probe: Some(probe_rope_theta),
    },
    CarriageRule {
        leaf: "layer_rope_theta",
        reaches: Carriage::Lowered,
        site: "Component.attention[].position, per layer → AttentionOp.position",
        probe: Some(probe_layer_rope_theta),
    },
    CarriageRule {
        leaf: "rope_type",
        reaches: Carriage::Represented,
        // PositionPolicy is `Rope { theta } | Yarn { theta, scaling } |
        // None`: unscaled rotary, YaRN-scaled rotary (frequencies AND the
        // attention amplitude), or no position encoding. Any other declared
        // rope class (llama3, dynamic, ...) still has no variant and
        // mismatches here — represented, not lowered: the interpreter and
        // the lowering refuse a YaRN layer until A-9.3/A-9.4 execute it.
        site: "Component.attention[].position (PositionPolicy::Rope | Yarn)",
        probe: Some(probe_rope_type),
    },
    // The YaRN block's own leaves, each carried on `PositionPolicy::Yarn`
    // and answered from it. A checkpoint that declares them without
    // declaring `rope_type: yarn` gets no answer, which is right — the
    // leaves mean nothing outside that block.
    CarriageRule {
        leaf: "factor",
        reaches: Carriage::Represented,
        site: "Component.attention[].position (PositionPolicy::Yarn.scaling.factor)",
        probe: Some(probe_yarn_factor),
    },
    CarriageRule {
        leaf: "beta_fast",
        reaches: Carriage::Represented,
        site: "Component.attention[].position (PositionPolicy::Yarn.scaling.beta_fast)",
        probe: Some(probe_yarn_beta_fast),
    },
    CarriageRule {
        leaf: "beta_slow",
        reaches: Carriage::Represented,
        site: "Component.attention[].position (PositionPolicy::Yarn.scaling.beta_slow)",
        probe: Some(probe_yarn_beta_slow),
    },
    CarriageRule {
        leaf: "truncate",
        reaches: Carriage::Represented,
        site: "Component.attention[].position (PositionPolicy::Yarn.scaling.truncate)",
        probe: Some(probe_yarn_truncate),
    },
    CarriageRule {
        leaf: "original_max_position_embeddings",
        reaches: Carriage::Represented,
        site: "Component.attention[].position (PositionPolicy::Yarn.scaling.original_max_position_embeddings)",
        probe: Some(probe_yarn_original_max),
    },
    // ── Span policy ─────────────────────────────────────────────────
    CarriageRule {
        leaf: "layer_types",
        reaches: Carriage::Lowered,
        site: "Component.attention[].span → AttentionOp.span",
        probe: Some(probe_layer_types),
    },
    CarriageRule {
        leaf: "sliding_window",
        reaches: Carriage::Lowered,
        site: "Component.attention[].window → AttentionOp.window",
        probe: Some(probe_sliding_window),
    },
    // ── Norms ───────────────────────────────────────────────────────
    CarriageRule {
        leaf: "rms_norm_eps",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.norm.pre.eps → NormOp.eps",
        probe: Some(probe_pre_norm_eps),
    },
    CarriageRule {
        leaf: "layer_norm_eps",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.norm.pre.eps → NormOp.eps",
        probe: Some(probe_pre_norm_eps),
    },
    CarriageRule {
        leaf: "norm_epsilon",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.norm.pre.eps → NormOp.eps",
        probe: Some(probe_pre_norm_eps),
    },
    CarriageRule {
        leaf: "post_norm_eps",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.norm.post.eps → NormOp.eps at the post sites",
        probe: Some(probe_post_norm_eps),
    },
    // ── FFN ─────────────────────────────────────────────────────────
    CarriageRule {
        leaf: "hidden_act",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.ffn.activation → FfnOp.activation",
        probe: Some(probe_activation),
    },
    CarriageRule {
        leaf: "hidden_activation",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.ffn.activation → FfnOp.activation",
        probe: Some(probe_activation),
    },
    CarriageRule {
        leaf: "swiglu_limit",
        reaches: Carriage::Represented,
        // GPT-OSS's clamped GLU: `gate.min(limit)`, `up.clamp(±limit)`,
        // `(up + 1) * gate * sigmoid(alpha * gate)`. Carried as a gate
        // *policy* rather than an activation variant, and judged here by
        // the limit it carries. Represented, not lowered: the interpreter
        // and the lowering refuse a ClampedGlu FFN until A-9.3/A-9.4.
        site: "ExecutionSurface.ffn.gate_policy (ExpertGatePolicy::ClampedGlu.limit) → FfnOp.gate_policy",
        probe: Some(probe_swiglu_limit),
    },
    // ── Attention/output scaling ────────────────────────────────────
    CarriageRule {
        leaf: "qk_scale_factor",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.attention.query_scale → AttentionOp.query_scale",
        probe: Some(probe_query_scale),
    },
    CarriageRule {
        leaf: "query_pre_attn_scalar",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.attention.score_scale → AttentionOp.score_scale",
        probe: Some(probe_score_scale),
    },
    CarriageRule {
        leaf: "attn_logit_softcapping",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.attention.logit_softcapping → AttentionOp.logit_softcapping",
        probe: Some(probe_attn_softcap),
    },
    CarriageRule {
        leaf: "final_logit_softcapping",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.head.final_logit_softcapping → OutputOp.softcapping",
        probe: Some(probe_final_softcap),
    },
    CarriageRule {
        leaf: "output_multiplier",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.head.output_multiplier → OutputOp.multiplier",
        probe: Some(probe_output_multiplier),
    },
    // ── Facts that stop at the parser, reviewed ─────────────────────
    CarriageRule {
        leaf: "attention_bias",
        reaches: Carriage::Parsed,
        // VINDEX3 has no `attention_bias` field; what it has instead is
        // operand closure, which refuses any bias tensor it cannot
        // classify into a declared op. For a model that declares `false`
        // the two agree trivially. For one that declares `true` the bias
        // operands themselves block at G5b — a stronger check than a
        // boolean, and the reason this is judged rather than a hole.
        // MOE1 gives the projections explicit bias operands.
        site: "no schema field — carried instead as operand evidence, gated by G5b closure",
        probe: None,
    },
    CarriageRule {
        leaf: "max_position_embeddings",
        reaches: Carriage::Parsed,
        // A serving/KV-allocation bound, not a forward-pass semantic: no
        // op reads it, and two checkpoints differing only here compute
        // identical logits for any prompt both can hold. Recorded so the
        // absence is a judgement on the report rather than a silence.
        site: "no schema field — a KV-allocation bound, read by no generic op",
        probe: None,
    },
];

/// The rule governing a config leaf, if any.
pub fn rule_for(leaf: &str) -> Option<&'static CarriageRule> {
    CARRIAGE_RULES.iter().find(|rule| rule.leaf == leaf)
}

// ── Probes ──────────────────────────────────────────────────────────
//
// Each reads what the *built graph* holds, so a rule's claim is checked
// against the schema rather than believed. They return `None` when the
// component has no surface or table to answer from.

/// The uniform rope base across the attention table, when there is one.
/// A per-layer split (Muse-Glimmer's `layer_rope_theta`) answers `None`
/// here and is checked by [`probe_layer_rope_theta`] instead.
fn probe_rope_theta(component: &Component) -> Option<Value> {
    let table = component.attention.as_ref()?;
    let mut thetas = table.iter().filter_map(|l| l.position.rope_theta());
    let first = thetas.next()?;
    thetas.all(|t| t == first).then(|| json!(first))
}

/// Every layer's rope base in layer order, with NoPE layers as `0` —
/// the same sentinel spelling the checkpoints use.
fn probe_layer_rope_theta(component: &Component) -> Option<Value> {
    let table = component.attention.as_ref()?;
    Some(Value::Array(
        table
            .iter()
            .map(|l| json!(l.position.rope_theta().unwrap_or(0.0)))
            .collect(),
    ))
}

/// The rope *class* the table actually carries, in the checkpoint's own
/// spelling: `yarn` when any rotating layer holds a YaRN block, else
/// `default`. A table can not mix the two — the block is a checkpoint-wide
/// fact — so the first rotating layer answers for all.
fn probe_rope_type(component: &Component) -> Option<Value> {
    let table = component.attention.as_ref()?;
    let scaled = table.iter().any(|l| l.position.yarn().is_some());
    Some(json!(if scaled { "yarn" } else { "default" }))
}

/// The YaRN block the table carries, when it carries one. `None` when the
/// table has no scaled layer — the caller's leaf then has nothing to be
/// judged against, which is the right answer for a checkpoint that
/// declares the leaf outside a `yarn` block.
fn yarn_block(component: &Component) -> Option<larql_models::YarnRopeScaling> {
    component
        .attention
        .as_ref()?
        .iter()
        .find_map(|l| l.position.yarn())
}

fn probe_yarn_factor(component: &Component) -> Option<Value> {
    Some(json!(yarn_block(component)?.factor))
}

fn probe_yarn_beta_fast(component: &Component) -> Option<Value> {
    Some(json!(yarn_block(component)?.beta_fast))
}

fn probe_yarn_beta_slow(component: &Component) -> Option<Value> {
    Some(json!(yarn_block(component)?.beta_slow))
}

fn probe_yarn_truncate(component: &Component) -> Option<Value> {
    Some(json!(yarn_block(component)?.truncate))
}

fn probe_yarn_original_max(component: &Component) -> Option<Value> {
    Some(json!(
        yarn_block(component)?.original_max_position_embeddings
    ))
}

/// Per-layer span kinds in the checkpoint's own vocabulary, so the
/// comparison is against the declared spelling rather than a rendering
/// this probe invents.
fn probe_layer_types(component: &Component) -> Option<Value> {
    let table = component.attention.as_ref()?;
    Some(Value::Array(
        table
            .iter()
            .map(|l| json!(l.span.declared_name()))
            .collect(),
    ))
}

/// The uniform sliding window across sliding layers, when there is one.
fn probe_sliding_window(component: &Component) -> Option<Value> {
    let table = component.attention.as_ref()?;
    let mut windows = table.iter().filter_map(|l| l.window);
    let first = windows.next()?;
    windows.all(|w| w == first).then(|| json!(first))
}

fn probe_pre_norm_eps(component: &Component) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.norm.pre.eps))
}

fn probe_post_norm_eps(component: &Component) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.norm.post?.eps))
}

fn probe_activation(component: &Component) -> Option<Value> {
    let activation = component.execution.as_ref()?.ffn.activation;
    serde_json::to_value(activation).ok()
}

/// The clamp bound the FFN surface carries, when its gate policy is the
/// clamped GLU. A plain-gated surface has no limit to answer with — a
/// checkpoint declaring `swiglu_limit` that resolved to plain gating is
/// then reported as unrepresented, which is the truth.
fn probe_swiglu_limit(component: &Component) -> Option<Value> {
    match component.execution.as_ref()?.ffn.gate_policy {
        larql_models::ExpertGatePolicy::ClampedGlu { limit, .. } => Some(json!(limit)),
        larql_models::ExpertGatePolicy::Gated => None,
    }
}

fn probe_query_scale(component: &Component) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.attention.query_scale?))
}

fn probe_score_scale(component: &Component) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.attention.score_scale))
}

fn probe_attn_softcap(component: &Component) -> Option<Value> {
    Some(json!(
        component.execution.as_ref()?.attention.logit_softcapping?
    ))
}

fn probe_final_softcap(component: &Component) -> Option<Value> {
    Some(json!(
        component
            .execution
            .as_ref()?
            .head
            .as_ref()?
            .final_logit_softcapping?
    ))
}

fn probe_output_multiplier(component: &Component) -> Option<Value> {
    Some(json!(
        component
            .execution
            .as_ref()?
            .head
            .as_ref()?
            .output_multiplier?
    ))
}
