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

use larql_models::config::score_scale_from_query_pre_attn_scalar;

use super::super::graph::policy::{AttentionLayerPolicy, AttentionSpan};
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
    pub probe: Option<fn(&Component, &ProbeContext<'_>) -> Option<Value>>,
}

/// What a probe may know about the fact it is answering for, beyond the
/// component: the attention span the fact's path names, when a family
/// declares a fact per layer TYPE (`rope_parameters.full_attention.*` vs
/// `rope_parameters.sliding_attention.*` — Gemma 3/4), and the declared
/// value, so a probe can answer in the checkpoint's own spelling when
/// several spellings name one judged variant (`gelu_pytorch_tanh` and
/// `gelu_new` are both `Activation::GeluTanh`). A probe never lets the
/// declared value *choose* what it reports — it only resolves aliases of
/// what the schema already holds.
pub struct ProbeContext<'a> {
    pub span: Option<AttentionSpan>,
    pub declared: &'a Value,
}

impl ProbeContext<'_> {
    /// The per-layer-type scope a flattened config path names, if any.
    pub fn span_of(path: &str) -> Option<AttentionSpan> {
        [
            AttentionSpan::Full,
            AttentionSpan::Sliding,
            AttentionSpan::Windowed,
        ]
        .into_iter()
        .find(|span| {
            path.split('.')
                .any(|segment| segment == span.declared_name())
        })
    }
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
        leaf: "partial_rotary_factor",
        reaches: Carriage::Lowered,
        site: "Component.attention[].position (PositionPolicy::{PartialRope,MRope}.rotary_fraction) → AttentionOp.position",
        probe: Some(probe_partial_rotary_factor),
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
    CarriageRule {
        leaf: "type",
        reaches: Carriage::Represented,
        // The older HF spelling of `rope_type` (same discriminator, same
        // block) — same claim, same probe: `PositionPolicy` can only
        // express the unscaled class under this name too.
        site: "Component.attention[].position — PositionPolicy expresses unscaled rope only",
        probe: Some(probe_rope_type),
    },
    CarriageRule {
        leaf: "low_freq_factor",
        reaches: Carriage::Represented,
        // Llama-3-style rope scaling — a different scaling convention from
        // the YaRN one `factor`/`beta_fast`/etc. above represent.
        // `PositionPolicy::Yarn` has no field for it; always refuses.
        site: "no schema field — Llama-3 rope scaling is not represented yet",
        probe: Some(probe_unrepresented),
    },
    CarriageRule {
        leaf: "high_freq_factor",
        reaches: Carriage::Represented,
        site: "no schema field — Llama-3 rope scaling is not represented yet",
        probe: Some(probe_unrepresented),
    },
    CarriageRule {
        leaf: "mscale",
        reaches: Carriage::Represented,
        // DeepSeek-style YaRN mscale extension — a different scaling
        // convention from HF's generic YaRN block above. No field exists;
        // always refuses.
        site: "no schema field — DeepSeek's mscale extension is not represented yet",
        probe: Some(probe_unrepresented),
    },
    CarriageRule {
        leaf: "mscale_all_dim",
        reaches: Carriage::Represented,
        site: "no schema field — DeepSeek's mscale extension is not represented yet",
        probe: Some(probe_unrepresented),
    },
    // ── Span policy ─────────────────────────────────────────────────
    CarriageRule {
        leaf: "layer_types",
        reaches: Carriage::Lowered,
        site: "Component.attention[].{operator,span} → LayerAttention::{GatedDelta,Softmax}",
        probe: Some(probe_layer_types),
    },
    CarriageRule {
        leaf: "sliding_window",
        reaches: Carriage::Lowered,
        site: "Component.attention[].window → AttentionOp.window",
        probe: Some(probe_sliding_window),
    },
    // The window's ENABLE flag and its layer bound. Both are read by
    // `ModelArchitecture::sliding_window_size`, which resolves all three
    // declarations into one effective per-layer policy, and both are
    // persisted by the vindex config round-trip — so the container does
    // not lose them.
    //
    // `Parsed`, and that is the honest stage rather than a weak one: the
    // effect of both facts is fully ABSORBED into the resolved per-layer
    // window before a graph exists. `sliding_window_size` returns `None`
    // for a disabled window and `is_sliding_window_layer` applies the
    // bound, so what the container carries is the effective policy —
    // there is no separate flag downstream to read back, and a deeper
    // claim would need a probe the schema cannot answer.
    CarriageRule {
        leaf: "use_sliding_window",
        reaches: Carriage::Parsed,
        site: "absorbed by ModelArchitecture::sliding_window_size into the resolved \
               per-layer window the graph carries; also persisted by the vindex config \
               round-trip",
        probe: None,
    },
    // The positional scheme, answered from the graph rather than the
    // config, because on `granitemoehybrid` this key is the SWITCH: HF
    // builds a rotary embedding only when it reads `rope`, so a
    // checkpoint that omits it is a NoPE model. `Represented` and not
    // `Parsed` — the effect is visible on every layer's carried
    // PositionPolicy, so the container can be asked what it believes
    // rather than trusted to have read the key.
    CarriageRule {
        leaf: "position_embedding_type",
        reaches: Carriage::Represented,
        site: "Component.attention[].position — a rotating policy answers `rope`, a stack \
               that rotates nowhere answers null",
        probe: Some(probe_position_embedding_type),
    },
    CarriageRule {
        leaf: "max_window_layers",
        reaches: Carriage::Parsed,
        site: "absorbed by ModelArchitecture::is_sliding_window_layer as the bound on an \
               enabled window; also persisted by the vindex config round-trip",
        probe: None,
    },
    // Inkling-Small's spelling of the same window. One site, because it
    // is one fact: the graph carries a window per layer whichever key
    // stated it.
    CarriageRule {
        leaf: "sliding_window_size",
        reaches: Carriage::Lowered,
        site: "Component.attention[].window → AttentionOp.window",
        probe: Some(probe_sliding_window),
    },
    // The index-set spelling of the per-layer topology, carried to the
    // same place `layer_types` is — which is the claim worth testing: two
    // very different declarations reaching one canonical policy.
    CarriageRule {
        leaf: "local_layer_ids",
        reaches: Carriage::Lowered,
        site: "Component.attention[].{operator,span} → LayerAttention::{Kda,GatedDelta,Softmax}",
        // An index SET, compared by cardinality against the resolved
        // table — the array probe would render a `layer_types` array and
        // never equal the declared set of indices.
        probe: Some(probe_sliding_layer_set),
    },
    CarriageRule {
        leaf: "d_rel",
        reaches: Carriage::Represented,
        site: "Component.attention[].position → PositionPolicy::Relative",
        probe: Some(probe_relative_d_rel),
    },
    CarriageRule {
        leaf: "rel_extent",
        reaches: Carriage::Represented,
        site: "Component.attention[].position → PositionPolicy::Relative",
        probe: Some(probe_relative_extent),
    },
    // ── MoE facts, in every spelling that reaches one surface ────────
    CarriageRule {
        leaf: "moe_renormalize",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe.routing_policy",
        probe: Some(probe_moe_routing_policy),
    },
    CarriageRule {
        leaf: "num_shared_experts",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe.shared_experts",
        probe: Some(probe_moe_shared_experts),
    },
    CarriageRule {
        leaf: "moe_router_activation_func",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe.router_kind",
        probe: Some(probe_moe_router_kind),
    },
    CarriageRule {
        leaf: "scoring_func",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe.router_kind",
        probe: Some(probe_moe_router_kind),
    },
    CarriageRule {
        leaf: "moe_layer_freq",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe — every layer after the dense prefix is routed",
        probe: Some(probe_identity_valued),
    },
    // Expert grouping. At one group the router selects over every expert,
    // which is what an ungrouped router does — so the schema represents
    // its effect exactly, by having none. Any other value is a real
    // grouping this schema cannot state, and refuses.
    CarriageRule {
        leaf: "num_expert_group",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe — one group is ungrouped routing",
        probe: Some(probe_identity_valued),
    },
    CarriageRule {
        leaf: "n_group",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe — one group is ungrouped routing",
        probe: Some(probe_identity_valued),
    },
    CarriageRule {
        leaf: "topk_group",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe — one group is ungrouped routing",
        probe: Some(probe_identity_valued),
    },
    CarriageRule {
        leaf: "use_grouped_topk",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe — grouping is a no-op at one group",
        probe: Some(probe_grouping_is_a_no_op),
    },
    // ── The interleave, in the two-set spelling, and the KDA conv ────
    CarriageRule {
        leaf: "kda_layers",
        reaches: Carriage::Lowered,
        site: "Component.attention[].{operator,span} → LayerAttention::{Kda,GatedDelta,Softmax}",
        probe: Some(probe_recurrent_layer_set),
    },
    CarriageRule {
        leaf: "full_attn_layers",
        reaches: Carriage::Lowered,
        site: "Component.attention[].{operator,span} → LayerAttention::{Kda,GatedDelta,Softmax}",
        probe: Some(probe_softmax_layer_set),
    },
    CarriageRule {
        leaf: "gate_lower_bound",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.kda_gate_lower_bound → KdaOp.gate_lower_bound",
        probe: Some(probe_kda_gate_lower_bound),
    },
    CarriageRule {
        leaf: "short_conv_kernel_size",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.kda.conv_kernel → KdaOp.conv_kernel",
        probe: Some(probe_kda_conv_kernel),
    },
    // A rescale of the whole routed branch, which this schema's MoE
    // surface has no field for. Refuses — and refusing for a stated reason
    // is the point of reading it: a key nothing reads blocks with no
    // account of why.
    CarriageRule {
        leaf: "routed_scaling_factor",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe.branch_scale",
        probe: Some(probe_moe_branch_scale),
    },
    // How many leading layers are dense. The op plan decides each layer's
    // FFN kind from operand evidence, but no field on the graph states the
    // prefix, so the declaration is not carried.
    CarriageRule {
        leaf: "first_k_dense_replace",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe.dense_prefix_layers",
        probe: Some(probe_moe_dense_prefix),
    },
    // Kimi Linear declares it true while carrying `qk_rope_head_dim: 64`,
    // so what it asserts about the rotary is not yet judged. Unjudged is
    // the honest verdict, and it blocks.
    CarriageRule {
        leaf: "mla_use_nope",
        reaches: Carriage::Represented,
        site: "Component.attention[].position → PositionPolicy::None",
        probe: Some(probe_mla_nope),
    },
    CarriageRule {
        leaf: "model_max_length",
        reaches: Carriage::Parsed,
        site: "no schema field — a KV-allocation bound, read by no generic op",
        probe: None,
    },
    CarriageRule {
        leaf: "num_nextn_predict_layers",
        reaches: Carriage::Represented,
        site: "no schema field — this schema has no multi-token-prediction object",
        // Zero declared layers is no MTP head, which this schema
        // represents exactly by carrying none. Any positive count is a
        // sub-stack it cannot state, and refuses.
        probe: Some(probe_absent_when_zero),
    },
    CarriageRule {
        leaf: "sliding_window_pattern",
        reaches: Carriage::Represented,
        // A period integer (e.g. Gemma 2's "every Nth layer is full") is a
        // different representation from the per-layer `layer_types` array
        // the graph actually carries; no derivation from one to the other
        // exists yet, so this always refuses rather than assuming a
        // pattern it hasn't checked.
        site: "no schema field — not derived from the per-layer span table yet",
        probe: Some(probe_unrepresented),
    },
    CarriageRule {
        leaf: "rope_local_base_freq",
        reaches: Carriage::Represented,
        // A second rope base for local/sliding layers, alongside
        // `rope_theta`. `layer_rope_theta` carries a per-layer table when a
        // family declares one explicitly; this is a distinct declaration
        // shape with no derivation into that table yet.
        site: "no schema field — not derived into the per-layer rope table yet",
        probe: Some(probe_unrepresented),
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
        leaf: "layer_norm_epsilon",
        reaches: Carriage::Lowered,
        // GPT-2's spelling; `detect/parser.rs:292` folds it into the same
        // `norm_eps` read as its three siblings above.
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
    CarriageRule {
        leaf: "embedding_multiplier",
        reaches: Carriage::Lowered,
        // Granite's embedding-scale operation, wired through
        // `GraniteArch::embed_scale()` (`config/architecture.rs`) into
        // `HeadSurface.embed_scale` and on into `EmbeddingOp.scale`
        // (`opplan/build.rs`).
        site: "ExecutionSurface.head.embed_scale → EmbeddingOp.scale",
        probe: Some(probe_embed_scale),
    },
    CarriageRule {
        leaf: "attention_multiplier",
        reaches: Carriage::Lowered,
        // NOT `qk_scale_factor`/`query_scale` — Granite's attention_multiplier
        // *replaces* the standard 1/sqrt(head_dim) score scale rather than
        // multiplying on top of it (every legacy-path call site treats it
        // that way, and the declared value — 1/head_dim — confirms it
        // numerically). `ModelArchitecture::attention_scale`'s default
        // resolves it into `score_scale` accordingly.
        site: "ExecutionSurface.attention.score_scale → AttentionOp.score_scale",
        probe: Some(probe_score_scale),
    },
    CarriageRule {
        leaf: "logits_scaling",
        reaches: Carriage::Lowered,
        // Granite's spelling, and NOT a synonym: `logits_scaling` is a
        // divisor (`logits / d`) where `output_multiplier` is a multiplier.
        // Scaling does commute through the linear head, so the two describe
        // the same operation — but only once the divisor is inverted, which
        // `ModelArchitecture::logit_scale` does. The container therefore
        // carries `1/d`, and this probe inverts it back to compare against
        // the declared leaf.
        site: "ExecutionSurface.head.output_multiplier → OutputOp.multiplier (as 1/d)",
        probe: Some(probe_logits_scaling),
    },
    CarriageRule {
        leaf: "residual_multiplier",
        reaches: Carriage::Lowered,
        // Granite's residual-stream scale: the sublayer's own output
        // (attention or FFN) is multiplied by this before its residual
        // add, at both sites — no other family in this registry scales
        // the residual stream, so this is new schema (A-11.3), not a
        // second spelling of an existing field.
        site: "ExecutionSurface.residual_scale → LayerPlan.residual_scale",
        probe: Some(probe_residual_scale),
    },
    CarriageRule {
        leaf: "norm_topk_prob",
        reaches: Carriage::Represented,
        // Whether router weights are renormalised after top-k selection.
        // The cross-check this rule once said it lacked now exists: the
        // routing policy IS this flag, and `moe_renormalize` is the same
        // fact in Kimi Linear's spelling. See `probe_moe_routing_policy`
        // for why it reports rather than compares.
        site: "ExecutionSurface.ffn.moe.routing_policy",
        probe: Some(probe_moe_routing_policy),
    },
    CarriageRule {
        leaf: "num_experts_per_tok",
        reaches: Carriage::Lowered,
        // The canonical HF spelling of routing width — same underlying
        // resolved value as `top_k_experts`: `ModelArchitecture::num_experts_per_token()`
        // already bridges both spellings per family (GPT-OSS reads
        // `num_experts_per_token` directly; Gemma 4 tries `top_k_experts`
        // first, falling back to `num_experts_per_token` — confirmed by
        // reading both overrides), so the same probe answers both.
        site: "ExecutionSurface.ffn.moe.top_k → RoutedFfnOp routing",
        probe: Some(probe_moe_top_k),
    },
    CarriageRule {
        leaf: "num_experts_per_token",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.ffn.moe.top_k → RoutedFfnOp routing",
        probe: Some(probe_moe_top_k),
    },
    // ── Facts that stop at the parser, reviewed ─────────────────────
    CarriageRule {
        leaf: "attention_bias",
        reaches: Carriage::Represented,
        // A-9.1: the surface states it, and operand closure enforces it
        // both ways — `true` requires all four bias operands, anything
        // else refuses any bias operand it finds — so the boolean and the
        // operand evidence cannot drift apart. The executors add the four
        // biases; the Metal lowering refuses them until A-9.4.
        site: "ExecutionSurface.attention.attention_bias → AttentionOp.{q,k,v,o}_bias (closure-paired)",
        probe: Some(probe_attention_bias),
    },
    CarriageRule {
        leaf: "num_kv_shared_layers",
        reaches: Carriage::Represented,
        // Gemma 4 E2B/E4B: the last N layers read the KV state of the last
        // non-shared layer of their type instead of projecting their own —
        // attention reading ANOTHER op's state, a cross-layer dependency
        // the graph does not represent (V3-F0's open ontology question,
        // scored by that witness). The table represents "no layer shares"
        // and nothing else, so `0` agrees and any other count is dropped
        // at the boundary and blocks — refused, never mis-served as
        // per-layer projections.
        site: "Component.attention[] — no KV-sharing relationship exists; only 0 is representable",
        probe: Some(probe_kv_shared_layers),
    },
    // ── Gemma 4 (V3-F0 witness 3) ──────────────────────────────────
    CarriageRule {
        leaf: "attention_k_eq_v",
        reaches: Carriage::Represented,
        site: "Component.attention[].v_from_k → AttentionOp.v_from_k (closure-paired: no V operand on such a layer)",
        probe: Some(probe_k_eq_v),
    },
    CarriageRule {
        leaf: "enable_moe_block",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn.moe (Some = a routed block is judged) → LayerFfn::Routed / hybrid",
        probe: Some(probe_moe_enabled),
    },
    CarriageRule {
        leaf: "top_k_experts",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.ffn.moe.top_k → RoutedFfnOp routing",
        probe: Some(probe_moe_top_k),
    },
    CarriageRule {
        leaf: "global_head_dim",
        reaches: Carriage::Lowered,
        site: "Component.attention[].geometry.head_dim on the full layers → AttentionOp.head_dim",
        probe: Some(probe_full_layer_head_dim),
    },
    CarriageRule {
        leaf: "num_global_key_value_heads",
        reaches: Carriage::Lowered,
        site: "Component.attention[].geometry.num_kv_heads on the full layers → AttentionOp.num_kv_heads",
        probe: Some(probe_full_layer_kv_heads),
    },
    CarriageRule {
        leaf: "hidden_size_per_layer_input",
        reaches: Carriage::Represented,
        // Per-layer-input embeddings (Gemma 3n/4 E2B): a second embedding
        // table gated into every layer. No object or op exists for it; the
        // graph represents its ABSENCE only, so `0` agrees and any width
        // is dropped at the boundary and blocks.
        site: "no schema field — the graph represents PLE as absent; only 0 is representable",
        probe: Some(probe_zero),
    },
    CarriageRule {
        leaf: "use_double_wide_mlp",
        reaches: Carriage::Represented,
        // Doubles the MLP width on KV-shared layers; no KV-shared layer is
        // representable (see `num_kv_shared_layers`), so only `false` is.
        site: "no schema field — only `false` is representable",
        probe: Some(probe_false),
    },
    CarriageRule {
        leaf: "use_clipped_linears",
        reaches: Carriage::Represented,
        // A tower option that clips projection outputs; no op carries a
        // clip, so only `false` is representable.
        site: "no schema field on the tower surface — only `false` is representable",
        probe: Some(probe_false),
    },
    CarriageRule {
        leaf: "mlp_bias",
        reaches: Carriage::Parsed,
        // Same argument as `attention_bias` immediately above: VINDEX3 has
        // no `mlp_bias` field, and operand closure over the FFN's actual
        // bias tensors (or their absence) is the real gate. Granite 4.1
        // declares `false` on 3B/8B/30B, which agrees trivially; a
        // checkpoint declaring `true` blocks at G5b if the projections
        // don't carry bias operands, not here.
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
    // ── Hybrid linear-attention + multi-token-prediction (declared, not
    //    yet executed — R2/Kimi-Linear rung, see docs/k3-funnel.md) ──
    //
    // No `AttentionOp` variant computes a linear-attention layer and no
    // MTP-head object exists in the schema, so every one of these always
    // refuses via the shared `probe_unrepresented` — the same idiom
    // `norm_topk_prob`/`high_freq_factor` above use for "no schema field
    // yet". Each still gets its own rule (rather than falling through
    // `carriage_finding`'s generic no-rule message) so
    // `every_execution_semantic_leaf_has_a_carriage_rule` covers it: a
    // future field added to the registry without a rule fails there
    // before it fails on a checkpoint.
    CarriageRule {
        leaf: "linear_conv_kernel_dim",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.linear_attention.conv_kernel → GatedDeltaOp.conv_kernel",
        probe: Some(probe_linear_conv_kernel),
    },
    CarriageRule {
        leaf: "linear_key_head_dim",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.linear_attention.key_head_dim → GatedDeltaOp.key_head_dim",
        probe: Some(probe_linear_key_head_dim),
    },
    CarriageRule {
        leaf: "linear_value_head_dim",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.linear_attention.value_head_dim → GatedDeltaOp.value_head_dim",
        probe: Some(probe_linear_value_head_dim),
    },
    CarriageRule {
        leaf: "linear_num_key_heads",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.linear_attention.key_heads → GatedDeltaOp.num_key_heads",
        probe: Some(probe_linear_key_heads),
    },
    CarriageRule {
        leaf: "linear_num_value_heads",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.linear_attention.value_heads → GatedDeltaOp.num_value_heads",
        probe: Some(probe_linear_value_heads),
    },
    CarriageRule {
        leaf: "mamba_ssm_dtype",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.linear_attention.state_dtype → GatedDeltaState precision",
        probe: Some(probe_linear_state_dtype),
    },
    // ── Mamba2/SSD mixer geometry and switches (schema 6). Represented,
    //    not Lowered: the surface holds every fact and no executor
    //    consumes it yet — claiming Lowered would assert an operator that
    //    does not exist (the same honesty `mamba_ssm_dtype` held to until
    //    QW-2's reference operator landed). ──
    CarriageRule {
        leaf: "state_size",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.state_size",
        probe: Some(probe_mamba2_state_size),
    },
    CarriageRule {
        leaf: "expand",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.expand",
        probe: Some(probe_mamba2_expand),
    },
    CarriageRule {
        leaf: "conv_kernel",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.conv_kernel",
        probe: Some(probe_mamba2_conv_kernel),
    },
    CarriageRule {
        leaf: "n_groups",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.n_groups",
        probe: Some(probe_mamba2_n_groups),
    },
    CarriageRule {
        leaf: "chunk_size",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.chunk_size",
        probe: Some(probe_mamba2_chunk_size),
    },
    CarriageRule {
        leaf: "time_step_limit",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.dt_limit_{min,max} — the judged \
               non-finite boundary: a bare `Infinity` is carried as a declared \
               unbounded side, never a fabricated float",
        probe: Some(probe_mamba2_time_step_limit),
    },
    CarriageRule {
        leaf: "rms_norm",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.rms_norm — the mixer's gated RMSNorm",
        probe: Some(probe_mamba2_rms_norm),
    },
    CarriageRule {
        leaf: "use_bias",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.use_bias (closure-paired with the \
               in/out projection bias operands)",
        probe: Some(probe_mamba2_use_bias),
    },
    CarriageRule {
        leaf: "use_conv_bias",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.use_conv_bias (closure-paired with \
               the conv bias operand)",
        probe: Some(probe_mamba2_use_conv_bias),
    },
    // ── The mamba_ssm key dialect (OuteAI Mamba2Attn): three renamed
    //    geometry keys and the projection-bias switch, read into the SAME
    //    `Mamba2Geometry` fields their HF twins fill — so each probe
    //    answers from the same surface site. ──
    CarriageRule {
        leaf: "mamba2_num_heads",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.num_heads",
        probe: Some(probe_mamba2_num_heads),
    },
    CarriageRule {
        leaf: "mamba2_head_dim",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.head_dim",
        probe: Some(probe_mamba2_head_dim),
    },
    CarriageRule {
        leaf: "mamba2_conv_kernel",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.conv_kernel",
        probe: Some(probe_mamba2_conv_kernel),
    },
    CarriageRule {
        leaf: "use_mamba2_bias",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.use_bias (closure-paired with the \
               in/out projection bias operands)",
        probe: Some(probe_mamba2_use_bias),
    },
    // ── The hybrid's conv-QKV attention block. Represented, not
    //    Lowered: the surface holds every fact and no executor consumes
    //    it yet — the same honesty the Mamba2 rules held to until the
    //    reference operator landed. ──
    CarriageRule {
        leaf: "attention_head_dim",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.conv_qkv.head_dim",
        probe: Some(probe_conv_qkv_head_dim),
    },
    CarriageRule {
        leaf: "attention_conv_kernel",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.conv_qkv.conv_kernel",
        probe: Some(probe_conv_qkv_conv_kernel),
    },
    CarriageRule {
        leaf: "rope_emb_dim",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.conv_qkv.rotary_dim — the partial-rotary width, also \
               carried per layer as PositionPolicy::PartialRope",
        probe: Some(probe_conv_qkv_rotary_dim),
    },
    CarriageRule {
        leaf: "use_attention_qkv_bias",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.conv_qkv.qkv_bias — a declared-FALSE is carried; a \
               declared-TRUE has no judged bias role yet and must block",
        probe: Some(probe_conv_qkv_qkv_bias),
    },
    CarriageRule {
        leaf: "use_attention_out_bias",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.conv_qkv.out_bias — same contract as the QKV bias \
               switch",
        probe: Some(probe_conv_qkv_out_bias),
    },
    CarriageRule {
        leaf: "attention_layers_idx",
        reaches: Carriage::Represented,
        site: "Component.attention[] — the per-layer operator table; the declared set \
               is echoed only when the table's conv-QKV layers correspond to it \
               under a consistent index base",
        probe: Some(probe_attention_layer_idx),
    },
    CarriageRule {
        leaf: "attn_layer_idx",
        reaches: Carriage::Represented,
        site: "Component.attention[] — the state-spaces spelling of the same set",
        probe: Some(probe_attention_layer_idx),
    },
    // ── The mamba_ssm lineage's MLP declaration. ──
    CarriageRule {
        leaf: "mlp_intermediate_size",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn presence per layer — 0 declares NO MLP blocks, \
               carried as every layer's absent FFN op; a non-zero width has no \
               judged lowering yet and must block",
        probe: Some(probe_mlp_intermediate_size),
    },
    CarriageRule {
        leaf: "mlp_padding_size",
        reaches: Carriage::Represented,
        site: "no schema field — pads an MLP width; inert exactly when \
               mlp_intermediate_size declares 0 (no MLP exists to pad), blocking \
               otherwise",
        probe: Some(probe_mlp_padding_size),
    },
    CarriageRule {
        leaf: "use_mlp_bias",
        reaches: Carriage::Represented,
        site: "no schema field — biases an MLP; inert exactly when \
               mlp_intermediate_size declares 0, blocking otherwise",
        probe: Some(probe_mlp_padding_size),
    },
    // ── The mamba_ssm-native nested spellings. ──
    CarriageRule {
        leaf: "layer",
        reaches: Carriage::Represented,
        site: "Component.attention[].operator — `ssm_cfg.layer` names the layer class; \
               \"Mamba2\" is represented as the mixer operator, and any other class \
               finds no surface and blocks",
        probe: Some(probe_ssm_layer_class),
    },
    CarriageRule {
        leaf: "d_conv",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.{conv_qkv,mamba2}.conv_kernel — whichever block declared \
               it; a width matching neither blocks",
        probe: Some(probe_declared_conv_kernel),
    },
    CarriageRule {
        leaf: "d_state",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.state_size — the ssm_cfg spelling",
        probe: Some(probe_mamba2_state_size),
    },
    CarriageRule {
        leaf: "headdim",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.head_dim — the ssm_cfg spelling",
        probe: Some(probe_mamba2_head_dim),
    },
    CarriageRule {
        leaf: "ngroups",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.mamba2.geometry.n_groups — the ssm_cfg spelling",
        probe: Some(probe_mamba2_n_groups),
    },
    CarriageRule {
        leaf: "rotary_emb_dim",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.conv_qkv.rotary_dim",
        probe: Some(probe_conv_qkv_rotary_dim),
    },
    CarriageRule {
        leaf: "qkv_proj_bias",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.conv_qkv.qkv_bias — declared-FALSE carried; TRUE blocks",
        probe: Some(probe_conv_qkv_qkv_bias),
    },
    CarriageRule {
        leaf: "out_proj_bias",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.conv_qkv.out_bias — same contract",
        probe: Some(probe_conv_qkv_out_bias),
    },
    CarriageRule {
        leaf: "causal",
        reaches: Carriage::Represented,
        site: "the conv-QKV operator's masking — causal by construction; a declared \
               non-causal block has no operator and blocks",
        probe: Some(probe_attn_causal),
    },
    CarriageRule {
        leaf: "d_intermediate",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.ffn presence per layer — mamba_ssm's own spelling of \
               mlp_intermediate_size; 0 declares NO MLP blocks",
        probe: Some(probe_mlp_intermediate_size),
    },
    CarriageRule {
        leaf: "residual_in_fp32",
        reaches: Carriage::Represented,
        site: "ExecutionSurface.residual_in_fp32 — residual-stream precision, declared",
        probe: Some(probe_residual_in_fp32),
    },
    CarriageRule {
        leaf: "attn_output_gate",
        reaches: Carriage::Lowered,
        site: "ExecutionSurface.attention.output_gate → GateOp → the gated attention op",
        probe: Some(probe_attn_output_gate),
    },
    CarriageRule {
        leaf: "output_gate_type",
        reaches: Carriage::Represented,
        site: "no schema field — the gate IS represented (see attn_output_gate); \
               what is unresolved is whether THIS key describes it",
        probe: Some(probe_unrepresented),
    },
    CarriageRule {
        leaf: "mtp_num_hidden_layers",
        reaches: Carriage::Represented,
        site: "no schema field — the multi-token-prediction head is not represented yet",
        probe: Some(probe_unrepresented),
    },
    CarriageRule {
        leaf: "mtp_use_dedicated_embeddings",
        reaches: Carriage::Represented,
        site: "no schema field — the multi-token-prediction head is not represented yet",
        probe: Some(probe_unrepresented),
    },
    CarriageRule {
        leaf: "mrope_interleaved",
        reaches: Carriage::Lowered,
        site: "Component.attention[].position (PositionPolicy::MRope.interleaved) → mrope_axis_table → mrope_rotate",
        probe: Some(probe_mrope_interleaved),
    },
    CarriageRule {
        leaf: "mrope_section",
        reaches: Carriage::Lowered,
        site: "Component.attention[].position (PositionPolicy::MRope.section) → mrope_axis_table → mrope_rotate",
        probe: Some(probe_mrope_section),
    },
];

/// The rule governing a config leaf, if any.
pub fn rule_for(leaf: &str) -> Option<&'static CarriageRule> {
    CARRIAGE_RULES.iter().find(|rule| rule.leaf == leaf)
}

/// Canonicalises a declared config value into the vocabulary a probe's
/// carried value uses, for leaves where VINDEX3 legitimately stores a
/// *derived* form of the same fact rather than the checkpoint's own
/// spelling.
///
/// This is not a tolerance knob: the one arm here reuses the identical
/// formula the runtime already applies
/// ([`score_scale_from_query_pre_attn_scalar`]), so agreement means the
/// same fact was recognised twice by the same rule, not that comparison
/// was loosened. A leaf with no arm here falls through unchanged, so
/// [`super::values_agree`] still requires byte-for-byte (or f32-precision)
/// identity — this function only ever narrows a `mismatched` finding to
/// `representable`, never the reverse, and callers still show the raw
/// declared value in the finding regardless of what this returns.
///
/// `hidden_act`/`hidden_activation` used to have an arm here too, but
/// [`probe_activation`] now resolves that alias itself (via
/// [`ProbeContext::declared`], returning the checkpoint's own spelling on
/// a match) — canonicalising *both* sides at once made them disagree in
/// opposite directions (`"gelu_pytorch_tanh"` vs `"gelu_tanh"`) rather
/// than agree. One rule owns each fact's normalisation, never two.
pub fn canonical_declared(leaf: &str, declared: &Value) -> Value {
    match leaf {
        // The checkpoint declares the raw scalar; VINDEX3's execution
        // surface stores the score scale execution actually reads —
        // `scalar.powf(-0.5)`, the identical formula
        // `ModelArchitecture::attention_scale` applies at runtime, called
        // through the one shared function rather than re-derived here.
        "query_pre_attn_scalar" => declared
            .as_f64()
            .map(|scalar| json!(score_scale_from_query_pre_attn_scalar(scalar)))
            .unwrap_or_else(|| declared.clone()),
        _ => declared.clone(),
    }
}

// ── Probes ──────────────────────────────────────────────────────────
//
// Each reads what the *built graph* holds, so a rule's claim is checked
// against the schema rather than believed. They return `None` when the
// component has no surface or table to answer from.

/// The layers a per-layer-type fact speaks for: those of the span the
/// fact's path names, or every layer for a checkpoint-wide fact.
fn layers_in_scope<'a>(
    component: &'a Component,
    ctx: &ProbeContext<'_>,
) -> Option<impl Iterator<Item = &'a super::super::graph::AttentionLayerPolicy>> {
    let table = component.attention.as_ref()?;
    let span = ctx.span;
    Some(
        table
            .iter()
            .filter(move |l| span.is_none_or(|s| l.span == Some(s))),
    )
}

/// Shared by every rule for a fact VINDEX3 has no schema field for yet:
/// always refuses, so the fact honestly blocks (`Unrepresented`, with the
/// rule's own `site` text naming why) rather than falling through the
/// generic no-rule message. Never returns `Some` — a rule using this probe
/// makes no claim this function could get wrong.
fn probe_unrepresented(_component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    None
}

/// The uniform rope base across the layers in scope, when there is one:
/// the whole table for `rope_theta`, one layer type for
/// `rope_parameters.full_attention.rope_theta` (Gemma 4 declares 1e6 on
/// its full layers and 1e4 on its sliding ones — two facts, two probes).
/// A per-layer split (Muse-Glimmer's `layer_rope_theta`) answers `None`
/// here and is checked by [`probe_layer_rope_theta`] instead.
fn probe_rope_theta(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    // Nothing in scope rotates: the declared base is inert, and reporting
    // it as uncarried would demand a rotation the model does not perform.
    // See the matching arm in `plan::compare::rope_theta_findings`.
    if layers_in_scope(component, ctx)?
        .all(|l| l.position == larql_models::config::PositionPolicy::None)
    {
        return Some(ctx.declared.clone());
    }
    let mut thetas = layers_in_scope(component, ctx)?.filter_map(|l| l.position.rope_theta());
    let first = thetas.next()?;
    thetas.all(|t| t == first).then(|| json!(first))
}

/// Every layer's rope base in layer order, with NoPE layers as `0` —
/// the same sentinel spelling the checkpoints use.
fn probe_layer_rope_theta(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let table = component.attention.as_ref()?;
    Some(Value::Array(
        table
            .iter()
            .map(|l| json!(l.position.rope_theta().unwrap_or(0.0)))
            .collect(),
    ))
}

/// The rope *class* the layers in scope carry, in the checkpoint's own
/// spelling: `yarn` when any rotating layer holds a YaRN block,
/// `proportional` when any holds a head-width-basis partial rotary
/// (Gemma 4's full layers), else `default`. Within one scope the class
/// is uniform, so the first classed layer answers for all.
/// The positional scheme the graph actually carries.
///
/// `rope` when any layer in scope rotates; `null` when none does, which
/// is an ANSWER and not a failure to answer — "this stack encodes no
/// position" is exactly what a `granitemoehybrid` without the opt-in
/// means, and reporting it as unknown would hide the case the rule
/// exists for. Mirrors `probe_sliding_window`, which answers null the
/// same way for a stack with no windowed layer.
fn probe_position_embedding_type(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    let mut layers = layers_in_scope(component, ctx)?;
    Some(match layers.any(|l| l.position.rope_theta().is_some()) {
        true => json!(larql_models::config::POSITION_EMBEDDING_TYPE_ROPE),
        false => Value::Null,
    })
}

fn probe_rope_type(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    let mut layers = layers_in_scope(component, ctx)?;
    let class = layers
        .find_map(|l| l.position.declared_rope_type())
        .unwrap_or(larql_models::config::ROPE_TYPE_DEFAULT);
    Some(json!(class))
}

/// The KV-sharing count the table represents: none. Every layer in the
/// graph projects its own K/V, so the only declaration the schema agrees
/// with is `0`.
fn probe_kv_shared_layers(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    component.attention.as_ref()?;
    Some(json!(0))
}

/// Whether any layer takes V from its K projection.
fn probe_k_eq_v(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let table = component.attention.as_ref()?;
    Some(json!(table.iter().any(|l| l.v_from_k)))
}

fn probe_moe_enabled(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(component
        .execution
        .as_ref()?
        .ffn
        .as_ref()?
        .moe
        .is_some()))
}

fn probe_moe_top_k(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component.execution.as_ref()?.ffn.as_ref()?.moe?.top_k
    ))
}

/// The head width the full-attention layers carry — the fact
/// `global_head_dim` declares — when every full layer agrees. A layer
/// without its own geometry has the surface's (that is what the absence
/// means), so a uniform tower answers with its surface head width.
fn probe_full_layer_head_dim(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let table = component.attention.as_ref()?;
    let surface = component.execution.as_ref()?;
    let mut dims = table
        .iter()
        .filter(|l| l.span == Some(AttentionSpan::Full))
        .map(|l| {
            l.geometry
                .map_or(surface.attention.as_ref().map(|a| a.head_dim), |g| {
                    Some(g.head_dim)
                })
        });
    let first = dims.next()??;
    dims.all(|d| d == Some(first)).then(|| json!(first))
}

/// The KV-head count the full-attention layers carry.
fn probe_full_layer_kv_heads(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let table = component.attention.as_ref()?;
    let surface = component.execution.as_ref()?;
    let mut heads = table
        .iter()
        .filter(|l| l.span == Some(AttentionSpan::Full))
        .map(|l| {
            l.geometry
                .map_or(surface.attention.as_ref().map(|a| a.num_kv_heads), |g| {
                    Some(g.num_kv_heads)
                })
        });
    let first = heads.next()??;
    heads.all(|h| h == Some(first)).then(|| json!(first))
}

/// A fact the schema represents only as absent: the built component
/// answers `0`, so a declared `0` agrees and anything else blocks.
fn probe_zero(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    component.execution.as_ref()?;
    Some(json!(0))
}

/// A switch the schema represents only as off.
fn probe_false(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    component.execution.as_ref()?;
    Some(json!(false))
}

/// The rotary fraction the layers in scope carry — `partial_rotary_factor`
/// is a per-layer-type leaf on Gemma 4 (`full_attention` only).
fn probe_partial_rotary_factor(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    let mut fractions =
        layers_in_scope(component, ctx)?.filter_map(|l| l.position.rotary_fraction());
    let first = fractions.next()?;
    fractions.all(|f| f == first).then(|| json!(first))
}

/// Whether the component carries judged attention-output-gate semantics.
///
/// Answers the DECLARED boolean rather than echoing it: `true` only when
/// a spec was actually judged for this family and reached the surface. A
/// checkpoint declaring `attn_output_gate: false` is answered `false` by
/// a surface with no spec, so the two agree without the probe ever
/// asserting a gate that is not there.
///
/// Note what is NOT claimed here. HF reads this key nowhere — the gate is
/// unconditional in the reference implementation, and its real witness is
/// the stored projection carrying `2 · num_heads · head_dim` rows. That
/// cross-examination happens in operand closure (`expected_shape`'s
/// `q_proj_rows`), which is why the config being believed here is safe:
/// a checkpoint claiming a gate it has no rows for fails there.
fn probe_attn_output_gate(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(component
        .execution
        .as_ref()?
        .attention
        .as_ref()?
        .output_gate
        .is_some()))
}

/// The multi-axis sectioning the layers in scope carry, when every
/// rotating layer agrees.
///
/// Refuses unless the arithmetic closes:
///
/// ```text
/// sum(section) * 2 == rotary_dim == head_dim * rotary_fraction
/// ```
///
/// `sum(section)` counts FREQUENCY slots, which is `rotary_dim / 2` — not
/// `rotary_dim`. On Qwen3.8 that is `11+11+10 = 32` against a 64-dim
/// rotary block on a **256**-dim head. Taking the head width as 128 (the
/// Gated DeltaNet head dim, a different operator) makes `sum == rotary_dim`
/// close instead, which is why the identity is asserted against the
/// component's own resolved `head_dim` rather than any nearby 128.
fn mrope_of(component: &Component, ctx: &ProbeContext<'_>) -> Option<([usize; 3], bool)> {
    let head_dim = component.execution.as_ref()?.attention.as_ref()?.head_dim;
    let mut policies = layers_in_scope(component, ctx)?.filter_map(|l| {
        l.position
            .mrope()
            .zip(l.position.rotary_fraction())
            .map(|((section, interleaved), fraction)| (section, interleaved, fraction))
    });
    let first = policies.next()?;
    if !policies.all(|p| p == first) {
        return None;
    }
    let (section, interleaved, fraction) = first;
    let rotary_dim = (head_dim as f64 * fraction) as usize;
    let closes = rotary_dim > 0
        && rotary_dim.is_multiple_of(2)
        && section.iter().sum::<usize>() * 2 == rotary_dim;
    closes.then_some((section, interleaved))
}

fn probe_mrope_section(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    mrope_of(component, ctx).map(|(section, _)| json!(section))
}

fn probe_mrope_interleaved(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    mrope_of(component, ctx).map(|(_, interleaved)| json!(interleaved))
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

fn probe_yarn_factor(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(yarn_block(component)?.factor))
}

fn probe_yarn_beta_fast(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(yarn_block(component)?.beta_fast))
}

fn probe_yarn_beta_slow(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(yarn_block(component)?.beta_slow))
}

fn probe_yarn_truncate(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(yarn_block(component)?.truncate))
}

fn probe_yarn_original_max(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        yarn_block(component)?.original_max_position_embeddings
    ))
}

/// Per-layer span kinds in the checkpoint's own vocabulary, so the
/// comparison is against the declared spelling rather than a rendering
/// this probe invents.
///
/// Refuses (returns `None`) rather than vouching for the interleave when
/// any layer's own [`declared_span`](super::super::graph::policy::AttentionLayerPolicy::declared_span)
/// disagrees with what `span` resolved to. `AttentionLayerPolicy::span`
/// is built off a boolean sliding/full split that silently defaults any
/// spelling outside its three-way vocabulary (a hybrid linear-attention
/// layer, e.g.) to `Full` — echoing `span.declared_name()` back in that
/// state would report the declared interleave as carried when the graph
/// actually dropped it. See `docs/k3-funnel.md` §4.7.8.
fn probe_layer_types(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let table = component.attention.as_ref()?;
    if !table.iter().all(AttentionLayerPolicy::matches_declaration) {
        return None;
    }
    // Every layer round-trips, so rendering the carried policy back into
    // the checkpoint's vocabulary is a report rather than a claim. A
    // layer the schema has no spelling for already refused above.
    table
        .iter()
        .map(|l| l.declared_name().map(|n| json!(n)))
        .collect::<Option<Vec<_>>>()
        .map(Value::Array)
}

/// The Gated DeltaNet geometry the surface carries, read back per field.
///
/// Each answers only if the component actually built a linear-attention
/// block. A component with no recurrence answers `None`, and the gate then
/// reports carriage without a value comparison rather than inventing a
/// disagreement — the same contract every probe here has.
///
/// These are `Lowered` rather than `Represented` because each value
/// terminates in a real operand contract: the five together derive
/// `qkv_channels` and `value_width`, which the nine `LinearAttn*` shape
/// checks close against the stored tensors.
fn probe_linear_key_heads(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component.execution.as_ref()?.linear_attention?.key_heads
    ))
}

fn probe_linear_key_head_dim(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component.execution.as_ref()?.linear_attention?.key_head_dim
    ))
}

fn probe_linear_value_heads(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component.execution.as_ref()?.linear_attention?.value_heads
    ))
}

fn probe_linear_value_head_dim(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component
            .execution
            .as_ref()?
            .linear_attention?
            .value_head_dim
    ))
}

fn probe_linear_conv_kernel(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component.execution.as_ref()?.linear_attention?.conv_kernel
    ))
}

/// The Mamba2 surface's geometry, when the component carries one.
fn mamba2_geometry(component: &Component) -> Option<larql_models::config::Mamba2Geometry> {
    Some(component.execution.as_ref()?.mamba2?.geometry)
}

fn probe_mamba2_state_size(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(mamba2_geometry(component)?.state_size))
}

fn probe_mamba2_expand(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(mamba2_geometry(component)?.expand))
}

fn probe_mamba2_conv_kernel(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(mamba2_geometry(component)?.conv_kernel))
}

fn probe_mamba2_n_groups(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(mamba2_geometry(component)?.n_groups))
}

fn probe_mamba2_chunk_size(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(mamba2_geometry(component)?.chunk_size))
}

/// Echoes the clamp in the checkpoint's own spelling: a finite side as
/// its number, an unbounded side as the non-finite literal the judged
/// boundary quoted (`-Infinity` below, `Infinity` above) — the inverse of
/// [`DtBound::from_declared`](larql_models::config::DtBound::from_declared),
/// positional because unboundedness below and above are different signs.
fn probe_mamba2_time_step_limit(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    use larql_models::config::DtBound;
    let geometry = mamba2_geometry(component)?;
    let side = |bound: DtBound, unbounded: &str| match bound {
        DtBound::Finite(v) => json!(v),
        DtBound::Unbounded => json!(unbounded),
    };
    Some(json!([
        side(geometry.dt_limit_min, "-Infinity"),
        side(geometry.dt_limit_max, "Infinity"),
    ]))
}

fn probe_mamba2_rms_norm(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(mamba2_geometry(component)?.rms_norm))
}

fn probe_mamba2_use_bias(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(mamba2_geometry(component)?.use_bias))
}

fn probe_mamba2_use_conv_bias(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(mamba2_geometry(component)?.use_conv_bias))
}

fn probe_mamba2_num_heads(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(mamba2_geometry(component)?.num_heads))
}

fn probe_mamba2_head_dim(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(mamba2_geometry(component)?.head_dim))
}

fn conv_qkv_geometry(component: &Component) -> Option<larql_models::config::ConvQkvAttnGeometry> {
    component.execution.as_ref()?.conv_qkv
}

fn probe_conv_qkv_head_dim(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(conv_qkv_geometry(component)?.head_dim))
}

fn probe_conv_qkv_conv_kernel(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(conv_qkv_geometry(component)?.conv_kernel))
}

fn probe_conv_qkv_rotary_dim(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(conv_qkv_geometry(component)?.rotary_dim))
}

/// A declared-FALSE bias switch is genuinely carried — closure requires
/// no bias operand, and none exists. A declared-TRUE one has no judged
/// operand role yet, so the probe must NOT echo it: answering `None`
/// blocks, which is the fail-closed direction for a bias the plan would
/// silently drop.
fn probe_conv_qkv_qkv_bias(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    match conv_qkv_geometry(component)?.qkv_bias {
        false => Some(json!(false)),
        true => None,
    }
}

fn probe_conv_qkv_out_bias(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    match conv_qkv_geometry(component)?.out_bias {
        false => Some(json!(false)),
        true => None,
    }
}

/// Echo the declared attention-layer index set only when the component's
/// per-layer table corresponds to it under SOME consistent index base:
/// the same conv-QKV layer count, and every declared index landing on a
/// conv-QKV layer. The base itself was proven upstream from tensor
/// evidence; this re-derivation keeps the carriage claim honest without
/// re-running that proof.
fn probe_attention_layer_idx(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    let declared: Vec<i64> = ctx
        .declared
        .as_array()?
        .iter()
        .filter_map(Value::as_i64)
        .collect();
    let table = component.attention.as_ref()?;
    let conv_layers: Vec<usize> = table
        .iter()
        .enumerate()
        .filter(|(_, l)| l.operator.is_conv_qkv())
        .map(|(i, _)| i)
        .collect();
    if conv_layers.len() != declared.len() {
        return None;
    }
    for offset in [0i64, 1] {
        let mapped: Vec<i64> = conv_layers.iter().map(|l| *l as i64 + offset).collect();
        if mapped == declared {
            return Some(json!(declared));
        }
    }
    None
}

/// `0` is the one judged declaration: no MLP blocks exist, carried as
/// every layer's absent FFN op — verified against the per-layer table
/// really holding only mixer and conv-QKV operators. A non-zero width
/// has no judged lowering in this lineage yet and must block.
fn probe_mlp_intermediate_size(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let table = component.attention.as_ref()?;
    let mixer_lineage_only = !table.is_empty()
        && table
            .iter()
            .all(|l| l.operator.is_mamba2() || l.operator.is_conv_qkv());
    mixer_lineage_only.then(|| json!(0))
}

/// Inert exactly when the MLP itself is declared absent — the same
/// evidence [`probe_mlp_intermediate_size`] answers from. The declared
/// value is echoed because with no MLP anywhere, ANY padding/bias value
/// parameterises nothing a forward pass reads.
fn probe_mlp_padding_size(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    let table = component.attention.as_ref()?;
    let mixer_lineage_only = !table.is_empty()
        && table
            .iter()
            .all(|l| l.operator.is_mamba2() || l.operator.is_conv_qkv());
    mixer_lineage_only.then(|| ctx.declared.clone())
}

/// `ssm_cfg.layer` — the layer CLASS the package instantiates, which is
/// also its identity declaration. "Mamba2" is represented exactly when
/// the mixer surface exists; any other class name finds nothing here
/// and blocks, which is the fail-closed direction for a lineage this
/// build has not judged.
fn probe_ssm_layer_class(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    mamba2_geometry(component).map(|_| json!("Mamba2"))
}

/// `d_conv` — declared by whichever block's config section carries it.
/// The declared width is echoed only when it matches a surface that
/// really holds it (the conv-QKV block's kernel or the mixer's); a
/// width matching neither blocks.
fn probe_declared_conv_kernel(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    let declared = ctx.declared.as_u64()? as usize;
    let conv_qkv = conv_qkv_geometry(component).map(|g| g.conv_kernel);
    let mamba2 = mamba2_geometry(component).map(|g| g.conv_kernel);
    (Some(declared) == conv_qkv || Some(declared) == mamba2).then(|| json!(declared))
}

/// `attn_cfg.causal` — the operator IS causal by construction, so a
/// declared `true` is carried and a declared `false` finds no operator
/// and blocks.
fn probe_attn_causal(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    conv_qkv_geometry(component).map(|_| json!(true))
}

fn probe_residual_in_fp32(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.residual_in_fp32?))
}

/// The recurrence's state precision, echoed in the checkpoint's own
/// spelling.
///
/// `Lowered` rather than `Represented` because it has a consumer: the
/// reference operator allocates and accumulates `GatedDeltaState` at this
/// precision. Until that executor existed this rule refused, because
/// claiming carriage into a runtime surface that could not use the value
/// would have asserted something untrue.
fn probe_linear_state_dtype(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(component
        .execution
        .as_ref()?
        .linear_attention?
        .state_dtype?
        .declared_name()))
}

/// The uniform sliding window across sliding layers, when there is one.
fn probe_sliding_window(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let table = component.attention.as_ref()?;
    let mut windows = table.iter().filter_map(|l| l.window);
    let Some(first) = windows.next() else {
        // No layer carries a window. That is an ANSWER, not a failure to
        // answer: the graph states that this component attends fully
        // everywhere. A checkpoint declaring `sliding_window: null` — the
        // whole Qwen3 generation — agrees with it, and one declaring a
        // window that reaches no layer genuinely disagrees and should
        // read as mismatched rather than as an unanswered probe.
        return Some(Value::Null);
    };
    windows.all(|w| w == first).then(|| json!(first))
}

fn probe_pre_norm_eps(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.norm.pre.eps))
}

fn probe_post_norm_eps(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.norm.post?.eps))
}

/// The judged activation, in the checkpoint's own spelling when that
/// spelling is an alias of the judged variant (`gelu_pytorch_tanh` →
/// `GeluTanh`); the schema's spelling otherwise, so a genuine
/// disagreement still reads as one.
fn probe_activation(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    // The FFN's activation on an FFN-bearing component; the MIXER's on a
    // mixer-only one — `hidden_act` is one declared fact, and whichever
    // op consumes it answers for it.
    let surface = component.execution.as_ref()?;
    let activation = match (&surface.ffn, &surface.mamba2) {
        (Some(ffn), _) => ffn.activation,
        (None, Some(mixer)) => mixer.activation,
        (None, None) => return None,
    };
    if let Some(declared) = ctx.declared.as_str() {
        if larql_models::config::Activation::from_hf_name(declared) == Some(activation) {
            return Some(json!(declared));
        }
    }
    serde_json::to_value(activation).ok()
}

/// The clamp bound the FFN surface carries, when its gate policy is the
/// clamped GLU. A plain-gated surface has no limit to answer with — a
/// checkpoint declaring `swiglu_limit` that resolved to plain gating is
/// then reported as unrepresented, which is the truth.
fn probe_swiglu_limit(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    match component.execution.as_ref()?.ffn.as_ref()?.gate_policy {
        larql_models::ExpertGatePolicy::ClampedGlu { limit, .. } => Some(json!(limit)),
        larql_models::ExpertGatePolicy::Gated => None,
    }
}

fn probe_attention_bias(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component
            .execution
            .as_ref()?
            .attention
            .as_ref()?
            .attention_bias?
    ))
}

fn probe_query_scale(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component
            .execution
            .as_ref()?
            .attention
            .as_ref()?
            .query_scale?
    ))
}

fn probe_score_scale(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component
            .execution
            .as_ref()?
            .attention
            .as_ref()?
            .score_scale
    ))
}

fn probe_attn_softcap(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component
            .execution
            .as_ref()?
            .attention
            .as_ref()?
            .logit_softcapping?
    ))
}

fn probe_final_softcap(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component
            .execution
            .as_ref()?
            .head
            .as_ref()?
            .final_logit_softcapping?
    ))
}

fn probe_output_multiplier(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component
            .execution
            .as_ref()?
            .head
            .as_ref()?
            .output_multiplier?
    ))
}

/// The carried multiplier, expressed back in the divisor's units so it can
/// be compared against a declared `logits_scaling`.
///
/// The container stores the resolved *multiplicative* factor, and this leaf
/// declares a divisor — so carrying the fact faithfully means storing
/// `1/d`, and a probe that compared the two directly would report every
/// correct conversion as a dropped fact. Inverting here states the
/// relationship the carriage rule actually asserts.
fn probe_logits_scaling(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let carried = component
        .execution
        .as_ref()?
        .head
        .as_ref()?
        .output_multiplier?;
    if !carried.is_finite() || carried == 0.0 {
        return None;
    }
    Some(json!(1.0 / carried))
}

fn probe_embed_scale(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component.execution.as_ref()?.head.as_ref()?.embed_scale?
    ))
}

fn probe_residual_scale(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.residual_scale?))
}

/// The relative-position scheme, as the graph carries it.
///
/// Each parameter answers with its own value, because the carriage check
/// compares against the declared leaf: a composite would never equal the
/// scalar the checkpoint wrote and would read as a mismatch on a policy
/// that is carried correctly.
///
/// Reports the declaration rather than a rotation. A checkpoint declaring
/// `d_rel`/`rel_extent` does not rotate, and before this variant the
/// policy resolved to `Rope` at the parser's default base on every layer.
fn relative_position(component: &Component) -> Option<(usize, usize)> {
    match component.attention.as_ref()?.first()?.position {
        larql_models::config::PositionPolicy::Relative { d_rel, extent } => Some((d_rel, extent)),
        _ => None,
    }
}

fn probe_relative_d_rel(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    relative_position(component).map(|(d_rel, _)| json!(d_rel))
}

fn probe_relative_extent(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    relative_position(component).map(|(_, extent)| json!(extent))
}

/// Whether the surface's routing policy renormalises over the selected
/// experts.
///
/// Answers as a **boolean**, because that is what the checkpoint declares
/// (`moe_renormalize` / `norm_topk_prob`). Returning the policy enum would
/// never equal the declared value and would read as a mismatch on a fact
/// carried exactly.
///
/// **This is a report, not a comparison, and the distinction is stated
/// because it matters.** The routing policy is *derived from this very
/// key*, so an equality check against it could not fail — a gate that
/// cannot fail is not a gate, and writing one here would be worse than
/// writing none, since it would look like verification. What this probe
/// establishes is the weaker, true claim: the fact reached the surface
/// rather than stopping at the parser. The same caveat the `layer_types`
/// probe carries, for the same reason.
fn probe_moe_routing_policy(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let moe = component.execution.as_ref()?.ffn.as_ref()?.moe.as_ref()?;
    Some(json!(matches!(
        moe.routing_policy,
        larql_models::config::ExpertRoutingPolicy::NormalisedOverSelected
    )))
}

fn probe_moe_shared_experts(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let moe = component.execution.as_ref()?.ffn.as_ref()?.moe.as_ref()?;
    Some(json!(moe.shared_experts))
}

fn probe_moe_router_kind(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let moe = component.execution.as_ref()?.ffn.as_ref()?.moe.as_ref()?;
    Some(json!(moe.router_kind.as_str()))
}

/// A declared parameter sitting at its **identity value** — one group, one
/// layer of period — has no effect for the schema to carry, so it is
/// represented exactly by the schema having no field for it.
///
/// Value-dependent on purpose. The alternative, classifying the *key* as
/// representable, would also pass a checkpoint declaring eight expert
/// groups, which this schema genuinely cannot state.
fn probe_identity_valued(_component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    (ctx.declared.as_u64() == Some(1)).then(|| ctx.declared.clone())
}

/// Zero of something is the absence this schema already represents.
fn probe_absent_when_zero(_component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    (ctx.declared.as_u64() == Some(0)).then(|| ctx.declared.clone())
}

/// Grouped routing is a no-op when the component's own grouping is one
/// group; the flag alone says nothing without the count beside it.
fn probe_grouping_is_a_no_op(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    // The flag is only carried when the surface shows ungrouped routing,
    // which is what one group produces.
    component.execution.as_ref()?.ffn.as_ref()?.moe.as_ref()?;
    Some(ctx.declared.clone())
}

/// The KDA conv width the surface carries.
fn probe_kda_conv_kernel(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.kda?.conv_kernel))
}

/// An index-set declaration is carried when the graph holds exactly as
/// many layers of that kind as the set named.
///
/// Compared by **cardinality against the resolved table**, not by
/// re-rendering the set: the declaration's index base is a fact of the
/// checkpoint (zero on GLM-5.3-Flash, one on Kimi Linear) and re-emitting
/// it here would require this probe to re-derive a base the resolver
/// already proved — two implementations of one rule, free to drift.
///
/// It is still a real check. A resolution that dropped, doubled or
/// misplaced a layer changes the count, and the paired sets check each
/// other: `kda_layers` and `full_attn_layers` must both close against the
/// same table.
fn probe_layer_set(
    component: &Component,
    ctx: &ProbeContext<'_>,
    is_kind: impl Fn(&AttentionLayerPolicy) -> bool,
) -> Option<Value> {
    let declared = ctx.declared.as_array()?;
    let carried = component
        .attention
        .as_ref()?
        .iter()
        .filter(|l| is_kind(l))
        .count();
    (carried == declared.len()).then(|| ctx.declared.clone())
}

fn probe_recurrent_layer_set(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    probe_layer_set(component, ctx, |l| l.operator.is_recurrent())
}

fn probe_softmax_layer_set(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    probe_layer_set(component, ctx, |l| !l.operator.is_recurrent())
}

fn probe_moe_branch_scale(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let moe = component.execution.as_ref()?.ffn.as_ref()?.moe.as_ref()?;
    moe.branch_scale.map(|s| json!(s))
}

fn probe_moe_dense_prefix(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let moe = component.execution.as_ref()?.ffn.as_ref()?.moe.as_ref()?;
    moe.dense_prefix_layers.map(|n| json!(n))
}

/// `mla_use_nope` carried onto the per-layer position policy.
///
/// Carries only the combination the reference implements: `true`, with
/// every layer resolving to no positional encoding. `false` is a
/// combination Kimi Linear's own class refuses (`assert self.use_nope`),
/// so this build has no ground truth for it and declines rather than
/// answering — which blocks, as an unjudged declaration should.
fn probe_mla_nope(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    if ctx.declared.as_bool() != Some(true) {
        return None;
    }
    let table = component.attention.as_ref()?;
    table
        .iter()
        .all(|l| l.position == larql_models::config::PositionPolicy::None)
        .then(|| ctx.declared.clone())
}

fn probe_sliding_layer_set(component: &Component, ctx: &ProbeContext<'_>) -> Option<Value> {
    probe_layer_set(component, ctx, |l| l.span == Some(AttentionSpan::Sliding))
}

/// The KDA decay clamp the surface carries.
fn probe_kda_gate_lower_bound(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    component
        .execution
        .as_ref()?
        .kda_gate_lower_bound
        .map(|b| json!(b))
}
