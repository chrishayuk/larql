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

use super::super::graph::policy::AttentionSpan;
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
        site: "Component.attention[].position (PositionPolicy::PartialRope.rotary_fraction) → AttentionOp.position",
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
        site: "Component.attention[].span → AttentionOp.span",
        probe: Some(probe_layer_types),
    },
    CarriageRule {
        leaf: "sliding_window",
        reaches: Carriage::Lowered,
        site: "Component.attention[].window → AttentionOp.window",
        probe: Some(probe_sliding_window),
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
        // Granite's spelling of `output_multiplier` — algebraically the
        // same operation (scaling commutes through the linear head, so
        // "before the vocab projection" and "on the logits" are the same
        // number), resolved by `ModelArchitecture::output_multiplier`'s
        // default the same way `attention_multiplier` resolves above.
        site: "ExecutionSurface.head.output_multiplier → OutputOp.multiplier",
        probe: Some(probe_output_multiplier),
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
        // Whether router weights are renormalised after top-k selection —
        // `RoutedFfnOp.routing_policy` judges the routing math itself
        // (`MoeRouterKind`/`ExpertRoutingPolicy`), but no field states this
        // flag directly; always refuses rather than assuming it agrees
        // with whatever the judged policy happens to imply.
        site: "no schema field — not yet cross-checked against routing_policy",
        probe: Some(probe_unrepresented),
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
            .filter(move |l| span.is_none_or(|s| l.span == s)),
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
    Some(json!(component.execution.as_ref()?.ffn.moe.is_some()))
}

fn probe_moe_top_k(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.ffn.moe?.top_k))
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
        .filter(|l| l.span == AttentionSpan::Full)
        .map(|l| {
            l.geometry
                .map_or(surface.attention.head_dim, |g| g.head_dim)
        });
    let first = dims.next()?;
    dims.all(|d| d == first).then(|| json!(first))
}

/// The KV-head count the full-attention layers carry.
fn probe_full_layer_kv_heads(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let table = component.attention.as_ref()?;
    let surface = component.execution.as_ref()?;
    let mut heads = table
        .iter()
        .filter(|l| l.span == AttentionSpan::Full)
        .map(|l| {
            l.geometry
                .map_or(surface.attention.num_kv_heads, |g| g.num_kv_heads)
        });
    let first = heads.next()?;
    heads.all(|h| h == first).then(|| json!(first))
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
fn probe_layer_types(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let table = component.attention.as_ref()?;
    Some(Value::Array(
        table
            .iter()
            .map(|l| json!(l.span.declared_name()))
            .collect(),
    ))
}

/// The uniform sliding window across sliding layers, when there is one.
fn probe_sliding_window(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    let table = component.attention.as_ref()?;
    let mut windows = table.iter().filter_map(|l| l.window);
    let first = windows.next()?;
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
    let activation = component.execution.as_ref()?.ffn.activation;
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
    match component.execution.as_ref()?.ffn.gate_policy {
        larql_models::ExpertGatePolicy::ClampedGlu { limit, .. } => Some(json!(limit)),
        larql_models::ExpertGatePolicy::Gated => None,
    }
}

fn probe_attention_bias(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component.execution.as_ref()?.attention.attention_bias?
    ))
}

fn probe_query_scale(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.attention.query_scale?))
}

fn probe_score_scale(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.attention.score_scale))
}

fn probe_attn_softcap(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component.execution.as_ref()?.attention.logit_softcapping?
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

fn probe_embed_scale(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(
        component.execution.as_ref()?.head.as_ref()?.embed_scale?
    ))
}

fn probe_residual_scale(component: &Component, _ctx: &ProbeContext<'_>) -> Option<Value> {
    Some(json!(component.execution.as_ref()?.residual_scale?))
}
