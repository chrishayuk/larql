//! The execution surface: what generic operations need in order to
//! execute a component (V3-G5a).
//!
//! [`super::Component`] answers *what part of the system this is*; the
//! surface answers *what the generic ops need to run it*. Fields are
//! grouped **by operation**, because the completeness contract derives
//! from the operations a component's objects imply — never from what any
//! particular architecture happens to declare:
//!
//! ```text
//! DecoderStack / PerceptionTower object  →  attention, ffn, norm
//! Embedding / OutputHead object          →  head
//! ```
//!
//! Every value is **fully resolved**: defaulting decisions (an absent
//! `hidden_act`, a post-norm epsilon shared with `norm_eps`, canonical
//! 1/√d attention scaling) are applied at build/judgment time and
//! persisted. A generic executor reads these fields; it never defaults an
//! absent one — absence is a completeness defect that refuses encoding,
//! not a branch at run time.

use larql_models::config::{
    Activation, AttentionGateSpec, AttentionSinkSpec, EmbeddingNorm, ExpertFormat,
    ExpertRoutingPolicy, FfnType, GateUpLayout, MoeRouterKind, NormSpec, ParameterFreeQkNorm,
    QkNormScope,
};
use larql_models::inventory::ArchitectureInventory;
use serde::{Deserialize, Serialize};

use super::object::LogicalObject;

mod linear_mla;
mod nested;
pub use linear_mla::*;
pub use nested::*;

/// Tensor-name fragments evidencing a gated FFN under a binding. Evidence,
/// not a family fact: presence of gate weights decides, whoever ships
/// them. One definition, shared by the builder and the G4 re-derivation.
const GATE_TENSOR_FRAGMENTS: &[&str] = &["gate_proj", "gate_up"];

/// Whether any tensor bound by `object` carries gate-FFN evidence.
pub fn gate_evidence(inventory: &ArchitectureInventory, object: &LogicalObject) -> bool {
    object.source_bindings.iter().any(|binding| {
        inventory
            .tensors
            .tensors
            .iter()
            .filter(|t| t.name.starts_with(&binding.tensor_prefix))
            .any(|t| {
                GATE_TENSOR_FRAGMENTS
                    .iter()
                    .any(|fragment| t.name.contains(fragment))
            })
    })
}

/// What the attention op reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionSurface {
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// The query-scale operation: a multiplier on the (normalised) query
    /// states before position encoding. `None` = the op is absent, which
    /// is a different claim from `Some(1.0)`. Kept separate from
    /// [`Self::score_scale`]: folding them is algebra-equivalent but not
    /// fp-equivalent, and the executor must place each multiply where
    /// the judged semantics put it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_scale: Option<f64>,
    /// Canonical score-time multiplier on QK^T.
    pub score_scale: f64,
    /// Attention-logit softcap; `None` = the op is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logit_softcapping: Option<f32>,
    /// QK-norm scope, read when QK-norm weights exist in the stack.
    pub qk_norm_scope: QkNormScope,
    pub qk_norm_weight_offset: f32,
    /// Parameter-free QK normalisation (no weight tensors) — judged
    /// semantics no tensor evidence can reveal.
    #[serde(default)]
    pub parameter_free_qk_norm: ParameterFreeQkNorm,
    /// Judged attention-output-gate semantics; `None` = no judgment
    /// exists — **never "no gate"**. A stack shipping an
    /// [`OperandRole::AttnOutputGate`](super::roles::OperandRole) operand
    /// while this is `None` fails operand closure — the primitive exists
    /// in the IR, but its semantics for that model have not been judged,
    /// and closure refuses rather than guessing an activation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_gate: Option<AttentionGateSpec>,
    /// Judged attention-sink semantics; `None` = no judgment exists —
    /// **never "no sinks"**. A stack shipping an
    /// [`OperandRole::AttnSinks`](super::roles::OperandRole) operand while
    /// this is `None` fails operand closure; a surface stating a spec
    /// while the operand is absent fails it too. Absent from the
    /// serialised surface when `None`, so every pre-A-9.1 container reads
    /// back unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sinks: Option<AttentionSinkSpec>,
    /// Whether the Q/K/V/O projections carry additive biases, as the
    /// checkpoint declares (`attention_bias`). `None` = undeclared, which
    /// is not "no bias": bias operands under `None`/`Some(false)` fail
    /// operand closure, and `Some(true)` requires all four operands. The
    /// executor adds each bias after its projection, before QK-norm /
    /// rope (Q, K), before caching (V) and after the output projection (O).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_bias: Option<bool>,
    /// Whether the Q/K/V projections carry biases and the output
    /// projection does not (`qkv_bias`; Qwen2's shape). `Some(true)`
    /// requires the three Q/K/V bias operands and refuses an output bias;
    /// otherwise, under `attention_bias` alone, any bias operand fails
    /// closure as above. Declaring it beside `attention_bias: true` is a
    /// contradiction closure refuses. Additive: absent on every container
    /// written before it, none of which could carry a Q/K/V-only bias.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub qkv_bias: Option<bool>,
}

/// What the FFN op reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FfnSurface {
    /// The DENSE FFN's intermediate width.
    ///
    /// `None` on a wholly-routed stack, which has no dense FFN to size —
    /// `Qwen3_5MoeTextConfig` is `@strict` and declares no
    /// `intermediate_size` at all, because every one of its layers is a
    /// routed block. Absence is the fact; a zero here would be a width,
    /// and every consumer that needs a dense width would take it.
    ///
    /// Added additively within GRAPH_SCHEMA 6: a container written before
    /// this carries the number and still reads as `Some`, and only a
    /// wholly-routed component — which could not be represented at all
    /// before this — omits it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intermediate_size: Option<usize>,
    /// Per-layer dense width, when the checkpoint declares one — a
    /// derived static-shard container stores physically narrower
    /// gate/up/down tensors for some layers and says so here. One entry
    /// per layer; the planner checks every layer's tensors against it and
    /// states the width on that layer's `FfnOp`. `None` = every layer at
    /// `intermediate_size`. Additive within GRAPH_SCHEMA 6.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intermediate_size_by_layer: Option<Vec<usize>>,
    pub activation: Activation,
    pub ffn_type: FfnType,
    /// How the gate combines with the up branch: plain `activation(gate) *
    /// up`, or GPT-OSS's clamped GLU (`swiglu_limit`, `alpha`). A distinct
    /// policy rather than an `Activation` variant, because the clamp and
    /// the `+1` change the model, not the nonlinearity — carried so the
    /// declared `swiglu_limit` has a container site to be judged against
    /// (A-9.0). Defaults for containers written before it existed.
    #[serde(default)]
    pub gate_policy: larql_models::ExpertGatePolicy,
    /// The routed-FFN judgment, when the component's FFN is a mixture of
    /// experts. `None` = dense — and, as everywhere on the surface, a stack
    /// shipping router or expert-bank operands under `None` fails operand
    /// closure rather than running them as something else. Absent from the
    /// serialised surface when `None`, so every dense container reads back
    /// unchanged. Which layers are routed is operand evidence: a layer with
    /// an expert bank is routed, one with dense FFN operands is dense, and
    /// the two may interleave.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe: Option<MoeSurface>,
    /// Whether the dense FFN's up and down projections carry additive
    /// biases (`nn.Linear`/`Conv1D` bias). `Some(true)` requires the up
    /// and down bias operands; under `None` or `Some(false)` any FFN bias
    /// operand fails closure by name. The executor adds the up bias before
    /// the activation and the down bias after the down projection.
    /// Additive within GRAPH_SCHEMA 6: absent on every container written
    /// before it, none of which could carry an FFN bias.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bias: Option<bool>,
}

/// The routed-FFN (mixture-of-experts) semantics of a component, lifted
/// from the family's judgment. Every field is something the executor
/// reads; none is re-derived from operand names.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MoeSurface {
    /// Routed experts per layer.
    pub experts: usize,
    /// Experts selected per token.
    pub top_k: usize,
    /// Per-expert intermediate width (the down projection's input).
    pub expert_intermediate_size: usize,
    /// How router logits become selected experts and weights.
    pub router_kind: MoeRouterKind,
    /// Whether the selected weights are normalised to sum to one.
    pub routing_policy: ExpertRoutingPolicy,
    /// Whether the router carries an additive bias on its logits — the
    /// `MoeRouterBias` operand is required iff this is set.
    pub router_bias: bool,
    /// How the experts are stored; decides which expert-bank operand roles
    /// closure requires (packed MXFP4: blocks + scales + bias per
    /// projection).
    pub expert_format: ExpertFormat,
    /// How a fused `gate_up` operand's rows split into gate and up.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate_up_layout: Option<GateUpLayout>,
    /// Always-active experts alongside the routed ones.
    pub shared_experts: usize,
    /// That branch's own intermediate width, resolved once by the
    /// architecture. `None` iff [`Self::shared_experts`] is zero.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_expert_intermediate_size: Option<usize>,
    /// The gate on that branch's output, where the family runs one.
    /// `None` = summed unscaled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_expert_gate: Option<larql_models::config::SharedExpertGateSpec>,
    /// Multiplier on the routed-expert branch (`routed_scaling_factor`).
    /// `None` when undeclared — not 1.0, which is a different claim, and
    /// a wrong one would rescale the whole branch.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch_scale: Option<f64>,
    /// Leading layers running a dense MLP instead of the routed block
    /// (`first_k_dense_replace`): 1 on Kimi Linear, 3 on GLM-5.3-Flash.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dense_prefix_layers: Option<usize>,
    /// A dense MLP summed with the expert block every layer.
    pub hybrid: bool,
    /// The bottleneck the ROUTED experts run behind, when the family
    /// declares one. `None` = the experts consume the block input at
    /// `hidden` and their weighted sum is already in the residual
    /// stream's space.
    ///
    /// Absent from the serialised surface when `None`, so every
    /// non-latent container reads back unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latent: Option<MoeLatent>,
}

/// The routed branch's bottleneck: down to [`Self::width`], experts,
/// weighted aggregate, optional norm, back up to `hidden`.
///
/// Nested rather than two flat fields on [`MoeSurface`] because the
/// reference nests them — `if self.use_latent_moe:` encloses `if
/// self.latent_moe_use_norm:` — so a norm without a width builds nothing
/// at all. Nesting makes that state unrepresentable instead of merely
/// wrong.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MoeLatent {
    /// The width the routed experts' projections are sized from. A third
    /// independently declared number — not `hidden / 2`, not the expert
    /// intermediate width.
    pub width: usize,
    /// The norm on the weighted aggregate, between summation and the
    /// up-projection. `None` = the family declares none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm: Option<LatentNorm>,
}

/// The routed-expert norm's own epsilon.
///
/// Carried rather than defaulted, and deliberately not shared with the
/// MLA low-rank norms: in this same family `q_a_layernorm` and
/// `kv_a_layernorm` run at `KimiRMSNorm`'s class default `1e-6` while
/// this one is constructed with `eps=config.rms_norm_eps` and runs at the
/// layer's `1e-5`. Same shape of fact, different authority, ten times
/// apart.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LatentNorm {
    /// The epsilon, from the layer's `rms_norm_eps`.
    pub eps: f64,
}

impl MoeSurface {
    /// The width the routed experts' own projections are sized from —
    /// the bottleneck when one is declared, `hidden` otherwise.
    ///
    /// THE single authority for routed-bank geometry. Every expert-bank
    /// shape contract asks this rather than reaching for `hidden`, so the
    /// packed and per-expert call sites cannot disagree about where the
    /// bank lives, and a future storage format inherits the answer rather
    /// than restating it.
    ///
    /// The router and the shared experts do NOT ask this: both read the
    /// un-projected block input, and the shared branch is summed after
    /// the up-projection.
    pub fn routed_expert_input_width(&self, hidden: usize) -> usize {
        self.latent.map_or(hidden, |l| l.width)
    }

    /// The declared latent width that cannot be executed, if there is
    /// one.
    ///
    /// `routed_expert_hidden_size: 0` SELECTS the latent form — the
    /// reference tests `is not None`, not truthiness — and then describes
    /// a bottleneck of no width. Naming it here lets the op plan refuse
    /// the DECLARATION, rather than letting every bank contract refuse a
    /// zero-column operand and send a reader to a tensor whose stored
    /// width is not the thing that is wrong.
    ///
    /// Falling back to the uniform form is the one answer that must not
    /// be given: it would execute a different model than the checkpoint
    /// declares, silently.
    pub fn degenerate_latent_width(&self) -> Option<usize> {
        self.latent.map(|l| l.width).filter(|w| *w == 0)
    }
}

/// What the norm op reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NormSurface {
    /// Complete spec for the pre-attention and pre-FFN sites.
    pub pre: NormSpec,
    /// Complete spec for the post-attention and post-FFN sites.
    /// `None` = unjudged — nothing has established it, and a four-norm
    /// [`Self::placement`] in that state fails closure rather than
    /// inheriting [`Self::pre`]. Muse-Glimmer's differ by three orders
    /// of magnitude in epsilon (1e-5 pre, 1e-8 post).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post: Option<NormSpec>,
    /// Complete spec for the final norm before the head. Its own spec
    /// because a family may use a different convention there and a
    /// single model-scope answer would silently break one site to fix
    /// the others — Muse-Glimmer's layers are centred (`1 + w`) while
    /// its final norm is not.
    pub final_norm: NormSpec,
    /// Norm placement around attention/FFN, judged from operand evidence
    /// ([`super::roles::norm_placement_evidence`]) — never from a family
    /// default, which is exactly the fact the generic fallback got wrong
    /// on the first real four-norm stack. Count is not semantics;
    /// placement is. `None` only for components with no decoder stack.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub placement: Option<super::roles::NormPlacement>,
    /// Whether every weighted norm site — every layer site and the final
    /// norm — carries an additive bias (LayerNorm's affine `bias`).
    /// `Some(true)` requires a bias operand beside every norm weight;
    /// under `None` or `Some(false)` any norm bias operand fails closure
    /// by name. Additive within GRAPH_SCHEMA 6.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bias: Option<bool>,
}

/// What embedding lookup and the output head read. Present iff the
/// component owns embedding/output-head objects.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeadSurface {
    pub vocab_size: usize,
    /// Normalisation applied to embedding-table output. `None` = no such
    /// operation. Weightless, so no operand evidences it and no closure
    /// check can infer it — it arrives only as a family judgment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_norm: Option<EmbeddingNorm>,
    /// The embedding-scale operation, applied after lookup. `None` = the
    /// op is absent, distinct from `Some(1.0)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed_scale: Option<f32>,
    /// The output-multiplier operation, applied before the vocabulary
    /// projection. `None` = the op is absent, distinct from `Some(1.0)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_multiplier: Option<f64>,
    /// Final-logit softcap; `None` = the op is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_logit_softcapping: Option<f32>,
    /// Whether a missing standalone output-head *object* means "reuse the
    /// embedding object" rather than "this component cannot generate".
    /// From [`ResolvedExecution::head_reuses_embedding`](larql_models::inventory::ResolvedExecution::head_reuses_embedding) —
    /// carried here so `opplan::build` can answer the question from the
    /// surface alone, with no re-interpretation of the source checkpoint.
    #[serde(default)]
    pub head_reuses_embedding: bool,
    /// How position enters at the input: a learned absolute table whose
    /// row `p` is added to the token embedding at position `p` (GPT-2's
    /// `wpe`). `None` = nothing is added at the input — which is a claim,
    /// checked by closure: a component owning a position-embedding object
    /// with no declaration here is refused, and a declaration with no
    /// object is refused. Never folded into the per-layer position
    /// policy, which states what each layer's attention rotates (nothing,
    /// on this lineage). Additive within GRAPH_SCHEMA 6.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_position: Option<larql_models::config::InputPositionEncoding>,
}

/// The complete per-component execution surface.
///
/// Since GRAPH_SCHEMA 6, `attention` and `ffn` are present **iff the
/// component's program runs those operations** — presence means semantic
/// presence, never "the file was written". A pure-SSM stack (mamba2)
/// carries neither: fabricating an attention surface for it is the
/// ontology drill's F1 finding, and its FFN twin is the same defect one
/// op over (the mixer is the whole block; no `intermediate_size` exists).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionSurface {
    /// How far the component's program is declared to run — the
    /// checkpoint's `max_position_embeddings`, or another family's
    /// spelling of the same fact.
    ///
    /// **On the component, not on `attention`.** Context extent is a
    /// property of the execution programme, not of softmax: a Gated
    /// DeltaNet, KDA or Mamba stack has one without attending at all,
    /// and Qwen3.8 runs forty-eight recurrent layers to sixteen
    /// attending ones. Hanging it off the attention surface would make
    /// it unreachable for exactly the architectures that most need it.
    ///
    /// Added additively within GRAPH_SCHEMA 6 — new information, not a
    /// reinterpretation of existing bytes, so a v6 graph written before
    /// this field still reads and a reader without it still parses one
    /// that has it.
    ///
    /// `tokenizer_config.json`'s `model_max_length` is a serving bound
    /// on the tokenizer and is **not** the authority for this. The two
    /// usually agree; when they disagree the execution semantic wins,
    /// because that is the one that changes what a forward pass does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u64>,
    /// What the attention op reads — present iff any layer of the
    /// component's program attends (softmax or MLA).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention: Option<AttentionSurface>,
    /// What the FFN op reads — present iff the component's program runs
    /// an FFN (every attention-class family today; a mixer-only stack
    /// does not).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ffn: Option<FfnSurface>,
    pub norm: NormSurface,
    /// Present iff the component owns embedding/output-head objects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub head: Option<HeadSurface>,
    /// Residual-stream scaling: the attention/FFN sublayer's own output is
    /// multiplied by this before its residual add, at both sites with the
    /// same value (Granite's `residual_multiplier`). `None` = the op is
    /// absent, distinct from `Some(1.0)`. Component-wide like every other
    /// field here, applied at every layer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual_scale: Option<f32>,
    /// Geometry the Gated DeltaNet operator consumes, on a component whose
    /// layers include linear attention. `None` on a wholly-softmax stack.
    ///
    /// Deliberately NOT every `linear_*` config field: the surface carries
    /// the subset an operator reads, not a second copy of `ModelConfig`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_attention: Option<LinearAttentionSurface>,
    /// Geometry the KDA operator consumes, on a component whose layers
    /// include Kimi Delta Attention. `None` otherwise.
    ///
    /// Beside [`Self::linear_attention`] rather than sharing it: the two
    /// operators' geometries are not interchangeable, and a single field
    /// would force every reader to ask which one it holds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kda: Option<larql_models::config::KdaGeometry>,
    /// Lower bound clamped onto KDA's decay gate (`gate_lower_bound`).
    /// Carried beside the geometry because it changes what the operator
    /// computes, and `None` must mean "the checkpoint declared none",
    /// never a silently-chosen default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kda_gate_lower_bound: Option<f32>,
    /// Which decay gate the KDA operator computes — the family's judgment
    /// of what its reference does with
    /// [`Self::kda_gate_lower_bound`], which the value alone cannot
    /// answer (two checkpoints declare `-5.0` and disagree). `None` is
    /// unjudged and must reach a refusal, never a chosen form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kda_gate_form: Option<larql_models::config::KdaGateForm>,
    /// The FORM of KDA's output gate (`linear_attn_config.use_full_rank_gate`):
    /// `Some(true)` = one full-rank `g_proj` of `[Hv·Dv, hidden]` (Kimi-K3),
    /// `Some(false)` = the low-rank `g_a_proj`/`g_b_proj` pair, `None` =
    /// undeclared, which the reference reads as the pair. Carried beside the
    /// geometry, not inside it: the geometry is all-three-or-none, and the
    /// form is a separate declared fact the op plan holds the shipped
    /// operands to. Only the gate's projection changes with it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kda_use_full_rank_gate: Option<bool>,
    /// Geometry the Multi-Latent Attention operator consumes, on a
    /// component whose full-attention layers run it. `None` otherwise —
    /// including a family that DECLARES `uses_mla` but whose geometry did
    /// not fully resolve, which stays `None` here while the layer still
    /// classifies as [`super::policy::LayerOperator::Mla`] (see
    /// [`larql_models::inventory::report::MlaExecution`]'s docs).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mla: Option<MlaSurface>,
    /// What the Mamba2/SSD mixer reads, on a component whose layers run
    /// it. `None` otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mamba2: Option<Mamba2Surface>,
    /// What the hybrid's conv-QKV attention block reads, on a component
    /// whose full layers run it (`LayerOperator::ConvQkvAttention`).
    /// `None` otherwise. Reused from the architectural record directly,
    /// the way [`Self::kda`] reuses `KdaGeometry` — every field is
    /// something the operator reads, and the struct already refuses
    /// partial declarations at the parse boundary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conv_qkv: Option<larql_models::config::ConvQkvAttnGeometry>,
    /// Whether the residual stream is kept at fp32 against a
    /// lower-precision model (`residual_in_fp32`) — declared, never
    /// chosen by an executor. `None` = the checkpoint declares nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual_in_fp32: Option<bool>,
    /// How this component's residual stream is shaped and recombined.
    ///
    /// A COMPONENT fact rather than a per-layer one: once the residual
    /// means `[..., streams, d]`, the embedding, every branch operator
    /// and the head must all agree about it, and a per-layer flag would
    /// let a stack claim a bundle while its embedding assumed one vector.
    ///
    /// Defaults to `SingleStream` when absent, which is what every
    /// container written before this field carried.
    #[serde(default = "single_stream")]
    pub residual_topology: larql_models::config::ResidualTopology,
}

/// The topology every family judged before hyper-connections uses, and
/// what a container written before this field meant.
fn single_stream() -> larql_models::config::ResidualTopology {
    larql_models::config::ResidualTopology::SingleStream
}
