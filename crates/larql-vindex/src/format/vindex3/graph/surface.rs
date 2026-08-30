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
    ExpertRoutingPolicy, FfnType, GateUpLayout, MoeRouterKind, NormSpec, NormType,
    ParameterFreeQkNorm, QkNormScope,
};
use larql_models::inventory::components::ComponentTopology;
use larql_models::inventory::ArchitectureInventory;
use serde::{Deserialize, Serialize};

use super::object::LogicalObject;

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
}

/// What the FFN op reads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FfnSurface {
    pub intermediate_size: usize,
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
}

/// What the Gated DeltaNet operator reads.
///
/// Mirrors [`LinearAttentionTopology`](larql_models::inventory::report::LinearAttentionTopology)
/// rather than reusing it, for the same reason [`AttentionSurface`] does not
/// reuse the resolved topology: the surface is the executor's contract and
/// may diverge from the architectural record. It does not, today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinearAttentionSurface {
    /// Hk — query/key-side head count (16 on Qwen3.8).
    pub key_heads: usize,
    /// Dk (128).
    pub key_head_dim: usize,
    /// Hv — value-side head count (48). Distinct from [`Self::key_heads`]
    /// on purpose; no single head count describes this operator.
    pub value_heads: usize,
    /// Dv (128).
    pub value_head_dim: usize,
    /// Depthwise causal convolution width over the fused q|k|v channels (4).
    pub conv_kernel: usize,
    /// The precision the recurrence keeps its state at. Consumed: the
    /// reference operator allocates and accumulates its state at this
    /// precision rather than the model's bulk dtype.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_dtype: Option<larql_models::inventory::report::RecurrentStateDtype>,
}

impl LinearAttentionSurface {
    /// `2·Hk·Dk + Hv·Dv` — the fused projection's row count, and the
    /// channel count the depthwise convolution runs over. Derived so it
    /// cannot drift from the head counts.
    pub fn qkv_channels(self) -> usize {
        self.key_heads * self.key_head_dim * 2 + self.value_heads * self.value_head_dim
    }

    /// `Hv·Dv` — the value/gate width.
    pub fn value_width(self) -> usize {
        self.value_heads * self.value_head_dim
    }
}

/// What the Multi-Latent Attention operator reads.
///
/// Mirrors [`MlaExecution`](larql_models::inventory::report::MlaExecution)
/// rather than reusing it, for the same reason [`LinearAttentionSurface`]
/// does not reuse `LinearAttentionTopology`: the surface is the executor's
/// contract, and may diverge from the architectural record. It does not,
/// today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MlaSurface {
    /// Query/output head count — the decompressed K/V side always
    /// produces this many heads' worth of output.
    pub num_heads: usize,
    /// Compressed KV latent width.
    pub kv_lora_rank: usize,
    /// Non-RoPE portion of the query/key head width.
    pub qk_nope_head_dim: usize,
    /// RoPE portion of the query/key head width, one SHARED projection
    /// (MQA-style) across every head.
    pub qk_rope_head_dim: usize,
    /// Value head width — independent of the query/key head width.
    pub v_head_dim: usize,
}

impl MlaSurface {
    /// `qk_nope_head_dim + qk_rope_head_dim` — one query/key head's full
    /// width, and the row width `self_attn.q_proj.weight` is fused at
    /// (`num_heads · q_head_dim`).
    pub fn q_head_dim(self) -> usize {
        self.qk_nope_head_dim + self.qk_rope_head_dim
    }
}

/// What the Mamba2/SSD mixer reads.
///
/// The geometry is reused from the architectural record directly (the
/// same way [`ExecutionSurface::kda`] reuses
/// [`KdaGeometry`](larql_models::config::KdaGeometry)) — every field is
/// something the operator reads, and the struct already refuses partial
/// declarations at the parse boundary. The activation sits beside it
/// because a mixer-only component has no FFN surface to carry
/// `hidden_act`, and the mixer genuinely consumes it (the conv branch and
/// the output gate both apply it).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Mamba2Surface {
    pub geometry: larql_models::config::Mamba2Geometry,
    /// The mixer's nonlinearity (`hidden_act`, SiLU on every judged
    /// checkpoint — read, never assumed).
    pub activation: Activation,
}

/// Build the surface for a text-path component (target/drafter) from its
/// inventory's resolution. Returns the missing source facts when the
/// surface cannot be completed — the caller turns those into blocking
/// findings, never into defaults.
pub fn surface_from_resolved(
    inventory: &ArchitectureInventory,
) -> Result<ExecutionSurface, Vec<String>> {
    let resolved = &inventory.resolved;
    let Some(execution) = &resolved.execution else {
        return Err(vec![
            "resolved.execution (pre-v3 inventory — re-run inspect-hf)".to_string(),
        ]);
    };
    // Presence follows the program (schema 6). A layer that declares no
    // per-layer kind is an attention-class layer — that is what absence
    // means on every judged transformer config — so the surface is
    // present unless EVERY layer declares a recurrence. The FFN follows
    // the declared width: a family with no `intermediate_size` has no
    // FFN op to describe (the mixer-only case), and writing one anyway
    // is the F1 fabrication one op over.
    let attends = resolved.layers.is_empty()
        || resolved.layers.iter().any(|l| {
            !matches!(
                l.declared_kind,
                Some(larql_models::config::LayerKind::Recurrent(_))
            )
        });
    let has_ffn = resolved.intermediate_size > 0;
    // The surface carries the component's declared head geometry; a
    // family that varies it by layer (Gemma 4's global layers) records
    // each layer's geometry on its `AttentionLayerPolicy`, and the op
    // plan reads the layer's, so nothing here is averaged away.
    Ok(ExecutionSurface {
        // Carried from the architectural record, not re-derived. `None`
        // when the model declares no recurrence — every layer attends by
        // softmax and the operator is never reached.
        linear_attention: resolved.linear_attention.map(|t| LinearAttentionSurface {
            key_heads: t.key_heads,
            key_head_dim: t.key_head_dim,
            value_heads: t.value_heads,
            value_head_dim: t.value_head_dim,
            conv_kernel: t.conv_kernel,
            state_dtype: t.state_dtype,
        }),
        kda: resolved.kda,
        kda_gate_lower_bound: resolved.kda_gate_lower_bound,
        mamba2: resolved.mamba2.map(|geometry| Mamba2Surface {
            geometry,
            activation: execution.activation,
        }),
        conv_qkv: resolved.conv_qkv_attn,
        residual_in_fp32: execution.residual_in_fp32,
        mla: execution.mla.map(|m| MlaSurface {
            num_heads: m.num_heads,
            kv_lora_rank: m.kv_lora_rank,
            qk_nope_head_dim: m.qk_nope_head_dim,
            qk_rope_head_dim: m.qk_rope_head_dim,
            v_head_dim: m.v_head_dim,
        }),
        attention: attends.then_some(AttentionSurface {
            num_q_heads: resolved.num_q_heads,
            num_kv_heads: resolved.num_kv_heads,
            head_dim: resolved.head_dim,
            query_scale: execution.query_scale,
            score_scale: execution.score_scale,
            logit_softcapping: execution.attn_logit_softcapping,
            qk_norm_scope: execution.qk_norm_scope,
            qk_norm_weight_offset: execution.qk_norm_weight_offset,
            parameter_free_qk_norm: execution.parameter_free_qk_norm,
            // Judged per model; never inferred from operand presence.
            output_gate: execution.attention_output_gate,
            sinks: execution.attention_sinks,
            attention_bias: execution.attention_bias,
        }),
        ffn: has_ffn.then(|| FfnSurface {
            intermediate_size: resolved.intermediate_size,
            activation: execution.activation,
            ffn_type: execution.ffn_type,
            gate_policy: execution.gate_policy,
            moe: execution.moe.map(|m| MoeSurface {
                branch_scale: m.branch_scale,
                dense_prefix_layers: m.dense_prefix_layers,
                experts: m.experts,
                top_k: m.top_k,
                expert_intermediate_size: m.expert_intermediate_size,
                router_kind: m.router_kind,
                routing_policy: m.routing_policy,
                router_bias: m.router_bias,
                expert_format: m.expert_format,
                gate_up_layout: m.gate_up_layout,
                shared_experts: m.shared_experts,
                hybrid: m.hybrid,
            }),
        }),
        norm: NormSurface {
            pre: execution.norm_pre,
            post: execution.norm_post,
            final_norm: execution.norm_final,
            // From operand evidence via `attach_stack_evidence`, once the
            // builder knows the component's stack object.
            placement: None,
        },
        // Attached by the builder once it knows the component's objects.
        head: None,
        residual_scale: execution.residual_scale,
    })
}

/// Attach the facts only the stack's operand estate can state: norm
/// placement from the norm-role evidence across the stack's bindings.
/// Shared by the builder and the G4 re-derivation, so the two can never
/// judge the same bytes differently.
pub fn attach_stack_evidence(
    surface: &mut ExecutionSurface,
    inventory: &ArchitectureInventory,
    stack: &LogicalObject,
) -> Result<(), Vec<String>> {
    let relative: Vec<String> = stack
        .source_bindings
        .iter()
        .flat_map(|binding| {
            inventory
                .tensors
                .tensors
                .iter()
                .filter_map(|t| t.name.strip_prefix(&binding.tensor_prefix))
                .map(|rest| rest.trim_start_matches('.').to_string())
        })
        .collect();
    // A mixer-only program (every layer declared a Mamba2 recurrence)
    // reads its own placement evidence: one pre-mixer norm per layer, no
    // attention/FFN wrap norms. The choice is made from the DECLARED
    // program, so a transformer stack that lost its norms still fails the
    // transformer evidence rather than sliding into the mixer's.
    // A hybrid's attention layers carry the same single pre-mixer norm
    // (the mamba_ssm lineage wraps EVERY block, mixer or attention, in
    // one `norm.weight`), so a Full layer counts as mixer-normed exactly
    // when the conv-QKV block is declared — a transformer's Full layer
    // still reads the transformer evidence.
    let mixer_only = !inventory.resolved.layers.is_empty()
        && inventory
            .resolved
            .layers
            .iter()
            .all(|l| match l.declared_kind {
                Some(larql_models::config::LayerKind::Recurrent(
                    larql_models::config::RecurrenceFamily::Mamba2,
                )) => true,
                Some(larql_models::config::LayerKind::Full) => {
                    inventory.resolved.conv_qkv_attn.is_some()
                }
                _ => false,
            });
    let evidence = if mixer_only {
        super::roles::mixer_norm_placement_evidence(relative.iter().map(String::as_str))
    } else {
        super::roles::norm_placement_evidence(relative.iter().map(String::as_str))
    };
    match evidence {
        Ok(placement) => {
            surface.norm.placement = Some(placement);
            Ok(())
        }
        Err(reason) => Err(vec![format!("norm placement ({reason})")]),
    }
}

/// The head surface for a component that owns embedding/output-head
/// objects, from the same resolution.
pub fn head_from_resolved(inventory: &ArchitectureInventory) -> Result<HeadSurface, Vec<String>> {
    let resolved = &inventory.resolved;
    let mut missing = Vec::new();
    let Some(execution) = &resolved.execution else {
        return Err(vec![
            "resolved.execution (pre-v3 inventory — re-run inspect-hf)".to_string(),
        ]);
    };
    let Some(vocab_size) = resolved.vocab_size else {
        missing.push("vocab_size".to_string());
        return Err(missing);
    };
    Ok(HeadSurface {
        vocab_size,
        embedding_norm: execution.embedding_norm,
        embed_scale: execution.embed_scale,
        output_multiplier: execution.output_multiplier,
        final_logit_softcapping: execution.final_logit_softcapping,
        head_reuses_embedding: execution.head_reuses_embedding,
    })
}

/// Build the surface for a perception component from its nested config
/// reading plus tensor evidence.
///
/// A nested component has no detection trait to resolve through, so the
/// judged derivations live here, in one place: MHA (kv = q) unless
/// declared otherwise, `head_dim = hidden/heads` when divisible, canonical
/// 1/√d scaling, norm kind from which epsilon spelling the config
/// declares, and FFN gating from whether gate tensors exist under the
/// tower (`has_gate_tensors` — evidence, not a family fact).
pub fn surface_from_nested(
    nested: &ComponentTopology,
    has_gate_tensors: bool,
) -> Result<ExecutionSurface, Vec<String>> {
    let mut missing = Vec::new();
    let hidden = nested.hidden_size.unwrap_or(0);
    let heads = match nested.num_attention_heads {
        Some(h) if h > 0 => h,
        _ => {
            missing.push("num_attention_heads".to_string());
            0
        }
    };
    let head_dim = match nested.head_dim {
        Some(d) => d,
        None if heads > 0 && hidden.is_multiple_of(heads) => hidden / heads,
        None => {
            // Two different defects reach here and must not read alike.
            // `heads == 0` is the sentinel for "no readable head count",
            // so formatting it into the arithmetic produced Qwen3.8's
            // "hidden 1152 not divisible by 0 heads" — a nonsense sum
            // standing in for an unread config spelling. A genuine
            // indivisibility is a different fact and keeps its own words.
            missing.push(if heads == 0 {
                "head_dim (no readable attention-head count to derive it from)".to_string()
            } else {
                format!("head_dim (hidden {hidden} not divisible by {heads} heads)")
            });
            0
        }
    };
    let intermediate_size = match nested.intermediate_size {
        Some(i) => i,
        None => {
            missing.push("intermediate_size".to_string());
            0
        }
    };
    let activation = match nested
        .hidden_act
        .as_deref()
        .and_then(Activation::from_hf_name)
    {
        Some(a) => a,
        None => {
            missing.push(format!(
                "hidden_act (declared {:?}, judged mapping required)",
                nested.hidden_act
            ));
            Activation::Gelu
        }
    };
    // The epsilon spelling the config declares names the norm kind; a
    // component declaring neither has no norm surface to persist.
    let (kind, eps) = match (nested.norm_kind, nested.norm_eps) {
        (Some(kind), Some(eps)) => (kind, eps),
        _ => {
            missing.push("norm_eps".to_string());
            (NormType::LayerNorm, 0.0)
        }
    };
    if !missing.is_empty() {
        return Err(missing);
    }
    Ok(ExecutionSurface {
        // No judged perception tower declares a linear-attention
        // recurrence, Multi-Latent Attention, or an SSM mixer.
        linear_attention: None,
        kda: None,
        kda_gate_lower_bound: None,
        mla: None,
        mamba2: None,
        conv_qkv: None,
        residual_in_fp32: None,
        attention: Some(AttentionSurface {
            num_q_heads: heads,
            num_kv_heads: nested.num_key_value_heads.unwrap_or(heads),
            head_dim,
            // No perception tower has declared a query-scale operation.
            // `None` says that; `Some(1.0)` would claim the source
            // specifies a multiply by one.
            query_scale: None,
            score_scale: (head_dim as f64).powf(-0.5),
            logit_softcapping: None,
            qk_norm_scope: QkNormScope::PerHead,
            qk_norm_weight_offset: 0.0,
            parameter_free_qk_norm: ParameterFreeQkNorm::default(),
            output_gate: None,
            sinks: None,
            // What the tower declares about its projection biases, when
            // it declares anything (Gemma 4 vision: `false`); the loader's
            // tensor-presence check answers otherwise, as for text.
            attention_bias: nested.tower.attention_bias,
        }),
        ffn: Some(FfnSurface {
            intermediate_size,
            activation,
            ffn_type: if has_gate_tensors {
                FfnType::Gated
            } else {
                FfnType::Standard
            },
            // Nested towers declare no gate policy; plain gating is the
            // fact, not a fallback.
            gate_policy: larql_models::ExpertGatePolicy::Gated,
            moe: None,
        }),
        norm: NormSurface {
            pre: NormSpec {
                kind,
                eps,
                weight_offset: 0.0,
            },
            // Unjudged, not "the same as `pre`". Perception towers have
            // no four-norm placement here, so nothing consumes it — and
            // claiming an equivalence nobody established is the
            // inherited-default failure this shape exists to prevent.
            post: None,
            final_norm: NormSpec {
                kind,
                eps,
                weight_offset: 0.0,
            },
            // Perception towers keep their own norm topology; placement
            // is a decoder-stack concept until the perception op set (5d)
            // defines its own.
            placement: None,
        },
        head: None,
        // No perception tower has declared a residual-scale operation.
        residual_scale: None,
    })
}
