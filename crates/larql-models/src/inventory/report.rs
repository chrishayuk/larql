//! Output types for the architecture inventory — the JSON `larql inspect-hf`
//! emits.
//!
//! Two deliberately separate sections carry the same subject matter from two
//! authorities:
//!
//! - **`config_keys`** — what the checkpoint *declares*, verbatim, with a
//!   per-key statement of whether this build's config parser reads it.
//! - **`resolved`** — what this build's detection *does* with it: the
//!   per-layer policy table the serving path would actually run.
//!
//! Disagreement between the two sections is the instrument's whole point.
//! A config fact the parser never reads (`status: unconsumed`) is the
//! [§4.7.8] failure shape *before* it ships: a field with one behavioural
//! default silently answering for every family. Five distinct serving bugs
//! have had that shape; this report makes the sixth visible from
//! `config.json` alone, with no forward pass.
//!
//! [§4.7.8]: ../../../../docs/k3-funnel.md

use serde::{Deserialize, Serialize};

/// Current inventory schema. Bump on any breaking change to these types.
///
/// v2: [`LayerPolicy`] carries a [`PositionPolicy`](crate::config::PositionPolicy)
/// instead of a bare `rope_base` — NoPE layers are a policy variant, not a
/// numeric sentinel.
///
/// v3: [`ResolvedTopology`] carries [`ResolvedExecution`] — the execution
/// scalars a generic executor reads, fully resolved. A v2 inventory
/// deserialises with `execution: None`, which downstream completeness
/// gates treat as *incomplete*, never as defaults.
///
/// v4: attention scaling splits into `query_scale` (applied to the
/// normalised query, as the reference implementations do) and
/// `score_scale` (the canonical score-time multiply) — folding them is
/// algebra-equivalent but not fp-equivalent, and moving a multiply
/// across RoPE/matmul is exactly the kind of silent normalisation strict
/// parity would later expose. Adds the judged attention-gate spec and
/// parameter-free QK norm. A v3 inventory fails to parse rather than
/// answering with a folded scale.
///
/// v5: [`LayerPolicy`] carries `declared_span` — the checkpoint's own
/// `layer_types` entry for the layer, verbatim, alongside the resolved
/// `"sliding"`/`"full"` boolean. Closes the gap a hybrid linear-attention
/// interleave (Qwen3.5, Kimi-Linear-style) fell into: `is_sliding_window_
/// layer` only ever answers a boolean, so every layer that was not
/// literally `"sliding_attention"` resolved to `"full"` with nothing
/// downstream able to tell a genuine full-attention layer from a
/// defaulted one. A v4 inventory deserialises with `declared_span: None`.
pub const INVENTORY_SCHEMA: u32 = 5;

/// Machine-readable architecture inventory of one HF checkpoint directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitectureInventory {
    /// Always [`INVENTORY_SCHEMA`].
    pub schema: u32,
    /// The inspected directory, as given by the caller.
    pub path: String,
    pub identity: Identity,
    pub detection: Detection,
    pub resolved: ResolvedTopology,
    /// Nested sibling components (`vision_config`, …), each read into a
    /// typed topology by the generic component reader. Empty for
    /// single-component checkpoints.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nested_components: Vec<crate::inventory::components::ComponentTopology>,
    /// The checkpoint's declared stored representation
    /// (`quantization_config`), read by
    /// [`super::representation::read_stored_representation`]. `None` for
    /// an unquantised checkpoint. Decides what raw-byte tensors *mean* —
    /// GPT-OSS's `U8` expert blocks and scales are MXFP4 by this
    /// declaration alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stored_representation: Option<crate::inventory::representation::StoredRepresentation>,
    /// The joins between components — special-token roles, soft-token
    /// counts, declared-absent towers, bidirectional masking — as the
    /// interface reader recorded them. `None` on a text-only checkpoint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multimodal_interface: Option<crate::inventory::interfaces::MultimodalInterface>,
    /// Every leaf in `config.json`, flattened to a dot path, classified.
    pub config_keys: Vec<ConfigKeyFact>,
    /// Declared cross-component interfaces (see
    /// [`super::config_keys::KNOWN_INTERFACE_KEYS`]).
    pub interfaces: Vec<InterfaceFact>,
    pub tensors: TensorInventory,
}

/// What the checkpoint says it is, before any interpretation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    /// `model_type`, read from `text_config` first, then top level.
    pub model_type: String,
    /// The *other* declaration, when a nested text config supplied
    /// [`Self::model_type`] and the container declares one too.
    ///
    /// Kept because the two are not interchangeable facts and the reader
    /// above throws one of them away. Kimi K3 declares `kimi_k3` at the
    /// container and `kimi_linear` under `text_config`, and taking the
    /// text declaration alone dispatches a 93-layer, 1.56 TB model to the
    /// Kimi-Linear-48B implementation.
    ///
    /// K3-ARCH-1 registered `kimi_k3` and had it declare `kimi_linear` as
    /// its text component, so the container reading no longer falls to
    /// the generic architecture: it resolves to a registry entry that is
    /// identified and deliberately not executable. That makes the gate's
    /// job narrower, not unnecessary — a container declaring a family it
    /// does NOT relate to its text component still refuses, and only both
    /// declarations surviving lets the gate tell the two cases apart.
    ///
    /// Registry lookup only. `detect_from_json` still prefers
    /// `text_config.model_type` and hands a real K3 config to the
    /// ancestor; that gap is pinned in `architectures::kimi_k3` and is
    /// the reason this field cannot be collapsed into one reading.
    ///
    /// `None` when the config is flat, or when only one level declares.
    /// Usually equal-in-meaning rather than conflicting — 27 of the 28
    /// checkpoints in the conformance corpus that declare at both levels
    /// use the `<container>_text` suffix form — so the divergence that
    /// matters is not string inequality but the two resolving to
    /// different architectures.
    pub container_model_type: Option<String>,
    /// HF `architectures` list, verbatim.
    pub architectures: Vec<String>,
    /// Checkpoint dtype (`dtype` or `torch_dtype`).
    pub dtype: Option<String>,
    pub transformers_version: Option<String>,
    /// Nested component configs present (`text_config`, `vision_config`, …).
    /// A flat config reports none.
    pub components: Vec<String>,
}

/// What this build's detection resolves the checkpoint to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Detection {
    /// `ModelArchitecture::family()` of the detected implementation.
    pub family: String,
    /// True when `model_type` matched no registry entry and detection fell
    /// through to the generic architecture. A generic fallback on a model
    /// with unconsumed config keys is the loudest red flag this report can
    /// raise: the model will load and serve, and serve wrong.
    pub generic_fallback: bool,
    /// Attention kind from the registry entry, when one matched.
    pub attention_kind: Option<String>,
    /// `ModelArchitecture::validate()` findings, carried as data — an
    /// inventory of an unsupported model must describe it, not refuse it.
    pub validation_errors: Vec<String>,
}

/// The topology the serving path would run, including the per-layer
/// attention policy table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedTopology {
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    /// Per-layer dense-FFN width a derived checkpoint declares; `None`
    /// means every layer runs at `intermediate_size`. Additive: an
    /// inventory JSON written before it reads as `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ffn_intermediate_size_by_layer: Option<Vec<usize>>,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: Option<usize>,
    pub sliding_window: Option<usize>,
    /// How far the programme is declared to run — `max_position_embeddings`
    /// or a family's spelling of it. Read here so the graph can record it:
    /// the encoder already judged it execution-semantic, and a judged fact
    /// with nowhere to land is the same as an unread one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<usize>,
    pub attention: AttentionSummary,
    /// One entry per layer, in layer order.
    pub layers: Vec<LayerPolicy>,
    /// Execution scalars, fully resolved. `None` only when read from a
    /// pre-v3 inventory JSON — a fact for completeness gates, not a
    /// licence to default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<ResolvedExecution>,
    /// Geometry of the model's linear-attention layers, when it declares
    /// any. `None` on a model whose every layer attends by softmax.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub linear_attention: Option<LinearAttentionTopology>,
    /// The KDA block's declared geometry, when the checkpoint declares one
    /// (`linear_attn_config`). Disjoint from [`Self::linear_attention`] in
    /// every observed checkpoint: the two describe different operators, so
    /// carrying them in one field would force a reader to guess which.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kda: Option<crate::config::KdaGeometry>,
    /// KDA's decay-gate lower bound (`linear_attn_config.gate_lower_bound`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kda_gate_lower_bound: Option<f32>,
    /// What the family's reference actually does with
    /// [`Self::kda_gate_lower_bound`] — an architecture fact, resolved by
    /// [`ModelArchitecture::kda_gate_form`](crate::config::ModelArchitecture::kda_gate_form),
    /// because the declared value does not determine it. `None` = the
    /// family has not been judged, which must reach a refusal.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kda_gate_form: Option<crate::config::KdaGateForm>,
    /// The FORM of KDA's output gate (`linear_attn_config.use_full_rank_gate`),
    /// verbatim: `Some(true)` = one full-rank `g_proj`, `Some(false)` = the
    /// low-rank pair, `None` = undeclared (the reference's own default is the
    /// pair). Carried as declared so an executor can hold the shipped
    /// operands to it rather than infer the form from them. K3-REP-GATE-1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kda_use_full_rank_gate: Option<bool>,
    /// The Mamba2 mixer's declared geometry, when the checkpoint declares
    /// one. Disjoint from [`Self::linear_attention`] and [`Self::kda`] in
    /// every observed checkpoint — a third recurrence family, and the
    /// geometries are not interchangeable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mamba2: Option<crate::config::Mamba2Geometry>,
    /// How the Mamba2 geometry was read: dialect and recorded family
    /// defaults. Present exactly when [`Self::mamba2`] is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mamba2_provenance: Option<crate::config::Mamba2Provenance>,
    /// The hybrid stack's conv-QKV attention geometry, when declared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conv_qkv_attn: Option<crate::config::ConvQkvAttnGeometry>,
    /// How the conv-QKV geometry was read: dialect and recorded family
    /// defaults. Present exactly when [`Self::conv_qkv_attn`] is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conv_qkv_provenance: Option<crate::config::ConvQkvProvenance>,
    /// `pad_vocab_size_multiple` — the embedding rows are the declared
    /// vocab rounded UP to this multiple (mamba_ssm lineage). The head
    /// genuinely emits the padded width; the declared vocab names the
    /// meaningful prefix.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pad_vocab_size_multiple: Option<usize>,
}

/// The precision a recurrent state is kept at.
///
/// Fail-closed by construction: a spelling this build does not represent
/// answers `None` and stays blocking, rather than resolving to the model's
/// bulk dtype. Only `float32` is represented, because that is the only
/// value any judged checkpoint declares — adding speculative variants
/// would claim representation this engine has never executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecurrentStateDtype {
    Float32,
}

impl RecurrentStateDtype {
    pub fn from_declared(value: &str) -> Option<Self> {
        match value {
            "float32" | "f32" => Some(Self::Float32),
            _ => None,
        }
    }

    pub fn declared_name(self) -> &'static str {
        match self {
            Self::Float32 => "float32",
        }
    }
}

/// Gated DeltaNet geometry — an architectural fact, like the head counts
/// beside it.
///
/// The key and value sides are separate facts and stay separate: Qwen3.8
/// declares 16 key heads and 48 value heads, so no single `num_heads`
/// describes this operator. Anything that folded them would have to pick
/// one, and picking either is wrong.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinearAttentionTopology {
    /// `linear_num_key_heads` — Hk (16 on Qwen3.8).
    pub key_heads: usize,
    /// `linear_key_head_dim` — Dk (128).
    pub key_head_dim: usize,
    /// `linear_num_value_heads` — Hv (48). Larger than [`Self::key_heads`]
    /// by design, not a GQA sharing ratio: the value side is the axis the
    /// recurrent state is blocked along.
    pub value_heads: usize,
    /// `linear_value_head_dim` — Dv (128).
    pub value_head_dim: usize,
    /// `linear_conv_kernel_dim` — the depthwise causal convolution width
    /// over the fused q|k|v channels (4).
    pub conv_kernel: usize,
    /// `mamba_ssm_dtype` — the precision the recurrence keeps its state at,
    /// which Qwen3.8 declares as `float32` against a bf16 model. `None`
    /// when undeclared or spelled in a way this build does not represent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub state_dtype: Option<RecurrentStateDtype>,
}

impl LinearAttentionTopology {
    /// Channels the fused q|k|v projection emits: query and key at the key
    /// geometry, value at the value geometry.
    ///
    /// Derived, never stored, so it cannot drift from the head counts it
    /// is computed from. On Qwen3.8 this closes exactly:
    /// `2·16·128 + 48·128 = 10240`, the observed `in_proj_qkv` row count.
    pub fn qkv_channels(self) -> usize {
        self.key_heads * self.key_head_dim * 2 + self.value_heads * self.value_head_dim
    }

    /// Width of the value/gate side: `Hv·Dv` (6144 on Qwen3.8).
    pub fn value_width(self) -> usize {
        self.value_heads * self.value_head_dim
    }

    /// Every declared dimension present, or `None` — a partially declared
    /// recurrence is refused rather than completed with defaults.
    pub fn from_config(cfg: &crate::config::ModelConfig) -> Option<Self> {
        Some(Self {
            key_heads: cfg.linear_num_key_heads?,
            key_head_dim: cfg.linear_key_head_dim?,
            value_heads: cfg.linear_num_value_heads?,
            value_head_dim: cfg.linear_value_head_dim?,
            conv_kernel: cfg.linear_conv_kernel_dim?,
            state_dtype: cfg
                .mamba_ssm_dtype
                .as_deref()
                .and_then(RecurrentStateDtype::from_declared),
        })
    }
}

/// The execution-scalar surface resolution produces: everything a generic
/// executor reads beyond topology and the per-layer table, **fully
/// resolved**. Every defaulting decision (an absent `hidden_act`, a shared
/// post-norm epsilon) is applied here, once, by the same detection surface
/// the serving path runs — an executor consuming these values never
/// defaults anything.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolvedExecution {
    /// The query-scale operation: a multiplier on the (normalised) query
    /// states before position encoding. `None` = the model declares no
    /// such operation, which is a different claim from `Some(1.0)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query_scale: Option<f64>,
    /// Canonical score-time multiplier on QK^T:
    /// (`query_pre_attn_scalar` or `head_dim`)^-0.5. Kept separate from
    /// [`Self::query_scale`]: folding them is algebra-equivalent but not
    /// fp-equivalent.
    pub score_scale: f64,
    /// Attention-logit softcap; `None` = the op is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attn_logit_softcapping: Option<f32>,
    /// Scope of QK normalisation, when QK-norm weights exist in the stack.
    pub qk_norm_scope: crate::config::QkNormScope,
    /// Offset added to QK-norm weights at runtime (Gemma 2/3: 1.0).
    pub qk_norm_weight_offset: f32,
    /// Parameter-free QK normalisation (no weight tensors) — a judged
    /// semantic fact no tensor evidence can reveal.
    #[serde(default)]
    pub parameter_free_qk_norm: crate::config::ParameterFreeQkNorm,
    /// Judged attention-output-gate semantics; `None` = no judgment
    /// exists (a shipped gate operand then fails operand closure).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_output_gate: Option<crate::config::AttentionGateSpec>,
    /// Judged attention-sink semantics; `None` = no judgment exists (a
    /// shipped `self_attn.sinks` operand then fails operand closure).
    /// Defaults for inventories written before it was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_sinks: Option<crate::config::AttentionSinkSpec>,
    /// Whether the Q/K/V/O projections carry additive biases, as the
    /// checkpoint declares (`attention_bias`). `None` = undeclared, which
    /// is not "no bias": bias operands shipped under `None` fail operand
    /// closure; `Some(true)` requires all four. Defaults for inventories
    /// written before it was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_bias: Option<bool>,
    /// The routed-FFN facts when the family declares experts; `None` = a
    /// dense-FFN model. Defaults for inventories written before it was
    /// recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub moe: Option<MoeExecution>,
    /// Multi-Latent Attention geometry when the family declares one
    /// (`ModelArchitecture::uses_mla`); `None` = ordinary per-position K/V
    /// attention. Applies to every [`LayerPolicy`] whose declared kind is
    /// [`crate::config::LayerKind::Full`] — MLA compresses the KV cache,
    /// not the FFN, so it is orthogonal to `moe`'s dense-prefix layers and
    /// uniform across a family's non-recurrent layers, the same way KDA's
    /// geometry is uniform across its recurrent ones.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mla: Option<MlaExecution>,
    pub activation: crate::config::Activation,
    pub ffn_type: crate::config::FfnType,
    /// How the FFN's gate combines with its up branch. `Gated` is the
    /// plain `activation(gate) * up`; GPT-OSS's clamped GLU is a distinct
    /// policy, not an activation variant — see `ExpertGatePolicy`.
    /// Defaults for inventories written before it was recorded.
    #[serde(default)]
    pub gate_policy: crate::config::ExpertGatePolicy,
    /// Complete norm spec for the pre-attention / pre-FFN sites.
    pub norm_pre: crate::config::NormSpec,
    /// Complete norm spec for the post-attention / post-FFN sites.
    /// `None` = unjudged; a four-norm stack in that state is refused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub norm_post: Option<crate::config::NormSpec>,
    /// Complete norm spec for the final norm before the head. Separate
    /// because a family may use a different convention there — Glimmer
    /// uses a centred norm in its layers and an ordinary one here.
    pub norm_final: crate::config::NormSpec,
    /// Normalisation applied to embedding-table output. `None` = no such
    /// operation. Weightless, so no operand evidences it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding_norm: Option<crate::config::EmbeddingNorm>,
    /// Whether layers carry separate post-norms around attention/FFN
    /// (Gemma-style four-norm layers) in addition to pre-norms.
    pub post_norms: bool,
    /// The embedding-scale operation, applied after lookup. `None` = the
    /// model declares no such operation, distinct from `Some(1.0)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embed_scale: Option<f32>,
    /// The output-multiplier operation, applied before the vocabulary
    /// projection. `None` = the model declares no such operation,
    /// distinct from `Some(1.0)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_multiplier: Option<f64>,
    /// Final-logit softcap; `None` = the op is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_logit_softcapping: Option<f32>,
    /// Residual-stream scaling: the attention/FFN sublayer's own output is
    /// multiplied by this before its residual add, at both sites with the
    /// same value (Granite's `residual_multiplier`). `None` = the model
    /// declares no such operation, distinct from `Some(1.0)`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual_scale: Option<f32>,
    /// Whether the residual stream is kept at fp32 against a
    /// lower-precision model (`residual_in_fp32`), verbatim. `None` = the
    /// checkpoint declares nothing, which is not `false` — an executor
    /// must not choose a residual precision the checkpoint never stated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual_in_fp32: Option<bool>,
    /// How the residual stream is shaped and recombined across this
    /// component. A component fact, because every consumer of the
    /// residual — embedding, branch operators, head — must agree.
    ///
    /// `None` means the checkpoint declared the topology PARTIALLY and
    /// nothing resolved it. That is deliberately not the same as
    /// `Some(SingleStream)`: defaulting a half-declared bundle to one
    /// stream is precisely the failure this field exists to prevent, so
    /// the surface refuses to build instead.
    #[serde(default = "single_stream_topology")]
    pub residual_topology: Option<crate::config::ResidualTopology>,
    /// Why [`Self::residual_topology`] is absent, verbatim from the one
    /// authority that decided it
    /// ([`ModelArchitecture::residual_topology`](crate::config::ModelArchitecture::residual_topology)).
    ///
    /// Two different declarations resolve to nothing and they mean
    /// opposite things to whoever acts on them: a Sinkhorn declaration
    /// missing a parameter may be a topology this build has not judged,
    /// while a checkpoint declaring TWO topologies has told this build
    /// two incompatible things about one residual. A single hardcoded
    /// reason downstream sent a reader of the second case to look for a
    /// missing iteration count, so the reason travels with the absence
    /// rather than being re-invented where it is printed.
    ///
    /// `Some` exactly when [`Self::residual_topology`] is `None` on an
    /// inventory this build resolved; both `None` on a pre-existing
    /// inventory JSON written before the field existed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual_topology_refusal: Option<String>,
    /// Whether a missing standalone output-head tensor means "tied to the
    /// embedding matrix" rather than "lost". See
    /// [`ModelArchitecture::output_head_reuses_embedding`](crate::config::ModelArchitecture::output_head_reuses_embedding).
    /// `#[serde(default)]` so a pre-existing inventory JSON without this
    /// field deserialises to `false` — the conservative "not proven tied"
    /// reading, not a silent claim either way.
    #[serde(default)]
    pub head_reuses_embedding: bool,
}

/// Counts over [`ResolvedTopology::layers`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionSummary {
    pub sliding_layers: usize,
    pub full_layers: usize,
}

/// Per-layer attention policy as the architecture trait resolves it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerPolicy {
    pub layer: usize,
    /// `"sliding"` or `"full"` — the boolean split
    /// `ModelArchitecture::is_sliding_window_layer` resolves to.
    /// [`Self::declared_span`] is the finer-grained fact when the
    /// checkpoint states one.
    pub attention: String,
    /// The checkpoint's own per-layer declaration for this layer, in
    /// `layer_types` vocabulary.
    ///
    /// Verbatim from `layer_types` when the checkpoint writes that array.
    /// When it does not, the equivalent declaration in the index-set
    /// spelling (`linear_attn_config.{kda_layers, full_attn_layers}`)
    /// answers instead, normalised into the same vocabulary — the fact is
    /// the same one, and carrying two spellings downstream would grow a
    /// second code path in every consumer. `None` only when the checkpoint
    /// states no per-layer split at all (an implicit stride, or a plain
    /// single-attention-type model) — [`Self::attention`] is then the only
    /// source of truth.
    ///
    /// Carries a vocabulary [`Self::attention`] cannot: a hybrid
    /// linear-attention interleave (`"linear_attention"`) is neither
    /// `"sliding"` nor `"full"`, and the boolean split alone cannot tell
    /// a defaulted layer from a genuine one. See `docs/k3-funnel.md`
    /// §4.7.8.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_span: Option<String>,
    /// The same declaration in the canonical vocabulary, carrying what the
    /// `layer_types` spelling cannot: which recurrence family, and a
    /// sliding layer's window. `None` when the checkpoint declares no
    /// per-layer topology, or declared one this build could not resolve.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub declared_kind: Option<crate::config::LayerKind>,
    /// Window size when sliding, `None` on full-attention layers.
    pub window: Option<usize>,
    /// How this layer encodes position — rotary at a base, or not at all.
    pub position: crate::config::PositionPolicy,
    pub head_dim: usize,
    pub num_kv_heads: usize,
    /// The value projection IS the key projection on this layer: no
    /// `v_proj` tensor exists and V is the raw K projection (before the
    /// key's norm and rotation) — Gemma 4 `attention_k_eq_v` on its full
    /// layers. Per layer, because the same checkpoint keeps `v_proj` on
    /// its sliding layers. Defaults for inventories written before it was
    /// recorded.
    #[serde(default)]
    pub v_from_k: bool,
    /// Source-name prefix of this layer's packed expert bank (the parent of
    /// its `gate_up`/`down` operands), when the layer's FFN is routed —
    /// `model.layers.3.mlp.experts`. `None` = a dense-FFN layer. Per layer,
    /// so an arbitrary dense/routed schedule needs no dedicated field.
    /// Defaults for inventories written before it was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_bank: Option<String>,
}

/// The routed-FFN (mixture-of-experts) execution facts, resolved once
/// from the architecture. Every field is a judged semantic the executor
/// reads — none is re-derived from operand names downstream.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MoeExecution {
    /// Multiplier on the routed-expert branch (`routed_scaling_factor`).
    /// `None` when undeclared — never 1.0, which is a different claim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch_scale: Option<f64>,
    /// Leading layers that run a dense MLP instead of the routed block
    /// (`first_k_dense_replace`). `None` when the checkpoint declares no
    /// prefix; `Some(0)` is a declared absence of one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dense_prefix_layers: Option<usize>,
    /// Routed experts per layer.
    pub experts: usize,
    /// Experts selected per token.
    pub top_k: usize,
    /// Per-expert intermediate width.
    pub expert_intermediate_size: usize,
    /// How router logits become selected experts and weights.
    pub router_kind: crate::config::MoeRouterKind,
    /// Whether the selected weights are normalised to sum to one.
    pub routing_policy: crate::config::ExpertRoutingPolicy,
    /// Whether the router carries an additive bias on its logits.
    pub router_bias: bool,
    /// How the experts are stored (per-expert tensors, packed MXFP4, packed
    /// BF16).
    pub expert_format: crate::config::ExpertFormat,
    /// How a fused `gate_up` operand splits into its branches; `None` when
    /// no fused operand exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gate_up_layout: Option<crate::config::GateUpLayout>,
    /// Always-active experts alongside the routed ones.
    pub shared_experts: usize,
    /// The width of that always-active branch, resolved once by
    /// [`ModelArchitecture::shared_expert_intermediate_size`](crate::config::ModelArchitecture::shared_expert_intermediate_size)
    /// — the checkpoint's own key where it declares one, the
    /// count-times-routed-width convention where it declares a count
    /// instead. `None` iff [`Self::shared_experts`] is zero.
    ///
    /// Carried rather than re-derived downstream: the two conventions
    /// disagree by 4x on Qwen1.5-MoE, so a second derivation is a second
    /// answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_expert_intermediate_size: Option<usize>,
    /// The gate on that branch's output, where the family runs one
    /// (Qwen MoE's `sigmoid(shared_expert_gate(x)) *`). `None` = summed
    /// unscaled, the DeepSeek/Kimi form.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_expert_gate: Option<crate::config::SharedExpertGateSpec>,
    /// A dense MLP summed with the expert block every layer (Gemma 4 A4B).
    pub hybrid: bool,
}

/// Multi-Latent Attention geometry, resolved once from the architecture.
/// `None` on [`ResolvedExecution::mla`] means the family runs ordinary
/// per-position K/V, not "MLA with defaulted dimensions" — every field
/// here is load-bearing for the compressed-KV operand shapes, and a
/// wrong-but-plausible default would silently accept a mis-shaped tensor.
///
/// Kimi Linear ships no `q_lora_rank` (`assert self.q_lora_rank is None`
/// in the checkpoint's own `modeling_kimi.py` — Q is one dense
/// projection, only K/V are low-rank compressed). A family that DOES
/// compress Q needed its own extension rather than a guess from this
/// one; [`MlaQueryForm`](crate::config::MlaQueryForm) is that extension,
/// carried in [`Self::query`] as a form rather than as a bare rank, so
/// that a layer without the factorisation cannot describe one.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MlaExecution {
    // (`query`'s serde default lives at `direct_query`, below.)
    /// Query/output head count (`num_attention_heads`) — the decompressed
    /// K/V side always produces this many heads' worth of output, not
    /// `num_key_value_heads`: MLA's compression is the efficiency
    /// mechanism, not a GQA-style head reduction after decompression.
    pub num_heads: usize,
    /// Compressed KV latent width (`kv_lora_rank`).
    pub kv_lora_rank: usize,
    /// Non-RoPE portion of the query/key head width.
    pub qk_nope_head_dim: usize,
    /// RoPE portion of the query/key head width, shared (MQA-style)
    /// across every head from the SAME compressed projection.
    pub qk_rope_head_dim: usize,
    /// Value head width — independent of the query/key head width; MLA's
    /// asymmetry is structural, not an approximation.
    pub v_head_dim: usize,
    /// Epsilon of the latent norm (`kv_a_layernorm`) that stands between
    /// the compressed cache and its decompression — the family's own
    /// reference value, NOT `rms_norm_eps` (see
    /// [`ModelArchitecture::mla_kv_a_norm_eps`](crate::config::ModelArchitecture::mla_kv_a_norm_eps)).
    ///
    /// `None` = this family's reference has not been read for it, and an
    /// executor must refuse rather than substitute the layer eps.
    /// Defaults for inventories written before it was recorded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_a_norm_eps: Option<f64>,
    /// Which query these layers build — one dense `q_proj`, or Kimi-K3's
    /// `q_a_proj` -> `q_a_layernorm` -> `q_b_proj` factorisation under a
    /// declared `q_lora_rank`.
    ///
    /// Resolved from the DECLARATION
    /// ([`ModelArchitecture::mla_query_form`](crate::config::ModelArchitecture::mla_query_form)),
    /// never from which operands the checkpoint ships: `q_proj` and
    /// `q_b_proj` share a row count and differ only in their columns.
    /// Defaults to `Direct` for inventories written before it was
    /// recorded, which is what those checkpoints declared.
    #[serde(default = "direct_query")]
    pub query: crate::config::MlaQueryForm,
    /// The output gate the checkpoint declares on its MLA layers
    /// (`mla_use_output_gate: true`), as the generic gated-attention
    /// operation it implements — the same spec the softmax family's
    /// `attn_output_gate` resolves to, judged from Kimi-K3's own reference
    /// ([`AttentionGateSpec::from_attention_input_sigmoid_before_output_projection`]).
    /// `None` = no gate: undeclared, or declared `false`. Defaults for
    /// inventories written before it was recorded. K3-REP-GATE-1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_gate: Option<crate::config::AttentionGateSpec>,
}

/// One flattened `config.json` leaf.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigKeyFact {
    /// Dot path from the config root, e.g. `text_config.qk_scale_factor`.
    pub path: String,
    /// The leaf value, verbatim. Arrays are leaves.
    pub value: serde_json::Value,
    pub status: KeyStatus,
}

/// Whether this build's config parser reads a key.
///
/// Scope: classification is against `larql-models`' `ModelConfig` parser
/// only. Keys other subsystems read (tokenizer ids, vision loaders) are not
/// credited here — `metadata` covers the identity/training facts known to be
/// inert for a text forward pass, and everything else unread is
/// `unconsumed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KeyStatus {
    /// The config parser reads this key (directly or via an alias list).
    Consumed,
    /// Identity or training-time fact with no bearing on a forward pass.
    Metadata,
    /// Declared by the checkpoint, read by nothing in the config parser.
    /// Every entry here is a potential silent-default serving bug.
    Unconsumed,
}

/// A config field that declares a cross-component interface — e.g. a
/// drafter's `target_layer_ids` naming the target hidden states it consumes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceFact {
    pub path: String,
    pub value: serde_json::Value,
}

/// Tensor-level inventory from safetensors headers. Shapes and dtypes come
/// from the shard headers alone; no tensor data is read.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInventory {
    /// Shard filenames scanned, relative to the model dir, sorted.
    pub files: Vec<String>,
    pub total_tensors: usize,
    pub total_bytes: u64,
    /// Totals grouped by name prefix (path up to the first numeric
    /// segment), sorted by prefix.
    pub groups: Vec<TensorGroup>,
    /// Every tensor, sorted by name.
    pub tensors: Vec<TensorFact>,
}

/// Aggregate over one name prefix, e.g. `model.language_model.layers`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorGroup {
    pub prefix: String,
    pub tensors: usize,
    pub bytes: u64,
}

/// One stored tensor, as its shard header describes it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorFact {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: u64,
    /// Shard filename holding the bytes, relative to the model dir.
    pub file: String,
}

/// Containers written before residual topology was a field carried one
/// stream, which is what every family judged then used.
fn single_stream_topology() -> Option<crate::config::ResidualTopology> {
    Some(crate::config::ResidualTopology::SingleStream)
}

/// `serde` default for [`MlaExecution::query`]: what every inventory
/// written before the query form was recorded declared.
fn direct_query() -> crate::config::MlaQueryForm {
    crate::config::MlaQueryForm::Direct
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The recurrent state dtype round-trips, and refuses what it
    /// does not represent.**
    ///
    /// `None` means undeclared or spelled in a way this build cannot
    /// represent — never "float32 by default". Qwen3.8 declares
    /// `float32` against a bf16 model, so a defaulted answer would put
    /// the recurrence at the model's precision and quietly change the
    /// operator.
    #[test]
    fn the_recurrent_state_dtype_round_trips_and_refuses_the_unrepresented() {
        for spelling in ["float32", "f32"] {
            assert_eq!(
                RecurrentStateDtype::from_declared(spelling),
                Some(RecurrentStateDtype::Float32),
                "`{spelling}` is a spelling of the same dtype"
            );
        }
        assert_eq!(
            RecurrentStateDtype::Float32.declared_name(),
            "float32",
            "the canonical spelling is what a container records"
        );
        assert_eq!(
            RecurrentStateDtype::from_declared(RecurrentStateDtype::Float32.declared_name()),
            Some(RecurrentStateDtype::Float32),
            "the canonical spelling must parse back"
        );
        for unknown in ["bfloat16", "float16", "fp32", "", "FLOAT32"] {
            assert_eq!(
                RecurrentStateDtype::from_declared(unknown),
                None,
                "`{unknown}` is not represented, so it must be refused rather than \
                 approximated by the one variant that exists"
            );
        }
    }

    /// **The linear-attention widths are DERIVED, so they cannot drift
    /// from the head counts they come from.**
    ///
    /// Checked at Qwen3.8's real geometry, where the key and value sides
    /// genuinely differ: `2·16·128 + 48·128 = 10240`, the observed
    /// `in_proj_qkv` row count. A build that folded the two sides into
    /// one head count would have to pick one, and either choice misses.
    #[test]
    fn the_linear_attention_widths_are_derived_from_both_sides() {
        let qwen38 = LinearAttentionTopology {
            key_heads: 16,
            key_head_dim: 128,
            value_heads: 48,
            value_head_dim: 128,
            conv_kernel: 4,
            state_dtype: Some(RecurrentStateDtype::Float32),
        };
        assert_eq!(
            qwen38.qkv_channels(),
            10240,
            "q and k at the KEY geometry, v at the value's"
        );
        assert_eq!(qwen38.value_width(), 6144);
        // Folding the sides would give a different number either way,
        // which is why they stay separate.
        assert_ne!(
            qwen38.qkv_channels(),
            3 * qwen38.key_heads * qwen38.key_head_dim
        );
        assert_ne!(qwen38.qkv_channels(), 3 * qwen38.value_width());
    }

    #[test]
    fn key_status_serialises_lowercase() {
        assert_eq!(
            serde_json::to_string(&KeyStatus::Unconsumed).unwrap(),
            "\"unconsumed\""
        );
        assert_eq!(
            serde_json::to_string(&KeyStatus::Consumed).unwrap(),
            "\"consumed\""
        );
        assert_eq!(
            serde_json::to_string(&KeyStatus::Metadata).unwrap(),
            "\"metadata\""
        );
    }

    #[test]
    fn key_status_round_trips() {
        for status in [
            KeyStatus::Consumed,
            KeyStatus::Metadata,
            KeyStatus::Unconsumed,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let back: KeyStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(back, status);
        }
    }
}
