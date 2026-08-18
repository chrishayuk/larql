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
pub const INVENTORY_SCHEMA: u32 = 4;

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
    /// Text-decoder features the graph represents only as absent, read
    /// verbatim so a checkpoint turning them on blocks on the value.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text_features: Option<crate::inventory::text_features::TextFeatures>,
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
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: Option<usize>,
    pub sliding_window: Option<usize>,
    pub attention: AttentionSummary,
    /// One entry per layer, in layer order.
    pub layers: Vec<LayerPolicy>,
    /// Execution scalars, fully resolved. `None` only when read from a
    /// pre-v3 inventory JSON — a fact for completeness gates, not a
    /// licence to default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<ResolvedExecution>,
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
    /// `"sliding"` or `"full"`.
    pub attention: String,
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
    /// A dense MLP summed with the expert block every layer (Gemma 4 A4B).
    pub hybrid: bool,
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

#[cfg(test)]
mod tests {
    use super::*;

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
