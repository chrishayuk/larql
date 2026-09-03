//! Qwen architecture (Qwen 2, 2.5, 3, MoE variants).
//!
//! Mostly Llama-compatible but with these differences:
//! - Qwen2/2.5: attention Q/K/V bias terms
//! - Qwen3: QK norms (no bias), optional MoE FFN
//! - Qwen3 MoE: router at `mlp.gate.weight`, per-expert `mlp.experts.{E}.{gate,up,down}_proj.weight`

use crate::config::{
    AttentionGateSpec, ExpertFormat, GateActivation, GateCombine, GatePlacement, GateSource,
    GateUpLayout, ModelArchitecture, ModelConfig, SharedExpertGateSource, SharedExpertGateSpec,
};
use crate::tensor_keys::{attn_bias, moe_experts, qk_norm};

/// Model types whose RMSNorm stores the weight as an OFFSET FROM ONE.
///
/// `Qwen3_5RMSNorm` initialises its weight to **zeros** and applies
/// `x_normed * (1.0 + weight)`. `Qwen3RMSNorm` initialises to **ones** and
/// applies `weight * x_normed`. Same family name, opposite conventions,
/// and the saved tensors are not interchangeable.
///
/// Qwen3.8 is the evidence: its `input_layernorm` weight has norm 3.83,
/// and HF's normed output has norm 75.5 — only reachable through
/// `(1 + w)`, since `w * x_normed` would land near 3.83. Applying the
/// weight directly made every decoder norm wrong and put the model's
/// first diverging plane at layer 0.
///
/// Read from the upstream classes rather than inferred: `qwen3`,
/// `qwen3_moe` and `qwen3_vl` are ones-initialised and stay at offset 0.
/// `qwen3_next` shares Qwen3.5's convention.
const PLUS_ONE_NORM_FAMILIES: &[&str] = &["qwen3_5", "qwen3_next"];

/// Whether this model type's saved norm weights are offsets from one.
fn stores_norm_weight_as_offset(model_type: &str) -> bool {
    PLUS_ONE_NORM_FAMILIES
        .iter()
        .any(|family| model_type.starts_with(family))
}

/// Model types whose expert bank is ONE stacked tensor per projection
/// rather than a tensor per expert.
///
/// `Qwen3_5MoeExperts` holds `gate_up_proj[num_experts, 2 ·
/// moe_intermediate_size, hidden_size]` and `down_proj[num_experts,
/// hidden_size, moe_intermediate_size]` as plain parameters, and the
/// checkpoints ship exactly that: `mlp.experts.gate_up_proj` with no
/// expert index, where `qwen2_moe` and `qwen3_moe` ship
/// `mlp.experts.{id}.{gate,up,down}_proj.weight`. Same family name,
/// different storage, and reading one as the other finds no operands at
/// all.
///
/// Prefix-matched like [`PLUS_ONE_NORM_FAMILIES`], so the nested
/// `qwen3_5_moe_text` spelling resolves too. `qwen3_5` (dense) is not a
/// prefix of this and keeps its own answer.
const PACKED_EXPERT_FAMILIES: &[&str] = &["qwen3_5_moe"];

/// Whether this model type stacks its experts into one tensor per
/// projection.
fn stacks_experts(model_type: &str) -> bool {
    PACKED_EXPERT_FAMILIES
        .iter()
        .any(|family| model_type.starts_with(family))
}

pub struct QwenArch {
    config: ModelConfig,
}

impl QwenArch {
    pub fn from_config(config: ModelConfig) -> Self {
        Self { config }
    }
}

impl ModelArchitecture for QwenArch {
    fn family(&self) -> &str {
        &self.config.model_type
    }

    fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// Qwen3.5/3.8's fused attention output gate.
    ///
    /// Judged from HF `Qwen3_5Attention.forward`, not from the config's
    /// own description of itself: HF reads neither `attn_output_gate` nor
    /// `output_gate_type` (zero references across every `qwen3_5` source
    /// file), so the gate is unconditional in the reference
    /// implementation and its real witness is the tensor geometry —
    /// `q_proj` carries `2 · num_heads · head_dim` rows.
    ///
    /// `attn_output_gate` is still what this reads, because a container
    /// must be able to state the fact without shipping weights, and the
    /// operand closure check cross-examines it against the actual rows.
    ///
    /// The activation is `sigmoid`. The config says `output_gate_type:
    /// "swish"`, which would be `x · silu(g)` and is NOT what HF computes
    /// (`x · sigmoid(g)`). That key is deliberately NOT consulted here:
    /// its semantic owner is unresolved — there is a genuine silu gate
    /// elsewhere in this model, in DeltaNet's gated RMSNorm — and
    /// resolving it on resemblance is the ownership error the plan's
    /// `Unrepresented` verdict exists to keep visible.
    fn attention_output_gate(&self) -> Option<AttentionGateSpec> {
        self.config
            .attn_output_gate
            .filter(|on| *on)
            .map(|_| AttentionGateSpec {
                source: GateSource::FusedQueryProjection,
                activation: GateActivation::Sigmoid,
                combine: GateCombine::ElementwiseMultiply,
                placement: GatePlacement::AfterAggregationBeforeOutputProjection,
            })
    }

    /// See [`PLUS_ONE_NORM_FAMILIES`]. Covers the decoder norms and the
    /// final norm, which are the same class upstream.
    fn norm_weight_offset(&self) -> f32 {
        if stores_norm_weight_as_offset(&self.config.model_type) {
            1.0
        } else {
            0.0
        }
    }

    /// The per-head Q/K norms are that same class too — `Qwen3_5Attention`
    /// builds them as `Qwen3_5RMSNorm(head_dim)` — so they share the
    /// convention. Declared separately because the two offsets are
    /// independent facts and a family could differ.
    fn qk_norm_weight_offset(&self) -> f32 {
        self.norm_weight_offset()
    }

    // ── MoE (Qwen3-MoE, Qwen2-MoE) ──

    fn is_moe(&self) -> bool {
        self.config.num_experts.unwrap_or(0) > 0
    }

    fn num_experts(&self) -> usize {
        self.config.num_experts.unwrap_or(0)
    }

    fn num_experts_per_token(&self) -> usize {
        self.config
            .num_experts_per_token
            .or(self.config.top_k_experts)
            .unwrap_or(0)
    }

    fn moe_intermediate_size(&self) -> usize {
        self.config.moe_intermediate_size.unwrap_or(0)
    }

    fn moe_router_key(&self, layer: usize) -> Option<String> {
        if !self.is_moe() {
            return None;
        }
        moe_experts::router(&self.layer_prefix(layer))
    }

    fn expert_ffn_gate_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        if !self.is_moe() {
            return None;
        }
        moe_experts::gate_proj(&self.layer_prefix(layer), expert_id)
    }

    fn expert_ffn_up_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        if !self.is_moe() {
            return None;
        }
        moe_experts::up_proj(&self.layer_prefix(layer), expert_id)
    }

    fn expert_ffn_down_key(&self, layer: usize, expert_id: usize) -> Option<String> {
        if !self.is_moe() {
            return None;
        }
        moe_experts::down_proj(&self.layer_prefix(layer), expert_id)
    }

    /// Qwen3.5-MoE stores one stacked tensor per projection; every other
    /// Qwen MoE stores a tensor per expert. See [`PACKED_EXPERT_FAMILIES`].
    fn expert_format(&self) -> ExpertFormat {
        if stacks_experts(&self.config.model_type) {
            ExpertFormat::PackedBF16
        } else {
            ExpertFormat::PerExpert
        }
    }

    /// `Qwen3_5MoeExperts.forward` splits the fused projection with
    /// `.chunk(2, dim=-1)` — the first half is the gate. Declared only
    /// where a fused operand exists to split; a per-expert bank has no
    /// fused layout to describe.
    fn gate_up_layout(&self) -> Option<GateUpLayout> {
        stacks_experts(&self.config.model_type).then_some(GateUpLayout::ContiguousHalves)
    }

    // ── The gated shared expert (Qwen2-MoE, Qwen3.5-MoE) ──
    //
    // `Qwen2MoeSparseMoeBlock` and `Qwen3_5MoeSparseMoeBlock` build
    // exactly ONE always-on branch, sized by its own key rather than by
    // the routed width, and scale it by a sigmoid of a one-logit
    // projection before summing it with the routed output. Qwen3-MoE
    // declares no such width and builds no such branch, so the declared
    // width is what tells the two apart — not a list of model names.

    /// One, when the checkpoint declares a width for it. The count is not
    /// declared by any Qwen config: it is read from the reference block,
    /// which constructs a single `shared_expert`.
    fn num_shared_experts(&self) -> usize {
        usize::from(self.config.shared_expert_intermediate_size.is_some())
    }

    fn shared_expert_branch_gate(&self) -> Option<SharedExpertGateSpec> {
        self.config
            .shared_expert_intermediate_size
            .map(|_| SharedExpertGateSpec {
                source: SharedExpertGateSource::HiddenStateToScalar,
                activation: GateActivation::Sigmoid,
                combine: GateCombine::ElementwiseMultiply,
            })
    }

    fn shared_expert_branch_gate_key(&self, layer: usize) -> Option<String> {
        self.config
            .shared_expert_intermediate_size
            .map(|_| format!("{}mlp.shared_expert_gate.weight", self.layer_prefix(layer)))
    }

    // The branch's own SwiGLU projections. Singular `shared_expert` — the
    // DeepSeek/Kimi lineage spells the same role `shared_experts`, and a
    // shared spelling would bind neither.

    fn shared_expert_gate_key(&self, layer: usize) -> Option<String> {
        self.config.shared_expert_intermediate_size.map(|_| {
            format!(
                "{}mlp.shared_expert.gate_proj.weight",
                self.layer_prefix(layer)
            )
        })
    }

    fn shared_expert_up_key(&self, layer: usize) -> Option<String> {
        self.config.shared_expert_intermediate_size.map(|_| {
            format!(
                "{}mlp.shared_expert.up_proj.weight",
                self.layer_prefix(layer)
            )
        })
    }

    fn shared_expert_down_key(&self, layer: usize) -> Option<String> {
        self.config.shared_expert_intermediate_size.map(|_| {
            format!(
                "{}mlp.shared_expert.down_proj.weight",
                self.layer_prefix(layer)
            )
        })
    }

    // ── QK norms (Qwen3) ──
    // Returning keys for models that don't have them is harmless —
    // the forward pass checks if the vector exists before using it.

    fn attn_q_norm_key(&self, layer: usize) -> Option<String> {
        qk_norm::q(&self.layer_prefix(layer))
    }

    fn attn_k_norm_key(&self, layer: usize) -> Option<String> {
        qk_norm::k(&self.layer_prefix(layer))
    }

    // ── Attention bias (Qwen2/2.5 only; absent in Qwen3) ──
    // Returning keys for absent tensors is harmless.

    fn attn_q_bias_key(&self, layer: usize) -> Option<String> {
        attn_bias::q(&self.layer_prefix(layer))
    }

    fn attn_k_bias_key(&self, layer: usize) -> Option<String> {
        attn_bias::k(&self.layer_prefix(layer))
    }

    fn attn_v_bias_key(&self, layer: usize) -> Option<String> {
        attn_bias::v(&self.layer_prefix(layer))
    }
}
