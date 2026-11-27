//! Which op a missing role belongs to, and stack geometry.

use super::super::super::graph::surface::LinearAttentionSurface;
use super::super::super::graph::surface::Mamba2Surface;
use super::super::super::graph::surface::MlaSurface;
use super::super::super::graph::{NormPlacement, OperandRole};
use larql_models::config::{HyperConnection, MoeRouterKind};

#[allow(unused_imports)]
use super::*;

/// The primitive a found operand requires when the surface does not carry
/// its op. `None` when the operand is consumed by a declared op.
pub(super) fn absent_op(role: OperandRole, ops: &LayerOps) -> Option<&'static str> {
    match role {
        // A site operand on a component whose residual is ONE vector: the
        // declaration and the estate disagree, and the operand implies an
        // operation the surface never declared. Paired with
        // `required_roles`, which demands all six iff the topology is
        // declared, so presence and requirement cannot desync.
        OperandRole::HcAttnMixFn
        | OperandRole::HcAttnBase
        | OperandRole::HcAttnScale
        | OperandRole::HcFfnMixFn
        | OperandRole::HcFfnBase
        | OperandRole::HcFfnScale
            if !ops.hyper_connection =>
        {
            Some(HC_SITE_ON_SINGLE_STREAM)
        }
        // The same operand on a one-sublayer block, under the topology:
        // no judged form exists, so it is a stray rather than a site.
        OperandRole::HcAttnMixFn
        | OperandRole::HcAttnBase
        | OperandRole::HcAttnScale
        | OperandRole::HcFfnMixFn
        | OperandRole::HcFfnBase
        | OperandRole::HcFfnScale
            if ops.operator.is_mamba2() || ops.operator.is_conv_qkv() =>
        {
            Some(HC_SITE_ON_MIXER_LAYER)
        }
        // The same reasoning for the attention-residual pairs: the
        // topology names its two sites after the two sublayers a
        // transformer block has, and a one-sublayer block has neither.
        // Nothing observed declares this combination; the arm exists so
        // that if something does, it blocks by name instead of planning a
        // layer with no sites under a topology that says every layer has
        // two. There is no single-stream arm beside it — those operands
        // never become roles without the declaration.
        OperandRole::AttnResAttentionNorm
        | OperandRole::AttnResAttentionProj
        | OperandRole::AttnResMlpNorm
        | OperandRole::AttnResMlpProj
            if ops.operator.is_mamba2() || ops.operator.is_conv_qkv() =>
        {
            Some(ATTN_RES_SITE_ON_MIXER_LAYER)
        }
        // A mixer-only layer runs neither attention nor an FFN; any
        // transformer-shaped operand on it is a stray, whatever its name.
        OperandRole::AttnQ
        | OperandRole::AttnK
        | OperandRole::AttnV
        | OperandRole::AttnO
        | OperandRole::FfnGate
        | OperandRole::FfnUp
        | OperandRole::FfnDown
        | OperandRole::PreAttentionNorm
        | OperandRole::PostAttentionNorm
        | OperandRole::PreFfnNorm
        | OperandRole::PostFfnNorm
            if ops.operator.is_mamba2() =>
        {
            Some("a mixer-only Mamba2 layer (no attention, no FFN)")
        }
        OperandRole::Mamba2Conv1dBias if !ops.mamba2.is_some_and(|m| m.geometry.use_conv_bias) => {
            Some("a conv bias (`use_conv_bias` declares none)")
        }
        OperandRole::Mamba2GatedNorm if !ops.mamba2.is_some_and(|m| m.geometry.rms_norm) => {
            Some("the mixer's gated RMSNorm (`rms_norm` declares none)")
        }
        OperandRole::AttnOutputGate if !ops.output_gate => {
            Some("attention output gate (judged semantics)")
        }
        // The two K3 gates, each held to its declaration from both sides:
        // the undeclared form is an operand implying an op the component
        // never chose, and the declared form's absence is reported by
        // `required_roles` as a missing operand.
        OperandRole::KdaGProj if !ops.kda_full_rank_gate => Some(KDA_FULL_RANK_GATE_UNDECLARED),
        OperandRole::KdaGAProj | OperandRole::KdaGBProj if ops.kda_full_rank_gate => {
            Some(KDA_LOW_RANK_GATE_UNDER_FULL_RANK)
        }
        OperandRole::MlaOutputGate if !ops.mla_output_gate => Some(MLA_OUTPUT_GATE_UNDECLARED),
        // The query form, held to its declaration from both sides.
        OperandRole::MlaQAProj | OperandRole::MlaQANorm | OperandRole::MlaQBProj
            if !ops.mla_q_lora =>
        {
            Some(MLA_Q_LORA_UNDECLARED)
        }
        OperandRole::MlaQProj if ops.mla_q_lora => Some(MLA_Q_PROJ_UNDER_Q_LORA),
        OperandRole::AttnQBias | OperandRole::AttnKBias | OperandRole::AttnVBias
            if !ops.attention_bias && !ops.qkv_bias =>
        {
            Some("attention projection bias (declared `attention_bias` or `qkv_bias`)")
        }
        OperandRole::AttnOBias if !ops.attention_bias => {
            Some("attention output-projection bias (declared `attention_bias`; `qkv_bias` biases Q/K/V only)")
        }
        OperandRole::AttnSinks if !ops.sinks => Some("attention sinks (judged semantics)"),
        OperandRole::AttnV if ops.v_from_k => {
            Some("value projection (this layer's V is its K projection — `attention_k_eq_v`)")
        }
        // MLA has no plain query/key/value/output projection to bind: its
        // `q_proj`/`o_proj` SUFFIXES are intercepted by `MLA_ROLE_TABLE`
        // before they ever reach these roles, and it never ships
        // `k_proj`/`v_proj` at all (K/V arrive only through the
        // compressed path). Without this guard a stray `k_proj` on an
        // MLA layer classified as plain `AttnK` and was checked against
        // the SOFTMAX contract (`num_kv_heads · head_dim`) — the wrong
        // question, not the right refusal.
        OperandRole::AttnQ | OperandRole::AttnK | OperandRole::AttnV | OperandRole::AttnO
            if ops.operator.is_mla() =>
        {
            Some("MLA (Multi-Latent Attention) — no plain query/key/value/output projection")
        }
        OperandRole::FfnGate if !ops.routed && !ops.gated_ffn => Some("gated FFN"),
        OperandRole::PreAttentionNormBias
        | OperandRole::PostAttentionNormBias
        | OperandRole::PreFfnNormBias
        | OperandRole::PostFfnNormBias
            if !ops.norm_bias =>
        {
            Some("a norm bias (declared `norm.bias`)")
        }
        OperandRole::FfnUpBias | OperandRole::FfnDownBias if !ops.ffn_bias => {
            Some("an FFN projection bias (declared `ffn.bias`)")
        }
        OperandRole::FfnUpBias | OperandRole::FfnDownBias if ops.routed && !ops.hybrid => {
            Some("dense FFN (this layer is routed)")
        }
        OperandRole::FfnGate | OperandRole::FfnUp | OperandRole::FfnDown
            if ops.routed && !ops.hybrid =>
        {
            Some("dense FFN (this layer is routed)")
        }
        OperandRole::MoeRouterScale | OperandRole::MoeRouterPerExpertScale
            if !ops
                .moe
                .is_some_and(|m| m.router_kind == MoeRouterKind::TopKRenormScaled) =>
        {
            Some("Gemma 4 router conditioning (router kind gemma4_top_k_softmax)")
        }
        OperandRole::PreExpertsNorm
        | OperandRole::PostDenseFfnNorm
        | OperandRole::PostExpertsNorm
            if !ops.hybrid =>
        {
            Some("hybrid dense+routed FFN (judged semantics)")
        }
        OperandRole::MoeRouterWeight
        | OperandRole::MoeRouterBias
        | OperandRole::MoeRouterScale
        | OperandRole::MoeRouterPerExpertScale
        | OperandRole::ExpertGateUp
        | OperandRole::ExpertGateUpScales
        | OperandRole::ExpertGateUpBias
        | OperandRole::ExpertDown
        | OperandRole::ExpertDownScales
        | OperandRole::ExpertDownBias
        | OperandRole::PerExpertGate(_)
        | OperandRole::PerExpertUp(_)
        | OperandRole::PerExpertDown(_)
        | OperandRole::SharedExpertGate
        | OperandRole::SharedExpertUp
        | OperandRole::SharedExpertDown
            if !ops.routed =>
        {
            Some("routed FFN (judged semantics)")
        }
        OperandRole::MoeRouterBias if ops.moe.is_some_and(|m| !m.router_bias) => {
            Some("router bias (declared by the routed-FFN judgment)")
        }
        // A `PerExpert` operand whose index the routed-FFN judgment does
        // not declare — the set-closure half of expert-bank carving: an
        // index beyond `moe.experts` is as much a defect as one missing
        // from `0..experts` ([`required_roles`] states the other half).
        OperandRole::PerExpertGate(expert)
        | OperandRole::PerExpertUp(expert)
        | OperandRole::PerExpertDown(expert)
            if ops.moe.is_some_and(|m| expert as usize >= m.experts) =>
        {
            Some("an expert index the routed-FFN judgment does not declare")
        }
        OperandRole::SharedExpertGate
        | OperandRole::SharedExpertUp
        | OperandRole::SharedExpertDown
            if ops.moe.is_some_and(|m| m.shared_experts == 0) =>
        {
            Some("a shared expert (the routed-FFN judgment declares none)")
        }
        OperandRole::SharedExpertBranchGate
            if ops
                .moe
                .is_some_and(|m| m.shared_experts == 0 || m.shared_expert_gate.is_none()) =>
        {
            Some("a gate on the shared-expert branch (the judgment declares none, so the branch is summed unscaled)")
        }
        // The latent wrapper, refused from the other side. The
        // declaration chooses the form; a shipped
        // `routed_expert_down_proj` may CONFIRM it and must never select
        // it, or a checkpoint carrying a stray wrapper tensor would be
        // executed as a different model — the experts behind a bottleneck
        // that its config never declared.
        OperandRole::MoeLatentDownProj | OperandRole::MoeLatentUpProj
            if ops.moe.is_some_and(|m| m.latent.is_none()) =>
        {
            Some(MOE_LATENT_BRANCH_UNDECLARED)
        }
        // Nested exactly as the reference nests it: no wrapper means no
        // norm to speak of, and a wrapper without `latent_moe_use_norm`
        // means the aggregate goes to the up-projection unnormalised.
        // Two different reasons, one refusal, and the message says which.
        OperandRole::MoeLatentNorm if ops.moe.is_some_and(|m| m.latent.is_none()) => {
            Some(MOE_LATENT_NORM_WITHOUT_BRANCH)
        }
        OperandRole::MoeLatentNorm
            if ops
                .moe
                .is_some_and(|m| m.latent.is_some_and(|l| l.norm.is_none())) =>
        {
            Some(MOE_LATENT_NORM_UNDER_FALSE_FLAG)
        }
        OperandRole::ExpertGateUpScales | OperandRole::ExpertDownScales
            if ops
                .moe
                .is_some_and(|m| !m.expert_format.has_split_scale_streams()) =>
        {
            Some("a scaled expert format (this format carries no separate scales)")
        }
        OperandRole::PreFfnNorm
        | OperandRole::PostFfnNorm
        | OperandRole::PreFfnNormBias
        | OperandRole::PostFfnNormBias
            if ops.placement == NormPlacement::PreOnly =>
        {
            Some("four-norm placement")
        }
        // Paired the other way: a post-norm stack refuses the two
        // pre-sublayer norms, so a checkpoint shipping one under this
        // placement is a disagreement rather than a spare tensor.
        OperandRole::PreAttentionNorm | OperandRole::PreFfnNorm
            if ops.placement == NormPlacement::PostOnly =>
        {
            Some("a pre-sublayer norm (post-norm placement normalises each sublayer's output)")
        }
        _ => None,
    }
}

/// The geometry one layer's stack operands are checked against — the
/// layer's own head geometry under the component's query-head count.
pub(super) struct StackGeometry {
    pub(super) hidden: usize,
    /// `num_q_heads · head_dim` — the ATTENTION width. What `o_proj`
    /// consumes, and what the query half occupies.
    pub(super) q_rows: usize,
    /// Rows the stored query projection actually carries.
    ///
    /// Equal to [`Self::q_rows`] on an ordinary stack, and **twice** it
    /// when the component's output gate is sourced from the query
    /// projection: that projection emits `2 · head_dim` per head, query
    /// and gate interleaved. Kept as its own field rather than doubling
    /// `q_rows`, because `o_proj` and the query-bias contract are still
    /// sized by the attention width — conflating the two would silently
    /// demand a 12288-wide `o_proj` on Qwen3.8, which carries 6144.
    pub(super) q_proj_rows: usize,
    pub(super) kv_rows: usize,
    pub(super) intermediate: usize,
    pub(super) head_dim: usize,
    pub(super) num_q_heads: usize,
    pub(super) num_kv_heads: usize,
    pub(super) qk_scope: larql_models::config::QkNormScope,
    /// The recurrence's geometry, on a component that declares one. Kept
    /// beside the softmax fields rather than folded into them: the key and
    /// value sides carry different head counts, so `num_q_heads`/`head_dim`
    /// cannot describe this operator.
    pub(super) linear: Option<LinearAttentionSurface>,
    /// The KDA block's geometry, on a component that declares one.
    /// Disjoint from [`Self::linear`]: the two describe different
    /// operators, and a stack carrying KDA operands against a Gated
    /// DeltaNet geometry would validate the wrong contracts.
    pub(super) kda: Option<larql_models::config::KdaGeometry>,
    /// The MLA operator's geometry, on a component whose full-attention
    /// layers run it. Disjoint from every field above it — MLA is neither
    /// the softmax fields' uniform per-head width nor a recurrence.
    pub(super) mla: Option<MlaSurface>,
    /// The Mamba2 mixer's surface, on a component whose layers run it.
    /// Disjoint from every field above for the same reason each of them
    /// is from the others.
    pub(super) mamba2: Option<Mamba2Surface>,
    /// The hybrid's conv-QKV attention geometry, on a component whose
    /// full layers run it.
    pub(super) conv_qkv: Option<larql_models::config::ConvQkvAttnGeometry>,
    /// The declared hyper-connection topology, whose stream count sizes
    /// the site operands: `[(2 + hc)·hc, hc·hidden]`, `[(2 + hc)·hc]`,
    /// `[3]`. `None` on a single-stream component, where a site operand
    /// is refused by `absent_op` before its shape is ever asked.
    pub(super) hyper_connection: Option<HyperConnection>,
}

/// Whether a stored shape satisfies a contract.
///
/// Exact equality, **plus one narrow equivalence**: a contract for a
/// *vector* is satisfied by the same values carrying broadcast singleton
/// dimensions. Kimi Linear stores its per-head decay as
/// `A_log: [1, 1, 32, 1]` — the shape its reference broadcasts against
/// `[B, T, H, D]` — where the contract says `[32]`. Those are the same 32
/// numbers in the same order.
///
/// Deliberately **not** a general squeeze. The equivalence applies only
/// when the contract is one-dimensional, so it can never quietly accept a
/// re-laid-out matrix: `[2, 16]` still fails `[32]`, and a `[4096, 128]`
/// contract is unaffected by anything here. A blanket "drop all ones"
/// would accept a genuine relayout as readily as a broadcast form, and the
/// point of a shape contract is to refuse exactly that.
pub(in super::super) fn shape_satisfies(actual: &[usize], expected: &[usize]) -> bool {
    if actual == expected {
        return true;
    }
    expected.len() == 1 && actual.iter().filter(|d| **d != 1).eq(expected.iter())
}
