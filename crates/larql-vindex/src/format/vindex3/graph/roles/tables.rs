//! Norm placement and the operand role tables.

use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use super::*;

impl NormPlacement {
    /// Why this build cannot lower this placement, when it cannot.
    ///
    /// **One authority.** The op plan refuses on it. The plan report read
    /// it too, naming a refused placement as an unsupported component so
    /// the two could never disagree — a report that calls a component
    /// admissible while the op plan refuses to build it is the
    /// looks-supported failure in its purest form — and stopped in wave
    /// 19, when no judged placement (nor topology) refused any more and
    /// the reader had nothing left to read. A variant that returns `Some`
    /// again must bring that reader back beside it.
    ///
    /// `None` is a claim, not an absence: this build lowers the
    /// placement and an executor reads its operands.
    pub fn unimplemented_reason(self) -> Option<&'static str> {
        match self {
            // Every judged placement lowers. `PostOnly` joined them in
            // wave 12: the generic executor already applied the wrap
            // norms to each sublayer's OUTPUT before the residual add,
            // and what it lacked was the ability to run with NO
            // pre-sublayer norm. It has that now, on both the batch and
            // the decode path, and the epsilon its QK norm runs at moved
            // off the pre-norm site onto the layer's own declared field.
            Self::PreOnly | Self::PrePost | Self::PreMixer | Self::PostOnly => None,
        }
    }
}

/// Suffix → role. Exact matches on the layer-relative suffix (after
/// `{layer}.`), so a new upstream spelling classifies as *nothing* and
/// blocks, rather than fuzzy-matching into the wrong op.
pub(super) const ROLE_TABLE: &[(&str, OperandRole)] = &[
    ("self_attn.q_proj.weight", OperandRole::AttnQ),
    ("self_attn.k_proj.weight", OperandRole::AttnK),
    ("self_attn.v_proj.weight", OperandRole::AttnV),
    ("self_attn.o_proj.weight", OperandRole::AttnO),
    ("self_attn.gate_proj.weight", OperandRole::AttnOutputGate),
    ("self_attn.q_proj.bias", OperandRole::AttnQBias),
    ("self_attn.k_proj.bias", OperandRole::AttnKBias),
    ("self_attn.v_proj.bias", OperandRole::AttnVBias),
    ("self_attn.o_proj.bias", OperandRole::AttnOBias),
    ("self_attn.sinks", OperandRole::AttnSinks),
    ("self_attn.q_norm.weight", OperandRole::AttnQNorm),
    ("self_attn.k_norm.weight", OperandRole::AttnKNorm),
    // Gated DeltaNet (Qwen3.8 `linear_attention` layers). Nine operands,
    // sharing nothing with the softmax set above: the recurrence has no
    // per-position key or value to retain, so none of the Attn* roles
    // apply. Exact suffixes, like every entry here — a DeltaNet layer's
    // `linear_attn.norm.weight` must never be mistaken for a decoder norm.
    (
        "linear_attn.in_proj_qkv.weight",
        OperandRole::LinearAttnInProjQkv,
    ),
    (
        "linear_attn.in_proj_a.weight",
        OperandRole::LinearAttnInProjA,
    ),
    (
        "linear_attn.in_proj_b.weight",
        OperandRole::LinearAttnInProjB,
    ),
    (
        "linear_attn.in_proj_z.weight",
        OperandRole::LinearAttnInProjZ,
    ),
    ("linear_attn.conv1d.weight", OperandRole::LinearAttnConv1d),
    ("linear_attn.A_log", OperandRole::LinearAttnALog),
    ("linear_attn.dt_bias", OperandRole::LinearAttnDtBias),
    ("linear_attn.norm.weight", OperandRole::LinearAttnNorm),
    (
        "linear_attn.out_proj.weight",
        OperandRole::LinearAttnOutProj,
    ),
    ("input_layernorm.weight", OperandRole::PreAttentionNorm),
    // LFM2's spelling of the same two-norm estate. `Lfm2DecoderLayer.
    // forward` is `residual = h; h = mixer(operator_norm(h)); h = h +
    // residual; h = h + feed_forward(ffn_norm(h))` — structurally the
    // two-norm PRE-only stack, with the mixer being attention on the
    // layers named in `full_attn_idxs` and a short convolution
    // elsewhere.
    //
    // `ffn_norm` binds to `PostAttentionNorm` and that reads oddly until
    // you know the rule this module already states: in a TWO-norm layer
    // `post_attention_layernorm` IS the pre-FFN norm, and the role keeps
    // the historical name. Binding LFM2's honestly-named `ffn_norm` to
    // the honestly-named `PreFfnNorm` would instead resolve the estate
    // as a partial FOUR-norm stack and refuse it.
    ("operator_norm.weight", OperandRole::PreAttentionNorm),
    ("ffn_norm.weight", OperandRole::PostAttentionNorm),
    (
        "post_attention_layernorm.weight",
        OperandRole::PostAttentionNorm,
    ),
    ("pre_feedforward_layernorm.weight", OperandRole::PreFfnNorm),
    (
        "post_feedforward_layernorm.weight",
        OperandRole::PostFfnNorm,
    ),
    ("mlp.gate_proj.weight", OperandRole::FfnGate),
    ("mlp.up_proj.weight", OperandRole::FfnUp),
    ("mlp.down_proj.weight", OperandRole::FfnDown),
    // Norm and FFN biases in the canonical spellings. Classifying them is
    // not admitting them: closure requires the surface's declaration.
    ("input_layernorm.bias", OperandRole::PreAttentionNormBias),
    (
        "post_attention_layernorm.bias",
        OperandRole::PostAttentionNormBias,
    ),
    (
        "pre_feedforward_layernorm.bias",
        OperandRole::PreFfnNormBias,
    ),
    (
        "post_feedforward_layernorm.bias",
        OperandRole::PostFfnNormBias,
    ),
    ("mlp.up_proj.bias", OperandRole::FfnUpBias),
    ("mlp.down_proj.bias", OperandRole::FfnDownBias),
    // GPT-2's lineage (`GPT2Block`, and GPT-Neo's `GPTNeoBlock`, which
    // spells its norms and MLP identically): `ln_1` normalises before
    // attention, `ln_2` before the MLP — the two-norm pre-only estate, so
    // `ln_2` binds the historical `PostAttentionNorm` role exactly as
    // LFM2's `ffn_norm` does. The MLP is non-gated: `c_fc` is the up
    // projection, `c_proj` the down. GPT-2's fused `c_attn` never reaches
    // this table — the inventory splits it into `self_attn.{q,k,v}_proj`
    // (see `larql_models::inventory::derive`) — while its output
    // projection keeps its own name.
    ("ln_1.weight", OperandRole::PreAttentionNorm),
    ("ln_1.bias", OperandRole::PreAttentionNormBias),
    ("ln_2.weight", OperandRole::PostAttentionNorm),
    ("ln_2.bias", OperandRole::PostAttentionNormBias),
    ("mlp.c_fc.weight", OperandRole::FfnUp),
    ("mlp.c_fc.bias", OperandRole::FfnUpBias),
    ("mlp.c_proj.weight", OperandRole::FfnDown),
    ("mlp.c_proj.bias", OperandRole::FfnDownBias),
    ("attn.c_proj.weight", OperandRole::AttnO),
    ("attn.c_proj.bias", OperandRole::AttnOBias),
    // GPT-Neo's attention (`GPTNeoSelfAttention`, under `attn.attention`):
    // separate unbiased Q/K/V and a biased output projection. The Q/K/V
    // bias spellings are classified so a checkpoint shipping one is
    // refused for the op it implies (GPT-Neo declares `output_bias`),
    // not as an unknown tensor.
    ("attn.attention.q_proj.bias", OperandRole::AttnQBias),
    ("attn.attention.k_proj.bias", OperandRole::AttnKBias),
    ("attn.attention.v_proj.bias", OperandRole::AttnVBias),
    ("attn.attention.q_proj.weight", OperandRole::AttnQ),
    ("attn.attention.k_proj.weight", OperandRole::AttnK),
    ("attn.attention.v_proj.weight", OperandRole::AttnV),
    ("attn.attention.out_proj.weight", OperandRole::AttnO),
    ("attn.attention.out_proj.bias", OperandRole::AttnOBias),
    ("mlp.router.weight", OperandRole::MoeRouterWeight),
    // The Qwen MoE lineage's router (`Qwen2MoeSparseMoeBlock.gate`,
    // `[num_experts, hidden]`). Distinct from the dense `mlp.gate_proj`.
    ("mlp.gate.weight", OperandRole::MoeRouterWeight),
    ("mlp.router.bias", OperandRole::MoeRouterBias),
    // Packed MXFP4 (GPT-OSS): blocks + scales + bias per projection.
    ("mlp.experts.gate_up_proj_blocks", OperandRole::ExpertGateUp),
    (
        "mlp.experts.gate_up_proj_scales",
        OperandRole::ExpertGateUpScales,
    ),
    (
        "mlp.experts.gate_up_proj_bias",
        OperandRole::ExpertGateUpBias,
    ),
    ("mlp.experts.down_proj_blocks", OperandRole::ExpertDown),
    (
        "mlp.experts.down_proj_scales",
        OperandRole::ExpertDownScales,
    ),
    ("mlp.experts.down_proj_bias", OperandRole::ExpertDownBias),
    // Packed BF16 (Gemma 4 A4B): one unquantised operand per projection,
    // in both spellings seen — the checkpoint's own (`experts.…`, no
    // `mlp.` — the experts sit beside the dense `mlp`, not inside it) and
    // the `mlp.experts.…` form.
    ("mlp.experts.gate_up_proj", OperandRole::ExpertGateUp),
    ("mlp.experts.down_proj", OperandRole::ExpertDown),
    ("experts.gate_up_proj", OperandRole::ExpertGateUp),
    ("experts.down_proj", OperandRole::ExpertDown),
    // Gemma 4 hybrid block: router beside the dense mlp, its two scales,
    // the three extra branch norms, and the layer scalar.
    ("router.proj.weight", OperandRole::MoeRouterWeight),
    ("router.scale", OperandRole::MoeRouterScale),
    (
        "router.per_expert_scale",
        OperandRole::MoeRouterPerExpertScale,
    ),
    (
        "pre_feedforward_layernorm_2.weight",
        OperandRole::PreExpertsNorm,
    ),
    (
        "post_feedforward_layernorm_1.weight",
        OperandRole::PostDenseFfnNorm,
    ),
    (
        "post_feedforward_layernorm_2.weight",
        OperandRole::PostExpertsNorm,
    ),
    ("layer_scalar", OperandRole::LayerScalar),
    // Kimi Linear: router beside its own component name (not `mlp.`), its
    // bias-corrected-selection tensor kept apart from the router weight
    // (see module docs on `MoeRouterBias`), and the always-active shared
    // expert under the same component.
    ("block_sparse_moe.gate.weight", OperandRole::MoeRouterWeight),
    (
        "block_sparse_moe.gate.e_score_correction_bias",
        OperandRole::MoeRouterBias,
    ),
    (
        "block_sparse_moe.shared_experts.gate_proj.weight",
        OperandRole::SharedExpertGate,
    ),
    (
        "block_sparse_moe.shared_experts.up_proj.weight",
        OperandRole::SharedExpertUp,
    ),
    (
        "block_sparse_moe.shared_experts.down_proj.weight",
        OperandRole::SharedExpertDown,
    ),
    // Kimi-K3's latent routed branch, under the same component as the
    // router and the shared experts — and deliberately NOT under
    // `block_sparse_moe.experts.`, because these wrap the bank rather
    // than belonging to it: one of each per routed layer, whatever the
    // expert count.
    (
        "block_sparse_moe.routed_expert_down_proj.weight",
        OperandRole::MoeLatentDownProj,
    ),
    (
        "block_sparse_moe.routed_expert_norm.weight",
        OperandRole::MoeLatentNorm,
    ),
    (
        "block_sparse_moe.routed_expert_up_proj.weight",
        OperandRole::MoeLatentUpProj,
    ),
    // Qwen MoE: the branch is singular (`shared_expert`) where the
    // DeepSeek/Kimi lineage spells it `shared_experts`, and it carries a
    // scalar output gate the other lineage has no operand for.
    (
        "mlp.shared_expert.gate_proj.weight",
        OperandRole::SharedExpertGate,
    ),
    (
        "mlp.shared_expert.up_proj.weight",
        OperandRole::SharedExpertUp,
    ),
    (
        "mlp.shared_expert.down_proj.weight",
        OperandRole::SharedExpertDown,
    ),
    (
        "mlp.shared_expert_gate.weight",
        OperandRole::SharedExpertBranchGate,
    ),
    // Sinkhorn hyper-connection sites, as DeepSeek-V4 and GLM-5.3-Flash
    // both spell them — read from the checkpoints' own safetensors
    // headers, not from a reference implementation. Bare leaves with no
    // `.weight`: `layers.N.hc_attn_fn` on DeepSeek,
    // `model.language_model.layers.N.hc_attn_fn` on GLM, and the stack
    // prefix is what differs between the two, never the suffix. Layer-
    // blind, like the norms: a KDA layer and an MLA layer of the same
    // GLM stack each carry all six.
    //
    // Hy4-preview's `hc_attn_layer.hc_pre.hc_fn` is deliberately NOT
    // here. It spells a Sinkhorn-free topology this build has not
    // judged (HC-PREPOST), and binding it to a Sinkhorn role would be
    // matching on the substring `hc_` rather than on the operation.
    ("hc_attn_fn", OperandRole::HcAttnMixFn),
    ("hc_attn_base", OperandRole::HcAttnBase),
    ("hc_attn_scale", OperandRole::HcAttnScale),
    ("hc_ffn_fn", OperandRole::HcFfnMixFn),
    ("hc_ffn_base", OperandRole::HcFfnBase),
    ("hc_ffn_scale", OperandRole::HcFfnScale),
];

/// Suffix → role **under the attention-residual topology**, consulted
/// before every other table by [`classify_stack_tensor_under`] and by
/// NOTHING else.
///
/// K3 spells all four as `.weight` leaves directly under the layer
/// (`language_model.model.layers.{L}.self_attention_res_norm.weight`),
/// read from the checkpoint's own safetensors headers rather than from a
/// reference implementation. The stack prefix is what differs between
/// checkpoints of this dialect, never the suffix — the same argument the
/// hyper-connection site spellings carry.
///
/// The table is gated on the DECLARATION rather than on the operator,
/// which is the one structural difference from [`KDA_ROLE_TABLE`] and
/// friends. The operator cannot answer here: K3 carries these four on
/// its KDA layers and its MLA layers alike, and a softmax stack of the
/// same dialect would carry them too. What decides whether they are
/// site operands is whether the component declares
/// `attn_res_block_size` — and a build that read the topology off the
/// names instead would let any checkpoint acquire a residual programme
/// by spelling.
pub(super) const ATTENTION_RESIDUAL_ROLE_TABLE: &[(&str, OperandRole)] = &[
    (
        "self_attention_res_norm.weight",
        OperandRole::AttnResAttentionNorm,
    ),
    (
        "self_attention_res_proj.weight",
        OperandRole::AttnResAttentionProj,
    ),
    ("mlp_res_norm.weight", OperandRole::AttnResMlpNorm),
    ("mlp_res_proj.weight", OperandRole::AttnResMlpProj),
];

/// The attention-residual EXIT's two tensor groups, as K3 spells them
/// (`language_model.model.output_attn_res_{norm,proj}`).
///
/// Held here, beside the roles, because two independent consumers ask
/// about them and must never disagree: the graph builder's placement
/// vocabulary matches these as name fragments (ordered BEFORE its
/// generic `norm` fragment, which otherwise swallows the exit norm into
/// the component's final-norm object), and the op plan classifies the
/// object's tensors through [`ATTENTION_RESIDUAL_EXIT_TABLE`].
pub const ATTENTION_RESIDUAL_EXIT_LEAVES: &[&str] =
    &["output_attn_res_norm", "output_attn_res_proj"];

/// The attention-residual exit's own operands — a component-level
/// object, not a stack operand.
///
/// The stack's last layer leaves a prefix sum and a history of
/// snapshots; `_apply_output_attn_res` reduces them to the ONE vector
/// the final norm and the head read. Its pair has a site's geometry and
/// a site's arithmetic, and is still not a site: it runs once, at the
/// stack's end, over the whole history, and a component declaring the
/// topology must ship it (K3 does). That is why it is its own object
/// with its own closure rather than a third pair on some layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionResidualExitOperand {
    /// `output_attn_res_norm.weight`, `[hidden]`.
    Norm,
    /// `output_attn_res_proj.weight`, `[1, hidden]`.
    Proj,
}

/// The exit's spellings, object-relative. Two groups under one common
/// segment prefix means the container names them `output_attn_res_*.
/// weight`, whatever the checkpoint's stack prefix was.
pub(super) const ATTENTION_RESIDUAL_EXIT_TABLE: &[(&str, AttentionResidualExitOperand)] = &[
    (
        "output_attn_res_norm.weight",
        AttentionResidualExitOperand::Norm,
    ),
    (
        "output_attn_res_proj.weight",
        AttentionResidualExitOperand::Proj,
    ),
];

/// Classify one tensor of the attention-residual exit object by its
/// object-relative name. Exact, like every classifier in this module: a
/// spelling not in the table is `None` and blocks.
pub fn classify_attention_residual_exit_tensor(
    relative_name: &str,
) -> Option<AttentionResidualExitOperand> {
    ATTENTION_RESIDUAL_EXIT_TABLE
        .iter()
        .find(|(name, _)| *name == relative_name)
        .map(|(_, operand)| *operand)
}

/// Whether `relative_name` is one of the four attention-residual SITE
/// operands, asked WITHOUT the declaration.
///
/// The graph builder never calls this — placement of the per-layer pairs
/// is the decoder stack's, as it already was. The op plan calls it in
/// exactly one place: to tell a stray from an unrecognised spelling when
/// a component that does NOT declare the topology ships these names, so
/// the defect can say what the operand implies instead of only that
/// nothing classified it. Recognition is not ownership — the same
/// separation the hyper-connection head's placement arm makes.
pub fn is_attention_residual_site_operand(relative_name: &str) -> bool {
    layer_and_suffix(relative_name).is_some_and(|(_, suffix)| {
        ATTENTION_RESIDUAL_ROLE_TABLE
            .iter()
            .any(|(name, _)| *name == suffix)
    })
}

/// The hyper-connection HEAD's own operands — a component-level object,
/// not a stack operand, and a different operation from a site's.
///
/// `ParallelHead.hc_head` runs no Sinkhorn: `reduce_fn` has ONE row per
/// stream where a site's `mix_fn` has `(2 + hc)·hc`, and `scale` is a
/// single scalar where a site carries three. Wave 17 recorded the
/// difference in the executor ([`HeadWeights`](crate::format::vindex3::opplan::exec::hyper_connection::HeadWeights));
/// this is the same difference on the addressing side, so a checkpoint
/// that stored a site's operands under the head's names would fail the
/// head's geometry rather than bind.
///
/// ```text
/// reduce_fn   [hc, hc·hidden]
/// base        [hc]
/// scale       [1]
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HcHeadOperand {
    ReduceFn,
    Base,
    Scale,
}

/// The head's spellings, as DeepSeek-V4 writes them: three BARE
/// top-level tensors with no `model.` prefix. GLM-5.3-Flash ships none
/// (its `mhc` flag sits unexplained beside that absence and is not read
/// as meaning anything here). Hy4-preview's `model.hc_head.hc_head_fn`
/// is not listed for the same reason its site spelling is not: a
/// different topology's dialect.
pub(super) const HC_HEAD_TABLE: &[(&str, HcHeadOperand)] = &[
    ("hc_head_fn", HcHeadOperand::ReduceFn),
    ("hc_head_base", HcHeadOperand::Base),
    ("hc_head_scale", HcHeadOperand::Scale),
];

/// Classify one tensor of the hyper-connection head object by its
/// object-relative name. Exact, like every classifier in this module:
/// a spelling not in [`HC_HEAD_TABLE`] is `None` and blocks.
pub fn classify_hyper_connection_head_tensor(relative_name: &str) -> Option<HcHeadOperand> {
    HC_HEAD_TABLE
        .iter()
        .find(|(name, _)| *name == relative_name)
        .map(|(_, operand)| *operand)
}

/// Whether `group_prefix` is one of the head's bare tensor groups — the
/// graph builder's placement question, answered from the same table the
/// op plan classifies by so the two cannot disagree about which names
/// are the head's.
pub fn is_hyper_connection_head_group(group_prefix: &str) -> bool {
    HC_HEAD_TABLE.iter().any(|(name, _)| *name == group_prefix)
}

/// One leaf spelling after the expert index (`"w1.weight"`) and the role
/// constructor it binds.
pub(super) type IndexedExpertLeaf = (&'static str, fn(u16) -> OperandRole);

/// One `ExpertFormat::PerExpert` family's indexed-operand spelling: the
/// fixed text surrounding the expert index, and which role each of the
/// family's per-expert leaf names maps to.
///
/// A second vocabulary beside [`ROLE_TABLE`] rather than an attempt to fold
/// indexed suffixes into it, because [`ROLE_TABLE`] matches by exact
/// string equality — the whole point of it never fuzzy-matching a new
/// spelling into the wrong role — and an expert index is exactly the kind
/// of value no static string can stand in for.
pub(super) struct IndexedExpertFamily {
    /// Text before the expert index, including its trailing `.`
    /// (`"block_sparse_moe.experts."`).
    pub(super) prefix: &'static str,
    /// This family's leaf spellings.
    pub(super) leaves: &'static [IndexedExpertLeaf],
}

/// Every `ExpertFormat::PerExpert` family this build recognises.
///
/// Kimi Linear's `w1`/`w2`/`w3`, checked against `KimiBlockSparseMLP.forward`
/// in the checkpoint's `modeling_kimi.py` (`w1`/`w3` feed the gated
/// product, `w2` reads it — NOT alphabetic gate/up/down order), and the
/// Qwen MoE lineage's `mlp.experts.{id}.gate_proj/up_proj/down_proj`,
/// checked against `Qwen2MoeMLP.forward` (`down_proj(act(gate_proj(x)) *
/// up_proj(x))`). Mixtral's `block_sparse_moe.experts.{id}.w1/w2/w3` is
/// Kimi's leaf spelling under Kimi's prefix and is deliberately NOT
/// judged here: it is the held-out architecture of E8, which must onboard
/// through its own entry after the freeze, never a guess from this one.
pub(super) const INDEXED_EXPERT_FAMILIES: &[IndexedExpertFamily] = &[
    IndexedExpertFamily {
        prefix: "block_sparse_moe.experts.",
        leaves: &[
            ("w1.weight", OperandRole::PerExpertGate),
            ("w3.weight", OperandRole::PerExpertUp),
            ("w2.weight", OperandRole::PerExpertDown),
        ],
    },
    IndexedExpertFamily {
        prefix: "mlp.experts.",
        leaves: &[
            ("gate_proj.weight", OperandRole::PerExpertGate),
            ("up_proj.weight", OperandRole::PerExpertUp),
            ("down_proj.weight", OperandRole::PerExpertDown),
        ],
    },
];
