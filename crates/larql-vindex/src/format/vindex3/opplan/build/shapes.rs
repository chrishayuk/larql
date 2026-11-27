//! The shape each role's operand must have.

use super::super::super::encode::segment::SegmentTensor;
use super::super::super::graph::surface::MoeSurface;
use super::super::super::graph::OperandRole;
use super::super::exec::hyper_connection::HC_SCALE_LEN;
use larql_models::config::ExpertFormat;
use larql_models::config::HyperConnectionWeights;

#[allow(unused_imports)]
use super::*;

/// Expected stored shape per role, from the surface's geometry. `None`
/// for roles whose shape contract is not yet pinned.
pub(super) fn expected_shape(
    role: OperandRole,
    g: &StackGeometry,
    moe: Option<&MoeSurface>,
) -> Option<Vec<usize>> {
    use larql_models::config::QkNormScope;
    let StackGeometry {
        hidden,
        q_rows,
        q_proj_rows,
        kv_rows,
        intermediate,
        head_dim,
        num_q_heads,
        num_kv_heads: _,
        qk_scope,
        linear,
        kda,
        mla,
        mamba2,
        conv_qkv,
        hyper_connection,
    } = *g;
    match role {
        // Sinkhorn hyper-connection sites. Every contract follows from the
        // component's DECLARED stream count closing over its width — the
        // same derivation the executor's `mix_rows_for` runs — and none
        // from the tensor: a `[2·hc, hc·hidden]` Sinkhorn-free operand
        // (Hy4-preview's shape) fails here rather than binding to a split
        // it does not parameterise. `hyper_connection` absent while such
        // an operand exists is unreachable past `absent_op`, and answers
        // `None` for the same reason the other family absences do.
        OperandRole::HcAttnMixFn | OperandRole::HcFfnMixFn => {
            let hc = hyper_connection?;
            Some(vec![
                HyperConnectionWeights::mix_rows_for(hc.streams),
                hc.streams * hidden,
            ])
        }
        OperandRole::HcAttnBase | OperandRole::HcFfnBase => {
            Some(vec![HyperConnectionWeights::mix_rows_for(
                hyper_connection?.streams,
            )])
        }
        OperandRole::HcAttnScale | OperandRole::HcFfnScale => Some(vec![HC_SCALE_LEN]),
        // Attention-residual sites. Both contracts close over the
        // component's width alone — the block size parameterises the
        // SCHEDULE, not any operand's shape — and the pair's asymmetry is
        // the contract: a `[hidden]` norm and a `[1, hidden]` projection,
        // multiplied elementwise into one score vector. `[1, hidden]` is
        // checked as the two-dimensional shape it is, so a `[hidden]`
        // tensor stored under the projection's name fails rather than
        // satisfying it by the one-dimensional broadcast equivalence.
        OperandRole::AttnResAttentionNorm | OperandRole::AttnResMlpNorm => Some(vec![hidden]),
        OperandRole::AttnResAttentionProj | OperandRole::AttnResMlpProj => Some(vec![1, hidden]),
        // Mamba2/SSD. Every contract follows from the mixer's own
        // declared geometry closing over the component width; none from
        // the softmax fields, which are zero on a mixer-only stack.
        // `mamba2` absent while such an operand exists is a refusal, for
        // the same reason `linear`/`kda`/`mla` absences are.
        OperandRole::Mamba2InProj => Some(vec![mamba2?.geometry.in_proj_rows(hidden), hidden]),
        OperandRole::Mamba2Conv1d => {
            let g = mamba2?.geometry;
            Some(vec![g.conv_dim(hidden), 1, g.conv_kernel])
        }
        OperandRole::Mamba2Conv1dBias => Some(vec![mamba2?.geometry.conv_dim(hidden)]),
        // Per-head scalars — the axis that separates this family from
        // KDA's per-channel `dt_bias`.
        OperandRole::Mamba2ALog | OperandRole::Mamba2D | OperandRole::Mamba2DtBias => {
            Some(vec![mamba2?.geometry.num_heads])
        }
        // Over the FULL inner width — unlike DeltaNet's per-head norm.
        OperandRole::Mamba2GatedNorm => Some(vec![mamba2?.geometry.d_inner(hidden)]),
        OperandRole::Mamba2OutProj => Some(vec![hidden, mamba2?.geometry.d_inner(hidden)]),
        OperandRole::Mamba2PreMixerNorm => Some(vec![hidden]),
        // Conv-QKV attention. Every contract follows from the hybrid
        // block's own declared geometry; `conv_qkv` absent while such an
        // operand exists is a refusal, for the same reason the other
        // family absences are.
        OperandRole::ConvQkvInProj => Some(vec![conv_qkv?.qkv_rows(), hidden]),
        OperandRole::ConvQkvConv1d => {
            let a = conv_qkv?;
            Some(vec![a.qkv_rows(), 1, a.conv_kernel])
        }
        OperandRole::ConvQkvConv1dBias => Some(vec![conv_qkv?.qkv_rows()]),
        OperandRole::ConvQkvOutProj => Some(vec![hidden, conv_qkv?.attn_out_width()]),
        OperandRole::AttnQ => Some(vec![q_proj_rows, hidden]),
        OperandRole::AttnK | OperandRole::AttnV => Some(vec![kv_rows, hidden]),
        OperandRole::AttnO => Some(vec![hidden, q_rows]),
        OperandRole::PreAttentionNorm
        | OperandRole::PostAttentionNorm
        | OperandRole::PreFfnNorm
        | OperandRole::PostFfnNorm
        | OperandRole::PreAttentionNormBias
        | OperandRole::PostAttentionNormBias
        | OperandRole::PreFfnNormBias
        | OperandRole::PostFfnNormBias
        | OperandRole::PreExpertsNorm
        | OperandRole::PostDenseFfnNorm
        | OperandRole::PostExpertsNorm
        | OperandRole::MoeRouterScale => Some(vec![hidden]),
        OperandRole::MoeRouterPerExpertScale => Some(vec![moe?.experts]),
        OperandRole::LayerScalar => Some(vec![1]),
        OperandRole::AttnQNorm | OperandRole::AttnKNorm => match qk_scope {
            QkNormScope::PerHead => Some(vec![head_dim]),
            // Full-projection shape contract unpinned until a real
            // instance is judged.
            QkNormScope::FullProjection => None,
        },
        // Gated DeltaNet. Every shape follows from the recurrence's own
        // geometry, and none from the softmax fields above — the key and
        // value sides carry different head counts (16 and 48 on Qwen3.8),
        // so nothing there stands in for them.
        //
        // `linear` absent while such an operand exists is a refusal, not a
        // waiver: the stack ships a recurrence whose geometry the component
        // never declared, and closure must not accept an operand it cannot
        // state a contract for.
        OperandRole::LinearAttnInProjQkv => Some(vec![linear?.qkv_channels(), hidden]),
        OperandRole::LinearAttnInProjA | OperandRole::LinearAttnInProjB => {
            Some(vec![linear?.value_heads, hidden])
        }
        OperandRole::LinearAttnInProjZ => Some(vec![linear?.value_width(), hidden]),
        // Depthwise over the fused channels: one kernel per channel.
        OperandRole::LinearAttnConv1d => {
            let l = linear?;
            Some(vec![l.qkv_channels(), 1, l.conv_kernel])
        }
        // Per-value-head scalars.
        OperandRole::LinearAttnALog | OperandRole::LinearAttnDtBias => {
            Some(vec![linear?.value_heads])
        }
        // Gated RMSNorm over ONE value head's width, not the full value
        // side — the norm is applied per head.
        OperandRole::LinearAttnNorm => Some(vec![linear?.value_head_dim]),
        OperandRole::LinearAttnOutProj => Some(vec![hidden, linear?.value_width()]),
        // Kimi Delta Attention. Every contract below follows from the KDA
        // block's own geometry; none from the softmax fields, which on a
        // hybrid checkpoint describe the OTHER layers of the same stack.
        //
        // `kda` absent while a KDA operand exists is a refusal for the
        // same reason `linear` is: an operand whose contract the component
        // never declared cannot be checked, and accepting it unchecked is
        // how a wrong binding survives.
        OperandRole::KdaQProj | OperandRole::KdaKProj | OperandRole::KdaVProj => {
            Some(vec![kda?.value_width(), hidden])
        }
        // Depthwise, one kernel per channel — three independent convs, not
        // one over fused channels. This is a structural difference from
        // Gated DeltaNet, not a parameterisation of it.
        OperandRole::KdaQConv1d | OperandRole::KdaKConv1d | OperandRole::KdaVConv1d => {
            let k = kda?;
            Some(vec![k.value_width(), 1, k.conv_kernel])
        }
        OperandRole::KdaBProj => Some(vec![kda?.num_heads, hidden]),
        OperandRole::KdaALog => Some(vec![kda?.num_heads]),
        // The discriminator: per CHANNEL, where Gated DeltaNet's is per
        // head. A checkpoint whose `dt_bias` is `[Hv]` is not a KDA block,
        // and this is the contract that says so.
        OperandRole::KdaDtBias => Some(vec![kda?.value_width()]),
        OperandRole::KdaONorm => Some(vec![kda?.head_dim]),
        OperandRole::KdaOutProj => Some(vec![hidden, kda?.value_width()]),
        // The full-rank gate is pinned by geometry alone — one projection
        // from `hidden` to the value width — unlike the low-rank pair,
        // whose rank no config declares.
        OperandRole::KdaGProj => Some(vec![kda?.value_width(), hidden]),
        // The f and g gates are low-rank and the config declares no rank,
        // so no per-operand contract can be stated from geometry alone.
        // Their agreement is a CLOSURE fact between the pair — `f_a` is
        // `[rank, hidden]` and `f_b` is `[Hv·Dv, rank]` for one rank — and
        // is checked there rather than invented here. `None` is the same
        // "contract not pinned" answer `QkNormScope::FullProjection` gives.
        OperandRole::KdaFAProj
        | OperandRole::KdaFBProj
        | OperandRole::KdaGAProj
        | OperandRole::KdaGBProj => None,
        // Multi-Latent Attention. Every contract below follows from the
        // MLA block's own geometry — none from the softmax fields, which
        // on a hybrid checkpoint describe the KDA layers of the same
        // stack. `mla` absent while an MLA operand exists is a refusal
        // for the same reason `kda`/`linear` are: an operand whose
        // contract the component never declared cannot be checked.
        OperandRole::MlaQProj => Some(vec![mla?.num_heads * mla?.q_head_dim(), hidden]),
        // The factorised query (K3-MLA-Q-LORA-1). `MlaQBProj` has the
        // SAME row count as `MlaQProj` above — `Hq*q_head_dim`, 18432 on
        // K3 — and differs only in its COLUMN count: the declared rank
        // against `hidden`. That column is the whole discriminator, and
        // its authority is the declared form, so a rank that is absent
        // here answers `None` and the operand is refused rather than
        // checked against a width nobody declared.
        OperandRole::MlaQAProj => Some(vec![mla?.query.rank()?, hidden]),
        OperandRole::MlaQANorm => Some(vec![mla?.query.rank()?]),
        OperandRole::MlaQBProj => {
            let m = mla?;
            Some(vec![m.num_heads * m.q_head_dim(), m.query.rank()?])
        }
        OperandRole::MlaKvAProj => Some(vec![mla?.kv_lora_rank + mla?.qk_rope_head_dim, hidden]),
        // Fused per-head nope-K + V, decompressed from the latent.
        OperandRole::MlaKvBProj => {
            let m = mla?;
            Some(vec![
                m.num_heads * (m.qk_nope_head_dim + m.v_head_dim),
                m.kv_lora_rank,
            ])
        }
        OperandRole::MlaKvANorm => Some(vec![mla?.kv_lora_rank]),
        OperandRole::MlaOutProj => Some(vec![hidden, mla?.num_heads * mla?.v_head_dim]),
        // Same numbers as `MlaOutProj`, transposed: the gate reads `hidden`
        // and writes the aggregated value's width. Identical to
        // `KdaGProj`'s contract on Kimi-K3, which is why only the layer's
        // operator can tell the two spellings apart.
        OperandRole::MlaOutputGate => Some(vec![mla?.num_heads * mla?.v_head_dim, hidden]),
        OperandRole::FfnGate | OperandRole::FfnUp => Some(vec![intermediate, hidden]),
        OperandRole::FfnDown => Some(vec![hidden, intermediate]),
        OperandRole::FfnUpBias => Some(vec![intermediate]),
        OperandRole::FfnDownBias => Some(vec![hidden]),
        // Linear(hidden -> q_heads*head_dim), per the judged spec.
        OperandRole::AttnOutputGate => Some(vec![q_rows, hidden]),
        // A bias is one value per output row of its projection.
        OperandRole::AttnQBias => Some(vec![q_rows]),
        OperandRole::AttnKBias | OperandRole::AttnVBias => Some(vec![kv_rows]),
        OperandRole::AttnOBias => Some(vec![hidden]),
        // One logit per query head, per the judged spec.
        OperandRole::AttnSinks => Some(vec![num_q_heads]),
        // Routed FFN: every shape follows from the judgment's expert count,
        // width and storage format; with no judgment there is no contract
        // (the operand is refused by `absent_op` before this is asked).
        OperandRole::MoeRouterWeight => Some(vec![moe?.experts, hidden]),
        OperandRole::MoeRouterBias => Some(vec![moe?.experts]),
        // The latent wrapper. Both projections cross BETWEEN the two
        // widths, so each names `hidden` on one axis and the latent width
        // on the other — which is what makes them the only routed
        // operands whose contract needs both. `None` when the component
        // declares no latent form: there is then no contract to state,
        // and the operand is refused by name before this is asked.
        OperandRole::MoeLatentDownProj => Some(vec![moe?.latent?.width, hidden]),
        OperandRole::MoeLatentNorm => Some(vec![moe?.latent?.width]),
        OperandRole::MoeLatentUpProj => Some(vec![hidden, moe?.latent?.width]),
        OperandRole::ExpertGateUp => {
            let m = moe?;
            Some(packed_shape(
                m,
                FUSED_BRANCHES * m.expert_intermediate_size,
                m.routed_expert_input_width(hidden),
            ))
        }
        OperandRole::ExpertGateUpScales => {
            let m = moe?;
            Some(scales_shape(
                m,
                FUSED_BRANCHES * m.expert_intermediate_size,
                m.routed_expert_input_width(hidden),
            ))
        }
        OperandRole::ExpertGateUpBias => {
            let m = moe?;
            Some(vec![m.experts, FUSED_BRANCHES * m.expert_intermediate_size])
        }
        OperandRole::ExpertDown => {
            let m = moe?;
            Some(packed_shape(
                m,
                m.routed_expert_input_width(hidden),
                m.expert_intermediate_size,
            ))
        }
        OperandRole::ExpertDownScales => {
            let m = moe?;
            Some(scales_shape(
                m,
                m.routed_expert_input_width(hidden),
                m.expert_intermediate_size,
            ))
        }
        OperandRole::ExpertDownBias => {
            let m = moe?;
            Some(vec![m.experts, m.routed_expert_input_width(hidden)])
        }
        // `ExpertFormat::PerExpert`: one `[inter, hidden]` gate/up and one
        // `[hidden, inter]` down PER EXPERT — the index carried on the role
        // picks which expert's operand this is, not which shape.
        OperandRole::PerExpertGate(_) | OperandRole::PerExpertUp(_) => {
            let m = moe?;
            Some(vec![
                m.expert_intermediate_size,
                m.routed_expert_input_width(hidden),
            ])
        }
        OperandRole::PerExpertDown(_) => {
            let m = moe?;
            Some(vec![
                m.routed_expert_input_width(hidden),
                m.expert_intermediate_size,
            ])
        }
        // Always-active shared expert(s): the same gated-FFN shape as a
        // routed expert, at the width the judgment declares. Sized from
        // `shared_expert_intermediate_size` and NOT re-derived here —
        // Kimi's `KimiSparseMoeBlock.__init__` sizes one wider `KimiMLP`
        // at `moe_intermediate_size * num_shared_experts` while Qwen's
        // block sizes it from its own key, and Qwen1.5-MoE's two answers
        // differ fourfold (5632 declared against 1408 derived).
        OperandRole::SharedExpertGate | OperandRole::SharedExpertUp => {
            Some(vec![shared_expert_width(moe?)?, hidden])
        }
        OperandRole::SharedExpertDown => Some(vec![hidden, shared_expert_width(moe?)?]),
        // The scalar that gates the branch: one logit per token.
        OperandRole::SharedExpertBranchGate => Some(vec![SCALAR_GATE_ROWS, hidden]),
    }
}

/// Gate and up: the two branches sharing one fused operand.
pub(super) const FUSED_BRANCHES: usize = larql_models::quant::mxfp4::FUSED_HALVES;

/// The shared-expert branch gate projects to ONE logit per token, so its
/// operand carries a single row. Named rather than written as a literal
/// `1`, which at that position would read as a placeholder.
pub(super) const SCALAR_GATE_ROWS: usize = 1;

/// The width of the always-active shared branch, or `None` when the
/// judgment declares no shared expert.
///
/// The single place this build answers the question. The two lineages
/// size the branch differently and the architecture already resolved
/// which applies (`ModelArchitecture::shared_expert_intermediate_size`);
/// re-deriving it here from the routed width would put a second answer
/// beside that one, and on Qwen1.5-MoE the two differ fourfold. A graph
/// that declares the branch by count alone has had its width filled from
/// the stored gate tensor by [`resolve_shared_expert_width`] before this
/// is asked, so a declared branch never answers `None` here.
pub(super) fn shared_expert_width(moe: &MoeSurface) -> Option<usize> {
    (moe.shared_experts > 0).then_some(moe.shared_expert_intermediate_size)?
}

/// A routed-FFN judgment whose shared branch has a width, wherever the
/// graph left it: the graph's own declaration when it carries one, else
/// the stored shape of the branch's gate tensor, `[width, hidden]`.
///
/// Graphs written before the width was part of the surface (schema 6,
/// before 2026-09-03) declare the branch by count only. Reading a missing
/// optional as "no branch" planned those models WITHOUT the shared expert
/// their graph declares and their closure counted — a silent omission
/// that every later parity and residency number would have inherited.
/// The gate tensor is the container's own authority on the width: this
/// is not a re-derivation from the routed width and assumes no lineage
/// convention. The shape check then holds up and down to the same width,
/// and refuses a branch whose three tensors disagree. A declared branch
/// whose gate is absent is refused by [`required_roles`] before any of
/// this matters.
pub(super) fn resolve_shared_expert_width(
    moe: MoeSurface,
    gate: Option<&SegmentTensor>,
) -> MoeSurface {
    if moe.shared_experts == 0 || moe.shared_expert_intermediate_size.is_some() {
        return moe;
    }
    let Some(gate) = gate else {
        return moe;
    };
    let width = match gate.shape.as_slice() {
        [width, _hidden] => Some(*width),
        _ => None,
    };
    MoeSurface {
        shared_expert_intermediate_size: width,
        ..moe
    }
}

/// Stored shape of a packed `[experts, rows, k]` projection under the
/// judged format: MXFP4 packs `k` as `k/32` groups of 16 bytes (32
/// nibbles); an unquantised packed store keeps `[experts, rows, k]`.
pub(super) fn packed_shape(moe: &MoeSurface, rows: usize, k: usize) -> Vec<usize> {
    use larql_models::quant::mxfp4::{MXFP4_GROUP_BYTES, MXFP4_GROUP_ELEMS};
    match moe.expert_format {
        ExpertFormat::PackedMxfp4 => {
            vec![moe.experts, rows, k / MXFP4_GROUP_ELEMS, MXFP4_GROUP_BYTES]
        }
        ExpertFormat::PackedBF16 | ExpertFormat::PerExpert => vec![moe.experts, rows, k],
    }
}

/// Stored shape of the companion scales stream: one E8M0 byte per group.
pub(super) fn scales_shape(moe: &MoeSurface, rows: usize, k: usize) -> Vec<usize> {
    use larql_models::quant::mxfp4::MXFP4_GROUP_ELEMS;
    match moe.expert_format {
        ExpertFormat::PackedMxfp4 => vec![moe.experts, rows, k / MXFP4_GROUP_ELEMS],
        ExpertFormat::PackedBF16 | ExpertFormat::PerExpert => vec![moe.experts, rows],
    }
}
