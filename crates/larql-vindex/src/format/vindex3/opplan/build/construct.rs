//! Phase 4 of [`plan_component_ops`]: build the plan once closure holds,
//! so every operand lookup is total. Reads only the facts the closure
//! phases established, carried in [`ClosedComponent`].

use super::super::super::encode::segment::SegmentTensor;
use super::super::super::graph::surface::MoeSurface;
use super::super::super::graph::{LogicalObject, NormPlacement, ObjectKind, OperandRole};
use super::super::{
    AttentionOp, AttentionResidualExitOp, AttentionResidualLayerOp, AttnResSiteOp, ComponentOpPlan,
    EmbeddingOp, ExpertBank, FfnOp, GateOp, GatedDeltaOp, HcSiteOp, HybridFfnOp,
    HyperConnectionHeadOp, HyperConnectionLayerOp, KdaOp, KdaOutputGate, LatentBranchOp,
    LatentNormOp, LayerAttention, LayerFfn, LayerPlan, Mamba2Op, MlaOp, MlaQueryProjection, NormOp,
    OperandRef, OutputOp, PackedProjection, PositionTableOp, QkNormOp, RoutedFfnOp,
    SharedExpertBranchGateOp, SharedExpertOp, SinkOp,
};
use larql_models::config::ExpertFormat;
use larql_models::config::MoeRouterKind;
use std::collections::BTreeMap;

use super::super::super::graph::component::Component;
use super::super::super::graph::surface::{AttentionSurface, ExecutionSurface, FfnSurface};
use super::super::super::graph::AttentionLayerPolicy;
use super::absent::StackGeometry;

#[allow(unused_imports)]
use super::*;

/// What closure established, for [`construct_plan`]: every fact the plan is
/// built from, already checked.
pub(super) struct ClosedComponent<'a> {
    pub(super) surface: &'a ExecutionSurface,
    pub(super) component: &'a Component,
    pub(super) placement: NormPlacement,
    pub(super) post_norm: Option<larql_models::config::NormSpec>,
    pub(super) hyper_connection: Option<larql_models::config::HyperConnection>,
    pub(super) attention_residual: bool,
    pub(super) attention_table: &'a [AttentionLayerPolicy],
    pub(super) tables: &'a BTreeMap<ObjectKind, (&'a LogicalObject, Vec<SegmentTensor>)>,
    pub(super) attn: Option<&'a AttentionSurface>,
    pub(super) ffn_surface: Option<&'a FfnSurface>,
    pub(super) ffn_moe: Option<MoeSurface>,
    pub(super) gated_ffn: bool,
    pub(super) by_layer: &'a BTreeMap<usize, BTreeMap<OperandRole, SegmentTensor>>,
    pub(super) bank_by_layer: &'a BTreeMap<usize, BTreeMap<OperandRole, SegmentTensor>>,
    pub(super) layer_geometry: &'a dyn Fn(usize) -> StackGeometry,
    pub(super) inter_for: &'a dyn Fn(usize) -> Option<usize>,
    pub(super) vocab: Option<usize>,
    pub(super) embedding_tensor: Option<(String, SegmentTensor)>,
    pub(super) final_norm_tensor: Option<(String, SegmentTensor, Option<SegmentTensor>)>,
    pub(super) position_tensor: Option<(
        larql_models::config::InputPositionEncoding,
        String,
        SegmentTensor,
    )>,
    pub(super) head_tensor: Option<(String, SegmentTensor)>,
    pub(super) hc_head_tensors: Option<(String, SegmentTensor, SegmentTensor, SegmentTensor)>,
    pub(super) attn_res_exit_tensors: Option<(String, SegmentTensor, SegmentTensor)>,
}

pub(super) fn construct_plan(c: ClosedComponent<'_>) -> ComponentOpPlan {
    let ClosedComponent {
        surface,
        component,
        placement,
        post_norm,
        hyper_connection,
        attention_residual,
        attention_table,
        tables,
        attn,
        ffn_surface,
        ffn_moe,
        gated_ffn,
        by_layer,
        bank_by_layer,
        layer_geometry,
        inter_for,
        vocab,
        embedding_tensor,
        final_norm_tensor,
        position_tensor,
        head_tensor,
        hc_head_tensors,
        attn_res_exit_tensors,
    } = c;
    // ── Plan construction (closure holds; lookups are now total) ──
    let operand = |object: &str, tensor: &SegmentTensor| OperandRef {
        object: object.to_string(),
        tensor: tensor.name.clone(),
        dtype: tensor.dtype.clone(),
        shape: tensor.shape.clone(),
    };
    // The spec travels whole: kind, epsilon and weight offset all come
    // from the site being built, never from a model-scope answer.
    let norm_op =
        |spec: larql_models::config::NormSpec, object: &str, tensor: &SegmentTensor| NormOp {
            kind: spec.kind,
            eps: spec.eps,
            weight_offset: spec.weight_offset,
            weight: operand(object, tensor),
            bias: None,
        };

    let stack_id = tables
        .get(&ObjectKind::DecoderStack)
        .map(|(o, _)| o.id.clone())
        .unwrap_or_default();
    let mut layers = Vec::with_capacity(component.num_layers);
    for layer in 0..component.num_layers {
        let slot = &by_layer[&layer];
        let get = |role: OperandRole| &slot[&role];
        // A trunk norm site: its weight, and the bias beside it when the
        // surface declares one (closure required it then).
        let trunk_norm = |spec: larql_models::config::NormSpec, role: OperandRole| NormOp {
            bias: super::roles::norm_bias_role(role)
                .and_then(|b| slot.get(&b))
                .map(|t| operand(&stack_id, t)),
            ..norm_op(spec, &stack_id, get(role))
        };
        let policy = &attention_table[layer];
        let geometry = layer_geometry(layer);
        // A mixer-only layer, on operand evidence: the fused five-way
        // `in_proj` is the discriminator (no other operator's role table
        // can put it in `slot`). Its whole program is the mixer — one
        // pre-block norm, no attention wrap, no FFN — so it is built
        // here and the transformer shape below is never consulted.
        if slot.contains_key(&OperandRole::Mamba2InProj) {
            let mixer = surface.mamba2.unwrap_or_else(|| {
                panic!(
                    "layer {layer} ships a Mamba2 operand while the component declares no \
                     mixer surface; closure should have refused this before the plan was built"
                )
            });
            let consumed = slot.len();
            layers.push(LayerPlan {
                declared_norm_eps: surface.norm.pre.eps,
                layer,
                pre_attention_norm: Some(norm_op(
                    surface.norm.pre,
                    &stack_id,
                    get(OperandRole::Mamba2PreMixerNorm),
                )),
                attention: LayerAttention::Mamba2(Box::new(Mamba2Op {
                    geometry: mixer.geometry,
                    activation: mixer.activation,
                    residual_in_fp32: surface.residual_in_fp32,
                    in_proj: operand(&stack_id, get(OperandRole::Mamba2InProj)),
                    conv1d: operand(&stack_id, get(OperandRole::Mamba2Conv1d)),
                    conv1d_bias: slot
                        .get(&OperandRole::Mamba2Conv1dBias)
                        .map(|t| operand(&stack_id, t)),
                    a_log: operand(&stack_id, get(OperandRole::Mamba2ALog)),
                    d: operand(&stack_id, get(OperandRole::Mamba2D)),
                    dt_bias: operand(&stack_id, get(OperandRole::Mamba2DtBias)),
                    gated_norm: slot
                        .get(&OperandRole::Mamba2GatedNorm)
                        .map(|t| norm_op(surface.norm.pre, &stack_id, t)),
                    out_proj: operand(&stack_id, get(OperandRole::Mamba2OutProj)),
                })),
                post_attention_norm: None,
                pre_ffn_norm: None,
                ffn: None,
                post_ffn_norm: None,
                layer_scale: None,
                hyper_connection: None,
                // A one-sublayer block carries neither of the topology's
                // two sites; `absent_op` refuses the operands on it by
                // name rather than planning a layer with none.
                attention_residual: None,
                residual_scale: surface.residual_scale,
                operands_accounted: consumed,
                operands_present: consumed,
            });
            continue;
        }
        // A conv-QKV attention layer, on operand evidence: its fused QKV
        // `in_proj` role is the discriminator (only the conv-QKV table
        // can put it in `slot`). Its whole program is the block — one
        // pre-block norm, no attention wrap, no FFN — the same shape as
        // the mixer arm above.
        if slot.contains_key(&OperandRole::ConvQkvInProj) {
            let attn_geometry = surface.conv_qkv.unwrap_or_else(|| {
                panic!(
                    "layer {layer} ships a conv-QKV operand while the component declares no \
                     conv-QKV surface; closure should have refused this before the plan was built"
                )
            });
            let consumed = slot.len();
            layers.push(LayerPlan {
                declared_norm_eps: surface.norm.pre.eps,
                layer,
                pre_attention_norm: Some(norm_op(
                    surface.norm.pre,
                    &stack_id,
                    get(OperandRole::Mamba2PreMixerNorm),
                )),
                attention: LayerAttention::ConvQkv(Box::new(super::super::conv_qkv::ConvQkvOp {
                    geometry: attn_geometry,
                    residual_in_fp32: surface.residual_in_fp32,
                    in_proj: operand(&stack_id, get(OperandRole::ConvQkvInProj)),
                    conv1d: operand(&stack_id, get(OperandRole::ConvQkvConv1d)),
                    conv1d_bias: slot
                        .get(&OperandRole::ConvQkvConv1dBias)
                        .map(|t| operand(&stack_id, t)),
                    out_proj: operand(&stack_id, get(OperandRole::ConvQkvOutProj)),
                })),
                post_attention_norm: None,
                pre_ffn_norm: None,
                ffn: None,
                post_ffn_norm: None,
                layer_scale: None,
                hyper_connection: None,
                // A one-sublayer block carries neither of the topology's
                // two sites; `absent_op` refuses the operands on it by
                // name rather than planning a layer with none.
                attention_residual: None,
                residual_scale: surface.residual_scale,
                operands_accounted: consumed,
                operands_present: consumed,
            });
            continue;
        }
        // Every non-mixer layer's program includes attention wrap norms
        // and an FFN, so their surface groups are present when closure
        // held — the panics state the invariant, mirroring KDA's below.
        let ffn_s = ffn_surface.unwrap_or_else(|| {
            panic!(
                "layer {layer} requires an FFN op while the surface carries no ffn group; \
                 closure should have refused this before the plan was built"
            )
        });
        // Q/K/V are biased under either declaration; the output only
        // under `attention_bias` — `qkv_bias` is Qwen2's unbiased `o_proj`.
        let all_four = attn.and_then(|a| a.attention_bias) == Some(true);
        let qkv_only = attn.and_then(|a| a.qkv_bias) == Some(true);
        let bias = |role: OperandRole| {
            let declared = match role {
                OperandRole::AttnOBias => all_four,
                _ => all_four || qkv_only,
            };
            declared.then(|| operand(&stack_id, get(role)))
        };
        let qk_norm = match (attn, slot.contains_key(&OperandRole::AttnQNorm)) {
            (Some(a), true) => Some(QkNormOp {
                scope: a.qk_norm_scope,
                weight_offset: a.qk_norm_weight_offset,
                q: operand(&stack_id, get(OperandRole::AttnQNorm)),
                k: operand(&stack_id, get(OperandRole::AttnKNorm)),
            }),
            _ => None,
        };
        // Placement decides which operand feeds the pre-FFN norm: the
        // dedicated one under four-norm, the overloaded
        // `post_attention_layernorm` under two-norm.
        // Which norm operand feeds each site. `None` for the pre-FFN
        // slot means the FFN reads the raw residual — the post-norm
        // program — and is not the same as a missing operand.
        // Which pre-sublayer norm operands this placement HAS. `None`
        // means the site does not exist, and the operand is not there to
        // be fetched — a post-norm stack ships neither.
        let pre_attn_role = match placement {
            NormPlacement::PostOnly => None,
            _ => Some(OperandRole::PreAttentionNorm),
        };
        let (post_attention_norm, pre_ffn_role, post_ffn_norm) = match placement {
            NormPlacement::PrePost => {
                let spec = post_norm.expect("PrePost resolves or returns above");
                (
                    Some(trunk_norm(spec, OperandRole::PostAttentionNorm)),
                    Some(OperandRole::PreFfnNorm),
                    Some(trunk_norm(spec, OperandRole::PostFfnNorm)),
                )
            }
            // Both wrap norms, no pre-FFN norm: each sublayer reads the
            // raw residual and its OUTPUT is normalised before the add.
            NormPlacement::PostOnly => {
                let spec = post_norm.expect("PostOnly resolves or returns above");
                (
                    Some(trunk_norm(spec, OperandRole::PostAttentionNorm)),
                    None,
                    Some(trunk_norm(spec, OperandRole::PostFfnNorm)),
                )
            }
            NormPlacement::PreOnly => (None, Some(OperandRole::PostAttentionNorm), None),
            NormPlacement::PreMixer => panic!(
                "layer {layer} carries transformer operands under a mixer-only norm \
                 placement; closure should have refused this before the plan was built"
            ),
        };
        let bank_slot = bank_by_layer.get(&layer);
        let bank_id = tables
            .get(&ObjectKind::ExpertBank)
            .map(|(o, _)| o.id.clone())
            .unwrap_or_default();
        let dense_op = || FfnOp {
            intermediate_size: inter_for(layer).unwrap_or_else(|| {
                panic!(
                    "component {} plans a dense FFN layer with no declared dense width; \
                     closure should have refused this before the plan was built",
                    component.id
                )
            }),
            activation: ffn_s.activation,
            gate_policy: ffn_s.gate_policy,
            gate: gated_ffn.then(|| operand(&stack_id, get(OperandRole::FfnGate))),
            up: operand(&stack_id, get(OperandRole::FfnUp)),
            down: operand(&stack_id, get(OperandRole::FfnDown)),
            up_bias: slot
                .get(&OperandRole::FfnUpBias)
                .map(|t| operand(&stack_id, t)),
            down_bias: slot
                .get(&OperandRole::FfnDownBias)
                .map(|t| operand(&stack_id, t)),
        };
        let ffn = match (ffn_moe, bank_slot) {
            // A declared-dense prefix layer plans dense even if stray bank
            // tensors exist (the mismatch defect above already refuses
            // the plan); a declared-routed layer with no bank plans dense
            // only as a placeholder behind its own defect.
            (Some(moe), Some(bank)) if declared_routed(ffn_moe.as_ref(), layer) == Some(true) => {
                let moe =
                    resolve_shared_expert_width(moe, slot.get(&OperandRole::SharedExpertGate));
                let bank_operand = |role: OperandRole| operand(&bank_id, &bank[&role]);
                let optional = |role: OperandRole| bank.get(&role).map(|t| operand(&bank_id, t));
                let gemma4_router = moe.router_kind == MoeRouterKind::TopKRenormScaled;
                // `ExpertFormat::PerExpert`: no fused operand exists, so the
                // bank is `experts` independent gate/up/down triples rather
                // than one `PackedProjection` per branch — see
                // `ExpertBank`'s docs for why this cannot reuse the packed
                // shape with a placeholder.
                let bank = if moe.expert_format == ExpertFormat::PerExpert {
                    let per_expert = |ctor: fn(u16) -> OperandRole| -> Vec<OperandRef> {
                        (0..moe.experts as u16)
                            .map(|e| bank_operand(ctor(e)))
                            .collect()
                    };
                    ExpertBank::PerExpert {
                        gate: per_expert(OperandRole::PerExpertGate),
                        up: per_expert(OperandRole::PerExpertUp),
                        down: per_expert(OperandRole::PerExpertDown),
                    }
                } else {
                    ExpertBank::Packed {
                        gate_up: Box::new(PackedProjection {
                            weights: bank_operand(OperandRole::ExpertGateUp),
                            scales: optional(OperandRole::ExpertGateUpScales),
                            bias: optional(OperandRole::ExpertGateUpBias),
                        }),
                        down: Box::new(PackedProjection {
                            weights: bank_operand(OperandRole::ExpertDown),
                            scales: optional(OperandRole::ExpertDownScales),
                            bias: optional(OperandRole::ExpertDownBias),
                        }),
                    }
                };
                // Always-active, unscaled — Kimi's `KimiSparseMoeBlock.
                // forward`: `y = moe(...); y = y + shared_experts(identity)`.
                // `required_roles`/`absent_op` paired `Some` here with
                // `moe.shared_experts > 0` exactly, so this cannot desync
                // from the closure pass that admitted the layer.
                let shared = shared_expert_width(&moe).map(|width| SharedExpertOp {
                    intermediate_size: width,
                    activation: ffn_s.activation,
                    gate_policy: ffn_s.gate_policy,
                    gate: operand(&stack_id, get(OperandRole::SharedExpertGate)),
                    up: operand(&stack_id, get(OperandRole::SharedExpertUp)),
                    down: operand(&stack_id, get(OperandRole::SharedExpertDown)),
                    branch_gate: moe.shared_expert_gate.map(|spec| SharedExpertBranchGateOp {
                        spec,
                        weight: operand(&stack_id, get(OperandRole::SharedExpertBranchGate)),
                    }),
                });
                // The latent wrapper, paired with `required_roles` and
                // `absent_op` the same way the shared branch is: `Some`
                // here exactly when the surface declares the form, so the
                // op cannot carry a bottleneck the closure pass did not
                // admit operands for — nor miss one it did.
                let latent = moe.latent.map(|l| LatentBranchOp {
                    width: l.width,
                    down: operand(&stack_id, get(OperandRole::MoeLatentDownProj)),
                    up: operand(&stack_id, get(OperandRole::MoeLatentUpProj)),
                    norm: l.norm.map(|n| LatentNormOp {
                        weight: operand(&stack_id, get(OperandRole::MoeLatentNorm)),
                        eps: n.eps,
                    }),
                });
                let routed = RoutedFfnOp {
                    experts: moe.experts,
                    top_k: moe.top_k,
                    expert_intermediate_size: moe.expert_intermediate_size,
                    router_kind: moe.router_kind,
                    routing_policy: moe.routing_policy,
                    branch_scale: moe.branch_scale,
                    activation: ffn_s.activation,
                    gate_policy: ffn_s.gate_policy,
                    expert_format: moe.expert_format,
                    gate_up_layout: moe.gate_up_layout,
                    router: operand(&stack_id, get(OperandRole::MoeRouterWeight)),
                    router_bias: moe
                        .router_bias
                        .then(|| operand(&stack_id, get(OperandRole::MoeRouterBias))),
                    bank,
                    shared,
                    latent,
                    router_scale: gemma4_router
                        .then(|| operand(&stack_id, get(OperandRole::MoeRouterScale))),
                    router_per_expert_scale: gemma4_router
                        .then(|| operand(&stack_id, get(OperandRole::MoeRouterPerExpertScale))),
                    // The router's scale-less norm uses the layer's norm
                    // epsilon (HF: `Gemma4RMSNorm(eps=config.rms_norm_eps,
                    // with_scale=False)`).
                    router_norm_eps: gemma4_router.then_some(surface.norm.pre.eps),
                };
                if moe.hybrid {
                    LayerFfn::Hybrid(Box::new(HybridFfnOp {
                        dense: dense_op(),
                        routed,
                        pre_experts_norm: norm_op(
                            surface.norm.pre,
                            &stack_id,
                            get(OperandRole::PreExpertsNorm),
                        ),
                        post_dense_norm: norm_op(
                            surface.norm.pre,
                            &stack_id,
                            get(OperandRole::PostDenseFfnNorm),
                        ),
                        post_experts_norm: norm_op(
                            surface.norm.pre,
                            &stack_id,
                            get(OperandRole::PostExpertsNorm),
                        ),
                    }))
                } else {
                    LayerFfn::Routed(Box::new(routed))
                }
            }
            _ => LayerFfn::Dense(Box::new(dense_op())),
        };
        let layer_scale = slot
            .get(&OperandRole::LayerScalar)
            .map(|t| operand(&stack_id, t));
        // The two sites, bound iff the component declares the topology.
        // Built HERE, in the transformer arm, and not above the mixer
        // arms: closure required all six on every transformer layer, so
        // the lookups are total for this layer kind — and a mixer layer
        // under the topology never reaches this point, because closure
        // refused it as unjudged. Binding eagerly for every layer kind
        // would turn that refusal's absence into an index panic instead
        // of the named defect.
        let hc_site = |mix_fn: OperandRole, base: OperandRole, scale: OperandRole| HcSiteOp {
            mix_fn: operand(&stack_id, get(mix_fn)),
            base: operand(&stack_id, get(base)),
            scale: operand(&stack_id, get(scale)),
        };
        // The topology's two site pairs, bound iff the component declares
        // the period. Built HERE for the same reason the hyper-connection
        // sites are: closure required all four on every transformer
        // layer, so the lookups are total for this layer kind, and a
        // mixer layer under the topology never reaches this point.
        let attn_res_site = |norm: OperandRole, proj: OperandRole| AttnResSiteOp {
            norm: operand(&stack_id, get(norm)),
            proj: operand(&stack_id, get(proj)),
        };
        let attention_residual_sites = attention_residual.then(|| AttentionResidualLayerOp {
            attention: attn_res_site(
                OperandRole::AttnResAttentionNorm,
                OperandRole::AttnResAttentionProj,
            ),
            ffn: attn_res_site(OperandRole::AttnResMlpNorm, OperandRole::AttnResMlpProj),
        });
        let hyper_connection_sites = hyper_connection.map(|_| HyperConnectionLayerOp {
            attention: hc_site(
                OperandRole::HcAttnMixFn,
                OperandRole::HcAttnBase,
                OperandRole::HcAttnScale,
            ),
            ffn: hc_site(
                OperandRole::HcFfnMixFn,
                OperandRole::HcFfnBase,
                OperandRole::HcFfnScale,
            ),
        });
        let consumed = slot.len() + bank_slot.map_or(0, |b| b.len());
        layers.push(LayerPlan {
            declared_norm_eps: surface.norm.pre.eps,
            layer,
            pre_attention_norm: pre_attn_role.map(|role| trunk_norm(surface.norm.pre, role)),
            // Which attention-class operator this layer runs, decided on
            // OPERAND EVIDENCE: a layer holding the fused q|k|v projection
            // of a recurrence is a DeltaNet layer, whatever else is
            // declared. Roles arrive only through exact ROLE_TABLE
            // suffixes, so nothing reaches here by lexical fallback.
            attention: if slot.contains_key(&OperandRole::KdaDtBias) {
                // KDA, on operand evidence. `dt_bias` is the discriminator
                // and not by accident: Gated DeltaNet carries one too, at
                // `[Hv]` against KDA's `[Hv·Dv]`, so the two are separated
                // by the operand whose GEOMETRY differs rather than by a
                // name either could have used. The role only reached this
                // slot because the layer's operator said KDA, so this is a
                // second, independent agreement rather than a restatement.
                let k = surface.kda.unwrap_or_else(|| {
                    panic!(
                        "layer {layer} ships a KDA operand while the component declares no KDA \
                         geometry; closure should have refused this before the plan was built"
                    )
                });
                // The rank the config never declares, resolved ONCE from
                // the operand that carries it and then stated on the op.
                // `f_a_proj` is `[rank, hidden]`; taking the row count is
                // the only place this is read, so no consumer has to
                // recover it from a shape later.
                let gate_rank = get(OperandRole::KdaFAProj)
                    .shape
                    .first()
                    .copied()
                    .unwrap_or(0);
                LayerAttention::Kda(Box::new(KdaOp {
                    num_heads: k.num_heads,
                    head_dim: k.head_dim,
                    conv_kernel: k.conv_kernel,
                    gate_rank,
                    gate_lower_bound: surface.kda_gate_lower_bound,
                    gate_form: surface.kda_gate_form,
                    q_proj: operand(&stack_id, get(OperandRole::KdaQProj)),
                    k_proj: operand(&stack_id, get(OperandRole::KdaKProj)),
                    v_proj: operand(&stack_id, get(OperandRole::KdaVProj)),
                    q_conv1d: operand(&stack_id, get(OperandRole::KdaQConv1d)),
                    k_conv1d: operand(&stack_id, get(OperandRole::KdaKConv1d)),
                    v_conv1d: operand(&stack_id, get(OperandRole::KdaVConv1d)),
                    f_a_proj: operand(&stack_id, get(OperandRole::KdaFAProj)),
                    f_b_proj: operand(&stack_id, get(OperandRole::KdaFBProj)),
                    // The form is the DECLARATION's; closure has already
                    // held the shipped operands to it, so the role read
                    // here is the one the declaration made required.
                    output_gate: if surface.kda_use_full_rank_gate == Some(true) {
                        KdaOutputGate::FullRank {
                            g_proj: operand(&stack_id, get(OperandRole::KdaGProj)),
                        }
                    } else {
                        KdaOutputGate::LowRank {
                            g_a_proj: operand(&stack_id, get(OperandRole::KdaGAProj)),
                            g_b_proj: operand(&stack_id, get(OperandRole::KdaGBProj)),
                        }
                    },
                    b_proj: operand(&stack_id, get(OperandRole::KdaBProj)),
                    a_log: operand(&stack_id, get(OperandRole::KdaALog)),
                    dt_bias: operand(&stack_id, get(OperandRole::KdaDtBias)),
                    o_norm: operand(&stack_id, get(OperandRole::KdaONorm)),
                    out_proj: operand(&stack_id, get(OperandRole::KdaOutProj)),
                }))
            } else if slot.contains_key(&OperandRole::LinearAttnInProjQkv) {
                let l = surface.linear_attention.unwrap_or_else(|| {
                    panic!(
                        "layer {layer} ships a Gated DeltaNet operand while the component \
                         declares no linear-attention geometry; closure should have refused \
                         this before the plan was built"
                    )
                });
                LayerAttention::GatedDelta(Box::new(GatedDeltaOp {
                    num_key_heads: l.key_heads,
                    num_value_heads: l.value_heads,
                    key_head_dim: l.key_head_dim,
                    value_head_dim: l.value_head_dim,
                    conv_kernel: l.conv_kernel,
                    state_dtype: l.state_dtype,
                    in_proj_qkv: operand(&stack_id, get(OperandRole::LinearAttnInProjQkv)),
                    in_proj_a: operand(&stack_id, get(OperandRole::LinearAttnInProjA)),
                    in_proj_b: operand(&stack_id, get(OperandRole::LinearAttnInProjB)),
                    in_proj_z: operand(&stack_id, get(OperandRole::LinearAttnInProjZ)),
                    conv1d: operand(&stack_id, get(OperandRole::LinearAttnConv1d)),
                    a_log: operand(&stack_id, get(OperandRole::LinearAttnALog)),
                    dt_bias: operand(&stack_id, get(OperandRole::LinearAttnDtBias)),
                    norm: operand(&stack_id, get(OperandRole::LinearAttnNorm)),
                    out_proj: operand(&stack_id, get(OperandRole::LinearAttnOutProj)),
                }))
            } else if slot.contains_key(&OperandRole::MlaKvAProj) {
                // MLA, on operand evidence. `kv_a_proj_with_mqa` is the
                // discriminator: no other operator's role table can put it
                // in `slot`, so its presence alone proves the layer's
                // operator said MLA — the same "role only reached here
                // because the graph already decided" reasoning KDA's
                // branch states above.
                let m = surface.mla.unwrap_or_else(|| {
                    panic!(
                        "layer {layer} ships an MLA operand while the component declares no \
                         MLA geometry; closure should have refused this before the plan was \
                         built"
                    )
                });
                LayerAttention::Mla(Box::new(MlaOp {
                    num_heads: m.num_heads,
                    kv_lora_rank: m.kv_lora_rank,
                    qk_nope_head_dim: m.qk_nope_head_dim,
                    qk_rope_head_dim: m.qk_rope_head_dim,
                    v_head_dim: m.v_head_dim,
                    // The form is the DECLARATION's; closure has already
                    // held the shipped operands to it, so the roles read
                    // here are the ones the declaration made required.
                    query: match m.query.rank() {
                        None => MlaQueryProjection::Direct {
                            q_proj: operand(&stack_id, get(OperandRole::MlaQProj)),
                        },
                        Some(_) => MlaQueryProjection::LowRank {
                            q_a_proj: operand(&stack_id, get(OperandRole::MlaQAProj)),
                            q_a_norm: operand(&stack_id, get(OperandRole::MlaQANorm)),
                            q_b_proj: operand(&stack_id, get(OperandRole::MlaQBProj)),
                            q_a_norm_eps: m.query.norm_eps(),
                        },
                    },
                    kv_a_proj: operand(&stack_id, get(OperandRole::MlaKvAProj)),
                    kv_b_proj: operand(&stack_id, get(OperandRole::MlaKvBProj)),
                    kv_a_norm: operand(&stack_id, get(OperandRole::MlaKvANorm)),
                    out_proj: operand(&stack_id, get(OperandRole::MlaOutProj)),
                    // Present exactly when the surface declares the gate;
                    // an undeclared `g_proj` never reaches here (closure
                    // refuses it by name).
                    output_gate: m
                        .output_gate
                        .map(|_| operand(&stack_id, get(OperandRole::MlaOutputGate))),
                    kv_a_norm_eps: m.kv_a_norm_eps,
                }))
            } else {
                let a = attn.unwrap_or_else(|| {
                    panic!(
                        "layer {layer} ships softmax attention operands while the surface \
                         carries no attention group; closure should have refused this before \
                         the plan was built"
                    )
                });
                LayerAttention::Softmax(Box::new(AttentionOp {
                    num_q_heads: geometry.num_q_heads,
                    num_kv_heads: geometry.num_kv_heads,
                    head_dim: geometry.head_dim,
                    query_scale: a.query_scale,
                    score_scale: a.score_scale,
                    logit_softcapping: a.logit_softcapping,
                    // The graph carries no span exactly when it recorded
                    // a recurrence for this layer. Reaching here means the
                    // layer ships softmax operands anyway — the checkpoint
                    // contradicting itself, config against tensors — and
                    // the mirror of the panic above: an invariant the
                    // builder upholds, not a case to paper over with a
                    // default span.
                    span: policy.span.unwrap_or_else(|| {
                        panic!(
                            "layer {layer} ships softmax attention operands while the graph \
                             records a recurrence for it (no span); the checkpoint's \
                             layer_types and its tensors disagree"
                        )
                    }),
                    window: policy.window,
                    position: policy.position,
                    qk_norm,
                    parameter_free_qk_norm: a.parameter_free_qk_norm,
                    q: operand(&stack_id, get(OperandRole::AttnQ)),
                    k: operand(&stack_id, get(OperandRole::AttnK)),
                    // On a K≡V layer the value operand IS the key operand:
                    // the op reads one matrix twice, and says so.
                    v: operand(
                        &stack_id,
                        get(if policy.v_from_k {
                            OperandRole::AttnK
                        } else {
                            OperandRole::AttnV
                        }),
                    ),
                    v_from_k: policy.v_from_k,
                    o: operand(&stack_id, get(OperandRole::AttnO)),
                    // On a fused source the gate has NO operand of its
                    // own: it is the per-head second half of the query
                    // projection, so the op names `q_proj` and reads one
                    // matrix for both roles — the same "one matrix, two
                    // roles" statement `v_from_k` makes for K≡V layers.
                    output_gate: a.output_gate.map(|spec| GateOp {
                        spec,
                        projection: operand(
                            &stack_id,
                            get(match spec.source {
                                larql_models::config::GateSource::AttentionInput => {
                                    OperandRole::AttnOutputGate
                                }
                                larql_models::config::GateSource::FusedQueryProjection => {
                                    OperandRole::AttnQ
                                }
                            }),
                        ),
                    }),
                    // Closure held, so `Some(true)` means all four are here
                    // and anything else means none is.
                    q_bias: bias(OperandRole::AttnQBias),
                    k_bias: bias(OperandRole::AttnKBias),
                    v_bias: bias(OperandRole::AttnVBias),
                    o_bias: bias(OperandRole::AttnOBias),
                    sinks: a.sinks.map(|spec| SinkOp {
                        spec,
                        logits: operand(&stack_id, get(OperandRole::AttnSinks)),
                    }),
                }))
            },
            post_attention_norm,
            // Absent under post-norm placement: the FFN reads the raw
            // residual there, and `post_ffn_norm` carries the site that
            // does exist.
            pre_ffn_norm: pre_ffn_role.map(|role| trunk_norm(surface.norm.pre, role)),
            ffn: Some(ffn),
            post_ffn_norm,
            layer_scale,
            hyper_connection: hyper_connection_sites,
            attention_residual: attention_residual_sites,
            residual_scale: surface.residual_scale,
            operands_accounted: consumed,
            operands_present: consumed,
        });
    }

    let plan = ComponentOpPlan {
        component: component.id.clone(),
        residual_topology: surface.residual_topology,
        attention_residual_exit: attn_res_exit_tensors.map(|(object, norm, proj)| {
            AttentionResidualExitOp {
                norm: operand(&object, &norm),
                proj: operand(&object, &proj),
            }
        }),
        hyper_connection_head: hc_head_tensors.map(|(object, reduce_fn, base, scale)| {
            HyperConnectionHeadOp {
                reduce_fn: operand(&object, &reduce_fn),
                base: operand(&object, &base),
                scale: operand(&object, &scale),
            }
        }),
        embedding: embedding_tensor.map(|(object, tensor)| EmbeddingOp {
            table: operand(&object, &tensor),
            norm: surface.head.as_ref().and_then(|h| h.embedding_norm),
            scale: surface.head.as_ref().and_then(|h| h.embed_scale),
            vocab_size: vocab.unwrap_or(0),
            position: position_tensor.map(|(encoding, object, tensor)| PositionTableOp {
                encoding,
                table: operand(&object, &tensor),
            }),
        }),
        layers,
        final_norm: final_norm_tensor.map(|(object, weight, bias)| NormOp {
            bias: bias.map(|b| operand(&object, &b)),
            ..norm_op(surface.norm.final_norm, &object, &weight)
        }),
        output: head_tensor.map(|(object, tensor)| OutputOp {
            projection: operand(&object, &tensor),
            multiplier: surface.head.as_ref().and_then(|h| h.output_multiplier),
            softcapping: surface
                .head
                .as_ref()
                .and_then(|h| h.final_logit_softcapping),
        }),
    };
    plan
}
