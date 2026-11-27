//! The roles a layer's ops require.

use super::super::super::graph::{NormPlacement, OperandRole};
use larql_models::config::ExpertFormat;
use larql_models::config::MoeRouterKind;

#[allow(unused_imports)]
use super::*;

/// Roles every layer must supply, given the surface's ops.
pub(super) fn required_roles(ops: &LayerOps) -> Vec<OperandRole> {
    // A mixer-only layer's program shares nothing with the transformer
    // shape below — one pre-mixer norm, no attention wrap, no FFN — so
    // it returns its own complete set rather than threading exemptions
    // through every clause that follows.
    if ops.operator.is_mamba2() {
        let mut roles = vec![
            OperandRole::Mamba2PreMixerNorm,
            OperandRole::Mamba2InProj,
            OperandRole::Mamba2Conv1d,
            OperandRole::Mamba2ALog,
            OperandRole::Mamba2D,
            OperandRole::Mamba2DtBias,
            OperandRole::Mamba2OutProj,
        ];
        if let Some(mixer) = ops.mamba2 {
            if mixer.geometry.use_conv_bias {
                roles.push(OperandRole::Mamba2Conv1dBias);
            }
            if mixer.geometry.rms_norm {
                roles.push(OperandRole::Mamba2GatedNorm);
            }
        }
        return roles;
    }
    // A conv-QKV attention layer likewise: the hybrid lineage wraps it
    // in ONE pre-mixer norm (no attention wrap, no FFN), and its operand
    // set is its own — fused QKV, the conv over it, the output
    // projection. The declared bias flags decide the bias operands the
    // same way the mixer's do; the observed checkpoint declares all
    // three projections bias-free and the conv biased.
    if ops.operator.is_conv_qkv() {
        let mut roles = vec![
            OperandRole::Mamba2PreMixerNorm,
            OperandRole::ConvQkvInProj,
            OperandRole::ConvQkvConv1d,
            OperandRole::ConvQkvOutProj,
        ];
        if let Some(attn) = ops.conv_qkv {
            // The conv-bias switch is the mixer's `use_conv_bias` — one
            // flag governs both block kinds' convs in this lineage.
            if ops.mamba2.is_some_and(|m| m.geometry.use_conv_bias) {
                roles.push(OperandRole::ConvQkvConv1dBias);
            }
            // qkv_bias / out_bias operands have no roles yet: the
            // observed checkpoint declares both false, and a role must
            // be judged from a real instance, not invented ahead of one.
            let _ = attn;
        }
        return roles;
    }
    // The attention block's two trunk norms, required per placement. A
    // post-norm stack has no pre-attention norm to require, and
    // requiring one would report a missing operand for a tensor the
    // checkpoint correctly never shipped.
    let mut roles = if ops.placement == NormPlacement::PostOnly {
        vec![OperandRole::PostAttentionNorm]
    } else {
        vec![
            OperandRole::PreAttentionNorm,
            OperandRole::PostAttentionNorm,
        ]
    };
    // Two sites per hyper-connected layer, three operands each, on EVERY
    // transformer layer of the component — DeepSeek-V4 and GLM-5.3-Flash
    // both carry all six on all 43 / 45 layers. A layer missing one is
    // not a partially hyper-connected layer; it is a bundle the traversal
    // cannot reduce or expand at that site.
    if ops.hyper_connection {
        roles.extend([
            OperandRole::HcAttnMixFn,
            OperandRole::HcAttnBase,
            OperandRole::HcAttnScale,
            OperandRole::HcFfnMixFn,
            OperandRole::HcFfnBase,
            OperandRole::HcFfnScale,
        ]);
    }
    // Two sites per attention-residual layer, a norm and a projection
    // each, on EVERY transformer layer of the component — K3 carries all
    // four on all 93. A layer missing one is not a partially
    // attention-residual layer; it is a site whose score vector has one
    // factor, and there is no judged form for that.
    if ops.attention_residual {
        roles.extend([
            OperandRole::AttnResAttentionNorm,
            OperandRole::AttnResAttentionProj,
            OperandRole::AttnResMlpNorm,
            OperandRole::AttnResMlpProj,
        ]);
    }
    if ops.operator.is_kda() {
        // Fifteen operands, and all fifteen are required: a KDA layer
        // missing one is not a partially-specified attention layer, it is
        // an operator that cannot run. None of the softmax roles apply —
        // the recurrence retains no per-position key or value.
        roles.extend([
            OperandRole::KdaQProj,
            OperandRole::KdaKProj,
            OperandRole::KdaVProj,
            OperandRole::KdaQConv1d,
            OperandRole::KdaKConv1d,
            OperandRole::KdaVConv1d,
            OperandRole::KdaFAProj,
            OperandRole::KdaFBProj,
            OperandRole::KdaBProj,
            OperandRole::KdaALog,
            OperandRole::KdaDtBias,
            OperandRole::KdaONorm,
            OperandRole::KdaOutProj,
        ]);
        // The output gate's operands follow its DECLARED form, so a layer
        // shipping the other form reports the declared one missing (and
        // the shipped one as implying an absent op — see `absent_op`).
        if ops.kda_full_rank_gate {
            roles.push(OperandRole::KdaGProj);
        } else {
            roles.extend([OperandRole::KdaGAProj, OperandRole::KdaGBProj]);
        }
    } else if ops.operator.is_gated_delta() {
        // A recurrence has no query, key, value or output projection —
        // demanding them made all 48 of Qwen3.8's linear layers report
        // four missing operands each for tensors that correctly do not
        // exist. Its nine operands are required instead, so the layer is
        // still fully pinned rather than merely exempted.
        roles.extend([
            OperandRole::LinearAttnInProjQkv,
            OperandRole::LinearAttnInProjA,
            OperandRole::LinearAttnInProjB,
            OperandRole::LinearAttnInProjZ,
            OperandRole::LinearAttnConv1d,
            OperandRole::LinearAttnALog,
            OperandRole::LinearAttnDtBias,
            OperandRole::LinearAttnNorm,
            OperandRole::LinearAttnOutProj,
        ]);
    } else if ops.operator.is_mla() {
        // No K/V projection exists to require — the compressed latent and
        // its decompression are the only KV path, so demanding AttnK/AttnV
        // would report two missing operands per layer for tensors the
        // checkpoint never shipped, the same shape GatedDelta's roles fix
        // for its own operands above.
        roles.extend([
            OperandRole::MlaKvAProj,
            OperandRole::MlaKvBProj,
            OperandRole::MlaKvANorm,
            OperandRole::MlaOutProj,
        ]);
        // The query's operands follow the DECLARED form, so a declared
        // factorisation missing any member names that member rather
        // than reporting a `q_proj` the checkpoint never shipped.
        if ops.mla_q_lora {
            roles.extend([
                OperandRole::MlaQAProj,
                OperandRole::MlaQANorm,
                OperandRole::MlaQBProj,
            ]);
        } else {
            roles.push(OperandRole::MlaQProj);
        }
        if ops.mla_output_gate {
            roles.push(OperandRole::MlaOutputGate);
        }
    } else {
        roles.extend([OperandRole::AttnQ, OperandRole::AttnK, OperandRole::AttnO]);
        if !ops.v_from_k {
            roles.push(OperandRole::AttnV);
        }
    }
    match ops.placement {
        NormPlacement::PrePost => {
            roles.push(OperandRole::PreFfnNorm);
            roles.push(OperandRole::PostFfnNorm);
        }
        // The FFN reads the raw residual; only its output is normed.
        NormPlacement::PostOnly => roles.push(OperandRole::PostFfnNorm),
        NormPlacement::PreOnly | NormPlacement::PreMixer => {}
    }
    if ops.output_gate {
        roles.push(OperandRole::AttnOutputGate);
    }
    if ops.attention_bias || ops.qkv_bias {
        roles.extend([
            OperandRole::AttnQBias,
            OperandRole::AttnKBias,
            OperandRole::AttnVBias,
        ]);
    }
    if ops.attention_bias {
        roles.push(OperandRole::AttnOBias);
    }
    if ops.sinks {
        roles.push(OperandRole::AttnSinks);
    }
    if ops.routed {
        if let Some(moe) = ops.moe {
            roles.push(OperandRole::MoeRouterWeight);
            if moe.expert_format == ExpertFormat::PerExpert {
                // No fused bank tensor exists to bind — the checkpoint
                // ships one gate/up/down triple PER EXPERT, so closure
                // requires the complete indexed set, not one flat role.
                // `absent_op` is what turns a stray expert beyond
                // `moe.experts` into a defect; this only states what MUST
                // be present.
                roles.extend((0..moe.experts as u16).flat_map(|expert| {
                    [
                        OperandRole::PerExpertGate(expert),
                        OperandRole::PerExpertUp(expert),
                        OperandRole::PerExpertDown(expert),
                    ]
                }));
            } else {
                roles.push(OperandRole::ExpertGateUp);
                roles.push(OperandRole::ExpertDown);
            }
            if moe.router_bias {
                roles.push(OperandRole::MoeRouterBias);
            }
            if moe.expert_format.has_split_scale_streams() {
                roles.push(OperandRole::ExpertGateUpScales);
                roles.push(OperandRole::ExpertDownScales);
            }
            // The latent wrapper, required by the DECLARATION and never
            // by the presence of its own tensors. Down and up always,
            // because a bottleneck the experts run behind has to be
            // entered and left; the norm only when the form carries one,
            // which is why the form nests it. `absent_op` refuses all
            // three where no latent form is declared, so the rule holds
            // from both sides.
            if let Some(latent) = moe.latent {
                roles.push(OperandRole::MoeLatentDownProj);
                roles.push(OperandRole::MoeLatentUpProj);
                if latent.norm.is_some() {
                    roles.push(OperandRole::MoeLatentNorm);
                }
            }
            // Gemma 4's router conditions its input and its selected
            // weights with two learned scales; the kind implies both.
            if moe.router_kind == MoeRouterKind::TopKRenormScaled {
                roles.push(OperandRole::MoeRouterScale);
                roles.push(OperandRole::MoeRouterPerExpertScale);
            }
            // Always-active alongside the routed selection — required
            // whenever the judgment declares one, on every routed layer
            // (Kimi/DeepSeek run it beside the routed block, never as a
            // Gemma-4-style hybrid dense branch).
            if moe.shared_experts > 0 {
                roles.extend([
                    OperandRole::SharedExpertGate,
                    OperandRole::SharedExpertUp,
                    OperandRole::SharedExpertDown,
                ]);
                // Paired with the judgment, both ways: a declared gate
                // must find its operand, and the `absent_op` arm below
                // refuses the operand where no gate is declared.
                if moe.shared_expert_gate.is_some() {
                    roles.push(OperandRole::SharedExpertBranchGate);
                }
            }
        }
    }
    if !ops.routed || ops.hybrid {
        roles.push(OperandRole::FfnUp);
        roles.push(OperandRole::FfnDown);
        if ops.gated_ffn {
            roles.push(OperandRole::FfnGate);
        }
        if ops.ffn_bias {
            roles.push(OperandRole::FfnUpBias);
            roles.push(OperandRole::FfnDownBias);
        }
    }
    if ops.hybrid {
        roles.extend([
            OperandRole::PreExpertsNorm,
            OperandRole::PostDenseFfnNorm,
            OperandRole::PostExpertsNorm,
        ]);
    }
    // A declared norm bias sits beside EVERY trunk norm weight the layer
    // requires — derived from the weights required above, so the two
    // sets cannot disagree about which sites exist.
    if ops.norm_bias {
        let biases: Vec<OperandRole> = roles.iter().filter_map(|r| norm_bias_role(*r)).collect();
        roles.extend(biases);
    }
    roles
}

/// The bias role paired with a trunk norm weight role, if it has one.
pub(super) fn norm_bias_role(weight: OperandRole) -> Option<OperandRole> {
    match weight {
        OperandRole::PreAttentionNorm => Some(OperandRole::PreAttentionNormBias),
        OperandRole::PostAttentionNorm => Some(OperandRole::PostAttentionNormBias),
        OperandRole::PreFfnNorm => Some(OperandRole::PreFfnNormBias),
        OperandRole::PostFfnNorm => Some(OperandRole::PostFfnNormBias),
        _ => None,
    }
}
