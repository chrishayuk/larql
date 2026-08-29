//! MoE construction for pipeline layers: resolving one layer's
//! `MoeLayerWeights` from a loaded model — expert byte tables, router
//! tensors, bias tables and the typed gate/routing rules — plus the
//! remote-MoE stub patching used by `--moe-shards` deployments.
//!
//! Split from `pipeline_layer` (file-size cap): everything here is the
//! MoE half of layer construction; the attention/FFN resolvers stay in
//! the parent.

use super::*;

pub fn build_moe_weights<'a>(
    weights: &'a ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> Option<MoeLayerWeights<'a>> {
    // Pure MoE (GraniteMoE, OLMoE) builds identically to hybrid — the expert
    // store, router and per-expert byte tables are the same. Hybrid differs
    // only in having a parallel dense slab, which lives outside this struct.
    if !(arch.is_moe() || arch.is_hybrid_moe()) {
        return None;
    }
    let router_key = arch.moe_router_key(layer)?;
    let router_proj = weights.vectors.get(&router_key)?.as_slice();

    // Build per-expert byte tables. Per-layer Q4_K reads each expert from
    // its own offset-table entry; legacy BF16 slices the monolith by stride.
    let num_experts = arch.num_experts();
    let moe_inter = arch.moe_intermediate_size();
    let hidden = weights.hidden_size;
    let (experts_gate_up, experts_down, expert_data_format): (Vec<&[u8]>, Vec<&[u8]>, _) =
        if weights.has_per_layer_ffn() {
            let mut gu_table = Vec::with_capacity(num_experts);
            let mut dn_table = Vec::with_capacity(num_experts);
            for e in 0..num_experts {
                let (gu, dn) = weights.get_layer_entry_bytes(layer, e)?;
                gu_table.push(gu);
                dn_table.push(dn);
            }
            // The layer file's own header is the format authority — a
            // Q6_K store (MXFP4-transcoded experts, GPT-OSS) decoded as
            // Q4_K is plausible-looking garbage. Absent tag = a vindex
            // loaded by a pre-format-threading loader, which only ever
            // wrote Q4_K. An unknown tag is a defect, not a fallback.
            let format = match weights.per_layer_ffn_format_tag(layer) {
                Some(tag) => QuantFormat::from_registry_tag(tag).unwrap_or_else(|| {
                    panic!(
                        "layer {layer}: per-layer FFN store declares format \
                         {tag:?} which compute has no decoder for"
                    )
                }),
                None => crate::QuantFormat::Q4_K,
            };
            (gu_table, dn_table, format)
        } else {
            // Legacy BF16 monolithic blob: split into per-expert strides.
            let gate_up_key = arch.packed_experts_gate_up_key(layer)?;
            let down_key = arch.packed_experts_down_key(layer)?;
            let gu_all = weights.get_packed_bytes(&gate_up_key)?;
            let dn_all = weights.get_packed_bytes(&down_key)?;
            let gu_stride = 2 * moe_inter * hidden * 2; // BF16 = 2 bytes
            let dn_stride = hidden * moe_inter * 2;
            let gu_table: Vec<&[u8]> = (0..num_experts)
                .map(|e| &gu_all[e * gu_stride..(e + 1) * gu_stride])
                .collect();
            let dn_table: Vec<&[u8]> = (0..num_experts)
                .map(|e| &dn_all[e * dn_stride..(e + 1) * dn_stride])
                .collect();
            (gu_table, dn_table, crate::QuantFormat::BF16)
        };

    let router_scale = arch
        .moe_router_scale_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let router_per_expert_scale = arch
        .moe_router_per_expert_scale_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let pre_experts_norm = arch
        .moe_pre_experts_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let post_ffn1_norm = arch
        .moe_post_ffn1_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let post_experts_norm = arch
        .moe_post_experts_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let router_norm = arch
        .moe_router_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .map(|v| v.as_slice())
        .unwrap_or(&[]);
    let router_norm_parameter_free = arch.moe_router_norm_parameter_free();
    let router_input_scalar = arch.moe_router_input_scalar().unwrap_or(1.0);

    // Bias tables. Resolution is by the arch's own keys, so an architecture
    // without them (Gemma 4, OLMoE) gets empty slices — the pre-bias
    // behaviour, bit for bit. An architecture that DECLARES them served
    // from a vindex that lacks them (extracted before the writer stored
    // biases) is a wrong-but-plausible forward: warn once per layer 0.
    let resolve_bias = |key: Option<String>| -> &'a [f32] {
        key.and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    };
    let router_bias = resolve_bias(arch.moe_router_bias_key(layer));
    let experts_gate_up_bias = resolve_bias(arch.packed_gate_up_bias_key(layer));
    let experts_down_bias = resolve_bias(arch.packed_down_bias_key(layer));
    if layer == 0
        && arch.moe_router_bias_key(0).is_some()
        && router_bias.is_empty()
        && weights.has_per_layer_ffn()
    {
        eprintln!(
            "warning: {} declares MoE biases but this vindex carries none — \
             extracted before biases were stored; re-extract for a faithful forward",
            arch.family()
        );
    }

    Some(MoeLayerWeights {
        experts_gate_up,
        experts_down,
        routing_policy: moe_routing_policy(arch.moe_router_kind()),
        weight_layout: MoeWeightLayout::default(),
        // Both statements are about the VINDEX2/legacy store this function
        // reads, not about the architecture. That store is written by an
        // extraction path which de-interleaves the fused rows and keeps
        // k-quant scales inside the blocks — including for GPT-OSS, whose
        // MXFP4 checkpoint is transcoded on the way in. A bank that kept
        // the checkpoint's own arrangement arrives through the container
        // route instead, which reads both facts off the region schema.
        expert_scales: MoeExpertScales::Inline,
        fused_row_layout: MoeFusedRowLayout::ContiguousHalves,
        expert_data_format,
        router_proj,
        router_bias,
        experts_gate_up_bias,
        experts_down_bias,
        router_scale,
        router_per_expert_scale,
        router_norm,
        router_norm_parameter_free,
        router_input_scalar,
        pre_experts_norm,
        post_ffn1_norm,
        post_experts_norm,
        num_experts: arch.num_experts(),
        top_k: arch.num_experts_per_token(),
        intermediate_size: arch.moe_intermediate_size(),
        gate_rule: crate::MoeGateRule::from_arch(arch.expert_gate_policy(), arch.activation()),
    })
}

pub fn patch_pipeline_layers_for_remote_moe<'a>(
    layers: &mut [FullPipelineLayer<'a>],
    weights: &'a ModelWeights,
) {
    let arch = &*weights.arch;
    if !arch.is_hybrid_moe() {
        return;
    }
    for (i, layer) in layers.iter_mut().enumerate() {
        if layer.moe.is_some() {
            continue;
        }
        if arch.moe_router_key(i).is_none() {
            continue;
        }
        layer.moe = Some(build_moe_stub(weights, arch, i));
    }
}

fn build_moe_stub<'a>(
    weights: &'a ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> MoeLayerWeights<'a> {
    let sl = |k: Option<String>| -> &'a [f32] {
        k.and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    };
    // expert_data_format is never read when moe_fn fires (remote path); match
    // what build_moe_weights would use so any fallback cpu_moe_forward still
    // decodes correctly if it ever runs.
    let expert_data_format = if weights.has_per_layer_ffn() {
        QuantFormat::Q4_K
    } else {
        QuantFormat::BF16
    };
    MoeLayerWeights {
        experts_gate_up: vec![],
        experts_down: vec![],
        routing_policy: moe_routing_policy(arch.moe_router_kind()),
        weight_layout: MoeWeightLayout::default(),
        // Both statements are about the VINDEX2/legacy store this function
        // reads, not about the architecture. That store is written by an
        // extraction path which de-interleaves the fused rows and keeps
        // k-quant scales inside the blocks — including for GPT-OSS, whose
        // MXFP4 checkpoint is transcoded on the way in. A bank that kept
        // the checkpoint's own arrangement arrives through the container
        // route instead, which reads both facts off the region schema.
        expert_scales: MoeExpertScales::Inline,
        fused_row_layout: MoeFusedRowLayout::ContiguousHalves,
        expert_data_format,
        router_proj: &[],
        router_bias: sl(arch.moe_router_bias_key(layer)),
        experts_gate_up_bias: sl(arch.packed_gate_up_bias_key(layer)),
        experts_down_bias: sl(arch.packed_down_bias_key(layer)),
        router_scale: sl(arch.moe_router_scale_key(layer)),
        router_per_expert_scale: sl(arch.moe_router_per_expert_scale_key(layer)),
        router_norm: sl(arch.moe_router_norm_key(layer)),
        router_norm_parameter_free: arch.moe_router_norm_parameter_free(),
        router_input_scalar: arch.moe_router_input_scalar().unwrap_or(1.0),
        pre_experts_norm: sl(arch.moe_pre_experts_norm_key(layer)),
        post_ffn1_norm: sl(arch.moe_post_ffn1_norm_key(layer)),
        post_experts_norm: sl(arch.moe_post_experts_norm_key(layer)),
        num_experts: arch.num_experts(),
        top_k: arch.num_experts_per_token(),
        intermediate_size: arch.moe_intermediate_size(),
        gate_rule: crate::MoeGateRule::from_arch(arch.expert_gate_policy(), arch.activation()),
    }
}

/// Map an architecture's routing rule onto the compute-side policy.
///
/// **Exhaustive on purpose.** This was a `match` over raw strings with a
/// `_ =>` default, so `gpt_oss`'s router — which selects top-k *then*
/// softmaxes over just those — silently took the ordinary policy. A new
/// variant now fails to compile here instead of quietly computing the wrong
/// expert weights. See `docs/k3-funnel.md` §4.7.8 for the same shape three
/// times over.
pub(crate) fn moe_routing_policy(kind: larql_models::MoeRouterKind) -> MoeRoutingPolicy {
    match kind {
        larql_models::MoeRouterKind::Gemma4Hybrid => MoeRoutingPolicy::gemma4_hybrid(),
        larql_models::MoeRouterKind::TopKSoftmax => MoeRoutingPolicy::top_k_softmax(),
        // Selected-then-normalised: the weights sum to 1 over the chosen
        // experts. Distinct from `top_k_softmax`, which does not.
        larql_models::MoeRouterKind::TopKThenSoftmax => MoeRoutingPolicy::top_k_then_softmax(),
        // Per-expert sigmoid gating. No policy here computes it, and every
        // policy that does exist normalises across experts in a way
        // sigmoid does not — so any of them would produce plausible,
        // wrong expert weights, which is the exact failure this match was
        // made exhaustive to prevent.
        //
        // Refusing loudly is the considered answer, not an oversight: the
        // families that declare it (Kimi Linear, GLM-5.3-Flash,
        // Inkling-Small) are not servable by this path at all yet.
        larql_models::MoeRouterKind::Sigmoid => unimplemented!(
            "sigmoid expert routing is represented but not executable: no routing policy \
             implements per-expert sigmoid gating"
        ),
    }
}
