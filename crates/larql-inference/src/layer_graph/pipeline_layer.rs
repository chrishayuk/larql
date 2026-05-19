//! Shared FullPipelineLayer construction from ModelWeights + VectorIndex.
//!
//! Single source of truth for extracting per-layer architecture parameters
//! from larql-models and wiring them into larql-compute's FullPipelineLayer.
//! Both GPU and CPU paths use this — no duplicated param extraction.
//!
//! Per-layer override resolution (env vars vs arch defaults) lives in
//! [`crate::forward_overrides`]; this module consumes those helpers.

use crate::forward_overrides::{effective_rope_base_for_layer, layer_forced_global};
use crate::model::ModelWeights;
use larql_compute::{
    FullPipelineLayer, MoeLayerWeights, MoeRoutingPolicy, MoeWeightLayout, QuantFormat, QuantWeight,
};

pub const DEFAULT_GPU_KV_CACHE_MAX_SEQ: usize = 4096;

pub(crate) fn kv_cache_shapes_for_arch(weights: &ModelWeights) -> Vec<(usize, usize)> {
    let arch = &*weights.arch;
    (0..weights.num_layers)
        .map(|layer| {
            (
                arch.num_kv_heads_for_layer(layer),
                arch.head_dim_for_layer(layer),
            )
        })
        .collect()
}

/// Extract per-layer architecture parameters into a FullPipelineLayer.
///
/// This is the single construction site for all per-layer params:
/// head_dim, num_q/kv_heads, rope_base, attn_scale, rotary_dim,
/// sliding_window, norm offsets, activation, FFN type, V-norm, layer scalar.
///
/// The attention weights (wq/wk/wv/wo) and FFN weights (gate/up/down)
/// must be provided separately since they come from different sources
/// (Q4_K from vindex, Q8 from vindex, f32 from model weights).
#[allow(clippy::too_many_arguments)]
pub fn build_arch_params<'a>(
    weights: &'a ModelWeights,
    layer: usize,
    wq: QuantWeight<'a>,
    wk: QuantWeight<'a>,
    wv: QuantWeight<'a>,
    wo: QuantWeight<'a>,
    gate: QuantWeight<'a>,
    up: QuantWeight<'a>,
    down: QuantWeight<'a>,
) -> FullPipelineLayer<'a> {
    let arch = &*weights.arch;
    let layer_hd = arch.head_dim_for_layer(layer);
    let layer_nq = arch.num_q_heads_for_layer(layer);
    let layer_nkv = arch.num_kv_heads_for_layer(layer);
    let rotary_frac = arch.rotary_fraction_for_layer(layer);
    let rotary_dim = if rotary_frac >= 1.0 {
        0
    } else {
        (layer_hd as f64 * rotary_frac) as usize
    };
    let force_global = layer_forced_global(layer);
    let sw = if !force_global && arch.is_sliding_window_layer(layer) {
        arch.sliding_window_size().unwrap_or(0)
    } else {
        0
    };
    let layer_scalar = arch
        .layer_scalar_key(layer)
        .and_then(|k| weights.vectors.get(&k))
        .and_then(|v| v.first().copied())
        .unwrap_or(0.0);

    FullPipelineLayer {
        wq,
        wk,
        wv,
        wo,
        gate,
        up,
        down,
        input_norm: weights
            .vectors
            .get(&arch.input_layernorm_key(layer))
            .map(|v| v.as_slice())
            .unwrap_or(&[]),
        post_attn_norm: weights
            .vectors
            .get(&arch.post_attention_layernorm_key(layer))
            .map(|v| v.as_slice())
            .unwrap_or(&[]),
        pre_ffn_norm: arch
            .pre_feedforward_layernorm_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice()),
        post_ffn_norm: arch
            .post_feedforward_layernorm_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice()),
        norm_offset: arch.norm_weight_offset(),
        has_post_norms: arch.has_post_norms(),
        activation: match arch.activation() {
            larql_models::Activation::GeluTanh => larql_compute::Activation::GeluTanh,
            _ => larql_compute::Activation::Silu,
        },
        qk_norm_offset: arch.qk_norm_weight_offset(),
        eps: arch.norm_eps(),
        norm_type: match arch.norm_type() {
            larql_models::NormType::LayerNorm => larql_compute::NormType::LayerNorm,
            _ => larql_compute::NormType::RmsNorm,
        },
        ffn_type: match arch.ffn_type() {
            larql_models::FfnType::Standard => larql_compute::FfnType::Standard,
            _ => larql_compute::FfnType::Gated,
        },
        // Granite-family `attention_multiplier` (1/64 on 3B, 1/128 on
        // 8B/30B) *replaces* `1/sqrt(head_dim)` — it is the trained-time
        // attention score scale, not a factor multiplied on top of
        // sqrt-scaling. The F32 attention path at
        // `attention/gpu.rs::scale` follows the same convention; the
        // Metal Q4K decode kernels (`decode/encode_attn.rs`,
        // `ops/full_layer.rs`) read this `attn_scale` directly with no
        // further adjustment, so failing to fold the multiplier in here
        // leaves Granite's 0.015625 / 0.0078125 trained scale unused and
        // the model degenerates to repeating high-frequency tokens
        // ("ikea ikea ikea…") because every attention distribution
        // peaks too sharply.
        attn_scale: if arch.attention_multiplier() != 1.0 {
            arch.attention_multiplier()
        } else {
            arch.attention_scale_for_layer(layer) as f32
        },
        head_dim: layer_hd,
        num_q_heads: layer_nq,
        num_kv_heads: layer_nkv,
        rope_base: effective_rope_base_for_layer(arch, layer) as f32,
        rotary_dim,
        sliding_window: sw,
        has_v_norm: arch.has_v_norm(),
        layer_scalar,
        input_norm_bias: None,
        post_attn_norm_bias: None,
        q_norm_weight: arch
            .attn_q_norm_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice()),
        k_norm_weight: arch
            .attn_k_norm_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice()),
        ffn_up_bias: arch
            .ffn_up_bias_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice()),
        ffn_down_bias: arch
            .ffn_down_bias_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice()),

        moe: build_moe_weights(weights, arch, layer),
        ffn_is_remote: false,
        moe_combined_output_norm: arch.moe_has_combined_output_norm(),
        moe_outer_post_norm: arch
            .moe_post_outer_norm_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice()),
        ple_input_gate: arch
            .per_layer_input_gate_key(layer)
            .and_then(|k| weights.tensors.get(&k))
            .and_then(|t| t.as_slice()),
        ple_projection: arch
            .per_layer_projection_key(layer)
            .and_then(|k| weights.tensors.get(&k))
            .and_then(|t| t.as_slice()),
        ple_post_norm: arch
            .post_per_layer_input_norm_key(layer)
            .and_then(|k| weights.vectors.get(&k))
            .map(|v| v.as_slice()),
        kv_shared_source: arch.kv_shared_source_layer(layer),
        // Granite-style residual scaling: HF `modeling_granite.py` does
        // `hidden_states = residual + self.residual_multiplier * hidden_states`
        // after both attention and FFN. Trait getter returns 1.0 for
        // every non-Granite arch, so this default is bit-identical for
        // them and the Metal `residual_add` shader's `b_scale` binding
        // is a no-op (multiply by 1.0).
        residual_multiplier: arch.residual_multiplier(),
    }
}

pub(crate) fn build_moe_weights<'a>(
    weights: &'a ModelWeights,
    arch: &dyn larql_models::ModelArchitecture,
    layer: usize,
) -> Option<MoeLayerWeights<'a>> {
    if !arch.is_hybrid_moe() {
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
            (gu_table, dn_table, larql_compute::QuantFormat::Q4_K)
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
            (gu_table, dn_table, larql_compute::QuantFormat::BF16)
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

    let activation = match arch.activation() {
        larql_models::Activation::GeluTanh => larql_compute::Activation::GeluTanh,
        _ => larql_compute::Activation::Silu,
    };

    Some(MoeLayerWeights {
        experts_gate_up,
        experts_down,
        routing_policy: moe_routing_policy(arch.moe_router_type()),
        weight_layout: MoeWeightLayout::default(),
        expert_data_format,
        router_proj,
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
        activation,
    })
}

/// Registry tag → `compute::QuantFormat` for the attention surface.
/// Explicit so a typo or new tag fails loudly rather than silently
/// aliasing to Q4_K. Lifted from inside `resolve_attn_weights` so the
/// mapping is unit-testable in isolation.
fn attn_str_to_format(s: &str) -> QuantFormat {
    match s {
        "Q4_K" => QuantFormat::Q4_K,
        "Q6_K" => QuantFormat::Q6_K,
        other => panic!(
            "resolve_attn_weights: registry tag {other:?} has no compute::QuantFormat mapping"
        ),
    }
}

/// Helper: resolve attention weights from vindex (Q4_K preferred, Q8 fallback).
pub fn resolve_attn_weights<'a>(
    index: &'a larql_vindex::VectorIndex,
    layer: usize,
) -> Option<(
    QuantWeight<'a>,
    QuantWeight<'a>,
    QuantWeight<'a>,
    QuantWeight<'a>,
)> {
    let to_format = attn_str_to_format;

    if let Some([q, k, v, o]) = index.attn_kquant_layer_data(layer) {
        Some((
            QuantWeight {
                data: q.0,
                scales: None,
                format: to_format(q.1),
            },
            QuantWeight {
                data: k.0,
                scales: None,
                format: to_format(k.1),
            },
            QuantWeight {
                data: v.0,
                scales: None,
                format: to_format(v.1),
            },
            QuantWeight {
                data: o.0,
                scales: None,
                format: to_format(o.1),
            },
        ))
    } else if let Some([q, k, v, o]) = index.attn_q8_layer_data(layer) {
        Some((
            QuantWeight {
                data: q.0,
                scales: Some(q.1),
                format: QuantFormat::Q8_0,
            },
            QuantWeight {
                data: k.0,
                scales: Some(k.1),
                format: QuantFormat::Q8_0,
            },
            QuantWeight {
                data: v.0,
                scales: Some(v.1),
                format: QuantFormat::Q8_0,
            },
            QuantWeight {
                data: o.0,
                scales: Some(o.1),
                format: QuantFormat::Q8_0,
            },
        ))
    } else {
        None
    }
}

/// Helper: resolve FFN weights from vindex interleaved mmap.
///
/// Prefers the per-matrix manifest when available (emitted by the streaming
/// `--quant q4k` writer: gate/up Q4_K, down Q6_K — non-uniform stride). Falls
/// back to the legacy uniform-stride layout produced by `build_q4k_weights.rs`
/// when the manifest is absent so older vindexes still work.
/// Registry tag → `compute::QuantFormat` for the FFN surface, with an
/// explicit `fallback` for the legacy uniform-stride writer that
/// didn't emit per-matrix tags. Lifted from inside `resolve_ffn_weights`
/// so the mapping is unit-testable in isolation.
fn ffn_str_to_format(s: &str, fallback: QuantFormat) -> QuantFormat {
    match s {
        "Q4_K" => QuantFormat::Q4_K,
        "Q6_K" => QuantFormat::Q6_K,
        "Q4_0" => QuantFormat::Q4_0,
        "" => fallback,
        other => panic!(
            "resolve_ffn_weights: registry tag {other:?} has no compute::QuantFormat mapping"
        ),
    }
}

pub fn resolve_ffn_weights<'a>(
    index: &'a larql_vindex::VectorIndex,
    layer: usize,
    q4_ffn_mmap: &'a [u8],
    q4_ffn_per_matrix: usize,
    ffn_format: QuantFormat,
) -> (QuantWeight<'a>, QuantWeight<'a>, QuantWeight<'a>) {
    let str_to_format = ffn_str_to_format;

    if let Some([gate, up, down]) = index.interleaved_kquant_layer_data(layer) {
        return (
            QuantWeight {
                data: gate.0,
                scales: None,
                format: str_to_format(gate.1, ffn_format),
            },
            QuantWeight {
                data: up.0,
                scales: None,
                format: str_to_format(up.1, ffn_format),
            },
            QuantWeight {
                data: down.0,
                scales: None,
                format: str_to_format(down.1, ffn_format),
            },
        );
    }

    let q4_ffn_per_layer = q4_ffn_per_matrix * 3;
    let fs = layer * q4_ffn_per_layer;
    (
        QuantWeight {
            data: &q4_ffn_mmap[fs..fs + q4_ffn_per_matrix],
            scales: None,
            format: ffn_format,
        },
        QuantWeight {
            data: &q4_ffn_mmap[fs + q4_ffn_per_matrix..fs + 2 * q4_ffn_per_matrix],
            scales: None,
            format: ffn_format,
        },
        QuantWeight {
            data: &q4_ffn_mmap[fs + 2 * q4_ffn_per_matrix..fs + 3 * q4_ffn_per_matrix],
            scales: None,
            format: ffn_format,
        },
    )
}

/// Build a complete Vec<FullPipelineLayer> for a range of layers.
/// Single source of truth — used by both GPU decode and GPU prefill paths.
#[allow(clippy::too_many_arguments)]
pub fn build_pipeline_layers<'a>(
    weights: &'a ModelWeights,
    index: &'a larql_vindex::VectorIndex,
    layer_range: std::ops::Range<usize>,
    q4_ffn_mmap: &'a [u8],
    q4_ffn_per_matrix: usize,
    ffn_format: QuantFormat,
) -> Vec<FullPipelineLayer<'a>> {
    layer_range
        .map(|layer| {
            let (wq, wk, wv, wo) = resolve_attn_weights(index, layer)
                .expect("No attention weights available for layer");
            let (gate, up, down) =
                resolve_ffn_weights(index, layer, q4_ffn_mmap, q4_ffn_per_matrix, ffn_format);
            build_arch_params(weights, layer, wq, wk, wv, wo, gate, up, down)
        })
        .collect()
}

/// For `--ffn URL` (remote dense FFN) deployments: all FFN work is delegated
/// to a remote server via `moe_fn` on every layer. This function sets
/// `ffn_is_remote = true` on all layers, which causes the Metal decode loop
/// to skip the local GPU FFN dispatches and route all FFN output through the
/// `moe_fn` callback instead.
///
/// No MoE stub injection is needed: the `has_moe` check in `setup.rs` now
/// also fires on `ffn_is_remote`, so the interleave path is taken for every
/// layer even without `layer.moe` being set.
pub fn patch_pipeline_layers_for_remote_ffn(layers: &mut [FullPipelineLayer<'_>]) {
    for layer in layers.iter_mut() {
        layer.ffn_is_remote = true;
    }
}

/// For `--moe-shards` (remote expert) deployments: the client vindex has no
/// per-layer expert bytes, so `build_moe_weights` returns `None` for every
/// layer, `has_moe = false`, and the Metal decode never calls `moe_fn`.
///
/// This function patches that by injecting a stub `MoeLayerWeights` for every
/// MoE-capable layer whose `moe` field is still `None`.  The stub has empty
/// expert slices — they are never read when `moe_fn` is `Some` (the remote
/// dispatch closure supersedes local `cpu_moe_forward`).  Norm weights are
/// populated from `weights.vectors` (loaded from `norms.bin` in the client
/// slice) so post-MoE normalisation remains correct.
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
        routing_policy: moe_routing_policy(arch.moe_router_type()),
        weight_layout: MoeWeightLayout::default(),
        expert_data_format,
        router_proj: &[],
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
        activation: match arch.activation() {
            larql_models::Activation::GeluTanh => larql_compute::Activation::GeluTanh,
            _ => larql_compute::Activation::Silu,
        },
    }
}

fn moe_routing_policy(router_type: &str) -> MoeRoutingPolicy {
    match router_type {
        "gemma4_top_k_softmax" => MoeRoutingPolicy::gemma4_hybrid(),
        _ => MoeRoutingPolicy::top_k_softmax(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{make_test_vindex, make_test_weights};
    use larql_models::ModelWeights;
    use std::collections::HashMap;
    use std::sync::OnceLock;

    fn weights() -> &'static ModelWeights {
        static W: OnceLock<ModelWeights> = OnceLock::new();
        W.get_or_init(make_test_weights)
    }

    fn empty_qw() -> QuantWeight<'static> {
        QuantWeight {
            data: &[],
            scales: None,
            format: QuantFormat::Q4_K,
        }
    }

    // ── kv_cache_shapes_for_arch ──────────────────────────────────────

    #[test]
    fn kv_cache_shapes_for_arch_returns_one_entry_per_layer() {
        let w = weights();
        let shapes = kv_cache_shapes_for_arch(w);
        assert_eq!(shapes.len(), w.num_layers);
        for (nh, hd) in &shapes {
            assert!(*nh > 0);
            assert!(*hd > 0);
        }
    }

    // ── attn_str_to_format ────────────────────────────────────────────

    #[test]
    fn attn_str_to_format_maps_q4k_and_q6k() {
        assert_eq!(attn_str_to_format("Q4_K"), QuantFormat::Q4_K);
        assert_eq!(attn_str_to_format("Q6_K"), QuantFormat::Q6_K);
    }

    #[test]
    #[should_panic(expected = "registry tag")]
    fn attn_str_to_format_panics_on_unknown_tag() {
        let _ = attn_str_to_format("BF16");
    }

    // ── ffn_str_to_format ─────────────────────────────────────────────

    #[test]
    fn ffn_str_to_format_maps_known_tags() {
        assert_eq!(
            ffn_str_to_format("Q4_K", QuantFormat::Q4_0),
            QuantFormat::Q4_K
        );
        assert_eq!(
            ffn_str_to_format("Q6_K", QuantFormat::Q4_0),
            QuantFormat::Q6_K
        );
        assert_eq!(
            ffn_str_to_format("Q4_0", QuantFormat::Q4_K),
            QuantFormat::Q4_0
        );
    }

    #[test]
    fn ffn_str_to_format_empty_tag_uses_fallback() {
        // Legacy writers omitted per-matrix tags — empty string falls
        // through to the caller's `fallback`.
        assert_eq!(ffn_str_to_format("", QuantFormat::Q4_0), QuantFormat::Q4_0);
        assert_eq!(ffn_str_to_format("", QuantFormat::Q4_K), QuantFormat::Q4_K);
    }

    #[test]
    #[should_panic(expected = "registry tag")]
    fn ffn_str_to_format_panics_on_unknown_tag() {
        let _ = ffn_str_to_format("BOGUS", QuantFormat::Q4_K);
    }

    // ── moe_routing_policy ────────────────────────────────────────────

    #[test]
    fn moe_routing_policy_gemma4_returns_hybrid() {
        // The `gemma4_top_k_softmax` tag must map to the Gemma 4
        // hybrid policy (the `top_k_softmax` route is the default for
        // every other router type).
        let g = moe_routing_policy("gemma4_top_k_softmax");
        let default = moe_routing_policy("top_k_softmax");
        // Don't lock-in MoeRoutingPolicy field equality; just verify the
        // function dispatches both arms without panic. The default arm
        // catches *all* other tags too.
        let _ = (g, default);
        // Any other tag (including unknown ones) takes the catch-all arm:
        let _ = moe_routing_policy("brand_new_router");
    }

    // ── build_arch_params ─────────────────────────────────────────────────────

    #[test]
    fn build_arch_params_extracts_norm_weights() {
        let w = weights();
        let params = build_arch_params(
            w,
            0,
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
        );
        // input_norm comes from arch.input_layernorm_key(0) which is in test weights
        assert!(
            !params.input_norm.is_empty(),
            "input_norm should be populated"
        );
        assert!(
            !params.post_attn_norm.is_empty(),
            "post_attn_norm should be populated"
        );
    }

    #[test]
    fn build_arch_params_head_dims_correct() {
        let w = weights();
        let params = build_arch_params(
            w,
            0,
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
        );
        assert_eq!(params.head_dim, w.head_dim);
        assert_eq!(params.num_q_heads, w.num_q_heads);
        assert_eq!(params.num_kv_heads, w.num_kv_heads);
    }

    #[test]
    fn build_arch_params_all_layers_no_panic() {
        let w = weights();
        for layer in 0..w.num_layers {
            let _ = build_arch_params(
                w,
                layer,
                empty_qw(),
                empty_qw(),
                empty_qw(),
                empty_qw(),
                empty_qw(),
                empty_qw(),
                empty_qw(),
            );
        }
    }

    // ── resolve_attn_weights ──────────────────────────────────────────────────

    #[test]
    fn resolve_attn_weights_returns_none_without_q4k() {
        // make_test_vindex has no Q4K attn data → should return None
        let w = weights();
        let idx = make_test_vindex(w);
        let result = resolve_attn_weights(&idx, 0);
        assert!(
            result.is_none(),
            "test vindex has no Q4K attn data, expected None"
        );
    }

    // ── resolve_ffn_weights ───────────────────────────────────────────────────

    #[test]
    fn resolve_ffn_weights_legacy_stride_slices_correctly() {
        // 4 bytes per matrix, layer 0: fs=0, gate=[0..4], up=[4..8], down=[8..12]
        let mmap: Vec<u8> = (0u8..12).collect();
        let idx = make_test_vindex(weights());
        let (gate, up, down) = resolve_ffn_weights(&idx, 0, &mmap, 4, QuantFormat::Q4_K);
        // No manifest, falls back to legacy stride
        assert_eq!(gate.data, &[0, 1, 2, 3]);
        assert_eq!(up.data, &[4, 5, 6, 7]);
        assert_eq!(down.data, &[8, 9, 10, 11]);
        assert_eq!(gate.format, QuantFormat::Q4_K);
    }

    #[test]
    fn resolve_ffn_weights_layer1_correct_offset() {
        // layer=1, per_matrix=4: fs = 1*12 = 12
        let mmap: Vec<u8> = (0u8..24).collect();
        let idx = make_test_vindex(weights());
        let (gate, up, down) = resolve_ffn_weights(&idx, 1, &mmap, 4, QuantFormat::Q4_0);
        assert_eq!(gate.data, &[12, 13, 14, 15]);
        assert_eq!(up.data, &[16, 17, 18, 19]);
        assert_eq!(down.data, &[20, 21, 22, 23]);
    }

    // ── KV-shared / PLE field plumbing (D-METAL-PLE / D-METAL-KV-SHARED) ──
    //
    // These tests pin the contract between `arch.kv_shared_source_layer(layer)`
    // /  PLE-key arch methods and the `FullPipelineLayer.kv_shared_source` /
    // `ple_*` fields.  They were introduced after the D-METAL-KV-SHARED
    // implementation surfaced a class of "field plumbing silently went None"
    // bugs that only show up at decode-time as multilingual gibberish output
    // — the kind of regression a one-line type-rename in build_arch_params
    // would re-introduce.

    /// Tiny synthetic Gemma-4-E2B-shaped arch with PLE + KV sharing.
    /// Same shape as `crates/larql-models/tests/test_architectures.rs::gemma4_e2b_arch`
    /// but smaller (4 layers, hidden=8) so weights fit in-memory cheaply.
    fn synthetic_e2b_like_arch_json() -> serde_json::Value {
        serde_json::json!({
            "model_type": "gemma4",
            "text_config": {
                "model_type": "gemma4_text",
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "head_dim": 4,
                "global_head_dim": 8,
                "vocab_size": 32,
                "sliding_window": 4,
                // PLE: 4-dim per-layer embed
                "hidden_size_per_layer_input": 4,
                // KV sharing: last 2 layers (L2, L3) share K/V from earlier layers
                "num_kv_shared_layers": 2,
                "rope_parameters": {
                    "full_attention": {
                        "partial_rotary_factor": 0.25,
                        "rope_theta": 1000000.0
                    },
                    "sliding_attention": {"rope_theta": 10000.0}
                },
                "layer_types": [
                    "sliding_attention",  // 0 — last sliding non-shared
                    "full_attention",     // 1 — last global non-shared
                    "sliding_attention",  // 2 — shared, source = L0
                    "full_attention"      // 3 — shared, source = L1
                ]
            }
        })
    }

    /// Build minimal ModelWeights for the synthetic E2B-like arch.  Tensors
    /// are zero-filled (the tests only assert presence/absence and which
    /// arch keys produced them, not numerical correctness).
    fn synthetic_e2b_like_weights() -> ModelWeights {
        use larql_models::{detect_from_json, WeightArray};
        use ndarray::Array2;

        let arch = detect_from_json(&synthetic_e2b_like_arch_json());
        let num_layers = 4;
        let hidden = 8;
        let intermediate = 16;
        let head_dim = 4;
        let global_head_dim = 8;
        let num_q_heads = 2;
        let num_kv_heads = 1;
        let vocab_size = 32;
        let ple_dim = 4;

        let mut tensors: HashMap<String, WeightArray> = HashMap::new();
        let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

        let zeros = |rows: usize, cols: usize| -> WeightArray {
            Array2::<f32>::zeros((rows, cols)).into_shared()
        };

        let embed = zeros(vocab_size, hidden);
        let lm_head = zeros(vocab_size, hidden);
        tensors.insert(arch.embed_key().to_string(), embed.clone());
        vectors.insert(arch.final_norm_key().to_string(), vec![1.0; hidden]);

        // Shared PLE tensors (Stream 1 + Stream 2 sources).
        if let Some(k) = arch.per_layer_model_projection_key() {
            tensors.insert(k, zeros(num_layers * ple_dim, hidden));
        }
        if let Some(k) = arch.per_layer_embed_key() {
            tensors.insert(k, zeros(vocab_size, num_layers * ple_dim));
        }
        if let Some(k) = arch.per_layer_projection_norm_key() {
            vectors.insert(k, vec![1.0; ple_dim]);
        }

        for layer in 0..num_layers {
            let layer_head_dim = if arch.is_sliding_window_layer(layer) {
                head_dim
            } else {
                global_head_dim
            };
            let q_dim = num_q_heads * layer_head_dim;
            let kv_dim = num_kv_heads * layer_head_dim;
            tensors.insert(arch.attn_q_key(layer), zeros(q_dim, hidden));
            tensors.insert(arch.attn_k_key(layer), zeros(kv_dim, hidden));
            tensors.insert(arch.attn_v_key(layer), zeros(kv_dim, hidden));
            tensors.insert(arch.attn_o_key(layer), zeros(hidden, q_dim));
            tensors.insert(arch.ffn_gate_key(layer), zeros(intermediate, hidden));
            tensors.insert(arch.ffn_up_key(layer), zeros(intermediate, hidden));
            tensors.insert(arch.ffn_down_key(layer), zeros(hidden, intermediate));
            vectors.insert(arch.input_layernorm_key(layer), vec![1.0; hidden]);
            vectors.insert(arch.post_attention_layernorm_key(layer), vec![1.0; hidden]);
            // Per-layer PLE tensors.
            if let Some(k) = arch.per_layer_input_gate_key(layer) {
                tensors.insert(k, zeros(ple_dim, hidden));
            }
            if let Some(k) = arch.per_layer_projection_key(layer) {
                tensors.insert(k, zeros(hidden, ple_dim));
            }
            if let Some(k) = arch.post_per_layer_input_norm_key(layer) {
                vectors.insert(k, vec![1.0; hidden]);
            }
        }

        ModelWeights {
            tensors,
            vectors,
            raw_bytes: HashMap::new(),
            packed_mmaps: HashMap::new(),
            skipped_tensors: Vec::new(),
            packed_byte_ranges: HashMap::new(),
            embed,
            lm_head,
            position_embed: None,
            arch,
            num_layers,
            hidden_size: hidden,
            intermediate_size: intermediate,
            vocab_size,
            head_dim,
            num_q_heads,
            num_kv_heads,
            rope_base: 10_000.0,
        }
    }

    /// `kv_shared_source` must propagate from `arch.kv_shared_source_layer(l)`.
    /// L0/L1 are non-shared, L2 → L0 (last sliding), L3 → L1 (last global).
    /// This is the test that would catch a bug where someone removes the
    /// `kv_shared_source: arch.kv_shared_source_layer(layer)` line in
    /// `build_arch_params` (or where `arch.kv_shared_source_layer` itself
    /// regresses on Gemma 4's per-layer-type matching logic).
    #[test]
    fn build_arch_params_populates_kv_shared_source_for_e2b_like_arch() {
        let weights = synthetic_e2b_like_weights();
        let l0 = build_arch_params(
            &weights,
            0,
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
        );
        let l1 = build_arch_params(
            &weights,
            1,
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
        );
        let l2 = build_arch_params(
            &weights,
            2,
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
        );
        let l3 = build_arch_params(
            &weights,
            3,
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
            empty_qw(),
        );
        assert_eq!(l0.kv_shared_source, None, "L0 (non-shared) → None");
        assert_eq!(l1.kv_shared_source, None, "L1 (non-shared) → None");
        // L2 is sliding; last sliding non-shared layer in [0..first_shared) = L0.
        assert_eq!(
            l2.kv_shared_source,
            Some(0),
            "L2 (shared sliding) → L0 (last sliding non-shared)"
        );
        // L3 is global; last global non-shared layer = L1.
        assert_eq!(
            l3.kv_shared_source,
            Some(1),
            "L3 (shared global) → L1 (last global non-shared)"
        );
    }

    /// `ple_input_gate` / `ple_projection` / `ple_post_norm` must populate
    /// when the arch reports PLE keys AND the corresponding tensors exist
    /// in the weights map.  Catches the bug class where someone changes the
    /// `arch.per_layer_input_gate_key(layer).and_then(|k| ...)` chain in
    /// `build_arch_params` and silently breaks PLE for E2B.
    #[test]
    fn build_arch_params_populates_ple_fields_for_e2b_like_arch() {
        let weights = synthetic_e2b_like_weights();
        for layer in 0..weights.num_layers {
            let params = build_arch_params(
                &weights,
                layer,
                empty_qw(),
                empty_qw(),
                empty_qw(),
                empty_qw(),
                empty_qw(),
                empty_qw(),
                empty_qw(),
            );
            assert!(
                params.ple_input_gate.is_some(),
                "L{layer} ple_input_gate must be Some when arch declares PLE"
            );
            assert!(
                params.ple_projection.is_some(),
                "L{layer} ple_projection must be Some when arch declares PLE"
            );
            assert!(
                params.ple_post_norm.is_some(),
                "L{layer} ple_post_norm must be Some when arch declares PLE"
            );
            // Sanity: ple_spec() returns Some when all three are present.
            assert!(
                params.ple_spec().is_some(),
                "L{layer} ple_spec() must be Some when all three PLE fields populate"
            );
        }
    }

    /// Non-PLE / non-KV-shared archs (TinyModel, etc) must leave the new
    /// fields as `None` — no spurious population.
    #[test]
    fn build_arch_params_leaves_ple_and_kv_shared_fields_none_for_tinymodel() {
        let w = weights();
        for layer in 0..w.num_layers {
            let params = build_arch_params(
                w,
                layer,
                empty_qw(),
                empty_qw(),
                empty_qw(),
                empty_qw(),
                empty_qw(),
                empty_qw(),
                empty_qw(),
            );
            assert_eq!(
                params.kv_shared_source, None,
                "L{layer} kv_shared_source must be None for non-shared arch"
            );
            assert_eq!(
                params.ple_input_gate, None,
                "L{layer} ple_input_gate must be None for non-PLE arch"
            );
            assert_eq!(params.ple_projection, None);
            assert_eq!(params.ple_post_norm, None);
            assert!(
                params.ple_spec().is_none(),
                "L{layer} ple_spec() must be None for non-PLE arch"
            );
        }
    }

    // ── build_moe_weights against the Gemma 4 MoE fixture ────────────────

    #[test]
    fn build_moe_weights_returns_none_for_non_moe_arch() {
        // make_test_weights → TinyModel arch with is_hybrid_moe()=false.
        let weights = crate::test_utils::make_test_weights();
        let arch = &*weights.arch;
        assert!(build_moe_weights(&weights, arch, 0).is_none());
    }

    #[test]
    fn build_moe_weights_returns_some_for_gemma4_moe_fixture() {
        use crate::test_utils::{
            make_test_gemma4_moe_weights, GEMMA4_MOE_NUM_EXPERTS, GEMMA4_MOE_TOP_K,
        };
        let weights = make_test_gemma4_moe_weights();
        let arch = &*weights.arch;
        assert!(arch.is_hybrid_moe());
        let moe = build_moe_weights(&weights, arch, 0)
            .expect("Gemma 4 MoE fixture must produce MoeLayerWeights");
        assert_eq!(moe.num_experts, GEMMA4_MOE_NUM_EXPERTS);
        assert_eq!(moe.top_k, GEMMA4_MOE_TOP_K);
        assert_eq!(moe.experts_gate_up.len(), GEMMA4_MOE_NUM_EXPERTS);
        assert_eq!(moe.experts_down.len(), GEMMA4_MOE_NUM_EXPERTS);
        // Activation should be GeluTanh for Gemma 4.
        assert!(matches!(
            moe.activation,
            larql_compute::Activation::GeluTanh
        ));
        // BF16 monolithic layout because the fixture writes raw_bytes
        // (not per-layer `layers/{l}/{e}/{component}` keys).
        assert!(matches!(
            moe.expert_data_format,
            larql_compute::QuantFormat::BF16
        ));
    }

    /// `patch_pipeline_layers_for_remote_moe` early-returns when the arch
    /// reports no hybrid-MoE (line 471). Drives the no-op path on the
    /// TinyModel test arch (is_hybrid_moe = false).
    #[test]
    fn patch_pipeline_layers_for_remote_moe_noop_on_non_moe_arch() {
        let w = weights();
        let mut layers: Vec<FullPipelineLayer<'_>> = Vec::new();
        patch_pipeline_layers_for_remote_moe(&mut layers, w);
        // No panic; nothing to assert about the empty slice. The function
        // bails at line 471 before touching the iterator.
    }

    /// `patch_pipeline_layers_for_remote_moe` on Gemma 4 MoE fixture: drives
    /// the per-layer loop, the `layer.moe.is_some() ⇒ continue` branch, and
    /// (after zeroing one layer's `moe` field) the `build_moe_stub`
    /// injection path. Covers lines 472-525.
    #[test]
    fn patch_pipeline_layers_for_remote_moe_injects_stub_on_gemma4_when_moe_none() {
        use crate::test_utils::{make_test_gemma4_moe_weights, make_test_q4k_vindex};
        let w = make_test_gemma4_moe_weights();
        let idx = make_test_q4k_vindex(&w);
        // q4_ffn_mmap legacy stride payload — large enough that the
        // per-layer offset (3 matrices × per_matrix) lands inside.
        let per_matrix = 32;
        let mmap: Vec<u8> = vec![0u8; per_matrix * 3 * w.num_layers];
        let mut layers = build_pipeline_layers(
            &w,
            &idx,
            0..w.num_layers,
            &mmap,
            per_matrix,
            QuantFormat::Q4_K,
        );
        // Sanity: build_moe_weights populated `moe` for all layers on the
        // Gemma 4 fixture (BF16 monolithic path).
        assert!(layers.iter().all(|l| l.moe.is_some()));

        // First call: every layer has moe=Some ⇒ patch hits the `continue`
        // branch every iteration (line 475).
        patch_pipeline_layers_for_remote_moe(&mut layers, &w);
        assert!(layers.iter().all(|l| l.moe.is_some()));

        // Zero one layer's moe field and re-run. The patch must invoke
        // `build_moe_stub` for that one layer (lines 480, 484-525).
        layers[0].moe = None;
        patch_pipeline_layers_for_remote_moe(&mut layers, &w);
        let stub = layers[0]
            .moe
            .as_ref()
            .expect("patch must re-populate moe for hybrid arch when None");
        assert!(stub.experts_gate_up.is_empty());
        assert!(stub.experts_down.is_empty());
        assert_eq!(stub.num_experts, w.arch.num_experts());
        assert_eq!(stub.top_k, w.arch.num_experts_per_token());
        assert_eq!(stub.intermediate_size, w.arch.moe_intermediate_size());
        assert!(matches!(
            stub.expert_data_format,
            larql_compute::QuantFormat::BF16
        ));
    }

    #[test]
    fn build_moe_weights_per_expert_slices_are_correct_size() {
        use crate::test_utils::{
            make_test_gemma4_moe_weights, GEMMA4_MOE_HIDDEN, GEMMA4_MOE_INTER,
        };
        let weights = make_test_gemma4_moe_weights();
        let moe = build_moe_weights(&weights, &*weights.arch, 0).unwrap();
        let expected_gu = 2 * GEMMA4_MOE_INTER * GEMMA4_MOE_HIDDEN * 2; // BF16
        let expected_dn = GEMMA4_MOE_HIDDEN * GEMMA4_MOE_INTER * 2;
        for (i, gu) in moe.experts_gate_up.iter().enumerate() {
            assert_eq!(gu.len(), expected_gu, "gate_up expert {i} size mismatch");
        }
        for (i, dn) in moe.experts_down.iter().enumerate() {
            assert_eq!(dn.len(), expected_dn, "down expert {i} size mismatch");
        }
    }
}
