//! Shared FullPipelineLayer construction from ModelWeights + VectorIndex.
//!
//! Single source of truth for extracting per-layer architecture parameters
//! from larql-models and wiring them into larql-compute's FullPipelineLayer.
//! Both GPU and CPU paths use this — no duplicated param extraction.

use crate::model::ModelWeights;
use larql_compute::{FullPipelineLayer, MoeLayerWeights, QuantFormat, QuantWeight};

pub const DEFAULT_GPU_KV_CACHE_MAX_SEQ: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttentionGeometry {
    pub q_dim: usize,
    pub kv_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rope_base: f32,
}

pub fn attention_geometry_for_arch_layer(
    weights: &ModelWeights,
    layer: usize,
) -> AttentionGeometry {
    let arch = &*weights.arch;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_q_heads = arch.num_q_heads_for_layer(layer);
    let num_kv_heads = arch.num_kv_heads_for_layer(layer);
    AttentionGeometry {
        q_dim: num_q_heads * head_dim,
        kv_dim: num_kv_heads * head_dim,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rope_base: arch.rope_base_for_layer(layer) as f32,
    }
}

pub(crate) fn attention_geometry_for_pipeline_layer(
    layer: &FullPipelineLayer<'_>,
) -> AttentionGeometry {
    AttentionGeometry {
        q_dim: layer.num_q_heads * layer.head_dim,
        kv_dim: layer.num_kv_heads * layer.head_dim,
        num_q_heads: layer.num_q_heads,
        num_kv_heads: layer.num_kv_heads,
        head_dim: layer.head_dim,
        rope_base: layer.rope_base,
    }
}

pub fn kv_cache_shapes_for_arch(weights: &ModelWeights) -> Vec<(usize, usize)> {
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
    let sw = if arch.is_sliding_window_layer(layer) {
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
        attn_scale: arch.attention_scale_for_layer(layer) as f32,
        head_dim: layer_hd,
        num_q_heads: layer_nq,
        num_kv_heads: layer_nkv,
        rope_base: arch.rope_base_for_layer(layer) as f32,
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
    // Registry tag → compute::QuantFormat. Explicit so a typo or new
    // tag fails loudly rather than silently aliasing to Q4_K.
    fn to_format(s: &str) -> QuantFormat {
        match s {
            "Q4_K" => QuantFormat::Q4_K,
            "Q6_K" => QuantFormat::Q6_K,
            other => panic!(
                "resolve_attn_weights: registry tag {other:?} has no compute::QuantFormat mapping"
            ),
        }
    }

    if let Some([q, k, v, o]) = index.attn_q4k_layer_data(layer) {
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
pub fn resolve_ffn_weights<'a>(
    index: &'a larql_vindex::VectorIndex,
    layer: usize,
    q4_ffn_mmap: &'a [u8],
    q4_ffn_per_matrix: usize,
    ffn_format: QuantFormat,
) -> (QuantWeight<'a>, QuantWeight<'a>, QuantWeight<'a>) {
    // Registry tag → compute::QuantFormat. The fallback exists for the
    // legacy uniform-stride path (`build_q4k_weights.rs` writer didn't
    // emit per-matrix tags); pass an explicit fallback rather than
    // silently aliasing unknown tags to Q4_K.
    fn str_to_format(s: &str, fallback: QuantFormat) -> QuantFormat {
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

    if let Some([gate, up, down]) = index.interleaved_q4k_layer_data(layer) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::test_utils::{make_test_vindex, make_test_weights};
    use larql_models::ModelWeights;
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
}
