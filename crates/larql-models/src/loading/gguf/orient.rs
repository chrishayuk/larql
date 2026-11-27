//! Tensor orientation helpers — orient_in_place, orient_ffn_tensors,
//! orient_attention_tensors, split_fused_qkv, orient_embedding.

use std::collections::HashMap;

pub(super) fn orient_embedding(
    embed: crate::WeightArray,
    hidden_size: usize,
    vocab_size: Option<usize>,
) -> crate::WeightArray {
    let shape = embed.shape();
    let rows = shape[0];
    let cols = shape[1];

    if cols == hidden_size || vocab_size.is_some_and(|vocab| rows == vocab) {
        return embed;
    }
    if rows == hidden_size || vocab_size.is_some_and(|vocab| cols == vocab) {
        let mut out = ndarray::Array2::<f32>::zeros((cols, rows));
        out.assign(&embed.t());
        return out.into_shared();
    }

    embed
}

/// Walk per-layer FFN tensors and ensure they're in canonical orientation.
///
/// Canonical (Llama / nn.Linear convention):
/// - gate / up:  shape `(intermediate, hidden)`
/// - down:       shape `(hidden, intermediate)`
///
/// Some GGUF converters (notably non-standard GPT-2 builds where Conv1D
/// weights weren't transposed) store FFN weights in the inverse layout.
/// If a tensor's loaded shape matches the inverse of the canonical
/// orientation — and the two dimensions differ so orientation is
/// unambiguous — transpose it. Otherwise leave it untouched.
///
/// Driven entirely by `ModelArchitecture` keys and `ModelConfig` dimensions
/// — no family-specific branching.
pub(super) fn orient_ffn_tensors(
    tensors: &mut HashMap<String, crate::WeightArray>,
    arch: &dyn crate::config::ModelArchitecture,
) {
    let cfg = arch.config();
    let hidden = cfg.hidden_size;
    let dense_inter = cfg.intermediate_size;
    if cfg.num_layers == 0 || hidden == 0 {
        return;
    }

    let moe_inter = if arch.is_moe() || arch.is_hybrid_moe() {
        let m = arch.moe_intermediate_size();
        (m > 0).then_some(m)
    } else {
        None
    };
    let n_experts = if moe_inter.is_some() {
        arch.num_experts()
    } else {
        0
    };

    for layer in 0..cfg.num_layers {
        // Dense FFN tensors
        if dense_inter > 0 {
            orient_in_place(tensors, &arch.ffn_gate_key(layer), dense_inter, hidden);
            orient_in_place(tensors, &arch.ffn_up_key(layer), dense_inter, hidden);
            orient_in_place(tensors, &arch.ffn_down_key(layer), hidden, dense_inter);
        }

        // The shared branch is sized by the architecture's own answer,
        // not by the dense width. The two coincide on Qwen1.5-MoE
        // (5632 either way) and nowhere else: DeepSeek's branch is
        // `moe_intermediate_size * n_shared_experts` against a much wider
        // dense FFN, and Qwen3.5-MoE has no dense width at all — under
        // the old rule its shared expert was left unoriented entirely.
        if let Some(shared_inter) = arch.shared_expert_intermediate_size() {
            if let Some(key) = arch.shared_expert_gate_key(layer) {
                orient_in_place(tensors, &key, shared_inter, hidden);
            }
            if let Some(key) = arch.shared_expert_up_key(layer) {
                orient_in_place(tensors, &key, shared_inter, hidden);
            }
            if let Some(key) = arch.shared_expert_down_key(layer) {
                orient_in_place(tensors, &key, hidden, shared_inter);
            }
            // One logit per token; the row count is the gate's own, not
            // the branch's.
            if let Some(key) = arch.shared_expert_branch_gate_key(layer) {
                orient_in_place(tensors, &key, 1, hidden);
            }
        }

        // Per-expert MoE FFN tensors use the per-expert intermediate dim.
        if let Some(mf) = moe_inter {
            for expert in 0..n_experts {
                if let Some(key) = arch.expert_ffn_gate_key(layer, expert) {
                    orient_in_place(tensors, &key, mf, hidden);
                }
                if let Some(key) = arch.expert_ffn_up_key(layer, expert) {
                    orient_in_place(tensors, &key, mf, hidden);
                }
                if let Some(key) = arch.expert_ffn_down_key(layer, expert) {
                    orient_in_place(tensors, &key, hidden, mf);
                }
            }
        }
    }
}

/// Transpose `tensors[key]` if it's currently shaped `(expected_cols, expected_rows)`
/// while the canonical shape is `(expected_rows, expected_cols)`. No-op when the
/// tensor is missing, already canonical, the dimensions are equal (ambiguous),
/// or the shape matches neither orientation.
pub(super) fn orient_in_place(
    tensors: &mut HashMap<String, crate::WeightArray>,
    key: &str,
    expected_rows: usize,
    expected_cols: usize,
) {
    if expected_rows == 0 || expected_cols == 0 || expected_rows == expected_cols {
        return;
    }
    let arr = match tensors.get(key) {
        Some(a) => a,
        None => return,
    };
    let shape = arr.shape();
    if shape.len() != 2 {
        return;
    }
    if shape[0] == expected_rows && shape[1] == expected_cols {
        return;
    }
    if shape[0] == expected_cols && shape[1] == expected_rows {
        let mut out = ndarray::Array2::<f32>::zeros((expected_rows, expected_cols));
        out.assign(&arr.t());
        tensors.insert(key.to_string(), out.into_shared());
    }
}

/// Walk per-layer attention tensors and ensure they're in canonical orientation.
///
/// Canonical (Linear convention):
/// - q_proj:   shape `(num_q_heads * head_dim, hidden_size)`
/// - k_proj:   shape `(num_kv_heads * head_dim, hidden_size)`
/// - v_proj:   shape `(num_kv_heads * head_dim, hidden_size)`
/// - o_proj:   shape `(hidden_size, num_q_heads * head_dim)`
/// - qkv_proj: shape `(q_dim + 2 * kv_dim, hidden_size)` — used by fused-QKV
///   architectures (GPT-2). Split happens in `split_fused_qkv` after this.
///
/// `orient_in_place` is a no-op when the two dimensions are equal, so square
/// tensors (e.g. GPT-2 with `q_dim == kv_dim == hidden`) survive untouched.
/// The fused-QKV tensor is asymmetric (`3*hidden vs hidden`) and orientable.
pub(super) fn orient_attention_tensors(
    tensors: &mut HashMap<String, crate::WeightArray>,
    arch: &dyn crate::config::ModelArchitecture,
) {
    let cfg = arch.config();
    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    if cfg.num_layers == 0 || hidden == 0 || head_dim == 0 {
        return;
    }
    let q_dim = cfg.num_q_heads * head_dim;
    let kv_dim = cfg.num_kv_heads * head_dim;

    for layer in 0..cfg.num_layers {
        if q_dim > 0 {
            orient_in_place(tensors, &arch.attn_q_key(layer), q_dim, hidden);
            orient_in_place(tensors, &arch.attn_o_key(layer), hidden, q_dim);
        }
        if kv_dim > 0 {
            orient_in_place(tensors, &arch.attn_k_key(layer), kv_dim, hidden);
            orient_in_place(tensors, &arch.attn_v_key(layer), kv_dim, hidden);
        }
        if let Some(key) = arch.fused_qkv_key(layer) {
            let total = q_dim + 2 * kv_dim;
            if total > 0 {
                orient_in_place(tensors, &key, total, hidden);
            }
        }
    }
}

/// Materialise per-projection q/k/v tensors (and biases) from a fused QKV
/// matrix, when the architecture declares one via `fused_qkv_key`.
///
/// The fused weight is assumed to be in canonical orientation
/// `(q_dim + 2 * kv_dim, hidden_size)` — `orient_attention_tensors` runs
/// first to enforce that. Rows split into:
/// - `0 .. q_dim`                       → `attn_q_key`
/// - `q_dim .. q_dim + kv_dim`          → `attn_k_key`
/// - `q_dim + kv_dim .. q_dim + 2*kv_dim` → `attn_v_key`
///
/// The fused bias (1D, length `q_dim + 2 * kv_dim`) splits identically into
/// the per-projection bias keys returned by the trait.
///
/// Driven entirely by `ModelArchitecture` keys + `ModelConfig` dimensions —
/// no family-specific branching.
pub(super) fn split_fused_qkv(
    tensors: &mut HashMap<String, crate::WeightArray>,
    vectors: &mut HashMap<String, Vec<f32>>,
    arch: &dyn crate::config::ModelArchitecture,
) {
    let cfg = arch.config();
    let hidden = cfg.hidden_size;
    let head_dim = cfg.head_dim;
    if cfg.num_layers == 0 || hidden == 0 || head_dim == 0 {
        return;
    }
    let q_dim = cfg.num_q_heads * head_dim;
    let kv_dim = cfg.num_kv_heads * head_dim;
    let total = q_dim + 2 * kv_dim;
    if total == 0 {
        return;
    }

    for layer in 0..cfg.num_layers {
        let Some(weight_key) = arch.fused_qkv_key(layer) else {
            continue;
        };

        if let Some(fused) = tensors.remove(&weight_key) {
            let shape = fused.shape();
            if shape.len() == 2 && shape[0] == total && shape[1] == hidden {
                if q_dim > 0 {
                    let q = fused.slice(ndarray::s![..q_dim, ..]).to_owned();
                    tensors.insert(arch.attn_q_key(layer), q.into_shared());
                }
                if kv_dim > 0 {
                    let k = fused
                        .slice(ndarray::s![q_dim..q_dim + kv_dim, ..])
                        .to_owned();
                    let v = fused
                        .slice(ndarray::s![q_dim + kv_dim..total, ..])
                        .to_owned();
                    tensors.insert(arch.attn_k_key(layer), k.into_shared());
                    tensors.insert(arch.attn_v_key(layer), v.into_shared());
                }
            } else {
                // Shape doesn't match expected fused layout — put it back so
                // the caller can surface the mismatch via missing-tensor errors.
                tensors.insert(weight_key, fused);
            }
        }

        if let Some(bias_key) = arch.fused_qkv_bias_key(layer) {
            if let Some(fused_b) = vectors.remove(&bias_key) {
                if fused_b.len() == total {
                    if let (Some(qb_key), true) = (arch.attn_q_bias_key(layer), q_dim > 0) {
                        vectors.insert(qb_key, fused_b[..q_dim].to_vec());
                    }
                    if kv_dim > 0 {
                        if let Some(kb_key) = arch.attn_k_bias_key(layer) {
                            vectors.insert(kb_key, fused_b[q_dim..q_dim + kv_dim].to_vec());
                        }
                        if let Some(vb_key) = arch.attn_v_bias_key(layer) {
                            vectors.insert(vb_key, fused_b[q_dim + kv_dim..total].to_vec());
                        }
                    }
                } else {
                    vectors.insert(bias_key, fused_b);
                }
            }
        }
    }
}

/// Build a minimal Gpt2-shaped ModelConfig for orientation/split tests.
#[cfg(test)]
fn synth_gpt2_config(
    num_layers: usize,
    hidden: usize,
    head_dim: usize,
    n_heads: usize,
) -> crate::config::ModelConfig {
    crate::config::ModelConfig {
        model_type: "gpt2".into(),
        norm_eps: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: 4 * hidden,
        ffn_intermediate_size_by_layer: None,
        head_dim,
        num_q_heads: n_heads,
        num_kv_heads: n_heads,
        vocab_size: Some(8),
        rope_base: 10_000.0,
        layer_rope_theta: None,
        rope_local_base: None,
        sliding_window: None,
        use_sliding_window: None,
        position_embedding_type: None,
        no_rope_layers: None,
        no_rope_layer_interval: None,
        rope_interleaved: None,
        use_mrope: None,
        ffn_shape_name: None,
        is_llama_config: None,
        max_window_layers: None,
        num_experts: None,
        num_experts_per_token: None,
        num_shared_experts: None,
        shared_expert_intermediate_size: None,
        hc_streams: None,
        hc_sinkhorn_iters: None,
        hc_eps: None,
        attn_res_block_size: None,
        enable_moe_block: false,
        top_k_experts: None,
        moe_intermediate_size: None,
        swiglu_limit: None,
        norm_topk_prob: None,
        routed_expert_hidden_size: None,
        latent_moe_use_norm: None,
        kv_lora_rank: None,
        q_lora_rank: None,
        qk_nope_head_dim: None,
        qk_rope_head_dim: None,
        v_head_dim: None,
        index_topk: None,
        index_n_heads: None,
        index_head_dim: None,
        rope_scaling: None,
        attn_logit_softcapping: None,
        final_logit_softcapping: None,
        query_pre_attn_scalar: None,
        embedding_multiplier: None,
        residual_multiplier: None,
        attention_multiplier: None,
        logits_scaling: None,
        global_head_dim: None,
        num_global_kv_heads: None,
        partial_rotary_factor: None,
        sliding_window_pattern: None,
        layer_types: None,
        attention_k_eq_v: false,
        per_layer_embed_dim: None,
        num_kv_shared_layers: None,
        has_vision_config: false,
        tie_word_embeddings: None,
        qk_scale_factor: None,
        output_multiplier: None,
        post_norm_eps: None,
        attention_bias: None,
        qkv_bias: None,
        declares_rotary: false,
        scale_attn_weights: None,
        scale_attn_by_inverse_layer_idx: None,
        reorder_and_upcast_attn: None,
        add_cross_attention: None,
        mlp_bias: None,
        hidden_act: None,
        activation_situ_beta: None,
        activation_situ_linear_beta: None,
        max_position_embeddings: None,
        image_token_id: None,
        video_token_id: None,
        out_hidden_size: None,
        projector_hidden_size: None,
        projector_hidden_act: None,
        target_layer_ids: None,
        draft_block_size: None,
        mask_token_id: None,
        use_double_wide_mlp: None,
        vocab_size_per_layer_input: None,
        linear_conv_kernel_dim: None,
        linear_key_head_dim: None,
        linear_value_head_dim: None,
        linear_num_key_heads: None,
        linear_num_value_heads: None,
        linear_attn_interleave: crate::config::DeclaredInterleave::Absent,
        mtp_interleave: crate::config::DeclaredInterleave::Absent,
        kda_geometry: None,
        kda_gate_lower_bound: None,
        kda_safe_gate: None,
        kda_use_full_rank_gate: None,
        mla_use_output_gate: None,
        router_activation: None,
        routed_scaling_factor: None,
        expert_groups: None,
        topk_group: None,
        use_grouped_topk: None,
        moe_layer_freq: None,
        first_k_dense_replace: None,
        mla_use_nope: None,
        model_max_length: None,
        d_rel: None,
        rel_extent: None,
        mamba_ssm_dtype: None,
        mamba2_geometry: None,
        mamba2_provenance: None,
        conv_qkv_attn: None,
        conv_qkv_provenance: None,
        attn_causal: None,
        pad_vocab_size_multiple: None,
        fused_add_norm: None,
        mlp_intermediate_size: None,
        mlp_padding_size: None,
        use_mlp_bias: None,
        residual_in_fp32: None,
        attn_output_gate: None,
        output_gate_type: None,
        mtp_num_hidden_layers: None,
        mtp_use_dedicated_embeddings: None,
        mrope_interleaved: None,
        mrope_section: None,
    }
}

#[cfg(test)]
mod tests;
