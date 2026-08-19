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

        // Shared-expert FFN tensors share dense intermediate dim.
        if dense_inter > 0 {
            if let Some(key) = arch.shared_expert_gate_key(layer) {
                orient_in_place(tensors, &key, dense_inter, hidden);
            }
            if let Some(key) = arch.shared_expert_up_key(layer) {
                orient_in_place(tensors, &key, dense_inter, hidden);
            }
            if let Some(key) = arch.shared_expert_down_key(layer) {
                orient_in_place(tensors, &key, hidden, dense_inter);
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
        head_dim,
        num_q_heads: n_heads,
        num_kv_heads: n_heads,
        vocab_size: Some(8),
        rope_base: 10_000.0,
        layer_rope_theta: None,
        rope_local_base: None,
        sliding_window: None,
        num_experts: None,
        num_experts_per_token: None,
        num_shared_experts: None,
        enable_moe_block: false,
        top_k_experts: None,
        moe_intermediate_size: None,
        swiglu_limit: None,
        norm_topk_prob: None,
        kv_lora_rank: None,
        q_lora_rank: None,
        qk_nope_head_dim: None,
        qk_rope_head_dim: None,
        v_head_dim: None,
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
        hidden_act: None,
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orient_in_place_transposes_inverse_layout() {
        use ndarray::Array2;

        let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
        // Inverse layout: stored (cols, rows) when canonical is (rows, cols).
        // Canonical for ffn_down is (hidden, intermediate).
        let stored = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .into_shared();
        tensors.insert("layers.0.mlp.down_proj.weight".to_string(), stored);

        // Canonical (hidden=2, intermediate=3): expect shape (2, 3) after orient.
        orient_in_place(&mut tensors, "layers.0.mlp.down_proj.weight", 2, 3);

        let oriented = tensors.get("layers.0.mlp.down_proj.weight").unwrap();
        assert_eq!(oriented.shape(), &[2, 3]);
        // Transpose maps (i,j) → (j,i): row-major buffer becomes 1,3,5,2,4,6.
        assert_eq!(oriented[[0, 0]], 1.0);
        assert_eq!(oriented[[0, 1]], 3.0);
        assert_eq!(oriented[[0, 2]], 5.0);
        assert_eq!(oriented[[1, 0]], 2.0);
        assert_eq!(oriented[[1, 1]], 4.0);
        assert_eq!(oriented[[1, 2]], 6.0);
    }

    #[test]
    fn test_orient_in_place_leaves_canonical_layout_untouched() {
        use ndarray::Array2;

        let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
        let canonical = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .into_shared();
        let original_ptr = canonical.as_ptr();
        tensors.insert("layers.0.mlp.down_proj.weight".to_string(), canonical);

        orient_in_place(&mut tensors, "layers.0.mlp.down_proj.weight", 2, 3);

        let after = tensors.get("layers.0.mlp.down_proj.weight").unwrap();
        // No clone-and-replace: same backing buffer.
        assert_eq!(after.as_ptr(), original_ptr);
    }

    #[test]
    fn test_orient_in_place_skips_ambiguous_square_dims() {
        use ndarray::Array2;

        let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
        let square = Array2::from_shape_vec((4, 4), (0..16).map(|x| x as f32).collect())
            .unwrap()
            .into_shared();
        tensors.insert("layers.0.mlp.up_proj.weight".to_string(), square);

        orient_in_place(&mut tensors, "layers.0.mlp.up_proj.weight", 4, 4);

        let after = tensors.get("layers.0.mlp.up_proj.weight").unwrap();
        // Untouched — orientation can't be inferred when rows == cols.
        assert_eq!(after.shape(), &[4, 4]);
        assert_eq!(after[[0, 0]], 0.0);
        assert_eq!(after[[3, 3]], 15.0);
    }

    #[test]
    fn test_orient_attention_tensors_fixes_inverse_fused_qkv_layout() {
        use ndarray::Array2;

        // hidden=4, head_dim=2, n_heads=2 → q_dim=kv_dim=4, total=12.
        let cfg = synth_gpt2_config(1, 4, 2, 2);
        let arch = crate::architectures::gpt2::Gpt2Arch::from_config(cfg);

        let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
        // Inverse layout: stored (hidden=4, total=12) instead of (12, 4).
        let inverse = Array2::<f32>::zeros((4, 12)).into_shared();
        tensors.insert("layers.0.self_attn.qkv_proj.weight".into(), inverse);

        orient_attention_tensors(&mut tensors, &arch);

        let oriented = tensors.get("layers.0.self_attn.qkv_proj.weight").unwrap();
        assert_eq!(oriented.shape(), &[12, 4]);
    }

    #[test]
    fn test_split_fused_qkv_materialises_per_projection_tensors_and_biases() {
        use ndarray::Array2;

        // hidden=4, head_dim=2, n_heads=2 → q_dim=kv_dim=4, total=12.
        let cfg = synth_gpt2_config(1, 4, 2, 2);
        let arch = crate::architectures::gpt2::Gpt2Arch::from_config(cfg);

        let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
        let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();

        // Fused weight: row r has constant value r so we can verify slices.
        let mut data = Vec::with_capacity(12 * 4);
        for r in 0..12 {
            for _c in 0..4 {
                data.push(r as f32);
            }
        }
        let fused_w = Array2::from_shape_vec((12, 4), data).unwrap().into_shared();
        tensors.insert("layers.0.self_attn.qkv_proj.weight".into(), fused_w);

        // Fused bias: 12 distinct values.
        let fused_b: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        vectors.insert("layers.0.self_attn.qkv_proj.bias".into(), fused_b);

        split_fused_qkv(&mut tensors, &mut vectors, &arch);

        // Fused tensor + bias removed.
        assert!(!tensors.contains_key("layers.0.self_attn.qkv_proj.weight"));
        assert!(!vectors.contains_key("layers.0.self_attn.qkv_proj.bias"));

        let q = tensors.get("layers.0.self_attn.q_proj.weight").unwrap();
        let k = tensors.get("layers.0.self_attn.k_proj.weight").unwrap();
        let v = tensors.get("layers.0.self_attn.v_proj.weight").unwrap();
        assert_eq!(q.shape(), &[4, 4]);
        assert_eq!(k.shape(), &[4, 4]);
        assert_eq!(v.shape(), &[4, 4]);
        // Row r maps to constant r in the fused layout. q rows 0..4, k 4..8, v 8..12.
        assert_eq!(q[[0, 0]], 0.0);
        assert_eq!(q[[3, 3]], 3.0);
        assert_eq!(k[[0, 0]], 4.0);
        assert_eq!(k[[3, 3]], 7.0);
        assert_eq!(v[[0, 0]], 8.0);
        assert_eq!(v[[3, 3]], 11.0);

        let qb = vectors.get("layers.0.self_attn.q_proj.bias").unwrap();
        let kb = vectors.get("layers.0.self_attn.k_proj.bias").unwrap();
        let vb = vectors.get("layers.0.self_attn.v_proj.bias").unwrap();
        assert_eq!(qb.len(), 4);
        assert_eq!(kb.len(), 4);
        assert_eq!(vb.len(), 4);
        assert!((qb[0] - 0.0).abs() < 1e-6);
        assert!((kb[0] - 0.4).abs() < 1e-6);
        assert!((vb[0] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_split_fused_qkv_no_op_when_arch_has_no_fused_key() {
        use ndarray::Array2;

        // Llama-style arch — no fused QKV.
        let cfg = synth_gpt2_config(1, 4, 2, 2);
        let arch = crate::architectures::llama::LlamaArch::from_config(cfg);

        let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
        let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
        let q = Array2::<f32>::zeros((4, 4)).into_shared();
        tensors.insert("layers.0.self_attn.q_proj.weight".into(), q);

        split_fused_qkv(&mut tensors, &mut vectors, &arch);

        // Untouched.
        assert!(tensors.contains_key("layers.0.self_attn.q_proj.weight"));
    }

    #[test]
    fn test_orient_ffn_tensors_fixes_gpt2_style_inverse_layout() {
        use crate::config::ModelConfig;
        use ndarray::Array2;

        let cfg = ModelConfig {
            model_type: "gpt2".into(),
            norm_eps: None,
            num_layers: 1,
            hidden_size: 4,
            intermediate_size: 12,
            head_dim: 2,
            num_q_heads: 2,
            num_kv_heads: 2,
            vocab_size: Some(8),
            rope_base: 10_000.0,
            layer_rope_theta: None,
            rope_local_base: None,
            sliding_window: None,
            num_experts: None,
            num_experts_per_token: None,
            num_shared_experts: None,
            enable_moe_block: false,
            top_k_experts: None,
            moe_intermediate_size: None,
            swiglu_limit: None,
            norm_topk_prob: None,
            kv_lora_rank: None,
            q_lora_rank: None,
            qk_nope_head_dim: None,
            qk_rope_head_dim: None,
            v_head_dim: None,
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
            hidden_act: None,
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
        };
        let arch = crate::architectures::gpt2::Gpt2Arch::from_config(cfg);

        // Inverse layouts: ffn_up stored (hidden, inter) instead of (inter, hidden);
        // ffn_down stored (inter, hidden) instead of (hidden, inter).
        let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
        let up_inverse = Array2::<f32>::zeros((4, 12)).into_shared();
        let down_inverse = Array2::<f32>::zeros((12, 4)).into_shared();
        tensors.insert("layers.0.mlp.up_proj.weight".into(), up_inverse);
        tensors.insert("layers.0.mlp.down_proj.weight".into(), down_inverse);

        orient_ffn_tensors(&mut tensors, &arch);

        let up = tensors.get("layers.0.mlp.up_proj.weight").unwrap();
        let down = tensors.get("layers.0.mlp.down_proj.weight").unwrap();
        assert_eq!(up.shape(), &[12, 4]);
        assert_eq!(down.shape(), &[4, 12]);
    }

    #[test]
    fn orient_embedding_transposes_when_hidden_is_rows() {
        use ndarray::Array2;
        let hidden = 4;
        let vocab = 10;
        let embed = Array2::<f32>::zeros((hidden, vocab)).into_shared();
        let result = orient_embedding(embed, hidden, Some(vocab));
        assert_eq!(result.shape(), &[vocab, hidden]);
    }

    #[test]
    fn orient_embedding_noop_when_already_canonical() {
        use ndarray::Array2;
        let hidden = 4;
        let vocab = 10;
        let embed = Array2::<f32>::zeros((vocab, hidden)).into_shared();
        let result = orient_embedding(embed, hidden, Some(vocab));
        assert_eq!(result.shape(), &[vocab, hidden]);
    }

    #[test]
    fn orient_embedding_ambiguous_passthrough() {
        use ndarray::Array2;
        let embed = Array2::<f32>::zeros((7, 9)).into_shared();
        let result = orient_embedding(embed, 4, Some(10));
        assert_eq!(result.shape(), &[7, 9]);
    }

    #[test]
    fn orient_ffn_tensors_noop_for_zero_layers() {
        let cfg = synth_gpt2_config(0, 4, 2, 2);
        let arch = crate::architectures::llama::LlamaArch::from_config(cfg);
        let mut tensors = HashMap::new();
        orient_ffn_tensors(&mut tensors, &arch);
        assert!(tensors.is_empty());
    }

    #[test]
    fn orient_attention_tensors_noop_for_zero_head_dim() {
        let cfg = synth_gpt2_config(1, 4, 0, 2);
        let arch = crate::architectures::llama::LlamaArch::from_config(cfg);
        let mut tensors = HashMap::new();
        orient_attention_tensors(&mut tensors, &arch);
        assert!(tensors.is_empty());
    }

    #[test]
    fn split_fused_qkv_noop_for_zero_head_dim() {
        let cfg = synth_gpt2_config(1, 4, 0, 2);
        let arch = crate::architectures::gpt2::Gpt2Arch::from_config(cfg);
        let mut tensors = HashMap::new();
        let mut vectors = HashMap::new();
        split_fused_qkv(&mut tensors, &mut vectors, &arch);
        assert!(tensors.is_empty());
    }

    #[test]
    fn split_fused_qkv_puts_back_wrong_shape_tensor() {
        use ndarray::Array2;
        // hidden=4, head_dim=2, n_heads=2 → q_dim=kv_dim=4, total=12
        let arch = crate::detect_from_json(&serde_json::json!({
            "model_type": "gpt2",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2
        }));
        let fused_key = arch.fused_qkv_key(0).unwrap();

        let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
        let mut vectors = HashMap::new();
        // Insert a tensor with wrong shape (5x4 instead of 12x4)
        let wrong = Array2::<f32>::zeros((5, 4)).into_shared();
        tensors.insert(fused_key.clone(), wrong);

        split_fused_qkv(&mut tensors, &mut vectors, &*arch);
        // Should be put back under original key
        assert!(
            tensors.contains_key(&fused_key),
            "wrong-shape tensor should be restored"
        );
        assert_eq!(tensors[&fused_key].shape(), &[5, 4]);
    }

    #[test]
    fn split_fused_qkv_puts_back_wrong_length_bias() {
        use ndarray::Array2;
        let arch = crate::detect_from_json(&serde_json::json!({
            "model_type": "gpt2",
            "hidden_size": 4,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "num_attention_heads": 2
        }));
        let fused_key = arch.fused_qkv_key(0).unwrap();
        let bias_key = arch.fused_qkv_bias_key(0).unwrap();

        let mut tensors: HashMap<String, crate::WeightArray> = HashMap::new();
        let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
        // Insert correct weight so it splits, but wrong-length bias
        let correct = Array2::<f32>::zeros((12, 4)).into_shared();
        tensors.insert(fused_key, correct);
        vectors.insert(bias_key.clone(), vec![1.0; 7]); // should be 12

        split_fused_qkv(&mut tensors, &mut vectors, &*arch);
        // Bias should be put back under original key
        assert!(
            vectors.contains_key(&bias_key),
            "wrong-length bias should be restored"
        );
        assert_eq!(vectors[&bias_key].len(), 7);
    }
}
