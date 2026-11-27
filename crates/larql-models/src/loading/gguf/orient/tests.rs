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
        ffn_intermediate_size_by_layer: None,
        head_dim: 2,
        num_q_heads: 2,
        num_kv_heads: 2,
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
