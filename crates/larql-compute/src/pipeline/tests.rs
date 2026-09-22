use super::*;

fn minimal_qw(data: &[u8]) -> QuantWeight<'_> {
    QuantWeight {
        data,
        scales: None,
        format: QuantFormat::Q4_0,
    }
}

fn minimal_layer<'a>(
    data: &'a [u8],
    norms: &'a [f32],
    ffn_type: FfnType,
    moe: Option<MoeLayerWeights<'a>>,
) -> FullPipelineLayer<'a> {
    let qw = minimal_qw(data);
    // Spread `..Default::default()` collapses the 30-field boilerplate
    // to just the fields this test actually exercises. Pin this pattern
    // so any future test that wants a minimal layer copies it.
    FullPipelineLayer {
        wq: qw,
        wk: qw,
        wv: qw,
        wo: qw,
        gate: qw,
        up: qw,
        down: qw,
        input_norm: norms,
        post_attn_norm: norms,
        ffn_type,
        attn_scale: 0.5,
        head_dim: 4,
        num_q_heads: 1,
        num_kv_heads: 1,
        moe,
        ..FullPipelineLayer::default()
    }
}

#[test]
fn activation_from_bool() {
    assert_eq!(Activation::from(true), Activation::GeluTanh);
    assert_eq!(Activation::from(false), Activation::Silu);
}

#[test]
fn registry_tag_round_trips_every_variant() {
    // registry_tag() is the exact inverse of from_registry_tag() for
    // every variant — including the new ternary I2_S.
    for f in [
        QuantFormat::Q4_0,
        QuantFormat::Q4_K,
        QuantFormat::Q4_KF,
        QuantFormat::Q5_K,
        QuantFormat::Q6_K,
        QuantFormat::Q8_0,
        QuantFormat::BF16,
        QuantFormat::F16,
        QuantFormat::F32,
        QuantFormat::I2S,
    ] {
        assert_eq!(QuantFormat::from_registry_tag(f.registry_tag()), Some(f));
    }
}

#[test]
fn i2s_tag_and_family_predicates() {
    assert_eq!(
        QuantFormat::from_registry_tag("I2_S"),
        Some(QuantFormat::I2S)
    );
    assert_eq!(QuantFormat::I2S.registry_tag(), "I2_S");
    assert!(QuantFormat::I2S.is_ternary());
    assert!(!QuantFormat::Q4_K.is_ternary());
    // I2_S is none of the block-quant families and has no flat block layout.
    assert!(!QuantFormat::I2S.is_kquant_family());
    assert!(!QuantFormat::I2S.is_legacy_q8());
    assert!(QuantFormat::I2S.packed_block_layout().is_none());
}

#[test]
fn is_gated_matches_ffn_type() {
    let norms = [1.0f32; 4];
    let gated = minimal_layer(&[], &norms, FfnType::Gated, None);
    let standard = minimal_layer(&[], &norms, FfnType::Standard, None);
    assert!(gated.is_gated());
    assert!(!standard.is_gated());
}

#[test]
fn is_hybrid_moe_reflects_option() {
    let norms = [1.0f32; 4];
    let no_moe = minimal_layer(&[], &norms, FfnType::Gated, None);
    assert!(!no_moe.is_hybrid_moe());

    let moe = MoeLayerWeights {
        experts_gate_up: Vec::new(),
        experts_down: Vec::new(),
        routing_policy: MoeRoutingPolicy::default(),
        weight_layout: MoeWeightLayout::default(),
        router_proj: &[],
        router_scale: &[],
        router_per_expert_scale: &[],
        router_norm: &[],
        router_norm_parameter_free: false,
        router_input_scalar: 1.0,
        pre_experts_norm: &[],
        post_ffn1_norm: &[],
        post_experts_norm: &[],
        num_experts: 2,
        top_k: 1,
        intermediate_size: 4,
        activation: Activation::Silu,
        expert_data_format: QuantFormat::BF16,
    };
    let with_moe = minimal_layer(&[], &norms, FfnType::Gated, Some(moe));
    assert!(with_moe.is_hybrid_moe());
}

#[test]
fn quant_format_equality() {
    assert_eq!(QuantFormat::Q4_K, QuantFormat::Q4_K);
    assert_ne!(QuantFormat::Q4_K, QuantFormat::Q6_K);
    assert_ne!(QuantFormat::Q4_0, QuantFormat::Q4_KF);
}

/// Pin the k-quant family taxonomy. Adding a new format requires
/// updating exactly one of these classifiers.
#[test]
fn quant_format_classifiers() {
    // k-quant family (256-element super-blocks)
    assert!(QuantFormat::Q4_K.is_kquant_family());
    assert!(QuantFormat::Q4_KF.is_kquant_family());
    assert!(QuantFormat::Q5_K.is_kquant_family());
    assert!(QuantFormat::Q6_K.is_kquant_family());
    // Legacy block-32 Q8 path
    assert!(QuantFormat::Q4_0.is_legacy_q8());
    assert!(QuantFormat::Q8_0.is_legacy_q8());
    // Float-input formats are neither
    for fmt in [QuantFormat::BF16, QuantFormat::F16, QuantFormat::F32] {
        assert!(!fmt.is_kquant_family());
        assert!(!fmt.is_legacy_q8());
    }
    // Q4_KF is a subset of the k-quant family
    assert!(QuantFormat::Q4_KF.is_q4kf());
    assert!(!QuantFormat::Q4_K.is_q4kf());
    assert!(!QuantFormat::Q6_K.is_q4kf());
}

/// Being a k-quant and having a Metal kernel are separate questions, and
/// Q5_K is the format that separates them. Conflating the two is what
/// would route its 176-byte super-block into the Q4_K kernel's 144-byte
/// stride.
#[test]
fn kquant_family_does_not_imply_metal_kernel() {
    assert!(QuantFormat::Q5_K.is_kquant_family());
    assert!(!QuantFormat::Q5_K.has_metal_kernel());

    for fmt in [QuantFormat::Q4_K, QuantFormat::Q4_KF, QuantFormat::Q6_K] {
        assert!(fmt.is_kquant_family());
        assert!(fmt.has_metal_kernel());
    }
}

/// Q5_K's super-block geometry must round-trip through the format enum —
/// a wrong `bytes_per_block` here silently mis-strides every row read.
#[test]
fn q5k_block_layout_and_tag() {
    assert_eq!(QuantFormat::Q5_K.packed_block_layout(), Some((256, 176)));
    assert_eq!(QuantFormat::Q5_K.registry_tag(), "Q5_K");
    assert_eq!(
        QuantFormat::from_registry_tag("Q5_K"),
        Some(QuantFormat::Q5_K)
    );
    // 2560 columns = 10 super-blocks → 1760 bytes per row.
    assert_eq!(QuantFormat::Q5_K.packed_matrix_bytes(1, 2560), Some(1760));
    assert!(!QuantFormat::Q5_K.is_legacy_q8());
    assert!(!QuantFormat::Q5_K.is_ternary());
    assert!(!QuantFormat::Q5_K.is_q4kf());
}

#[test]
fn quant_format_reports_packed_matrix_bytes() {
    assert_eq!(QuantFormat::Q4_0.packed_matrix_bytes(2, 32), Some(36));
    assert_eq!(QuantFormat::Q4_K.packed_matrix_bytes(2, 256), Some(288));
    assert_eq!(QuantFormat::Q4_KF.packed_matrix_bytes(2, 256), Some(320));
    assert_eq!(QuantFormat::Q6_K.packed_matrix_bytes(2, 256), Some(420));
    assert_eq!(QuantFormat::F16.packed_matrix_bytes(2, 256), None);
}

#[test]
fn from_registry_tag_round_trips_known_tags_and_rejects_unknown() {
    for (tag, fmt) in [
        ("Q4_0", QuantFormat::Q4_0),
        ("Q4_K", QuantFormat::Q4_K),
        ("Q4_KF", QuantFormat::Q4_KF),
        ("Q6_K", QuantFormat::Q6_K),
        ("Q8_0", QuantFormat::Q8_0),
        ("BF16", QuantFormat::BF16),
        ("F16", QuantFormat::F16),
        ("F32", QuantFormat::F32),
    ] {
        assert_eq!(QuantFormat::from_registry_tag(tag), Some(fmt));
    }
    assert_eq!(QuantFormat::from_registry_tag("Q42_X"), None);
    assert_eq!(QuantFormat::from_registry_tag("q4_k"), None); // case-sensitive
    assert_eq!(QuantFormat::from_registry_tag(""), None);
}

/// The two string-keyed matvec dispatchers derive their packed row
/// stride as `(cols / block_elems) * block_bytes`; pin that this equals
/// what `from_registry_tag` + `packed_block_layout` produce for the
/// Q4_K / Q6_K tags they accept (256-multiple cols).
#[test]
fn registry_tag_block_layout_matches_dispatcher_stride() {
    let cols = 512usize;
    for (tag, raw_block_bytes) in [("Q4_K", 144usize), ("Q6_K", 210usize)] {
        let fmt = QuantFormat::from_registry_tag(tag).unwrap();
        let (block_elems, block_bytes) = fmt.packed_block_layout().unwrap();
        assert_eq!(block_elems, 256);
        assert_eq!(block_bytes, raw_block_bytes);
        assert_eq!(
            (cols / block_elems) * block_bytes,
            (cols / 256) * raw_block_bytes
        );
    }
}

/// `..Default::default()` must work with stack-local borrowed data —
/// the compiler reborrows the `'static` defaults at the caller's
/// shorter lifetime. Pin the pattern.
#[test]
fn default_layer_accepts_local_borrows_via_spread() {
    let data: Vec<u8> = vec![0, 1, 2];
    let norms: Vec<f32> = vec![1.0; 4];

    let layer = FullPipelineLayer {
        input_norm: &norms,
        post_attn_norm: &norms,
        wq: QuantWeight {
            data: &data,
            ..Default::default()
        },
        head_dim: 4,
        num_q_heads: 1,
        num_kv_heads: 1,
        ..Default::default()
    };

    // Defaulted fields carry through.
    assert_eq!(layer.eps, RMSNORM_EPSILON_DEFAULT);
    assert_eq!(layer.norm_type, NormType::RmsNorm);
    assert_eq!(layer.ffn_type, FfnType::Gated);
    assert_eq!(layer.activation, Activation::Silu);
    assert!(!layer.has_v_norm);
    assert!(layer.moe.is_none());

    // Explicit fields are honoured.
    assert_eq!(layer.input_norm.len(), 4);
    assert_eq!(layer.wq.data.len(), 3);
    assert_eq!(layer.head_dim, 4);
}

#[test]
fn layer_spec_views_preserve_flat_field_values() {
    let data: Vec<u8> = vec![0, 1, 2, 3];
    let norms: Vec<f32> = vec![1.0; 8];
    let q_norm: Vec<f32> = vec![0.5; 4];

    let qw = QuantWeight {
        data: &data,
        scales: None,
        format: QuantFormat::Q4_K,
    };
    let layer = FullPipelineLayer {
        wq: qw,
        wk: qw,
        wv: qw,
        wo: qw,
        gate: qw,
        up: qw,
        down: qw,
        input_norm: &norms,
        post_attn_norm: &norms,
        norm_offset: 1.0,
        qk_norm_offset: 0.25,
        eps: 1e-5,
        has_post_norms: true,
        activation: Activation::GeluTanh,
        attn_scale: 0.125,
        head_dim: 4,
        num_q_heads: 2,
        num_kv_heads: 1,
        rope_base: ROPE_BASE_GLOBAL,
        rotary_dim: 2,
        sliding_window: 32,
        has_v_norm: true,
        q_norm_weight: Some(&q_norm),
        ffn_is_remote: true,
        moe_combined_output_norm: true,
        moe_outer_post_norm: Some(&norms),
        ..FullPipelineLayer::default()
    };

    let weights = layer.weights();
    assert_eq!(weights.attention.wq.format, QuantFormat::Q4_K);
    assert_eq!(weights.ffn.down.data.len(), data.len());

    let norms_view = layer.norms();
    assert_eq!(norms_view.input_norm.len(), norms.len());
    assert_eq!(norms_view.norm_offset, 1.0);
    assert_eq!(norms_view.qk_norm_offset, 0.25);
    assert_eq!(norms_view.eps, 1e-5);
    assert!(norms_view.has_post_norms);

    let attn = layer.attention_spec();
    assert_eq!(attn.head_dim, 4);
    assert_eq!(attn.num_q_heads, 2);
    assert_eq!(attn.num_kv_heads, 1);
    assert_eq!(attn.rope_base, ROPE_BASE_GLOBAL);
    assert_eq!(attn.rotary_dim, 2);
    assert_eq!(attn.sliding_window, 32);
    assert!(attn.has_v_norm);
    assert!(attn.q_norm_enabled);
    assert!(!attn.k_norm_enabled);

    assert_eq!(layer.ffn_spec().activation, Activation::GeluTanh);
    assert!(layer.remote_ffn_spec().is_remote);
    assert!(layer.moe_spec().combined_output_norm);
    assert_eq!(layer.moe_spec().outer_post_norm.unwrap().len(), norms.len());
}

/// `MoeWeightLayout` constructors + both `down_cols` policy arms.
#[test]
fn moe_weight_layout_down_cols_policies() {
    let unpadded = MoeWeightLayout::unpadded();
    assert_eq!(unpadded.down_cols(704, QuantFormat::Q4_K), 704);

    let padded = MoeWeightLayout::quant_block_padded_down();
    // 704 rounds up to the next 256-elem Q4_K super-block boundary.
    assert_eq!(padded.down_cols(704, QuantFormat::Q4_K), 768);
    // Already block-aligned stays put.
    assert_eq!(padded.down_cols(512, QuantFormat::Q4_K), 512);
}
