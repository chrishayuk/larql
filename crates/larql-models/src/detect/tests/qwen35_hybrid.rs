//! Qwen3.5-style hybrid linear-attention configs — `model_type: "qwen3_5"`
//! (or `"qwen3_5_text"` when nested under `text_config`) matches the
//! `qwen`-prefix family route and is served by [`QwenArch`], so no new
//! registry entry is needed. What *is* new: `text_config` declares hybrid
//! linear-attention block geometry, a multi-token-prediction head, and
//! mRoPE sectioning that `ModelConfig` did not carry before — this module
//! pins that every one of those facts reaches `ModelConfig` verbatim
//! (R2/Kimi-Linear-rung prep, `docs/k3-funnel.md`).

use crate::detect::*;

fn qwen35_shaped_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "hidden_size": 5120,
            "num_hidden_layers": 8,
            "intermediate_size": 17408,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248320,
            "full_attention_interval": 4,
            "layer_types": [
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
                "linear_attention", "linear_attention", "linear_attention", "full_attention",
            ],
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "mamba_ssm_dtype": "float32",
            "attn_output_gate": true,
            "output_gate_type": "swish",
            "mtp_num_hidden_layers": 1,
            "mtp_use_dedicated_embeddings": false,
            "partial_rotary_factor": 0.25,
            "rope_parameters": {
                "rope_theta": 10000000,
                "rope_type": "default",
                "partial_rotary_factor": 0.25,
                "mrope_interleaved": true,
                "mrope_section": [11, 11, 10]
            }
        }
    })
}

/// The declared-but-previously-unparsed hybrid fields all land on
/// `ModelConfig`, verbatim.
#[test]
fn hybrid_linear_attention_fields_are_parsed_verbatim() {
    let arch = detect_from_json(&qwen35_shaped_config());
    let cfg = arch.config();

    assert_eq!(cfg.model_type, "qwen3_5_text");
    assert_eq!(cfg.linear_conv_kernel_dim, Some(4));
    assert_eq!(cfg.linear_key_head_dim, Some(128));
    assert_eq!(cfg.linear_value_head_dim, Some(128));
    assert_eq!(cfg.linear_num_key_heads, Some(16));
    assert_eq!(cfg.linear_num_value_heads, Some(48));
    assert_eq!(cfg.mamba_ssm_dtype.as_deref(), Some("float32"));
    assert_eq!(cfg.attn_output_gate, Some(true));
    assert_eq!(cfg.output_gate_type.as_deref(), Some("swish"));
    assert_eq!(cfg.mtp_num_hidden_layers, Some(1));
    assert_eq!(cfg.mtp_use_dedicated_embeddings, Some(false));
    assert_eq!(cfg.mrope_interleaved, Some(true));
    assert_eq!(cfg.mrope_section, Some(vec![11, 11, 10]));
    assert_eq!(
        cfg.layer_types.as_deref(),
        Some(
            [
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
                "linear_attention",
                "linear_attention",
                "full_attention",
            ]
            .map(str::to_string)
            .as_slice()
        )
    );
}

/// `qwen3_5*` routes through the existing `qwen`-prefix match — no new
/// registry entry, matching the family's own Qwen2/2.5/3 convention.
#[test]
fn qwen35_uses_the_qwen_prefix_route() {
    let arch = detect_from_json(&qwen35_shaped_config());
    assert_eq!(arch.family(), "qwen3_5_text");
}

/// `partial_rotary_factor` is read when the checkpoint declares it at
/// both the flat `text_config` spot and under `rope_parameters` — the
/// Qwen3.8 shape, which this fixture mirrors.
#[test]
fn partial_rotary_factor_is_read_from_the_declared_spot() {
    let arch = detect_from_json(&qwen35_shaped_config());
    assert_eq!(arch.config().partial_rotary_factor, Some(0.25));
}

/// **The real Qwen3.5 shape declares the fraction under `rope_parameters`
/// only.** Every Qwen3.5 checkpoint on the Hub (0.8B through 397B-A17B)
/// omits the flat key; Qwen3.8 happens to write both, which is the only
/// reason the test above ever passed — it pinned a coincidence of the
/// fixture, not the parser. transformers 5.5.0 reads the fraction from
/// `rope_parameters` (`modeling_rope_utils.py`, every `_compute_*`
/// function: `rope_parameters_dict.get("partial_rotary_factor", 1.0)`),
/// so a parser that only reads the flat spelling rotates all 256 head
/// dims of a Qwen3.5 layer where the checkpoint asks for 64.
#[test]
fn the_nested_only_partial_rotary_spelling_is_read() {
    let mut config = qwen35_shaped_config();
    config["text_config"]
        .as_object_mut()
        .unwrap()
        .remove("partial_rotary_factor");
    assert!(config["text_config"]["rope_parameters"]["partial_rotary_factor"].is_number());
    let arch = detect_from_json(&config);
    assert_eq!(arch.config().partial_rotary_factor, Some(0.25));
}

/// When both spellings are present the flat one wins, because that is
/// what the reference does: `PreTrainedConfig::standardize_rope_params`
/// copies a top-level `partial_rotary_factor` INTO `rope_parameters`,
/// overwriting whatever the block declared. VINDEX3 blocks the
/// disagreement separately (`two_disagreeing_partial_rotary_spellings_block`);
/// this pins which value the engine path resolves so that the two doors
/// agree on what "the declared fraction" is.
#[test]
fn the_flat_partial_rotary_spelling_overrides_the_nested_one() {
    let mut config = qwen35_shaped_config();
    config["text_config"]["partial_rotary_factor"] = serde_json::json!(0.5);
    let arch = detect_from_json(&config);
    assert_eq!(arch.config().partial_rotary_factor, Some(0.5));
}

/// An absent hybrid block (an ordinary dense Qwen3 config) leaves every
/// new field `None` — presence, not a family default, is what turns them
/// into a fact.
#[test]
fn a_dense_qwen_config_declares_none_of_the_hybrid_fields() {
    let config = serde_json::json!({
        "model_type": "qwen3",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    });
    let arch = detect_from_json(&config);
    let cfg = arch.config();
    assert_eq!(cfg.linear_conv_kernel_dim, None);
    assert_eq!(cfg.linear_key_head_dim, None);
    assert_eq!(cfg.linear_value_head_dim, None);
    assert_eq!(cfg.linear_num_key_heads, None);
    assert_eq!(cfg.linear_num_value_heads, None);
    assert_eq!(cfg.mamba_ssm_dtype, None);
    assert_eq!(cfg.attn_output_gate, None);
    assert_eq!(cfg.output_gate_type, None);
    assert_eq!(cfg.mtp_num_hidden_layers, None);
    assert_eq!(cfg.mtp_use_dedicated_embeddings, None);
    assert_eq!(cfg.mrope_interleaved, None);
    assert_eq!(cfg.mrope_section, None);
}

/// **The output gate is a resolved OPERATOR, not a repeated flag.**
///
/// `attn_output_gate: true` has to become the four facts a forward pass
/// needs — where the gate values come from, what activates them, how
/// they combine, and at which point in the attention block. A consumer
/// handed only the boolean would have to invent all four, and inventing
/// the placement is the one that silently changes the arithmetic:
/// gating after the output projection is a different function from
/// gating before it.
#[test]
fn the_qwen_output_gate_resolves_to_a_complete_operator() {
    use crate::config::attention_gate::{GateActivation, GateCombine, GatePlacement, GateSource};

    let arch = detect_from_json(&qwen35_shaped_config());
    let gate = arch
        .attention_output_gate()
        .expect("`attn_output_gate: true` must resolve to an operator");
    assert_eq!(gate.source, GateSource::FusedQueryProjection);
    assert_eq!(gate.activation, GateActivation::Sigmoid);
    assert_eq!(gate.combine, GateCombine::ElementwiseMultiply);
    assert_eq!(
        gate.placement,
        GatePlacement::AfterAggregationBeforeOutputProjection
    );

    // Declared FALSE is not the same as declared absent, and neither
    // yields an operator: a gate that multiplies by nothing is not a
    // gate, and inventing one would change every attention output.
    let mut off = qwen35_shaped_config();
    off["text_config"]["attn_output_gate"] = serde_json::json!(false);
    assert!(
        detect_from_json(&off).attention_output_gate().is_none(),
        "an explicitly disabled gate must not resolve to an operator"
    );

    let mut absent = qwen35_shaped_config();
    absent["text_config"]
        .as_object_mut()
        .expect("text_config")
        .remove("attn_output_gate");
    assert!(
        detect_from_json(&absent).attention_output_gate().is_none(),
        "an undeclared gate must not be invented"
    );
}

/// **Qwen stores its norm weights as an OFFSET from one.**
///
/// `w` on this family means `1 + w`, so a consumer applying the raw
/// stored value normalises by roughly the wrong factor everywhere —
/// plausible output, wrong model. The offset is a property of the
/// family's spelling, so it is answered from `model_type` rather than
/// from anything a caller passes in.
#[test]
fn qwen_norm_weights_are_stored_as_an_offset_from_one() {
    let arch = detect_from_json(&qwen35_shaped_config());
    assert_eq!(
        arch.norm_weight_offset(),
        1.0,
        "the qwen family stores `w` meaning `1 + w`"
    );
    // The per-head Q/K norms are the same class, so they must agree —
    // a build that offset one and not the other would be internally
    // inconsistent in a way no single test would catch.
    assert_eq!(arch.qk_norm_weight_offset(), arch.norm_weight_offset());
}

/// **A partially declared recurrence yields no topology.**
///
/// Every dimension must be present. Completing a missing one with a
/// default would produce widths that are arithmetically self-consistent
/// and physically wrong — `qkv_channels` would close against a head
/// count the checkpoint never declared, and the fused projection would
/// be sliced in the wrong places.
///
/// Driven through the real detection path rather than a hand-built
/// config, so the fields under test are the ones a checkpoint actually
/// produces.
#[test]
fn a_partially_declared_recurrence_yields_no_topology() {
    use crate::inventory::report::{LinearAttentionTopology, RecurrentStateDtype};

    let full = detect_from_json(&qwen35_shaped_config());
    let got = LinearAttentionTopology::from_config(full.config())
        .expect("the fixture declares every dimension");
    // Qwen3.5's geometry: q and k at the KEY side, v at the value's.
    assert_eq!(got.qkv_channels(), 2 * 16 * 128 + 48 * 128);
    assert_eq!(got.value_width(), 48 * 128);
    assert_eq!(got.conv_kernel, 4);
    assert_eq!(got.state_dtype, Some(RecurrentStateDtype::Float32));

    // Drop each required dimension in turn; each one alone refuses.
    for key in [
        "linear_num_key_heads",
        "linear_key_head_dim",
        "linear_num_value_heads",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
    ] {
        let mut cfg = qwen35_shaped_config();
        cfg["text_config"]
            .as_object_mut()
            .expect("text_config")
            .remove(key);
        let arch = detect_from_json(&cfg);
        assert!(
            LinearAttentionTopology::from_config(arch.config()).is_none(),
            "a recurrence missing `{key}` must be refused, not completed with a default"
        );
    }

    // The state dtype is the ONE optional fact: undeclared, or spelled
    // in a way this build does not represent, still yields a topology —
    // with `None`, which says so rather than assuming the model's own
    // precision.
    for spelling in [None, Some("bfloat16")] {
        let mut cfg = qwen35_shaped_config();
        match spelling {
            Some(s) => cfg["text_config"]["mamba_ssm_dtype"] = serde_json::json!(s),
            None => {
                cfg["text_config"]
                    .as_object_mut()
                    .expect("text_config")
                    .remove("mamba_ssm_dtype");
            }
        }
        let arch = detect_from_json(&cfg);
        let got = LinearAttentionTopology::from_config(arch.config())
            .expect("the dtype is optional; the dimensions are not");
        assert_eq!(got.state_dtype, None, "for {spelling:?}");
    }
}
