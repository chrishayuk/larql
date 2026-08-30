//! Two independent checkpoints must resolve to the **same KDA vocabulary
//! at different geometry** — that is what makes the operator generic
//! rather than "Kimi support".
//!
//! Kimi Linear proves the operator exists. GLM-5.3-Flash proves it is not
//! shaped around Kimi: different head count, different hidden size,
//! different index base, same fifteen operands and same contracts. A width
//! or rank assumption baked in for one is exposed by the other here,
//! before any execution code is written.

use super::linear_attn::KdaGeometry;

/// Kimi Linear 48B-A3B: 32 heads × 128, conv kernel 4.
fn kimi_section() -> serde_json::Value {
    serde_json::json!({
        "num_heads": 32,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
    })
}

/// GLM-5.3-Flash: 64 heads × 128, conv kernel 4.
fn glm_section() -> serde_json::Value {
    serde_json::json!({
        "num_heads": 64,
        "head_dim": 128,
        "short_conv_kernel_size": 4,
        "gate_lower_bound": -5.0,
    })
}

#[test]
fn both_checkpoints_resolve_the_same_vocabulary_at_different_geometry() {
    let kimi = KdaGeometry::read(&kimi_section()).expect("Kimi declares a KDA block");
    let glm = KdaGeometry::read(&glm_section()).expect("GLM declares a KDA block");

    // Same vocabulary: every field answers for both.
    assert_eq!(kimi.head_dim, glm.head_dim, "both are 128-wide per head");
    assert_eq!(kimi.conv_kernel, glm.conv_kernel);

    // Different geometry: the head count, and everything derived from it.
    assert_ne!(kimi.num_heads, glm.num_heads);
    assert_eq!(kimi.value_width(), 32 * 128);
    assert_eq!(glm.value_width(), 64 * 128);
    assert_ne!(
        kimi.value_width(),
        glm.value_width(),
        "a width assumption baked in for one model must not satisfy the other"
    );

    // The state a KV planner would size, per layer. GLM's is exactly twice
    // Kimi's because only the head count differs — stated so a change to
    // either side of the product is visible here.
    assert_eq!(kimi.state_elements(), 32 * 128 * 128);
    assert_eq!(glm.state_elements(), 64 * 128 * 128);
}

/// `dt_bias` is the operand whose geometry separates KDA from Gated
/// DeltaNet: `[Hv·Dv]` against `[Hv]`. Pinned on both checkpoints, because
/// it is the contract a merge of the two operators would have to break.
#[test]
fn the_dt_bias_contract_is_per_channel_on_both_checkpoints() {
    for (name, section, heads) in [
        ("Kimi Linear", kimi_section(), 32usize),
        ("GLM-5.3-Flash", glm_section(), 64),
    ] {
        let g = KdaGeometry::read(&section).expect("declared");
        assert_eq!(
            g.value_width(),
            heads * 128,
            "{name}: dt_bias is per channel, not per head"
        );
        assert_ne!(
            g.value_width(),
            g.num_heads,
            "{name}: a per-head dt_bias would make this a Gated DeltaNet block"
        );
    }
}

/// A partial declaration is refused, never completed with defaults — the
/// same contract `LinearAttentionTopology::from_config` holds.
#[test]
fn a_partial_declaration_is_refused() {
    for missing in ["num_heads", "head_dim", "short_conv_kernel_size"] {
        let mut section = kimi_section();
        section.as_object_mut().unwrap().remove(missing);
        assert!(
            KdaGeometry::read(&section).is_none(),
            "a block missing `{missing}` must not resolve"
        );
    }
}

/// A zero dimension is not a declaration. GLM-5.3-Flash writes
/// `head_dim: 0` at the *text* level for exactly this reason — MLA carries
/// its geometry elsewhere — so a zero reaching the KDA reader means the
/// wrong section was handed to it.
#[test]
fn a_zero_dimension_is_refused() {
    for zeroed in ["num_heads", "head_dim", "short_conv_kernel_size"] {
        let mut section = kimi_section();
        section[zeroed] = serde_json::json!(0);
        assert!(
            KdaGeometry::read(&section).is_none(),
            "`{zeroed}: 0` must not resolve"
        );
    }
}

/// A Gated DeltaNet checkpoint presented to the KDA reader must resolve
/// nothing.
///
/// Qwen3.8 declares its recurrence in `linear_*` keys at the text level
/// and carries no `linear_attn_config` section at all, so the reader is
/// handed an absent value. If this ever starts resolving, the two
/// operators have been merged by accident.
#[test]
fn a_gated_deltanet_checkpoint_resolves_no_kda_block() {
    let qwen_text_config = serde_json::json!({
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "mamba_ssm_dtype": "float32",
    });
    assert!(
        KdaGeometry::read(&qwen_text_config["linear_attn_config"]).is_none(),
        "Gated DeltaNet's geometry must not read as a KDA block"
    );
    // And the reverse: KDA's section declares none of Qwen3.8's keys.
    let kimi = kimi_section();
    for qwen_key in [
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_conv_kernel_dim",
    ] {
        assert!(kimi.get(qwen_key).is_none(), "{qwen_key} is not a KDA key");
    }
}
