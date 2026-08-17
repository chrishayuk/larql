//! `rope_scaling` / `rope_parameters` parsing — covers bugs 1 and 3 from
//! the shannon cross-engine diagnostic, plus the transformers-5.x flat
//! `rope_parameters` form the Muse-Glimmer inventory caught falling through
//! to the default.

use super::support::minimal_llama_config;
use crate::detect::*;

#[test]
fn gemma3_rope_scaling_structured_per_layer_type_parses() {
    // HF's `Gemma3TextConfig` expands the flat `rope_scaling = {factor: 8,
    // rope_type: linear}` to a structured per-layer-type dict. Some on-disk
    // dumps include the structured form directly; the parser must lift the
    // `full_attention` slot and mark `gemma3_global_only`.
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "gemma3",
        "text_config": {
            "model_type": "gemma3_text",
            "hidden_size": 2560,
            "head_dim": 256,
            "num_hidden_layers": 34,
            "num_attention_heads": 8,
            "intermediate_size": 10240,
            "sliding_window": 1024,
            "rope_scaling": {
                "full_attention": {"rope_type": "linear", "factor": 8.0},
                "sliding_attention": {"rope_type": "default"},
            },
        },
    }));
    let rs = arch
        .config()
        .rope_scaling
        .as_ref()
        .expect("rope_scaling parsed");
    assert_eq!(rs.scaling_type, "linear");
    assert_eq!(rs.factor, 8.0);
    assert!(
        rs.gemma3_global_only,
        "structured form must set gemma3_global_only"
    );
}

#[test]
fn gemma3_arch_rope_position_divisor_only_on_global_layers() {
    // The Gemma 3 4B fix: divide RoPE positions by `factor` only on
    // full-attention (global) layers — layers 5, 11, 17, 23, 29 in the
    // standard 34-layer pattern (every 6th). Sliding layers stay at 1.0.
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "gemma3",
        "text_config": {
            "model_type": "gemma3_text",
            "hidden_size": 2560,
            "head_dim": 256,
            "num_hidden_layers": 34,
            "num_attention_heads": 8,
            "intermediate_size": 10240,
            "sliding_window": 1024,
            "rope_scaling": {
                "full_attention": {"rope_type": "linear", "factor": 8.0},
                "sliding_attention": {"rope_type": "default"},
            },
        },
    }));
    // Layer 5: global (5 + 1 = 6, multiple of 6) → factor.
    assert_eq!(arch.rope_position_divisor_for_layer(5), 8.0);
    assert_eq!(arch.rope_position_divisor_for_layer(11), 8.0);
    // Layer 0, 1, 4: sliding → 1.0 (no scaling).
    assert_eq!(arch.rope_position_divisor_for_layer(0), 1.0);
    assert_eq!(arch.rope_position_divisor_for_layer(4), 1.0);
    assert_eq!(arch.rope_position_divisor_for_layer(6), 1.0);
}

#[test]
fn llama3_rope_scaling_parsed_with_all_four_fields() {
    // Llama-3.2's config ships the full wavelength-band parameter set.
    // The parser must capture all four optional fields on RopeScaling.
    let arch = detect_from_json(&minimal_llama_config(serde_json::json!({
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    })));
    let rs = arch
        .config()
        .rope_scaling
        .as_ref()
        .expect("rope_scaling parsed");
    assert_eq!(rs.scaling_type, "llama3");
    assert_eq!(rs.factor, 32.0);
    assert_eq!(rs.llama3_low_freq_factor, Some(1.0));
    assert_eq!(rs.llama3_high_freq_factor, Some(4.0));
    assert_eq!(rs.llama3_original_max_position_embeddings, Some(8192.0));
    assert!(!rs.gemma3_global_only);
}

#[test]
fn llama_arch_returns_llama3_rope_scaling_when_configured() {
    // The arch method exposes the parsed scaling to the forward path. With
    // `rope_type=llama3`, the four wavelength-band parameters must flow
    // through unchanged. With `rope_type=default`, the method returns None.
    let arch = detect_from_json(&minimal_llama_config(serde_json::json!({
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        },
    })));
    let scaling = arch
        .llama3_rope_scaling()
        .expect("llama3 scaling exposed by arch");
    assert_eq!(scaling.factor, 32.0);
    assert_eq!(scaling.low_freq_factor, 1.0);
    assert_eq!(scaling.high_freq_factor, 4.0);
    assert_eq!(scaling.original_max_position_embeddings, 8192.0);

    // Non-llama3 rope_type → arch returns None.
    let arch_linear = detect_from_json(&minimal_llama_config(serde_json::json!({
        "rope_scaling": {"rope_type": "linear", "factor": 2.0},
    })));
    assert!(arch_linear.llama3_rope_scaling().is_none());
}

#[test]
fn transformers5_flat_rope_parameters_is_read() {
    // The transformers-5.x flat form: `rope_parameters: {rope_theta: N}`
    // replaces the legacy top-level `rope_theta`. This fell through to the
    // 10000 default until the Muse-Glimmer inventory caught a checkpoint
    // declaring θ=500000 and resolving θ=10000.
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "some_transformers5_model",
        "hidden_size": 6656,
        "num_hidden_layers": 52,
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "intermediate_size": 19968,
        "rope_parameters": { "rope_theta": 500000.0, "rope_type": "default" }
    }));
    assert_eq!(arch.config().rope_base, 500_000.0);
}

#[test]
fn structured_rope_parameters_beat_the_flat_form() {
    // Gemma-4-style structured per-layer-type declaration is more specific
    // than a flat rope_theta sitting next to it; specificity wins.
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "some_transformers5_model",
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "intermediate_size": 4096,
        "rope_parameters": {
            "rope_theta": 10000.0,
            "full_attention": { "rope_theta": 1000000.0 }
        }
    }));
    assert_eq!(arch.config().rope_base, 1_000_000.0);
}

#[test]
fn layer_rope_theta_resolves_per_layer_policies_with_nope() {
    use crate::config::PositionPolicy;
    // A hybrid checkpoint declaring per-layer thetas, with the upstream
    // 0.0 NoPE sentinel on the last layer. The sentinel is interpreted at
    // the trait boundary — nothing downstream sees a zero theta.
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "some_hybrid_nope_model",
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "layer_rope_theta": [500000.0, 500000.0, 500000.0, 0.0]
    }));
    assert_eq!(
        arch.position_policy_for_layer(0),
        PositionPolicy::Rope { theta: 500000.0 }
    );
    assert_eq!(arch.position_policy_for_layer(3), PositionPolicy::None);
    // Beyond the declared array (defensive): falls back to the uniform base.
    assert_eq!(
        arch.position_policy_for_layer(99),
        PositionPolicy::Rope {
            theta: arch.rope_base_for_layer(99)
        }
    );
}

#[test]
fn no_layer_array_means_uniform_rotary_policy() {
    use crate::config::PositionPolicy;
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "num_attention_heads": 8,
        "num_key_value_heads": 8,
        "rope_theta": 10000.0
    }));
    for layer in 0..2 {
        assert_eq!(
            arch.position_policy_for_layer(layer),
            PositionPolicy::Rope { theta: 10000.0 }
        );
    }
}

#[test]
fn flat_rope_parameters_beat_the_legacy_top_level_field() {
    // A checkpoint carrying both spellings is declaring the 5.x form as
    // current; the legacy field is the migration leftover.
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "some_transformers5_model",
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "intermediate_size": 4096,
        "rope_theta": 10000.0,
        "rope_parameters": { "rope_theta": 500000.0 }
    }));
    assert_eq!(arch.config().rope_base, 500_000.0);
}

#[test]
fn transformers5_rope_parameters_yarn_block_is_read_as_scaling() {
    // The 5.x block carries theta AND scaling. Before this was read, a
    // checkpoint declaring YaRN here had its theta honoured and its scaling
    // dropped at parse — served at the unscaled frequencies and the wrong
    // amplitude — which the VINDEX3 carriage gate caught on a
    // Glimmer-shaped fixture. Every leaf must land, not just the type.
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "some_transformers5_model",
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "intermediate_size": 4096,
        "rope_parameters": {
            "rope_theta": 150000.0,
            "rope_type": "yarn",
            "factor": 32.0,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "truncate": false,
            "original_max_position_embeddings": 4096
        }
    }));
    assert_eq!(arch.config().rope_base, 150_000.0);
    let yarn = arch
        .yarn_rope_scaling()
        .expect("a complete yarn block under rope_parameters is scaling");
    assert_eq!(yarn.factor, 32.0);
    assert_eq!(yarn.beta_fast, 32.0);
    assert_eq!(yarn.beta_slow, 1.0);
    assert!(!yarn.truncate);
    assert_eq!(yarn.original_max_position_embeddings, 4096.0);
    assert!(matches!(
        arch.position_policy_for_layer(0),
        crate::config::PositionPolicy::Yarn { theta, .. } if theta == 150_000.0
    ));
}

#[test]
fn unscaled_rope_parameters_leave_legacy_rope_scaling_reachable() {
    // A 5.x block that declares no scaling (`default`, no `factor`) is not
    // an answer of "no scaling" that silences the legacy key — the two are
    // consulted in specificity order, and a legacy block still parses.
    let arch = detect_from_json(&serde_json::json!({
        "model_type": "some_transformers5_model",
        "hidden_size": 1024,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "intermediate_size": 4096,
        "rope_parameters": { "rope_theta": 500000.0, "rope_type": "default" },
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 4.0,
            "original_max_position_embeddings": 8192
        }
    }));
    assert_eq!(arch.config().rope_base, 500_000.0);
    let yarn = arch.yarn_rope_scaling().expect("legacy block still read");
    assert_eq!(yarn.factor, 4.0);
    assert_eq!(yarn.original_max_position_embeddings, 8192.0);
}
