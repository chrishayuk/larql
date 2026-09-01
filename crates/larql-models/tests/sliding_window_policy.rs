//! **A declared sliding window that is declared off is off.**
//!
//! Qwen states two independent facts about this feature and VINDEX3 read
//! only one of them:
//!
//! ```text
//! sliding_window       32768   the window, when there is one
//! use_sliding_window   false    whether it applies at all
//! max_window_layers    24       how far up the stack it applies
//! ```
//!
//! `sliding_window_size()` returned the window unconditionally. That was
//! not a live wrong answer on any checkpoint measured — Qwen2.5 declares
//! no sliding `layer_types`, so no layer reached the window — but the
//! safety was accidental, and the adversarial case below is an ordinary
//! config a family could ship tomorrow.
//!
//! The controls matter as much as the fix: it must remain possible to
//! *enable* a window, or "honour the disable flag" quietly becomes
//! "ignore sliding-window declarations".

use larql_models::detect::detect_from_json;
use serde_json::json;

/// Qwen-shaped config with the sliding-window declarations under test.
fn config(
    sliding_window: serde_json::Value,
    use_sliding_window: Option<bool>,
    max_window_layers: Option<usize>,
    layer_types: Option<Vec<&str>>,
) -> serde_json::Value {
    let mut config = json!({
        "model_type": "qwen3",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "sliding_window": sliding_window,
    });
    if let Some(flag) = use_sliding_window {
        config["use_sliding_window"] = json!(flag);
    }
    if let Some(bound) = max_window_layers {
        config["max_window_layers"] = json!(bound);
    }
    if let Some(types) = layer_types {
        config["layer_types"] = json!(types);
    }
    config
}

#[test]
fn qwen25_declares_a_window_and_disables_it() {
    // The real Qwen2.5-0.5B shape: a 32768 window, switched off.
    let arch = detect_from_json(&config(json!(32768), Some(false), Some(24), None));
    assert_eq!(
        arch.sliding_window_size(),
        None,
        "a window declared beside `use_sliding_window: false` is not in effect"
    );
}

#[test]
fn a_disabled_window_beats_a_sliding_layer_schedule() {
    // The adversarial case, and the reason this is a correctness fix
    // rather than admission bookkeeping: a checkpoint stating BOTH a
    // sliding interleave and the disable flag. Before, `layer_types`
    // would have made layers sliding and `sliding_window_size` would have
    // handed them a window the checkpoint said not to use.
    let arch = detect_from_json(&config(
        json!(32768),
        Some(false),
        Some(4),
        Some(vec![
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "full_attention",
        ]),
    ));
    assert_eq!(arch.sliding_window_size(), None);
    for layer in 0..4 {
        assert!(
            !arch.is_sliding_window_layer(layer),
            "layer {layer} must not slide when the checkpoint disables the feature"
        );
    }
}

#[test]
fn control_an_enabled_window_still_slides() {
    // Without this the fix could degenerate into ignoring the feature.
    let arch = detect_from_json(&config(
        json!(4096),
        Some(true),
        None,
        Some(vec![
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "full_attention",
        ]),
    ));
    assert_eq!(arch.sliding_window_size(), Some(4096));
    assert!(arch.is_sliding_window_layer(0));
    assert!(arch.is_sliding_window_layer(1));
    assert!(!arch.is_sliding_window_layer(2));
}

#[test]
fn control_a_family_stating_no_flag_is_unaffected() {
    // `None` is not `Some(false)`. Every family before Qwen leaves the
    // question to the window's own presence, and must keep doing so.
    let arch = detect_from_json(&config(
        json!(1024),
        None,
        None,
        Some(vec![
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ]),
    ));
    assert_eq!(arch.sliding_window_size(), Some(1024));
    assert!(arch.is_sliding_window_layer(0));
    assert!(!arch.is_sliding_window_layer(1));
}

#[test]
fn a_declared_null_window_is_deliberate_absence() {
    // Qwen3's shape: the key is present and explicitly null.
    let arch = detect_from_json(&config(json!(null), Some(false), Some(4), None));
    assert_eq!(arch.sliding_window_size(), None);
    for layer in 0..4 {
        assert!(!arch.is_sliding_window_layer(layer));
    }
}

#[test]
fn max_window_layers_bounds_an_enabled_window() {
    // The bottom `n` layers slide; the rest attend fully.
    let arch = detect_from_json(&config(
        json!(4096),
        Some(true),
        Some(2),
        Some(vec!["sliding_attention"; 4]),
    ));
    assert_eq!(arch.sliding_window_size(), Some(4096));
    assert!(arch.is_sliding_window_layer(0));
    assert!(arch.is_sliding_window_layer(1));
    assert!(
        !arch.is_sliding_window_layer(2),
        "layer 2 is above max_window_layers and must attend fully"
    );
    assert!(!arch.is_sliding_window_layer(3));
}
