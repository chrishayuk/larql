//! The flat `gemma3_text` config (Gemma 3 270M / 1B: no `text_config`
//! wrapper) states its sliding/full interleave at the root, either as an
//! explicit `layer_types` array (270M) or as the `sliding_window_pattern`
//! period (1B). `Gemma3Arch` resolves the interleave in HF's order of
//! authority — array, then declared period, then the class default of 6
//! — instead of hard-wiring the default, so a declaration that differs
//! from the default is honoured rather than overridden.

use larql_models::{
    config::is_full_at_period, detect_from_json, ModelArchitecture, LAYER_TYPE_FULL_ATTENTION,
    LAYER_TYPE_SLIDING_ATTENTION,
};

/// The 270M/1B geometry, flat at the root.
const LAYERS: usize = 18;
const GLOBAL_THETA: f64 = 1_000_000.0;
const LOCAL_THETA: f64 = 10_000.0;
/// A period other than the class default, so honouring the declaration
/// and falling back to the default answer differently.
const DECLARED_PERIOD: usize = 3;
/// HF's class-default period (`Gemma3TextConfig`).
const CLASS_DEFAULT_PERIOD: usize = 6;

fn flat_gemma3(extra: serde_json::Value) -> Box<dyn ModelArchitecture> {
    let mut config = serde_json::json!({
        "model_type": "gemma3_text",
        "hidden_size": 640, "num_hidden_layers": LAYERS, "intermediate_size": 2048,
        "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 256,
        "sliding_window": 512,
        "rope_theta": GLOBAL_THETA,
        "rope_local_base_freq": LOCAL_THETA,
    });
    for (key, value) in extra.as_object().unwrap() {
        config[key] = value.clone();
    }
    detect_from_json(&config)
}

fn full_layers(arch: &dyn ModelArchitecture) -> Vec<usize> {
    (0..LAYERS)
        .filter(|&layer| !arch.is_sliding_window_layer(layer))
        .collect()
}

fn layers_at_period(period: usize) -> Vec<usize> {
    (0..LAYERS)
        .filter(|&layer| is_full_at_period(layer, period))
        .collect()
}

#[test]
fn the_flat_spelling_resolves_to_the_gemma3_family() {
    assert_eq!(flat_gemma3(serde_json::json!({})).family(), "gemma3");
}

#[test]
fn an_undeclared_interleave_takes_the_class_default_period() {
    let arch = flat_gemma3(serde_json::json!({}));
    assert_eq!(
        full_layers(arch.as_ref()),
        layers_at_period(CLASS_DEFAULT_PERIOD)
    );
}

#[test]
fn a_declared_period_is_honoured_over_the_class_default() {
    let arch = flat_gemma3(serde_json::json!({ "sliding_window_pattern": DECLARED_PERIOD }));
    assert_eq!(
        full_layers(arch.as_ref()),
        layers_at_period(DECLARED_PERIOD)
    );
}

/// A zero period is not a stride: the class default answers instead.
#[test]
fn a_zero_period_falls_back_to_the_class_default() {
    let arch = flat_gemma3(serde_json::json!({ "sliding_window_pattern": 0 }));
    assert_eq!(
        full_layers(arch.as_ref()),
        layers_at_period(CLASS_DEFAULT_PERIOD)
    );
}

/// The array outranks the period, as in `Gemma3TextConfig.__post_init__`.
#[test]
fn a_declared_layer_types_array_outranks_the_period() {
    let types: Vec<&str> = (0..LAYERS)
        .map(|layer| {
            if is_full_at_period(layer, DECLARED_PERIOD) {
                LAYER_TYPE_FULL_ATTENTION
            } else {
                LAYER_TYPE_SLIDING_ATTENTION
            }
        })
        .collect();
    let arch = flat_gemma3(serde_json::json!({
        "layer_types": types,
        "sliding_window_pattern": CLASS_DEFAULT_PERIOD,
    }));
    assert_eq!(
        full_layers(arch.as_ref()),
        layers_at_period(DECLARED_PERIOD)
    );
}

/// The root `rope_theta` is the global base and `rope_local_base_freq`
/// the sliding one — the two halves of HF's per-layer-type rope table.
#[test]
fn the_root_rope_bases_split_by_layer_type() {
    let arch = flat_gemma3(serde_json::json!({}));
    for layer in 0..LAYERS {
        let expected = if arch.is_sliding_window_layer(layer) {
            LOCAL_THETA
        } else {
            GLOBAL_THETA
        };
        assert_eq!(arch.rope_base_for_layer(layer), expected, "layer {layer}");
    }
}

#[test]
fn a_zero_period_selects_no_layer() {
    assert!((0..LAYERS).all(|layer| !is_full_at_period(layer, 0)));
}
