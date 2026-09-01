//! **`no_rope_layers` is named for what it disables and its values say
//! the opposite.**
//!
//! SmolLM3's config documents it as
//!
//! ```text
//! no_rope_layers (`List[int]`, *optional*):
//!     A `1` at an index position indicates that the corresponding layer
//!     will use RoPE, while a `0` indicates that it's a NoPE layer.
//! ```
//!
//! and both SmolLM3 and Llama 4 read it as
//! `self.use_rope = config.no_rope_layers[layer_idx]`. So a reader who
//! trusts the key's name inverts the entire schedule.
//!
//! That is the whole reason this file exists. On SmolLM3-3B the declared
//! mask is `[1,1,1,0, 1,1,1,0, …]` over 36 layers, so an inverted reading
//! rotates the 27 layers that must not rotate and leaves unrotated the 9
//! that must — and the model still emits fluent text, because NoPE layers
//! are a minority and attention degrades gracefully. Nothing but a test
//! that pins the polarity against the reference catches it.
//!
//! The second fact under test is precedence: both references build the
//! mask from `no_rope_layer_interval` only `if no_rope_layers is None`,
//! so an explicit mask SUPERSEDES a declared interval rather than being
//! reconciled with it.

use larql_models::config::PositionPolicy;
use larql_models::detect::detect_from_json;
use serde_json::json;

const THETA: f64 = 5_000_000.0;

/// SmolLM3-shaped config; the schedule declarations vary.
fn config(
    layers: usize,
    no_rope_layers: Option<Vec<i64>>,
    interval: Option<usize>,
) -> serde_json::Value {
    let mut config = json!({
        "model_type": "smollm3",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": layers,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": THETA,
    });
    if let Some(mask) = no_rope_layers {
        config["no_rope_layers"] = json!(mask);
    }
    if let Some(interval) = interval {
        config["no_rope_layer_interval"] = json!(interval);
    }
    config
}

/// `true` where the resolved policy rotates.
fn rotates(layers: usize, no_rope_layers: Option<Vec<i64>>, interval: Option<usize>) -> Vec<bool> {
    let arch = detect_from_json(&config(layers, no_rope_layers, interval));
    (0..layers)
        .map(|l| arch.position_policy_for_layer(l) != PositionPolicy::None)
        .collect()
}

#[test]
fn a_one_means_the_layer_rotates_and_a_zero_means_it_does_not() {
    // THE POLARITY, stated as baldly as possible. Reading the key's name
    // instead of the reference produces exactly the inverse of this.
    assert_eq!(
        rotates(4, Some(vec![1, 1, 1, 0]), None),
        vec![true, true, true, false],
        "`1` uses RoPE and `0` is NoPE — the key's name is the opposite of its values"
    );
}

#[test]
fn the_real_smollm3_schedule_leaves_every_fourth_layer_unrotated() {
    // SmolLM3-3B's actual declaration: 36 layers, NoPE every 4th.
    let layers = 36;
    let mask: Vec<i64> = (0..layers).map(|i| i64::from((i + 1) % 4 != 0)).collect();
    let rotates = rotates(layers, Some(mask), Some(4));

    let unrotated: Vec<usize> = (0..layers).filter(|&i| !rotates[i]).collect();
    assert_eq!(
        unrotated,
        vec![3, 7, 11, 15, 19, 23, 27, 31, 35],
        "every 4th layer is NoPE; got {unrotated:?}"
    );
    assert_eq!(
        rotates.iter().filter(|r| **r).count(),
        27,
        "the other 27 rotate — the count an inverted reading would swap"
    );
}

#[test]
fn the_interval_alone_generates_the_same_schedule() {
    // The fallback both references use when the mask is absent:
    // `int((layer_idx + 1) % interval != 0)`.
    assert_eq!(
        rotates(8, None, Some(4)),
        rotates(8, Some(vec![1, 1, 1, 0, 1, 1, 1, 0]), None),
        "interval and mask must describe the same schedule"
    );
}

#[test]
fn an_explicit_mask_supersedes_a_disagreeing_interval() {
    // Precedence, not reconciliation. Upstream consults the interval only
    // `if no_rope_layers is None`, so a checkpoint declaring both runs the
    // MASK. Preferring the interval, or intersecting the two, would put
    // this build on a schedule the reference never runs.
    let mask = vec![1, 1, 1, 1];
    assert_eq!(
        rotates(4, Some(mask), Some(2)),
        vec![true, true, true, true],
        "the mask wins; an interval of 2 would have unrotated layers 1 and 3"
    );
}

#[test]
fn a_stack_with_no_schedule_declared_rotates_throughout() {
    // The control that bounds the change: almost every checkpoint in the
    // corpus declares neither key, and must be completely unaffected.
    assert_eq!(rotates(4, None, None), vec![true; 4]);
}

#[test]
fn a_scheduled_layer_still_rotates_at_the_declared_base() {
    // The schedule answers WHETHER a layer rotates, not HOW. A rotating
    // layer must still pick up the checkpoint's base — resolving it to a
    // bare `Rope { theta: default }` would be a silent re-basing of every
    // rotating layer.
    let arch = detect_from_json(&config(4, Some(vec![1, 0, 1, 0]), None));
    assert_eq!(
        arch.position_policy_for_layer(0),
        PositionPolicy::Rope { theta: THETA }
    );
    assert_eq!(arch.position_policy_for_layer(1), PositionPolicy::None);
}

#[test]
fn a_mask_shorter_than_the_stack_says_nothing_about_the_rest() {
    // Both references document "at least the same length as the number of
    // layers" and index directly. A short mask is malformed; the layers it
    // does not mention keep the unscheduled behaviour rather than
    // acquiring a guessed one.
    assert_eq!(
        rotates(4, Some(vec![0, 0]), None),
        vec![false, false, true, true],
        "layers past the mask's end are not described by it"
    );
}
