//! The lowered session's rotary tables for a linear-scaled layer: the
//! `inv_freq` the Metal kernel rotates with is the interpreter's own
//! (plain series over the factor), and the table key separates a scaled
//! layer from a plain one at the same base — the pair Gemma 3's global
//! and sliding layers would otherwise be if they shared a theta.

use larql_models::config::PositionPolicy;

use super::super::resident::{rope_inv_freq_table, rope_table_key};

const HEAD_DIM: usize = 16;
/// Gemma 3's global base and declared factor.
const THETA: f64 = 1_000_000.0;
const FACTOR: f64 = 8.0;

#[test]
fn the_linear_table_is_the_plain_series_over_the_factor() {
    let plain = rope_inv_freq_table(&PositionPolicy::Rope { theta: THETA }, HEAD_DIM);
    let linear = rope_inv_freq_table(
        &PositionPolicy::Linear {
            theta: THETA,
            factor: FACTOR,
        },
        HEAD_DIM,
    );
    assert_eq!(plain.len(), HEAD_DIM / 2);
    assert_eq!(linear.len(), plain.len());
    for (i, (p, l)) in plain.iter().zip(&linear).enumerate() {
        let expected = p / FACTOR as f32;
        assert!(
            (l - expected).abs() <= f32::EPSILON * expected.abs().max(1.0),
            "slot {i}: linear {l} vs plain/factor {expected}"
        );
    }
    // And it is a different table: a kernel handed the plain one would
    // rotate the global layers eight times too fast.
    assert_ne!(plain, linear);
}

#[test]
fn the_linear_key_separates_scaled_from_plain_and_factor_from_factor() {
    let plain = rope_table_key(&PositionPolicy::Rope { theta: THETA }, HEAD_DIM);
    let eight = rope_table_key(
        &PositionPolicy::Linear {
            theta: THETA,
            factor: FACTOR,
        },
        HEAD_DIM,
    );
    let four = rope_table_key(
        &PositionPolicy::Linear {
            theta: THETA,
            factor: 4.0,
        },
        HEAD_DIM,
    );
    assert!(plain.is_some() && eight.is_some() && four.is_some());
    assert_ne!(plain, eight, "scaled and plain must not share a table");
    assert_ne!(eight, four, "two factors must not share a table");
    // Same policy, same key: the table is shared across layers that
    // genuinely rotate alike.
    assert_eq!(
        eight,
        rope_table_key(
            &PositionPolicy::Linear {
                theta: THETA,
                factor: FACTOR,
            },
            HEAD_DIM,
        )
    );
}
