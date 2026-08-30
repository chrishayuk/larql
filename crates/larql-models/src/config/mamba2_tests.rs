//! Tests for [`super::mamba2`] — the declared Mamba2 mixer geometry.

use super::mamba2::{DtBound, Mamba2Geometry};
use serde_json::json;

fn mamba2_780m() -> serde_json::Value {
    json!({
        "state_size": 128,
        "num_heads": 48,
        "head_dim": 64,
        "expand": 2,
        "conv_kernel": 4,
        "n_groups": 1,
        "chunk_size": 256,
        "time_step_limit": [0.0, "Infinity"],
        "rms_norm": true,
        "use_bias": false,
        "use_conv_bias": true
    })
}

/// **The declared geometry closes over the real estate exactly.** These
/// are the observed mamba2-780m tensor shapes, not derived round-trips:
/// `mixer.in_proj.weight` is `[6448, 1536]`, `mixer.conv1d.weight` is
/// `[3328, 1, 4]`, and the state is 48 heads of `64 × 128`.
#[test]
fn the_derived_widths_close_over_the_real_estate() {
    let g = Mamba2Geometry::read(&mamba2_780m()).expect("complete declaration");
    let hidden = 1536;
    assert_eq!(g.d_inner(hidden), 3072);
    assert_eq!(g.conv_dim(hidden), 3328, "the conv1d row count");
    assert_eq!(g.in_proj_rows(hidden), 6448, "the in_proj row count");
    assert_eq!(g.state_elements(), 48 * 64 * 128);
    assert!(g.geometry_defects(hidden).is_empty());
}

/// **The judged non-finite boundary reaches the semantics.** The config's
/// bare `Infinity` arrives as the string it spells; here it becomes a
/// declared unbounded clamp side — never a fabricated float.
#[test]
fn the_unbounded_dt_clamp_is_a_declared_fact() {
    let g = Mamba2Geometry::read(&mamba2_780m()).unwrap();
    assert_eq!(g.dt_limit_min, DtBound::Finite(0.0));
    assert_eq!(g.dt_limit_max, DtBound::Unbounded);
    assert_eq!(
        DtBound::from_declared(&json!("-Infinity")),
        Some(DtBound::Unbounded)
    );
}

/// **`NaN` bounds nothing and is refused**, as is any other string.
#[test]
fn nan_and_unknown_spellings_refuse() {
    assert_eq!(DtBound::from_declared(&json!("NaN")), None);
    assert_eq!(DtBound::from_declared(&json!("inf")), None);
    assert_eq!(DtBound::from_declared(&json!(null)), None);
    let mut config = mamba2_780m();
    config["time_step_limit"] = json!([0.0, "NaN"]);
    assert_eq!(Mamba2Geometry::read(&config), None);
}

/// **All fields or none.** Removing any single field refuses the whole
/// declaration rather than completing it with a default.
#[test]
fn a_partial_declaration_is_refused() {
    for field in [
        "state_size",
        "num_heads",
        "head_dim",
        "expand",
        "conv_kernel",
        "n_groups",
        "chunk_size",
        "time_step_limit",
        "rms_norm",
        "use_bias",
        "use_conv_bias",
    ] {
        let mut config = mamba2_780m();
        config.as_object_mut().unwrap().remove(field);
        assert_eq!(
            Mamba2Geometry::read(&config),
            None,
            "removing `{field}` must refuse the declaration"
        );
    }
    // A malformed limit array refuses too — two bounds or nothing.
    let mut config = mamba2_780m();
    config["time_step_limit"] = json!([0.0]);
    assert_eq!(Mamba2Geometry::read(&config), None);
}

/// **A geometry that does not close names its defect.** 47 heads of 64
/// cannot tile a 3072-wide inner dimension.
#[test]
fn a_non_closing_geometry_names_its_defect() {
    let mut config = mamba2_780m();
    config["num_heads"] = json!(47);
    let g = Mamba2Geometry::read(&config).unwrap();
    let defects = g.geometry_defects(1536);
    assert_eq!(defects.len(), 1);
    assert!(defects[0].contains("does not close over d_inner"));
}

/// The geometry round-trips through serde, `Unbounded` included — the
/// container must be able to carry the unbounded side without an IEEE
/// infinity appearing in JSON.
#[test]
fn the_geometry_round_trips_through_serde() {
    let g = Mamba2Geometry::read(&mamba2_780m()).unwrap();
    let text = serde_json::to_string(&g).unwrap();
    assert!(!text.contains("inf"), "no IEEE infinity in the wire form");
    let back: Mamba2Geometry = serde_json::from_str(&text).unwrap();
    assert_eq!(back, g);
}
