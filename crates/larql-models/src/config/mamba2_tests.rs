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

/// The OuteAI Mamba2Attn declaration, verbatim where it matters: three
/// renamed geometry keys, `use_mamba2_bias`, and NO `n_groups`/`rms_norm`.
fn oute_250m() -> serde_json::Value {
    json!({
        "state_size": 128,
        "mamba2_num_heads": 32,
        "mamba2_head_dim": 64,
        "expand": 2,
        "mamba2_conv_kernel": 4,
        "chunk_size": 256,
        "time_step_limit": [0.0, "Infinity"],
        "use_mamba2_bias": false,
        "use_conv_bias": true
    })
}

/// **The mamba_ssm dialect reads into the same geometry, with its two
/// package defaults RECORDED, never silent.** A wrong default is then
/// findable: it is named in the provenance and still subject to the
/// cross-field closure a declared value faces.
#[test]
fn the_mamba_ssm_dialect_records_its_family_defaults() {
    let (g, provenance) = Mamba2Geometry::read_with_provenance(&oute_250m()).unwrap();
    assert_eq!(provenance.dialect, super::mamba2::Mamba2Dialect::MambaSsm);
    assert_eq!(g.num_heads, 32);
    assert_eq!(g.head_dim, 64);
    assert_eq!(g.conv_kernel, 4);
    assert!(!g.use_bias);
    // The two fields the dialect never spells, filled from mamba_ssm's
    // own defaults — each one on the record.
    assert_eq!(g.n_groups, 1);
    assert!(g.rms_norm);
    let defaulted: Vec<&str> = provenance
        .family_defaults
        .iter()
        .map(|d| d.key.as_str())
        .collect();
    assert_eq!(defaulted, ["n_groups", "rms_norm"]);
    // And the geometry closes over the real widths: conv_dim 2304,
    // in_proj rows 4384 on hidden 1024.
    assert_eq!(g.conv_dim(1024), 2304);
    assert_eq!(g.in_proj_rows(1024), 4384);
}

/// A declared value outranks the family default and leaves no record —
/// there is nothing defaulted to record.
#[test]
fn a_declared_value_outranks_the_family_default() {
    let mut config = oute_250m();
    config["n_groups"] = json!(2);
    config["rms_norm"] = json!(false);
    let (g, provenance) = Mamba2Geometry::read_with_provenance(&config).unwrap();
    assert_eq!(g.n_groups, 2);
    assert!(!g.rms_norm);
    assert!(provenance.family_defaults.is_empty());
}

/// The HF spelling wins with an empty default record, and a partial
/// mamba_ssm declaration is refused — dropping the bias switch must not
/// let the dialect read complete with a third silent default.
#[test]
fn the_hf_spelling_reads_with_no_defaults_and_a_partial_dialect_refuses() {
    let (_, provenance) = Mamba2Geometry::read_with_provenance(&mamba2_780m()).unwrap();
    assert_eq!(provenance.dialect, super::mamba2::Mamba2Dialect::Hf);
    assert!(provenance.family_defaults.is_empty());

    let mut partial = oute_250m();
    partial.as_object_mut().unwrap().remove("use_mamba2_bias");
    assert!(Mamba2Geometry::read_with_provenance(&partial).is_none());
}

/// **The conv-QKV attention geometry: all fields or none**, with the
/// derived widths that tell an attention mixer apart from a Mamba2 mixer
/// in the tensor estate.
#[test]
fn the_conv_qkv_geometry_reads_whole_and_derives_its_widths() {
    use super::ConvQkvAttnGeometry;
    let config = json!({
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "attention_head_dim": 128,
        "attention_conv_kernel": 4,
        "rope_emb_dim": 64,
        "rope_theta": 10000.0,
        "use_attention_qkv_bias": false,
        "use_attention_out_bias": false
    });
    let a = ConvQkvAttnGeometry::read(&config).unwrap();
    assert_eq!(a.qkv_rows(), 6144);
    assert_eq!(a.attn_out_width(), 2048);
    assert!(a.geometry_defects().is_empty());

    let mut partial = config.clone();
    partial.as_object_mut().unwrap().remove("rope_emb_dim");
    assert!(ConvQkvAttnGeometry::read(&partial).is_none());

    let mut wide = config;
    wide["rope_emb_dim"] = json!(256);
    let defects = ConvQkvAttnGeometry::read(&wide).unwrap().geometry_defects();
    assert_eq!(defects.len(), 1);
    assert!(defects[0].contains("exceeds"));
}

/// The other two cross-field defects, each named: a query-head count
/// that no GQA grouping divides, and an odd rotary width (rotation
/// pairs dims).
#[test]
fn the_conv_qkv_defects_name_grouping_and_odd_rotation() {
    use super::ConvQkvAttnGeometry;
    let base = json!({
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "attention_head_dim": 128,
        "attention_conv_kernel": 4,
        "rope_emb_dim": 64,
        "rope_theta": 10000.0,
        "use_attention_qkv_bias": false,
        "use_attention_out_bias": false
    });

    let mut ungrouped = base.clone();
    ungrouped["num_key_value_heads"] = json!(3);
    let defects = ConvQkvAttnGeometry::read(&ungrouped)
        .unwrap()
        .geometry_defects();
    assert_eq!(defects.len(), 1);
    assert!(defects[0].contains("not a multiple"), "{defects:?}");

    let mut odd = base;
    odd["rope_emb_dim"] = json!(63);
    let defects = ConvQkvAttnGeometry::read(&odd).unwrap().geometry_defects();
    assert_eq!(defects.len(), 1);
    assert!(defects[0].contains("odd"), "{defects:?}");
}
