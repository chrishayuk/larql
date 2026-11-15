//! `locate_stored_tensor`: names resolve exactly as a weight edit resolves them, without a patch's refusals.

use larql_vindex::format::vindex3::represent::codec::codecs::float::FloatDtype;
use larql_vindex::format::vindex3::stored_tensor::{locate_stored_tensor, StoredTensorError};
use larql_vindex::format::vindex3::weight_patch::{bind_source, qualified, WeightPatchError};

use crate::support::*;

#[test]
fn a_bare_name_finds_its_one_canonical_holder() {
    let c = bf16_packed();
    let t = locate_stored_tensor(&c.root, MATRIX).unwrap();
    assert_eq!(t.representation, CANONICAL);
    assert!(t.canonical);
    assert_eq!(t.tensor, MATRIX);
    assert_eq!(t.dtype, FloatDtype::Bf16);
    assert_eq!(t.shape, vec![MATRIX_ROWS, MATRIX_COLS]);
    assert_eq!(t.element_count(), MATRIX_ROWS * MATRIX_COLS);
}

#[test]
fn a_bare_ambiguous_name_is_refused_with_the_weight_edit_hint() {
    let c = bf16_packed();
    let err = locate_stored_tensor(&c.root, AMBIGUOUS).unwrap_err();
    let StoredTensorError::Ambiguous(WeightPatchError::AmbiguousTensor { holders, .. }) = &err
    else {
        panic!("{err}");
    };
    assert!(holders.contains(&EMBEDDING.to_string()), "{holders:?}");
    let edit = bind_source(&c.root, AMBIGUOUS).unwrap_err();
    assert_eq!(err.to_string(), edit.to_string());
    assert!(
        err.to_string().contains("<representation>::weight"),
        "{err}"
    );
}

#[test]
fn a_qualified_name_finds_the_representation_it_names() {
    let c = bf16_packed();
    let t = locate_stored_tensor(&c.root, &qualified(EMBEDDING, AMBIGUOUS)).unwrap();
    assert_eq!(t.representation, EMBEDDING);
    assert_eq!(t.tensor, AMBIGUOUS);
    let head = locate_stored_tensor(&c.root, &qualified(OUTPUT_HEAD, AMBIGUOUS)).unwrap();
    assert_eq!(head.representation, OUTPUT_HEAD);
}

#[test]
fn a_qualified_derived_float_tensor_is_found() {
    // The Q8_0 pack keeps the norm gains as BF16: a plain float tensor of a derived representation reads.
    let c = bf16_packed();
    let t = locate_stored_tensor(&c.root, &qualified(DERIVED, GAIN)).unwrap();
    assert_eq!(t.representation, DERIVED);
    assert!(!t.canonical);
    assert_eq!(t.dtype, FloatDtype::Bf16);
}

#[test]
fn a_packed_quantised_tensor_is_refused() {
    let c = bf16_packed();
    let err = locate_stored_tensor(&c.root, &qualified(DERIVED, MATRIX)).unwrap_err();
    assert!(
        matches!(&err, StoredTensorError::PackedDtype { dtype, .. } if dtype == "Q8_0"),
        "{err}"
    );
    assert!(err.to_string().contains("Q8_0"), "{err}");
}

#[test]
fn unknown_names_are_refused_by_kind() {
    let c = bf16_packed();
    let err = locate_stored_tensor(&c.root, "9.mlp.down_proj.weight").unwrap_err();
    assert!(matches!(err, StoredTensorError::UnknownBare(_)), "{err}");
    assert!(err.to_string().contains("SHOW TENSORS"), "{err}");
    // The decoder stack does not hold `weight`; a representation that does not exist holds nothing.
    for name in [
        qualified(CANONICAL, AMBIGUOUS),
        qualified("target.nothing@BF16", MATRIX),
    ] {
        let err = locate_stored_tensor(&c.root, &name).unwrap_err();
        assert!(
            matches!(err, StoredTensorError::UnknownQualified(_)),
            "{name}: {err}"
        );
    }
}

#[test]
fn a_read_is_not_refused_where_a_patch_would_be() {
    // A weight edit of the embedding is refused while a derived pack depends on another object
    // (`DerivedFromOther`); reading it is not an edit.
    let c = bf16_packed();
    let name = qualified(EMBEDDING, AMBIGUOUS);
    assert!(larql_vindex::format::vindex3::weight_patch::read_bf16_tensor(&c.root, &name).is_err());
    assert!(locate_stored_tensor(&c.root, &name).is_ok());
}

#[test]
fn an_f32_tensor_is_found_as_f32() {
    let c = f32_miniature();
    let t = locate_stored_tensor(&c.root, F32_MATRIX).unwrap();
    assert_eq!(t.representation, F32_CANONICAL);
    assert_eq!(t.dtype, FloatDtype::F32);
}
