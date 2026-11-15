//! `list_tensors`: every tensor of every representation, and what its bare name finds.

use larql_vindex::format::vindex3::stored_tensor::{list_tensors, BareName};
use larql_vindex::format::vindex3::weight_patch::qualified;

use crate::support::*;

#[test]
fn every_representation_is_listed_canonical_and_derived() {
    let c = bf16_packed();
    let listings = list_tensors(&c.root).unwrap();
    for representation in [CANONICAL, DERIVED, EMBEDDING, FINAL_NORM, OUTPUT_HEAD] {
        assert!(
            listings.iter().any(|l| l.representation == representation),
            "{representation} missing"
        );
    }
    let derived = listings
        .iter()
        .find(|l| l.representation == DERIVED && l.tensor == MATRIX)
        .unwrap();
    assert!(!derived.canonical);
    assert_eq!(derived.dtype, "Q8_0");
    assert_eq!(derived.shape, vec![MATRIX_ROWS, MATRIX_COLS]);
    assert_eq!(derived.bare_name, BareName::Derived);
}

#[test]
fn the_qualified_name_is_representation_then_tensor() {
    let c = bf16_packed();
    let listings = list_tensors(&c.root).unwrap();
    let embedding = listings
        .iter()
        .find(|l| l.representation == EMBEDDING)
        .unwrap();
    assert_eq!(embedding.qualified_name(), qualified(EMBEDDING, AMBIGUOUS));
    assert_eq!(embedding.qualified_name(), "target.embedding@BF16::weight");
}

#[test]
fn a_name_held_by_several_canonical_representations_is_marked_ambiguous() {
    let c = bf16_packed();
    let listings = list_tensors(&c.root).unwrap();
    let holders: Vec<_> = listings.iter().filter(|l| l.tensor == AMBIGUOUS).collect();
    assert_eq!(holders.len(), 3);
    assert!(holders.iter().all(|l| l.bare_name == BareName::Ambiguous));
    let matrix = listings
        .iter()
        .find(|l| l.representation == CANONICAL && l.tensor == MATRIX)
        .unwrap();
    assert_eq!(matrix.bare_name, BareName::Resolves);
    assert!(matrix.canonical);
}

#[test]
fn the_labels_are_distinct_words() {
    let labels = [
        BareName::Resolves.label(),
        BareName::Ambiguous.label(),
        BareName::Derived.label(),
    ];
    assert_eq!(labels, ["yes", "ambiguous", "no (derived)"]);
}

#[test]
fn a_directory_that_is_not_a_container_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    assert!(list_tensors(dir.path()).is_err());
    std::fs::write(dir.path().join("index.json"), "not json").unwrap();
    assert!(list_tensors(dir.path()).is_err());
}
