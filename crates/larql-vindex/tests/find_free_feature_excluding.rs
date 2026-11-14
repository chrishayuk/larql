//! Free-slot search that also skips slots a caller has already handed out.
//!
//! A caller placing several features on one layer before applying any of them
//! (the server choosing slots for a patch's INSERT ops) must not be given the
//! same slot twice; the overlay cannot know about slots that are not applied
//! yet.

use std::collections::HashSet;

use larql_vindex::{ndarray::Array2, PatchedVindex, VectorIndex};

const LAYER: usize = 0;
const FEATURES: usize = 3;
const HIDDEN: usize = 4;

/// One layer of `FEATURES` slots, none with metadata, so every slot is free.
fn empty_layer() -> PatchedVindex {
    let gate = Array2::<f32>::zeros((FEATURES, HIDDEN));
    let meta = vec![None; FEATURES];
    PatchedVindex::new(VectorIndex::new(
        vec![Some(gate)],
        vec![Some(meta)],
        1,
        HIDDEN,
    ))
}

#[test]
fn with_nothing_claimed_it_agrees_with_find_free_feature() {
    let patched = empty_layer();
    assert_eq!(
        patched.find_free_feature_excluding(LAYER, &HashSet::new()),
        patched.find_free_feature(LAYER)
    );
}

#[test]
fn a_claimed_slot_is_not_handed_out_again() {
    let patched = empty_layer();
    let mut claimed = HashSet::new();
    for _ in 0..FEATURES {
        let slot = patched
            .find_free_feature_excluding(LAYER, &claimed)
            .expect("a free slot remains");
        assert!(claimed.insert(slot), "slot {slot} was handed out twice");
    }
    assert_eq!(claimed.len(), FEATURES);
}

#[test]
fn when_every_slot_is_claimed_there_is_none() {
    let patched = empty_layer();
    let claimed: HashSet<usize> = (0..FEATURES).collect();
    assert_eq!(patched.find_free_feature_excluding(LAYER, &claimed), None);
}
