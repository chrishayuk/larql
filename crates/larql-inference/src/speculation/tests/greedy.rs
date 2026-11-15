//! The shared greedy choice.

use crate::speculation::greedy_id;

#[test]
fn the_highest_logit_wins() {
    assert_eq!(greedy_id(&[0.1, 2.0, -1.0]), Some(1));
}

#[test]
fn a_tie_goes_to_the_lowest_id() {
    assert_eq!(greedy_id(&[1.0, 3.0, 3.0, 0.0]), Some(1));
}

#[test]
fn empty_or_non_finite_logits_have_no_greedy_id() {
    assert_eq!(greedy_id(&[]), None);
    assert_eq!(greedy_id(&[1.0, f32::NAN]), None);
    assert_eq!(greedy_id(&[f32::INFINITY, 1.0]), None);
    assert_eq!(greedy_id(&[1.0, f32::NEG_INFINITY]), None);
}
