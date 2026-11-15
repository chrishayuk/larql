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
fn negative_infinity_is_a_score_that_is_never_chosen() {
    assert_eq!(
        greedy_id(&[f32::NEG_INFINITY, -5.0, f32::NEG_INFINITY]),
        Some(1)
    );
    assert_eq!(greedy_id(&[f32::NEG_INFINITY, f32::NEG_INFINITY]), None);
}

#[test]
fn without_a_defined_choice_there_is_no_greedy_id() {
    assert_eq!(greedy_id(&[]), None);
    assert_eq!(greedy_id(&[1.0, f32::NAN]), None);
    assert_eq!(greedy_id(&[f32::INFINITY, 1.0]), None);
}
