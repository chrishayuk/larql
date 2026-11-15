//! Where a generated sequence ends.

use std::collections::HashSet;

use crate::speculation::truncate_at_stop;

#[test]
fn the_generated_ids_end_at_the_first_stop_which_is_kept() {
    let stop = HashSet::from([7]);
    let mut out = vec![1, 2, 7, 3, 7];
    truncate_at_stop(&mut out, &stop, 10);
    assert_eq!(out, vec![1, 2, 7]);
}

#[test]
fn the_count_still_bounds_a_run_that_stops_later_than_it() {
    let stop = HashSet::from([7]);
    let mut out = vec![1, 2, 3, 7];
    truncate_at_stop(&mut out, &stop, 2);
    assert_eq!(out, vec![1, 2]);
}

#[test]
fn without_a_stop_in_the_ids_only_the_count_applies() {
    let mut out = vec![1, 2, 3];
    truncate_at_stop(&mut out, &HashSet::from([7]), 10);
    assert_eq!(out, vec![1, 2, 3]);
    let mut out = vec![7, 7];
    truncate_at_stop(&mut out, &HashSet::new(), 1);
    assert_eq!(out, vec![7]);
}
