//! Prompt-lookup proposer and acceptance.

use super::super::prompt_lookup::{accepted_prefix, propose};

#[test]
fn proposes_from_the_earliest_prompt_occurrence() {
    // prompt = [1, 2, 3, 4, 1, 2, 9, 5]; tail [1, 2] occurs at 0 and 4.
    let ctx = [1, 2, 3, 4, 1, 2, 9, 5, 7, 1, 2];
    assert_eq!(propose(&ctx, 8, 3), vec![3, 4, 1]);
}

#[test]
fn prefers_the_longest_suffix() {
    // [7, 1, 2] pins the occurrence at 4 over the bigram's earlier one at 0.
    let ctx = [1, 2, 3, 0, 7, 1, 2, 8, 6, 7, 1, 2];
    assert_eq!(propose(&ctx, 9, 1), vec![8]);
}

#[test]
fn ignores_matches_in_the_generated_output() {
    // [5, 6] occurs only after the prompt (length 3).
    let ctx = [9, 9, 9, 5, 6, 4, 5, 6];
    assert!(propose(&ctx, 3, 4).is_empty());
}

#[test]
fn a_unigram_match_does_not_propose() {
    let ctx = [5, 6, 7, 8, 5];
    assert!(propose(&ctx, 4, 4).is_empty());
}

#[test]
fn proposal_stops_at_the_end_of_the_prompt_and_at_max() {
    let ctx = [1, 2, 3, 4, 0, 1, 2];
    assert_eq!(propose(&ctx, 4, 7), vec![3, 4]);
    assert_eq!(propose(&ctx, 4, 1), vec![3]);
    assert!(propose(&ctx, 4, 0).is_empty());
    assert!(propose(&[1], 1, 4).is_empty());
}

#[test]
fn accepted_prefix_stops_at_the_first_disagreement() {
    assert_eq!(accepted_prefix(&[4, 5, 6], &[4, 5, 9, 1]), 2);
    assert_eq!(accepted_prefix(&[4, 5, 6], &[3, 5, 6, 1]), 0);
    assert_eq!(accepted_prefix(&[4, 5, 6], &[4, 5, 6, 1]), 3);
}
