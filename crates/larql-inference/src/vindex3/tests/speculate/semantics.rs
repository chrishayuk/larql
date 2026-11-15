//! The loop against a scripted target.

use std::collections::HashSet;

use crate::error::InferenceError;
use crate::vindex3::speculate::{
    greedy_speculate, prefill, BlockTarget, Proposer, Source, SpeculativeRun,
};

/// Vocabulary of the scripted model.
const VOCAB: u32 = 23;
/// Widths under test.
const WIDTHS: [usize; 3] = [2, 4, 8];
/// Ids generated in the long runs.
const GENERATE: usize = 40;
/// Constants of the scripted model's next-id rule.
const MULTIPLIER: u32 = 7;
const LENGTH_WEIGHT: u32 = 3;

/// The scripted model's greedy next id after `ctx`.
fn next_of(ctx: &[u32]) -> u32 {
    let last = *ctx.last().expect("non-empty context");
    (last * MULTIPLIER + ctx.len() as u32 * LENGTH_WEIGHT + 1) % VOCAB
}

fn one_hot(id: u32) -> Vec<f32> {
    let mut row = vec![0.0; VOCAB as usize];
    row[id as usize] = 1.0;
    row
}

/// A block target whose greedy choice after any context is `next_of`.
#[derive(Default)]
struct Scripted {
    committed: Vec<u32>,
    pending: Option<Vec<u32>>,
    /// When set, the logits of this committed position are non-finite.
    poison_at: Option<usize>,
    /// Every block verified, in order.
    verified: Vec<Vec<u32>>,
}

impl Scripted {
    fn logits(&self, ctx: &[u32]) -> Vec<f32> {
        if self.poison_at == Some(ctx.len()) {
            return vec![f32::NAN; VOCAB as usize];
        }
        one_hot(next_of(ctx))
    }
}

impl BlockTarget for Scripted {
    fn verify(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>, InferenceError> {
        assert!(self.pending.is_none(), "verify while a block is pending");
        let mut ctx = self.committed.clone();
        let mut rows = Vec::new();
        for token in tokens {
            ctx.push(*token);
            rows.push(self.logits(&ctx));
        }
        self.pending = Some(tokens.to_vec());
        self.verified.push(tokens.to_vec());
        Ok(rows)
    }
    fn commit(&mut self, count: usize) -> Result<(), InferenceError> {
        let block = self
            .pending
            .take()
            .expect("commit without a verified block");
        assert!(count <= block.len());
        self.committed.extend(&block[..count]);
        Ok(())
    }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, InferenceError> {
        assert!(self.pending.is_none(), "step while a block is pending");
        self.committed.push(token);
        Ok(self.logits(&self.committed.clone()))
    }
    fn position(&self) -> usize {
        self.committed.len()
    }
}

fn prompt() -> Vec<u32> {
    (0..12).map(|i| (i * 5 + 2) % VOCAB).collect()
}

/// Sequential greedy decoding: the reference every run must reproduce.
fn reference(prompt: &[u32], n: usize) -> Vec<u32> {
    let mut ctx = prompt.to_vec();
    let mut out = Vec::new();
    while out.len() < n {
        let id = next_of(&ctx);
        out.push(id);
        ctx.push(id);
    }
    out
}

/// Proposes the true greedy continuation of `ctx`.
fn oracle(ctx: &[u32], _source_len: usize, max: usize) -> Vec<u32> {
    let mut ctx = ctx.to_vec();
    (0..max)
        .map(|_| {
            let id = next_of(&ctx);
            ctx.push(id);
            id
        })
        .collect()
}

fn run_on(
    target: &mut Scripted,
    width: usize,
    n: usize,
    stop: &HashSet<u32>,
    proposer: &mut dyn Proposer,
) -> SpeculativeRun {
    let prompt = prompt();
    let first = prefill(target, &prompt, width).unwrap();
    let run = greedy_speculate(target, &prompt, first, n, width, stop, proposer).unwrap();
    run.reconcile().unwrap();
    assert_eq!(run.final_position, target.position());
    run
}

fn run(width: usize, n: usize, stop: &HashSet<u32>, proposer: &mut dyn Proposer) -> SpeculativeRun {
    run_on(&mut Scripted::default(), width, n, stop, proposer)
}

#[test]
fn prefill_returns_the_prompts_greedy_id_at_any_width() {
    let prompt = prompt();
    for width in [1, 3, 8, 64] {
        let mut target = Scripted::default();
        assert_eq!(
            prefill(&mut target, &prompt, width).unwrap(),
            next_of(&prompt)
        );
        assert_eq!(target.position(), prompt.len());
    }
}

#[test]
fn every_proposal_accepted_keeps_whole_blocks_and_matches_greedy() {
    for width in WIDTHS {
        let run = run(width, GENERATE, &HashSet::new(), &mut oracle);
        assert_eq!(run.ids, reference(&prompt(), GENERATE));
        // Only the last wanted id can need a single step: with one id left
        // there is nothing to propose.
        assert!(run.single_steps <= 1, "width {width}");
        assert!(run
            .blocks
            .iter()
            .all(|b| b.rejected_at.is_none() && b.accepted == b.proposed));
        // Full blocks except possibly the last, which is cut to what is still wanted.
        let full = run.blocks.iter().filter(|b| b.width == width).count();
        assert!(full + 1 >= run.blocks.len(), "width {width}");
    }
}

#[test]
fn nothing_proposed_means_single_steps_only_and_matches_greedy() {
    for width in WIDTHS {
        let run = run(width, GENERATE, &HashSet::new(), &mut |_: &[u32], _, _| {
            Vec::new()
        });
        assert_eq!(run.ids, reference(&prompt(), GENERATE));
        assert!(run.blocks.is_empty());
        assert_eq!(run.single_steps, GENERATE - 1);
    }
}

#[test]
fn a_corrupted_proposal_is_rejected_exactly_where_it_is_wrong() {
    for width in WIDTHS {
        for offset in 0..width - 1 {
            let mut corrupt = |ctx: &[u32], source_len: usize, max: usize| {
                let mut guess = oracle(ctx, source_len, max);
                if let Some(token) = guess.get_mut(offset) {
                    *token = (*token + 1) % VOCAB;
                }
                guess
            };
            let run = run(width, GENERATE, &HashSet::new(), &mut corrupt);
            assert_eq!(
                run.ids,
                reference(&prompt(), GENERATE),
                "width {width} offset {offset}"
            );
            for block in &run.blocks {
                if block.proposed > offset {
                    assert_eq!(
                        block.rejected_at,
                        Some(offset),
                        "width {width} offset {offset}"
                    );
                    assert_eq!(block.accepted, offset);
                } else {
                    assert_eq!(block.rejected_at, None);
                }
            }
        }
    }
}

#[test]
fn proposals_accepted_only_sometimes_still_match_greedy() {
    for width in WIDTHS {
        let mut calls = 0usize;
        let mut sometimes = |ctx: &[u32], source_len: usize, max: usize| {
            calls += 1;
            match calls % 3 {
                0 => Vec::new(),
                1 => oracle(ctx, source_len, max),
                _ => oracle(ctx, source_len, max)
                    .into_iter()
                    .map(|t| (t + 2) % VOCAB)
                    .collect(),
            }
        };
        let run = run(width, GENERATE, &HashSet::new(), &mut sometimes);
        assert_eq!(run.ids, reference(&prompt(), GENERATE));
        assert!(run.single_steps > 0 && !run.blocks.is_empty());
    }
}

#[test]
fn a_stop_anywhere_ends_generation_there_and_later_ids_are_not_emitted() {
    let expected = reference(&prompt(), GENERATE);
    for width in WIDTHS {
        for at in 0..GENERATE {
            let stop = HashSet::from([expected[at]]);
            let end = expected.iter().position(|id| *id == expected[at]).unwrap() + 1;
            let mut target = Scripted::default();
            let run = run_on(&mut target, width, GENERATE, &stop, &mut oracle);
            assert_eq!(run.ids, expected[..end], "width {width} stop at {at}");
            // Nothing past the stop was verified or committed: after the
            // prefill blocks, no verified block holds a stop id.
            let prefill_blocks = prompt().len().div_ceil(width);
            assert!(
                target.verified[prefill_blocks..]
                    .iter()
                    .all(|block| !block.iter().any(|t| stop.contains(t))),
                "width {width} stop at {at}: a stop id was verified inside a block"
            );
            assert_eq!(target.position(), prompt().len() + end - 1);
            let emitted = run.events.iter().filter(|e| e.emitted).count();
            assert_eq!(emitted, end);
        }
    }
}

#[test]
fn the_count_is_never_exceeded_and_no_proposal_outruns_it() {
    for width in WIDTHS {
        for n in 1..=2 * width + 1 {
            let run = run(width, n, &HashSet::new(), &mut oracle);
            assert_eq!(run.ids, reference(&prompt(), n), "width {width} n {n}");
            // Before each block, at most n - generated - 1 tokens may be
            // proposed: the block's own correction is the last wanted id.
            let mut generated = 1;
            for block in &run.blocks {
                assert!(block.proposed < n - generated, "width {width} n {n}");
                generated += block.accepted + 1;
            }
            assert!(
                run.events.iter().all(|e| e.emitted),
                "nothing beyond n was produced"
            );
        }
    }
}

#[test]
fn every_event_has_one_source_and_the_ledger_reconciles() {
    let run = run(4, GENERATE, &HashSet::new(), &mut oracle);
    assert_eq!(run.events[0].source, Source::Prefill);
    assert!(run.events[1..].iter().all(|e| e.source != Source::Prefill));
    let swap = (1..run.events.len() - 1)
        .find(|&i| run.events[i].id != run.events[i + 1].id)
        .expect("the run has two different neighbouring ids");
    let mut tampered = run.clone();
    tampered.events.swap(swap, swap + 1);
    assert!(tampered.reconcile().is_err(), "reordered events reconciled");
    let mut miscounted = run.clone();
    miscounted.blocks[0].accepted += 1;
    assert!(
        miscounted.reconcile().is_err(),
        "a miscounted block reconciled"
    );
    // A block record that is self-consistent (fewer accepted, rejected
    // there) but disagrees with the block's own events.
    let k = run
        .blocks
        .iter()
        .position(|b| b.accepted > 0)
        .expect("a block accepted something");
    let mut understated = run.clone();
    understated.blocks[k].accepted -= 1;
    understated.blocks[k].rejected_at = Some(understated.blocks[k].accepted);
    assert!(
        understated.reconcile().is_err(),
        "a block understating its events reconciled"
    );
    let mut moved = run.clone();
    moved.final_position += 1;
    assert!(
        moved.reconcile().is_err(),
        "a wrong final position reconciled"
    );
    let mut shifted = run.clone();
    shifted.blocks[0].start += 1;
    assert!(
        shifted.reconcile().is_err(),
        "a wrong block start reconciled"
    );
    let mut dropped = run.clone();
    dropped.events.pop();
    assert!(dropped.reconcile().is_err());
    let mut relabelled = run;
    relabelled.events[1].source = Source::SingleStep;
    assert!(relabelled.reconcile().is_err());
}

#[test]
fn what_the_loop_cannot_do_is_refused() {
    let prompt = prompt();
    let mut target = Scripted::default();
    assert!(prefill(&mut target, &[], 4).is_err());
    assert!(prefill(&mut target, &prompt, 0).is_err());
    let first = prefill(&mut target, &prompt, 4).unwrap();
    let error = greedy_speculate(
        &mut target,
        &prompt,
        first,
        5,
        1,
        &HashSet::new(),
        &mut oracle,
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("at least two"), "{error}");
    let none = greedy_speculate(
        &mut target,
        &prompt,
        first,
        0,
        4,
        &HashSet::new(),
        &mut oracle,
    )
    .unwrap();
    assert!(none.ids.is_empty() && none.events.is_empty());
    assert_eq!(none.final_position, prompt.len());
    none.reconcile().unwrap();
    // Non-finite logits have no greedy id: the loop stops with an error.
    let mut poisoned = Scripted {
        poison_at: Some(prompt.len() + 2),
        ..Default::default()
    };
    let first = prefill(&mut poisoned, &prompt, 4).unwrap();
    assert!(greedy_speculate(
        &mut poisoned,
        &prompt,
        first,
        10,
        4,
        &HashSet::new(),
        &mut oracle
    )
    .is_err());
}

#[test]
fn a_proposer_that_returns_too_much_is_cut_back_to_the_block() {
    for width in WIDTHS {
        let mut asks_for_more =
            |ctx: &[u32], source_len: usize, max: usize| oracle(ctx, source_len, max + 5);
        let run = run(width, GENERATE, &HashSet::new(), &mut asks_for_more);
        assert_eq!(run.ids, reference(&prompt(), GENERATE));
        assert!(run.blocks.iter().all(|b| b.width <= width), "width {width}");
    }
}

#[test]
fn a_session_that_already_held_earlier_input_reconciles() {
    // A system prefix committed before the prompt: the proposer copies from
    // the prompt alone, and positions are judged from where the run began.
    let prefix: Vec<u32> = (0..5).map(|i| (i * 3 + 1) % VOCAB).collect();
    for width in WIDTHS {
        let mut target = Scripted::default();
        prefill(&mut target, &prefix, width).unwrap();
        let prompt = prompt();
        let first = prefill(&mut target, &prompt, width).unwrap();
        let run = greedy_speculate(
            &mut target,
            &prompt,
            first,
            GENERATE,
            width,
            &HashSet::new(),
            &mut |ctx: &[u32], _: usize, max: usize| {
                let mut full = prefix.clone();
                full.extend(ctx);
                oracle(&full, 0, max)
            },
        )
        .unwrap();
        assert_eq!(run.start_position, prefix.len() + prompt.len());
        run.reconcile().unwrap();
        let mut whole = prefix.clone();
        whole.extend(&prompt);
        assert_eq!(run.ids, reference(&whole, GENERATE), "width {width}");
    }
}
