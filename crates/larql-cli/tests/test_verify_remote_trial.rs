//! The VERIFY-REMOTE trial's own pieces (`examples/support/verify.rs`),
//! against a fake target: a deterministic "model" whose greedy next id
//! depends only on the token fed, so prompt lookup can propose correctly,
//! and which logs one exchange per routed layer for every unit of work,
//! like the timed transport.

#[allow(dead_code)]
#[path = "../examples/support/mod.rs"]
mod support;

use larql_inference::error::InferenceError;
use larql_inference::speculation::propose;
use larql_inference::vindex3::speculate::{greedy_speculate, prefill, BlockTarget};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use support::timing::{Exchange, Recorder, MULTI_ROW, SINGLE_ROW};
use support::verify::{ledger_json, sequential, Corrupt, Tagged, TrialProposer, STEP, VERIFY};

const VOCAB: u32 = 16;
const GENERATE: usize = 24;
const WIDTHS: [usize; 3] = [2, 4, 8];
const PREFILL_WIDTH: usize = 8;
/// Routed layers the fake contacts per unit of work.
const LAYERS: usize = 3;

/// The fake model's greedy next id after `token`.
fn next(token: u32) -> u32 {
    (token * 5 + 3) % VOCAB
}

fn one_hot(id: u32) -> Vec<f32> {
    let mut row = vec![0.0; VOCAB as usize];
    row[id as usize] = 1.0;
    row
}

struct Fake {
    position: usize,
    pending: Option<usize>,
    exchanges: Arc<Mutex<Vec<Exchange>>>,
}

impl Fake {
    fn log(&self, route: &'static str, rows: usize) {
        let mut exchanges = self.exchanges.lock().unwrap();
        for layer in 0..LAYERS {
            exchanges.push(Exchange {
                route,
                layer,
                rows,
                pairs: rows,
                request_bytes: 0,
                response_bytes: 0,
                roundtrip_ns: 0,
            });
        }
    }
}

impl BlockTarget for Fake {
    fn verify(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>, InferenceError> {
        assert!(self.pending.is_none(), "verify with a block pending");
        self.log(MULTI_ROW, tokens.len());
        self.pending = Some(tokens.len());
        Ok(tokens.iter().map(|&t| one_hot(next(t))).collect())
    }

    fn commit(&mut self, count: usize) -> Result<(), InferenceError> {
        let width = self.pending.take().expect("commit with no block");
        assert!(count <= width);
        self.position += count;
        Ok(())
    }

    fn step(&mut self, token: u32) -> Result<Vec<f32>, InferenceError> {
        assert!(self.pending.is_none(), "step with a block pending");
        self.log(SINGLE_ROW, 1);
        self.position += 1;
        Ok(one_hot(next(token)))
    }

    fn position(&self) -> usize {
        self.position
    }
}

fn tagged() -> Tagged<Fake> {
    let exchanges = Arc::new(Mutex::new(Vec::new()));
    Tagged {
        inner: Fake {
            position: 0,
            pending: None,
            exchanges: exchanges.clone(),
        },
        exchanges,
        recorder: Recorder::install(false).unwrap(),
        units: Vec::new(),
    }
}

/// A prompt that follows the model, so lookups find the true continuation,
/// then a token that breaks the chain.
fn prompt() -> Vec<u32> {
    let mut ids = vec![1];
    for _ in 0..19 {
        ids.push(next(*ids.last().unwrap()));
    }
    ids.push(7);
    ids
}

/// Prompt lookup, as a plain function pointer.
type Lookup = fn(&[u32], usize, usize) -> Vec<u32>;

fn trial_proposer(corrupt: Corrupt, stop: &HashSet<u32>) -> TrialProposer<'_, Lookup> {
    TrialProposer {
        lookup: propose,
        corrupt,
        stop,
        changed: Vec::new(),
    }
}

fn reference(prompt: &[u32], n: usize) -> Vec<u32> {
    let mut ids = vec![next(*prompt.last().unwrap())];
    while ids.len() < n {
        ids.push(next(*ids.last().unwrap()));
    }
    ids
}

#[test]
fn sequential_generates_the_reference_by_single_steps() {
    let mut target = tagged();
    let first = prefill(&mut target, &prompt(), PREFILL_WIDTH).unwrap();
    let from = target.units.len();
    let ids = sequential(&mut target, first, GENERATE, &HashSet::new()).unwrap();
    assert_eq!(ids, reference(&prompt(), GENERATE));
    let units = &target.units[from..];
    assert_eq!(units.len(), GENERATE - 1, "the first id is prefill's");
    assert!(units
        .iter()
        .all(|u| u.kind == STEP && u.width == 1 && u.committed == Some(1)));
}

#[test]
fn sequential_keeps_the_stop_id_and_nothing_after_it() {
    let mut target = tagged();
    let first = prefill(&mut target, &prompt(), PREFILL_WIDTH).unwrap();
    let full = reference(&prompt(), GENERATE);
    let stop: HashSet<u32> = [full[5]].into();
    let ids = sequential(&mut target, first, GENERATE, &stop).unwrap();
    let end = full.iter().position(|id| stop.contains(id)).unwrap();
    assert_eq!(ids, full[..=end]);
    assert!(sequential(&mut tagged(), first, 0, &HashSet::new())
        .unwrap()
        .is_empty());
}

#[test]
fn every_arm_and_control_matches_sequential() {
    let reference = reference(&prompt(), GENERATE);
    for width in WIDTHS {
        let mut clean_accepted = None;
        for corrupt in [
            Corrupt::None,
            Corrupt::First,
            Corrupt::Middle,
            Corrupt::Last,
        ] {
            let mut target = tagged();
            let first = prefill(&mut target, &prompt(), PREFILL_WIDTH).unwrap();
            let none = HashSet::new();
            let mut proposer = trial_proposer(corrupt, &none);
            let run = greedy_speculate(
                &mut target,
                &prompt(),
                first,
                GENERATE,
                width,
                &HashSet::new(),
                &mut proposer,
            )
            .unwrap();
            let label = format!("width {width} corrupt {}", corrupt.name());
            assert_eq!(run.ids, reference, "{label}");
            assert_eq!(ledger_json(&run)["reconcile"], "ok", "{label}");
            assert!(!run.blocks.is_empty(), "{label}: lookup must propose");
            // A clean lookup can itself be wrong (it copies across the
            // prompt's break), so a corrupted block rejects at its recorded
            // changed offset or earlier, never later, and acceptance falls.
            assert_eq!(proposer.changed.len(), run.blocks.len(), "{label}");
            for (block, changed) in run.blocks.iter().zip(&proposer.changed) {
                assert_eq!(*changed, corrupt.offset(block.proposed), "{label}");
                if let Some(at) = changed {
                    assert!(
                        block.rejected_at.is_some_and(|r| r <= *at),
                        "{label}: {block:?}"
                    );
                }
            }
            let accepted: usize = run.blocks.iter().map(|b| b.accepted).sum();
            match clean_accepted {
                None => {
                    assert!(accepted > 0, "{label}: the clean lookup must hit");
                    clean_accepted = Some(accepted);
                }
                Some(clean) => assert!(accepted < clean, "{label}: {accepted} vs {clean}"),
            }
        }
    }
}

#[test]
fn units_chain_and_own_their_exchanges() {
    for width in WIDTHS {
        let mut target = tagged();
        let first = prefill(&mut target, &prompt(), PREFILL_WIDTH).unwrap();
        let none = HashSet::new();
        let mut proposer = trial_proposer(Corrupt::Middle, &none);
        let run = greedy_speculate(
            &mut target,
            &prompt(),
            first,
            GENERATE,
            width,
            &HashSet::new(),
            &mut proposer,
        )
        .unwrap();
        let exchanges = target.exchanges.lock().unwrap();
        let mut position = 0;
        let mut exchange = 0;
        for unit in &target.units {
            assert_eq!(unit.start, position, "width {width}");
            let committed = unit.committed.expect("every unit is committed");
            assert!(committed >= 1 && committed <= unit.width);
            position += committed;
            assert_eq!(unit.exchanges, (exchange, exchange + LAYERS));
            exchange += LAYERS;
            let route = if unit.kind == VERIFY {
                MULTI_ROW
            } else {
                SINGLE_ROW
            };
            for logged in &exchanges[unit.exchanges.0..unit.exchanges.1] {
                assert_eq!(logged.rows, unit.width);
                assert_eq!(logged.route, route);
            }
        }
        assert_eq!(exchange, exchanges.len());
        assert_eq!(position, target.position());
        assert_eq!(run.final_position, target.position());
        let verifies = target.units.iter().filter(|u| u.kind == VERIFY).count();
        let prefill_blocks = prompt().len().div_ceil(PREFILL_WIDTH);
        assert_eq!(verifies, prefill_blocks + run.blocks.len());
        let steps = target.units.iter().filter(|u| u.kind == STEP).count();
        assert_eq!(steps, run.single_steps);
    }
}

#[test]
fn a_commit_needs_a_pending_verify_block() {
    let mut target = tagged();
    assert!(target.commit(1).is_err(), "nothing verified");
    target.verify(&[1, 2]).unwrap();
    target.commit(2).unwrap();
    assert!(target.commit(1).is_err(), "already committed");
    target.step(3).unwrap();
    assert!(target.commit(1).is_err(), "a step is not a block");
}

#[test]
fn corruption_changes_exactly_one_token_and_stays_in_range() {
    let none = HashSet::new();
    let base = vec![5u32, 0, 9];
    for (corrupt, at) in [
        (Corrupt::First, 0),
        (Corrupt::Middle, 1),
        (Corrupt::Last, 2),
    ] {
        let mut guess = base.clone();
        assert_eq!(corrupt.apply(&mut guess, &none), Some(at));
        for (i, (a, b)) in base.iter().zip(&guess).enumerate() {
            assert_eq!(a != b, i == at, "{}", corrupt.name());
        }
    }
    let mut zero = vec![0u32];
    Corrupt::First.apply(&mut zero, &none);
    assert_eq!(zero, [1]);
    let mut empty = Vec::new();
    assert_eq!(Corrupt::Last.apply(&mut empty, &none), None);
    let mut untouched = base.clone();
    assert_eq!(Corrupt::None.apply(&mut untouched, &none), None);
    assert_eq!(untouched, base);
    for name in ["none", "first", "middle", "last"] {
        assert_eq!(Corrupt::parse(name).unwrap().name(), name);
    }
    assert!(Corrupt::parse("all").is_err());
}

#[test]
fn corruption_lands_on_a_verified_token_and_never_makes_a_stop() {
    // The proposal is cut before its stop id first, as the loop would cut
    // it, so the changed offset is one the target verifies.
    let stop: HashSet<u32> = [7].into();
    let mut guess = vec![3, 4, 7, 9];
    assert_eq!(Corrupt::Last.apply(&mut guess, &stop), Some(1));
    assert_eq!(guess, [3, 3]);
    // The replacement skips stop ids: 8's neighbour below is a stop.
    let mut guess = vec![8];
    Corrupt::First.apply(&mut guess, &stop);
    assert_eq!(guess, [6]);
    // A clean proposal is cut the same way and left otherwise unchanged.
    let mut guess = vec![3, 7, 9];
    assert_eq!(Corrupt::None.apply(&mut guess, &stop), None);
    assert_eq!(guess, [3]);
    // A proposal that starts with a stop id becomes empty: no block, no record.
    let mut proposer = TrialProposer {
        lookup: |_: &[u32], _: usize, _: usize| vec![7, 1],
        corrupt: Corrupt::First,
        stop: &stop,
        changed: Vec::new(),
    };
    use larql_inference::vindex3::speculate::Proposer;
    assert!(proposer.propose(&[1], 1, 4).is_empty());
    assert!(proposer.changed.is_empty());
}

#[test]
fn a_control_with_stop_ids_still_matches_sequential() {
    let full = reference(&prompt(), GENERATE);
    let stop: HashSet<u32> = [full[GENERATE - 4]].into();
    let end = full.iter().position(|id| stop.contains(id)).unwrap();
    for width in WIDTHS {
        for corrupt in [Corrupt::None, Corrupt::Middle, Corrupt::Last] {
            let mut target = tagged();
            let first = prefill(&mut target, &prompt(), PREFILL_WIDTH).unwrap();
            let mut proposer = trial_proposer(corrupt, &stop);
            let run = greedy_speculate(
                &mut target,
                &prompt(),
                first,
                GENERATE,
                width,
                &stop,
                &mut proposer,
            )
            .unwrap();
            let label = format!("width {width} corrupt {}", corrupt.name());
            assert_eq!(run.ids, full[..=end], "{label}");
            assert_eq!(ledger_json(&run)["reconcile"], "ok", "{label}");
            assert_eq!(proposer.changed.len(), run.blocks.len(), "{label}");
            for (block, changed) in run.blocks.iter().zip(&proposer.changed) {
                assert_eq!(*changed, corrupt.offset(block.proposed), "{label}");
            }
        }
    }
}

#[test]
fn the_ledger_names_every_source() {
    let mut target = tagged();
    let first = prefill(&mut target, &prompt(), PREFILL_WIDTH).unwrap();
    let none = HashSet::new();
    let mut proposer = trial_proposer(Corrupt::Last, &none);
    let run = greedy_speculate(
        &mut target,
        &prompt(),
        first,
        GENERATE,
        4,
        &HashSet::new(),
        &mut proposer,
    )
    .unwrap();
    let ledger = ledger_json(&run);
    let sources: HashSet<&str> = ledger["events"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["source"].as_str().unwrap())
        .collect();
    assert!(sources.contains("prefill"));
    assert!(sources.contains("accepted-proposal"));
    assert!(sources.contains("target-correction"));
    assert!(!sources.contains("unknown"));
    assert_eq!(ledger["blocks"].as_array().unwrap().len(), run.blocks.len());
}
