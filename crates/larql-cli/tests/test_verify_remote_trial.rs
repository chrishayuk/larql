//! The VERIFY-REMOTE trial's own pieces (`examples/support/verify.rs`),
//! against a fake target: a deterministic "model" whose greedy next id
//! depends only on the token fed, so prompt lookup can propose correctly,
//! and which logs one exchange per unit of work like the timed transport.

#[allow(dead_code)]
#[path = "../examples/support/mod.rs"]
mod support;

use larql_inference::error::InferenceError;
use larql_inference::speculation::propose;
use larql_inference::vindex3::speculate::{greedy_speculate, prefill, BlockTarget};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use support::timing::{Exchange, Recorder, MULTI_ROW, SINGLE_ROW};
use support::verify::{ledger_json, sequential, Corrupt, Tagged, STEP, VERIFY};

const VOCAB: u32 = 16;
const GENERATE: usize = 24;
const WIDTHS: [usize; 3] = [2, 4, 8];
const PREFILL_WIDTH: usize = 8;

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
        self.exchanges.lock().unwrap().push(Exchange {
            route,
            layer: 0,
            rows,
            pairs: rows,
            request_bytes: 0,
            response_bytes: 0,
            roundtrip_ns: 0,
        });
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
            let mut proposer = |ctx: &[u32], source_len: usize, max: usize| {
                let mut guess = propose(ctx, source_len, max);
                corrupt.apply(&mut guess);
                guess
            };
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
            // prompt's break), so a corrupted block rejects at its changed
            // offset or earlier, never later, and acceptance falls.
            for block in &run.blocks {
                if let Some(at) = corrupt.offset(block.proposed) {
                    assert!(
                        block.rejected_at.is_some_and(|r| r <= at),
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
        let mut proposer = |ctx: &[u32], source_len: usize, max: usize| {
            let mut guess = propose(ctx, source_len, max);
            Corrupt::Middle.apply(&mut guess);
            guess
        };
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
            assert_eq!(unit.exchanges, (exchange, exchange + 1));
            exchange += 1;
            let logged = &exchanges[unit.exchanges.0];
            assert_eq!(logged.rows, unit.width);
            let route = if unit.kind == VERIFY {
                MULTI_ROW
            } else {
                SINGLE_ROW
            };
            assert_eq!(logged.route, route);
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
    let base = [5u32, 0, 9];
    for (corrupt, at) in [
        (Corrupt::First, 0),
        (Corrupt::Middle, 1),
        (Corrupt::Last, 2),
    ] {
        let mut guess = base;
        corrupt.apply(&mut guess);
        for (i, (a, b)) in base.iter().zip(&guess).enumerate() {
            assert_eq!(a != b, i == at, "{}", corrupt.name());
        }
    }
    let mut zero = [0u32];
    Corrupt::First.apply(&mut zero);
    assert_eq!(zero, [1]);
    let mut empty: [u32; 0] = [];
    Corrupt::Last.apply(&mut empty);
    let mut untouched = base;
    Corrupt::None.apply(&mut untouched);
    assert_eq!(untouched, base);
    for name in ["none", "first", "middle", "last"] {
        assert_eq!(Corrupt::parse(name).unwrap().name(), name);
    }
    assert!(Corrupt::parse("all").is_err());
}

#[test]
fn the_ledger_names_every_source() {
    let mut target = tagged();
    let first = prefill(&mut target, &prompt(), PREFILL_WIDTH).unwrap();
    let mut proposer = |ctx: &[u32], source_len: usize, max: usize| {
        let mut guess = propose(ctx, source_len, max);
        Corrupt::Last.apply(&mut guess);
        guess
    };
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
