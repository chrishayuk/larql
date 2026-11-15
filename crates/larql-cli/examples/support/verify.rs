//! The VERIFY-REMOTE trial's own pieces, kept out of `main` so they are
//! tested (`crates/larql-cli/tests/test_verify_remote_trial.rs`): the
//! work-tagging target wrapper, the independent sequential loop, the
//! negative-control proposal corruption, and the ledger's JSON form.

use super::timing::{Exchange, Recorder};
use larql_inference::error::InferenceError;
use larql_inference::speculation::greedy_id;
use larql_inference::vindex3::speculate::{BlockTarget, Source, SpeculativeRun};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub const VERIFY: &str = "verify";
pub const STEP: &str = "step";

/// Where the negative control changes each proposal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Corrupt {
    None,
    First,
    Middle,
    Last,
}

impl Corrupt {
    pub fn parse(value: &str) -> Result<Self, String> {
        match value {
            "none" => Ok(Self::None),
            "first" => Ok(Self::First),
            "middle" => Ok(Self::Middle),
            "last" => Ok(Self::Last),
            other => Err(format!(
                "--corrupt takes none|first|middle|last, not {other}"
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::First => "first",
            Self::Middle => "middle",
            Self::Last => "last",
        }
    }

    /// The offset this control changes in a proposal of `len` tokens.
    pub fn offset(self, len: usize) -> Option<usize> {
        match self {
            Self::None => None,
            _ if len == 0 => None,
            Self::First => Some(0),
            Self::Middle => Some(len / 2),
            Self::Last => Some(len - 1),
        }
    }

    /// Change one token of `guess` to the id below it (above it for id 0),
    /// so it differs and stays in the vocabulary.
    pub fn apply(self, guess: &mut [u32]) {
        if let Some(at) = self.offset(guess.len()) {
            guess[at] = if guess[at] == 0 { 1 } else { guess[at] - 1 };
        }
    }
}

/// One unit of target work: a verify block or a single step.
#[derive(Debug)]
pub struct Unit {
    pub kind: &'static str,
    /// The committed position before the unit ran.
    pub start: usize,
    pub width: usize,
    /// How many of its positions were kept; `None` until committed.
    pub committed: Option<usize>,
    pub run_ns: u64,
    pub commit_ns: u64,
    /// Exchange and worker-record index ranges, over the whole run.
    pub exchanges: (usize, usize),
    pub records: (usize, usize),
}

impl Unit {
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "kind": self.kind, "start": self.start, "width": self.width,
            "committed": self.committed, "run_ns": self.run_ns, "commit_ns": self.commit_ns,
            "exchanges": [self.exchanges.0, self.exchanges.1],
            "worker_records": [self.records.0, self.records.1],
        })
    }
}

/// A target that notes, for every unit of work, its shape, wall time and
/// the exchanges and worker records it caused. It changes nothing the
/// target does.
pub struct Tagged<T> {
    pub inner: T,
    pub exchanges: Arc<Mutex<Vec<Exchange>>>,
    pub recorder: &'static Recorder,
    pub units: Vec<Unit>,
}

impl<T: BlockTarget> Tagged<T> {
    /// Exchanges and worker records so far.
    pub fn counts(&self) -> (usize, usize) {
        (
            self.exchanges.lock().unwrap().len(),
            self.recorder.records.lock().unwrap().len(),
        )
    }

    fn run<R>(
        &mut self,
        kind: &'static str,
        width: usize,
        work: impl FnOnce(&mut T) -> Result<R, InferenceError>,
    ) -> Result<R, InferenceError> {
        let start = self.inner.position();
        let (e0, r0) = self.counts();
        let started = Instant::now();
        let out = work(&mut self.inner)?;
        let run_ns = started.elapsed().as_nanos() as u64;
        let (e1, r1) = self.counts();
        self.units.push(Unit {
            kind,
            start,
            width,
            committed: (kind == STEP).then_some(1),
            run_ns,
            commit_ns: 0,
            exchanges: (e0, e1),
            records: (r0, r1),
        });
        Ok(out)
    }
}

impl<T: BlockTarget> BlockTarget for Tagged<T> {
    fn verify(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>, InferenceError> {
        self.run(VERIFY, tokens.len(), |inner| inner.verify(tokens))
    }

    fn commit(&mut self, count: usize) -> Result<(), InferenceError> {
        let pending = self
            .units
            .last()
            .is_some_and(|u| u.kind == VERIFY && u.committed.is_none());
        if !pending {
            return Err(InferenceError::Parse(
                "commit with no verified block".into(),
            ));
        }
        let started = Instant::now();
        self.inner.commit(count)?;
        let unit = self.units.last_mut().expect("checked above");
        unit.committed = Some(count);
        unit.commit_ns = started.elapsed().as_nanos() as u64;
        Ok(())
    }

    fn step(&mut self, token: u32) -> Result<Vec<f32>, InferenceError> {
        self.run(STEP, 1, |inner| inner.step(token))
    }

    fn position(&self) -> usize {
        self.inner.position()
    }
}

/// Sequential greedy generation: at most `n` ids after and including
/// `first`, ending after the first stop id. Deliberately independent of
/// `greedy_speculate`, so the identity gate never compares a loop with
/// itself.
pub fn sequential<T: BlockTarget + ?Sized>(
    target: &mut T,
    first: u32,
    n: usize,
    stop: &HashSet<u32>,
) -> Result<Vec<u32>, InferenceError> {
    let mut ids = Vec::with_capacity(n);
    if n == 0 {
        return Ok(ids);
    }
    ids.push(first);
    while ids.len() < n {
        let last = *ids.last().expect("seeded");
        if stop.contains(&last) {
            break;
        }
        let logits = target.step(last)?;
        ids.push(greedy_id(&logits).ok_or_else(|| InferenceError::Parse("no greedy id".into()))?);
    }
    Ok(ids)
}

pub fn source_name(source: Source) -> &'static str {
    match source {
        Source::Prefill => "prefill",
        Source::AcceptedProposal => "accepted-proposal",
        Source::TargetCorrection => "target-correction",
        Source::SingleStep => "single-step",
        _ => "unknown",
    }
}

/// The ledger as the trial writes it, with the result of `reconcile`.
pub fn ledger_json(run: &SpeculativeRun) -> serde_json::Value {
    serde_json::json!({
        "events": run.events.iter().map(|e| serde_json::json!({
            "id": e.id, "source": source_name(e.source), "block": e.block, "emitted": e.emitted,
        })).collect::<Vec<_>>(),
        "blocks": run.blocks.iter().map(|b| serde_json::json!({
            "start": b.start, "width": b.width, "proposed": b.proposed,
            "accepted": b.accepted, "rejected_at": b.rejected_at,
        })).collect::<Vec<_>>(),
        "single_steps": run.single_steps,
        "prompt_len": run.prompt_len,
        "start_position": run.start_position,
        "final_position": run.final_position,
        "reconcile": match run.reconcile() {
            Ok(()) => "ok".to_string(),
            Err(why) => why,
        },
    })
}
