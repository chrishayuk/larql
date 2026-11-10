//! Observing a COMPOSE install's balance pass without taking part in it.
//!
//! The balance pass is a small controller: it reads the canonical prompt's target probability,
//! then amplifies, shrinks, stops or rolls back the slot's down vector. When an install ends
//! somewhere unsafe, the final state alone cannot say *why*: whether no safe scale was ever
//! tried, a safe one was passed over, or a non-monotone response was misread. A
//! [`BalanceObserver`] sees every step of that controller as it happens.
//!
//! The contract is narrow:
//!
//! - **Observation only.** The observer is called after the controller has read its probability
//!   and chosen what to do, and before it does it. It receives shared references; it cannot
//!   change a decision, a vector or the session.
//! - **Disabled means absent.** With no observer set the install runs the same operations in the
//!   same order as before.
//! - **Enabled changes nothing.** The final overlay and every prediction are bit-identical with
//!   and without an observer. Extra forward passes an observer makes through
//!   [`StepContext`] read the composed program as it stands and write nothing back.
//! - **A failing observer is detached, not propagated.** A panic is caught, the observer is
//!   removed for the rest of the session, the failure is counted
//!   ([`Session::balance_observer_failures`](crate::Session::balance_observer_failures)), and the
//!   install completes exactly as it would have.
//!
//! What the controller read is reported as it read it. Nothing here interprets a token or
//! re-implements the target match, so the same observer works with any version of the matcher.

use std::sync::{Arc, Mutex};

use super::vindex3::{
    compose_overrides, encode_raw_prompt, top_k_probs, v3_continuation, V3Runtime,
};
use crate::error::LqlError;
use larql_vindex::format::vindex3::knowledge::KnowledgeOverlay;
use larql_vindex::tokenizers::Tokenizer;

/// What the controller decided at one iteration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BalanceDecision {
    /// The probability is inside the band: the install keeps its current state and stops.
    InBand,
    /// Below the band: the down vector is multiplied by `factor` and the loop continues.
    Amplify { factor: f32 },
    /// Above the band: the down vector is multiplied by `factor` and the loop continues.
    Shrink { factor: f32 },
    /// Below the band with no improvement for too many iterations: stop, and roll back to the
    /// best amplified state if there was one.
    Saturated,
}

/// Why the loop ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BalanceTermination {
    InBand,
    Saturated,
    IterationsExhausted,
}

/// One iteration, reported before the controller acts on its decision.
#[derive(Debug, Clone, PartialEq)]
pub struct BalanceStep {
    pub iteration: usize,
    pub layer: usize,
    pub feature: usize,
    /// The target probability the controller read for the canonical prompt.
    pub controller_target_prob: f64,
    /// L2 norm of the slot's down vector when that probability was read.
    pub down_norm: f32,
    pub decision: BalanceDecision,
}

/// The loop's end, reported after any rollback.
#[derive(Debug, Clone, PartialEq)]
pub struct BalanceEnd {
    /// Iterations that read a probability.
    pub iterations: usize,
    pub termination: BalanceTermination,
    /// Whether the down vector was restored to an earlier best state.
    pub rolled_back: bool,
    /// L2 norm of the slot's down vector that the install keeps.
    pub final_down_norm: f32,
}

/// Receives the balance pass of every COMPOSE install in a session while set.
pub trait BalanceObserver: Send {
    /// Called once per iteration, before the controller acts.
    fn step(&mut self, step: &BalanceStep, context: &StepContext<'_>);
    /// Called once when the loop has ended.
    fn end(&mut self, end: &BalanceEnd, context: &StepContext<'_>);
}

/// Read-only access to the install being balanced.
pub struct StepContext<'a> {
    runtime: &'a V3Runtime,
    tokenizer: &'a Tokenizer,
    overlay: &'a KnowledgeOverlay,
    bos: Option<u32>,
    entity: &'a str,
    relation: &'a str,
    target: &'a str,
    canonical_prompt: &'a str,
}

impl<'a> StepContext<'a> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        runtime: &'a V3Runtime,
        tokenizer: &'a Tokenizer,
        overlay: &'a KnowledgeOverlay,
        bos: Option<u32>,
        entity: &'a str,
        relation: &'a str,
        target: &'a str,
        canonical_prompt: &'a str,
    ) -> Self {
        Self {
            runtime,
            tokenizer,
            overlay,
            bos,
            entity,
            relation,
            target,
            canonical_prompt,
        }
    }

    pub fn entity(&self) -> &str {
        self.entity
    }
    pub fn relation(&self) -> &str {
        self.relation
    }
    pub fn target(&self) -> &str {
        self.target
    }
    pub fn canonical_prompt(&self) -> &str {
        self.canonical_prompt
    }

    /// The top `k` next-token `(id, probability)` pairs for `prompt` through the composed
    /// program as it stands at this step. A forward pass; writes nothing.
    pub fn top_k(&self, prompt: &str, k: usize) -> Result<Vec<(u32, f32)>, LqlError> {
        let ids = encode_raw_prompt(self.tokenizer, prompt, self.bos)?;
        let continuation = v3_continuation(self.runtime)?;
        let output = match compose_overrides(self.runtime, self.overlay)? {
            Some(overrides) => self.runtime.execute_streaming_overlaid(
                &ids,
                &overrides,
                &continuation,
                &mut |_| Ok(()),
            ),
            None => self
                .runtime
                .execute_streaming(&ids, &continuation, &mut |_| Ok(())),
        }
        .map_err(|e| LqlError::exec("balance trace: forward pass failed", e))?;
        let logits = output
            .logits
            .ok_or_else(|| LqlError::Execution("balance trace: no output head".into()))?;
        Ok(top_k_probs(&logits, k))
    }

    /// The vocabulary entry for `id`, decoded on its own (leading space kept).
    pub fn decode(&self, id: u32) -> String {
        self.tokenizer.decode(&[id], false).unwrap_or_default()
    }
}

/// One recorded event.
#[derive(Debug, Clone, PartialEq)]
pub enum TraceEvent {
    Step(BalanceStep),
    End(BalanceEnd),
}

/// An observer that keeps what it sees. Clone the handle before boxing it into a session to read
/// the events afterwards.
#[derive(Clone, Default)]
pub struct TraceRecorder {
    events: Arc<Mutex<Vec<TraceEvent>>>,
}

impl TraceRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn events(&self) -> Vec<TraceEvent> {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn push(&self, event: TraceEvent) {
        self.events
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(event);
    }
}

impl BalanceObserver for TraceRecorder {
    fn step(&mut self, step: &BalanceStep, _context: &StepContext<'_>) {
        self.push(TraceEvent::Step(step.clone()));
    }

    fn end(&mut self, end: &BalanceEnd, _context: &StepContext<'_>) {
        self.push(TraceEvent::End(end.clone()));
    }
}

/// The session's observer and its failure count.
#[derive(Default)]
pub(crate) struct ObserverSlot {
    observer: Option<Box<dyn BalanceObserver>>,
    failures: usize,
}

impl ObserverSlot {
    pub(crate) fn set(&mut self, observer: Option<Box<dyn BalanceObserver>>) {
        self.observer = observer;
    }

    pub(crate) fn failures(&self) -> usize {
        self.failures
    }

    pub(crate) fn is_set(&self) -> bool {
        self.observer.is_some()
    }

    /// Runs `call` on the observer. A panic detaches the observer and is counted; it never
    /// reaches the caller.
    fn guarded(&mut self, call: impl FnOnce(&mut dyn BalanceObserver)) {
        let Some(observer) = self.observer.as_mut() else {
            return;
        };
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            call(observer.as_mut());
        }));
        if outcome.is_err() {
            self.failures += 1;
            // The observer's state is unknown after a panic. Dropping it may panic again, and that
            // must not reach the install either.
            if let Some(broken) = self.observer.take() {
                let _ =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || drop(broken)));
            }
        }
    }

    pub(crate) fn step(&mut self, step: &BalanceStep, context: &StepContext<'_>) {
        self.guarded(|o| o.step(step, context));
    }

    pub(crate) fn end(&mut self, end: &BalanceEnd, context: &StepContext<'_>) {
        self.guarded(|o| o.end(end, context));
    }
}
