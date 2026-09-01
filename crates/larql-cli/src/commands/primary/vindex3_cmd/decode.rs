//! Greedy autoregressive decode over a [`DecodeSession`] — the loop
//! itself, with the ids handed out as they are produced.
//!
//! Sampling is greedy argmax on purpose: generation doubles as a
//! fixture (same ids in → same ids out per backend), and a sampler
//! would put a source of randomness between two runs of a parity
//! comparison. Ids go in and come out as ids; what a caller does with
//! them — print them, decode them to text, compare them — is the
//! caller's wrapper, not the loop's.
//!
//! The wrapper sees every id through a sink *before* the next step
//! runs, and can halt the loop from there. That is how an end-of-
//! sequence token ends a chat turn without the loop knowing what a
//! tokenizer is.

use std::time::Instant;

use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::cpu::{ledger, PhysicalProjectionPlan, PlanTally};
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::timing;

type BoxErr = Box<dyn std::error::Error>;

/// What a sink says after seeing an id: carry on, or stop before the
/// next step runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Flow {
    Continue,
    Halt,
}

/// Where each generated id goes the moment it exists. Receives the id
/// and the logit it was chosen on; an error aborts the decode.
pub(crate) type TokenSink<'s> = dyn FnMut(u32, f32) -> Result<Flow, BoxErr> + 's;

/// One steady step's weight traffic, priced by the process-wide ledger.
pub(crate) type PricedStep = (f64, Vec<(PhysicalProjectionPlan, PlanTally)>);

/// What one greedy decode produced, with its phases timed separately —
/// prompt ingestion and steady decode are different costs and
/// conflating them is how a decode number lies.
pub(crate) struct Decoded {
    /// Every id handed to the sink, in order — including the one the
    /// sink halted on.
    pub(crate) generated: Vec<u32>,
    pub(crate) prompt_seconds: f64,
    /// Wall seconds of each decode step after the first generated token
    /// (which the prompt phase produces).
    pub(crate) step_seconds: Vec<f64>,
    /// The step the ledger priced, when the decode ran long enough to
    /// reach it.
    pub(crate) priced_step: Option<PricedStep>,
}

/// Ingest the prompt one position at a time, then append the argmax of
/// each step's logits until `new_tokens` have been produced or the sink
/// halts.
///
/// The session is the caller's: it may be fresh, or continue a
/// sequence the caller already fed. Nothing is loaded here.
pub(crate) fn greedy_decode<B: PlanBackend>(
    session: &mut DecodeSession<'_, B>,
    prompt: &[u32],
    new_tokens: usize,
    sink: &mut TokenSink<'_>,
) -> Result<Decoded, BoxErr> {
    if prompt.is_empty() {
        return Err("prompt holds no tokens — nothing to condition on".into());
    }
    // Prompt ingestion: every position must pass through the stack to
    // fill the continuation state; only the last position's logits are
    // consumed.
    let prompt_started = Instant::now();
    let mut logits = None;
    for &token in prompt {
        logits = session.step(token)?.logits;
    }
    let prompt_seconds = prompt_started.elapsed().as_secs_f64();
    let logits = logits.ok_or(NO_HEAD)?;
    let (mut next, mut value) = argmax(&logits).ok_or(NO_LOGITS)?;

    let mut generated = Vec::with_capacity(new_tokens);
    let mut step_seconds = Vec::with_capacity(new_tokens);
    let mut priced_step = None;
    for step in 0..new_tokens {
        let id = u32::try_from(next)?;
        generated.push(id);
        if sink(id, value)? == Flow::Halt || step + 1 == new_tokens {
            break;
        }
        // Price ONE steady step's weight traffic. Reset immediately
        // before the step it belongs to: the ledger is process-wide and
        // has been counting since the prompt.
        let price_this_step = step + 1 == new_tokens.saturating_sub(1);
        if price_this_step {
            ledger().reset();
            timing::ledger().reset();
        }
        let started = Instant::now();
        let logits = session.step(id)?.logits.ok_or(NO_HEAD)?;
        (next, value) = argmax(&logits).ok_or(NO_LOGITS)?;
        step_seconds.push(started.elapsed().as_secs_f64());
        if price_this_step {
            priced_step = Some((*step_seconds.last().expect("just pushed"), read_ledger()));
        }
    }
    Ok(Decoded {
        generated,
        prompt_seconds,
        step_seconds,
        priced_step,
    })
}

const NO_HEAD: &str = "plan carries no output head — cannot generate";
const NO_LOGITS: &str = "output head produced no logits";

/// Snapshot every plan's tally.
fn read_ledger() -> Vec<(PhysicalProjectionPlan, PlanTally)> {
    ledger().all().to_vec()
}

/// Index and value of the largest logit; ties keep the first, matching
/// the summary path's fold.
pub(crate) fn argmax(logits: &[f32]) -> Option<(usize, f32)> {
    logits
        .iter()
        .enumerate()
        .fold(None, |best, (index, &value)| match best {
            Some((_, best_value)) if value <= best_value => best,
            _ => Some((index, value)),
        })
}

/// The steady-state window is the tail half of the decode steps — after
/// the page cache and device buffer pools have warmed on the early ones.
const STEADY_TAIL_DIVISOR: usize = 2;

/// Steady-decode timing over the per-step seconds (prompt ingestion and
/// weight load are reported separately by the caller).
#[derive(Debug, PartialEq)]
pub(crate) struct DecodeReport {
    pub(crate) decode_tokens: usize,
    pub(crate) decode_seconds: f64,
    pub(crate) mean_seconds_per_token: f64,
    pub(crate) steady_seconds_per_token: f64,
}

impl DecodeReport {
    /// `None` when no decode step beyond the first token ran — a single
    /// forward has no decode rate to report.
    pub(crate) fn from_steps(step_seconds: &[f64]) -> Option<Self> {
        if step_seconds.is_empty() {
            return None;
        }
        let decode_seconds: f64 = step_seconds.iter().sum();
        let steady_len = (step_seconds.len() / STEADY_TAIL_DIVISOR).max(1);
        let steady = &step_seconds[step_seconds.len() - steady_len..];
        Some(Self {
            decode_tokens: step_seconds.len(),
            decode_seconds,
            mean_seconds_per_token: decode_seconds / step_seconds.len() as f64,
            steady_seconds_per_token: steady.iter().sum::<f64>() / steady.len() as f64,
        })
    }
}
