//! `LARQL_EXEC_POLICY` — the one way to arm the seam from outside the
//! process, so a bytes-and-latency A/B is a change of environment
//! rather than a change of code.
//!
//! # Grammar
//!
//! ```text
//! skip-layers:<L>[,<L>...]            every decode token
//! skip-layers:<L>[,<L>...]:every-<N>  every Nth decode token
//! skip-layers:<L>[,<L>...]:token-<N>  exactly decode token N
//! trace:<path>                        replay a recorded oracle trace
//! ```
//!
//! Every form is DECODE-ONLY and cannot be made otherwise from the
//! environment. Skipping expert groups during prefill perturbs the
//! prompt's own representation, which is a different experiment with a
//! different control; making it reachable by a typo in a benchmark
//! command is not worth the flexibility.
//!
//! # A malformed value is a hard error, never a silent no-op
//!
//! [`from_env`] returns `Err` rather than `None` for anything it cannot
//! parse, and the CLI is expected to exit on it. This is the whole
//! reason the function exists in this shape: the failure mode of a
//! silent fallback is an A/B that compares canonical against canonical
//! and reports "no change" — an instrument that cannot fail on known-
//! different input, which is exactly the class of error the BW10 ledger
//! was built to prevent. A typo must stop the run, not quietly answer
//! the wrong question.

use std::sync::Arc;

use super::policies::{LayerStepMask, StepSelector, TraceReplay};
use super::trace::Trace;
use super::ExecutionPolicy;
use crate::movement_ledger::Phase;
use crate::options;

const SKIP_LAYERS: &str = "skip-layers:";
const TRACE: &str = "trace:";
const EVERY: &str = "every-";
const TOKEN: &str = "token-";

/// One-line summary of the grammar, reused in every error so a failed
/// run tells the operator what to type instead.
const USAGE: &str = "expected `skip-layers:<layer>[,<layer>...][:every-<n>|:token-<n>]` \
     or `trace:<path>`";

/// Build the policy named by `spec`.
pub fn parse(spec: &str) -> Result<Arc<dyn ExecutionPolicy>, String> {
    let spec = spec.trim();
    if let Some(rest) = spec.strip_prefix(SKIP_LAYERS) {
        return parse_skip_layers(rest);
    }
    if let Some(path) = spec.strip_prefix(TRACE) {
        let trace = Trace::read(path.trim())?;
        if trace.is_empty() {
            return Err(format!(
                "exec trace {path:?} records no skips — replaying it is identical to \
                 canonical execution, which is almost certainly not what was intended"
            ));
        }
        return Ok(Arc::new(TraceReplay::new(
            Phase::Decode,
            trace.skips.iter().copied(),
        )));
    }
    Err(format!("unrecognised exec policy {spec:?} — {USAGE}"))
}

fn parse_skip_layers(rest: &str) -> Result<Arc<dyn ExecutionPolicy>, String> {
    let (layer_list, steps) = match rest.split_once(':') {
        Some((l, s)) => (l, parse_steps(s)?),
        None => (rest, StepSelector::Every),
    };
    let mut layers = Vec::new();
    for field in layer_list.split(',') {
        let field = field.trim();
        if field.is_empty() {
            return Err(format!("empty layer index in {layer_list:?} — {USAGE}"));
        }
        layers.push(
            field
                .parse::<usize>()
                .map_err(|_| format!("layer {field:?} is not an integer — {USAGE}"))?,
        );
    }
    if layers.is_empty() {
        return Err(format!("no layers named — {USAGE}"));
    }
    Ok(Arc::new(
        LayerStepMask::new(layers)
            .with_phase(Phase::Decode)
            .with_steps(steps),
    ))
}

fn parse_steps(s: &str) -> Result<StepSelector, String> {
    let s = s.trim();
    if let Some(n) = s.strip_prefix(EVERY) {
        let n: u64 = n
            .parse()
            .map_err(|_| format!("period {n:?} is not an integer — {USAGE}"))?;
        if n == 0 {
            return Err("period must be >= 1: `every-0` would skip nothing".to_string());
        }
        return Ok(StepSelector::EveryNth(n));
    }
    if let Some(n) = s.strip_prefix(TOKEN) {
        let n: u64 = n
            .parse()
            .map_err(|_| format!("token index {n:?} is not an integer — {USAGE}"))?;
        return Ok(StepSelector::Exactly(n));
    }
    Err(format!("unrecognised step selector {s:?} — {USAGE}"))
}

/// Install the policy named by `LARQL_EXEC_POLICY`, if any.
///
/// `Ok(None)` means the variable is unset and execution stays canonical.
/// `Err` means it was set to something unparseable and the caller should
/// STOP — see the module doc for why this must not degrade to canonical.
///
/// The returned guard uninstalls on drop, so a caller holds it for the
/// span it wants the policy to cover.
#[must_use = "the policy is uninstalled when the guard drops"]
pub fn from_env() -> Result<Option<super::PolicyGuard>, String> {
    let Some(spec) = options::env_nonempty_value(options::ENV_EXEC_POLICY) else {
        return Ok(None);
    };
    let policy = parse(&spec)?;
    // Announced unconditionally, on stderr, even when the ledger is off.
    // A run whose expert groups are being deleted must never look like
    // an ordinary run in the scrollback — this is the one line that
    // stops a policy-on benchmark being quoted as a baseline.
    eprintln!(
        "[exec-policy] INSTALLED {} — expert groups will be SKIPPED on the decode path. \
         Output and throughput are NOT this model's canonical behaviour.",
        policy.name()
    );
    Ok(Some(super::install(policy)))
}

#[cfg(test)]
#[path = "tests/spec.rs"]
mod tests;
