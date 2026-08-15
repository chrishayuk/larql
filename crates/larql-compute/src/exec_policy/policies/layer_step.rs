//! A static `(layer × step)` mask — the deliberately stupid policy.
//!
//! It predicts nothing. Its whole purpose is to prove the seam works:
//! that a named layer's expert group can be deleted on the production
//! dispatch path, that the ledger sees exactly the bytes that were
//! avoided, and that removing the policy restores canonical execution.
//! BW-C1/C2 already falsified router weight and contribution norm as
//! predictors, so anything that looked like a predictor here would be
//! unearned; layer depth is the only signal with evidence behind it
//! (safe% rising 65.6% → 74.0% → 82.3% early→late), and "which layers"
//! is exactly what this policy takes as an argument rather than guesses.

use std::fmt::Write as _;

use crate::exec_policy::{ExecutionPolicy, ExecutionStrategy, ExpertGroupSite};
use crate::movement_ledger::Phase;

/// Which token indices within a phase a mask applies to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepSelector {
    /// Every token, including one with no declared index.
    Every,
    /// Exactly one token — the single-shot form the gate test uses, and
    /// the only one that makes "one known layer, one known token" an
    /// exact statement rather than a first-visit approximation.
    Exactly(u64),
    /// Every `n`-th token (`step % n == 0`). `n == 0` selects nothing
    /// rather than dividing by zero.
    EveryNth(u64),
    /// An explicit set of token indices.
    OneOf(Vec<u64>),
}

impl StepSelector {
    /// Whether this selector covers `step`.
    ///
    /// Every variant except [`Self::Every`] REFUSES an undeclared step
    /// (`None`) rather than treating it as 0 — 0 is a legitimate index a
    /// caller can select, so guessing it would make "skip on token 0"
    /// fire on every unattributed boundary as well.
    pub fn selects(&self, step: Option<u64>) -> bool {
        match self {
            StepSelector::Every => true,
            StepSelector::Exactly(want) => step == Some(*want),
            StepSelector::EveryNth(n) => match (step, n) {
                (Some(s), n) if *n > 0 => s % n == 0,
                _ => false,
            },
            StepSelector::OneOf(set) => step.is_some_and(|s| set.contains(&s)),
        }
    }

    fn label(&self) -> String {
        match self {
            StepSelector::Every => "every".to_string(),
            StepSelector::Exactly(s) => format!("exactly({s})"),
            StepSelector::EveryNth(n) => format!("every-{n}th"),
            StepSelector::OneOf(set) => format!("one-of({} steps)", set.len()),
        }
    }
}

/// Skip the routed expert group at a fixed set of layers, on the tokens a
/// [`StepSelector`] names, optionally restricted to one phase.
pub struct LayerStepMask {
    name: String,
    layers: Vec<usize>,
    phase: Option<Phase>,
    steps: StepSelector,
}

impl LayerStepMask {
    /// Skip at `layers`, on every token of every phase. Narrow it with
    /// [`Self::with_phase`] / [`Self::with_steps`].
    pub fn new(layers: impl IntoIterator<Item = usize>) -> Self {
        let mut layers: Vec<usize> = layers.into_iter().collect();
        layers.sort_unstable();
        layers.dedup();
        let mut mask = Self {
            name: String::new(),
            layers,
            phase: None,
            steps: StepSelector::Every,
        };
        mask.rebuild_name();
        mask
    }

    /// Restrict to one generation phase. Without this the mask fires
    /// during prefill too, which for a routed MoE model on this engine
    /// means ~130 extra firings on `gpt-oss-20b`'s chat template before
    /// decode step 0 — see [`crate::exec_policy::step`].
    pub fn with_phase(mut self, phase: Phase) -> Self {
        self.phase = Some(phase);
        self.rebuild_name();
        self
    }

    /// Restrict to the tokens `steps` names.
    pub fn with_steps(mut self, steps: StepSelector) -> Self {
        self.steps = steps;
        self.rebuild_name();
        self
    }

    /// The name is derived from the configuration, not supplied, so the
    /// report line can never describe a policy other than the one that
    /// ran.
    fn rebuild_name(&mut self) {
        let mut n = String::from("layer-step-mask{layers=[");
        for (i, l) in self.layers.iter().enumerate() {
            if i > 0 {
                n.push(',');
            }
            let _ = write!(n, "{l}");
        }
        let _ = write!(
            n,
            "],phase={},steps={}}}",
            match self.phase {
                Some(p) => p.label(),
                None => "any",
            },
            self.steps.label(),
        );
        self.name = n;
    }
}

impl ExecutionPolicy for LayerStepMask {
    fn name(&self) -> &str {
        &self.name
    }

    fn expert_group(&self, site: &ExpertGroupSite) -> ExecutionStrategy {
        // A phase-restricted mask refuses an undeclared phase rather than
        // assuming decode — the same refusal the ledger makes, for the
        // same reason: assuming decode is exactly how prefill positions
        // got counted as decode steps once already.
        if let Some(want) = self.phase {
            if site.phase != Some(want) {
                return ExecutionStrategy::Canonical;
            }
        }
        if !self.layers.contains(&site.layer) || !self.steps.selects(site.step) {
            return ExecutionStrategy::Canonical;
        }
        ExecutionStrategy::Skip
    }
}

#[cfg(test)]
#[path = "../tests/layer_step.rs"]
mod tests;
