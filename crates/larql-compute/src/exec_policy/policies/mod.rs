//! Policies that ship with the seam. All of them are deliberately
//! stupid.
//!
//! None of these predicts anything, and that is the point at this stage:
//! the job today is to prove that an expert group can be deleted on the
//! production path and that the ledger sees exactly the bytes it avoided
//! — not to invent the predictor. BW-C1 and BW-C2 have already falsified
//! the two obvious candidates (router weight, `point-biserial = -0.078`;
//! contribution-over-residual norm, `pearson = 0.012 / spearman = 0.038`)
//! on 576 real interventions, so a predictor shipped here now would be
//! guessing against measured nulls.
//!
//! What exists instead is one static mask and one trace replay:
//!
//! - [`LayerStepMask`] — skip at named layers on named tokens. Covers
//!   "explicit layer mask" and "every N tokens" through its
//!   [`StepSelector`], and is what the gate test arms.
//! - [`TraceReplay`] — replay BW-C5's recorded oracle decisions on the
//!   serve path, so the offline ceiling and the production path can be
//!   compared in the same currency.

pub mod layer_step;
pub mod trace_replay;

pub use layer_step::{LayerStepMask, StepSelector};
pub use trace_replay::TraceReplay;
