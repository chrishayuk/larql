//! A block session over remote routed experts: verify several positions in
//! one traversal, then keep an accepted prefix and discard the rest.
//!
//! It adds no execution of its own. A block runs through the decode
//! session's existing causal block traversal
//! ([`DecodeSession::step_many_logits`]), whose routed layers send each
//! contacted worker one multi-row frame per layer, and a discarded suffix
//! is removed with the continuation's own rollback
//! ([`ContinuationProvider::truncate`]).
//!
//! What it owns is the boundary between verified and committed:
//!
//! ```text
//! verify(tokens)   run the block; return every position's logits; nothing committed
//! commit(count)    keep the first `count` positions of that block, discard the rest
//! step(token)      run one position on the single-row route and commit it
//! position()       the committed position
//! ```
//!
//! `step` exists so a caller with nothing to verify does not pay for a
//! one-row block: it runs the decode session's ordinary `step`, which sends
//! each contacted worker one single-row frame per routed layer, over the
//! same continuation, and commits the position it adds.
//!
//! Acceptance is the caller's. The session reports logits and keeps or
//! discards what it is told to; it holds no drafting or acceptance policy.
//!
//! Scope of this first form, each refused at construction, before any
//! state exists:
//!
//! - coordinator operands with placed routed experts only;
//! - a continuation provider that can be rewound, which among the shipped
//!   providers is the row provider only. The session asks the selected
//!   provider itself, by rewinding a fresh one to where it already is:
//!   rollback is a capability a provider states by implementing it, so a
//!   provider that cannot honour it refuses here by name;
//! - no recurrent or latent continuation state, which cannot be rewound;
//! - an output head and a single-stream residual: the block traversal
//!   returns every position's logits only for those, so a plan without
//!   them could step but never verify, and is refused before it can do
//!   either.
//!
//! Failure is the single-position session's rule: any failed block
//! invalidates the session. Its committed position does not move, and the
//! way back is a fresh session replaying the committed prefix.
use crate::error::InferenceError;
use larql_vindex::format::vindex3::opplan::{
    exec::{
        backend::PlanBackend,
        continuation::{plan_continuation_geometry, LayerContinuationGeometry},
        continuation_registry::{BoxedContinuation, SelectedContinuation},
        decode::DecodeSession,
        prepared::{ExecutionSlice, PreparedOperands},
    },
    ComponentOpPlan,
};

pub struct RoutedBlockSession<'a, B: PlanBackend> {
    plan: &'a ComponentOpPlan,
    ops: &'a PreparedOperands,
    backend: &'a B,
    state: BoxedContinuation,
    committed: usize,
    /// Positions of the verified block still awaiting [`Self::commit`].
    pending: Option<usize>,
    failed: bool,
}

impl<'a, B: PlanBackend> RoutedBlockSession<'a, B> {
    pub fn new(
        plan: &'a ComponentOpPlan,
        ops: &'a PreparedOperands,
        backend: &'a B,
        continuation: &SelectedContinuation,
    ) -> Result<Self, InferenceError> {
        // What block verification itself cannot do is refused first, for
        // any operands: these refusals hold whatever the slice.
        if ops.carries_hyper_connection() || ops.carries_attention_residual() {
            return Err(InferenceError::Parse(
                "a routed block session needs a single-stream residual: this component leaves \
                 the stack through a per-position reduction, which block verification does not \
                 implement"
                    .into(),
            ));
        }
        if !ops.has_output() {
            return Err(InferenceError::Parse(
                "a routed block session needs an output head: these operands have no logits to \
                 return"
                    .into(),
            ));
        }
        if !matches!(ops.slice(), ExecutionSlice::RoutedExpertCoordinator) {
            return Err(InferenceError::Parse(
                "a routed block session requires routed-expert coordinator operands".into(),
            ));
        }
        let geometry = plan_continuation_geometry(plan).map_err(InferenceError::Parse)?;
        if let Some(layer) = geometry
            .iter()
            .position(|g| !matches!(g, LayerContinuationGeometry::Kv(_)))
        {
            return Err(InferenceError::Parse(format!(
                "a routed block session supports K/V continuation only; layer {layer} keeps \
                 recurrent or latent state, which cannot be rewound"
            )));
        }
        let mut state = continuation.build();
        // A fresh provider stands at position 0, so this rewind changes
        // nothing in one that supports it.
        state.truncate(0).map_err(|e| {
            InferenceError::Parse(format!(
                "a routed block session requires a continuation provider that can be \
                 rewound: {e}"
            ))
        })?;
        Ok(Self {
            plan,
            ops,
            backend,
            state,
            committed: 0,
            pending: None,
            failed: false,
        })
    }

    /// The committed position: how many positions later blocks build on.
    /// A verified block does not move it until it is committed.
    pub fn position(&self) -> usize {
        self.committed
    }

    /// Run `tokens` from the committed position as one block and return
    /// every position's logits: row `i` is the model's output after
    /// consuming `tokens[..=i]`. Nothing is committed.
    ///
    /// The whole block succeeds or the session is invalidated: no logits
    /// are returned for a block any layer of which failed.
    pub fn verify(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>, InferenceError> {
        if self.failed {
            return Err(InferenceError::Parse(
                "routed block session is invalid after a failed block; start a fresh session \
                 and replay committed input"
                    .into(),
            ));
        }
        if self.pending.is_some() {
            return Err(InferenceError::Parse(
                "a verified block is awaiting commit; commit it (or commit 0 to discard it) \
                 before verifying another"
                    .into(),
            ));
        }
        if tokens.is_empty() {
            return Err(InferenceError::Parse(
                "a block must carry at least one token".into(),
            ));
        }
        let result = (|| {
            let mut session =
                DecodeSession::over_prepared(self.plan, self.ops, self.backend, &mut *self.state)?;
            session.step_many_logits(tokens)
        })();
        match result {
            Ok(logits) => {
                self.pending = Some(tokens.len());
                Ok(logits)
            }
            Err(error) => {
                // Earlier layers have already appended this block's rows.
                // Nothing of it is committed, and nothing more will run here.
                self.failed = true;
                Err(error.into())
            }
        }
    }

    /// Run one position from the committed position on the single-row
    /// route, commit it, and return its logits. Refused while a verified
    /// block awaits commit.
    ///
    /// The position succeeds or the session is invalidated, as for a block.
    /// Construction refuses operands with no output head, so a step without
    /// logits cannot happen; if one ever did, the position would already be
    /// in the continuation with nothing to return, and the session is
    /// invalidated rather than left ahead of its caller.
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>, InferenceError> {
        if self.failed {
            return Err(InferenceError::Parse(
                "routed block session is invalid after a failed block; start a fresh session \
                 and replay committed input"
                    .into(),
            ));
        }
        if self.pending.is_some() {
            return Err(InferenceError::Parse(
                "a verified block is awaiting commit; commit it (or commit 0 to discard it) \
                 before stepping"
                    .into(),
            ));
        }
        let result = (|| {
            let mut session =
                DecodeSession::over_prepared(self.plan, self.ops, self.backend, &mut *self.state)?;
            session.step(token)
        })();
        match result {
            Ok(output) => match output.logits {
                Some(logits) => {
                    self.committed += 1;
                    Ok(logits)
                }
                None => {
                    self.failed = true;
                    Err(InferenceError::Parse(
                        "a routed block session step produced no logits: the plan has no output \
                         head"
                            .into(),
                    ))
                }
            },
            Err(error) => {
                self.failed = true;
                Err(error.into())
            }
        }
    }

    /// Keep the first `count` positions of the verified block and discard
    /// the rest. `commit(0)` discards the whole block. Afterwards the
    /// continuation is exactly what decoding those `count` positions alone
    /// would have left.
    pub fn commit(&mut self, count: usize) -> Result<(), InferenceError> {
        if self.failed {
            return Err(InferenceError::Parse(
                "routed block session is invalid after a failed block".into(),
            ));
        }
        let Some(verified) = self.pending else {
            return Err(InferenceError::Parse(
                "there is no verified block to commit".into(),
            ));
        };
        if count > verified {
            return Err(InferenceError::Parse(format!(
                "cannot commit {count} positions of a block of {verified}"
            )));
        }
        // The rewind validates everything before changing anything, so a
        // refusal leaves the block pending and the session usable.
        self.state
            .truncate(self.committed + count)
            .map_err(|e| InferenceError::Parse(e.to_string()))?;
        self.committed += count;
        self.pending = None;
        Ok(())
    }
}
