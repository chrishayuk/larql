//! Checked input and exact replay for VINDEX3 sessions.
use super::LogitsSession;
use crate::error::InferenceError;
use larql_vindex::format::vindex3::opplan::{
    exec::{
        backend::PlanBackend, continuation_registry::SelectedContinuation, decode::DecodeSession,
        prepared::PreparedOperands,
    },
    ComponentOpPlan,
};

/// One sequential input position. Embeddings have already passed their
/// external connector/scaling; token positions use the container's lookup.
#[derive(Clone, Debug)]
pub enum InputPosition {
    Token(u32),
    Embedding(Vec<f32>),
}

/// Validate the entire prompt before any continuation state advances.
pub fn validate_inputs<B: PlanBackend>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    inputs: &[InputPosition],
) -> Result<(), InferenceError> {
    if inputs.is_empty() {
        return Err(InferenceError::Parse(
            "input must contain at least one position".into(),
        ));
    }
    for input in inputs {
        match input {
            InputPosition::Token(id) => {
                ops.embed_token(plan, backend, *id)?;
            }
            InputPosition::Embedding(row) => {
                if row.len() != ops.hidden() || row.iter().any(|x| !x.is_finite()) {
                    return Err(InferenceError::Parse(format!(
                        "external embedding must contain {} finite values",
                        ops.hidden()
                    )));
                }
            }
        }
    }
    Ok(())
}

pub fn step_input<B: PlanBackend>(
    session: &mut DecodeSession<'_, B>,
    input: &InputPosition,
) -> Result<Vec<f32>, InferenceError> {
    let output = match input {
        InputPosition::Token(id) => session.step(*id)?,
        InputPosition::Embedding(row) => session.step_embedding(row)?,
    };
    output
        .logits
        .ok_or_else(super::session::missing_logits_error)
}

/// No retained KV/recurrent state: replay the complete input history through
/// the canonical tokenwise executor on each step. Exact, deliberately slower.
/// The owned history includes external embeddings, so image prefixes replay too.
///
/// A replay MODE, not a provider: each step's transient state is a fresh
/// provider built from the caller's selection (CONTINUATION-PLUGIN-1, C3).
pub struct ReplaySession<'a, B: PlanBackend> {
    plan: &'a ComponentOpPlan,
    ops: &'a PreparedOperands,
    backend: &'a B,
    continuation: SelectedContinuation,
    history: Vec<InputPosition>,
}
impl<'a, B: PlanBackend> ReplaySession<'a, B> {
    pub fn new(
        plan: &'a ComponentOpPlan,
        ops: &'a PreparedOperands,
        backend: &'a B,
        continuation: SelectedContinuation,
    ) -> Self {
        Self {
            plan,
            ops,
            backend,
            continuation,
            history: Vec::new(),
        }
    }
    pub fn extend_inputs(&mut self, inputs: &[InputPosition]) -> Result<Vec<f32>, InferenceError> {
        validate_inputs(self.plan, self.ops, self.backend, inputs)?;
        let mut state = self.continuation.build();
        let mut session =
            DecodeSession::over_prepared(self.plan, self.ops, self.backend, &mut *state)?;
        let mut logits = None;
        for input in self.history.iter().chain(inputs) {
            logits = Some(step_input(&mut session, input)?);
        }
        // Failure leaves the previous logical history intact.
        self.history.extend_from_slice(inputs);
        logits.ok_or_else(super::session::missing_logits_error)
    }
}
impl<B: PlanBackend> LogitsSession for ReplaySession<'_, B> {
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, InferenceError> {
        self.extend_inputs(
            &tokens
                .iter()
                .copied()
                .map(InputPosition::Token)
                .collect::<Vec<_>>(),
        )
    }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, InferenceError> {
        self.extend_inputs(&[InputPosition::Token(token)])
    }
    fn position(&self) -> usize {
        self.history.len()
    }
}

/// Cached input session; state is supplied by the caller (row or canonical).
pub struct CachedInputSession<'a, B: PlanBackend> {
    plan: &'a ComponentOpPlan,
    ops: &'a PreparedOperands,
    backend: &'a B,
    inner: DecodeSession<'a, B>,
}
impl<'a, B: PlanBackend> CachedInputSession<'a, B> {
    pub fn new(
        plan: &'a ComponentOpPlan,
        ops: &'a PreparedOperands,
        backend: &'a B,
        state: &'a mut dyn larql_vindex::format::vindex3::opplan::exec::kv::KvState,
    ) -> Result<Self, InferenceError> {
        Ok(Self {
            plan,
            ops,
            backend,
            inner: DecodeSession::over_prepared(plan, ops, backend, state)?,
        })
    }
    pub fn extend_inputs(&mut self, inputs: &[InputPosition]) -> Result<Vec<f32>, InferenceError> {
        validate_inputs(self.plan, self.ops, self.backend, inputs)?;
        let mut logits = None;
        for input in inputs {
            logits = Some(step_input(&mut self.inner, input)?);
        }
        logits.ok_or_else(super::session::missing_logits_error)
    }
}
impl<B: PlanBackend> LogitsSession for CachedInputSession<'_, B> {
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, InferenceError> {
        self.extend_inputs(
            &tokens
                .iter()
                .copied()
                .map(InputPosition::Token)
                .collect::<Vec<_>>(),
        )
    }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, InferenceError> {
        self.extend_inputs(&[InputPosition::Token(token)])
    }
    fn position(&self) -> usize {
        self.inner.position()
    }
}
