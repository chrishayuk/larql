//! The format-neutral session contract and its VINDEX3 realisation.

use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession;
use larql_vindex::format::vindex3::opplan::exec::kv::KvState;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandSource;
use larql_vindex::format::vindex3::opplan::exec::prepared::PreparedOperands;
use larql_vindex::format::vindex3::opplan::ComponentOpPlan;

use crate::error::InferenceError;

/// The three refusal messages, built outside the generic impls so each
/// exists (and is counted) once, not once per backend instantiation.
fn headless_plan_error(component: &str) -> InferenceError {
    InferenceError::Parse(format!(
        "component `{component}` carries no output head — a logits session cannot serve it"
    ))
}

fn empty_prompt_error() -> InferenceError {
    InferenceError::Parse("prefill requires at least one prompt token".to_string())
}

/// Defensive: unreachable while construction refuses headless plans,
/// kept so a future plan mutation fails closed instead of panicking.
pub(super) fn missing_logits_error() -> InferenceError {
    InferenceError::Parse(
        "decode step produced no logits despite the plan's output head".to_string(),
    )
}

/// Logits and state progression — everything generation needs from a
/// model runtime, and nothing else. Deliberately not another
/// transformer abstraction: an implementation may be a VINDEX3 plan
/// interpreter, a V2 layer graph, or anything that can advance one
/// token and price a vocabulary.
pub trait LogitsSession {
    /// Consume the prompt and return the logits for its last position.
    /// Errors on an empty prompt — there is no position to price.
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, InferenceError>;

    /// Advance one token and return the logits for the position just
    /// consumed.
    fn step(&mut self, token: u32) -> Result<Vec<f32>, InferenceError>;

    /// Positions consumed so far.
    fn position(&self) -> usize;
}

/// [`LogitsSession`] over the canonical VINDEX3 incremental executor.
///
/// Borrows the plan, operand store, and backend — typically owned by a
/// [`Vindex3Runtime`](super::Vindex3Runtime), which hands sessions
/// out. Construction loads every operand once, in the backend's
/// declared weight format (weights stay resident for the session's
/// lifetime, which is what lets a pointer-keyed device buffer cache
/// hold the model on the GPU).
pub struct Vindex3Session<'a, B: PlanBackend> {
    inner: DecodeSession<'a, B>,
}

impl<'a, B: PlanBackend> Vindex3Session<'a, B> {
    /// Load the plan's operands and open an incremental session at
    /// position zero. The plan must carry an output head — a session
    /// that cannot produce logits cannot serve generation.
    pub fn new<'s>(
        plan: &'a ComponentOpPlan,
        store: impl Into<OperandSource<'s>>,
        backend: &'a B,
    ) -> Result<Self, InferenceError> {
        if plan.output.is_none() {
            return Err(headless_plan_error(&plan.component));
        }
        Ok(Self {
            inner: DecodeSession::new(plan, store, backend)?,
        })
    }

    /// Like [`new`](Self::new), but continuation state lives in — and
    /// outlives the session as — the caller's [`KvState`] provider
    /// (VI3-INF-2). The session continues from `kv.position()`: an
    /// empty provider starts fresh, a batch-prefilled or previously
    /// decoded one resumes (VI3-INF-3) — the provider is the only
    /// position authority. It is `prepare`d with the plan's per-layer
    /// KV geometry, so residency and windowing policy read explicit
    /// program properties, never a family registry.
    pub fn with_kv_state<'s>(
        plan: &'a ComponentOpPlan,
        store: impl Into<OperandSource<'s>>,
        backend: &'a B,
        kv: &'a mut dyn KvState,
    ) -> Result<Self, InferenceError> {
        if plan.output.is_none() {
            return Err(headless_plan_error(&plan.component));
        }
        Ok(Self {
            inner: DecodeSession::with_kv_state(plan, store, backend, kv)?,
        })
    }
}

impl<'a, B: PlanBackend> Vindex3Session<'a, B> {
    /// Open a session over operands the caller already prepared — the
    /// resident-model path. Nothing is loaded here; the session is
    /// per-request state over a per-model image.
    pub fn over_prepared(
        plan: &'a ComponentOpPlan,
        operands: &'a PreparedOperands,
        backend: &'a B,
        kv: &'a mut dyn KvState,
    ) -> Result<Self, InferenceError> {
        if plan.output.is_none() {
            return Err(headless_plan_error(&plan.component));
        }
        Ok(Self {
            inner: DecodeSession::over_prepared(plan, operands, backend, kv)?,
        })
    }
}

impl<B: PlanBackend> LogitsSession for Vindex3Session<'_, B> {
    /// Tokenwise prompt ingestion (VI3-INF-0): every position passes
    /// through the stack to fill the session's KV state; only the last
    /// position's logits are returned. Slow on long prompts by design
    /// at this rung — the batch interpreter becomes the prefill path
    /// at VI3-INF-3.
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, InferenceError> {
        let (&last, rest) = tokens.split_last().ok_or_else(empty_prompt_error)?;
        for &token in rest {
            self.step(token)?;
        }
        self.step(last)
    }

    fn step(&mut self, token: u32) -> Result<Vec<f32>, InferenceError> {
        self.inner
            .step(token)?
            .logits
            .ok_or_else(missing_logits_error)
    }

    fn position(&self) -> usize {
        self.inner.position()
    }
}

impl<B: PlanBackend> Vindex3Session<'_, B> {
    /// [`LogitsSession::step`] with a subscriber on the canonical
    /// step's operation boundaries (LQL-2 TRACE). Observation is
    /// observational: the executor's own parity gate pins that the
    /// observed and unobserved paths are bit-identical.
    pub fn step_observed(
        &mut self,
        token: u32,
        observer: &mut dyn larql_vindex::format::vindex3::opplan::exec::observe::StepObserver,
    ) -> Result<Vec<f32>, InferenceError> {
        self.inner
            .step_observed(token, observer)?
            .logits
            .ok_or_else(missing_logits_error)
    }

    /// The step with interventions armed (V3-INTERVENE-1 carrier
    /// addresses, V3-INTERVENE-2 head addresses): both plans are
    /// admitted against the prepared image before the token executes,
    /// and the firings say which declared interventions applied.
    pub fn step_intervened(
        &mut self,
        token: u32,
        observer: &mut dyn larql_vindex::format::vindex3::opplan::exec::observe::StepObserver,
        interventions: &larql_vindex::format::vindex3::opplan::exec::intervene::InterventionPlan,
        head_interventions: &larql_vindex::format::vindex3::opplan::exec::intervene_heads::HeadInterventionPlan,
    ) -> Result<
        larql_vindex::format::vindex3::opplan::exec::intervene::InterventionStepOutput,
        InferenceError,
    > {
        Ok(self
            .inner
            .step_intervened(token, observer, interventions, head_interventions)?)
    }
}
