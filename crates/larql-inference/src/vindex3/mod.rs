//! VINDEX3 inference runtime — the seam where the executable model
//! program meets the generation machinery (VI3-INF-0/1).
//!
//! A VINDEX3 container is not "reorganised weights to reconstruct a
//! transformer from" — it is a closed executable program
//! (`ComponentOpPlan`) plus its operands. The canonical interpreter in
//! `larql-vindex` owns model *meaning* (operation order, residual
//! placement, optional operations, span policy); a `PlanBackend` owns
//! only the arithmetic. This module gives that program a home inside
//! the inference runtime **without translating it back into a
//! V2-shaped model**: no [`ModelWeights`](crate::ModelWeights), no
//! family detection, no `ModelArchitecture` reconstruction.
//!
//! The seam is deliberately tiny. Generation needs logits and state
//! progression, so that is the whole contract:
//!
//! ```text
//! generate (sampler / EOS / streaming)
//!        │
//!   LogitsSession        ← format-neutral: prefill, step, position
//!        │
//!  Vindex3Session        ← wraps the canonical DecodeSession
//!        │
//!   PlanBackend          ← reference / production / Metal arithmetic
//! ```
//!
//! What deliberately does **not** exist here: a `load_vindex3() ->
//! ModelWeights` bridge, or a `match version {2 => …, 3 => …}` inside
//! `open_inference_vindex`. The two formats have different authority
//! models and converge only above [`LogitsSession`].
//!
//! Rung status: VI3-INF-0/1 — [`LogitsSession::prefill`] feeds the
//! prompt tokenwise through [`LogitsSession::step`] (a semantic
//! integration gate, not a fast path). VI3-INF-2 — continuation state
//! is a caller-side [`KvState`] provider (`session_with_kv`),
//! `prepare`d with the plan's explicit per-layer KV geometry
//! ([`LayerKvGeometry`]): row width and sliding/full window come from
//! the executable program, never from `ModelArchitecture` inference.
//! VI3-INF-3 — batch prefill populates the **same** caller provider
//! ([`Vindex3Runtime::prefill_into`]), which also owns the logical
//! continuation position, and `session_with_kv` resumes from it; no
//! batch-state → decode-state translation exists anywhere.

mod explain;
mod generate;
mod runtime;
mod session;

#[cfg(test)]
mod tests;

pub use generate::{
    continue_session, continue_session_masked, generate_session, LogitsMask, SessionGeneration,
};
pub use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
pub use runtime::{open_component, OpenPolicy, OpenedComponent, PreparedVindex3, Vindex3Runtime};
pub use session::{LogitsSession, Vindex3Session};

pub use explain::{
    ExplainAttention, ExplainEmbedding, ExplainFfn, ExplainKvGeometry, ExplainLayer,
    ExplainOperand, ExplainOutput, ExplainPlan,
};

// The continuation-state seam, re-exported so engine authors reach it
// from the runtime module without deep `larql_vindex` paths.
pub use larql_vindex::format::vindex3::opplan::exec::kv::{
    plan_kv_geometry, KvState, LayerKvGeometry, RowKvState,
};
// The observation seam (LQL-2 TRACE): subscribers to the canonical
// executor's step boundaries — one execution path, many consumers.
pub use larql_vindex::format::vindex3::opplan::exec::observe::{
    RecordingObserver, StepEvent, StepObserver,
};
// The batch-execution taps (V3-LQL-3B): plane events streamed from the
// one traversal, consumed by residual capture and retrieval keys.
pub use larql_vindex::format::vindex3::opplan::exec::{FinalOutput, LayerTrace, PlaneEvent};
// The operand-source seam (V3-LQL-3B compose): base representation +
// overlay override → effective operand — how a mutation reaches
// execution without touching the container.
pub use larql_vindex::format::vindex3::opplan::exec::operands::{
    OperandEdit, OperandOverrides, OperandSource,
};
