//! Opening a VINDEX3 container as an inference runtime.
//!
//! [`open_component`] is the **one** opening authority in the workspace:
//! inspect the container, close the component's plan, bind its operands.
//! `larql serve`, `larql run` and `larql vindex3 exec` all come through
//! it, so the same container under the same [`OpenPolicy`] means the same
//! program bound to the same bytes whichever front door was used. What
//! *executes* that program — CPU, Metal, an interpreter or a lowering —
//! is not decided here: that is a realisation, chosen by the caller and
//! handed in as the backend.

use std::path::Path;

use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::kv::KvState;
use larql_vindex::format::vindex3::opplan::exec::operands::{
    OperandOverrides, OperandSource, OperandStore, RepresentationSource,
};
use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use larql_vindex::format::vindex3::opplan::exec::{
    execute_plan_streaming, prefill_plan, prefill_prepared, FinalOutput, PlaneEvent,
};
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ClosureDefect, ComponentOpPlan};

use crate::error::InferenceError;

use super::session::Vindex3Session;

/// How a component's operands are bound when it is opened.
///
/// Two separate questions, deliberately: `want` is *what* encoding
/// execution asks for — `None` executes the canonical bytes and never
/// looks for a compiled pack, so adding packs to a container cannot
/// change what such a caller runs — and `source` is *whether* the runtime
/// may manufacture that encoding at load or must find it already
/// compiled. The default is the canonical program, which is what a
/// server binds.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct OpenPolicy {
    /// The stored encoding to serve from, if one is compiled.
    pub want: Option<String>,
    pub source: RepresentationSource,
}

/// What opening a component yields: the executable plan, its operand
/// store, the two identities the container declares about itself, and
/// the encoding the store was asked for.
pub struct OpenedComponent {
    pub plan: ComponentOpPlan,
    pub store: OperandStore,
    /// The container's self-declared model name (`index.model`) — the
    /// identity authority. Callers must not fall back to a directory
    /// name when this is non-empty.
    pub model_name: String,
    /// The model family the container declares.
    pub family: String,
    /// [`OpenPolicy::want`], carried so a caller can report what it
    /// asked for beside what the store bound.
    pub want: Option<String>,
}

/// Inspect the container, plan `component`'s operations, and open the
/// operand store under `policy` — solely from the container's own
/// contents. Kept outside the generic impl so the whole opening path
/// (and its refusals) is one instantiation regardless of backend.
///
/// A component whose stack does not close refuses to open, with every
/// defect in the error: an unclosed program is never "best-effort"
/// executed by anyone.
pub fn open_component(
    container: &Path,
    component: &str,
    policy: OpenPolicy,
) -> Result<OpenedComponent, InferenceError> {
    let inspection = inspect_container(container, false)?;
    let outcome = plan_component_ops(&inspection, container, component)?;
    if !outcome.closed() {
        return Err(unclosed_component(component, &outcome.defects));
    }
    let plan = outcome.plan.ok_or_else(|| {
        InferenceError::Parse(format!("component `{component}` produced no plan"))
    })?;
    let store = OperandStore::open_for(
        container,
        &inspection,
        policy.want.as_deref(),
        policy.source,
    )?;
    // The container names itself (`index.model`) — identity travels
    // with the artifact, never a sidecar or a directory name — and
    // declares its own family, which is the only authority a V3
    // binding has (there is no architecture registry entry to ask).
    Ok(OpenedComponent {
        plan,
        store,
        model_name: inspection.index.model.clone(),
        family: inspection.index.family.clone(),
        want: policy.want,
    })
}

/// Built outside the generic impl so the refusal exists (and is
/// counted) once, not once per backend instantiation. Defensive:
/// unreachable while every encoded text component carries a head,
/// kept so a headless component fails closed instead of panicking.
pub(super) fn headless_prefill_error() -> InferenceError {
    InferenceError::Parse(
        "prefill produced no logits — the component carries no output head".to_string(),
    )
}

/// A component whose stack does not fully classify into the declared
/// operations refuses to open, with the defects in the error. An
/// unclosed program must not be "best-effort" executed.
fn unclosed_component(component: &str, defects: &[ClosureDefect]) -> InferenceError {
    let listed: Vec<String> = defects.iter().map(|d| d.to_string()).collect();
    InferenceError::Parse(format!(
        "component `{component}` does not close: {}",
        listed.join("; ")
    ))
}

/// One opened container component: the executable plan, its operand
/// store, and the arithmetic backend. Owns what a [`Vindex3Session`]
/// borrows, so sessions can be created, dropped, and re-created (fresh
/// conversations) without re-planning the container.
pub struct Vindex3Runtime<B: PlanBackend> {
    plan: ComponentOpPlan,
    store: OperandStore,
    backend: B,
    model_name: String,
    family: String,
}

impl<B: PlanBackend> Vindex3Runtime<B> {
    /// Open `component` from the container as its canonical program —
    /// [`open_with`](Self::open_with) under the default [`OpenPolicy`].
    pub fn open(container: &Path, component: &str, backend: B) -> Result<Self, InferenceError> {
        Self::open_with(container, component, backend, OpenPolicy::default())
    }

    /// Open `component` from the container with its operands bound under
    /// `policy`, refusing any closure defect (see [`unclosed_component`]'s
    /// doc). The backend is the realisation; the policy is what it runs.
    pub fn open_with(
        container: &Path,
        component: &str,
        backend: B,
        policy: OpenPolicy,
    ) -> Result<Self, InferenceError> {
        let OpenedComponent {
            plan,
            store,
            model_name,
            family,
            want: _,
        } = open_component(container, component, policy)?;
        Ok(Self {
            plan,
            store,
            backend,
            model_name,
            family,
        })
    }

    /// The container's browse view (V3-LQL-3A): the query surface's
    /// semantic roles bound to this runtime's own plan and operand
    /// store, so the queryable view and the executed program cannot
    /// name different bytes. `tokenizer` decodes feature annotations.
    pub fn knowledge_view(
        &self,
        tokenizer: &larql_vindex::tokenizers::Tokenizer,
    ) -> Result<larql_vindex::format::vindex3::knowledge::KnowledgeView, InferenceError> {
        Ok(
            larql_vindex::format::vindex3::knowledge::KnowledgeView::from_plan(
                &self.plan,
                &self.store,
                tokenizer,
            )?,
        )
    }

    /// The container's self-declared model name (`index.model`) — the
    /// identity authority; callers must not fall back to directory
    /// names when this is non-empty.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// The model family the container declares.
    ///
    /// Read from the same inspection that produced the plan — a V3
    /// binding has no architecture registry to ask, so the container's
    /// own declaration is the authority.
    pub fn family(&self) -> &str {
        &self.family
    }

    /// Open an incremental session at position zero. Each call loads
    /// the operands in the backend's declared weight format.
    pub fn session(&self) -> Result<Vindex3Session<'_, B>, InferenceError> {
        Vindex3Session::new(&self.plan, &self.store, &self.backend)
    }

    /// [`session`](Self::session) with a mutation overlay's operand
    /// edits applied — the session captures the effective operands at
    /// construction.
    pub fn session_overlaid(
        &self,
        overrides: &OperandOverrides,
    ) -> Result<Vindex3Session<'_, B>, InferenceError> {
        Vindex3Session::new(
            &self.plan,
            OperandSource::overlaid(&self.store, overrides),
            &self.backend,
        )
    }

    /// Open a session whose continuation state lives in — and outlives
    /// the session as — the caller's [`KvState`] provider (VI3-INF-2).
    /// The session continues from `kv.position()`, so this is also the
    /// resume path after [`prefill_into`](Self::prefill_into). See
    /// [`Vindex3Session::with_kv_state`] for the provider contract.
    pub fn session_with_kv<'a>(
        &'a self,
        kv: &'a mut dyn KvState,
    ) -> Result<Vindex3Session<'a, B>, InferenceError> {
        Vindex3Session::with_kv_state(&self.plan, &self.store, &self.backend, kv)
    }

    /// Batch-prefill `tokens` into the caller's provider (VI3-INF-3)
    /// and return the last position's logits, so generation can sample
    /// the first continuation token before resuming decode over the
    /// **same** provider via [`session_with_kv`](Self::session_with_kv).
    /// A provider already holding state is extended from its logical
    /// position — a long prompt can prefill in chunks.
    pub fn prefill_into(
        &self,
        tokens: &[u32],
        kv: &mut dyn KvState,
    ) -> Result<Vec<f32>, InferenceError> {
        let out = prefill_plan(&self.plan, &self.store, tokens, &self.backend, kv)?;
        out.logits.ok_or_else(headless_prefill_error)
    }

    /// One observed analysis pass over the component's plan (V3-LQL-3B):
    /// execute `tokens` and stream every plane event — the embedded
    /// rows and each layer's residual taps — to `sink`, returning the
    /// final output. This is the **same** `traverse` every other entry
    /// point runs (observation is subscription, never a second
    /// executor); no continuation state is kept, so use it for
    /// analyses (residual capture, retrieval keys), not generation.
    pub fn execute_streaming(
        &self,
        tokens: &[u32],
        sink: &mut dyn FnMut(PlaneEvent) -> Result<(), larql_vindex::VindexError>,
    ) -> Result<FinalOutput, InferenceError> {
        Ok(execute_plan_streaming(
            &self.plan,
            &self.store,
            tokens,
            &self.backend,
            None,
            sink,
        )?)
    }

    /// The runtime's operand resolver (base representation, no
    /// overlay) — for analyses that read stored operands through the
    /// same resolution execution uses (e.g. the compose install's
    /// layer-norm statistics).
    pub fn operands(&self) -> OperandSource<'_> {
        (&self.store).into()
    }

    /// [`execute_streaming`](Self::execute_streaming) with a mutation
    /// overlay's operand edits applied (V3-LQL-3B compose): the same
    /// canonical traversal, resolving operands as base + override →
    /// effective. An empty overrides value is bit-identical to the
    /// plain call.
    pub fn execute_streaming_overlaid(
        &self,
        tokens: &[u32],
        overrides: &OperandOverrides,
        sink: &mut dyn FnMut(PlaneEvent) -> Result<(), larql_vindex::VindexError>,
    ) -> Result<FinalOutput, InferenceError> {
        Ok(execute_plan_streaming(
            &self.plan,
            OperandSource::overlaid(&self.store, overrides),
            tokens,
            &self.backend,
            None,
            sink,
        )?)
    }

    /// [`prefill_into`](Self::prefill_into) with a mutation overlay's
    /// operand edits applied — generation over an edited program.
    pub fn prefill_into_overlaid(
        &self,
        tokens: &[u32],
        overrides: &OperandOverrides,
        kv: &mut dyn KvState,
    ) -> Result<Vec<f32>, InferenceError> {
        let out = prefill_plan(
            &self.plan,
            OperandSource::overlaid(&self.store, overrides),
            tokens,
            &self.backend,
            kv,
        )?;
        out.logits.ok_or_else(headless_prefill_error)
    }

    /// [`session_with_kv`](Self::session_with_kv) with a mutation
    /// overlay's operand edits applied. The session captures the
    /// effective operands at construction, so the overlay may change
    /// afterwards without affecting a running continuation.
    pub fn session_with_kv_overlaid<'a>(
        &'a self,
        kv: &'a mut dyn KvState,
        overrides: &OperandOverrides,
    ) -> Result<Vindex3Session<'a, B>, InferenceError> {
        Vindex3Session::with_kv_state(
            &self.plan,
            OperandSource::overlaid(&self.store, overrides),
            &self.backend,
            kv,
        )
    }

    /// The component's executable plan — the model-meaning authority.
    pub fn plan(&self) -> &ComponentOpPlan {
        &self.plan
    }

    /// The arithmetic backend this runtime executes with.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Lower this component's operands into the backend's execution
    /// form, once, and hand back the prepared model.
    ///
    /// This is the boundary between model lifetime and request
    /// lifetime. Opening a runtime plans the component and maps its
    /// store — cheap, milliseconds. *Preparing* it converts every
    /// operand into the arithmetic the backend will actually run and
    /// gives the backend its chance to place them on a device — the
    /// expensive step, and the one that must happen once per served
    /// model rather than once per request.
    ///
    /// Consumes the runtime so there is one answer to "is this model
    /// prepared", but *keeps* the operand store: it is only mmap
    /// handles, and holding it leaves the browse view
    /// ([`knowledge_view`](PreparedVindex3::knowledge_view)) and future
    /// slice preparations available on a prepared model.
    pub fn prepare(self) -> Result<PreparedVindex3<B>, InferenceError> {
        self.prepare_slice(ExecutionSlice::Full)
    }

    /// [`prepare`](Self::prepare) for part of the component: only the
    /// slice's operands are lowered. A layer-range shard pays for its
    /// own layers and nothing else.
    pub fn prepare_slice(
        self,
        slice: ExecutionSlice,
    ) -> Result<PreparedVindex3<B>, InferenceError> {
        let operands = PreparedOperands::load(&self.plan, &self.store, &self.backend, slice)?;
        Ok(PreparedVindex3 {
            plan: self.plan,
            store: self.store,
            backend: self.backend,
            operands,
            model_name: self.model_name,
            family: self.family,
        })
    }
}

/// A component whose operands are already in the backend's execution
/// form: the model as a server holds it.
///
/// Sessions and batch prefill both read these operands; neither loads
/// anything. That is the whole point — before this existed, a serve
/// path that batch-prefilled and then decoded materialised the model
/// twice per request.
///
/// Immutable, and therefore shareable: every concurrent request on a
/// model reads one image. Per-request state (continuation K/V,
/// sampling, masks, and eventually patch overlays) lives on the session
/// instead, so nothing a request does can disturb another's weights.
pub struct PreparedVindex3<B: PlanBackend> {
    plan: ComponentOpPlan,
    store: OperandStore,
    backend: B,
    operands: PreparedOperands,
    model_name: String,
    family: String,
}

impl<B: PlanBackend> PreparedVindex3<B> {
    /// Open an incremental session over the resident operands, with the
    /// caller's continuation state. Cheap: no operand touches disk.
    pub fn session_with_kv<'a>(
        &'a self,
        kv: &'a mut dyn KvState,
    ) -> Result<Vindex3Session<'a, B>, InferenceError> {
        Vindex3Session::over_prepared(&self.plan, &self.operands, &self.backend, kv)
    }

    /// Batch-prefill `tokens` into the caller's provider over the
    /// resident operands, returning the last position's logits.
    pub fn prefill_into(
        &self,
        tokens: &[u32],
        kv: &mut dyn KvState,
    ) -> Result<Vec<f32>, InferenceError> {
        let out = prefill_prepared(&self.plan, &self.operands, tokens, &self.backend, kv)?;
        out.logits.ok_or_else(headless_prefill_error)
    }

    /// The component's executable plan — the model-meaning authority.
    pub fn plan(&self) -> &ComponentOpPlan {
        &self.plan
    }

    /// The container's self-declared model name (`index.model`).
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// The model family the container declares.
    pub fn family(&self) -> &str {
        &self.family
    }

    /// The container's browse view, bound to this model's own plan and
    /// operand store — the same surface an unprepared runtime offers,
    /// so preparing a model does not cost it its queryability.
    pub fn knowledge_view(
        &self,
        tokenizer: &larql_vindex::tokenizers::Tokenizer,
    ) -> Result<larql_vindex::format::vindex3::knowledge::KnowledgeView, InferenceError> {
        Ok(
            larql_vindex::format::vindex3::knowledge::KnowledgeView::from_plan(
                &self.plan,
                &self.store,
                tokenizer,
            )?,
        )
    }

    /// The arithmetic backend these operands were lowered for.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// The prepared operands themselves — slice, layer count, whether a
    /// head is present.
    pub fn operands(&self) -> &PreparedOperands {
        &self.operands
    }
}
