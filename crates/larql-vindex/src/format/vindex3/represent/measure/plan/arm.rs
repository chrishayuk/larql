//! **An arm**: one realization of one container, able to teacher-force a
//! sample and say what it bound.
//!
//! The procedure never knows which kind of arm it holds. The interpreter
//! arm lives here, and runs on every platform with the reference backend.
//! The lowered Metal arm lives in `larql-cli` beside `LoweredSession`.
//! Both answer the same two questions: the logits of every position, and
//! the physical facts the attribution proof reads.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::format::vindex3::opplan::exec::backend::PlanBackend;
use crate::format::vindex3::opplan::exec::continuation_registry::SelectedContinuation;
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::operands::OperandStore;
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::ComponentOpPlan;

/// What an arm bound for one object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoundObject {
    /// The encoding of the bytes opened, e.g. `BF16` or `NVFP4`.
    pub encoding: String,
    /// Whether they came from a compiled pack.
    pub stored: bool,
}

/// An arm's account of itself, read after preparation and before any
/// sample is scored.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArmDescription {
    /// The execution arm, as `vindex3 exec --backend` spells it.
    pub arm: String,
    /// The container it opened.
    pub container: PathBuf,
    /// The pack encoding the arm asked for, `None` for canonical.
    pub requested_pack: Option<String>,
    /// Whether the arm was forbidden to manufacture a representation.
    pub stored_only: bool,
    /// Per object: what was bound.
    pub objects: BTreeMap<String, BoundObject>,
    /// Tensors quantised at load. Must be zero for a `stored_only` arm.
    pub runtime_quantised: u64,
}

/// One realization, ready to score samples.
pub trait TeacherForcedArm {
    /// What the arm bound.
    fn describe(&self) -> &ArmDescription;

    /// Every position's full-vocabulary logits for `ids`, from a fresh
    /// state. Position `i` predicts token `i + 1`; the last is included.
    fn score(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, String>;
}

/// The plan interpreter over a prepared image, with any backend.
pub struct InterpreterArm<B: PlanBackend> {
    plan: ComponentOpPlan,
    ops: PreparedOperands,
    backend: B,
    description: ArmDescription,
    /// Who holds each scored sample's state — the caller's selection.
    continuation: SelectedContinuation,
}

impl<B: PlanBackend> InterpreterArm<B> {
    /// Prepare `plan` from `store` on `backend`, and describe what bound.
    /// Each scored sample runs over a fresh provider from `continuation`.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        arm: &str,
        container: PathBuf,
        requested_pack: Option<String>,
        stored_only: bool,
        plan: ComponentOpPlan,
        store: &OperandStore,
        backend: B,
        continuation: SelectedContinuation,
    ) -> Result<Self, String> {
        let ops = PreparedOperands::load(&plan, store, &backend, ExecutionSlice::Full)
            .map_err(|e| format!("{arm}: preparation refused: {e}"))?;
        let objects = store
            .selection()
            .iter()
            .map(|(object, selected)| {
                (
                    object.clone(),
                    BoundObject {
                        encoding: selected.encoding.clone(),
                        stored: selected.stored,
                    },
                )
            })
            .collect();
        let description = ArmDescription {
            arm: arm.to_string(),
            container,
            requested_pack,
            stored_only,
            objects,
            runtime_quantised: store.runtime_quantised(),
        };
        Ok(Self {
            plan,
            ops,
            backend,
            description,
            continuation,
        })
    }
}

impl<B: PlanBackend> TeacherForcedArm for InterpreterArm<B> {
    fn describe(&self) -> &ArmDescription {
        &self.description
    }

    fn score(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        let mut kv = self.continuation.build();
        let mut session =
            DecodeSession::over_prepared(&self.plan, &self.ops, &self.backend, &mut *kv)
                .map_err(|e| e.to_string())?;
        ids.iter()
            .enumerate()
            .map(|(i, &id)| {
                session
                    .step(id)
                    .map_err(|e| format!("position {i}: {e}"))?
                    .logits
                    .ok_or_else(|| format!("position {i}: the plan carries no output head"))
            })
            .collect()
    }
}

/// **An interpreter arm over a container, built in this crate** — the
/// executor's arm builder (MEASURE-PLAN-2).
///
/// The backend is requested by lowering identity from the shipped
/// registry, never constructed here, and the continuation provider is
/// selected by identity from the caller's registry for this arm's own
/// plan geometry. With a `pack`, the arm reads that stored pack and
/// nothing is quantised at load; without one it reads the canonical
/// bytes.
pub fn interpreter_arm_for(
    arm: &str,
    container: &std::path::Path,
    pack: Option<&str>,
    component: &str,
    lowering: &crate::format::vindex3::opplan::exec::lowering::LoweringIdentity,
    continuations: &crate::format::vindex3::opplan::exec::continuation_registry::ContinuationRegistry,
    continuation: &crate::format::vindex3::opplan::exec::continuation_identity::ContinuationIdentity,
) -> Result<InterpreterArm<crate::format::vindex3::opplan::exec::lowering::SharedProvider>, String>
{
    use crate::format::vindex3::inspect::inspect_container;
    use crate::format::vindex3::opplan::exec::continuation::plan_continuation_geometry;
    use crate::format::vindex3::opplan::exec::continuation_authority::ContinuationConfig;
    use crate::format::vindex3::opplan::exec::lowering::LoweringRegistry;
    use crate::format::vindex3::opplan::exec::operands::{OperandStore, RepresentationSource};
    use crate::format::vindex3::opplan::plan_component_ops;

    let inspection =
        inspect_container(container, false).map_err(|e| format!("{arm}: inspect: {e}"))?;
    let plan = plan_component_ops(&inspection, container, component)
        .map_err(|e| format!("{arm}: plan: {e}"))?
        .plan
        .ok_or_else(|| format!("{arm}: component `{component}` has no executable plan"))?;
    let source = match pack {
        Some(_) => RepresentationSource::Stored,
        None => RepresentationSource::Auto,
    };
    let store = OperandStore::open_for(container, &inspection, pack, source)
        .map_err(|e| format!("{arm}: operands: {e}"))?;
    let geometry =
        plan_continuation_geometry(&plan).map_err(|e| format!("{arm}: continuation: {e}"))?;
    let selected = continuations
        .select(continuation, &ContinuationConfig::empty(), &geometry)
        .map_err(|e| format!("{arm}: continuation: {e}"))?;
    let backend = LoweringRegistry::shipped()
        .provider_shared(lowering)
        .map_err(|e| format!("{arm}: lowering: {e}"))?;
    InterpreterArm::prepare(
        arm,
        container.to_path_buf(),
        pack.map(str::to_string),
        pack.is_some(),
        plan,
        &store,
        backend,
        selected,
    )
}
