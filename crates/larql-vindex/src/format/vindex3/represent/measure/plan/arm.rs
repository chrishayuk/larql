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
use crate::format::vindex3::opplan::exec::decode::DecodeSession;
use crate::format::vindex3::opplan::exec::kv::RowKvState;
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
}

impl<B: PlanBackend> InterpreterArm<B> {
    /// Prepare `plan` from `store` on `backend`, and describe what bound.
    pub fn prepare(
        arm: &str,
        container: PathBuf,
        requested_pack: Option<String>,
        stored_only: bool,
        plan: ComponentOpPlan,
        store: &OperandStore,
        backend: B,
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
        })
    }
}

impl<B: PlanBackend> TeacherForcedArm for InterpreterArm<B> {
    fn describe(&self) -> &ArmDescription {
        &self.description
    }

    fn score(&mut self, ids: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        let mut kv = RowKvState::default();
        let mut session =
            DecodeSession::over_prepared(&self.plan, &self.ops, &self.backend, &mut kv)
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
