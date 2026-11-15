//! Shared pieces of the live-view tests: one op of each kind, a bound component, and what it computes.
//!
//! "Live" is a session executing its pending stored-weight ops before they are compiled. Nothing here opens an
//! LQL session: the overlay, the plan and the operand store are driven directly, by token id, through the same
//! `OperandSource::overlaid` seam the session's `INFER` uses.

use std::path::{Path, PathBuf};

use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::knowledge::{ExecutionBinding, KnowledgeOverlay};
use larql_vindex::format::vindex3::opplan::exec::execute_plan;
use larql_vindex::format::vindex3::opplan::exec::lowering::LoweringIdentity;
use larql_vindex::format::vindex3::opplan::exec::operands::{
    OperandOverrides, OperandSource, OperandStore, RepresentationSource,
};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, OperandRef};
use larql_vindex::format::vindex3::weight_patch::structured::{OperandBits, ScalarBits};
use larql_vindex::format::vindex3::weight_patch::{
    apply_declared_patches, declare_patches, read_bf16_tensor, record_op, LowRankColumnsOp,
    StructuredWeightOp, WeightEdit, WeightOp, WeightOpRef,
};
use larql_vindex::{PatchOp, VindexPatch};

use crate::fixture::*;

/// The component every fixture binds its text stack under.
pub const COMPONENT: &str = "target";
/// The object that holds the decoder's matrices.
pub const DECODER: &str = "target.decoder_stack";
/// The prompt every execution here runs, as token ids of the fixture's vocabulary.
pub const TOKENS: [u32; 4] = [3, 17, 5, 42];
pub const PROFILE: &str = "f64-seq-rne1/v1";
pub const F64_ONE: u64 = 0x3ff0_0000_0000_0000;
pub const F64_EIGHTH: u64 = 0x3fc0_0000_0000_0000;
/// The row and the delta of the element edits.
pub const EDIT_ROW: usize = 5;
pub const EDIT_DELTA: f32 = 0.0625;
/// Two disjoint column sets: patch A and patch B of the switching tests.
pub const COLUMNS_A: [usize; 3] = [8, 9, 33];
pub const COLUMNS_B: [usize; 2] = [100, 101];
/// The model name the fixture's containers declare.
pub const MODEL: &str = "wsp";

/// The three kinds of stored-weight op.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Delta,
    RemoveDirection,
    LowRank,
}

pub const KINDS: [Kind; 3] = [Kind::Delta, Kind::RemoveDirection, Kind::LowRank];

/// One op of `kind` on `columns` of the fixture's tensor, bound to `container`.
pub fn op(container: &Path, kind: Kind, columns: &[usize]) -> WeightOp {
    let ([rows, _], _) = read_bf16_tensor(container, TENSOR).unwrap();
    match kind {
        Kind::Delta => {
            let edits: Vec<WeightEdit> = columns
                .iter()
                .map(|&col| WeightEdit {
                    row: EDIT_ROW,
                    col,
                    delta: EDIT_DELTA,
                })
                .collect();
            WeightOp::Delta(record_op(container, TENSOR, &edits).unwrap())
        }
        Kind::RemoveDirection => {
            let mut u = vec![0u64; rows];
            u[0] = F64_ONE;
            WeightOp::Structured(
                StructuredWeightOp::bind(
                    container,
                    TENSOR,
                    PROFILE,
                    &OperandBits::F64(u),
                    ScalarBits::F64(F64_ONE),
                    columns.to_vec(),
                )
                .unwrap(),
            )
        }
        Kind::LowRank => {
            let a = OperandBits::F64(vec![F64_EIGHTH; rows]);
            let b = OperandBits::F64(vec![F64_ONE; columns.len()]);
            WeightOp::LowRankColumns(
                LowRankColumnsOp::bind(container, TENSOR, PROFILE, 1, &a, &b, columns.to_vec())
                    .unwrap(),
            )
        }
    }
}

/// Stage `op` directly in the session.
pub fn stage(overlay: &mut KnowledgeOverlay, op: WeightOp) {
    match op {
        WeightOp::Delta(op) => overlay.stage_weight_op(op).unwrap(),
        WeightOp::Structured(op) => overlay.stage_structured_op(op).unwrap(),
        WeightOp::LowRankColumns(op) => overlay.stage_low_rank_columns_op(op).unwrap(),
    }
}

/// A `.vlp` patch named `name` carrying `ops`.
pub fn patch(name: &str, ops: Vec<WeightOp>) -> VindexPatch {
    VindexPatch {
        version: 2,
        base_model: MODEL.into(),
        base_checksum: None,
        created_at: String::new(),
        description: Some(name.into()),
        author: None,
        tags: vec![],
        operations: ops
            .into_iter()
            .map(|op| match op {
                WeightOp::Delta(op) => PatchOp::WeightDelta(op),
                WeightOp::Structured(op) => PatchOp::StructuredWeight(op),
                WeightOp::LowRankColumns(op) => PatchOp::LowRankColumnsWeight(op),
            })
            .collect(),
    }
}

/// A container's component as a session binds it: the plan and the operand store.
pub struct Bound {
    pub container: PathBuf,
    pub plan: ComponentOpPlan,
    pub store: OperandStore,
}

/// Bind `container` the way an LQL session does: the canonical program, under `stored`.
pub fn bind(container: &Path) -> Bound {
    bind_as(container, None, RepresentationSource::Stored)
}

pub fn bind_as(container: &Path, want: Option<&str>, source: RepresentationSource) -> Bound {
    let inspection = inspect_container(container, false).unwrap();
    let plan = plan_component_ops(&inspection, container, COMPONENT)
        .unwrap()
        .plan
        .expect("the fixture plans");
    let store = OperandStore::open_for(container, &inspection, want, source).unwrap();
    Bound {
        container: container.to_path_buf(),
        plan,
        store,
    }
}

impl Bound {
    /// What the session's overlay is refreshed against.
    pub fn binding(&self) -> ExecutionBinding<'_> {
        self.binding_through(LoweringIdentity::cpu_production())
    }

    pub fn binding_through(&self, lowering: LoweringIdentity) -> ExecutionBinding<'_> {
        ExecutionBinding {
            container: &self.container,
            plan: &self.plan,
            store: &self.store,
            lowering,
        }
    }

    /// The bits of the last position's logits and final hidden state for [`TOKENS`], over `overrides`.
    pub fn compute(&self, overrides: Option<&OperandOverrides>) -> Computed {
        let backend = ProductionBackend::new();
        let trace = match overrides {
            Some(overrides) => execute_plan(
                &self.plan,
                OperandSource::overlaid(&self.store, overrides),
                &TOKENS,
                &backend,
            ),
            None => execute_plan(&self.plan, &self.store, &TOKENS, &backend),
        }
        .unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        Computed {
            hidden: bits(trace.final_hidden()),
            logits: bits(&trace.logits.expect("the fixture carries a head")),
        }
    }

    /// What a session holding `overlay` computes: over its live replacements, or the stored weights when it
    /// holds no stored-weight op.
    pub fn compute_session(&self, overlay: &KnowledgeOverlay) -> Computed {
        let overrides = overlay.live_weight_overrides().unwrap();
        let computed = self.compute(overrides.as_ref());
        if let Some(overrides) = &overrides {
            overrides.ensure_patched_read().unwrap();
        }
        computed
    }

    /// The plan's operand for the fixture's tensor.
    pub fn tensor_operand(&self) -> OperandRef {
        self.plan
            .planned_operands()
            .into_iter()
            .map(|p| p.operand)
            .find(|o| o.object == DECODER && o.tensor == TENSOR)
            .expect("the plan projects through the fixture's tensor")
    }
}

/// Exact output: every logit and every final-hidden element, as bits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Computed {
    pub logits: Vec<u32>,
    pub hidden: Vec<u32>,
}

/// Compile `ops` against `src` exactly as `COMPILE CURRENT INTO VINDEX` does: declare, then apply.
pub fn compile(src: &Source, ops: &[WeightOpRef<'_>], name: &str) -> PathBuf {
    let (patches, origins) = declare_patches(&src.container, ops).unwrap();
    let out = src.out(name);
    apply_declared_patches(&src.container, &patches, &origins, &out).unwrap();
    out
}

/// An overlay holding `ops`, staged directly, with its live view built against `bound`.
pub fn session_with(bound: &Bound, ops: Vec<WeightOp>) -> KnowledgeOverlay {
    let mut overlay = KnowledgeOverlay::new();
    for op in ops {
        stage(&mut overlay, op);
    }
    overlay.refresh_live_weights(&bound.binding()).unwrap();
    overlay
}
