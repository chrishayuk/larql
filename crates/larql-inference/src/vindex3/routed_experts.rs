//! Placement of selected expert transforms under V3 execution authority.
pub use super::dense_ffn::DenseFfnSession as RoutedExpertSession;
use super::{dense_ffn::identities, distributed::artifact_identity, PreparedVindex3};
use crate::error::InferenceError;
use larql_router_protocol::{vindex3, vindex3_experts::Binding};
use larql_vindex::{
    error::VindexError,
    format::vindex3::opplan::{
        exec::{
            backend::PlanBackend,
            lowering::LoweringIdentity,
            operands::OperandSource,
            prepared::{select_realizations, ExecutionSlice, PreparedOperands},
            routed_experts::{PreparedRoutedExperts, RoutedExpertProvider},
        },
        ComponentOpPlan,
    },
};
use std::{path::Path, sync::Arc};

pub trait ExpertTransport: Send + Sync {
    fn bindings(&self) -> Vec<Binding>;
    fn forward(
        &self,
        shard: usize,
        layer: usize,
        experts: &[usize],
        row: &[f32],
    ) -> Result<Vec<ExpertOutput>, String>;
}
fn cpu<B: PlanBackend + ?Sized>(backend: &B) -> Result<(), InferenceError> {
    if backend.identity() != LoweringIdentity::cpu_production() {
        return Err(InferenceError::Parse(
            "routed placement requires CPU production lowering".into(),
        ));
    }
    Ok(())
}
fn expected(
    path: &Path,
    plan: &ComponentOpPlan,
    source: OperandSource<'_>,
    backend: &impl PlanBackend,
    slice: &ExecutionSlice,
) -> Result<Binding, InferenceError> {
    cpu(backend)?;
    let regions = PreparedRoutedExperts::regions_for(plan, source, slice)?;
    let ExecutionSlice::RoutedExperts {
        start,
        end,
        expert_start,
        expert_end,
    } = slice
    else {
        return Err(InferenceError::Parse("routed expert slice required".into()));
    };
    let pins = select_realizations(plan, source, backend, slice)?;
    Ok(Binding {
        program: vindex3::Binding {
            schema: vindex3::SCHEMA,
            artifact: artifact_identity(path, plan)?,
            backend: "cpu".into(),
            lowering: backend.identity().to_string(),
            start: *start,
            end: *end,
            layers: plan.layers.len(),
            hidden: plan
                .embedding
                .as_ref()
                .ok_or_else(|| InferenceError::Parse("missing embedding".into()))?
                .table
                .shape[1],
        },
        expert_start: *expert_start,
        expert_end: *expert_end,
        operands: identities(&pins)?,
        regions: serde_json::to_string(&regions)
            .map_err(|e| InferenceError::Parse(e.to_string()))?,
    })
}
pub struct BoundExpertWorker<B: PlanBackend> {
    runtime: Arc<PreparedVindex3<B>>,
    binding: Binding,
}
impl<B: PlanBackend> BoundExpertWorker<B> {
    pub fn new(path: &Path, runtime: Arc<PreparedVindex3<B>>) -> Result<Self, InferenceError> {
        cpu(runtime.backend())?;
        let ops = runtime.operands();
        ops.ensure_providers_in(ops.registry())?;
        ops.ensure_lowered_by(runtime.backend())?;
        let ExecutionSlice::RoutedExperts {
            start,
            end,
            expert_start,
            expert_end,
        } = ops.slice()
        else {
            return Err(InferenceError::Parse("expert-only worker required".into()));
        };
        let worker = ops
            .routed_experts()
            .ok_or_else(|| InferenceError::Parse("missing expert worker image".into()))?;
        let binding = Binding {
            program: vindex3::Binding {
                schema: vindex3::SCHEMA,
                artifact: artifact_identity(path, runtime.plan())?,
                backend: "cpu".into(),
                lowering: runtime.backend().identity().to_string(),
                start: *start,
                end: *end,
                layers: runtime.plan().layers.len(),
                hidden: ops.hidden(),
            },
            expert_start: *expert_start,
            expert_end: *expert_end,
            operands: identities(ops.realizations())?,
            regions: serde_json::to_string(worker.regions())
                .map_err(|e| InferenceError::Parse(e.to_string()))?,
        };
        binding.validate().map_err(InferenceError::Parse)?;
        Ok(Self { runtime, binding })
    }
    pub fn binding(&self) -> &Binding {
        &self.binding
    }
    pub fn max_selected(&self) -> usize {
        self.runtime
            .plan()
            .layers
            .iter()
            .filter_map(|l| l.ffn.as_ref().and_then(|f| f.routed()).map(|r| r.top_k))
            .max()
            .unwrap_or(0)
    }
    pub fn apply(
        &self,
        layer: usize,
        experts: &[usize],
        row: &[f32],
    ) -> Result<Vec<ExpertOutput>, InferenceError> {
        Ok(self
            .runtime
            .operands()
            .routed_experts()
            .expect("bound worker")
            .apply(self.runtime.backend(), layer, experts, row)?)
    }
}
struct Grid<T> {
    transport: T,
    owners: Vec<Vec<usize>>,
    bindings: Vec<Binding>,
}
impl<T: ExpertTransport> RoutedExpertProvider for Grid<T> {
    fn apply(
        &self,
        layer: usize,
        input: &[f32],
        experts: &[usize],
    ) -> Result<Vec<ExpertOutput>, VindexError> {
        let owners = self
            .owners
            .get(layer)
            .ok_or_else(|| VindexError::Parse("unowned expert layer".into()))?;
        let mut groups = std::collections::BTreeMap::<usize, Vec<usize>>::new();
        for expert in experts {
            let owner = *owners
                .get(*expert)
                .ok_or_else(|| VindexError::Parse("unowned selected expert".into()))?;
            groups.entry(owner).or_default().push(*expert);
        }
        std::thread::scope(|scope| {
            let pending: Vec<_> = groups
                .into_iter()
                .map(|(shard, ids)| {
                    scope.spawn(move || {
                        let rows = self
                            .transport
                            .forward(shard, layer, &ids, input)
                            .map_err(VindexError::Parse)?;
                        let binding = &self.bindings[shard];
                        let mut seen = std::collections::BTreeSet::new();
                        if rows.len() != ids.len()
                            || rows.iter().any(|r| {
                                !ids.contains(&r.expert)
                                    || !(binding.expert_start..binding.expert_end)
                                        .contains(&r.expert)
                                    || !seen.insert(r.expert)
                                    || r.row.len() != binding.program.hidden
                                    || r.row.iter().any(|v| !v.is_finite())
                            })
                        {
                            return Err(VindexError::Parse(
                                "expert response ownership, count, width or finiteness mismatch"
                                    .into(),
                            ));
                        }
                        Ok(rows)
                    })
                })
                .collect();
            // Join every dispatch even when one fails: no request outlives
            // this step, and a second panic cannot escape scope's auto-join.
            let joined: Vec<_> = pending
                .into_iter()
                .map(|p| {
                    p.join()
                        .map_err(|_| VindexError::Parse("expert dispatch panicked".into()))
                        .and_then(|r| r)
                })
                .collect();
            let mut rows = Vec::new();
            for result in joined {
                rows.extend(result?);
            }
            Ok(rows)
        })
    }
}
pub fn prepare_coordinator<B: PlanBackend, T: ExpertTransport + 'static>(
    path: &Path,
    plan: &ComponentOpPlan,
    source: OperandSource<'_>,
    backend: &B,
    transport: T,
) -> Result<PreparedOperands, InferenceError> {
    cpu(backend)?;
    if source.stamp() != OperandSource::from(source.store()).stamp() {
        return Err(InferenceError::Parse(
            "routed placement requires base artifact operands".into(),
        ));
    }
    let bindings = transport.bindings();
    let mut owners = plan
        .layers
        .iter()
        .map(|l| {
            l.ffn
                .as_ref()
                .and_then(|f| f.routed())
                .map(|r| vec![usize::MAX; r.experts])
                .ok_or_else(|| InferenceError::Parse("routed layer required".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (shard, binding) in bindings.iter().enumerate() {
        binding.validate().map_err(InferenceError::Parse)?;
        let b = &binding.program;
        let slice = ExecutionSlice::RoutedExperts {
            start: b.start,
            end: b.end,
            expert_start: binding.expert_start,
            expert_end: binding.expert_end,
        };
        let want = expected(path, plan, source, backend, &slice)?;
        if b.artifact != want.program.artifact {
            return Err(InferenceError::Parse(
                "routed expert artifact/plan mismatch".into(),
            ));
        }
        if b.backend != want.program.backend || b.lowering != want.program.lowering {
            return Err(InferenceError::Parse(
                "routed expert numerical provider mismatch".into(),
            ));
        }
        if b.layers != want.program.layers || b.hidden != want.program.hidden {
            return Err(InferenceError::Parse(
                "routed expert dimensions mismatch".into(),
            ));
        }
        if binding.operands != want.operands || binding.regions != want.regions {
            return Err(InferenceError::Parse(
                "routed expert representation or owned byte-region mismatch".into(),
            ));
        }
        for layer in &mut owners[b.start..b.end] {
            for owner in &mut layer[binding.expert_start..binding.expert_end] {
                if *owner != usize::MAX {
                    return Err(InferenceError::Parse("overlapping expert coverage".into()));
                }
                *owner = shard;
            }
        }
    }
    if owners.is_empty()
        || owners
            .iter()
            .any(|l| l.is_empty() || l.contains(&usize::MAX))
    {
        return Err(InferenceError::Parse("incomplete expert coverage".into()));
    }
    let mut ops = PreparedOperands::load(
        plan,
        source,
        backend,
        ExecutionSlice::RoutedExpertCoordinator,
    )?;
    ops.bind_routed_expert_provider(Arc::new(Grid {
        transport,
        owners,
        bindings,
    }))?;
    Ok(ops)
}

pub use larql_vindex::format::vindex3::opplan::exec::routed_experts::ExpertOutput;
