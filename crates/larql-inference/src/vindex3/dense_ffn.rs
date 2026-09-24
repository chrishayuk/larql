//! Artifact-bound placement of dense FFN operations. The canonical interpreter
//! owns attention, local continuation, norms and residual ordering.
use super::distributed::artifact_identity;
use crate::error::InferenceError;
use larql_router_protocol::{
    vindex3,
    vindex3_ffn::{Binding, OperandIdentity, Response},
};
pub use larql_vindex::format::vindex3::opplan::exec::profile;
use larql_vindex::{
    error::VindexError,
    format::vindex3::opplan::{
        exec::{
            backend::PlanBackend,
            dense_ffn::DenseFfnProvider,
            lowering::LoweringIdentity,
            operands::OperandSource,
            prepared::{select_realizations, ExecutionSlice, PreparedOperands},
            realization::RealizationRecord,
        },
        ComponentOpPlan,
    },
};
use std::path::Path;
use std::sync::Arc;

fn identities(records: &[RealizationRecord]) -> Result<Vec<OperandIdentity>, InferenceError> {
    records
        .iter()
        .map(|r| {
            Ok(OperandIdentity {
                layer: r
                    .planned
                    .layer
                    .ok_or_else(|| InferenceError::Parse("FFN operand has no layer".into()))?,
                operand: serde_json::to_string(&r.planned.operand)
                    .map_err(|e| InferenceError::Parse(e.to_string()))?,
                representation: r.representation.clone(),
                codec: format!("{:?}", r.codec_provider),
                realization: format!("{:?}", r.selection.realization),
                extent: format!("{:?}", r.extent),
                dependencies: format!("{:?}", r.dependencies),
            })
        })
        .collect()
}
fn make_binding(
    artifact: String,
    plan: &ComponentOpPlan,
    start: usize,
    end: usize,
    hidden: usize,
    records: &[RealizationRecord],
) -> Result<Binding, InferenceError> {
    Ok(Binding {
        program: vindex3::Binding {
            schema: vindex3::SCHEMA,
            artifact,
            backend: "cpu".into(),
            lowering: LoweringIdentity::cpu_production().to_string(),
            start,
            end,
            layers: plan.layers.len(),
            hidden,
        },
        operands: identities(records)?,
    })
}
fn ensure_cpu<B: PlanBackend + ?Sized>(backend: &B) -> Result<(), InferenceError> {
    if backend.identity() != LoweringIdentity::cpu_production() {
        return Err(InferenceError::Parse(
            "dense FFN RPC requires CPU production lowering".into(),
        ));
    }
    Ok(())
}
pub fn binding(
    path: &Path,
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
) -> Result<Binding, InferenceError> {
    let ExecutionSlice::DenseFfns { start, end } = ops.slice() else {
        return Err(InferenceError::Parse(
            "requires dense FFN worker operands".into(),
        ));
    };
    make_binding(
        artifact_identity(path, plan)?,
        plan,
        *start,
        *end,
        ops.hidden(),
        ops.realizations(),
    )
}
pub fn forward<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    binding: &Binding,
    layer: usize,
    row: &[f32],
) -> Result<Response, InferenceError> {
    forward_timed(plan, ops, backend, binding, layer, row, None)
}

/// Same execution authority with an optional timer around the worker transform.
/// Binding validation is outside `ffn_ns`; operand application checks are inside.
pub fn forward_timed<B: PlanBackend + ?Sized>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    binding: &Binding,
    layer: usize,
    row: &[f32],
    ffn_ns: Option<&mut u64>,
) -> Result<Response, InferenceError> {
    ensure_cpu(backend)?;
    ops.ensure_providers_in(ops.registry())?;
    ops.ensure_lowered_by(backend)?;
    binding
        .validate_row(layer, row)
        .map_err(InferenceError::Parse)?;
    if ops.slice()
        != &(ExecutionSlice::DenseFfns {
            start: binding.program.start,
            end: binding.program.end,
        })
        || binding.program.hidden != ops.hidden()
        || binding.program.layers != plan.layers.len()
        || binding.program.lowering != backend.identity().to_string()
        || binding.operands != identities(ops.realizations())?
    {
        return Err(InferenceError::Parse(
            "dense FFN binding disagrees with prepared operands".into(),
        ));
    }
    let started = ffn_ns.as_ref().map(|_| std::time::Instant::now());
    let row = ops
        .dense_ffns()
        .ok_or_else(|| InferenceError::Parse("missing worker image".into()))?
        .apply(plan, backend, layer, row)?;
    if let (Some(ns), Some(started)) = (ffn_ns, started) {
        *ns = started.elapsed().as_nanos() as u64;
    }
    Ok(Response {
        binding: binding.clone(),
        layer,
        row,
    })
}

pub trait FfnTransport: Send + Sync {
    fn bindings(&self) -> Vec<Binding>;
    fn forward(&self, shard: usize, layer: usize, normalized: &[f32]) -> Result<Response, String>;
}
struct BoundProvider<T> {
    transport: T,
    bindings: Vec<Binding>,
    owners: Vec<usize>,
}
impl<T: FfnTransport> DenseFfnProvider for BoundProvider<T> {
    fn apply(&self, layer: usize, row: &[f32]) -> Result<Vec<f32>, VindexError> {
        let shard = *self
            .owners
            .get(layer)
            .ok_or_else(|| VindexError::Parse("unknown FFN layer".into()))?;
        let expected = &self.bindings[shard];
        expected
            .validate_row(layer, row)
            .map_err(VindexError::Parse)?;
        let response = self
            .transport
            .forward(shard, layer, row)
            .map_err(VindexError::Parse)?;
        if response.binding != *expected || response.layer != layer {
            return Err(VindexError::Parse(
                "dense FFN response changed binding or layer".into(),
            ));
        }
        expected
            .validate_row(layer, &response.row)
            .map_err(VindexError::Parse)?;
        Ok(response.row)
    }
}
/// Check complete ownership and effective realizations before loading local
/// operands or accepting a token. Selecting the remote pins does not load the
/// FFN matrices into the coordinator.
pub fn prepare_coordinator<B: PlanBackend + ?Sized, T: FfnTransport + 'static>(
    path: &Path,
    plan: &ComponentOpPlan,
    source: OperandSource<'_>,
    backend: &B,
    transport: T,
) -> Result<PreparedOperands, InferenceError> {
    ensure_cpu(backend)?;
    if source.stamp() != OperandSource::from(source.store()).stamp() {
        return Err(InferenceError::Parse(
            "dense FFN placement requires base artifact operands".into(),
        ));
    }
    let artifact = artifact_identity(path, plan)?;
    let bindings = transport.bindings();
    let mut owners = vec![usize::MAX; plan.layers.len()];
    let hidden = plan
        .embedding
        .as_ref()
        .ok_or_else(|| InferenceError::Parse("missing embedding".into()))?
        .table
        .shape[1];
    for (index, binding) in bindings.iter().enumerate() {
        binding.validate().map_err(InferenceError::Parse)?;
        let b = &binding.program;
        if b.artifact != artifact {
            return Err(InferenceError::Parse(
                "dense FFN artifact/plan mismatch".into(),
            ));
        }
        if b.lowering != backend.identity().to_string() || b.backend != "cpu" {
            return Err(InferenceError::Parse(
                "dense FFN numerical provider mismatch".into(),
            ));
        }
        if b.layers != plan.layers.len() || b.hidden != hidden {
            return Err(InferenceError::Parse(
                "dense FFN dimensions mismatch".into(),
            ));
        }
        let slice = ExecutionSlice::DenseFfns {
            start: b.start,
            end: b.end,
        };
        let pins = select_realizations(plan, source, backend, &slice)?;
        let expected = make_binding(artifact.clone(), plan, b.start, b.end, hidden, &pins)?;
        if expected.operands != binding.operands {
            return Err(InferenceError::Parse(
                "dense FFN representation/realization mismatch".into(),
            ));
        }
        for owner in &mut owners[b.start..b.end] {
            if *owner != usize::MAX {
                return Err(InferenceError::Parse(
                    "overlapping dense FFN coverage".into(),
                ));
            }
            *owner = index;
        }
    }
    if owners.is_empty() || owners.contains(&usize::MAX) {
        return Err(InferenceError::Parse(
            "incomplete dense FFN coverage".into(),
        ));
    }
    let mut ops =
        PreparedOperands::load(plan, source, backend, ExecutionSlice::DenseFfnCoordinator)?;
    ops.bind_dense_ffn_provider(Arc::new(BoundProvider {
        transport,
        bindings,
        owners,
    }))?;
    Ok(ops)
}

/// Owns local KV. Any failed extension invalidates this session, so partially
/// advanced attention state cannot be reused. Start a fresh session and replay
/// committed input to recover; remote workers retain no continuation state.
pub struct DenseFfnSession<'a, B: PlanBackend> {
    plan: &'a ComponentOpPlan,
    ops: &'a PreparedOperands,
    backend: &'a B,
    state: larql_vindex::format::vindex3::opplan::exec::kv::RowKvState,
    committed: usize,
    failed: bool,
}
impl<'a, B: PlanBackend> DenseFfnSession<'a, B> {
    pub fn new(
        plan: &'a ComponentOpPlan,
        ops: &'a PreparedOperands,
        backend: &'a B,
    ) -> Result<Self, InferenceError> {
        if ops.slice() != &ExecutionSlice::DenseFfnCoordinator {
            return Err(InferenceError::Parse(
                "dense FFN session requires coordinator operands".into(),
            ));
        }
        Ok(Self {
            plan,
            ops,
            backend,
            state: Default::default(),
            committed: 0,
            failed: false,
        })
    }
    pub fn extend_inputs(
        &mut self,
        inputs: &[super::input::InputPosition],
    ) -> Result<Vec<f32>, InferenceError> {
        if self.failed {
            return Err(InferenceError::Parse("dense FFN session is invalid after a failed step; create a fresh session and replay committed input".into()));
        }
        super::input::validate_inputs(self.plan, self.ops, self.backend, inputs)?;
        let result = (|| {
            let mut session =
                larql_vindex::format::vindex3::opplan::exec::decode::DecodeSession::over_prepared(
                    self.plan,
                    self.ops,
                    self.backend,
                    &mut self.state,
                )?;
            let mut logits = None;
            for input in inputs {
                logits = Some(super::input::step_input(&mut session, input)?);
            }
            logits.ok_or_else(super::session::missing_logits_error)
        })();
        match result {
            Ok(logits) => {
                self.committed += inputs.len();
                Ok(logits)
            }
            Err(error) => {
                self.failed = true;
                Err(error)
            }
        }
    }
}
impl<B: PlanBackend> super::LogitsSession for DenseFfnSession<'_, B> {
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, InferenceError> {
        self.extend_inputs(
            &tokens
                .iter()
                .copied()
                .map(super::input::InputPosition::Token)
                .collect::<Vec<_>>(),
        )
    }
    fn step(&mut self, token: u32) -> Result<Vec<f32>, InferenceError> {
        self.extend_inputs(&[super::input::InputPosition::Token(token)])
    }
    fn position(&self) -> usize {
        self.committed
    }
}
