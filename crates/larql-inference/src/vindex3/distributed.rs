//! Stateless layer-prefix execution and a coordinator over the same V3 plan.
//! The initial RPC supports single-stream softmax stacks on CPU. Each step
//! recomputes the whole prefix, so remote failure cannot desynchronise KV state.
use super::{
    input::{validate_inputs, InputPosition},
    LogitsSession,
};
use crate::error::InferenceError;
use larql_router_protocol::vindex3::{Binding, Response, SCHEMA};
use larql_vindex::format::vindex3::{
    inspect::inspect_container,
    opplan::{
        exec::{
            self,
            backend::PlanBackend,
            lowering::LoweringIdentity,
            prepared::{ExecutionSlice, PreparedOperands},
            Plane, PlaneEvent, ResumePoint,
        },
        ComponentOpPlan,
    },
};
use sha2::{Digest, Sha256};
use std::path::Path;

/// Identity includes declared payload hashes, graph and exact operation plan.
/// It names declared bytes; ordinary container verification checks those bytes.
pub fn artifact_identity(path: &Path, plan: &ComponentOpPlan) -> Result<String, InferenceError> {
    let inspection = inspect_container(path, false)?;
    let bytes = serde_json::to_vec(&(inspection.index, inspection.graph, plan))
        .map_err(|e| InferenceError::Parse(e.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
pub fn ensure_supported(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
) -> Result<(), InferenceError> {
    if ops.carries_hyper_connection()
        || ops.carries_attention_residual()
        || plan.layers.iter().any(|l| l.attention.softmax().is_none())
    {
        return Err(InferenceError::Parse(
            "V3 layer RPC supports single-stream softmax stacks only".into(),
        ));
    }
    Ok(())
}
pub fn binding(
    path: &Path,
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
) -> Result<Binding, InferenceError> {
    ensure_supported(plan, ops)?;
    let ExecutionSlice::LayerRange { start, end } = ops.slice() else {
        return Err(InferenceError::Parse(
            "layer RPC requires a prepared layer-range shard".into(),
        ));
    };
    Ok(Binding {
        schema: SCHEMA,
        artifact: artifact_identity(path, plan)?,
        backend: "cpu".into(),
        lowering: LoweringIdentity::cpu_production().to_string(),
        start: *start,
        end: *end,
        layers: plan.layers.len(),
        hidden: ops.hidden(),
    })
}

/// Execute every position from zero; intermediate planes never become output
/// unless the entire selected layer range succeeds.
pub fn forward<B: PlanBackend>(
    plan: &ComponentOpPlan,
    ops: &PreparedOperands,
    backend: &B,
    binding: &Binding,
    rows: Vec<Vec<f32>>,
) -> Result<Response, InferenceError> {
    ensure_supported(plan, ops)?;
    ensure_cpu(backend)?;
    binding
        .validate_rows(&rows)
        .map_err(InferenceError::Parse)?;
    if ops.slice()
        != &(ExecutionSlice::LayerRange {
            start: binding.start,
            end: binding.end,
        })
        || binding.lowering != backend.identity().to_string()
        || ops.hidden() != binding.hidden
        || plan.layers.len() != binding.layers
    {
        return Err(InferenceError::Parse(
            "shard binding disagrees with its prepared image".into(),
        ));
    }
    let positions = rows.len();
    let mut output = None;
    exec::execute_prepared_streaming(
        plan,
        ops,
        &vec![0; positions],
        backend,
        Some(ResumePoint {
            next_layer: binding.start,
            hidden: Plane::Rows(rows),
        }),
        &mut |event| {
            if let PlaneEvent::Layer { index, trace } = event {
                if index + 1 == binding.end {
                    if let Plane::Rows(rows) = trace.post_layer {
                        output = Some(rows);
                    }
                }
            }
            Ok(())
        },
    )?;
    let rows =
        output.ok_or_else(|| InferenceError::Parse("shard produced no final plane".into()))?;
    binding
        .validate_rows(&rows)
        .map_err(InferenceError::Parse)?;
    Ok(Response {
        binding: binding.clone(),
        rows,
    })
}

/// A transport must return the server's binding, not manufacture one locally.
pub trait ShardTransport {
    fn bindings(&self) -> Vec<Binding>;
    fn forward(&self, shard: usize, rows: Vec<Vec<f32>>) -> Result<Response, String>;
}

pub struct DistributedSession<'a, B: PlanBackend, T: ShardTransport> {
    plan: &'a ComponentOpPlan,
    ops: &'a PreparedOperands,
    backend: &'a B,
    transport: T,
    bindings: Vec<Binding>,
    inputs: Vec<InputPosition>,
}
impl<'a, B: PlanBackend, T: ShardTransport> DistributedSession<'a, B, T> {
    pub fn new(
        plan: &'a ComponentOpPlan,
        ops: &'a PreparedOperands,
        backend: &'a B,
        artifact: &str,
        transport: T,
    ) -> Result<Self, InferenceError> {
        ensure_supported(plan, ops)?;
        ensure_cpu(backend)?;
        if ops.slice() != &ExecutionSlice::Endpoints {
            return Err(InferenceError::Parse(
                "distributed coordinator must prepare endpoints only".into(),
            ));
        }
        let bindings = transport.bindings();
        let mut next = 0;
        for b in &bindings {
            b.validate().map_err(InferenceError::Parse)?;
            if b.lowering != backend.identity().to_string()
                || b.artifact != artifact
                || b.hidden != ops.hidden()
                || b.layers != plan.layers.len()
                || b.start != next
            {
                return Err(InferenceError::Parse("shards must name the same artifact and cover every layer exactly once, in order".into()));
            }
            next = b.end;
        }
        if bindings.is_empty() || next != plan.layers.len() {
            return Err(InferenceError::Parse("shard coverage is incomplete".into()));
        }
        Ok(Self {
            plan,
            ops,
            backend,
            transport,
            bindings,
            inputs: Vec::new(),
        })
    }
    pub fn extend_inputs(&mut self, inputs: &[InputPosition]) -> Result<Vec<f32>, InferenceError> {
        validate_inputs(self.plan, self.ops, self.backend, inputs)?;
        let mut rows = self
            .inputs
            .iter()
            .chain(inputs)
            .map(|i| match i {
                InputPosition::Token(id) => self
                    .ops
                    .embed_token(self.plan, self.backend, *id)
                    .map_err(InferenceError::from),
                InputPosition::Embedding(row) => Ok(row.clone()),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let count = rows.len();
        for (index, binding) in self.bindings.iter().enumerate() {
            binding
                .validate_rows(&rows)
                .map_err(InferenceError::Parse)?;
            let response = self
                .transport
                .forward(index, rows)
                .map_err(InferenceError::Parse)?;
            if response.binding != *binding || response.rows.len() != count {
                return Err(InferenceError::Parse(
                    "shard response changed binding or position count".into(),
                ));
            }
            binding
                .validate_rows(&response.rows)
                .map_err(InferenceError::Parse)?;
            rows = response.rows;
        }
        let logits = self
            .ops
            .head_logits(self.backend, rows.last().expect("validated nonempty"))?
            .ok_or_else(super::session::missing_logits_error)?;
        self.inputs.extend_from_slice(inputs);
        Ok(logits)
    }
}
impl<B: PlanBackend, T: ShardTransport> LogitsSession for DistributedSession<'_, B, T> {
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
        self.inputs.len()
    }
}

fn ensure_cpu<B: PlanBackend>(backend: &B) -> Result<(), InferenceError> {
    if backend.identity() != LoweringIdentity::cpu_production() {
        return Err(InferenceError::Parse(
            "V3 layer RPC requires the CPU production lowering".into(),
        ));
    }
    Ok(())
}
