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
            identity::{ExecutionIdentity, ModelAuthority, ProcessArithmetic},
            lowering::LoweringIdentity,
            operands::OperandSource,
            portability::{carrier_form_of, ensure_portable},
            prepared::{select_realizations, ExecutionSlice, PreparedOperands},
            realization::RealizationRecord,
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
    _ops: &PreparedOperands,
) -> Result<(), InferenceError> {
    // Rows cross this boundary, so the carrier must be portable: the one
    // authority for that is `portability` (RESIDUAL-BUS-2 A3).
    ensure_portable(carrier_form_of(plan.residual_topology))?;
    // A separate limit, not a portability rule: the stateless worker runs
    // with no continuation provider, so it cannot execute a layer that
    // keeps state beyond softmax attention.
    if plan.layers.iter().any(|l| l.attention.softmax().is_none()) {
        return Err(InferenceError::Parse(
            "V3 layer RPC runs stateless workers, so every layer must be softmax attention".into(),
        ));
    }
    Ok(())
}
/// The one lowering an image's pins name, read from the pins and never
/// assumed (RESIDUAL-BUS-2 I3: the binding used to hardcode it).
pub fn pinned_lowering(records: &[RealizationRecord]) -> Result<String, InferenceError> {
    let mut lowering: Vec<String> = records
        .iter()
        .map(|r| r.lowering_provider.to_string())
        .collect();
    lowering.sort();
    lowering.dedup();
    match lowering.as_slice() {
        [one] => Ok(one.clone()),
        [] => Err(InferenceError::Parse(
            "a shard binding needs pinned operands to name its lowering".into(),
        )),
        many => Err(InferenceError::Parse(format!(
            "a shard binding names one lowering, and these pins name {}",
            many.join(", ")
        ))),
    }
}

/// The execution-identity digest a binding carries for `records` over
/// `slice` in THIS process (RESIDUAL-BUS-2 I1, I3). A worker states its
/// own; a coordinator derives the one it expects of a remote slice from
/// pins it selects itself, so a worker whose model, lowering, realizations
/// or process arithmetic differ presents a different digest. Refuses an
/// unanchored identity: exact routing never runs on one (D8).
pub fn identity_digest(
    artifact: &str,
    records: &[RealizationRecord],
    slice: &ExecutionSlice,
    overlaid: bool,
) -> Result<String, InferenceError> {
    let identity = ExecutionIdentity::from_records(
        records,
        slice,
        Some(ModelAuthority::new(artifact)?),
        overlaid,
        ProcessArithmetic::current()?,
    );
    identity.ensure_anchored()?;
    Ok(identity.digest())
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
    let artifact = artifact_identity(path, plan)?;
    Ok(Binding {
        schema: SCHEMA,
        execution_identity: identity_digest(
            &artifact,
            ops.realizations(),
            ops.slice(),
            ops.source_stamp().is_overlaid(),
        )?,
        artifact,
        backend: "cpu".into(),
        lowering: pinned_lowering(ops.realizations())?,
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

/// The layer-prefix transport seam lives in the protocol crate so a
/// transport can be written without this crate; re-exported here for the
/// original path.
pub use larql_router_protocol::vindex3_transport::ShardTransport;

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
        source: OperandSource<'_>,
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
            // What a shard computing exactly this slice must present:
            // its pins, selected here, under this process's arithmetic.
            let slice = ExecutionSlice::LayerRange {
                start: b.start,
                end: b.end,
            };
            let pins = select_realizations(plan, source, backend, &slice)?;
            let expected = identity_digest(artifact, &pins, &slice, source.stamp().is_overlaid())?;
            if b.execution_identity != expected {
                return Err(InferenceError::Parse(format!(
                    "shard {}..{} presents execution identity {}, and this process expects {} \
                     for that slice: the shard computes something else (a different model, \
                     lowering, realization or process arithmetic)",
                    b.start, b.end, b.execution_identity, expected
                )));
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
            .enumerate()
            .map(|(position, i)| match i {
                InputPosition::Token(id) => self
                    .ops
                    .embed_token(self.plan, self.backend, *id, position)
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
