//! What a capture's sidecar says about the model state that computed it: whether the stored weights executed or
//! patched ones, every stored-weight op in effect with its digest and the `.vlp` file it came from, the tensors
//! the live view replaced, and how the container was opened.

use larql_vindex::format::vindex3::knowledge::KnowledgeOverlay;
use larql_vindex::format::vindex3::opplan::exec::backend::PlanBackend;
use larql_vindex::format::vindex3::opplan::exec::residency_record::source_name;
use larql_vindex::format::vindex3::weight_patch::WeightOpRef;
use serde_json::{json, Value};

use super::super::V3Runtime;

/// The stored weights executed: no op or edit changed what the program computes with.
pub(super) const EXECUTES_STORED: &str = "stored";
/// Stored-weight ops were in effect: the program executed their patched bytes.
pub(super) const EXECUTES_PATCHED: &str = "patched";
/// Knowledge (compose) edits were in effect: the program executed with their operand overrides.
pub(super) const EXECUTES_COMPOSE_EDITED: &str = "compose-edited";
/// Where a directly staged op comes from, as the overlay names it.
const STAGED_IN_SESSION: &str = "this session";

/// A `.vlp` file as `APPLY PATCH` read it: kept beside the applied patch, in the same order, so a capture can
/// name the exact bytes that were applied.
#[derive(Debug, Clone)]
pub(crate) struct AppliedPatchFile {
    pub path: String,
    pub sha256: String,
}

/// What the program executed, given that a run resolved `overridden` overrides.
pub(super) fn executes(overlay: &KnowledgeOverlay, overridden: bool) -> &'static str {
    match (overridden, overlay.has_weight_ops()) {
        (false, _) => EXECUTES_STORED,
        (true, true) => EXECUTES_PATCHED,
        (true, false) => EXECUTES_COMPOSE_EDITED,
    }
}

fn op_kind(op: &WeightOpRef<'_>) -> &'static str {
    match op {
        WeightOpRef::Delta(_) => "weight_delta",
        WeightOpRef::Structured(_) => "remove_direction",
        WeightOpRef::LowRankColumns(_) => "add_low_rank_columns",
    }
}

fn op_record(op: &WeightOpRef<'_>) -> Value {
    json!({
        "kind": op_kind(op),
        "tensor": op.located_tensor(),
        "digest": op.digest(),
    })
}

/// Every applied patch (with its file's digest) and every op staged directly in the session.
pub(super) fn patches(overlay: &KnowledgeOverlay, files: &[AppliedPatchFile]) -> Value {
    let mut out: Vec<Value> = overlay
        .patches
        .iter()
        .enumerate()
        .map(|(i, patch)| {
            let ops: Vec<Value> = patch
                .operations
                .iter()
                .filter_map(larql_vindex::PatchOp::weight_op)
                .map(|op| op_record(&op))
                .collect();
            let file = files.get(i);
            json!({
                "source": "applied patch",
                "description": patch.description,
                "path": file.map(|f| f.path.clone()),
                "sha256": file.map(|f| f.sha256.clone()),
                "operations": patch.operations.len(),
                "weight_ops": ops,
            })
        })
        .collect();
    let staged: Vec<Value> = overlay
        .weight_ops()
        .iter()
        .filter(|(from, _)| from == STAGED_IN_SESSION)
        .map(|(_, op)| op_record(op))
        .collect();
    if !staged.is_empty() {
        out.push(json!({ "source": STAGED_IN_SESSION, "weight_ops": staged }));
    }
    Value::Array(out)
}

/// The tensors the live view executes from patched bytes, with each one's digest (equal to the same tensor's
/// digest in a container COMPILE writes from the same ops).
pub(super) fn live_tensors(overlay: &KnowledgeOverlay) -> Value {
    let tensors: Vec<Value> = match overlay.live_weights() {
        Ok(Some(view)) => view
            .tensors()
            .iter()
            .map(|t| json!({ "tensor": t.tensor, "object": t.object, "shape": t.shape, "sha256": t.sha256 }))
            .collect(),
        _ => Vec::new(),
    };
    Value::Array(tensors)
}

/// How the container was opened: the representation source, the lowering, and anything quantised at load.
pub(super) fn binding(runtime: &V3Runtime) -> Value {
    json!({
        "representation_source": source_name(runtime.store().representation_source()),
        "lowering": runtime.backend().identity().to_string(),
        "runtime_quantised_tensors": runtime.store().runtime_quantised_tensors(),
    })
}
