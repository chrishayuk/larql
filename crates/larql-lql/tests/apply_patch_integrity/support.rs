//! Shared helpers: run statements, snapshot what a session shows, and turn a
//! patch the session itself saved into each invalid variant.

use std::path::Path;

use larql_lql::{parse, Session};
use larql_vindex::patch::format::{decode_gate_vector, encode_gate_vector};
use larql_vindex::{PatchOp, VindexPatch};

/// Not valid base64, so the vector cannot be decoded at all.
pub const CORRUPT_VECTOR: &str = "!!!not-base64!!!";
/// A model name no fixture uses.
pub const OTHER_MODEL: &str = "some/other-model";
/// Past the end of any fixture's layers and of any layer's features.
pub const OUT_OF_BOUNDS: usize = 1_000_000;

/// A path as it is written inside an LQL string literal.
pub fn lql_path(path: impl AsRef<Path>) -> String {
    path.as_ref().display().to_string().replace('\\', "\\\\")
}

pub fn try_run(session: &mut Session, sql: &str) -> Result<Vec<String>, String> {
    let parsed = parse(sql).map_err(|e| format!("parse {sql:?}: {e}"))?;
    session.execute(&parsed).map_err(|e| e.to_string())
}

pub fn run(session: &mut Session, sql: &str) -> Vec<String> {
    try_run(session, sql).unwrap_or_else(|e| panic!("{sql}: {e}"))
}

/// Everything the given statements print, as one comparable value.
pub fn snapshot(session: &mut Session, statements: &[&str]) -> Vec<Vec<String>> {
    statements.iter().map(|sql| run(session, sql)).collect()
}

/// Write `patch` beside the fixture and return the statement that applies it.
pub fn apply_statement(dir: &Path, name: &str, patch: &VindexPatch) -> String {
    let path = dir.join(name);
    patch.save(&path).expect("write the patch");
    format!(r#"APPLY PATCH "{}";"#, lql_path(path))
}

/// The base64 vector fields of an op, in a fixed order.
fn vectors(op: &mut PatchOp) -> Vec<&mut Option<String>> {
    match op {
        PatchOp::Insert {
            gate_vector_b64,
            up_vector_b64,
            down_vector_b64,
            ..
        }
        | PatchOp::Update {
            gate_vector_b64,
            up_vector_b64,
            down_vector_b64,
            ..
        } => vec![gate_vector_b64, up_vector_b64, down_vector_b64],
        _ => vec![],
    }
}

/// Replace the first vector the patch carries. `hidden_size` builds one when
/// the patch's slot ops carry none, so the variant never depends on which
/// vectors a fixture's INSERT happens to record.
fn set_first_vector(patch: &mut VindexPatch, hidden_size: usize, f: impl Fn(Vec<f32>) -> String) {
    for op in &mut patch.operations {
        if let PatchOp::InsertKnn { key_vector_b64, .. } = op {
            let key = decode_gate_vector(key_vector_b64).expect("the saved key decodes");
            *key_vector_b64 = f(key);
            return;
        }
        if let Some(slot) = vectors(op).into_iter().next() {
            let current = match slot.as_deref() {
                Some(b64) => decode_gate_vector(b64).expect("the saved vector decodes"),
                None => vec![0.0; hidden_size],
            };
            *slot = Some(f(current));
            return;
        }
    }
    panic!("the patch has no operation that can carry a vector");
}

/// The patch with one vector that does not decode.
pub fn with_corrupt_vector(mut patch: VindexPatch, hidden_size: usize) -> VindexPatch {
    set_first_vector(&mut patch, hidden_size, |_| CORRUPT_VECTOR.to_string());
    patch
}

/// The patch with one vector a single element too wide.
pub fn with_wide_vector(mut patch: VindexPatch, hidden_size: usize) -> VindexPatch {
    set_first_vector(&mut patch, hidden_size, |mut v| {
        v.push(0.0);
        encode_gate_vector(&v)
    });
    patch
}

/// The patch with its first op moved to a layer the index does not have.
pub fn with_layer_out_of_bounds(mut patch: VindexPatch) -> VindexPatch {
    match patch.operations.first_mut().expect("the patch has an op") {
        PatchOp::Insert { layer, .. }
        | PatchOp::Update { layer, .. }
        | PatchOp::Delete { layer, .. }
        | PatchOp::InsertKnn { layer, .. } => *layer = OUT_OF_BOUNDS,
        other => panic!("the first op names no layer: {other:?}"),
    }
    patch
}

/// The patch with its first op replaced by one that names a feature the
/// layer does not have. Replaced, not edited: a fixture's INSERT may record a
/// KNN op, which names no feature slot.
pub fn with_feature_out_of_bounds(mut patch: VindexPatch) -> VindexPatch {
    *patch.operations.first_mut().expect("the patch has an op") = PatchOp::Delete {
        layer: 0,
        feature: OUT_OF_BOUNDS,
        reason: None,
    };
    patch
}

pub fn with_base_model(mut patch: VindexPatch, model: &str) -> VindexPatch {
    patch.base_model = model.to_string();
    patch
}

/// A valid op followed by an invalid one: the valid op must not be applied.
pub fn with_trailing_invalid_op(mut patch: VindexPatch) -> VindexPatch {
    patch.operations.push(PatchOp::Delete {
        layer: OUT_OF_BOUNDS,
        feature: 0,
        reason: None,
    });
    patch
}
