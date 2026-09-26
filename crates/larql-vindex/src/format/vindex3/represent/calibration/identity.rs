use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

use super::{json_digest, refused, SCHEMA};
use crate::error::VindexError;
use crate::format::vindex3::opplan::exec::operands::OperandSource;
use crate::format::vindex3::opplan::{ComponentOpPlan, LayerFfn, OperandRef};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Projection {
    Query,
    Key,
    Value,
    Gate,
    Up,
    Down,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Boundary {
    AttentionInput,
    FfnInput,
    FfnDownInput,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationSite {
    pub layer: usize,
    pub projection: Projection,
    pub boundary: Boundary,
    pub object: String,
    pub tensor: String,
    pub rows: usize,
    pub width: usize,
}

impl CalibrationSite {
    pub(super) fn resolve(
        plan: &ComponentOpPlan,
        layer: usize,
        projection: Projection,
    ) -> Result<Self, VindexError> {
        let op = plan
            .layers
            .get(layer)
            .ok_or_else(|| refused("capture layer outside plan"))?;
        let (operand, boundary) = match projection {
            Projection::Query | Projection::Key | Projection::Value => {
                let attention = op
                    .attention
                    .softmax()
                    .ok_or_else(|| refused("capture requires softmax attention"))?;
                (
                    match projection {
                        Projection::Query => &attention.q,
                        Projection::Key => &attention.k,
                        _ => &attention.v,
                    },
                    Boundary::AttentionInput,
                )
            }
            Projection::Gate | Projection::Up | Projection::Down => {
                let Some(LayerFfn::Dense(ffn)) = &op.ffn else {
                    return Err(refused("capture requires a dense FFN"));
                };
                match projection {
                    Projection::Gate => (
                        ffn.gate
                            .as_ref()
                            .ok_or_else(|| refused("no gate projection"))?,
                        Boundary::FfnInput,
                    ),
                    Projection::Up => (&ffn.up, Boundary::FfnInput),
                    _ => (&ffn.down, Boundary::FfnDownInput),
                }
            }
        };
        let [rows, width] = operand.shape.as_slice() else {
            return Err(refused("site is not a matrix"));
        };
        if *rows == 0 || *width == 0 {
            return Err(refused("empty site matrix"));
        }
        Ok(Self {
            layer,
            projection,
            boundary,
            object: operand.object.clone(),
            tensor: operand.tensor.clone(),
            rows: *rows,
            width: *width,
        })
    }
}

// Includes glue operands (norms, biases, sinks), which planned_operands()
// intentionally omits. Walk the plan's serialized OperandRef records so future
// optional glue is sealed too. This is capture provenance, not a replacement
// for RepresentationStateId or the canonical source semantic identity.
#[derive(Deserialize)]
struct RefRecord {
    object: String,
    tensor: String,
    dtype: String,
    shape: Vec<usize>,
}

fn refs(
    value: &serde_json::Value,
    out: &mut BTreeMap<(String, String), OperandRef>,
) -> Result<(), VindexError> {
    match value {
        serde_json::Value::Object(fields)
            if fields.contains_key("object") && fields.contains_key("tensor") =>
        {
            let r: RefRecord = serde_json::from_value(value.clone()).map_err(refused)?;
            let operand = OperandRef {
                object: r.object,
                tensor: r.tensor,
                dtype: r.dtype,
                shape: r.shape,
            };
            let key = (operand.object.clone(), operand.tensor.clone());
            if let Some(old) = out.insert(key, operand.clone()) {
                if old != operand {
                    return Err(refused("conflicting operand geometries in prefix"));
                }
            }
        }
        serde_json::Value::Object(fields) => {
            for child in fields.values() {
                refs(child, out)?;
            }
        }
        serde_json::Value::Array(items) => {
            for child in items {
                refs(child, out)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Seal plan plus effective decoded operand values, one tensor at a time.
/// Includes the target layer in full (conservative invalidation), never later
/// layers or the output head. Overlaid candidates use their effective values;
/// no caller-supplied candidate name or process-local source stamp is trusted.
pub(super) fn image_digest(
    plan: &ComponentOpPlan,
    source: OperandSource<'_>,
) -> Result<String, VindexError> {
    let serialized = serde_json::to_value(plan).map_err(refused)?;
    let mut operands = BTreeMap::new();
    refs(&serialized, &mut operands)?;
    if operands.is_empty() {
        return Err(refused("prefix contains no operands"));
    }
    let mut hash = Sha256::new();
    hash.update(SCHEMA.as_bytes());
    hash.update(json_digest(&serialized)?.as_bytes());
    for operand in operands.values() {
        hash.update(json_digest(operand)?.as_bytes());
        let dtype = source
            .stored_dtype(operand)
            .ok_or_else(|| refused("missing prefix operand"))?;
        hash.update(json_digest(&dtype)?.as_bytes());
        let values = source.load(operand)?;
        let expected = operand
            .shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .ok_or_else(|| refused("operand shape overflow"))?;
        if values.len() != expected || values.iter().any(|v| !v.is_finite()) {
            return Err(refused(
                "prefix operand has wrong shape or non-finite values",
            ));
        }
        hash.update((values.len() as u64).to_le_bytes());
        let mut buffer = [0u8; 65536];
        for chunk in values.chunks(buffer.len() / 4) {
            for (value, bytes) in chunk.iter().zip(buffer.chunks_exact_mut(4)) {
                bytes.copy_from_slice(&value.to_le_bytes());
            }
            hash.update(&buffer[..chunk.len() * 4]);
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}
