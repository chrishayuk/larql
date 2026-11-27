//! Closure of the head-side objects the GPT-2 lineage added: the learned
//! input position table and the final norm's bias, plus the attention
//! bias declarations' contradictions.

use std::collections::BTreeMap;

use super::super::super::encode::segment::SegmentTensor;
use super::super::super::graph::surface::AttentionSurface;
use super::super::super::graph::{LogicalObject, ObjectKind};
use super::super::ClosureDefect;
use larql_models::config::InputPositionEncoding;

/// Declaring two attention-bias forms at once is a contradiction, refused
/// rather than resolved in favour of either.
pub(super) fn bias_contradictions(
    attn: Option<&AttentionSurface>,
    component: &str,
    defects: &mut Vec<ClosureDefect>,
) {
    if attn.is_some_and(|a| {
        a.output_bias == Some(true) && (a.attention_bias == Some(true) || a.qkv_bias == Some(true))
    }) {
        defects.push(ClosureDefect::ContradictoryDeclaration {
            component: component.to_string(),
            detail: "`output_bias` (output projection alone biased) is declared beside \
                     `attention_bias` or `qkv_bias`, which bias Q/K/V"
                .to_string(),
        });
    }
    if attn.is_some_and(|a| a.attention_bias == Some(true) && a.qkv_bias == Some(true)) {
        defects.push(ClosureDefect::ContradictoryDeclaration {
            component: component.to_string(),
            detail: "`attention_bias` (Q/K/V and output biased) and `qkv_bias` (Q/K/V only, \
                     output unbiased) are both declared true"
                .to_string(),
        });
    }
}

/// The learned input position table: present iff the head surface
/// declares one, as exactly one `[max_positions, hidden]` tensor.
pub(super) fn position_table(
    input_position: Option<InputPositionEncoding>,
    tables: &BTreeMap<ObjectKind, (&LogicalObject, Vec<SegmentTensor>)>,
    component: &str,
    hidden: usize,
    defects: &mut Vec<ClosureDefect>,
) -> Option<(InputPositionEncoding, String, SegmentTensor)> {
    let object_id = || format!("{component}.{}", ObjectKind::PositionEmbedding.name());
    match (input_position, tables.get(&ObjectKind::PositionEmbedding)) {
        (Some(encoding), Some((object, tensors))) => {
            let [tensor] = tensors.as_slice() else {
                defects.push(ClosureDefect::ObjectShape {
                    object: object.id.clone(),
                    detail: format!("expected exactly 1 tensor, found {}", tensors.len()),
                });
                return None;
            };
            let expected = vec![encoding.max_positions(), hidden];
            if tensor.shape != expected {
                defects.push(ClosureDefect::GeometryMismatch {
                    tensor: format!("{}/{}", object.id, tensor.name),
                    expected,
                    actual: tensor.shape.clone(),
                });
            }
            Some((encoding, object.id.clone(), tensor.clone()))
        }
        (Some(_), None) => {
            defects.push(ClosureDefect::ObjectShape {
                object: object_id(),
                detail: "the head surface declares a learned input position table, and the \
                         component owns no position-embedding object"
                    .to_string(),
            });
            None
        }
        (None, Some(_)) => {
            defects.push(ClosureDefect::ObjectShape {
                object: object_id(),
                detail: "a position-embedding object is present and the head surface declares \
                         no input position encoding — the table would be read by nothing"
                    .to_string(),
            });
            None
        }
        (None, None) => None,
    }
}

/// The final norm's weight and bias, under a surface that declares norm
/// biases: exactly two `[hidden]` tensors, one per leaf.
pub(super) fn final_norm_with_bias(
    tables: &BTreeMap<ObjectKind, (&LogicalObject, Vec<SegmentTensor>)>,
    hidden: usize,
    defects: &mut Vec<ClosureDefect>,
) -> Option<(String, SegmentTensor, Option<SegmentTensor>)> {
    let (object, tensors) = tables.get(&ObjectKind::FinalNorm)?;
    let leaf = |t: &&SegmentTensor, name: &str| t.name.rsplit('.').next() == Some(name);
    let weight = tensors.iter().find(|t| leaf(t, "weight"));
    let bias = tensors.iter().find(|t| leaf(t, "bias"));
    let (Some(weight), Some(bias), 2) = (weight, bias, tensors.len()) else {
        defects.push(ClosureDefect::ObjectShape {
            object: object.id.clone(),
            detail: format!(
                "norm biases are declared: expected one `weight` and one `bias` tensor, found {:?}",
                tensors.iter().map(|t| t.name.as_str()).collect::<Vec<_>>()
            ),
        });
        return None;
    };
    for tensor in [weight, bias] {
        if tensor.shape != [hidden] {
            defects.push(ClosureDefect::GeometryMismatch {
                tensor: format!("{}/{}", object.id, tensor.name),
                expected: vec![hidden],
                actual: tensor.shape.clone(),
            });
        }
    }
    Some((object.id.clone(), weight.clone(), Some(bias.clone())))
}
