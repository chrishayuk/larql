//! The text of REMOVE PATCH's refusal over live session edits (Gate B, B2).
//!
//! Removal rebuilds the state from the bound vindex and the remaining
//! patches, so a knowledge edit no applied patch holds would be dropped.
//! REMOVE PATCH refuses instead, and this module says why: which patch, which
//! edits block it (in journal order), that edits held by applied patches do
//! not, and how to remove the patch without losing them.
//!
//! An edit is named by its fact id only when the op itself carries a fact key
//! (`insert_knn`, `update_knn_fact`, `delete_knn_fact`, a keyed compose
//! `insert`). A physical or weight op is named by its kind and its address,
//! never by a fact id derived for it.

use larql_vindex::{FactKey, PatchOp};

use super::PatchRecording;

/// How many blocking edits the refusal lists before summarising the rest.
pub(crate) const LISTED_EDITS_MAX: usize = 8;

/// One line naming a journal op: its kind, then its fact id or its address.
pub(crate) fn describe_op(op: &PatchOp) -> String {
    let fact = |entity: &str, relation: &str| FactKey::new(entity, relation).fact_id();
    match op {
        PatchOp::Insert {
            layer,
            feature,
            entity,
            relation: Some(relation),
            ..
        } => format!(
            "insert, fact {}, at layer {layer} feature {feature}",
            fact(entity, relation)
        ),
        PatchOp::Insert { layer, feature, .. } => {
            format!("insert at layer {layer} feature {feature}")
        }
        PatchOp::Update { layer, feature, .. } => {
            format!("update at layer {layer} feature {feature}")
        }
        PatchOp::Delete { layer, feature, .. } => {
            format!("delete at layer {layer} feature {feature}")
        }
        PatchOp::InsertKnn {
            layer,
            entity,
            relation,
            ..
        } => format!(
            "insert_knn, fact {}, at layer {layer}",
            fact(entity, relation)
        ),
        PatchOp::UpdateKnnFact {
            entity, relation, ..
        } => format!("update_knn_fact, fact {}", fact(entity, relation)),
        PatchOp::DeleteKnnFact { entity, relation } => {
            format!("delete_knn_fact, fact {}", fact(entity, relation))
        }
        PatchOp::DeleteKnn { entity } => format!("delete_knn of entity {entity:?}"),
        PatchOp::WeightDelta(w) => format!("weight_delta on tensor {}", w.tensor),
        PatchOp::StructuredWeight(w) => format!("structured_weight on tensor {}", w.tensor),
        PatchOp::LowRankColumnsWeight(w) => {
            format!("low_rank_columns_weight on tensor {}", w.tensor)
        }
    }
}

/// The refusal: the patch, the blocking edits in journal order, what does not
/// block, and a way to remove the patch that keeps them.
///
/// Every journal edit is also written to the open recording (one is started
/// if none is open), so each is either in that recording or in a file an
/// earlier SAVE PATCH wrote. The recovery saves the first and re-applies the
/// second.
pub(crate) fn refusal_text(
    path: &str,
    journal: &[&PatchOp],
    recording: Option<&PatchRecording>,
) -> String {
    let mut lines = vec![
        format!("REMOVE PATCH refused: {path:?}. Nothing was changed."),
        format!(
            "Removing a patch rebuilds the state from the vindex and the remaining patches. \
             This session holds {} knowledge edit(s) that no applied patch holds, and they \
             would be dropped:",
            journal.len()
        ),
    ];
    lines.extend(
        journal
            .iter()
            .take(LISTED_EDITS_MAX)
            .map(|op| format!("  {}", describe_op(op))),
    );
    if journal.len() > LISTED_EDITS_MAX {
        lines.push(format!("  +{} more", journal.len() - LISTED_EDITS_MAX));
    }
    lines.push("Edits held by applied patches do not block removal.".into());

    let save = match recording.filter(|r| !r.operations.is_empty()) {
        Some(r) if r.path.is_empty() => {
            "save the open recording with BEGIN PATCH \"file.vlp\" then SAVE PATCH; ".to_string()
        }
        Some(r) => format!("save the open recording to {} with SAVE PATCH; ", r.path),
        None => String::new(),
    };
    lines.push(format!(
        "To remove the patch and keep these edits: {save}bind the vindex again with USE; then \
         APPLY the patches to keep, including the file(s) this session's edits were saved to."
    ));
    lines.join("\n")
}
