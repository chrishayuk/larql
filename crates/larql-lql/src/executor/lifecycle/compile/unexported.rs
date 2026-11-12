//! What a legacy `COMPILE` refuses instead of leaving out.
//!
//! `COMPILE INTO MODEL` writes model weights. The only session edit it can
//! put into them is a compose-mode `INSERT`, through the MEMIT solve, and
//! the solve runs only when `LARQL_MEMIT_ENABLE` is set. Every other edit
//! (a `DELETE`, an `UPDATE`, a KNN-mode `INSERT`, an insert the solve
//! skipped) has no place in the export. A compile that answered `Compiled`
//! over any of them would publish the source's weights as the edited model,
//! so each is refused by name before a file is written.

use crate::ast::{CompileTarget, OutputFormat};
use crate::error::LqlError;
use larql_vindex::{PatchOp, PatchedVindex};

/// The switch that turns the MEMIT solve on.
pub(super) const MEMIT_ENABLE_ENV: &str = "LARQL_MEMIT_ENABLE";

/// How many skipped facts a refusal lists.
const SKIPPED_PREVIEW: usize = 5;

/// Whether the MEMIT solve is switched on for this process.
///
/// Read through the override-aware `larql_compute::options` helper so tests
/// can toggle it without `std::env::set_var` (which races concurrent
/// `getenv` on the decode path and SIGSEGVs libc).
pub(super) fn memit_enabled() -> bool {
    larql_compute::options::env_value(MEMIT_ENABLE_ENV)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Refuse a `FORMAT` clause the legacy compile does not write.
///
/// The clause used to be parsed and dropped, so `FORMAT GGUF` produced the
/// same files as no clause at all.
pub(super) fn refuse_unwritten_format(
    target: CompileTarget,
    format: Option<OutputFormat>,
) -> Result<(), LqlError> {
    match (target, format) {
        (_, None) | (CompileTarget::Model, Some(OutputFormat::Safetensors)) => Ok(()),
        (CompileTarget::Model, Some(OutputFormat::Gguf)) => Err(LqlError::Execution(
            "COMPILE INTO MODEL does not write FORMAT GGUF: nothing was compiled".into(),
        )),
        (CompileTarget::Vindex, Some(format)) => Err(LqlError::Execution(format!(
            "COMPILE INTO VINDEX takes no FORMAT clause (got {format:?}): nothing was compiled"
        ))),
    }
}

/// The session's edit operations of each kind the model export cannot hold.
#[derive(Default)]
struct EditCensus {
    deletes: usize,
    updates: usize,
    knn: usize,
    inserts: usize,
}

fn census(patched: &PatchedVindex, recording_ops: &[PatchOp]) -> EditCensus {
    let mut census = EditCensus::default();
    let applied = patched.patches.iter().flat_map(|p| p.operations.iter());
    for op in applied.chain(recording_ops) {
        match op {
            PatchOp::Delete { .. } => census.deletes += 1,
            PatchOp::Update { .. } => census.updates += 1,
            PatchOp::InsertKnn { .. } | PatchOp::DeleteKnn { .. } => census.knn += 1,
            PatchOp::Insert { .. } => census.inserts += 1,
            // Applies to a VINDEX3 container, which has its own compile.
            PatchOp::WeightDelta(_) | PatchOp::StructuredWeight(_) => {}
        }
    }
    // Overlay state no recorded operation accounts for is still an edit.
    census.deletes = census.deletes.max(patched.num_deleted());
    census.knn = census.knn.max(patched.knn_store.len());
    if census.inserts == 0 && census.updates == 0 && patched.has_vector_or_meta_overrides() {
        census.inserts = 1;
    }
    census
}

/// Refuse a model export that would not contain the session's edits.
///
/// `solved_facts` is how many inserts the MEMIT solve will write, and
/// `skipped` the lines describing the ones it will not.
pub(super) fn refuse_unexported_edits(
    patched: &PatchedVindex,
    recording_ops: &[PatchOp],
    solved_facts: usize,
    skipped: &[String],
) -> Result<(), LqlError> {
    let census = census(patched, recording_ops);
    let refuse = |what: String| {
        Err(LqlError::Execution(format!(
            "COMPILE INTO MODEL cannot export {what}: the exported weights would not \
             contain the edit, so nothing was compiled"
        )))
    };
    if census.deletes > 0 {
        return refuse(format!("{} DELETE edit(s)", census.deletes));
    }
    if census.updates > 0 {
        return refuse(format!("{} UPDATE edit(s)", census.updates));
    }
    if census.knn > 0 {
        return refuse(format!(
            "{} KNN-mode edit(s), which live in the KNN store and not in the weights",
            census.knn
        ));
    }
    if census.inserts == 0 {
        return Ok(());
    }
    if !memit_enabled() {
        return refuse(format!(
            "{} INSERT edit(s) while {MEMIT_ENABLE_ENV} is not set (the MEMIT solve is the \
             only path that writes an INSERT into exported weights)",
            census.inserts
        ));
    }
    if !skipped.is_empty() {
        let preview = skipped
            .iter()
            .take(SKIPPED_PREVIEW)
            .map(|line| line.trim())
            .collect::<Vec<_>>()
            .join("; ");
        return refuse(format!(
            "{} INSERT edit(s) the MEMIT solve skips ({preview})",
            skipped.len()
        ));
    }
    if solved_facts == 0 {
        return refuse(format!(
            "{} INSERT edit(s) that give the MEMIT solve no fact",
            census.inserts
        ));
    }
    Ok(())
}
