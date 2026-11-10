//! `UPDATE WEIGHTS` and the weight-edit side of `COMPILE` on a VINDEX3 binding.
//!
//! A weight edit names real elements of one stored tensor. It is staged in the
//! session overlay, bound to the exact container it was written against, and
//! written into a new container by `COMPILE CURRENT INTO VINDEX`, together
//! with every other pending edit (of any tensor of the same canonical
//! representation) as one patch. Staging
//! writes nothing and leaves the container untouched; nothing in this session
//! sees the edit until it is compiled.
//!
//! The edit lives in one of two places, both kept by the overlay: directly in
//! the session, or inside an applied patch (`APPLY PATCH` of a `.vlp` that
//! carries `weight_delta` ops). While a `BEGIN PATCH` recording is active,
//! a staged edit is also recorded into it, so `SAVE PATCH` can write it out.

use std::path::Path;

use larql_vindex::format::vindex3::weight_patch::{
    self as wp, bf16::bf16_to_f32, WeightEdit, WeightPatch,
};

use crate::error::LqlError;
use crate::executor::{Backend, Session};

/// How many edited elements a staging reply lists before summarising.
const SHOWN_ELEMENTS: usize = 8;

fn refused(e: impl std::fmt::Display) -> LqlError {
    LqlError::Execution(format!("UPDATE WEIGHTS refused: {e}"))
}

impl Session {
    pub(crate) fn exec_update_weights(
        &mut self,
        tensor: &str,
        row: usize,
        cols: &[usize],
        delta: f32,
    ) -> Result<Vec<String>, LqlError> {
        let path = match &self.backend {
            Backend::Vindex3 { path, .. } => path.clone(),
            Backend::Vindex { .. } => {
                return Err(LqlError::Execution(
                    "UPDATE WEIGHTS edits the stored weights of a VINDEX3 container; \
                     this session is bound to a VINDEX2 vindex"
                        .into(),
                ))
            }
            _ => return Err(LqlError::NoBackend),
        };
        let edits: Vec<WeightEdit> = cols
            .iter()
            .map(|&col| WeightEdit { row, col, delta })
            .collect();
        // Everything checkable is checked here, before anything is staged.
        let planned = wp::plan_weight_patch(
            &path,
            &WeightPatch {
                tensor: tensor.to_string(),
                edits: edits.clone(),
            },
        )
        .map_err(refused)?;
        let op = wp::record_op(&path, tensor, &edits).map_err(refused)?;
        let digest = op.digest.clone();
        let Backend::Vindex3 { overlay, .. } = &mut self.backend else {
            unreachable!("matched above")
        };
        overlay.stage_weight_op(op.clone()).map_err(refused)?;
        let total = overlay.weight_edit_count();
        if let Some(recording) = &mut self.patch_recording {
            recording
                .operations
                .push(larql_vindex::PatchOp::WeightDelta(op));
        }

        let mut out = vec![format!(
            "Staged: {} element(s) of {tensor}, row {row}, delta {delta} (BF16, nearest-even of the exact sum)",
            planned.len()
        )];
        for e in planned.iter().take(SHOWN_ELEMENTS) {
            out.push(format!(
                "  [{row}, {}]: {:.7} -> {:.7}",
                e.col,
                bf16_to_f32(e.old_bits),
                bf16_to_f32(e.new_bits)
            ));
        }
        if planned.len() > SHOWN_ELEMENTS {
            out.push(format!("  … and {} more", planned.len() - SHOWN_ELEMENTS));
        }
        out.push(format!("  op digest {}", &digest[..16]));
        out.push(format!(
            "  {total} weight element(s) pending in this session"
        ));
        match &self.patch_recording {
            Some(r) if !r.path.is_empty() => out.push(format!(
                "  recorded into patch \"{}\" ({} ops pending)",
                r.path,
                r.operations.len()
            )),
            Some(_) => out.push("  recorded into the anonymous patch".into()),
            None => {}
        }
        out.push(
            "  Takes effect in a container written by COMPILE CURRENT INTO VINDEX; \
             INFER in this session does not see it."
                .into(),
        );
        Ok(out)
    }

    /// `COMPILE CURRENT INTO VINDEX` when weight edits are pending: write a new
    /// container with every pending edit applied, or nothing.
    pub(crate) fn exec_compile_weights(&mut self, output: &str) -> Result<Vec<String>, LqlError> {
        let Backend::Vindex3 { path, overlay, .. } = &self.backend else {
            return Err(LqlError::NoBackend);
        };
        if overlay.has_knowledge_state() {
            return Err(LqlError::Execution(
                "COMPILE cannot combine stored-weight edits with knowledge edits (INSERT/UPDATE EDGES/DELETE) \
                 in one container. Compile them separately: REMOVE the knowledge patches, or compile the \
                 knowledge first and apply the weight edits to that container."
                    .into(),
            ));
        }
        let ops = overlay.weight_ops();
        // The container may have changed since staging; re-check every binding.
        for (_, op) in &ops {
            op.check_binding(path)
                .map_err(|e| LqlError::Execution(format!("COMPILE refused: {e}")))?;
        }
        // One patch per tensor, in the order each tensor was first edited.
        let mut patches: Vec<WeightPatch> = Vec::new();
        for (_, op) in &ops {
            let edits = op.weight_edits();
            match patches.iter_mut().find(|p| p.tensor == op.tensor) {
                Some(p) => p.edits.extend(edits),
                None => patches.push(WeightPatch {
                    tensor: op.tensor.clone(),
                    edits,
                }),
            }
        }
        let origins: Vec<String> = ops.iter().map(|(_, o)| o.digest.clone()).collect();
        let report = wp::apply_weight_patches(path, &patches, &origins, Path::new(output))
            .map_err(|e| LqlError::Execution(format!("COMPILE refused: {e}")))?;
        Ok(vec![
            format!(
                "Compiled: {output} (VINDEX3, clean container; weight patch of {} element(s) of {})",
                report.elements,
                report.tensors.join(", ")
            ),
            format!("  re-derived: {}", if report.rederived.is_empty() { "(none)".into() } else { report.rederived.join(", ") }),
            format!("  edited source (canonical) payload {}", &report.canonical.payload_sha256[..16]),
            format!("  ledger weight_patches.json ({} patch op(s) as origin)", report.ledger.origins.len()),
        ])
    }
}
