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

use larql_vindex::format::vindex3::weight_patch::structured::{OperandBits, Profile, ScalarBits};
use larql_vindex::format::vindex3::weight_patch::{
    self as wp, bf16::bf16_to_f32, StructuredWeightOp, WeightEdit, WeightPatch,
};
use serde::Deserialize;

use crate::error::LqlError;
use crate::executor::{Backend, Session};

/// The operand file a `REMOVE DIRECTION` reads: the direction's stored bits, and optionally where it came
/// from. Bits, never decimals: the file fixes the operand exactly.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OperandFile {
    bits: Vec<String>,
    #[serde(default)]
    origin: Option<String>,
}

fn refused_direction(e: impl std::fmt::Display) -> LqlError {
    LqlError::Execution(format!("REMOVE DIRECTION refused: {e}"))
}

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
        // One patch per tensor, in the order each tensor was first edited.
        // Delta ops stay deltas; structured ops resolve to the exact BF16 results the numerical profile
        // defines. Nothing is converted from one form to the other.
        let held: Vec<wp::WeightOpRef<'_>> = ops.iter().map(|(_, op)| *op).collect();
        let (patches, origins) = wp::declare_patches(path, &held)
            .map_err(|e| LqlError::Execution(format!("COMPILE refused: {e}")))?;
        let report = wp::apply_declared_patches(path, &patches, &origins, Path::new(output))
            .map_err(|e| LqlError::Execution(format!("COMPILE refused: {e}")))?;
        Ok(vec![
            format!(
                "Compiled: {output} (VINDEX3, clean container; weight patch of {} element(s) of {})",
                report.elements,
                report.tensors.join(", ")
            ),
            format!("  re-derived: {}", if report.rederived.is_empty() { "(none)".into() } else { report.rederived.join(", ") }),
            format!("  edited source (canonical) payload {}", &report.canonical.payload_sha256[..16]),
            format!(
                "  ledger weight_patches.json {} ({} patch op(s) as origin)",
                report.ledger.schema,
                report.ledger.origins.len()
            ),
        ])
    }

    /// `DISCARD WEIGHT EDITS`: drop every stored-weight op staged directly in this session, and the copies an
    /// open recording holds, so a mis-staged edit can be withdrawn without re-binding the container. Ops an
    /// applied patch carries stay with that patch.
    pub(crate) fn exec_discard_weight_edits(&mut self) -> Result<Vec<String>, LqlError> {
        let overlay = match &mut self.backend {
            Backend::Vindex3 { overlay, .. } => overlay,
            Backend::Vindex { .. } => return Err(LqlError::Execution(
                "DISCARD WEIGHT EDITS drops stored-weight edits staged on a VINDEX3 container; \
                     this session is bound to a VINDEX2 vindex"
                    .into(),
            )),
            _ => return Err(LqlError::NoBackend),
        };
        let discarded = overlay.discard_session_weight_ops();
        let from_patches = overlay.weight_ops().len();
        let mut unrecorded = 0;
        if let Some(recording) = &mut self.patch_recording {
            let before = recording.operations.len();
            recording.operations.retain(|op| op.weight_op().is_none());
            unrecorded = before - recording.operations.len();
        }
        let mut out = vec![if discarded == 0 {
            "Nothing to discard: no stored-weight op is staged in this session".to_string()
        } else {
            format!("Discarded: {discarded} stored-weight op(s) staged in this session")
        }];
        if unrecorded > 0 {
            out.push(format!(
                "  {unrecorded} op(s) removed from the open patch recording"
            ));
        }
        if from_patches > 0 {
            out.push(format!(
                "  {from_patches} stored-weight op(s) from applied patches remain pending; REMOVE PATCH removes them"
            ));
        }
        Ok(out)
    }

    /// `REMOVE DIRECTION`: stage a structured `remove_direction` on the bound container. Everything
    /// checkable is checked before anything is staged: the operand file, the profile, the tensor, the
    /// columns, and that the op, resolved by the frozen interpreter, edits at least one element.
    pub(crate) fn exec_remove_direction(
        &mut self,
        operand: &str,
        tensor: &str,
        columns: &[usize],
        scale: f64,
        numerics: &str,
    ) -> Result<Vec<String>, LqlError> {
        let path = match &self.backend {
            Backend::Vindex3 { path, .. } => path.clone(),
            Backend::Vindex { .. } => {
                return Err(LqlError::Execution(
                    "REMOVE DIRECTION edits the stored weights of a VINDEX3 container; \
                     this session is bound to a VINDEX2 vindex"
                        .into(),
                ))
            }
            _ => return Err(LqlError::NoBackend),
        };
        let profile = Profile::parse(Some(numerics)).map_err(refused_direction)?;
        if !profile.works_in_f64() {
            return Err(refused_direction(format!(
                "`{numerics}` works in f32; REMOVE DIRECTION takes f64-working profiles, so that the \
                 scale literal is not rounded on the way in"
            )));
        }
        if !scale.is_finite() {
            return Err(refused_direction("SCALE must be finite"));
        }
        let text = std::fs::read_to_string(operand)
            .map_err(|e| refused_direction(format!("cannot read operand file {operand:?}: {e}")))?;
        let file: OperandFile = serde_json::from_str(&text)
            .map_err(|e| refused_direction(format!("operand file {operand:?}: {e}")))?;
        let bits = file
            .bits
            .iter()
            .map(|h| {
                (h.len() == 16 && h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f')))
                    .then(|| u64::from_str_radix(h, 16).ok())
                    .flatten()
                    .ok_or_else(|| {
                        refused_direction(format!(
                            "`{h}` is not 16 lowercase hex digits (f64 bits)"
                        ))
                    })
            })
            .collect::<Result<Vec<u64>, _>>()?;
        let mut op = StructuredWeightOp::bind(
            &path,
            tensor,
            numerics,
            &OperandBits::F64(bits),
            ScalarBits::F64(scale.to_bits()),
            columns.to_vec(),
        )
        .map_err(refused_direction)?;
        if let Some(origin) = file.origin {
            op = op.with_origin(origin);
        }
        // Resolve now, so a bad op is refused when it is written down and not when it is compiled.
        let resolved = op.resolve(&path).map_err(refused_direction)?;
        let digest = op.digest.clone();
        let Backend::Vindex3 { overlay, .. } = &mut self.backend else {
            unreachable!("matched above")
        };
        overlay
            .stage_structured_op(op.clone())
            .map_err(refused_direction)?;
        let pending = overlay.structured_op_count();
        if let Some(recording) = &mut self.patch_recording {
            recording
                .operations
                .push(larql_vindex::PatchOp::StructuredWeight(op));
        }
        let mut out = vec![format!(
            "Staged: remove direction ({} component(s)) from {tensor}, column(s) {columns:?}, scale {scale} under {numerics}; \
             {} element(s) would change",
            file.bits.len(),
            resolved.edits.len()
        )];
        out.push(format!("  op digest {}", &digest[..16]));
        out.push("  results are BF16 as the profile defines them; no f32 delta is involved".into());
        out.push(format!(
            "  {pending} structured op(s) pending in this session; takes effect at COMPILE CURRENT INTO VINDEX"
        ));
        Ok(out)
    }
}
