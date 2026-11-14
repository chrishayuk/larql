//! `DIFF a b` on VINDEX3 containers (V3-LQL-3D): a **logical** report
//! — what model facts differ — with `DIFF … PHYSICAL` as the
//! subordinate storage-level mode.
//!
//! `CURRENT` means the session's model state: the bound container PLUS
//! its overlay (operand edits and L0 knowledge). That definition is
//! what makes the diff an instrument rather than a file comparator:
//! a session with an overlay and the clean container COMPILE bakes
//! from it report identically against any third side, and diff each
//! other as semantically empty — while the physical mode is free to
//! show that one stores its meaning in rewritten segments.

use std::path::PathBuf;

use crate::ast::VindexRef;
use crate::error::LqlError;
use crate::executor::{Backend, Session};
use larql_vindex::format::vindex3::diff::{physical_diff, semantic_diff, DiffSide};

impl Session {
    /// Build one side. `CURRENT` layers the session overlay on.
    fn v3_diff_side(&self, vref: &VindexRef) -> Result<DiffSide, LqlError> {
        match vref {
            VindexRef::Current => {
                let Backend::Vindex3 {
                    path,
                    runtime,
                    overlay,
                    ..
                } = &self.backend
                else {
                    return Err(LqlError::NoBackend);
                };
                let side = DiffSide::open(path, crate::executor::vindex3::V3_COMPONENT)
                    .map_err(|e| LqlError::exec("diff: reopen bound container", e))?;
                let overrides = crate::executor::vindex3::compose_overrides(runtime, overlay)?
                    .unwrap_or_default();
                Ok(side.with_overlay(overrides, overlay.knn_store.clone()))
            }
            VindexRef::Path(p) => {
                DiffSide::open(&PathBuf::from(p), crate::executor::vindex3::V3_COMPONENT)
                    .map_err(|e| LqlError::exec(format!("diff: open {p}"), e))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn exec_diff_v3(
        &self,
        a: &VindexRef,
        b: &VindexRef,
        layer_filter: Option<u32>,
        relation: Option<&str>,
        limit: Option<u32>,
        into_patch: Option<&str>,
        physical: bool,
    ) -> Result<Vec<String>, LqlError> {
        if into_patch.is_some() {
            return Err(crate::executor::vindex3::unsupported(
                "DIFF … INTO PATCH (extracting a diff as a patch)",
            ));
        }
        let side_a = self.v3_diff_side(a)?;
        let side_b = self.v3_diff_side(b)?;
        let name = |r: &VindexRef| match r {
            VindexRef::Current => "CURRENT".to_string(),
            VindexRef::Path(p) => p.clone(),
        };

        if physical {
            // Subordinate mode: how the meaning is stored. Overlays
            // have no physical form — CURRENT reports its bound
            // container's segments.
            let phys = physical_diff(side_a.index(), side_b.index());
            let mut out = vec![format!("Diff (physical): {} vs {}", name(a), name(b))];
            if phys.changed_segments.is_empty()
                && phys.only_in_a.is_empty()
                && phys.only_in_b.is_empty()
            {
                out.push("  (identical segments)".into());
                return Ok(out);
            }
            for (rep, sha_a, sha_b) in &phys.changed_segments {
                out.push(format!(
                    "  segment {rep}: {} → {}",
                    &sha_a[..12.min(sha_a.len())],
                    &sha_b[..12.min(sha_b.len())],
                ));
            }
            for rep in &phys.only_in_a {
                out.push(format!("  only in A: {rep}"));
            }
            for rep in &phys.only_in_b {
                out.push(format!("  only in B: {rep}"));
            }
            return Ok(out);
        }

        let diff =
            semantic_diff(&side_a, &side_b).map_err(|e| LqlError::exec("diff: compare", e))?;
        let mut out = vec![format!("Diff (logical): {} vs {}", name(a), name(b))];
        // A staged stored-weight op has no effect until COMPILE writes a container, so CURRENT is compared
        // without it. Say so: "equivalent" must not read as "the pending edit changed nothing".
        let pending = match (&self.backend, a, b) {
            (Backend::Vindex3 { overlay, .. }, VindexRef::Current, _)
            | (Backend::Vindex3 { overlay, .. }, _, VindexRef::Current) => {
                overlay.weight_ops().len()
            }
            _ => 0,
        };
        let pending_note = (pending > 0).then(|| {
            format!(
                "  note: {pending} stored-weight op(s) are pending COMPILE and are not part of CURRENT here; \
                 compile them and DIFF the container to compare the edited weights"
            )
        });
        if diff.is_empty() {
            out.push("  no semantic differences — the models are equivalent".into());
            out.extend(pending_note);
            return Ok(out);
        }
        out.extend(pending_note);

        let edges: Vec<_> = diff
            .knowledge_added
            .iter()
            .map(|e| ('+', e))
            .chain(diff.knowledge_removed.iter().map(|e| ('-', e)))
            .filter(|(_, (_, rel, _))| relation.is_none_or(|want| rel == want))
            .collect();
        if !edges.is_empty() {
            out.push("KNOWLEDGE".into());
            for (sign, (entity, rel, target)) in edges {
                out.push(format!("  {sign} {entity} —[{rel}]→ {target}"));
            }
        }

        let limit = limit.unwrap_or(20) as usize;
        let slots: Vec<_> = diff
            .features
            .iter()
            .filter(|s| layer_filter.is_none_or(|l| s.layer == l as usize))
            .collect();
        if !slots.is_empty() {
            out.push("FEATURES".into());
            for slot in slots.iter().take(limit) {
                let mut what = Vec::new();
                if slot.gate_changed {
                    what.push("gate_row");
                }
                if slot.up_changed {
                    what.push("up_row");
                }
                if slot.down_changed {
                    what.push("down_col");
                }
                out.push(format!(
                    "  L{} F{}   {} changed",
                    slot.layer,
                    slot.feature,
                    what.join(", ")
                ));
            }
            if slots.len() > limit {
                out.push(format!("  … {} more (LIMIT {limit})", slots.len() - limit));
            }
        }

        if !diff.changed_tensors.is_empty() {
            out.push("REPRESENTATIONS".into());
            for tensor in &diff.changed_tensors {
                out.push(format!("  {tensor}   semantic content changed"));
            }
        }
        if !diff.metadata.is_empty() {
            out.push("METADATA".into());
            for line in &diff.metadata {
                out.push(format!("  {line}"));
            }
        }
        out.push("SUMMARY".into());
        out.push(format!(
            "  feature slots changed: {}, knowledge +{}/−{}, other tensors: {}",
            diff.features.len(),
            diff.knowledge_added.len(),
            diff.knowledge_removed.len(),
            diff.changed_tensors.len(),
        ));
        Ok(out)
    }
}
