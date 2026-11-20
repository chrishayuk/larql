//! `COMPACT INTO VINDEX "out"` (VINDEX3): semantics-preserving
//! physical reorganisation of the bound container.
//!
//! COMPACT is not a second compiler: it refuses while the session
//! holds ANY overlay state, because removing overlays by materialising
//! meaning is COMPILE's job. The division keeps both operations
//! honest — COMPILE changes how meaning is stored to absorb edits;
//! COMPACT changes how the SAME meaning is stored, and the logical
//! DIFF is its proof instrument (`SemanticDiff(input, output) == ∅`).

use std::path::PathBuf;

use crate::error::LqlError;
use crate::executor::{Backend, Session};
use larql_vindex::format::vindex3::compact::compact_container_validated;
use larql_vindex::format::vindex3::diff::{semantic_diff, DiffSide};

impl Session {
    pub(crate) fn exec_compact_into(&mut self, output: &str) -> Result<Vec<String>, LqlError> {
        let Backend::Vindex3 { path, overlay, .. } = &self.backend else {
            return Err(LqlError::Execution(
                "COMPACT INTO VINDEX reorganises a VINDEX3 container; VINDEX2 compaction \
                 is the tiered COMPACT MINOR / COMPACT MAJOR"
                    .into(),
            ));
        };
        // Any pending edit, of knowledge or of stored weights: COMPACT writes the bound container's meaning
        // unchanged, so compacting past an edit would report success and lose it. Knowledge the container
        // already carries is not an edit and travels with it.
        if overlay.has_pending_edits() {
            let weight_ops = overlay.weight_ops().len();
            let held = if weight_ops > 0 {
                format!("{weight_ops} pending stored-weight op(s)")
            } else {
                "overlay state".to_string()
            };
            return Err(LqlError::Execution(format!(
                "the session holds {held} — COMPACT preserves meaning and cannot \
                 absorb edits; COMPILE CURRENT INTO VINDEX first (materialise the meaning), \
                 then COMPACT the result"
            )));
        }

        let out = PathBuf::from(output);
        // Before publication, the staged container must mean exactly what the
        // source means: the logical DIFF between them is empty (Gate B, B6
        // §3.3).
        let report =
            compact_container_validated(path, &out, |staged| require_same_meaning(path, staged))
                .map_err(|e| LqlError::exec("compact failed", e))?;

        let mut lines = vec![
            format!("Compacted: {} (VINDEX3)", out.display()),
            format!(
                "  {} segments carried byte-identically",
                report.carried_segments
            ),
        ];
        if report.dropped.is_empty() {
            lines.push("  nothing to drop — the container was already minimal".into());
        } else {
            lines.push(format!(
                "  {} unreferenced files dropped:",
                report.dropped.len()
            ));
            for name in &report.dropped {
                lines.push(format!("    - {name}"));
            }
        }
        Ok(lines)
    }
}

/// Gate B, B6 §3.3: the container at `staged` means exactly what the one at
/// `source` means. The logical DIFF between them is empty, or the error
/// counts what differs.
pub(super) fn require_same_meaning(
    source: &std::path::Path,
    staged: &std::path::Path,
) -> Result<(), larql_vindex::VindexError> {
    let component = crate::executor::vindex3::V3_COMPONENT;
    let before = DiffSide::open(source, component)?;
    let after = DiffSide::open(staged, component)?;
    let diff = semantic_diff(&before, &after)?;
    if diff.is_empty() {
        return Ok(());
    }
    Err(larql_vindex::VindexError::Parse(format!(
        "the compacted container does not mean what the source means: {} knowledge edge(s) \
         added, {} removed, {} feature(s), {} tensor(s) and {} metadata item(s) differ",
        diff.knowledge_added.len(),
        diff.knowledge_removed.len(),
        diff.features.len(),
        diff.changed_tensors.len(),
        diff.metadata.len(),
    )))
}
