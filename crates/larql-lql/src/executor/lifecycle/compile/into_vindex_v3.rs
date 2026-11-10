//! `COMPILE CURRENT INTO VINDEX` on a VINDEX3 binding (V3-LQL-3D):
//! bake the session overlay into a clean container.
//!
//! ```text
//! base objects + KnowledgeOverlay
//!     → OperandOverrides (the same derivation execution uses)
//!     → larql_vindex bake: rewrite touched segments, link the rest
//!     → knn_store.bin carries the L0 knowledge (V2's own compiled
//!       form — the file USE binds, not a patch)
//! ```
//!
//! The result binds with a **zero-override overlay**: the composed
//! behaviour comes from the stored bytes. What cannot bake refuses
//! loudly (`bake_blockers`): tombstones and meta-only relabels have no
//! physical form in a container whose annotations are derived from
//! weights — they remain overlay/patch state.

use std::path::PathBuf;

use crate::error::LqlError;
use crate::executor::{Backend, Session};
use larql_vindex::format::filenames::KNN_STORE_BIN;
use larql_vindex::format::vindex3::compile::bake_container_with;
use larql_vindex::VindexError;

impl Session {
    pub(crate) fn exec_compile_into_vindex_v3(
        &mut self,
        output: &str,
    ) -> Result<Vec<String>, LqlError> {
        let Backend::Vindex3 {
            path,
            runtime,
            overlay,
            ..
        } = &self.backend
        else {
            unreachable!("caller matched the backend");
        };

        let blockers = overlay.bake_blockers();
        if !blockers.is_empty() {
            return Err(LqlError::Execution(format!(
                "COMPILE cannot bake this overlay: {} — V3 annotations are derived \
                 from weights, so tombstones and meta-only relabels have no clean-container \
                 form; keep them as an applied patch, or drop them before compiling",
                blockers.join(", ")
            )));
        }

        let overrides =
            crate::executor::vindex3::compose_overrides(runtime, overlay)?.unwrap_or_default();
        let out = PathBuf::from(output);
        // The KNN store is written into the staged container, so a
        // failed save leaves nothing at `out` rather than a container
        // missing its L0 knowledge.
        let knn_entries = overlay.knn_store.len();
        let report = bake_container_with(path, &overrides, &out, |staged| {
            if knn_entries > 0 {
                overlay
                    .knn_store
                    .save(&staged.join(KNN_STORE_BIN))
                    .map_err(|e| VindexError::Parse(format!("save knn store: {e}")))?;
            }
            Ok(())
        })
        .map_err(|e| LqlError::exec("compile: bake failed", e))?;

        Ok(vec![
            format!("Compiled: {} (VINDEX3, clean container)", out.display()),
            format!(
                "  {} segments rewritten ({} tensors baked), {} hard-linked",
                report.rewritten_segments, report.rewritten_tensors, report.linked_segments,
            ),
            format!(
                "  L0 knowledge: {} KNN entries {}",
                knn_entries,
                if knn_entries > 0 {
                    "→ knn_store.bin"
                } else {
                    "(none)"
                },
            ),
            "  the composed behaviour is in the stored bytes — no overlay required".into(),
        ])
    }
}
