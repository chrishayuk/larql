//! APPLY PATCH against the fact catalogue (`docs/fact-id-1.md` §3.5 I-2).
//!
//! Two independently written patches can each assert a value for one key.
//! Applying the second must not silently change the fact: its ops are folded
//! over the current catalogue first, and the whole patch is refused when one
//! of them inserts a key that is already live, updates or deletes a key that
//! is not a live KNN fact, or would give two keys one id. A key the same patch
//! inserted earlier may be inserted again (legacy patches recorded duplicates;
//! the fact loads as `ambiguous`), and a key the patch deletes may then be
//! inserted.

use std::collections::HashSet;

use larql_vindex::{FactKey, PatchOp, VindexPatch};

use crate::error::LqlError;
use crate::executor::Session;

use super::fold::Source;

impl Session {
    /// Refuse `patch` if applying it would break fact identity. Reads only.
    pub(crate) fn check_patch_facts(
        &self,
        patch: &VindexPatch,
        name: &str,
    ) -> Result<(), LqlError> {
        let Ok(ledger) = self.ledger() else {
            return Ok(());
        };
        let mut catalogue = ledger.catalogue();
        let source = Source::Patch(name.to_string());
        let mut inserted_here: HashSet<FactKey> = HashSet::new();
        let mut run: Option<FactKey> = None;
        for (i, op) in patch.operations.iter().enumerate() {
            let refuse = |reason: String| {
                LqlError::Execution(format!(
                    "APPLY PATCH refused: op {i} {reason}. Nothing was applied; change a fact \
                     with UPDATE, or DELETE it before inserting it again"
                ))
            };
            match op {
                PatchOp::InsertKnn {
                    entity, relation, ..
                }
                | PatchOp::Insert {
                    entity,
                    relation: Some(relation),
                    ..
                } => {
                    let key = FactKey::new(entity, relation);
                    let same_install = run.as_ref() == Some(&key);
                    if catalogue.is_live(&key) && !same_install && !inserted_here.contains(&key) {
                        return Err(refuse(format!(
                            "asserts ({entity}, {relation}), which is already the live fact {}",
                            key.fact_id()
                        )));
                    }
                    if let Some(other) = catalogue.colliding_key(&key) {
                        return Err(refuse(super::collision(&key, &other)));
                    }
                    inserted_here.insert(key);
                }
                PatchOp::UpdateKnnFact {
                    entity, relation, ..
                }
                | PatchOp::DeleteKnnFact { entity, relation } => {
                    let key = FactKey::new(entity, relation);
                    if !catalogue.is_live_knn(&key) {
                        return Err(refuse(format!(
                            "changes ({entity}, {relation}), which is not a live KNN fact here"
                        )));
                    }
                }
                _ => {}
            }
            catalogue.fold_op(op, &source, &mut run);
        }
        Ok(())
    }
}
