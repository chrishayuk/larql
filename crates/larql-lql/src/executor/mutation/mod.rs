//! Mutation executor: INSERT, DELETE, UPDATE, MERGE, REBALANCE.
//!
//! All mutations go through the `PatchedVindex` overlay — base vindex
//! files on disk are never modified.

mod delete;
mod insert;
mod merge;
mod rebalance;
mod update;

use std::collections::HashMap;

use crate::ast::Condition;
use crate::error::LqlError;
use crate::executor::predicate::{EdgePredicate, PredicateScope};
use crate::executor::query::select::edges::resolve_edge_rows;
use crate::executor::Session;

type PatchVectorSnapshot = (Option<String>, Option<String>, Option<String>);

impl Session {
    /// The `(layer, feature)` slots a DELETE / UPDATE `WHERE` clause
    /// addresses.
    ///
    /// The clause is validated whole first ([`EdgePredicate::parse`]), so
    /// a condition this executor cannot evaluate refuses the statement
    /// instead of widening it. Everything here is a read: a caller that
    /// gets an `Err` back has changed neither the overlay nor the patch
    /// recording.
    ///
    /// The slots are those of the rows `SELECT * FROM EDGES WHERE <the
    /// same clause>` lists, before its ORDER BY and LIMIT — one resolver
    /// serves both ([`resolve_edge_rows`]), so a mutation cannot reach an
    /// edge its preview did not show. Four things differ deliberately,
    /// all in the narrowing direction:
    /// - `entity` together with `relation` is refused (SELECT ranks a
    ///   gate walk for that pair; see below);
    /// - a `relation` condition needs relation labels loaded (SELECT
    ///   lists nothing without them; a mutation refuses);
    /// - `layer` + `feature` together name one slot, which must exist
    ///   (SELECT lists nothing; a mutation refuses);
    /// - SELECT may reinterpret a relation that matched nothing as a
    ///   synonym of a known one; a mutation takes the relation literally.
    pub(super) fn resolve_mutation_targets(
        &self,
        conditions: &[Condition],
    ) -> Result<Vec<(usize, usize)>, LqlError> {
        let predicate = EdgePredicate::parse(conditions, PredicateScope::EDGES)?;
        if predicate.entity.is_some() && predicate.relation.is_some() {
            // SELECT answers `entity` + `relation` with an embedding walk:
            // the strongest gate hits for the entity whose label matches
            // the relation — hundreds of slots per layer, none of which
            // need mention the entity. That is a ranking to read, not a
            // set to destroy.
            return Err(LqlError::Execution(
                "unsupported WHERE clause: a mutation does not combine `entity` and \
                 `relation` — SELECT resolves that pair by a ranked gate walk, which is \
                 too broad to delete or update by. List the edges with SELECT and address \
                 each by `layer` and `feature`"
                    .into(),
            ));
        }
        let ctx = self.browse()?;
        let classifier = self.relation_classifier();
        if predicate.relation.is_some() && classifier.is_none() {
            return Err(LqlError::Execution(
                "relation filters require relation labels for the active vindex; \
                 target by layer/feature or omit relation"
                    .into(),
            ));
        }

        if let (Some(layer), Some(feature)) = (predicate.layer, predicate.feature) {
            let slots = ctx.source.num_features(layer);
            if feature >= slots {
                return Err(LqlError::Execution(format!(
                    "no such feature slot: L{layer} F{feature} (layer {layer} has {slots} \
                     feature slots)"
                )));
            }
        }

        // The one resolver SELECT uses: a mutation reaches exactly the
        // slots the matching SELECT lists.
        let targets = resolve_edge_rows(&ctx, classifier, &predicate)?
            .into_iter()
            .map(|row| (row.layer, row.feature))
            .collect();
        Ok(targets)
    }
}

impl Session {
    pub(crate) fn refresh_recorded_patch_ops_for_slots(
        &mut self,
        slots: &[(usize, usize)],
    ) -> Result<(), LqlError> {
        if slots.is_empty() || self.patch_recording.is_none() {
            return Ok(());
        }

        let mut snapshots: HashMap<(usize, usize), PatchVectorSnapshot> = HashMap::new();
        {
            let (_, _, patched) = self.require_vindex()?;
            for &(layer, feature) in slots {
                let gate = patched.overrides_gate_at(layer, feature).map(encode_vector);
                let up = patched.up_override_at(layer, feature).map(encode_vector);
                let down = patched.down_override_at(layer, feature).map(encode_vector);
                snapshots.insert((layer, feature), (gate, up, down));
            }
        }

        // Present: checked on entry, and nothing above clears it.
        let operations = self
            .patch_recording
            .iter_mut()
            .flat_map(|recording| recording.operations.iter_mut());
        for op in operations {
            if let larql_vindex::PatchOp::Insert {
                layer,
                feature,
                gate_vector_b64,
                up_vector_b64,
                down_vector_b64,
                ..
            } = op
            {
                if let Some((gate, up, down)) = snapshots.get(&(*layer, *feature)) {
                    if let Some(gate) = gate {
                        *gate_vector_b64 = Some(gate.clone());
                    }
                    if let Some(up) = up {
                        *up_vector_b64 = Some(up.clone());
                    }
                    if let Some(down) = down {
                        *down_vector_b64 = Some(down.clone());
                    }
                }
            }
        }

        Ok(())
    }
}

fn encode_vector(vec: &[f32]) -> String {
    larql_vindex::patch::core::encode_gate_vector(vec)
}
