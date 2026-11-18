//! UPDATE and DELETE by fact identity (`docs/fact-id-1.md` §5.3–5.4).
//!
//! Only `WHERE entity = … AND relation = …`, with nothing else, addresses a
//! fact (D-8). It reaches asserted facts and never discovered rows: a key
//! with no asserted fact is refused, not widened to the gate walk.
//! Everything is decided with read-only borrows first; only then is the
//! overlay changed and the op recorded, so a refusal changes nothing.

use larql_vindex::{FactKey, PatchOp};

use crate::ast::Condition;
use crate::error::LqlError;
use crate::executor::mutation::update_set::UpdateSet;
use crate::executor::predicate::{EdgePredicate, PredicateScope, RelationOp};
use crate::executor::{Backend, Session};

use super::{AssertedFact, FactStatus};

/// The `(entity, relation)` a WHERE clause addresses a fact by, when it has
/// exactly that shape (D-8); `None` for every other clause.
pub(crate) fn semantic_pair(
    conditions: &[Condition],
) -> Result<Option<(String, String)>, LqlError> {
    let predicate = EdgePredicate::parse(conditions, PredicateScope::EDGES)?;
    let pair = match (&predicate.entity, &predicate.relation) {
        (Some(entity), Some(relation))
            if relation.op == RelationOp::Eq
                && predicate.layer.is_none()
                && predicate.feature.is_none()
                && predicate.score.is_none() =>
        {
            Some((entity.to_string(), relation.wanted.to_string()))
        }
        _ => None,
    };
    Ok(pair)
}

/// Which statement is asking, for the refusal text.
#[derive(Clone, Copy)]
enum Verb {
    Update,
    Delete,
}

impl Verb {
    fn name(self) -> &'static str {
        match self {
            Verb::Update => "UPDATE",
            Verb::Delete => "DELETE",
        }
    }

    fn refused(self, reason: String) -> LqlError {
        let done = match self {
            Verb::Update => "updated",
            Verb::Delete => "deleted",
        };
        LqlError::Execution(format!(
            "{} refused: {reason}. Nothing was {done}.",
            self.name()
        ))
    }
}

impl Session {
    /// The live asserted fact `(entity, relation)` names, or the refusal.
    fn addressed_fact(
        &self,
        verb: Verb,
        entity: &str,
        relation: &str,
    ) -> Result<AssertedFact, LqlError> {
        let ledger = self.ledger()?;
        let single = |t: &str, id: u32| ledger.is_single(t, id);
        ledger
            .catalogue()
            .live(&FactKey::new(entity, relation), &single)
            .ok_or_else(|| {
                verb.refused(format!(
                    "no asserted fact ({entity}, {relation}). {} by entity and relation reaches \
                     facts LQL inserted, never discovered rows; address a discovered row by layer \
                     and feature",
                    verb.name()
                ))
            })
    }

    /// Why `fact` cannot take `verb`, if it cannot.
    fn status_refusal(verb: Verb, fact: &AssertedFact) -> Option<LqlError> {
        let allowed = match verb {
            Verb::Update => matches!(fact.status, FactStatus::Active),
            Verb::Delete => matches!(
                fact.status,
                FactStatus::Active | FactStatus::MultiToken | FactStatus::Ambiguous
            ),
        };
        if allowed {
            return None;
        }
        let why = match fact.status {
            FactStatus::Compose | FactStatus::Degraded => format!(
                "fact {} is a compose install ({}). No recorded op returns a compose slot to its \
                 base, and balancing ties installs together, so it is read-only by identity. \
                 REMOVE PATCH of the patch that installed it is the exact rollback; a DELETE by \
                 layer and feature tombstones a slot for whoever accepts that effect",
                fact.id,
                fact.status.label()
            ),
            FactStatus::Inconsistent => format!(
                "fact {} is inconsistent: the ledger and the live KNN store disagree about it, so \
                 no op can be stated against it",
                fact.id
            ),
            FactStatus::MultiToken => format!(
                "fact {} has a multi-token target {:?}, which its KNN entry recalls only the first \
                 token of. DELETE it and INSERT a single-token target",
                fact.id, fact.target
            ),
            FactStatus::Ambiguous => format!(
                "fact {} has more than one realization. DELETE it (which removes all of them) and \
                 INSERT it again",
                fact.id
            ),
            FactStatus::Active => unreachable!("active facts take both verbs"),
        };
        Some(verb.refused(why))
    }

    /// `UPDATE EDGES SET … WHERE entity = … AND relation = …`.
    pub(crate) fn exec_update_fact(
        &mut self,
        set: &UpdateSet,
        entity: &str,
        relation: &str,
    ) -> Result<Vec<String>, LqlError> {
        let fact = self.addressed_fact(Verb::Update, entity, relation)?;
        if let Some(refusal) = Self::status_refusal(Verb::Update, &fact) {
            return Err(refusal);
        }
        let (target, target_id) = match &set.target {
            Some(new_target) => {
                let ledger = self.ledger()?;
                let id = match ledger.tokenizer() {
                    Some(tok) => super::single_token_id(tok, new_target).ok_or_else(|| {
                        Verb::Update.refused(format!(
                            "the KNN target {new_target:?} is not a single token, and a KNN entry \
                             recalls one token"
                        ))
                    })?,
                    None => {
                        return Err(Verb::Update.refused(
                            "this container carries no tokenizer, so the new target cannot be \
                             tokenized"
                                .into(),
                        ))
                    }
                };
                (new_target.clone(), id)
            }
            None => (fact.target.clone(), fact.target_id.unwrap_or_default()),
        };
        let op = PatchOp::UpdateKnnFact {
            entity: fact.entity.clone(),
            relation: fact.relation.clone(),
            target: target.clone(),
            target_id,
            confidence: set.confidence,
        };

        let mut out = self.ensure_patch_session();
        self.advance_epoch();
        self.apply_fact_op(&op);
        self.record_knowledge_op(op);

        let layers = layer_list(&fact.layers);
        let mut change = Vec::new();
        if set.target.is_some() {
            change.push(format!("target {} → {target}", fact.target));
        }
        if let Some(c) = set.confidence {
            change.push(format!(
                "confidence {:.3} → {c:.3}",
                fact.confidence.unwrap_or_default()
            ));
        }
        out.push(format!(
            "Updated fact {} (v{} → v{}): {} (knn, {layers})",
            fact.id,
            fact.version,
            fact.version + 1,
            change.join(", ")
        ));
        Ok(out)
    }

    /// `DELETE FROM EDGES WHERE entity = … AND relation = …`.
    pub(crate) fn exec_delete_fact(
        &mut self,
        entity: &str,
        relation: &str,
    ) -> Result<Vec<String>, LqlError> {
        let fact = self.addressed_fact(Verb::Delete, entity, relation)?;
        if let Some(refusal) = Self::status_refusal(Verb::Delete, &fact) {
            return Err(refusal);
        }
        let op = PatchOp::DeleteKnnFact {
            entity: fact.entity.clone(),
            relation: fact.relation.clone(),
        };

        let mut out = self.ensure_patch_session();
        self.advance_epoch();
        self.apply_fact_op(&op);
        self.record_knowledge_op(op);

        out.push(format!(
            "Deleted fact {} (v{} → v{}): ({}, {}) → {} (knn, {})",
            fact.id,
            fact.version,
            fact.version + 1,
            fact.entity,
            fact.relation,
            fact.target,
            layer_list(&fact.layers)
        ));
        Ok(out)
    }

    /// Apply a fact op to the bound overlay's KNN store.
    fn apply_fact_op(&mut self, op: &PatchOp) {
        let store = match &mut self.backend {
            Backend::Vindex { patched, .. } => &mut patched.knn_store,
            Backend::Vindex3 { overlay, .. } => &mut overlay.knn_store,
            _ => unreachable!("ledger() refused every other backend"),
        };
        store.apply_fact_op(op);
    }
}

fn layer_list(layers: &[usize]) -> String {
    layers
        .iter()
        .map(|l| format!("L{l}"))
        .collect::<Vec<_>>()
        .join(",")
}
