//! Compaction executor: COMPACT MINOR, COMPACT MAJOR (Gate B, B6,
//! `docs/gate-b-b6.md` r3 §3.1 and §3.2).
//!
//! COMPACT MINOR promotes every KNN (L0) fact to a keyed compose install.
//! Each promotion is applied and recorded as `delete_knn_fact(K)` followed by
//! the keyed compose `Insert` for K, in that order, so the live state is the
//! one a replay of the recording produces. The statement is one transaction:
//! it refuses before any change if any fact cannot be promoted, and if a
//! promotion fails part-way everything it and the earlier promotions wrote is
//! put back.
//!
//! COMPACT MAJOR refuses before any work: a MEMIT cycle has no ledger
//! representation, so its result could not be replayed (ruling D2).

use larql_vindex::{FactKey, PatchOp};

use super::facts::{AssertedFact, FactRealization, FactStatus};
use super::install_fault::InstallFaultPoint;
use super::mutation::InsertSnapshot;
use super::{Backend, Session};
use crate::ast::InsertMode;
use crate::error::LqlError;

/// Why COMPACT MAJOR is refused (Gate B, B6 ruling D2).
pub(crate) const COMPACT_MAJOR_REFUSAL: &str = "COMPACT MAJOR is refused: a MEMIT cycle has no \
     ledger representation, so its result could not be replayed or reconstructed from a patch. \
     Restoring this statement needs a ledger-backed MEMIT operation under its own contract. \
     Nothing was changed.";

/// One planned promotion: a live KNN fact and the layer it occupies.
struct Promotion {
    layer: usize,
    key: FactKey,
    fact: AssertedFact,
}

/// What COMPACT MINOR can write, taken before it changes anything: the B5
/// install snapshot (overlay slots, session journal, caches, install
/// registry, recording length), widened to the KNN store, the recording
/// switch and the mutation counters.
struct CompactSnapshot {
    insert: InsertSnapshot,
    knn: larql_vindex::KnnStore,
    recording_open: bool,
    auto_patch: bool,
    epoch: u64,
    mutations_since_minor: usize,
    mutations_since_major: usize,
}

fn refused(reason: String) -> LqlError {
    LqlError::Execution(format!(
        "COMPACT MINOR refused, nothing was changed: {reason}"
    ))
}

fn describe(fact: &AssertedFact) -> String {
    format!(
        "fact {} ({}, {}) → {}",
        fact.id, fact.entity, fact.relation, fact.target
    )
}

impl Session {
    /// `COMPACT MINOR`: promote every L0 (KNN) fact to a keyed compose install.
    pub(crate) fn exec_compact_minor(&mut self) -> Result<Vec<String>, LqlError> {
        self.require_vindex()?;
        let plan = self.plan_minor_promotions()?;
        if plan.is_empty() {
            return Ok(vec![
                "COMPACT MINOR: L0 is empty, nothing to compact.".into()
            ]);
        }

        let snapshot = self.compact_snapshot();
        let mut out = self.ensure_patch_session();
        match self.promote_all(&plan) {
            Ok(lines) => {
                out.extend(lines);
                if let Some(note) = self.compose_identity_loss_line() {
                    out.push(note);
                }
                self.advance_epoch();
                Ok(out)
            }
            Err(e) => {
                self.restore_compact(snapshot);
                Err(LqlError::Execution(format!(
                    "COMPACT MINOR failed and was rolled back, nothing was changed: {e}"
                )))
            }
        }
    }

    /// The promotions COMPACT MINOR will run, in a defined order (layer,
    /// then fact key), or a refusal naming the first fact that cannot be
    /// promoted. Reads only.
    fn plan_minor_promotions(&self) -> Result<Vec<Promotion>, LqlError> {
        let ledger = self.ledger()?;
        let single = |t: &str, id: u32| ledger.is_single(t, id);
        let catalogue = ledger.catalogue();

        let (_, _, patched) = self.require_vindex()?;
        let mut keys: Vec<FactKey> = patched
            .knn_store
            .ordered_entries()
            .into_iter()
            .map(|(_, e)| FactKey::new(&e.entity, &e.relation))
            .collect();
        keys.sort();
        keys.dedup();

        let mut plan = Vec::with_capacity(keys.len());
        for key in keys {
            let Some(fact) = catalogue.live(&key, &single) else {
                return Err(refused(format!(
                    "the KNN store holds an entry for ({}, {}) that no live fact in the ledger \
                     accounts for",
                    key.entity(),
                    key.relation()
                )));
            };
            if fact.status != FactStatus::Active || fact.realization != FactRealization::Knn {
                return Err(refused(format!(
                    "{} is {} ({}); only an active KNN fact can be promoted",
                    describe(&fact),
                    fact.status.label(),
                    fact.realization.label()
                )));
            }
            let [layer] = fact.layers[..] else {
                return Err(refused(format!(
                    "{} occupies {} KNN layers; a promotion installs at one layer",
                    describe(&fact),
                    fact.layers.len()
                )));
            };
            plan.push(Promotion { layer, key, fact });
        }
        plan.sort_by(|a, b| (a.layer, &a.key).cmp(&(b.layer, &b.key)));
        Ok(plan)
    }

    /// Runs every promotion in order. On an error the caller restores.
    fn promote_all(&mut self, plan: &[Promotion]) -> Result<Vec<String>, LqlError> {
        self.inject_install_fault(InstallFaultPoint::CompactBeforePromotions)?;
        let mut out = vec![format!(
            "COMPACT MINOR: promoting {} fact(s) from KNN (L0) to compose (L1):",
            plan.len()
        )];
        for (i, p) in plan.iter().enumerate() {
            let promotion = i + 1;
            let fact = &p.fact;

            // Engine order equals recorded order: the KNN fact goes first,
            // so the install's capture never sees the old KNN override.
            let delete = PatchOp::DeleteKnnFact {
                entity: fact.entity.clone(),
                relation: fact.relation.clone(),
            };
            self.apply_fact_op(&delete);
            self.record_knowledge_op(delete);
            self.inject_install_fault(InstallFaultPoint::CompactAfterKnnRemoval { promotion })?;

            let version = self
                .check_fact_insert(&fact.entity, &fact.relation, &fact.target, false)
                .map_err(|e| LqlError::Execution(format!("{}: {e}", describe(fact))))?;
            self.exec_insert(
                &fact.entity,
                &fact.relation,
                &fact.target,
                Some(p.layer as u32),
                fact.confidence,
                None,
                InsertMode::Compose,
            )
            .map_err(|e| LqlError::Execution(format!("{}: {e}", describe(fact))))?;
            self.inject_install_fault(InstallFaultPoint::CompactAfterPromotion { promotion })?;

            out.push(format!(
                "  promoted {} at L{}: knn → compose (v{} → v{version})",
                describe(fact),
                p.layer,
                fact.version,
            ));
        }
        out.push(format!(
            "COMPACT MINOR complete: {} fact(s) promoted, recorded as delete_knn_fact then the \
             keyed compose insert for each.",
            plan.len()
        ));
        Ok(out)
    }

    fn compact_snapshot(&self) -> CompactSnapshot {
        let knn = match &self.backend {
            Backend::Vindex { patched, .. } => patched.knn_store.clone(),
            _ => unreachable!("COMPACT MINOR runs on a VINDEX2 binding only"),
        };
        CompactSnapshot {
            insert: self.insert_snapshot(true),
            knn,
            recording_open: self.patch_recording.is_some(),
            auto_patch: self.auto_patch,
            epoch: self.epoch,
            mutations_since_minor: self.mutations_since_minor,
            mutations_since_major: self.mutations_since_major,
        }
    }

    fn restore_compact(&mut self, snapshot: CompactSnapshot) {
        self.restore_insert(snapshot.insert);
        if let Backend::Vindex { patched, .. } = &mut self.backend {
            patched.knn_store = snapshot.knn;
        }
        if !snapshot.recording_open {
            self.patch_recording = None;
        }
        self.auto_patch = snapshot.auto_patch;
        self.epoch = snapshot.epoch;
        self.mutations_since_minor = snapshot.mutations_since_minor;
        self.mutations_since_major = snapshot.mutations_since_major;
    }

    /// `COMPACT MAJOR [FULL] [WITH LAMBDA = <f>]`: refused before any work
    /// (Gate B, B6 ruling D2). See [`COMPACT_MAJOR_REFUSAL`].
    pub(crate) fn exec_compact_major(
        &mut self,
        _full: bool,
        _lambda: Option<f32>,
    ) -> Result<Vec<String>, LqlError> {
        Err(LqlError::Execution(COMPACT_MAJOR_REFUSAL.into()))
    }
}
