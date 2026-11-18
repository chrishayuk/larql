//! Folding the ledger into the fact catalogue (`docs/fact-id-1.md` §3.3–3.4).
//!
//! The fold reads, in the order they took effect: the container's own KNN
//! entries, then the applied patches interleaved with the session journal
//! (each journal op carries how many patches were applied when it ran). It
//! mirrors exactly what each op does to the KNN store, then checks itself
//! against the live store: a key whose folded entries differ from the live
//! ones is `inconsistent`.

use std::collections::BTreeMap;

use larql_vindex::{FactKey, KnnStore, PatchOp, VindexPatch};

use super::{AssertedFact, FactDurability, FactRealization, FactStatus};

/// Where the fold is reading from.
#[derive(Debug, Clone)]
pub(super) enum Source {
    Container,
    Patch(String),
    Session,
}

impl Source {
    fn durability(&self) -> FactDurability {
        match self {
            Source::Container => FactDurability::Container,
            Source::Patch(_) => FactDurability::Patch,
            Source::Session => FactDurability::Session,
        }
    }

    fn origin(&self) -> String {
        match self {
            Source::Container => "container".into(),
            Source::Patch(name) => name.clone(),
            Source::Session => "session".into(),
        }
    }
}

/// One KNN entry as the fold sees it: what the store would hold.
#[derive(Debug, Clone, PartialEq)]
struct KnnRec {
    layer: usize,
    entity: String,
    target: String,
    target_id: u32,
    confidence: f32,
}

/// A keyed compose install.
#[derive(Debug, Clone, Default)]
struct ComposeRec {
    target: String,
    slots: Vec<(usize, usize)>,
    installs: u32,
    degraded: bool,
}

/// Everything the fold knows about one key.
#[derive(Debug, Clone)]
struct KeyState {
    entity: String,
    relation: String,
    knn: Vec<KnnRec>,
    compose: Option<ComposeRec>,
    version: u32,
    source: Source,
    inconsistent: bool,
}

impl KeyState {
    fn new(entity: &str, relation: &str, source: Source) -> Self {
        Self {
            entity: entity.to_string(),
            relation: relation.to_string(),
            knn: Vec::new(),
            compose: None,
            version: 0,
            source,
            inconsistent: false,
        }
    }

    fn is_live(&self) -> bool {
        !self.knn.is_empty() || self.compose.is_some()
    }

    fn touch(&mut self, entity: &str, relation: &str, source: &Source) {
        self.entity = entity.to_string();
        self.relation = relation.to_string();
        self.version += 1;
        self.source = source.clone();
    }
}

/// The folded ledger: every key it has seen, live or deleted.
#[derive(Debug, Clone, Default)]
pub(crate) struct FactCatalogue {
    states: BTreeMap<FactKey, KeyState>,
}

impl FactCatalogue {
    /// The live fact `key` names, as an [`AssertedFact`].
    pub(crate) fn live(
        &self,
        key: &FactKey,
        single: &dyn Fn(&str, u32) -> bool,
    ) -> Option<AssertedFact> {
        let state = self.states.get(key)?;
        state.is_live().then(|| to_fact(key, state, single))
    }

    /// Every live fact, ordered by key.
    pub(crate) fn all_live(&self, single: &dyn Fn(&str, u32) -> bool) -> Vec<AssertedFact> {
        self.states
            .iter()
            .filter(|(_, s)| s.is_live())
            .map(|(k, s)| to_fact(k, s, single))
            .collect()
    }

    /// The version `key` reached, live or deleted (0 if never seen).
    pub(crate) fn version(&self, key: &FactKey) -> u32 {
        self.states.get(key).map_or(0, |s| s.version)
    }

    /// Whether `key` is live (any status).
    pub(crate) fn is_live(&self, key: &FactKey) -> bool {
        self.states.get(key).is_some_and(KeyState::is_live)
    }

    /// Whether `key` is live as a KNN fact the store holds entries for.
    pub(crate) fn is_live_knn(&self, key: &FactKey) -> bool {
        self.states.get(key).is_some_and(|s| !s.knn.is_empty())
    }

    /// A different key the ledger has seen whose fact id equals `key`'s.
    pub(crate) fn colliding_key(&self, key: &FactKey) -> Option<FactKey> {
        let id = key.fact_id();
        self.states
            .keys()
            .find(|other| *other != key && other.fact_id() == id)
            .cloned()
    }

    /// Fold one op, read from `source`. `compose_run` is the key of the
    /// compose `Insert` just before this op in the same source, if any: the
    /// slots of one compose INSERT are one transition.
    pub(super) fn fold_op(
        &mut self,
        op: &PatchOp,
        source: &Source,
        compose_run: &mut Option<FactKey>,
    ) {
        let continuing = compose_run.take();
        match op {
            PatchOp::InsertKnn {
                layer,
                entity,
                relation,
                target,
                target_id,
                confidence,
                ..
            } => {
                let key = FactKey::new(entity, relation);
                let state = self.state(&key, entity, relation, source);
                state.knn.push(KnnRec {
                    layer: *layer,
                    entity: entity.clone(),
                    target: target.clone(),
                    target_id: *target_id,
                    confidence: confidence.unwrap_or(DEFAULT_REPLAY_CONFIDENCE),
                });
                state.touch(entity, relation, source);
            }
            PatchOp::UpdateKnnFact {
                entity,
                relation,
                target,
                target_id,
                confidence,
            } => {
                let key = FactKey::new(entity, relation);
                if let Some(state) = self.states.get_mut(&key).filter(|s| !s.knn.is_empty()) {
                    for rec in &mut state.knn {
                        rec.target = target.clone();
                        rec.target_id = *target_id;
                        if let Some(c) = confidence {
                            rec.confidence = *c;
                        }
                    }
                    state.touch(entity, relation, source);
                }
            }
            PatchOp::DeleteKnnFact { entity, relation } => {
                let key = FactKey::new(entity, relation);
                if let Some(state) = self.states.get_mut(&key).filter(|s| !s.knn.is_empty()) {
                    state.knn.clear();
                    state.touch(entity, relation, source);
                }
            }
            PatchOp::DeleteKnn { entity } => {
                // `KnnStore::remove_by_entity`: the raw entity, lowercased.
                let wanted = entity.to_lowercase();
                for state in self.states.values_mut() {
                    let before = state.knn.len();
                    state.knn.retain(|r| r.entity.to_lowercase() != wanted);
                    if state.knn.len() != before {
                        let (e, r) = (state.entity.clone(), state.relation.clone());
                        state.touch(&e, &r, source);
                    }
                }
            }
            PatchOp::Insert {
                layer,
                feature,
                entity,
                relation: Some(relation),
                target,
                ..
            } => {
                let slot = (*layer, *feature);
                let key = FactKey::new(entity, relation);
                self.degrade_slot_owners(slot, Some(&key));
                let same_install = continuing.as_ref() == Some(&key);
                let state = self.state(&key, entity, relation, source);
                let compose = state.compose.get_or_insert_with(ComposeRec::default);
                compose.target = target.clone();
                compose.slots.push(slot);
                if !same_install {
                    compose.installs += 1;
                    state.touch(entity, relation, source);
                }
                *compose_run = Some(key);
            }
            PatchOp::Insert { layer, feature, .. }
            | PatchOp::Update { layer, feature, .. }
            | PatchOp::Delete { layer, feature, .. } => {
                self.degrade_slot_owners((*layer, *feature), None);
            }
            PatchOp::WeightDelta(_)
            | PatchOp::StructuredWeight(_)
            | PatchOp::LowRankColumnsWeight(_) => {}
        }
    }

    fn state(
        &mut self,
        key: &FactKey,
        entity: &str,
        relation: &str,
        source: &Source,
    ) -> &mut KeyState {
        self.states
            .entry(key.clone())
            .or_insert_with(|| KeyState::new(entity, relation, source.clone()))
    }

    /// A physical op on `slot` degrades every compose fact that owns it,
    /// except `owner` (the fact installing it now).
    fn degrade_slot_owners(&mut self, slot: (usize, usize), owner: Option<&FactKey>) {
        for (key, state) in &mut self.states {
            if Some(key) == owner {
                continue;
            }
            if let Some(compose) = state.compose.as_mut() {
                if compose.slots.contains(&slot) {
                    compose.degraded = true;
                }
            }
        }
    }

    /// Mark every key whose folded entries differ from the live store's, and
    /// add live keys the ledger never produced.
    pub(super) fn check_against(&mut self, live: &KnnStore) {
        let mut live_by_key: BTreeMap<FactKey, Vec<KnnRec>> = BTreeMap::new();
        let mut spelling: BTreeMap<FactKey, (String, String)> = BTreeMap::new();
        for (layer, e) in live.ordered_entries() {
            let key = FactKey::new(&e.entity, &e.relation);
            spelling
                .entry(key.clone())
                .or_insert_with(|| (e.entity.clone(), e.relation.clone()));
            live_by_key.entry(key).or_default().push(KnnRec {
                layer,
                entity: e.entity.clone(),
                target: e.target_token.clone(),
                target_id: e.target_id,
                confidence: e.confidence,
            });
        }
        for (key, state) in &mut self.states {
            let live_recs = live_by_key.remove(key).unwrap_or_default();
            if !same_multiset(&state.knn, &live_recs) {
                state.inconsistent = true;
                // Show what executes; when nothing does, keep the ledger's
                // view so the divergence is listed rather than hidden.
                if !live_recs.is_empty() {
                    state.knn = live_recs;
                }
            }
        }
        for (key, recs) in live_by_key {
            let (entity, relation) = spelling.remove(&key).unwrap_or_default();
            let mut state = KeyState::new(&entity, &relation, Source::Container);
            state.knn = recs;
            state.inconsistent = true;
            self.states.insert(key, state);
        }
    }
}

/// `insert_knn` without a confidence replays as this (both overlays'
/// `confidence.unwrap_or(1.0)`).
const DEFAULT_REPLAY_CONFIDENCE: f32 = 1.0;

fn same_multiset(a: &[KnnRec], b: &[KnnRec]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut unmatched: Vec<&KnnRec> = b.iter().collect();
    for rec in a {
        match unmatched.iter().position(|other| *other == rec) {
            Some(i) => {
                unmatched.swap_remove(i);
            }
            None => return false,
        }
    }
    true
}

/// Fold the container entries, then the patches interleaved with the journal.
pub(super) fn fold_ledger(
    base: &KnnStore,
    patches: &[VindexPatch],
    journal: &[(usize, PatchOp)],
) -> FactCatalogue {
    let mut catalogue = FactCatalogue::default();
    let container = Source::Container;
    let mut seen_in_base: Vec<FactKey> = Vec::new();
    for (layer, e) in base.ordered_entries() {
        let key = FactKey::new(&e.entity, &e.relation);
        let state = catalogue.state(&key, &e.entity, &e.relation, &container);
        state.knn.push(KnnRec {
            layer,
            entity: e.entity.clone(),
            target: e.target_token.clone(),
            target_id: e.target_id,
            confidence: e.confidence,
        });
        if !seen_in_base.contains(&key) {
            seen_in_base.push(key);
            state.version = 1;
        }
    }

    let mut journal_ops = journal.iter().peekable();
    let session = Source::Session;
    for applied in 0..=patches.len() {
        let mut run = None;
        while let Some((_, op)) = journal_ops.next_if(|(stamp, _)| *stamp <= applied) {
            catalogue.fold_op(op, &session, &mut run);
        }
        if let Some(patch) = patches.get(applied) {
            let name = patch
                .description
                .clone()
                .unwrap_or_else(|| format!("patch #{}", applied + 1));
            let source = Source::Patch(name);
            let mut run = None;
            for op in &patch.operations {
                catalogue.fold_op(op, &source, &mut run);
            }
        }
    }
    // Anything stamped past the patch list (cannot happen while the overlay
    // clears its journal on removal) still folds, last.
    let mut run = None;
    for (_, op) in journal_ops {
        catalogue.fold_op(op, &session, &mut run);
    }
    catalogue
}

fn to_fact(key: &FactKey, state: &KeyState, single: &dyn Fn(&str, u32) -> bool) -> AssertedFact {
    let (realization, target, target_id, confidence, layers, slots) =
        match (&state.compose, state.knn.first()) {
            (Some(c), None) => (
                FactRealization::Compose,
                c.target.clone(),
                None,
                None,
                dedup_layers(c.slots.iter().map(|s| s.0)),
                c.slots.clone(),
            ),
            (_, Some(first)) => (
                FactRealization::Knn,
                first.target.clone(),
                Some(first.target_id),
                Some(first.confidence),
                dedup_layers(state.knn.iter().map(|r| r.layer)),
                Vec::new(),
            ),
            (None, None) => unreachable!("only live states become facts"),
        };
    let status = if state.inconsistent {
        FactStatus::Inconsistent
    } else if state.knn.len() > 1 || (state.compose.is_some() && !state.knn.is_empty()) {
        FactStatus::Ambiguous
    } else if let Some(c) = &state.compose {
        if c.degraded {
            FactStatus::Degraded
        } else if c.installs > 1 {
            FactStatus::Ambiguous
        } else {
            FactStatus::Compose
        }
    } else if !single(&target, target_id.unwrap_or_default()) {
        FactStatus::MultiToken
    } else {
        FactStatus::Active
    };
    let (durability, origin) = if state.inconsistent {
        (FactDurability::Unknown, "unknown".to_string())
    } else {
        (state.source.durability(), state.source.origin())
    };
    AssertedFact {
        id: key.fact_id(),
        entity: state.entity.clone(),
        relation: state.relation.clone(),
        target,
        target_id,
        confidence,
        realization,
        status,
        version: state.version.max(1),
        durability,
        origin,
        layers,
        slots,
    }
}

fn dedup_layers(layers: impl Iterator<Item = usize>) -> Vec<usize> {
    let mut v: Vec<usize> = layers.collect();
    v.sort_unstable();
    v.dedup();
    v
}
