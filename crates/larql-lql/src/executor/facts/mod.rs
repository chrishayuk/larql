//! FACT-ID-1: the identity and lifecycle of asserted facts
//! (`docs/fact-id-1.md`).
//!
//! An asserted fact is one LQL put there: an INSERT, or a KNN entry the bound
//! container carries. It is addressed by its fact key, the normal form of
//! `(entity, relation)`. There is no fact store: [`fold::fold_ledger`] folds
//! the container's entries, the applied patches and the session journal into a
//! [`FactCatalogue`] every time one is needed, and checks it against the live
//! KNN store. Discovered rows (features the gate walk finds) have no identity
//! here and are never reached by a semantic mutation.

mod apply_check;
mod fold;
mod mutate;
mod select;
mod token;

use std::borrow::Cow;

use larql_inference::tokenizers::Tokenizer;
use larql_vindex::{FactKey, KnnStore, PatchOp, VindexPatch};

use crate::error::LqlError;
use crate::executor::{Backend, Session};

pub(crate) use fold::FactCatalogue;
pub(crate) use mutate::semantic_pair;
pub(crate) use select::with_asserted_block;
pub(crate) use token::single_token_id;

/// How an asserted fact is realised.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactRealization {
    /// Retrieval entries in the KNN store.
    Knn,
    /// A compose install: gate/up/down vectors at feature slots.
    Compose,
}

/// What can be claimed and done about an asserted fact
/// (`docs/fact-id-1.md` §3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactStatus {
    /// One KNN entry with a single-token target: INFER behaviour is claimed,
    /// and UPDATE and DELETE by identity are allowed.
    Active,
    /// One KNN entry whose target text needs more than one token.
    MultiToken,
    /// One KNN entry whose target text cannot be checked against its token
    /// id, because the binding has no tokenizer. Nothing is claimed about
    /// its INFER behaviour; DELETE is allowed, UPDATE is refused.
    Unverified,
    /// More than one live realization for the key (a legacy ledger).
    Ambiguous,
    /// The ledger and the live KNN store disagree for this key.
    Inconsistent,
    /// A keyed compose install: read-only by identity.
    Compose,
    /// A compose fact a later physical statement deleted or overwrote a slot of.
    Degraded,
}

/// Where an asserted fact's identity is kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactDurability {
    /// In the bound container's `knn_store.bin`.
    Container,
    /// In an applied patch file.
    Patch,
    /// Only in this session: lost on exit unless a patch saving it is applied.
    Session,
    /// Not derivable from the ledger (the fact is `Inconsistent`).
    Unknown,
}

/// One asserted fact, as a structured result. SELECT formats these; no code
/// recovers them from text.
#[derive(Debug, Clone, PartialEq)]
pub struct AssertedFact {
    /// `f_` and 16 hex digits, from the fact key alone.
    pub id: String,
    /// The entity as the latest op on the key spelt it.
    pub entity: String,
    /// The relation as the latest op on the key spelt it.
    pub relation: String,
    pub target: String,
    /// The target token id (KNN facts).
    pub target_id: Option<u32>,
    /// The stored confidence (KNN facts).
    pub confidence: Option<f32>,
    pub realization: FactRealization,
    pub status: FactStatus,
    /// Logical transitions the key has been through (a container entry is 1).
    pub version: u32,
    pub durability: FactDurability,
    /// `container`, the applied patch's name, `session` or `unknown`.
    pub origin: String,
    /// The layers the realization occupies.
    pub layers: Vec<usize>,
    /// The feature slots of a compose realization.
    pub slots: Vec<(usize, usize)>,
}

impl AssertedFact {
    /// The fact key.
    pub fn key(&self) -> FactKey {
        FactKey::new(&self.entity, &self.relation)
    }
}

impl FactRealization {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Knn => "knn",
            Self::Compose => "compose",
        }
    }
}

impl FactStatus {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::MultiToken => "multi_token",
            Self::Unverified => "unverified",
            Self::Ambiguous => "ambiguous",
            Self::Inconsistent => "inconsistent",
            Self::Compose => "compose",
            Self::Degraded => "degraded",
        }
    }
}

impl FactDurability {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Container => "container",
            Self::Patch => "patch",
            Self::Session => "session",
            Self::Unknown => "unknown",
        }
    }
}

/// The ledger of the bound overlay, borrowed.
pub(crate) struct LedgerView<'a> {
    base: &'a KnnStore,
    patches: &'a [VindexPatch],
    journal: &'a [(usize, PatchOp)],
    live: &'a KnnStore,
    /// Borrowed from a VINDEX3 binding; loaded from the vindex for VINDEX2,
    /// whose binding keeps none.
    tokenizer: Option<Cow<'a, Tokenizer>>,
}

impl LedgerView<'_> {
    /// Fold the ledger and check it against the live store.
    pub(crate) fn catalogue(&self) -> FactCatalogue {
        let mut catalogue = fold::fold_ledger(self.base, self.patches, self.journal);
        catalogue.check_against(self.live);
        catalogue
    }

    /// Whether `target` is the single token `target_id`, or `None` when the
    /// binding has no tokenizer to check it with. An unchecked entry is never
    /// classified as checked (`docs/fact-id-1.md` §3.4, `unverified`).
    pub(crate) fn is_single(&self, target: &str, target_id: u32) -> Option<bool> {
        self.tokenizer
            .as_deref()
            .map(|tok| single_token_id(tok, target) == Some(target_id))
    }

    pub(crate) fn tokenizer(&self) -> Option<&Tokenizer> {
        self.tokenizer.as_deref()
    }
}

impl Session {
    /// Whether the binding has a knowledge overlay, and so a fact ledger.
    /// Checks that guard fact identity skip only these bindings, which cannot
    /// apply a patch or insert a fact either; on an overlay binding a ledger
    /// that cannot be read is an error, never a pass.
    pub(crate) fn holds_facts(&self) -> bool {
        matches!(
            self.backend,
            Backend::Vindex { .. } | Backend::Vindex3 { .. }
        )
    }

    /// The ledger of the bound overlay. VINDEX2 and VINDEX3 bindings only.
    pub(crate) fn ledger(&self) -> Result<LedgerView<'_>, LqlError> {
        match &self.backend {
            Backend::Vindex { path, patched, .. } => Ok(LedgerView {
                base: patched.base_knn(),
                patches: &patched.patches,
                journal: patched.session_ops(),
                live: &patched.knn_store,
                tokenizer: larql_vindex::load_vindex_tokenizer(path)
                    .ok()
                    .map(Cow::Owned),
            }),
            Backend::Vindex3 {
                overlay, tokenizer, ..
            } => Ok(LedgerView {
                base: overlay.base_knn(),
                patches: &overlay.patches,
                journal: overlay.session_ops(),
                live: &overlay.knn_store,
                tokenizer: tokenizer.as_ref().map(Cow::Borrowed),
            }),
            Backend::None => Err(LqlError::NoBackend),
            _ => Err(LqlError::Execution(
                "asserted facts live in a vindex's knowledge overlay; this session is not bound \
                 to a VINDEX2 or VINDEX3 container"
                    .into(),
            )),
        }
    }

    /// Every live asserted fact of the bound overlay, ordered by fact key.
    pub fn asserted_facts(&self) -> Result<Vec<AssertedFact>, LqlError> {
        let ledger = self.ledger()?;
        let single = |t: &str, id: u32| ledger.is_single(t, id);
        Ok(ledger.catalogue().all_live(&single))
    }

    /// The COMPILE notice for asserted compose facts, which the compiled
    /// artifact keeps as vectors and metadata but not as facts.
    pub(crate) fn compose_identity_loss_line(&self) -> Option<String> {
        let facts = self.asserted_facts().ok()?;
        let compose = facts
            .iter()
            .filter(|f| f.realization == FactRealization::Compose)
            .count();
        (compose > 0).then(|| {
            format!(
                "  note: {compose} asserted compose fact(s) lose their identity in the output — it keeps their vectors, not their entity and relation, so they read back as discovered rows (KNN facts keep theirs in knn_store.bin)"
            )
        })
    }

    /// Record a knowledge op the session has just applied to its overlay: into
    /// the session journal always, and into the open recording when there is
    /// one. Every direct knowledge mutation records through here, so the
    /// journal and the recording cannot drift apart.
    pub(crate) fn record_knowledge_op(&mut self, op: PatchOp) {
        match &mut self.backend {
            Backend::Vindex { patched, .. } => patched.record_session_op(op.clone()),
            Backend::Vindex3 { overlay, .. } => overlay.record_session_op(op.clone()),
            _ => {}
        }
        if let Some(recording) = self.patch_recording.as_mut() {
            recording.operations.push(op);
        }
    }

    /// Refuse an INSERT that would break fact identity, before anything runs:
    /// a key that is already live, an id collision, or (KNN) a target that is
    /// not one token. Returns the version the new fact will have.
    pub(crate) fn check_fact_insert(
        &self,
        entity: &str,
        relation: &str,
        target: &str,
        knn: bool,
    ) -> Result<u32, LqlError> {
        // A binding with no knowledge overlay holds no facts; INSERT itself
        // says what it requires.
        if !self.holds_facts() {
            return Ok(1);
        }
        let ledger = self.ledger()?;
        let catalogue = ledger.catalogue();
        let key = FactKey::new(entity, relation);
        if catalogue.is_live(&key) {
            return Err(refused(format!(
                "the fact ({entity}, {relation}) already exists as {}. Change it with UPDATE EDGES \
                 SET target = … WHERE entity = … AND relation = …, or DELETE it first",
                key.fact_id()
            )));
        }
        if let Some(other) = catalogue.colliding_key(&key) {
            return Err(refused(collision(&key, &other)));
        }
        if knn {
            // Unchecked is not admitted: with no tokenizer the single-token
            // rule cannot be shown to hold.
            let Some(tok) = ledger.tokenizer() else {
                return Err(refused(format!(
                    "the KNN target {target:?} cannot be checked against the single-token rule: \
                     this binding has no tokenizer"
                )));
            };
            if single_token_id(tok, target).is_none() {
                return Err(refused(format!(
                    "the KNN target {target:?} is not a single token, and a KNN entry recalls one \
                     token. Use a single-token target"
                )));
            }
        }
        Ok(catalogue.version(&key) + 1)
    }
}

/// Why `key` cannot be asserted beside `other`, whose fact id it shares.
pub(crate) fn collision(key: &FactKey, other: &FactKey) -> String {
    format!(
        "the fact id {} of ({}, {}) collides with that of ({}, {})",
        key.fact_id(),
        key.entity(),
        key.relation(),
        other.entity(),
        other.relation()
    )
}

fn refused(reason: String) -> LqlError {
    LqlError::Execution(format!("INSERT refused: {reason}. Nothing was inserted."))
}
