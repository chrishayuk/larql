//! **What a tensor costs at source precision, read from the authority
//! that already identifies the container.**
//!
//! Stage 4 could not serve `next_experiment` because pricing a
//! PROTECTED decision needs the source storage facts and nothing held
//! them:
//!
//! ```text
//! declared   SearchSemantics.physical_accounting = "logical-bytes/v1"
//! held       a version id, which names a procedure and is not one
//! missing    a Footprint oracle the snapshot can name
//! ```
//!
//! The three test fixtures that stood in for one all multiplied by two,
//! which is bf16 asserted rather than read. This module is the
//! procedure that identifier names.
//!
//! # The invariant
//!
//! > **Accounting does not discover new source facts. It projects
//! > accounting facts from the same validated authority already used to
//! > establish source identity.**
//!
//! ```text
//! CanonicalRepresentationAuthority     segment, segment_sha256, tensor_count
//!         ↓ opens exactly that segment, recomputes exactly that digest
//! SegmentTensor { name, dtype, shape, offset, len }
//!         ↓
//! SourceStorageFact { logical_bytes = len, dtype }
//! ```
//!
//! **This is a dereference, not a second parse.** The per-tensor table
//! is not in `index.json`; it is inside the segment file, which is why
//! `segment_sha256` had to enter [`SourceSemanticIdentity`] explicitly
//! in 4b-b2. The authority names the file and seals its contents, and
//! this module opens that file and proves the bytes it read are the
//! sealed ones before reading a single number out of them. A reader
//! that skipped the check would be authoring a second physical truth
//! next to the one the state id is built on.
//!
//! # `len` is the byte count. `dtype` explains it.
//!
//! ```text
//! logical_bytes = the table's `len`          AUTHORITATIVE
//! dtype         = the table's `dtype`        EXPLANATORY
//! ```
//!
//! A packed, padded or otherwise nontrivially stored tensor has a length
//! `shape × width(dtype)` does not predict — the whole reason stage 4
//! refused to price a `Source` decision by multiplying by two, and the
//! reason 4b-a moved `dtype` and `len` in two SEPARATE adversarial
//! tests. So [`SourceDType`] is deliberately opaque: it has no `width`,
//! no `size_of`, and no conversion to a number. The moment one exists,
//! `numel × width` becomes reachable, and the regression this module
//! exists to prevent is one refactor away.
//!
//! # What this module does NOT do
//!
//! It does not know what a surface is. The facts describe what the
//! CONTAINER stores; the surface is what REPRESENT enumerated. Those are
//! different populations, and finding out where they disagree is 4b-d's
//! entire job — a `TensorSurfaceId` recorded here would be claiming a
//! completeness nothing had checked.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::super::compile::hash_bytes;
use super::super::source_identity::{CanonicalRepresentationAuthority, SourceIdentity};
use super::realization::LogicalBytes;
use super::surface::{FIELD, SECTION};
use crate::error::VindexError;
use crate::format::vindex3::encode::segment::read_segment_header;

/// The procedure `SearchSemantics.physical_accounting` names.
///
/// Deliberately the string stage 1d already declares, and not a tidier
/// one. `SearchSemanticsId` digests this among five normative version
/// ids, so renaming it would move every stored snapshot's semantics id
/// and announce a changed PROCEDURE — and the procedure did not change
/// here, it merely started existing.
pub const PHYSICAL_ACCOUNTING_PROCEDURE: &str = "logical-bytes/v1";

/// Every refusal from [`read_source_storage`] is prefixed.
const REFUSED: &str = "source storage refused";

/// **A declared storage dtype, and nothing computable.**
///
/// It says what the bytes ARE, so a reader can tell bf16 from Q6_K
/// without decoding. It cannot say how many bytes there are — that is
/// [`SourceStorageFact::logical_bytes`], read from the table — and this
/// type carries no accessor that would let it try.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct SourceDType(String);

impl SourceDType {
    pub fn new(dtype: impl Into<String>) -> Self {
        Self(dtype.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// **The optimizer's tensor identity: `(object, tensor)`.**
///
/// The pair `PlanRoles` is keyed by and `CompilationLedger` seals
/// against, named here because 4b-d has to hand back the exact
/// identities an accounting bind is missing, and a bare tuple in that
/// position is a pair of strings a caller has to guess the order of.
///
/// **Aliased objects stay two identities.** A tied embedding and output
/// head are one payload and two objects; REPRESENT compiles one pack per
/// object and a map resolves per `(object, tensor)`, so collapsing them
/// here would price a decision against an agreement the compiler does
/// not enforce.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct TensorIdentity {
    pub object: String,
    pub tensor: String,
}

impl TensorIdentity {
    pub fn new(object: impl Into<String>, tensor: impl Into<String>) -> Self {
        Self {
            object: object.into(),
            tensor: tensor.into(),
        }
    }
}

impl std::fmt::Display for TensorIdentity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.object, self.tensor)
    }
}

/// What one tensor occupies in the source container, and as what.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceStorageFact {
    /// The segment table's `len`. THE source byte count.
    pub logical_bytes: LogicalBytes,
    /// The segment table's `dtype`. Explanatory.
    pub dtype: SourceDType,
}

/// **What "physical accounting" MEANS here**, as declared facts.
///
/// Digested rather than described so that a change to the procedure
/// moves the id even when the version string does not — 1c's rule, and
/// the reason `InstrumentSemanticsId` digests meaning and never a
/// version.
///
/// 1c had to state that a declaration can lie and that the module could
/// not enforce otherwise. Here it can: [`read_source_storage`] stamps
/// [`Self::logical_bytes_v1`] and `PhysicalAccountingFacts` has no other
/// constructor, so the facts carry the meaning of the code that built
/// them rather than a claim made beside it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalAccountingSemantics {
    /// Where a tensor's source byte count comes from.
    pub source_byte_authority: String,
    /// How the table it is read from is proved to be the sealed one.
    pub seal_verification: String,
    /// What the declared dtype is permitted to do.
    pub dtype_role: String,
}

impl PhysicalAccountingSemantics {
    /// **The one procedure this build implements.**
    pub fn logical_bytes_v1() -> Self {
        Self {
            source_byte_authority: "segment-table-len".into(),
            seal_verification: "segment-sha256-recomputed".into(),
            dtype_role: "explanatory-never-multiplied".into(),
        }
    }

    pub fn id(&self) -> PhysicalAccountingSemanticsId {
        PhysicalAccountingSemanticsId(hash_bytes(self.canonical().as_bytes()))
    }

    fn canonical(&self) -> String {
        format!(
            "{PHYSICAL_ACCOUNTING_PROCEDURE}{SECTION}source_byte_authority={}{FIELD}\
             seal_verification={}{FIELD}dtype_role={}",
            self.source_byte_authority, self.seal_verification, self.dtype_role
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct PhysicalAccountingSemanticsId(String);

impl PhysicalAccountingSemanticsId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// **Every tensor the source container stores, priced.**
///
/// Bound to the source it was read from by digest: a footprint computed
/// against one model's storage must not be served for another, and
/// carrying the identity is what lets that be checked rather than
/// assumed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhysicalAccountingFacts {
    semantics: PhysicalAccountingSemanticsId,
    source: String,
    source_storage: BTreeMap<TensorIdentity, SourceStorageFact>,
}

impl PhysicalAccountingFacts {
    pub fn semantics(&self) -> &PhysicalAccountingSemanticsId {
        &self.semantics
    }

    pub fn get(&self, tensor: &TensorIdentity) -> Option<&SourceStorageFact> {
        self.source_storage.get(tensor)
    }

    pub fn tensors(&self) -> impl Iterator<Item = (&TensorIdentity, &SourceStorageFact)> {
        self.source_storage.iter()
    }

    pub fn len(&self) -> usize {
        self.source_storage.len()
    }

    pub fn is_empty(&self) -> bool {
        self.source_storage.is_empty()
    }

    /// Whether these facts describe `model`'s storage.
    ///
    /// The semantic digest and not the artifact digest, so a
    /// byte-different export of the same container is still the source
    /// these facts belong to — 4b-b2's relation, applied to accounting.
    pub fn describe(&self, model: &SourceIdentity) -> bool {
        self.source == model.semantic_digest()
    }
}

/// **Read every stored tensor's source facts out of the sealed
/// segments `identity` names.**
///
/// `identity` must be the identity of `container` — the segment digests
/// are recomputed against it, so passing another container's identity
/// refuses on the first segment rather than pricing the wrong bytes.
///
/// Refuses rather than omits, on b1's doctrine: a footprint built over
/// the tensors that happened to read cleanly would price a model that
/// does not exist.
pub fn read_source_storage(
    container: &Path,
    identity: &SourceIdentity,
) -> Result<PhysicalAccountingFacts, VindexError> {
    let mut source_storage: BTreeMap<TensorIdentity, SourceStorageFact> = BTreeMap::new();
    for authority in &identity.semantic.representations {
        let segment = container.join(&authority.segment);
        verify_seal(&segment, authority)?;

        // Parsed by the writer's own reader. A second header parser
        // here would be the untyped-walk mistake of 4b-b1, one level
        // down.
        let (header, _) = read_segment_header(&segment)?;
        if header.tensors.len() != authority.tensor_count {
            return Err(VindexError::Parse(format!(
                "{REFUSED}: representation `{}` declares {} tensors and its sealed table \
                 holds {} — two sealed facts about one segment disagree",
                authority.representation,
                authority.tensor_count,
                header.tensors.len()
            )));
        }

        for tensor in header.tensors {
            let id = TensorIdentity::new(&authority.object, tensor.name);
            let fact = SourceStorageFact {
                logical_bytes: LogicalBytes::new(tensor.len),
                dtype: SourceDType::new(tensor.dtype),
            };
            // Two representations of ONE object give one tensor two
            // prices, and which of them is "the source" is a rule this
            // module does not hold. `compiled_from` is the obvious
            // discriminator and adopting it is a decision with its own
            // evidence, so the collision is named rather than resolved.
            if let Some(held) = source_storage.insert(id.clone(), fact.clone()) {
                return Err(VindexError::Parse(format!(
                    "{REFUSED}: `{id}` is stored twice — {} B as {} and {} B as {}, the \
                     second by representation `{}`. Which one is the source is not a fact \
                     this procedure holds",
                    held.logical_bytes.get(),
                    held.dtype.as_str(),
                    fact.logical_bytes.get(),
                    fact.dtype.as_str(),
                    authority.representation
                )));
            }
        }
    }

    Ok(PhysicalAccountingFacts {
        semantics: PhysicalAccountingSemantics::logical_bytes_v1().id(),
        source: identity.semantic_digest(),
        source_storage,
    })
}

/// Prove the segment about to be read is the one the source identity
/// sealed.
///
/// This is what makes the read a dereference of an established fact
/// rather than the discovery of a new one. Without it, accounting would
/// be reading whatever is on disk under a path an index mentions, and a
/// state id would be sealing a table nobody checked was still there.
fn verify_seal(
    segment: &Path,
    authority: &CanonicalRepresentationAuthority,
) -> Result<(), VindexError> {
    let bytes = std::fs::read(segment).map_err(|e| {
        VindexError::Parse(format!(
            "{REFUSED}: representation `{}` names segment `{}`, which cannot be read: {e}",
            authority.representation,
            segment.display()
        ))
    })?;
    let actual = format!("{:x}", Sha256::digest(&bytes));
    if actual != authority.segment_sha256 {
        return Err(VindexError::Parse(format!(
            "{REFUSED}: segment `{}` hashes to {} and representation `{}` seals {} — the \
             table these bytes carry is not the one the source identity sealed",
            authority.segment,
            &actual[..12],
            authority.representation,
            &authority.segment_sha256[..authority.segment_sha256.len().min(12)]
        )));
    }
    Ok(())
}
