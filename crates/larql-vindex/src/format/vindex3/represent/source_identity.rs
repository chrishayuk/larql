//! **What a source container IS, separated from what it was exported
//! as.**
//!
//! Two digests over one container, and conflating them is what made a
//! reformatted `index.json` a different search state:
//!
//! ```text
//! index.json raw bytes
//!         → ContainerArtifactDigest      provenance; MAY move on re-export
//!
//! typed Vindex3Index
//! + graph authority (by content)
//! + representation → segment/header/payload associations
//!         → SourceSemanticIdentity      what the container IS
//!         → RepresentationStateId
//! ```
//!
//! The invariant this buys:
//!
//! > **Reformatting or key-reordering an index changes artifact
//! > provenance and does not create a new physical search state.**
//!
//! # Why the raw bytes could not stay
//!
//! `manifest_hash` was `hash_bytes(index.json)`. Two indices carrying
//! identical values in a different serialisation — a re-export, a
//! pretty-print, a reordered key, any tool that rewrites the file —
//! therefore identified as different models, and one physical reality
//! arrived as a new state with none of its own evidence attached. That
//! is stage 1a's SPLIT direction: the search re-measures what it has
//! already refused, and every alternative route to a map reads as
//! novel.
//!
//! # What replaced them, and the trap that had to be avoided
//!
//! The raw bytes were doing real work: `index.json` records each
//! representation's `segment_sha256`, which is what seals the segment
//! header table — the per-tensor `dtype` and `len` a physical optimiser
//! prices a PROTECTED decision from. Byte-hashing the index sealed that
//! table by accident. So when the bytes left, the digest had to enter
//! the semantic identity **explicitly**, or this change would have
//! severed exactly the seal `state::tests::source_seal` had just
//! proved.
//!
//! It is [`CanonicalRepresentationAuthority::segment_sha256`], and
//! `source_semantics::the_semantic_identity_carries_the_segment_file_digest`
//! is the test that keeps it there.
//!
//! # Associations, not a multiset of hashes
//!
//! The projection preserves which representation claims which digests.
//! Hashing a sorted multiset of `segment_sha256`s would be insufficient
//! for a reason that is easy to miss: swapping two entries' authorities
//! preserves the multiset and changes the model. Each authority is one
//! record binding an entry's identity to its own facts, and the records
//! are ordered by that identity rather than by input order.
//!
//! # What is excluded, and why each one
//!
//! An identity that omits a load-bearing fact MERGES two states and
//! credits one's evidence to the other, so the test applied to every
//! field is *does changing this change the model or the search
//! reality?* — never *is it present in `index.json`?*
//!
//! ```text
//! system_graph         a FILENAME. The graph is sealed by CONTENT, as
//!                      graph_hash, so two containers naming different
//!                      files with identical contents are one model.
//! derived_from_model   an operator's hint for finding the authority a
//!                      deployment image was cut from. A locator.
//! encoder              the encode RECIPE, not the decode contract. The
//!                      bytes it produced are already sealed by
//!                      payload_sha256; the index itself says a
//!                      mismatch is never a refusal.
//! compiled_from        lineage: which representation these derived
//! source_representation_digest
//!                      bytes came from. The bytes are sealed either
//!                      way, and a wrong lineage claim changes nothing
//!                      about what loads or what it costs.
//! ```
//!
//! Everything else the validated index carries stays in, including
//! fields this build does not understand ([`Vindex3Index::extra`]).
//! Excluding is the dangerous direction, so the tail is sealed rather
//! than enumerated — see [`CATALOGUE`].
//!
//! **A registered gap, unchanged by this step**: `moe_manifest` is a
//! filename whose contents nothing hashes. It was sealed by name under
//! the byte digest and it is sealed by name here, so there is no
//! regression — but a routed programme manifest is exactly the kind of
//! document that alters interpretation, and sealing it by content is a
//! separate change with its own refusal path.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::compile::hash_bytes;
use super::nvfp4_pack::CodecIdentity;
use super::state::surface::{FIELD, RECORD, SECTION};
use crate::error::VindexError;
use crate::format::filenames::INDEX_JSON;
use crate::format::vindex3::index::{RepresentationEntry, Vindex3Index};

/// The canonical-form version the semantic identity is computed under.
///
/// Introduced rather than folded into the existing state-id version on
/// its own: this identity is a projection with its own rules, and a
/// later change to *which facts count* must be visible without
/// implying that the enclosing state algorithm also changed.
pub const SOURCE_SEMANTIC_ID_VERSION: &str = "source-semantic-id/v1";

/// Every refusal from [`read_source_identity`] is prefixed, so a caller
/// reading a message knows the container was rejected as an IDENTITY
/// rather than as anything else.
const REFUSED: &str = "source identity refused";

/// Fields of the serialised [`Vindex3Index`] the catalogue projection
/// removes, each for a reason stated in the module doc.
///
/// Named as constants and asserted against a real serialised index by
/// `source_semantics::every_provenance_exclusion_names_a_field_the_index_actually_has`
/// — a removal keyed on a stale name is a silent no-op, which would
/// quietly return the excluded fact to the seal.
const INDEX_REPRESENTATIONS: &str = "representations";
const INDEX_SYSTEM_GRAPH: &str = "system_graph";
const INDEX_DERIVED_FROM_MODEL: &str = "derived_from_model";

/// Fields of a serialised [`RepresentationEntry`] the per-entry
/// authority does not carry. Left out STRUCTURALLY — the authority
/// copies the fields it seals rather than deleting the ones it does not
/// — so these exist only for the test that keeps the names honest.
#[cfg(any(test, feature = "test-utils"))]
const ENTRY_ENCODER: &str = "encoder";
#[cfg(any(test, feature = "test-utils"))]
const ENTRY_COMPILED_FROM: &str = "compiled_from";
#[cfg(any(test, feature = "test-utils"))]
const ENTRY_SOURCE_REPRESENTATION_DIGEST: &str = "source_representation_digest";

/// **The container's bytes on this disk, as exported.**
///
/// Provenance and nothing else. It answers "is this the same FILE?",
/// which is a real question — *this is a byte-different export of the
/// same semantic source* is a useful diagnostic — and it is not the
/// question identity asks. It must never reach
/// [`super::state::RepresentationStateId`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Serialize, Deserialize)]
pub struct ContainerArtifactDigest(String);

impl ContainerArtifactDigest {
    pub fn new(digest: impl Into<String>) -> Self {
        Self(digest.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// **One representation entry's authority, as facts rather than text.**
///
/// The unit the semantic identity is built from, and the unit 4b-c
/// reads its `SourceStorageFacts` out of: one validated authority
/// serving both identity and accounting is what stops a second physical
/// truth being authored for the optimiser.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CanonicalRepresentationAuthority {
    /// `{object_id}@{encoding}` — the index's own key, and the stable
    /// semantic key these records are ordered by. Not input order,
    /// which is a `BTreeMap` walk today and could stop being one.
    pub representation: String,
    pub object: String,
    pub encoding: String,
    /// Where the sealed artifact is, relative to the container root.
    ///
    /// In, because the loader resolves a representation to bytes
    /// through this directory and nothing else — §12.1 forbids
    /// composing a filename by sniffing. A container that puts the same
    /// bytes somewhere else is a container a reader reaches differently.
    pub segment: String,
    pub tensor_count: usize,
    pub payload_bytes: u64,
    /// SHA-256 over the payload region.
    pub payload_sha256: String,
    /// SHA-256 over the whole segment file, header table included.
    ///
    /// **The fact that had to enter explicitly when the raw index bytes
    /// left.** It is the only seal over `SegmentTensor{name, dtype,
    /// shape, offset, len}`, and `len` — not `shape × width(dtype)` —
    /// is the authority for a source footprint.
    pub segment_sha256: String,
    /// The decode ABI. The same bytes under a different revision mean
    /// something different, which is why this is semantics and
    /// `encoder` is not.
    pub codec: Option<CodecIdentity>,
}

impl CanonicalRepresentationAuthority {
    fn read(representation: &str, entry: &RepresentationEntry) -> Self {
        Self {
            representation: representation.to_string(),
            object: entry.object.clone(),
            encoding: entry.encoding.clone(),
            segment: entry.segment.clone(),
            tensor_count: entry.tensor_count,
            payload_bytes: entry.payload_bytes,
            payload_sha256: entry.payload_sha256.clone(),
            segment_sha256: entry.segment_sha256.clone(),
            codec: entry.codec.clone(),
        }
    }

    /// One record. Every field appears, so an authority that gains a
    /// field and forgets this line fails to compile rather than
    /// silently leaving it out of the seal.
    fn canonical(&self) -> String {
        let Self {
            representation,
            object,
            encoding,
            segment,
            tensor_count,
            payload_bytes,
            payload_sha256,
            segment_sha256,
            codec,
        } = self;
        let codec = match codec {
            Some(codec) => canonical_json(&serde_json::json!(codec)),
            None => String::new(),
        };
        [
            representation.as_str(),
            object.as_str(),
            encoding.as_str(),
            segment.as_str(),
            &tensor_count.to_string(),
            &payload_bytes.to_string(),
            payload_sha256.as_str(),
            segment_sha256.as_str(),
            &codec,
        ]
        .join(&FIELD.to_string())
    }
}

/// **What a source container IS.**
///
/// Three levels, because they can disagree independently and an overlay
/// depends on all three: payload bytes alone would miss a changed
/// semantic graph over identical bytes, and both together would miss a
/// changed segment header table.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SourceSemanticIdentity {
    /// Content digest of the system graph the index names — operand
    /// identities, shapes, roles.
    pub graph_hash: String,
    /// Every representation's authority, ordered by
    /// [`CanonicalRepresentationAuthority::representation`].
    pub representations: Vec<CanonicalRepresentationAuthority>,
    /// The rest of the validated index, sealed. See [`CATALOGUE`].
    pub catalogue_hash: String,
}

/// The container-level tail: the validated index minus the
/// representation entries — sealed above, one fact in one place — and
/// minus the named provenance exclusions.
///
/// A digest rather than a field list on purpose. A hand-copied list of
/// container facts silently drops whatever field the index gains next,
/// and dropping a fact is the MERGE direction. Projecting from the
/// typed document instead means a new field is IN by default and
/// leaving it out takes a deliberate, named, tested removal.
pub const CATALOGUE: &str = "catalogue";

impl SourceSemanticIdentity {
    /// The exact string the digest is taken over.
    ///
    /// One function, so the canonical form is stated in a single place
    /// a reader can check against the doc comment rather than implied
    /// by the order of calls at three call sites.
    pub fn canonical(&self) -> String {
        let representations = self
            .representations
            .iter()
            .map(CanonicalRepresentationAuthority::canonical)
            .collect::<Vec<_>>()
            .join(&RECORD.to_string());
        format!(
            "{SOURCE_SEMANTIC_ID_VERSION}{SECTION}graph={}{SECTION}{CATALOGUE}={}\
             {SECTION}representations={representations}",
            self.graph_hash, self.catalogue_hash
        )
    }

    pub fn digest(&self) -> String {
        hash_bytes(self.canonical().as_bytes())
    }

    /// Segment path → the source index's own `payload_sha256`.
    ///
    /// Derived, never stored: two representations may share a segment,
    /// and [`read_source_identity`] has already refused the case where
    /// they disagree about it.
    pub fn segments(&self) -> BTreeMap<&str, &str> {
        self.representations
            .iter()
            .map(|r| (r.segment.as_str(), r.payload_sha256.as_str()))
            .collect()
    }
}

/// A container's identity: what it is, and what it was exported as.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SourceIdentity {
    /// Provenance. Moves on a re-export that changes no value, and is
    /// not consulted by identity or by verification.
    pub artifact: ContainerArtifactDigest,
    /// Identity. What [`super::state::RepresentationStateId`] reads.
    pub semantic: SourceSemanticIdentity,
}

impl SourceIdentity {
    pub fn graph_hash(&self) -> &str {
        &self.semantic.graph_hash
    }

    /// Segment path → `payload_sha256`.
    pub fn segments(&self) -> BTreeMap<&str, &str> {
        self.semantic.segments()
    }

    pub fn semantic_digest(&self) -> String {
        self.semantic.digest()
    }

    /// **A synthetic identity, for fixtures.**
    ///
    /// Real identities come from [`read_source_identity`] and nowhere
    /// else. This exists so a test needing two identities that differ
    /// in exactly one named fact does not have to write out a whole
    /// container's authority to get one; the segments it is given
    /// become one authority apiece, and every field it is not given is
    /// left empty. An empty field is a fact this identity does not
    /// carry — including the catalogue, so two synthetics differ only
    /// in what the caller made differ.
    #[cfg(any(test, feature = "test-utils"))]
    pub fn synthetic(
        artifact: impl Into<String>,
        graph_hash: impl Into<String>,
        segments: impl IntoIterator<Item = (String, String)>,
    ) -> Self {
        Self {
            artifact: ContainerArtifactDigest::new(artifact),
            semantic: SourceSemanticIdentity {
                graph_hash: graph_hash.into(),
                representations: segments
                    .into_iter()
                    .map(
                        |(segment, payload_sha256)| CanonicalRepresentationAuthority {
                            representation: segment.clone(),
                            segment,
                            payload_sha256,
                            ..Default::default()
                        },
                    )
                    .collect(),
                catalogue_hash: String::new(),
            },
        }
    }
}

/// **Read a container's identity from its own metadata — totally.**
///
/// Parsed through [`Vindex3Index`], the same validated schema every
/// other reader of `index.json` uses, and not through a second informal
/// walk over `serde_json::Value`. The doctrine is worth stating:
///
/// > **The thing computing identity must consume the same validated
/// > facts as the thing that consumes the container.**
///
/// An identity function may be stricter than the consumer it identifies
/// and must never be looser. The earlier untyped walk was looser in
/// four ways, each of which produced a confident-looking identity over
/// facts it had silently dropped:
///
/// ```text
/// no `representations` key      an identity sealing NO payload at all
/// entry missing segment/digest  that segment quietly left out of the seal
/// two entries, one segment      last writer wins, the other discarded
/// no `system_graph`             a filename assumed, and hashed as authority
/// ```
///
/// So this refuses instead — and refusal stays UNDERNEATH the canonical
/// projection. Malformed input is never canonicalised into a confident
/// identity; it is refused before there is anything to canonicalise.
pub fn read_source_identity(container: &Path) -> Result<SourceIdentity, VindexError> {
    // Read as text and parsed with `from_str`, which is what all seven
    // other readers of this index do. One deserialiser for one document
    // format, and the bytes hashed as provenance are the ones parsed.
    let index_text = std::fs::read_to_string(container.join(INDEX_JSON))?;
    let index: Vindex3Index = serde_json::from_str(&index_text)
        .map_err(|e| VindexError::Parse(format!("{REFUSED}: {INDEX_JSON} does not parse: {e}")))?;

    // Absence means "no graph recorded", never "the usual filename".
    // Hashing an assumed path would seal a document the container never
    // claimed as its authority.
    let graph_name = index.system_graph.as_deref().ok_or_else(|| {
        VindexError::Parse(format!(
            "{REFUSED}: the index declares no `{INDEX_SYSTEM_GRAPH}`, and the semantic \
             graph is one of the three things a source identity seals"
        ))
    })?;
    let graph_bytes = std::fs::read(container.join(graph_name))?;

    if index.representations.is_empty() {
        return Err(VindexError::Parse(format!(
            "{REFUSED}: the index declares no representations, so this identity \
             would seal no payload bytes at all"
        )));
    }

    let mut representations = Vec::with_capacity(index.representations.len());
    // Kept beside the authorities so a second reference to one segment
    // is checked against the WHOLE of what the first claimed, not only
    // the half the identity happens to be reading at the time.
    let mut claimed: BTreeMap<&str, (&str, &str)> = BTreeMap::new();
    for (id, entry) in &index.representations {
        for (field, value) in [
            ("segment", entry.segment.as_str()),
            ("payload_sha256", entry.payload_sha256.as_str()),
            ("segment_sha256", entry.segment_sha256.as_str()),
        ] {
            if value.is_empty() {
                return Err(VindexError::Parse(format!(
                    "{REFUSED}: representation `{id}` carries an empty `{field}`, \
                     which is a missing fact wearing a present field"
                )));
            }
        }
        let facts = (entry.payload_sha256.as_str(), entry.segment_sha256.as_str());
        match claimed.insert(entry.segment.as_str(), facts) {
            // Two representations may legitimately share one segment;
            // what they may not do is disagree about what is in it.
            Some(held) if held != facts => {
                return Err(VindexError::Parse(format!(
                    "{REFUSED}: representation `{id}` and an earlier one both name \
                     segment `{}` and disagree about its digests",
                    entry.segment
                )))
            }
            _ => {}
        }
        representations.push(CanonicalRepresentationAuthority::read(id, entry));
    }

    Ok(SourceIdentity {
        artifact: ContainerArtifactDigest::new(hash_bytes(index_text.as_bytes())),
        semantic: SourceSemanticIdentity {
            graph_hash: hash_bytes(&graph_bytes),
            representations,
            catalogue_hash: catalogue_hash(&index)?,
        },
    })
}

/// Seal everything the index says that the authorities above do not.
///
/// Projected from the TYPED document — `Vindex3Index` has already
/// validated it and carries even the fields this build does not
/// understand — so what reaches the digest is values, never
/// serialisation. Formatting, whitespace and key order are gone before
/// this is called, because they never survive the parse.
fn catalogue_hash(index: &Vindex3Index) -> Result<String, VindexError> {
    let mut document = serde_json::to_value(index)
        .map_err(|e| VindexError::Parse(format!("{REFUSED}: the index does not re-render: {e}")))?;
    let object = document
        .as_object_mut()
        .ok_or_else(|| VindexError::Parse(format!("{REFUSED}: an index is a JSON object")))?;
    for field in [
        INDEX_REPRESENTATIONS,
        INDEX_SYSTEM_GRAPH,
        INDEX_DERIVED_FROM_MODEL,
    ] {
        object.remove(field);
    }
    Ok(hash_bytes(canonical_json(&document).as_bytes()))
}

/// **Deterministic JSON: object keys sorted, no insignificant
/// whitespace.**
///
/// This is an ENCODING, not the projection. What counts as identity was
/// already decided by the typed structures above; this only has to turn
/// a settled set of values into bytes the same way every time, and
/// sorting explicitly rather than trusting a map's iteration order is
/// what makes that true regardless of how `serde_json` is configured.
///
/// Array order is preserved: an array's order is one of its values.
fn canonical_json(value: &serde_json::Value) -> String {
    let mut out = String::new();
    write_canonical_json(value, &mut out);
    out
}

fn write_canonical_json(value: &serde_json::Value, out: &mut String) {
    match value {
        serde_json::Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            out.push('{');
            for (position, key) in keys.into_iter().enumerate() {
                if position > 0 {
                    out.push(',');
                }
                out.push_str(&serde_json::Value::String(key.clone()).to_string());
                out.push(':');
                write_canonical_json(&map[key], out);
            }
            out.push('}');
        }
        serde_json::Value::Array(items) => {
            out.push('[');
            for (position, item) in items.iter().enumerate() {
                if position > 0 {
                    out.push(',');
                }
                write_canonical_json(item, out);
            }
            out.push(']');
        }
        leaf => out.push_str(&leaf.to_string()),
    }
}

/// Every key [`catalogue_hash`] removes, exposed so a test can assert
/// each one names a field the index actually serialises. A removal
/// keyed on a stale name is a silent no-op, and the fact it was meant
/// to drop returns to the seal without anything failing.
#[cfg(any(test, feature = "test-utils"))]
pub const CATALOGUE_REMOVALS: [&str; 3] = [
    INDEX_REPRESENTATIONS,
    INDEX_SYSTEM_GRAPH,
    INDEX_DERIVED_FROM_MODEL,
];

/// The per-entry fields the authority does not copy, for the same test.
#[cfg(any(test, feature = "test-utils"))]
pub const ENTRY_OMISSIONS: [&str; 3] = [
    ENTRY_ENCODER,
    ENTRY_COMPILED_FROM,
    ENTRY_SOURCE_REPRESENTATION_DIGEST,
];
