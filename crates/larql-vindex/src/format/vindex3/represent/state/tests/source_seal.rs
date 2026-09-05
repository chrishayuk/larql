//! **The other half of the accounting seal: the segment's own table.**
//!
//! [`super`] pins the payload half — change a segment's payload digest
//! and the state id moves. This pins the half above it. The facts a
//! physical optimiser needs to price a decision the map PROTECTS live
//! in the segment header, not in the payload:
//!
//! ```text
//! SegmentTensor { name, dtype, shape, offset, len }
//! ```
//!
//! and `len` is the authority, not `shape × width(dtype)`. A packed,
//! padded or otherwise nontrivially stored tensor has a length the
//! naive product does not predict, which is the whole reason stage 4
//! refused to price a `Source` decision by multiplying by two.
//!
//! So the question these tests answer is whether the header reality is
//! sealed into [`RepresentationStateId`] the way the payload reality
//! is:
//!
//! ```text
//! segment header table
//!     ↓ sealed by       segment_sha256   (whole file, header included)
//!     ↓ recorded in     index.json
//!     ↓ declared in     a representation's authority
//!     ↓ carried by      SourceSemanticIdentity
//!     ↓                 RepresentationStateId
//! ```
//!
//! It is. These tests exist so it stays that way once 4b starts
//! trusting it — the invariant being:
//!
//! > **A `RepresentationStateId` must never be reusable across two
//! > physical-accounting realities that price the same effective
//! > decision vector differently.**
//!
//! Each test moves ONE field of the table and nothing else. The payload
//! bytes are copied verbatim, so `payload_sha256` cannot move, and any
//! change in the state id has to have travelled through the header.

use std::collections::BTreeMap;

use super::super::super::compiler::read_source_identity;
use super::container;
use super::container::{restate_table, state_id, Restated};

/// Assert the shape every one of these mutations must have.
fn only_the_header_moved(restated: &Restated) {
    assert_eq!(
        restated.before.segments(),
        restated.after.segments(),
        "no payload digest may move — the payload half must do none of the work here"
    );
    assert_eq!(
        restated.before.graph_hash(),
        restated.after.graph_hash(),
        "the semantic graph is untouched"
    );
    assert_ne!(
        restated.before.semantic, restated.after.semantic,
        "the index records the segment digest, so restating the table must move the \
         SEMANTIC identity, and not merely the bytes it was exported as"
    );
    assert_eq!(
        restated.index_changes,
        vec!["representations.target.embedding@BF16.segment_sha256"],
        "exactly one index field may move, and it is the segment digest"
    );

    let (before, decisions_before) = state_id(&restated.before);
    let (after, decisions_after) = state_id(&restated.after);
    assert_eq!(
        decisions_before, decisions_after,
        "the effective decision vector is byte-identical — that is what makes this adversarial"
    );
    assert_ne!(
        before, after,
        "two physical-accounting realities sharing a state id would let the search \
         reuse a price it never measured"
    );
}

#[test]
fn a_changed_source_dtype_moves_the_state_id_though_no_payload_byte_moves() {
    let container = container::dense();
    let restated = restate_table(container.path(), |header| {
        header.tensors[0].dtype = "FP16".into();
    });
    only_the_header_moved(&restated);
}

#[test]
fn a_changed_source_length_moves_the_state_id_though_no_payload_byte_moves() {
    // `len` and not `shape × width(dtype)` is the authority for a
    // source footprint. A packed or padded storage has a length the
    // naive product does not predict, so a seal blind to this field
    // would let two different physical realities share one price.
    let container = container::dense();
    let restated = restate_table(container.path(), |header| {
        header.tensors[0].len -= 1;
    });
    only_the_header_moved(&restated);
}

#[test]
fn restating_the_table_unchanged_moves_nothing() {
    // The control. Without it, "the id moved" could just mean the
    // harness rewrites the file differently from the writer, and both
    // tests above would pass on a seal that sealed nothing.
    let container = container::dense();
    let restated = restate_table(container.path(), |_| {});

    assert_eq!(restated.before, restated.after, "identity is content");
    assert_eq!(
        restated.index_changes,
        Vec::<String>::new(),
        "a faithful restatement rewrites the file byte for byte"
    );
    let (before, _) = state_id(&restated.before);
    let (after, _) = state_id(&restated.after);
    assert_eq!(before, after);
}

#[test]
fn the_segments_map_carries_the_payload_digest_and_not_the_file_digest() {
    // Which is why the header needed its own test at all: `segments`
    // seals the payload region, and the table sits outside it.
    let container = container::dense();
    let identity = read_source_identity(container.path()).expect("identity");
    let index = container::read_index(container.path());
    let entry = &index["representations"]["target.embedding@BF16"];

    let recorded: BTreeMap<&str, &str> = BTreeMap::from([(
        entry["segment"].as_str().expect("segment"),
        entry["payload_sha256"].as_str().expect("hash"),
    )]);
    assert_eq!(identity.segments(), recorded);
    assert_ne!(
        entry["payload_sha256"], entry["segment_sha256"],
        "the two digests cover different regions, which is the point"
    );
}
