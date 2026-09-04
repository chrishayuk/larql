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
//!     ↓ digested as     manifest_hash
//!     ↓ carried by      SourceIdentity
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
use std::io::Read;
use std::path::Path;

use sha2::{Digest, Sha256};

use crate::format::vindex3::encode::segment::{SegmentHeader, SEGMENT_PAYLOAD_ALIGN};

use super::super::super::compiler::{read_source_identity, SourceIdentity};
use super::super::super::map::PrecisionMap;
use super::super::super::policy::Role;
use super::super::identity::{RepresentationState, RepresentationStateId};
use super::super::resolved::NoLayoutConstraint;
use super::super::surface::{SurfaceTensor, TensorSurface};
use super::container;

/// A container's identity before and after a restatement of its table.
struct Restated {
    before: SourceIdentity,
    after: SourceIdentity,
    /// Every `index.json` field whose value changed, as a dotted path.
    index_changes: Vec<String>,
}

/// **Rewrite one segment's tensor table, copying the payload verbatim.**
///
/// Models an honest writer: the table changes, the file digest is
/// recomputed, and `index.json` records the new one. What it must NOT
/// do is disturb the payload — asserted here rather than assumed, since
/// a harness that quietly moved a payload byte would make every test
/// below pass for the wrong reason.
fn restate_table(root: &Path, edit: impl FnOnce(&mut SegmentHeader)) -> Restated {
    // Put the index into this harness's own serialisation BEFORE the
    // baseline is taken. `manifest_hash` digests the file's bytes, not
    // its content — see
    // `reserialising_the_index_alone_moves_the_state_id` — so without
    // this the control below would fail on pretty-printing and prove
    // nothing about the seal.
    container::reserialise(root);
    let before = read_source_identity(root).expect("a container identity");
    let index_before = container::read_index(root);

    let (id, entry) = index_before["representations"]
        .as_object()
        .expect("representations")
        .iter()
        .next()
        .expect("the fixture writes one representation");
    let segment_path = root.join(entry["segment"].as_str().expect("a segment path"));

    // Split the file into its three parts.
    let mut file = std::fs::File::open(&segment_path).expect("open segment");
    let mut length = [0u8; 8];
    file.read_exact(&mut length).expect("length prefix");
    let mut header_bytes = vec![0u8; u64::from_le_bytes(length) as usize];
    file.read_exact(&mut header_bytes).expect("header");
    let mut payload = Vec::new();
    file.read_to_end(&mut payload).expect("payload");
    let payload_before = format!("{:x}", Sha256::digest(&payload));

    let mut header: SegmentHeader = serde_json::from_slice(&header_bytes).expect("header is JSON");
    edit(&mut header);

    // Re-serialise under the writer's own padding rule, so a no-op edit
    // reproduces the file byte for byte and the control test means
    // something.
    let mut restated = serde_json::to_vec(&header).expect("serialise header");
    let pad = (SEGMENT_PAYLOAD_ALIGN - (8 + restated.len()) % SEGMENT_PAYLOAD_ALIGN)
        % SEGMENT_PAYLOAD_ALIGN;
    restated.extend(std::iter::repeat_n(b' ', pad));

    let mut written = Vec::new();
    written.extend_from_slice(&(restated.len() as u64).to_le_bytes());
    written.extend_from_slice(&restated);
    written.extend_from_slice(&payload);
    std::fs::write(&segment_path, &written).expect("rewrite segment");

    let payload_after = {
        let mut file = std::fs::File::open(&segment_path).expect("reopen");
        let mut length = [0u8; 8];
        file.read_exact(&mut length).expect("length");
        let mut skip = vec![0u8; u64::from_le_bytes(length) as usize];
        file.read_exact(&mut skip).expect("header");
        let mut payload = Vec::new();
        file.read_to_end(&mut payload).expect("payload");
        format!("{:x}", Sha256::digest(&payload))
    };
    assert_eq!(
        payload_before, payload_after,
        "the harness moved a payload byte; every assertion below would pass for the wrong reason"
    );

    let mut index_after = index_before.clone();
    index_after["representations"][id]["segment_sha256"] =
        serde_json::json!(format!("{:x}", Sha256::digest(&written)));
    container::write_index(root, &index_after);

    Restated {
        after: read_source_identity(root).expect("a container identity"),
        before,
        index_changes: changed_paths(&index_before, &index_after),
    }
}

/// Every leaf whose value differs between two JSON documents.
fn changed_paths(before: &serde_json::Value, after: &serde_json::Value) -> Vec<String> {
    let mut changes = Vec::new();
    walk(before, after, String::new(), &mut changes);
    changes.sort();
    changes
}

fn walk(a: &serde_json::Value, b: &serde_json::Value, path: String, out: &mut Vec<String>) {
    match (a, b) {
        (serde_json::Value::Object(a), serde_json::Value::Object(b)) => {
            for key in a
                .keys()
                .chain(b.keys())
                .collect::<std::collections::BTreeSet<_>>()
            {
                let child = match path.is_empty() {
                    true => key.clone(),
                    false => format!("{path}.{key}"),
                };
                match (a.get(key), b.get(key)) {
                    (Some(a), Some(b)) => walk(a, b, child, out),
                    _ => out.push(child),
                }
            }
        }
        _ if a != b => out.push(path),
        _ => {}
    }
}

/// The same map, surface and layout under two container identities.
fn state_id(model: &SourceIdentity) -> (RepresentationStateId, String) {
    let surface = TensorSurface::new([SurfaceTensor::new(
        "target.embedding",
        "weight",
        Role::Embedding,
        vec![128, 64],
    )])
    .expect("one tensor");
    let map = PrecisionMap {
        name: "m".into(),
        encoding: "BF16".into(),
        roles: vec!["embedding".into()],
        exceptions: Vec::new(),
    };
    let resolved = RepresentationState::resolve(model, &surface, &map, &NoLayoutConstraint);
    let decisions = resolved.decisions().canonical_full();
    (resolved.id().clone(), decisions)
}

/// Assert the shape every one of these mutations must have.
fn only_the_header_moved(restated: &Restated) {
    assert_eq!(
        restated.before.segments, restated.after.segments,
        "no payload digest may move — the payload half must do none of the work here"
    );
    assert_eq!(
        restated.before.graph_hash, restated.after.graph_hash,
        "the semantic graph is untouched"
    );
    assert_ne!(
        restated.before.manifest_hash, restated.after.manifest_hash,
        "the index records the segment digest, so restating the table must move it"
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

    let recorded: BTreeMap<String, String> = BTreeMap::from([(
        entry["segment"].as_str().expect("segment").to_string(),
        entry["payload_sha256"].as_str().expect("hash").to_string(),
    )]);
    assert_eq!(identity.segments, recorded);
    assert_ne!(
        entry["payload_sha256"], entry["segment_sha256"],
        "the two digests cover different regions, which is the point"
    );
}

#[test]
fn reserialising_the_index_alone_moves_the_state_id() {
    // **A registered hazard, not a desired property.**
    //
    // `manifest_hash` is `hash_bytes(index.json)` — the file's BYTES.
    // Two indices carrying identical values in a different
    // serialisation (a re-export, a pretty-print, a reordered key, any
    // tool that rewrites the file) therefore identify as different
    // models, and the same physical reality arrives as a new search
    // state with no evidence attached to it.
    //
    // That is stage 1a's SPLIT direction: the search re-measures what
    // it has already refused. It is the cheaper failure of the two —
    // merging would credit one state's evidence to another — but it is
    // a failure, and 4b binds physical accounting to this identity, so
    // it is pinned here rather than rediscovered later.
    let container = container::dense();
    let before = read_source_identity(container.path()).expect("identity");
    container::reserialise(container.path());
    let after = read_source_identity(container.path()).expect("identity");

    assert_eq!(
        before.segments, after.segments,
        "not one payload byte moved"
    );
    assert_eq!(before.graph_hash, after.graph_hash);
    assert_ne!(
        before.manifest_hash, after.manifest_hash,
        "and the model identity moved anyway"
    );
    assert_ne!(
        state_id(&before).0,
        state_id(&after).0,
        "so a re-exported container is a different search state"
    );
}
