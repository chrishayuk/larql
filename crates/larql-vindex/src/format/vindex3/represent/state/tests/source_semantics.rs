//! **The equivalence relation, stated on one real container.**
//!
//! [`super::source_seal`] proved the segment header table reaches the
//! state id. It proved it through `hash_bytes(index.json)`, which also
//! made a re-export of the same container a different search state —
//! a FALSE SPLIT, pinned there and removed here.
//!
//! Removing it is not a bug fix; it changes which containers are the
//! same state, so the whole relation is written out rather than the one
//! case that moved:
//!
//! ```text
//! presentation bytes changed only     SAME physical state
//! header/storage reality changed      DIFFERENT physical state
//! payload reality changed             DIFFERENT physical state
//! graph reality changed               DIFFERENT physical state
//! ```
//!
//! The centrepiece is three siblings of ONE encoded container:
//!
//! ```text
//! A  original
//! B  semantically identical index, differently serialised
//! C  identical index and payload, one segment-table `len` changed
//!
//! artifact(A) != artifact(B)          a re-export IS a different file
//! semantic(A) == semantic(B)          and the same source
//! state(A)    == state(B)
//! semantic(A) != semantic(C)          a different physical reality
//! state(A)    != state(C)
//! ```
//!
//! Both halves are load-bearing. Without B the relation is the old one;
//! without C the fix has severed the seal 4b-a just proved, which is
//! the specific way this change could go wrong.

use super::super::super::compiler::read_source_identity;
use super::super::super::source_identity::{
    CanonicalRepresentationAuthority, CATALOGUE_REMOVALS, ENTRY_OMISSIONS,
    SOURCE_SEMANTIC_ID_VERSION,
};
use super::container;
use super::container::{sibling, state_id};

// ------------------------------------------------------ the centrepiece

#[test]
fn three_siblings_of_one_container_state_the_whole_relation() {
    let a = container::dense();
    let b = sibling(a.path());
    let c = sibling(a.path());

    // B: the same values, written by a different serialiser.
    container::reserialise(b.path());
    // C: one tensor's stored length restated, `segment_sha256` updated
    // to match, payload copied verbatim.
    let restated = container::restate_table(c.path(), |header| header.tensors[0].len -= 1);

    let a = read_source_identity(a.path()).expect("identity");
    let b = read_source_identity(b.path()).expect("identity");
    let c = read_source_identity(c.path()).expect("identity");

    assert_ne!(
        a.artifact, b.artifact,
        "a re-export is a different FILE, and provenance should say so"
    );
    assert_eq!(
        a.semantic, b.semantic,
        "and the same SOURCE: not one value in the index changed"
    );
    assert_eq!(
        state_id(&a).0,
        state_id(&b).0,
        "so it is not a new physical search state, and arrives carrying its own evidence"
    );

    assert_ne!(
        a.semantic, c.semantic,
        "a restated segment header table is a different physical reality"
    );
    assert_ne!(state_id(&a).0, state_id(&c).0);

    // And C moved in exactly one place: the authority for the segment
    // whose table was restated. Without this the test would pass on any
    // change at all, including one that had thrown the projection away.
    assert_eq!(a.graph_hash(), c.graph_hash());
    assert_eq!(a.segments(), c.segments(), "no payload byte moved");
    assert_eq!(
        differing_authority_fields(&a.semantic.representations, &c.semantic.representations),
        vec!["segment_sha256".to_string()]
    );
    assert_eq!(
        restated.index_changes,
        vec!["representations.target.embedding@BF16.segment_sha256".to_string()],
        "and the index says the same thing"
    );
}

/// Which fields differ between two authority lists, as field names.
fn differing_authority_fields(
    before: &[CanonicalRepresentationAuthority],
    after: &[CanonicalRepresentationAuthority],
) -> Vec<String> {
    assert_eq!(before.len(), after.len(), "the same representations");
    let mut moved = Vec::new();
    for (before, after) in before.iter().zip(after) {
        let before = serde_json::to_value(before).expect("authority");
        let after = serde_json::to_value(after).expect("authority");
        for (field, value) in before.as_object().expect("object") {
            if after[field] != *value && !moved.contains(field) {
                moved.push(field.clone());
            }
        }
    }
    moved.sort();
    moved
}

// -------------------------------------------- the rest of the relation

#[test]
fn a_changed_payload_digest_is_a_different_state() {
    // The half `source_seal` calls the payload half, restated against
    // the semantic identity rather than the index's bytes.
    let container = container::dense();
    let before = read_source_identity(container.path()).expect("identity");
    container::with_index(container.path(), |index| {
        let entry = container::a_representation(index);
        index["representations"][&entry]["payload_sha256"] = serde_json::json!("f".repeat(64));
    });
    let after = read_source_identity(container.path()).expect("identity");

    assert_ne!(before.segments(), after.segments());
    assert_ne!(before.semantic, after.semantic);
    assert_ne!(state_id(&before).0, state_id(&after).0);
}

#[test]
fn a_changed_graph_is_a_different_state() {
    let container = container::dense();
    let before = read_source_identity(container.path()).expect("identity");
    let graph = container.path().join(
        container::read_index(container.path())["system_graph"]
            .as_str()
            .expect("the fixture records a graph"),
    );
    let mut document: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&graph).expect("graph")).expect("graph is JSON");
    document["larql_test_marker"] = serde_json::json!(1);
    std::fs::write(&graph, document.to_string()).expect("rewrite graph");

    let after = read_source_identity(container.path()).expect("identity");
    assert_eq!(
        before.segments(),
        after.segments(),
        "identical payloads under a different graph"
    );
    assert_ne!(before.graph_hash(), after.graph_hash());
    assert_ne!(state_id(&before).0, state_id(&after).0);
}

#[test]
fn a_catalogue_fact_no_authority_carries_still_moves_the_state() {
    // The tail the authorities do not cover, and the reason it is
    // sealed rather than enumerated: `num_layers` is a container-level
    // structural fact, and nothing per-representation would notice it.
    let container = container::dense();
    let before = read_source_identity(container.path()).expect("identity");
    container::with_index(container.path(), |index| {
        let layers = index["num_layers"].as_u64().expect("num_layers");
        index["num_layers"] = serde_json::json!(layers + 1);
    });
    let after = read_source_identity(container.path()).expect("identity");

    assert_eq!(
        before.semantic.representations,
        after.semantic.representations
    );
    assert_eq!(before.graph_hash(), after.graph_hash());
    assert_ne!(
        before.semantic.catalogue_hash,
        after.semantic.catalogue_hash
    );
    assert_ne!(state_id(&before).0, state_id(&after).0);
}

// ------------------------------------------------ associations, not sets

#[test]
fn swapping_two_authorities_moves_the_identity_though_the_multiset_does_not() {
    // **The trap an over-aggressive canonicalisation falls into.**
    // Hashing a sorted multiset of digests would be blind to this:
    // every digest the container declares is still declared, and two
    // representations have exchanged which segment file each one is
    // sealed by. That is a different model.
    //
    // `segment_sha256` and not `payload_sha256`, deliberately —
    // swapping payloads also moves the segment→payload map, so it would
    // be caught by a projection that had lost every association.
    let container = container::glimmer();
    let before = read_source_identity(container.path()).expect("identity");
    let index = container::read_index(container.path());
    let (first, second) = two_representations(&index);

    container::with_index(container.path(), |index| {
        let a = index["representations"][&first]["segment_sha256"].clone();
        let b = index["representations"][&second]["segment_sha256"].clone();
        assert_ne!(a, b, "two entries that already agree would prove nothing");
        index["representations"][&first]["segment_sha256"] = b;
        index["representations"][&second]["segment_sha256"] = a;
    });
    let after = read_source_identity(container.path()).expect("identity");

    assert_eq!(
        multiset(&before, |a| &a.segment_sha256),
        multiset(&after, |a| &a.segment_sha256),
        "not one digest was added or removed"
    );
    assert_eq!(before.segments(), after.segments(), "nor any payload");
    assert_ne!(
        before.semantic, after.semantic,
        "and the container is still a different one"
    );
    assert_ne!(state_id(&before).0, state_id(&after).0);
}

fn multiset<'a>(
    identity: &'a super::super::super::compiler::SourceIdentity,
    field: impl Fn(&'a CanonicalRepresentationAuthority) -> &'a String,
) -> Vec<&'a String> {
    let mut values: Vec<&String> = identity
        .semantic
        .representations
        .iter()
        .map(field)
        .collect();
    values.sort();
    values
}

fn two_representations(index: &serde_json::Value) -> (String, String) {
    let mut ids = index["representations"]
        .as_object()
        .expect("representations")
        .keys();
    let first = ids.next().expect("one").clone();
    (first, ids.next().expect("two").clone())
}

#[test]
fn a_codec_revision_is_a_fact_about_what_the_bytes_mean() {
    // The ONE per-entry field that is not a digest and is still
    // semantics: `encoding` names a family, the codec names the decode
    // contract. The same bytes under a revision a reader implements
    // differently are different bytes as far as the model is concerned,
    // which is why this is in and `encoder` — the encode recipe — is
    // not.
    let container = container::dense();
    let before = read_source_identity(container.path()).expect("identity");
    container::with_index(container.path(), |index| {
        let entry = container::a_representation(index);
        index["representations"][&entry]["codec"] = serde_json::json!({
            "family": "nvfp4",
            "revision": 2,
            "group_elems": 16,
            "element": "e2m1",
            "group_scale": "e4m3",
            "tensor_scale": "f32",
            "layout": "scales-then-codes",
        });
    });
    let after = read_source_identity(container.path()).expect("identity");

    assert!(before.semantic.representations[0].codec.is_none());
    assert!(after.semantic.representations[0].codec.is_some());
    assert_eq!(before.segments(), after.segments(), "no payload moved");
    assert_ne!(before.semantic, after.semantic);
    assert_ne!(state_id(&before).0, state_id(&after).0);
}

#[test]
fn the_order_the_profiles_are_declared_in_is_a_fact() {
    // A profile list is a sequence, and the canonical form keeps
    // sequences in order for the same reason it sorts object keys: one
    // of those is a value and the other is a spelling. A container that
    // offers the same profiles in a different order offers a different
    // default.
    let container = container::dense();
    let before = read_source_identity(container.path()).expect("identity");
    container::with_index(container.path(), |index| {
        let first = index["profiles"][0].clone();
        assert_eq!(
            index["profiles"].as_array().expect("profiles").len(),
            1,
            "the fixture declares one, and this test needs two"
        );
        index["profiles"] = serde_json::json!([first, {"name": "second-profile"}]);
    });
    let two = read_source_identity(container.path()).expect("identity");
    assert_ne!(before.semantic, two.semantic, "a profile was added");

    container::with_index(container.path(), |index| {
        let profiles = index["profiles"].as_array().expect("profiles").clone();
        index["profiles"] = serde_json::json!([profiles[1], profiles[0]]);
    });
    let swapped = read_source_identity(container.path()).expect("identity");
    assert_ne!(
        two.semantic, swapped.semantic,
        "the same two profiles, declared the other way round"
    );
    assert_ne!(state_id(&two).0, state_id(&swapped).0);
}

// ----------------------------------------------- the seal that had to move

#[test]
fn the_semantic_identity_carries_the_segment_file_digest() {
    // **The non-negotiable invariant of this change.** When the raw
    // index bytes left the identity, `segment_sha256` had to enter it
    // explicitly — it is the only seal over the segment header table,
    // and the table is where a physical optimiser reads the `dtype` and
    // `len` it prices a PROTECTED decision from. Asserted on the
    // canonical form itself, so a future projection that quietly stops
    // writing the field fails here and not three steps later.
    let container = container::dense();
    let identity = read_source_identity(container.path()).expect("identity");
    let canonical = identity.semantic.canonical();

    for authority in &identity.semantic.representations {
        assert!(!authority.segment_sha256.is_empty());
        assert!(
            canonical.contains(&authority.segment_sha256),
            "the canonical form must write `{}`",
            authority.segment_sha256
        );
        assert!(canonical.contains(&authority.payload_sha256));
        assert!(canonical.contains(&authority.representation));
    }
}

#[test]
fn the_canonical_semantic_form_is_versioned() {
    // v1 of the state id folded in the index's bytes; v2 does not, and
    // the two answer differently for containers that already exist. A
    // stored graph must be recognisably stale rather than silently
    // colliding.
    let identity = read_source_identity(container::dense().path()).expect("identity");
    assert!(
        identity
            .semantic
            .canonical()
            .starts_with(SOURCE_SEMANTIC_ID_VERSION),
        "{}",
        identity.semantic.canonical()
    );
    assert_eq!(
        super::super::identity::STATE_ID_VERSION,
        "represent-state-id/v2",
        "removing the false split changed the equivalence relation, so it changed the version"
    );
    assert_eq!(identity.semantic.digest().len(), 64, "sha256 hex");
}

// ------------------------------------------------- provenance, not identity

#[test]
fn a_provenance_only_change_moves_the_artifact_and_not_the_state() {
    // Every excluded fact, changed at once. Each is lineage or a
    // locator: what the bytes are is already sealed, and where they
    // came from does not change what loads or what it costs.
    let container = container::dense();
    let before = read_source_identity(container.path()).expect("identity");

    let graph = container::read_index(container.path())["system_graph"]
        .as_str()
        .expect("a graph")
        .to_string();
    std::fs::rename(
        container.path().join(&graph),
        container.path().join("renamed_graph.json"),
    )
    .expect("rename the graph file");

    container::with_index(container.path(), |index| {
        index["system_graph"] = serde_json::json!("renamed_graph.json");
        index["derived_from_model"] = serde_json::json!("some/other/model");
        let entry = container::a_representation(index);
        index["representations"][&entry]["compiled_from"] = serde_json::json!("an-earlier-pack");
        index["representations"][&entry]["source_representation_digest"] =
            serde_json::json!("d".repeat(64));
        index["representations"][&entry]["encoder"] =
            serde_json::json!({"algorithm": "nvfp4-gptq", "revision": 7});
    });
    let after = read_source_identity(container.path()).expect("identity");

    assert_ne!(
        before.artifact, after.artifact,
        "the file changed, and provenance is what records that"
    );
    assert_eq!(
        before.semantic, after.semantic,
        "and none of it is a fact about what this container IS"
    );
    assert_eq!(state_id(&before).0, state_id(&after).0);
}

#[test]
fn every_omission_names_a_field_the_index_actually_serialises() {
    // A removal keyed on a stale name is a silent no-op that returns
    // the fact to the seal; a field the authority stopped copying is a
    // silent MERGE. Both are invisible without this.
    let container = container::dense();
    container::with_index(container.path(), |index| {
        index["derived_from_model"] = serde_json::json!("some/other/model");
        let entry = container::a_representation(index);
        index["representations"][&entry]["compiled_from"] = serde_json::json!("an-earlier-pack");
        index["representations"][&entry]["source_representation_digest"] =
            serde_json::json!("d".repeat(64));
        index["representations"][&entry]["encoder"] =
            serde_json::json!({"algorithm": "nvfp4-nearest", "revision": 1});
    });
    // Round-tripped through the typed schema, so what is asserted is
    // what the projection actually sees and not what a hand-written
    // document happened to carry.
    let index = container::read_index(container.path());
    let typed: crate::format::vindex3::index::Vindex3Index =
        serde_json::from_value(index).expect("a valid index");
    let rendered = serde_json::to_value(&typed).expect("render");

    for field in CATALOGUE_REMOVALS {
        assert!(
            rendered.get(field).is_some(),
            "the projection removes `{field}`, which this index does not have"
        );
    }
    let entry = rendered["representations"]
        .as_object()
        .expect("representations")
        .values()
        .next()
        .expect("an entry");
    for field in ENTRY_OMISSIONS {
        assert!(
            entry.get(field).is_some(),
            "the authority omits `{field}`, which an entry does not have"
        );
    }
}
