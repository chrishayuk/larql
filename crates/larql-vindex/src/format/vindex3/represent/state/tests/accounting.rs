//! **4b-c: the accounting facts, and the one arithmetic that must stay
//! unreachable.**
//!
//! 4b-a moved `dtype` and `len` in two SEPARATE adversarial tests
//! precisely so this file could assert their asymmetry:
//!
//! ```text
//! same shape, same dtype, changed len     → the price changes
//! same shape, changed dtype, same len     → the price does NOT change
//! ```
//!
//! The second is the load-bearing one. It permanently forecloses
//!
//! ```text
//! dtype → width → numel × width
//! ```
//!
//! which is the shape stage 4's three fixture footprints had, all three
//! multiplying by two and calling it bf16.

use super::super::super::compiler::read_source_identity;
use super::super::accounting::{
    read_source_storage, PhysicalAccountingSemantics, SourceDType, TensorIdentity,
    PHYSICAL_ACCOUNTING_PROCEDURE,
};
use sha2::Digest;

use super::container;

/// Read a container's storage facts through its own identity — the only
/// route this module offers, and the reason the two cannot drift.
fn facts(container: &std::path::Path) -> super::super::accounting::PhysicalAccountingFacts {
    let identity = read_source_identity(container).expect("a container identity");
    read_source_storage(container, &identity).expect("its storage facts")
}

fn refusal(container: &std::path::Path) -> String {
    let identity = read_source_identity(container).expect("a container identity");
    match read_source_storage(container, &identity) {
        Ok(facts) => panic!("expected a refusal, got {} tensors", facts.len()),
        Err(e) => e.to_string(),
    }
}

fn only_tensor(container: &std::path::Path) -> TensorIdentity {
    let facts = facts(container);
    let (id, _) = facts.tensors().next().expect("at least one tensor");
    id.clone()
}

// ---------------------------------------------- len is the byte count

#[test]
fn a_changed_stored_length_changes_the_price() {
    let container = container::dense();
    let before = facts(container.path());
    let tensor = only_tensor(container.path());
    let was = before.get(&tensor).expect("a fact").logical_bytes;

    container::restate_table(container.path(), |header| header.tensors[0].len -= 1);
    let after = facts(container.path());
    let now = after.get(&tensor).expect("a fact").logical_bytes;

    assert_eq!(now.get(), was.get() - 1, "the table's `len` is the price");
    assert_ne!(before, after);
}

#[test]
fn a_changed_dtype_does_not_change_the_price() {
    // **The regression guard.** Shape untouched, `len` untouched, only
    // the declared dtype moved — and the price is a number read from
    // the table, not a number computed from a type. Any implementation
    // that reaches `numel × width(dtype)` fails here, whatever it names
    // its helper.
    let container = container::dense();
    let tensor = only_tensor(container.path());
    let before = facts(container.path());
    let was = before.get(&tensor).expect("a fact").clone();
    assert_ne!(was.dtype, SourceDType::new("Q6_K"), "the fixture is BF16");

    container::restate_table(container.path(), |header| {
        header.tensors[0].dtype = "Q6_K".into();
    });
    let now = facts(container.path())
        .get(&tensor)
        .expect("a fact")
        .clone();

    assert_eq!(
        now.logical_bytes, was.logical_bytes,
        "a dtype is what the bytes ARE, never how many there are"
    );
    assert_eq!(now.dtype, SourceDType::new("Q6_K"), "and it is still read");
    assert_ne!(now, was, "the fact moved; only the price did not");
}

#[test]
fn the_shape_is_not_what_is_priced() {
    // The other half of the same guard, from the other side: a shape
    // that no longer matches the stored length changes nothing, because
    // the shape was never an input to the price.
    let container = container::dense();
    let tensor = only_tensor(container.path());
    let was = facts(container.path())
        .get(&tensor)
        .expect("a fact")
        .clone();

    let restated = container::restate_table(container.path(), |header| {
        header.tensors[0].shape = vec![1, 1];
    });
    assert_ne!(
        restated.before.semantic, restated.after.semantic,
        "the edit landed — without this the assertion below is vacuous"
    );
    assert_eq!(facts(container.path()).get(&tensor), Some(&was));
}

// ------------------------------------- the same authority, not a second one

#[test]
fn the_facts_come_from_the_segment_the_source_identity_seals() {
    // Every tensor priced traces to a representation whose
    // `segment_sha256` is in the semantic identity, and the count
    // matches what those authorities declare. Accounting reads the
    // sealed table and enumerates nothing of its own.
    let container = container::glimmer();
    let identity = read_source_identity(container.path()).expect("identity");
    let facts = read_source_storage(container.path(), &identity).expect("facts");

    let declared: usize = identity
        .semantic
        .representations
        .iter()
        .map(|a| a.tensor_count)
        .sum();
    assert!(declared > 0, "the fixture stores tensors");
    assert_eq!(facts.len(), declared, "one fact per declared tensor");

    let objects: std::collections::BTreeSet<&str> = identity
        .semantic
        .representations
        .iter()
        .map(|a| a.object.as_str())
        .collect();
    for (tensor, fact) in facts.tensors() {
        assert!(
            objects.contains(tensor.object.as_str()),
            "{tensor} names an object no authority declares"
        );
        assert!(!fact.dtype.as_str().is_empty());
    }

    assert!(facts.describe(&identity), "bound to the source it read");
    assert!(!facts.is_empty());
}

#[test]
fn a_segment_that_is_not_the_sealed_one_refuses_rather_than_pricing_it() {
    // **What makes this a dereference and not a second parse.** The
    // table is restated and the index is NOT told, so the bytes on disk
    // are no longer the bytes the identity sealed. Pricing them would
    // author a physical truth next to the one the state id is built on.
    let container = container::dense();
    let identity = read_source_identity(container.path()).expect("identity");
    let segment = container
        .path()
        .join(&identity.semantic.representations[0].segment);
    let mut bytes = std::fs::read(&segment).expect("segment");
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    std::fs::write(&segment, &bytes).expect("rewrite");

    let err = read_source_storage(container.path(), &identity).expect_err("must refuse");
    assert!(
        format!("{err}").contains("not the one the source identity sealed"),
        "{err}"
    );
}

#[test]
fn another_containers_identity_prices_nothing() {
    // The identity is an input, so it can be the wrong one. A sibling
    // of the same container names every segment at the same path, so
    // nothing but the seal can tell them apart — which is the point.
    // Two containers with different layouts would refuse on a missing
    // file and prove nothing about the check under test.
    let a = container::dense();
    let b = container::sibling(a.path());
    container::restate_table(b.path(), |header| header.tensors[0].len -= 1);
    let foreign = read_source_identity(b.path()).expect("identity");

    let own = read_source_identity(a.path()).expect("identity");
    assert_eq!(
        foreign.semantic.representations[0].segment, own.semantic.representations[0].segment,
        "same segment path, so only the digest separates them"
    );
    let err = read_source_storage(a.path(), &foreign).expect_err("must refuse");
    assert!(
        format!("{err}").contains("not the one the source identity sealed"),
        "{err}"
    );
}

#[test]
fn a_missing_segment_refuses_rather_than_synthesising_a_price() {
    let container = container::dense();
    let identity = read_source_identity(container.path()).expect("identity");
    std::fs::remove_file(
        container
            .path()
            .join(&identity.semantic.representations[0].segment),
    )
    .expect("remove the segment");
    let err = read_source_storage(container.path(), &identity).expect_err("must refuse");
    assert!(format!("{err}").contains("cannot be read"), "{err}");
}

#[test]
fn a_table_that_contradicts_its_own_declared_count_is_refused() {
    // `tensor_count` and the table are both sealed facts about one
    // segment. When they disagree the writer lied, and taking either
    // one would be choosing which lie to price.
    let container = container::dense();
    container::restate_table(container.path(), |header| {
        let extra = header.tensors[0].clone();
        header
            .tensors
            .push(super::super::super::super::encode::segment::SegmentTensor {
                name: format!("{}.duplicate", extra.name),
                ..extra
            });
    });
    let err = refusal(container.path());
    assert!(
        err.contains("two sealed facts about one segment disagree"),
        "{err}"
    );
}

#[test]
fn one_object_stored_twice_is_refused_rather_than_last_writer_wins() {
    // A container may hold a source pack and a compiled pack for ONE
    // object. Both declare the same `(object, tensor)` pairs at
    // different lengths, and picking one is a rule this procedure does
    // not hold — `compiled_from` is the obvious discriminator and
    // adopting it is a decision with its own evidence. Silently keeping
    // the last would price a model nobody built.
    //
    // Declared by pointing a second representation at the SAME segment,
    // which b1 permits precisely because the two agree about it.
    let container = container::dense();
    container::with_index(container.path(), |index| {
        let id = container::a_representation(index);
        let entry = index["representations"][&id].clone();
        index["representations"]["target.embedding@COPY"] = entry;
    });
    let err = refusal(container.path());
    assert!(err.contains("is stored twice"), "{err}");
    assert!(err.contains("not a fact this procedure holds"), "{err}");
}

#[test]
fn a_reserialised_container_is_still_the_source_these_facts_describe() {
    // 4b-b2's relation, applied to accounting: `describe` resolves on
    // the semantic digest, so re-exporting the index does not orphan a
    // footprint already computed against the container.
    let container = container::dense();
    let before = read_source_identity(container.path()).expect("identity");
    let facts = read_source_storage(container.path(), &before).expect("facts");

    container::reserialise(container.path());
    let after = read_source_identity(container.path()).expect("identity");
    assert_ne!(before.artifact, after.artifact, "a different file");
    assert!(
        facts.describe(&after),
        "and the same source, so the facts still belong to it"
    );

    // Where they stop belonging is a changed table, which is what
    // 4b-a proved moves the state id.
    container::restate_table(container.path(), |header| header.tensors[0].len -= 1);
    let restated = read_source_identity(container.path()).expect("identity");
    assert!(!facts.describe(&restated));
}

#[test]
fn a_sealed_segment_whose_header_does_not_parse_is_refused() {
    // The seal proves the bytes are the ones the identity sealed. It
    // does not prove they are a table — a writer can seal garbage —
    // so the parse is still allowed to refuse, and the refusal names
    // the file rather than being swallowed into an empty price.
    let container = container::dense();
    let identity = read_source_identity(container.path()).expect("identity");
    let representation = identity.semantic.representations[0].clone();
    let segment = container.path().join(&representation.segment);

    let mut bytes = std::fs::read(&segment).expect("segment");
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("prefix")) as usize;
    bytes[8..8 + header_len].fill(b'?');
    std::fs::write(&segment, &bytes).expect("rewrite");

    // Re-seal it, so the check under test is the parse and not the
    // digest — otherwise this test passes on the previous guard.
    container::with_index(container.path(), |index| {
        let id = container::a_representation(index);
        index["representations"][&id]["segment_sha256"] =
            serde_json::json!(format!("{:x}", sha2::Sha256::digest(&bytes)));
    });
    let resealed = read_source_identity(container.path()).expect("identity");
    let err = read_source_storage(container.path(), &resealed).expect_err("must refuse");
    assert!(format!("{err}").contains("segment header"), "{err}");
}

#[test]
fn the_facts_survive_a_round_trip_through_json() {
    // **They did not.** `source_storage` was a `BTreeMap` keyed by a
    // STRUCT, which derives `Serialize` happily and then fails at
    // runtime — "key must be a string" — the moment it holds anything.
    // The snapshot is written as JSON, so a record carrying accounting
    // authority could not be stored at all, and nothing here noticed
    // because every test read the facts in memory.
    let container = container::glimmer();
    let before = facts(container.path());
    assert!(before.len() > 1);

    let text = serde_json::to_string(&before).expect("accounting facts must serialise");
    let after: super::super::accounting::PhysicalAccountingFacts =
        serde_json::from_str(&text).expect("and reload");
    assert_eq!(before, after);

    // Through `Value` as well as `from_str`: the borrowed-string bug in
    // `Role::deserialize` passed one and failed the other.
    let value = serde_json::to_value(&before).expect("to value");
    let from_value: super::super::accounting::PhysicalAccountingFacts =
        serde_json::from_value(value).expect("from value");
    assert_eq!(before, from_value);
}

// ------------------------------------------------------------- aliasing

#[test]
fn two_objects_sharing_a_tensor_name_are_two_facts() {
    // `(object, tensor)` is the identity, so an alias — a tied
    // embedding and output head, one payload and two objects — is two
    // facts and not one. Keying on the tensor name alone would price
    // one of them and silently drop the other.
    let container = container::glimmer();
    let facts = facts(container.path());
    let mut names: Vec<&str> = facts.tensors().map(|(t, _)| t.tensor.as_str()).collect();
    names.sort_unstable();
    let shared = names.windows(2).find(|w| w[0] == w[1]);

    match shared {
        Some(w) => {
            let objects: Vec<&str> = facts
                .tensors()
                .filter(|(t, _)| t.tensor == w[0])
                .map(|(t, _)| t.object.as_str())
                .collect();
            assert!(objects.len() > 1, "one name, several objects");
            assert_eq!(
                objects
                    .iter()
                    .collect::<std::collections::BTreeSet<_>>()
                    .len(),
                objects.len(),
                "and each object appears once"
            );
        }
        None => panic!(
            "the fixture no longer shares a tensor name across objects, so this test \
             asserts nothing — pick a fixture that does"
        ),
    }
}

// --------------------------------------------- the declaration is the code

#[test]
fn the_declared_procedure_is_the_one_that_runs() {
    // Stage 4 refused `next_experiment` because
    // `SearchSemantics.physical_accounting` named a procedure that did
    // not exist. It exists now, and the two names are checked against
    // each other rather than merely agreeing by habit.
    let semantics = super::super::fixtures::semantics();
    assert_eq!(
        semantics.physical_accounting, PHYSICAL_ACCOUNTING_PROCEDURE,
        "the snapshot declares the procedure this module implements"
    );

    // And the id is over the MEANING, so a changed procedure moves it
    // with the version string standing still.
    let declared = PhysicalAccountingSemantics::logical_bytes_v1();
    assert_eq!(
        facts(container::dense().path()).semantics(),
        &declared.id(),
        "the facts carry the meaning of the code that built them"
    );
    let mut multiplied = declared.clone();
    multiplied.dtype_role = "width-times-numel".into();
    assert_ne!(declared.id(), multiplied.id());
    assert_eq!(declared.id().as_str().len(), 64, "sha256 hex");
}
