//! **Identity construction is total, or it refuses.**
//!
//! [`read_source_identity`] used to walk `index.json` as untyped JSON
//! and take what it found:
//!
//! ```text
//! if let (Some(seg), Some(hash)) =
//!     (entry["segment"].as_str(), entry["payload_sha256"].as_str())
//! ```
//!
//! An entry missing either key was skipped, an index with no
//! `representations` key produced an identity sealing nothing, two
//! entries naming one segment collapsed to whichever was read last, and
//! an absent `system_graph` fell back to a filename the container never
//! declared. In every one of those cases the function returned `Ok` —
//! a confident identity over facts it had dropped.
//!
//! That is the wrong direction for an identity function:
//!
//! > **The thing computing identity must consume the same validated
//! > facts as the thing that consumes the container.** It may be
//! > stricter than that consumer. It must never be looser.
//!
//! The container parser refuses a malformed entry; the identity reader
//! quietly omitted it. These tests pin the asymmetry closed.
//!
//! What they deliberately do NOT test is any change to which VALID
//! containers identify as equal. Reformatting an index still moves the
//! identity — a false split, registered in [`super::source_seal`] and
//! left standing here, because removing it changes the equivalence
//! relation every piece of held evidence is keyed by and belongs under
//! its own identity version.

use super::super::super::compiler::read_source_identity;
use super::container;

/// The refusal every message from identity construction carries.
const REFUSED: &str = "source identity refused";

fn refusal(container: &std::path::Path) -> String {
    match read_source_identity(container) {
        Ok(identity) => panic!("expected a refusal, got {identity:?}"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn a_whole_container_identifies_from_its_own_declared_facts() {
    // The control. Every refusal below is only meaningful because this
    // container identifies without complaint, and seals every segment
    // its index declares.
    let container = container::glimmer();
    let identity = read_source_identity(container.path()).expect("a valid container");
    let index = container::read_index(container.path());
    let entries = index["representations"].as_object().expect("entries");

    assert_eq!(entries.len(), 8, "the fixture is not a single-entry one");
    assert_eq!(identity.segments().len(), entries.len());
    for entry in entries.values() {
        assert_eq!(
            identity
                .segments()
                .get(entry["segment"].as_str().expect("segment")),
            entry["payload_sha256"].as_str().as_ref(),
            "every declared segment is sealed by its own payload digest"
        );
    }
}

#[test]
fn deleting_one_entrys_required_fact_refuses_rather_than_identifying_the_rest() {
    // **The central test.** The old walk would have returned an
    // identity over the surviving seven, which is a different physical
    // reality wearing a confident digest — and evidence keyed by it
    // would be credited to a container nobody has.
    let container = container::glimmer();
    let index = container::read_index(container.path());
    let id = container::a_representation(&index);

    for field in ["payload_sha256", "segment_sha256", "segment"] {
        let container = container::glimmer();
        container::with_index(container.path(), |index| {
            index["representations"][&id]
                .as_object_mut()
                .expect("entry")
                .remove(field);
        });
        let refusal = refusal(container.path());
        assert!(refusal.contains(REFUSED), "{field}: {refusal}");
        assert!(
            refusal.contains(field),
            "the refusal must name the missing fact: {refusal}"
        );
    }
}

#[test]
fn an_empty_digest_is_a_missing_fact_wearing_a_present_field() {
    let container = container::glimmer();
    let index = container::read_index(container.path());
    let id = container::a_representation(&index);

    for field in ["payload_sha256", "segment_sha256", "segment"] {
        let container = container::glimmer();
        container::with_index(container.path(), |index| {
            index["representations"][&id][field] = serde_json::json!("");
        });
        let refusal = refusal(container.path());
        assert!(refusal.contains(REFUSED), "{field}: {refusal}");
        assert!(refusal.contains(field), "{refusal}");
        assert!(refusal.contains(&id), "and which entry: {refusal}");
    }
}

#[test]
fn an_index_declaring_no_representations_seals_nothing_and_is_refused() {
    let container = container::glimmer();
    container::with_index(container.path(), |index| {
        index["representations"] = serde_json::json!({});
    });

    // The old walk returned `Ok` here with an empty `segments` map: an
    // identity that pins a model's graph and its index text, and not one
    // byte of its weights.
    let refusal = refusal(container.path());
    assert!(refusal.contains(REFUSED), "{refusal}");
    assert!(refusal.contains("no representations"), "{refusal}");
}

#[test]
fn an_index_that_names_no_system_graph_is_refused_rather_than_assumed() {
    let container = container::glimmer();
    container::with_index(container.path(), |index| {
        index.as_object_mut().expect("index").remove("system_graph");
    });

    // `Vindex3Index::system_graph` is `Option` and its own documentation
    // says absence means "no graph recorded", never "the usual
    // filename". Hashing an assumed path would seal a document the
    // container never claimed as its authority.
    let refusal = refusal(container.path());
    assert!(refusal.contains(REFUSED), "{refusal}");
    assert!(refusal.contains("system_graph"), "{refusal}");
}

#[test]
fn an_index_naming_a_graph_file_that_is_not_there_is_refused() {
    // The graph is named and absent, which is not the same as unnamed:
    // b1 refuses the second because a filename would have to be
    // assumed, and this one because the authority the container DOES
    // name cannot be read. Neither may be identified around.
    let container = container::glimmer();
    let graph = container::read_index(container.path())["system_graph"]
        .as_str()
        .expect("the fixture records a graph")
        .to_string();
    std::fs::remove_file(container.path().join(&graph)).expect("remove the graph");
    assert!(read_source_identity(container.path()).is_err());
}

#[test]
fn a_malformed_entry_is_refused_by_the_schema_rather_than_stepped_over() {
    let container = container::glimmer();
    let index = container::read_index(container.path());
    let id = container::a_representation(&index);
    container::with_index(container.path(), |index| {
        index["representations"][&id]["payload_bytes"] = serde_json::json!("not a number");
    });

    // Parsing through `Vindex3Index` means the schema refuses this, and
    // identity construction inherits the refusal instead of maintaining
    // a second, laxer idea of what an entry is.
    let refusal = refusal(container.path());
    assert!(refusal.contains(REFUSED), "{refusal}");
    assert!(refusal.contains("does not parse"), "{refusal}");
}

#[test]
fn two_representations_may_share_a_segment_when_they_agree_about_it() {
    let container = container::glimmer();
    let index = container::read_index(container.path());
    let (first, second) = two_representations(&index);
    let shared = index["representations"][&first].clone();

    container::with_index(container.path(), |index| {
        // The second entry now names the first's segment and carries the
        // first's digests — the same file, described twice.
        for field in [
            "segment",
            "payload_sha256",
            "segment_sha256",
            "payload_bytes",
        ] {
            index["representations"][&second][field] = shared[field].clone();
        }
    });

    let identity = read_source_identity(container.path()).expect("agreement is not a conflict");
    assert_eq!(
        identity.segments().len(),
        7,
        "a shared segment appears once; sharing is deduplicated, not forbidden"
    );
}

#[test]
fn two_representations_naming_one_segment_and_disagreeing_are_refused() {
    let container = container::glimmer();
    let index = container::read_index(container.path());
    let (first, second) = two_representations(&index);
    let segment = index["representations"][&first]["segment"].clone();

    container::with_index(container.path(), |index| {
        // The same segment, and a different account of what is in it.
        // The old walk kept whichever it read last and reported nothing.
        index["representations"][&second]["segment"] = segment;
    });

    let refusal = refusal(container.path());
    assert!(refusal.contains(REFUSED), "{refusal}");
    assert!(refusal.contains("disagree"), "{refusal}");
}

/// Two distinct representation ids, in the index's own order.
fn two_representations(index: &serde_json::Value) -> (String, String) {
    let mut ids = index["representations"]
        .as_object()
        .expect("representations")
        .keys()
        .cloned();
    let first = ids.next().expect("one");
    let second = ids
        .next()
        .expect("two — the fixture must not be single-entry");
    (first, second)
}
