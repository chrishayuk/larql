//! What the reference table promises: an address, shared or not, judged
//! before anything asks it a question.

use super::super::*;
use super::{address, reference, BOOKS, CODEBOOK, STACK};

#[test]
fn a_container_with_no_table_declares_no_dependency() {
    let table = ReferenceTable::empty();
    assert!(table.is_empty());
    assert_eq!(table.len(), 0);
    assert_eq!(
        table.target(&address(STACK, "0.mlp.down_proj.weight"), CODEBOOK),
        None
    );
    assert!(table
        .auxiliaries_of(&address(STACK, "0.mlp.down_proj.weight"))
        .is_empty());
}

/// The point of keying by `(owner, name)`: one target serves many owners,
/// which a per-tensor naming rule could not express.
#[test]
fn one_target_serves_many_owners_and_each_owner_names_it_itself() {
    let down = address(STACK, "0.mlp.down_proj.weight");
    let up = address(STACK, "0.mlp.up_proj.weight");
    let shared = address(BOOKS, "shared.codebook");
    let table = AuxiliaryReferences::new(vec![
        reference(
            (STACK, "0.mlp.down_proj.weight"),
            CODEBOOK,
            (BOOKS, "shared.codebook"),
        ),
        reference(
            (STACK, "0.mlp.up_proj.weight"),
            CODEBOOK,
            (BOOKS, "shared.codebook"),
        ),
    ])
    .judge()
    .unwrap();

    assert_eq!(table.target(&down, CODEBOOK), Some(&shared));
    assert_eq!(table.target(&up, CODEBOOK), Some(&shared));
    assert_eq!(table.len(), 2, "two references, one target");
    // And the sharing is readable from the other end, which is what the
    // accounting will ask: who depends on this object?
    let mut owners = table.owners_of(&shared);
    owners.sort();
    assert_eq!(owners, vec![&down, &up]);
    // A name the owner does not declare resolves to nothing rather than to
    // something plausible.
    assert_eq!(table.target(&down, "palette"), None);
}

#[test]
fn an_owner_may_declare_several_dependencies_by_name() {
    let owner = address(STACK, "0.mlp.down_proj.weight");
    let table = AuxiliaryReferences::new(vec![
        reference((STACK, "0.mlp.down_proj.weight"), CODEBOOK, (BOOKS, "a")),
        reference((STACK, "0.mlp.down_proj.weight"), "palette", (BOOKS, "b")),
    ])
    .judge()
    .unwrap();
    let declared = table.auxiliaries_of(&owner);
    assert_eq!(
        declared
            .iter()
            .map(|(name, target)| (*name, target.tensor.as_str()))
            .collect::<Vec<_>>(),
        vec![(CODEBOOK, "a"), ("palette", "b")],
        "in name order, so a refusal lists them the same way twice"
    );
}

#[test]
fn a_schema_another_build_wrote_is_refused_naming_both_versions() {
    let stored = AuxiliaryReferences {
        schema: AUXILIARY_REFERENCES_SCHEMA + 7,
        references: vec![reference((STACK, "w"), CODEBOOK, (BOOKS, "cb"))],
    };
    let err = stored.judge().unwrap_err().to_string();
    assert!(
        err.contains(&(AUXILIARY_REFERENCES_SCHEMA + 7).to_string())
            && err.contains(&AUXILIARY_REFERENCES_SCHEMA.to_string()),
        "{err}"
    );
}

#[test]
fn the_same_dependency_declared_twice_is_refused_rather_than_resolved() {
    let stored = AuxiliaryReferences::new(vec![
        reference((STACK, "w"), CODEBOOK, (BOOKS, "first")),
        reference((STACK, "w"), CODEBOOK, (BOOKS, "second")),
    ]);
    let err = stored.judge().unwrap_err().to_string();
    assert!(err.contains("first") && err.contains("second"), "{err}");
    assert!(err.contains(CODEBOOK), "{err}");
}

#[test]
fn an_operand_cannot_be_its_own_dependency() {
    let stored = AuxiliaryReferences::new(vec![reference(
        (STACK, "0.mlp.down_proj.weight"),
        CODEBOOK,
        (STACK, "0.mlp.down_proj.weight"),
    )]);
    let err = stored.judge().unwrap_err().to_string();
    assert!(err.contains("its own") && err.contains(CODEBOOK), "{err}");
}

#[test]
fn an_empty_address_or_name_is_refused_where_it_is_declared() {
    for (stored, expected) in [
        (
            AuxiliaryReferences::new(vec![reference(("", "w"), CODEBOOK, (BOOKS, "cb"))]),
            "owner",
        ),
        (
            AuxiliaryReferences::new(vec![reference((STACK, "w"), CODEBOOK, (BOOKS, "   "))]),
            "target",
        ),
        (
            AuxiliaryReferences::new(vec![reference((STACK, "w"), "  ", (BOOKS, "cb"))]),
            "empty auxiliary",
        ),
    ] {
        let err = stored.judge().unwrap_err().to_string();
        assert!(err.contains(expected), "{expected}: {err}");
    }
}

/// The table addresses; it does not describe. A row that carried shape or
/// dtype would be a second authority able to disagree with the segment.
#[test]
fn a_row_carries_an_address_and_nothing_the_segment_already_states() {
    let stored = AuxiliaryReferences::new(vec![reference(
        (STACK, "0.mlp.down_proj.weight"),
        CODEBOOK,
        (BOOKS, "shared.codebook"),
    )]);
    let json = serde_json::to_value(&stored).unwrap();
    let row = &json["references"][0];
    assert_eq!(
        row["owner"].as_object().unwrap().keys().collect::<Vec<_>>(),
        vec!["object", "tensor"]
    );
    assert!(row.get("shape").is_none() && row.get("dtype").is_none());
    // And it round-trips, including through a table that was judged first.
    let back: AuxiliaryReferences = serde_json::from_value(json).unwrap();
    assert_eq!(back, stored);
    assert_eq!(back.judge().unwrap().stored(), stored);
}

#[test]
fn a_table_is_read_from_the_root_the_index_names() {
    let dir = tempfile::tempdir().unwrap();
    let stored = AuxiliaryReferences::new(vec![reference(
        (STACK, "0.mlp.down_proj.weight"),
        CODEBOOK,
        (BOOKS, "shared.codebook"),
    )]);
    let name = crate::format::filenames::AUXILIARY_REFERENCES_JSON;
    std::fs::write(
        dir.path().join(name),
        serde_json::to_string_pretty(&stored).unwrap(),
    )
    .unwrap();
    let table = AuxiliaryReferences::read(dir.path(), name).unwrap();
    assert_eq!(
        table.target(&address(STACK, "0.mlp.down_proj.weight"), CODEBOOK),
        Some(&address(BOOKS, "shared.codebook"))
    );
    // A table the index names and the container does not hold is a
    // refusal, not an empty table: the index said it was there.
    let err = AuxiliaryReferences::read(dir.path(), "not_written.json")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("named by the index") && err.contains("not_written.json"),
        "{err}"
    );
    // Nor is unreadable content quietly ignored.
    std::fs::write(dir.path().join("broken.json"), "{ not json").unwrap();
    let err = AuxiliaryReferences::read(dir.path(), "broken.json")
        .unwrap_err()
        .to_string();
    assert!(err.contains("not readable as one"), "{err}");
}
