//! The overlay's metadata edits reach a written `down_meta.bin`.
//!
//! `INSERT`, `UPDATE` and `DELETE` change metadata in the patch overlay
//! only. A compile that copied the source's `down_meta.bin` dropped all
//! three. These gates pin the vindex half of the fix:
//!
//! - `save_effective_down_meta` writes what the patched session shows,
//!   one record per base feature, with a deleted slot empty;
//! - an edit the file cannot hold is reported, not written wrongly.

mod common;

use common::{build_synthetic_index, tokenizer, FIXTURE_FEATURES, FIXTURE_LAYERS};
use larql_models::TopKEntry;
use larql_vindex::format::down_meta;
use larql_vindex::{FeatureMeta, LabelStoredAsToken, PatchedVindex, UnstorableMetaEdit};

/// The fixture's score on every feature.
const BASE_SCORE: f32 = 0.5;
/// A score no fixture feature carries.
const EDITED_SCORE: f32 = 0.125;
/// A token id no fixture feature carries (the fixture uses `0..FEATURES`).
const NEW_TOKEN_ID: u32 = 9;
/// The slots the tests edit.
const LAYER: usize = 1;
const DELETED: usize = 0;
const UPDATED: usize = 1;
const INSERTED: usize = 2;
const UNTOUCHED: usize = 3;
/// The top-k width the source declares.
const SOURCE_TOP_K: usize = 1;

/// The text the fixture gives a token id (`tok{id}`).
fn fixture_token(id: u32) -> Option<String> {
    Some(format!("tok{id}"))
}

/// An index with no tokenizer to derive text from.
fn no_tokenizer(_id: u32) -> Option<String> {
    None
}

fn patched() -> PatchedVindex {
    PatchedVindex::new(build_synthetic_index())
}

fn meta(token: &str, token_id: u32, score: f32, top_k: usize) -> FeatureMeta {
    FeatureMeta {
        top_token: token.to_string(),
        top_token_id: token_id,
        c_score: score,
        top_k: (0..top_k)
            .map(|rank| TopKEntry {
                token: token.to_string(),
                token_id,
                logit: score - rank as f32 * 0.01,
            })
            .collect(),
    }
}

/// Write the patched index's effective metadata and read it back.
fn round_trip(
    patched: &PatchedVindex,
    min_top_k: usize,
) -> (Vec<Option<Vec<Option<FeatureMeta>>>>, usize, usize) {
    let dir = tempfile::tempdir().unwrap();
    let written = patched
        .save_effective_down_meta(dir.path(), min_top_k)
        .unwrap();
    let (records, _) = down_meta::read_binary(dir.path(), &tokenizer()).unwrap();
    let top_k_width = top_k_width_on_disk(dir.path());
    (records, written, top_k_width)
}

/// The top-k width in the file header (the fourth little-endian u32).
fn top_k_width_on_disk(dir: &std::path::Path) -> usize {
    const WIDTH_FIELD: std::ops::Range<usize> = 12..16;
    let bytes = std::fs::read(dir.join(larql_vindex::format::filenames::DOWN_META_BIN)).unwrap();
    u32::from_le_bytes(bytes[WIDTH_FIELD].try_into().unwrap()) as usize
}

#[test]
fn an_unedited_overlay_reports_no_metadata_edits() {
    let patched = patched();

    assert!(!patched.has_meta_edits());
    assert!(!patched.is_deleted(LAYER, DELETED));
    assert!(patched.unstorable_meta_edits().is_empty());
}

#[test]
fn the_written_file_holds_the_delete_the_update_and_the_insert() {
    let mut patched = patched();
    patched.delete_feature(LAYER, DELETED);
    patched.update_feature_meta(
        LAYER,
        UPDATED,
        meta("tok1", UPDATED as u32, EDITED_SCORE, 0),
    );
    patched.insert_feature(
        LAYER,
        INSERTED,
        Vec::new(),
        meta("new", NEW_TOKEN_ID, EDITED_SCORE, 0),
    );

    assert!(patched.has_meta_edits());
    assert!(patched.is_deleted(LAYER, DELETED));
    assert!(!patched.is_deleted(LAYER, UPDATED));
    assert!(patched.unstorable_meta_edits().is_empty());

    let (records, written, _) = round_trip(&patched, SOURCE_TOP_K);

    assert_eq!(records.len(), FIXTURE_LAYERS);
    let edited = records[LAYER].as_ref().expect("edited layer");
    assert_eq!(
        edited.len(),
        FIXTURE_FEATURES,
        "one record per base feature"
    );
    assert!(edited[DELETED].is_none(), "a deleted slot is empty");
    let updated = edited[UPDATED].as_ref().expect("updated slot");
    assert_eq!(updated.c_score, EDITED_SCORE);
    assert_eq!(updated.top_token_id, UPDATED as u32);
    let inserted = edited[INSERTED].as_ref().expect("inserted slot");
    assert_eq!(inserted.top_token_id, NEW_TOKEN_ID);
    assert_eq!(inserted.c_score, EDITED_SCORE);
    let untouched = edited[UNTOUCHED].as_ref().expect("untouched slot");
    assert_eq!(untouched.c_score, BASE_SCORE);
    assert_eq!(untouched.top_token_id, UNTOUCHED as u32);

    let other_layer = records[0].as_ref().expect("unedited layer");
    assert!(other_layer.iter().all(|m| m.is_some()));
    assert_eq!(
        written,
        FIXTURE_LAYERS * FIXTURE_FEATURES - 1,
        "every record but the deleted one"
    );
}

#[test]
fn the_file_is_at_least_as_wide_as_the_source_and_never_truncates() {
    const WIDE: usize = 3;
    let mut patched = patched();

    let (_, _, width) = round_trip(&patched, SOURCE_TOP_K);
    assert_eq!(width, SOURCE_TOP_K, "the fixture's records carry no top-k");

    patched.update_feature_meta(
        LAYER,
        UPDATED,
        meta("tok1", UPDATED as u32, EDITED_SCORE, WIDE),
    );
    let (records, _, width) = round_trip(&patched, SOURCE_TOP_K);
    assert_eq!(width, WIDE, "an edited record wider than the source");
    let updated = records[LAYER].as_ref().unwrap()[UPDATED].as_ref().unwrap();
    assert_eq!(updated.top_k.len(), WIDE);
}

#[test]
fn a_source_that_declares_no_width_gets_the_default() {
    const UNDECLARED: usize = 0;
    let (_, _, width) = round_trip(&patched(), UNDECLARED);

    assert!(width > 0, "a zero-width file could hold no top-k at all");
}

#[test]
fn a_text_change_that_keeps_the_token_id_is_reported() {
    let mut patched = patched();
    patched.update_feature_meta(
        LAYER,
        UPDATED,
        meta("changed", UPDATED as u32, BASE_SCORE, 0),
    );

    let unstorable = patched.unstorable_meta_edits();

    assert_eq!(
        unstorable,
        vec![UnstorableMetaEdit::TextWithoutTokenId {
            layer: LAYER,
            feature: UPDATED,
            stored_text: "tok1".into(),
            edited_text: "changed".into(),
        }]
    );
    let message = unstorable[0].to_string();
    assert!(
        message.contains(&format!("L{LAYER}/F{UPDATED}")),
        "{message}"
    );
    assert!(message.contains("changed"), "{message}");
    assert!(message.contains("token id"), "{message}");
    assert!(
        patched
            .labels_stored_as_their_token(&fixture_token)
            .is_empty(),
        "a refused edit is not also listed as stored"
    );
}

#[test]
fn text_that_is_not_its_token_is_stored_and_listed() {
    // A multi-token target is stored as its first token; the record is
    // written, and reloads as that token.
    let mut patched = patched();
    patched.update_feature_meta(LAYER, UPDATED, meta("changed", NEW_TOKEN_ID, BASE_SCORE, 0));

    assert!(patched.unstorable_meta_edits().is_empty());
    let labels = patched.labels_stored_as_their_token(&fixture_token);
    assert_eq!(
        labels,
        vec![LabelStoredAsToken {
            layer: LAYER,
            feature: UPDATED,
            session_text: "changed".into(),
            stored_text: fixture_token(NEW_TOKEN_ID).unwrap(),
        }]
    );
    let line = labels[0].to_string();
    assert!(
        line.contains("changed") && line.contains("reloads as"),
        "{line}"
    );
}

#[test]
fn text_that_is_its_token_is_not_listed() {
    let mut patched = patched();
    let token = fixture_token(NEW_TOKEN_ID).unwrap();
    patched.update_feature_meta(LAYER, UPDATED, meta(&token, NEW_TOKEN_ID, BASE_SCORE, 0));
    patched.delete_feature(LAYER, DELETED);
    patched.insert_feature(
        LAYER,
        FIXTURE_FEATURES,
        Vec::new(),
        meta("outside", NEW_TOKEN_ID, BASE_SCORE, 0),
    );

    assert!(patched
        .labels_stored_as_their_token(&fixture_token)
        .is_empty());
}

#[test]
fn without_token_text_no_label_is_listed() {
    let mut patched = patched();
    patched.update_feature_meta(LAYER, UPDATED, meta("changed", NEW_TOKEN_ID, BASE_SCORE, 0));

    assert!(patched
        .labels_stored_as_their_token(&no_tokenizer)
        .is_empty());
    assert!(patched
        .labels_stored_as_their_token(&|_| Some(String::new()))
        .is_empty());
}

#[test]
fn a_record_the_file_reads_back_as_empty_is_reported() {
    let mut patched = patched();
    patched.update_feature_meta(LAYER, UPDATED, meta("", 0, 0.0, 0));

    let unstorable = patched.unstorable_meta_edits();

    assert_eq!(
        unstorable,
        vec![UnstorableMetaEdit::ReadsBackEmpty {
            layer: LAYER,
            feature: UPDATED,
        }]
    );
    assert!(unstorable[0].to_string().contains("empty slot"));
}

#[test]
fn deleted_slots_are_listed_in_order() {
    let mut patched = patched();
    patched.delete_feature(LAYER, UPDATED);
    patched.delete_feature(0, INSERTED);

    assert_eq!(
        patched.deleted_slots(),
        vec![(0, INSERTED), (LAYER, UPDATED)]
    );
    assert_eq!(patched.num_deleted(), 2);
    assert!(patched.has_vector_or_meta_overrides());
}

#[test]
fn an_edit_outside_the_layer_is_reported_and_gets_no_record() {
    let mut patched = patched();
    patched.insert_feature(
        LAYER,
        FIXTURE_FEATURES,
        Vec::new(),
        meta("new", NEW_TOKEN_ID, EDITED_SCORE, 0),
    );
    patched.delete_feature(LAYER, FIXTURE_FEATURES + 1);

    let unstorable = patched.unstorable_meta_edits();

    assert_eq!(
        unstorable,
        vec![
            UnstorableMetaEdit::OutOfRange {
                layer: LAYER,
                feature: FIXTURE_FEATURES,
                num_features: FIXTURE_FEATURES,
            },
            UnstorableMetaEdit::OutOfRange {
                layer: LAYER,
                feature: FIXTURE_FEATURES + 1,
                num_features: FIXTURE_FEATURES,
            },
        ],
        "in slot order"
    );
    let message = unstorable[0].to_string();
    assert!(
        message.contains(&format!("L{LAYER}/F{FIXTURE_FEATURES}")),
        "{message}"
    );
    assert!(message.contains("outside"), "{message}");

    let (records, _, _) = round_trip(&patched, SOURCE_TOP_K);
    assert_eq!(records[LAYER].as_ref().unwrap().len(), FIXTURE_FEATURES);
}

#[test]
fn writing_over_a_hard_link_leaves_the_linked_file_untouched() {
    use larql_vindex::format::filenames::DOWN_META_BIN;

    let source = tempfile::tempdir().unwrap();
    let dest = tempfile::tempdir().unwrap();
    let unedited = patched();
    unedited
        .save_effective_down_meta(source.path(), SOURCE_TOP_K)
        .unwrap();
    let source_file = source.path().join(DOWN_META_BIN);
    let before = std::fs::read(&source_file).unwrap();
    std::fs::hard_link(&source_file, dest.path().join(DOWN_META_BIN)).unwrap();

    let mut edited = patched();
    edited.delete_feature(LAYER, DELETED);
    edited
        .save_effective_down_meta(dest.path(), SOURCE_TOP_K)
        .unwrap();

    assert_eq!(
        std::fs::read(&source_file).unwrap(),
        before,
        "the source's file must not change"
    );
    assert_ne!(
        std::fs::read(dest.path().join(DOWN_META_BIN)).unwrap(),
        before
    );
}
