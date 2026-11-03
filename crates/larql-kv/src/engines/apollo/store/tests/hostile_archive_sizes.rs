//! `.npz` members whose declared sizes are not to be believed, through the
//! public `ApolloStore::load`.
//!
//! The first two tests are controls: they prove the hand-written encoder makes
//! archives the loader accepts, so a refusal below is about the size and not
//! about a malformed test archive.

use super::archive_errors::{
    assert_archive_error, load_err, one_entry, store_with_archive, ENTRIES_NPZ, WINDOW_TOKENS_NPZ,
};
use super::raw_archive::{overlapping_archive, raw_archive, RawMember, METHOD_DEFLATED};
use super::*;

const HIDDEN: usize = 4;
const POW_2_46: u64 = 1 << 46;
const POW_2_62: u64 = 1 << 62;
const POW_2_63: u64 = 1 << 63;
const TOP_OF_U64: u64 = u64::MAX;
/// Below the ZIP64 sentinel, so it is carried in the 32-bit field itself.
const LARGEST_ZIP32_SIZE: u64 = 0xFFFF_FFFE;
const TOKENS_MEMBER: &str = "0.npy";
const ENTRIES_MEMBER: &str = "entries.npy";
const TOKENS: [u32; 3] = [0, 1, 2];

fn tokens_member() -> RawMember {
    RawMember::stored(TOKENS_MEMBER, synth_u32_npy(&TOKENS))
}

fn entries_member() -> RawMember {
    RawMember::stored(ENTRIES_MEMBER, synth_structured_entries_npy(&one_entry()))
}

fn store_with_member(file: &str, member: RawMember) -> TempDir {
    store_with_archive(file, &raw_archive(&[member]))
}

fn assert_inconsistent(err: StoreLoadError, member: &RawMember) {
    match err {
        StoreLoadError::MemberSizeInconsistent {
            declared, stored, ..
        } => {
            assert_eq!(declared, member.declared());
            assert_eq!(stored, member.data_len());
        }
        other => panic!("expected MemberSizeInconsistent, got {other:?}"),
    }
}

/// `$member`'s declared size is `$declared`; `$zip64` picks the carrying field.
macro_rules! declared_size_is_refused {
    ($($test:ident: $file:ident, $member:ident, $declared:expr, $zip64:expr;)*) => {$(
        #[test]
        fn $test() {
            let member = $member().declaring($declared);
            let member = if $zip64 { member.zip64() } else { member };
            let err = load_err(&store_with_member($file, member.clone()));
            assert_inconsistent(err, &member);
        }
    )*};
}

declared_size_is_refused! {
    tokens_zip64_declared_2_46: WINDOW_TOKENS_NPZ, tokens_member, POW_2_46, true;
    tokens_zip64_declared_2_62: WINDOW_TOKENS_NPZ, tokens_member, POW_2_62, true;
    tokens_zip64_declared_2_63: WINDOW_TOKENS_NPZ, tokens_member, POW_2_63, true;
    tokens_zip64_declared_top_of_u64: WINDOW_TOKENS_NPZ, tokens_member, TOP_OF_U64, true;
    tokens_zip32_declared_largest: WINDOW_TOKENS_NPZ, tokens_member, LARGEST_ZIP32_SIZE, false;
    entries_zip64_declared_2_46: ENTRIES_NPZ, entries_member, POW_2_46, true;
    entries_zip64_declared_2_62: ENTRIES_NPZ, entries_member, POW_2_62, true;
    entries_zip64_declared_2_63: ENTRIES_NPZ, entries_member, POW_2_63, true;
    entries_zip64_declared_top_of_u64: ENTRIES_NPZ, entries_member, TOP_OF_U64, true;
    entries_zip32_declared_largest: ENTRIES_NPZ, entries_member, LARGEST_ZIP32_SIZE, false;
}

#[test]
fn control_honest_raw_archives_load() {
    let dir = store_with_member(WINDOW_TOKENS_NPZ, tokens_member());
    std::fs::write(
        dir.path().join(ENTRIES_NPZ),
        raw_archive(&[entries_member()]),
    )
    .unwrap();
    let store = ApolloStore::load(dir.path()).expect("honest archives load");
    assert_eq!(store.window_tokens, vec![TOKENS.to_vec()]);
    assert_eq!(store.entries.len(), one_entry().len());
}

/// ZIP64 size fields are legal on a small member, so carrying the sizes there
/// must not by itself cause a refusal.
#[test]
fn control_honest_zip64_sizes_load() {
    let dir = store_with_member(WINDOW_TOKENS_NPZ, tokens_member().zip64());
    std::fs::write(
        dir.path().join(ENTRIES_NPZ),
        raw_archive(&[entries_member().zip64()]),
    )
    .unwrap();
    let store = ApolloStore::load(dir.path()).expect("honest ZIP64 archives load");
    assert_eq!(store.window_tokens, vec![TOKENS.to_vec()]);
}

#[test]
fn tokens_declared_one_byte_below_the_data_is_refused() {
    let member = tokens_member();
    let member = member.clone().declaring(member.data_len() - 1);
    let err = load_err(&store_with_member(WINDOW_TOKENS_NPZ, member.clone()));
    assert_inconsistent(err, &member);
}

#[test]
fn tokens_declared_one_byte_above_the_data_is_refused() {
    let member = tokens_member();
    let member = member.clone().declaring(member.data_len() + 1);
    let err = load_err(&store_with_member(WINDOW_TOKENS_NPZ, member.clone()));
    assert_inconsistent(err, &member);
}

#[test]
fn entries_declared_one_byte_below_the_data_is_refused() {
    let member = entries_member();
    let member = member.clone().declaring(member.data_len() - 1);
    let err = load_err(&store_with_member(ENTRIES_NPZ, member.clone()));
    assert_inconsistent(err, &member);
}

#[test]
fn entries_declared_one_byte_above_the_data_is_refused() {
    let member = entries_member();
    let member = member.clone().declaring(member.data_len() + 1);
    let err = load_err(&store_with_member(ENTRIES_NPZ, member.clone()));
    assert_inconsistent(err, &member);
}

fn assert_exceeds_archive(err: StoreLoadError, claimed_at_least: u64, file: &str, dir: &TempDir) {
    let file_len = std::fs::metadata(dir.path().join(file)).unwrap().len();
    match err {
        StoreLoadError::MemberExceedsArchive {
            claimed,
            archive_len,
            ..
        } => {
            assert_eq!(archive_len, file_len);
            assert!(claimed >= claimed_at_least, "claimed {claimed}");
            assert!(claimed > archive_len, "{claimed} within {archive_len}");
        }
        other => panic!("expected MemberExceedsArchive, got {other:?}"),
    }
}

/// Both size fields agree, so only the file's real length can refuse them.
macro_rules! both_sizes_past_the_file_are_refused {
    ($($test:ident: $file:ident, $member:ident, $size:expr;)*) => {$(
        #[test]
        fn $test() {
            let member = $member().declaring($size).occupying($size).zip64();
            let dir = store_with_member($file, member);
            let err = load_err(&dir);
            assert_exceeds_archive(err, $size, $file, &dir);
        }
    )*};
}

both_sizes_past_the_file_are_refused! {
    tokens_both_sizes_2_62: WINDOW_TOKENS_NPZ, tokens_member, POW_2_62;
    tokens_both_sizes_top_of_u64: WINDOW_TOKENS_NPZ, tokens_member, TOP_OF_U64;
    entries_both_sizes_2_62: ENTRIES_NPZ, entries_member, POW_2_62;
    entries_both_sizes_top_of_u64: ENTRIES_NPZ, entries_member, TOP_OF_U64;
}

/// A claim of exactly the file's length is within the rule, which is inclusive, so
/// it is the next check that refuses it: this data cannot fit before the directory.
#[test]
fn claim_of_exactly_the_archive_length_passes_the_budget() {
    // Field widths don't change with the value, so the length is known from a dry run.
    let archive_len = raw_archive(&[tokens_member().zip64()]).len() as u64;
    let member = tokens_member()
        .declaring(archive_len)
        .occupying(archive_len)
        .zip64();
    let err = load_err(&store_with_member(WINDOW_TOKENS_NPZ, member));
    assert_archive_error(
        &err,
        &ArchiveError::BadLocalHeader("data overlaps the central directory"),
    );
}

/// Each directory record is individually within the file, but they all share
/// one payload, so a small file would otherwise decode to many times its length.
#[test]
fn tokens_members_overlapping_one_payload_are_refused() {
    let names = ["0.npy", "1.npy", "2.npy"];
    let dir = TempDir::new().unwrap();
    write_minimal_store(dir.path(), names.len(), HIDDEN, TOKENS.len(), &one_entry());
    let member = tokens_member();
    let archive = overlapping_archive(&member, &names);
    let shared_payload_claims = member.data_len() * names.len() as u64;
    assert!(
        shared_payload_claims > archive.len() as u64,
        "the fixture must over-claim the file for this test to mean anything"
    );
    std::fs::write(dir.path().join(WINDOW_TOKENS_NPZ), archive).unwrap();
    let err = load_err(&dir);
    assert_exceeds_archive(err, member.data_len(), WINDOW_TOKENS_NPZ, &dir);
}

/// The reader decodes stored members only, so a compressed one is refused by
/// name whichever features `zip` is built with.
#[test]
fn deflated_tokens_member_is_refused_as_unsupported() {
    let member = RawMember::deflate_wrapped(TOKENS_MEMBER, synth_u32_npy(&TOKENS));
    let err = load_err(&store_with_member(WINDOW_TOKENS_NPZ, member));
    assert_archive_error(&err, &ArchiveError::UnsupportedCompression(METHOD_DEFLATED));
}

#[test]
fn deflated_entries_member_is_refused_as_unsupported() {
    let member =
        RawMember::deflate_wrapped(ENTRIES_MEMBER, synth_structured_entries_npy(&one_entry()));
    let err = load_err(&store_with_member(ENTRIES_NPZ, member));
    assert_archive_error(&err, &ArchiveError::UnsupportedCompression(METHOD_DEFLATED));
}

/// A running sum that would wrap past `u64::MAX` back to within the file: the first
/// member is honest and the second claims everything. Saturating stays over the limit.
#[test]
fn claims_that_would_wrap_u64_back_into_the_file_are_refused() {
    const WINDOWS: usize = 2;
    let dir = TempDir::new().unwrap();
    write_minimal_store(dir.path(), WINDOWS, HIDDEN, TOKENS.len(), &one_entry());
    let first = tokens_member();
    let second = RawMember::stored("1.npy", Vec::new())
        .declaring(u64::MAX)
        .occupying(u64::MAX)
        .zip64();
    std::fs::write(
        dir.path().join(WINDOW_TOKENS_NPZ),
        raw_archive(&[first, second]),
    )
    .unwrap();
    let err = load_err(&dir);
    assert_exceeds_archive(err, u64::MAX, WINDOW_TOKENS_NPZ, &dir);
}
