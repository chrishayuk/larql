//! The `.npz` reader's ZIP handling, through the public loader: forms real
//! writers produce must load, and malformed structure must be a typed error.

use super::archive_errors::{
    assert_archive_error, assert_names, load_err, one_entry, store_with_archive, ENTRIES_NPZ,
    WINDOW_TOKENS_NPZ,
};
use super::raw_archive::{
    archive_with_surroundings, raw_archive, zip64_directory_archive, RawMember,
};
use super::*;

const WINDOWS: usize = 1;
const HIDDEN: usize = 4;
const WINDOW_SIZE: usize = 3;
const TOKENS: [u32; 3] = [0, 1, 2];
const MEMBER: &str = "0.npy";
const ENTRIES_MEMBER: &str = "entries.npy";
/// Over the 24 characters of the longest window name.
const LONGER_THAN_ANY_WINDOW_NAME: usize = 40;
const END_RECORD_LEN: usize = 22;
const EXTRA_HEADER_LEN: usize = 4;
const CENTRAL_RECORD_LEN: usize = 46;
const ZIP64_LOCATOR_LEN: usize = 20;
const ZIP64_END_RECORD_LEN: usize = 56;
const END_RECORD_SIGNATURE: &[u8] = b"PK\x05\x06";
const CENTRAL_SIGNATURE: &[u8] = b"PK\x01\x02";
const FLAG_ENCRYPTED: u16 = 1;
const FLAG_PATCHED_DATA: u16 = 1 << 5;
const FLAG_STRONG_ENCRYPTION: u16 = 1 << 6;
/// Bit 3: sizes follow the data, as an unseekable `zipfile` stream writes.
const FLAG_DATA_DESCRIPTOR: u16 = 1 << 3;

fn member() -> RawMember {
    RawMember::stored(MEMBER, synth_u32_npy(&TOKENS))
}

fn assert_loads(archive: &[u8]) {
    let dir = store_with_archive(WINDOW_TOKENS_NPZ, archive);
    let store = ApolloStore::load(dir.path()).expect("legitimate archive loads");
    assert_eq!(store.window_tokens, vec![TOKENS.to_vec()]);
}

fn archive_err(archive: &[u8]) -> StoreLoadError {
    let err = load_err(&store_with_archive(WINDOW_TOKENS_NPZ, archive));
    assert_names(&err, WINDOW_TOKENS_NPZ);
    err
}

fn position_of(haystack: &[u8], needle: &[u8]) -> usize {
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
        .expect("needle present")
}

#[test]
fn prefixed_self_extracting_archive_loads() {
    assert_loads(&archive_with_surroundings(b"#!/bin/sh\n", &[member()], b""));
}

#[test]
fn archive_comment_loads() {
    assert_loads(&archive_with_surroundings(b"", &[member()], b"a comment"));
}

/// Only the last signature is used, so one in a comment is the end record, and a bogus
/// one is refused. `numpy` refuses this as well.
#[test]
fn comment_planting_an_end_signature_is_refused() {
    let archive = archive_with_surroundings(b"", &[member()], END_RECORD_SIGNATURE);
    assert_archive_error(&archive_err(&archive), &ArchiveError::NoEndRecord);
}

/// A comment hiding a whole second directory and an end record this reader would call
/// inconsistent. `numpy` reads the inner one, so falling back to the outer read other
/// data. Each way of being inconsistent is refused the same way.
#[test]
fn inconsistent_end_record_in_a_comment_is_refused_rather_than_skipped() {
    const DISK_AT: usize = 4;
    const RECORDS_ON_DISK_AT: usize = 8;
    const COMMENT_LEN_AT: usize = 20;
    for field in [DISK_AT, RECORDS_ON_DISK_AT, COMMENT_LEN_AT] {
        let mut inner = raw_archive(&[RawMember::stored("x", Vec::new())]);
        let end_at = inner.len() - END_RECORD_LEN;
        inner[end_at + field] = 9;
        let archive = archive_with_surroundings(b"", &[member()], &inner);
        assert_archive_error(&archive_err(&archive), &ArchiveError::NoEndRecord);
    }
}

/// An end signature with too few bytes after it to be a record is still the last one.
#[test]
fn truncated_end_signature_after_an_archive_is_refused() {
    let mut archive = raw_archive(&[member()]);
    archive.extend_from_slice(END_RECORD_SIGNATURE);
    assert_archive_error(&archive_err(&archive), &ArchiveError::NoEndRecord);
}

/// A last end record that fails its checks mustn't fall back to an earlier one:
/// `zipfile` commits to the last signature, so the two readers would see two archives.
#[test]
fn inconsistent_last_end_record_does_not_select_an_earlier_archive() {
    const DISK_AT: usize = 4;
    let earlier = raw_archive(&[member()]);
    let mut archive = [earlier, raw_archive(&[member()])].concat();
    let last_end_at = archive.len() - END_RECORD_LEN;
    archive[last_end_at + DISK_AT] = 1;
    assert_archive_error(&archive_err(&archive), &ArchiveError::NoEndRecord);
}

/// The same with a signature after a complete archive and outside its comment.
#[test]
fn end_signature_past_the_comment_is_refused() {
    let mut archive = raw_archive(&[member()]);
    archive.extend_from_slice(END_RECORD_SIGNATURE);
    archive.extend_from_slice(&[0xAB; END_RECORD_LEN]);
    assert_archive_error(&archive_err(&archive), &ArchiveError::NoEndRecord);
}

/// A directory offset whose little-endian bytes are the end record's signature
/// puts those bytes in the record's own fields. The record at the zero-comment
/// position is the real one; `zipfile` looks there first too. A hole keeps the
/// file cheap, and the bytes before the directory are never read.
#[test]
fn directory_offset_spelling_the_end_signature_loads() {
    use std::io::{Seek, SeekFrom, Write};
    const SIGNATURE_AS_OFFSET: u32 = u32::from_le_bytes(*b"PK\x05\x06");
    const DIRECTORY_OFFSET_AT: usize = 16;
    let mut archive = raw_archive(&[member()]);
    let end_at = archive.len() - END_RECORD_LEN;
    let offset_at = end_at + DIRECTORY_OFFSET_AT;
    let directory_at =
        u32::from_le_bytes(archive[offset_at..offset_at + 4].try_into().unwrap()) as usize;
    archive[offset_at..offset_at + 4].copy_from_slice(&SIGNATURE_AS_OFFSET.to_le_bytes());

    let dir = store_with_archive(WINDOW_TOKENS_NPZ, &archive[..directory_at]);
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(dir.path().join(WINDOW_TOKENS_NPZ))
        .unwrap();
    file.seek(SeekFrom::Start(u64::from(SIGNATURE_AS_OFFSET)))
        .unwrap();
    file.write_all(&archive[directory_at..]).unwrap();
    drop(file);

    let store = ApolloStore::load(dir.path()).expect("a real archive with this offset loads");
    assert_eq!(store.window_tokens, vec![TOKENS.to_vec()]);
}

/// A last record whose comment runs past the file is refused, with no fallback.
/// This doesn't pin the zero-comment check itself: a nonzero comment at that
/// position always overruns the file, so the scan refuses it the same way.
#[test]
fn record_with_a_comment_overrunning_the_file_is_refused() {
    let mut archive = raw_archive(&[member()]);
    archive.extend_from_slice(END_RECORD_SIGNATURE);
    archive.extend_from_slice(&[0u8; END_RECORD_LEN - END_RECORD_SIGNATURE.len() - 2]);
    archive.extend_from_slice(&1u16.to_le_bytes());
    assert_archive_error(&archive_err(&archive), &ArchiveError::NoEndRecord);
}

#[test]
fn bytes_after_the_comment_load() {
    let mut archive = raw_archive(&[member()]);
    archive.extend_from_slice(b"trailing");
    assert_loads(&archive);
}

#[test]
fn data_descriptor_flag_loads_from_central_sizes() {
    assert_loads(&raw_archive(&[member().with_flags(FLAG_DATA_DESCRIPTOR)]));
}

#[test]
fn directory_and_other_members_are_ignored() {
    let extras = RawMember::stored("meta/", Vec::new());
    let readme = RawMember::stored("README.txt", b"hello".to_vec());
    assert_loads(&raw_archive(&[extras, readme, member()]));
}

#[test]
fn zip64_end_records_load() {
    assert_loads(&zip64_directory_archive(b"", &member(), 1, b""));
}

#[test]
fn zip64_end_records_with_a_prefix_load() {
    assert_loads(&zip64_directory_archive(b"prefix", &member(), 1, b""));
}

#[test]
fn zip64_extensible_sector_loads_without_a_prefix() {
    assert_loads(&zip64_directory_archive(b"", &member(), 1, b"sector"));
}

/// Newly unsupported, pinned: with a prefix the ZIP64 record is found by assuming
/// its standard length, which an extensible sector breaks. `zipfile` assumes the same.
#[test]
fn zip64_extensible_sector_with_a_prefix_finds_no_end_record() {
    let archive = zip64_directory_archive(b"prefix", &member(), 1, b"sector");
    assert_archive_error(&archive_err(&archive), &ArchiveError::NoEndRecord);
}

/// Newly unsupported, pinned: the directory must end where the end record starts.
/// The `zip` crate scans for it instead; no writer tested here leaves such a gap.
#[test]
fn gap_between_directory_and_end_record_finds_no_end_record() {
    let mut archive = raw_archive(&[member()]);
    let end_at = archive.len() - END_RECORD_LEN;
    let end_record = archive.split_off(end_at);
    archive.extend_from_slice(b"gap");
    archive.extend_from_slice(&end_record);
    assert_archive_error(&archive_err(&archive), &ArchiveError::NoEndRecord);
}

/// A ZIP64 end record is only believed if a real locator points to it.
#[test]
fn zip64_end_record_without_a_locator_signature_finds_no_end_record() {
    const LOCATOR_FROM_END: usize = END_RECORD_LEN + ZIP64_LOCATOR_LEN;
    let mut archive = zip64_directory_archive(b"", &member(), 1, b"");
    let locator_at = archive.len() - LOCATOR_FROM_END;
    archive[locator_at] = b'X';
    assert_archive_error(&archive_err(&archive), &ArchiveError::NoEndRecord);
}

/// A locator pointing so near the end that a ZIP64 record there would run off the
/// file is simply not a candidate, not an I/O failure.
#[test]
fn zip64_locator_pointing_past_a_readable_record_finds_no_end_record() {
    const LOCATOR_FROM_END: usize = END_RECORD_LEN + ZIP64_LOCATOR_LEN;
    const LOCATOR_OFFSET_FIELD_AT: usize = 8;
    // Remove the ZIP64 record too, so the adjacent guess can't rescue it.
    let mut archive = zip64_directory_archive(b"", &member(), 1, b"");
    let locator_at = archive.len() - LOCATOR_FROM_END;
    let just_before_locator = (locator_at - 1) as u64;
    archive[locator_at + LOCATOR_OFFSET_FIELD_AT..locator_at + LOCATOR_OFFSET_FIELD_AT + 8]
        .copy_from_slice(&just_before_locator.to_le_bytes());
    archive[locator_at - ZIP64_END_RECORD_LEN] = b'X';
    assert_archive_error(&archive_err(&archive), &ArchiveError::NoEndRecord);
}

/// Each 32-bit field defers to ZIP64 alone when it is the only one at its sentinel,
/// and the end record's other fields then hold their real values.
#[test]
fn any_single_sentinel_in_the_end_record_defers_to_zip64() {
    const RECORDS_ON_DISK_AT: usize = 8;
    const RECORDS_AT: usize = 10;
    const DIRECTORY_LEN_AT: usize = 12;
    const DIRECTORY_OFFSET_AT: usize = 16;
    const REAL_RECORDS: u16 = 1;
    /// The one-member archive's real directory length and offset are not needed:
    /// any value other than the sentinel leaves that field not deferring.
    const NOT_SENTINEL: u32 = 0;
    let base = zip64_directory_archive(b"", &member(), 1, b"");
    let end_at = base.len() - END_RECORD_LEN;
    let with_fields = |records: u16, len: u32, offset: u32| {
        let mut archive = base.clone();
        let end = &mut archive[end_at..];
        end[RECORDS_ON_DISK_AT..RECORDS_ON_DISK_AT + 2].copy_from_slice(&records.to_le_bytes());
        end[RECORDS_AT..RECORDS_AT + 2].copy_from_slice(&records.to_le_bytes());
        end[DIRECTORY_LEN_AT..DIRECTORY_LEN_AT + 4].copy_from_slice(&len.to_le_bytes());
        end[DIRECTORY_OFFSET_AT..DIRECTORY_OFFSET_AT + 4].copy_from_slice(&offset.to_le_bytes());
        archive
    };
    assert_loads(&with_fields(u16::MAX, NOT_SENTINEL, NOT_SENTINEL));
    assert_loads(&with_fields(REAL_RECORDS, u32::MAX, NOT_SENTINEL));
    assert_loads(&with_fields(REAL_RECORDS, NOT_SENTINEL, u32::MAX));
}

/// The record count passes the minimum-size check because names are long, so the
/// directory runs out exactly where another record's fixed part should start.
#[test]
fn directory_running_out_between_records_is_refused() {
    const RECORDS_ON_DISK_AT: usize = 8;
    const RECORDS_AT: usize = 10;
    const CLAIMED_RECORDS: u16 = 3;
    let long_name = "n".repeat(CENTRAL_RECORD_LEN);
    let mut archive = raw_archive(&[
        RawMember::stored(&long_name, Vec::new()),
        RawMember::stored(&format!("{long_name}2"), Vec::new()),
    ]);
    let end_at = archive.len() - END_RECORD_LEN;
    for field in [RECORDS_ON_DISK_AT, RECORDS_AT] {
        archive[end_at + field..end_at + field + 2].copy_from_slice(&CLAIMED_RECORDS.to_le_bytes());
    }
    assert_archive_error(
        &archive_err(&archive),
        &ArchiveError::BadDirectoryRecord {
            index: 2,
            reason: "runs past the directory",
        },
    );
}

#[test]
fn zip64_declared_record_count_beyond_the_directory_finds_no_end_record() {
    const POW_2_62: u64 = 1 << 62;
    let archive = zip64_directory_archive(b"", &member(), POW_2_62, b"");
    assert_archive_error(&archive_err(&archive), &ArchiveError::NoEndRecord);
}

#[test]
fn zip64_record_count_that_overflows_the_arithmetic_finds_no_end_record() {
    let archive = zip64_directory_archive(b"", &member(), u64::MAX, b"");
    assert_archive_error(&archive_err(&archive), &ArchiveError::NoEndRecord);
}

#[test]
fn file_shorter_than_an_end_record_finds_no_end_record() {
    assert_archive_error(&archive_err(b"PK"), &ArchiveError::NoEndRecord);
}

#[test]
fn comment_running_past_the_file_finds_no_end_record() {
    let mut archive = raw_archive(&[member()]);
    let comment_len_at = archive.len() - 2;
    archive[comment_len_at..].copy_from_slice(&u16::MAX.to_le_bytes());
    assert_archive_error(&archive_err(&archive), &ArchiveError::NoEndRecord);
}

#[test]
fn directory_length_past_the_file_start_finds_no_end_record() {
    const DIRECTORY_LEN_AT: usize = 12;
    let mut archive = raw_archive(&[member()]);
    let end_at = archive.len() - END_RECORD_LEN;
    let oversized = u32::try_from(archive.len()).unwrap();
    archive[end_at + DIRECTORY_LEN_AT..end_at + DIRECTORY_LEN_AT + 4]
        .copy_from_slice(&oversized.to_le_bytes());
    assert_archive_error(&archive_err(&archive), &ArchiveError::NoEndRecord);
}

#[test]
fn multi_disk_archive_finds_no_end_record() {
    const DISK_AT: usize = 4;
    let mut archive = raw_archive(&[member()]);
    let end_at = archive.len() - END_RECORD_LEN;
    archive[end_at + DISK_AT] = 1;
    assert_archive_error(&archive_err(&archive), &ArchiveError::NoEndRecord);
}

#[test]
fn record_counts_that_disagree_find_no_end_record() {
    const RECORDS_ON_DISK_AT: usize = 8;
    let mut archive = raw_archive(&[member()]);
    let end_at = archive.len() - END_RECORD_LEN;
    archive[end_at + RECORDS_ON_DISK_AT] = 0;
    assert_archive_error(&archive_err(&archive), &ArchiveError::NoEndRecord);
}

/// `zip` reads as many records as the count says and `zipfile` as many as fit the
/// length, so a count short of the directory has two different member lists.
#[test]
fn record_count_short_of_the_directory_is_refused() {
    const RECORDS_ON_DISK_AT: usize = 8;
    const RECORDS_AT: usize = 10;
    const CLAIMED_RECORDS: u16 = 1;
    let mut archive = raw_archive(&[member(), RawMember::stored("x", Vec::new())]);
    let end_at = archive.len() - END_RECORD_LEN;
    for field in [RECORDS_ON_DISK_AT, RECORDS_AT] {
        archive[end_at + field..end_at + field + 2].copy_from_slice(&CLAIMED_RECORDS.to_le_bytes());
    }
    assert_archive_error(
        &archive_err(&archive),
        &ArchiveError::BadDirectoryRecord {
            index: 1,
            reason: "directory continues past its record count",
        },
    );
}

#[test]
fn second_record_with_a_bad_signature_is_refused_by_index() {
    let mut archive = raw_archive(&[member(), RawMember::stored("1.npy", Vec::new())]);
    let first = position_of(&archive, CENTRAL_SIGNATURE);
    let second = first + 1 + position_of(&archive[first + 1..], CENTRAL_SIGNATURE);
    archive[second] = b'X';
    assert_archive_error(
        &archive_err(&archive),
        &ArchiveError::BadDirectoryRecord {
            index: 1,
            reason: "bad signature",
        },
    );
}

#[test]
fn record_whose_name_runs_past_the_directory_is_refused() {
    const NAME_LEN_AT: usize = 28;
    let mut archive = raw_archive(&[member()]);
    let record = position_of(&archive, CENTRAL_SIGNATURE);
    archive[record + NAME_LEN_AT..record + NAME_LEN_AT + 2]
        .copy_from_slice(&u16::MAX.to_le_bytes());
    assert_archive_error(
        &archive_err(&archive),
        &ArchiveError::BadDirectoryRecord {
            index: 0,
            reason: "runs past the directory",
        },
    );
}

#[test]
fn directory_shorter_than_its_records_is_refused() {
    const DIRECTORY_LEN_AT: usize = 12;
    const OFFSET_AT: usize = 16;
    const ONE_RECORD_MINIMUM: u32 = CENTRAL_RECORD_LEN as u32;
    let mut archive = raw_archive(&[member()]);
    let end_at = archive.len() - END_RECORD_LEN;
    let start = position_of(&archive, CENTRAL_SIGNATURE) as u32;
    let actual = u32::from_le_bytes(archive[end_at + 12..end_at + 16].try_into().unwrap());
    // Exactly one fixed record long, so the name no longer fits inside it.
    let moved = start + actual - ONE_RECORD_MINIMUM;
    archive[end_at + DIRECTORY_LEN_AT..end_at + DIRECTORY_LEN_AT + 4]
        .copy_from_slice(&ONE_RECORD_MINIMUM.to_le_bytes());
    archive[end_at + OFFSET_AT..end_at + OFFSET_AT + 4].copy_from_slice(&moved.to_le_bytes());
    let err = archive_err(&archive);
    assert!(
        matches!(
            err,
            StoreLoadError::Archive {
                source: ArchiveError::NoEndRecord | ArchiveError::BadDirectoryRecord { .. },
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn zip64_extra_too_short_for_its_sentinels_is_refused() {
    const EXTRA_LEN_FIELD_AT: usize = 2;
    let archive = raw_archive(&[member().zip64()]);
    let record = position_of(&archive, CENTRAL_SIGNATURE);
    let extra_at = record + CENTRAL_RECORD_LEN + MEMBER.len();
    let mut truncated = archive;
    truncated[extra_at + EXTRA_LEN_FIELD_AT..extra_at + EXTRA_LEN_FIELD_AT + 2]
        .copy_from_slice(&8u16.to_le_bytes());
    assert_archive_error(
        &archive_err(&truncated),
        &ArchiveError::BadDirectoryRecord {
            index: 0,
            reason: "ZIP64 extra field too short",
        },
    );
}

#[test]
fn extra_field_overrunning_its_record_is_refused() {
    const EXTRA_LEN_FIELD_AT: usize = 2;
    let archive = raw_archive(&[member().zip64()]);
    let record = position_of(&archive, CENTRAL_SIGNATURE);
    let extra_at = record + CENTRAL_RECORD_LEN + MEMBER.len();
    let mut overrun = archive;
    overrun[extra_at + EXTRA_LEN_FIELD_AT..extra_at + EXTRA_LEN_FIELD_AT + 2]
        .copy_from_slice(&u16::MAX.to_le_bytes());
    assert_archive_error(
        &archive_err(&overrun),
        &ArchiveError::BadDirectoryRecord {
            index: 0,
            reason: "extra field overruns its record",
        },
    );
}

#[test]
fn encrypted_and_unsupported_flags_are_refused() {
    for flag in [FLAG_ENCRYPTED, FLAG_PATCHED_DATA, FLAG_STRONG_ENCRYPTION] {
        let archive = raw_archive(&[member().with_flags(flag)]);
        assert_archive_error(&archive_err(&archive), &ArchiveError::Encrypted);
    }
}

#[test]
fn local_name_differing_from_the_directory_is_refused() {
    let mut archive = raw_archive(&[member()]);
    let local_name_at = position_of(&archive, MEMBER.as_bytes());
    archive[local_name_at] = b'9';
    assert_archive_error(
        &archive_err(&archive),
        &ArchiveError::BadLocalHeader("name differs from the directory"),
    );
}

#[test]
fn local_name_of_a_different_length_is_refused() {
    const NAME_LEN_AT: usize = 26;
    let mut archive = raw_archive(&[member()]);
    archive[NAME_LEN_AT] -= 1;
    assert_archive_error(
        &archive_err(&archive),
        &ArchiveError::BadLocalHeader("name differs from the directory"),
    );
}

/// ZIP64 fields naming any disk other than the first describe a split archive.
#[test]
fn zip64_nonzero_disk_fields_find_no_end_record() {
    const ZIP64_RECORD_FROM_END: usize = END_RECORD_LEN + ZIP64_LOCATOR_LEN + ZIP64_END_RECORD_LEN;
    const LOCATOR_FROM_END: usize = END_RECORD_LEN + ZIP64_LOCATOR_LEN;
    const RECORD_DISK_AT: usize = 16;
    const RECORD_DIRECTORY_DISK_AT: usize = 20;
    const LOCATOR_DISK_AT: usize = 4;
    const LOCATOR_TOTAL_DISKS_AT: usize = 16;
    const SECOND_DISK: u8 = 2;
    for field_from_end in [
        ZIP64_RECORD_FROM_END - RECORD_DISK_AT,
        ZIP64_RECORD_FROM_END - RECORD_DIRECTORY_DISK_AT,
        LOCATOR_FROM_END - LOCATOR_DISK_AT,
        LOCATOR_FROM_END - LOCATOR_TOTAL_DISKS_AT,
    ] {
        let mut archive = zip64_directory_archive(b"", &member(), 1, b"");
        let at = archive.len() - field_from_end;
        archive[at] = SECOND_DISK;
        assert_archive_error(&archive_err(&archive), &ArchiveError::NoEndRecord);
    }
}

/// A locator next to an end record that doesn't defer to it. `zipfile` honours it and
/// `zip` doesn't, so if they would read different directories it must be refused.
#[test]
fn zip64_locator_disagreeing_with_a_plain_end_record_finds_no_end_record() {
    const RECORDS_ON_DISK_AT: usize = 8;
    const RECORDS_AT: usize = 10;
    const DISAGREEING_RECORDS: u16 = 2;
    let mut archive = plain_end_record_after_agreeing_zip64(b"");
    let end_at = archive.len() - END_RECORD_LEN;
    let end = &mut archive[end_at..];
    end[RECORDS_ON_DISK_AT..RECORDS_ON_DISK_AT + 2]
        .copy_from_slice(&DISAGREEING_RECORDS.to_le_bytes());
    end[RECORDS_AT..RECORDS_AT + 2].copy_from_slice(&DISAGREEING_RECORDS.to_le_bytes());
    assert_archive_error(&archive_err(&archive), &ArchiveError::NoEndRecord);
}

/// Sentinels, a locator stating one ZIP64 record and another immediately before the
/// locator: `zipfile` reads the adjacent one and `zip` the stated one.
#[test]
fn zip64_stated_and_adjacent_records_differing_finds_no_end_record() {
    let mut adjacent_lookalike = b"PK\x06\x06".to_vec();
    adjacent_lookalike.resize(ZIP64_END_RECORD_LEN, 0);
    let archive = zip64_directory_archive(b"", &member(), 1, &adjacent_lookalike);
    assert_archive_error(&archive_err(&archive), &ArchiveError::NoEndRecord);
}

/// A plain end record's agreeing ZIP64 record must be adjacent. `zipfile` looks only
/// there, so one found only at the stated offset would be read by `zip` alone.
#[test]
fn zip64_agreeing_with_a_plain_end_record_must_be_adjacent() {
    let archive = plain_end_record_after_agreeing_zip64(b"sector");
    assert_archive_error(&archive_err(&archive), &ArchiveError::NoEndRecord);
}

/// `zip` reads all three values from a ZIP64 extra field long enough to hold them,
/// even where the record's own fields aren't sentinels. `zipfile` reads only sentinels.
#[test]
fn full_zip64_extra_disagreeing_with_the_record_is_refused() {
    const ZIP64_EXTRA_ID: [u8; 2] = [1, 0];
    const FULL_BODY_LEN: u16 = 24;
    const OTHER_VALUE: u64 = 7;
    let extra = [
        ZIP64_EXTRA_ID.as_slice(),
        &FULL_BODY_LEN.to_le_bytes(),
        &OTHER_VALUE.to_le_bytes(),
        &OTHER_VALUE.to_le_bytes(),
        &OTHER_VALUE.to_le_bytes(),
    ]
    .concat();
    let archive = raw_archive(&[member().with_leading_extra(&extra)]);
    assert_archive_error(
        &archive_err(&archive),
        &ArchiveError::BadDirectoryRecord {
            index: 0,
            reason: "ZIP64 extra field disagrees with the record",
        },
    );
}

/// ZIP64 records carrying the same values as a plain end record are harmless. Python's
/// `zipfile` honours them and this reader uses the plain record, so the two agree.
#[test]
fn zip64_records_agreeing_with_a_plain_end_record_load() {
    assert_loads(&plain_end_record_after_agreeing_zip64(b""));
}

/// `archive` with a directory, agreeing ZIP64 records, then a plain end record. The
/// directory is found from the end record, so its stated length covers the ZIP64 records.
fn plain_end_record_after_agreeing_zip64(sector: &[u8]) -> Vec<u8> {
    const ZIP64_END_SIGNATURE: &[u8] = b"PK\x06\x06";
    const ZIP64_LOCATOR_SIGNATURE: &[u8] = b"PK\x06\x07";
    const ZIP64_END_FIXED_SIZE: u64 = 44;
    const VERSION: u16 = 45;
    let plain = raw_archive(&[member()]);
    let end_at = plain.len() - END_RECORD_LEN;
    let directory_at = position_of(&plain, CENTRAL_SIGNATURE) as u64;
    let directory_len = end_at as u64 - directory_at;
    let zip64_at = end_at as u64;
    let mut out = plain[..end_at].to_vec();
    out.extend_from_slice(ZIP64_END_SIGNATURE);
    out.extend_from_slice(&(ZIP64_END_FIXED_SIZE + sector.len() as u64).to_le_bytes());
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&[0; 8]);
    out.extend_from_slice(&1u64.to_le_bytes());
    out.extend_from_slice(&1u64.to_le_bytes());
    out.extend_from_slice(&directory_len.to_le_bytes());
    out.extend_from_slice(&directory_at.to_le_bytes());
    out.extend_from_slice(sector);
    out.extend_from_slice(ZIP64_LOCATOR_SIGNATURE);
    out.extend_from_slice(&[0; 4]);
    out.extend_from_slice(&zip64_at.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&plain[end_at..]);
    out
}

// ── Which member is which ─────────────────────────────────────────────

fn entries_with_token(token_id: u32) -> Vec<u8> {
    let mut rows = one_entry();
    rows[0].token_id = token_id;
    synth_structured_entries_npy(&rows)
}

fn entries_archive_err(archive: &[u8]) -> StoreLoadError {
    load_err(&store_with_archive(ENTRIES_NPZ, archive))
}

/// `numpy` and `zip` take the last duplicate and a reader taking the first would see
/// other data, so a duplicate has no one right answer.
#[test]
fn duplicate_entries_member_is_refused() {
    let archive = raw_archive(&[
        RawMember::stored(ENTRIES_MEMBER, entries_with_token(1)),
        RawMember::stored(ENTRIES_MEMBER, entries_with_token(2)),
    ]);
    match entries_archive_err(&archive) {
        StoreLoadError::Archive {
            source: ArchiveError::AmbiguousMemberName(name),
            ..
        } => assert_eq!(name, ENTRIES_MEMBER),
        other => panic!("expected AmbiguousMemberName, got {other:?}"),
    }
}

#[test]
fn decoy_before_the_entries_member_is_not_read() {
    const REAL_TOKEN: u32 = 2;
    let archive = raw_archive(&[
        RawMember::stored("entries_old.npy", entries_with_token(1)),
        RawMember::stored(ENTRIES_MEMBER, entries_with_token(REAL_TOKEN)),
    ]);
    let dir = store_with_archive(ENTRIES_NPZ, &archive);
    let store = ApolloStore::load(dir.path()).expect("exact name is found");
    assert_eq!(store.entries[0].token_id, REAL_TOKEN);
}

/// `numpy` looks up the key `entries` as a bare member first, ahead of `entries.npy`.
#[test]
fn bare_entries_beside_entries_npy_is_refused() {
    for names in [["entries", ENTRIES_MEMBER], [ENTRIES_MEMBER, "entries"]] {
        let archive = raw_archive(&[
            RawMember::stored(names[0], entries_with_token(1)),
            RawMember::stored(names[1], entries_with_token(2)),
        ]);
        assert_ambiguous(&entries_archive_err(&archive), ENTRIES_MEMBER);
    }
}

/// `zipfile` cuts a name at its first NUL, so this is `entries.npy` to `numpy`, which
/// takes the last. It's refused anywhere in either archive.
#[test]
fn nul_in_a_member_name_is_refused_in_both_archives() {
    let cut_at_nul = format!("{ENTRIES_MEMBER}\0x");
    let archive = raw_archive(&[
        RawMember::stored(ENTRIES_MEMBER, entries_with_token(1)),
        RawMember::stored(&cut_at_nul, entries_with_token(2)),
    ]);
    assert_ambiguous(&entries_archive_err(&archive), &cut_at_nul);

    let cut_at_nul = format!("{MEMBER}\0x");
    let archive = raw_archive(&[
        member(),
        RawMember::stored(&cut_at_nul, synth_u32_npy(&TOKENS)),
    ]);
    assert_ambiguous(&archive_err(&archive), &cut_at_nul);
}

/// A Unicode Path field. `zip` and `zipfile` read this name in place of the record's.
fn unicode_path_extra(path: &str) -> Vec<u8> {
    const UNICODE_PATH_ID: u16 = 0x7075;
    const VERSION_AND_CRC: [u8; 5] = [1, 0, 0, 0, 0];
    let body_len = u16::try_from(VERSION_AND_CRC.len() + path.len()).unwrap();
    [
        UNICODE_PATH_ID.to_le_bytes().as_slice(),
        &body_len.to_le_bytes(),
        &VERSION_AND_CRC,
        path.as_bytes(),
    ]
    .concat()
}

#[test]
fn unicode_path_naming_a_member_differently_is_refused() {
    const OTHER_NAME: &str = "a.txt";
    let renamed_to_entries = RawMember::stored(OTHER_NAME, entries_with_token(1))
        .with_leading_extra(&unicode_path_extra(ENTRIES_MEMBER));
    let real = RawMember::stored(ENTRIES_MEMBER, entries_with_token(2));
    assert_ambiguous(
        &entries_archive_err(&raw_archive(&[renamed_to_entries, real])),
        OTHER_NAME,
    );

    let entries_renamed_away = RawMember::stored(ENTRIES_MEMBER, entries_with_token(1))
        .with_leading_extra(&unicode_path_extra(OTHER_NAME));
    assert_ambiguous(
        &entries_archive_err(&raw_archive(&[entries_renamed_away])),
        ENTRIES_MEMBER,
    );
}

/// A plain name with a non-ASCII path differs under `zip` and `zipfile` too.
#[test]
fn unicode_path_non_ascii_on_a_plain_name_is_refused() {
    let renamed = RawMember::stored(ENTRIES_MEMBER, entries_with_token(1))
        .with_leading_extra(&unicode_path_extra("entriés.npy"));
    assert_ambiguous(
        &entries_archive_err(&raw_archive(&[renamed])),
        ENTRIES_MEMBER,
    );
}

/// What Info-ZIP writes for a non-ASCII name, which must keep loading: the encodings
/// differ, and neither can be a window or `entries.npy`.
#[test]
fn unicode_path_on_a_non_ascii_name_loads() {
    const CP437_NAME_BYTES: &[u8] = b"caf\x82.txt";
    const UTF8_NAME: &str = "café.txt";
    const PLACEHOLDER_NAME: &str = "cafe.txt";
    let other = RawMember::stored(PLACEHOLDER_NAME, b"hi".to_vec())
        .with_leading_extra(&unicode_path_extra(UTF8_NAME));
    let mut archive = raw_archive(&[other, member()]);
    let mut from = 0;
    while let Some(at) = archive[from..]
        .windows(PLACEHOLDER_NAME.len())
        .position(|w| w == PLACEHOLDER_NAME.as_bytes())
    {
        archive[from + at..from + at + CP437_NAME_BYTES.len()].copy_from_slice(CP437_NAME_BYTES);
        from += at + CP437_NAME_BYTES.len();
    }
    assert_loads(&archive);
}

fn assert_ambiguous(err: &StoreLoadError, expected: &str) {
    match err {
        StoreLoadError::Archive {
            source: ArchiveError::AmbiguousMemberName(name),
            ..
        } => assert_eq!(name, expected),
        other => panic!("expected AmbiguousMemberName, got {other:?}"),
    }
}

#[test]
fn only_a_similarly_named_member_is_missing_file() {
    for decoy in [
        "entries_old.npy",
        "entries",
        "entries.npy.npy",
        "ENTRIES.NPY",
        "entries.npy ",
        "./entries.npy",
        "../entries.npy",
        "dir\\entries.npy",
        "/entries.npy",
        "/../entries.npy",
        "C:\\entries.npy",
    ] {
        let archive = raw_archive(&[RawMember::stored(decoy, entries_with_token(1))]);
        let err = entries_archive_err(&archive);
        assert!(
            matches!(err, StoreLoadError::MissingFile(_)),
            "{decoy}: {err:?}"
        );
    }
}

/// Names an extractor would write outside its destination. The load never extracts:
/// it matches member names exactly, so none of these is ever read in place of the
/// real member, in either archive.
const ESCAPING_NAME_PREFIXES: [&str; 4] = ["../", "/", "/../", "C:\\"];

#[test]
fn escaping_names_beside_the_real_member_are_never_read() {
    const DECOY_TOKEN: u32 = 7;
    const REAL_TOKEN: u32 = 2;
    for prefix in ESCAPING_NAME_PREFIXES {
        let window_decoy = format!("{prefix}{MEMBER}");
        let archive = raw_archive(&[
            RawMember::stored(&window_decoy, synth_u32_npy(&[DECOY_TOKEN])),
            member(),
            RawMember::stored(&window_decoy, synth_u32_npy(&[DECOY_TOKEN])),
        ]);
        assert_loads(&archive);

        let entries_decoy = format!("{prefix}{ENTRIES_MEMBER}");
        let archive = raw_archive(&[
            RawMember::stored(&entries_decoy, entries_with_token(DECOY_TOKEN)),
            RawMember::stored(ENTRIES_MEMBER, entries_with_token(REAL_TOKEN)),
            RawMember::stored(&entries_decoy, entries_with_token(DECOY_TOKEN)),
        ]);
        let dir = store_with_archive(ENTRIES_NPZ, &archive);
        let store = ApolloStore::load(dir.path()).expect("exact name is found");
        assert_eq!(store.entries[0].token_id, REAL_TOKEN, "{entries_decoy}");
    }
}

#[test]
fn duplicate_window_member_is_a_window_archive_error() {
    let archive = raw_archive(&[member(), member()]);
    let err = archive_err_any(&archive);
    assert!(matches!(err, StoreLoadError::WindowArchive(_)), "{err:?}");
}

fn archive_err_any(archive: &[u8]) -> StoreLoadError {
    load_err(&store_with_archive(WINDOW_TOKENS_NPZ, archive))
}

#[test]
fn non_canonical_window_names_are_ambiguous() {
    for name in ["00.npy", "+0.npy", "0", "0.npy.npy", "01.npy"] {
        let archive = raw_archive(&[RawMember::stored(name, synth_u32_npy(&TOKENS))]);
        match archive_err_any(&archive) {
            StoreLoadError::Archive {
                source: ArchiveError::AmbiguousMemberName(found),
                ..
            } => assert_eq!(found, name),
            other => panic!("{name}: expected AmbiguousMemberName, got {other:?}"),
        }
    }
}

/// A central record with its name and extra field both at their largest. The buffer
/// that reads them has to hold the pair.
#[test]
fn largest_name_and_extra_together_load() {
    const NON_ZIP64_EXTRA_ID: [u8; 2] = [0x34, 0x12];
    let name = "n".repeat(usize::from(u16::MAX));
    let extra_body = vec![0u8; usize::from(u16::MAX) - EXTRA_HEADER_LEN];
    let body_len = u16::try_from(extra_body.len()).unwrap();
    let extra = [
        NON_ZIP64_EXTRA_ID.as_slice(),
        &body_len.to_le_bytes(),
        &extra_body,
    ]
    .concat();
    let other = RawMember::stored(&name, Vec::new()).with_leading_extra(&extra);
    assert_loads(&raw_archive(&[other, member()]));
}

/// A last record with no name ends exactly at the directory's end, which is valid.
#[test]
fn record_ending_exactly_at_the_directory_end_loads() {
    assert_loads(&raw_archive(&[member(), RawMember::stored("", Vec::new())]));
}

/// A name too long to be a window is some other member, and doesn't get retained.
#[test]
fn long_member_names_are_ignored() {
    let long_name = "9".repeat(LONGER_THAN_ANY_WINDOW_NAME);
    let other = RawMember::stored(&long_name, b"hello".to_vec());
    assert_loads(&raw_archive(&[other, member()]));
}

#[test]
fn directory_over_the_record_limit_is_refused_before_it_is_read() {
    let archive = raw_archive(&[member(), RawMember::stored("x", Vec::new())]);
    let dir = store_with_archive(WINDOW_TOKENS_NPZ, &archive);
    let limits = StoreLimits {
        max_records: 1,
        ..StoreLimits::default()
    };
    match ApolloStore::load_with_limits(dir.path(), &limits).unwrap_err() {
        StoreLoadError::TooManyRecords { records, limit, .. } => {
            assert_eq!((records, limit), (2, 1));
        }
        other => panic!("expected TooManyRecords, got {other:?}"),
    }
}

#[test]
fn member_whose_data_reaches_into_the_directory_is_refused() {
    // Both sizes agree and fit the file, but the data would run over the directory.
    let extended = member();
    let size = extended.data_len() + 1;
    let archive = raw_archive(&[extended.declaring(size).occupying(size)]);
    assert_archive_error(
        &archive_err(&archive),
        &ArchiveError::BadLocalHeader("data overlaps the central directory"),
    );
}

#[test]
fn local_offset_at_the_directory_or_beyond_is_refused() {
    const LOCAL_OFFSET_AT: usize = 42;
    let mut archive = raw_archive(&[member()]);
    let record = position_of(&archive, CENTRAL_SIGNATURE);
    let past = u32::try_from(archive.len() * 2).unwrap();
    archive[record + LOCAL_OFFSET_AT..record + LOCAL_OFFSET_AT + 4]
        .copy_from_slice(&past.to_le_bytes());
    assert_archive_error(
        &archive_err(&archive),
        &ArchiveError::BadLocalHeader("offset past the central directory"),
    );
}

/// A ZIP64 offset field that a `SeekFrom::Start` could not represent.
#[test]
fn zip64_local_offset_of_two_to_the_63_is_refused_without_seeking() {
    const POW_2_63: u64 = 1 << 63;
    let archive = raw_archive(&[member().with_zip64_local_offset(POW_2_63)]);
    assert_archive_error(
        &archive_err(&archive),
        &ArchiveError::BadLocalHeader("offset past the central directory"),
    );
}

#[test]
fn zip64_local_offset_that_overflows_with_a_prefix_is_refused() {
    let archive = archive_with_surroundings(
        b"prefix",
        &[member().with_zip64_local_offset(u64::MAX)],
        b"",
    );
    assert_archive_error(
        &archive_err(&archive),
        &ArchiveError::BadLocalHeader("offset overflows"),
    );
}

/// A correct ZIP64 offset field must be honoured, as for a member past 4 GiB.
#[test]
fn zip64_local_offset_that_is_correct_loads() {
    assert_loads(&raw_archive(&[member().with_zip64_local_offset(0)]));
}

/// Fields other than ZIP64 come first in some writers' extras, and are skipped.
#[test]
fn unknown_extra_before_the_zip64_field_is_skipped() {
    let unknown = [0x34, 0x12, 2, 0, 0xAA, 0xBB];
    assert_loads(&raw_archive(&[member()
        .zip64()
        .with_leading_extra(&unknown)]));
}

/// The end-record search is limited to the final 64 KiB, so no number of planted
/// signatures can make it expensive. Most of these defer to ZIP64 and so cost
/// extra reads, which is the worst case for it.
#[test]
fn planted_end_signatures_are_refused_within_a_bounded_search() {
    const DEFERRING_CANDIDATE_LEN: usize = 22;
    const CANDIDATES: usize = (u16::MAX as usize) / DEFERRING_CANDIDATE_LEN;
    let mut candidate = END_RECORD_SIGNATURE.to_vec();
    candidate.extend_from_slice(&[0; 4]);
    candidate.extend_from_slice(&[0xFF; 12]);
    candidate.extend_from_slice(&[0; 2]);
    assert_archive_error(
        &archive_err(&candidate.repeat(CANDIDATES)),
        &ArchiveError::NoEndRecord,
    );
}

#[test]
fn non_utf8_member_name_is_ignored_rather_than_an_error() {
    const NON_UTF8_BYTE: u8 = 0xFF;
    let readme = RawMember::stored("x.txt", b"hello".to_vec());
    let mut archive = raw_archive(&[readme, member()]);
    let name_at = position_of(&archive, b"x.txt");
    archive[name_at] = NON_UTF8_BYTE;
    let record = position_of(&archive, CENTRAL_SIGNATURE);
    let name_in_directory = record + CENTRAL_RECORD_LEN;
    archive[name_in_directory] = NON_UTF8_BYTE;
    assert_loads(&archive);
}

#[test]
fn entries_archive_with_no_members_is_missing_file() {
    let dir = TempDir::new().unwrap();
    write_minimal_store(dir.path(), WINDOWS, HIDDEN, WINDOW_SIZE, &one_entry());
    std::fs::write(dir.path().join(ENTRIES_NPZ), raw_archive(&[])).unwrap();
    let err = load_err(&dir);
    assert!(matches!(err, StoreLoadError::MissingFile(_)), "{err:?}");
}
