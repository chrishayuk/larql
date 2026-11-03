//! A reader for the stored (uncompressed) ZIP archives `.npz` files are.
//!
//! `zip::ZipArchive::new` reserves capacity for the record count an archive
//! declares, limited only by the file's logical length and with an infallible
//! `Vec::with_capacity`. A sparse file of a few KiB can therefore abort the
//! process. There is no entry point on its seekable reader that avoids this,
//! and `read_zipfile_from_stream` cannot read data-descriptor or prefixed
//! archives, so the load path does not call into `zip` at all.
//!
//! Only stored members are supported, which is all this crate ever decoded.
//!
//! The directory is held in one fallible allocation of fixed-size records, capped
//! by `StoreLimits::max_records`, so its cost does not depend on the file's length
//! or on how long member names are. Sorting it is in place.
//!
//! The central directory is authoritative for sizes, and a member's local header
//! must agree with it on the name. On opening, every member's whole extent, from
//! local header to end of data, must be disjoint from every other's and end before
//! the directory. So members cannot share a payload, even with padding or holes
//! leaving the sum of their sizes within the file.

use std::fmt;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use super::store::StoreLoadError;
use super::store_file::{read_exactly, try_reserve_for, RegularFile};

const END_RECORD_SIGNATURE: [u8; 4] = *b"PK\x05\x06";
const ZIP64_END_RECORD_SIGNATURE: [u8; 4] = *b"PK\x06\x06";
const ZIP64_LOCATOR_SIGNATURE: [u8; 4] = *b"PK\x06\x07";
const CENTRAL_RECORD_SIGNATURE: [u8; 4] = *b"PK\x01\x02";
const LOCAL_RECORD_SIGNATURE: [u8; 4] = *b"PK\x03\x04";
const SIGNATURE_LEN: usize = 4;

const END_RECORD_LEN: u64 = 22;
const ZIP64_LOCATOR_LEN: u64 = 20;
/// Without the optional extensible data sector.
const ZIP64_END_RECORD_LEN: u64 = 56;
const CENTRAL_RECORD_LEN: usize = 46;
const LOCAL_RECORD_LEN: usize = 30;
/// The end record's comment length is a `u16`, so it starts this close to the end.
const MAX_END_RECORD_DISTANCE: u64 = END_RECORD_LEN + u16::MAX as u64;
/// A central record's name and extra field together, each with a `u16` length.
const MAX_VARIABLE_LEN: usize = 2 * u16::MAX as usize;

const ZIP64_EXTRA_ID: u16 = 0x0001;
const ZIP64_U16_SENTINEL: u16 = u16::MAX;
const ZIP64_U32_SENTINEL: u32 = u32::MAX;
const METHOD_STORED: u16 = 0;
const FLAG_ENCRYPTED: u16 = 1;
/// Bits a reader of stored data cannot honour: compressed patched data and strong
/// encryption. `zipfile` refuses both.
const FLAGS_UNSUPPORTED: u16 = (1 << 5) | (1 << 6);
/// A window's name is at most `usize` digits and `.npy`, and `entries.npy` is
/// shorter. Longer names can't be either, so they are kept as "other".
const MAX_RETAINED_NAME: usize = 24;

/// Why a file is not an archive this reader can use.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ArchiveError {
    #[error("no consistent end-of-central-directory record")]
    NoEndRecord,
    #[error("central directory record {index} is malformed: {reason}")]
    BadDirectoryRecord { index: u64, reason: &'static str },
    #[error("member's local header is malformed: {0}")]
    BadLocalHeader(&'static str),
    #[error("members overlap: one ends at offset {end}, the next starts at {next_start}")]
    OverlappingMembers { end: u64, next_start: u64 },
    #[error("member name {0:?} is ambiguous")]
    AmbiguousMemberName(String),
    #[error("member uses compression method {0}; only stored members are supported")]
    UnsupportedCompression(u16),
    #[error("member is encrypted or uses unsupported flags")]
    Encrypted,
    #[error("member's data fails its CRC")]
    CrcMismatch,
}

/// One member, in a fixed-size record with no allocation of its own.
#[derive(Debug, Clone, Copy)]
struct Member {
    declared: u64,
    stored: u64,
    /// Absolute offset of the local header.
    header_at: u64,
    /// Absolute offset of the data.
    data_at: u64,
    crc32: u32,
    method: u16,
    flags: u16,
    name: [u8; MAX_RETAINED_NAME],
    /// `None` for a name longer than [`MAX_RETAINED_NAME`].
    name_len: Option<u8>,
}

impl Member {
    fn name(&self) -> Option<&[u8]> {
        self.name_len.map(|len| &self.name[..usize::from(len)])
    }
}

/// Where the central directory is, from a validated end record.
#[derive(Debug, Clone, Copy)]
struct Directory {
    start: u64,
    len: u64,
    records: u64,
    /// Bytes before the archive proper, as in a self-extracting file.
    prefix: u64,
}

#[derive(Debug)]
pub(super) struct StoredArchive {
    file: File,
    path: String,
    members: Vec<Member>,
}

/// `archive::member`, built only if an error needs it.
struct MemberPath<'a> {
    archive: &'a str,
    name: Option<&'a [u8]>,
}

impl fmt::Display for MemberPath<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.name {
            Some(name) => write!(f, "{}::{}", self.archive, String::from_utf8_lossy(name)),
            None => write!(f, "{}::<long name>", self.archive),
        }
    }
}

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([bytes[at], bytes[at + 1]])
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn u64_at(bytes: &[u8], at: usize) -> u64 {
    let low = u64::from(u32_at(bytes, at));
    let high = u64::from(u32_at(bytes, at + 4));
    low | (high << u32::BITS)
}

/// `Ok(false)` at end of file, so a candidate that runs off the end is simply
/// not a candidate.
fn read_at(file: &mut File, at: u64, buf: &mut [u8]) -> std::io::Result<bool> {
    file.seek(SeekFrom::Start(at))?;
    match file.read_exact(buf) {
        Ok(()) => Ok(true),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => Ok(false),
        Err(e) => Err(e),
    }
}

/// The values in a ZIP64 end record, located through its locator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Zip64End {
    at: u64,
    directory_len: u64,
    directory_offset: u64,
    records: u64,
}

/// What a locator just before the end record leads to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Zip64Location {
    /// No locator, or none describing a single-disk ZIP64 record.
    Absent,
    Found {
        end: Zip64End,
        /// Whether the record sits immediately before the locator, where `zipfile`
        /// always looks. A record found only at the stated offset has an extensible
        /// data sector after it.
        adjacent: bool,
    },
    /// The locator's stated record and the adjacent one are different records.
    /// `zipfile` reads the adjacent one and `zip` the stated one.
    Conflicting,
}

fn zip64_record_at(
    file: &mut File,
    at: u64,
) -> std::io::Result<Option<[u8; ZIP64_END_RECORD_LEN as usize]>> {
    let mut record = [0u8; ZIP64_END_RECORD_LEN as usize];
    let found =
        read_at(file, at, &mut record)? && record[..SIGNATURE_LEN] == ZIP64_END_RECORD_SIGNATURE;
    Ok(found.then_some(record))
}

/// The ZIP64 end record whose locator sits just before `end_at`.
fn zip64_end(file: &mut File, end_at: u64) -> std::io::Result<Zip64Location> {
    /// Field offsets in the ZIP64 end record.
    const DISK_AT: usize = 16;
    const DIRECTORY_DISK_AT: usize = 20;
    const RECORDS_ON_DISK_AT: usize = 24;
    const RECORDS_AT: usize = 32;
    const DIRECTORY_LEN_AT: usize = 40;
    const DIRECTORY_OFFSET_AT: usize = 48;
    /// Field offsets in the locator.
    const END_RECORD_DISK_AT: usize = 4;
    const END_RECORD_OFFSET_AT: usize = 8;
    const TOTAL_DISKS_AT: usize = 16;
    /// Some writers leave the total at zero for a single disk.
    const MAX_TOTAL_DISKS: u32 = 1;

    let Some(locator_at) = end_at.checked_sub(ZIP64_LOCATOR_LEN) else {
        return Ok(Zip64Location::Absent);
    };
    let mut locator = [0u8; ZIP64_LOCATOR_LEN as usize];
    if !read_at(file, locator_at, &mut locator)?
        || locator[..SIGNATURE_LEN] != ZIP64_LOCATOR_SIGNATURE
    {
        return Ok(Zip64Location::Absent);
    }
    if u32_at(&locator, END_RECORD_DISK_AT) != 0
        || u32_at(&locator, TOTAL_DISKS_AT) > MAX_TOTAL_DISKS
    {
        return Ok(Zip64Location::Absent);
    }
    // The locator's offset is right unless the file has a prefix, in which case the
    // record normally sits immediately before the locator.
    let stated_at = Some(u64_at(&locator, END_RECORD_OFFSET_AT)).filter(|at| *at < locator_at);
    let adjacent_at = locator_at.checked_sub(ZIP64_END_RECORD_LEN);
    let stated = match stated_at {
        Some(at) => zip64_record_at(file, at)?.map(|record| (at, record)),
        None => None,
    };
    let adjacent = match adjacent_at {
        Some(at) => zip64_record_at(file, at)?.map(|record| (at, record)),
        None => None,
    };
    let (at, record, is_adjacent) = match (stated, adjacent) {
        (Some((stated_at, _)), Some((adjacent_at, _))) if stated_at != adjacent_at => {
            return Ok(Zip64Location::Conflicting);
        }
        (_, Some((at, record))) => (at, record, true),
        (Some((at, record)), None) => (at, record, false),
        (None, None) => return Ok(Zip64Location::Absent),
    };
    let records = u64_at(&record, RECORDS_AT);
    if u64_at(&record, RECORDS_ON_DISK_AT) != records
        || u32_at(&record, DISK_AT) != 0
        || u32_at(&record, DIRECTORY_DISK_AT) != 0
    {
        return Ok(Zip64Location::Absent);
    }
    Ok(Zip64Location::Found {
        end: Zip64End {
            at,
            directory_len: u64_at(&record, DIRECTORY_LEN_AT),
            directory_offset: u64_at(&record, DIRECTORY_OFFSET_AT),
            records,
        },
        adjacent: is_adjacent,
    })
}

/// The directory an end record at `end_at` describes, if it is self-consistent.
/// A record can't need more bytes than the directory has, and its first record
/// must really be where the arithmetic says, so a declared count is never
/// believed beyond the file.
///
/// A locator next to an end record that doesn't defer to it is legal, and the directory
/// then ends at the ZIP64 record. `zipfile` honours the adjacent record's values while
/// `zip` uses the end record's. That is only one answer if the ZIP64 record is adjacent
/// and they agree, so anything else is refused.
fn directory_from_end_record(
    file: &mut File,
    end: &[u8],
    end_at: u64,
    file_len: u64,
) -> std::io::Result<Option<Directory>> {
    /// Field offsets in the end record.
    const DISK_AT: usize = 4;
    const DIRECTORY_DISK_AT: usize = 6;
    const RECORDS_ON_DISK_AT: usize = 8;
    const RECORDS_AT: usize = 10;
    const DIRECTORY_LEN_AT: usize = 12;
    const DIRECTORY_OFFSET_AT: usize = 16;
    const COMMENT_LEN_AT: usize = 20;

    let comment_len = u64::from(u16_at(end, COMMENT_LEN_AT));
    if end_at + END_RECORD_LEN + comment_len > file_len
        || u16_at(end, DISK_AT) != u16_at(end, DIRECTORY_DISK_AT)
    {
        return Ok(None);
    }
    let records = u16_at(end, RECORDS_AT);
    let directory_len = u32_at(end, DIRECTORY_LEN_AT);
    let directory_offset = u32_at(end, DIRECTORY_OFFSET_AT);
    let defers_to_zip64 = records == ZIP64_U16_SENTINEL
        || directory_len == ZIP64_U32_SENTINEL
        || directory_offset == ZIP64_U32_SENTINEL;
    let used_zip64 = match zip64_end(file, end_at)? {
        Zip64Location::Absent => None,
        Zip64Location::Conflicting => return Ok(None),
        Zip64Location::Found {
            end: zip64,
            adjacent,
        } => {
            let agrees = zip64.directory_len == u64::from(directory_len)
                && zip64.directory_offset == u64::from(directory_offset)
                && zip64.records == u64::from(records);
            if !defers_to_zip64 && !(adjacent && agrees) {
                return Ok(None);
            }
            Some(zip64)
        }
    };
    let (directory_end, len, offset, records) = used_zip64.map_or(
        (
            end_at,
            u64::from(directory_len),
            u64::from(directory_offset),
            u64::from(records),
        ),
        |zip64| {
            (
                zip64.at,
                zip64.directory_len,
                zip64.directory_offset,
                zip64.records,
            )
        },
    );
    if used_zip64.is_none() && u16_at(end, RECORDS_ON_DISK_AT) != u16_at(end, RECORDS_AT) {
        return Ok(None);
    }
    let Some(start) = directory_end.checked_sub(len) else {
        return Ok(None);
    };
    let Some(prefix) = start.checked_sub(offset) else {
        return Ok(None);
    };
    let fits = records
        .checked_mul(CENTRAL_RECORD_LEN as u64)
        .is_some_and(|minimum| minimum <= len);
    if !fits {
        return Ok(None);
    }
    if records > 0 {
        let mut signature = [0u8; SIGNATURE_LEN];
        if !read_at(file, start, &mut signature)? || signature != CENTRAL_RECORD_SIGNATURE {
            return Ok(None);
        }
    }
    Ok(Some(Directory {
        start,
        len,
        records,
        prefix,
    }))
}

/// Use the last end-record signature in the final 64 KiB, the only place a valid one
/// can be, or nothing. There is no fallback to an earlier one: which record is used
/// mustn't depend on which this reader calls consistent. A comment holding a bogus
/// signature is refused, as `zipfile` does.
fn find_directory(regular: &mut RegularFile) -> Result<Directory, StoreLoadError> {
    let path = regular.path.as_str();
    let io_err = |source| StoreLoadError::Io {
        path: path.to_string(),
        source,
    };
    let tail_len = regular.len.min(MAX_END_RECORD_DISTANCE);
    let tail_at = regular.len - tail_len;
    regular
        .file
        .seek(SeekFrom::Start(tail_at))
        .map_err(io_err)?;
    let tail = read_exactly(&mut regular.file, tail_len, &path)?;
    let last_signature = tail
        .windows(SIGNATURE_LEN)
        .rposition(|window| window == END_RECORD_SIGNATURE);
    let end_record = last_signature.and_then(|index| Some((index, tail.get(index..)?)));
    if let Some((index, end)) = end_record.filter(|(_, end)| end.len() >= END_RECORD_LEN as usize) {
        let end_at = tail_at + index as u64;
        let found = directory_from_end_record(&mut regular.file, end, end_at, regular.len)
            .map_err(io_err)?;
        if let Some(directory) = found {
            return Ok(directory);
        }
    }
    Err(StoreLoadError::Archive {
        path: path.to_string(),
        source: ArchiveError::NoEndRecord,
    })
}

/// A Unicode Path extra field naming the member differently, when either name is plain
/// ASCII. `zip` and `zipfile` use this name and this reader doesn't, so a window or
/// `entries.npy` could be two different members. Info-ZIP writes one only for a
/// non-ASCII name, whose other encoding is expected to differ and can't be a window
/// or `entries.npy`.
fn unicode_path_conflicts(extra: &[u8], name: &[u8]) -> bool {
    const EXTRA_HEADER_LEN: usize = 4;
    const UNICODE_PATH_ID: u16 = 0x7075;
    /// A version byte and the CRC of the name it replaces.
    const UNICODE_PATH_NAME_AT: usize = 5;
    let mut rest = extra;
    while rest.len() >= EXTRA_HEADER_LEN {
        let id = u16_at(rest, 0);
        let end = EXTRA_HEADER_LEN + usize::from(u16_at(rest, 2));
        let Some(body) = rest.get(EXTRA_HEADER_LEN..end) else {
            return false;
        };
        if id == UNICODE_PATH_ID {
            let path = body.get(UNICODE_PATH_NAME_AT..).unwrap_or_default();
            if path != name && (path.is_ascii() || name.is_ascii()) {
                return true;
            }
        }
        rest = &rest[end..];
    }
    false
}

/// Sizes from the ZIP64 extra field. It holds only the fields whose 32-bit
/// values were the sentinel, in this order. `zip` instead reads all three in order
/// from any field at least that long, so if it would see other values it's refused.
fn apply_zip64_extra(
    extra: &[u8],
    declared: &mut u64,
    stored: &mut u64,
    offset: &mut u64,
) -> Result<(), &'static str> {
    const EXTRA_HEADER_LEN: usize = 4;
    const U64_LEN: usize = 8;
    const ALL_THREE_FIELDS_LEN: usize = 3 * U64_LEN;
    let sentinel = u64::from(ZIP64_U32_SENTINEL);
    let mut rest = extra;
    while rest.len() >= EXTRA_HEADER_LEN {
        let id = u16_at(rest, 0);
        let len = usize::from(u16_at(rest, 2));
        let Some(body) = rest.get(EXTRA_HEADER_LEN..EXTRA_HEADER_LEN + len) else {
            return Err("extra field overruns its record");
        };
        if id == ZIP64_EXTRA_ID {
            let mut at = 0;
            for field in [&mut *declared, &mut *stored, &mut *offset] {
                if *field == sentinel {
                    let Some(bytes) = body.get(at..at + U64_LEN) else {
                        return Err("ZIP64 extra field too short");
                    };
                    *field = u64_at(bytes, 0);
                    at += U64_LEN;
                }
            }
            if body.len() >= ALL_THREE_FIELDS_LEN {
                let as_zip_reads = [
                    u64_at(body, 0),
                    u64_at(body, U64_LEN),
                    u64_at(body, 2 * U64_LEN),
                ];
                if as_zip_reads != [*declared, *stored, *offset] {
                    return Err("ZIP64 extra field disagrees with the record");
                }
            }
            return Ok(());
        }
        rest = &rest[EXTRA_HEADER_LEN + len..];
    }
    Ok(())
}

/// Validate a member's local header: its name must match the directory's and its
/// data must end before the directory. Returns the offset of the data, or why the
/// header is unusable; a failed read is an I/O error, not a verdict on the header.
fn validate_local_header(
    file: &mut File,
    directory: Directory,
    header_at: u64,
    stored: u64,
    central_name: &[u8],
    scratch: &mut [u8],
) -> std::io::Result<Result<u64, &'static str>> {
    const NAME_LEN_AT: usize = 26;
    const EXTRA_LEN_AT: usize = 28;
    const NAME_DIFFERS: &str = "name differs from the directory";
    if header_at >= directory.start {
        return Ok(Err("offset past the central directory"));
    }
    let mut header = [0u8; LOCAL_RECORD_LEN];
    if !read_at(file, header_at, &mut header)? || header[..SIGNATURE_LEN] != LOCAL_RECORD_SIGNATURE
    {
        return Ok(Err("bad signature"));
    }
    let name_len = usize::from(u16_at(&header, NAME_LEN_AT));
    if name_len != central_name.len() {
        return Ok(Err(NAME_DIFFERS));
    }
    let local_name = &mut scratch[..name_len];
    let name_at = header_at + LOCAL_RECORD_LEN as u64;
    if !read_at(file, name_at, local_name)? || *local_name != *central_name {
        return Ok(Err(NAME_DIFFERS));
    }
    let data_at = name_at + name_len as u64 + u64::from(u16_at(&header, EXTRA_LEN_AT));
    let ends_before_directory = data_at
        .checked_add(stored)
        .is_some_and(|end| end <= directory.start);
    if !ends_before_directory {
        return Ok(Err("data overlaps the central directory"));
    }
    Ok(Ok(data_at))
}

fn read_members(
    regular: &mut RegularFile,
    directory: Directory,
    max_records: u64,
) -> Result<Vec<Member>, StoreLoadError> {
    /// Field offsets in a central directory record.
    const FLAGS_AT: usize = 8;
    const METHOD_AT: usize = 10;
    const CRC_AT: usize = 16;
    const STORED_AT: usize = 20;
    const DECLARED_AT: usize = 24;
    const NAME_LEN_AT: usize = 28;
    const EXTRA_LEN_AT: usize = 30;
    const COMMENT_LEN_AT: usize = 32;
    const LOCAL_OFFSET_AT: usize = 42;

    let path = regular.path.as_str();
    if directory.records > max_records {
        return Err(StoreLoadError::TooManyRecords {
            path: path.to_string(),
            records: directory.records,
            limit: max_records,
        });
    }
    let io_err = |source| StoreLoadError::Io {
        path: path.to_string(),
        source,
    };
    let records =
        usize::try_from(directory.records).map_err(|_| StoreLoadError::TooManyRecords {
            path: path.to_string(),
            records: directory.records,
            limit: max_records,
        })?;
    let mut members = Vec::new();
    try_reserve_for(&mut members, records, &path)?;
    // Two buffers of a fixed size, reused for every record.
    let mut variable = Vec::new();
    try_reserve_for(&mut variable, MAX_VARIABLE_LEN, &path)?;
    variable.resize(MAX_VARIABLE_LEN, 0u8);
    let mut local_name = Vec::new();
    try_reserve_for(&mut local_name, usize::from(u16::MAX), &path)?;
    local_name.resize(usize::from(u16::MAX), 0u8);

    let directory_end = directory.start + directory.len;
    let mut cursor = directory.start;
    for index in 0..directory.records {
        let bad = |reason| StoreLoadError::Archive {
            path: path.to_string(),
            source: ArchiveError::BadDirectoryRecord { index, reason },
        };
        let truncated = "runs past the directory";
        let after_fixed = cursor + CENTRAL_RECORD_LEN as u64;
        if after_fixed > directory_end {
            return Err(bad(truncated));
        }
        let mut record = [0u8; CENTRAL_RECORD_LEN];
        if !read_at(&mut regular.file, cursor, &mut record).map_err(io_err)? {
            return Err(bad(truncated));
        }
        if record[..SIGNATURE_LEN] != CENTRAL_RECORD_SIGNATURE {
            return Err(bad("bad signature"));
        }
        let name_len = usize::from(u16_at(&record, NAME_LEN_AT));
        let extra_len = usize::from(u16_at(&record, EXTRA_LEN_AT));
        let comment_len = usize::from(u16_at(&record, COMMENT_LEN_AT));
        let next = after_fixed + (name_len + extra_len + comment_len) as u64;
        if next > directory_end {
            return Err(bad(truncated));
        }
        let name_and_extra = &mut variable[..name_len + extra_len];
        if !read_at(&mut regular.file, after_fixed, name_and_extra).map_err(io_err)? {
            return Err(bad(truncated));
        }
        let (central_name, extra) = name_and_extra.split_at(name_len);
        // `zipfile` cuts a name at its first NUL, so this can be another member's name.
        if central_name.contains(&0) || unicode_path_conflicts(extra, central_name) {
            return Err(StoreLoadError::Archive {
                path: path.to_string(),
                source: ArchiveError::AmbiguousMemberName(
                    String::from_utf8_lossy(central_name).into_owned(),
                ),
            });
        }
        let mut declared = u64::from(u32_at(&record, DECLARED_AT));
        let mut stored = u64::from(u32_at(&record, STORED_AT));
        let mut local_offset = u64::from(u32_at(&record, LOCAL_OFFSET_AT));
        apply_zip64_extra(extra, &mut declared, &mut stored, &mut local_offset).map_err(bad)?;
        let header_at =
            local_offset
                .checked_add(directory.prefix)
                .ok_or_else(|| StoreLoadError::Archive {
                    path: path.to_string(),
                    source: ArchiveError::BadLocalHeader("offset overflows"),
                })?;
        let data_at = validate_local_header(
            &mut regular.file,
            directory,
            header_at,
            stored,
            central_name,
            &mut local_name,
        )
        .map_err(io_err)?
        .map_err(|reason| StoreLoadError::Archive {
            path: path.to_string(),
            source: ArchiveError::BadLocalHeader(reason),
        })?;
        let mut name = [0u8; MAX_RETAINED_NAME];
        let retained = (central_name.len() <= MAX_RETAINED_NAME).then(|| {
            name[..central_name.len()].copy_from_slice(central_name);
            central_name.len() as u8
        });
        members.push(Member {
            declared,
            stored,
            header_at,
            data_at,
            crc32: u32_at(&record, CRC_AT),
            method: u16_at(&record, METHOD_AT),
            flags: u16_at(&record, FLAGS_AT),
            name,
            name_len: retained,
        });
        cursor = next;
    }
    // `zip` stops at the record count and `zipfile` at the directory's length, so a
    // directory with bytes left over means something different to each.
    if cursor != directory_end {
        return Err(StoreLoadError::Archive {
            path: path.to_string(),
            source: ArchiveError::BadDirectoryRecord {
                index: directory.records,
                reason: "directory continues past its record count",
            },
        });
    }
    Ok(members)
}

/// Refuse any two members whose extents overlap. Sorting in place allocates nothing,
/// and it's `O(n log n)`, against a directory already capped by `max_records`.
fn refuse_overlaps(members: &mut [Member], path: &str) -> Result<(), StoreLoadError> {
    members.sort_unstable_by_key(|member| member.header_at);
    for pair in members.windows(2) {
        let end = pair[0].data_at + pair[0].stored;
        if end > pair[1].header_at {
            return Err(StoreLoadError::Archive {
                path: path.to_string(),
                source: ArchiveError::OverlappingMembers {
                    end,
                    next_start: pair[1].header_at,
                },
            });
        }
    }
    Ok(())
}

impl StoredArchive {
    pub(super) fn open(mut regular: RegularFile, max_records: u64) -> Result<Self, StoreLoadError> {
        let directory = find_directory(&mut regular)?;
        let mut members = read_members(&mut regular, directory, max_records)?;
        refuse_overlaps(&mut members, &regular.path)?;
        Ok(Self {
            file: regular.file,
            path: regular.path,
            members,
        })
    }

    /// Member names, in the order of their data. `None` for a name too long to be
    /// a window or the entries. Pass a position to [`Self::read_member`].
    pub(super) fn names(&self) -> impl Iterator<Item = Option<&[u8]>> {
        self.members.iter().map(Member::name)
    }

    fn member_path(&self, index: usize) -> MemberPath<'_> {
        MemberPath {
            archive: &self.path,
            name: self.members[index].name(),
        }
    }

    /// The name to report for the member at `index`.
    pub(super) fn member_display(&self, index: usize) -> impl fmt::Display + '_ {
        self.member_path(index)
    }

    /// Read the member at `index` in [`Self::names`] and check its CRC. Its declared
    /// size must equal the size it occupies: only stored members are supported, and
    /// for those they are the same number.
    pub(super) fn read_member(&mut self, index: usize) -> Result<Vec<u8>, StoreLoadError> {
        let member = self.members[index];
        let archive_err = |source| StoreLoadError::Archive {
            path: self.member_path(index).to_string(),
            source,
        };
        if member.method != METHOD_STORED {
            return Err(archive_err(ArchiveError::UnsupportedCompression(
                member.method,
            )));
        }
        if member.flags & (FLAG_ENCRYPTED | FLAGS_UNSUPPORTED) != 0 {
            return Err(archive_err(ArchiveError::Encrypted));
        }
        if member.declared != member.stored {
            return Err(StoreLoadError::MemberSizeInconsistent {
                path: self.member_path(index).to_string(),
                declared: member.declared,
                stored: member.stored,
            });
        }
        let mut file = &self.file;
        file.seek(SeekFrom::Start(member.data_at))
            .map_err(|source| StoreLoadError::Io {
                path: self.member_path(index).to_string(),
                source,
            })?;
        let bytes = read_exactly(file, member.stored, &self.member_path(index))?;
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&bytes);
        if hasher.finalize() != member.crc32 {
            return Err(archive_err(ArchiveError::CrcMismatch));
        }
        Ok(bytes)
    }
}
