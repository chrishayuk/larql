//! A reader for the stored (uncompressed) ZIP archives `.npz` files are.
//!
//! `zip::ZipArchive::new` reserves capacity for the record count an archive
//! declares, limited only by the file's logical length and with an infallible
//! `Vec::with_capacity`. A sparse file of a few KiB can therefore abort the
//! process. There is no entry point on its seekable reader that avoids this,
//! and `read_zipfile_from_stream` cannot read data-descriptor or prefixed
//! archives, so the load path does not call into `zip` at all.
//!
//! Only stored members are supported, which is all this crate ever decoded. The
//! parser allocates per record it has actually read, each of which needs 46 real
//! bytes on disk: a hole in the file reads as zeros and fails the signature check.
//!
//! The central directory is authoritative; a member's local header is used only
//! to find where its data starts.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

use super::store::StoreLoadError;
use super::store_file::{read_exactly, try_reserve_for, RegularFile};

const END_RECORD_SIGNATURE: [u8; 4] = *b"PK\x05\x06";
const ZIP64_END_RECORD_SIGNATURE: [u8; 4] = *b"PK\x06\x06";
const ZIP64_LOCATOR_SIGNATURE: [u8; 4] = *b"PK\x06\x07";
const CENTRAL_RECORD_SIGNATURE: [u8; 4] = *b"PK\x01\x02";
const LOCAL_RECORD_SIGNATURE: [u8; 4] = *b"PK\x03\x04";

const END_RECORD_LEN: u64 = 22;
const ZIP64_LOCATOR_LEN: u64 = 20;
/// Without the optional extensible data sector.
const ZIP64_END_RECORD_LEN: u64 = 56;
const CENTRAL_RECORD_LEN: usize = 46;
const LOCAL_RECORD_LEN: usize = 30;
/// The end record's comment length is a `u16`, so it starts this close to the end.
const MAX_END_RECORD_DISTANCE: u64 = END_RECORD_LEN + u16::MAX as u64;

const ZIP64_EXTRA_ID: u16 = 0x0001;
const ZIP64_U16_SENTINEL: u16 = u16::MAX;
const ZIP64_U32_SENTINEL: u32 = u32::MAX;
const METHOD_STORED: u16 = 0;
const FLAG_ENCRYPTED: u16 = 1;

/// Why a file is not an archive this reader can use.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ArchiveError {
    #[error("no consistent end-of-central-directory record")]
    NoEndRecord,
    #[error("central directory record {index} is malformed: {reason}")]
    BadDirectoryRecord { index: u64, reason: &'static str },
    #[error("member's local header is malformed: {0}")]
    BadLocalHeader(&'static str),
    #[error("member uses compression method {0}; only stored members are supported")]
    UnsupportedCompression(u16),
    #[error("member is encrypted")]
    Encrypted,
    #[error("member's data fails its CRC")]
    CrcMismatch,
}

#[derive(Debug)]
struct Member {
    name: String,
    declared: u64,
    stored: u64,
    /// Relative to the start of the archive proper, before any prefix.
    local_offset: u64,
    crc32: u32,
    method: u16,
    flags: u16,
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
    archive_len: u64,
    directory: Directory,
    members: Vec<Member>,
    claimed: u64,
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

/// The ZIP64 values a candidate's sentinel fields defer to, if a locator sits
/// just before it: `(end of directory, directory length, offset, records)`.
fn zip64_fields(file: &mut File, end_at: u64) -> std::io::Result<Option<(u64, u64, u64, u64)>> {
    /// Field offsets in the ZIP64 end record.
    const RECORDS_ON_DISK_AT: usize = 24;
    const RECORDS_AT: usize = 32;
    const DIRECTORY_LEN_AT: usize = 40;
    const DIRECTORY_OFFSET_AT: usize = 48;
    /// Field offset in the locator.
    const END_RECORD_OFFSET_AT: usize = 8;

    let Some(locator_at) = end_at.checked_sub(ZIP64_LOCATOR_LEN) else {
        return Ok(None);
    };
    let mut locator = [0u8; ZIP64_LOCATOR_LEN as usize];
    if !read_at(file, locator_at, &mut locator)? || locator[..4] != ZIP64_LOCATOR_SIGNATURE {
        return Ok(None);
    }
    // The locator's offset is right unless the file has a prefix, in which case
    // the record normally sits immediately before the locator.
    let stated = u64_at(&locator, END_RECORD_OFFSET_AT);
    let adjacent = locator_at.checked_sub(ZIP64_END_RECORD_LEN);
    for at in [Some(stated), adjacent].into_iter().flatten() {
        if at >= locator_at {
            continue;
        }
        let mut record = [0u8; ZIP64_END_RECORD_LEN as usize];
        if read_at(file, at, &mut record)? && record[..4] == ZIP64_END_RECORD_SIGNATURE {
            let records = u64_at(&record, RECORDS_AT);
            if u64_at(&record, RECORDS_ON_DISK_AT) != records {
                return Ok(None);
            }
            return Ok(Some((
                at,
                u64_at(&record, DIRECTORY_LEN_AT),
                u64_at(&record, DIRECTORY_OFFSET_AT),
                records,
            )));
        }
    }
    Ok(None)
}

/// The directory an end record at `end_at` describes, if it is self-consistent.
/// A record can't need more bytes than the directory has, and its first record
/// must really be where the arithmetic says, so a declared count is never
/// believed beyond the file.
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
    let zip64 = if defers_to_zip64 {
        zip64_fields(file, end_at)?
    } else {
        None
    };
    let (directory_end, len, offset, records) = zip64.unwrap_or((
        end_at,
        u64::from(directory_len),
        u64::from(directory_offset),
        u64::from(records),
    ));
    if zip64.is_none() && u16_at(end, RECORDS_ON_DISK_AT) != u16_at(end, RECORDS_AT) {
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
        let mut signature = [0u8; 4];
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

/// Find the last self-consistent end record. Candidates are limited to the final
/// 64 KiB, the only place a valid one can be, so the search is bounded however
/// large the file claims to be and however many signatures it plants.
fn find_directory(regular: &mut RegularFile) -> Result<Directory, StoreLoadError> {
    let path = regular.path.clone();
    let io_err = |source| StoreLoadError::Io {
        path: path.clone(),
        source,
    };
    let tail_len = regular.len.min(MAX_END_RECORD_DISTANCE);
    let tail_at = regular.len - tail_len;
    regular
        .file
        .seek(SeekFrom::Start(tail_at))
        .map_err(io_err)?;
    let tail = read_exactly(&mut regular.file, tail_len, &path)?;
    for (index, window) in tail
        .windows(END_RECORD_LEN as usize)
        .enumerate()
        .rev()
        .filter(|(_, window)| window[..4] == END_RECORD_SIGNATURE)
    {
        let end_at = tail_at + index as u64;
        let found = directory_from_end_record(&mut regular.file, window, end_at, regular.len)
            .map_err(io_err)?;
        if let Some(directory) = found {
            return Ok(directory);
        }
    }
    Err(StoreLoadError::Archive {
        path,
        source: ArchiveError::NoEndRecord,
    })
}

/// Sizes from the ZIP64 extra field. It holds only the fields whose 32-bit
/// values were the sentinel, in this order.
fn apply_zip64_extra(
    extra: &[u8],
    declared: &mut u64,
    stored: &mut u64,
    offset: &mut u64,
) -> Result<(), &'static str> {
    const EXTRA_HEADER_LEN: usize = 4;
    const U64_LEN: usize = 8;
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
            return Ok(());
        }
        rest = &rest[EXTRA_HEADER_LEN + len..];
    }
    Ok(())
}

fn read_members(
    regular: &mut RegularFile,
    directory: Directory,
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

    let path = regular.path.clone();
    let io_err = |source| StoreLoadError::Io {
        path: path.clone(),
        source,
    };
    regular
        .file
        .seek(SeekFrom::Start(directory.start))
        .map_err(io_err)?;
    let mut reader = BufReader::new((&mut regular.file).take(directory.len));
    let mut members = Vec::new();
    let mut variable = Vec::new();
    for index in 0..directory.records {
        let bad = |reason| StoreLoadError::Archive {
            path: path.clone(),
            source: ArchiveError::BadDirectoryRecord { index, reason },
        };
        let mut record = [0u8; CENTRAL_RECORD_LEN];
        let truncated = "runs past the directory";
        match reader.read_exact(&mut record) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Err(bad(truncated)),
            Err(e) => return Err(io_err(e)),
        }
        if record[..4] != CENTRAL_RECORD_SIGNATURE {
            return Err(bad("bad signature"));
        }
        let name_len = usize::from(u16_at(&record, NAME_LEN_AT));
        let extra_len = usize::from(u16_at(&record, EXTRA_LEN_AT));
        let comment_len = usize::from(u16_at(&record, COMMENT_LEN_AT));
        let variable_len = name_len + extra_len + comment_len;
        variable.clear();
        try_reserve_for(&mut variable, variable_len, &path)?;
        let read = (&mut reader)
            .take(variable_len as u64)
            .read_to_end(&mut variable)
            .map_err(io_err)?;
        if read != variable_len {
            return Err(bad(truncated));
        }
        let mut declared = u64::from(u32_at(&record, DECLARED_AT));
        let mut stored = u64::from(u32_at(&record, STORED_AT));
        let mut local_offset = u64::from(u32_at(&record, LOCAL_OFFSET_AT));
        apply_zip64_extra(
            &variable[name_len..name_len + extra_len],
            &mut declared,
            &mut stored,
            &mut local_offset,
        )
        .map_err(bad)?;
        let lossy = String::from_utf8_lossy(&variable[..name_len]);
        let mut name = String::new();
        name.try_reserve_exact(lossy.len())
            .map_err(|_| StoreLoadError::AllocationRefused {
                path: path.clone(),
                bytes: lossy.len() as u64,
            })?;
        name.push_str(&lossy);
        try_reserve_for(&mut members, 1, &path)?;
        members.push(Member {
            name,
            declared,
            stored,
            local_offset,
            crc32: u32_at(&record, CRC_AT),
            method: u16_at(&record, METHOD_AT),
            flags: u16_at(&record, FLAGS_AT),
        });
    }
    Ok(members)
}

impl StoredArchive {
    pub(super) fn open(mut regular: RegularFile) -> Result<Self, StoreLoadError> {
        let directory = find_directory(&mut regular)?;
        let members = read_members(&mut regular, directory)?;
        Ok(Self {
            file: regular.file,
            path: regular.path,
            archive_len: regular.len,
            directory,
            members,
            claimed: 0,
        })
    }

    /// Member names in directory order. Pass a position to [`Self::read_member`].
    pub(super) fn names(&self) -> impl Iterator<Item = &str> {
        self.members.iter().map(|member| member.name.as_str())
    }

    fn archive_err(&self, member_path: &str, source: ArchiveError) -> StoreLoadError {
        StoreLoadError::Archive {
            path: member_path.to_string(),
            source,
        }
    }

    /// Enforce the size rule before anything is allocated for a member:
    ///
    /// 1. Its declared size must equal the size it occupies. Only stored members
    ///    are supported, and for those they are the same number.
    /// 2. The bytes occupied, summed over the members read so far, must fit in the
    ///    file's logical length, taken from `fstat`. Members of a real archive are
    ///    disjoint regions of it, so any honest writer passes, while directory
    ///    records sharing one payload do not.
    ///
    /// This limits member data to the file's logical length, not to what is
    /// allocated on disk: a hole reads as zeros and costs real memory.
    fn admit(
        &mut self,
        member_path: &str,
        declared: u64,
        stored: u64,
    ) -> Result<(), StoreLoadError> {
        if declared != stored {
            return Err(StoreLoadError::MemberSizeInconsistent {
                path: member_path.to_string(),
                declared,
                stored,
            });
        }
        let claimed = self.claimed.saturating_add(stored);
        if claimed > self.archive_len {
            return Err(StoreLoadError::MemberExceedsArchive {
                path: member_path.to_string(),
                claimed,
                archive_len: self.archive_len,
            });
        }
        self.claimed = claimed;
        Ok(())
    }

    /// Where a member's data starts. It must end before the central directory.
    fn locate_data(&mut self, index: usize, member_path: &str) -> Result<u64, StoreLoadError> {
        const NAME_LEN_AT: usize = 26;
        const EXTRA_LEN_AT: usize = 28;
        let member = &self.members[index];
        let (local_offset, stored) = (member.local_offset, member.stored);
        let bad = |reason| StoreLoadError::Archive {
            path: member_path.to_string(),
            source: ArchiveError::BadLocalHeader(reason),
        };
        let header_at = local_offset
            .checked_add(self.directory.prefix)
            .ok_or_else(|| bad("offset overflows"))?;
        if header_at >= self.directory.start {
            return Err(bad("offset past the central directory"));
        }
        let mut header = [0u8; LOCAL_RECORD_LEN];
        let found = read_at(&mut self.file, header_at, &mut header).map_err(|source| {
            StoreLoadError::Io {
                path: member_path.to_string(),
                source,
            }
        })?;
        if !found || header[..4] != LOCAL_RECORD_SIGNATURE {
            return Err(bad("bad signature"));
        }
        let data_at = header_at
            + LOCAL_RECORD_LEN as u64
            + u64::from(u16_at(&header, NAME_LEN_AT))
            + u64::from(u16_at(&header, EXTRA_LEN_AT));
        let ends_before_directory = data_at
            .checked_add(stored)
            .is_some_and(|end| end <= self.directory.start);
        if !ends_before_directory {
            return Err(bad("data overlaps the central directory"));
        }
        Ok(data_at)
    }

    /// Read the member at `index` in [`Self::names`] and check its CRC.
    pub(super) fn read_member(&mut self, index: usize) -> Result<Vec<u8>, StoreLoadError> {
        let member = &self.members[index];
        let member_path = format!("{}::{}", self.path, member.name);
        let (declared, stored, crc32, method, flags) = (
            member.declared,
            member.stored,
            member.crc32,
            member.method,
            member.flags,
        );
        if method != METHOD_STORED {
            return Err(
                self.archive_err(&member_path, ArchiveError::UnsupportedCompression(method))
            );
        }
        if flags & FLAG_ENCRYPTED != 0 {
            return Err(self.archive_err(&member_path, ArchiveError::Encrypted));
        }
        self.admit(&member_path, declared, stored)?;
        let data_at = self.locate_data(index, &member_path)?;
        self.file
            .seek(SeekFrom::Start(data_at))
            .map_err(|source| StoreLoadError::Io {
                path: member_path.clone(),
                source,
            })?;
        let bytes = read_exactly(&mut self.file, stored, &member_path)?;
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&bytes);
        if hasher.finalize() != crc32 {
            return Err(self.archive_err(&member_path, ArchiveError::CrcMismatch));
        }
        Ok(bytes)
    }
}
