//! A hand-written ZIP encoder for tests that need an archive to lie.
//!
//! `zip::ZipWriter` only writes the truth, so it cannot produce a member whose
//! declared sizes disagree with its data. Only the fields the reader consults
//! are populated; timestamps are the DOS epoch.

const LOCAL_HEADER_SIGNATURE: u32 = 0x0403_4b50;
const CENTRAL_HEADER_SIGNATURE: u32 = 0x0201_4b50;
const END_OF_DIRECTORY_SIGNATURE: u32 = 0x0605_4b50;
/// The value that defers a 32-bit field to the ZIP64 extra field.
const ZIP64_SIZE_SENTINEL: u32 = u32::MAX;
const ZIP64_EXTRA_ID: u16 = 0x0001;
/// Uncompressed size then compressed size, `u64` each.
const ZIP64_EXTRA_SIZES_LEN: u16 = 16;
const VERSION_ZIP32: u16 = 20;
const VERSION_ZIP64: u16 = 45;
const NO_FLAGS: u16 = 0;
const DOS_TIME_MIDNIGHT: u16 = 0;
/// 1980-01-01, the earliest date a ZIP timestamp can hold.
const DOS_DATE_EPOCH: u16 = 0x21;
const METHOD_STORED: u16 = 0;
pub(super) const METHOD_DEFLATED: u16 = 8;
const CRC32_REFLECTED_POLYNOMIAL: u32 = 0xEDB8_8320;

/// One central-directory record, honest until told otherwise.
#[derive(Clone)]
pub(super) struct RawMember {
    name: String,
    data: Vec<u8>,
    /// CRC of what the reader will see, which differs from `data` when deflated.
    crc_of: Vec<u8>,
    declared: u64,
    stored: u64,
    method: u16,
    flags: u16,
    zip64: bool,
    /// Carried in the ZIP64 extra in place of the record's real offset.
    zip64_local_offset: Option<u64>,
    leading_extra: Vec<u8>,
}

impl RawMember {
    pub(super) fn stored(name: &str, data: Vec<u8>) -> Self {
        let len = data.len() as u64;
        Self {
            name: name.to_string(),
            crc_of: data.clone(),
            data,
            declared: len,
            stored: len,
            method: METHOD_STORED,
            flags: NO_FLAGS,
            zip64: false,
            zip64_local_offset: None,
            leading_extra: Vec::new(),
        }
    }

    /// `plain` wrapped in one uncompressed DEFLATE block: a valid member of
    /// method 8, needing no compressor in the test.
    pub(super) fn deflate_wrapped(name: &str, plain: Vec<u8>) -> Self {
        /// BFINAL=1, BTYPE=00.
        const FINAL_STORED_BLOCK: u8 = 0x01;
        let len = u16::try_from(plain.len()).expect("fits one DEFLATE block");
        let mut data = vec![FINAL_STORED_BLOCK];
        data.extend_from_slice(&len.to_le_bytes());
        data.extend_from_slice(&(!len).to_le_bytes());
        data.extend_from_slice(&plain);
        Self {
            name: name.to_string(),
            declared: plain.len() as u64,
            stored: data.len() as u64,
            crc_of: plain,
            data,
            method: METHOD_DEFLATED,
            flags: NO_FLAGS,
            zip64: false,
            zip64_local_offset: None,
            leading_extra: Vec::new(),
        }
    }

    pub(super) fn declaring(mut self, size: u64) -> Self {
        self.declared = size;
        self
    }

    pub(super) fn occupying(mut self, size: u64) -> Self {
        self.stored = size;
        self
    }

    pub(super) fn zip64(mut self) -> Self {
        self.zip64 = true;
        self
    }

    pub(super) fn with_zip64_local_offset(mut self, offset: u64) -> Self {
        self.zip64 = true;
        self.zip64_local_offset = Some(offset);
        self
    }

    pub(super) fn with_leading_extra(mut self, extra: &[u8]) -> Self {
        self.leading_extra = extra.to_vec();
        self
    }

    pub(super) fn with_flags(mut self, flags: u16) -> Self {
        self.flags = flags;
        self
    }

    pub(super) fn named(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    pub(super) fn data_len(&self) -> u64 {
        self.data.len() as u64
    }

    pub(super) fn declared(&self) -> u64 {
        self.declared
    }

    fn version(&self) -> u16 {
        if self.zip64 {
            VERSION_ZIP64
        } else {
            VERSION_ZIP32
        }
    }

    /// `(declared, stored)` as the 32-bit header fields carry them.
    fn header_sizes(&self) -> (u32, u32) {
        if self.zip64 {
            (ZIP64_SIZE_SENTINEL, ZIP64_SIZE_SENTINEL)
        } else {
            (
                u32::try_from(self.declared).expect("non-ZIP64 size fits 32 bits"),
                u32::try_from(self.stored).expect("non-ZIP64 size fits 32 bits"),
            )
        }
    }

    fn extra(&self) -> Vec<u8> {
        let mut extra = self.leading_extra.clone();
        if !self.zip64 {
            return extra;
        }
        let offset = self.zip64_local_offset.map(u64::to_le_bytes);
        let body_len = ZIP64_EXTRA_SIZES_LEN + offset.map_or(0, |o| o.len() as u16);
        extra.extend_from_slice(&ZIP64_EXTRA_ID.to_le_bytes());
        extra.extend_from_slice(&body_len.to_le_bytes());
        extra.extend_from_slice(&self.declared.to_le_bytes());
        extra.extend_from_slice(&self.stored.to_le_bytes());
        if let Some(offset) = offset {
            extra.extend_from_slice(&offset);
        }
        extra
    }

    /// Version through extra-field length: the run the local and central
    /// headers share.
    fn common_fields(&self) -> Vec<u8> {
        let (declared, stored) = self.header_sizes();
        let mut out = Vec::new();
        for field in [
            self.version(),
            self.flags,
            self.method,
            DOS_TIME_MIDNIGHT,
            DOS_DATE_EPOCH,
        ] {
            out.extend_from_slice(&field.to_le_bytes());
        }
        out.extend_from_slice(&crc32(&self.crc_of).to_le_bytes());
        out.extend_from_slice(&stored.to_le_bytes());
        out.extend_from_slice(&declared.to_le_bytes());
        let name_len = u16::try_from(self.name.len()).expect("short test name");
        let extra_len = u16::try_from(self.extra().len()).expect("short extra");
        out.extend_from_slice(&name_len.to_le_bytes());
        out.extend_from_slice(&extra_len.to_le_bytes());
        out
    }

    fn local_record(&self) -> Vec<u8> {
        let mut out = LOCAL_HEADER_SIGNATURE.to_le_bytes().to_vec();
        out.extend_from_slice(&self.common_fields());
        out.extend_from_slice(self.name.as_bytes());
        out.extend_from_slice(&self.extra());
        out.extend_from_slice(&self.data);
        out
    }

    fn central_record(&self, local_offset: u32) -> Vec<u8> {
        /// Comment length, disk number, internal attributes.
        const ZEROED_U16_FIELDS: [u8; 6] = [0; 6];
        const EXTERNAL_ATTRIBUTES: u32 = 0;
        let mut out = CENTRAL_HEADER_SIGNATURE.to_le_bytes().to_vec();
        out.extend_from_slice(&self.version().to_le_bytes());
        out.extend_from_slice(&self.common_fields());
        out.extend_from_slice(&ZEROED_U16_FIELDS);
        out.extend_from_slice(&EXTERNAL_ATTRIBUTES.to_le_bytes());
        let local_offset = if self.zip64_local_offset.is_some() {
            ZIP64_SIZE_SENTINEL
        } else {
            local_offset
        };
        out.extend_from_slice(&local_offset.to_le_bytes());
        out.extend_from_slice(self.name.as_bytes());
        out.extend_from_slice(&self.extra());
        out
    }
}

fn end_of_directory(records: usize, directory_len: usize, directory_offset: usize) -> Vec<u8> {
    const ZERO_U16: [u8; 2] = [0; 2];
    let records = u16::try_from(records).expect("few test records");
    let mut out = END_OF_DIRECTORY_SIGNATURE.to_le_bytes().to_vec();
    out.extend_from_slice(&ZERO_U16); // this disk
    out.extend_from_slice(&ZERO_U16); // directory disk
    out.extend_from_slice(&records.to_le_bytes());
    out.extend_from_slice(&records.to_le_bytes());
    out.extend_from_slice(&u32::try_from(directory_len).unwrap().to_le_bytes());
    out.extend_from_slice(&u32::try_from(directory_offset).unwrap().to_le_bytes());
    out.extend_from_slice(&ZERO_U16); // comment length
    out
}

/// The members laid out one after another, as an honest writer does.
pub(super) fn raw_archive(members: &[RawMember]) -> Vec<u8> {
    let mut body = Vec::new();
    let mut directory = Vec::new();
    for member in members {
        let offset = u32::try_from(body.len()).unwrap();
        body.extend_from_slice(&member.local_record());
        directory.extend_from_slice(&member.central_record(offset));
    }
    finish(body, directory, members.len())
}

/// `prefix` before the archive and a `comment` after its end record. Offsets stay
/// relative to the archive proper, as they do in a self-extracting file.
pub(super) fn archive_with_surroundings(
    prefix: &[u8],
    members: &[RawMember],
    comment: &[u8],
) -> Vec<u8> {
    let mut out = prefix.to_vec();
    out.extend_from_slice(&raw_archive(members));
    const COMMENT_LEN_LEN: usize = 2;
    let comment_len_at = out.len() - COMMENT_LEN_LEN;
    let comment_len = u16::try_from(comment.len()).expect("short comment");
    out[comment_len_at..].copy_from_slice(&comment_len.to_le_bytes());
    out.extend_from_slice(comment);
    out
}

/// One member described by ZIP64 end records. `sector` pads the ZIP64 record's
/// extensible data, which moves it away from where a prefix-tolerant reader looks.
pub(super) fn zip64_directory_archive(
    prefix: &[u8],
    member: &RawMember,
    declared_records: u64,
    sector: &[u8],
) -> Vec<u8> {
    const ZIP64_END_SIGNATURE: u32 = 0x0606_4b50;
    const ZIP64_LOCATOR_SIGNATURE: u32 = 0x0706_4b50;
    /// The ZIP64 end record's size field excludes its first 12 bytes.
    const ZIP64_END_FIXED_SIZE: u64 = 44;
    const SINGLE_DISK: u32 = 1;
    let body = member.local_record();
    let directory = member.central_record(0);
    let zip64_end_at = (body.len() + directory.len()) as u64;

    let mut out = prefix.to_vec();
    out.extend_from_slice(&body);
    out.extend_from_slice(&directory);
    out.extend_from_slice(&ZIP64_END_SIGNATURE.to_le_bytes());
    out.extend_from_slice(&(ZIP64_END_FIXED_SIZE + sector.len() as u64).to_le_bytes());
    out.extend_from_slice(&VERSION_ZIP64.to_le_bytes());
    out.extend_from_slice(&VERSION_ZIP64.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes()); // this disk
    out.extend_from_slice(&0u32.to_le_bytes()); // directory disk
    out.extend_from_slice(&declared_records.to_le_bytes());
    out.extend_from_slice(&declared_records.to_le_bytes());
    out.extend_from_slice(&(directory.len() as u64).to_le_bytes());
    out.extend_from_slice(&(body.len() as u64).to_le_bytes());
    out.extend_from_slice(sector);
    out.extend_from_slice(&ZIP64_LOCATOR_SIGNATURE.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&zip64_end_at.to_le_bytes());
    out.extend_from_slice(&SINGLE_DISK.to_le_bytes());
    out.extend_from_slice(&END_OF_DIRECTORY_SIGNATURE.to_le_bytes());
    out.extend_from_slice(&[0; 4]); // disks
    out.extend_from_slice(&u16::MAX.to_le_bytes());
    out.extend_from_slice(&u16::MAX.to_le_bytes());
    out.extend_from_slice(&ZIP64_SIZE_SENTINEL.to_le_bytes());
    out.extend_from_slice(&ZIP64_SIZE_SENTINEL.to_le_bytes());
    out.extend_from_slice(&[0; 2]); // comment length
    out
}

/// One local record that every name in `names` points at, so the directory
/// claims each member's bytes again without the file growing by more than a
/// directory record.
pub(super) fn overlapping_archive(member: &RawMember, names: &[&str]) -> Vec<u8> {
    const FIRST_RECORD_OFFSET: u32 = 0;
    let body = member.local_record();
    let mut directory = Vec::new();
    for name in names {
        directory.extend_from_slice(
            &member
                .clone()
                .named(name)
                .central_record(FIRST_RECORD_OFFSET),
        );
    }
    finish(body, directory, names.len())
}

fn finish(mut body: Vec<u8>, directory: Vec<u8>, records: usize) -> Vec<u8> {
    let directory_offset = body.len();
    let directory_len = directory.len();
    body.extend_from_slice(&directory);
    body.extend_from_slice(&end_of_directory(records, directory_len, directory_offset));
    body
}

fn crc32(bytes: &[u8]) -> u32 {
    !bytes.iter().fold(u32::MAX, |crc, byte| {
        (0..u8::BITS).fold(crc ^ u32::from(*byte), |crc, _| {
            if crc & 1 == 1 {
                (crc >> 1) ^ CRC32_REFLECTED_POLYNOMIAL
            } else {
                crc >> 1
            }
        })
    })
}
