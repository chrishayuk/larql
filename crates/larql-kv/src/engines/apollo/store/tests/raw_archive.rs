//! A hand-written ZIP encoder for tests that need an archive to lie.
//!
//! `zip::ZipWriter` only writes the truth, so it cannot produce a member whose
//! declared sizes disagree with its data. Only the fields the reader consults
//! are populated; timestamps are the DOS epoch.

const LOCAL_HEADER_SIGNATURE: u32 = 0x0403_4b50;
const CENTRAL_HEADER_SIGNATURE: u32 = 0x0201_4b50;
const END_OF_DIRECTORY_SIGNATURE: u32 = 0x0605_4b50;
/// The value that defers a 32-bit size field to the ZIP64 extra field.
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
const METHOD_DEFLATED: u16 = 8;
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
    zip64: bool,
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
            zip64: false,
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
            zip64: false,
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
        if !self.zip64 {
            return Vec::new();
        }
        let mut extra = Vec::new();
        extra.extend_from_slice(&ZIP64_EXTRA_ID.to_le_bytes());
        extra.extend_from_slice(&ZIP64_EXTRA_SIZES_LEN.to_le_bytes());
        extra.extend_from_slice(&self.declared.to_le_bytes());
        extra.extend_from_slice(&self.stored.to_le_bytes());
        extra
    }

    /// Version through extra-field length: the run the local and central
    /// headers share.
    fn common_fields(&self) -> Vec<u8> {
        let (declared, stored) = self.header_sizes();
        let mut out = Vec::new();
        for field in [
            self.version(),
            NO_FLAGS,
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
