//! A NumPy `.npy` writer, format version 1.0, little-endian, C order: the file `numpy.load` reads.
//!
//! The header is the magic string, the version, a little-endian u16 header length, then a Python dict literal
//! padded with spaces and ended by a newline so that the data starts on a multiple of [`HEADER_ALIGN`] bytes.
//! The body is streamed: the writer never holds the array. Every byte written is hashed as it goes, so the digest
//! is of exactly the file on disk.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use sha2::{Digest, Sha256};

/// `\x93NUMPY`.
const MAGIC: &[u8] = b"\x93NUMPY";
/// Format version 1.0: a u16 header length.
const VERSION: [u8; 2] = [1, 0];
/// The data starts on a multiple of this many bytes.
const HEADER_ALIGN: usize = 64;
/// Bytes before the dict: magic, version, the u16 length.
const PREAMBLE_BYTES: usize = MAGIC.len() + VERSION.len() + std::mem::size_of::<u16>();
/// Buffered output, so a selection streamed row by row is not a write per row.
const WRITE_BUFFER_BYTES: usize = 1 << 20;

/// The element type of the array, as NumPy's `descr` names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum NpyDescr {
    /// Little-endian float32.
    F4,
    /// Little-endian uint16.
    U2,
    /// Little-endian uint32.
    U4,
}

impl NpyDescr {
    pub(super) fn label(self) -> &'static str {
        match self {
            Self::F4 => "<f4",
            Self::U2 => "<u2",
            Self::U4 => "<u4",
        }
    }
}

/// The header bytes for an array of `descr` and `shape`.
fn header(descr: NpyDescr, shape: &[usize]) -> std::io::Result<Vec<u8>> {
    let dims = match shape {
        [only] => format!("({only},)"),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {dims}, }}",
        descr.label()
    );
    // Pad with spaces so the newline that ends the dict is the last byte before an aligned data start.
    let unpadded = PREAMBLE_BYTES + dict.len() + 1;
    dict.push_str(&" ".repeat(unpadded.next_multiple_of(HEADER_ALIGN) - unpadded));
    dict.push('\n');
    let length = u16::try_from(dict.len()).map_err(|_| {
        std::io::Error::other(format!(
            "an .npy 1.0 header holds at most {} bytes; this shape needs {}",
            u16::MAX,
            dict.len()
        ))
    })?;
    let mut bytes = Vec::with_capacity(PREAMBLE_BYTES + dict.len());
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&VERSION);
    bytes.extend_from_slice(&length.to_le_bytes());
    bytes.extend_from_slice(dict.as_bytes());
    Ok(bytes)
}

/// An `.npy` file being written. Created only where no file exists.
pub(super) struct NpyWriter {
    out: BufWriter<File>,
    hash: Sha256,
}

impl NpyWriter {
    /// Create `path`, which must not exist, and write the header.
    pub(super) fn create_new(
        path: &Path,
        descr: NpyDescr,
        shape: &[usize],
    ) -> std::io::Result<Self> {
        let header = header(descr, shape)?;
        let file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?;
        let mut writer = Self {
            out: BufWriter::with_capacity(WRITE_BUFFER_BYTES, file),
            hash: Sha256::new(),
        };
        writer.write(&header)?;
        Ok(writer)
    }

    /// Append body bytes: elements already little-endian, of the header's `descr`.
    pub(super) fn write(&mut self, bytes: &[u8]) -> std::io::Result<()> {
        self.hash.update(bytes);
        self.out.write_all(bytes)
    }

    /// Flush to disk and give the SHA-256 of the whole file, lowercase hex.
    pub(super) fn finish(self) -> std::io::Result<String> {
        let file = self.out.into_inner().map_err(|e| e.into_error())?;
        file.sync_all()?;
        Ok(format!("{:x}", self.hash.finalize()))
    }
}
