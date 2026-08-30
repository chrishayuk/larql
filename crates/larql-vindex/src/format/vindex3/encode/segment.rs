//! One representation's segment: self-describing framing + payloads.
//!
//! Framing mirrors the proven safetensors shape, with our own header:
//!
//! ```text
//! [u64 LE header length][header JSON][payload bytes…]
//! ```
//!
//! The header carries the representation id and a tensor table (name
//! relative to the object, dtype, shape, offset, length — offsets relative
//! to the payload region). Table order is the payload order and is
//! deterministic: sorted by relative name. Two hashes are produced in the
//! single writing pass: one over the payload region (the source-side
//! byte-equivalence anchor) and one over the whole file as written.

use std::io::Write;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::error::VindexError;

/// Current segment header schema. Bump on any breaking change.
pub const SEGMENT_HEADER_SCHEMA: u32 = 1;

/// **Canonical output alignment**: what this encoder writes payloads
/// on. NOT the conformance requirement for reading one.
///
/// Two different numbers, deliberately:
///
/// * `SEGMENT_PAYLOAD_ALIGN` (16) — the encoder's own policy. Sixteen
///   covers every element type a segment carries and every SIMD load a
///   kernel might make over one, and costs at most fifteen bytes of
///   header padding, so there is no reason to write less.
/// * the *execution* minimum — 4 bytes, MEASURED against the real
///   kernel (`larql_compute_metal::buffers::WEIGHT_BINDING_ALIGN`).
///   That is the widest element a weight buffer is bound as, and a
///   region satisfying it binds zero-copy correctly.
///
/// A segment aligned to only the execution minimum is therefore
/// **conforming and directly bindable**, merely not what this encoder
/// would write today. That distinction is load-bearing rather than
/// pedantic: the Kimi container's 94 GB expert segment starts at
/// 2,438,284 — a multiple of four and not of sixteen — and is retained
/// and executed as-is. Raising the execution requirement to sixteen
/// would declare it non-conforming and force a 94 GB rewrite that
/// changes not one byte of model content.
pub const SEGMENT_PAYLOAD_ALIGN: usize = 16;

/// The segment header, as serialised into the file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentHeader {
    /// Always [`SEGMENT_HEADER_SCHEMA`].
    pub schema: u32,
    /// Representation id this segment materialises (`object@encoding`).
    pub representation: String,
    /// Tensor table; order is payload order.
    pub tensors: Vec<SegmentTensor>,
}

/// One tensor within a segment.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentTensor {
    /// Name relative to the logical object (never artifact-global).
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Offset within the payload region.
    pub offset: u64,
    pub len: u64,
}

/// What one planned tensor needs from the writer.
pub struct PlannedTensor {
    /// Object-relative name (table key).
    pub relative_name: String,
    /// Full source name (payload lookup key).
    pub source_name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub len: u64,
}

/// Result of writing one segment.
pub struct WrittenSegment {
    pub tensor_count: usize,
    pub payload_bytes: u64,
    pub payload_sha256: String,
    pub segment_sha256: String,
}

/// Deterministic payload order: sorted by object-relative name. The single
/// definition of segment order, shared by the writer and the G4 source-side
/// re-hash so the two can never disagree about what "the same bytes" means.
pub fn sort_into_payload_order(tensors: &mut [PlannedTensor]) {
    tensors.sort_by(|a, b| a.relative_name.cmp(&b.relative_name));
}

/// Write one segment file: header first (offsets are known from the plan),
/// then every payload streamed through both hashers.
pub fn write_segment(
    path: &Path,
    representation: &str,
    mut tensors: Vec<PlannedTensor>,
    mut stream_payload: impl FnMut(
        &str,
        &mut dyn Write,
        &mut dyn FnMut(&[u8]),
    ) -> Result<u64, VindexError>,
) -> Result<WrittenSegment, VindexError> {
    sort_into_payload_order(&mut tensors);
    let mut offset = 0u64;
    let table: Vec<SegmentTensor> = tensors
        .iter()
        .map(|t| {
            let entry = SegmentTensor {
                name: t.relative_name.clone(),
                dtype: t.dtype.clone(),
                shape: t.shape.clone(),
                offset,
                len: t.len,
            };
            offset += t.len;
            entry
        })
        .collect();
    let header = SegmentHeader {
        schema: SEGMENT_HEADER_SCHEMA,
        representation: representation.to_string(),
        tensors: table,
    };
    let mut header_bytes = serde_json::to_vec(&header)
        .map_err(|e| VindexError::Parse(format!("serialise segment header: {e}")))?;
    // **Pad so the payload starts on the canonical boundary.**
    //
    // `payload_start = 8 + header_len`, and a JSON header is whatever
    // length the tensor table happens to serialise to — odd as often as
    // not. A misaligned payload makes every tensor in the segment
    // misaligned too, and a compute backend that binds an mmap'd region
    // zero-copy then hands its kernel a misaligned pointer: on Metal a
    // `device const ushort*` at an odd address reads garbage with no
    // error at all.
    //
    // Padding to `SEGMENT_PAYLOAD_ALIGN` rather than to the execution
    // minimum because this is the writer, and the writer has no reason
    // to emit the least conforming thing it can.
    //
    // Measured on the real Kimi container, whose `decoder_stack`
    // payload started at 56,925: every dense/shared expert dispatch
    // returned NaN while the command buffer reported success.
    //
    // The padding is trailing whitespace, which JSON ignores, so a
    // reader that predates this still parses the header — and the
    // payload bytes and their hash are untouched.
    let pad = (SEGMENT_PAYLOAD_ALIGN - (8 + header_bytes.len()) % SEGMENT_PAYLOAD_ALIGN)
        % SEGMENT_PAYLOAD_ALIGN;
    header_bytes.extend(std::iter::repeat_n(b' ', pad));

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let file = std::fs::File::create(path)?;
    let mut writer = std::io::BufWriter::new(file);
    let mut file_hasher = Sha256::new();
    let len_prefix = (header_bytes.len() as u64).to_le_bytes();
    writer.write_all(&len_prefix)?;
    file_hasher.update(len_prefix);
    writer.write_all(&header_bytes)?;
    file_hasher.update(&header_bytes);

    let mut payload_hasher = Sha256::new();
    let mut payload_bytes = 0u64;
    for tensor in &tensors {
        let copied = stream_payload(&tensor.source_name, &mut writer, &mut |chunk| {
            payload_hasher.update(chunk);
            file_hasher.update(chunk);
        })?;
        if copied != tensor.len {
            return Err(VindexError::Parse(format!(
                "segment `{representation}`: tensor `{}` copied {copied} bytes, \
                 planned {} — source changed underneath the encode",
                tensor.source_name, tensor.len
            )));
        }
        payload_bytes += copied;
    }
    writer.flush()?;
    Ok(WrittenSegment {
        tensor_count: tensors.len(),
        payload_bytes,
        payload_sha256: format!("{:x}", payload_hasher.finalize()),
        segment_sha256: format!("{:x}", file_hasher.finalize()),
    })
}

/// Read a segment's header (framing only; no payload I/O).
/// **Rewrite a segment so its payload starts on
/// [`SEGMENT_PAYLOAD_ALIGN`], copying the payload bytes verbatim.**
///
/// For containers encoded before the encoder padded its header. The
/// tensor table, the payload bytes and therefore the payload hash
/// recorded in the container's index are all unchanged — only the
/// header grows by up to fifteen spaces — so a candidate overlay
/// compiled against the container still verifies against it.
///
/// Returns `(payload_start_before, payload_start_after)`. A segment
/// that is already aligned is left untouched and reports the same value
/// twice.
pub fn realign_segment(path: &Path, out: &Path) -> Result<(u64, u64), VindexError> {
    use std::io::{Read, Seek, SeekFrom};
    let (_, payload_start) = read_segment_header(path)?;
    let mut file = std::fs::File::open(path)?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes)?;
    let pad = (SEGMENT_PAYLOAD_ALIGN - (payload_start as usize) % SEGMENT_PAYLOAD_ALIGN)
        % SEGMENT_PAYLOAD_ALIGN;
    if pad == 0 && path == out {
        return Ok((payload_start, payload_start));
    }
    header_bytes.extend(std::iter::repeat_n(b' ', pad));

    if let Some(parent) = out.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = std::io::BufWriter::new(std::fs::File::create(out)?);
    writer.write_all(&(header_bytes.len() as u64).to_le_bytes())?;
    writer.write_all(&header_bytes)?;
    file.seek(SeekFrom::Start(payload_start))?;
    let copied = std::io::copy(&mut file, &mut writer)?;
    writer.flush()?;
    let after = 8 + header_bytes.len() as u64;
    if !after.is_multiple_of(SEGMENT_PAYLOAD_ALIGN as u64) {
        return Err(VindexError::Parse(format!(
            "realigned payload still starts at {after}"
        )));
    }
    let _ = copied;
    Ok((payload_start, after))
}

pub fn read_segment_header(path: &Path) -> Result<(SegmentHeader, u64), VindexError> {
    use std::io::Read;
    /// Bound mirrors the writer's practical header sizes.
    const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;
    let mut file = std::fs::File::open(path)?;
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)?;
    let header_len = u64::from_le_bytes(len_bytes);
    if header_len > MAX_HEADER_BYTES {
        return Err(VindexError::Parse(format!(
            "{}: segment header claims {header_len} bytes — corrupt",
            path.display()
        )));
    }
    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes)?;
    let header: SegmentHeader = serde_json::from_slice(&header_bytes)
        .map_err(|e| VindexError::Parse(format!("{}: segment header: {e}", path.display())))?;
    Ok((header, 8 + header_len))
}
