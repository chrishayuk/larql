//! Minimal numpy `.npy` v1.0 reader for the dtypes the Apollo store uses.
//!
//! We avoid `ndarray-npy` because it depends on ndarray 0.17 while the
//! workspace pins 0.16. The format is simple enough to parse directly:
//!
//! ```text
//! 6 bytes  magic        "\x93NUMPY"
//! 2 bytes  version      0x01 0x00   (v1.0; v2.0 uses u32 header_len)
//! 2 bytes  header_len   u16 little-endian
//! N bytes  header       ASCII Python dict literal
//! remaining data        row-major contiguous, little-endian
//! ```
//!
//! Supported dtype strings (only what apollo11_store uses):
//!   - `'<f4'` → f32
//!   - `'<u4'` → u32
//!   - structured dtypes are parsed by the `apollo::store` module directly.

#[derive(Debug)]
pub struct NpyHeader<'a> {
    /// The dtype exactly as written, quotes and all, for comparing with a known form.
    pub descr: &'a str,
    pub shape: Vec<usize>,
}

#[derive(Debug, thiserror::Error)]
pub enum NpyError {
    #[error("file is not a valid .npy (bad magic)")]
    BadMagic,
    #[error("unsupported .npy version {0}.{1} (need 1.x)")]
    UnsupportedVersion(u8, u8),
    #[error("truncated .npy header")]
    TruncatedHeader,
    #[error("header is not valid UTF-8: {0}")]
    InvalidUtf8(std::str::Utf8Error),
    #[error("could not parse header field '{field}' from: {snippet}")]
    ParseField {
        field: &'static str,
        snippet: String,
    },
    #[error("dtype mismatch: expected {expected}, got {actual}")]
    DtypeMismatch {
        expected: &'static str,
        actual: String,
    },
    #[error("data length {got} does not match expected {expected} ({shape:?} × {stride} bytes)")]
    DataLength {
        got: usize,
        expected: usize,
        shape: Vec<usize>,
        stride: usize,
    },
    #[error("shape {shape:?} × {stride} bytes overflows the address space")]
    ShapeOverflow { shape: Vec<usize>, stride: usize },
    #[error("allocation of {bytes} bytes refused")]
    AllocationRefused { bytes: u64 },
    #[error("{0}")]
    StructuredLayout(String),
}

const F4_DESCR: &str = "'<f4'";
const U4_DESCR: &str = "'<u4'";

/// Bytes per element of both `<f4` and `<u4`.
const ELEMENT_STRIDE: usize = 4;

/// Byte length a `shape` of `stride`-byte rows occupies, or
/// [`NpyError::ShapeOverflow`] if it does not fit in `usize`.
///
/// Unchecked, this product wraps in release builds: `2^62 × 4` becomes 0, which
/// matches an empty data section and lets the shape reach the allocator.
pub fn checked_data_len(shape: &[usize], stride: usize) -> Result<usize, NpyError> {
    shape
        .iter()
        .try_fold(stride, |len, dim| len.checked_mul(*dim))
        .ok_or_else(|| NpyError::ShapeOverflow {
            shape: shape.to_vec(),
            stride,
        })
}

/// An empty `Vec` with room for `n` elements, refusing with a typed error where
/// `Vec::with_capacity` would abort.
pub fn try_vec_with_capacity<T>(n: usize) -> Result<Vec<T>, NpyError> {
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|_| NpyError::AllocationRefused {
            bytes: u64::try_from(n.saturating_mul(size_of::<T>())).unwrap_or(u64::MAX),
        })?;
    Ok(out)
}

/// Decode a checked data section of little-endian four-byte elements.
fn decode_elements<T>(data: &[u8], decode: fn([u8; 4]) -> T) -> Result<Vec<T>, NpyError> {
    let mut out = try_vec_with_capacity(data.len() / ELEMENT_STRIDE)?;
    let (elements, _) = data.as_chunks::<ELEMENT_STRIDE>();
    for element in elements {
        out.push(decode(*element));
    }
    Ok(out)
}

/// `data`, once it is exactly as long as `shape` says. Every allocation sized
/// from a shape happens after this, so it is bounded by bytes actually present.
fn checked_data<'a>(data: &'a [u8], shape: &[usize], stride: usize) -> Result<&'a [u8], NpyError> {
    let expected = checked_data_len(shape, stride)?;
    if data.len() != expected {
        return Err(NpyError::DataLength {
            got: data.len(),
            expected,
            shape: shape.to_vec(),
            stride,
        });
    }
    Ok(data)
}

/// The dictionary numpy's `_write_array_header` writes: keys sorted, each followed by
/// `, `, then `}`. `descr` and `shape` are `repr`s, and only C-order arrays are read.
const HEADER_OPEN: &str = "{'descr': ";
const HEADER_AFTER_DESCR: &str = ", 'fortran_order': False, 'shape': ";
const HEADER_CLOSE: &str = ", }";
/// `_wrap_header` pads with spaces and ends with one newline.
const HEADER_PADDING: char = ' ';
const HEADER_END: char = '\n';

/// The header as numpy's writer lays it out for format 1.0, and nothing else.
///
/// numpy reads a header as a Python literal, so repeated keys, comments, other quoting,
/// escapes, concatenation and more can all mean something other than they look. Rather
/// than recognise those, this accepts only the one form a writer produces. Python 2's
/// `3L` shapes aren't accepted, and `descr` is compared whole by the caller.
fn parse_writer_form(header: &str) -> Option<NpyHeader<'_>> {
    let body = header.strip_suffix(HEADER_END)?;
    let body = body.trim_end_matches(HEADER_PADDING);
    let body = body.strip_prefix(HEADER_OPEN)?;
    let body = body.strip_suffix(HEADER_CLOSE)?;
    let (descr, shape) = body.split_once(HEADER_AFTER_DESCR)?;
    Some(NpyHeader {
        descr,
        shape: parse_shape_repr(shape)?,
    })
}

/// A Python `repr` of a tuple of non-negative integers: `()`, `(3,)` or `(1, 1, 4)`.
fn parse_shape_repr(shape: &str) -> Option<Vec<usize>> {
    const SEPARATOR: &str = ", ";
    let inner = shape.strip_prefix('(')?.strip_suffix(')')?;
    if inner.is_empty() {
        return Some(Vec::new());
    }
    let (list, single) = match inner.strip_suffix(',') {
        Some(only) => (only, true),
        None => (inner, false),
    };
    let mut dims = Vec::new();
    for dim in list.split(SEPARATOR) {
        let canonical =
            dim.bytes().all(|byte| byte.is_ascii_digit()) && (dim == "0" || !dim.starts_with('0'));
        if !canonical {
            return None;
        }
        dims.push(dim.parse().ok()?);
    }
    // A one-element tuple has the trailing comma and only that one has.
    (single == (dims.len() == 1)).then_some(dims)
}

/// Parse the `.npy` header. Returns `(header, data_offset)` where `data_offset`
/// is the byte index at which raw array data begins.
pub fn parse_header(bytes: &[u8]) -> Result<(NpyHeader<'_>, usize), NpyError> {
    if bytes.len() < 10 {
        return Err(NpyError::TruncatedHeader);
    }
    if &bytes[..6] != b"\x93NUMPY" {
        return Err(NpyError::BadMagic);
    }
    let major = bytes[6];
    let minor = bytes[7];
    if major != 1 {
        return Err(NpyError::UnsupportedVersion(major, minor));
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header_end = 10 + header_len;
    if bytes.len() < header_end {
        return Err(NpyError::TruncatedHeader);
    }
    let header_str = std::str::from_utf8(&bytes[10..header_end]).map_err(NpyError::InvalidUtf8)?;
    let header = parse_writer_form(header_str).ok_or_else(|| NpyError::ParseField {
        field: "header",
        snippet: header_str.to_string(),
    })?;
    Ok((header, header_end))
}

/// Read an `<f4` 1D array from .npy bytes.
pub fn read_f32_1d(bytes: &[u8]) -> Result<Vec<f32>, NpyError> {
    let (header, data_off) = parse_header(bytes)?;
    check_dtype(header.descr, F4_DESCR)?;
    if header.shape.len() != 1 {
        return Err(NpyError::ParseField {
            field: "shape",
            snippet: format!("expected 1D, got {:?}", header.shape),
        });
    }
    let data = checked_data(&bytes[data_off..], &header.shape, ELEMENT_STRIDE)?;
    decode_elements(data, f32::from_le_bytes)
}

/// Read an `<f4` multi-D array as a flat Vec (row-major) plus shape.
pub fn read_f32_flat(bytes: &[u8]) -> Result<(Vec<f32>, Vec<usize>), NpyError> {
    let (header, data_off) = parse_header(bytes)?;
    check_dtype(header.descr, F4_DESCR)?;
    let data = checked_data(&bytes[data_off..], &header.shape, ELEMENT_STRIDE)?;
    Ok((decode_elements(data, f32::from_le_bytes)?, header.shape))
}

/// Read an `<u4` 1D array from .npy bytes.
pub fn read_u32_1d(bytes: &[u8]) -> Result<Vec<u32>, NpyError> {
    let (header, data_off) = parse_header(bytes)?;
    check_dtype(header.descr, U4_DESCR)?;
    if header.shape.len() != 1 {
        return Err(NpyError::ParseField {
            field: "shape",
            snippet: format!("expected 1D, got {:?}", header.shape),
        });
    }
    let data = checked_data(&bytes[data_off..], &header.shape, ELEMENT_STRIDE)?;
    decode_elements(data, u32::from_le_bytes)
}

fn check_dtype(got: &str, expected: &'static str) -> Result<(), NpyError> {
    if got != expected {
        Err(NpyError::DtypeMismatch {
            expected,
            actual: got.to_string(),
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal .npy v1.0 blob for an f32 1D array of given values.
    fn synth_f32_1d(values: &[f32]) -> Vec<u8> {
        let header = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
            values.len()
        );
        // Pad header to 64-byte alignment (numpy convention).
        let mut padded = header.into_bytes();
        let total = 10 + padded.len();
        let pad_to = (total + 63) & !63;
        while 10 + padded.len() + 1 < pad_to {
            padded.push(b' ');
        }
        padded.push(b'\n');
        let header_len = padded.len();

        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY");
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(header_len as u16).to_le_bytes());
        out.extend_from_slice(&padded);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn parse_1d_f32_roundtrip() {
        let vals = [1.0f32, 2.0, 3.0, -4.5, 0.125];
        let blob = synth_f32_1d(&vals);
        let parsed = read_f32_1d(&blob).expect("parse");
        assert_eq!(parsed, vals.to_vec());
    }

    #[test]
    fn dtype_mismatch_reports_what_was_found() {
        let vals = [1.0f32, 2.0];
        let blob = synth_f32_1d(&vals);
        let result = read_u32_1d(&blob);
        let err = result.unwrap_err();
        assert!(matches!(err, NpyError::DtypeMismatch { .. }));
    }

    /// Build a minimal .npy v1.0 blob for a u32 1D array of given values.
    fn synth_u32_1d(values: &[u32]) -> Vec<u8> {
        let header = format!(
            "{{'descr': '<u4', 'fortran_order': False, 'shape': ({},), }}",
            values.len()
        );
        let mut padded = header.into_bytes();
        let total = 10 + padded.len();
        let pad_to = (total + 63) & !63;
        while 10 + padded.len() + 1 < pad_to {
            padded.push(b' ');
        }
        padded.push(b'\n');
        let header_len = padded.len();
        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY");
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(header_len as u16).to_le_bytes());
        out.extend_from_slice(&padded);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// Build a 2D .npy blob for read_f32_flat tests.
    fn synth_f32_flat(values: &[f32], rows: usize, cols: usize) -> Vec<u8> {
        let header =
            format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
        let mut padded = header.into_bytes();
        let total = 10 + padded.len();
        let pad_to = (total + 63) & !63;
        while 10 + padded.len() + 1 < pad_to {
            padded.push(b' ');
        }
        padded.push(b'\n');
        let header_len = padded.len();
        let mut out = Vec::new();
        out.extend_from_slice(b"\x93NUMPY");
        out.push(1);
        out.push(0);
        out.extend_from_slice(&(header_len as u16).to_le_bytes());
        out.extend_from_slice(&padded);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn read_u32_1d_roundtrip() {
        let vals = [1u32, 7, 42, 999, u32::MAX];
        let blob = synth_u32_1d(&vals);
        assert_eq!(read_u32_1d(&blob).unwrap(), vals.to_vec());
    }

    #[test]
    fn read_f32_flat_2d_roundtrip() {
        let vals = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let blob = synth_f32_flat(&vals, 2, 3);
        let (parsed, shape) = read_f32_flat(&blob).unwrap();
        assert_eq!(parsed, vals.to_vec());
        assert_eq!(shape, vec![2, 3]);
    }

    // ── Error paths ───────────────────────────────────────────────────────

    #[test]
    fn parse_header_rejects_short_input() {
        let err = parse_header(&[0u8; 5]).unwrap_err();
        assert!(matches!(err, NpyError::TruncatedHeader));
    }

    #[test]
    fn parse_header_rejects_bad_magic() {
        let mut bad = vec![0u8; 16];
        bad[..6].copy_from_slice(b"\x00WRONG");
        let err = parse_header(&bad).unwrap_err();
        assert!(matches!(err, NpyError::BadMagic));
    }

    #[test]
    fn parse_header_rejects_v2() {
        let mut blob = vec![0u8; 16];
        blob[..6].copy_from_slice(b"\x93NUMPY");
        blob[6] = 2; // major
        blob[7] = 0; // minor
        let err = parse_header(&blob).unwrap_err();
        match err {
            NpyError::UnsupportedVersion(2, 0) => {}
            other => panic!("expected UnsupportedVersion(2,0), got {other:?}"),
        }
    }

    #[test]
    fn parse_header_truncated_after_header_len() {
        // Magic + ver + a header_len that points past the buffer end.
        let mut blob = Vec::new();
        blob.extend_from_slice(b"\x93NUMPY");
        blob.push(1);
        blob.push(0);
        blob.extend_from_slice(&(9999u16).to_le_bytes());
        // No actual header bytes.
        let err = parse_header(&blob).unwrap_err();
        assert!(matches!(err, NpyError::TruncatedHeader));
    }

    #[test]
    fn parse_header_invalid_utf8_in_header() {
        let mut blob = Vec::new();
        blob.extend_from_slice(b"\x93NUMPY");
        blob.push(1);
        blob.push(0);
        let header = vec![0xff, 0xfe, 0xfd, 0xfc];
        blob.extend_from_slice(&(header.len() as u16).to_le_bytes());
        blob.extend_from_slice(&header);
        let err = parse_header(&blob).unwrap_err();
        assert!(matches!(err, NpyError::InvalidUtf8(_)));
    }

    #[test]
    fn read_f32_1d_data_length_mismatch() {
        let mut blob = synth_f32_1d(&[1.0f32, 2.0, 3.0]);
        // Drop two bytes from the data tail.
        blob.truncate(blob.len() - 2);
        let err = read_f32_1d(&blob).unwrap_err();
        match err {
            NpyError::DataLength {
                expected,
                stride: 4,
                ..
            } => assert_eq!(expected, 12),
            other => panic!("expected DataLength, got {other:?}"),
        }
    }

    #[test]
    fn read_f32_1d_rejects_2d() {
        // 2D blob handed to a 1D reader → ParseField on shape.
        let blob = synth_f32_flat(&[1.0f32, 2.0, 3.0, 4.0], 2, 2);
        let err = read_f32_1d(&blob).unwrap_err();
        match err {
            NpyError::ParseField { field, .. } => assert_eq!(field, "shape"),
            other => panic!("expected ParseField, got {other:?}"),
        }
    }

    #[test]
    fn read_u32_1d_rejects_2d() {
        let header = "{'descr': '<u4', 'fortran_order': False, 'shape': (2, 2), }".to_string();
        let mut padded = header.into_bytes();
        let total = 10 + padded.len();
        let pad_to = (total + 63) & !63;
        while 10 + padded.len() + 1 < pad_to {
            padded.push(b' ');
        }
        padded.push(b'\n');
        let header_len = padded.len();
        let mut blob = Vec::new();
        blob.extend_from_slice(b"\x93NUMPY");
        blob.push(1);
        blob.push(0);
        blob.extend_from_slice(&(header_len as u16).to_le_bytes());
        blob.extend_from_slice(&padded);
        for v in [1u32, 2, 3, 4] {
            blob.extend_from_slice(&v.to_le_bytes());
        }
        let err = read_u32_1d(&blob).unwrap_err();
        match err {
            NpyError::ParseField { field, .. } => assert_eq!(field, "shape"),
            other => panic!("expected ParseField, got {other:?}"),
        }
    }

    // ── Header field-parsing helpers ──────────────────────────────────────
}
