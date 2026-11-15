//! Arrays on the wire: a whole `.npy` file, base64 in the JSON.
//!
//! Out: the file `larql_lql::operations::npy_file_bytes` builds, the bytes the matching LQL statement writes,
//! so its SHA-256 is the `npy_sha256` in the provenance. In: a 2-D, C-order, little-endian float64 or float32
//! array (`numpy.save` of an operand), read strictly. Anything else is refused rather than converted.

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use larql_lql::operations::{npy_file_bytes, NpyDescr};

use super::protocol::Failure;

const MAGIC: &[u8] = b"\x93NUMPY";
/// Magic, then the major and minor version bytes.
const VERSION_END: usize = MAGIC.len() + 2;

/// An array as a response carries it, and the digest the audit log records.
pub struct EncodedArray {
    pub value: Value,
    /// The `.npy` file's size, which the response limit counts.
    pub bytes: u64,
    pub sha256: String,
}

/// The `.npy` file of `descr` and `shape` holding `body`, for a response.
pub fn encode(descr: NpyDescr, shape: &[usize], body: &[u8]) -> Result<EncodedArray, Failure> {
    let file = npy_file_bytes(descr, shape, body).map_err(Failure::failed)?;
    let sha256 = format!("{:x}", Sha256::digest(&file));
    Ok(EncodedArray {
        value: json!({
            "npy_base64": STANDARD.encode(&file),
            "sha256": sha256,
            "dtype": descr.label(),
            "shape": shape,
        }),
        bytes: file.len() as u64,
        sha256,
    })
}

/// The values of a 2-D operand array.
#[derive(Debug, Clone, PartialEq)]
pub enum Operand {
    F64(Vec<f64>),
    F32(Vec<f32>),
}

/// A decoded operand: its values, row-major, and its shape.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedOperand {
    pub values: Operand,
    pub shape: [usize; 2],
    /// The SHA-256 of the `.npy` bytes received.
    pub sha256: String,
}

/// Decode the base64 `.npy` operand `which`.
pub fn decode_operand(which: &str, base64_text: &str) -> Result<DecodedOperand, Failure> {
    let bad = |why: String| Failure::invalid(format!("`{which}`: {why}"));
    let file = STANDARD
        .decode(base64_text)
        .map_err(|e| bad(format!("not base64: {e}")))?;
    let sha256 = format!("{:x}", Sha256::digest(&file));
    if file.len() < VERSION_END || &file[..MAGIC.len()] != MAGIC {
        return Err(bad("not a .npy file".into()));
    }
    let (header_len, header_start) = match file[MAGIC.len()] {
        1 => header_length(&file, 2).map(|n| (n, VERSION_END + 2)),
        2 | 3 => header_length(&file, 4).map(|n| (n, VERSION_END + 4)),
        v => return Err(bad(format!("unsupported .npy version {v}"))),
    }
    .ok_or_else(|| bad("truncated .npy header".into()))?;
    let body_start = header_start
        .checked_add(header_len)
        .filter(|&end| end <= file.len())
        .ok_or_else(|| bad("truncated .npy header".into()))?;
    let header = std::str::from_utf8(&file[header_start..body_start])
        .map_err(|_| bad("the .npy header is not text".into()))?;
    let (descr, fortran, shape) = parse_header(header).map_err(bad)?;
    if fortran {
        return Err(bad(
            "Fortran-order arrays are refused; pass a C-order array".into(),
        ));
    }
    let [rows, cols] = match shape[..] {
        [r, c] => [r, c],
        _ => return Err(bad(format!("must be 2-D, got shape {shape:?}"))),
    };
    let body = &file[body_start..];
    let width = match descr.as_str() {
        "<f8" => 8,
        "<f4" => 4,
        other => {
            return Err(bad(format!(
                "dtype `{other}` is refused; pass little-endian float64 (`<f8`) or float32 (`<f4`)"
            )))
        }
    };
    let expected = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(width))
        .ok_or_else(|| bad("the shape overflows".into()))?;
    if body.len() != expected {
        return Err(bad(format!(
            "holds {} body byte(s); shape {rows}x{cols} of {descr} needs {expected}",
            body.len()
        )));
    }
    let values = if width == 8 {
        Operand::F64(
            body.chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().expect("8 bytes")))
                .collect(),
        )
    } else {
        Operand::F32(
            body.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
                .collect(),
        )
    };
    Ok(DecodedOperand {
        values,
        shape: [rows, cols],
        sha256,
    })
}

fn header_length(file: &[u8], width: usize) -> Option<usize> {
    let bytes = file.get(VERSION_END..VERSION_END + width)?;
    Some(match width {
        2 => usize::from(u16::from_le_bytes([bytes[0], bytes[1]])),
        _ => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize,
    })
}

/// The header dict's `descr`, `fortran_order` and `shape`, from the literal `numpy.save` writes:
/// `{'descr': '<f8', 'fortran_order': False, 'shape': (3, 1), }`.
fn parse_header(header: &str) -> Result<(String, bool, Vec<usize>), String> {
    let body = header
        .trim_end()
        .strip_prefix('{')
        .and_then(|h| h.strip_suffix('}'))
        .ok_or("the .npy header is not a dict")?;
    let descr = field(body, "descr")?;
    let descr = descr
        .strip_prefix('\'')
        .and_then(|d| d.strip_suffix('\''))
        .ok_or("`descr` is not a string")?
        .to_string();
    let fortran = match field(body, "fortran_order")? {
        "False" => false,
        "True" => true,
        other => return Err(format!("`fortran_order` is `{other}`")),
    };
    let shape = field(body, "shape")?;
    let dims = shape
        .strip_prefix('(')
        .and_then(|s| s.strip_suffix(')'))
        .ok_or("`shape` is not a tuple")?;
    let shape = dims
        .split(',')
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .map(|d| d.parse::<usize>().map_err(|_| format!("`shape` has `{d}`")))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((descr, fortran, shape))
}

/// The text of `key`'s value in the dict body: up to the next top-level comma.
fn field<'a>(body: &'a str, key: &str) -> Result<&'a str, String> {
    let quoted = format!("'{key}':");
    let start = body
        .find(&quoted)
        .ok_or_else(|| format!("the .npy header has no `{key}`"))?
        + quoted.len();
    let rest = body[start..].trim_start();
    let end = if rest.starts_with('(') {
        rest.find(')').map(|i| i + 1)
    } else {
        rest.find(',')
    }
    .unwrap_or(rest.len());
    Ok(rest[..end].trim())
}
