//! Multi-row selected-expert transforms: a separate, explicitly versioned
//! frame pair beside the single-row [`crate::vindex3_experts`] contract.
//!
//! One request carries several input rows, each with its own caller-assigned
//! row id and its own selected experts. The response carries one unweighted
//! output per `(row, expert)` pair. No routing weights are on the wire, and a
//! row id means nothing to the worker: whether rows are positions of one
//! session or something else is the coordinator's business.
//!
//! This is not an extension of `VEX1`/`VEY1`. It has its own path, content
//! type and magics, so a worker that predates it has no such route, and a
//! frame sent to the wrong endpoint fails that endpoint's magic check. A
//! response is whole or it is an error: nothing here lets a decoder return a
//! subset of what was asked for.
//!
//! ```text
//! header (44 bytes, both directions)
//!   magic[4]  handle[16]  sequence u64  layer u32  hidden u32  rows u32  pairs u32
//! request  VXR1, then per row:   row u32, count u32, count x expert u32, hidden x f32
//! response VXS1, then per pair:  row u32, expert u32, hidden x f32
//! ```
//!
//! All integers and F32 bits are little-endian, with no padding.
use crate::vindex3_experts::Handle;

pub const ROWS_PATH: &str = "/v1/vindex3/experts/rows";
pub const CONTENT_TYPE: &str = "application/vnd.larql.v3-expert-rows.f32";
pub const REQUEST_MAGIC: [u8; 4] = *b"VXR1";
pub const RESPONSE_MAGIC: [u8; 4] = *b"VXS1";
pub const HEADER_BYTES: usize = 44;
/// Most rows one frame may carry. A decoder checks this before it allocates
/// anything sized by the frame, so a hostile header cannot size a buffer.
pub const MAX_ROWS: usize = 256;

const WORD: usize = 4;
const HANDLE_END: usize = 20;
const SEQUENCE_END: usize = 28;
const LAYER_AT: usize = 28;
const HIDDEN_AT: usize = 32;
const ROWS_AT: usize = 36;
const PAIRS_AT: usize = 40;

/// One input row and the experts it selected, as the caller named them.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    pub id: u32,
    pub experts: Vec<usize>,
    pub input: Vec<f32>,
}

/// One unweighted expert output for one row.
#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    pub row: u32,
    pub expert: usize,
    pub values: Vec<f32>,
}

#[derive(Debug)]
pub struct Request {
    pub handle: Handle,
    pub sequence: u64,
    pub layer: usize,
    pub rows: Vec<Row>,
}

#[derive(Debug)]
pub struct Response {
    pub handle: Handle,
    pub sequence: u64,
    pub layer: usize,
    /// Rows the request carried, echoed so a reply is tied to its shape.
    pub rows: usize,
    pub outputs: Vec<Output>,
}

fn word(value: usize) -> Result<[u8; WORD], String> {
    u32::try_from(value)
        .map(u32::to_le_bytes)
        .map_err(|_| "expert rows frame field overflow".to_string())
}

fn word_at(bytes: &[u8], offset: usize) -> usize {
    u32::from_le_bytes(bytes[offset..offset + WORD].try_into().unwrap()) as usize
}

fn header(
    magic: [u8; 4],
    handle: Handle,
    sequence: u64,
    layer: usize,
    hidden: usize,
    rows: usize,
    pairs: usize,
) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(HEADER_BYTES);
    out.extend_from_slice(&magic);
    out.extend_from_slice(&handle);
    out.extend_from_slice(&sequence.to_le_bytes());
    for field in [layer, hidden, rows, pairs] {
        out.extend_from_slice(&word(field)?);
    }
    Ok(out)
}

struct Header {
    handle: Handle,
    sequence: u64,
    layer: usize,
    rows: usize,
    pairs: usize,
}

fn read_header(
    bytes: &[u8],
    magic: [u8; 4],
    hidden: usize,
    max_rows: usize,
    max_selected: usize,
) -> Result<Header, String> {
    if bytes.len() < HEADER_BYTES || bytes[..WORD] != magic {
        return Err("invalid expert rows frame header".into());
    }
    let rows = word_at(bytes, ROWS_AT);
    let pairs = word_at(bytes, PAIRS_AT);
    let most_pairs = rows
        .checked_mul(max_selected)
        .ok_or("expert rows frame count overflow")?;
    if hidden == 0
        || word_at(bytes, HIDDEN_AT) != hidden
        || rows == 0
        || rows > max_rows.min(MAX_ROWS)
        || pairs < rows
        || pairs > most_pairs
    {
        return Err("expert rows frame width or count mismatch".into());
    }
    Ok(Header {
        handle: bytes[WORD..HANDLE_END].try_into().unwrap(),
        sequence: u64::from_le_bytes(bytes[HANDLE_END..SEQUENCE_END].try_into().unwrap()),
        layer: word_at(bytes, LAYER_AT),
        rows,
        pairs,
    })
}

fn append_values(out: &mut Vec<u8>, values: &[f32], hidden: usize) -> Result<(), String> {
    if values.len() != hidden || values.iter().any(|v| !v.is_finite()) {
        return Err("expert rows carrier width or finiteness mismatch".into());
    }
    for value in values {
        out.extend_from_slice(&value.to_bits().to_le_bytes());
    }
    Ok(())
}

fn read_values(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let values: Vec<f32> = bytes
        .chunks_exact(WORD)
        .map(|b| f32::from_bits(u32::from_le_bytes(b.try_into().unwrap())))
        .collect();
    if values.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite expert rows carrier".into());
    }
    Ok(values)
}

/// Exact body length of a request for `pairs` selections over `rows` rows.
pub fn request_len(hidden: usize, rows: usize, pairs: usize) -> Result<usize, String> {
    let per_row = hidden
        .checked_mul(WORD)
        .and_then(|n| n.checked_add(2 * WORD));
    per_row
        .and_then(|n| n.checked_mul(rows))
        .and_then(|n| n.checked_add(pairs.checked_mul(WORD)?))
        .and_then(|n| n.checked_add(HEADER_BYTES))
        .ok_or_else(|| "expert rows frame length overflow".into())
}

/// Exact body length of a response carrying `pairs` outputs.
pub fn response_len(hidden: usize, pairs: usize) -> Result<usize, String> {
    hidden
        .checked_mul(WORD)
        .and_then(|n| n.checked_add(2 * WORD))
        .and_then(|n| n.checked_mul(pairs))
        .and_then(|n| n.checked_add(HEADER_BYTES))
        .ok_or_else(|| "expert rows frame length overflow".into())
}

/// Row ids must be distinct, and so must the experts within a row.
fn check_rows(rows: &[Row], max_selected: usize) -> Result<usize, String> {
    if rows.is_empty() || rows.len() > MAX_ROWS {
        return Err("expert rows request must carry between one row and the frame limit".into());
    }
    let mut ids = std::collections::BTreeSet::new();
    let mut pairs = 0usize;
    for row in rows {
        if !ids.insert(row.id) {
            return Err("duplicate row id in expert rows request".into());
        }
        if row.experts.is_empty() || row.experts.len() > max_selected {
            return Err("expert rows selection count outside bound".into());
        }
        let distinct: std::collections::BTreeSet<_> = row.experts.iter().collect();
        if distinct.len() != row.experts.len() {
            return Err("duplicate expert within a row".into());
        }
        pairs += row.experts.len();
    }
    Ok(pairs)
}

pub fn encode_request(
    handle: Handle,
    sequence: u64,
    layer: usize,
    hidden: usize,
    rows: &[Row],
) -> Result<Vec<u8>, String> {
    let pairs = check_rows(rows, usize::MAX)?;
    let mut out = header(
        REQUEST_MAGIC,
        handle,
        sequence,
        layer,
        hidden,
        rows.len(),
        pairs,
    )?;
    out.reserve(request_len(hidden, rows.len(), pairs)? - HEADER_BYTES);
    for row in rows {
        out.extend_from_slice(&row.id.to_le_bytes());
        out.extend_from_slice(&word(row.experts.len())?);
        for expert in &row.experts {
            out.extend_from_slice(&word(*expert)?);
        }
        append_values(&mut out, &row.input, hidden)?;
    }
    Ok(out)
}

pub fn decode_request(
    bytes: &[u8],
    hidden: usize,
    max_rows: usize,
    max_selected: usize,
) -> Result<Request, String> {
    let head = read_header(bytes, REQUEST_MAGIC, hidden, max_rows, max_selected)?;
    if bytes.len() != request_len(hidden, head.rows, head.pairs)? {
        return Err("expert rows request length mismatch".into());
    }
    let mut rows = Vec::with_capacity(head.rows);
    let mut at = HEADER_BYTES;
    for _ in 0..head.rows {
        // The total length is already exact, but a row's own count decides
        // where the next row starts, so bound each read against the frame.
        let id = bytes
            .get(at..at + WORD)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .ok_or("expert rows request truncated")?;
        let count = bytes
            .get(at + WORD..at + 2 * WORD)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
            .ok_or("expert rows request truncated")?;
        if count == 0 || count > max_selected {
            return Err("expert rows selection count outside bound".into());
        }
        let experts_end = at + 2 * WORD + count * WORD;
        let input_end = experts_end + hidden * WORD;
        let (experts, input) = bytes
            .get(at + 2 * WORD..experts_end)
            .zip(bytes.get(experts_end..input_end))
            .ok_or("expert rows request truncated")?;
        rows.push(Row {
            id,
            experts: experts
                .chunks_exact(WORD)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
                .collect(),
            input: read_values(input)?,
        });
        at = input_end;
    }
    if at != bytes.len() || check_rows(&rows, max_selected)? != head.pairs {
        return Err("expert rows request counts disagree with its header".into());
    }
    Ok(Request {
        handle: head.handle,
        sequence: head.sequence,
        layer: head.layer,
        rows,
    })
}

pub fn encode_response(
    handle: Handle,
    sequence: u64,
    layer: usize,
    hidden: usize,
    rows: usize,
    outputs: &[Output],
) -> Result<Vec<u8>, String> {
    if outputs.is_empty() || rows == 0 || rows > outputs.len() {
        return Err("expert rows response must answer every row".into());
    }
    let mut out = header(
        RESPONSE_MAGIC,
        handle,
        sequence,
        layer,
        hidden,
        rows,
        outputs.len(),
    )?;
    out.reserve(response_len(hidden, outputs.len())? - HEADER_BYTES);
    for output in outputs {
        out.extend_from_slice(&output.row.to_le_bytes());
        out.extend_from_slice(&word(output.expert)?);
        append_values(&mut out, &output.values, hidden)?;
    }
    Ok(out)
}

pub fn decode_response(
    bytes: &[u8],
    hidden: usize,
    max_rows: usize,
    max_selected: usize,
) -> Result<Response, String> {
    let head = read_header(bytes, RESPONSE_MAGIC, hidden, max_rows, max_selected)?;
    if bytes.len() != response_len(hidden, head.pairs)? {
        return Err("expert rows response length mismatch".into());
    }
    let outputs = bytes[HEADER_BYTES..]
        .chunks_exact(2 * WORD + hidden * WORD)
        .map(|chunk| {
            Ok(Output {
                row: u32::from_le_bytes(chunk[..WORD].try_into().unwrap()),
                expert: word_at(chunk, WORD),
                values: read_values(&chunk[2 * WORD..])?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Response {
        handle: head.handle,
        sequence: head.sequence,
        layer: head.layer,
        rows: head.rows,
        outputs,
    })
}

/// Every requested `(row, expert)` pair answered exactly once, and nothing
/// else. Order is free; identity is not. A caller runs this on a complete
/// response before using any of it, so a short, padded or substituted reply
/// is an error for the whole frame and never a partial result.
pub fn check_correspondence(rows: &[Row], outputs: &[Output], hidden: usize) -> Result<(), String> {
    let mut wanted = std::collections::BTreeSet::new();
    for row in rows {
        for expert in &row.experts {
            wanted.insert((row.id, *expert));
        }
    }
    if outputs.len() != wanted.len() {
        return Err("expert rows response pair count mismatch".into());
    }
    for output in outputs {
        if output.values.len() != hidden || output.values.iter().any(|v| !v.is_finite()) {
            return Err("expert rows output width or finiteness mismatch".into());
        }
        if !wanted.remove(&(output.row, output.expert)) {
            return Err("unsolicited or duplicated (row, expert) in expert rows response".into());
        }
    }
    Ok(())
}
