//! Artifact-bound selected-expert transforms. No routing weights on the wire.
use serde::{Deserialize, Serialize};
pub const PATH: &str = "/v1/vindex3/experts";
pub const OPEN_PATH: &str = "/v1/vindex3/experts/open";
pub const BINARY_PATH: &str = "/v1/vindex3/experts/binary";
pub const CONTENT_TYPE: &str = "application/vnd.larql.v3-experts.f32";
pub const HEADER_BYTES: usize = 40;
/// Optional HTTP diagnostics only: never part of the exact numerical frame.
pub const PROFILE_HEADER: &str = "x-larql-expert-profile";
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WorkerTiming {
    pub decode_ns: u64,
    pub queue_ns: u64,
    pub execute_ns: u64,
    pub experts_ns: u64,
    pub encode_ns: u64,
    pub handler_ns: u64,
}
pub type Handle = [u8; 16];
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub program: crate::vindex3::Binding,
    pub expert_start: usize,
    pub expert_end: usize,
    pub operands: Vec<crate::vindex3_ffn::OperandIdentity>,
    /// Canonical serialized byte-range ledger, including expert biases.
    pub regions: String,
}
impl Binding {
    pub fn validate(&self) -> Result<(), String> {
        self.program.validate()?;
        if self.expert_start >= self.expert_end
            || self.operands.is_empty()
            || self.regions.is_empty()
        {
            return Err("invalid expert binding ownership or operands".into());
        }
        Ok(())
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Opened {
    pub version: u32,
    pub handle: Handle,
    pub binding: Binding,
}

pub struct Request {
    pub handle: Handle,
    pub sequence: u64,
    pub layer: usize,
    pub experts: Vec<usize>,
    pub row: Vec<f32>,
}
pub struct Response {
    pub handle: Handle,
    pub sequence: u64,
    pub layer: usize,
    pub rows: Vec<(usize, Vec<f32>)>,
}
fn header(
    magic: &[u8; 4],
    handle: Handle,
    sequence: u64,
    layer: usize,
    hidden: usize,
    count: usize,
) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(HEADER_BYTES);
    out.extend_from_slice(magic);
    out.extend_from_slice(&handle);
    out.extend_from_slice(&sequence.to_le_bytes());
    for n in [layer, hidden, count] {
        out.extend_from_slice(
            &u32::try_from(n)
                .map_err(|_| "expert frame dimension overflow")?
                .to_le_bytes(),
        );
    }
    Ok(out)
}
fn read_header(
    bytes: &[u8],
    magic: &[u8; 4],
    hidden: usize,
    max_count: usize,
) -> Result<(Handle, u64, usize, usize), String> {
    if bytes.len() < HEADER_BYTES || &bytes[..4] != magic {
        return Err("invalid expert frame header".into());
    }
    let u32_at =
        |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap()) as usize;
    let count = u32_at(36);
    if hidden == 0 || u32_at(32) != hidden || count == 0 || count > max_count {
        return Err("expert frame width or count mismatch".into());
    }
    Ok((
        bytes[4..20].try_into().unwrap(),
        u64::from_le_bytes(bytes[20..28].try_into().unwrap()),
        u32_at(28),
        count,
    ))
}
fn append_row(out: &mut Vec<u8>, row: &[f32]) -> Result<(), String> {
    if row.is_empty() || row.iter().any(|x| !x.is_finite()) {
        return Err("expert carrier must be finite and nonempty".into());
    }
    for x in row {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    Ok(())
}
fn read_row(bytes: &[u8]) -> Result<Vec<f32>, String> {
    let row: Vec<_> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_bits(u32::from_le_bytes(b.try_into().unwrap())))
        .collect();
    if row.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite expert carrier".into());
    }
    Ok(row)
}
pub fn response_len(hidden: usize, count: usize) -> Result<usize, String> {
    hidden
        .checked_mul(4)
        .and_then(|n| n.checked_add(4))
        .and_then(|n| n.checked_mul(count))
        .and_then(|n| n.checked_add(HEADER_BYTES))
        .ok_or("expert frame length overflow".into())
}
pub fn encode_request(
    handle: Handle,
    sequence: u64,
    layer: usize,
    experts: &[usize],
    row: &[f32],
) -> Result<Vec<u8>, String> {
    if experts.is_empty() {
        return Err("empty expert selection".into());
    }
    let mut out = header(b"VEX1", handle, sequence, layer, row.len(), experts.len())?;
    let mut seen = std::collections::BTreeSet::new();
    for id in experts {
        if !seen.insert(id) {
            return Err("duplicate expert request".into());
        }
        out.extend_from_slice(
            &u32::try_from(*id)
                .map_err(|_| "expert ID overflow")?
                .to_le_bytes(),
        );
    }
    append_row(&mut out, row)?;
    Ok(out)
}
pub fn decode_request(bytes: &[u8], hidden: usize, max_count: usize) -> Result<Request, String> {
    let (handle, sequence, layer, count) = read_header(bytes, b"VEX1", hidden, max_count)?;
    let length = count
        .checked_add(hidden)
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| n.checked_add(HEADER_BYTES))
        .ok_or("expert frame length overflow")?;
    if bytes.len() != length {
        return Err("expert request length mismatch".into());
    }
    let experts: Vec<_> = bytes[HEADER_BYTES..HEADER_BYTES + count * 4]
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
        .collect();
    if experts
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len()
        != count
    {
        return Err("duplicate expert request".into());
    }
    Ok(Request {
        handle,
        sequence,
        layer,
        experts,
        row: read_row(&bytes[HEADER_BYTES + count * 4..])?,
    })
}
pub fn encode_response(
    handle: Handle,
    sequence: u64,
    layer: usize,
    hidden: usize,
    rows: &[(usize, Vec<f32>)],
) -> Result<Vec<u8>, String> {
    if rows.is_empty() {
        return Err("empty expert response".into());
    }
    let mut out = header(b"VEY1", handle, sequence, layer, hidden, rows.len())?;
    for (id, row) in rows {
        if row.len() != hidden {
            return Err("expert output width mismatch".into());
        }
        out.extend_from_slice(
            &u32::try_from(*id)
                .map_err(|_| "expert ID overflow")?
                .to_le_bytes(),
        );
        append_row(&mut out, row)?;
    }
    Ok(out)
}
pub fn decode_response(bytes: &[u8], hidden: usize, max_count: usize) -> Result<Response, String> {
    let (handle, sequence, layer, count) = read_header(bytes, b"VEY1", hidden, max_count)?;
    if bytes.len() != response_len(hidden, count)? {
        return Err("expert response length mismatch".into());
    }
    let mut rows = Vec::with_capacity(count);
    for chunk in bytes[HEADER_BYTES..].chunks_exact(4 + hidden * 4) {
        rows.push((
            u32::from_le_bytes(chunk[..4].try_into().unwrap()) as usize,
            read_row(&chunk[4..])?,
        ));
    }
    Ok(Response {
        handle,
        sequence,
        layer,
        rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_carriers_and_strict_frame_bounds() {
        let row = [0.0, -0.0, f32::from_bits(1), f32::MAX];
        let request = encode_request([7; 16], 91, 3, &[9, 2], &row).unwrap();
        let decoded = decode_request(&request, 4, 2).unwrap();
        assert_eq!(decoded.handle, [7; 16]);
        assert_eq!(decoded.sequence, 91);
        assert_eq!(decoded.layer, 3);
        assert_eq!(decoded.experts, [9, 2]);
        assert_eq!(
            decoded.row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            row.map(f32::to_bits)
        );
        let response =
            encode_response([7; 16], 91, 3, 4, &[(2, row.to_vec()), (9, row.to_vec())]).unwrap();
        let decoded = decode_response(&response, 4, 2).unwrap();
        assert_eq!(decoded.rows[0].0, 2);
        assert_eq!(
            decoded.rows[0]
                .1
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            row.map(f32::to_bits)
        );
        for n in 0..request.len() {
            assert!(decode_request(&request[..n], 4, 2).is_err());
        }
        for n in 0..response.len() {
            assert!(decode_response(&response[..n], 4, 2).is_err());
        }
        for original in [&request, &response] {
            let decode = |bytes: &[u8]| {
                if original[2] == b'X' {
                    decode_request(bytes, 4, 2).map(|_| ())
                } else {
                    decode_response(bytes, 4, 2).map(|_| ())
                }
            };
            let mut bad = original.clone();
            bad.push(0);
            assert!(decode(&bad).is_err());
            for offset in [0, 32, 36] {
                let mut bad = original.clone();
                bad[offset] ^= 0x80;
                assert!(decode(&bad).is_err());
            }
            let mut bad = original.clone();
            let end = bad.len();
            bad[end - 4..].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
            assert!(decode(&bad).is_err());
        }
        let mut duplicate = request.clone();
        duplicate[44..48].copy_from_slice(&9u32.to_le_bytes());
        assert!(decode_request(&duplicate, 4, 2).is_err());
        assert!(encode_request([0; 16], 0, 0, &[1, 1], &row).is_err());
        assert!(decode_request(&request, 4, 1).is_err());
        assert!(decode_response(&response, 3, 2).is_err());
        assert!(response_len(usize::MAX, 2).is_err());
    }
}
