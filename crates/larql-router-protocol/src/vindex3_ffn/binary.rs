//! Exact f32 wire v1. All integers and IEEE-754 bits are little-endian.
//! Header: magic[4], worker handle[16], sequence u64, layer u32, width u32.
use super::Binding;
pub const OPEN_PATH: &str = "/v1/vindex3/ffn/open";
pub const PATH: &str = "/v1/vindex3/ffn/binary";
pub const CONTENT_TYPE: &str = "application/vnd.larql.v3-ffn.f32";
pub const HEADER_BYTES: usize = 36;
pub const VERSION: u32 = 1;
pub type Handle = [u8; 16];

#[derive(Debug, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Opened {
    pub version: u32,
    pub handle: Handle,
    pub binding: Binding,
}
#[derive(Clone, Copy)]
pub enum Direction {
    Request,
    Response,
}
impl Direction {
    fn magic(self) -> &'static [u8; 4] {
        match self {
            Self::Request => b"VFF1",
            Self::Response => b"VFR1",
        }
    }
}
#[derive(Debug)]
pub struct Frame {
    pub handle: Handle,
    pub sequence: u64,
    pub layer: u32,
    pub row: Vec<f32>,
}
pub fn encode(
    direction: Direction,
    handle: Handle,
    sequence: u64,
    layer: usize,
    row: &[f32],
) -> Result<Vec<u8>, String> {
    let layer = u32::try_from(layer).map_err(|_| "layer exceeds wire range")?;
    let width = u32::try_from(row.len()).map_err(|_| "width exceeds wire range")?;
    let len = row
        .len()
        .checked_mul(4)
        .and_then(|n| n.checked_add(HEADER_BYTES))
        .ok_or("frame length overflow")?;
    if row.is_empty() || row.iter().any(|x| !x.is_finite()) {
        return Err("carrier must contain finite f32 values".into());
    }
    let mut out = Vec::with_capacity(len);
    out.extend_from_slice(direction.magic());
    out.extend_from_slice(&handle);
    out.extend_from_slice(&sequence.to_le_bytes());
    out.extend_from_slice(&layer.to_le_bytes());
    out.extend_from_slice(&width.to_le_bytes());
    for x in row {
        out.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    Ok(out)
}
pub fn decode(bytes: &[u8], direction: Direction, hidden: usize) -> Result<Frame, String> {
    let expected = hidden
        .checked_mul(4)
        .and_then(|n| n.checked_add(HEADER_BYTES))
        .ok_or("frame length overflow")?;
    if hidden == 0
        || bytes.len() != expected
        || bytes.get(..4) != Some(direction.magic().as_slice())
    {
        return Err("invalid FFN frame version, direction or length".into());
    }
    let width = u32::from_le_bytes(bytes[32..36].try_into().unwrap()) as usize;
    if width != hidden {
        return Err("FFN frame width disagrees with binding".into());
    }
    let row: Vec<f32> = bytes[HEADER_BYTES..]
        .chunks_exact(4)
        .map(|b| f32::from_bits(u32::from_le_bytes(b.try_into().unwrap())))
        .collect();
    if row.iter().any(|x| !x.is_finite()) {
        return Err("non-finite FFN carrier".into());
    }
    Ok(Frame {
        handle: bytes[4..20].try_into().unwrap(),
        sequence: u64::from_le_bytes(bytes[20..28].try_into().unwrap()),
        layer: u32::from_le_bytes(bytes[28..32].try_into().unwrap()),
        row,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_layout_and_finite_bits_are_exact() {
        let row = [0.0, -0.0, f32::from_bits(1), f32::MIN, f32::MAX, 1.25];
        let bytes = encode(
            Direction::Request,
            [7; 16],
            0x0807060504030201,
            0x04030201,
            &row,
        )
        .unwrap();
        assert_eq!(&bytes[20..32], &[1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4]);
        assert_eq!(bytes.len(), HEADER_BYTES + row.len() * 4);
        let decoded = decode(&bytes, Direction::Request, row.len()).unwrap();
        assert_eq!(
            decoded.row.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            row.map(f32::to_bits)
        );
        assert!(decode(&bytes, Direction::Response, row.len()).is_err());
        assert!(decode(&bytes, Direction::Request, row.len() + 1).is_err());
        for end in 0..bytes.len() {
            assert!(decode(&bytes[..end], Direction::Request, row.len()).is_err());
        }
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(decode(&extra, Direction::Request, row.len()).is_err());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(encode(Direction::Request, [0; 16], 0, 0, &[value]).is_err());
            let mut corrupt = bytes.clone();
            corrupt[HEADER_BYTES..HEADER_BYTES + 4].copy_from_slice(&value.to_bits().to_le_bytes());
            assert!(decode(&corrupt, Direction::Request, row.len()).is_err());
        }
    }
}
