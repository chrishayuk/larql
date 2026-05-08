//! Versioned binary wire format for `KvCache` snapshots.
//!
//! Used by `/v1/kv-cache/snapshot` and `/v1/kv-cache/restore` (HTTP)
//! and the gRPC equivalents. Round-trips a full per-layer cache —
//! both FP32 and RotorQuant-compressed slots — through a self-
//! describing byte blob.
//!
//! The format is **not stable across versions**. The 16-bit `version`
//! field is the only forward contract: a server SHALL reject snapshots
//! whose version it does not recognise.
//!
//! ## Layout
//!
//! ```text
//! 0x00  u32 le   magic = 0x4C41_5141 ('LAQA' little-endian)
//! 0x04  u16 le   version = 1
//! 0x06  u16 le   flags
//!                 bit 0  any layer is RotorQuant-compressed
//!                 bit 1  any layer is FP32-populated
//!                 bit 2..15 reserved (must be 0)
//! 0x08  u32 le   num_layers
//! 0x0C  u32 le   max_window  (0 ⇒ unbounded)
//! 0x10  u32 le   next_position
//! 0x14  u32 le   kv_format    (0 ⇒ none, 1 = Planar3, 2 = Planar4,
//!                              3 = Iso3, 4 = Iso4)
//! 0x18  u32 le   promote_on_read_count_lo
//! 0x1C  u32 le   promote_on_read_count_hi
//! 0x20  per-layer offsets: [u64 le; num_layers]
//!         each offset is from the start of the blob; points at the
//!         layer's tagged payload.
//! ...   payload, packed in layer order:
//!         u8 tag   (0 = empty, 1 = fp32, 2 = quantized)
//!         if fp32:
//!           u32 le  k_rows
//!           u32 le  k_cols
//!           [f32 le; k_rows × k_cols]
//!           u32 le  v_rows
//!           u32 le  v_cols
//!           [f32 le; v_rows × v_cols]
//!         if quantized:
//!           u8      kv_format byte (1..=4)
//!           — K block —
//!           u32 le  n_rows
//!           u32 le  head_dim
//!           u32 le  codes_len
//!           [u8;    codes_len]
//!           u32 le  norms_len  (== n_rows)
//!           [f32;   norms_len]
//!           u32 le  rot_len    (== n_rows × n_blocks_per_row)
//!           [u16;   rot_len]
//!           — V block (same layout) —
//! ```
//!
//! Endianness is little-endian throughout. All multi-byte fields are
//! aligned to their natural width (the offsets table guarantees the
//! per-layer payload starts on an 8-byte boundary in practice, but
//! readers SHOULD NOT rely on alignment beyond the explicit u64
//! offsets).

use larql_inference::attention::decode::KvCache;
use larql_rotorquant::{KvFormat, QuantizedKv};
use ndarray::Array2;
use thiserror::Error;

/// Magic number `'LAQA'` (LARQL Attention snapshoT, capital A) in
/// little-endian byte order.
pub const SNAPSHOT_MAGIC: u32 = 0x4C41_5141;

/// Current snapshot version. Bumped on any wire-format change.
pub const SNAPSHOT_VERSION: u16 = 1;

const HEADER_BYTES: usize = 0x20;

const TAG_EMPTY: u8 = 0;
const TAG_FP32: u8 = 1;
const TAG_QUANTIZED: u8 = 2;

const FLAG_HAS_QUANTIZED: u16 = 1 << 0;
const FLAG_HAS_FP32: u16 = 1 << 1;

const FORMAT_NONE: u32 = 0;
const FORMAT_PLANAR3: u32 = 1;
const FORMAT_PLANAR4: u32 = 2;
const FORMAT_ISO3: u32 = 3;
const FORMAT_ISO4: u32 = 4;

#[derive(Error, Debug)]
pub enum SnapshotError {
    #[error("snapshot too short: have {have} bytes, need at least {need}")]
    TooShort { have: usize, need: usize },
    #[error("magic mismatch: want 0x{want:08X}, got 0x{got:08X}")]
    MagicMismatch { want: u32, got: u32 },
    #[error("snapshot version {got} is not supported by this server (supported: {supported:?})")]
    UnsupportedVersion { got: u16, supported: Vec<u16> },
    #[error("unknown layer tag {tag} at offset {offset}")]
    UnknownTag { tag: u8, offset: usize },
    #[error("unknown kv_format byte {byte}")]
    UnknownKvFormat { byte: u8 },
    #[error("payload would read past end (layer {layer}, op {op})")]
    Truncated { layer: usize, op: &'static str },
    #[error("dimension mismatch: {detail}")]
    DimensionMismatch { detail: String },
}

fn fmt_to_u32(fmt: KvFormat) -> u32 {
    match fmt {
        KvFormat::Planar3 => FORMAT_PLANAR3,
        KvFormat::Planar4 => FORMAT_PLANAR4,
        KvFormat::Iso3 => FORMAT_ISO3,
        KvFormat::Iso4 => FORMAT_ISO4,
    }
}

fn u32_to_fmt(byte: u32) -> Result<Option<KvFormat>, SnapshotError> {
    Ok(match byte {
        FORMAT_NONE => None,
        FORMAT_PLANAR3 => Some(KvFormat::Planar3),
        FORMAT_PLANAR4 => Some(KvFormat::Planar4),
        FORMAT_ISO3 => Some(KvFormat::Iso3),
        FORMAT_ISO4 => Some(KvFormat::Iso4),
        other => {
            // u32 path bottoms out at u8 representation; downstream
            // callers see whichever byte was provided.
            return Err(SnapshotError::UnknownKvFormat { byte: other as u8 });
        }
    })
}

fn put_u8(buf: &mut Vec<u8>, v: u8) {
    buf.push(v);
}

fn put_u16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_f32_slice(buf: &mut Vec<u8>, vs: &[f32]) {
    for v in vs {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

fn put_u16_slice(buf: &mut Vec<u8>, vs: &[u16]) {
    for v in vs {
        buf.extend_from_slice(&v.to_le_bytes());
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn need(&self, n: usize) -> Result<(), SnapshotError> {
        if self.pos + n > self.bytes.len() {
            return Err(SnapshotError::TooShort {
                have: self.bytes.len(),
                need: self.pos + n,
            });
        }
        Ok(())
    }

    fn read_u8(&mut self) -> Result<u8, SnapshotError> {
        self.need(1)?;
        let v = self.bytes[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_u16(&mut self) -> Result<u16, SnapshotError> {
        self.need(2)?;
        let v = u16::from_le_bytes(self.bytes[self.pos..self.pos + 2].try_into().unwrap());
        self.pos += 2;
        Ok(v)
    }

    fn read_u32(&mut self) -> Result<u32, SnapshotError> {
        self.need(4)?;
        let v = u32::from_le_bytes(self.bytes[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_u64(&mut self) -> Result<u64, SnapshotError> {
        self.need(8)?;
        let v = u64::from_le_bytes(self.bytes[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        Ok(v)
    }

    fn read_bytes(&mut self, n: usize) -> Result<&'a [u8], SnapshotError> {
        self.need(n)?;
        let s = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    fn read_f32_vec(&mut self, n: usize) -> Result<Vec<f32>, SnapshotError> {
        self.need(4 * n)?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.read_f32()?);
        }
        Ok(out)
    }

    fn read_f32(&mut self) -> Result<f32, SnapshotError> {
        self.need(4)?;
        let v = f32::from_le_bytes(self.bytes[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        Ok(v)
    }

    fn read_u16_vec(&mut self, n: usize) -> Result<Vec<u16>, SnapshotError> {
        self.need(2 * n)?;
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.read_u16()?);
        }
        Ok(out)
    }
}

/// Serialise a `KvCache` into the v1 wire format.
pub fn serialize(cache: &KvCache) -> Vec<u8> {
    let num_layers = cache.layers.len();
    let mut flags: u16 = 0;
    if cache.layers.iter().any(|l| l.is_some()) {
        flags |= FLAG_HAS_FP32;
    }
    if cache.quantized_kv.iter().any(|q| q.is_some()) {
        flags |= FLAG_HAS_QUANTIZED;
    }

    // Build payload first to know offsets.
    let mut payload = Vec::<u8>::new();
    let mut offsets = Vec::<u64>::with_capacity(num_layers);
    let header_and_offsets_len = HEADER_BYTES + 8 * num_layers;

    for layer in 0..num_layers {
        let layer_offset = (header_and_offsets_len + payload.len()) as u64;
        offsets.push(layer_offset);

        let fp32 = cache.layers.get(layer).and_then(|s| s.as_ref());
        let quantized = cache.quantized_kv.get(layer).and_then(|s| s.as_ref());

        match (fp32, quantized) {
            (Some((k, v)), _) => {
                put_u8(&mut payload, TAG_FP32);
                serialize_fp32_pair(&mut payload, k, v);
            }
            (None, Some((k, v))) => {
                put_u8(&mut payload, TAG_QUANTIZED);
                serialize_quantized_pair(&mut payload, k, v);
            }
            (None, None) => {
                put_u8(&mut payload, TAG_EMPTY);
            }
        }
    }

    let mut out = Vec::with_capacity(header_and_offsets_len + payload.len());
    // Header.
    put_u32(&mut out, SNAPSHOT_MAGIC);
    put_u16(&mut out, SNAPSHOT_VERSION);
    put_u16(&mut out, flags);
    put_u32(&mut out, num_layers as u32);
    put_u32(&mut out, cache.max_window.unwrap_or(0) as u32);
    put_u32(&mut out, cache.next_position as u32);
    put_u32(
        &mut out,
        cache.kv_format.map(fmt_to_u32).unwrap_or(FORMAT_NONE),
    );
    put_u32(&mut out, (cache.promote_on_read_count & 0xffff_ffff) as u32);
    put_u32(&mut out, (cache.promote_on_read_count >> 32) as u32);
    debug_assert_eq!(out.len(), HEADER_BYTES);

    // Offsets.
    for off in &offsets {
        put_u64(&mut out, *off);
    }

    out.extend_from_slice(&payload);
    out
}

fn serialize_fp32_pair(buf: &mut Vec<u8>, k: &Array2<f32>, v: &Array2<f32>) {
    serialize_array2(buf, k);
    serialize_array2(buf, v);
}

fn serialize_array2(buf: &mut Vec<u8>, a: &Array2<f32>) {
    let (rows, cols) = a.dim();
    put_u32(buf, rows as u32);
    put_u32(buf, cols as u32);
    // ndarray default layout is row-major (C order); to_owned() copies
    // into a contiguous standard layout, and `as_slice()` is then Some.
    let owned;
    let slice = match a.as_slice() {
        Some(s) => s,
        None => {
            owned = a.to_owned();
            owned.as_slice().expect("standard layout after to_owned")
        }
    };
    put_f32_slice(buf, slice);
}

fn serialize_quantized_pair(buf: &mut Vec<u8>, k: &QuantizedKv, v: &QuantizedKv) {
    debug_assert_eq!(k.format, v.format);
    put_u8(buf, fmt_to_u32(k.format) as u8);
    serialize_quantized(buf, k);
    serialize_quantized(buf, v);
}

fn serialize_quantized(buf: &mut Vec<u8>, q: &QuantizedKv) {
    put_u32(buf, q.n_rows as u32);
    put_u32(buf, q.head_dim as u32);
    put_u32(buf, q.codes.len() as u32);
    buf.extend_from_slice(&q.codes);
    put_u32(buf, q.norms.len() as u32);
    put_f32_slice(buf, &q.norms);
    put_u32(buf, q.rotation_indices.len() as u32);
    put_u16_slice(buf, &q.rotation_indices);
}

/// Deserialise a v1 snapshot back into a `KvCache`.
pub fn deserialize(bytes: &[u8]) -> Result<KvCache, SnapshotError> {
    if bytes.len() < HEADER_BYTES {
        return Err(SnapshotError::TooShort {
            have: bytes.len(),
            need: HEADER_BYTES,
        });
    }
    let mut r = Reader::new(bytes);

    let magic = r.read_u32()?;
    if magic != SNAPSHOT_MAGIC {
        return Err(SnapshotError::MagicMismatch {
            want: SNAPSHOT_MAGIC,
            got: magic,
        });
    }
    let version = r.read_u16()?;
    if version != SNAPSHOT_VERSION {
        return Err(SnapshotError::UnsupportedVersion {
            got: version,
            supported: vec![SNAPSHOT_VERSION],
        });
    }
    let _flags = r.read_u16()?;
    let num_layers = r.read_u32()? as usize;
    let max_window_raw = r.read_u32()? as usize;
    let next_position = r.read_u32()? as usize;
    let kv_format_raw = r.read_u32()?;
    let promote_lo = r.read_u32()? as u64;
    let promote_hi = r.read_u32()? as u64;
    let promote_on_read_count = (promote_hi << 32) | promote_lo;

    let kv_format = u32_to_fmt(kv_format_raw)?;

    let mut offsets = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        offsets.push(r.read_u64()? as usize);
    }

    let mut layers: Vec<Option<larql_inference::attention::SharedKV>> = vec![None; num_layers];
    let mut quantized_kv: Vec<Option<(QuantizedKv, QuantizedKv)>> = vec![None; num_layers];

    for (layer, &offset) in offsets.iter().enumerate() {
        if offset > bytes.len() {
            return Err(SnapshotError::Truncated {
                layer,
                op: "layer offset",
            });
        }
        r.pos = offset;
        let tag = r.read_u8()?;
        match tag {
            TAG_EMPTY => {}
            TAG_FP32 => {
                let k = read_array2(&mut r, layer)?;
                let v = read_array2(&mut r, layer)?;
                layers[layer] = Some((k, v));
            }
            TAG_QUANTIZED => {
                let fmt_byte = r.read_u8()?;
                let format = u32_to_fmt(fmt_byte as u32)?
                    .ok_or(SnapshotError::UnknownKvFormat { byte: fmt_byte })?;
                let k = read_quantized(&mut r, format, layer)?;
                let v = read_quantized(&mut r, format, layer)?;
                quantized_kv[layer] = Some((k, v));
            }
            other => return Err(SnapshotError::UnknownTag { tag: other, offset }),
        }
    }

    Ok(KvCache {
        layers,
        max_window: if max_window_raw == 0 {
            None
        } else {
            Some(max_window_raw)
        },
        next_position,
        kv_format,
        quantized_kv,
        promote_on_read_count,
    })
}

fn read_array2(r: &mut Reader<'_>, layer: usize) -> Result<Array2<f32>, SnapshotError> {
    let rows = r.read_u32()? as usize;
    let cols = r.read_u32()? as usize;
    let total = rows
        .checked_mul(cols)
        .ok_or(SnapshotError::DimensionMismatch {
            detail: format!("layer {layer} rows*cols overflow"),
        })?;
    let data = r.read_f32_vec(total)?;
    Array2::from_shape_vec((rows, cols), data).map_err(|e| SnapshotError::DimensionMismatch {
        detail: format!("layer {layer}: {e}"),
    })
}

fn read_quantized(
    r: &mut Reader<'_>,
    format: KvFormat,
    layer: usize,
) -> Result<QuantizedKv, SnapshotError> {
    let n_rows = r.read_u32()? as usize;
    let head_dim = r.read_u32()? as usize;
    let codes_len = r.read_u32()? as usize;
    let codes = r.read_bytes(codes_len)?.to_vec();
    let norms_len = r.read_u32()? as usize;
    if norms_len != n_rows {
        return Err(SnapshotError::DimensionMismatch {
            detail: format!("layer {layer}: norms_len {norms_len} != n_rows {n_rows}"),
        });
    }
    let norms = r.read_f32_vec(norms_len)?;
    let rot_len = r.read_u32()? as usize;
    let rotation_indices = r.read_u16_vec(rot_len)?;
    Ok(QuantizedKv {
        format,
        n_rows,
        head_dim,
        codes,
        norms,
        rotation_indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn make_fp32_cache(num_layers: usize, seq_len: usize, kv_dim: usize) -> KvCache {
        let mut cache = KvCache::with_layers(num_layers);
        for layer in 0..num_layers {
            let k = Array2::<f32>::from_shape_fn((seq_len, kv_dim), |(r, c)| {
                (layer * 1000 + r * kv_dim + c) as f32 * 0.001
            });
            let v = Array2::<f32>::from_shape_fn((seq_len, kv_dim), |(r, c)| {
                -((layer * 1000 + r * kv_dim + c) as f32) * 0.001
            });
            cache.layers[layer] = Some((k, v));
        }
        cache.next_position = seq_len;
        cache
    }

    #[test]
    fn magic_and_version_are_present_at_known_offsets() {
        let cache = make_fp32_cache(2, 3, 4);
        let bytes = serialize(&cache);
        assert_eq!(bytes[0], 0x41);
        assert_eq!(bytes[1], 0x51);
        assert_eq!(bytes[2], 0x41);
        assert_eq!(bytes[3], 0x4C);
        assert_eq!(bytes[4], 0x01);
        assert_eq!(bytes[5], 0x00);
    }

    #[test]
    fn round_trip_fp32_is_byte_identical() {
        let cache = make_fp32_cache(3, 5, 8);
        let bytes = serialize(&cache);
        let restored = deserialize(&bytes).expect("deserialize");
        assert_eq!(restored.layers.len(), 3);
        assert_eq!(restored.next_position, 5);
        for layer in 0..3 {
            let (k0, v0) = cache.layers[layer].as_ref().unwrap();
            let (k1, v1) = restored.layers[layer].as_ref().unwrap();
            assert_eq!(k0, k1, "layer {layer} K mismatch");
            assert_eq!(v0, v1, "layer {layer} V mismatch");
        }
    }

    #[test]
    fn round_trip_preserves_max_window_and_promote_count() {
        let mut cache = KvCache::with_window(2, 64);
        cache.next_position = 10;
        cache.promote_on_read_count = 0x1_0000_0007;
        let bytes = serialize(&cache);
        let restored = deserialize(&bytes).expect("deserialize");
        assert_eq!(restored.max_window, Some(64));
        assert_eq!(restored.next_position, 10);
        assert_eq!(restored.promote_on_read_count, 0x1_0000_0007);
    }

    #[test]
    fn round_trip_compressed_layer() {
        let mut cache = make_fp32_cache(2, 4, 8);
        cache.set_kv_format(KvFormat::Iso4);
        // Compress layer 0; layer 1 stays FP32.
        cache.quantize_layer(0);
        assert!(cache.is_layer_compressed(0));
        assert!(!cache.is_layer_compressed(1));

        let bytes = serialize(&cache);
        let restored = deserialize(&bytes).expect("deserialize");

        assert_eq!(restored.kv_format, Some(KvFormat::Iso4));
        assert!(restored.layers[0].is_none());
        assert!(restored.quantized_kv[0].is_some());
        assert!(restored.layers[1].is_some());
        assert!(restored.quantized_kv[1].is_none());

        // Compressed K and V byte payloads should be byte-identical.
        let (k0, v0) = cache.quantized_kv[0].as_ref().unwrap();
        let (k1, v1) = restored.quantized_kv[0].as_ref().unwrap();
        assert_eq!(k0.format, k1.format);
        assert_eq!(k0.n_rows, k1.n_rows);
        assert_eq!(k0.head_dim, k1.head_dim);
        assert_eq!(k0.codes, k1.codes);
        assert_eq!(k0.norms, k1.norms);
        assert_eq!(k0.rotation_indices, k1.rotation_indices);
        assert_eq!(v0.codes, v1.codes);
        assert_eq!(v0.norms, v1.norms);
        assert_eq!(v0.rotation_indices, v1.rotation_indices);
    }

    #[test]
    fn empty_layers_round_trip() {
        let cache = KvCache::with_layers(4);
        let bytes = serialize(&cache);
        let restored = deserialize(&bytes).expect("deserialize");
        assert_eq!(restored.layers.len(), 4);
        for layer in 0..4 {
            assert!(restored.layers[layer].is_none());
            assert!(restored.quantized_kv[layer].is_none());
        }
    }

    #[test]
    fn magic_mismatch_is_rejected() {
        let cache = make_fp32_cache(1, 1, 4);
        let mut bytes = serialize(&cache);
        // Corrupt magic.
        bytes[0] = 0xFF;
        bytes[1] = 0xFF;
        match deserialize(&bytes) {
            Err(SnapshotError::MagicMismatch { .. }) => {}
            other => panic!("expected MagicMismatch, got {other:?}"),
        }
    }

    #[test]
    fn unknown_version_is_rejected() {
        let cache = make_fp32_cache(1, 1, 4);
        let mut bytes = serialize(&cache);
        // Bump version to 0xFF.
        bytes[4] = 0xFF;
        bytes[5] = 0x00;
        match deserialize(&bytes) {
            Err(SnapshotError::UnsupportedVersion { got, supported }) => {
                assert_eq!(got, 0xFF);
                assert_eq!(supported, vec![SNAPSHOT_VERSION]);
            }
            other => panic!("expected UnsupportedVersion, got {other:?}"),
        }
    }

    #[test]
    fn truncated_input_is_rejected_cleanly() {
        let cache = make_fp32_cache(2, 3, 4);
        let bytes = serialize(&cache);
        let truncated = &bytes[..bytes.len() / 2];
        // Should error, not panic.
        assert!(deserialize(truncated).is_err());
    }
}
