//! GGUF binary **writer** — the inverse of [`reader`](super::reader) /
//! [`parser`](super::parser).
//!
//! Serializes metadata key/values + tensors into a GGUF v3 byte stream that
//! the larql reader (and llama.cpp) can load. This is the format spine behind
//! `larql convert vindex-to-gguf`; tensor selection / naming / quant repacking
//! for a specific architecture lives in the CLI converter, not here.
//!
//! Layout (must match [`parser::open_single`](super::parser) byte-for-byte):
//! ```text
//!   magic u32 = "GGUF"   | version u32 = 3
//!   tensor_count u64     | metadata_kv_count u64
//!   metadata_kv[]: key:gguf_string, value_type:u32, value:payload
//!   tensor_info[]: name:gguf_string, n_dims:u32, dims:u64[n_dims],
//!                  type:u32, offset:u64  (offset is data-section-relative)
//!   <pad to 32>          | tensor_data[]  (each tensor at its 32-aligned offset)
//! ```
//! The reader hardcodes a 32-byte alignment for the data section, so we use
//! the same here (llama.cpp's default `general.alignment` is also 32).
//!
//! [`GgufWriter`] holds every tensor in RAM (`GgufTensor::data`), which is
//! fine for fixtures and small models and unusable for a 20 GB export.
//! [`write_streaming`] is the sibling for that case: same bytes, constant
//! memory, payloads supplied by a callback as the file is written.

use std::io::{self, Write};
use std::path::Path;

use super::constants::{
    GGUF_MAGIC, GGUF_TYPE_ARRAY, GGUF_TYPE_BOOL, GGUF_TYPE_FLOAT32, GGUF_TYPE_FLOAT64,
    GGUF_TYPE_INT16, GGUF_TYPE_INT32, GGUF_TYPE_INT64, GGUF_TYPE_INT8, GGUF_TYPE_STRING,
    GGUF_TYPE_UINT16, GGUF_TYPE_UINT32, GGUF_TYPE_UINT64, GGUF_TYPE_UINT8,
};
use super::types::GgufValue;

const GGUF_VERSION: u32 = 3;
const GGUF_ALIGNMENT: u64 = 32;

/// One tensor to serialize.
///
/// `dims` are in **GGUF order** — fastest-varying axis first. For a row-major
/// `[rows, cols]` matrix that is `[cols, rows]` (llama.cpp's convention). The
/// caller is responsible for `ggml_type` (see [`crate::quant::ggml`]) matching
/// the byte layout in `data`.
pub struct GgufTensor {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub data: Vec<u8>,
}

/// Accumulates metadata + tensors, then serializes a GGUF v3 stream.
#[derive(Default)]
pub struct GgufWriter {
    metadata: Vec<(String, GgufValue)>,
    tensors: Vec<GgufTensor>,
}

impl GgufWriter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a metadata key/value. Insertion order is preserved (GGUF does not
    /// require a particular order, but stable ordering keeps output diffable).
    pub fn meta(&mut self, key: impl Into<String>, value: GgufValue) -> &mut Self {
        self.metadata.push((key.into(), value));
        self
    }

    /// Add a tensor to be written. Tensors are emitted in insertion order.
    pub fn tensor(&mut self, tensor: GgufTensor) -> &mut Self {
        self.tensors.push(tensor);
        self
    }

    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Serialize to an in-memory byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();

        // ── Header ──
        write_u32(&mut out, GGUF_MAGIC);
        write_u32(&mut out, GGUF_VERSION);
        write_u64(&mut out, self.tensors.len() as u64);
        write_u64(&mut out, self.metadata.len() as u64);

        // ── Metadata KVs ──
        for (key, value) in &self.metadata {
            write_string(&mut out, key);
            write_value(&mut out, value);
        }

        // ── Tensor info table ──
        // Data-section-relative offsets, each aligned to GGUF_ALIGNMENT.
        let mut offsets = Vec::with_capacity(self.tensors.len());
        let mut running = 0u64;
        for t in &self.tensors {
            offsets.push(running);
            running = align_up(running + t.data.len() as u64, GGUF_ALIGNMENT);
        }
        for (t, &offset) in self.tensors.iter().zip(&offsets) {
            write_string(&mut out, &t.name);
            write_u32(&mut out, t.dims.len() as u32);
            for &d in &t.dims {
                write_u64(&mut out, d);
            }
            write_u32(&mut out, t.ggml_type);
            write_u64(&mut out, offset);
        }

        // ── Data section ── starts at the next 32-byte boundary.
        let data_start = align_up(out.len() as u64, GGUF_ALIGNMENT);
        out.resize(data_start as usize, 0);
        for (t, &offset) in self.tensors.iter().zip(&offsets) {
            let want = data_start + offset;
            // `offset` is aligned by construction, so this only ever pads
            // forward from the previous tensor's unaligned end.
            out.resize(want as usize, 0);
            out.extend_from_slice(&t.data);
        }

        out
    }

    /// Serialize directly to `path`.
    pub fn write_to_file(&self, path: &Path) -> io::Result<()> {
        let bytes = self.to_bytes();
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(&bytes)?;
        f.flush()
    }
}

/// One tensor to stream: everything the header needs, without the bytes.
///
/// `len` must equal the payload the emitter will write. It is declared
/// up front because GGUF puts every tensor's offset in a table before
/// any payload — the format requires knowing the sizes in advance, so a
/// wrong `len` is a corrupt file rather than a short write.
pub struct GgufTensorDescriptor {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub len: u64,
}

/// Serialize a GGUF to `path` without holding tensor payloads in memory.
///
/// Byte-identical to [`GgufWriter::to_bytes`] for the same inputs — the
/// header, metadata and tensor table are built in memory (kilobytes),
/// and only the data section streams. `emit` is called once per tensor,
/// in order, with its index and the output sink; it must write exactly
/// `descriptors[i].len` bytes.
///
/// The length is verified after each tensor. An emitter that writes the
/// wrong amount produces a file whose every subsequent offset is wrong,
/// and a reader would follow those offsets into the middle of another
/// tensor and decode plausible nonsense — so this fails loudly at the
/// point the mistake is made.
pub fn write_streaming<F>(
    metadata: &[(String, GgufValue)],
    descriptors: &[GgufTensorDescriptor],
    path: &Path,
    mut emit: F,
) -> io::Result<()>
where
    F: FnMut(usize, &mut dyn Write) -> io::Result<()>,
{
    let mut head = Vec::new();
    write_u32(&mut head, GGUF_MAGIC);
    write_u32(&mut head, GGUF_VERSION);
    write_u64(&mut head, descriptors.len() as u64);
    write_u64(&mut head, metadata.len() as u64);
    for (key, value) in metadata {
        write_string(&mut head, key);
        write_value(&mut head, value);
    }

    let mut offsets = Vec::with_capacity(descriptors.len());
    let mut running = 0u64;
    for d in descriptors {
        offsets.push(running);
        running = align_up(running + d.len, GGUF_ALIGNMENT);
    }
    for (d, &offset) in descriptors.iter().zip(&offsets) {
        write_string(&mut head, &d.name);
        write_u32(&mut head, d.dims.len() as u32);
        for &dim in &d.dims {
            write_u64(&mut head, dim);
        }
        write_u32(&mut head, d.ggml_type);
        write_u64(&mut head, offset);
    }

    let data_start = align_up(head.len() as u64, GGUF_ALIGNMENT);
    let mut f = io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&head)?;
    // Pad to the data section, then to each tensor's declared offset.
    let mut written = head.len() as u64;
    let pad_to = |f: &mut dyn Write, written: &mut u64, target: u64| -> io::Result<()> {
        while *written < target {
            let chunk = (target - *written).min(4096) as usize;
            f.write_all(&vec![0u8; chunk])?;
            *written += chunk as u64;
        }
        Ok(())
    };
    pad_to(&mut f, &mut written, data_start)?;

    for (i, (d, &offset)) in descriptors.iter().zip(&offsets).enumerate() {
        pad_to(&mut f, &mut written, data_start + offset)?;
        let mut counter = CountingWriter {
            inner: &mut f,
            n: 0,
        };
        emit(i, &mut counter)?;
        let n = counter.n;
        if n != d.len {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "tensor `{}` declared {} bytes but its emitter wrote {n} —                      every later offset in this file would be wrong",
                    d.name, d.len
                ),
            ));
        }
        written += n;
    }
    f.flush()
}

/// Counts what passes through, so a declared length can be enforced.
struct CountingWriter<'a, W: Write> {
    inner: &'a mut W,
    n: u64,
}

impl<W: Write> Write for CountingWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.n += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn align_up(n: u64, align: u64) -> u64 {
    n.div_ceil(align) * align
}

// ── Byte writers (mirror `reader`'s read_* helpers) ──

fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    write_u64(out, s.len() as u64);
    out.extend_from_slice(s.as_bytes());
}

/// The GGUF metadata-value type tag for a value (matches `reader::read_value`).
fn value_type_tag(v: &GgufValue) -> u32 {
    match v {
        GgufValue::U8(_) => GGUF_TYPE_UINT8,
        GgufValue::I8(_) => GGUF_TYPE_INT8,
        GgufValue::U16(_) => GGUF_TYPE_UINT16,
        GgufValue::I16(_) => GGUF_TYPE_INT16,
        GgufValue::U32(_) => GGUF_TYPE_UINT32,
        GgufValue::I32(_) => GGUF_TYPE_INT32,
        GgufValue::F32(_) => GGUF_TYPE_FLOAT32,
        GgufValue::Bool(_) => GGUF_TYPE_BOOL,
        GgufValue::String(_) => GGUF_TYPE_STRING,
        GgufValue::Array(_) => GGUF_TYPE_ARRAY,
        GgufValue::U64(_) => GGUF_TYPE_UINT64,
        GgufValue::I64(_) => GGUF_TYPE_INT64,
        GgufValue::F64(_) => GGUF_TYPE_FLOAT64,
    }
}

/// Write a tagged metadata value: `value_type:u32` then payload.
fn write_value(out: &mut Vec<u8>, v: &GgufValue) {
    write_u32(out, value_type_tag(v));
    write_value_payload(out, v);
}

/// Write only the payload (no type tag) — used for array elements, which
/// carry one shared element-type tag for the whole array.
fn write_value_payload(out: &mut Vec<u8>, v: &GgufValue) {
    match v {
        GgufValue::U8(x) => out.push(*x),
        GgufValue::I8(x) => out.push(*x as u8),
        GgufValue::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::Bool(x) => out.push(if *x { 1 } else { 0 }),
        GgufValue::String(s) => write_string(out, s),
        GgufValue::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
        GgufValue::Array(elems) => {
            // All elements share one type tag (GGUF arrays are homogeneous).
            // Empty arrays default to UINT32 — harmless, llama.cpp tolerates it.
            let elem_tag = elems
                .first()
                .map(value_type_tag)
                .unwrap_or(GGUF_TYPE_UINT32);
            write_u32(out, elem_tag);
            write_u64(out, elems.len() as u64);
            for e in elems {
                write_value_payload(out, e);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loading::gguf::GgufFile;

    /// Build a GGUF with mixed metadata + two tensors, write to a temp file,
    /// reopen with the production reader, and assert everything round-trips.
    #[test]
    fn round_trips_through_the_reader() {
        let dir = std::env::temp_dir().join(format!("gguf_writer_rt_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rt.gguf");

        let mut w = GgufWriter::new();
        w.meta("general.architecture", GgufValue::String("gemma3".into()))
            .meta("test.u32", GgufValue::U32(4096))
            .meta("test.f32", GgufValue::F32(1.5))
            .meta("test.bool", GgufValue::Bool(true))
            .meta(
                "test.u32_array",
                GgufValue::Array(vec![
                    GgufValue::U32(10),
                    GgufValue::U32(20),
                    GgufValue::U32(30),
                ]),
            )
            .meta(
                "tokenizer.ggml.tokens",
                GgufValue::Array(vec![
                    GgufValue::String("<bos>".into()),
                    GgufValue::String("hello".into()),
                ]),
            );

        // F32 tensor type id from the shared ggml constants.
        let f32_ty = crate::quant::ggml::TYPE_F32;
        // Tensor A: 2x2 f32 [1,2,3,4]; dims in GGUF order [cols, rows] = [2, 2].
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let a_bytes: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.tensor(GgufTensor {
            name: "blk.0.attn_q.weight".into(),
            dims: vec![2, 2],
            ggml_type: f32_ty,
            data: a_bytes.clone(),
        });
        // Tensor B: 3-vector f32, odd byte length to exercise inter-tensor pad.
        let b: Vec<f32> = vec![5.0, 6.0, 7.0];
        let b_bytes: Vec<u8> = b.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.tensor(GgufTensor {
            name: "output_norm.weight".into(),
            dims: vec![3],
            ggml_type: f32_ty,
            data: b_bytes.clone(),
        });

        w.write_to_file(&path).unwrap();

        // Reopen with the production reader.
        let f = GgufFile::open(&path).unwrap();

        // Metadata.
        assert_eq!(
            f.metadata.get("general.architecture").unwrap().as_str(),
            Some("gemma3")
        );
        assert_eq!(f.metadata.get("test.u32").unwrap().as_u32(), Some(4096));
        assert_eq!(f.metadata.get("test.f32").unwrap().as_f64(), Some(1.5));
        match f.metadata.get("test.bool").unwrap() {
            GgufValue::Bool(b) => assert!(*b),
            other => panic!("expected Bool, got {other:?}"),
        }
        match f.metadata.get("test.u32_array").unwrap() {
            GgufValue::Array(a) => {
                assert_eq!(a.len(), 3);
                assert_eq!(a[0].as_u32(), Some(10));
                assert_eq!(a[2].as_u32(), Some(30));
            }
            other => panic!("expected Array, got {other:?}"),
        }
        match f.metadata.get("tokenizer.ggml.tokens").unwrap() {
            GgufValue::Array(a) => {
                assert_eq!(a.len(), 2);
                assert_eq!(a[0].as_str(), Some("<bos>"));
                assert_eq!(a[1].as_str(), Some("hello"));
            }
            other => panic!("expected Array, got {other:?}"),
        }

        // Tensor infos.
        assert_eq!(f.tensor_infos.len(), 2);
        let ta = f
            .tensor_infos
            .iter()
            .find(|t| t.name() == "blk.0.attn_q.weight")
            .unwrap();
        assert_eq!(ta.dims(), &[2, 2]);
        assert_eq!(ta.tensor_type(), f32_ty);
        let tb = f
            .tensor_infos
            .iter()
            .find(|t| t.name() == "output_norm.weight")
            .unwrap();
        assert_eq!(tb.dims(), &[3]);

        // Tensor data — read back from data_offset + per-tensor offset.
        let raw = std::fs::read(&path).unwrap();
        let read_tensor = |info: &crate::loading::gguf::GgufTensorInfo, n: usize| -> Vec<u8> {
            let start = (f.data_offset + info.offset()) as usize;
            raw[start..start + n].to_vec()
        };
        assert_eq!(read_tensor(ta, a_bytes.len()), a_bytes);
        assert_eq!(read_tensor(tb, b_bytes.len()), b_bytes);

        // Per-tensor offsets must be 32-aligned (llama.cpp requirement).
        for info in &f.tensor_infos {
            assert_eq!(
                info.offset() % GGUF_ALIGNMENT,
                0,
                "tensor offset not aligned"
            );
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_writer_has_valid_header() {
        let w = GgufWriter::new();
        let bytes = w.to_bytes();
        // magic + version + 2×u64 counts = 24 bytes minimum.
        assert_eq!(&bytes[0..4], &GGUF_MAGIC.to_le_bytes());
        assert_eq!(&bytes[4..8], &GGUF_VERSION.to_le_bytes());
        assert_eq!(&bytes[8..16], &0u64.to_le_bytes()); // n_tensors
        assert_eq!(&bytes[16..24], &0u64.to_le_bytes()); // n_metadata
    }

    #[test]
    fn serializes_every_scalar_value_type_and_counts_tensors() {
        // `round_trips_through_the_reader` exercises String / U32 / F32 / Bool /
        // Array; this pins the remaining integer + float widths so
        // `value_type_tag` and `write_value_payload` are covered for every
        // `GgufValue` arm (and `tensor_count` for both empty and non-empty).
        let mut w = GgufWriter::new();
        w.meta("t.u8", GgufValue::U8(0x12))
            .meta("t.i8", GgufValue::I8(-3))
            .meta("t.u16", GgufValue::U16(0x1234))
            .meta("t.i16", GgufValue::I16(-300))
            .meta("t.i32", GgufValue::I32(-70_000))
            .meta("t.u64", GgufValue::U64(0x1122_3344_5566_7788))
            .meta("t.i64", GgufValue::I64(-5_000_000_000))
            .meta("t.f64", GgufValue::F64(123.456_789))
            // Array of a non-default element type recurses the payload writer.
            .meta(
                "t.i16_array",
                GgufValue::Array(vec![GgufValue::I16(-1), GgufValue::I16(2)]),
            );

        assert_eq!(w.tensor_count(), 0);
        w.tensor(GgufTensor {
            name: "blk.0.weight".into(),
            dims: vec![4, 4],
            ggml_type: 0, // F32
            data: vec![0u8; 4 * 4 * 4],
        });
        assert_eq!(w.tensor_count(), 1);

        // Serializing drives `write_value` (tag + payload) for every metadata
        // type above; the header counts must reflect what we added.
        let bytes = w.to_bytes();
        assert_eq!(&bytes[0..4], &GGUF_MAGIC.to_le_bytes());
        assert_eq!(&bytes[8..16], &1u64.to_le_bytes()); // n_tensors
        assert_eq!(&bytes[16..24], &9u64.to_le_bytes()); // n_metadata
    }

    /// **The streaming writer and the buffering writer must agree.**
    ///
    /// Not "produce a valid file" — produce the *same* file. The
    /// streaming path exists because a 20 GB export cannot be built in a
    /// `Vec`, and the moment it diverges from the reference
    /// implementation it becomes a second format nobody is testing.
    #[test]
    fn streaming_and_buffered_writers_produce_identical_bytes() {
        let metadata: Vec<(String, GgufValue)> = vec![
            (
                "general.architecture".into(),
                GgufValue::String("qwen35".into()),
            ),
            ("general.name".into(), GgufValue::String("fixture".into())),
            ("qwen35.block_count".into(), GgufValue::U32(3)),
            (
                "qwen35.layer_kinds".into(),
                GgufValue::Array(vec![
                    GgufValue::U32(0),
                    GgufValue::U32(1),
                    GgufValue::U32(0),
                ]),
            ),
        ];
        // Deliberately unaligned lengths, so the inter-tensor padding is
        // exercised rather than coincidentally zero.
        let payloads: Vec<Vec<u8>> = vec![
            (0..37u8).collect(),
            (0..64u8).map(|b| b ^ 0x5a).collect(),
            (0..5u8).collect(),
        ];
        let names = [
            "blk.0.attn_q.weight",
            "blk.1.ffn_down.weight",
            "output_norm.weight",
        ];
        let dims: Vec<Vec<u64>> = vec![vec![37], vec![8, 8], vec![5]];
        let types = [30u32, 40, 0];

        let mut w = GgufWriter::new();
        for (k, v) in &metadata {
            w.meta(k.clone(), v.clone());
        }
        for i in 0..3 {
            w.tensor(GgufTensor {
                name: names[i].into(),
                dims: dims[i].clone(),
                ggml_type: types[i],
                data: payloads[i].clone(),
            });
        }
        let buffered = w.to_bytes();

        let descriptors: Vec<GgufTensorDescriptor> = (0..3)
            .map(|i| GgufTensorDescriptor {
                name: names[i].into(),
                dims: dims[i].clone(),
                ggml_type: types[i],
                len: payloads[i].len() as u64,
            })
            .collect();
        let dir = std::env::temp_dir().join(format!("gguf-stream-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("streamed.gguf");
        write_streaming(&metadata, &descriptors, &path, |i, out| {
            out.write_all(&payloads[i])
        })
        .expect("stream");
        let streamed = std::fs::read(&path).unwrap();
        std::fs::remove_dir_all(&dir).ok();

        assert_eq!(
            streamed.len(),
            buffered.len(),
            "streamed {} bytes, buffered {}",
            streamed.len(),
            buffered.len()
        );
        assert_eq!(streamed, buffered, "the two writers disagree byte-for-byte");
    }

    /// A declared length the emitter does not honour corrupts every
    /// later offset, so it is caught where the mistake happens.
    #[test]
    fn an_emitter_that_writes_the_wrong_length_is_refused() {
        let descriptors = vec![GgufTensorDescriptor {
            name: "blk.0.weight".into(),
            dims: vec![4],
            ggml_type: 0,
            len: 16,
        }];
        let dir = std::env::temp_dir().join(format!("gguf-badlen-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.gguf");
        let err = write_streaming(&[], &descriptors, &path, |_, out| out.write_all(&[0u8; 8]))
            .expect_err("a short emitter must refuse");
        std::fs::remove_dir_all(&dir).ok();
        assert!(
            err.to_string().contains("declared 16 bytes") && err.to_string().contains("wrote 8"),
            "the refusal must name both numbers: {err}"
        );
    }
}
