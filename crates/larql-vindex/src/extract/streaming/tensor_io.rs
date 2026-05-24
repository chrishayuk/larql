//! Mmap-based tensor source for the streaming extraction pipeline.
//!
//! Named `tensor_io` rather than `safetensors` to avoid shadowing the
//! external `safetensors` crate inside the parent module.
//!
//! Two backing formats:
//!  - **safetensors**: HF-canonical layout, key already in HF form after
//!    `normalize_key` strips the architecture-specific prefix.
//!  - **GGUF**: llama.cpp blockwise quants (Q4_K, Q4_0, BF16, F32, …),
//!    keyed by GGUF tensor name (`blk.L.attn_q.weight`) and mapped back
//!    to HF form via [`larql_models::loading::gguf::normalize_gguf_key`].
//!
//! Types:
//!  - [`MmapShard`] keeps a safetensors file handle + mmap alive.
//!  - [`TensorSource`] dispatches `get_tensor_f32` over either backend.
//!  - [`GateSink`] is the gate-vector writer abstraction (real file or
//!    `/dev/null` when `--drop-gate-vectors` is set).
//!  - `normalize_key` strips a fixed prefix list from a safetensors key.

use std::collections::HashMap;
use std::io::{BufWriter, Write};

use ndarray::Array2;

use crate::error::VindexError;

/// Mmap'd safetensors file — kept alive for the duration of extraction.
pub(super) struct MmapShard {
    pub(super) _file: std::fs::File,
    pub(super) mmap: memmap2::Mmap,
}

/// Tensor source for the streaming pipeline. Dispatches `get_tensor_f32`
/// between mmap'd safetensors and mmap'd GGUF blockwise-quantised tensors.
///
/// The MXFP4 raw-pair path (packed expert gate_up/down in DeepSeek-V4)
/// is safetensors-only — GGUF stores the same expert tensors as standard
/// blockwise quants (Q4_0/Q4_K/…) handled by `get_tensor_f32` directly.
pub(super) enum TensorSource {
    Safetensors {
        shards: Vec<MmapShard>,
        /// hf_key → (shard_idx, safetensors-internal name)
        index: HashMap<String, (usize, String)>,
    },
    Gguf(GgufTensorSource),
}

/// GGUF backend for [`TensorSource`]. Owns the parsed `GgufFile` (multi-
/// shard aware) plus one mmap per shard. Eager mmap is cheap (virtual
/// address space, not RAM) so the OS only pages in tensors we touch.
///
/// Holds the architecture config so `get_tensor_f32` can apply the same
/// canonical orientation the in-memory loader runs via `orient_in_place`
/// (GGUF stores ffn tensors in PyTorch shape which may be the transpose
/// of the HF Linear convention — without correction, `tensor.shape()[0]`
/// is `hidden_size` instead of `intermediate_size` and downstream
/// matmul produces NaN).
pub(super) struct GgufTensorSource {
    pub(super) gguf: larql_models::loading::gguf::GgufFile,
    pub(super) shard_mmaps: Vec<memmap2::Mmap>,
    /// hf_key (after `normalize_gguf_key`) → index into `gguf.tensor_infos`.
    pub(super) index: HashMap<String, usize>,
    /// Canonical hidden_size — used to orient ffn tensors to HF Linear shape.
    hidden_size: usize,
    /// Canonical intermediate_size (dense FFN). Some MoE architectures
    /// have a separate `moe_intermediate_size` — that's handled at the
    /// call site since the streaming pipeline matches the in-memory
    /// loader's behavior of skipping 3D-packed expert tensors anyway.
    intermediate_size: usize,
}

impl TensorSource {
    /// Read a 2D tensor by HF-canonical key, dequantising to f32. Returns
    /// `Ok(None)` if the key is absent, the tensor is not 2D, or the
    /// dtype is unsupported by this code path (matching the safetensors
    /// historical contract).
    pub(super) fn get_tensor_f32(&self, key: &str) -> Result<Option<Array2<f32>>, VindexError> {
        match self {
            TensorSource::Safetensors { shards, index } => get_tensor_f32(shards, index, key),
            TensorSource::Gguf(g) => g.get_tensor_f32(key),
        }
    }

    /// Borrow the safetensors backing data for raw-pair access (MXFP4
    /// expert gate_up/down). Returns `None` for the GGUF variant — GGUF
    /// has no MXFP4-packed format.
    pub(super) fn safetensors_view(
        &self,
    ) -> Option<(&[MmapShard], &HashMap<String, (usize, String)>)> {
        match self {
            TensorSource::Safetensors { shards, index } => Some((shards, index)),
            TensorSource::Gguf(_) => None,
        }
    }

    /// Borrow the safetensors shards as `&[u8]` slices, for callers that
    /// need to hand the raw mmap to a separate writer subsystem
    /// (`StreamingWeights`). Returns `None` for the GGUF variant.
    pub(super) fn safetensors_mmap_refs(&self) -> Option<Vec<&[u8]>> {
        match self {
            TensorSource::Safetensors { shards, .. } => {
                Some(shards.iter().map(|s| s.mmap.as_ref()).collect())
            }
            TensorSource::Gguf(_) => None,
        }
    }

    /// Borrow the safetensors tensor index. Returns `None` for the GGUF
    /// variant.
    pub(super) fn safetensors_index(&self) -> Option<&HashMap<String, (usize, String)>> {
        match self {
            TensorSource::Safetensors { index, .. } => Some(index),
            TensorSource::Gguf(_) => None,
        }
    }

}

impl GgufTensorSource {
    /// Build a `GgufTensorSource` from an already-opened `GgufFile`. The
    /// caller is responsible for parsing the header (`GgufFile::open`);
    /// this just mmaps each shard and builds the hf_key → tensor_info
    /// index.
    ///
    /// `hidden_size` and `intermediate_size` come from the detected
    /// architecture and drive the canonical FFN orientation applied
    /// inside `get_tensor_f32` (mirrors `orient_in_place` in the
    /// in-memory loader).
    pub(super) fn from_gguf(
        gguf: larql_models::loading::gguf::GgufFile,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Result<Self, VindexError> {
        let mut shard_mmaps: Vec<memmap2::Mmap> = Vec::with_capacity(gguf.shards.len());
        for shard in &gguf.shards {
            let file = std::fs::File::open(&shard.path)
                .map_err(|e| VindexError::Parse(format!("open {}: {e}", shard.path.display())))?;
            // SAFETY: shard files are owned, immutable input.
            let mmap = unsafe { memmap2::Mmap::map(&file) }
                .map_err(|e| VindexError::Parse(format!("mmap {}: {e}", shard.path.display())))?;
            shard_mmaps.push(mmap);
        }

        let mut index: HashMap<String, usize> = HashMap::with_capacity(gguf.tensor_infos.len());
        for (i, info) in gguf.tensor_infos.iter().enumerate() {
            let hf_key = larql_models::loading::gguf::normalize_gguf_key(info.name());
            // First-write wins: if the same hf_key normalises twice (it
            // shouldn't, but defensively), keep the earlier one rather
            // than silently flipping shard_idx on the caller.
            index.entry(hf_key).or_insert(i);
        }

        Ok(Self {
            gguf,
            shard_mmaps,
            index,
            hidden_size,
            intermediate_size,
        })
    }

    /// Inspect a normalised HF key and return the canonical `(rows, cols)`
    /// shape expected by downstream stages — or `None` for keys that are
    /// already shape-correct as stored in GGUF (embed, lm_head, norms).
    ///
    /// The pattern matching mirrors `load_gguf`'s `orient_in_place` calls
    /// for ffn_gate / ffn_up / ffn_down (+ MoE per-expert variants the
    /// streaming path would reach if GGUF stored experts 2D — currently
    /// they're 3D-packed and skipped here as in the in-memory loader).
    fn expected_shape(&self, key: &str) -> Option<(usize, usize)> {
        let h = self.hidden_size;
        let m = self.intermediate_size;
        if h == 0 || m == 0 {
            return None;
        }
        // Order matters: check the more specific `down_proj` before
        // generic `proj` to avoid the gate/up branches catching down.
        if key.ends_with("down_proj.weight") || key.contains(".down_proj.") {
            return Some((h, m));
        }
        if key.ends_with("gate_proj.weight")
            || key.contains(".gate_proj.")
            || key.ends_with("up_proj.weight")
            || key.contains(".up_proj.")
        {
            return Some((m, h));
        }
        None
    }

    /// Transpose if the current shape is `(cols, rows)` instead of the
    /// expected `(rows, cols)`. Mirror of `load_gguf::orient_in_place`.
    fn orient(&self, arr: Array2<f32>, expected: (usize, usize)) -> Array2<f32> {
        let (er, ec) = expected;
        if er == ec {
            return arr;
        }
        let s = arr.shape();
        if s[0] == er && s[1] == ec {
            return arr;
        }
        if s[0] == ec && s[1] == er {
            let mut out = Array2::<f32>::zeros((er, ec));
            out.assign(&arr.t());
            return out;
        }
        // Shape doesn't match either orientation — return as-is and let
        // the caller's shape check surface the mismatch.
        arr
    }

    fn get_tensor_f32(&self, key: &str) -> Result<Option<Array2<f32>>, VindexError> {
        let info_idx = match self.index.get(key) {
            Some(&i) => i,
            None => return Ok(None),
        };
        let info = &self.gguf.tensor_infos[info_idx];
        if info.n_dims() != 2 {
            return Ok(None);
        }

        let shard_idx = info.shard_idx();
        let mmap = &self.shard_mmaps[shard_idx];
        let data_offset = self.gguf.shards[shard_idx].data_offset;
        let abs_offset = data_offset
            .checked_add(info.offset())
            .ok_or_else(|| VindexError::Parse(format!(
                "gguf tensor {}: data_offset {data_offset} + offset {} overflows",
                info.name(),
                info.offset(),
            )))?;

        let dims = info.dims();
        let n_elements: u64 = dims.iter().product();
        let n_elements_usize = usize::try_from(n_elements).map_err(|_| {
            VindexError::Parse(format!(
                "gguf tensor {}: n_elements {n_elements} exceeds usize",
                info.name(),
            ))
        })?;

        let data_size =
            larql_models::quant::ggml::tensor_data_size(info.tensor_type(), n_elements_usize)
                .map_err(|e| VindexError::Parse(e.to_string()))?;
        let abs_offset_usize = usize::try_from(abs_offset).map_err(|_| {
            VindexError::Parse(format!(
                "gguf tensor {}: abs_offset {abs_offset} exceeds usize",
                info.name(),
            ))
        })?;
        let end = abs_offset_usize
            .checked_add(data_size)
            .ok_or_else(|| VindexError::Parse(format!(
                "gguf tensor {}: offset+size overflows",
                info.name(),
            )))?;
        if end > mmap.len() {
            return Err(VindexError::Parse(format!(
                "gguf tensor {} out of bounds: end {end} > shard len {}",
                info.name(),
                mmap.len(),
            )));
        }

        let raw = &mmap[abs_offset_usize..end];
        let floats = larql_models::quant::ggml::dequantize(raw, info.tensor_type(), n_elements_usize)
            .map_err(|e| VindexError::Parse(e.to_string()))?;

        // GGUF dim ordering: dims[0] = columns (innermost/fastest),
        // dims[1] = rows (outermost). Raw bytes are contiguous along
        // dims[0], so reshaping to (rows, cols) = (dims[1], dims[0])
        // gives ndarray's row-major layout matching the HF convention.
        let ne0 = dims[0] as usize;
        let ne1 = dims[1] as usize;
        let arr = Array2::from_shape_vec((ne1, ne0), floats)
            .map_err(|e| VindexError::Parse(format!("gguf tensor {}: {e}", info.name())))?;

        // Apply canonical orientation when the key is a known FFN
        // tensor — matches `orient_in_place` in the in-memory loader.
        let arr = match self.expected_shape(key) {
            Some(expected) => self.orient(arr, expected),
            None => arr,
        };
        Ok(Some(arr))
    }
}

/// Sink for gate-vector bytes. With `--drop-gate-vectors` the writer
/// still walks every layer (so `layer_infos` is populated for
/// `index.json`) but redirects bytes to `/dev/null` — they're
/// recoverable from `interleaved_kquant.bin` at load time.
pub(super) enum GateSink {
    File(BufWriter<std::fs::File>),
    Discard(std::io::Sink),
}

impl Write for GateSink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            GateSink::File(f) => f.write(buf),
            GateSink::Discard(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            GateSink::File(f) => f.flush(),
            GateSink::Discard(s) => s.flush(),
        }
    }
}

/// Get a 2D tensor from mmap'd safetensors, dequantizing to f32.
pub(super) fn get_tensor_f32(
    shards: &[MmapShard],
    index: &HashMap<String, (usize, String)>,
    key: &str,
) -> Result<Option<Array2<f32>>, VindexError> {
    let (shard_idx, tensor_name) = match index.get(key) {
        Some(v) => v,
        None => return Ok(None),
    };

    let st = safetensors::SafeTensors::deserialize(&shards[*shard_idx].mmap)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    let view = st
        .tensor(tensor_name)
        .map_err(|e| VindexError::Parse(e.to_string()))?;

    let shape = view.shape();
    if shape.len() != 2 {
        return Ok(None);
    }

    // MXFP4 (DeepSeek-V4 expert weights): the streaming extract has its own
    // f32 decoder separate from `larql-models::loading::safetensors`, so the
    // I8+F8_E8M0 pairing has to be detected here too. Without it,
    // `gate_vectors.bin` came out 0 bytes for V4 models because every layer's
    // gate fell into the catch-all `_ => return Ok(None)` below.
    //
    // Detection contract: `.weight` tensor with `I8` dtype and a `.scale`
    // companion of dtype `F8_E8M0` whose row count matches and whose column
    // count divides the unpacked-cols evenly into a sane group size
    // {16, 32, 64, 128}. Anything else falls through to the dtype match.
    if view.dtype() == safetensors::Dtype::I8 && tensor_name.ends_with(".weight") {
        let scale_name = tensor_name.replacen(".weight", ".scale", 1);
        if let Ok(scale_view) = st.tensor(&scale_name) {
            if scale_view.dtype() == safetensors::Dtype::F8_E8M0 {
                let s_shape = scale_view.shape();
                if s_shape.len() == 2 && s_shape[0] == shape[0] {
                    let cols_unpacked = shape[1] * 2;
                    if s_shape[1] > 0 && cols_unpacked % s_shape[1] == 0 {
                        let group_size = cols_unpacked / s_shape[1];
                        if [16usize, 32, 64, 128].contains(&group_size) {
                            let unpacked = larql_models::quant::mxfp4::dequantize_expert(
                                view.data(),
                                scale_view.data(),
                                shape[0],
                                s_shape[1],
                            )
                            .map_err(|e| VindexError::Parse(e.to_string()))?;
                            let arr = Array2::from_shape_vec((shape[0], cols_unpacked), unpacked)
                                .map_err(|e| VindexError::Parse(e.to_string()))?;
                            return Ok(Some(arr));
                        }
                    }
                }
            }
        }
    }

    let data = match view.dtype() {
        safetensors::Dtype::F32 => view
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        safetensors::Dtype::F16 => crate::format::quant::half::decode_f16(view.data()),
        safetensors::Dtype::BF16 => crate::format::quant::half::decode_bf16(view.data()),
        _ => return Ok(None), // skip non-float (and non-MXFP4) tensors
    };

    let arr = Array2::from_shape_vec((shape[0], shape[1]), data)
        .map_err(|e| VindexError::Parse(e.to_string()))?;
    Ok(Some(arr))
}

pub(super) fn normalize_key(key: &str, prefixes: &[&str]) -> String {
    for prefix in prefixes {
        if let Some(stripped) = key.strip_prefix(prefix) {
            return stripped.to_string();
        }
    }
    key.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::Write;

    // ── Synthetic safetensors fixture ─────────────────────────────────────
    //
    // Tests below want to drive `get_tensor_f32` against a real mmap'd
    // safetensors file rather than mock the SafeTensors trait. The helper
    // here builds a tempfile from a list of `(name, shape, dtype, bytes)`
    // entries via `safetensors::tensor::serialize`, mmaps it, and yields
    // a single-shard `(MmapShard, tensor_index)` pair the tests can hand
    // to `get_tensor_f32` directly.

    /// One tensor's raw payload, in the layout the safetensors crate
    /// expects (little-endian floats / packed nibble bytes / etc.).
    struct FixtureTensor {
        name: String,
        shape: Vec<usize>,
        dtype: safetensors::Dtype,
        bytes: Vec<u8>,
    }

    /// Write `tensors` to a fresh `.safetensors` file in `dir`, mmap it,
    /// and return the `MmapShard` plus a `tensor_index` keyed by the
    /// tensor names (single-shard fixtures use the same name for the
    /// "logical key" and the safetensors-internal name).
    fn write_fixture(
        dir: &std::path::Path,
        tensors: Vec<FixtureTensor>,
    ) -> (Vec<MmapShard>, HashMap<String, (usize, String)>) {
        let path = dir.join("fixture.safetensors");
        let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
            .iter()
            .map(|t| {
                (
                    t.name.clone(),
                    safetensors::tensor::TensorView::new(t.dtype, t.shape.clone(), &t.bytes)
                        .expect("invalid synthetic tensor view"),
                )
            })
            .collect();
        let bytes = safetensors::tensor::serialize(views, None)
            .expect("synthetic fixture serialisation failed");
        let mut f = std::fs::File::create(&path).expect("create fixture");
        f.write_all(&bytes).expect("write fixture");
        f.sync_all().ok();

        let file = std::fs::File::open(&path).expect("reopen fixture");
        let mmap = unsafe { memmap2::Mmap::map(&file).expect("mmap fixture") };
        let mut index = HashMap::new();
        for t in &tensors {
            index.insert(t.name.clone(), (0_usize, t.name.clone()));
        }
        (vec![MmapShard { _file: file, mmap }], index)
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn f16_bytes(values: &[f32]) -> Vec<u8> {
        crate::format::quant::half::encode_f16(values)
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        crate::format::quant::half::encode_bf16(values)
    }

    // ── get_tensor_f32 ────────────────────────────────────────────────────

    #[test]
    fn get_tensor_f32_returns_none_when_key_missing() {
        let dir = tempfile::tempdir().unwrap();
        let (shards, index) = write_fixture(
            dir.path(),
            vec![FixtureTensor {
                name: "present.weight".to_string(),
                shape: vec![2, 2],
                dtype: safetensors::Dtype::F32,
                bytes: f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
            }],
        );
        let out = get_tensor_f32(&shards, &index, "missing.key").unwrap();
        assert!(out.is_none(), "missing logical key must return Ok(None)");
    }

    #[test]
    fn get_tensor_f32_decodes_f32_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let values = [0.5f32, -1.25, 3.0, -4.75];
        let (shards, index) = write_fixture(
            dir.path(),
            vec![FixtureTensor {
                name: "w.f32".to_string(),
                shape: vec![2, 2],
                dtype: safetensors::Dtype::F32,
                bytes: f32_bytes(&values),
            }],
        );
        let arr = get_tensor_f32(&shards, &index, "w.f32").unwrap().unwrap();
        assert_eq!(arr.shape(), &[2, 2]);
        assert_eq!(arr.as_slice().unwrap(), values);
    }

    #[test]
    fn get_tensor_f32_decodes_f16_tensor() {
        let dir = tempfile::tempdir().unwrap();
        // f16 representable values — exact round-trip through half::f16.
        let values = [0.5f32, -1.0, 2.0, 0.0];
        let (shards, index) = write_fixture(
            dir.path(),
            vec![FixtureTensor {
                name: "w.f16".to_string(),
                shape: vec![2, 2],
                dtype: safetensors::Dtype::F16,
                bytes: f16_bytes(&values),
            }],
        );
        let arr = get_tensor_f32(&shards, &index, "w.f16").unwrap().unwrap();
        assert_eq!(arr.shape(), &[2, 2]);
        for (got, want) in arr.as_slice().unwrap().iter().zip(values.iter()) {
            assert!((got - want).abs() < 1e-3, "{got} != {want} (f16 rt)");
        }
    }

    #[test]
    fn get_tensor_f32_decodes_bf16_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let values = [1.0f32, -2.0, 4.0, 0.5];
        let (shards, index) = write_fixture(
            dir.path(),
            vec![FixtureTensor {
                name: "w.bf16".to_string(),
                shape: vec![2, 2],
                dtype: safetensors::Dtype::BF16,
                bytes: bf16_bytes(&values),
            }],
        );
        let arr = get_tensor_f32(&shards, &index, "w.bf16").unwrap().unwrap();
        assert_eq!(arr.shape(), &[2, 2]);
        for (got, want) in arr.as_slice().unwrap().iter().zip(values.iter()) {
            assert!(
                (got - want).abs() < 0.05,
                "{got} != {want} (bf16 rt has 7-bit mantissa)"
            );
        }
    }

    #[test]
    fn get_tensor_f32_returns_none_for_non_2d_tensor() {
        // 1D tensors aren't supported by the streaming gate-vector path —
        // it expects [num_features, hidden] matrices.
        let dir = tempfile::tempdir().unwrap();
        let (shards, index) = write_fixture(
            dir.path(),
            vec![FixtureTensor {
                name: "v".to_string(),
                shape: vec![4],
                dtype: safetensors::Dtype::F32,
                bytes: f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
            }],
        );
        let out = get_tensor_f32(&shards, &index, "v").unwrap();
        assert!(out.is_none(), "1D tensor must return Ok(None)");
    }

    #[test]
    fn get_tensor_f32_returns_none_for_non_float_dtype() {
        // I64 (or any other dtype the match arm doesn't list, with no
        // I8+F8_E8M0 companion to trigger the MXFP4 path) falls into the
        // catch-all `_ => return Ok(None)`.
        let dir = tempfile::tempdir().unwrap();
        let bytes: Vec<u8> = (0..32).collect(); // 4 i64 values
        let (shards, index) = write_fixture(
            dir.path(),
            vec![FixtureTensor {
                name: "ids".to_string(),
                shape: vec![2, 2],
                dtype: safetensors::Dtype::I64,
                bytes,
            }],
        );
        let out = get_tensor_f32(&shards, &index, "ids").unwrap();
        assert!(out.is_none(), "I64 dtype must return Ok(None)");
    }

    #[test]
    fn get_tensor_f32_i8_without_scale_companion_returns_none() {
        // I8 .weight without an I8+F8_E8M0 companion must fall through to
        // the dtype match — and I8 isn't in {F32,F16,BF16}, so Ok(None).
        let dir = tempfile::tempdir().unwrap();
        let (shards, index) = write_fixture(
            dir.path(),
            vec![FixtureTensor {
                name: "experts.0.w1.weight".to_string(),
                shape: vec![2, 16],
                dtype: safetensors::Dtype::I8,
                bytes: vec![0u8; 32],
            }],
        );
        let out = get_tensor_f32(&shards, &index, "experts.0.w1.weight").unwrap();
        assert!(
            out.is_none(),
            "I8 weight without F8_E8M0 scale must return Ok(None)"
        );
    }

    #[test]
    fn get_tensor_f32_i8_with_wrong_scale_dtype_falls_through() {
        // The companion `.scale` exists but is BF16 instead of F8_E8M0 —
        // the MXFP4 detector requires F8_E8M0 specifically. The detector
        // should bail and the catch-all `_ => Ok(None)` fires.
        let dir = tempfile::tempdir().unwrap();
        let (shards, index) = write_fixture(
            dir.path(),
            vec![
                FixtureTensor {
                    name: "experts.0.w1.weight".to_string(),
                    shape: vec![2, 16],
                    dtype: safetensors::Dtype::I8,
                    bytes: vec![0u8; 32],
                },
                FixtureTensor {
                    name: "experts.0.w1.scale".to_string(),
                    shape: vec![2, 1],
                    dtype: safetensors::Dtype::BF16,
                    bytes: bf16_bytes(&[1.0, 1.0]),
                },
            ],
        );
        let out = get_tensor_f32(&shards, &index, "experts.0.w1.weight").unwrap();
        assert!(out.is_none(), "BF16 scale companion must not trigger MXFP4");
    }

    #[test]
    fn get_tensor_f32_i8_with_bad_scale_shape_falls_through() {
        // Scale row count must match weight row count. Mismatch → bail.
        let dir = tempfile::tempdir().unwrap();
        let (shards, index) = write_fixture(
            dir.path(),
            vec![
                FixtureTensor {
                    name: "experts.0.w1.weight".to_string(),
                    shape: vec![2, 16],
                    dtype: safetensors::Dtype::I8,
                    bytes: vec![0u8; 32],
                },
                FixtureTensor {
                    name: "experts.0.w1.scale".to_string(),
                    shape: vec![3, 1], // rows=3, weight has rows=2 → mismatch
                    dtype: safetensors::Dtype::F8_E8M0,
                    bytes: vec![127u8; 3], // 2^0 = 1.0
                },
            ],
        );
        let out = get_tensor_f32(&shards, &index, "experts.0.w1.weight").unwrap();
        assert!(out.is_none(), "row-mismatch scale must abort MXFP4");
    }

    #[test]
    fn get_tensor_f32_i8_with_unsupported_group_size_falls_through() {
        // group_size = cols_unpacked / s_shape[1]. Allowed: {16, 32, 64,
        // 128}. weight cols=16 → cols_unpacked=32. With s_shape[1]=4 →
        // group_size=8 → not supported → bail.
        let dir = tempfile::tempdir().unwrap();
        let (shards, index) = write_fixture(
            dir.path(),
            vec![
                FixtureTensor {
                    name: "experts.0.w1.weight".to_string(),
                    shape: vec![2, 16],
                    dtype: safetensors::Dtype::I8,
                    bytes: vec![0u8; 32],
                },
                FixtureTensor {
                    name: "experts.0.w1.scale".to_string(),
                    shape: vec![2, 4], // groups=4 → group_size=8, not in {16,32,64,128}
                    dtype: safetensors::Dtype::F8_E8M0,
                    bytes: vec![127u8; 8],
                },
            ],
        );
        let out = get_tensor_f32(&shards, &index, "experts.0.w1.weight").unwrap();
        assert!(out.is_none(), "group_size=8 must abort MXFP4");
    }

    #[test]
    fn get_tensor_f32_i8_mxfp4_dequantizes_when_companion_matches() {
        // The full MXFP4 happy path. MXFP4 packs 2 FP4 nibbles per byte,
        // 32-element blocks → each (row, block) consumes 16 bytes of
        // weight + 1 byte of F8_E8M0 scale.
        //
        // Layout: weight [rows=2, packed_cols=32] (=2 blocks of 16 bytes
        // each per row), scale [rows=2, groups=2]. cols_unpacked = 64,
        // group_size = 64 / 2 = 32 ✓ (in the allowed set).
        //
        // All nibbles zero + scale=1.0 (F8_E8M0 byte=127) → all output
        // zero (FP4 lookup of 0 is 0.0).
        let dir = tempfile::tempdir().unwrap();
        let weight_bytes = vec![0u8; 2 * 32]; // 64 bytes
        let scale_bytes = vec![127u8; 2 * 2]; // 4 bytes, 2^0 = 1.0
        let (shards, index) = write_fixture(
            dir.path(),
            vec![
                FixtureTensor {
                    name: "experts.0.w1.weight".to_string(),
                    shape: vec![2, 32],
                    dtype: safetensors::Dtype::I8,
                    bytes: weight_bytes,
                },
                FixtureTensor {
                    name: "experts.0.w1.scale".to_string(),
                    shape: vec![2, 2],
                    dtype: safetensors::Dtype::F8_E8M0,
                    bytes: scale_bytes,
                },
            ],
        );
        let arr = get_tensor_f32(&shards, &index, "experts.0.w1.weight")
            .unwrap()
            .unwrap();
        // Output shape: [rows=2, cols_unpacked = 32 * 2 = 64].
        assert_eq!(arr.shape(), &[2, 64]);
        for v in arr.iter() {
            assert_eq!(*v, 0.0);
        }
    }

    // ── normalize_key (existing tests; left untouched) ───────────────────

    #[test]
    fn normalize_strips_first_matching_prefix() {
        let prefixes = ["model.", "transformer."];
        assert_eq!(
            normalize_key("model.layers.0.mlp.gate_proj.weight", &prefixes),
            "layers.0.mlp.gate_proj.weight",
        );
    }

    #[test]
    fn normalize_keeps_key_when_no_prefix_matches() {
        let prefixes = ["model.", "transformer."];
        assert_eq!(
            normalize_key("layers.0.mlp.gate_proj.weight", &prefixes),
            "layers.0.mlp.gate_proj.weight",
        );
    }

    #[test]
    fn normalize_uses_first_match_only() {
        // First matching prefix wins; the second isn't applied to the
        // residue. Pinning this matters because the safetensors loader
        // walks tensors with a fixed prefix list — re-stripping would
        // mangle keys like "model.model.embed_tokens.weight".
        let prefixes = ["model.", "model.model."];
        assert_eq!(
            normalize_key("model.model.embed_tokens.weight", &prefixes),
            "model.embed_tokens.weight",
        );
    }

    #[test]
    fn normalize_with_empty_prefix_list_is_identity() {
        assert_eq!(normalize_key("anything", &[]), "anything");
    }

    #[test]
    fn normalize_handles_empty_input() {
        let prefixes = ["model."];
        assert_eq!(normalize_key("", &prefixes), "");
    }
}
