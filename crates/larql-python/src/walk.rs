//! WalkModel — mmap-backed model weights + vindex for walk FFN inference.
//!
//! Weight files are mmap'd, not loaded to heap. Array2 views point directly
//! at mmap'd memory. Only the pages touched during inference are paged in.
//! Peak RSS: ~one layer of weights at a time (OS manages page eviction).

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::collections::HashMap;
use std::path::Path;

use larql_inference::ffn::FfnBackend;
use larql_inference::forward::{
    capture_donor_state_with_ffn, embedding_neighbors as li_embedding_neighbors,
    embedding_row as li_embedding_row, embedding_row_scaled as li_embedding_row_scaled,
    logit_lens_topk, patch_and_trace_with_ffn,
    project_through_unembed as li_project_through_unembed, trace_forward_full_hooked,
    track_race as li_track_race, track_token as li_track_token,
    unembedding_row as li_unembedding_row, RecordHook, SteerHook, ZeroAblateHook,
};
use larql_inference::{predict_with_ffn, ModelWeights, WalkFfn};
use larql_kv::generation::generate_cached_hooked;
use larql_vindex::format::filenames::{
    ATTN_WEIGHTS_BIN, DOWN_WEIGHTS_BIN, EMBEDDINGS_BIN, GATE_VECTORS_BIN, LM_HEAD_BIN,
    MODEL_WEIGHTS_BIN, NORMS_BIN, UP_WEIGHTS_BIN, WEIGHT_MANIFEST_JSON,
};
use larql_vindex::{
    load_vindex_config, load_vindex_tokenizer, tokenizers, SilentLoadCallbacks, VectorIndex,
};

use crate::trace_py;

/// Mmap'd weight file — kept alive so Array2 views remain valid.
struct WeightMmap {
    _file: std::fs::File,
    mmap: memmap2::Mmap,
}

/// Create ModelWeights backed by mmap'd files.
/// Tensors are Array2<f32> created from mmap'd memory.
/// For f32 vindexes: zero-copy (pointer into mmap).
/// For f16 vindexes: decoded to f32 (one allocation per tensor).
fn load_mmap_weights(dir: &Path) -> Result<(ModelWeights, Vec<WeightMmap>), String> {
    let config = load_vindex_config(dir).map_err(|e| e.to_string())?;

    if !config.has_model_weights {
        return Err("No model weights. Extract with --level all".into());
    }

    let model_cfg = config
        .model_config
        .as_ref()
        .ok_or("Missing model_config in index.json")?;

    let arch_json = serde_json::json!({
        "model_type": model_cfg.model_type,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_layers,
        "intermediate_size": config.intermediate_size,
        "head_dim": model_cfg.head_dim,
        "num_attention_heads": model_cfg.num_q_heads,
        "num_key_value_heads": model_cfg.num_kv_heads,
        "rope_theta": model_cfg.rope_base,
        "sliding_window": model_cfg.sliding_window,
        "vocab_size": config.vocab_size,
    });
    let arch = larql_models::detect_from_json(&arch_json);

    // Mmap weight files
    let mut mmaps: Vec<WeightMmap> = Vec::new();
    let mut mmap_index: HashMap<String, usize> = HashMap::new();

    let weight_files = [
        ATTN_WEIGHTS_BIN,
        UP_WEIGHTS_BIN,
        DOWN_WEIGHTS_BIN,
        NORMS_BIN,
        LM_HEAD_BIN,
    ];
    for fname in &weight_files {
        let path = dir.join(fname);
        if path.exists() {
            let file = std::fs::File::open(&path).map_err(|e| e.to_string())?;
            let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| e.to_string())?;
            mmap_index.insert(fname.to_string(), mmaps.len());
            mmaps.push(WeightMmap { _file: file, mmap });
        }
    }

    // Mmap embeddings
    let embed_file = std::fs::File::open(dir.join(EMBEDDINGS_BIN)).map_err(|e| e.to_string())?;
    let embed_mmap = unsafe { memmap2::Mmap::map(&embed_file) }.map_err(|e| e.to_string())?;
    let embed_idx = mmaps.len();
    mmaps.push(WeightMmap {
        _file: embed_file,
        mmap: embed_mmap,
    });

    // Mmap gate_vectors
    let gate_file = std::fs::File::open(dir.join(GATE_VECTORS_BIN)).map_err(|e| e.to_string())?;
    let gate_mmap = unsafe { memmap2::Mmap::map(&gate_file) }.map_err(|e| e.to_string())?;
    let gate_idx = mmaps.len();
    mmaps.push(WeightMmap {
        _file: gate_file,
        mmap: gate_mmap,
    });

    // Read manifest
    let manifest_text =
        std::fs::read_to_string(dir.join(WEIGHT_MANIFEST_JSON)).map_err(|e| e.to_string())?;

    #[derive(serde::Deserialize)]
    struct Entry {
        key: String,
        kind: String,
        shape: Vec<usize>,
        offset: u64,
        length: u64,
        #[serde(default)]
        file: String,
    }
    let entries: Vec<Entry> = serde_json::from_str(&manifest_text).map_err(|e| e.to_string())?;

    let is_f32 = config.dtype == larql_vindex::StorageDtype::F32;

    // Build tensors from mmap'd memory
    let mut tensors: HashMap<String, larql_models::WeightArray> = HashMap::new();
    let mut vectors: HashMap<String, Vec<f32>> = HashMap::new();
    let mut lm_head_arr: Option<larql_models::WeightArray> = None;

    for entry in &entries {
        let fname = if entry.file.is_empty() {
            MODEL_WEIGHTS_BIN
        } else {
            &entry.file
        };
        let mmap_idx = match mmap_index.get(fname) {
            Some(idx) => *idx,
            None => continue,
        };
        let mmap_data = &mmaps[mmap_idx].mmap;

        let offset = entry.offset as usize;
        let length = entry.length as usize;
        if offset + length > mmap_data.len() {
            continue;
        }

        let raw = &mmap_data[offset..offset + length];

        match entry.kind.as_str() {
            "tensor" if entry.shape.len() == 2 => {
                let (rows, cols) = (entry.shape[0], entry.shape[1]);
                let arr = if is_f32 {
                    // Zero-copy: create Array2 from a Vec that points at mmap memory.
                    //
                    // SAFETY:
                    // - The mmap is held alive in `_mmaps` for the lifetime of PyWalkModel
                    // - The pointer is aligned (f32, mmap page-aligned)
                    // - len and capacity are correct (checked by offset+length bounds above)
                    // - We use a leaked Vec (mem::forget prevents dealloc) so ndarray
                    //   won't free the mmap'd memory when the Array2 is dropped
                    // - The PyWalkModel drops _mmaps AFTER weights, so the memory is valid
                    //   for the entire lifetime of the arrays
                    let count = rows * cols;
                    let ptr = raw.as_ptr() as *mut f32;
                    let vec = unsafe { Vec::from_raw_parts(ptr, count, count) };
                    let arr = Array2::from_shape_vec((rows, cols), vec)
                        .map_err(|e| e.to_string())?
                        .into_shared();
                    // Leak an extra Arc ref to prevent the Vec from being freed
                    // when the ArcArray2 drops — the mmap owns this memory
                    std::mem::forget(arr.clone());
                    arr
                } else {
                    let floats = larql_vindex::config::dtype::decode_floats(raw, config.dtype);
                    Array2::from_shape_vec((rows, cols), floats)
                        .map_err(|e| e.to_string())?
                        .into_shared()
                };

                if entry.key == "lm_head.weight" {
                    lm_head_arr = Some(arr);
                } else {
                    tensors.insert(entry.key.clone(), arr);
                }
            }
            "vector" => {
                // Vectors are small (norms) — copy is fine
                let floats = if is_f32 {
                    unsafe {
                        std::slice::from_raw_parts(raw.as_ptr() as *const f32, entry.shape[0])
                    }
                    .to_vec()
                } else {
                    larql_vindex::config::dtype::decode_floats(raw, config.dtype)
                };
                vectors.insert(entry.key.clone(), floats);
            }
            _ => {}
        }
    }

    // Embeddings from mmap — zero-copy for f32
    let embed_data = &mmaps[embed_idx].mmap;
    let embed = if is_f32 {
        let count = config.vocab_size * config.hidden_size;
        let ptr = embed_data.as_ptr() as *mut f32;
        let vec = unsafe { Vec::from_raw_parts(ptr, count, count) };
        let arr = Array2::from_shape_vec((config.vocab_size, config.hidden_size), vec)
            .map_err(|e| e.to_string())?
            .into_shared();
        std::mem::forget(arr.clone());
        arr
    } else {
        let floats = larql_vindex::config::dtype::decode_floats(embed_data, config.dtype);
        Array2::from_shape_vec((config.vocab_size, config.hidden_size), floats)
            .map_err(|e| e.to_string())?
            .into_shared()
    };

    // Gate vectors from mmap — zero-copy for f32
    let gate_data = &mmaps[gate_idx].mmap;
    let bpf = larql_vindex::config::dtype::bytes_per_float(config.dtype);
    for info in &config.layers {
        let float_offset = info.offset as usize / bpf;
        let float_count = info.num_features * config.hidden_size;

        let gate_arr = if is_f32 {
            let ptr = unsafe { (gate_data.as_ptr() as *const f32).add(float_offset) as *mut f32 };
            if float_offset + float_count > gate_data.len() / 4 {
                continue;
            }
            let vec = unsafe { Vec::from_raw_parts(ptr, float_count, float_count) };
            let arr = Array2::from_shape_vec((info.num_features, config.hidden_size), vec)
                .map_err(|e| e.to_string())?
                .into_shared();
            std::mem::forget(arr.clone());
            arr
        } else {
            let byte_offset = info.offset as usize;
            let byte_length = info.length as usize;
            if byte_offset + byte_length > gate_data.len() {
                continue;
            }
            let floats = larql_vindex::config::dtype::decode_floats(
                &gate_data[byte_offset..byte_offset + byte_length],
                config.dtype,
            );
            Array2::from_shape_vec((info.num_features, config.hidden_size), floats)
                .map_err(|e| e.to_string())?
                .into_shared()
        };
        tensors.insert(arch.ffn_gate_key(info.layer), gate_arr);
    }

    let lm_head = lm_head_arr.unwrap_or_else(|| embed.clone());

    let weights = ModelWeights {
        tensors,
        vectors,
        raw_bytes: std::collections::HashMap::new(),
        skipped_tensors: Vec::new(),
        packed_mmaps: std::collections::HashMap::new(),
        packed_byte_ranges: std::collections::HashMap::new(),
        embed,
        lm_head,
        position_embed: None,
        num_layers: config.num_layers,
        hidden_size: config.hidden_size,
        intermediate_size: config.intermediate_size,
        vocab_size: config.vocab_size,
        head_dim: model_cfg.head_dim,
        num_q_heads: model_cfg.num_q_heads,
        num_kv_heads: model_cfg.num_kv_heads,
        rope_base: model_cfg.rope_base,
        arch,
    };

    Ok((weights, mmaps))
}

// ── InferState: lazy-loaded mmap'd weights for vindex.infer() ──

/// Mmap'd model weights, reusable across infer() calls.
/// Created lazily on first infer(), held by PyVindex.
pub struct InferState {
    pub weights: ModelWeights,
    _mmaps: Vec<WeightMmap>,
}

impl InferState {
    pub fn load(dir: &Path) -> Result<Self, String> {
        let (weights, mmaps) = load_mmap_weights(dir)?;
        Ok(Self {
            weights,
            _mmaps: mmaps,
        })
    }
}

// ── Python class ──

#[pyclass(name = "WalkModel", unsendable)]
pub struct PyWalkModel {
    weights: ModelWeights,
    index: VectorIndex,
    tokenizer: tokenizers::Tokenizer,
    top_k: usize,
    path: String,
    // Hold mmaps alive — weight arrays reference this memory
    _mmaps: Vec<WeightMmap>,
}

#[pymethods]
impl PyWalkModel {
    /// Load a walk model from a vindex directory.
    ///
    /// Weight files are mmap'd, not loaded to heap. Only the pages
    /// touched during inference are paged into RSS by the OS.
    #[new]
    #[pyo3(signature = (path, top_k=8192))]
    fn new(path: &str, top_k: usize) -> PyResult<Self> {
        let dir = std::path::Path::new(path);

        let mut load_cb = SilentLoadCallbacks;
        let index = VectorIndex::load_vindex(dir, &mut load_cb)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let (weights, mmaps) =
            load_mmap_weights(dir).map_err(pyo3::exceptions::PyIOError::new_err)?;

        let tokenizer = load_vindex_tokenizer(dir)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        Ok(Self {
            weights,
            index,
            tokenizer,
            top_k,
            path: path.to_string(),
            _mmaps: mmaps,
        })
    }

    /// Run full forward pass with walk FFN. Returns [(token, probability)].
    #[pyo3(signature = (prompt, top_k_predictions=5))]
    fn predict(&self, prompt: &str, top_k_predictions: usize) -> PyResult<Vec<(String, f64)>> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);
        let result = predict_with_ffn(
            &self.weights,
            &self.tokenizer,
            &token_ids,
            top_k_predictions,
            &walk_ffn,
        );

        Ok(result.predictions)
    }

    /// Run walk FFN for a single layer.
    ///
    /// Accepts raw f32 bytes (from MLX memoryview), returns raw f32 bytes.
    /// No numpy: MLX → bytes → Rust → bytes → MLX.
    fn ffn_layer<'py>(
        &self,
        py: Python<'py>,
        layer: usize,
        x_bytes: &[u8],
        seq_len: usize,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let hidden = self.weights.hidden_size;
        let expected = seq_len * hidden * 4;
        if x_bytes.len() != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} bytes ({}x{}xf32), got {}",
                expected,
                seq_len,
                hidden,
                x_bytes.len()
            )));
        }

        let floats: &[f32] =
            unsafe { std::slice::from_raw_parts(x_bytes.as_ptr() as *const f32, seq_len * hidden) };
        let x_arr = ndarray::ArrayView2::from_shape((seq_len, hidden), floats)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);
        let output = walk_ffn.forward(layer, &x_arr.to_owned());

        let out_slice = output.as_slice().unwrap();
        let out_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(out_slice.as_ptr() as *const u8, out_slice.len() * 4)
        };
        Ok(PyBytes::new(py, out_bytes))
    }

    /// Feature selection only — returns indices for MLX sparse matmul.
    ///
    /// Runs gate KNN on the vindex for each sequence position, returns the
    /// union of top-K feature indices. MLX uses these to gather rows and
    /// do the matmul on Metal GPU.
    ///
    /// Args:
    ///     layer: layer index
    ///     x_bytes: raw f32 bytes (seq_len × hidden) from MLX
    ///     seq_len: number of sequence positions
    ///     top_k: features to select per position (default: self.top_k)
    ///
    /// Returns:
    ///     List of feature indices (sorted, deduplicated union across positions)
    #[pyo3(signature = (layer, x_bytes, seq_len, top_k=None))]
    fn gate_select(
        &self,
        layer: usize,
        x_bytes: &[u8],
        seq_len: usize,
        top_k: Option<usize>,
    ) -> PyResult<Vec<usize>> {
        let hidden = self.weights.hidden_size;
        let expected = seq_len * hidden * 4;
        if x_bytes.len() != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} bytes ({}x{}xf32), got {}",
                expected,
                seq_len,
                hidden,
                x_bytes.len()
            )));
        }

        let k = top_k.unwrap_or(self.top_k);
        let floats: &[f32] =
            unsafe { std::slice::from_raw_parts(x_bytes.as_ptr() as *const f32, seq_len * hidden) };

        // Collect features across all positions
        let mut seen = std::collections::HashSet::new();
        for s in 0..seq_len {
            let row = &floats[s * hidden..(s + 1) * hidden];
            let arr = ndarray::Array1::from_vec(row.to_vec());
            let hits = self.index.gate_knn(layer, &arr, k);
            for (idx, _score) in hits {
                seen.insert(idx);
            }
        }

        let mut indices: Vec<usize> = seen.into_iter().collect();
        indices.sort_unstable();
        Ok(indices)
    }

    /// Feature selection returning indices and gate scores.
    ///
    /// Like gate_select but also returns the max gate score per feature
    /// (useful for debugging / weighted sparse FFN).
    #[pyo3(signature = (layer, x_bytes, seq_len, top_k=None))]
    fn gate_select_scored(
        &self,
        layer: usize,
        x_bytes: &[u8],
        seq_len: usize,
        top_k: Option<usize>,
    ) -> PyResult<(Vec<usize>, Vec<f32>)> {
        let hidden = self.weights.hidden_size;
        let expected = seq_len * hidden * 4;
        if x_bytes.len() != expected {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} bytes ({}x{}xf32), got {}",
                expected,
                seq_len,
                hidden,
                x_bytes.len()
            )));
        }

        let k = top_k.unwrap_or(self.top_k);
        let floats: &[f32] =
            unsafe { std::slice::from_raw_parts(x_bytes.as_ptr() as *const f32, seq_len * hidden) };

        let mut best: std::collections::HashMap<usize, f32> = std::collections::HashMap::new();
        for s in 0..seq_len {
            let row = &floats[s * hidden..(s + 1) * hidden];
            let arr = ndarray::Array1::from_vec(row.to_vec());
            let hits = self.index.gate_knn(layer, &arr, k);
            for (idx, score) in hits {
                let entry = best.entry(idx).or_insert(0.0f32);
                if score.abs() > entry.abs() {
                    *entry = score;
                }
            }
        }

        let mut pairs: Vec<(usize, f32)> = best.into_iter().collect();
        pairs.sort_unstable_by_key(|(idx, _)| *idx);
        let indices = pairs.iter().map(|(i, _)| *i).collect();
        let scores = pairs.iter().map(|(_, s)| *s).collect();
        Ok((indices, scores))
    }

    #[getter]
    fn num_layers(&self) -> usize {
        self.weights.num_layers
    }

    #[getter]
    fn hidden_size(&self) -> usize {
        self.weights.hidden_size
    }

    #[getter]
    fn intermediate_size(&self) -> usize {
        self.weights.intermediate_size
    }

    #[getter]
    fn top_k(&self) -> usize {
        self.top_k
    }

    /// Capture a complete residual stream trace.
    ///
    /// Runs a full forward pass through WalkFfn, recording the residual,
    /// attn_delta, and post-attention ffn_delta at every layer. Returns a
    /// ResidualTrace object.
    ///
    /// Args:
    ///     prompt: Input text
    ///     positions: "last" (default) or "all"
    ///
    /// Example:
    ///     t = walk_model.trace("The capital of France is")
    ///     t.answer_trajectory("Paris")
    #[pyo3(signature = (prompt, positions="last"))]
    fn trace(&self, prompt: &str, positions: &str) -> PyResult<trace_py::PyResidualTrace> {
        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);
        trace_py::capture_trace_with_ffn(
            &self.weights,
            &self.tokenizer,
            prompt,
            positions,
            &walk_ffn,
        )
    }

    // ── Mechanistic interp surface (lazarus parity) ────────────────────────
    //
    // These methods mirror the chuk-mcp-lazarus tool surface. They run a
    // forward pass with a `LayerHook` registered and return numpy tensors
    // ready for Python-side analysis.

    /// Tokenize then capture last-token residual at each requested layer.
    ///
    /// Returns `dict[layer_index] -> numpy.ndarray (hidden_size,)`.
    #[pyo3(signature = (prompt, layers))]
    fn capture_residuals<'py>(
        &self,
        py: Python<'py>,
        prompt: &str,
        layers: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let token_ids = self.encode(prompt)?;
        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);
        let mut hook = RecordHook::for_layers(layers.iter().copied());
        let _ = trace_forward_full_hooked(
            &self.weights,
            &token_ids,
            &layers,
            false,
            0,
            false,
            &walk_ffn,
            &mut hook,
        );

        let out = PyDict::new(py);
        for (layer, mat) in hook.post_layer.iter() {
            // Last-token row only — matches the convention everywhere else
            // in larql_inference. Full matrix available via
            // `forward_with_capture` if a caller needs every position.
            let last = mat.row(mat.nrows() - 1).to_vec();
            out.set_item(*layer, last.into_pyarray(py))?;
        }
        Ok(out)
    }

    /// Run a forward pass with a [`RecordHook`] and return the **full**
    /// `(seq_len, hidden_size)` post-layer residual at each requested
    /// layer. Larger than `capture_residuals` — only call when you need
    /// per-position activations (patching, full causal trace).
    ///
    /// Returns `dict[layer_index] -> numpy.ndarray (seq_len, hidden_size)`.
    #[pyo3(signature = (prompt, layers))]
    fn forward_with_capture<'py>(
        &self,
        py: Python<'py>,
        prompt: &str,
        layers: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let token_ids = self.encode(prompt)?;
        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);
        let mut hook = RecordHook::for_layers(layers.iter().copied());
        let _ = trace_forward_full_hooked(
            &self.weights,
            &token_ids,
            &layers,
            false,
            0,
            false,
            &walk_ffn,
            &mut hook,
        );

        let out = PyDict::new(py);
        for (layer, mat) in hook.post_layer.iter() {
            out.set_item(*layer, mat.clone().into_pyarray(py))?;
        }
        Ok(out)
    }

    /// Zero-ablate the post-layer residual at the listed `ablate_layers`,
    /// then capture last-token residuals at `capture_layers`. Mirrors
    /// lazarus's `ablate_layers` + measurement workflow.
    ///
    /// Returns `dict[layer_index] -> numpy.ndarray (hidden_size,)` for
    /// each capture layer (post-ablation).
    #[pyo3(signature = (prompt, ablate_layers, capture_layers))]
    fn forward_ablate<'py>(
        &self,
        py: Python<'py>,
        prompt: &str,
        ablate_layers: Vec<usize>,
        capture_layers: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let token_ids = self.encode(prompt)?;
        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);
        let mut ablate = ZeroAblateHook::for_layers(ablate_layers);
        let trace = trace_forward_full_hooked(
            &self.weights,
            &token_ids,
            &capture_layers,
            false,
            0,
            false,
            &walk_ffn,
            &mut ablate,
        );

        let out = PyDict::new(py);
        for (layer, residual) in trace.residuals {
            out.set_item(layer, residual.into_pyarray(py))?;
        }
        Ok(out)
    }

    /// Add `alpha * v` to the last-token row of the post-layer residual at
    /// each (layer, vector, alpha) entry, then capture last-token
    /// residuals at `capture_layers`. Mirrors lazarus's `steer_and_generate`
    /// at the residual-readback level.
    ///
    /// `steers` is a list of `(layer, numpy_vector, alpha)` tuples.
    #[pyo3(signature = (prompt, steers, capture_layers))]
    fn forward_steer<'py>(
        &self,
        py: Python<'py>,
        prompt: &str,
        steers: Vec<(usize, PyReadonlyArray1<f32>, f32)>,
        capture_layers: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let token_ids = self.encode(prompt)?;
        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);

        let mut steer = SteerHook::new();
        for (layer, vec, alpha) in steers {
            let arr = Array1::from_vec(vec.as_slice()?.to_vec());
            steer = steer.add(layer, arr, alpha);
        }
        let trace = trace_forward_full_hooked(
            &self.weights,
            &token_ids,
            &capture_layers,
            false,
            0,
            false,
            &walk_ffn,
            &mut steer,
        );

        let out = PyDict::new(py);
        for (layer, residual) in trace.residuals {
            out.set_item(layer, residual.into_pyarray(py))?;
        }
        Ok(out)
    }

    /// Activation patching. Run `donor_prompt`, capture post-layer
    /// residuals at the `(layer, position)` coords in `coords`, then run
    /// `recipient_prompt` with those residuals patched in at the same
    /// coords. Returns last-token residuals at `capture_layers` (post-
    /// patch).
    ///
    /// Mirrors lazarus's `patch_activations`. Uses the vindex WalkFfn path
    /// so patches are measured against the same mechanism as inference.
    #[pyo3(signature = (donor_prompt, recipient_prompt, coords, capture_layers))]
    fn patch_activations<'py>(
        &self,
        py: Python<'py>,
        donor_prompt: &str,
        recipient_prompt: &str,
        coords: Vec<(usize, usize)>,
        capture_layers: Vec<usize>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let donor_tokens = self.encode(donor_prompt)?;
        let recipient_tokens = self.encode(recipient_prompt)?;

        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);
        let donor = capture_donor_state_with_ffn(&self.weights, &donor_tokens, &coords, &walk_ffn);
        let trace = patch_and_trace_with_ffn(
            &self.weights,
            &recipient_tokens,
            &donor,
            &capture_layers,
            &walk_ffn,
        );

        let out = PyDict::new(py);
        for (layer, residual) in trace.residuals {
            out.set_item(layer, residual.into_pyarray(py))?;
        }
        Ok(out)
    }

    // ── Logit lens / vocab projection ──────────────────────────────────────

    /// Project `residual` through final norm + lm_head + softcap and
    /// return the top-`k` `(token_id, probability)` pairs.
    #[pyo3(signature = (residual, k=10))]
    fn logit_lens(&self, residual: PyReadonlyArray1<f32>, k: usize) -> PyResult<Vec<(u32, f32)>> {
        Ok(logit_lens_topk(&self.weights, residual.as_slice()?, k))
    }

    /// Probability of `target_token_id` at the residual.
    fn track_token_at(
        &self,
        residual: PyReadonlyArray1<f32>,
        target_token_id: u32,
    ) -> PyResult<f32> {
        Ok(li_track_token(
            &self.weights,
            residual.as_slice()?,
            target_token_id,
        ))
    }

    /// Top-k per layer for a `dict[layer] -> residual` mapping.
    /// Returns `dict[layer] -> List[(token_id, prob)]`.
    #[pyo3(signature = (residuals, k=5))]
    fn track_race<'py>(
        &self,
        py: Python<'py>,
        residuals: &Bound<'py, PyDict>,
        k: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let mut pairs: Vec<(usize, Vec<f32>)> = Vec::with_capacity(residuals.len());
        for (key, val) in residuals.iter() {
            let layer: usize = key.extract()?;
            let arr: PyReadonlyArray1<f32> = val.extract()?;
            pairs.push((layer, arr.as_slice()?.to_vec()));
        }
        let race = li_track_race(&self.weights, &pairs, k);
        let out = PyDict::new(py);
        for (layer, top) in race {
            out.set_item(layer, top)?;
        }
        Ok(out)
    }

    /// Top-`k` vocab tokens by cosine similarity to `query` against `W_E`.
    /// Returns `[(token_id, cosine), ...]` descending.
    #[pyo3(signature = (query, k=10))]
    fn embedding_neighbors(
        &self,
        query: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<Vec<(u32, f32)>> {
        Ok(li_embedding_neighbors(&self.weights, query.as_slice()?, k))
    }

    /// Raw `lm_head @ vec` projection — top-`k` `(token_id, logit)` pairs.
    /// **No final norm, no softcap, no softmax.** This is the DLA
    /// primitive — apply it to a head's contribution or any direction
    /// you want to read out as a vocabulary distribution without the
    /// model's final-stage normalisation.
    #[pyo3(signature = (vec, k=10))]
    fn project_through_unembed(
        &self,
        vec: PyReadonlyArray1<f32>,
        k: usize,
    ) -> PyResult<Vec<(u32, f32)>> {
        Ok(li_project_through_unembed(
            &self.weights,
            vec.as_slice()?,
            k,
        ))
    }

    /// Embedding row for `token_id`. `scaled=True` (default) returns the
    /// row multiplied by `embed_scale` so it matches what the forward
    /// pass writes into the residual. `scaled=False` returns the raw
    /// matrix row.
    #[pyo3(signature = (token_id, scaled=true))]
    fn embedding_for<'py>(
        &self,
        py: Python<'py>,
        token_id: u32,
        scaled: bool,
    ) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        let row = if scaled {
            li_embedding_row_scaled(&self.weights, token_id)
        } else {
            li_embedding_row(&self.weights, token_id)
        };
        Ok(row.map(|r| r.into_pyarray(py)))
    }

    /// Unembedding (`lm_head`) row for `token_id` — the direction whose
    /// dot product with the final residual gives the raw logit for that
    /// token (before any norm/softcap/scaling).
    fn unembedding_for<'py>(
        &self,
        py: Python<'py>,
        token_id: u32,
    ) -> PyResult<Option<Bound<'py, PyArray1<f32>>>> {
        Ok(li_unembedding_row(&self.weights, token_id).map(|r| r.into_pyarray(py)))
    }

    /// Multi-token generation with a `LayerHook` active on **every layer
    /// of every step** (prefill + each decode step). Mirrors lazarus's
    /// `steer_and_generate` and `ablate_and_generate` workflows.
    ///
    /// Pass an `ablate_layers` list to zero the post-layer residual at
    /// those layers, and/or a `steers` list of `(layer, vector, alpha)`
    /// triples to add `alpha * v` to the last-token row at those layers.
    /// Both apply on every step. Returns the generated string and the
    /// raw token ids.
    ///
    /// **Backend note**: this routes to the CPU KV-cache path. The
    /// Metal-fast `predict` is hook-free by design (kernel pipeline is
    /// fused). For mech-interp use cases hooks-on-CPU is the right
    /// trade.
    #[pyo3(signature = (prompt, max_new_tokens, ablate_layers=None, steers=None))]
    fn generate_with_hooks(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        ablate_layers: Option<Vec<usize>>,
        steers: Option<Vec<(usize, PyReadonlyArray1<f32>, f32)>>,
    ) -> PyResult<(String, Vec<u32>)> {
        let token_ids = self.encode(prompt)?;
        let walk_ffn = WalkFfn::new(&self.weights, &self.index, self.top_k);

        // Build the active hook(s). When both ablate + steer are present,
        // wrap them in a CompositeHook; otherwise pass the single hook
        // directly so we don't pay for the extra dispatch.
        let mut ablate = ZeroAblateHook::for_layers(ablate_layers.unwrap_or_default());
        let mut steer = SteerHook::new();
        if let Some(steers) = steers {
            for (layer, vec, alpha) in steers {
                let arr = Array1::from_vec(vec.as_slice()?.to_vec());
                steer = steer.add(layer, arr, alpha);
            }
        }

        let mut composite = larql_inference::forward::CompositeHook::new(vec![
            &mut ablate as &mut dyn larql_inference::forward::LayerHook,
            &mut steer as &mut dyn larql_inference::forward::LayerHook,
        ]);

        let mut generated_text = String::new();
        let ids = generate_cached_hooked(
            &self.weights,
            &self.tokenizer,
            &walk_ffn,
            &token_ids,
            max_new_tokens,
            None,
            None,
            &mut composite,
            |_id, text| generated_text.push_str(text),
        );
        Ok((generated_text, ids))
    }

    fn __repr__(&self) -> String {
        format!(
            "WalkModel(path='{}', layers={}, hidden={}, top_k={})",
            self.path, self.weights.num_layers, self.weights.hidden_size, self.top_k
        )
    }
}

impl PyWalkModel {
    /// Tokenize a prompt to ids, raising a Python ValueError on failure.
    fn encode(&self, prompt: &str) -> PyResult<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }
}
