//! Typed VINDEX3 operations on a `Session`: tensors, tokens, captures, predictions and patches, with NumPy
//! arrays in and out.
//!
//! Every method calls the same typed Rust operation the LQL statement's executor calls (see
//! `larql_lql::operations`), so a result here equals what the statement prints or writes: no LQL text is built
//! and no console output is read. Arrays handed out are owned copies. Every call runs detached from the GIL,
//! holding the session's lock, as `Session.query` does.

use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use larql_lql::operations::{
    operand_bits_from_values, operand_origin, CaptureContinuation, CaptureLayers, CapturePositions,
    CaptureSite, CaptureSpec, NpyDescr, OperandValues,
};
use larql_lql::LqlError;

use crate::json_py::json_to_py;
use crate::session::PySession;
use crate::sync::lock;

/// The origin an op staged from NumPy arrays records, before the arrays' shapes and digests.
const ARRAY_ORIGIN: &str = "python arrays";
/// The `positions=` spellings, as `CAPTURE … POSITIONS` spells them.
const POSITION_WORDS: &str = "LAST, PROMPT_END, REPLY or ALL";

fn failed(e: LqlError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// An owned array and the provenance record of the values in it.
#[pyclass(name = "Readout", frozen)]
pub struct PyReadout {
    array: Py<PyAny>,
    provenance: Py<PyAny>,
}

#[pymethods]
impl PyReadout {
    /// The values: an owned `numpy.ndarray`.
    #[getter]
    fn array(&self, py: Python<'_>) -> Py<PyAny> {
        self.array.clone_ref(py)
    }

    /// Where the values came from: the fields the matching LQL statement's sidecar records, with
    /// `statement` None and `npy_sha256` the digest of the `.npy` file that statement would write.
    #[getter]
    fn provenance(&self, py: Python<'_>) -> Py<PyAny> {
        self.provenance.clone_ref(py)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let array = self.array.bind(py);
        Ok(format!(
            "Readout(shape={}, dtype={})",
            array.getattr("shape")?.repr()?,
            array.getattr("dtype")?.str()?
        ))
    }
}

/// Little-endian elements as an owned array of `shape`.
fn le_array<'py, T, const N: usize>(
    py: Python<'py>,
    bytes: &[u8],
    shape: &[usize],
    from: fn([u8; N]) -> T,
) -> PyResult<Bound<'py, PyAny>>
where
    T: numpy::Element,
{
    let values: Vec<T> = bytes
        .chunks_exact(N)
        .map(|c| from(c.try_into().expect("chunks_exact gives N bytes")))
        .collect();
    let array = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), values)
        .map_err(|e| PyRuntimeError::new_err(format!("array shape: {e}")))?;
    Ok(array.into_pyarray(py).into_any())
}

fn readout(
    py: Python<'_>,
    array: Bound<'_, PyAny>,
    provenance: &serde_json::Value,
) -> PyResult<PyReadout> {
    Ok(PyReadout {
        array: array.unbind(),
        provenance: json_to_py(py, provenance)?.unbind(),
    })
}

fn positions_arg(positions: &Bound<'_, PyAny>) -> PyResult<CapturePositions> {
    if let Ok(word) = positions.extract::<String>() {
        return match word.to_ascii_uppercase().as_str() {
            "LAST" => Ok(CapturePositions::Last),
            "PROMPT_END" => Ok(CapturePositions::PromptEnd),
            "REPLY" => Ok(CapturePositions::Reply),
            "ALL" => Ok(CapturePositions::All),
            _ => Err(PyValueError::new_err(format!(
                "positions={word:?}: name {POSITION_WORDS}, or pass a list of positions"
            ))),
        };
    }
    positions
        .extract::<Vec<usize>>()
        .map(CapturePositions::List)
        .map_err(|_| {
            PyTypeError::new_err(format!(
                "positions is {POSITION_WORDS}, or a list of non-negative ints"
            ))
        })
}

fn layers_arg(layer: Option<usize>, layers: Option<Vec<usize>>) -> PyResult<Option<CaptureLayers>> {
    match (layer, layers) {
        (Some(_), Some(_)) => Err(PyValueError::new_err("pass layer= or layers=, not both")),
        (Some(one), None) => Ok(Some(CaptureLayers::One(one))),
        (None, Some(many)) => Ok(Some(CaptureLayers::Many(many))),
        (None, None) => Ok(None),
    }
}

fn continuation_arg(
    text: Option<String>,
    tokens: Option<Vec<u32>>,
    generate: Option<u32>,
) -> PyResult<Option<CaptureContinuation>> {
    match (text, tokens, generate) {
        (None, None, None) => Ok(None),
        (Some(t), None, None) => Ok(Some(CaptureContinuation::Text(t))),
        (None, Some(ids), None) => Ok(Some(CaptureContinuation::Tokens(ids))),
        (None, None, Some(n)) => Ok(Some(CaptureContinuation::Generate(n))),
        _ => Err(PyValueError::new_err(
            "pass at most one of continue_text=, continue_tokens= and generate=",
        )),
    }
}

/// A 2-D operand array as its values, row-major, and its shape. float64 and float32 are taken as they are.
fn operand_values(array: &Bound<'_, PyAny>, which: &str) -> PyResult<(Operand, [usize; 2])> {
    if let Ok(a) = array.extract::<PyReadonlyArray2<f64>>() {
        let view = a.as_array();
        let shape = [view.nrows(), view.ncols()];
        return Ok((Operand::F64(view.iter().copied().collect()), shape));
    }
    if let Ok(a) = array.extract::<PyReadonlyArray2<f32>>() {
        let view = a.as_array();
        let shape = [view.nrows(), view.ncols()];
        return Ok((Operand::F32(view.iter().copied().collect()), shape));
    }
    Err(PyTypeError::new_err(format!(
        "`{which}` must be a 2-D numpy array of float64 or float32"
    )))
}

/// Operand values copied out of NumPy, so the GIL can be released while they are staged.
enum Operand {
    F64(Vec<f64>),
    F32(Vec<f32>),
}

impl Operand {
    fn values(&self) -> OperandValues<'_> {
        match self {
            Operand::F64(v) => OperandValues::F64(v),
            Operand::F32(v) => OperandValues::F32(v),
        }
    }
}

#[pymethods]
impl PySession {
    /// Every stored tensor of the bound VINDEX3 container, as `SHOW TENSORS ["filter"]` lists it.
    ///
    /// Returns a list of dicts: `name` (the qualified name, which always works), `representation`,
    /// `object`, `tensor`, `canonical`, `dtype`, `shape` and `bare` (whether the bare name works:
    /// "yes", "no" or "ambiguous").
    #[pyo3(signature = (filter=None))]
    fn tensors<'py>(
        &self,
        py: Python<'py>,
        filter: Option<String>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let listings = py
            .detach(|| lock(&self.session).tensors(filter.as_deref()))
            .map_err(failed)?;
        listings
            .iter()
            .map(|l| {
                let d = PyDict::new(py);
                d.set_item("name", l.qualified_name())?;
                d.set_item("representation", &l.representation)?;
                d.set_item("object", &l.object)?;
                d.set_item("tensor", &l.tensor)?;
                d.set_item("canonical", l.canonical)?;
                d.set_item("dtype", &l.dtype)?;
                d.set_item("shape", l.shape.clone())?;
                d.set_item("bare", l.bare_name.label())?;
                Ok(d)
            })
            .collect()
    }

    /// One stored tensor, or a selection of its rows and columns, as `EXPORT WEIGHT` exports it.
    ///
    /// The values are float32, widened exactly from the stored dtype; with `bits=True` they are the
    /// stored element bits (uint16 for BF16/F16, uint32 for F32). They are the container's stored bytes:
    /// edits staged or applied in this session are not included.
    ///
    /// Example: `s.tensor("target.embedding@BF16::weight", rows=[9079]).array`
    #[pyo3(signature = (name, rows=None, cols=None, bits=false))]
    fn tensor(
        &self,
        py: Python<'_>,
        name: &str,
        rows: Option<Vec<usize>>,
        cols: Option<Vec<usize>>,
        bits: bool,
    ) -> PyResult<PyReadout> {
        let read = py
            .detach(|| {
                lock(&self.session).read_weight(name, rows.as_deref(), cols.as_deref(), bits)
            })
            .map_err(failed)?;
        let array = match read.descr {
            NpyDescr::F4 => le_array(py, &read.bytes, &read.shape, f32::from_le_bytes)?,
            NpyDescr::U2 => le_array(py, &read.bytes, &read.shape, u16::from_le_bytes)?,
            NpyDescr::U4 => le_array(py, &read.bytes, &read.shape, u32::from_le_bytes)?,
        };
        readout(py, array, &read.provenance)
    }

    /// The token ids `text` encodes to.
    ///
    /// By default this is how `INFER` and `CAPTURE` encode a prompt (INFER's raw-prompt encoder: the
    /// tokenizer's own special tokens, plus the container's declared BOS where the tokenizer did not add
    /// it; no chat template). With `raw=True` it is the text's tokens alone, no special tokens: how
    /// `CAPTURE … CONTINUE "<text>"` encodes its continuation.
    #[pyo3(signature = (text, raw=false))]
    fn tokenize(&self, py: Python<'_>, text: &str, raw: bool) -> PyResult<Vec<u32>> {
        py.detach(|| lock(&self.session).tokenize(text, !raw))
            .map_err(failed)
    }

    /// The text `ids` decode to, special tokens kept. `s.decode([i])` is the token a capture's provenance
    /// records for a position holding `i`.
    fn decode(&self, py: Python<'_>, ids: Vec<u32>) -> PyResult<String> {
        py.detach(|| lock(&self.session).decode(&ids))
            .map_err(failed)
    }

    /// The values the session's program computes at one site, as `CAPTURE` writes them.
    ///
    /// `site` is one of residual, post_attention, ffn.input, ffn.down_input, ffn.down_output (inside a
    /// layer: pass `layer=` or `layers=`), final or logits. The sequence is the prompt, then at most one of
    /// `continue_text=`, `continue_tokens=` or `generate=` (INFER's greedy tokens). `positions` is "LAST"
    /// (the default), "PROMPT_END", "REPLY", "ALL" or a list. The array is `[positions, width]`, or
    /// `[layers, positions, width]` with `layers=`.
    ///
    /// Example: `s.capture("logits", "The capital of France is").array`
    #[pyo3(signature = (
        site, prompt, layer=None, layers=None, continue_text=None, continue_tokens=None, generate=None,
        positions=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn capture(
        &self,
        py: Python<'_>,
        site: &str,
        prompt: &str,
        layer: Option<usize>,
        layers: Option<Vec<usize>>,
        continue_text: Option<String>,
        continue_tokens: Option<Vec<u32>>,
        generate: Option<u32>,
        positions: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyReadout> {
        let site = CaptureSite::from_name(site).ok_or_else(|| {
            let names: Vec<&str> = CaptureSite::ALL.iter().map(|s| s.name()).collect();
            PyValueError::new_err(format!(
                "unknown site {site:?}; sites are {}",
                names.join(", ")
            ))
        })?;
        let layers = layers_arg(layer, layers)?;
        let continuation = continuation_arg(continue_text, continue_tokens, generate)?;
        let positions = match positions {
            Some(p) => positions_arg(p)?,
            None => CapturePositions::Last,
        };
        let captured = py
            .detach(|| {
                lock(&self.session).capture(&CaptureSpec {
                    site,
                    layers: layers.as_ref(),
                    prompt,
                    continuation: continuation.as_ref(),
                    positions: &positions,
                })
            })
            .map_err(failed)?;
        let values = captured.values();
        let array = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&captured.shape), values)
            .map_err(|e| PyRuntimeError::new_err(format!("array shape: {e}")))?;
        readout(py, array.into_pyarray(py).into_any(), &captured.provenance)
    }

    /// The next-token prediction `INFER "<prompt>" TOP <top>` prints, as values.
    ///
    /// Returns a dict: `prompt_ids`, `predictions` (a list of `{"token_id", "token", "probability"}`, most
    /// probable first) and `knn_override` (None, or the session KNN store's `{"token", "cosine", "layer"}`
    /// where it overrides the model's top token).
    #[pyo3(signature = (prompt, top=5))]
    fn infer<'py>(
        &self,
        py: Python<'py>,
        prompt: &str,
        top: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let predicted = py
            .detach(|| lock(&self.session).predict(prompt, top))
            .map_err(failed)?;
        let d = PyDict::new(py);
        d.set_item("prompt_ids", predicted.prompt_ids)?;
        let rows: Vec<Bound<'py, PyDict>> = predicted
            .top
            .iter()
            .map(|p| {
                let row = PyDict::new(py);
                row.set_item("token_id", p.token_id)?;
                row.set_item("token", &p.token)?;
                row.set_item("probability", p.probability)?;
                Ok(row)
            })
            .collect::<PyResult<_>>()?;
        d.set_item("predictions", rows)?;
        match &predicted.knn_override {
            Some(ovr) => {
                let o = PyDict::new(py);
                o.set_item("token", &ovr.token)?;
                o.set_item("cosine", ovr.cosine)?;
                o.set_item("layer", ovr.layer)?;
                d.set_item("knn_override", o)?;
            }
            None => d.set_item("knn_override", py.None())?,
        }
        Ok(d)
    }

    /// The greedy continuation `INFER "<prompt>" GENERATE <n>` prints, as values: a dict of `prompt_ids`,
    /// `token_ids` (at most `n`; a stop token ends generation and is not included) and `text`.
    fn generate<'py>(
        &self,
        py: Python<'py>,
        prompt: &str,
        n: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let generated = py
            .detach(|| lock(&self.session).generate(prompt, n))
            .map_err(failed)?;
        let d = PyDict::new(py);
        d.set_item("prompt_ids", generated.prompt_ids)?;
        d.set_item("token_ids", generated.token_ids)?;
        d.set_item("text", generated.text)?;
        Ok(d)
    }

    /// Open a patch recording that `save_patch()` writes to `path` (`BEGIN PATCH "<path>"`).
    fn begin_patch(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| lock(&self.session).begin_patch(path))
            .map_err(failed)
    }

    /// Stage a rank-`k` update `a @ b` on `columns` of `tensor` (`ADD LOW RANK`), from NumPy arrays.
    ///
    /// `a` is `rows × k` and `b` is `k × len(columns)`. They are converted, in Rust, to the operand bits
    /// of the profile `numerics`: an f64-working profile takes float64 (float32 is widened exactly), the
    /// f32 one takes float32. The op records its origin as "python arrays" with each array's shape and
    /// SHA-256. An open recording records it. Returns a dict: `digest`, `changed_elements`, `sham`,
    /// `pending_structured_ops` and `origin`.
    fn add_low_rank<'py>(
        &self,
        py: Python<'py>,
        tensor: &str,
        a: &Bound<'py, PyAny>,
        b: &Bound<'py, PyAny>,
        columns: Vec<usize>,
        numerics: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (a_values, a_shape) = operand_values(a, "a")?;
        let (b_values, b_shape) = operand_values(b, "b")?;
        let rank = a_shape[1];
        if b_shape != [rank, columns.len()] {
            return Err(PyValueError::new_err(format!(
                "`b` is {}x{}; for `a` of rank {rank} over {} column(s) it must be {rank}x{}",
                b_shape[0],
                b_shape[1],
                columns.len(),
                columns.len()
            )));
        }
        let staged = py
            .detach(|| {
                let a = operand_bits_from_values(numerics, a_values.values(), "a")?;
                let b = operand_bits_from_values(numerics, b_values.values(), "b")?;
                let origin = operand_origin(ARRAY_ORIGIN, &a, a_shape, &b, b_shape);
                lock(&self.session).stage_low_rank(
                    tensor,
                    &columns,
                    numerics,
                    rank,
                    &a,
                    &b,
                    Some(origin),
                )
            })
            .map_err(failed)?;
        let d = PyDict::new(py);
        d.set_item("digest", &staged.op.digest)?;
        d.set_item("changed_elements", staged.changed_elements)?;
        d.set_item("sham", staged.sham)?;
        d.set_item("pending_structured_ops", staged.pending_structured_ops)?;
        d.set_item("origin", staged.op.origin.clone())?;
        Ok(d)
    }

    /// Stage `delta` on `cols` of `row` of `tensor` (`UPDATE WEIGHTS`). Returns a dict: `digest`,
    /// `elements` (a list of `{"col", "old_bits", "new_bits"}`, BF16 bits) and `pending_elements`.
    fn update_weights<'py>(
        &self,
        py: Python<'py>,
        tensor: &str,
        row: usize,
        cols: Vec<usize>,
        delta: f32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let staged = py
            .detach(|| lock(&self.session).stage_weight_delta(tensor, row, &cols, delta))
            .map_err(failed)?;
        let d = PyDict::new(py);
        d.set_item("digest", &staged.digest)?;
        let elements: Vec<Bound<'py, PyDict>> = staged
            .elements
            .iter()
            .map(|e| {
                let row = PyDict::new(py);
                row.set_item("col", e.col)?;
                row.set_item("old_bits", e.old_bits)?;
                row.set_item("new_bits", e.new_bits)?;
                Ok(row)
            })
            .collect::<PyResult<_>>()?;
        d.set_item("elements", elements)?;
        d.set_item("pending_elements", staged.pending_elements)?;
        Ok(d)
    }

    /// Write the open recording (`SAVE PATCH`). Returns a dict: `path`, `operations`, `inserts`,
    /// `updates`, `deletes` and `weight_edits`.
    fn save_patch<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let saved = py
            .detach(|| lock(&self.session).save_patch())
            .map_err(failed)?;
        let d = PyDict::new(py);
        d.set_item("path", saved.path.to_string_lossy().into_owned())?;
        d.set_item("operations", saved.operations)?;
        d.set_item("inserts", saved.inserts)?;
        d.set_item("updates", saved.updates)?;
        d.set_item("deletes", saved.deletes)?;
        d.set_item("weight_edits", saved.weight_edits)?;
        Ok(d)
    }

    /// Apply the `.vlp` at `path` through the session overlay (`APPLY PATCH`); the container is not
    /// touched. Returns a dict: `path`, `sha256` (of the bytes applied), `operations`, `inserts`,
    /// `updates`, `deletes`, `weight_edits` and `structured_ops`.
    fn apply_patch<'py>(&self, py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
        let applied = py
            .detach(|| lock(&self.session).apply_patch(path))
            .map_err(failed)?;
        let d = PyDict::new(py);
        d.set_item("path", &applied.path)?;
        d.set_item("sha256", &applied.sha256)?;
        d.set_item("operations", applied.operations)?;
        d.set_item("inserts", applied.inserts)?;
        d.set_item("updates", applied.updates)?;
        d.set_item("deletes", applied.deletes)?;
        d.set_item("weight_edits", applied.weight_edits)?;
        d.set_item("structured_ops", applied.structured_ops)?;
        Ok(d)
    }

    /// Remove the applied patch `path` names (`REMOVE PATCH`). Returns a dict: `number` (its 1-based
    /// place in `patches()`) and `carried_weight_ops`.
    fn remove_patch<'py>(&self, py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
        let removed = py
            .detach(|| lock(&self.session).remove_patch(path))
            .map_err(failed)?;
        let d = PyDict::new(py);
        d.set_item("number", removed.number)?;
        d.set_item("carried_weight_ops", removed.carried_weight_ops)?;
        Ok(d)
    }

    /// Drop every stored-weight op staged directly in this session, and the copies an open recording holds
    /// (`DISCARD WEIGHT EDITS`); ops an applied patch carries stay with it. Returns a dict: `discarded`,
    /// `unrecorded` (ops removed from the open recording) and `from_patches` (weight ops applied patches
    /// still carry).
    fn discard_weight_edits<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let discarded = py
            .detach(|| lock(&self.session).discard_weight_edits())
            .map_err(failed)?;
        let d = PyDict::new(py);
        d.set_item("discarded", discarded.discarded)?;
        d.set_item("unrecorded", discarded.unrecorded)?;
        d.set_item("from_patches", discarded.from_patches)?;
        Ok(d)
    }

    /// What `SHOW PATCHES` reports, as values: a dict of `applied` (a list of `{"name", "path", "sha256",
    /// "operations", "inserts", "updates", "deletes"}`), `session_overrides`, `weight_ops`,
    /// `weight_elements` and `recording` (None, or `{"path", "operations"}`; the path is empty for the
    /// anonymous recording).
    fn patches<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let report = py
            .detach(|| lock(&self.session).patches())
            .map_err(failed)?;
        let d = PyDict::new(py);
        let applied: Vec<Bound<'py, PyDict>> = report
            .applied
            .iter()
            .map(|p| {
                let row = PyDict::new(py);
                row.set_item("name", p.name.clone())?;
                row.set_item("path", p.path.clone())?;
                row.set_item("sha256", p.sha256.clone())?;
                row.set_item("operations", p.operations)?;
                row.set_item("inserts", p.inserts)?;
                row.set_item("updates", p.updates)?;
                row.set_item("deletes", p.deletes)?;
                Ok(row)
            })
            .collect::<PyResult<_>>()?;
        d.set_item("applied", applied)?;
        d.set_item("session_overrides", report.session_overrides)?;
        d.set_item("weight_ops", report.weight_ops)?;
        d.set_item("weight_elements", report.weight_elements)?;
        match &report.recording {
            Some((path, operations)) => {
                let r = PyDict::new(py);
                r.set_item("path", path)?;
                r.set_item("operations", *operations)?;
                d.set_item("recording", r)?;
            }
            None => d.set_item("recording", py.None())?,
        }
        Ok(d)
    }
}
