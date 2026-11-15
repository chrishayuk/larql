use pyo3::prelude::*;

mod f32_bytes;
mod json_py;
mod session;
mod session_ops;
mod sync;
mod testing;
mod trace_py;
mod vindex;
mod walk;

use session::PySession;
use session_ops::PyReadout;
use trace_py::{
    PyAnswerWaypoint, PyBoundaryStore, PyBoundaryWriter, PyLayerSummary, PyResidualTrace,
    PyTraceStore,
};
use vindex::{PyDescribeEdge, PyFeatureMeta, PyRelation, PyVindex, PyWalkHit};
use walk::PyWalkModel;

mod graph;
mod graph_fns;
pub use graph::*;
use graph_fns::*;

// ── Vindex top-level functions ──

/// Load a vindex from a directory path.
///
/// Returns a Vindex object with gate vectors, embeddings, and tokenizer.
/// Supports numpy array access for gate vectors and embeddings.
///
/// Example:
///     vindex = larql.load_vindex("gemma3-4b.vindex")
///     embed = vindex.embed("France")
///     hits = vindex.entity_knn("France", layer=26, top_k=10)
#[pyfunction]
fn load_vindex(py: Python<'_>, path: &str) -> PyResult<PyVindex> {
    PyVindex::open_detached(py, path)
}

/// Create an LQL session connected to a vindex.
///
/// The session provides both LQL query execution and direct vindex access:
///     session = larql.session("gemma3-4b.vindex")
///     session.query("DESCRIBE 'France'")         # LQL queries
///     session.vindex.embed("France")              # numpy arrays
#[pyfunction]
fn create_session(py: Python<'_>, path: &str) -> PyResult<PySession> {
    PySession::create(py, path)
}

// ── Module ──

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Graph types (existing)
    m.add_class::<PyEdge>()?;
    m.add_class::<PyNode>()?;
    m.add_class::<PyGraph>()?;

    // Vindex types (new)
    m.add_class::<PyVindex>()?;
    m.add_class::<PyFeatureMeta>()?;
    m.add_class::<PyWalkHit>()?;
    m.add_class::<PyDescribeEdge>()?;
    m.add_class::<PyRelation>()?;
    m.add_class::<PySession>()?;
    m.add_class::<PyReadout>()?;
    m.add_class::<PyWalkModel>()?;
    m.add_class::<PyResidualTrace>()?;
    m.add_class::<PyAnswerWaypoint>()?;
    m.add_class::<PyLayerSummary>()?;
    m.add_class::<PyTraceStore>()?;
    m.add_class::<PyBoundaryStore>()?;
    m.add_class::<PyBoundaryWriter>()?;

    // Graph functions (existing)
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(save, m)?)?;
    m.add_function(wrap_pyfunction!(shortest_path, m)?)?;
    m.add_function(wrap_pyfunction!(merge_graphs, m)?)?;
    m.add_function(wrap_pyfunction!(merge_graphs_with_strategy, m)?)?;
    m.add_function(wrap_pyfunction!(diff, m)?)?;
    m.add_function(wrap_pyfunction!(pagerank, m)?)?;
    m.add_function(wrap_pyfunction!(bfs_traversal, m)?)?;
    m.add_function(wrap_pyfunction!(dfs_traversal, m)?)?;
    m.add_function(wrap_pyfunction!(load_csv, m)?)?;
    m.add_function(wrap_pyfunction!(save_csv, m)?)?;
    m.add_function(wrap_pyfunction!(weight_walk, m)?)?;
    m.add_function(wrap_pyfunction!(attention_walk, m)?)?;

    // Vindex functions (new)
    m.add_function(wrap_pyfunction!(load_vindex, m)?)?;
    m.add_function(wrap_pyfunction!(create_session, m)?)?;
    m.add_function(wrap_pyfunction!(testing::py_write_test_container, m)?)?;

    Ok(())
}
