//! Python bindings for LQL sessions.
//!
//! Wraps larql_lql::Session to provide LQL query execution from Python.
//! Several interfaces, one session:
//! - session.query("DESCRIBE 'France'") — LQL string queries
//! - session.vindex — direct PyVindex access for numpy arrays (V2)
//! - typed VINDEX3 operations (tensor, capture, infer, patches) — see `session_ops`
//!
//! Threading: statements execute inside `Python::detach` (an INFER or a
//! USE load can run for seconds). The LQL session is behind a `Mutex`, so
//! statements from different Python threads run one at a time.

use pyo3::prelude::*;
use std::sync::Mutex;

use crate::sync::lock;
use crate::vindex::PyVindex;
use larql_lql::{parse, Session, Statement};
use larql_vindex::format::generation::ContainerGeneration;

// ── PySession ──

/// The cached direct-array view and the artifact path it was opened from.
type ArrayView = Option<(std::path::PathBuf, Py<PyVindex>)>;

#[pyclass(name = "Session", frozen)]
pub struct PySession {
    pub(crate) session: Mutex<Session>,
    vindex_obj: Mutex<ArrayView>,
    path: Mutex<String>,
}

impl PySession {
    /// Create a session (Rust-callable). The initial USE runs detached.
    pub fn create(py: Python<'_>, path: &str) -> PyResult<Self> {
        py.detach(|| Self::bind(path))
    }

    fn bind(path: &str) -> PyResult<Self> {
        let mut session = Session::new();

        // Execute USE to connect the LQL session to the vindex
        let stmt = larql_lql::Statement::Use {
            target: larql_lql::ast::UseTarget::Vindex(path.to_string()),
        };
        session
            .execute(&stmt)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("USE failed: {e}")))?;

        // Direct arrays are a separate V2 capability, loaded only on access.
        // A V3 session must never pass through the VectorIndex loader.
        Ok(Self {
            session: Mutex::new(session),
            vindex_obj: Mutex::new(None),
            path: Mutex::new(path.to_string()),
        })
    }
}

#[pymethods]
impl PySession {
    /// Create a session connected to a vindex.
    #[new]
    fn new(py: Python<'_>, path: &str) -> PyResult<Self> {
        Self::create(py, path)
    }

    /// Execute an LQL query string. Returns list of output lines.
    ///
    /// Examples:
    ///   session.query("DESCRIBE 'France'")
    ///   session.query("WALK 'The capital of France is' TOP 10")
    ///   session.query("STATS")
    ///   session.query("SELECT entity, target FROM EDGES WHERE relation = 'capital' LIMIT 10")
    fn query(&self, py: Python<'_>, lql: &str) -> PyResult<Vec<String>> {
        // Add semicolon if missing
        let input = if lql.trim_end().ends_with(';') {
            lql.to_string()
        } else {
            format!("{};", lql)
        };

        let stmt = parse(&input)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Parse error: {e}")))?;

        // USE can rebind to another artifact or a remote backend. Invalidate
        // the direct view even on a failed bind; it is cheap to reopen lazily.
        // The view is taken out while attached and dropped after the query.
        let _stale_view =
            matches!(stmt, Statement::Use { .. }).then(|| lock(&self.vindex_obj).take());
        py.detach(|| {
            let mut session = lock(&self.session);
            let result = session.execute(&stmt).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!("Execution error: {e}"))
            })?;
            *lock(&self.path) = session
                .local_artifact()
                .map(|(path, _)| path.to_string_lossy().into_owned())
                .unwrap_or_default();
            Ok(result)
        })
    }

    /// Execute an LQL query and return results as a single string.
    fn query_text(&self, py: Python<'_>, lql: &str) -> PyResult<String> {
        let lines = self.query(py, lql)?;
        Ok(lines.join("\n"))
    }

    /// Direct NumPy arrays for a local V2 artifact. V3 uses the LQL interface.
    #[getter]
    fn vindex(&self, py: Python<'_>) -> PyResult<Py<PyVindex>> {
        let artifact = py.detach(|| {
            lock(&self.session)
                .local_artifact()
                .map(|(path, generation)| (path.to_path_buf(), generation))
        });
        let Some((path, generation)) = artifact else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Direct array access requires a local V2 artifact",
            ));
        };
        if generation == ContainerGeneration::V3 {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "VINDEX3 has no direct VectorIndex array view; use Session.query(), or the \
                 typed methods such as Session.tensor() and Session.capture()",
            ));
        }
        // `vindex_obj` is only ever locked while attached and never across
        // a detach, so the GIL already orders its (brief) critical sections.
        let cached = lock(&self.vindex_obj)
            .as_ref()
            .filter(|(cached_path, _)| *cached_path == path)
            .map(|(_, view)| view.clone_ref(py));
        if let Some(view) = cached {
            return Ok(view);
        }
        let view = Py::new(py, PyVindex::open_detached(py, &path.to_string_lossy())?)?;
        let stale = lock(&self.vindex_obj).replace((path, view.clone_ref(py)));
        drop(stale);
        Ok(view)
    }

    /// Current local artifact path; empty after binding a remote or weight backend.
    #[getter]
    fn path(&self, py: Python<'_>) -> String {
        py.detach(|| lock(&self.path).clone())
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!("Session(path='{}')", self.path(py))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_vindex::format::vindex3::fixtures::{
        encode_fixture_container, miniature_glimmer, G_VOCAB,
    };

    #[test]
    fn v3_session_queries_without_a_v2_array_view_and_rebinds() {
        let checkpoint = tempfile::tempdir().unwrap();
        let container = tempfile::tempdir().unwrap();
        encode_fixture_container(
            miniature_glimmer,
            checkpoint.path(),
            container.path(),
            "python-v3",
        );
        std::fs::write(
            container.path().join("tokenizer.json"),
            larql_inference::test_utils::synthetic_tokenizer_json(G_VOCAB),
        )
        .unwrap();
        Python::attach(|py| {
            let session = PySession::create(py, container.path().to_str().unwrap()).unwrap();
            assert!(session.query_text(py, "STATS").unwrap().contains("VINDEX3"));
            let output = session.query_text(py, "INFER \"[3]\" GENERATE 4").unwrap();
            assert!(output.contains("ids:"), "{output}");
            assert!(session
                .vindex(py)
                .unwrap_err()
                .is_instance_of::<pyo3::exceptions::PyNotImplementedError>(py));
            let escaped = container
                .path()
                .to_str()
                .unwrap()
                .replace('\\', "\\\\")
                .replace('"', "\\\"");
            session.query(py, &format!("USE \"{escaped}\"")).unwrap();
            assert_eq!(session.path(py), container.path().to_str().unwrap());
            assert!(session.query_text(py, "SHOW LAYERS").is_ok());
        });
    }
}
