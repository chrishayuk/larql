//! Python bindings for LQL sessions.
//!
//! Wraps larql_lql::Session to provide LQL query execution from Python.
//! Two interfaces, one session:
//! - session.query("DESCRIBE 'France'") — LQL string queries
//! - session.vindex — direct PyVindex access for numpy arrays

use pyo3::prelude::*;

use crate::vindex::PyVindex;
use larql_lql::{parse, Session, Statement};
use larql_vindex::format::generation::ContainerGeneration;

// ── PySession ──

#[pyclass(name = "Session", unsendable)]
pub struct PySession {
    session: Session,
    vindex_obj: Option<(std::path::PathBuf, Py<PyVindex>)>,
    path: String,
}

impl PySession {
    /// Create a session (Rust-callable).
    pub fn create(_py: Python<'_>, path: &str) -> PyResult<Self> {
        let mut session = Session::new();

        // Execute USE to connect the LQL session to the vindex
        let use_stmt = format!(
            "USE \"{}\";",
            path.replace('\\', "\\\\").replace('"', "\\\"")
        );
        let stmt = parse(&use_stmt)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Parse error: {e}")))?;
        session
            .execute(&stmt)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("USE failed: {e}")))?;

        // Direct arrays are a separate V2 capability, loaded only on access.
        // A V3 session must never pass through the VectorIndex loader.
        Ok(Self {
            session,
            vindex_obj: None,
            path: path.to_string(),
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
    fn query(&mut self, lql: &str) -> PyResult<Vec<String>> {
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
        if matches!(stmt, Statement::Use { .. }) {
            self.vindex_obj = None;
        }
        let result = self.session.execute(&stmt).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Execution error: {e}"))
        })?;
        if let Some((path, _)) = self.session.local_artifact() {
            self.path = path.to_string_lossy().into_owned();
        } else {
            self.path.clear();
        }
        Ok(result)
    }

    /// Execute an LQL query and return results as a single string.
    fn query_text(&mut self, lql: &str) -> PyResult<String> {
        let lines = self.query(lql)?;
        Ok(lines.join("\n"))
    }

    /// Direct NumPy arrays for a local V2 artifact. V3 uses the LQL interface.
    #[getter]
    fn vindex(&mut self, py: Python<'_>) -> PyResult<Py<PyVindex>> {
        let Some((path, generation)) = self.session.local_artifact() else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Direct array access requires a local V2 artifact",
            ));
        };
        if generation == ContainerGeneration::V3 {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "VINDEX3 has no direct VectorIndex array view; use Session.query()",
            ));
        }
        if self
            .vindex_obj
            .as_ref()
            .is_none_or(|(cached_path, _)| cached_path != path)
        {
            self.vindex_obj = Some((
                path.to_path_buf(),
                Py::new(py, PyVindex::open(&path.to_string_lossy())?)?,
            ));
        }
        Ok(self
            .vindex_obj
            .as_ref()
            .expect("array view loaded")
            .1
            .clone_ref(py))
    }

    /// Current local artifact path; empty after binding a remote or weight backend.
    #[getter]
    fn path(&self) -> &str {
        &self.path
    }

    fn __repr__(&self) -> String {
        format!("Session(path='{}')", self.path)
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
            let mut session = PySession::create(py, container.path().to_str().unwrap()).unwrap();
            assert!(session.query_text("STATS").unwrap().contains("VINDEX3"));
            let output = session.query_text("INFER \"[3]\" GENERATE 4").unwrap();
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
            session.query(&format!("USE \"{escaped}\"")).unwrap();
            assert_eq!(session.path(), container.path().to_str().unwrap());
            assert!(session.query_text("SHOW LAYERS").is_ok());
        });
    }
}
