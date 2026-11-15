//! A `serde_json::Value` as plain Python objects: `dict`, `list`, `str`, `int`, `float`, `bool` and `None`.
//!
//! Provenance records are built once, in Rust, as the record a sidecar holds; this hands the same record to
//! Python without a text round trip.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::Value;

pub(crate) fn json_to_py<'py>(py: Python<'py>, value: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match value {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => b.into_pyobject(py)?.to_owned().into_any(),
        Value::Number(n) => match (n.as_u64(), n.as_i64()) {
            (Some(u), _) => u.into_pyobject(py)?.into_any(),
            (None, Some(i)) => i.into_pyobject(py)?.into_any(),
            _ => n.as_f64().unwrap_or(f64::NAN).into_pyobject(py)?.into_any(),
        },
        Value::String(s) => s.into_pyobject(py)?.into_any(),
        Value::Array(items) => {
            let list = PyList::empty(py);
            for item in items {
                list.append(json_to_py(py, item)?)?;
            }
            list.into_any()
        }
        Value::Object(map) => {
            let dict = PyDict::new(py);
            for (key, item) in map {
                dict.set_item(key, json_to_py(py, item)?)?;
            }
            dict.into_any()
        }
    })
}
