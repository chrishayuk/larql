//! A tiny synthetic VINDEX3 container for the binding's tests, so they run anywhere without model files.
//!
//! It is the container the LQL `CAPTURE` tests run on: BF16 canonical weights (two layers, hidden 64) with a
//! Q8_0 pack, and a word-level tokenizer in which each of "The capital of France is Paris" is one token and
//! every other id is `[i]`.

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use larql_vindex::format::vindex3::fixtures::{bf16_container_with_q8_pack, DENSE_VOCAB};

/// The words the tokenizer knows besides `[i]`, and the id the first one takes.
const WORDS: [&str; 6] = ["The", "capital", "of", "France", "is", "Paris"];
const FIRST_WORD_ID: usize = 10;
const TOKENIZER_JSON: &str = "tokenizer.json";

fn token(i: usize) -> String {
    match i.checked_sub(FIRST_WORD_ID) {
        Some(w) if w < WORDS.len() => WORDS[w].to_string(),
        _ => format!("[{i}]"),
    }
}

fn tokenizer_json() -> String {
    let entries: serde_json::Map<String, serde_json::Value> =
        (0..DENSE_VOCAB).map(|i| (token(i), i.into())).collect();
    serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": entries, "unk_token": "[0]"}
    })
    .to_string()
}

/// Build the test container under `work` (an existing, empty directory) and return its path.
pub(crate) fn write_test_container(work: &std::path::Path) -> std::io::Result<std::path::PathBuf> {
    let container = bf16_container_with_q8_pack(work);
    std::fs::write(container.join(TOKENIZER_JSON), tokenizer_json())?;
    Ok(container)
}

/// Build the binding tests' synthetic VINDEX3 container under `work` and return its path. A test
/// helper: the container is tiny and its weights are synthetic.
#[pyfunction]
#[pyo3(name = "_write_test_container")]
pub(crate) fn py_write_test_container(py: Python<'_>, work: &str) -> PyResult<String> {
    py.detach(|| write_test_container(std::path::Path::new(work)))
        .map(|p| p.to_string_lossy().into_owned())
        .map_err(|e| PyRuntimeError::new_err(format!("write the test container: {e}")))
}
