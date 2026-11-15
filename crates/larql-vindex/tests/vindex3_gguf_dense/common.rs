//! Shared fixture plumbing.

use std::path::{Path, PathBuf};

use larql_models::loading::gguf::{GgufFile, GgufValue};
use larql_vindex::format::vindex3::fixtures::gemma3_text_container;
use larql_vindex::format::vindex3::gguf::dense::{export_dense, DenseExport, DenseExportOptions};

/// A freshly encoded fixture container, kept alive with its work dir.
pub struct Fixture {
    pub work: tempfile::TempDir,
    pub container: PathBuf,
}

impl Fixture {
    pub fn new() -> Self {
        let work = tempfile::tempdir().unwrap();
        let container = gemma3_text_container(work.path());
        Self { work, container }
    }

    pub fn out(&self, name: &str) -> PathBuf {
        self.work.path().join(name)
    }

    pub fn export(&self, name: &str) -> (DenseExport, PathBuf) {
        let out = self.out(name);
        let report = export_dense(&self.container, &out, &DenseExportOptions::default())
            .expect("the fixture exports");
        (report, out)
    }

    /// Rewrite a JSON file inside the container.
    pub fn edit_json(&self, name: &str, edit: impl FnOnce(&mut serde_json::Value)) {
        edit_json(&self.container.join(name), edit);
    }
}

pub fn edit_json(path: &Path, edit: impl FnOnce(&mut serde_json::Value)) {
    let mut v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    edit(&mut v);
    std::fs::write(path, serde_json::to_string_pretty(&v).unwrap()).unwrap();
}

pub fn meta<'a>(f: &'a GgufFile, key: &str) -> &'a GgufValue {
    f.metadata
        .get(key)
        .unwrap_or_else(|| panic!("`{key}` missing"))
}

pub fn strings(v: &GgufValue) -> Vec<String> {
    match v {
        GgufValue::Array(a) => a.iter().map(|x| x.as_str().unwrap().to_string()).collect(),
        other => panic!("expected an array, got {other:?}"),
    }
}

pub fn tensor_bytes(f: &GgufFile, path: &Path, name: &str, len: usize) -> Vec<u8> {
    let info = f
        .tensor_infos
        .iter()
        .find(|t| t.name() == name)
        .unwrap_or_else(|| panic!("tensor `{name}` missing"));
    let raw = std::fs::read(path).unwrap();
    let start = (f.data_offset + info.offset()) as usize;
    raw[start..start + len].to_vec()
}
