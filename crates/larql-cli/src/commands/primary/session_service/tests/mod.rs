//! The session service, driven the way a client drives it: request lines in, response lines out, on the
//! synthetic VINDEX3 container (two BF16 layers, hidden 64, a word-level tokenizer in which each word of
//! "The capital of France is Paris" is one token). Every array is held bit for bit to the typed operation run
//! in-process on a session of its own.

mod budget;
mod limits;
mod npy;
mod operations;
mod paths;
mod policy;
mod protocol;
mod server;

use std::path::{Path, PathBuf};

use base64::Engine;
use serde_json::{json, Value};

use larql_lql::Session;
use larql_vindex::format::vindex3::fixtures::{bf16_container_with_q8_pack, DENSE_VOCAB};

use super::policy::{Policy, POLICY_VERSION};
use super::protocol::{Op, PROTOCOL};
use super::service::Service;

pub(super) const PROMPT: &str = "The capital of France is";
pub(super) const CONTINUATION: &str = " Paris";
pub(super) const TENSOR: &str = "1.mlp.down_proj.weight";
pub(super) const F64_PROFILE: &str = "f64-seq-rne1/v1";
pub(super) const COLUMNS: [usize; 3] = [8, 9, 33];
/// A delta on row 5 large enough to move what the model computes.
pub(super) const ROW: usize = 5;
pub(super) const DELTA: f32 = 64.0;

const WORDS: [&str; 6] = ["The", "capital", "of", "France", "is", "Paris"];
const FIRST_WORD_ID: usize = 10;

fn token(i: usize) -> String {
    match i.checked_sub(FIRST_WORD_ID) {
        Some(w) if w < WORDS.len() => WORDS[w].to_string(),
        _ => format!("[{i}]"),
    }
}

fn tokenizer_json() -> String {
    let entries: serde_json::Map<String, Value> =
        (0..DENSE_VOCAB).map(|i| (token(i), i.into())).collect();
    json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": entries, "unk_token": "[0]"}
    })
    .to_string()
}

/// A container, a workdir and a place for the audit log, each in its own temporary directory.
pub(super) struct Fixture {
    _models: tempfile::TempDir,
    pub container: PathBuf,
    pub work: tempfile::TempDir,
    pub logs: tempfile::TempDir,
}

impl Fixture {
    pub fn new() -> Self {
        let models = tempfile::tempdir().unwrap();
        let container = bf16_container_with_q8_pack(models.path());
        std::fs::write(container.join("tokenizer.json"), tokenizer_json()).unwrap();
        Self {
            _models: models,
            container,
            work: tempfile::tempdir().unwrap(),
            logs: tempfile::tempdir().unwrap(),
        }
    }

    pub fn workdir(&self) -> &Path {
        self.work.path()
    }

    pub fn log(&self) -> PathBuf {
        self.logs.path().join("audit.jsonl")
    }

    /// A policy allowing every operation, with `budget` edited executions.
    pub fn policy(&self, budget: u64) -> Value {
        json!({
            "policy": POLICY_VERSION,
            "container": self.container,
            "workdir": self.workdir(),
            "audit_log": self.log(),
            "operations": Op::names(),
            "budget": {"edited_executions": budget},
            "staging_dir": self.logs.path(),
        })
    }

    pub fn start_with(&self, policy: &Value) -> Result<Service, String> {
        Service::start(Policy::from_bytes(policy.to_string().as_bytes())?)
    }

    pub fn start(&self, budget: u64) -> Service {
        self.start_with(&self.policy(budget)).unwrap()
    }

    /// An in-process session on the same container: what the service's answers are held to.
    pub fn session(&self) -> Session {
        let mut s = Session::new();
        let stmt = larql_lql::Statement::Use {
            target: larql_lql::ast::UseTarget::Vindex(self.container.display().to_string()),
        };
        s.execute(&stmt).unwrap();
        s
    }

    /// The audit log's records.
    pub fn records(&self) -> Vec<Value> {
        std::fs::read_to_string(self.log())
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect()
    }

    /// A `.vlp` in the workdir carrying the delta [`DELTA`] on row [`ROW`], recorded in-process.
    pub fn patch(&self, name: &str) -> PathBuf {
        let path = self.workdir().join(name);
        let mut s = self.session();
        s.begin_patch(path.to_str().unwrap()).unwrap();
        s.stage_weight_delta(TENSOR, ROW, &[0, 1, 2, 3, 4, 5, 6, 7], DELTA)
            .unwrap();
        s.save_patch().unwrap();
        path
    }
}

/// A provenance record as the service sends it: the container's path replaced by its label.
pub(super) fn redacted(mut provenance: Value) -> Value {
    if provenance.get("container").is_some() {
        provenance["container"] = json!(super::service::CONTAINER_LABEL);
    }
    provenance
}

/// A request line.
pub(super) fn line(op: &str, args: Value) -> Vec<u8> {
    serde_json::to_vec(&json!({"protocol": PROTOCOL, "id": 7, "op": op, "args": args})).unwrap()
}

/// Send one request, give the parsed response.
pub(super) fn call(service: &mut Service, op: &str, args: Value) -> Value {
    serde_json::from_slice(&service.handle_line(&line(op, args))).unwrap()
}

/// The result of a request that must succeed.
pub(super) fn ok(service: &mut Service, op: &str, args: Value) -> Value {
    let r = call(service, op, args.clone());
    assert_eq!(r["ok"], true, "{op} {args}: {r}");
    r["result"].clone()
}

/// The error of a request that must fail: its kind and reason.
pub(super) fn err(service: &mut Service, op: &str, args: Value) -> (String, String) {
    let r = call(service, op, args.clone());
    assert_eq!(r["ok"], false, "{op} {args}: {r}");
    (
        r["error"]["kind"].as_str().unwrap().to_string(),
        r["error"]["reason"].as_str().unwrap().to_string(),
    )
}

/// The `.npy` bytes an array in a result carries, its digest checked.
pub(super) fn npy_bytes(array: &Value) -> Vec<u8> {
    use sha2::{Digest, Sha256};
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(array["npy_base64"].as_str().unwrap())
        .unwrap();
    assert_eq!(array["sha256"], format!("{:x}", Sha256::digest(&bytes)));
    bytes
}

/// `values` as the `npy_base64` argument of an operand: a 2-D float64 `.npy`.
pub(super) fn operand(values: &[f64], shape: [usize; 2]) -> Value {
    let body: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let header = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': ({}, {}), }}",
        shape[0], shape[1]
    );
    json!({"npy_base64": base64::engine::general_purpose::STANDARD.encode(npy_v1(&header, &body))})
}

/// A version 1.0 `.npy` file with this header dict and body, padded as NumPy pads.
pub(super) fn npy_v1(header: &str, body: &[u8]) -> Vec<u8> {
    let mut dict = header.to_string();
    let unpadded = 10 + dict.len() + 1;
    dict.push_str(&" ".repeat(unpadded.next_multiple_of(64) - unpadded));
    dict.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(dict.len() as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.extend_from_slice(body);
    out
}

/// The rank-1 `a` factor over the tensor's rows.
pub(super) fn a_factor(rows: usize, scale: f64) -> Vec<f64> {
    vec![scale; rows]
}

pub(super) fn tensor_rows(session: &Session) -> usize {
    session
        .tensors(None)
        .unwrap()
        .into_iter()
        .find(|t| t.tensor == TENSOR && t.canonical)
        .unwrap()
        .shape[0]
}
