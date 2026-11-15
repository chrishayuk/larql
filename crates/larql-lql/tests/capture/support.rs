//! A small BF16 container that INFER runs on, a session over it, INFER's own pass for comparison, and a reader
//! for what CAPTURE writes.

use std::path::{Path, PathBuf};

use larql_inference::vindex3::{encode_raw_prompt, OpenPolicy, Vindex3Runtime};
use larql_kv::{shipped_continuations, CanonicalFactory};
use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::fixtures::{bf16_container_with_q8_pack, DENSE_VOCAB};
use larql_vindex::format::vindex3::opplan::exec::continuation_authority::ContinuationConfig;
use larql_vindex::format::vindex3::opplan::exec::continuation_registry::ContinuationFactory;
use larql_vindex::format::vindex3::opplan::exec::lowering::{LoweringIdentity, LoweringRegistry};
use larql_vindex::format::vindex3::opplan::exec::operands::RepresentationSource;

/// The words the fixture's tokenizer knows besides `[i]`, and the id the first one takes.
const WORDS: [&str; 6] = ["The", "capital", "of", "France", "is", "Paris"];
const FIRST_WORD_ID: usize = 10;
pub const PROMPT: &str = "The capital of France is";
/// The prompt's tokens: one per word.
pub const PROMPT_TOKENS: usize = 5;
pub const CONTINUATION: &str = " Paris";
pub const PARIS_ID: u32 = (FIRST_WORD_ID + 5) as u32;
pub const LAYERS: usize = 2;
pub const HIDDEN: usize = 64;
pub const INTERMEDIATE: usize = 256;
pub const VOCAB: usize = DENSE_VOCAB;
/// The tensor the patch edits, and an edit large enough to move what the model computes.
pub const TENSOR: &str = "1.mlp.down_proj.weight";
pub const STAGE: &str =
    "UPDATE WEIGHTS SET DELTA = 64.0 WHERE TENSOR = \"1.mlp.down_proj.weight\" \
                         AND ROW = 5 AND COL IN (0, 1, 2, 3, 4, 5, 6, 7);";
const TOKENIZER_JSON: &str = "tokenizer.json";
const NPY_MAGIC: &[u8] = b"\x93NUMPY";
const NPY_PREAMBLE: usize = 10;

fn token(i: usize) -> String {
    match i.checked_sub(FIRST_WORD_ID) {
        Some(w) if w < WORDS.len() => WORDS[w].to_string(),
        _ => format!("[{i}]"),
    }
}

/// Whitespace-split word-level vocabulary: each word one token.
fn tokenizer_json() -> String {
    let entries: serde_json::Map<String, serde_json::Value> =
        (0..VOCAB).map(|i| (token(i), i.into())).collect();
    serde_json::json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "WhitespaceSplit"}, "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": entries, "unk_token": "[0]"}
    })
    .to_string()
}

pub struct Fixture {
    pub dir: tempfile::TempDir,
    pub container: PathBuf,
}

impl Fixture {
    pub fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let container = bf16_container_with_q8_pack(dir.path());
        std::fs::write(container.join(TOKENIZER_JSON), tokenizer_json()).unwrap();
        Self { dir, container }
    }

    /// Without a tokenizer.
    pub fn tokenless() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let container = bf16_container_with_q8_pack(dir.path());
        Self { dir, container }
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }

    pub fn session(&self) -> Session {
        session_on(&self.container)
    }

    /// A `.vlp` carrying [`STAGE`], recorded in a session of its own.
    pub fn patch(&self, name: &str) -> PathBuf {
        let vlp = self.path(name);
        let mut s = self.session();
        run(&mut s, &format!("BEGIN PATCH \"{}\";", q(&vlp)));
        run(&mut s, STAGE);
        run(&mut s, "SAVE PATCH;");
        vlp
    }
}

pub fn session_on(container: &Path) -> Session {
    let mut s = Session::new();
    run(&mut s, &format!("USE \"{}\";", q(container)));
    s
}

pub fn q(path: impl AsRef<Path>) -> String {
    path.as_ref().display().to_string().replace('\\', "\\\\")
}

pub fn run(s: &mut Session, stmt: &str) -> Vec<String> {
    try_run(s, stmt).unwrap_or_else(|e| panic!("{stmt}: {e}"))
}

pub fn try_run(s: &mut Session, stmt: &str) -> Result<Vec<String>, String> {
    let parsed = parse(stmt).map_err(|e| format!("parse: {e}"))?;
    s.execute(&parsed).map_err(|e| e.to_string())
}

/// `CAPTURE <clauses> TO "<out>";`.
pub fn capture(s: &mut Session, clauses: &str, out: &Path) -> Vec<String> {
    run(s, &format!("CAPTURE {clauses} TO \"{}\";", q(out)))
}

pub fn try_capture(s: &mut Session, clauses: &str, out: &Path) -> Result<Vec<String>, String> {
    try_run(s, &format!("CAPTURE {clauses} TO \"{}\";", q(out)))
}

/// A `.npy` file CAPTURE wrote: its shape and its f32 values.
pub struct Npy {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
    pub bytes: Vec<u8>,
}

pub fn read_npy(path: &Path) -> Npy {
    let bytes = std::fs::read(path).unwrap();
    assert_eq!(&bytes[..NPY_MAGIC.len()], NPY_MAGIC);
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[NPY_PREAMBLE..NPY_PREAMBLE + header_len]).unwrap();
    assert!(header.contains("'descr': '<f4'"), "{header}");
    let dims = header.split("'shape': (").nth(1).unwrap();
    let dims = &dims[..dims.find(')').unwrap()];
    let shape = dims
        .split(',')
        .map(str::trim)
        .filter(|d| !d.is_empty())
        .map(|d| d.parse().unwrap())
        .collect();
    let values = bytes[NPY_PREAMBLE + header_len..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Npy {
        shape,
        values,
        bytes,
    }
}

pub fn sidecar(npy: &Path) -> serde_json::Value {
    let path = format!("{}.json", npy.display());
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

pub fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

/// The container opened exactly as `USE` opens it.
pub fn runtime(
    container: &Path,
) -> Vindex3Runtime<larql_vindex::format::vindex3::opplan::exec::lowering::SharedProvider> {
    Vindex3Runtime::open_with_via(
        container,
        "target",
        std::sync::Arc::new(LoweringRegistry::shipped()),
        &LoweringIdentity::cpu_production(),
        OpenPolicy {
            want: None,
            source: RepresentationSource::Stored,
        },
    )
    .unwrap()
}

/// The logits INFER ranks for `text`: its encoder, then its one-shot pass (`execute_streaming`).
pub fn infer_logits(container: &Path, text: &str) -> Vec<f32> {
    let runtime = runtime(container);
    let tokenizer = larql_vindex::load_vindex_tokenizer(container).unwrap();
    let bos = larql_inference::vindex3::declared_prompt_bos(container);
    let ids = encode_raw_prompt(&tokenizer, text, bos).unwrap();
    let continuation = runtime
        .select_continuation(
            &shipped_continuations(),
            &CanonicalFactory.identity(),
            &ContinuationConfig::empty(),
        )
        .unwrap();
    runtime
        .execute_streaming(&ids, &continuation, &mut |_| Ok(()))
        .unwrap()
        .logits
        .unwrap()
}

/// The rows of a `[positions, width]` array, or of block `b` of a `[layers, positions, width]` one.
pub fn rows(npy: &Npy, block: usize) -> Vec<Vec<f32>> {
    let (positions, width) = match npy.shape.as_slice() {
        [p, w] => (*p, *w),
        [_, p, w] => (*p, *w),
        other => panic!("shape {other:?}"),
    };
    let start = block * positions * width;
    npy.values[start..start + positions * width]
        .chunks_exact(width)
        .map(<[f32]>::to_vec)
        .collect()
}

pub fn has(lines: &[String], needle: &str) -> bool {
    lines.iter().any(|l| l.contains(needle))
}
