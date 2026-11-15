//! Fixtures, a session, and a direct view of the segment bytes.

use std::path::{Path, PathBuf};

use larql_lql::{parse, Session};
use larql_vindex::format::vindex3::encode::segment::read_segment_header;
use larql_vindex::format::vindex3::fixtures::{
    bf16_container_with_q8_pack, encode_fixture_container, miniature_glimmer,
};

/// A BF16 matrix, `[64, 256]`.
pub const MATRIX: &str = "1.mlp.down_proj.weight";
pub const MATRIX_ROWS: usize = 64;
pub const MATRIX_COLS: usize = 256;
/// A 1-D norm gain, `[64]`.
pub const GAIN: &str = "0.input_layernorm.weight";
pub const GAIN_LEN: usize = 64;
/// The name the embedding, the final norm and the output head all hold.
pub const AMBIGUOUS: &str = "weight";
pub const CANONICAL: &str = "target.decoder_stack@BF16";
pub const DERIVED: &str = "target.decoder_stack@Q8_0";
pub const EMBEDDING: &str = "target.embedding@BF16";
pub const EMBEDDING_SHAPE: [usize; 2] = [128, 64];
/// The F32 miniature's down projection, `[12, 20]`.
pub const F32_MATRIX: &str = "0.mlp.down_proj.weight";
pub const F32_CANONICAL: &str = "target.decoder_stack@F32";
pub const F32_SHAPE: [usize; 2] = [12, 20];
pub const BF16_BYTES: usize = 2;

pub struct Fixture {
    _dirs: Vec<tempfile::TempDir>,
    pub root: PathBuf,
    /// Where the test writes its exports.
    pub out: tempfile::TempDir,
}

impl Fixture {
    pub fn bf16() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = bf16_container_with_q8_pack(dir.path());
        Self {
            _dirs: vec![dir],
            root,
            out: tempfile::tempdir().unwrap(),
        }
    }

    pub fn f32() -> Self {
        let checkpoint = tempfile::tempdir().unwrap();
        let container = tempfile::tempdir().unwrap();
        encode_fixture_container(
            miniature_glimmer,
            checkpoint.path(),
            container.path(),
            "tensor-export",
        );
        let root = container.path().to_path_buf();
        Self {
            _dirs: vec![checkpoint, container],
            root,
            out: tempfile::tempdir().unwrap(),
        }
    }

    pub fn path(&self, name: &str) -> PathBuf {
        self.out.path().join(name)
    }

    pub fn session(&self) -> Session {
        let mut s = Session::new();
        run(&mut s, &format!("USE \"{}\";", q(&self.root)));
        s
    }
}

/// A path as it is written inside an LQL string literal.
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

/// The raw bytes of one tensor of one representation, read straight from its segment.
pub fn segment_bytes(root: &Path, representation: &str, tensor: &str) -> Vec<u8> {
    let index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("index.json")).unwrap()).unwrap();
    let segment = root.join(
        index["representations"][representation]["segment"]
            .as_str()
            .unwrap(),
    );
    let (header, start) = read_segment_header(&segment).unwrap();
    let t = header.tensors.iter().find(|t| t.name == tensor).unwrap();
    let raw = std::fs::read(&segment).unwrap();
    raw[(start + t.offset) as usize..(start + t.offset + t.len) as usize].to_vec()
}

/// BF16 bits, little-endian bytes in.
pub fn bf16_bits(bytes: &[u8]) -> Vec<u16> {
    bytes
        .chunks_exact(BF16_BYTES)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

/// f32 values, little-endian bytes in.
pub fn f32_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// The exact f32 of a BF16 bit pattern.
pub fn widen(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

pub fn sidecar(npy: &Path) -> serde_json::Value {
    let path = format!("{}.json", npy.display());
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

pub fn export(tensor: &str, selection: &str, out: &Path, suffix: &str) -> String {
    format!(
        "EXPORT WEIGHT \"{tensor}\"{selection} TO \"{}\"{suffix};",
        q(out)
    )
}
