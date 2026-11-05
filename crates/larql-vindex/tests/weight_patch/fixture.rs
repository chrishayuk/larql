//! A small BF16 container with a compiled Q8_0 pack.

use std::path::{Path, PathBuf};

use larql_vindex::format::vindex3::encode::segment::read_segment_header;
use larql_vindex::format::vindex3::fixtures::{DENSE_HIDDEN, DENSE_INTERMEDIATE};
use larql_vindex::format::vindex3::index::Vindex3Index;

/// The tensor the tests edit: layer 1's down projection, `[64, 256]`.
pub const TENSOR: &str = "1.mlp.down_proj.weight";
pub const TENSOR_ROWS: usize = DENSE_HIDDEN;
pub const TENSOR_COLS: usize = DENSE_INTERMEDIATE;
pub const CANONICAL: &str = "target.decoder_stack@BF16";
pub const DERIVED: &str = "target.decoder_stack@Q8_0";
pub const Q8_BLOCK_BYTES: usize = 34;
pub const Q8_BLOCK_ELEMS: usize = 32;

pub struct Source {
    pub dir: tempfile::TempDir,
    pub container: PathBuf,
}

impl Source {
    pub fn out(&self, name: &str) -> PathBuf {
        self.dir.path().join(name)
    }
}

/// A BF16 container with its Q8_0 pack compiled by the production compiler.
pub fn source_with_pack() -> Source {
    let dir = tempfile::tempdir().unwrap();
    let container =
        larql_vindex::format::vindex3::fixtures::bf16_container_with_q8_pack(dir.path());
    Source { dir, container }
}

pub fn index_of(container: &Path) -> Vindex3Index {
    serde_json::from_str(&std::fs::read_to_string(container.join("index.json")).unwrap()).unwrap()
}

/// The raw bytes of one tensor in one representation's segment.
pub fn tensor_bytes(container: &Path, representation: &str, tensor: &str) -> Vec<u8> {
    let entry = index_of(container).representations[representation].clone();
    let path = container.join(&entry.segment);
    let (header, start) = read_segment_header(&path).unwrap();
    let t = header.tensors.iter().find(|t| t.name == tensor).unwrap();
    let raw = std::fs::read(&path).unwrap();
    raw[(start + t.offset) as usize..(start + t.offset + t.len) as usize].to_vec()
}

/// A hash of the whole directory's file contents, for "untouched" assertions.
pub fn tree_digest(root: &Path) -> Vec<(String, u64, String)> {
    use sha2::{Digest, Sha256};
    let mut out = Vec::new();
    fn walk(root: &Path, dir: &Path, out: &mut Vec<(String, u64, String)>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let e = e.unwrap();
            if e.file_type().unwrap().is_dir() {
                walk(root, &e.path(), out);
            } else {
                let bytes = std::fs::read(e.path()).unwrap();
                out.push((
                    e.path()
                        .strip_prefix(root)
                        .unwrap()
                        .to_string_lossy()
                        .to_string(),
                    bytes.len() as u64,
                    format!("{:x}", Sha256::digest(&bytes)),
                ));
            }
        }
    }
    walk(root, root, &mut out);
    out.sort();
    out
}
