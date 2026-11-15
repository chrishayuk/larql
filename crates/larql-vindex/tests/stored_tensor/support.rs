//! Fixtures and a direct view of the segment bytes, independent of the module under test.

use std::path::{Path, PathBuf};

use larql_vindex::format::vindex3::encode::segment::read_segment_header;
use larql_vindex::format::vindex3::fixtures::{
    bf16_container_with_q8_pack, encode_fixture_container, miniature_glimmer,
};
use larql_vindex::format::vindex3::index::Vindex3Index;

/// A BF16 matrix, `[64, 256]`.
pub const MATRIX: &str = "1.mlp.down_proj.weight";
pub const MATRIX_ROWS: usize = 64;
pub const MATRIX_COLS: usize = 256;
/// A 1-D norm gain, `[64]`.
pub const GAIN: &str = "0.input_layernorm.weight";
/// The name the embedding, the final norm and the output head all hold.
pub const AMBIGUOUS: &str = "weight";
pub const CANONICAL: &str = "target.decoder_stack@BF16";
pub const DERIVED: &str = "target.decoder_stack@Q8_0";
pub const EMBEDDING: &str = "target.embedding@BF16";
pub const FINAL_NORM: &str = "target.final_norm@BF16";
pub const OUTPUT_HEAD: &str = "target.output_head@BF16";
/// The F32 miniature's down projection.
pub const F32_MATRIX: &str = "0.mlp.down_proj.weight";
pub const F32_CANONICAL: &str = "target.decoder_stack@F32";

pub struct Container {
    _dirs: Vec<tempfile::TempDir>,
    pub root: PathBuf,
}

/// The BF16 container with its Q8_0 pack.
pub fn bf16_packed() -> Container {
    let dir = tempfile::tempdir().unwrap();
    let root = bf16_container_with_q8_pack(dir.path());
    Container {
        _dirs: vec![dir],
        root,
    }
}

/// The F32 miniature container.
pub fn f32_miniature() -> Container {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_glimmer,
        checkpoint.path(),
        container.path(),
        "stored-tensor",
    );
    let root = container.path().to_path_buf();
    Container {
        _dirs: vec![checkpoint, container],
        root,
    }
}

/// The raw bytes of one tensor of one representation, read straight from its segment.
pub fn segment_bytes(root: &Path, representation: &str, tensor: &str) -> Vec<u8> {
    let index: Vindex3Index =
        serde_json::from_str(&std::fs::read_to_string(root.join("index.json")).unwrap()).unwrap();
    let path = root.join(&index.representations[representation].segment);
    let (header, start) = read_segment_header(&path).unwrap();
    let t = header.tensors.iter().find(|t| t.name == tensor).unwrap();
    let raw = std::fs::read(&path).unwrap();
    raw[(start + t.offset) as usize..(start + t.offset + t.len) as usize].to_vec()
}

/// Everything a read hands its sink, concatenated.
pub fn read_all(
    stored: &larql_vindex::format::vindex3::stored_tensor::StoredTensor,
    selection: &larql_vindex::format::vindex3::stored_tensor::Selection,
) -> Vec<u8> {
    let mut out = Vec::new();
    stored
        .read(selection, &mut |chunk| {
            out.extend_from_slice(chunk);
            Ok(())
        })
        .unwrap();
    out
}
