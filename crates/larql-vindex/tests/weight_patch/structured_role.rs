//! The token embedding is refused by what the CONTAINER says it is (its graph object's kind), not by what its tensor is
//! called. A family that names the embedding something other than `embed_tokens` must be refused all the same.

use std::path::{Path, PathBuf};

use larql_vindex::format::vindex3::encode::segment::read_segment_header;
use larql_vindex::format::vindex3::weight_patch::structured::{OperandBits, ScalarBits};
use larql_vindex::format::vindex3::weight_patch::{
    read_bf16_tensor, StructuredWeightOp, WeightPatchError,
};
use sha2::{Digest, Sha256};

use crate::fixture::*;

const PROFILE: &str = "f64-seq-rne1/v1";
const F64_ONE: u64 = 0x3ff0_0000_0000_0000;
const ALPHA_NEAR_ONE: u64 = 0x3ff0_0000_4000_0000;
/// The fixture's embedding tensor is called `weight`, like the final norm's, so it is ambiguous by name. A name of the
/// SAME LENGTH keeps the segment header's size, and so every offset in the file, unchanged.
const ORIGINAL_NAME: &str = "weight";
pub(crate) const UNIQUE_NAME: &str = "embeds";
const EMBEDDING_SEGMENT: &str = "segments/target.embedding.bin";
const EMBEDDING_REPRESENTATION: &str = "target.embedding@BF16";
/// The BF16-only container the fixture builder leaves beside the packed one. In a container WITH a derived pack an
/// embedding patch is already refused on a structural rule (`DerivedFromOther`), so the role matters only without one.
const BF16_ONLY: &str = "canonical.vindex3";

fn copy_tree(from: &Path, to: &Path) {
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let path = entry.unwrap().path();
        let target = to.join(path.file_name().unwrap());
        if path.is_dir() {
            copy_tree(&path, &target);
        } else {
            std::fs::copy(&path, &target).unwrap();
        }
    }
}

/// A copy of the fixture container in which the embedding tensor has a unique name, with the segment hash re-sealed so
/// the container still reads as intact.
pub(crate) fn container_with_addressable_embedding(src: &Source) -> PathBuf {
    let dir = src.out("renamed.vindex3");
    copy_tree(&src.dir.path().join(BF16_ONLY), &dir);
    let segment = dir.join(EMBEDDING_SEGMENT);
    let (_, payload_start) = read_segment_header(&segment).unwrap();
    let mut bytes = std::fs::read(&segment).unwrap();
    let needle = format!("\"{ORIGINAL_NAME}\"").into_bytes();
    let header = &mut bytes[..payload_start as usize];
    let at: Vec<usize> = (0..header.len() - needle.len())
        .filter(|&i| header[i..i + needle.len()] == needle[..])
        .collect();
    assert_eq!(
        at.len(),
        1,
        "the embedding segment header names its tensor once"
    );
    header[at[0] + 1..at[0] + 1 + UNIQUE_NAME.len()].copy_from_slice(UNIQUE_NAME.as_bytes());
    std::fs::write(&segment, &bytes).unwrap();
    let index_path = dir.join("index.json");
    let mut index: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    index["representations"][EMBEDDING_REPRESENTATION]["segment_sha256"] = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect::<String>()
        .into();
    std::fs::write(&index_path, serde_json::to_string(&index).unwrap()).unwrap();
    dir
}

fn unit_row(rows: usize, row: usize) -> OperandBits {
    OperandBits::F64(
        (0..rows)
            .map(|r| if r == row { F64_ONE } else { 0 })
            .collect(),
    )
}

fn op_on(container: &Path, tensor: &str, column: usize) -> StructuredWeightOp {
    let ([rows, cols], base) = read_bf16_tensor(container, tensor).unwrap();
    let row = (0..rows)
        .find(|&r| base[r * cols + column] & 0x7fff != 0)
        .expect("a nonzero element in the column");
    StructuredWeightOp::bind(
        container,
        tensor,
        PROFILE,
        &unit_row(rows, row),
        ScalarBits::F64(ALPHA_NEAR_ONE),
        vec![column],
    )
    .unwrap()
}

#[test]
fn an_embedding_under_a_name_that_is_not_embed_tokens_is_still_refused() {
    let src = source_with_pack();
    let container = container_with_addressable_embedding(&src);
    let op = op_on(&container, UNIQUE_NAME, 3);
    match op.resolve(&container) {
        Err(WeightPatchError::Structured(why)) => {
            assert!(why.contains("tensor_refused"), "refused for the wrong reason: {why}")
        }
        other => panic!("an embedding named {UNIQUE_NAME} must be refused as the token embedding, got {other:?}"),
    }
}

#[test]
fn a_decoder_tensor_in_the_same_container_is_not_refused() {
    let src = source_with_pack();
    let container = container_with_addressable_embedding(&src);
    op_on(&container, TENSOR, 33)
        .resolve(&container)
        .expect("only the embedding is refused");
}

#[test]
fn the_embedding_under_its_stock_name_is_ambiguous_with_the_final_norm_and_is_refused_too() {
    let src = source_with_pack();
    let err = StructuredWeightOp::bind(
        &src.container,
        ORIGINAL_NAME,
        PROFILE,
        &unit_row(1, 0),
        ScalarBits::F64(ALPHA_NEAR_ONE),
        vec![0],
    )
    .unwrap_err();
    assert!(
        matches!(err, WeightPatchError::AmbiguousTensor { .. }),
        "{err:?}"
    );
}
