//! Golden: the Rust implementation against the independent Python prototype
//! on the real Qwen3-0.6B container. Ignored by default (needs the 1.3 GB
//! container); run with
//!
//!   WSP_GOLDEN_SOURCE=<shipped container> WSP_GOLDEN_EXPECT=<prototype's edited container> \
//!   cargo test -p larql-vindex --test weight_patch golden -- --ignored --nocapture
//!
//! The prototype shares no code with this implementation. Agreement on the
//! segment digests means two separately written patchers produced the same
//! bytes from the same declared edit.

use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::weight_patch::{apply_weight_patch, WeightEdit, WeightPatch};

use crate::fixture::{index_of, CANONICAL, DERIVED};

/// Stage 1's manifest: seven elements of one Q8_0 block, +2^-7, the block's
/// maximum (column 205) untouched.
const TENSOR: &str = "26.mlp.down_proj.weight";
const ROW: usize = 100;
const COLS: [usize; 7] = [200, 201, 202, 203, 204, 206, 207];
const DELTA: f32 = 0.0078125;

#[test]
#[ignore]
fn matches_the_prototype_byte_for_byte_on_the_real_container() {
    let (Ok(source), Ok(expect)) = (
        std::env::var("WSP_GOLDEN_SOURCE"),
        std::env::var("WSP_GOLDEN_EXPECT"),
    ) else {
        panic!("set WSP_GOLDEN_SOURCE and WSP_GOLDEN_EXPECT");
    };
    let scratch = tempfile::tempdir().unwrap();
    let out = scratch.path().join("edited.vindex3");
    let patch = WeightPatch {
        tensor: TENSOR.to_string(),
        edits: COLS
            .iter()
            .map(|&col| WeightEdit {
                row: ROW,
                col,
                delta: DELTA,
            })
            .collect(),
    };
    let started = std::time::Instant::now();
    let report = apply_weight_patch(std::path::Path::new(&source), &patch, &out).unwrap();
    println!(
        "patched in {:.1?}; {} elements; re-derived {:?}",
        started.elapsed(),
        report.elements,
        report.rederived
    );

    assert!(inspect_container(&out, true).unwrap().is_coherent());
    let (mine, theirs) = (index_of(&out), index_of(std::path::Path::new(&expect)));
    for id in [CANONICAL, DERIVED] {
        let (a, b) = (&mine.representations[id], &theirs.representations[id]);
        println!(
            "{id}: payload {} segment {}",
            &a.payload_sha256[..16],
            &a.segment_sha256[..16]
        );
        assert_eq!(a.payload_sha256, b.payload_sha256, "{id} payload digest");
        assert_eq!(a.segment_sha256, b.segment_sha256, "{id} segment digest");
    }
    assert_eq!(mine.model, theirs.model);
    let shipped = index_of(std::path::Path::new(&source));
    assert_ne!(
        mine.representations[CANONICAL].payload_sha256,
        shipped.representations[CANONICAL].payload_sha256
    );
}
