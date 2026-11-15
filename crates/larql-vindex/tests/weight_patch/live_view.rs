//! The live view's bytes: what a session executes for its pending stored-weight ops is, byte for byte, what
//! `COMPILE CURRENT INTO VINDEX` stores for them.

use larql_vindex::format::vindex3::knowledge::KnowledgeOverlay;
use larql_vindex::format::vindex3::weight_patch::structured::{OperandBits, ScalarBits};
use larql_vindex::format::vindex3::weight_patch::{
    apply_declared_patches, declare_patches, materialise_declared_patches, read_bf16_tensor,
    DeclaredPatch, LowRankColumnsOp, ResultEdit, ResultPatch, StructuredWeightOp, WeightEdit,
    WeightOp, WeightPatch, WeightPatchError,
};
use sha2::{Digest, Sha256};

use crate::fixture::*;
use crate::live_support::*;

/// A second tensor of the same canonical representation.
const OTHER_TENSOR: &str = "0.mlp.down_proj.weight";
/// A directory a container does not hold.
const STRAY_DIRECTORY: &str = "notes";

fn sha256(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[test]
fn every_kind_of_op_materialises_to_the_bytes_compile_stores() {
    for kind in KINDS {
        let src = source_with_pack();
        let bound = bind(&src.container);
        let staged = op(&src.container, kind, &COLUMNS_A);
        let compiled = compile(&src, &[staged.as_ref()], "compiled.vindex3");
        let overlay = session_with(&bound, vec![staged]);

        let view = overlay.live_weights().unwrap().expect("an op is in effect");
        assert_eq!(view.tensors().len(), 1, "{kind:?}");
        let live = &view.tensors()[0];
        let stored = tensor_bytes(&compiled, CANONICAL, TENSOR);
        assert_eq!(
            &live.bytes[..],
            &stored[..],
            "{kind:?}: the live bytes are the compiled container's"
        );
        assert_eq!(live.sha256, sha256(&stored), "{kind:?}");
        assert_eq!(
            (live.object.as_str(), live.tensor.as_str()),
            (DECODER, TENSOR)
        );
        assert_eq!(live.shape, [TENSOR_ROWS, TENSOR_COLS]);
        assert_ne!(
            &live.bytes[..],
            &tensor_bytes(&src.container, CANONICAL, TENSOR)[..],
            "{kind:?}: and they are not the source's"
        );
    }
}

#[test]
fn several_tensors_materialise_together_each_to_its_compiled_bytes() {
    let src = source_with_pack();
    let edit = |tensor: &str, col: usize| {
        DeclaredPatch::Deltas(WeightPatch {
            tensor: tensor.to_string(),
            edits: vec![WeightEdit {
                row: EDIT_ROW,
                col,
                delta: EDIT_DELTA,
            }],
        })
    };
    let patches = [edit(TENSOR, 8), edit(OTHER_TENSOR, 9)];
    let live = materialise_declared_patches(&src.container, &patches).unwrap();
    let out = src.out("two.vindex3");
    apply_declared_patches(&src.container, &patches, &[], &out).unwrap();
    assert_eq!(live.len(), 2);
    for (tensor, name) in live.iter().zip([TENSOR, OTHER_TENSOR]) {
        assert_eq!(tensor.tensor, name);
        assert_eq!(tensor.representation, CANONICAL);
        assert_eq!(tensor.object, DECODER);
        assert_eq!(tensor.bytes, tensor_bytes(&out, CANONICAL, name));
        assert_eq!(tensor.sha256(), sha256(&tensor.bytes));
    }
}

/// The view is built when the op set changes and shared by every execution after: each execution's overrides
/// hold the view's bytes, not a copy and not a fresh materialisation.
#[test]
fn every_execution_shares_the_bytes_materialised_once() {
    /// Holders of the bytes with the view alone: the tensor it lists and its own replacements.
    const HOLDERS_OF_A_VIEW: usize = 2;
    let src = source_with_pack();
    let bound = bind(&src.container);
    let overlay = session_with(&bound, vec![op(&src.container, Kind::Delta, &COLUMNS_A)]);
    let bytes = &overlay.live_weights().unwrap().unwrap().tensors()[0].bytes;
    assert_eq!(std::sync::Arc::strong_count(bytes), HOLDERS_OF_A_VIEW);
    let (first, second) = (
        overlay.live_weight_overrides().unwrap().unwrap(),
        overlay.live_weight_overrides().unwrap().unwrap(),
    );
    assert_eq!(std::sync::Arc::strong_count(bytes), HOLDERS_OF_A_VIEW + 2);
    assert_ne!(
        first.version(),
        second.version(),
        "each execution's set is its own"
    );

    let shown = format!("{:?}", overlay.live_weights().unwrap().unwrap());
    assert!(
        shown.contains(TENSOR) && shown.len() < bytes.len(),
        "a view is shown by its tensors' names and digests, not their bytes"
    );
}

#[test]
fn materialising_writes_nothing() {
    let src = source_with_pack();
    let before = tree_digest(src.dir.path());
    let bound = bind(&src.container);
    session_with(&bound, vec![op(&src.container, Kind::LowRank, &COLUMNS_A)]);
    assert_eq!(tree_digest(src.dir.path()), before);
}

/// What compile refuses before it writes, the pure path refuses with the same error.
#[test]
fn materialising_refuses_what_compile_refuses_before_it_writes() {
    let src = source_with_pack();
    let one = |tensor: &str, row: usize, col: usize| {
        DeclaredPatch::Deltas(WeightPatch {
            tensor: tensor.to_string(),
            edits: vec![WeightEdit {
                row,
                col,
                delta: EDIT_DELTA,
            }],
        })
    };
    let compiled = compile(
        &src,
        &[op(&src.container, Kind::Delta, &COLUMNS_A).as_ref()],
        "already.vindex3",
    );
    let stray = src.out("stray.vindex3");
    copy_tree(&src.container, &stray);
    std::fs::create_dir(stray.join(STRAY_DIRECTORY)).unwrap();
    let no_result = DeclaredPatch::Results(ResultPatch {
        tensor: TENSOR.to_string(),
        edits: Vec::<ResultEdit>::new(),
    });

    let cases: Vec<(&str, std::path::PathBuf, Vec<DeclaredPatch>)> = vec![
        ("no patch", src.container.clone(), vec![]),
        ("a patch of nothing", src.container.clone(), vec![no_result]),
        (
            "one tensor twice",
            src.container.clone(),
            vec![one(TENSOR, 0, 0), one(TENSOR, 0, 1)],
        ),
        (
            "an unknown tensor",
            src.container.clone(),
            vec![one("no.such.weight", 0, 0)],
        ),
        (
            "an element out of bounds",
            src.container.clone(),
            vec![one(TENSOR, TENSOR_ROWS, 0)],
        ),
        (
            "a vector",
            src.container.clone(),
            vec![one("1.input_layernorm.weight", 0, 0)],
        ),
        (
            "a source already patched",
            compiled,
            vec![one(TENSOR, 0, 0)],
        ),
        (
            "a source holding a stray directory",
            stray,
            vec![one(TENSOR, 0, 0)],
        ),
    ];
    for (what, source, patches) in cases {
        let live = materialise_declared_patches(&source, &patches).unwrap_err();
        let out = src.out("refused.vindex3");
        let written = apply_declared_patches(&source, &patches, &[], &out).unwrap_err();
        assert_eq!(live.to_string(), written.to_string(), "{what}");
        assert!(!out.exists(), "{what}: a refused compile leaves nothing");
    }
}

#[test]
fn a_stray_directory_is_refused_by_name() {
    let src = source_with_pack();
    let stray = src.out("stray.vindex3");
    copy_tree(&src.container, &stray);
    std::fs::create_dir(stray.join(STRAY_DIRECTORY)).unwrap();
    let patch = DeclaredPatch::Deltas(WeightPatch {
        tensor: TENSOR.to_string(),
        edits: vec![WeightEdit {
            row: 0,
            col: 0,
            delta: EDIT_DELTA,
        }],
    });
    assert!(matches!(
        materialise_declared_patches(&stray, &[patch]),
        Err(WeightPatchError::UnexpectedEntry(name)) if name == STRAY_DIRECTORY
    ));
}

/// The sham control. A declared sham (every term has a zero factor) is accepted, as compile accepts it, and
/// executes the base's bytes; an op that is not zero-valued and changes nothing is refused, as compile refuses
/// it, and the session then refuses to execute at all rather than run the stored weights.
#[test]
fn a_declared_sham_holds_the_bases_bytes_and_an_op_that_changes_nothing_stays_refused() {
    let src = source_with_pack();
    let bound = bind(&src.container);
    let ([rows, _], _) = read_bf16_tensor(&src.container, TENSOR).unwrap();

    let sham = LowRankColumnsOp::bind(
        &src.container,
        TENSOR,
        PROFILE,
        1,
        &OperandBits::F64(vec![0; rows]),
        &OperandBits::F64(vec![F64_ONE; COLUMNS_A.len()]),
        COLUMNS_A.to_vec(),
    )
    .unwrap();
    let overlay = session_with(&bound, vec![WeightOp::LowRankColumns(sham)]);
    let view = overlay.live_weights().unwrap().unwrap();
    assert_eq!(
        &view.tensors()[0].bytes[..],
        &tensor_bytes(&src.container, CANONICAL, TENSOR)[..]
    );
    assert_eq!(bound.compute_session(&overlay), bound.compute(None));

    // Scale zero: every element rounds back to itself. Not zero-valued by declaration, so not a sham.
    let mut u = vec![0u64; rows];
    u[0] = F64_ONE;
    let nothing = StructuredWeightOp::bind(
        &src.container,
        TENSOR,
        PROFILE,
        &OperandBits::F64(u),
        ScalarBits::F64(0),
        COLUMNS_A.to_vec(),
    )
    .unwrap();
    assert!(matches!(
        declare_patches(
            &src.container,
            &[WeightOp::Structured(nothing.clone()).as_ref()]
        ),
        Err(WeightPatchError::Empty)
    ));
    let mut overlay = KnowledgeOverlay::new();
    overlay.stage_structured_op(nothing).unwrap();
    let refused = overlay.refresh_live_weights(&bound.binding()).unwrap_err();
    assert!(
        refused.to_string().contains("the patch edits no elements"),
        "{refused}"
    );
    assert!(
        overlay.live_weight_overrides().is_err(),
        "with an op pending the session never executes the stored weights"
    );
}

fn copy_tree(from: &std::path::Path, to: &std::path::Path) {
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
