//! **Fine-grained FP8, end to end through a real container.**
//!
//! The codec's numbers are settled elsewhere: `fp8_finegrained`'s unit
//! tests pin the tiling, and `scripts/glm_fp8_dequant_gate.py` shows the
//! decode is bit-identical to the upstream reference over 125,829,120
//! real GLM values. What neither can see is everything between a
//! checkpoint and a kernel — whether the encoder places the
//! `weight_scale_inv` sibling at all, whether the loader finds it,
//! whether it binds the RIGHT one, and whether the tile survives.
//!
//! ```text
//! checkpoint (F8_E4M3 + F32 grid)
//!   -> encode_system (declares `scales`) -> OperandStore
//!   -> select: cpu:direct/FusedFp8Block, grid RETAINED
//!   -> load_weight(Fp8Block) -> WeightSlice -> WeightRows -> FusedFp8Block
//!                                                          the candidate
//!
//! the SAME source bytes -> fp8_finegrained::dequantize -> scalar dot
//!                                                        the authority
//! ```
//!
//! The authority reads the CHECKPOINT's bytes, not the container's, so
//! the two arms share nothing downstream of the source file. A carriage
//! defect — a dropped grid, a transposed grid, a tile read off the
//! config — moves one and not the other.
//!
//! Since the format became a codec (`F8_E4M3`, family `fp8-block`), the
//! grid is a DECLARED dependency the encoder writes into the reference
//! table, the canonical decode is the registry's, and the kernel is
//! reached by selection rather than by a caller that knew the answer.
//! The tests below hold each of those.

use crate::format::vindex3::auxiliary_references::OperandAddress;
use crate::format::vindex3::encode::segment::read_segment_header;
use crate::format::vindex3::fixtures::{dense_fp8_model, encode_fixture_container};
use crate::format::vindex3::index::Vindex3Index;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::{PlanBackend, WeightFormat, WeightSlice};
use crate::format::vindex3::opplan::exec::cpu::kernels::FusedFp8Block;
use crate::format::vindex3::opplan::exec::cpu::physical::PhysicalProjectionPlan;
use crate::format::vindex3::opplan::exec::cpu::projector::DenseProjector;
use crate::format::vindex3::opplan::exec::execute_plan;
use crate::format::vindex3::opplan::exec::operands::{
    OperandSource, OperandStore, RepresentationSource,
};
use crate::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::realization::{
    DependencyLifetime, RealizationForm, RealizationId, SelectionReason,
};
use crate::format::vindex3::opplan::exec::reference::ReferenceBackend;
use crate::format::vindex3::opplan::exec::weights::{load_weight, LoadedWeight};
use crate::format::vindex3::opplan::{plan_component_ops, OperandRef};
use crate::format::vindex3::represent::codec::codecs::fp8_block::{DTYPE_FP8_BLOCK, SCALES};
use larql_models::quant::fp8_finegrained::{dequantize, Fp8Grid};

const TARGET: &str = "0.mlp.gate_proj.weight";
const COMPONENT: &str = "target";
/// A prompt over the dense fixture's 128-token vocabulary.
const TOKENS: [u32; 5] = [3, 17, 28, 0, 11];

/// The container, and the checkpoint it came from.
fn encoded(tmp: &tempfile::TempDir) -> (std::path::PathBuf, std::path::PathBuf) {
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    let container = tmp.path().join("out.vindex3");
    encode_fixture_container(dense_fp8_model, &checkpoint, &container, "target");
    (checkpoint, container)
}

/// The FP8 tensor in the container, as an operand — found by DTYPE, so
/// the test cannot pass by binding something that merely has the name.
fn fp8_operand(container: &std::path::Path) -> OperandRef {
    let index: Vindex3Index = serde_json::from_str(
        &std::fs::read_to_string(container.join(crate::format::filenames::INDEX_JSON)).unwrap(),
    )
    .unwrap();
    for entry in index.representations.values() {
        let Ok((header, _)) = read_segment_header(&container.join(&entry.segment)) else {
            continue;
        };
        if let Some(t) = header.tensors.iter().find(|t| t.dtype == DTYPE_FP8_BLOCK) {
            return OperandRef {
                object: entry.object.clone(),
                tensor: t.name.clone(),
                dtype: t.dtype.clone(),
                shape: t.shape.clone(),
            };
        }
    }
    panic!("no F8_E4M3 tensor survived the encode — the codes were not carried");
}

fn open(container: &std::path::Path) -> OperandStore {
    let inspection = inspect_container(container, false).unwrap();
    OperandStore::open_for(container, &inspection, None, RepresentationSource::Stored).unwrap()
}

/// The authority: the CHECKPOINT's own bytes, dequantised and dotted.
fn from_the_checkpoint(checkpoint: &std::path::Path, x: &[f32]) -> Vec<f32> {
    let (grid, w) = dequantised_from_the_checkpoint(checkpoint);
    (0..grid.rows)
        .map(|r| {
            w[r * grid.cols..(r + 1) * grid.cols]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect()
}

/// The CHECKPOINT's own bytes, dequantised by the reference and nothing
/// of this crate's — the values every container-side reading must equal.
fn dequantised_from_the_checkpoint(checkpoint: &std::path::Path) -> (Fp8Grid, Vec<f32>) {
    let raw = std::fs::read(checkpoint.join("model-fp8.safetensors")).unwrap();
    let hlen = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&raw[8..8 + hlen]).unwrap();
    let body = &raw[8 + hlen..];
    let get = |name: &str| -> (Vec<usize>, &[u8]) {
        let e = &header[name];
        let off = e["data_offsets"].as_array().unwrap();
        let (a, b) = (
            off[0].as_u64().unwrap() as usize,
            off[1].as_u64().unwrap() as usize,
        );
        let shape = e["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        (shape, &body[a..b])
    };
    let (wshape, codes) = get("model.layers.0.mlp.gate_proj.weight");
    let (sshape, sbytes) = get("model.layers.0.mlp.gate_proj.weight_scale_inv");
    let scales: Vec<f32> = sbytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let grid = Fp8Grid {
        rows: wshape[0],
        cols: wshape[1],
        scale_rows: sshape[0],
        scale_cols: sshape[1],
    };
    let w = dequantize(codes, &scales, grid).unwrap();
    (grid, w)
}

/// The container DECLARES the grid as the codes' dependency: the encoder
/// consumed the checkpoint's naming convention once and wrote a reference,
/// and nothing downstream spells `weight_scale_inv`.
#[test]
fn the_encoder_declares_the_scale_grid_as_the_codes_dependency() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, container) = encoded(&tmp);
    let operand = fp8_operand(&container);
    assert!(
        operand.tensor.ends_with(TARGET),
        "found `{}`, expected the FP8 gate projection",
        operand.tensor
    );

    let store = open(&container);
    let owner = OperandAddress::new(&operand.object, &operand.tensor);
    let grid = store
        .references()
        .target(&owner, SCALES)
        .expect("the encoder declared the grid under the codec's name")
        .clone();
    assert_eq!(
        grid.object, operand.object,
        "the grid is in the codes' object"
    );
    assert!(
        grid.tensor.ends_with("weight_scale_inv"),
        "the reference addresses the checkpoint's grid, found `{}`",
        grid.tensor
    );
    let shape = store
        .stored_shape(&grid)
        .expect("the container holds the grid");
    assert_eq!(shape.len(), 2, "the grid is two-dimensional");
    let raw = store
        .load_raw(&OperandRef {
            object: grid.object.clone(),
            tensor: grid.tensor.clone(),
            dtype: String::new(),
            shape: shape.clone(),
        })
        .unwrap();
    assert_eq!(raw.dtype, "F32");
    assert_eq!(
        raw.bytes.len(),
        shape.iter().product::<usize>() * 4,
        "the grid's bytes match its declared shape"
    );
    // And the only dependency: the codec requires one, the table declares one.
    assert_eq!(store.references().auxiliaries_of(&owner).len(), 1);
}

/// The registry's canonical decode of the container's bytes IS the
/// reference dequantisation of the checkpoint's — bit for bit.
#[test]
fn the_canonical_decode_is_the_reference_dequantisation_bit_for_bit() {
    let tmp = tempfile::tempdir().unwrap();
    let (checkpoint, container) = encoded(&tmp);
    let operand = fp8_operand(&container);
    let store = open(&container);
    let decoded = store
        .load(&operand)
        .expect("the codec decodes through its dependency");
    let (grid, want) = dequantised_from_the_checkpoint(&checkpoint);
    assert_eq!(decoded.len(), grid.elements());
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    assert_eq!(bits(&decoded), bits(&want));
}

fn with_prepared<B: PlanBackend>(
    container: &std::path::Path,
    backend: &B,
    check: impl FnOnce(&PreparedOperands),
) {
    let inspection = inspect_container(container, false).unwrap();
    let plan = plan_component_ops(&inspection, container, COMPONENT)
        .unwrap()
        .plan
        .expect("the FP8 fixture plans");
    let store = open(container);
    let prepared = PreparedOperands::load(&plan, &store, backend, ExecutionSlice::Full)
        .expect("the FP8 operand has an admissible realization");
    check(&prepared);
}

/// Selection reaches the FP8 kernel from the codec's declaration — no
/// privileged caller — and the pin RETAINS the grid, which is the
/// dependency lifetime a direct kernel over codes has and the ledger
/// prices. The reference oracle decodes the same operand and is finished
/// with the grid once it has its image.
#[test]
fn selection_pins_the_fp8_kernel_and_retains_its_grid() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, container) = encoded(&tmp);
    let operand = fp8_operand(&container);
    let record_for = |prepared: &PreparedOperands| {
        prepared
            .realizations()
            .iter()
            .find(|r| r.planned.operand.tensor == operand.tensor)
            .cloned()
            .expect("the FP8 projection is planned")
    };

    with_prepared(&container, &ProductionBackend::new(), |prepared| {
        let record = record_for(prepared);
        assert_eq!(record.representation, DTYPE_FP8_BLOCK);
        assert_eq!(
            record.provider.as_ref().map(|p| p.family.as_str()),
            Some("fp8-block")
        );
        assert_eq!(
            record.selection.realization,
            RealizationId::cpu(RealizationForm::Direct(
                PhysicalProjectionPlan::FusedFp8Block
            ))
        );
        assert_eq!(record.selection.reason, SelectionReason::DirectDeclared);
        assert_eq!(record.dependencies.len(), 1, "one dependency: the grid");
        let grid = &record.dependencies[0];
        assert_eq!(grid.name, SCALES);
        assert_eq!(grid.lifetime, DependencyLifetime::Retained);
        assert_eq!(grid.label, "F32");
        assert!(grid.elements > 1, "the grid is priced by its elements");
    });

    with_prepared(&container, &ReferenceBackend::new(), |prepared| {
        let record = record_for(prepared);
        assert_eq!(
            record.selection.realization,
            RealizationId::cpu(RealizationForm::Decode(PhysicalProjectionPlan::ScalarF32))
        );
        assert_eq!(record.selection.reason, SelectionReason::ReferenceOracle);
        assert_eq!(
            record.dependencies[0].lifetime,
            DependencyLifetime::PreparationOnly,
            "a decode is finished with the grid once it has an f32 image"
        );
    });
}

/// The whole forward through selection: the production backend runs the
/// FP8 kernel over the stored codes, the reference decodes them, and the
/// two agree on the same container — which neither could execute at all
/// before the format was a codec.
#[test]
fn production_and_reference_agree_on_the_fp8_container_through_selection() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, container) = encoded(&tmp);
    let inspection = inspect_container(&container, false).unwrap();
    let plan = plan_component_ops(&inspection, &container, COMPONENT)
        .unwrap()
        .plan
        .unwrap();
    let store = open(&container);
    let logits = |backend: &dyn PlanBackend| {
        execute_plan(&plan, &store, &TOKENS, backend)
            .expect("executes")
            .logits
            .expect("the dense fixture carries a head")
    };
    let production = logits(&ProductionBackend::new());
    let reference = logits(&ReferenceBackend::new());
    assert_eq!(production.len(), reference.len());
    let scale = reference.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = production
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst <= scale * 1e-4,
        "production and reference disagree on the FP8 container: worst {worst:e} of {scale:e}"
    );
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
    };
    assert_eq!(argmax(&production), argmax(&reference));
}

/// The whole path, against an authority that shares nothing with it below
/// the source file.
#[test]
fn a_container_bound_fp8_projection_agrees_with_the_source_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    let (checkpoint, container) = encoded(&tmp);
    let operand = fp8_operand(&container);
    let store = open(&container);

    let loaded = load_weight(
        OperandSource::from(&store),
        &operand,
        WeightFormat::Fp8Block,
    )
    .expect("a container-carried FP8 pair binds");

    // The bytes are the checkpoint's: nothing was widened on the way in.
    assert!(
        !loaded.is_widened_f32(),
        "FP8 must not reach residency as a widened f32 image"
    );
    let (rows, cols) = (operand.shape[0], operand.shape[1]);
    assert!(
        matches!(loaded, LoadedWeight::Fp8Block { .. }),
        "the loader produced another representation"
    );
    assert_eq!(
        loaded.resident_bytes(),
        rows * cols + {
            let LoadedWeight::Fp8Block { scales, .. } = &loaded else {
                unreachable!()
            };
            scales.len() * 4
        },
        "a resident FP8 operand costs its codes plus its scales"
    );

    let slice = loaded.slice();
    assert!(matches!(slice, WeightSlice::Fp8Block { .. }));
    let rows_view = slice.rows(rows, cols).expect("geometry closes");

    let x: Vec<f32> = (0..cols)
        .map(|i| ((i as f32) * 0.031).cos() * 0.7)
        .collect();
    let mut got = vec![0.0f32; rows];
    FusedFp8Block.project_rows(rows_view, &x, &mut got);

    let want = from_the_checkpoint(&checkpoint, &x);
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
        .fold(0.0f32, f32::max);
    assert!(
        worst <= 1e-6,
        "container-bound FP8 disagrees with the source bytes: worst {worst:e}"
    );
}

/// **The control.** A projection bound against ANOTHER matrix's scale
/// grid must not agree — otherwise the test above would pass for a
/// loader that found any grid at all.
#[test]
fn the_wrong_scale_grid_does_not_agree() {
    let tmp = tempfile::tempdir().unwrap();
    let (checkpoint, container) = encoded(&tmp);
    let operand = fp8_operand(&container);
    let store = open(&container);
    let loaded = load_weight(
        OperandSource::from(&store),
        &operand,
        WeightFormat::Fp8Block,
    )
    .unwrap();
    let LoadedWeight::Fp8Block {
        codes,
        scales,
        block_rows,
        block_cols,
        scale_cols,
    } = &loaded
    else {
        unreachable!()
    };

    let (rows, cols) = (operand.shape[0], operand.shape[1]);
    let x: Vec<f32> = (0..cols)
        .map(|i| ((i as f32) * 0.031).cos() * 0.7)
        .collect();
    let want = from_the_checkpoint(&checkpoint, &x);

    // Same codes, grid rotated by one entry.
    let mut wrong = scales.clone();
    wrong.rotate_left(1);
    let view = crate::format::vindex3::opplan::exec::cpu::projector::WeightRows::Fp8Block {
        codes: &codes.as_slice()[..rows * cols],
        scales: &wrong,
        block_rows: *block_rows,
        block_cols: *block_cols,
        scale_cols: *scale_cols,
        row_in_tile: 0,
    };
    let mut got = vec![0.0f32; rows];
    FusedFp8Block.project_rows(view, &x, &mut got);
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs() / b.abs().max(1.0))
        .fold(0.0f32, f32::max);
    assert!(
        worst > 1e-3,
        "a rotated scale grid produced the same answer ({worst:e}) — the scales \
         are not reaching the arithmetic"
    );
}

/// **The boundary of the claim, asserted rather than described.**
///
/// This build reproduces fine-grained FP8 *storage* exactly and
/// implements none of its *compute* path. A checkpoint that declares
/// `activation_scheme` is asking for an FP8 GEMM against a run-time
/// quantised activation; this build dequantises weights and runs f32,
/// which is numerically close and not the same route.
///
/// So the same fixture must encode WITHOUT that key and refuse WITH it.
/// Without this test, "named and refused" is a comment; with it, the
/// refusal is a property.
#[test]
fn declaring_an_fp8_activation_scheme_is_refused() {
    let tmp = tempfile::tempdir().unwrap();
    let checkpoint = tmp.path().join("ckpt");
    std::fs::create_dir_all(&checkpoint).unwrap();
    dense_fp8_model(&checkpoint);

    // The storage-only fixture is admissible — the control that makes the
    // refusal below attributable to this key and nothing else.
    let ok_dir = tmp.path().join("ok.vindex3");
    let inventory = larql_models::inventory::build_inventory(&checkpoint).unwrap();
    crate::format::vindex3::encode::encode_system(&[("target".to_string(), inventory)], &ok_dir)
        .expect("FP8 storage alone is admissible");

    let cfg_path = checkpoint.join("config.json");
    let mut cfg: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
    cfg["quantization_config"]["activation_scheme"] = serde_json::json!("dynamic");
    std::fs::write(&cfg_path, cfg.to_string()).unwrap();

    let bad_dir = tmp.path().join("bad.vindex3");
    let inventory = larql_models::inventory::build_inventory(&checkpoint).unwrap();
    let err = crate::format::vindex3::encode::encode_system(
        &[("target".to_string(), inventory)],
        &bad_dir,
    )
    .expect_err("an FP8 compute path this build does not implement must not encode");
    let msg = err.to_string();
    assert!(
        msg.contains("inadmissible"),
        "the refusal should name the plan, got: {msg}"
    );
}

/// A container encoded before the encoder declared dependencies holds the
/// FP8 pair and no table. It is refused LOUDLY — the grid is an
/// unclassified operand, so the plan does not close — and it migrates
/// with the encoder's own rule applied late, after which it plans and
/// executes exactly as a fresh encode does. The rule is never applied by
/// a reader: a container that declares nothing depends on nothing.
#[test]
fn a_container_encoded_before_dependencies_were_declared_refuses_and_then_migrates() {
    use crate::format::filenames::{AUXILIARY_REFERENCES_JSON, INDEX_JSON};
    use crate::format::vindex3::encode::declare_references;

    let tmp = tempfile::tempdir().unwrap();
    let (_, container) = encoded(&tmp);
    let operand = fp8_operand(&container);

    // Undeclare: drop the table and the index's name for it, which is
    // exactly what an older encoder left behind.
    let index_path = container.join(INDEX_JSON);
    let mut index: Vindex3Index =
        serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    assert!(
        index.auxiliary_references.is_some(),
        "a fresh encode declares"
    );
    index.auxiliary_references = None;
    std::fs::write(&index_path, serde_json::to_string_pretty(&index).unwrap()).unwrap();
    std::fs::remove_file(container.join(AUXILIARY_REFERENCES_JSON)).unwrap();

    let inspection = inspect_container(&container, false).unwrap();
    let outcome = plan_component_ops(&inspection, &container, COMPONENT).unwrap();
    assert!(
        outcome.plan.is_none() && !outcome.defects.is_empty(),
        "an undeclared grid must not plan: {:?}",
        outcome.defects
    );
    let listed: Vec<String> = outcome.defects.iter().map(|d| d.to_string()).collect();
    assert!(
        listed.iter().any(|d| d.contains("weight_scale_inv")),
        "the defect names the orphaned grid: {listed:?}"
    );
    // And the decode has no dependency to resolve through.
    let store = open(&container);
    let err = store.load(&operand).unwrap_err().to_string();
    assert!(err.contains(SCALES), "{err}");

    // Migrate: the encoder's rule, applied to the headers on disk.
    assert_eq!(declare_references(&container).unwrap(), 1);
    let again = declare_references(&container).unwrap_err().to_string();
    assert!(again.contains("already declares"), "{again}");

    let inspection = inspect_container(&container, false).unwrap();
    assert!(inspection.index.auxiliary_references.is_some());
    let outcome = plan_component_ops(&inspection, &container, COMPONENT).unwrap();
    assert!(outcome.plan.is_some(), "{:?}", outcome.defects);
    let store = open(&container);
    let (_, want) = dequantised_from_the_checkpoint(&tmp.path().join("ckpt"));
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    assert_eq!(bits(&store.load(&operand).unwrap()), bits(&want));
}
