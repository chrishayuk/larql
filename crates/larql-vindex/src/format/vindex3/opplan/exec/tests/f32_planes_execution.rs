//! A progressive operand read out of a real container: the streams an
//! extent needs are opened and the ones it does not are never touched.
//!
//! The claim a bit-plane codec makes is not about arithmetic — its unit
//! tests settle that — but about I/O: *decoding at depth d does not read
//! the planes above d*. That is a statement about files, so it is checked
//! against files, two ways. The store counts every payload read, so a
//! shallow decode's count is one where a terminal decode's is three; and
//! the same decode succeeds against a container that does not CONTAIN the
//! deeper plane, which no accounting can fake.
//!
//! The container itself is built the way rung 2's witness builds its
//! candidate: one fixture checkpoint, one segment rewritten, the index
//! re-recorded. The refinements are sibling tensors named as the loader's
//! convention names them — the operand's tensor plus the stream's declared
//! name — so nothing here teaches the loader what a plane is.

use std::path::Path;

use super::super::accounting::{
    expectations, BlockGeometry, RepresentationFloor, ResidencyBudget, ResourceLedger,
};
use super::super::operands::OperandStore;
use super::super::prepared::{select_realizations_within, ExecutionSlice};
use super::super::production::ProductionBackend;
use super::super::realization::RealizationRecord;
use crate::format::filenames::INDEX_JSON;
use crate::format::vindex3::encode::segment::{read_segment_header, write_segment, PlannedTensor};
use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::index::Vindex3Index;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, OperandRef};
use crate::format::vindex3::represent::codec::codecs::f32_planes::{
    F32PlanesCodec, DTYPE_F32_PLANES, REFINE_8A, REFINE_8B, TERMINAL_DEPTH,
};
use crate::format::vindex3::represent::codec::RepresentationExtent;

/// The tensors this witness stores as planes: the FFN projections, which
/// the dense fixture holds as f32 and the plan addresses by name.
const PLANE_MARK: &str = "mlp.";
/// The dtype a refinement plane is stored under: bytes, whose meaning is
/// the codec's declaration and not the container's.
const REFINEMENT_DTYPE: &str = "U8";

/// One tensor now stored as planes, as the plan will address it.
struct Planed {
    object: String,
    tensor: String,
    shape: Vec<usize>,
    /// The values the planes were cut from — the foreign reference every
    /// extent is measured against.
    source: Vec<f32>,
}

impl Planed {
    fn operand(&self) -> OperandRef {
        OperandRef {
            object: self.object.clone(),
            tensor: self.tensor.clone(),
            dtype: String::new(),
            shape: self.shape.clone(),
        }
    }
}

/// How much of a progressive tensor the container keeps.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Keep {
    /// Every plane: the artifact is complete.
    EveryPlane,
    /// The base and the first refinement only — an artifact that was
    /// truncated, so a terminal decode has nothing to read.
    ThroughFirstRefinement,
}

/// Rewrite the container's FFN projections into planes, keeping `keep` of
/// them, and re-record each segment in `index.json`.
fn store_as_planes(root: &Path, keep: Keep) -> Vec<Planed> {
    let index_path = root.join(INDEX_JSON);
    let mut index: Vindex3Index =
        serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    let mut out = Vec::new();
    for entry in index.representations.values_mut() {
        let path = root.join(&entry.segment);
        let (header, payload_start) = read_segment_header(&path).unwrap();
        if !header.tensors.iter().any(|t| t.name.contains(PLANE_MARK)) {
            continue;
        }
        let file = std::fs::read(&path).unwrap();
        let payload = &file[payload_start as usize..];
        let mut planned = Vec::new();
        let mut bytes_by_name = std::collections::BTreeMap::new();
        for t in &header.tensors {
            let stored = &payload[t.offset as usize..(t.offset + t.len) as usize];
            if !t.name.contains(PLANE_MARK) {
                planned.push(PlannedTensor {
                    relative_name: t.name.clone(),
                    source_name: t.name.clone(),
                    dtype: t.dtype.clone(),
                    shape: t.shape.clone(),
                    len: stored.len() as u64,
                });
                bytes_by_name.insert(t.name.clone(), stored.to_vec());
                continue;
            }
            let values: Vec<f32> = stored
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let (base, refine_a, refine_b) = F32PlanesCodec::encode_planes(&values);
            let mut planes = vec![
                (t.name.clone(), DTYPE_F32_PLANES, t.shape.clone(), base),
                (
                    OperandStore::sibling_stream_tensor(&t.name, REFINE_8A.name),
                    REFINEMENT_DTYPE,
                    t.shape.clone(),
                    refine_a,
                ),
                (
                    OperandStore::sibling_stream_tensor(&t.name, REFINE_8B.name),
                    REFINEMENT_DTYPE,
                    t.shape.clone(),
                    refine_b,
                ),
            ];
            if keep == Keep::ThroughFirstRefinement {
                planes.pop();
            }
            for (name, dtype, shape, bytes) in planes {
                planned.push(PlannedTensor {
                    relative_name: name.clone(),
                    source_name: name.clone(),
                    dtype: dtype.to_string(),
                    shape,
                    len: bytes.len() as u64,
                });
                bytes_by_name.insert(name, bytes);
            }
            out.push(Planed {
                object: entry.object.clone(),
                tensor: t.name.clone(),
                shape: t.shape.clone(),
                source: values,
            });
        }
        let written = write_segment(&path, &header.representation, planned, |name, w, hash| {
            let bytes = &bytes_by_name[name];
            std::io::Write::write_all(w, bytes).map_err(crate::error::VindexError::Io)?;
            hash(bytes);
            Ok(bytes.len() as u64)
        })
        .unwrap();
        entry.payload_bytes = written.payload_bytes;
        entry.payload_sha256 = written.payload_sha256;
        entry.segment_sha256 = written.segment_sha256;
        entry.tensor_count = written.tensor_count;
    }
    std::fs::write(&index_path, serde_json::to_string_pretty(&index).unwrap()).unwrap();
    assert!(!out.is_empty(), "the fixture holds FFN projections");
    out
}

/// A container whose FFN projections are stored as planes.
fn planed_container(keep: Keep) -> (tempfile::TempDir, Vec<Planed>) {
    let dir = tempfile::tempdir().unwrap();
    let checkpoint = dir.path().join("checkpoint");
    let container = dir.path().join("container");
    std::fs::create_dir_all(&checkpoint).unwrap();
    encode_fixture_container(dense_f32_model, &checkpoint, &container, "planes");
    let planed = store_as_planes(&container, keep);
    (dir, planed)
}

fn store_of(dir: &Path) -> OperandStore {
    let container = dir.join("container");
    let inspection = inspect_container(&container, false).expect("the rewritten container reads");
    OperandStore::open(&container, &inspection).expect("the rewritten container opens")
}

fn plan_and_store(dir: &Path) -> (ComponentOpPlan, OperandStore) {
    let container = dir.join("container");
    let inspection = inspect_container(&container, false).unwrap();
    let outcome = plan_component_ops(&inspection, &container, "target").unwrap();
    let plan = outcome
        .plan
        .unwrap_or_else(|| panic!("the planed fixture must plan: {:?}", outcome.defects));
    (plan, store_of(dir))
}

/// Select under `budget`, and price what was selected.
fn select(
    plan: &ComponentOpPlan,
    store: &OperandStore,
    budget: &ResidencyBudget,
) -> Result<(Vec<RealizationRecord>, ResourceLedger), String> {
    let records = select_realizations_within(
        plan,
        store.into(),
        &ProductionBackend::new(),
        &ExecutionSlice::Full,
        budget,
    )
    .map_err(|e| e.to_string())?;
    let priced = expectations(&records, |o| store.stored_len(o), BlockGeometry::executor());
    let ledger = ResourceLedger::aggregate(&priced);
    Ok((records, ledger))
}

/// The pinned depths of the progressive operands, tensor by tensor.
fn pinned_depths(records: &[RealizationRecord]) -> Vec<(String, u32)> {
    records
        .iter()
        .filter(|r| r.extent.is_progressive())
        .map(|r| (r.planned.operand.tensor.clone(), r.extent.selected.depth))
        .collect()
}

#[test]
fn a_depth_opens_its_own_planes_and_no_others() {
    let (dir, planed) = planed_container(Keep::EveryPlane);
    let store = store_of(dir.path());
    let operand = planed[0].operand();
    for (depth, expected_reads) in (0..=TERMINAL_DEPTH).zip(1u64..) {
        let before = store.load_count();
        let values = store
            .load_at(&operand, RepresentationExtent::at_depth(depth))
            .unwrap_or_else(|e| panic!("depth {depth}: {e}"));
        let reads = store.load_count() - before;
        assert_eq!(
            reads, expected_reads,
            "depth {depth} read {reads} payloads, not {expected_reads}"
        );
        assert_eq!(values.len(), planed[0].source.len());
    }
    // The default loader asks for everything, which for this operand is
    // three planes and not the one a terminal codec would read.
    let before = store.load_count();
    let whole = store.load(&operand).unwrap();
    assert_eq!(store.load_count() - before, u64::from(TERMINAL_DEPTH) + 1);
    // And everything is the source, bit for bit.
    for (source, out) in planed[0].source.iter().zip(&whole) {
        assert_eq!(source.to_bits(), out.to_bits());
    }
}

#[test]
fn a_shallow_extent_reads_a_container_that_does_not_hold_the_deeper_plane() {
    let (dir, planed) = planed_container(Keep::ThroughFirstRefinement);
    let store = store_of(dir.path());
    let operand = planed[0].operand();
    // Depths 0 and 1 need only the planes this container kept.
    for depth in 0..TERMINAL_DEPTH {
        let values = store
            .load_at(&operand, RepresentationExtent::at_depth(depth))
            .unwrap_or_else(|e| panic!("depth {depth}: {e}"));
        assert_eq!(values.len(), planed[0].source.len());
    }
    // The terminal extent needs a plane that is not there, and says which.
    let err = store
        .load_at(&operand, RepresentationExtent::at_depth(TERMINAL_DEPTH))
        .unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains(REFINE_8B.name) && message.contains("no tensor"),
        "{message}"
    );
}

/// The three arms the extent dimension exists for: the same plan, the same
/// artifact, three answers — and the middle one is a REFUSAL, because a
/// budget may not spend quality the caller did not offer.
#[test]
fn the_pin_takes_the_depth_the_budget_and_the_floor_leave_it() {
    let (dir, planed) = planed_container(Keep::EveryPlane);
    let (plan, store) = plan_and_store(dir.path());
    assert!(!planed.is_empty());

    // A — nothing pressing, exact reconstruction required: every
    // progressive pin is on the whole artifact.
    let generous = ResidencyBudget::UNBOUNDED;
    let (whole_records, whole) = select(&plan, &store, &generous).expect("nothing to refuse");
    let depths = pinned_depths(&whole_records);
    assert!(!depths.is_empty(), "the fixture has progressive operands");
    assert!(
        depths.iter().all(|(_, depth)| *depth == TERMINAL_DEPTH),
        "{depths:?}"
    );

    // B — the same plan under a preparation budget it cannot meet, with
    // the default floor: refused before any payload byte, naming the
    // deficit AND the requirement that made it irreducible.
    let tight = ResidencyBudget::UNBOUNDED.with_prepare_bytes(whole.read_to_prepare * 3 / 4);
    let refusal = select(&plan, &store, &tight).expect_err("the budget cannot be met exactly");
    assert!(
        refusal.contains("preparation opens") && refusal.contains("exact reconstruction"),
        "{refusal}"
    );

    // C — the same budget, with a floor that admits a shallower extent:
    // now the pin moves, and only as far as it had to.
    let relaxed = tight.with_fidelity(RepresentationFloor::RelativeRms(5e-3));
    let (shallow_records, shallow) = select(&plan, &store, &relaxed).expect("a floor with room");
    let moved = pinned_depths(&shallow_records);
    assert!(
        moved.iter().any(|(_, depth)| *depth < TERMINAL_DEPTH),
        "some pin took a shallower extent: {moved:?}"
    );
    assert!(
        shallow.read_to_prepare <= tight.prepare_bytes.unwrap(),
        "the preparation now fits: {} vs {}",
        shallow.read_to_prepare,
        tight.prepare_bytes.unwrap()
    );

    // What moved and what did not — the accounting half of the claim.
    assert_eq!(
        shallow.stored, whole.stored,
        "the artifact still holds every plane"
    );
    assert_eq!(
        shallow.resident, whole.resident,
        "canonical decode widens to f32 at every depth"
    );
    assert_eq!(
        shallow.touch_per_token, whole.touch_per_token,
        "the image the executor streams is the same image"
    );
    assert!(
        shallow.read_to_prepare < whole.read_to_prepare,
        "what changed is what is OPENED: {} vs {}",
        shallow.read_to_prepare,
        whole.read_to_prepare
    );
}

/// A floor no shallow extent satisfies leaves only the terminal one, and
/// the budget is then refused rather than approximated.
#[test]
fn a_floor_finer_than_any_shallow_extent_leaves_only_the_terminal_one() {
    let (dir, _) = planed_container(Keep::EveryPlane);
    let (plan, store) = plan_and_store(dir.path());
    let (_, whole) = select(&plan, &store, &ResidencyBudget::UNBOUNDED).unwrap();
    // Finer than depth 1's declared radius, so only the exact extent
    // qualifies and the preparation cannot shrink.
    let budget = ResidencyBudget::UNBOUNDED
        .with_prepare_bytes(whole.read_to_prepare * 3 / 4)
        .with_fidelity(RepresentationFloor::RelativeRms(1e-9));
    let refusal = select(&plan, &store, &budget).expect_err("no extent is fine enough");
    assert!(refusal.contains("relative RMS at or under"), "{refusal}");
}

#[test]
fn each_extent_decodes_the_source_it_can_reach() {
    let (dir, planed) = planed_container(Keep::EveryPlane);
    let store = store_of(dir.path());
    let subject = &planed[0];
    let mut worst_before = f64::INFINITY;
    for depth in 0..=TERMINAL_DEPTH {
        let values = store
            .load_at(&subject.operand(), RepresentationExtent::at_depth(depth))
            .unwrap();
        let worst = subject
            .source
            .iter()
            .zip(&values)
            .filter(|(s, _)| s.is_normal())
            .map(|(s, v)| ((f64::from(*v) - f64::from(*s)) / f64::from(*s)).abs())
            .fold(0.0f64, f64::max);
        assert!(
            worst < worst_before || worst == 0.0,
            "depth {depth} is no closer to the source than the extent before it"
        );
        worst_before = worst;
    }
    assert_eq!(worst_before, 0.0, "the terminal extent is the source");
}
