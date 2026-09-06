//! Two operands, one codebook, one container: the dependency is resolved
//! through the table the container declares, not through a name anyone
//! guessed.
//!
//! The arm this file is: `VQ8_SHARED` codes for two FFN projections, an
//! f32 codebook stored once beside them, and an `auxiliary_references`
//! table that says which is which. What it proves is that the loader
//! resolves a dependency it was told about, that both owners get the same
//! object, and that every way the declaration can be wrong is refused
//! with the container in hand rather than in a unit test's imagination.

use std::path::Path;

use super::super::accounting::{
    expectations, BlockGeometry, Expectation, ResidencyBudget, ResourceLedger,
};
use super::super::operands::{AuxiliaryExtents, OperandStore};
use super::super::prepared::{select_realizations_within, ExecutionSlice};
use super::super::production::ProductionBackend;
use super::super::realization::{DependencyLifetime, RealizationRecord};
use crate::format::filenames::{AUXILIARY_REFERENCES_JSON, INDEX_JSON};
use crate::format::vindex3::auxiliary_references::{
    AuxiliaryReference, AuxiliaryReferences, OperandAddress,
};
use crate::format::vindex3::encode::segment::{read_segment_header, write_segment, PlannedTensor};
use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::index::Vindex3Index;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, OperandRef};
use crate::format::vindex3::represent::codec::codecs::f32_planes::{
    F32PlanesCodec, DTYPE_F32_PLANES, REFINE_8A, REFINE_8B, TERMINAL_DEPTH,
};
use crate::format::vindex3::represent::codec::codecs::vq8_shared::{
    Vq8SharedCodec, CODEBOOK, DTYPE_VQ8_SHARED, VQ_CODEBOOK_ENTRIES, VQ_VECTOR_ELEMS,
};
use crate::format::vindex3::represent::codec::RepresentationExtent;

/// The two tensors this arm stores as codes — one layer's gate and up
/// projections, so the shared codebook has two genuinely different owners.
const OWNERS: [&str; 2] = ["0.mlp.gate_proj.weight", "0.mlp.up_proj.weight"];
/// Where the codebook is written: a tensor of the same segment, named
/// nothing like its owners, because a dependency is addressed and not
/// spelled.
const CODEBOOK_TENSOR: &str = "shared.vq8.codebook";
/// The dtype a raw f32 codebook is stored under — arm one: the dependency
/// has the most ordinary representation there is.
const CODEBOOK_DTYPE: &str = "F32";
/// The dtype a refinement plane of a progressive codebook is stored under.
const PLANE_DTYPE: &str = "U8";

/// How the codebook itself is stored — arm one raw, arm two progressive.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Codebook {
    /// Raw f32: the dependency has the most ordinary representation there
    /// is, so arm one isolates resolution.
    Raw,
    /// `F32_PLANES`: the dependency has a representation, extents and a
    /// provider of its own, so arm two can move its extent while every
    /// code stays exactly where it was.
    Progressive,
}

/// How the container's declaration is wrong, for the refusal arms.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Declared {
    /// Both owners point at the one codebook: the arm itself.
    Correctly,
    /// No table at all — the codec requires a dependency nobody declared.
    NotAtAll,
    /// A table pointing at a tensor the container does not hold.
    AtSomethingAbsent,
    /// A codebook of the wrong width: judged from metadata, before bytes.
    AtTheWrongShape,
}

struct Built {
    dir: tempfile::TempDir,
    /// The values each owner was coded from, in owner order.
    sources: Vec<Vec<f32>>,
    shape: Vec<usize>,
    object: String,
    codebook: Vec<f32>,
}

impl Built {
    fn operand(&self, owner: &str) -> OperandRef {
        OperandRef {
            object: self.object.clone(),
            tensor: owner.to_string(),
            dtype: String::new(),
            shape: self.shape.clone(),
        }
    }

    fn plan_and_store(&self) -> (ComponentOpPlan, OperandStore) {
        let container = self.dir.path().join("container");
        let inspection = inspect_container(&container, false).unwrap();
        let outcome = plan_component_ops(&inspection, &container, "target").unwrap();
        let plan = outcome
            .plan
            .unwrap_or_else(|| panic!("the coded fixture must plan: {:?}", outcome.defects));
        (plan, self.store().unwrap())
    }

    fn store(&self) -> Result<OperandStore, crate::error::VindexError> {
        let container = self.dir.path().join("container");
        let inspection = inspect_container(&container, false)?;
        OperandStore::open(&container, &inspection)
    }
}

/// A codebook spanning the fixture's value range, so nearest-entry coding
/// is a real assignment.
fn codebook_over(values: &[f32]) -> Vec<f32> {
    let low = values.iter().copied().fold(f32::INFINITY, f32::min);
    let high = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = high - low;
    (0..VQ_CODEBOOK_ENTRIES * VQ_VECTOR_ELEMS)
        .map(|i| {
            let entry = (i / VQ_VECTOR_ELEMS) as f32;
            low + entry * span / (VQ_CODEBOOK_ENTRIES - 1) as f32
        })
        .collect()
}

/// Rewrite the fixture's two projections as codes, store one codebook
/// beside them, and declare the references `declared` describes.
fn build(declared: Declared) -> Built {
    build_with(declared, Codebook::Raw)
}

fn build_with(declared: Declared, stored_as: Codebook) -> Built {
    let dir = tempfile::tempdir().unwrap();
    let checkpoint = dir.path().join("checkpoint");
    let container = dir.path().join("container");
    std::fs::create_dir_all(&checkpoint).unwrap();
    encode_fixture_container(dense_f32_model, &checkpoint, &container, "vq");

    let index_path = container.join(INDEX_JSON);
    let mut index: Vindex3Index =
        serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    let mut sources = Vec::new();
    let mut shape = Vec::new();
    let mut object = String::new();
    let mut codebook = Vec::new();

    for entry in index.representations.values_mut() {
        let path = container.join(&entry.segment);
        let (header, payload_start) = read_segment_header(&path).unwrap();
        if !header
            .tensors
            .iter()
            .any(|t| OWNERS.contains(&t.name.as_str()))
        {
            continue;
        }
        object = entry.object.clone();
        let file = std::fs::read(&path).unwrap();
        let payload = &file[payload_start as usize..];
        let read = |t: &crate::format::vindex3::encode::segment::SegmentTensor| -> Vec<f32> {
            payload[t.offset as usize..(t.offset + t.len) as usize]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };
        // One codebook over everything both owners hold, so the two really
        // do share it rather than each getting a private best fit.
        let together: Vec<f32> = header
            .tensors
            .iter()
            .filter(|t| OWNERS.contains(&t.name.as_str()))
            .flat_map(&read)
            .collect();
        codebook = codebook_over(&together);
        let stored_codebook = if declared == Declared::AtTheWrongShape {
            // A codebook one lane narrow: the shape rule refuses it from
            // metadata, before a value is read.
            vec![0.0f32; VQ_CODEBOOK_ENTRIES * (VQ_VECTOR_ELEMS - 1)]
        } else {
            codebook.clone()
        };
        let codebook_shape = vec![
            VQ_CODEBOOK_ENTRIES,
            if declared == Declared::AtTheWrongShape {
                VQ_VECTOR_ELEMS - 1
            } else {
                VQ_VECTOR_ELEMS
            },
        ];

        let mut planned = Vec::new();
        let mut bytes_by_name = std::collections::BTreeMap::new();
        for t in &header.tensors {
            let stored = &payload[t.offset as usize..(t.offset + t.len) as usize];
            if !OWNERS.contains(&t.name.as_str()) {
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
            let values = read(t);
            shape = t.shape.clone();
            sources.push(values.clone());
            let codes = Vq8SharedCodec::encode_codes(&values, &codebook);
            planned.push(PlannedTensor {
                relative_name: t.name.clone(),
                source_name: t.name.clone(),
                dtype: DTYPE_VQ8_SHARED.to_string(),
                shape: t.shape.clone(),
                len: codes.len() as u64,
            });
            bytes_by_name.insert(t.name.clone(), codes);
        }
        // The codebook: one object, whatever the number of owners, and
        // stored however this arm asks. Its own representation is its
        // business — the owner sees values either way.
        match stored_as {
            Codebook::Raw => {
                let bytes: Vec<u8> = stored_codebook
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect();
                planned.push(PlannedTensor {
                    relative_name: CODEBOOK_TENSOR.to_string(),
                    source_name: CODEBOOK_TENSOR.to_string(),
                    dtype: CODEBOOK_DTYPE.to_string(),
                    shape: codebook_shape,
                    len: bytes.len() as u64,
                });
                bytes_by_name.insert(CODEBOOK_TENSOR.to_string(), bytes);
            }
            Codebook::Progressive => {
                let (base, refine_a, refine_b) = F32PlanesCodec::encode_planes(&stored_codebook);
                for (name, dtype, bytes) in [
                    (CODEBOOK_TENSOR.to_string(), DTYPE_F32_PLANES, base),
                    (
                        OperandStore::sibling_stream_tensor(CODEBOOK_TENSOR, REFINE_8A.name),
                        PLANE_DTYPE,
                        refine_a,
                    ),
                    (
                        OperandStore::sibling_stream_tensor(CODEBOOK_TENSOR, REFINE_8B.name),
                        PLANE_DTYPE,
                        refine_b,
                    ),
                ] {
                    planned.push(PlannedTensor {
                        relative_name: name.clone(),
                        source_name: name.clone(),
                        dtype: dtype.to_string(),
                        shape: codebook_shape.clone(),
                        len: bytes.len() as u64,
                    });
                    bytes_by_name.insert(name, bytes);
                }
            }
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
    assert_eq!(sources.len(), OWNERS.len(), "both owners were rewritten");

    if declared != Declared::NotAtAll {
        let target = match declared {
            Declared::AtSomethingAbsent => "nothing.of.the.kind",
            _ => CODEBOOK_TENSOR,
        };
        let table = AuxiliaryReferences::new(
            OWNERS
                .iter()
                .map(|owner| AuxiliaryReference {
                    owner: OperandAddress::new(&object, *owner),
                    auxiliary: CODEBOOK.to_string(),
                    target: OperandAddress::new(&object, target),
                })
                .collect(),
        );
        std::fs::write(
            container.join(AUXILIARY_REFERENCES_JSON),
            serde_json::to_string_pretty(&table).unwrap(),
        )
        .unwrap();
        index.auxiliary_references = Some(AUXILIARY_REFERENCES_JSON.to_string());
    }
    std::fs::write(&index_path, serde_json::to_string_pretty(&index).unwrap()).unwrap();

    Built {
        dir,
        sources,
        shape,
        object,
        codebook,
    }
}

fn container_of(dir: &Path) -> std::path::PathBuf {
    dir.join("container")
}

/// The arm: both owners decode through the one codebook the container
/// declares, and the values are the codebook's entries — not the source's.
#[test]
fn two_operands_decode_through_the_one_codebook_the_container_declares() {
    let built = build(Declared::Correctly);
    let store = built.store().unwrap();
    for (owner, source) in OWNERS.iter().zip(&built.sources) {
        let decoded = store
            .load(&built.operand(owner))
            .unwrap_or_else(|e| panic!("{owner}: {e}"));
        assert_eq!(decoded.len(), source.len());
        // Every decoded vector is exactly some codebook entry: the values
        // came from the dependency, not from the codes.
        for vector in decoded.chunks(VQ_VECTOR_ELEMS) {
            let found = built
                .codebook
                .chunks(VQ_VECTOR_ELEMS)
                .any(|entry| entry == vector);
            assert!(found, "{owner}: {vector:?} is not a codebook entry");
        }
        // And it is a real assignment, checked as one: every vector came
        // back as the NEAREST codebook entry to its source, which is what
        // the encoder chose and what the decoder must reproduce. (How
        // close that is belongs to the codebook and the data — this
        // fixture's entries are constant vectors, so the residual is the
        // spread within each four-weight vector. Not a quality claim.)
        assert!(decoded.iter().zip(source).any(|(d, s)| d != s));
        for (vector, decoded_vector) in source
            .chunks(VQ_VECTOR_ELEMS)
            .zip(decoded.chunks(VQ_VECTOR_ELEMS))
        {
            let nearest = built
                .codebook
                .chunks(VQ_VECTOR_ELEMS)
                .min_by(|a, b| {
                    let distance = |entry: &[f32]| -> f64 {
                        entry
                            .iter()
                            .zip(vector)
                            .map(|(e, v)| {
                                let d = f64::from(*e) - f64::from(*v);
                                d * d
                            })
                            .sum()
                    };
                    distance(a).total_cmp(&distance(b))
                })
                .expect("the codebook has entries");
            assert_eq!(decoded_vector, nearest, "{owner}");
        }
    }
}

/// One object, two owners: the store reads the codebook for each decode
/// that needs it, and the two owners get the same values from it.
#[test]
fn both_owners_resolve_the_same_object() {
    let built = build(Declared::Correctly);
    let store = built.store().unwrap();
    let table = store.references();
    let target = OperandAddress::new(&built.object, CODEBOOK_TENSOR);
    let mut owners = table.owners_of(&target);
    owners.sort();
    assert_eq!(owners.len(), 2, "one target, two owners: {owners:?}");
    // Each owner's decode reads its own codes and the shared codebook —
    // two payload reads apiece, and the second is the same tensor.
    let before = store.load_count();
    store.load(&built.operand(OWNERS[0])).unwrap();
    assert_eq!(store.load_count() - before, 2);
    let before = store.load_count();
    store.load(&built.operand(OWNERS[1])).unwrap();
    assert_eq!(store.load_count() - before, 2);
}

/// Every way the container's declaration can be wrong, refused with the
/// container in hand.
#[test]
fn a_wrong_declaration_is_refused_through_the_container() {
    // Nobody declared the dependency the codec requires.
    let built = build(Declared::NotAtAll);
    let store = built.store().unwrap();
    let err = store
        .load(&built.operand(OWNERS[0]))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains(CODEBOOK) && err.contains("no reference"),
        "{err}"
    );

    // The table points at a tensor the container does not hold.
    let built = build(Declared::AtSomethingAbsent);
    let store = built.store().unwrap();
    let err = store
        .load(&built.operand(OWNERS[0]))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("no such tensor") && err.contains("nothing.of.the.kind"),
        "{err}"
    );

    // The codebook is the wrong shape — refused from METADATA, so the
    // refusal does not depend on anything having read it.
    let built = build(Declared::AtTheWrongShape);
    let store = built.store().unwrap();
    let err = store
        .load(&built.operand(OWNERS[0]))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("is unusable") && err.contains(CODEBOOK),
        "{err}"
    );
}

/// A table the index names and the container does not hold is refused at
/// OPEN, not at the first decode that needed it.
#[test]
fn a_declared_table_that_is_not_there_is_refused_when_the_store_opens() {
    let built = build(Declared::Correctly);
    std::fs::remove_file(container_of(built.dir.path()).join(AUXILIARY_REFERENCES_JSON)).unwrap();
    let Err(err) = built.store() else {
        panic!("the index names a table the container does not hold");
    };
    let err = err.to_string();
    assert!(
        err.contains("named by the index") && err.contains(AUXILIARY_REFERENCES_JSON),
        "{err}"
    );
}

/// A decode at an extent the codec does not declare is refused before the
/// dependency is resolved at all.
#[test]
fn an_extent_the_codec_does_not_declare_is_refused() {
    let built = build(Declared::Correctly);
    let store = built.store().unwrap();
    let err = store
        .load_at(&built.operand(OWNERS[0]), RepresentationExtent::at_depth(1))
        .unwrap_err()
        .to_string();
    assert!(err.contains("no extent at depth 1"), "{err}");
}

/// **Arm two.** The codebook is stored progressively, and only ITS extent
/// moves: the codes are the same bytes, the reference table is the same
/// table, and what changes is how much of the dependency was read.
///
/// This is what makes a dependency a represented OBJECT rather than a
/// blob: it has a representation of its own, that representation has
/// extents, and choosing among them is a decision about the dependency
/// that its owner never sees the mechanics of.
#[test]
fn only_the_codebooks_extent_moves_and_the_values_move_with_it() {
    let built = build_with(Declared::Correctly, Codebook::Progressive);
    let store = built.store().unwrap();
    let operand = built.operand(OWNERS[0]);
    let whole = AuxiliaryExtents::whole();
    let shallow = AuxiliaryExtents::whole().with(CODEBOOK, RepresentationExtent::BASE);

    let at_terminal = store
        .load_with(&operand, RepresentationExtent::BASE, &whole)
        .unwrap();
    let at_base = store
        .load_with(&operand, RepresentationExtent::BASE, &shallow)
        .unwrap();
    assert_eq!(at_terminal.len(), at_base.len());

    // The values moved, and moved the way the codebook's own extent says:
    // every one is the depth-0 truncation of the value the whole codebook
    // gave. The codes chose the same entries; the entries say less.
    assert!(at_terminal.iter().zip(&at_base).any(|(a, b)| a != b));
    for (whole_value, shallow_value) in at_terminal.iter().zip(&at_base) {
        let truncated = f32::from_bits(whole_value.to_bits() & 0xffff_0000);
        assert_eq!(*shallow_value, truncated);
    }

    // And the READING moved with it: a whole codebook is three planes, a
    // base-extent one is a single plane, while the codes are one payload
    // either way.
    let before = store.load_count();
    store
        .load_with(&operand, RepresentationExtent::BASE, &whole)
        .unwrap();
    let whole_reads = store.load_count() - before;
    let before = store.load_count();
    store
        .load_with(&operand, RepresentationExtent::BASE, &shallow)
        .unwrap();
    let shallow_reads = store.load_count() - before;
    assert_eq!(
        (whole_reads, shallow_reads),
        (1 + u64::from(TERMINAL_DEPTH) + 1, 2),
        "codes plus the planes the chosen extent reads"
    );
}

/// The owner is unchanged by any of it: `VQ8_SHARED` neither knows nor
/// can ask how its codebook is stored.
#[test]
fn the_owner_decodes_identically_however_its_codebook_is_stored() {
    let raw = build_with(Declared::Correctly, Codebook::Raw);
    let progressive = build_with(Declared::Correctly, Codebook::Progressive);
    for owner in OWNERS {
        let from_raw = raw.store().unwrap().load(&raw.operand(owner)).unwrap();
        let from_planes = progressive
            .store()
            .unwrap()
            .load(&progressive.operand(owner))
            .unwrap();
        assert_eq!(
            from_raw, from_planes,
            "{owner}: the terminal extent of the planes IS the raw image"
        );
    }
}

/// **The accounting.** A shared codebook is ONE object: its footprint and
/// the reading that prepares it count once, however many owners resolve
/// it. Under canonical decode nothing of it stays resident and no token
/// touches it — and the ledger says so because the REALIZATION says so,
/// not because it is an auxiliary.
#[test]
fn a_shared_dependency_is_counted_once_and_costs_what_its_lifetime_says() {
    let built = build(Declared::Correctly);
    let (plan, store) = built.plan_and_store();
    let records = select_realizations_within(
        &plan,
        (&store).into(),
        &ProductionBackend::new(),
        &ExecutionSlice::Full,
        &ResidencyBudget::UNBOUNDED,
    )
    .unwrap();

    // Both owners pin the same dependency, by the name their codec declared.
    let owners: Vec<&RealizationRecord> = records
        .iter()
        .filter(|r| r.representation == DTYPE_VQ8_SHARED)
        .collect();
    assert_eq!(owners.len(), OWNERS.len(), "both owners were planned");
    for record in &owners {
        assert_eq!(record.dependencies.len(), 1);
        let pin = &record.dependencies[0];
        assert_eq!(pin.name, CODEBOOK);
        assert_eq!(pin.tensor, CODEBOOK_TENSOR);
        assert_eq!(
            pin.lifetime,
            DependencyLifetime::PreparationOnly,
            "a decode is finished with its codebook once it has an f32 image"
        );
    }

    let priced = expectations(&records, |o| store.stored_len(o), BlockGeometry::executor());
    let ledger = ResourceLedger::aggregate(&priced);
    let codebook_bytes = owners[0].dependencies[0].stored_bytes.unwrap();
    assert!(codebook_bytes > 0);

    // Counted ONCE: the ledger's footprint moves by one codebook, not two.
    let without_dependencies: Vec<Expectation> = priced
        .iter()
        .cloned()
        .map(|mut e| {
            e.dependencies.clear();
            e
        })
        .collect();
    let bare = ResourceLedger::aggregate(&without_dependencies);
    assert_eq!(
        ledger.stored - bare.stored,
        codebook_bytes,
        "one codebook, two owners"
    );
    assert_eq!(
        ledger.read_to_prepare - bare.read_to_prepare,
        codebook_bytes
    );
    // And nothing of it is held or streamed, because the pinned
    // realization decodes.
    assert_eq!(ledger.resident, bare.resident);
    assert_eq!(ledger.touch_per_token, bare.touch_per_token);
}

/// The control, priced and never selected: a realization that RETAINED
/// its codebook would pay residency once and touch per use — which is how
/// the ledger shows that lifetime is the realization's to declare.
///
/// Nothing in this build produces such a pin. It is constructed here, at
/// the PRICING layer, and the first assertion checks the backend never
/// offers one: a declaration with no implementation must not be
/// selectable.
#[test]
fn a_retaining_realization_would_pay_residency_once_and_touch_per_use() {
    let built = build(Declared::Correctly);
    let (plan, store) = built.plan_and_store();
    let records = select_realizations_within(
        &plan,
        (&store).into(),
        &ProductionBackend::new(),
        &ExecutionSlice::Full,
        &ResidencyBudget::UNBOUNDED,
    )
    .unwrap();
    assert!(
        records
            .iter()
            .flat_map(|r| &r.dependencies)
            .all(|d| d.lifetime == DependencyLifetime::PreparationOnly),
        "no realization in this build declares a retained dependency"
    );

    let priced = expectations(&records, |o| store.stored_len(o), BlockGeometry::executor());
    let decoded = ResourceLedger::aggregate(&priced);
    let retained: Vec<Expectation> = priced
        .iter()
        .cloned()
        .map(|mut e| {
            for dependency in &mut e.dependencies {
                dependency.lifetime = DependencyLifetime::Retained;
            }
            e
        })
        .collect();
    let held = ResourceLedger::aggregate(&retained);

    let elements = VQ_CODEBOOK_ENTRIES * VQ_VECTOR_ELEMS;
    let image = (elements * std::mem::size_of::<f32>()) as u64;
    assert_eq!(
        held.resident - decoded.resident,
        image,
        "resident once, whoever keeps it"
    );
    assert_eq!(
        held.touch_per_token - decoded.touch_per_token,
        image * OWNERS.len() as u64,
        "touched once per owner that reads it"
    );
    // The footprint is a container fact and does not move with lifetime.
    assert_eq!(held.stored, decoded.stored);
    assert_eq!(held.read_to_prepare, decoded.read_to_prepare);
}
