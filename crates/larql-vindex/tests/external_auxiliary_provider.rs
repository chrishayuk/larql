//! VQ-1's extensibility proof: a crate that is not `larql-vindex`
//! registers a representation that DEPENDS ON ANOTHER REPRESENTED OBJECT,
//! and a second representation for the object it depends on, and the
//! whole thing plans, resolves, executes and invalidates through exported
//! API alone.
//!
//! Nothing here is the shipped `VQ8_SHARED`. The parent is `PALETTE8`,
//! its dependency is named `swatches` rather than `codebook`, and the
//! dependency is stored under `SWATCH16` — a codec with its own family
//! and revision. If any of that had to be known by the planner, the
//! loader or the accounting, the plugin plane would be cosmetic; a test
//! at the bottom of this file greps those files to prove it is not.
//!
//! The forecast this answers is
//! `docs/represent/forecasts/represent-vq-1.json` (W8, W9).

use std::collections::BTreeMap;
use std::ops::Range;
use std::path::PathBuf;

use larql_vindex::error::VindexError;
use larql_vindex::format::filenames::{AUXILIARY_REFERENCES_JSON, INDEX_JSON};
use larql_vindex::format::vindex3::auxiliary_references::{
    AuxiliaryReference, AuxiliaryReferences, OperandAddress,
};
use larql_vindex::format::vindex3::encode::segment::{
    read_segment_header, write_segment, PlannedTensor,
};
use larql_vindex::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use larql_vindex::format::vindex3::index::Vindex3Index;
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::accounting::{
    expectations, BlockGeometry, ResidencyBudget, ResourceLedger,
};
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prepared::{
    select_realizations_within, ExecutionSlice, PreparedOperands,
};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, OperandRef};
use larql_vindex::format::vindex3::represent::codec::codecs::{
    bf16_zlib, f32_planes, float, kquant, mxfp4, nvfp4, vq8_shared,
};
use larql_vindex::format::vindex3::represent::codec::streams::VALUES;
use larql_vindex::format::vindex3::represent::codec::{
    AccessGranularity, AuxiliaryMetadata, AuxiliarySpec, CodecCapabilities, CodecError,
    CodecOperands, CodecRegistry, ExtentCertificate, RepresentationCodec, RepresentationExtent,
    ResidencyProfile, StreamSpec,
};
use larql_vindex::format::vindex3::represent::nvfp4_pack::CodecIdentity;

/// The parent this crate registers: one byte per pair of weights.
const PARENT_LABEL: &str = "PALETTE8";
/// Its dependency's name — deliberately NOT `codebook`.
const SWATCHES: &str = "swatches";
/// The dependency's own representation: another external codec, so the
/// auxiliary has a provider identity of its own to lose.
const SWATCH_LABEL: &str = "SWATCH16";

/// Weights one palette index stands for.
const PAIR: usize = 2;
/// Entries a `u8` index can address.
const ENTRIES: usize = 256;
/// Where the swatch table lives in the container.
const SWATCH_TENSOR: &str = "external.swatches";
/// The projections this crate stores as palette indices.
const OWNERS: [&str; 2] = ["0.mlp.gate_proj.weight", "0.mlp.up_proj.weight"];

// ── The external parent ──────────────────────────────────────────────

struct Palette8;

const PARENT_STREAMS: [StreamSpec; 1] = [VALUES];
const PARENT_REQUIRES: [AuxiliarySpec; 1] = [AuxiliarySpec::new(SWATCHES)];
const SWATCH_SHAPE: [usize; 2] = [ENTRIES, PAIR];

impl RepresentationCodec for Palette8 {
    fn encoding_label(&self) -> &'static str {
        PARENT_LABEL
    }
    fn identity(&self) -> CodecIdentity {
        CodecIdentity {
            family: PARENT_LABEL.into(),
            revision: 1,
            group_elems: PAIR,
            element: "u8-index".into(),
            group_scale: "none".into(),
            tensor_scale: "none".into(),
            layout: "row-major/palette8".into(),
        }
    }
    fn streams(&self) -> &'static [StreamSpec] {
        &PARENT_STREAMS
    }
    fn required_auxiliaries(&self, _: RepresentationExtent) -> &'static [AuxiliarySpec] {
        &PARENT_REQUIRES
    }
    fn validate_auxiliary(
        &self,
        name: &str,
        target: &AuxiliaryMetadata,
        _: &[usize],
        _: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        target.require_shape(&SWATCH_SHAPE, tensor, PARENT_LABEL, name)
    }
    fn capabilities(&self) -> CodecCapabilities {
        CodecCapabilities {
            access: AccessGranularity::RowRandom,
            group_elems: PAIR,
            row_align_elems: PAIR,
            physical_align_bytes: 1,
        }
    }
    fn extents(&self) -> Vec<ExtentCertificate> {
        vec![ExtentCertificate::terminal(4.0)]
    }
    fn stored_bytes(
        &self,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<u64, CodecError> {
        self.certificate_at(extent, tensor)?;
        Ok((elements(shape) / PAIR) as u64)
    }
    fn validate(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        self.certificate_at(extent, tensor)?;
        let codes = operands.stream(VALUES, PARENT_LABEL, tensor)?;
        let need = elements(shape) / PAIR;
        if codes.len() != need {
            return Err(CodecError::StreamLength {
                tensor: tensor.into(),
                label: PARENT_LABEL.into(),
                stream: VALUES.name.into(),
                need,
                have: codes.len(),
            });
        }
        operands
            .auxiliaries
            .require(SWATCHES, PARENT_LABEL, tensor)
            .map(|_| ())
    }
    fn decode_rows(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        rows: Range<usize>,
        _: RepresentationExtent,
        dst: &mut [f32],
        tensor: &str,
    ) -> Result<(), CodecError> {
        let k = shape.last().copied().unwrap_or(1);
        let codes = operands.stream(VALUES, PARENT_LABEL, tensor)?;
        let swatches = operands
            .auxiliaries
            .require(SWATCHES, PARENT_LABEL, tensor)?;
        for (out, element) in dst.iter_mut().zip(rows.start * k..) {
            let entry = usize::from(codes[element / PAIR]);
            *out = swatches.values[entry * PAIR + element % PAIR];
        }
        Ok(())
    }
    fn decode_residency(&self) -> ResidencyProfile {
        ResidencyProfile::DECODED_F32
    }
}

fn elements(shape: &[usize]) -> usize {
    shape.iter().product::<usize>().max(1)
}

/// Nearest-entry coding against a swatch table — the fixture side.
fn encode_indices(values: &[f32], swatches: &[f32]) -> Vec<u8> {
    values
        .chunks(PAIR)
        .map(|pair| {
            (0..ENTRIES)
                .min_by(|a, b| {
                    let distance = |entry: &usize| -> f64 {
                        (0..PAIR)
                            .map(|lane| {
                                let d = f64::from(swatches[entry * PAIR + lane])
                                    - f64::from(pair[lane]);
                                d * d
                            })
                            .sum()
                    };
                    distance(a).total_cmp(&distance(b))
                })
                .unwrap_or(0) as u8
        })
        .collect()
}

// ── The external dependency's own codec ──────────────────────────────

/// The swatch table's representation: f32 values stored as their high
/// halves, so `SWATCH16` is a real codec with its own bytes rather than
/// a relabelled float.
struct Swatch16;

const SWATCH_STREAMS: [StreamSpec; 1] = [VALUES];

impl RepresentationCodec for Swatch16 {
    fn encoding_label(&self) -> &'static str {
        SWATCH_LABEL
    }
    fn identity(&self) -> CodecIdentity {
        CodecIdentity {
            family: SWATCH_LABEL.into(),
            revision: 1,
            group_elems: 1,
            element: "f32-high16".into(),
            group_scale: "none".into(),
            tensor_scale: "none".into(),
            layout: "row-major-le/high16".into(),
        }
    }
    fn streams(&self) -> &'static [StreamSpec] {
        &SWATCH_STREAMS
    }
    fn capabilities(&self) -> CodecCapabilities {
        CodecCapabilities {
            access: AccessGranularity::ElementRandom,
            group_elems: 1,
            row_align_elems: 1,
            physical_align_bytes: 1,
        }
    }
    fn extents(&self) -> Vec<ExtentCertificate> {
        vec![ExtentCertificate::terminal(16.0)]
    }
    fn stored_bytes(
        &self,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<u64, CodecError> {
        self.certificate_at(extent, tensor)?;
        Ok((elements(shape) * 2) as u64)
    }
    fn validate(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        _: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        operands
            .stream_of_len(VALUES, elements(shape) * 2, SWATCH_LABEL, tensor)
            .map(|_| ())
    }
    fn decode_rows(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        rows: Range<usize>,
        _: RepresentationExtent,
        dst: &mut [f32],
        tensor: &str,
    ) -> Result<(), CodecError> {
        let k = shape.last().copied().unwrap_or(1);
        let bytes = operands.stream(VALUES, SWATCH_LABEL, tensor)?;
        for (out, element) in dst.iter_mut().zip(rows.start * k..) {
            let high = u16::from_le_bytes([bytes[element * 2], bytes[element * 2 + 1]]);
            *out = f32::from_bits(u32::from(high) << 16);
        }
        Ok(())
    }
    fn decode_residency(&self) -> ResidencyProfile {
        ResidencyProfile::DECODED_F32
    }
}

fn encode_swatches(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// Every shipped codec plus this crate's two.
fn registry_with_providers() -> &'static CodecRegistry {
    Box::leak(Box::new(
        shipped()
            .register(Box::new(Palette8))
            .and_then(|r| r.register(Box::new(Swatch16)))
            .expect("two new labels"),
    ))
}

/// The shipped set, and the swatch provider WITHOUT the parent — for the
/// arm where the dependency's codec is what goes missing.
fn registry_without_swatches() -> &'static CodecRegistry {
    Box::leak(Box::new(
        shipped()
            .register(Box::new(Palette8))
            .expect("one new label"),
    ))
}

fn shipped() -> CodecRegistry {
    CodecRegistry::new()
        .register(Box::new(float::BF16))
        .and_then(|r| r.register(Box::new(float::F16)))
        .and_then(|r| r.register(Box::new(float::F32)))
        .and_then(|r| r.register(Box::new(kquant::Q4_K)))
        .and_then(|r| r.register(Box::new(kquant::Q6_K)))
        .and_then(|r| r.register(Box::new(kquant::Q8_0)))
        .and_then(|r| r.register(Box::new(nvfp4::NVFP4)))
        .and_then(|r| r.register(Box::new(mxfp4::MXFP4)))
        .and_then(|r| r.register(Box::new(bf16_zlib::BF16_ZLIB)))
        .and_then(|r| r.register(Box::new(f32_planes::F32_PLANES)))
        .and_then(|r| r.register(Box::new(vq8_shared::VQ8_SHARED)))
        .expect("the shipped labels are distinct")
}

// ── One container, built through exported API only ───────────────────

struct Built {
    dir: tempfile::TempDir,
    object: String,
    shape: Vec<usize>,
    swatches: Vec<f32>,
    sources: Vec<Vec<f32>>,
}

impl Built {
    fn container(&self) -> PathBuf {
        self.dir.path().join("container")
    }

    fn operand(&self, owner: &str) -> OperandRef {
        OperandRef {
            object: self.object.clone(),
            tensor: owner.to_string(),
            dtype: String::new(),
            shape: self.shape.clone(),
        }
    }

    fn open(&self, registry: &'static CodecRegistry) -> (ComponentOpPlan, OperandStore) {
        let container = self.container();
        let inspection = inspect_container(&container, false).unwrap();
        let outcome = plan_component_ops(&inspection, &container, "target").unwrap();
        let plan = outcome
            .plan
            .unwrap_or_else(|| panic!("the external fixture must plan: {:?}", outcome.defects));
        let store = OperandStore::open(&container, &inspection)
            .unwrap()
            .with_registry(registry);
        (plan, store)
    }

    fn prepared(&self, registry: &'static CodecRegistry) -> Result<PreparedOperands, VindexError> {
        let (plan, store) = self.open(registry);
        PreparedOperands::load(
            &plan,
            &store,
            &ProductionBackend::new(),
            ExecutionSlice::Full,
        )
    }
}

/// What the container holds, so the read counter has a control.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Shape {
    /// The arm: palette indices, a swatch table, a reference table.
    External,
    /// The control: the same fixture with the owners left as plain f32
    /// and no swatch table at all. Identical in every other tensor, so
    /// the difference between the two preparations' read counts IS the
    /// dependency's reading and nothing else.
    Plain,
    /// One owner coded, one left plain — the discrimination. A read count
    /// that is a constant and a read count that is per-owner agree at two
    /// owners and disagree here.
    OneOwner,
}

impl Shape {
    /// Whether this shape codes `owner` against the swatch table.
    fn codes(self, owner: &str) -> bool {
        match self {
            Shape::External => true,
            Shape::Plain => false,
            Shape::OneOwner => owner == OWNERS[0],
        }
    }

    /// The owners whose declaration the reference table carries.
    fn owners(self) -> &'static [&'static str] {
        match self {
            Shape::External => &OWNERS,
            Shape::Plain => &[],
            Shape::OneOwner => &OWNERS[..1],
        }
    }
}

fn build() -> Built {
    build_as(Shape::External)
}

fn build_as(shape_of: Shape) -> Built {
    let dir = tempfile::tempdir().unwrap();
    let checkpoint = dir.path().join("checkpoint");
    let container = dir.path().join("container");
    std::fs::create_dir_all(&checkpoint).unwrap();
    encode_fixture_container(dense_f32_model, &checkpoint, &container, "external");

    let index_path = container.join(INDEX_JSON);
    let mut index: Vindex3Index =
        serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    let (mut object, mut shape, mut swatches, mut sources) =
        (String::new(), Vec::new(), Vec::new(), Vec::new());

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
        let read = |t: &larql_vindex::format::vindex3::encode::segment::SegmentTensor| -> Vec<f32> {
            payload[t.offset as usize..(t.offset + t.len) as usize]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };
        let together: Vec<f32> = header
            .tensors
            .iter()
            .filter(|t| OWNERS.contains(&t.name.as_str()))
            .flat_map(&read)
            .collect();
        let low = together.iter().copied().fold(f32::INFINITY, f32::min);
        let high = together.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        swatches = (0..ENTRIES * PAIR)
            .map(|i| {
                let entry = (i / PAIR) as f32;
                low + entry * (high - low) / (ENTRIES - 1) as f32
            })
            .collect();
        // Stored through SWATCH16, so what the parent reads is what that
        // codec decodes — including its rounding.
        let swatch_bytes = encode_swatches(&swatches);
        let stored_swatches: Vec<f32> = swatch_bytes
            .chunks_exact(2)
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
            .collect();

        let mut planned = Vec::new();
        let mut bytes_by_name = BTreeMap::new();
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
            let (dtype, bytes) = if shape_of.codes(&t.name) {
                (
                    PARENT_LABEL.to_string(),
                    encode_indices(&values, &stored_swatches),
                )
            } else {
                (t.dtype.clone(), stored.to_vec())
            };
            planned.push(PlannedTensor {
                relative_name: t.name.clone(),
                source_name: t.name.clone(),
                dtype,
                shape: t.shape.clone(),
                len: bytes.len() as u64,
            });
            bytes_by_name.insert(t.name.clone(), bytes);
        }
        if !shape_of.owners().is_empty() {
            planned.push(PlannedTensor {
                relative_name: SWATCH_TENSOR.to_string(),
                source_name: SWATCH_TENSOR.to_string(),
                dtype: SWATCH_LABEL.to_string(),
                shape: SWATCH_SHAPE.to_vec(),
                len: swatch_bytes.len() as u64,
            });
            bytes_by_name.insert(SWATCH_TENSOR.to_string(), swatch_bytes);
        }
        swatches = stored_swatches;

        let written = write_segment(&path, &header.representation, planned, |name, w, hash| {
            let bytes = &bytes_by_name[name];
            std::io::Write::write_all(w, bytes).map_err(VindexError::Io)?;
            hash(bytes);
            Ok(bytes.len() as u64)
        })
        .unwrap();
        entry.payload_bytes = written.payload_bytes;
        entry.payload_sha256 = written.payload_sha256;
        entry.segment_sha256 = written.segment_sha256;
        entry.tensor_count = written.tensor_count;
    }

    if !shape_of.owners().is_empty() {
        let table = AuxiliaryReferences::new(
            shape_of
                .owners()
                .iter()
                .map(|owner| AuxiliaryReference {
                    owner: OperandAddress::new(&object, *owner),
                    auxiliary: SWATCHES.to_string(),
                    target: OperandAddress::new(&object, SWATCH_TENSOR),
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
        object,
        shape,
        swatches,
        sources,
    }
}

// ── The arms ─────────────────────────────────────────────────────────

/// An out-of-tree parent, an out-of-tree dependency with its own provider
/// identity, a dependency name this build never heard of — and it decodes.
#[test]
fn an_external_representation_resolves_an_external_dependency_and_decodes() {
    let built = build();
    let (_, store) = built.open(registry_with_providers());
    for (owner, source) in OWNERS.iter().zip(&built.sources) {
        let decoded = store.load(&built.operand(owner)).unwrap();
        assert_eq!(decoded.len(), source.len());
        // Every decoded pair is an entry of the swatch table as SWATCH16
        // decodes it — the parent read values, never bytes.
        for pair in decoded.chunks(PAIR) {
            assert!(
                built.swatches.chunks(PAIR).any(|entry| entry == pair),
                "{owner}: {pair:?} is not a swatch"
            );
        }
        assert!(decoded.iter().zip(source).any(|(d, s)| d != s));
    }
    // Neither label is one the shipped registry knows.
    assert!(CodecRegistry::builtin().by_label(PARENT_LABEL).is_none());
    assert!(CodecRegistry::builtin().by_label(SWATCH_LABEL).is_none());
}

/// Losing the DEPENDENCY's provider invalidates the prepared image, by
/// that provider's name — the parent's own codec is still registered and
/// that is not enough.
#[test]
fn losing_the_dependencys_provider_invalidates_the_image_by_name() {
    let built = build();
    let prepared = built.prepared(registry_with_providers()).unwrap();
    prepared
        .ensure_providers_in(registry_with_providers())
        .expect("nothing changed");

    let Err(err) = prepared.ensure_providers_in(registry_without_swatches()) else {
        panic!("the swatch provider is gone and the image is not executable");
    };
    let err = err.to_string();
    assert!(err.contains(SWATCH_LABEL), "{err}");
    assert!(err.contains("re-prepare"), "{err}");
    // And preparing against that registry refuses too, before any
    // execution: the dependency has no decode.
    let refusal = built
        .prepared(registry_without_swatches())
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(refusal.contains(SWATCH_LABEL), "{refusal}");
}

/// The store's counter and the ledger, differenced against a container
/// identical but for the dependency.
///
/// They do NOT agree, and the disagreement is the finding: the ledger
/// prices a shared object once because one object is one footprint, while
/// the loader — which has no dependency cache, caching being out of
/// VQ-1's scope — reads it once per owner. Asserting the exact gap, with
/// a one-owner arm to prove the cost is per-owner and not a constant, is
/// what makes a later cache show up as a failing test rather than as a
/// silently improved number.
#[test]
fn the_counter_reads_a_shared_dependency_per_owner_the_ledger_prices_it_once() {
    let built = build();
    let (plan, store) = built.open(registry_with_providers());
    let records = select_realizations_within(
        &plan,
        (&store).into(),
        &ProductionBackend::new(),
        &ExecutionSlice::Full,
        &ResidencyBudget::UNBOUNDED,
    )
    .unwrap();
    let priced = expectations(&records, |o| store.stored_len(o), BlockGeometry::executor());
    let ledger = ResourceLedger::aggregate(&priced);

    let pins: Vec<_> = records
        .iter()
        .flat_map(|r| &r.dependencies)
        .filter(|d| d.name == SWATCHES)
        .collect();
    assert_eq!(pins.len(), OWNERS.len(), "both owners pin it");
    let swatch_bytes = pins[0].stored_bytes.unwrap();
    assert_eq!(pins[0].label, SWATCH_LABEL);

    // The ledger counts it once...
    let bare: u64 = priced.iter().map(|e| e.read_to_prepare).sum::<u64>();
    assert_eq!(
        ledger.read_to_prepare,
        bare + swatch_bytes,
        "one swatch table, two owners"
    );

    // ...and the counter is read against the PREPARATION PATH the ledger
    // prices, differenced against a container identical but for the
    // dependency. `load_count` is process-global, so the arm alone says
    // nothing; the difference from the control is the swatch table's
    // reading and nothing else.
    let reads = |built: &Built| -> u64 {
        let (plan, store) = built.open(registry_with_providers());
        let before = store.load_count();
        let prepared = PreparedOperands::load(
            &plan,
            &store,
            &ProductionBackend::new(),
            ExecutionSlice::Full,
        )
        .unwrap();
        assert!(prepared.realizations().len() >= OWNERS.len());
        store.load_count() - before
    };
    let control = build_as(Shape::Plain);
    let floor = reads(&control);
    let swatch_reads = reads(&built) - floor;
    // The discrimination: one owner, one read. A constant cost of
    // resolving "the dependency plane at all" would have shown the same
    // number here as with two owners.
    assert_eq!(
        reads(&build_as(Shape::OneOwner)) - floor,
        1,
        "one owner reads it once"
    );

    // WHAT THIS MEASURES, AND WHAT IT DOES NOT: the ledger prices the
    // shared object once, because one object is one footprint. The loader
    // has no dependency cache — caching is out of VQ-1's scope — so it
    // reads the table once per owner that resolves it. Both numbers are
    // correct about different questions, and this asserts the exact gap
    // rather than the agreement, so that the day a cache lands this test
    // fails and says so.
    assert_eq!(
        swatch_reads,
        OWNERS.len() as u64,
        "the loader reads a shared dependency once per owner"
    );
    assert_eq!(
        ledger.read_to_prepare - bare,
        swatch_bytes,
        "and the ledger prices it once"
    );
}

/// The structural proof: no file in the planner, the loader or the
/// accounting mentions this crate's labels or its dependency's name.
///
/// A grep is a blunt instrument and that is why it is convincing here —
/// if the plugin plane were cosmetic, one of these spellings would have
/// had to appear in one of these files for any of the arms above to pass.
#[test]
fn no_core_file_names_this_crates_representation_or_its_dependency() {
    // The stripper's own control, first: a check that returns "absent"
    // is worth nothing until it is shown it could return "present".
    // Code before a comment survives; the comment does not.
    assert_eq!(strip("let codebook = 1; // codebook"), "let codebook = 1; ");
    assert_eq!(strip("// codebook"), "");

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/format/vindex3");
    let core = [
        root.join("opplan/build.rs"),
        root.join("opplan/exec/operands.rs"),
        root.join("opplan/exec/prepared.rs"),
        root.join("opplan/exec/accounting.rs"),
        root.join("opplan/exec/realization.rs"),
        root.join("auxiliary_references/mod.rs"),
        root.join("auxiliary_references/closure.rs"),
    ];
    for path in core {
        let text =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        // CODE, not prose. A doc comment is free to say "a codebook is
        // read once" as an illustration, and several of these files do —
        // that is a description of a use case, not a dependency on one.
        // What would make the plane cosmetic is an identifier, a match
        // arm or a string literal, and stripping comments is what tells
        // the two apart.
        let code: String = text.lines().map(strip).collect::<Vec<_>>().join("\n");
        assert!(
            code.contains("pub "),
            "{} stripped to nothing",
            path.display()
        );
        for spelling in [
            // This crate's, which the core cannot possibly know...
            PARENT_LABEL,
            SWATCH_LABEL,
            SWATCHES,
            // ...and the SHIPPED dependency-bearing codec's, which it
            // could have known and must not: the machinery knows about
            // requirements, never about codebooks.
            "VQ8_SHARED",
            "Vq8Shared",
            "codebook",
        ] {
            assert!(
                !code.contains(spelling),
                "{} names `{spelling}` in code",
                path.display()
            );
        }
    }
    assert!(root.exists());
}

/// A line's code, without its comment.
fn strip(line: &str) -> &str {
    line.split("//").next().unwrap_or("")
}
