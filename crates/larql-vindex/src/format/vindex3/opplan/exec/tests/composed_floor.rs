//! **B3's forcing case, through the real planner.**
//!
//! [`attested_fidelity`](super::super::attested_fidelity) proves the
//! composition algebra against hand-built options. This file proves the
//! CARRIAGE: that `select_realizations_within` — the function a caller
//! actually plans through — hands the fidelity floor the COMPOSED bound
//! and not the codec's declared one.
//!
//! The two are different claims, and the gap between them is where a
//! guarantee goes missing. Until carriage, `shallowest_saving` read
//! `option.certificate.radius` exactly as the codec declared it, with no
//! knowledge of the dependency the extent decodes through. That was
//! invisible rather than benign: no SHIPPED codec both declares a radius
//! and requires an auxiliary, so nothing in the suite could tell the two
//! readings apart. These codecs exist precisely to tell them apart.
//!
//! ```text
//! parent radius (declared, depth 0): 0.003
//! coarse codebook:                   0.003  -> composed 0.006 -> refused
//! fine codebook:                     0.001  -> composed 0.004 -> selected
//! floor:                             0.005
//! ```
//!
//! Neither half decides it: 0.003 is inside the floor and so is each
//! codebook's bound. Only the composition separates them, which is what
//! makes this a witness for carriage rather than for arithmetic.

use std::ops::Range;

use super::super::accounting::{
    expectations, BlockGeometry, RepresentationFloor, ResidencyBudget, ResourceLedger,
};
use super::super::operands::OperandStore;
use super::super::prepared::{select_realizations_within, ExecutionSlice};
use super::super::production::ProductionBackend;
use super::super::realization::RealizationRecord;
use crate::format::filenames::{
    AUXILIARY_REFERENCES_JSON, INDEX_JSON, REPRESENTATION_ATTESTATIONS_JSON,
};
use crate::format::vindex3::auxiliary_references::{
    AuxiliaryReference, AuxiliaryReferences, OperandAddress,
};
use crate::format::vindex3::encode::segment::{read_segment_header, write_segment, PlannedTensor};
use crate::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use crate::format::vindex3::index::Vindex3Index;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use crate::format::vindex3::represent::codec::auxiliary::AuxiliaryMetadata;
use crate::format::vindex3::represent::codec::codecs::{float, kquant, mxfp4, nvfp4};
use crate::format::vindex3::represent::codec::{
    AccessGranularity, AuxiliarySpec, CodecCapabilities, CodecError, CodecOperands, CodecRegistry,
    ExtentCertificate, FidelityCertificate, RepresentationCodec, RepresentationExtent,
    ResidencyProfile, StreamRole, StreamSpec,
};
use crate::format::vindex3::represent::nvfp4_pack::CodecIdentity;
use crate::format::vindex3::representation_attestations::recognition::RecognisedMethods;
use crate::format::vindex3::representation_attestations::{
    content_digest, terminal_baseline, AttestationBinding, AttestationMethod,
    RepresentationAttestations, StoredAttestation, StoredId,
};

// ── The premises, as `const` so they cannot drift into a tautology ───

/// What the OWNER's shallow extent declares on its own.
const PARENT: f64 = 0.003;
/// What the coarse codebook certifies.
const COARSE: f64 = 0.003;
/// What the fine codebook certifies.
const FINE: f64 = 0.001;
/// What execution requires.
const FLOOR: f64 = 0.005;
/// What an ENCODER measured on this instance, against the coarse
/// codebook — better than the scheme's declaration, because the scheme
/// has to hold for every tensor and this number was measured on one.
const ATTESTED: f64 = 0.001;

const AUTHORITY: &str = "larql-encoder";
const METHOD: &str = "measured-rms";

/// Neither half decides it, and the composition decides it both ways. If
/// someone edits these so that one term alone settles the outcome, the
/// build fails rather than the test quietly proving nothing.
const _: () = {
    assert!(PARENT < FLOOR);
    assert!(COARSE < FLOOR);
    assert!(FINE < FLOOR);
    assert!(PARENT + COARSE > FLOOR);
    assert!(PARENT + FINE < FLOOR);
    // And the attested arm: the SAME coarse codebook, admitted only
    // because the measured claim replaced the declared one.
    assert!(ATTESTED + COARSE < FLOOR);
};

const OWNER_LABEL: &str = "GRADED_PALETTE";
const COARSE_LABEL: &str = "COARSEBOOK";
const FINE_LABEL: &str = "FINEBOOK";
const CODEBOOK: &str = "codebook";
const CODEBOOK_TENSOR: &str = "shared.palette.codebook";
const ENTRIES: usize = 256;
const REFINE_DTYPE: &str = "U8";

/// The projections stored under the owner's codec.
const OWNERS: [&str; 2] = ["0.mlp.gate_proj.weight", "0.mlp.up_proj.weight"];

const VALUES: StreamSpec = StreamSpec {
    name: "codes",
    role: StreamRole::Values,
};
const REFINE: StreamSpec = StreamSpec {
    name: "residual",
    role: StreamRole::Refinement { depth: 1 },
};
const OWNER_STREAMS: [StreamSpec; 2] = [VALUES, REFINE];
const BOOK_STREAMS: [StreamSpec; 1] = [VALUES];
const OWNER_AUXILIARIES: [AuxiliarySpec; 1] = [AuxiliarySpec::new(CODEBOOK)];

fn elements(shape: &[usize]) -> usize {
    shape.iter().product::<usize>().max(1)
}

// ── The owner: graded extents AND a dependency ───────────────────────

/// A palette codec with two extents and a required codebook — the
/// combination no shipped codec has, and the only combination under which
/// the declared and composed readings of a floor differ.
struct GradedPalette;

impl RepresentationCodec for GradedPalette {
    fn encoding_label(&self) -> &'static str {
        OWNER_LABEL
    }
    fn identity(&self) -> CodecIdentity {
        CodecIdentity {
            family: OWNER_LABEL.into(),
            revision: 1,
            group_elems: 1,
            element: "u8-index".into(),
            group_scale: "none".into(),
            tensor_scale: "none".into(),
            layout: "row-major".into(),
        }
    }
    fn streams(&self) -> &'static [StreamSpec] {
        &OWNER_STREAMS
    }
    fn required_auxiliaries(&self, _: RepresentationExtent) -> &'static [AuxiliarySpec] {
        // At every depth: a code means nothing without the palette.
        &OWNER_AUXILIARIES
    }
    fn validate_auxiliary(
        &self,
        name: &str,
        target: &AuxiliaryMetadata,
        _: &[usize],
        _: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        // The palette's shape is this codec's business: one entry per
        // code, and a code is a byte.
        target.require_shape(&[ENTRIES], tensor, OWNER_LABEL, name)
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
        vec![
            ExtentCertificate::certified(
                0,
                8.0,
                FidelityCertificate::relative_rms(PARENT).expect("a well-formed radius"),
            ),
            // Terminal: the residual plane restores the values exactly, so
            // it declares no radius and the floor admits it unconditionally.
            ExtentCertificate {
                extent: RepresentationExtent::at_depth(1),
                bits_per_weight: 16.0,
                radius: None,
            },
        ]
    }
    fn stored_bytes(
        &self,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<u64, CodecError> {
        self.certificate_at(extent, tensor)?;
        let per = if extent.depth == 0 { 1 } else { 2 };
        Ok((elements(shape) * per) as u64)
    }
    fn validate(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        operands.stream_of_len(VALUES, elements(shape), OWNER_LABEL, tensor)?;
        if extent.depth >= 1 {
            operands.stream_of_len(REFINE, elements(shape), OWNER_LABEL, tensor)?;
        }
        Ok(())
    }
    fn decode_rows(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        rows: Range<usize>,
        extent: RepresentationExtent,
        dst: &mut [f32],
        tensor: &str,
    ) -> Result<(), CodecError> {
        let k = shape.last().copied().unwrap_or(1);
        let codes = operands.stream(VALUES, OWNER_LABEL, tensor)?;
        let palette = operands
            .auxiliaries
            .require(CODEBOOK, OWNER_LABEL, tensor)?
            .values;
        let residual = if extent.depth >= 1 {
            Some(operands.stream(REFINE, OWNER_LABEL, tensor)?)
        } else {
            None
        };
        for (out, element) in dst.iter_mut().zip(rows.start * k..) {
            let base = palette[usize::from(codes[element])];
            *out = match residual {
                // The residual plane is a signed byte of the palette step,
                // which is what makes depth 1 terminal rather than merely
                // better.
                Some(bytes) => base + f32::from(bytes[element] as i8) * PALETTE_STEP,
                None => base,
            };
        }
        Ok(())
    }
    fn decode_residency(&self) -> ResidencyProfile {
        ResidencyProfile::DECODED_F32
    }
}

/// The quantum a residual byte corrects by.
const PALETTE_STEP: f32 = 1.0 / 512.0;

// ── The dependency, at two fidelities ────────────────────────────────

/// A codebook codec that CERTIFIES its own reconstruction — the term the
/// owner's bound has to be widened by.
struct Codebook {
    label: &'static str,
    radius: f64,
}

const COARSEBOOK: Codebook = Codebook {
    label: COARSE_LABEL,
    radius: COARSE,
};
const FINEBOOK: Codebook = Codebook {
    label: FINE_LABEL,
    radius: FINE,
};

impl RepresentationCodec for Codebook {
    fn encoding_label(&self) -> &'static str {
        self.label
    }
    fn identity(&self) -> CodecIdentity {
        CodecIdentity {
            family: self.label.into(),
            revision: 1,
            group_elems: 1,
            element: "f32".into(),
            group_scale: "none".into(),
            tensor_scale: "none".into(),
            layout: "row-major".into(),
        }
    }
    fn streams(&self) -> &'static [StreamSpec] {
        &BOOK_STREAMS
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
        // One extent, which is therefore its terminal one — and unlike
        // `ExtentCertificate::terminal` it CERTIFIES that extent, because
        // a lossy codebook's error is what the owner must inherit.
        vec![ExtentCertificate::certified(
            0,
            32.0,
            FidelityCertificate::relative_rms(self.radius).expect("a well-formed radius"),
        )]
    }
    fn stored_bytes(
        &self,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<u64, CodecError> {
        self.certificate_at(extent, tensor)?;
        Ok((elements(shape) * 4) as u64)
    }
    fn validate(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        _: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        operands
            .stream_of_len(VALUES, elements(shape) * 4, self.label, tensor)
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
        let bytes = operands.stream(VALUES, self.label, tensor)?;
        for (out, element) in dst.iter_mut().zip(rows.start * k..) {
            let at = element * 4;
            *out = f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
        }
        Ok(())
    }
    fn decode_residency(&self) -> ResidencyProfile {
        ResidencyProfile::DECODED_F32
    }
}

/// Enough shipped codecs for the fixture's other tensors, plus the
/// owner and BOTH codebooks — one registry, so the arms differ only in
/// what the container stores.
fn registry() -> &'static CodecRegistry {
    use std::sync::OnceLock;
    static ONCE: OnceLock<CodecRegistry> = OnceLock::new();
    ONCE.get_or_init(|| {
        CodecRegistry::new()
            .register(Box::new(float::BF16))
            .and_then(|r| r.register(Box::new(float::F16)))
            .and_then(|r| r.register(Box::new(float::F32)))
            .and_then(|r| r.register(Box::new(kquant::Q4_K)))
            .and_then(|r| r.register(Box::new(kquant::Q6_K)))
            .and_then(|r| r.register(Box::new(kquant::Q8_0)))
            .and_then(|r| r.register(Box::new(nvfp4::NVFP4)))
            .and_then(|r| r.register(Box::new(mxfp4::MXFP4)))
            .and_then(|r| r.register(Box::new(GradedPalette)))
            .and_then(|r| r.register(Box::new(COARSEBOOK)))
            .and_then(|r| r.register(Box::new(FINEBOOK)))
            .expect("distinct labels")
    })
}

// ── One container, with the codebook stored at one fidelity or other ──

struct Built {
    dir: tempfile::TempDir,
    object: String,
    /// Each owner's stored bytes, which is exactly what verification
    /// hashes — taken from what was written, never recomputed.
    codes: std::collections::BTreeMap<String, Vec<u8>>,
    shapes: std::collections::BTreeMap<String, Vec<usize>>,
}

impl Built {
    fn container(&self) -> std::path::PathBuf {
        self.dir.path().join("container")
    }

    fn plan_and_store(&self) -> (ComponentOpPlan, OperandStore) {
        self.plan_and_store_trusting(RecognisedMethods::none())
    }

    fn plan_and_store_trusting(
        &self,
        recognised: RecognisedMethods,
    ) -> (ComponentOpPlan, OperandStore) {
        let inspection = inspect_container(&self.container(), false).unwrap();
        let plan = plan_component_ops(&inspection, &self.container(), "target")
            .unwrap()
            .plan
            .unwrap();
        let store = OperandStore::open(&self.container(), &inspection)
            .unwrap()
            .with_registry(registry())
            .with_recognised(recognised);
        (plan, store)
    }

    /// Write an attestation binding each owner's depth-0 extent to
    /// `radius`, against the bytes the container actually holds.
    fn attest(&self, radius: f64) {
        let table = RepresentationAttestations::new(
            OWNERS
                .iter()
                .map(|owner| StoredAttestation {
                    binding: AttestationBinding {
                        operand: OperandAddress::new(&self.object, *owner),
                        extent_depth: 0,
                        codec_family: OWNER_LABEL.into(),
                        codec_revision: 1,
                        shape: self.shapes[*owner].clone(),
                        content_digest: content_digest(&self.codes[*owner]),
                        source_digest: "sha256:checkpoint".into(),
                        auxiliary_baselines: std::collections::BTreeMap::from([(
                            CODEBOOK.to_string(),
                            terminal_baseline(COARSE_LABEL, 1),
                        )]),
                        recipe: "uniform-palette@256".into(),
                    },
                    method: AttestationMethod::new(AUTHORITY, StoredId::new(METHOD, 1)),
                    metric: StoredId::new("relative-rms", 1),
                    domain: StoredId::new("finite-normals", 1),
                    radius,
                })
                .collect(),
        );
        let container = self.container();
        std::fs::write(
            container.join(REPRESENTATION_ATTESTATIONS_JSON),
            serde_json::to_string_pretty(&table).unwrap(),
        )
        .unwrap();
        let index_path = container.join(INDEX_JSON);
        let mut index: Vindex3Index =
            serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
        index.representation_attestations = Some(REPRESENTATION_ATTESTATIONS_JSON.to_string());
        std::fs::write(&index_path, serde_json::to_string_pretty(&index).unwrap()).unwrap();
    }

    /// Plan trusting `AUTHORITY`, and hand back the owner's records.
    fn select_trusting(
        &self,
        budget: &ResidencyBudget,
        recognised: RecognisedMethods,
    ) -> Result<Vec<RealizationRecord>, String> {
        let (plan, store) = self.plan_and_store_trusting(recognised);
        select_realizations_within(
            &plan,
            crate::format::vindex3::opplan::exec::operands::OperandSource::from(&store),
            &ProductionBackend::new(),
            &ExecutionSlice::Full,
            budget,
        )
        .map_err(|e| e.to_string())
    }

    /// Plan under `budget`, with what the plan is priced at.
    fn select(
        &self,
        budget: &ResidencyBudget,
    ) -> Result<(Vec<RealizationRecord>, ResourceLedger), String> {
        let (plan, store) = self.plan_and_store();
        let records = select_realizations_within(
            &plan,
            crate::format::vindex3::opplan::exec::operands::OperandSource::from(&store),
            &ProductionBackend::new(),
            &ExecutionSlice::Full,
            budget,
        )
        .map_err(|e| e.to_string())?;
        let priced = expectations(&records, |o| store.stored_len(o), BlockGeometry::executor());
        let ledger = ResourceLedger::aggregate(&priced);
        Ok((records, ledger))
    }
}

/// A container whose owners are palette-coded and whose codebook is
/// stored under `book`.
fn build(book: &'static str) -> Built {
    let dir = tempfile::tempdir().unwrap();
    let checkpoint = dir.path().join("checkpoint");
    let container = dir.path().join("container");
    std::fs::create_dir_all(&checkpoint).unwrap();
    encode_fixture_container(dense_f32_model, &checkpoint, &container, "palette");

    let index_path = container.join(INDEX_JSON);
    let mut index: Vindex3Index =
        serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    let mut object = String::new();
    let mut codes_by_owner = std::collections::BTreeMap::new();
    let mut shapes_by_owner = std::collections::BTreeMap::new();

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
        let together: Vec<f32> = header
            .tensors
            .iter()
            .filter(|t| OWNERS.contains(&t.name.as_str()))
            .flat_map(&read)
            .collect();
        let palette = palette_over(&together);

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
            let codes: Vec<u8> = values.iter().map(|v| nearest(&palette, *v)).collect();
            let residual: Vec<u8> = values
                .iter()
                .zip(&codes)
                .map(|(v, c)| {
                    let step = (v - palette[usize::from(*c)]) / PALETTE_STEP;
                    step.round().clamp(-128.0, 127.0) as i8 as u8
                })
                .collect();
            planned.push(PlannedTensor {
                relative_name: t.name.clone(),
                source_name: t.name.clone(),
                dtype: OWNER_LABEL.to_string(),
                shape: t.shape.clone(),
                len: codes.len() as u64,
            });
            codes_by_owner.insert(t.name.clone(), codes.clone());
            shapes_by_owner.insert(t.name.clone(), t.shape.clone());
            bytes_by_name.insert(t.name.clone(), codes);
            let sibling = OperandStore::sibling_stream_tensor(&t.name, REFINE.name);
            planned.push(PlannedTensor {
                relative_name: sibling.clone(),
                source_name: sibling.clone(),
                dtype: REFINE_DTYPE.to_string(),
                shape: t.shape.clone(),
                len: residual.len() as u64,
            });
            bytes_by_name.insert(sibling, residual);
        }
        let bytes: Vec<u8> = palette.iter().flat_map(|v| v.to_le_bytes()).collect();
        planned.push(PlannedTensor {
            relative_name: CODEBOOK_TENSOR.to_string(),
            source_name: CODEBOOK_TENSOR.to_string(),
            dtype: book.to_string(),
            shape: vec![ENTRIES],
            len: bytes.len() as u64,
        });
        bytes_by_name.insert(CODEBOOK_TENSOR.to_string(), bytes);

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

    let table = AuxiliaryReferences::new(
        OWNERS
            .iter()
            .map(|owner| AuxiliaryReference {
                owner: OperandAddress::new(&object, *owner),
                auxiliary: CODEBOOK.to_string(),
                target: OperandAddress::new(&object, CODEBOOK_TENSOR),
            })
            .collect(),
    );
    std::fs::write(
        container.join(AUXILIARY_REFERENCES_JSON),
        serde_json::to_string_pretty(&table).unwrap(),
    )
    .unwrap();
    index.auxiliary_references = Some(AUXILIARY_REFERENCES_JSON.to_string());
    std::fs::write(&index_path, serde_json::to_string_pretty(&index).unwrap()).unwrap();

    Built {
        dir,
        object,
        codes: codes_by_owner,
        shapes: shapes_by_owner,
    }
}

/// An evenly spaced palette over the values' range.
fn palette_over(values: &[f32]) -> Vec<f32> {
    let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let step = (hi - lo) / (ENTRIES - 1) as f32;
    (0..ENTRIES).map(|i| lo + step * i as f32).collect()
}

fn nearest(palette: &[f32], value: f32) -> u8 {
    (0..ENTRIES)
        .min_by(|a, b| {
            (palette[*a] - value)
                .abs()
                .total_cmp(&(palette[*b] - value).abs())
        })
        .unwrap_or(0) as u8
}

/// The owner's shallow option, as the planner built it.
fn shallow(
    records: &[RealizationRecord],
) -> &crate::format::vindex3::opplan::exec::realization::ExtentOption {
    records
        .iter()
        .find(|r| r.representation == OWNER_LABEL)
        .expect("the projections are the owner's")
        .extent
        .options
        .iter()
        .find(|o| o.certificate.extent == RepresentationExtent::BASE)
        .expect("the owner declares a shallow extent")
}

// ── The regression ───────────────────────────────────────────────────

/// **The decisive one.** The floor's answer moves on the composition, in
/// the planner, with one artifact and one floor.
#[test]
fn the_floor_judges_the_composed_bound_and_not_the_codecs_declared_one() {
    let terminal = RepresentationExtent::at_depth(1);
    let floor = RepresentationFloor::RelativeRms(FLOOR);

    let (coarse, _) = build(COARSE_LABEL)
        .select(&ResidencyBudget::UNBOUNDED)
        .unwrap();
    let option = shallow(&coarse);
    let radius = option
        .certificate
        .radius
        .as_ref()
        .expect("the composition is available");
    assert!(
        (radius.radius() - (PARENT + COARSE)).abs() < 1e-12,
        "expected the composed bound, got {radius:?}"
    );
    assert!(
        !floor.admits(option, terminal),
        "0.006 must not satisfy a 0.005 floor"
    );

    let (fine, _) = build(FINE_LABEL)
        .select(&ResidencyBudget::UNBOUNDED)
        .unwrap();
    let option = shallow(&fine);
    let radius = option
        .certificate
        .radius
        .as_ref()
        .expect("the composition is available");
    assert!(
        (radius.radius() - (PARENT + FINE)).abs() < 1e-12,
        "expected the composed bound, got {radius:?}"
    );
    assert!(
        floor.admits(option, terminal),
        "0.004 must satisfy a 0.005 floor"
    );
}

/// The over-promise, stated as a fact: the codec's DECLARED bound is the
/// same in both containers and inside the floor in both, so a planner
/// reading it — which is what this build did until carriage — would have
/// admitted the coarse artifact too.
#[test]
fn the_declared_bound_alone_would_have_admitted_both() {
    let declared = registry()
        .by_label(OWNER_LABEL)
        .expect("registered")
        .extents()
        .into_iter()
        .find(|c| c.extent == RepresentationExtent::BASE)
        .and_then(|c| c.radius)
        .expect("the owner declares a radius at depth 0");
    assert!(
        (declared.radius() - PARENT).abs() < 1e-12,
        "the declaration is the same whatever the codebook"
    );
    assert!(
        declared.radius() <= FLOOR,
        "which is why reading it alone was an over-promise"
    );
}

/// And the end of it: under a preparation budget the terminal extent
/// cannot meet, the fine container gives up depth and the coarse one is
/// REFUSED — because the only extent that would have saved anything is
/// one its composed bound does not admit.
#[test]
fn a_budget_may_only_spend_fidelity_the_composition_still_admits() {
    let fine = build(FINE_LABEL);
    let (whole_records, whole) = fine.select(&ResidencyBudget::UNBOUNDED).unwrap();
    for record in whole_records
        .iter()
        .filter(|r| r.representation == OWNER_LABEL)
    {
        assert_eq!(
            record.extent.selected,
            RepresentationExtent::at_depth(1),
            "an unbudgeted plan pins the whole artifact"
        );
    }

    // A deficit ONE owner's move can cover, so the arms differ in whether
    // the move is permitted and not in whether it would have been enough.
    let saving = whole_records
        .iter()
        .filter(|r| r.representation == OWNER_LABEL)
        .map(|r| r.planned.operand.shape.iter().product::<usize>() as u64)
        .max()
        .expect("the owner is in the plan");
    let budget = ResidencyBudget::UNBOUNDED
        .with_prepare_bytes(whole.read_to_prepare - saving)
        .with_fidelity(RepresentationFloor::RelativeRms(FLOOR));

    let (shallower, priced) = fine.select(&budget).expect("0.004 is inside the floor");
    assert!(
        shallower
            .iter()
            .filter(|r| r.representation == OWNER_LABEL)
            .any(|r| r.extent.selected == RepresentationExtent::BASE),
        "the fine container may spend the depth"
    );
    assert!(
        priced.read_to_prepare < whole.read_to_prepare,
        "and spending it opened less"
    );

    // The same budget, the same owner codec, the same declared radius —
    // and a refusal, because the only extent that would have saved
    // anything is one the COMPOSED bound does not admit.
    let refusal = build(COARSE_LABEL)
        .select(&budget)
        .expect_err("0.006 is outside the floor, so there is nothing to give up");
    assert!(
        refusal.contains("relative RMS at or under 5.000e-3"),
        "the refusal names the floor it could not meet: {refusal}"
    );
    assert!(
        !refusal.contains("extent depth 1 → 0"),
        "and no shallower extent was ever taken: {refusal}"
    );
}

/// The wave's exit claim, in the planner: a MEASURED error on this
/// instance reaches the floor that gates selection.
///
/// The same coarse codebook whose composition the floor refuses above.
/// Nothing about the artifact changed except that an encoder measured it
/// and said so: 0.001 against the scheme's conservative 0.003, composed
/// with the codebook's 0.003, is 0.004 and inside the floor. A lossy
/// artifact satisfies a quality requirement without anyone inventing a
/// number.
#[test]
fn a_verified_measurement_replaces_the_declared_bound_and_reaches_the_floor() {
    let built = build(COARSE_LABEL);
    built.attest(ATTESTED);
    let trusting = RecognisedMethods::none()
        .with_authority(AUTHORITY)
        .with_method(METHOD, 1);

    let records = built
        .select_trusting(&ResidencyBudget::UNBOUNDED, trusting)
        .unwrap();
    let option = shallow(&records);
    let radius = option
        .certificate
        .radius
        .as_ref()
        .expect("the measurement composed");
    assert!(
        (radius.radius() - (ATTESTED + COARSE)).abs() < 1e-12,
        "the attested claim REPLACES the declared one rather than widening it: {radius:?}"
    );
    assert!(
        radius.radius() < PARENT + COARSE,
        "and it is the better of the two, which is the whole point"
    );
    assert!(
        RepresentationFloor::RelativeRms(FLOOR).admits(option, RepresentationExtent::at_depth(1)),
        "0.004 satisfies a 0.005 floor"
    );
}

/// And the control that makes the arm above mean something: the same
/// container, the same attestation, the same bytes — read by a build that
/// recognises nobody.
///
/// The guarantee is UNAVAILABLE, not optimistic: selection falls back to
/// what the codec declares, the composition is 0.006, and the floor
/// refuses it. An attestation is carried and checked either way; whether
/// it may influence a decision is a trust question, and the container
/// making the claim does not get to answer it.
#[test]
fn an_unrecognised_measurement_leaves_the_declared_bound_standing() {
    let built = build(COARSE_LABEL);
    built.attest(ATTESTED);

    let records = built
        .select_trusting(&ResidencyBudget::UNBOUNDED, RecognisedMethods::none())
        .unwrap();
    let option = shallow(&records);
    let radius = option
        .certificate
        .radius
        .as_ref()
        .expect("the declared bound still composes");
    assert!(
        (radius.radius() - (PARENT + COARSE)).abs() < 1e-12,
        "an unrecognised measurement must not be acted on: {radius:?}"
    );
    assert!(
        !RepresentationFloor::RelativeRms(FLOOR).admits(option, RepresentationExtent::at_depth(1)),
        "0.006 is refused exactly as if nothing had been attested"
    );
}

/// Verification is not free, and the reads are real: hashing the owners'
/// payloads moves the store's consumption ledger.
///
/// Priced honestly rather than absorbed — a plan that admits nothing
/// reads nothing, and one that admits two operands reads both. (The
/// `ResourceLedger`'s own `prepare` figure is computed from declared
/// extents and does not yet include these; that is debt D1, named by the
/// freeze and not paid here.)
#[test]
fn verifying_evidence_is_recorded_as_preparation_work() {
    let built = build(COARSE_LABEL);
    built.attest(ATTESTED);
    let trusting = RecognisedMethods::none()
        .with_authority(AUTHORITY)
        .with_method(METHOD, 1);

    let (plan, store) = built.plan_and_store_trusting(trusting);
    let before = store.load_count();
    select_realizations_within(
        &plan,
        crate::format::vindex3::opplan::exec::operands::OperandSource::from(&store),
        &ProductionBackend::new(),
        &ExecutionSlice::Full,
        &ResidencyBudget::UNBOUNDED,
    )
    .unwrap();
    assert_eq!(
        store.load_count() - before,
        OWNERS.len() as u64,
        "one payload read per attested operand, and not one more"
    );

    // The same plan with nothing attested opens nothing to plan it.
    let quiet = build(COARSE_LABEL);
    let (plan, store) = quiet.plan_and_store();
    let before = store.load_count();
    select_realizations_within(
        &plan,
        crate::format::vindex3::opplan::exec::operands::OperandSource::from(&store),
        &ProductionBackend::new(),
        &ExecutionSlice::Full,
        &ResidencyBudget::UNBOUNDED,
    )
    .unwrap();
    assert_eq!(store.load_count(), before, "planning read nothing");
}
