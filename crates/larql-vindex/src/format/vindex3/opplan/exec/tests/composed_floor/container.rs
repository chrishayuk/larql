//! One container, built through the same path a real encoder takes, with
//! the codebook stored at one fidelity or the other and an attestation
//! written only when an arm asks for one.

use super::codecs::registry;
use super::vocabulary::*;
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
use crate::format::vindex3::opplan::exec::accounting::{
    expectations, BlockGeometry, ResidencyBudget, ResourceLedger,
};
use crate::format::vindex3::opplan::exec::operands::{OperandSource, OperandStore};
use crate::format::vindex3::opplan::exec::prepared::{select_realizations_within, ExecutionSlice};
use crate::format::vindex3::opplan::exec::production::ProductionBackend;
use crate::format::vindex3::opplan::exec::realization::RealizationRecord;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use crate::format::vindex3::represent::codec::RepresentationExtent;
use crate::format::vindex3::representation_attestations::recognition::RecognisedMethods;
use crate::format::vindex3::representation_attestations::{
    content_digest, terminal_baseline, AttestationBinding, AttestationMethod,
    RepresentationAttestations, StoredAttestation, StoredId,
};

pub(super) struct Built {
    dir: tempfile::TempDir,
    object: String,
    /// Each owner's stored bytes, which is exactly what verification
    /// hashes — taken from what was written, never recomputed.
    codes: std::collections::BTreeMap<String, Vec<u8>>,
    shapes: std::collections::BTreeMap<String, Vec<usize>>,
}

impl Built {
    pub(super) fn container(&self) -> std::path::PathBuf {
        self.dir.path().join("container")
    }

    pub(super) fn plan_and_store(&self) -> (ComponentOpPlan, OperandStore) {
        self.plan_and_store_trusting(RecognisedMethods::none())
    }

    pub(super) fn plan_and_store_trusting(
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
    pub(super) fn attest(&self, radius: f64) {
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
    pub(super) fn select_trusting(
        &self,
        budget: &ResidencyBudget,
        recognised: RecognisedMethods,
    ) -> Result<Vec<RealizationRecord>, String> {
        let (plan, store) = self.plan_and_store_trusting(recognised);
        select_realizations_within(
            &plan,
            OperandSource::from(&store),
            &ProductionBackend::new(),
            &ExecutionSlice::Full,
            budget,
        )
        .map_err(|e| e.to_string())
    }

    /// Plan under `budget`, with what the plan is priced at.
    pub(super) fn select(
        &self,
        budget: &ResidencyBudget,
    ) -> Result<(Vec<RealizationRecord>, ResourceLedger), String> {
        let (plan, store) = self.plan_and_store();
        let records = select_realizations_within(
            &plan,
            OperandSource::from(&store),
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
pub(super) fn build(book: &'static str) -> Built {
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
pub(super) fn shallow(
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
