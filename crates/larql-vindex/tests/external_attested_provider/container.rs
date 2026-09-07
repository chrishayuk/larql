//! One container written through exported API only, plus the two sidecars
//! the plane depends on: the reference table that binds the role to an
//! object, and the attestation that measures the instance.

use std::collections::BTreeMap;
use std::path::PathBuf;

use larql_vindex::format::filenames::{
    AUXILIARY_REFERENCES_JSON, INDEX_JSON, REPRESENTATION_ATTESTATIONS_JSON,
};
use larql_vindex::format::vindex3::auxiliary_references::{
    AuxiliaryReference, AuxiliaryReferences, OperandAddress,
};
use larql_vindex::format::vindex3::encode::segment::{
    read_segment_header, write_segment, PlannedTensor, SegmentTensor,
};
use larql_vindex::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use larql_vindex::format::vindex3::index::Vindex3Index;
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::accounting::{
    expectations, BlockGeometry, ResidencyBudget, ResourceLedger,
};
use larql_vindex::format::vindex3::opplan::exec::operands::{OperandSource, OperandStore};
use larql_vindex::format::vindex3::opplan::exec::prepared::{
    select_realizations_within, ExecutionSlice, PreparedOperands,
};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::realization::RealizationRecord;
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use larql_vindex::format::vindex3::represent::codec::CodecRegistry;
use larql_vindex::format::vindex3::representation_attestations::recognition::RecognisedMethods;
use larql_vindex::format::vindex3::representation_attestations::{
    content_digest, terminal_baseline, AttestationBinding, AttestationMethod,
    RepresentationAttestations, StoredAttestation, StoredId,
};

use crate::provider::*;

/// The tensors stored under the external parent representation.
pub const OWNERS: [&str; 2] = ["0.mlp.gate_proj.weight", "0.mlp.up_proj.weight"];
/// Where the anchor table is written — named nothing like its owners,
/// because a dependency is ADDRESSED and never spelled.
pub const ANCHOR_TENSOR: &str = "acme.shared.anchor_table";

/// How the attestation should relate to the bytes actually stored.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Attest {
    /// No sidecar at all.
    None,
    /// Bound to the bytes that are there.
    Truthful,
    /// Bound to bytes that are NOT there — the payload check must catch
    /// it, and nothing before the payload check can.
    Tampered,
}

pub struct Built {
    dir: tempfile::TempDir,
}

impl Built {
    pub fn container(&self) -> PathBuf {
        self.dir.path().join("container")
    }

    pub fn open(
        &self,
        registry: &'static CodecRegistry,
        recognised: RecognisedMethods,
    ) -> (ComponentOpPlan, OperandStore) {
        let inspection = inspect_container(&self.container(), false).unwrap();
        let plan = plan_component_ops(&inspection, &self.container(), "target")
            .unwrap()
            .plan
            .unwrap();
        let store = OperandStore::open(&self.container(), &inspection)
            .unwrap()
            .with_registry(registry)
            .with_recognised(recognised);
        (plan, store)
    }

    /// Plan under `budget`, trusting `recognised`.
    pub fn select(
        &self,
        registry: &'static CodecRegistry,
        recognised: RecognisedMethods,
        budget: &ResidencyBudget,
    ) -> Result<(Vec<RealizationRecord>, ResourceLedger), String> {
        let (plan, store) = self.open(registry, recognised);
        let records = select_realizations_within(
            &plan,
            OperandSource::from(&store),
            &ProductionBackend::new(),
            &ExecutionSlice::Full,
            budget,
        )
        .map_err(|e| e.to_string())?;
        let priced = expectations(&records, |o| store.stored_len(o), BlockGeometry::executor());
        Ok((records, ResourceLedger::aggregate(&priced)))
    }

    pub fn prepared(&self, registry: &'static CodecRegistry) -> Result<PreparedOperands, String> {
        let (plan, store) = self.open(registry, recognising());
        PreparedOperands::load(
            &plan,
            &store,
            &ProductionBackend::new(),
            ExecutionSlice::Full,
        )
        .map_err(|e| e.to_string())
    }
}

/// A reader that takes this authority's word, by this exact method.
pub fn recognising() -> RecognisedMethods {
    RecognisedMethods::none()
        .with_authority(AUTHORITY)
        .with_method(METHOD, METHOD_VERSION)
}

/// An evenly spaced anchor table over the values' range.
fn anchors_over(values: &[f32]) -> Vec<f32> {
    let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let step = (hi - lo) / (ENTRIES - 1) as f32;
    (0..ENTRIES).map(|i| lo + step * i as f32).collect()
}

fn nearest(anchors: &[f32], value: f32) -> u8 {
    (0..ENTRIES)
        .min_by(|a, b| {
            (anchors[*a] - value)
                .abs()
                .total_cmp(&(anchors[*b] - value).abs())
        })
        .unwrap_or(0) as u8
}

pub fn build(attest: Attest) -> Built {
    let dir = tempfile::tempdir().unwrap();
    let checkpoint = dir.path().join("checkpoint");
    let container = dir.path().join("container");
    std::fs::create_dir_all(&checkpoint).unwrap();
    encode_fixture_container(dense_f32_model, &checkpoint, &container, "acme");

    let index_path = container.join(INDEX_JSON);
    let mut index: Vindex3Index =
        serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    let mut object = String::new();
    let mut codes_by_owner: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    let mut shapes_by_owner: BTreeMap<String, Vec<usize>> = BTreeMap::new();

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
        let read = |t: &SegmentTensor| -> Vec<f32> {
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
        let anchors = anchors_over(&together);

        let mut planned = Vec::new();
        let mut bytes_by_name: BTreeMap<String, Vec<u8>> = BTreeMap::new();
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
            let codes: Vec<u8> = values.iter().map(|v| nearest(&anchors, *v)).collect();
            let residual: Vec<u8> = values
                .iter()
                .zip(&codes)
                .map(|(v, c)| {
                    let step = (v - anchors[usize::from(*c)]) / LATTICE_STEP;
                    step.round().clamp(-128.0, 127.0) as i8 as u8
                })
                .collect();
            planned.push(PlannedTensor {
                relative_name: t.name.clone(),
                source_name: t.name.clone(),
                dtype: LATTICE.to_string(),
                shape: t.shape.clone(),
                len: codes.len() as u64,
            });
            codes_by_owner.insert(t.name.clone(), codes.clone());
            shapes_by_owner.insert(t.name.clone(), t.shape.clone());
            bytes_by_name.insert(t.name.clone(), codes);
            let sibling = OperandStore::sibling_stream_tensor(&t.name, "acme_residual");
            planned.push(PlannedTensor {
                relative_name: sibling.clone(),
                source_name: sibling.clone(),
                dtype: REFINE_DTYPE.to_string(),
                shape: t.shape.clone(),
                len: residual.len() as u64,
            });
            bytes_by_name.insert(sibling, residual);
        }
        let bytes: Vec<u8> = anchors.iter().flat_map(|v| v.to_le_bytes()).collect();
        planned.push(PlannedTensor {
            relative_name: ANCHOR_TENSOR.to_string(),
            source_name: ANCHOR_TENSOR.to_string(),
            dtype: ANCHORS.to_string(),
            shape: vec![ENTRIES],
            len: bytes.len() as u64,
        });
        bytes_by_name.insert(ANCHOR_TENSOR.to_string(), bytes);

        let written = write_segment(&path, &header.representation, planned, |name, w, hash| {
            let bytes = &bytes_by_name[name];
            std::io::Write::write_all(w, bytes).map_err(larql_vindex::error::VindexError::Io)?;
            hash(bytes);
            Ok(bytes.len() as u64)
        })
        .unwrap();
        entry.payload_bytes = written.payload_bytes;
        entry.payload_sha256 = written.payload_sha256;
        entry.segment_sha256 = written.segment_sha256;
        entry.tensor_count = written.tensor_count;
    }
    assert_eq!(codes_by_owner.len(), OWNERS.len(), "both owners rewritten");

    let table = AuxiliaryReferences::new(
        OWNERS
            .iter()
            .map(|owner| AuxiliaryReference {
                owner: OperandAddress::new(&object, *owner),
                auxiliary: ANCHOR_ROLE.to_string(),
                target: OperandAddress::new(&object, ANCHOR_TENSOR),
            })
            .collect(),
    );
    std::fs::write(
        container.join(AUXILIARY_REFERENCES_JSON),
        serde_json::to_string_pretty(&table).unwrap(),
    )
    .unwrap();
    index.auxiliary_references = Some(AUXILIARY_REFERENCES_JSON.to_string());

    if attest != Attest::None {
        let attestations = RepresentationAttestations::new(
            OWNERS
                .iter()
                .map(|owner| {
                    let bytes = &codes_by_owner[*owner];
                    // The tampered arm binds to bytes that differ in ONE
                    // position: same length, same shape, same codec, same
                    // baselines — so every metadata check still passes and
                    // only the payload can catch it.
                    let digest = match attest {
                        Attest::Tampered => {
                            let mut altered = bytes.clone();
                            altered[bytes.len() / 2] ^= 0x01;
                            content_digest(&altered)
                        }
                        _ => content_digest(bytes),
                    };
                    StoredAttestation {
                        binding: AttestationBinding {
                            operand: OperandAddress::new(&object, *owner),
                            extent_depth: 0,
                            codec_family: LATTICE.into(),
                            codec_revision: 1,
                            shape: shapes_by_owner[*owner].clone(),
                            content_digest: digest,
                            source_digest: "sha256:acme-checkpoint".into(),
                            auxiliary_baselines: BTreeMap::from([(
                                ANCHOR_ROLE.to_string(),
                                terminal_baseline(ANCHORS, 1),
                            )]),
                            recipe: "acme-lattice-fit@256".into(),
                        },
                        method: AttestationMethod::new(
                            AUTHORITY,
                            StoredId::new(METHOD, METHOD_VERSION),
                        ),
                        metric: StoredId::new("relative-rms", 1),
                        domain: StoredId::new("finite-normals", 1),
                        radius: ATTESTED,
                    }
                })
                .collect(),
        );
        std::fs::write(
            container.join(REPRESENTATION_ATTESTATIONS_JSON),
            serde_json::to_string_pretty(&attestations).unwrap(),
        )
        .unwrap();
        index.representation_attestations = Some(REPRESENTATION_ATTESTATIONS_JSON.to_string());
    }
    std::fs::write(&index_path, serde_json::to_string_pretty(&index).unwrap()).unwrap();

    Built { dir }
}
