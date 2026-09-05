//! Rung 3d's extensibility proof at the CODEC plane: a crate that is not
//! `larql-vindex` registers a representation this build does not ship
//! and executes a container stored in it — bit-exact to the same bytes
//! under the shipped label — without editing the planner's or the
//! executor's selection. An integration test is a separate crate in
//! cargo's model, so this file sees only what `larql-vindex` exports;
//! anything the proof had needed from the crate's internals would have
//! been a leak in the contract, and it needed nothing.
//!
//! `F32X` is f32 under another name. Its bytes are the fixture's own f32
//! image, so the candidate container differs from the control ONLY in
//! the label of its projection tensors, and every difference in what the
//! executor does is the registry's doing: the same selection, the same
//! logits, and — with the provider absent — a refusal by name before any
//! byte, plus the invalidation of an image prepared while it was present.
//!
//! The forecast this answers is `docs/represent/forecasts/rung3-planned-realizations.json`.

use std::ops::Range;
use std::path::Path;

use larql_vindex::error::VindexError;
use larql_vindex::format::filenames::INDEX_JSON;
use larql_vindex::format::vindex3::encode::segment::{
    read_segment_header, write_segment, PlannedTensor,
};
use larql_vindex::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use larql_vindex::format::vindex3::index::Vindex3Index;
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::cpu::physical::PhysicalProjectionPlan;
use larql_vindex::format::vindex3::opplan::exec::execute_plan;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prepared::{ExecutionSlice, PreparedOperands};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::realization::RealizationRecord;
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use larql_vindex::format::vindex3::represent::codec::codecs::{
    bf16_zlib, float, kquant, mxfp4, nvfp4,
};
use larql_vindex::format::vindex3::represent::codec::streams::VALUES;
use larql_vindex::format::vindex3::represent::codec::{
    Acceleration, AccessGranularity, CodecCapabilities, CodecError, CodecOperands, CodecRegistry,
    ExtentCertificate, RepresentationCodec, RepresentationExtent, ResidencyProfile, StreamSpec,
};
use larql_vindex::format::vindex3::represent::nvfp4_pack::CodecIdentity;

/// The label this crate registers and `larql-vindex` does not.
const LABEL: &str = "F32X";
/// The shipped label the same bytes carry in the control container.
const SHIPPED_LABEL: &str = "F32";
const F32_WIDTH: usize = std::mem::size_of::<f32>();
const F32_BITS: f64 = 32.0;
/// Prompt over the dense fixture's 128-token vocabulary — the rung-2
/// witness's prompt, so the two proofs read the same execution.
const TOKENS: [u32; 5] = [3, 17, 28, 0, 11];
/// The projections the candidate relabels.
const PROJECTION_SUFFIX: &str = "_proj.weight";

// ── The external provider ────────────────────────────────────────────

/// f32 under a label of its own: one row-major little-endian stream,
/// element-random access, a direct BLAS realization declared exactly as
/// the shipped f32 codec declares it.
struct ExternalF32;

impl ExternalF32 {
    fn rows_and_k(shape: &[usize], tensor: &str) -> Result<(usize, usize), CodecError> {
        match shape {
            [rows, rest @ ..] => Ok((*rows, rest.iter().product::<usize>().max(1))),
            [] => Err(CodecError::Geometry {
                tensor: tensor.into(),
                label: LABEL.into(),
                shape: shape.to_vec(),
                why: "a scalar has no rows".into(),
            }),
        }
    }
}

impl RepresentationCodec for ExternalF32 {
    fn encoding_label(&self) -> &'static str {
        LABEL
    }

    fn identity(&self) -> CodecIdentity {
        CodecIdentity {
            family: "external-f32x".into(),
            revision: 1,
            group_elems: 1,
            element: "f32".into(),
            group_scale: "none".into(),
            tensor_scale: "none".into(),
            layout: "row-major-le".into(),
        }
    }

    fn streams(&self) -> &'static [StreamSpec] {
        &[VALUES]
    }

    fn capabilities(&self) -> CodecCapabilities {
        CodecCapabilities {
            access: AccessGranularity::ElementRandom,
            group_elems: 1,
            row_align_elems: 1,
            physical_align_bytes: F32_WIDTH,
        }
    }

    fn extents(&self) -> Vec<ExtentCertificate> {
        vec![ExtentCertificate::terminal(F32_BITS)]
    }

    fn stored_bytes(
        &self,
        shape: &[usize],
        _: RepresentationExtent,
        tensor: &str,
    ) -> Result<u64, CodecError> {
        let (rows, k) = Self::rows_and_k(shape, tensor)?;
        Ok((rows * k * F32_WIDTH) as u64)
    }

    fn validate(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        let need = self.stored_bytes(shape, extent, tensor)? as usize;
        operands.stream_of_len(VALUES, need, LABEL, tensor)?;
        Ok(())
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
        let (_, k) = Self::rows_and_k(shape, tensor)?;
        let bytes = operands.stream_of_len(VALUES, rows.end * k * F32_WIDTH, LABEL, tensor)?;
        let region = &bytes[rows.start * k * F32_WIDTH..rows.end * k * F32_WIDTH];
        if dst.len() != rows.len() * k {
            return Err(CodecError::Destination {
                tensor: tensor.into(),
                need: rows.len() * k,
                have: dst.len(),
            });
        }
        for (out, chunk) in dst.iter_mut().zip(region.chunks_exact(F32_WIDTH)) {
            *out = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(())
    }

    fn accelerations(&self) -> Vec<Acceleration> {
        let stored = ResidencyProfile::stored(F32_BITS);
        vec![
            Acceleration::cpu(PhysicalProjectionPlan::BlasF32, stored),
            Acceleration::cpu(PhysicalProjectionPlan::ScalarF32, stored),
        ]
    }

    fn decode_residency(&self) -> ResidencyProfile {
        ResidencyProfile::DECODED_F32
    }
}

/// Every shipped codec plus the external one — what a build that linked
/// this crate as a provider would register.
fn registry_with_provider() -> &'static CodecRegistry {
    let registry = CodecRegistry::new()
        .register(Box::new(float::BF16))
        .and_then(|r| r.register(Box::new(float::F16)))
        .and_then(|r| r.register(Box::new(float::F32)))
        .and_then(|r| r.register(Box::new(kquant::Q4_K)))
        .and_then(|r| r.register(Box::new(kquant::Q6_K)))
        .and_then(|r| r.register(Box::new(kquant::Q8_0)))
        .and_then(|r| r.register(Box::new(nvfp4::NVFP4)))
        .and_then(|r| r.register(Box::new(mxfp4::MXFP4)))
        .and_then(|r| r.register(Box::new(bf16_zlib::BF16_ZLIB)))
        .and_then(|r| r.register(Box::new(ExternalF32)))
        .expect("nine shipped labels and one new one");
    Box::leak(Box::new(registry))
}

// ── Two containers, one image ────────────────────────────────────────

/// Relabel every tensor `select` names from `F32` to `LABEL`, bytes
/// untouched: the segment is rewritten so its header, lengths and
/// checksums are what the container records for the new label.
fn relabel(root: &Path, select: impl Fn(&str, &[usize]) -> bool) -> Vec<(String, String, usize)> {
    let index_path = root.join(INDEX_JSON);
    let mut index: Vindex3Index =
        serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    let mut out = Vec::new();
    for entry in index.representations.values_mut() {
        let path = root.join(&entry.segment);
        let (header, payload_start) = read_segment_header(&path).unwrap();
        if !header.tensors.iter().any(|t| select(&t.name, &t.shape)) {
            continue;
        }
        let file = std::fs::read(&path).unwrap();
        let payload = &file[payload_start as usize..];
        let mut planned = Vec::new();
        let mut bytes_by_name = std::collections::BTreeMap::new();
        for t in &header.tensors {
            let stored = payload[t.offset as usize..(t.offset + t.len) as usize].to_vec();
            let dtype = if select(&t.name, &t.shape) {
                assert_eq!(t.dtype, SHIPPED_LABEL, "the fixture stores f32");
                out.push((entry.object.clone(), t.name.clone(), stored.len()));
                LABEL.to_string()
            } else {
                t.dtype.clone()
            };
            planned.push(PlannedTensor {
                relative_name: t.name.clone(),
                source_name: t.name.clone(),
                dtype,
                shape: t.shape.clone(),
                len: stored.len() as u64,
            });
            bytes_by_name.insert(t.name.clone(), stored);
        }
        let written = write_segment(&path, &header.representation, planned, |name, w, hash| {
            let bytes = &bytes_by_name[name];
            w.write_all(bytes).map_err(VindexError::Io)?;
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
    out
}

fn is_projection(name: &str, shape: &[usize]) -> bool {
    shape.len() == 2 && name.ends_with(PROJECTION_SUFFIX)
}

struct Container {
    _src: tempfile::TempDir,
    dir: tempfile::TempDir,
    /// (object, tensor, stored bytes) of every relabelled tensor.
    relabelled: Vec<(String, String, usize)>,
}

impl Container {
    fn build(relabelled: bool) -> Self {
        let src = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        encode_fixture_container(dense_f32_model, src.path(), dir.path(), "dense");
        let relabelled = if relabelled {
            let done = relabel(dir.path(), is_projection);
            assert!(!done.is_empty(), "the dense fixture has projections");
            done
        } else {
            Vec::new()
        };
        Self {
            _src: src,
            dir,
            relabelled,
        }
    }

    fn open(&self, registry: &'static CodecRegistry) -> (ComponentOpPlan, OperandStore) {
        let inspection = inspect_container(self.dir.path(), false).unwrap();
        let plan = plan_component_ops(&inspection, self.dir.path(), "target")
            .unwrap()
            .plan
            .expect("the dense fixture plans");
        let store = OperandStore::open(self.dir.path(), &inspection)
            .unwrap()
            .with_registry(registry);
        (plan, store)
    }

    fn logits_and_hidden(&self, registry: &'static CodecRegistry) -> (Vec<u32>, Vec<u32>) {
        let (plan, store) = self.open(registry);
        let trace = execute_plan(&plan, &store, &TOKENS, &ProductionBackend::new()).unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        let hidden = bits(trace.final_hidden());
        let logits = bits(&trace.logits.expect("the dense fixture carries a head"));
        (logits, hidden)
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

/// What the selector decided for one operand — everything in the record
/// except the representation and its provider, which are the two facts
/// the relabel changes on purpose.
fn decision(r: &RealizationRecord) -> (String, String, String, String) {
    (
        r.planned.operand.tensor.clone(),
        format!("{:?}", r.planned.operation),
        format!("{:?}", r.selection.realization),
        format!("{:?}", r.selection.reason),
    )
}

// ── The proof ────────────────────────────────────────────────────────

/// Registration alone: the candidate executes bit-exact to the control,
/// and every selection the planner made is the one it made for the
/// shipped label — only the record's representation and provider differ.
#[test]
fn an_external_codec_executes_through_registration_alone_with_the_shipped_selection() {
    let registry = registry_with_provider();
    let control = Container::build(false);
    let candidate = Container::build(true);
    assert_eq!(
        control.logits_and_hidden(registry),
        candidate.logits_and_hidden(registry),
        "the same bytes under the external label compute the same bits"
    );

    let shipped = control.prepared(registry).unwrap();
    let external = candidate.prepared(registry).unwrap();
    let shipped_decisions: Vec<_> = shipped.realizations().iter().map(decision).collect();
    let external_decisions: Vec<_> = external.realizations().iter().map(decision).collect();
    assert_eq!(
        shipped_decisions, external_decisions,
        "selection is unchanged"
    );

    let relabelled: std::collections::BTreeSet<&str> = candidate
        .relabelled
        .iter()
        .map(|(_, t, _)| t.as_str())
        .collect();
    let external_records: Vec<_> = external
        .realizations()
        .iter()
        .filter(|r| relabelled.contains(r.planned.operand.tensor.as_str()))
        .collect();
    assert_eq!(external_records.len(), candidate.relabelled.len());
    for r in &external_records {
        assert_eq!(r.representation, LABEL, "{r:?}");
        assert_eq!(r.provider.as_ref(), Some(&ExternalF32.identity()), "{r:?}");
    }
    // And the bytes really are the shipped image: each relabelled tensor
    // is stored at exactly f32's width.
    for (_, tensor, stored) in &candidate.relabelled {
        let r = external_records
            .iter()
            .find(|r| &r.planned.operand.tensor == tensor)
            .unwrap();
        assert_eq!(*stored, r.planned.logical_elements * F32_WIDTH, "{tensor}");
    }
}

/// Without the provider the same container is refused by name before any
/// byte is read, and an image prepared while the provider was registered
/// is invalidated — never executed — once it is gone.
#[test]
fn without_the_provider_the_candidate_is_refused_by_name_and_a_prepared_image_is_invalidated() {
    let candidate = Container::build(true);
    let Err(err) = candidate.prepared(CodecRegistry::builtin()) else {
        panic!("the built-in registry does not know the label");
    };
    let err = err.to_string();
    assert!(err.contains("unregistered representation"), "{err}");
    assert!(err.contains(LABEL), "{err}");
    assert!(
        err.contains(SHIPPED_LABEL),
        "the refusal names the registered ones: {err}"
    );

    let prepared = candidate.prepared(registry_with_provider()).unwrap();
    let Err(err) = prepared.ensure_providers_in(CodecRegistry::builtin()) else {
        panic!("the provider is gone");
    };
    let err = err.to_string();
    assert!(err.contains(&format!("`{LABEL}`")), "{err}");
    assert!(err.contains("no registered codec"), "{err}");
    assert!(err.contains("re-prepare"), "{err}");
    // The control never needed the provider, and is untouched by its loss.
    Container::build(false)
        .prepared(CodecRegistry::builtin())
        .unwrap()
        .ensure_providers_in(CodecRegistry::builtin())
        .unwrap();
}
