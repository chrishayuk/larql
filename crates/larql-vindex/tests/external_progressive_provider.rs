//! PROGRESSIVE-1's extensibility proof: a crate that is not
//! `larql-vindex` registers a PROGRESSIVE representation this build does
//! not ship — its own streams, its own depths, its own certificates — and
//! the planner reaches those extents, prices them, pins one under a
//! budget and a fidelity floor, and executes the result bit-exactly.
//!
//! An integration test is a separate crate in cargo's model, so this file
//! sees only what `larql-vindex` exports. Nothing under `src/` was edited
//! for the provider: no match arm names it, no planner branch knows it
//! offers two extents, and the loader opens its second stream because the
//! codec DECLARES a refinement, not because anything here taught it what
//! a refinement is.
//!
//! `PLANES16` is deliberately not the shipped `F32_PLANES`: two planes
//! rather than three, its own stream names, its own radius. If the proof
//! only worked for the codec the core happens to ship, it would be
//! proving the core.
//!
//! The forecast this answers is
//! `docs/represent/forecasts/represent-progressive-1.json` (P6).

use std::ops::Range;
use std::path::Path;

use larql_vindex::format::filenames::INDEX_JSON;
use larql_vindex::format::vindex3::encode::segment::{
    read_segment_header, write_segment, PlannedTensor,
};
use larql_vindex::format::vindex3::fixtures::{dense_f32_model, encode_fixture_container};
use larql_vindex::format::vindex3::index::Vindex3Index;
use larql_vindex::format::vindex3::inspect::inspect_container;
use larql_vindex::format::vindex3::opplan::exec::accounting::{
    expectations, BlockGeometry, RepresentationFloor, ResidencyBudget, ResourceLedger,
};
use larql_vindex::format::vindex3::opplan::exec::execute_plan;
use larql_vindex::format::vindex3::opplan::exec::operands::OperandStore;
use larql_vindex::format::vindex3::opplan::exec::prepared::{
    select_realizations_within, ExecutionSlice,
};
use larql_vindex::format::vindex3::opplan::exec::production::ProductionBackend;
use larql_vindex::format::vindex3::opplan::exec::realization::RealizationRecord;
use larql_vindex::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan};
use larql_vindex::format::vindex3::represent::codec::codecs::{
    bf16_zlib, f32_planes, float, kquant, mxfp4, nvfp4,
};
use larql_vindex::format::vindex3::represent::codec::{
    AccessGranularity, CodecCapabilities, CodecError, CodecOperands, CodecRegistry, ErrorRadius,
    ExtentCertificate, RepresentationCodec, RepresentationExtent, ResidencyProfile, StreamRole,
    StreamSpec,
};
use larql_vindex::format::vindex3::represent::nvfp4_pack::CodecIdentity;

/// The label this crate registers and `larql-vindex` does not.
const LABEL: &str = "PLANES16";
const FAMILY: &str = "PLANES16";
const REVISION: u32 = 1;
/// The dtype the refinement plane is stored under.
const REFINEMENT_DTYPE: &str = "U8";
/// The projections the candidate stores as planes.
const PROJECTION_SUFFIX: &str = "_proj.weight";
/// The rung-2 witness's prompt, so the executions compare directly.
const TOKENS: [u32; 5] = [3, 17, 28, 0, 11];

const BASE: StreamSpec = StreamSpec {
    name: "high16",
    role: StreamRole::Values,
};
const TAIL: StreamSpec = StreamSpec {
    name: "low16",
    role: StreamRole::Refinement { depth: 1 },
};
const STREAMS: [StreamSpec; 2] = [BASE, TAIL];
const BASE_BYTES: usize = 2;
const TAIL_BYTES: usize = 2;
/// Truncating to the high half keeps 7 mantissa bits, so the residue is
/// under one ulp of those: `2^-7`, and its uniform RMS is that over √3.
const BASE_RELATIVE_RMS: f64 = 0.004_510_5;

/// An f32 image in two halves — the external crate's own progressive
/// representation.
struct Planes16;

impl Planes16 {
    fn elements(shape: &[usize]) -> usize {
        shape.iter().product::<usize>().max(1)
    }

    fn encode(values: &[f32]) -> (Vec<u8>, Vec<u8>) {
        let mut high = Vec::with_capacity(values.len() * BASE_BYTES);
        let mut low = Vec::with_capacity(values.len() * TAIL_BYTES);
        for value in values {
            let bits = value.to_bits();
            high.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
            low.extend_from_slice(&(bits as u16).to_le_bytes());
        }
        (high, low)
    }

    fn plane<'a>(
        operands: &CodecOperands<'a>,
        spec: StreamSpec,
        need: usize,
        tensor: &str,
    ) -> Result<&'a [u8], CodecError> {
        let bytes = operands.stream(spec, LABEL, tensor)?;
        if bytes.len() != need {
            return Err(CodecError::StreamLength {
                tensor: tensor.into(),
                label: LABEL.into(),
                stream: spec.name.into(),
                need,
                have: bytes.len(),
            });
        }
        Ok(bytes)
    }
}

impl RepresentationCodec for Planes16 {
    fn encoding_label(&self) -> &'static str {
        LABEL
    }

    fn identity(&self) -> CodecIdentity {
        CodecIdentity {
            family: FAMILY.into(),
            revision: REVISION,
            group_elems: 1,
            element: "f32".into(),
            group_scale: "none".into(),
            tensor_scale: "none".into(),
            layout: "row-major/halves-le".into(),
        }
    }

    fn streams(&self) -> &'static [StreamSpec] {
        &STREAMS
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
            ExtentCertificate {
                extent: RepresentationExtent::BASE,
                bits_per_weight: (BASE_BYTES * 8) as f64,
                radius: Some(ErrorRadius {
                    relative_rms: BASE_RELATIVE_RMS,
                }),
            },
            ExtentCertificate {
                extent: RepresentationExtent::at_depth(1),
                bits_per_weight: ((BASE_BYTES + TAIL_BYTES) * 8) as f64,
                radius: Some(ErrorRadius { relative_rms: 0.0 }),
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
        let per_element = BASE_BYTES + if extent.depth >= 1 { TAIL_BYTES } else { 0 };
        Ok((Self::elements(shape) * per_element) as u64)
    }

    fn validate(
        &self,
        operands: &CodecOperands<'_>,
        shape: &[usize],
        extent: RepresentationExtent,
        tensor: &str,
    ) -> Result<(), CodecError> {
        self.certificate_at(extent, tensor)?;
        let elements = Self::elements(shape);
        Self::plane(operands, BASE, elements * BASE_BYTES, tensor)?;
        if extent.depth >= 1 {
            Self::plane(operands, TAIL, elements * TAIL_BYTES, tensor)?;
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
        self.certificate_at(extent, tensor)?;
        let elements = Self::elements(shape);
        let k = shape.last().copied().unwrap_or(elements);
        let high = Self::plane(operands, BASE, elements * BASE_BYTES, tensor)?;
        let low = if extent.depth >= 1 {
            Some(Self::plane(operands, TAIL, elements * TAIL_BYTES, tensor)?)
        } else {
            None
        };
        for (out, element) in dst.iter_mut().zip(rows.start * k..) {
            let at = element * BASE_BYTES;
            let mut bits = u32::from(u16::from_le_bytes([high[at], high[at + 1]])) << 16;
            if let Some(low) = low {
                bits |= u32::from(u16::from_le_bytes([low[at], low[at + 1]]));
            }
            *out = f32::from_bits(bits);
        }
        Ok(())
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
        .and_then(|r| r.register(Box::new(f32_planes::F32_PLANES)))
        .and_then(|r| r.register(Box::new(Planes16)))
        .expect("the shipped labels and one new one");
    Box::leak(Box::new(registry))
}

fn is_projection(name: &str) -> bool {
    name.ends_with(PROJECTION_SUFFIX)
}

/// Store every projection as `PLANES16`'s two planes: the base under the
/// tensor's own name, the refinement as a sibling named the way the
/// loader's convention names it.
fn store_as_planes(root: &Path) -> Vec<String> {
    let index_path = root.join(INDEX_JSON);
    let mut index: Vindex3Index =
        serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    let mut done = Vec::new();
    for entry in index.representations.values_mut() {
        let path = root.join(&entry.segment);
        let (header, payload_start) = read_segment_header(&path).unwrap();
        if !header.tensors.iter().any(|t| is_projection(&t.name)) {
            continue;
        }
        let file = std::fs::read(&path).unwrap();
        let payload = &file[payload_start as usize..];
        let mut planned = Vec::new();
        let mut bytes_by_name = std::collections::BTreeMap::new();
        for t in &header.tensors {
            let stored = &payload[t.offset as usize..(t.offset + t.len) as usize];
            if !is_projection(&t.name) {
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
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let (high, low) = Planes16::encode(&values);
            let tail = OperandStore::sibling_stream_tensor(&t.name, TAIL.name);
            for (name, dtype, bytes) in
                [(t.name.clone(), LABEL, high), (tail, REFINEMENT_DTYPE, low)]
            {
                planned.push(PlannedTensor {
                    relative_name: name.clone(),
                    source_name: name.clone(),
                    dtype: dtype.to_string(),
                    shape: t.shape.clone(),
                    len: bytes.len() as u64,
                });
                bytes_by_name.insert(name, bytes);
            }
            done.push(t.name.clone());
        }
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
    std::fs::write(&index_path, serde_json::to_string_pretty(&index).unwrap()).unwrap();
    done
}

struct Container {
    _src: tempfile::TempDir,
    dir: tempfile::TempDir,
}

impl Container {
    fn build(as_planes: bool) -> Self {
        let src = tempfile::tempdir().unwrap();
        let dir = tempfile::tempdir().unwrap();
        encode_fixture_container(dense_f32_model, src.path(), dir.path(), "dense");
        if as_planes {
            assert!(
                !store_as_planes(dir.path()).is_empty(),
                "the dense fixture has projections"
            );
        }
        Self { _src: src, dir }
    }

    fn open(&self, registry: &'static CodecRegistry) -> (ComponentOpPlan, OperandStore) {
        let inspection = inspect_container(self.dir.path(), false).unwrap();
        let outcome = plan_component_ops(&inspection, self.dir.path(), "target").unwrap();
        let plan = outcome
            .plan
            .unwrap_or_else(|| panic!("the fixture must plan: {:?}", outcome.defects));
        let store = OperandStore::open(self.dir.path(), &inspection)
            .unwrap()
            .with_registry(registry);
        (plan, store)
    }

    fn logits(&self, registry: &'static CodecRegistry) -> Vec<u32> {
        let (plan, store) = self.open(registry);
        let trace = execute_plan(&plan, &store, &TOKENS, &ProductionBackend::new()).unwrap();
        trace
            .logits
            .expect("the dense fixture carries a head")
            .iter()
            .map(|x| x.to_bits())
            .collect()
    }

    fn select(
        &self,
        registry: &'static CodecRegistry,
        budget: &ResidencyBudget,
    ) -> Result<(Vec<RealizationRecord>, ResourceLedger), String> {
        let (plan, store) = self.open(registry);
        let records = select_realizations_within(
            &plan,
            (&store).into(),
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

#[test]
fn an_external_progressive_representation_executes_through_registration_alone() {
    let control = Container::build(false);
    let candidate = Container::build(true);
    let registry = registry_with_provider();
    // The terminal extent is the source, so the executions are the same
    // execution: any difference would be the registry's doing.
    assert_eq!(
        control.logits(registry),
        candidate.logits(registry),
        "the external representation's terminal extent is the f32 image"
    );
    // And the built-in registry does not know the label, so nothing here
    // rode on a codec the core ships.
    assert!(CodecRegistry::builtin().by_label(LABEL).is_none());
}

#[test]
fn an_external_provider_s_extents_reach_selection_and_a_budget_pins_one() {
    let candidate = Container::build(true);
    let registry = registry_with_provider();

    // What the artifact offers reaches the pin: two options, terminal
    // selected, priced from the provider's own declaration.
    let (whole_records, whole) = candidate
        .select(registry, &ResidencyBudget::UNBOUNDED)
        .expect("nothing to refuse");
    let progressive: Vec<&RealizationRecord> = whole_records
        .iter()
        .filter(|r| r.representation == LABEL)
        .collect();
    assert!(
        !progressive.is_empty(),
        "the projections are the provider's"
    );
    for record in &progressive {
        assert_eq!(
            record.extent.options.len(),
            2,
            "{}",
            record.planned.operand.tensor
        );
        assert_eq!(record.extent.selected, RepresentationExtent::at_depth(1));
        assert!(
            record.extent.touch_bytes().is_some(),
            "priced by the provider"
        );
    }

    // A preparation budget the exact extent cannot meet, under a floor
    // that admits the provider's shallow one: the pin moves, and what
    // moves with it is what is OPENED.
    let budget = ResidencyBudget::UNBOUNDED
        .with_prepare_bytes(whole.read_to_prepare * 3 / 4)
        .with_fidelity(RepresentationFloor::RelativeRms(5e-3));
    let (shallow_records, shallow) = candidate
        .select(registry, &budget)
        .expect("the provider's base extent satisfies the floor");
    assert!(
        shallow_records
            .iter()
            .filter(|r| r.representation == LABEL)
            .any(|r| r.extent.selected == RepresentationExtent::BASE),
        "some pin took the provider's base extent"
    );
    assert!(shallow.read_to_prepare < whole.read_to_prepare);
    assert_eq!(shallow.stored, whole.stored, "every plane is still stored");
    assert_eq!(shallow.resident, whole.resident, "decode widens either way");

    // The same budget with the default floor is refused instead: a
    // provider's extents are options, never permissions.
    let exact = ResidencyBudget::UNBOUNDED.with_prepare_bytes(whole.read_to_prepare * 3 / 4);
    let refusal = candidate
        .select(registry, &exact)
        .expect_err("exact reconstruction leaves nothing to give up");
    assert!(refusal.contains("exact reconstruction"), "{refusal}");
}
