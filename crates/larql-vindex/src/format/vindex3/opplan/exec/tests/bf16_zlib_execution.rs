//! Rung 2's execution witness: an entropy-coded, sequential, instance-
//! sized representation executes through REGISTRATION ALONE.
//!
//! Two containers are built from one dense fixture and differ only in how
//! their projection tensors are stored: CONTROL holds the rounded bf16
//! image raw, CANDIDATE holds the same image as one zlib stream per
//! tensor. Every projection is small enough that the residency policy
//! widens both to f32 (`choose_for` answers `BlasF32` below the cache
//! threshold), so the comparison isolates the codec and says nothing
//! about kernels — controls before parity. Nothing under `exec/` outside
//! `tests/` was edited for the candidate to run, except the packed-bank
//! preflight, which this file does not exercise.
//!
//! The forecast this answers is `docs/represent/forecasts/rung2-entropy-coded-bf16.json`.

use std::path::Path;

use super::super::cpu::physical::{compact_threshold_bytes, PhysicalProjectionPlan};
use super::super::execute_plan;
use super::super::operands::{widen, OperandStore};
use super::super::prepared::{ExecutionSlice, PreparedOperands, ResidencyCensus, SiteResidency};
use super::super::production::ProductionBackend;
use crate::error::VindexError;
use crate::format::checksums::sha256_file;
use crate::format::filenames::INDEX_JSON;
use crate::format::vindex3::encode::segment::{read_segment_header, write_segment, PlannedTensor};
use crate::format::vindex3::fixtures::{
    dense_f32_model, encode_bf16_zlib, encode_fixture_container,
};
use crate::format::vindex3::index::Vindex3Index;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::{plan_component_ops, ComponentOpPlan, OperandRef};
use crate::format::vindex3::represent::codec::codecs::bf16_zlib::{BF16_ZLIB, DTYPE_BF16_ZLIB};
use crate::format::vindex3::represent::codec::codecs::{float, kquant, mxfp4, nvfp4};
use crate::format::vindex3::represent::codec::{
    CodecError, CodecRegistry, RepresentationCodec, RepresentationExtent, ResidencyProfile,
};
use larql_models::quant::half::{encode_bf16, encode_f16};

/// Prompt over the dense fixture's 128-token vocabulary.
const TOKENS: [u32; 5] = [3, 17, 28, 0, 11];
/// A label nobody registers, carrying valid zlib bytes: the load-time
/// negative arm of "executable through registration".
const UNREGISTERED_LABEL: &str = "BF16_ZLIB_UNREGISTERED";
/// The projections this witness transcodes.
const PROJECTION_SUFFIX: &str = "_proj.weight";
const ATTENTION_MARK: &str = "self_attn.";
const FFN_MARK: &str = "mlp.";
const BF16_WIDTH: usize = std::mem::size_of::<u16>();
const F32_WIDTH: usize = std::mem::size_of::<f32>();

/// Which stored form a transcoded tensor takes. Every form rounds the
/// source f32 through the same bf16 grid first, so the IMAGES are
/// identical by construction and only their storage differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Transcode {
    /// The bf16 image, raw — the control.
    Bf16,
    /// The bf16 image as one RFC 1950 stream — the candidate.
    Bf16Zlib,
    /// The candidate's bytes under a label no codec claims.
    Unregistered,
    /// An f16 image: the pre-existing no-kernel codec, for the census.
    F16,
}

impl Transcode {
    fn dtype(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::Bf16Zlib => DTYPE_BF16_ZLIB,
            Self::Unregistered => UNREGISTERED_LABEL,
            Self::F16 => "F16",
        }
    }

    fn bytes(self, values: &[f32]) -> Vec<u8> {
        match self {
            Self::Bf16 => encode_bf16(values),
            Self::Bf16Zlib | Self::Unregistered => encode_bf16_zlib(values),
            Self::F16 => encode_f16(values),
        }
    }
}

/// One tensor the transcode rewrote, as the plan will address it.
#[derive(Clone, Debug)]
pub(super) struct Transcoded {
    pub object: String,
    pub tensor: String,
    pub shape: Vec<usize>,
    /// The length the container now records for it.
    pub stored_len: u64,
}

impl Transcoded {
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn operand(&self) -> OperandRef {
        OperandRef {
            object: self.object.clone(),
            tensor: self.tensor.clone(),
            dtype: String::new(),
            shape: self.shape.clone(),
        }
    }
}

/// Rewrite every segment under `root` so that each tensor `select` names
/// is stored as `into`, and re-record the segment in `index.json` exactly
/// as the compiler does after a rewrite. Returns what was transcoded.
pub(super) fn transcode(
    root: &Path,
    select: impl Fn(&str, &[usize]) -> bool,
    into: Transcode,
) -> Vec<Transcoded> {
    let index_path = root.join(INDEX_JSON);
    let mut index: Vindex3Index =
        serde_json::from_str(&std::fs::read_to_string(&index_path).unwrap()).unwrap();
    let mut out = Vec::new();
    for (rep_id, entry) in index.representations.iter_mut() {
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
            let stored = &payload[t.offset as usize..(t.offset + t.len) as usize];
            let (dtype, bytes) = if select(&t.name, &t.shape) {
                let values = widen(&t.dtype, stored, &t.name).unwrap();
                (into.dtype().to_string(), into.bytes(&values))
            } else {
                (t.dtype.clone(), stored.to_vec())
            };
            if select(&t.name, &t.shape) {
                out.push(Transcoded {
                    object: entry.object.clone(),
                    tensor: t.name.clone(),
                    shape: t.shape.clone(),
                    stored_len: bytes.len() as u64,
                });
            }
            planned.push(PlannedTensor {
                relative_name: t.name.clone(),
                source_name: t.name.clone(),
                dtype,
                shape: t.shape.clone(),
                len: bytes.len() as u64,
            });
            bytes_by_name.insert(t.name.clone(), bytes);
        }
        let written = write_segment(&path, &header.representation, planned, |name, w, hash| {
            let bytes = &bytes_by_name[name];
            w.write_all(bytes).map_err(VindexError::Io)?;
            hash(bytes);
            Ok(bytes.len() as u64)
        })
        .unwrap_or_else(|e| panic!("{rep_id}: {e}"));
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

struct Witness {
    _src: tempfile::TempDir,
    container: tempfile::TempDir,
    transcoded: Vec<Transcoded>,
}

impl Witness {
    fn build(into: Transcode) -> Self {
        let src = tempfile::tempdir().unwrap();
        let container = tempfile::tempdir().unwrap();
        encode_fixture_container(dense_f32_model, src.path(), container.path(), "dense");
        let transcoded = transcode(container.path(), is_projection, into);
        assert!(!transcoded.is_empty(), "the dense fixture has projections");
        Self {
            _src: src,
            container,
            transcoded,
        }
    }

    fn open(&self) -> (ComponentOpPlan, OperandStore) {
        let inspection = inspect_container(self.container.path(), false).unwrap();
        let plan = plan_component_ops(&inspection, self.container.path(), "target")
            .unwrap()
            .plan
            .expect("the dense fixture plans");
        let store = OperandStore::open(self.container.path(), &inspection).unwrap();
        (plan, store)
    }

    fn logits_and_hidden(&self) -> (Vec<u32>, Vec<u32>) {
        let (plan, store) = self.open();
        let trace = execute_plan(&plan, &store, &TOKENS, &ProductionBackend::new()).unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
        let hidden = bits(trace.final_hidden());
        (
            bits(&trace.logits.expect("the dense fixture carries a head")),
            hidden,
        )
    }

    fn census(&self) -> ResidencyCensus {
        let (plan, store) = self.open();
        PreparedOperands::load(
            &plan,
            &store,
            &ProductionBackend::new(),
            ExecutionSlice::Full,
        )
        .unwrap()
        .residency_census()
    }

    /// Transcoded elements per site, by the tensor's name.
    fn elements_at(&self, mark: &str) -> usize {
        self.transcoded
            .iter()
            .filter(|t| t.tensor.contains(mark))
            .map(Transcoded::elements)
            .sum()
    }
}

/// Whether one site's residency is what `profile` declares for
/// `elements` — the agreement the census witness turns on. Written as a
/// function of the PROFILE so a mutated declaration visibly breaks it.
fn agrees(profile: ResidencyProfile, elements: usize, site: SiteResidency) -> bool {
    let declared = (profile.bytes_per_weight * elements as f64).round() as usize;
    site.compact == 0 && site.widened_f32 == declared
}

// ── C3: the pre-registration control ─────────────────────────────────

#[test]
fn without_registration_the_label_is_refused_naming_the_eight_that_are() {
    let rung_one = CodecRegistry::new()
        .register(Box::new(float::BF16))
        .and_then(|r| r.register(Box::new(float::F16)))
        .and_then(|r| r.register(Box::new(float::F32)))
        .and_then(|r| r.register(Box::new(kquant::Q4_K)))
        .and_then(|r| r.register(Box::new(kquant::Q6_K)))
        .and_then(|r| r.register(Box::new(kquant::Q8_0)))
        .and_then(|r| r.register(Box::new(nvfp4::NVFP4)))
        .and_then(|r| r.register(Box::new(mxfp4::MXFP4)))
        .unwrap();
    let err = rung_one.resolve(DTYPE_BF16_ZLIB, "w").unwrap_err();
    assert_eq!(
        err,
        CodecError::UnknownEncoding {
            tensor: "w".into(),
            label: DTYPE_BF16_ZLIB.into(),
            registered: ["BF16", "F16", "F32", "Q4_K", "Q6_K", "Q8_0", "NVFP4", "MXFP4"]
                .map(String::from)
                .to_vec(),
        }
    );
    assert!(CodecRegistry::builtin()
        .resolve(DTYPE_BF16_ZLIB, "w")
        .is_ok());
}

#[test]
fn the_same_bytes_under_an_unregistered_label_are_refused_before_any_byte_by_name() {
    let alien = Witness::build(Transcode::Unregistered);
    let (plan, store) = alien.open();
    // Rung 3b moved the refusal from the load to the preparation: the
    // selector resolves the label against the registry before any byte
    // is read, and names every operand it refused.
    let before = store.load_count();
    let Err(err) = execute_plan(&plan, &store, &TOKENS, &ProductionBackend::new()) else {
        panic!("nothing is registered for the label");
    };
    let msg = err.to_string();
    assert!(
        msg.contains("unregistered representation") && msg.contains(UNREGISTERED_LABEL),
        "the refusal names the kind and the label: {msg}"
    );
    assert_eq!(
        store.load_count(),
        before,
        "refused before any byte was read"
    );
    // The registry's own refusal still stands behind it, listing what IS
    // registered — reached by asking the store directly.
    let operand = alien.transcoded[0].operand();
    let msg = store.load(&operand).unwrap_err().to_string();
    assert!(
        msg.contains(&format!(
            "representation `{UNREGISTERED_LABEL}` is not registered"
        )) && msg.contains(DTYPE_BF16_ZLIB),
        "{msg}"
    );
}

// ── P1 / P3 / P4: executable through registration, bit-exact ─────────

#[test]
fn the_candidate_executes_bit_exact_to_the_control_and_both_are_widened() {
    let control = Witness::build(Transcode::Bf16);
    let candidate = Witness::build(Transcode::Bf16Zlib);
    // The size condition that makes the comparison about the codec:
    // every transcoded matrix widens under the policy whether or not it
    // is stored bf16, so the control never reaches the bf16 kernel.
    for t in control.transcoded.iter().chain(&candidate.transcoded) {
        let elements = t.elements();
        assert!(
            elements * F32_WIDTH < compact_threshold_bytes(),
            "{}",
            t.tensor
        );
        for stored_bf16 in [true, false] {
            assert_eq!(
                PhysicalProjectionPlan::choose_for(None, elements, stored_bf16),
                PhysicalProjectionPlan::BlasF32,
                "{}",
                t.tensor
            );
        }
    }
    assert_eq!(control.logits_and_hidden(), candidate.logits_and_hidden());
}

#[test]
fn the_census_agrees_with_the_declared_profile_and_a_mutated_profile_would_not() {
    let candidate = Witness::build(Transcode::Bf16Zlib);
    let census = candidate.census();
    let declared = BF16_ZLIB.decode_residency();
    assert_eq!(declared, ResidencyProfile::DECODED_F32);
    for (mark, site) in [(ATTENTION_MARK, census.attention), (FFN_MARK, census.ffn)] {
        let elements = candidate.elements_at(mark);
        assert!(elements > 0, "{mark}");
        assert!(
            agrees(declared, elements, site),
            "{mark}: declared {declared:?} over {elements} elements vs {site:?}"
        );
        // Independence: had the codec declared a stored realization, the
        // same census would contradict it. The agreement is a check with
        // teeth, not two readings of the executor's f32 default.
        assert!(
            !agrees(ResidencyProfile::stored(16.0), elements, site),
            "{mark}"
        );
    }
    // The control and the F16 precedent classify identically: this is
    // the pre-existing no-kernel path, not new behaviour.
    let widened = |c: ResidencyCensus| {
        (
            c.attention.widened_f32,
            c.attention.compact,
            c.ffn.widened_f32,
            c.ffn.compact,
        )
    };
    assert_eq!(
        widened(census),
        widened(Witness::build(Transcode::Bf16).census())
    );
    assert_eq!(
        widened(census),
        widened(Witness::build(Transcode::F16).census())
    );
}

// ── P5 / P6: source touch is the recorded length, never the shape ─────

#[test]
fn the_source_touch_is_the_recorded_length_and_differs_from_the_image_in_both_containers() {
    let candidate = Witness::build(Transcode::Bf16Zlib);
    let (_, store) = candidate.open();
    let mut raw_total = 0;
    let mut stored_total = 0;
    for t in &candidate.transcoded {
        let raw = store.load_raw(&t.operand()).unwrap();
        assert_eq!(raw.dtype, DTYPE_BF16_ZLIB);
        assert_eq!(raw.bytes.len() as u64, t.stored_len, "{}", t.tensor);
        assert_ne!(raw.bytes.len(), t.elements() * BF16_WIDTH, "{}", t.tensor);
        raw_total += t.elements() * BF16_WIDTH;
        stored_total += raw.bytes.len();
        // And the codec itself will not price it from the shape.
        assert!(matches!(
            BF16_ZLIB.stored_bytes(&t.shape, RepresentationExtent::TERMINAL, &t.tensor),
            Err(CodecError::InstanceSized { .. })
        ));
    }
    assert_ne!(raw_total, stored_total);
    // The container's record is honest: the re-recorded hashes match the
    // rewritten segments.
    let index: Vindex3Index = serde_json::from_str(
        &std::fs::read_to_string(candidate.container.path().join(INDEX_JSON)).unwrap(),
    )
    .unwrap();
    let mut checked = 0;
    for entry in index.representations.values() {
        if candidate
            .transcoded
            .iter()
            .any(|t| t.object == entry.object)
        {
            let path = candidate.container.path().join(&entry.segment);
            assert_eq!(sha256_file(&path).unwrap(), entry.segment_sha256);
            checked += 1;
        }
    }
    assert!(checked > 0);
}

#[test]
fn the_two_containers_differ_only_in_the_transcoded_tensors() {
    let control = Witness::build(Transcode::Bf16);
    let candidate = Witness::build(Transcode::Bf16Zlib);
    let objects: std::collections::BTreeSet<&str> = candidate
        .transcoded
        .iter()
        .map(|t| t.object.as_str())
        .collect();
    let load = |w: &Witness| -> serde_json::Value {
        let mut index: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(w.container.path().join(INDEX_JSON)).unwrap(),
        )
        .unwrap();
        // The transcoded representations legitimately differ in what
        // they record about their bytes; nothing else may.
        for (_, entry) in index["representations"].as_object_mut().unwrap() {
            if objects.contains(entry["object"].as_str().unwrap()) {
                for field in ["payload_bytes", "payload_sha256", "segment_sha256"] {
                    entry[field] = serde_json::Value::Null;
                }
            }
        }
        index
    };
    assert_eq!(load(&control), load(&candidate));
    // Every file that is not a transcoded segment is byte-identical.
    let index: Vindex3Index = serde_json::from_str(
        &std::fs::read_to_string(control.container.path().join(INDEX_JSON)).unwrap(),
    )
    .unwrap();
    let transcoded_segments: std::collections::BTreeSet<String> = index
        .representations
        .values()
        .filter(|e| objects.contains(e.object.as_str()))
        .map(|e| e.segment.clone())
        .collect();
    // Every regular file under the container, `segments/` included, as a
    // path relative to the root.
    fn walk(root: &Path, dir: &Path, out: &mut std::collections::BTreeSet<String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if entry.file_type().unwrap().is_dir() {
                walk(root, &path, out);
            } else {
                // Joined with `/` regardless of platform: the index
                // records segment paths with forward slashes, and the
                // skip below compares against those strings.
                let relative: Vec<&str> = path
                    .strip_prefix(root)
                    .unwrap()
                    .components()
                    .map(|c| c.as_os_str().to_str().unwrap())
                    .collect();
                out.insert(relative.join("/"));
            }
        }
    }
    let files = |w: &Witness| {
        let mut out = std::collections::BTreeSet::new();
        walk(w.container.path(), w.container.path(), &mut out);
        out
    };
    let names = files(&control);
    assert_eq!(names, files(&candidate), "the same files, by path");
    let (mut compared, mut skipped) = (0, 0);
    for name in names {
        if name == INDEX_JSON || transcoded_segments.contains(&name) {
            skipped += 1;
            continue;
        }
        let a = std::fs::read(control.container.path().join(&name)).unwrap();
        let b = std::fs::read(candidate.container.path().join(&name)).unwrap();
        assert_eq!(a, b, "{name}");
        compared += 1;
    }
    assert!(
        compared > 0,
        "something other than the transcoded segments exists"
    );
    assert_eq!(
        skipped,
        1 + transcoded_segments.len(),
        "index.json and each transcoded segment"
    );
}
