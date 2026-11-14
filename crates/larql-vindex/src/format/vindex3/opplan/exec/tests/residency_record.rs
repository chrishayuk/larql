//! The residency record: what the loader read and what it left resident, per operand, and nothing while no
//! recording is open. A record belongs to one store, so these tests share no state.

use sha2::{Digest, Sha256};

use crate::format::vindex3::fixtures::bf16_container_with_q8_pack;
use crate::format::vindex3::inspect::inspect_container;
use crate::format::vindex3::opplan::exec::backend::WeightFormat;
use crate::format::vindex3::opplan::exec::operands::{OperandStore, RepresentationSource};
use crate::format::vindex3::opplan::exec::residency_record::{
    self as record, OperandRecord, F32_DIGEST_LAYOUT, Q8_DIGEST_LAYOUT, RULE_Q8_LOAD,
    RULE_STORED_BYTES, RULE_WIDEN_F32, SCHEMA,
};
use crate::format::vindex3::opplan::exec::weights::load_weight;
use crate::format::vindex3::opplan::{plan_component_ops, LayerFfn, OperandRef};

/// The loader's Q8 block length and code limit, restated here so the test pins the rule rather than reading it
/// back from the code under test.
const BLOCK: usize = 64;
const CODE_LIMIT: f32 = 127.0;

struct Fixture {
    _dir: tempfile::TempDir,
    container: std::path::PathBuf,
}

impl Fixture {
    fn build() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let container = bf16_container_with_q8_pack(dir.path());
        Self {
            _dir: dir,
            container,
        }
    }

    /// The canonical store under `source`, and layer 1's `down_proj`.
    fn open(&self, source: RepresentationSource) -> (OperandStore, OperandRef) {
        let inspection = inspect_container(&self.container, false).unwrap();
        let plan = plan_component_ops(&inspection, &self.container, "target")
            .unwrap()
            .plan
            .expect("the fixture plans");
        let Some(LayerFfn::Dense(ffn)) = &plan.layers[1].ffn else {
            panic!("the fixture's layer 1 has a dense FFN");
        };
        let store = OperandStore::open_for(&self.container, &inspection, None, source).unwrap();
        (store, ffn.down.clone())
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// `q8-load/1`, written out independently: the resident digest of a row-major `[rows, in_dim]` matrix.
fn q8_load_digest(values: &[f32], in_dim: usize) -> String {
    let mut codes: Vec<u8> = Vec::with_capacity(values.len());
    let mut scales: Vec<u8> = Vec::new();
    for row in values.chunks(in_dim) {
        for block in row.chunks(BLOCK) {
            let peak = block.iter().fold(0.0f32, |m, w| m.max(w.abs()));
            let scale = if peak > 0.0 { peak / CODE_LIMIT } else { 1.0 };
            scales.extend_from_slice(&scale.to_le_bytes());
            for w in block {
                codes.push((w / scale).round().clamp(-CODE_LIMIT, CODE_LIMIT) as i8 as u8);
            }
        }
    }
    codes.extend_from_slice(&scales);
    sha256_hex(&codes)
}

fn only(records: Vec<OperandRecord>) -> OperandRecord {
    assert_eq!(records.len(), 1, "{records:?}");
    records.into_iter().next().unwrap()
}

#[test]
fn a_quantised_load_is_recorded_as_manufactured_with_both_digests() {
    let fixture = Fixture::build();
    let (store, down) = fixture.open(RepresentationSource::Auto);
    let stored = store.load_raw(&down).unwrap();
    let widened = store.load(&down).unwrap();

    store.record_residency();
    load_weight((&store).into(), &down, WeightFormat::Q8).unwrap();
    let r = only(store.take_residency_record());

    assert_eq!(
        (r.object.as_str(), r.tensor.as_str()),
        (down.object.as_str(), down.tensor.as_str())
    );
    assert_eq!(r.shape, down.shape);
    assert_eq!(r.stored_dtype.as_deref(), Some("BF16"));
    assert_eq!(
        r.stored_sha256.as_deref(),
        Some(sha256_hex(&stored.bytes).as_str())
    );
    assert_eq!(r.representation_source, "auto");
    assert_eq!(
        (r.requested_format.as_str(), r.resident_format.as_str()),
        ("Q8", "Q8")
    );
    assert!(r.manufactured);
    assert_eq!(r.rule, RULE_Q8_LOAD);
    assert_eq!(r.resident_digest_layout, Some(Q8_DIGEST_LAYOUT));
    assert_eq!(
        r.resident_sha256.as_deref(),
        Some(q8_load_digest(&widened, down.shape[1]).as_str()),
        "the resident digest is the rule's, computed independently"
    );
    assert_ne!(r.resident_sha256, r.stored_sha256);
}

#[test]
fn a_stored_bytes_load_is_recorded_as_the_stored_bytes() {
    let fixture = Fixture::build();
    let (store, down) = fixture.open(RepresentationSource::Stored);

    store.record_residency();
    load_weight((&store).into(), &down, WeightFormat::Bf16).unwrap();
    let r = only(store.take_residency_record());

    assert_eq!(r.representation_source, "stored");
    assert_eq!(r.resident_format, "Bf16");
    assert!(!r.manufactured);
    assert_eq!(r.rule, RULE_STORED_BYTES);
    assert_eq!(r.resident_sha256, r.stored_sha256, "nothing was converted");
}

/// A copied tensor sits at the front of a page-padded buffer. The digest is of the tensor, not of the padding:
/// a matrix whose bytes do not fill a page must still record `resident = stored`. Found by the independent
/// verifier, which could not reproduce the padded digest.
#[test]
fn the_stored_bytes_digest_is_of_the_tensor_not_of_its_page_padding() {
    use crate::format::vindex3::opplan::LayerAttention;
    let fixture = Fixture::build();
    let inspection = inspect_container(&fixture.container, false).unwrap();
    let plan = plan_component_ops(&inspection, &fixture.container, "target")
        .unwrap()
        .plan
        .unwrap();
    let LayerAttention::Softmax(op) = &plan.layers[1].attention else {
        panic!("the fixture's layer 1 is softmax attention");
    };
    let small = op.k.clone();
    let store = OperandStore::open_for(
        &fixture.container,
        &inspection,
        None,
        RepresentationSource::Stored,
    )
    .unwrap();
    let stored = store.load_raw(&small).unwrap();
    assert!(
        stored.bytes.len() % crate::format::vindex3::opplan::exec::weights::DEVICE_PAGE_ALIGN != 0,
        "the case needs a tensor that does not fill a page: {} bytes",
        stored.bytes.len()
    );

    store.record_residency();
    load_weight((&store).into(), &small, WeightFormat::Bf16).unwrap();
    let r = only(store.take_residency_record());
    assert_eq!(
        r.resident_sha256.as_deref(),
        Some(sha256_hex(&stored.bytes).as_str())
    );
    assert_eq!(r.resident_sha256, r.stored_sha256);
}

#[test]
fn a_widened_load_is_recorded_with_the_digest_of_its_exact_binary32_values() {
    let fixture = Fixture::build();
    let (store, down) = fixture.open(RepresentationSource::Stored);
    let stored = store.load_raw(&down).unwrap();
    // The widening, written out: a BF16 pattern is the top half of a binary32.
    let widened: Vec<u8> = stored
        .bytes
        .chunks_exact(2)
        .flat_map(|h| ((u16::from_le_bytes([h[0], h[1]]) as u32) << 16).to_le_bytes())
        .collect();

    store.record_residency();
    load_weight((&store).into(), &down, WeightFormat::F32).unwrap();
    let r = only(store.take_residency_record());

    assert_eq!(r.resident_format, "F32");
    assert!(
        !r.manufactured,
        "an exact widening manufactures nothing lossy"
    );
    assert_eq!(r.rule, RULE_WIDEN_F32);
    assert_eq!(r.resident_digest_layout, Some(F32_DIGEST_LAYOUT));
    assert_eq!(
        r.resident_sha256.as_deref(),
        Some(sha256_hex(&widened).as_str())
    );
    assert_eq!(
        r.stored_sha256.as_deref(),
        Some(sha256_hex(&stored.bytes).as_str())
    );
}

#[test]
fn a_load_the_source_refuses_records_nothing() {
    let fixture = Fixture::build();
    let (store, down) = fixture.open(RepresentationSource::Stored);

    store.record_residency();
    let err = load_weight((&store).into(), &down, WeightFormat::Q8)
        .unwrap_err()
        .to_string();
    assert!(err.contains("forbids manufacturing"), "{err}");
    assert!(store.take_residency_record().is_empty());
}

#[test]
fn nothing_is_recorded_unless_a_recording_is_open() {
    let fixture = Fixture::build();
    let (store, down) = fixture.open(RepresentationSource::Auto);

    assert!(!store.records_residency());
    load_weight((&store).into(), &down, WeightFormat::Q8).unwrap();
    store.record_residency();
    assert!(store.records_residency());
    let taken = store.take_residency_record();
    assert!(
        taken.is_empty(),
        "a load before the recording opened: {taken:?}"
    );
    assert!(!store.records_residency());

    // A second recording starts empty: opening one discards what an abandoned one left.
    store.record_residency();
    load_weight((&store).into(), &down, WeightFormat::Bf16).unwrap();
    store.record_residency();
    assert!(store.take_residency_record().is_empty());
}

#[test]
fn a_record_belongs_to_one_store() {
    let fixture = Fixture::build();
    let (recording, down) = fixture.open(RepresentationSource::Auto);
    let (other, _) = fixture.open(RepresentationSource::Auto);
    recording.record_residency();
    load_weight((&other).into(), &down, WeightFormat::Q8).unwrap();
    assert!(
        recording.take_residency_record().is_empty(),
        "another store's load is not this store's record"
    );
    assert!(other.take_residency_record().is_empty());
}

#[test]
fn the_document_carries_the_rule_and_what_decided_the_selection() {
    let fixture = Fixture::build();
    let (store, down) = fixture.open(RepresentationSource::Auto);
    store.record_residency();
    load_weight((&store).into(), &down, WeightFormat::Q8).unwrap();
    let operands = store.take_residency_record();

    let doc = record::document("a.vindex3", "abc123", &operands);
    assert_eq!(doc["schema"], SCHEMA);
    assert_eq!(doc["container_index_sha256"], "abc123");
    assert!(doc["selection_inputs"]["l2_bytes"].as_u64().unwrap() > 0);
    assert_eq!(doc["rules"][RULE_Q8_LOAD]["block_elements"], BLOCK);
    assert_eq!(doc["operands"].as_array().unwrap().len(), 1);
    assert_eq!(doc["operands"][0]["rule"], RULE_Q8_LOAD);
    assert_eq!(doc["operands"][0]["order"], 0);
}

/// The committed fixture the independent verifier is tested on (`verifier/conformance/structured_patch/base`).
const VERIFIER_BASE: &str = "../../verifier/conformance/structured_patch/base";
const DIGESTS_OUT_VAR: &str = "LARQL_RESIDENCY_ENGINE_DIGESTS";
const DIGESTS_SCHEMA: &str = "larql-residency-engine-digests/1";
const CANONICAL: &str = "target.decoder_stack@BF16";

/// Writes, for every matrix of the committed verifier fixture, the resident digest this loader produces under
/// each rule: engine output for the verifier's tests to reproduce, never an oracle for this crate. Ignored
/// unless regenerating:
///
/// ```sh
/// LARQL_RESIDENCY_ENGINE_DIGESTS=$PWD/verifier/conformance/execution_residency/engine_digests.json \
///   cargo test -p larql-vindex --lib residency_record::the_loaders_digests -- --ignored
/// ```
#[test]
#[ignore = "regenerates the verifier's engine digests; set LARQL_RESIDENCY_ENGINE_DIGESTS"]
fn the_loaders_digests_for_the_committed_verifier_fixture() {
    use crate::format::vindex3::encode::segment::read_segment_header;
    use crate::format::vindex3::opplan::exec::residency_record::RULE_UNSPECIFIED;

    let out = std::env::var(DIGESTS_OUT_VAR).unwrap_or_else(|_| panic!("set {DIGESTS_OUT_VAR}"));
    let container = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(VERIFIER_BASE);
    let inspection = inspect_container(&container, false).unwrap();
    // `auto`, so that the loader may be asked for a Q8 residency directly.
    let store =
        OperandStore::open_for(&container, &inspection, None, RepresentationSource::Auto).unwrap();
    let index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(container.join("index.json")).unwrap()).unwrap();
    let entry = &index["representations"][CANONICAL];
    let (header, _) =
        read_segment_header(&container.join(entry["segment"].as_str().unwrap())).unwrap();

    let mut operands = Vec::new();
    for tensor in header.tensors.iter().filter(|t| t.shape.len() == 2) {
        let operand = OperandRef {
            object: entry["object"].as_str().unwrap().to_string(),
            tensor: tensor.name.clone(),
            dtype: tensor.dtype.clone(),
            shape: tensor.shape.clone(),
        };
        let mut resident = serde_json::Map::new();
        let mut stored_sha256 = None;
        for format in [WeightFormat::Q8, WeightFormat::F32, WeightFormat::Bf16] {
            store.record_residency();
            load_weight((&store).into(), &operand, format).unwrap();
            let r = only(store.take_residency_record());
            assert_ne!(r.rule, RULE_UNSPECIFIED);
            stored_sha256 = r.stored_sha256.clone();
            resident.insert(r.rule.to_string(), serde_json::json!(r.resident_sha256));
        }
        operands.push(serde_json::json!({
            "object": operand.object,
            "tensor": operand.tensor,
            "shape": operand.shape,
            "stored_sha256": stored_sha256,
            "resident": resident,
        }));
    }
    let doc = serde_json::json!({
        "schema": DIGESTS_SCHEMA,
        "generated_by": "larql-vindex opplan/exec/tests/residency_record.rs (engine output; not an oracle)",
        "container": "../structured_patch/base",
        "container_index_sha256": sha256_hex(&std::fs::read(container.join("index.json")).unwrap()),
        "operands": operands,
    });
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap() + "\n").unwrap();
}
