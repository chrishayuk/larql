//! **A container compiled from a repo is the container compiled from the
//! checkpoint.**
//!
//! That is the whole claim of the remote source, and it is the only claim
//! worth gating: everything else — the ratio of bytes fetched, the stub
//! size, the pretty progress line — is commentary on a container that
//! either is or is not the right one.
//!
//! So the subject here is a REAL fixture checkpoint, encoded twice: once
//! from the directory, once from a mock repo serving that same directory
//! by byte range. The two containers must agree on the bytes and on the
//! graph. Any difference — an off-by-one in the offset arithmetic, a
//! shard resolved to the wrong file, a range read that silently returned
//! a prefix — surfaces as a payload hash that does not match.
//!
//! The comparison has an obvious way to be vacuous: if the remote path
//! quietly fell back to reading the local directory, the two would agree
//! for the wrong reason. [`remote_source_reads_only_ranges`] closes that
//! by proving the remote source's staged directory holds no payload
//! bytes at all — the checkpoint is not reachable from where it read.

use std::collections::BTreeMap;

use larql_models::inventory::build_inventory;
use serial_test::serial;

use super::super::{RemoteArtifactSource, TensorSource};
use crate::format::huggingface::metadata_checkpoint::{
    stage_metadata_checkpoint, StagedCheckpoint,
};
use crate::format::huggingface::range::test_support::{
    MockRepo, RangeBehaviour, MOCK_REPO, MOCK_REVISION,
};
use crate::format::huggingface::range::{HfRangeClient, RetryPolicy};
use crate::format::vindex3::encode::source::staged_payload_bytes;
use crate::format::vindex3::encode::{encode_system, encode_system_from_sources, SEGMENTS_DIR};
use crate::format::vindex3::fixtures::miniature_glimmer;

/// The artifact name both encodes use. Identical on both sides, because
/// the name is recorded in the graph and a difference there would mask
/// the payload comparison with a trivially explained one.
const ARTIFACT: &str = "fixture";

/// Everything one arm of the comparison needs to stay alive.
struct Arm {
    container: tempfile::TempDir,
    /// Kept so the staged headers outlive the encode.
    _staging: Option<tempfile::TempDir>,
}

/// Encode the fixture from the local checkpoint directory.
fn encode_local(checkpoint: &std::path::Path) -> Arm {
    let container = tempfile::tempdir().unwrap();
    let inventory = build_inventory(checkpoint).unwrap();
    encode_system(&[(ARTIFACT.to_string(), inventory)], container.path()).unwrap();
    Arm {
        container,
        _staging: None,
    }
}

/// Stage the mock repo's headers, then encode over byte ranges.
///
/// Returns the arm and what the remote source measured: bytes fetched,
/// bytes declared, and the staged directory's own size.
fn encode_remote() -> (Arm, RemoteLedger) {
    let staging = tempfile::tempdir().unwrap();
    let client = HfRangeClient::new(MOCK_REPO, MOCK_REVISION).unwrap();
    let staged = stage_metadata_checkpoint(&client, staging.path()).unwrap();

    // Admission runs on the staged headers, exactly as it would on a real
    // checkpoint — this is the step that must not need a payload byte.
    let inventory = build_inventory(&staged.dir).unwrap();
    let source = RemoteArtifactSource::open(client, &staged).unwrap();

    let container = tempfile::tempdir().unwrap();
    let named = [(ARTIFACT.to_string(), inventory)];
    let sources: BTreeMap<&str, &dyn TensorSource> = [(ARTIFACT, &source as &dyn TensorSource)]
        .into_iter()
        .collect();
    encode_system_from_sources(&named, Some(&sources), container.path(), None).unwrap();

    let ledger = RemoteLedger {
        fetched: source.fetched(),
        declared: source.declared_bytes(),
        tensors: source.tensors(),
        staged_dir: staged.dir.clone(),
        stub_bytes: staged.stub_bytes,
        metadata_bytes: staged.metadata_bytes,
    };
    (
        Arm {
            container,
            _staging: Some(staging),
        },
        ledger,
    )
}

struct RemoteLedger {
    fetched: u64,
    declared: u64,
    tensors: u64,
    staged_dir: std::path::PathBuf,
    stub_bytes: u64,
    metadata_bytes: u64,
}

/// Offset of the first byte at which two segments disagree, if any.
fn first_difference(left: &[u8], right: &[u8]) -> Option<usize> {
    if let Some(at) = left.iter().zip(right).position(|(l, r)| l != r) {
        return Some(at);
    }
    (left.len() != right.len()).then(|| left.len().min(right.len()))
}

/// Every segment file of a container, by relative name.
fn segments(container: &std::path::Path) -> BTreeMap<String, Vec<u8>> {
    let dir = container.join(SEGMENTS_DIR);
    let mut out = BTreeMap::new();
    for entry in std::fs::read_dir(&dir).expect("segments dir") {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        out.insert(name, std::fs::read(&path).unwrap());
    }
    assert!(!out.is_empty(), "container wrote no segments");
    out
}

#[test]
#[serial]
fn remote_encode_is_byte_identical_to_local_encode() {
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_glimmer(checkpoint.path());
    let local = encode_local(checkpoint.path());

    let _repo = MockRepo::serve(checkpoint.path(), RangeBehaviour::Honour);
    let (remote, ledger) = encode_remote();

    let local_segments = segments(local.container.path());
    let remote_segments = segments(remote.container.path());
    assert_eq!(
        local_segments.keys().collect::<Vec<_>>(),
        remote_segments.keys().collect::<Vec<_>>(),
        "the two encodes wrote different segments"
    );
    for (name, local_bytes) in &local_segments {
        // Report the first differing offset rather than two whole
        // segments: an assert that prints a hundred kilobytes of bytes is
        // an assert nobody reads.
        let remote_bytes = &remote_segments[name];
        if let Some(at) = first_difference(local_bytes, remote_bytes) {
            panic!(
                "segment `{name}` differs between the local and remote encode \
                 at byte {at} (local {} B, remote {} B): {:?} vs {:?}",
                local_bytes.len(),
                remote_bytes.len(),
                local_bytes.get(at),
                remote_bytes.get(at),
            );
        }
    }

    // The graph is the semantic half of the container; identical payloads
    // under a different graph would still be a different model.
    let local_graph = std::fs::read(local.container.path().join("system_graph.json")).unwrap();
    let remote_graph = std::fs::read(remote.container.path().join("system_graph.json")).unwrap();
    assert_eq!(local_graph, remote_graph, "system graphs differ");

    assert!(ledger.tensors > 0, "the remote source streamed nothing");
    assert_eq!(
        ledger.fetched, ledger.declared,
        "this fixture's plan binds every tensor, so the fetched and \
         declared totals must agree"
    );
}

#[test]
#[serial]
fn remote_source_reads_only_ranges() {
    // The anti-vacuity check. If the remote arm could reach the real
    // checkpoint, the parity above would prove nothing about ranges.
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_glimmer(checkpoint.path());
    let _repo = MockRepo::serve(checkpoint.path(), RangeBehaviour::Honour);
    let (_arm, ledger) = encode_remote();

    let shard = ledger.staged_dir.join("model.safetensors");
    let staged_len = std::fs::metadata(&shard).unwrap().len();
    let real_len = std::fs::metadata(checkpoint.path().join("model.safetensors"))
        .unwrap()
        .len();
    assert!(
        staged_len < real_len,
        "the staged shard ({staged_len} B) is not smaller than the real one \
         ({real_len} B) — the stub is holding payload bytes"
    );
    assert_eq!(
        ledger.stub_bytes, staged_len,
        "the staging report does not match what is on disk"
    );
    // The staging ledger must account for the metadata separately, and
    // must not fold payload bytes into either figure.
    assert_eq!(
        ledger.metadata_bytes,
        std::fs::metadata(ledger.staged_dir.join("config.json"))
            .unwrap()
            .len(),
        "this fixture stages config.json and nothing else; the metadata \
         figure should be exactly its size"
    );
    assert!(
        ledger.declared > staged_len,
        "the fixture declares less payload ({}) than its own header costs \
         ({staged_len}) — too small to witness anything",
        ledger.declared
    );
}

#[test]
#[serial]
fn inventory_over_staged_headers_matches_the_real_checkpoint() {
    // The staging claim, stated directly: admission against headers gives
    // the verdict admission against weights would give. Everything the
    // remote path does downstream rests on this.
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_glimmer(checkpoint.path());
    let real = build_inventory(checkpoint.path()).unwrap();

    let _repo = MockRepo::serve(checkpoint.path(), RangeBehaviour::Honour);
    let staging = tempfile::tempdir().unwrap();
    let client = HfRangeClient::new(MOCK_REPO, MOCK_REVISION).unwrap();
    let staged = stage_metadata_checkpoint(&client, staging.path()).unwrap();
    let mut from_headers = build_inventory(&staged.dir).unwrap();

    // The one field that legitimately differs: where it was read from.
    assert_ne!(from_headers.path, real.path);
    from_headers.path = real.path.clone();
    assert_eq!(
        serde_json::to_value(&from_headers).unwrap(),
        serde_json::to_value(&real).unwrap(),
        "the header-only stub does not stand in for the checkpoint"
    );
}

#[test]
#[serial]
fn a_host_that_ignores_ranges_refuses_the_encode() {
    // The control for the whole path. A proxy that answers 200 with the
    // whole shard would hand the encoder plausible bytes from the wrong
    // offset; the encode must fail rather than seal them.
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_glimmer(checkpoint.path());
    let _repo = MockRepo::serve(checkpoint.path(), RangeBehaviour::Ignore);

    let staging = tempfile::tempdir().unwrap();
    // Impatient: what is under test is that the read is REFUSED, not how
    // long it waits first.
    let client = HfRangeClient::new(MOCK_REPO, MOCK_REVISION)
        .unwrap()
        .with_retry(RetryPolicy {
            attempts: 2,
            initial_delay: std::time::Duration::from_millis(1),
        });
    let err = stage_metadata_checkpoint(&client, staging.path())
        .expect_err("staging must refuse a host that does not honour ranges");
    assert!(
        err.to_string().contains("206"),
        "the refusal should name what was expected, got: {err}"
    );
}

/// Write a shard index beside the fixture whose `total_size` is short by
/// one tensor — what a checkpoint with TIED WEIGHTS looks like.
///
/// HF computes `metadata.total_size` from deduplicated parameter storage,
/// so a model whose embedding and output head were tied declares one of
/// them and serialises both. granite-4.2-3b really does this: it declares
/// 6,805,672,960 bytes while its own headers sum to 7,319,475,200, short
/// by exactly one 513,802,240-byte member.
fn write_short_index(dir: &std::path::Path, understate_by: u64) -> u64 {
    let shard = dir.join("model.safetensors");
    let mut file = std::fs::File::open(&shard).unwrap();
    let mut len = [0u8; 8];
    std::io::Read::read_exact(&mut file, &mut len).unwrap();
    let header_len = u64::from_le_bytes(len);
    let mut header = vec![0u8; header_len as usize];
    std::io::Read::read_exact(&mut file, &mut header).unwrap();
    let header: serde_json::Value = serde_json::from_slice(&header).unwrap();

    let mut weight_map = serde_json::Map::new();
    let mut total = 0u64;
    for (name, desc) in header.as_object().unwrap() {
        if name == "__metadata__" {
            continue;
        }
        let offsets = desc["data_offsets"].as_array().unwrap();
        total += offsets[1].as_u64().unwrap() - offsets[0].as_u64().unwrap();
        weight_map.insert(name.clone(), serde_json::json!("model.safetensors"));
    }
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({
            "metadata": { "total_size": total - understate_by },
            "weight_map": weight_map,
        }))
        .unwrap(),
    )
    .unwrap();
    total
}

#[test]
#[serial]
fn the_headers_are_the_payload_authority_not_the_index() {
    // The index and the headers can legitimately disagree, and only one of
    // them predicts what a range-read encode transfers. Reporting the
    // index's number would understate the transfer and make "standing in
    // for" and "fetched" differ by 7% for no visible reason.
    let checkpoint = tempfile::tempdir().unwrap();
    miniature_glimmer(checkpoint.path());
    let understated = 4096;
    let true_total = write_short_index(checkpoint.path(), understated);

    let _repo = MockRepo::serve(checkpoint.path(), RangeBehaviour::Honour);
    let staging = tempfile::tempdir().unwrap();
    let client = HfRangeClient::new(MOCK_REPO, MOCK_REVISION).unwrap();
    let staged = stage_metadata_checkpoint(&client, staging.path()).unwrap();

    assert_eq!(
        staged.declared_total_size,
        Some(true_total - understated),
        "the index's own declaration should be carried through verbatim"
    );
    assert_eq!(
        staged_payload_bytes(&staged).unwrap(),
        true_total,
        "the payload census must come from the headers, not the index"
    );

    // And the census must equal what the source will actually stream.
    let source = RemoteArtifactSource::open(client, &staged).unwrap();
    assert_eq!(source.declared_bytes(), true_total);
}

/// A stub directory holding one shard per entry, each announcing `header`.
fn stub_checkpoint(headers: &[(&str, serde_json::Value)]) -> (tempfile::TempDir, StagedCheckpoint) {
    let dir = tempfile::tempdir().unwrap();
    let mut shards = Vec::new();
    for (name, header) in headers {
        let json = serde_json::to_vec(header).unwrap();
        let mut stub = (json.len() as u64).to_le_bytes().to_vec();
        stub.extend_from_slice(&json);
        std::fs::write(dir.path().join(name), stub).unwrap();
        shards.push((*name).to_string());
    }
    let staged = StagedCheckpoint {
        dir: dir.path().to_path_buf(),
        commit: None,
        shards,
        metadata: Vec::new(),
        stub_bytes: 0,
        metadata_bytes: 0,
        declared_total_size: None,
    };
    (dir, staged)
}

fn tensor(name: &str, start: u64, end: u64) -> (String, serde_json::Value) {
    (
        name.to_string(),
        serde_json::json!({"dtype": "F32", "shape": [(end - start) / 4], "data_offsets": [start, end]}),
    )
}

#[test]
fn a_repo_whose_headers_declare_no_tensors_is_refused_at_open() {
    // Well-formed headers that announce nothing are the shape a repo has
    // when the range read landed on the right file and the wrong
    // revision. Opening the source anyway would defer the failure to the
    // first tensor, by which point the encode has already written an
    // index and a graph describing a model with no weights.
    let (_dir, staged) = stub_checkpoint(&[
        ("model-00001-of-00002.safetensors", serde_json::json!({})),
        ("model-00002-of-00002.safetensors", serde_json::json!({})),
    ]);
    let client = HfRangeClient::new(MOCK_REPO, MOCK_REVISION).unwrap();

    let err = RemoteArtifactSource::open(client, &staged)
        .err()
        .expect("headers that declare no tensors are not a checkpoint");
    let message = err.to_string();
    assert!(
        message.contains("no tensors") && message.contains("2 shard(s)"),
        "the refusal must state what it read and how much of it, got: {message}"
    );
}

#[test]
fn the_source_names_each_shard_once_however_many_tensors_it_holds() {
    // `shards()` is what the CLI reports as the repo's file set, and what
    // a caller would use to size the transfer. Keyed off tensors rather
    // than shards it would count a 300-tensor shard three hundred times.
    let (_dir, staged) = stub_checkpoint(&[
        (
            "b-second.safetensors",
            serde_json::Value::Object(
                [tensor("beta", 0, 16), tensor("gamma", 16, 32)]
                    .into_iter()
                    .collect(),
            ),
        ),
        (
            "a-first.safetensors",
            serde_json::Value::Object([tensor("alpha", 0, 16)].into_iter().collect()),
        ),
    ]);
    let client = HfRangeClient::new(MOCK_REPO, MOCK_REVISION).unwrap();
    let source = RemoteArtifactSource::open(client, &staged).unwrap();

    let shards: Vec<String> = source
        .shards()
        .iter()
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    assert_eq!(
        shards,
        vec!["a-first.safetensors", "b-second.safetensors"],
        "each shard once, in a stable order"
    );
    assert_eq!(source.declared_bytes(), 48);
}
