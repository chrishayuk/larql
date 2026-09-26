//! Every refusal names its reason, and the reader tolerates exactly what
//! the writer never produces: nothing more.

use std::io::{Read, Write};

use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use flate2::Compression;

use super::{hash_bytes, identity, read_stream, stream, stream_dir, write_stream, StreamError};

const MANIFEST: &str = "stream-manifest.json";
const BODY: &str = "observations.jsonl.gz";

/// A refusal message is part of the contract a later reader acts on, so
/// each variant's text is pinned, not just its variant.
#[test]
fn every_refusal_names_its_reason() {
    let cases = [
        (StreamError::Io("boom".into()), "stream io: boom"),
        (
            StreamError::Encoding("bad json".into()),
            "stream encoding: bad json",
        ),
        (
            StreamError::Tampered {
                expected: "aa".into(),
                found: "bb".into(),
            },
            "expected aa, found bb",
        ),
        (
            StreamError::CountMismatch {
                declared: 3,
                found: 2,
            },
            "manifest declares 3 observations, stream holds 2",
        ),
        (
            StreamError::UnknownFormat("v99".into()),
            "unknown stream format v99",
        ),
    ];
    for (err, expected) in cases {
        let text = err.to_string();
        assert!(
            text.contains(expected),
            "{text:?} should contain {expected:?}"
        );
    }
    assert!(StreamError::Tampered {
        expected: "aa".into(),
        found: "bb".into(),
    }
    .to_string()
    .contains("invalidates every bank derived from it"));
}

#[test]
fn a_missing_stream_is_an_io_refusal_not_a_panic() {
    let dir = tempfile::tempdir().unwrap();
    match read_stream(&dir.path().join("absent")) {
        Err(StreamError::Io(e)) => assert!(e.contains("No such file") || e.contains("cannot find")),
        other => panic!("expected an io refusal, got {other:?}"),
    }
}

#[test]
fn a_stream_lives_under_its_root_by_label() {
    let root = std::path::Path::new("/streams");
    assert_eq!(
        stream_dir(root, "flagship-selection-8192"),
        root.join("stream-flagship-selection-8192")
    );
}

/// The writer never emits a blank line; a reader that meets one skips it
/// rather than failing to parse an empty observation, and the count it
/// verifies is of OBSERVATIONS, not lines.
#[test]
fn a_blank_line_in_the_body_is_skipped_not_counted() {
    let dir = tempfile::tempdir().unwrap();
    let observations = stream(1, 3);
    write_stream(dir.path(), &identity(1, 3), &observations).unwrap();

    let mut text = String::new();
    GzDecoder::new(&std::fs::read(dir.path().join(BODY)).unwrap()[..])
        .read_to_string(&mut text)
        .unwrap();
    let with_blank = text.replacen('\n', "\n\n", 1);
    let mut gz = GzEncoder::new(Vec::new(), Compression::default());
    gz.write_all(with_blank.as_bytes()).unwrap();
    let body = gz.finish().unwrap();

    let manifest_path = dir.path().join(MANIFEST);
    let mut m: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    m["stream_sha256"] = serde_json::json!(hash_bytes(&body));
    std::fs::write(dir.path().join(BODY), &body).unwrap();
    std::fs::write(&manifest_path, serde_json::to_vec(&m).unwrap()).unwrap();

    let (manifest, loaded) = read_stream(dir.path()).unwrap();
    assert_eq!(manifest.observations, 3);
    assert_eq!(loaded, observations, "a blank line changes nothing");
}
