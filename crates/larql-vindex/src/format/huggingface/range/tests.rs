//! Range-read gates.
//!
//! The parity assertion here — "the span that comes back is the span that
//! was asked for" — is worth little on its own, because it passes for a
//! client that ignores the range as long as the fixture is small enough
//! that the whole file *is* the span. So each of the two ways a range
//! read goes wrong quietly has its own control, and both must FAIL.

use std::path::Path;

use serial_test::serial;

use super::test_support::{MockRepo, RangeBehaviour, MOCK_COMMIT, MOCK_REPO, MOCK_REVISION};
use super::{parse_spec, HfRangeClient, RetryPolicy, DEFAULT_REVISION};

/// The served fixture: long enough that a span is a strict subset, so a
/// client that ignored the range would be caught by length alone.
const FIXTURE_FILE: &str = "config.json";
const FIXTURE_LEN: usize = 4096;

/// Retry policy for the controls — the refusal is what is under test,
/// not the patience.
fn impatient() -> RetryPolicy {
    RetryPolicy {
        attempts: 2,
        initial_delay: std::time::Duration::from_millis(1),
    }
}

/// A directory holding one file of known, position-dependent bytes.
fn fixture_dir() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    let body: Vec<u8> = (0..FIXTURE_LEN).map(|i| (i % 251) as u8).collect();
    std::fs::write(dir.path().join(FIXTURE_FILE), &body).unwrap();
    dir
}

fn expected(start: usize, len: usize) -> Vec<u8> {
    (start..start + len).map(|i| (i % 251) as u8).collect()
}

fn client() -> HfRangeClient {
    HfRangeClient::new(MOCK_REPO, MOCK_REVISION).unwrap()
}

#[test]
fn spec_parses_repo_and_revision() {
    let parsed = parse_spec("hf://moonshotai/Kimi-K3").unwrap();
    assert_eq!(parsed.repo, "moonshotai/Kimi-K3");
    assert_eq!(parsed.revision, DEFAULT_REVISION);

    let pinned = parse_spec("hf://moonshotai/Kimi-K3@abc123").unwrap();
    assert_eq!(pinned.repo, "moonshotai/Kimi-K3");
    assert_eq!(pinned.revision, "abc123");
}

#[test]
fn spec_refuses_what_it_cannot_address() {
    for spec in ["moonshotai/Kimi-K3", "hf://", "hf://@main", "hf://repo@"] {
        assert!(
            parse_spec(spec).is_err(),
            "`{spec}` should not parse as a repo spec"
        );
    }
}

#[test]
#[serial]
fn range_read_returns_exactly_the_span() {
    let dir = fixture_dir();
    let _repo = MockRepo::serve(dir.path(), RangeBehaviour::Honour);
    let client = client();

    // A span from the middle: neither a prefix nor the whole file, so
    // both offset and length are actually under test.
    let start = 1000;
    let len = 512;
    let got = client
        .fetch_range(FIXTURE_FILE, start as u64, len as u64)
        .unwrap();
    assert_eq!(got.len(), len);
    assert_eq!(got, expected(start, len), "span content");
}

#[test]
#[serial]
fn range_read_reaches_the_last_byte() {
    let dir = fixture_dir();
    let _repo = MockRepo::serve(dir.path(), RangeBehaviour::Honour);
    let client = client();

    // The inclusive/exclusive boundary: `bytes=start-end` is inclusive on
    // both ends, and an off-by-one here would silently drop a tensor's
    // final byte.
    let got = client
        .fetch_range(FIXTURE_FILE, (FIXTURE_LEN - 1) as u64, 1)
        .unwrap();
    assert_eq!(got, expected(FIXTURE_LEN - 1, 1));
}

#[test]
#[serial]
fn control_a_host_that_ignores_the_range_is_refused() {
    let dir = fixture_dir();
    let _repo = MockRepo::serve(dir.path(), RangeBehaviour::Ignore);
    let client = client().with_retry(impatient());

    // 200 OK with the whole file. The bytes ARRIVE, and the first 512 of
    // them are a perfectly plausible tensor — from the wrong offset.
    // Accepting this is the failure this check exists to prevent.
    let err = client
        .fetch_range(FIXTURE_FILE, 1000, 512)
        .expect_err("a 200 answer to a range request must be refused");
    let message = err.to_string();
    assert!(
        message.contains("206"),
        "the refusal should name what was expected, got: {message}"
    );
}

#[test]
#[serial]
fn control_a_short_body_is_refused() {
    let dir = fixture_dir();
    let _repo = MockRepo::serve(dir.path(), RangeBehaviour::Truncate);
    let client = client().with_retry(impatient());

    let err = client
        .fetch_range(FIXTURE_FILE, 1000, 512)
        .expect_err("a short range body must be refused");
    let message = err.to_string();
    assert!(
        message.contains("511 of 512"),
        "the refusal should state what was measured, got: {message}"
    );
}

#[test]
#[serial]
fn absent_optional_file_is_absence_not_failure() {
    let dir = fixture_dir();
    let _repo = MockRepo::serve(dir.path(), RangeBehaviour::Honour);
    let client = client();

    assert!(
        client.fetch("generation_config.json").unwrap().is_none(),
        "a 404 on an optional metadata file is a fact, not an error"
    );
    assert!(client.fetch(FIXTURE_FILE).unwrap().is_some());
}

#[test]
#[serial]
fn revision_resolves_to_a_commit() {
    let dir = fixture_dir();
    let _repo = MockRepo::serve(dir.path(), RangeBehaviour::Honour);
    let client = client();

    assert_eq!(
        client.resolve_commit(FIXTURE_FILE).unwrap().as_deref(),
        Some(MOCK_COMMIT),
    );
}

#[test]
#[serial]
fn url_is_the_hub_resolve_shape() {
    let dir = fixture_dir();
    let _repo = MockRepo::serve(dir.path(), RangeBehaviour::Honour);
    let client = client();
    let url = client.url("model-00001-of-00002.safetensors");
    assert!(
        url.ends_with(&format!(
            "/{MOCK_REPO}/resolve/{MOCK_REVISION}/model-00001-of-00002.safetensors"
        )),
        "unexpected URL shape: {url}"
    );
}

/// The fixture dir is only ever read, never written — a guard against a
/// future edit teaching the mock to mutate the checkpoint under test.
#[test]
fn fixture_helper_writes_only_inside_its_tempdir() {
    let dir = fixture_dir();
    assert!(dir.path().join(FIXTURE_FILE).exists());
    assert!(Path::new(dir.path()).is_dir());
}

#[test]
#[serial]
fn a_span_larger_than_one_chunk_reassembles_in_order() {
    // The chunking path, which the default 64 MiB chunk hides: every
    // fixture here is smaller than one chunk, so without a small chunk
    // size this code would never run under test at all.
    let dir = fixture_dir();
    let _repo = MockRepo::serve(dir.path(), RangeBehaviour::Honour);
    let client = client().with_chunk_size(100);

    // 1000 bytes over a 100-byte chunk: ten requests, and any
    // mis-ordering or double-counted offset shows up immediately because
    // the fixture's bytes are position-dependent.
    let start = 250;
    let len = 1000;
    let got = client
        .fetch_range(FIXTURE_FILE, start as u64, len as u64)
        .unwrap();
    assert_eq!(got.len(), len);
    assert_eq!(got, expected(start, len), "chunks reassembled out of order");
}

#[test]
#[serial]
fn a_span_that_does_not_divide_evenly_ends_exactly() {
    // The final short chunk is where an off-by-one lives.
    let dir = fixture_dir();
    let _repo = MockRepo::serve(dir.path(), RangeBehaviour::Honour);
    let client = client().with_chunk_size(64);

    let start = 7;
    let len = 333; // 5 whole chunks + 13 bytes
    let got = client
        .fetch_range(FIXTURE_FILE, start as u64, len as u64)
        .unwrap();
    assert_eq!(got, expected(start, len));
}

#[test]
#[serial]
fn control_a_chunked_span_still_refuses_a_short_body() {
    // Chunking must not weaken the refusal: the per-chunk retry has to
    // enforce exact length the same way a single-shot read did.
    let dir = fixture_dir();
    let _repo = MockRepo::serve(dir.path(), RangeBehaviour::Truncate);
    let client = client().with_retry(impatient()).with_chunk_size(100);

    let err = client
        .fetch_range(FIXTURE_FILE, 250, 1000)
        .expect_err("a short chunk must be refused");
    assert!(
        err.to_string().contains("99 of 100"),
        "the refusal should state the CHUNK it measured, got: {err}"
    );
}

#[test]
fn a_spec_is_recognised_by_its_scheme_alone() {
    // The CLI branches on this before anything is parsed, so a local
    // directory whose name merely mentions the hub must not be taken for
    // a repo.
    assert!(super::is_hf_spec("hf://Qwen/Qwen3-4B"));
    assert!(super::is_hf_spec("hf://"));
    assert!(!super::is_hf_spec("./hf/Qwen3-4B"));
    assert!(!super::is_hf_spec("/models/hf://not-a-scheme"));
    assert!(!super::is_hf_spec("Qwen/Qwen3-4B"));
}

#[test]
fn a_client_opened_from_a_spec_reports_what_it_was_pinned_to() {
    // `from_spec` is the CLI's entry point, and the two accessors are how
    // every refusal message names the repo it was talking to. A client
    // that reported a different revision than it was opened with would
    // make every such message a lie.
    let client = HfRangeClient::from_spec("hf://Qwen/Qwen3-4B@refs/pr/1").unwrap();
    assert_eq!(client.repo(), "Qwen/Qwen3-4B");
    assert_eq!(client.revision(), "refs/pr/1");

    let defaulted = HfRangeClient::from_spec("hf://Qwen/Qwen3-4B").unwrap();
    assert_eq!(defaulted.revision(), DEFAULT_REVISION);

    // `is_err` rather than `expect_err`: the client holds the token, so
    // it deliberately does not derive Debug.
    assert!(
        HfRangeClient::from_spec("./local/checkpoint").is_err(),
        "a path is not a spec and must not open a client"
    );
}

#[test]
#[serial]
fn a_zero_length_span_costs_no_request() {
    // A tensor can legitimately declare zero bytes. Asking the hub for an
    // empty range is not a valid request, so the client must answer it
    // itself. The endpoint is pointed at a closed port rather than left
    // unset: with no override the client would reach the real hub, and a
    // regression here would leave the suite making network calls instead
    // of failing.
    let prev = std::env::var("HF_ENDPOINT").ok();
    std::env::set_var("HF_ENDPOINT", "http://127.0.0.1:1");
    let client = client();
    let mut sink = Vec::new();
    let copied = client
        .stream_range(FIXTURE_FILE, 0, 0, &mut sink, &mut |_| {})
        .expect("an empty span is an answer, not an error");
    assert_eq!(copied, 0);
    assert!(
        sink.is_empty(),
        "nothing was asked for, so nothing is written"
    );

    match prev {
        Some(prev) => std::env::set_var("HF_ENDPOINT", prev),
        None => std::env::remove_var("HF_ENDPOINT"),
    }
}

#[test]
#[serial]
fn a_server_error_is_refused_by_name_and_not_read_as_absence() {
    // 404 means "this repo does not carry that file" and is an ANSWER
    // (`Ok(None)`) — see `absent_optional_file_is_absence_not_failure`.
    // Every other failure status must NOT collapse into the same shape,
    // or a hub outage would look like a checkpoint that ships no
    // tokenizer, and the encode would proceed with a capability missing.
    let mut server = mockito::Server::new();
    let prev = std::env::var("HF_ENDPOINT").ok();
    std::env::set_var("HF_ENDPOINT", server.url());
    let _mock = server
        .mock(
            "GET",
            format!("/{MOCK_REPO}/resolve/{MOCK_REVISION}/{FIXTURE_FILE}").as_str(),
        )
        .with_status(503)
        .create();

    let err = client()
        .fetch(FIXTURE_FILE)
        .expect_err("a 503 is not an absence");
    let message = err.to_string();
    assert!(
        message.contains("503") && message.contains(FIXTURE_FILE),
        "the refusal must name the status and the URL it asked for, got: {message}"
    );

    match prev {
        Some(prev) => std::env::set_var("HF_ENDPOINT", prev),
        None => std::env::remove_var("HF_ENDPOINT"),
    }
}

#[test]
#[serial]
fn a_token_in_the_environment_is_sent_as_a_bearer() {
    // A gated repo answers 401 without this, and the failure would read
    // as "that model does not exist". The mock only matches when the
    // header is present, so an unauthenticated request 501s instead.
    let dir = fixture_dir();
    let prev_token = std::env::var("HF_TOKEN").ok();
    std::env::set_var("HF_TOKEN", "hf_test_token");
    let mut server = mockito::Server::new();
    let prev_endpoint = std::env::var("HF_ENDPOINT").ok();
    std::env::set_var("HF_ENDPOINT", server.url());
    let body = std::fs::read(dir.path().join(FIXTURE_FILE)).unwrap();
    let _mock = server
        .mock(
            "GET",
            format!("/{MOCK_REPO}/resolve/{MOCK_REVISION}/{FIXTURE_FILE}").as_str(),
        )
        .match_header("authorization", "Bearer hf_test_token")
        .with_status(200)
        .with_body(body.clone())
        .create();

    let got = client()
        .fetch(FIXTURE_FILE)
        .expect("the request must carry the token")
        .expect("the file is served");
    assert_eq!(got.len(), body.len());

    match prev_token {
        Some(prev) => std::env::set_var("HF_TOKEN", prev),
        None => std::env::remove_var("HF_TOKEN"),
    }
    match prev_endpoint {
        Some(prev) => std::env::set_var("HF_ENDPOINT", prev),
        None => std::env::remove_var("HF_ENDPOINT"),
    }
}
