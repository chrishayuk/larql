//! A mock HuggingFace repo backed by a local directory.
//!
//! Serves every file in a directory at the hub's `resolve` URL shape,
//! honouring `Range:` the way the hub does — `206 Partial Content` with
//! exactly the requested span. Files the directory does not carry answer
//! `404`, so an absent optional metadata file is a real absence and not a
//! mock artifact.
//!
//! The point of serving a REAL fixture checkpoint rather than synthetic
//! bytes is that it lets the remote path be compared against the local
//! one over the same checkpoint, which is the only comparison that says
//! anything.

use std::path::{Path, PathBuf};

use mockito::{Matcher, Mock, Server, ServerGuard};

use super::super::metadata_checkpoint::STAGED_METADATA_FILES;

/// Commit the mock reports for every revision.
pub(crate) const MOCK_COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

/// Header the hub sets naming the resolved commit.
const COMMIT_HEADER: &str = "X-Repo-Commit";

/// Repo id the mock serves under.
pub(crate) const MOCK_REPO: &str = "larql-test/fixture";

/// Revision the mock serves under.
pub(crate) const MOCK_REVISION: &str = "main";

/// How a served file answers a range request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RangeBehaviour {
    /// `206` with exactly the requested span — what the hub does.
    Honour,
    /// `200` with the WHOLE file, ignoring the range. A real failure mode
    /// of caches and proxies, and the one that would splice the head of a
    /// shard in as if it were a tensor.
    Ignore,
    /// `206`, but one byte short of what was asked for. What a
    /// rate-limiting host does.
    Truncate,
}

/// A running mock repo. Restores `HF_ENDPOINT` on drop.
pub(crate) struct MockRepo {
    _server: ServerGuard,
    _mocks: Vec<Mock>,
    prev_endpoint: Option<String>,
}

impl MockRepo {
    /// Serve every file in `dir`, plus 404s for the metadata files it
    /// does not carry.
    pub(crate) fn serve(dir: &Path, behaviour: RangeBehaviour) -> Self {
        let mut server = Server::new();
        let prev_endpoint = std::env::var("HF_ENDPOINT").ok();
        std::env::set_var("HF_ENDPOINT", server.url());

        let mut mocks = Vec::new();
        let mut served: Vec<String> = Vec::new();
        // Recursive: a VINDEX3 container keeps its payloads under
        // `segments/`, and a flat listing would serve its index and graph
        // while silently 404ing every object.
        for path in walk(dir) {
            let name = path
                .strip_prefix(dir)
                .expect("under the served dir")
                .to_string_lossy()
                .replace('\\', "/");
            mocks.extend(mock_file(
                &mut server,
                resolve_path(&name),
                &path,
                behaviour,
            ));
            mocks.extend(mock_file(&mut server, commit_path(&name), &path, behaviour));
            served.push(name);
        }
        for name in STAGED_METADATA_FILES {
            if !served.iter().any(|s| s == name) {
                for absent in [resolve_path(name), commit_path(name)] {
                    mocks.push(
                        server
                            .mock("GET", absent.as_str())
                            .with_status(404)
                            .expect_at_least(0)
                            .create(),
                    );
                }
            }
        }
        Self {
            _server: server,
            _mocks: mocks,
            prev_endpoint,
        }
    }
}

impl Drop for MockRepo {
    fn drop(&mut self) {
        match self.prev_endpoint.take() {
            Some(prev) => std::env::set_var("HF_ENDPOINT", prev),
            None => std::env::remove_var("HF_ENDPOINT"),
        }
    }
}

/// Every file under `dir`, depth-first, in a stable order.
fn walk(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(current) = stack.pop() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&current)
            .expect("served dir")
            .map(|e| e.expect("dir entry").path())
            .collect();
        entries.sort();
        for path in entries {
            if path.is_dir() {
                stack.push(path);
            } else if path.is_file() {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// The hub's resolve path for one repo-relative file.
fn resolve_path(file: &str) -> String {
    format!("/{MOCK_REPO}/resolve/{MOCK_REVISION}/{file}")
}

/// The same file under the resolved COMMIT, which is how the hub serves it
/// once a revision has been pinned.
///
/// Both spellings are served because that is what the hub does, and
/// because the code under test re-pins: `artifact::resolve` reads the
/// commit from the branch and then fetches everything under the commit,
/// so a mock offering only the branch would 501 on exactly the step that
/// makes headers and payloads come from the same checkpoint.
fn commit_path(file: &str) -> String {
    format!("/{MOCK_REPO}/resolve/{MOCK_COMMIT}/{file}")
}

/// Range-honouring and whole-file mocks for one file.
fn mock_file(
    server: &mut Server,
    url: String,
    path: &PathBuf,
    behaviour: RangeBehaviour,
) -> Vec<Mock> {
    let body = std::fs::read(path).expect("read fixture file");

    // Un-ranged fallback FIRST: mockito matches the most recently
    // registered mock, so the ranged mock below must be registered last
    // or it never sees a request.
    let unranged = server
        .mock("GET", url.as_str())
        .with_status(200)
        .with_header(COMMIT_HEADER, MOCK_COMMIT)
        .with_body(body.clone())
        .expect_at_least(0)
        .create();

    let ranged = match behaviour {
        RangeBehaviour::Ignore => server
            .mock("GET", url.as_str())
            .match_header("range", Matcher::Regex("bytes=.*".to_string()))
            .with_status(200)
            .with_header(COMMIT_HEADER, MOCK_COMMIT)
            .with_body(body)
            .expect_at_least(0)
            .create(),
        _ => {
            let whole = body;
            let truncate = behaviour == RangeBehaviour::Truncate;
            server
                .mock("GET", url.as_str())
                .match_header("range", Matcher::Regex("bytes=.*".to_string()))
                .with_status(206)
                .with_header(COMMIT_HEADER, MOCK_COMMIT)
                .with_body_from_request(move |request| {
                    let header = request
                        .header("range")
                        .first()
                        .expect("range header")
                        .to_str()
                        .expect("range header is text")
                        .to_string();
                    let mut span = slice_range(&whole, &header);
                    if truncate && !span.is_empty() {
                        span.pop();
                    }
                    span
                })
                .expect_at_least(0)
                .create()
        }
    };

    vec![unranged, ranged]
}

/// `bytes=START-END`, inclusive on both ends, as HTTP states it.
fn slice_range(body: &[u8], header: &str) -> Vec<u8> {
    let spec = header
        .trim()
        .strip_prefix("bytes=")
        .expect("range header is a byte range");
    let (start, end) = spec.split_once('-').expect("range has both ends");
    let start: usize = start.parse().expect("range start");
    let end: usize = end.parse().expect("range end");
    let stop = (end + 1).min(body.len());
    body[start.min(body.len())..stop].to_vec()
}
