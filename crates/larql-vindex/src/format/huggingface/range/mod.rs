//! Byte-range reads against a HuggingFace repo.
//!
//! The rest of [`super`] moves whole files: `hf-hub` snapshots a repo into
//! the local cache and callers open what landed. That is the right shape
//! for a vindex (tens of MB of metadata, a few GB of tensors) and the
//! wrong shape for a canonical BF16 checkpoint, where the point of
//! compiling a representation is precisely **not** to pay for all of it.
//!
//! This module reads spans. Given a repo, a revision and a file, it
//! answers `bytes [start, end]` — which is what lets a safetensors header
//! be read without its payload, and one tensor's payload be read without
//! its shard.
//!
//! # A short read is refused, never accepted
//!
//! Two ways a range read goes wrong quietly, and both are fatal to
//! anything built on top:
//!
//! - The host ignores `Range:` and answers `200 OK` with the whole file.
//!   A caller that only checked "did bytes arrive" would splice the head
//!   of a 40 GB shard into the position of one tensor.
//! - The host rate-limits and answers `206` with a short body, or an
//!   error page inside a `200`. HTTP succeeded; the bytes are wrong.
//!
//! So every read here asserts the status is `206 Partial Content` and the
//! body length is exactly what was asked for, retrying with backoff and
//! then failing by name. `scripts/hf_metadata_checkpoint.py` learned the
//! second of these the hard way; the first it cannot detect at all,
//! because `curl` hides the status.

use std::io::Read;
use std::time::Duration;

use crate::error::VindexError;

/// Hub endpoint used when `HF_ENDPOINT` is unset.
const DEFAULT_ENDPOINT: &str = "https://huggingface.co";

/// Endpoint override, honoured the same way `hf-hub` honours it so a test
/// or a mirror can redirect both paths together.
const ENDPOINT_ENV: &str = "HF_ENDPOINT";

/// Header the hub sets on a `resolve` response naming the commit the
/// revision resolved to. Recorded as provenance: `main` is a moving
/// target and a container compiled from it must say which `main`.
const COMMIT_HEADER: &str = "x-repo-commit";

/// Revision used when a spec names none.
pub const DEFAULT_REVISION: &str = "main";

/// URI scheme this module and [`super::download`] share.
const HF_SCHEME: &str = "hf://";

/// Separator between repo and revision in an `hf://` spec.
const REVISION_SEP: char = '@';

/// Attempts before a range read is called a failure.
const RETRY_ATTEMPTS: u32 = 4;

/// First backoff; doubled per attempt.
const RETRY_INITIAL_DELAY: Duration = Duration::from_secs(2);

/// How hard a range read tries before it is called a failure.
///
/// A knob rather than a constant for two reasons: a slow or heavily
/// rate-limited link wants more patience than the default, and the gates
/// that prove a short read is REFUSED have to be able to run in
/// milliseconds. A control that takes fourteen seconds to express is a
/// control that stops being run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    pub attempts: u32,
    pub initial_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            attempts: RETRY_ATTEMPTS,
            initial_delay: RETRY_INITIAL_DELAY,
        }
    }
}

/// Per-request ceiling. Generous: one span can be hundreds of MB on a
/// fused expert bank.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(600);

/// Bytes requested per range GET while streaming a payload.
///
/// Peak memory is bounded by THIS, not by the tensor: each chunk is its
/// own range request, verified and retried independently, then written
/// through. Qwen3-4B is why the bound exists — its 0.78 GB embedding is a
/// single tensor, and buffering a span whole cost its full size in RAM
/// (measured at 0.93 GB RSS). A fused expert bank would be far worse.
///
/// 64 MiB rather than the local source's 1 MiB because every chunk is an
/// HTTP round trip: a 7 GB tensor is ~110 requests here and ~7000 at
/// 1 MiB.
const RANGE_CHUNK: u64 = 64 * 1024 * 1024;

/// Read buffer inside one chunk's response body.
const STREAM_BUF: usize = 1 << 20;

/// A repo, pinned to a revision, readable by byte range.
pub struct HfRangeClient {
    endpoint: String,
    repo: String,
    revision: String,
    token: Option<String>,
    client: reqwest::blocking::Client,
    retry: RetryPolicy,
    chunk: u64,
}

/// Repo and revision parsed out of an `hf://org/name[@revision]` spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HfSpec {
    pub repo: String,
    pub revision: String,
}

/// Whether `spec` names an `hf://` source.
pub fn is_hf_spec(spec: &str) -> bool {
    spec.starts_with(HF_SCHEME)
}

/// Parse `hf://org/name[@revision]`.
///
/// The revision is whatever the user wrote — a branch, a tag or a sha.
/// Resolving it to a commit is [`HfRangeClient::resolve_commit`]'s job,
/// and deliberately a separate step: parsing must not require network.
pub fn parse_spec(spec: &str) -> Result<HfSpec, VindexError> {
    let rest = spec
        .strip_prefix(HF_SCHEME)
        .ok_or_else(|| VindexError::Parse(format!("not an {HF_SCHEME} spec: {spec}")))?;
    let (repo, revision) = match rest.split_once(REVISION_SEP) {
        Some((repo, revision)) => (repo, revision),
        None => (rest, DEFAULT_REVISION),
    };
    if repo.is_empty() || revision.is_empty() {
        return Err(VindexError::Parse(format!(
            "malformed {HF_SCHEME} spec `{spec}` — expected {HF_SCHEME}org/name[@revision]"
        )));
    }
    Ok(HfSpec {
        repo: repo.to_string(),
        revision: revision.to_string(),
    })
}

impl HfRangeClient {
    /// Open a client for one repo at one revision.
    pub fn new(repo: &str, revision: &str) -> Result<Self, VindexError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|e| VindexError::Parse(format!("HTTP client init failed: {e}")))?;
        Ok(Self {
            endpoint: std::env::var(ENDPOINT_ENV)
                .unwrap_or_else(|_| DEFAULT_ENDPOINT.to_string())
                .trim_end_matches('/')
                .to_string(),
            repo: repo.to_string(),
            revision: revision.to_string(),
            token: super::token::resolve(),
            client,
            retry: RetryPolicy::default(),
            chunk: RANGE_CHUNK,
        })
    }

    /// Replace the retry policy.
    pub fn with_retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    /// Replace the per-request chunk size.
    ///
    /// A knob for the same reason [`RetryPolicy`] is one: the gate that
    /// proves a multi-chunk span reassembles correctly has to be able to
    /// cross a chunk boundary, and a test cannot afford a 64 MiB fixture.
    /// Also legitimately operational — a constrained host may want less
    /// than [`RANGE_CHUNK`] in flight.
    pub fn with_chunk_size(mut self, chunk: u64) -> Self {
        self.chunk = chunk.max(1);
        self
    }

    /// Open a client from an `hf://` spec.
    pub fn from_spec(spec: &str) -> Result<Self, VindexError> {
        let parsed = parse_spec(spec)?;
        Self::new(&parsed.repo, &parsed.revision)
    }

    pub fn repo(&self) -> &str {
        &self.repo
    }

    pub fn revision(&self) -> &str {
        &self.revision
    }

    /// Resolve URL for one repo-relative file.
    pub fn url(&self, file: &str) -> String {
        format!(
            "{}/{}/resolve/{}/{}",
            self.endpoint, self.repo, self.revision, file
        )
    }

    fn get(&self, file: &str) -> reqwest::blocking::RequestBuilder {
        let request = self.client.get(self.url(file));
        match &self.token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }

    /// The commit `self.revision` currently resolves to.
    ///
    /// Read from the hub's own `X-Repo-Commit` on a one-byte read of
    /// `probe`, so it costs a request and not a download. A hub that does
    /// not send the header is not an error here — the caller records
    /// "unresolved" provenance rather than inventing a sha.
    pub fn resolve_commit(&self, probe: &str) -> Result<Option<String>, VindexError> {
        let response = self
            .get(probe)
            .header(reqwest::header::RANGE, "bytes=0-0")
            .send()
            .map_err(|e| VindexError::Parse(format!("resolve {}: {e}", self.url(probe))))?;
        Ok(response
            .headers()
            .get(COMMIT_HEADER)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string))
    }

    /// Whole contents of one small repo-relative file.
    ///
    /// `Ok(None)` when the file is absent — an optional metadata file a
    /// repo does not carry is a fact, not a failure. Any other
    /// non-success status is an error naming the status, because a
    /// rate-limit or an auth failure returning `None` would look
    /// identical to absence and silently narrow what gets staged.
    pub fn fetch(&self, file: &str) -> Result<Option<Vec<u8>>, VindexError> {
        let response = self
            .get(file)
            .send()
            .map_err(|e| VindexError::Parse(format!("fetch {}: {e}", self.url(file))))?;
        let status = response.status();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Ok(None);
        }
        if !status.is_success() {
            return Err(VindexError::Parse(format!(
                "fetch {}: HTTP {status}",
                self.url(file)
            )));
        }
        let bytes = response
            .bytes()
            .map_err(|e| VindexError::Parse(format!("read {}: {e}", self.url(file))))?;
        Ok(Some(bytes.to_vec()))
    }

    /// Bytes `[start, start + len)` of one file, retried until the length
    /// is exactly right.
    ///
    /// For spans small enough to hold whole — headers, index files. Tensor
    /// payloads go through [`Self::stream_range`].
    pub fn fetch_range(&self, file: &str, start: u64, len: u64) -> Result<Vec<u8>, VindexError> {
        let mut buffer = Vec::with_capacity(len as usize);
        self.stream_range(file, start, len, &mut buffer, &mut |_| {})?;
        Ok(buffer)
    }

    /// Stream bytes `[start, start + len)` into `write`, feeding every
    /// byte to `observe` on the way.
    ///
    /// The signature mirrors the local source's, so the two are
    /// substitutable at the encoder's callback without the encoder
    /// knowing which it holds.
    pub fn stream_range(
        &self,
        file: &str,
        start: u64,
        len: u64,
        write: &mut dyn std::io::Write,
        observe: &mut dyn FnMut(&[u8]),
    ) -> Result<u64, VindexError> {
        if len == 0 {
            return Ok(0);
        }
        // Chunked, so peak memory is `RANGE_CHUNK` and not the tensor.
        // Each chunk is its own range request and retries independently,
        // which is what makes writing through safe: a transient short
        // read costs that chunk again, not the whole span.
        let mut done = 0u64;
        while done < len {
            let want = self.chunk.min(len - done);
            let chunk = self.fetch_chunk(file, start + done, want)?;
            observe(&chunk);
            write.write_all(&chunk)?;
            done += want;
        }
        Ok(len)
    }

    /// One chunk, retried until its length is exactly right.
    ///
    /// Buffered — but bounded by [`RANGE_CHUNK`] — because a short or
    /// wrong-status response must not reach `write`. Exhausting the
    /// retries aborts the caller; a segment already partly written is
    /// discarded with the failed encode, as it was before chunking.
    fn fetch_chunk(&self, file: &str, start: u64, len: u64) -> Result<Vec<u8>, VindexError> {
        let mut delay = self.retry.initial_delay;
        let mut last: String = String::new();
        for attempt in 0..self.retry.attempts {
            let mut staged = Vec::with_capacity(len as usize);
            match self.attempt_range(file, start, len, &mut staged) {
                Ok(()) => return Ok(staged),
                Err(err) => last = err,
            }
            if attempt + 1 < self.retry.attempts {
                std::thread::sleep(delay);
                delay *= 2;
            }
        }
        Err(VindexError::Parse(format!(
            "range read {}..+{len} of {} failed after {} attempts: {last}",
            start,
            self.url(file),
            self.retry.attempts,
        )))
    }

    /// One range attempt. `Err(String)` is the retryable reason.
    fn attempt_range(
        &self,
        file: &str,
        start: u64,
        len: u64,
        into: &mut Vec<u8>,
    ) -> Result<(), String> {
        // HTTP states ranges inclusive on both ends.
        let end = start + len - 1;
        let response = self
            .get(file)
            .header(reqwest::header::RANGE, format!("bytes={start}-{end}"))
            .send()
            .map_err(|e| format!("{e}"))?;
        let status = response.status();
        // The load-bearing check. `200 OK` here means the host ignored
        // the range and is answering with the whole file — accepting it
        // would splice the head of the shard in as if it were the span.
        if status != reqwest::StatusCode::PARTIAL_CONTENT {
            return Err(format!(
                "expected 206 Partial Content, got HTTP {status} \
                 (the host did not honour the range)"
            ));
        }
        let mut reader = response;
        let mut chunk = vec![0u8; STREAM_BUF.min(len as usize)];
        let mut read_total = 0u64;
        while read_total < len {
            let want = ((len - read_total) as usize).min(chunk.len());
            let got = reader
                .read(&mut chunk[..want])
                .map_err(|e| format!("{e}"))?;
            if got == 0 {
                break;
            }
            into.extend_from_slice(&chunk[..got]);
            read_total += got as u64;
        }
        if read_total != len {
            into.clear();
            return Err(format!("read {read_total} of {len} bytes"));
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod test_support;
#[cfg(test)]
mod tests;
