//! Per-shard transport state + dispatch entry points.
//!
//! Three transports share one `Shard` struct:
//!
//! - **HTTP** — `reqwest::blocking::Client` over TCP. Default.
//! - **UDS** — manual HTTP/1.1 framing over a `UnixStream`; same wire body
//!   as TCP HTTP, just no kernel TCP stack (~50 µs/call faster on loopback).
//! - **gRPC** — tonic over a persistent HTTP/2 channel. Used for the
//!   bidirectional streaming dispatch path.
//!
//! ## Module layout (post-2026-05-10 split from a 924-LOC single file):
//!
//! - this file (`mod.rs`) — `Shard` struct + `ShardTransport` enum +
//!   constructor (`connect`) + small accessors + the UDS HTTP/1.1
//!   helpers that the per-method dispatch files share.
//! - [`stream`] — `open_stream` (gRPC bidi setup) and the dedicated
//!   tokio task that backs `ShardStream`.
//! - [`expert_batch`] — `call_batch` (single-expert batch dispatch).
//! - [`layer_batch`] — `call_layer_batch` (one-residual + K-experts batch
//!   with HTTP/UDS/gRPC transport branches).
//! - [`multi_layer`] — `call_multi_layer_batch` + the Q8K-prenormed
//!   variant (HTTP/UDS only).

mod expert_batch;
mod layer_batch;
mod multi_layer;
mod stream;

use super::config::ShardConfig;
use super::error::RemoteMoeError;

// ── Internal shard state ──────────────────────────────────────────────────────

pub(super) struct GrpcState {
    runtime: std::sync::Arc<tokio::runtime::Runtime>,
    client: larql_router_protocol::ExpertServiceClient<tonic::transport::Channel>,
}

pub(super) enum ShardTransport {
    Http(reqwest::blocking::Client),
    Grpc(std::sync::Arc<GrpcState>),
    /// Unix domain socket transport for same-host shards.  Holds one
    /// persistent stream per shard behind a `Mutex` (per-shard calls
    /// are sequential within a `forward_moe`, and across `forward_moe`
    /// calls in chat mode).  Manual HTTP/1.1 framing keeps the wire
    /// protocol identical to the TCP `Http` variant — server-side it's
    /// the same axum router on a `UnixListener`.
    ///
    /// Saves ~50 µs/call on loopback by skipping the kernel TCP stack
    /// (no Nagle, no delayed ACK, no socket buffer copies through the
    /// network stack).  Most of the saving is on the response path
    /// (server flushes complete writes immediately).
    ///
    /// Unix-only — `std::os::unix::net::UnixStream` doesn't exist on
    /// Windows, so the variant is `cfg(unix)`. `unix://` URLs return
    /// `RemoteMoeError::Client` on non-unix hosts at connect time.
    #[cfg(unix)]
    Uds(UdsState),
}

#[cfg(unix)]
pub(super) struct UdsState {
    /// Filesystem path of the socket.  Used in error messages.
    path: std::path::PathBuf,
    /// Persistent stream behind a mutex.  Reconnect lazily on disconnect.
    stream: std::sync::Mutex<Option<std::os::unix::net::UnixStream>>,
    /// The `Authorization` header value sent with every request, when the
    /// shard config carries a key.
    authorization: Option<String>,
}

pub(super) struct Shard {
    pub(super) config: ShardConfig,
    pub(super) transport: ShardTransport,
}

impl Shard {
    pub(super) fn connect(config: ShardConfig) -> Result<Self, RemoteMoeError> {
        // URL scheme dispatch:
        //   `grpc://host:port` → tonic gRPC over HTTP/2 persistent channel.
        //   `unix:///path/to/sock` → manual HTTP/1.1 over a Unix domain
        //     socket (same-host fast path; ~50 µs/call faster than TCP
        //     loopback).
        //   `http://host:port` → reqwest blocking HTTP/1.1 (default).
        let transport = if let Some(_uds_path) = config
            .url
            .strip_prefix("unix://")
            .or_else(|| config.url.strip_prefix("unix:"))
        {
            #[cfg(unix)]
            {
                // Strip the leading `///` of `unix:///abs/path` (the third `/`
                // is part of the path).  `unix:relative/path` also accepted.
                let path = std::path::PathBuf::from(_uds_path);
                // Open + health check.
                let stream = std::os::unix::net::UnixStream::connect(&path).map_err(|e| {
                    RemoteMoeError::Unreachable {
                        url: format!("unix://{}", path.display()),
                        cause: e.to_string(),
                    }
                })?;
                // Apply the configured timeout to read/write so a stuck shard
                // doesn't wedge the client forever.
                let _ = stream.set_read_timeout(Some(config.timeout));
                let _ = stream.set_write_timeout(Some(config.timeout));
                ShardTransport::Uds(UdsState {
                    path,
                    stream: std::sync::Mutex::new(Some(stream)),
                    authorization: config
                        .api_key
                        .as_deref()
                        .map(crate::ffn::shard_auth::bearer),
                })
            }
            #[cfg(not(unix))]
            {
                return Err(RemoteMoeError::Client(format!(
                    "unix:// transport not supported on this platform — \
                     `{}` requires a unix host; use http:// or grpc:// instead",
                    config.url,
                )));
            }
        } else if config.url.starts_with("grpc://") || config.url.starts_with("grpcs://") {
            // The gRPC transport has no place to put the key yet. Connecting
            // without it would be refused by a keyed shard on the first
            // call, or reach a keyless one the operator meant to protect.
            if config.api_key.is_some() {
                return Err(RemoteMoeError::Client(format!(
                    "`{}`: the gRPC shard transport does not send {} yet; use an \
                     https:// or http:// shard URL, or unset the key",
                    config.url,
                    crate::ffn::shard_auth::SHARD_API_KEY_ENV,
                )));
            }
            let use_tls = config.url.starts_with("grpcs://");
            let grpc_endpoint = if use_tls {
                config.url.replacen("grpcs://", "https://", 1)
            } else {
                config.url.replacen("grpc://", "http://", 1)
            };
            let rt = std::sync::Arc::new(
                tokio::runtime::Builder::new_multi_thread()
                    .worker_threads(2)
                    .enable_all()
                    .build()
                    .map_err(|e| RemoteMoeError::Client(e.to_string()))?,
            );
            let client = if use_tls {
                let endpoint = tonic::transport::Channel::from_shared(grpc_endpoint.clone())
                    .map_err(|e| RemoteMoeError::Client(e.to_string()))?
                    .tls_config(tonic::transport::ClientTlsConfig::new().with_webpki_roots())
                    .map_err(|e| RemoteMoeError::Client(e.to_string()))?;
                let channel =
                    rt.block_on(endpoint.connect())
                        .map_err(|e| RemoteMoeError::Unreachable {
                            url: grpc_endpoint,
                            cause: e.to_string(),
                        })?;
                larql_router_protocol::ExpertServiceClient::new(channel)
            } else {
                rt.block_on(larql_router_protocol::ExpertServiceClient::connect(
                    grpc_endpoint.clone(),
                ))
                .map_err(|e| RemoteMoeError::Unreachable {
                    url: grpc_endpoint,
                    cause: e.to_string(),
                })?
            };
            ShardTransport::Grpc(std::sync::Arc::new(GrpcState {
                runtime: rt,
                client,
            }))
        } else {
            let http = reqwest::blocking::Client::builder()
                .timeout(config.timeout)
                .pool_max_idle_per_host(64)
                .default_headers(
                    crate::ffn::shard_auth::default_headers(config.api_key.as_deref())
                        .map_err(RemoteMoeError::Client)?,
                )
                .build()
                .map_err(|e| RemoteMoeError::Client(e.to_string()))?;
            // Health check on HTTP shards only (gRPC connect already verifies).
            let health_url = format!("{}/v1/health", config.url);
            let resp = http
                .get(&health_url)
                .send()
                .map_err(|e| RemoteMoeError::Unreachable {
                    url: health_url.clone(),
                    cause: e.to_string(),
                })?;
            if !resp.status().is_success() {
                return Err(RemoteMoeError::ServerError {
                    status: resp.status().as_u16(),
                    body: resp.text().unwrap_or_default(),
                });
            }
            ShardTransport::Http(http)
        };

        Ok(Self { config, transport })
    }

    /// Layer-aware ownership check.  When the shard's `unit_set` is set
    /// (`--moe-units-manifest`), checks the explicit `(layer, expert_id)`
    /// membership; otherwise falls back to the layer-uniform range so
    /// existing `--moe-shards "0-63=URL"` configs keep working unchanged.
    pub(super) fn owns_unit(&self, layer: usize, expert_id: usize) -> bool {
        if let Some(units) = self.config.unit_set.as_ref() {
            return units.contains(&(layer, expert_id));
        }
        expert_id >= self.config.start && expert_id <= self.config.end
    }

    /// True if this shard uses gRPC transport (not HTTP or UDS).
    /// Used by `backend.rs` to decide whether to use the multi-layer fast path.
    pub(super) fn is_grpc(&self) -> bool {
        matches!(self.transport, ShardTransport::Grpc(_))
    }
}

// ── UDS HTTP/1.1 helpers ──────────────────────────────────────────────────────
//
// Hand-rolled because reqwest doesn't natively expose UDS, and pulling in
// hyperlocal + hyper for one request type would be heavier than the wire
// protocol itself.  We control both ends so framing is fixed:
//
//   POST <path> HTTP/1.1\r\n
//   Host: localhost\r\n
//   Content-Type: <ct>\r\n
//   Content-Length: <N>\r\n
//   Connection: keep-alive\r\n
//   \r\n
//   <body bytes>
//
// Response:
//   HTTP/1.1 200 OK\r\n
//   Content-Type: <ct>\r\n
//   Content-Length: <M>\r\n
//   ...other headers...
//   \r\n
//   <body bytes>
//
// Connections are persistent and reused across calls (the server's axum
// hyper accept loop honours keep-alive by default).

/// Send a single POST + read the response body via the persistent UDS
/// stream.  Reconnects on broken-pipe / read errors.
#[cfg(unix)]
pub(super) fn uds_call(
    uds: &UdsState,
    path: &str,
    content_type: &str,
    body: &[u8],
) -> Result<Vec<u8>, RemoteMoeError> {
    use std::io::{Read, Write};

    let mut guard = uds
        .stream
        .lock()
        .map_err(|_| RemoteMoeError::Client("UDS stream mutex poisoned".into()))?;

    // Try once; on transport error, reconnect and retry once.
    for attempt in 0..2 {
        // Establish the stream lazily / after disconnect.
        if guard.is_none() {
            let s = std::os::unix::net::UnixStream::connect(&uds.path).map_err(|e| {
                RemoteMoeError::Unreachable {
                    url: format!("unix://{}", uds.path.display()),
                    cause: e.to_string(),
                }
            })?;
            *guard = Some(s);
        }
        let stream = guard.as_mut().expect("just populated");

        // Build request header in a small Vec so the kernel sees one syscall
        // for the header (write_vectored could split header/body but for
        // small headers the difference is negligible; the bench result is
        // dominated by the body bytes).
        let mut req = Vec::with_capacity(160 + body.len());
        req.extend_from_slice(b"POST ");
        req.extend_from_slice(path.as_bytes());
        req.extend_from_slice(b" HTTP/1.1\r\n");
        req.extend_from_slice(b"Host: localhost\r\n");
        if let Some(authorization) = &uds.authorization {
            req.extend_from_slice(b"Authorization: ");
            req.extend_from_slice(authorization.as_bytes());
            req.extend_from_slice(b"\r\n");
        }
        req.extend_from_slice(b"Content-Type: ");
        req.extend_from_slice(content_type.as_bytes());
        req.extend_from_slice(b"\r\n");
        req.extend_from_slice(format!("Content-Length: {}\r\n", body.len()).as_bytes());
        req.extend_from_slice(b"Connection: keep-alive\r\n\r\n");
        req.extend_from_slice(body);

        // Send request.
        if let Err(e) = stream.write_all(&req) {
            if attempt == 0 {
                *guard = None; // force reconnect
                continue;
            }
            return Err(RemoteMoeError::Unreachable {
                url: format!("unix://{}", uds.path.display()),
                cause: format!("write: {e}"),
            });
        }

        // Read response: parse headers, find Content-Length, then read N bytes.
        let mut buf = Vec::with_capacity(8 * 1024);
        let mut tmp = [0u8; 4096];
        let body_start;
        let content_length;
        loop {
            match stream.read(&mut tmp) {
                Ok(0) => {
                    // Server closed; reconnect on next attempt.
                    if attempt == 0 {
                        *guard = None;
                    }
                    return Err(RemoteMoeError::BadResponse(
                        "UDS server closed connection mid-response".into(),
                    ));
                }
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
                Err(e) => {
                    if attempt == 0 {
                        *guard = None;
                    }
                    return Err(RemoteMoeError::BadResponse(format!("UDS read: {e}")));
                }
            }
            // Look for end-of-headers (\r\n\r\n).
            if let Some(idx) = find_header_end(&buf) {
                body_start = idx + 4;
                content_length = parse_content_length(&buf[..idx])?;
                break;
            }
            if buf.len() > 64 * 1024 {
                return Err(RemoteMoeError::BadResponse(
                    "UDS response headers exceed 64 KB — refusing to read further".into(),
                ));
            }
        }

        // Check status line — first 12 bytes are "HTTP/1.1 XXX".
        if buf.len() < 12 || &buf[..9] != b"HTTP/1.1 " {
            return Err(RemoteMoeError::BadResponse(
                "UDS response missing HTTP/1.1 status line".into(),
            ));
        }
        let status = std::str::from_utf8(&buf[9..12])
            .ok()
            .and_then(|s| s.parse::<u16>().ok())
            .unwrap_or(0);
        if !(200..300).contains(&status) {
            // Read body for the error message but cap to keep memory bounded.
            let body_end = (body_start + content_length).min(buf.len());
            let body_slice = &buf[body_start..body_end];
            return Err(RemoteMoeError::ServerError {
                status,
                body: String::from_utf8_lossy(body_slice).into_owned(),
            });
        }

        // Read remaining body bytes.
        let already_have = buf.len() - body_start;
        if already_have < content_length {
            let mut body_buf = vec![0u8; content_length - already_have];
            if let Err(e) = stream.read_exact(&mut body_buf) {
                return Err(RemoteMoeError::BadResponse(format!("UDS body read: {e}")));
            }
            buf.extend_from_slice(&body_buf);
        }

        return Ok(buf[body_start..body_start + content_length].to_vec());
    }
    Err(RemoteMoeError::Client("UDS retry exhausted".into()))
}

#[cfg(unix)]
fn find_header_end(buf: &[u8]) -> Option<usize> {
    if buf.len() < 4 {
        return None;
    }
    (0..=buf.len() - 4).find(|&i| &buf[i..i + 4] == b"\r\n\r\n")
}

#[cfg(unix)]
fn parse_content_length(headers: &[u8]) -> Result<usize, RemoteMoeError> {
    // Headers look like:
    //   HTTP/1.1 200 OK\r\nContent-Type: ...\r\nContent-Length: 11264\r\n
    // Search case-insensitively for "content-length:".
    let lower = headers
        .iter()
        .map(|&b| b.to_ascii_lowercase())
        .collect::<Vec<u8>>();
    let needle = b"content-length:";
    let pos = lower
        .windows(needle.len())
        .position(|w| w == needle)
        .ok_or_else(|| {
            RemoteMoeError::BadResponse("UDS response missing Content-Length header".into())
        })?;
    let mut start = pos + needle.len();
    while start < headers.len() && (headers[start] == b' ' || headers[start] == b'\t') {
        start += 1;
    }
    let mut end = start;
    while end < headers.len() && headers[end].is_ascii_digit() {
        end += 1;
    }
    let s = std::str::from_utf8(&headers[start..end])
        .map_err(|_| RemoteMoeError::BadResponse("UDS Content-Length value not UTF-8".into()))?;
    s.parse::<usize>()
        .map_err(|_| RemoteMoeError::BadResponse(format!("UDS Content-Length not a number: {s:?}")))
}

#[cfg(all(test, unix))]
mod uds_authorization_tests {
    //! The Unix-socket transport writes its own HTTP/1.1 request, so the
    //! key is a header line it has to write itself.

    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::os::unix::net::UnixListener;

    const BODY: &[u8] = b"request";
    const REPLY: &[u8] = b"reply";

    /// Serve one request on `listener` and return its header lines.
    fn serve_one(listener: UnixListener) -> std::thread::JoinHandle<Vec<String>> {
        std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut reader = BufReader::new(stream.try_clone().expect("clone"));
            let mut headers = Vec::new();
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).expect("header line");
                if line == "\r\n" {
                    break;
                }
                headers.push(line.trim_end().to_string());
            }
            let mut body = vec![0u8; BODY.len()];
            std::io::Read::read_exact(&mut reader, &mut body).expect("body");
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n",
                REPLY.len()
            )
            .expect("status");
            stream.write_all(REPLY).expect("reply");
            headers
        })
    }

    fn headers_sent(authorization: Option<String>) -> Vec<String> {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("shard.sock");
        let server = serve_one(UnixListener::bind(&path).expect("bind"));
        let uds = UdsState {
            path,
            stream: std::sync::Mutex::new(None),
            authorization,
        };

        let reply =
            uds_call(&uds, "/v1/expert/batch", "application/octet-stream", BODY).expect("uds call");

        assert_eq!(reply, REPLY);
        server.join().expect("server thread")
    }

    #[test]
    fn the_key_is_sent_as_an_authorization_header() {
        let headers = headers_sent(Some(crate::ffn::shard_auth::bearer("the-shard-key")));

        assert!(
            headers.contains(&"Authorization: Bearer the-shard-key".to_string()),
            "{headers:?}"
        );
    }

    #[test]
    fn no_header_is_sent_without_a_key() {
        let headers = headers_sent(None);

        assert!(
            !headers
                .iter()
                .any(|h| h.to_lowercase().starts_with("authorization")),
            "{headers:?}"
        );
    }
}
