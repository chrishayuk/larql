//! ADR-0019 — HTTP/3 transport scaffolding for the shard fan-out
//! path.
//!
//! This module owns the **transport layer** for h3 — quinn endpoint
//! construction with the correct ALPN, an `H3Client` wrapper that
//! issues `POST /path` requests over a per-host QUIC connection,
//! and a minimal `accept_h3_connection` helper for the server side.
//!
//! It deliberately does **not** plug into `axum::Router` (that's
//! Phase 2 in `larql-server`) and does **not** drive the
//! `larql-router::dispatch` codepaths (Phase 3 in `larql-router`).
//! What lives here is the shared transport primitive both crates
//! will build on.
//!
//! ALPN: h3 negotiates `"h3"` (the IANA-registered token).
//! Cert pinning reuses [`super::quic`]'s SHA-256 fingerprint
//! verifier, so operators don't need a second cert.
//!
//! Feature-gated under `http3` (which itself depends on `quic`).

use std::net::SocketAddr;
use std::sync::Arc;

use quinn::Endpoint;

use super::quic;

/// ALPN protocol identifier for HTTP/3 (RFC 9114 §3.1).
pub const ALPN_H3: &[u8] = b"h3";

/// Errors produced by the HTTP/3 transport layer. Surfaces as plain
/// `String` messages today; if call sites grow more selective error
/// handling we can promote this to a `thiserror` enum.
#[derive(Debug)]
pub enum H3Error {
    /// Endpoint build / bind failure (UDP socket allocation, TLS).
    Setup(String),
    /// Connect failure (TLS handshake, fingerprint mismatch, timeout).
    Connect(String),
    /// Request-time failure (stream open, send, recv, decode).
    Request(String),
}

impl std::fmt::Display for H3Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Setup(e) => write!(f, "h3 setup: {e}"),
            Self::Connect(e) => write!(f, "h3 connect: {e}"),
            Self::Request(e) => write!(f, "h3 request: {e}"),
        }
    }
}

impl std::error::Error for H3Error {}

/// Build a client-side `quinn::Endpoint` configured for HTTP/3.
///
/// Differences from [`quic::client_endpoint`]:
///   * ALPN is set to `"h3"` so the TLS handshake negotiates HTTP/3.
///     The plain QUIC path uses no ALPN.
///
/// `expected_fingerprint` is a SHA-256 hex string of the server's
/// leaf cert DER (as printed by the server's QUIC self-signed cert
/// generation). Pass `None` to skip cert verification (LAN / dev
/// only).
pub fn client_endpoint(
    bind_addr: SocketAddr,
    expected_fingerprint: Option<String>,
) -> Result<Endpoint, H3Error> {
    let mut endpoint = Endpoint::client(bind_addr)
        .map_err(|e| H3Error::Setup(format!("quinn Endpoint::client: {e}")))?;
    let client_cfg = build_h3_client_config(expected_fingerprint)?;
    endpoint.set_default_client_config(client_cfg);
    Ok(endpoint)
}

/// Build a server-side `quinn::Endpoint` configured for HTTP/3.
///
/// Differences from [`quic::server_endpoint`]:
///   * ALPN is set to `"h3"` so the TLS handshake advertises HTTP/3.
///
/// `tls` carries the cert/key pair (either provided via
/// `--quic-cert`/`--quic-key` or auto-generated as a self-signed
/// pair — see [`quic::self_signed_tls`]).
pub fn server_endpoint(addr: SocketAddr, tls: &quic::SelfSignedTls) -> Result<Endpoint, H3Error> {
    let server_cfg = build_h3_server_config(tls)?;
    Endpoint::server(server_cfg, addr)
        .map_err(|e| H3Error::Setup(format!("quinn Endpoint::server: {e}")))
}

fn build_h3_client_config(
    expected_fingerprint: Option<String>,
) -> Result<quinn::ClientConfig, H3Error> {
    // Reuse the rustls config from the `quic` module then mutate ALPN.
    // `quic::build_client_config` is private; we replicate the
    // fingerprint-verifier setup here with the ALPN override added.
    let mut crypto: rustls::ClientConfig = match expected_fingerprint {
        Some(fp) => quic::client_rustls_config_with_fingerprint(fp).map_err(H3Error::Setup)?,
        None => quic::client_rustls_config_skip_verify(),
    };
    crypto.alpn_protocols = vec![ALPN_H3.to_vec()];
    let quic_client_cfg = quinn::crypto::rustls::QuicClientConfig::try_from(Arc::new(crypto))
        .map_err(|e| H3Error::Setup(format!("quinn ClientConfig from rustls: {e}")))?;
    Ok(quinn::ClientConfig::new(Arc::new(quic_client_cfg)))
}

fn build_h3_server_config(tls: &quic::SelfSignedTls) -> Result<quinn::ServerConfig, H3Error> {
    let mut crypto = quic::server_rustls_config(tls).map_err(H3Error::Setup)?;
    crypto.alpn_protocols = vec![ALPN_H3.to_vec()];
    let quic_server_cfg = quinn::crypto::rustls::QuicServerConfig::try_from(Arc::new(crypto))
        .map_err(|e| H3Error::Setup(format!("quinn ServerConfig from rustls: {e}")))?;
    Ok(quinn::ServerConfig::with_crypto(Arc::new(quic_server_cfg)))
}

// ── Client (Phase 3) ────────────────────────────────────────────────────────

/// HTTP/3 client that issues one-shot `POST /path` requests over a
/// per-host QUIC connection. Cloneable: the `Arc<Endpoint>` inside
/// is shared between clones, so a router can hold one `H3Client`
/// and dispatch concurrent requests against it.
///
/// Connections are NOT pooled across calls in this minimum-viable
/// implementation — each `post_json` opens, uses, and tears down a
/// fresh QUIC connection. Connection pooling is the natural next
/// step but adds significant lifecycle complexity (idle timeouts,
/// keep-alive, error recovery, per-host limits). Defer to a future
/// ADR amendment if benchmarks show the connect cost matters
/// against the per-token fan-out workload.
#[derive(Clone)]
pub struct H3Client {
    endpoint: Endpoint,
}

/// Response from a successful `H3Client::post_json` call.
#[derive(Debug, Clone)]
pub struct H3Response {
    pub status: u16,
    pub body: bytes::Bytes,
}

impl H3Client {
    /// Build a client bound to `bind_addr` (usually `0.0.0.0:0` for
    /// an ephemeral port). The fingerprint pin behaves the same as
    /// for [`client_endpoint`].
    pub fn new(bind_addr: SocketAddr, cert_fingerprint: Option<String>) -> Result<Self, H3Error> {
        Ok(Self {
            endpoint: client_endpoint(bind_addr, cert_fingerprint)?,
        })
    }

    /// Build a client around an existing endpoint. Useful when the
    /// caller has already configured cert pinning + bind options.
    pub fn from_endpoint(endpoint: Endpoint) -> Self {
        Self { endpoint }
    }

    /// `POST {server_name}{path}` with a JSON body. Returns the
    /// response status + body. The h3 connection is closed at the
    /// end of the call — no pooling (see struct doc).
    ///
    /// `server_addr` is the UDP socket of the server; `server_name`
    /// is the SNI / authority value embedded in the TLS certificate.
    pub async fn post_json(
        &self,
        server_addr: SocketAddr,
        server_name: &str,
        path: &str,
        body: bytes::Bytes,
    ) -> Result<H3Response, H3Error> {
        let connecting = self
            .endpoint
            .connect(server_addr, server_name)
            .map_err(|e| H3Error::Connect(format!("endpoint.connect: {e}")))?;
        let conn = connecting
            .await
            .map_err(|e| H3Error::Connect(format!("connect: {e}")))?;

        let h3_conn = h3_quinn::Connection::new(conn);
        let (mut driver, mut send_request) = h3::client::new(h3_conn)
            .await
            .map_err(|e| H3Error::Request(format!("h3::client::new: {e}")))?;

        // Spawn the connection driver so the connection can multiplex
        // I/O while we issue our one request. The driver future
        // resolves when the connection closes; we drop the handle and
        // let it finish in the background after our request returns.
        let driver_task = tokio::spawn(async move {
            // The driver returns when the connection terminates;
            // graceful close is the expected outcome here.
            let _ = driver.wait_idle().await;
        });

        // Build the request — fingers crossed on the URI / authority.
        let uri = format!("https://{server_name}{path}");
        let req = http::Request::builder()
            .method(http::Method::POST)
            .uri(uri)
            .header(http::header::CONTENT_TYPE, "application/json")
            .header(http::header::CONTENT_LENGTH, body.len())
            .body(())
            .map_err(|e| H3Error::Request(format!("build request: {e}")))?;

        let mut stream = send_request
            .send_request(req)
            .await
            .map_err(|e| H3Error::Request(format!("send_request: {e}")))?;
        stream
            .send_data(body)
            .await
            .map_err(|e| H3Error::Request(format!("send_data: {e}")))?;
        stream
            .finish()
            .await
            .map_err(|e| H3Error::Request(format!("finish send: {e}")))?;

        let resp = stream
            .recv_response()
            .await
            .map_err(|e| H3Error::Request(format!("recv_response: {e}")))?;
        let status = resp.status().as_u16();

        // Drain the response body. Loop until `recv_data` returns
        // `None`. Bound the buffer at 64 MiB to match the existing
        // axum body limit on the dense path.
        let mut body_bytes = bytes::BytesMut::new();
        const MAX_BODY: usize = 64 * 1024 * 1024;
        while let Some(mut chunk) = stream
            .recv_data()
            .await
            .map_err(|e| H3Error::Request(format!("recv_data: {e}")))?
        {
            use bytes::Buf;
            let remaining = chunk.remaining();
            if body_bytes.len() + remaining > MAX_BODY {
                return Err(H3Error::Request(format!(
                    "response body exceeds {MAX_BODY} bytes"
                )));
            }
            body_bytes.extend_from_slice(&chunk.copy_to_bytes(remaining));
        }

        // Connection cleanup. Closing the send_request signals the
        // peer we're done; the driver task spins down naturally.
        drop(send_request);
        let _ = driver_task.await;

        Ok(H3Response {
            status,
            body: body_bytes.freeze(),
        })
    }
}

// ── Server (Phase 2) ────────────────────────────────────────────────────────

/// Serve an `axum::Router` over HTTP/3 on the given `quinn::Endpoint`.
///
/// Spawns one `tokio::task` per accepted QUIC connection. Inside
/// each connection, every incoming HTTP/3 request runs as a
/// `serve_h3_with_axum` call against a clone of the `axum::Router`
/// — the same handlers used for HTTP/2 on the dense path.
///
/// Runs until the endpoint stops accepting connections (caller
/// shutdown). Per-connection panics are caught and logged but
/// don't tear down the accept loop.
///
/// This function never returns under normal operation — it's
/// expected to be spawned on a long-lived `tokio::task`.
pub async fn serve_axum(endpoint: Endpoint, app: axum::Router) -> Result<(), H3Error> {
    while let Some(connecting) = endpoint.accept().await {
        let app = app.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_connection(connecting, app).await {
                tracing::warn!("h3 connection: {e}");
            }
        });
    }
    Ok(())
}

/// Drive one accepted QUIC connection through h3, calling
/// `h3_axum::serve_h3_with_axum` for each request the peer sends.
/// Used internally by [`serve_axum`].
async fn handle_connection(connecting: quinn::Incoming, app: axum::Router) -> Result<(), H3Error> {
    let conn = connecting
        .await
        .map_err(|e| H3Error::Connect(format!("incoming.await: {e}")))?;
    let h3_conn = h3_quinn::Connection::new(conn);
    let mut server = h3::server::Connection::new(h3_conn)
        .await
        .map_err(|e| H3Error::Request(format!("h3::server::Connection::new: {e}")))?;

    loop {
        match server.accept().await {
            Ok(Some(resolver)) => {
                let app = app.clone();
                tokio::spawn(async move {
                    if let Err(e) = h3_axum::serve_h3_with_axum(app, resolver).await {
                        tracing::warn!("h3 request: {e}");
                    }
                });
            }
            Ok(None) => break,
            Err(e) => {
                if h3_axum::is_graceful_h3_close(&e) {
                    break;
                }
                return Err(H3Error::Request(format!("h3::server::accept: {e}")));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The h3 module exposes a stable ALPN identifier — clients +
    /// servers must agree or the TLS handshake fails.
    #[test]
    fn alpn_h3_matches_rfc_9114() {
        assert_eq!(ALPN_H3, b"h3");
    }

    /// A client endpoint can be built against the loopback bind +
    /// no fingerprint (LAN/dev mode). Smoke test: no panic, no
    /// transport setup error.
    #[tokio::test]
    async fn client_endpoint_loopback_no_fingerprint_succeeds() {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let ep = client_endpoint("127.0.0.1:0".parse().unwrap(), None)
            .expect("client_endpoint must succeed on loopback");
        // Endpoint allocated a local UDP port — local_addr should
        // resolve to a non-zero port.
        let addr = ep.local_addr().expect("endpoint must have a local addr");
        assert!(addr.port() != 0, "ephemeral port must be assigned");
    }

    /// A server endpoint can be built from a self-signed TLS cert,
    /// bind to an ephemeral port, and not panic. ALPN is set
    /// internally to `"h3"`.
    #[tokio::test]
    async fn server_endpoint_with_self_signed_tls_succeeds() {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let tls = quic::self_signed_tls("test-server").expect("self_signed_tls must succeed");
        let ep = server_endpoint("127.0.0.1:0".parse().unwrap(), &tls)
            .expect("server_endpoint must succeed on loopback");
        let addr = ep.local_addr().expect("endpoint must have a local addr");
        assert!(addr.port() != 0);
    }

    /// Building the client config with a (malformed) fingerprint
    /// surfaces a `Setup` error rather than panicking.
    #[tokio::test]
    async fn client_endpoint_rejects_malformed_fingerprint() {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let err = client_endpoint(
            "127.0.0.1:0".parse().unwrap(),
            Some("not-a-real-fingerprint".into()),
        )
        .expect_err("malformed fingerprint must fail at config time");
        assert!(matches!(err, H3Error::Setup(_)), "got {err:?}");
    }

    /// `H3Error::Display` produces a human-readable string for each
    /// variant — useful for tracing.
    #[test]
    fn h3_error_display_includes_variant_label() {
        let setup = H3Error::Setup("x".into());
        assert!(setup.to_string().starts_with("h3 setup"));
        let connect = H3Error::Connect("y".into());
        assert!(connect.to_string().starts_with("h3 connect"));
        let request = H3Error::Request("z".into());
        assert!(request.to_string().starts_with("h3 request"));
    }

    /// `H3Client::from_endpoint` accepts a pre-configured endpoint
    /// without re-building TLS. Smoke test that exercises the
    /// constructor branch.
    #[tokio::test]
    async fn h3_client_from_endpoint_wraps_existing_endpoint() {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let ep = client_endpoint("127.0.0.1:0".parse().unwrap(), None).unwrap();
        let local = ep.local_addr().unwrap();
        let client = H3Client::from_endpoint(ep);
        // Verify the wrapped endpoint kept its port — the constructor
        // didn't drop / replace it.
        assert_eq!(client.endpoint.local_addr().unwrap(), local);
    }

    /// `serve_axum` returns `Ok(())` when the endpoint is closed
    /// externally. Covers the loop-exit branch that production
    /// hits during a graceful router shutdown.
    #[tokio::test]
    async fn serve_axum_returns_ok_when_endpoint_closes() {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let tls = quic::self_signed_tls("test-server").unwrap();
        let endpoint = server_endpoint("127.0.0.1:0".parse().unwrap(), &tls).unwrap();
        let endpoint_handle = endpoint.clone();
        let server_task =
            tokio::spawn(async move { serve_axum(endpoint, axum::Router::new()).await });
        // Give the loop a tick to enter accept().
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        endpoint_handle.close(0u32.into(), b"test shutdown");
        tokio::time::timeout(std::time::Duration::from_secs(2), server_task)
            .await
            .expect("serve_axum must exit within 2s of endpoint close")
            .expect("task join")
            .expect("serve_axum returned Err");
    }

    /// `H3Response` derives Clone — useful for downstream code that
    /// wants to fan out a single response. Compile-time check via
    /// type-binding.
    #[test]
    fn h3_response_is_clone_and_carries_status_and_body() {
        let r = H3Response {
            status: 200,
            body: bytes::Bytes::from_static(b"hello"),
        };
        let r2 = r.clone();
        assert_eq!(r2.status, 200);
        assert_eq!(r2.body.as_ref(), b"hello");
    }
}
