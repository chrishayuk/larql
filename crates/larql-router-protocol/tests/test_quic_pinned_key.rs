//! A pinned certificate is trusted only from a peer that holds its key.
//!
//! The fingerprint verifier checked that the server's certificate hashed
//! to the pinned value and then accepted any handshake signature. A
//! certificate is public: anyone who has seen it can present it. So a
//! peer with the pinned certificate and *some other* private key completed
//! the handshake, and the pin authenticated nobody.
//!
//! This stands up exactly that peer and checks the client refuses it.

#![cfg(feature = "quic")]

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use larql_router_protocol::transport::quic::client_endpoint;
use quinn::Endpoint;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::server::{ClientHello, ResolvesServerCert};
use rustls::sign::CertifiedKey;

const SERVER_NAME: &str = "larql-server";
const LOOPBACK_ANY_PORT: &str = "127.0.0.1:0";
/// A refused handshake fails at once; this bounds a hung one.
const HANDSHAKE_DEADLINE: Duration = Duration::from_secs(20);

/// A certificate, its SHA-256 fingerprint in hex, and a signing key.
struct Identity {
    cert: CertificateDer<'static>,
    fingerprint: String,
    key_der: Vec<u8>,
}

fn identity() -> Identity {
    use sha2::{Digest, Sha256};
    let generated =
        rcgen::generate_simple_self_signed(vec![SERVER_NAME.to_owned()]).expect("self-sign");
    let cert = generated.cert.der().clone();
    let fingerprint = Sha256::digest(cert.as_ref())
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Identity {
        cert,
        fingerprint,
        key_der: generated.key_pair.serialize_der(),
    }
}

/// Presents one certificate and signs with whatever key it was given,
/// whether or not the two belong together.
#[derive(Debug)]
struct Presents(Arc<CertifiedKey>);

impl ResolvesServerCert for Presents {
    fn resolve(&self, _hello: ClientHello<'_>) -> Option<Arc<CertifiedKey>> {
        Some(self.0.clone())
    }
}

/// A QUIC server that presents `cert` and signs the handshake with `key_der`.
fn server(cert: &CertificateDer<'static>, key_der: &[u8]) -> (Endpoint, SocketAddr) {
    let provider = rustls::crypto::ring::default_provider();
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(key_der.to_vec()));
    let signer = rustls::crypto::ring::sign::any_supported_type(&key).expect("signing key");
    let presented = Arc::new(CertifiedKey::new(vec![cert.clone()], signer));
    let tls = rustls::ServerConfig::builder_with_provider(Arc::new(provider))
        .with_protocol_versions(&[&rustls::version::TLS13])
        .expect("TLS 1.3")
        .with_no_client_auth()
        .with_cert_resolver(Arc::new(Presents(presented)));
    let crypto = quinn::crypto::rustls::QuicServerConfig::try_from(tls).expect("quic tls");
    let config = quinn::ServerConfig::with_crypto(Arc::new(crypto));
    let endpoint =
        Endpoint::server(config, LOOPBACK_ANY_PORT.parse().unwrap()).expect("server endpoint");
    let addr = endpoint.local_addr().expect("server addr");
    let accepting = endpoint.clone();
    tokio::spawn(async move {
        while let Some(incoming) = accepting.accept().await {
            tokio::spawn(async move {
                if let Ok(connection) = incoming.await {
                    connection.closed().await;
                }
            });
        }
    });
    (endpoint, addr)
}

/// Whether a client pinned to `fingerprint` completes a handshake with the
/// server at `addr`.
async fn handshake(addr: SocketAddr, fingerprint: &str) -> Result<(), String> {
    let client = client_endpoint(LOOPBACK_ANY_PORT.parse().unwrap(), Some(fingerprint.into()))
        .expect("client endpoint");
    let connecting = client
        .connect(addr, SERVER_NAME)
        .map_err(|e| e.to_string())?;
    match tokio::time::timeout(HANDSHAKE_DEADLINE, connecting).await {
        Ok(Ok(_connection)) => Ok(()),
        Ok(Err(e)) => Err(e.to_string()),
        Err(_) => Err("handshake timed out".into()),
    }
}

#[tokio::test]
async fn the_pinned_certificate_with_its_own_key_is_trusted() {
    // The control: the pin does not refuse everything.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let pinned = identity();
    let (_server, addr) = server(&pinned.cert, &pinned.key_der);

    handshake(addr, &pinned.fingerprint)
        .await
        .expect("the genuine server must be trusted");
}

#[tokio::test]
async fn the_pinned_certificate_without_its_key_is_refused() {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let pinned = identity();
    let impostor = identity();
    // The pinned certificate, signed for with a key that is not its own.
    let (_server, addr) = server(&pinned.cert, &impostor.key_der);

    let err = handshake(addr, &pinned.fingerprint)
        .await
        .expect_err("a peer without the pinned certificate's key must be refused");

    assert!(
        !err.contains("timed out"),
        "the handshake must be refused, not hang: {err}"
    );
}

#[tokio::test]
async fn another_certificate_is_refused() {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let pinned = identity();
    let other = identity();
    let (_server, addr) = server(&other.cert, &other.key_der);

    handshake(addr, &pinned.fingerprint)
        .await
        .expect_err("a different certificate must be refused");
}
