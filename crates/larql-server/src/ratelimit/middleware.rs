//! Axum middleware applying the per-IP [`RateLimiter`] to each request.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::Arc;

use axum::extract::ConnectInfo;
use axum::http::header::HeaderName;
use axum::http::{HeaderMap, Request, StatusCode};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};

use super::RateLimiter;
use crate::http::HEALTH_PATH;

/// The header a proxy chain records client addresses in.
const FORWARDED_FOR: &str = "x-forwarded-for";

/// The bucket for a request that carries no client address at all. Such
/// requests share one bucket; they are not let through unlimited.
const UNKNOWN_CLIENT: IpAddr = IpAddr::V4(Ipv4Addr::UNSPECIFIED);

/// Where the client address a request is limited by comes from.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ClientIpSource {
    /// The socket peer. Correct when clients connect directly.
    #[default]
    Socket,
    /// The first `X-Forwarded-For` entry. Correct only behind a proxy that
    /// removes the header a client sends: a proxy that appends to it leaves
    /// the first entry under the client's control.
    ForwardedFor,
    /// A header the proxy in front sets to the one client address it saw
    /// and does not accept from the client (`Fly-Client-IP`,
    /// `CF-Connecting-IP`, `X-Real-IP`).
    Header(HeaderName),
}

/// Marks a request that arrived over a transport with no client address
/// and only local peers (the Unix socket). It is not rate limited.
#[derive(Debug, Clone, Copy)]
pub struct LocalTransport;

/// Runtime configuration for rate-limit middleware.
pub struct RateLimitState {
    pub limiter: Arc<RateLimiter>,
    pub client_ip: ClientIpSource,
}

/// The address a proxy recorded for the client, if the configured source
/// names one and the request carries a value that parses.
fn proxied_client_ip(source: &ClientIpSource, headers: &HeaderMap) -> Option<IpAddr> {
    let value = match source {
        ClientIpSource::Socket => return None,
        ClientIpSource::ForwardedFor => headers
            .get(FORWARDED_FOR)?
            .to_str()
            .ok()?
            .split(',')
            .next()?,
        ClientIpSource::Header(name) => headers.get(name)?.to_str().ok()?,
    };
    value.trim().parse().ok()
}

/// Middleware that applies per-IP rate limiting.
///
/// The bucket is the proxy-recorded client address when one is configured
/// and present, and the socket peer otherwise. A request with neither is
/// limited in one shared bucket.
pub async fn rate_limit_middleware(
    axum::extract::State(state): axum::extract::State<Arc<RateLimitState>>,
    request: Request<axum::body::Body>,
    next: Next,
) -> Response {
    // Health check exempt from rate limiting, and so is the local socket.
    if request.uri().path() == HEALTH_PATH || request.extensions().get::<LocalTransport>().is_some()
    {
        return next.run(request).await;
    }

    let connect_ip = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|ci| ci.0.ip());
    let ip = proxied_client_ip(&state.client_ip, request.headers())
        .or(connect_ip)
        .unwrap_or(UNKNOWN_CLIENT);

    if !state.limiter.check(ip) {
        return (StatusCode::TOO_MANY_REQUESTS, "rate limit exceeded").into_response();
    }

    next.run(request).await
}
