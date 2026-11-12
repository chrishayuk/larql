//! RATELIMIT MIDDLEWARE

use super::*;

#[tokio::test]
async fn rate_limit_blocks_when_exhausted() {
    // 1/sec → first request with trusted X-Forwarded-For passes, second is rejected.
    let rl = Arc::new(RateLimiter::parse("1/sec").unwrap());
    let app1 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp1 = app1
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "1.2.3.4")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp1.status(), StatusCode::OK, "first request should pass");

    let app2 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp2 = app2
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "1.2.3.4")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp2.status(),
        StatusCode::TOO_MANY_REQUESTS,
        "second request should be rate-limited"
    );
}

#[tokio::test]
async fn rate_limit_health_exempt() {
    // Even with a 1/sec limiter exhausted, /v1/health is exempt.
    let rl = Arc::new(RateLimiter::parse("1/sec").unwrap());

    // Exhaust the limiter for 127.0.0.1 via X-Forwarded-For.
    let app1 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp1 = app1
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "127.0.0.1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp1.status(), StatusCode::OK);

    // Verify exhausted on /v1/stats.
    let app2 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp2 = app2
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "127.0.0.1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp2.status(), StatusCode::TOO_MANY_REQUESTS);

    // Health check is exempt — should still pass.
    let app3 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp3 = app3
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/health")
                .header("x-forwarded-for", "127.0.0.1")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp3.status(),
        StatusCode::OK,
        "/v1/health should be exempt from rate limiting"
    );
}

#[tokio::test]
async fn rate_limit_forwarded_for_header_used_as_ip_when_trusted() {
    // X-Forwarded-For: 10.0.0.1 → uses that IP, different from 10.0.0.2.
    let rl = Arc::new(RateLimiter::parse("1/sec").unwrap());
    let proxy_addr: SocketAddr = "192.0.2.10:443".parse().unwrap();

    // Exhaust 10.0.0.1 bucket.
    let app1 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let _ = app1
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "10.0.0.1")
                .extension(ConnectInfo(proxy_addr))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // 10.0.0.1 is now blocked.
    let app2 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp_blocked = app2
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "10.0.0.1")
                .extension(ConnectInfo(proxy_addr))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp_blocked.status(), StatusCode::TOO_MANY_REQUESTS);

    // 10.0.0.2 has its own bucket — should pass.
    let app3 = router_with_limiter_trust_forwarded_for(Arc::clone(&rl), true);
    let resp_other = app3
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", "10.0.0.2")
                .extension(ConnectInfo(proxy_addr))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        resp_other.status(),
        StatusCode::OK,
        "different IP should have its own bucket"
    );
}

/// One request to `/v1/stats`, optionally from a socket peer and with one
/// extra header, through a router keyed on `client_ip`.
async fn status_of(
    rl: &Arc<RateLimiter>,
    client_ip: ClientIpSource,
    peer: Option<SocketAddr>,
    header: Option<(&str, &str)>,
) -> StatusCode {
    let mut request = Request::builder().method("GET").uri("/v1/stats");
    if let Some((name, value)) = header {
        request = request.header(name, value);
    }
    if let Some(peer) = peer {
        request = request.extension(ConnectInfo(peer));
    }
    router_with_client_ip(Arc::clone(rl), client_ip)
        .oneshot(request.body(Body::empty()).unwrap())
        .await
        .unwrap()
        .status()
}

const CLIENT_IP_HEADER: &str = "fly-client-ip";

fn client_ip_header() -> ClientIpSource {
    ClientIpSource::Header(CLIENT_IP_HEADER.parse().unwrap())
}

fn proxy() -> Option<SocketAddr> {
    Some("192.0.2.10:443".parse().unwrap())
}

#[tokio::test]
async fn rate_limit_forwarded_for_header_ignored_by_default() {
    // One socket peer, a different X-Forwarded-For each time: by default
    // the header is not read, so the second request is the peer's second.
    let rl = Arc::new(RateLimiter::parse("1/sec").unwrap());
    let first = status_of(
        &rl,
        ClientIpSource::Socket,
        proxy(),
        Some(("x-forwarded-for", "10.0.0.1")),
    )
    .await;
    let second = status_of(
        &rl,
        ClientIpSource::Socket,
        proxy(),
        Some(("x-forwarded-for", "10.0.0.2")),
    )
    .await;

    assert_eq!(first, StatusCode::OK);
    assert_eq!(second, StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn a_request_with_no_client_address_is_limited_in_a_shared_bucket() {
    // No socket peer and no proxy header used to mean no limit at all.
    let rl = Arc::new(RateLimiter::parse("1/sec").unwrap());

    let first = status_of(&rl, ClientIpSource::Socket, None, None).await;
    let second = status_of(&rl, ClientIpSource::Socket, None, None).await;

    assert_eq!(first, StatusCode::OK);
    assert_eq!(second, StatusCode::TOO_MANY_REQUESTS);
}

#[tokio::test]
async fn the_client_ip_header_keys_the_bucket_and_forwarded_for_does_not() {
    // The proxy sets the client-address header itself. A client that
    // rotates X-Forwarded-For stays in one bucket.
    let rl = Arc::new(RateLimiter::parse("1/sec").unwrap());
    let request = |forwarded: &'static str, client: &'static str| {
        let rl = Arc::clone(&rl);
        async move {
            let mut request = Request::builder()
                .method("GET")
                .uri("/v1/stats")
                .header("x-forwarded-for", forwarded)
                .header(CLIENT_IP_HEADER, client)
                .extension(ConnectInfo(proxy().unwrap()));
            let _ = &mut request;
            router_with_client_ip(rl, client_ip_header())
                .oneshot(request.body(Body::empty()).unwrap())
                .await
                .unwrap()
                .status()
        }
    };

    assert_eq!(request("10.0.0.1", "203.0.113.5").await, StatusCode::OK);
    assert_eq!(
        request("10.0.0.2", "203.0.113.5").await,
        StatusCode::TOO_MANY_REQUESTS,
        "a new X-Forwarded-For must not open a new bucket"
    );
    assert_eq!(
        request("10.0.0.2", "203.0.113.6").await,
        StatusCode::OK,
        "another client has its own bucket"
    );
}

#[tokio::test]
async fn a_missing_or_unparseable_client_ip_header_falls_back_to_the_socket_peer() {
    let rl = Arc::new(RateLimiter::parse("1/sec").unwrap());

    let missing = status_of(&rl, client_ip_header(), proxy(), None).await;
    let garbage = status_of(
        &rl,
        client_ip_header(),
        proxy(),
        Some((CLIENT_IP_HEADER, "not-an-address")),
    )
    .await;

    assert_eq!(missing, StatusCode::OK);
    assert_eq!(
        garbage,
        StatusCode::TOO_MANY_REQUESTS,
        "both requests are the proxy peer's"
    );
}

#[tokio::test]
async fn the_first_forwarded_for_entry_is_the_one_trusted() {
    // Documented behaviour of --trust-forwarded-for: correct only behind a
    // proxy that removes the header a client sends.
    let rl = Arc::new(RateLimiter::parse("1/sec").unwrap());
    let forwarded = |value: &'static str| {
        status_of(
            &rl,
            ClientIpSource::ForwardedFor,
            proxy(),
            Some(("x-forwarded-for", value)),
        )
    };

    assert_eq!(forwarded("10.0.0.1, 192.0.2.1").await, StatusCode::OK);
    assert_eq!(
        forwarded("10.0.0.1, 192.0.2.2").await,
        StatusCode::TOO_MANY_REQUESTS
    );
}

#[tokio::test]
async fn the_local_transport_is_not_rate_limited() {
    let rl = Arc::new(RateLimiter::parse("1/sec").unwrap());
    for _ in 0..3 {
        let status = router_with_client_ip(Arc::clone(&rl), ClientIpSource::Socket)
            .layer(axum::Extension(LocalTransport))
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap()
            .status();
        assert_eq!(status, StatusCode::OK);
    }
}
