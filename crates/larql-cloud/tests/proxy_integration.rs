//! Integration test for the `larql-cloud-proxy` binary.
//!
//! Spawns:
//! - A mock OpenAI server that records requests and returns canned
//!   answers.
//! - The proxy binary, configured with `--provider local --base-url
//!   <mock>`.
//! - A test client that hits the proxy's `/v1/health`, `/v1/stats`,
//!   `/v1/infer`, `/v1/embeddings`, `/v1/chat/completions` and
//!   asserts the full plumbing works end to end.
//!
//! Verifies the contract pg_infer's `RemoteBackend` relies on:
//! `/v1/stats` reports `mode: "cloud-proxy"`, `/v1/infer` returns a
//! `predictions` array, `/v1/walk` returns 501 with a clear error.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use bytes::Bytes;
use http_body_util::Full;
use hyper::body::Incoming;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;

const PROXY_BIN: &str = env!("CARGO_BIN_EXE_larql-cloud-proxy");

type RequestLog = Arc<Mutex<Vec<(String, String)>>>;

async fn spawn_mock() -> (String, RequestLog) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("addr");
    let log: RequestLog = Arc::new(Mutex::new(Vec::new()));
    let log_bg = log.clone();
    tokio::spawn(async move {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
        loop {
            tokio::select! {
                _ = tokio::time::sleep_until(deadline) => return,
                res = listener.accept() => {
                    let (stream, _) = match res { Ok(v) => v, Err(_) => continue };
                    let log_conn = log_bg.clone();
                    tokio::spawn(async move {
                        let io = TokioIo::new(stream);
                        let svc = hyper::service::service_fn(move |req: Request<Incoming>| {
                            let log = log_conn.clone();
                            async move {
                                let path = req.uri().path().to_string();
                                let method = req.method().to_string();
                                log.lock().expect("lock").push((method, path.clone()));
                                let body = match path.as_str() {
                                    "/v1/chat/completions" => serde_json::json!({
                                        "id": "chat-1",
                                        "object": "chat.completion",
                                        "model": "test-model",
                                        "choices": [{
                                            "index": 0,
                                            "message": {"role": "assistant", "content": "Paris is the capital"},
                                            "finish_reason": "stop"
                                        }],
                                        "usage": {"prompt_tokens": 5, "completion_tokens": 4, "total_tokens": 9}
                                    }),
                                    "/v1/embeddings" => serde_json::json!({
                                        "object": "list",
                                        "model": "test-model",
                                        "data": [
                                            {"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0},
                                        ],
                                        "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1}
                                    }),
                                    _ => serde_json::json!({"error": "not found"}),
                                };
                                let bytes = serde_json::to_vec(&body).expect("vec");
                                Ok::<Response<Full<Bytes>>, std::io::Error>(
                                    Response::builder()
                                        .status(if path.starts_with("/v1/") { 200 } else { 404 })
                                        .header("content-type", "application/json")
                                        .body(Full::new(Bytes::from(bytes)))
                                        .expect("resp"))
                            }
                        });
                        let _ = hyper::server::conn::http1::Builder::new()
                            .serve_connection(io, svc).await;
                    });
                }
            }
        }
    });
    (format!("http://{addr}"), log)
}

fn pick_port() -> u16 {
    let l = std::net::TcpListener::bind("127.0.0.1:0").expect("port");
    let p = l.local_addr().expect("addr").port();
    drop(l);
    p
}

#[tokio::test]
async fn proxy_end_to_end_against_local_provider() {
    let (mock_url, log) = spawn_mock().await;
    let proxy_port = pick_port();

    // Spawn the proxy binary with stdout/stderr captured so we can
    // surface its log output if the test fails.
    let mut child = std::process::Command::new(PROXY_BIN)
        .args([
            "--provider",
            "local",
            "--base-url",
            &mock_url,
            "--model",
            "test-model",
            "--port",
            &proxy_port.to_string(),
            "--host",
            "127.0.0.1",
            "--log-level",
            "warn",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn proxy");

    // Wait up to 5s for the proxy to come up.
    let proxy_url = format!("http://127.0.0.1:{proxy_port}");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .expect("client");
    let mut up = false;
    for _ in 0..50 {
        if client
            .get(format!("{proxy_url}/v1/health"))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
        {
            up = true;
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    if !up {
        let _ = child.kill();
        let out = child.wait_with_output().ok();
        let stderr = out
            .as_ref()
            .map(|o| String::from_utf8_lossy(&o.stderr).into_owned())
            .unwrap_or_default();
        panic!("proxy did not come up; stderr=\n{stderr}");
    }

    // /v1/stats reports cloud-proxy mode.
    let stats: serde_json::Value = client
        .get(format!("{proxy_url}/v1/stats"))
        .send()
        .await
        .expect("stats")
        .json()
        .await
        .expect("stats json");
    assert_eq!(stats["mode"], "cloud-proxy");
    assert_eq!(stats["model"], "test-model");
    assert_eq!(stats["provider"], "openai-compat");

    // /v1/infer goes through to chat-completions on the mock.
    let infer: serde_json::Value = client
        .post(format!("{proxy_url}/v1/infer"))
        .json(&serde_json::json!({"prompt": "what is the capital of France?", "top": 3}))
        .send()
        .await
        .expect("infer")
        .json()
        .await
        .expect("infer json");
    let preds = infer["predictions"].as_array().expect("predictions array");
    assert_eq!(preds.len(), 3);
    assert_eq!(preds[0]["token"], "Paris");
    assert_eq!(preds[1]["token"], "is");
    assert_eq!(preds[2]["token"], "the");

    // /v1/embeddings returns OpenAI shape.
    let emb: serde_json::Value = client
        .post(format!("{proxy_url}/v1/embeddings"))
        .json(&serde_json::json!({"input": "hello"}))
        .send()
        .await
        .expect("embed")
        .json()
        .await
        .expect("embed json");
    assert_eq!(emb["object"], "list");
    let data = emb["data"].as_array().expect("data");
    assert_eq!(data.len(), 1);
    // f32 precision: 0.1 isn't exactly representable.  Allow a small tolerance.
    let v = data[0]["embedding"][0].as_f64().expect("f64");
    assert!((v - 0.1).abs() < 1e-6, "got {v}");

    // /v1/chat/completions passes through.
    let chat: serde_json::Value = client
        .post(format!("{proxy_url}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .expect("chat")
        .json()
        .await
        .expect("chat json");
    assert_eq!(
        chat["choices"][0]["message"]["content"],
        "Paris is the capital"
    );
    assert_eq!(chat["choices"][0]["finish_reason"], "stop");

    // /v1/walk returns 501 (vindex-only endpoint).
    let walk = client
        .get(format!("{proxy_url}/v1/walk?prompt=hello"))
        .send()
        .await
        .expect("walk");
    assert_eq!(walk.status().as_u16(), 501);

    // Mock saw exactly the requests we expected (3 chat + 1 embed).
    let logged = log.lock().expect("log");
    let chat_count = logged.iter().filter(|(_, p)| p == "/v1/chat/completions").count();
    let emb_count = logged.iter().filter(|(_, p)| p == "/v1/embeddings").count();
    assert_eq!(chat_count, 2, "expected 2 chat calls (infer + chat)");
    assert_eq!(emb_count, 1);

    let _ = child.kill();
    let _ = child.wait();
}
