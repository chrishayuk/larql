//! V3 EOS, V2-only refusals, backend selection, and HTTP layer workers.

use super::*;

/// **A container's declared end-of-turn token stops served generation.**
///
/// The V3 driver judges EOS on ids alone, so the server must hand it the
/// ids the container declares — an empty built-in set means every V3
/// completion runs to `max_tokens`. The direct arm says what the fixture
/// emits for PROMPT; its second id is declared as EOS; the same request
/// then finishes with `stop` after exactly the tokens before that id,
/// buffered and streamed, and a chat turn stops the same way. The
/// identical container without the declaration runs to `length`, which
/// is the control that the stop came from the declaration.
#[tokio::test]
async fn v3_generation_stops_on_the_containers_declared_eos_token() {
    // Control: no declaration, the budget is filled.
    let plain = v3_container();
    let emitted = direct_arm(plain.path(), NEW_TOKENS);
    let eos_id = emitted[1].0;
    let expected_len = emitted
        .iter()
        .position(|(id, _)| *id == eos_id)
        .expect("the declared id is one the fixture emits");
    let expected_text: String = emitted[..expected_len]
        .iter()
        .map(|(_, t)| t.as_str())
        .collect();
    let plain_app = larql_server::routes::single_model_router(v3_state(plain.path()));
    let resp = common::post_json(
        plain_app.clone(),
        "/v1/completions",
        serde_json::json!({"prompt": PROMPT, "max_tokens": NEW_TOKENS}),
    )
    .await;
    let json = common::body_json(resp.into_body()).await;
    assert_eq!(
        json["choices"][0]["finish_reason"], "length",
        "control: {json}"
    );
    assert_eq!(
        json["usage"]["completion_tokens"], NEW_TOKENS,
        "control: {json}"
    );
    let plain_chat = common::post_json(
        plain_app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "[1]"}],
            "max_tokens": NEW_TOKENS,
        }),
    )
    .await;
    let plain_chat = common::body_json(plain_chat.into_body()).await;
    assert_eq!(
        plain_chat["choices"][0]["finish_reason"], "length",
        "control: {plain_chat}"
    );
    let chat_ids = ids_in_surface(
        plain_chat["choices"][0]["message"]["content"]
            .as_str()
            .unwrap(),
    );
    assert!(
        chat_ids.len() >= 2,
        "control: the chat turn must emit ids: {plain_chat}"
    );
    let chat_eos_id = chat_ids[1];
    let chat_expected_len = chat_ids.iter().position(|&id| id == chat_eos_id).unwrap();

    // Declared: the same fixture with `generation_config.json`.
    let container = v3_container_declaring_eos(eos_id);
    let app = larql_server::routes::single_model_router(v3_state(container.path()));

    let resp = common::post_json(
        app.clone(),
        "/v1/completions",
        serde_json::json!({"prompt": PROMPT, "max_tokens": NEW_TOKENS}),
    )
    .await;
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let json = common::body_json(resp.into_body()).await;
    assert_eq!(
        json["choices"][0]["finish_reason"], "stop",
        "buffered: {json}"
    );
    assert_eq!(
        json["usage"]["completion_tokens"], expected_len,
        "buffered: {json}"
    );
    assert_eq!(
        json["choices"][0]["text"],
        expected_text.as_str(),
        "buffered: {json}"
    );

    let resp = common::post_json(
        app.clone(),
        "/v1/completions",
        serde_json::json!({"prompt": PROMPT, "max_tokens": NEW_TOKENS, "stream": true}),
    )
    .await;
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let chunks = sse_chunks(core::str::from_utf8(&bytes).unwrap());
    assert_eq!(
        chunks.len(),
        expected_len + 1,
        "streamed: one chunk per kept token plus the stop"
    );
    assert_eq!(
        chunks[expected_len]["choices"][0]["finish_reason"], "stop",
        "streamed: {chunks:?}"
    );

    // Chat: its own prompt, its own sequence, the same stop.
    let chat_container = v3_container_declaring_eos(chat_eos_id);
    let chat_app = larql_server::routes::single_model_router(v3_state(chat_container.path()));
    let resp = common::post_json(
        chat_app,
        "/v1/chat/completions",
        serde_json::json!({
            "messages": [{"role": "user", "content": "[1]"}],
            "max_tokens": NEW_TOKENS,
        }),
    )
    .await;
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let json = common::body_json(resp.into_body()).await;
    assert_eq!(json["choices"][0]["finish_reason"], "stop", "chat: {json}");
    assert_eq!(
        json["usage"]["completion_tokens"], chat_expected_len,
        "chat: {json}"
    );
    assert_eq!(
        ids_in_surface(json["choices"][0]["message"]["content"].as_str().unwrap()),
        chat_ids[..chat_expected_len],
        "chat: {json}"
    );
}

/// **A loaded-but-unsupported model never masquerades as absent.**
///
/// The V2-only surfaces resolve a VINDEX2 model; on a server that has
/// only a VINDEX3 container bound they used to answer 404 "no model
/// loaded" while `/v1/models` listed the container. The rule is three-
/// way: no model is 404, a V2 model takes the route, and a V3 model on
/// a V2-only capability is 501 naming VINDEX3 — a truthful refusal, not
/// V3 support, which those routes do not have.
#[tokio::test]
async fn v2_only_surfaces_refuse_a_v3_container_as_unsupported_not_absent() {
    let container = v3_container();
    let state = v3_state(container.path());
    let app = larql_server::routes::single_model_router(state.clone());

    let gets = [
        "/v1/describe?entity=%5B1%5D",
        "/v1/relations",
        "/v1/patches",
        "/v1/walk?prompt=%5B1%5D&top=1",
    ];
    for path in gets {
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(path)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = resp.status();
        let json = common::body_json(resp.into_body()).await;
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{path}: {json}");
        let error = json["error"].as_str().unwrap_or_default().to_string();
        assert!(
            error.contains("VINDEX3"),
            "{path}: refusal must name VINDEX3: {json}"
        );
        assert!(
            !error.contains("not found"),
            "{path}: a bound model is not absent: {json}"
        );
    }
    let resp = common::post_json(
        app.clone(),
        "/v1/infer",
        serde_json::json!({"prompt": "[1]"}),
    )
    .await;
    let status = resp.status();
    let json = common::body_json(resp.into_body()).await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "/v1/infer: {json}");
    assert!(
        json["error"]
            .as_str()
            .unwrap_or_default()
            .contains("VINDEX3"),
        "{json}"
    );

    // An OpenAI-shaped V2-only route answers in the OpenAI envelope.
    let resp = common::post_json(
        app.clone(),
        "/v1/embeddings",
        serde_json::json!({"input": "[1]"}),
    )
    .await;
    let status = resp.status();
    let json = common::body_json(resp.into_body()).await;
    assert_eq!(
        status,
        StatusCode::NOT_IMPLEMENTED,
        "/v1/embeddings: {json}"
    );
    assert_eq!(json["error"]["type"], "not_implemented_error", "{json}");
    assert!(
        json["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("VINDEX3"),
        "{json}"
    );

    // gRPC: the same rule, in gRPC's vocabulary.
    use larql_server::grpc::proto::vindex_service_server::VindexService;
    let svc = larql_server::grpc::VindexGrpcService {
        state: state.clone(),
    };
    let err = svc
        .get_stats(tonic::Request::new(
            larql_server::grpc::proto::StatsRequest {},
        ))
        .await
        .expect_err("a V3-only server refuses the V2 gRPC surface");
    assert_eq!(err.code(), tonic::Code::Unimplemented, "{err}");
    assert!(err.message().contains("VINDEX3"), "{err}");

    // Control: nothing bound is still "absent".
    let empty = larql_server::routes::single_model_router(empty_state());
    let resp = empty
        .oneshot(
            Request::builder()
                .method("GET")
                .uri("/v1/describe?entity=%5B1%5D")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    let status = resp.status();
    let json = common::body_json(resp.into_body()).await;
    assert_eq!(status, StatusCode::NOT_FOUND, "control: {json}");
    assert_eq!(json["error"], "no model loaded", "control: {json}");
}

#[tokio::test]
async fn lifecycle_reports_selected_backend_and_refuses_an_implicit_switch() {
    let container = v3_container();
    let state = common::state(vec![]);
    let app = larql_server::routes::single_model_router(state);
    let path = container.path().to_string_lossy();
    let loaded = common::post_json(
        app.clone(),
        "/v1/runtime/model",
        serde_json::json!({"path": path, "backend": "cpu"}),
    )
    .await;
    assert_eq!(loaded.status(), StatusCode::OK);
    let body = common::body_json(loaded.into_body()).await;
    assert_eq!(body["backend"]["selected"], "cpu");
    let conflict = common::post_json(
        app.clone(),
        "/v1/runtime/model",
        serde_json::json!({"path": path, "backend": "metal"}),
    )
    .await;
    assert_eq!(conflict.status(), StatusCode::CONFLICT);
    let bad = common::post_json(
        app,
        "/v1/runtime/model",
        serde_json::json!({"path": path, "backend": "typo"}),
    )
    .await;
    assert_eq!(bad.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[cfg(not(all(feature = "vindex3-metal", target_os = "macos")))]
#[test]
fn unavailable_metal_is_refused_instead_of_serving_on_cpu() {
    let container = v3_container();
    let result = load_artifact(
        container.path().to_str().unwrap(),
        LoadVindexOptions {
            v3_backend: larql_server::vindex3::V3Backend::Metal,
            ..Default::default()
        },
    );
    let err = result
        .err()
        .expect("Metal must not fall back to CPU")
        .to_string();
    assert!(err.contains("vindex3-metal"), "{err}");
}

/// Explicit opt-in: requires a real Metal device; it never passes by skipping
/// device creation or falling back to CPU. Run serially with vindex3-metal.
#[cfg(all(feature = "vindex3-metal", target_os = "macos"))]
#[test]
#[ignore = "requires a real Metal device"]
fn selected_metal_backend_executes_the_v3_fixture() {
    use larql_server::vindex3::{load_v3_model_with_backend, V3Backend};
    let container = v3_container();
    let model = load_v3_model_with_backend(container.path(), V3Backend::Metal).unwrap();
    assert_eq!(model.backend, V3Backend::Metal);
    let ids = model
        .tokenizer
        .encode(PROMPT, true)
        .unwrap()
        .get_ids()
        .to_vec();
    let result = generate_v3(
        &model,
        &ids,
        NEW_TOKENS,
        SamplingConfig::greedy(),
        &EosConfig::builtin(),
        |_, _| {},
    )
    .unwrap();
    let expected: Vec<_> = direct_arm(container.path(), NEW_TOKENS)
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(
        result.ids, expected,
        "fixture's greedy tokens must agree with CPU"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn v3_layer_workers_over_http_match_local_execution_and_refuse_bad_requests() {
    use larql_inference::vindex3::{
        distributed::{artifact_identity, DistributedSession},
        LogitsSession,
    };
    use larql_router::vindex3::HttpLayerShards;
    use larql_router_protocol::vindex3::{Binding, PATH};
    use larql_vindex::format::vindex3::opplan::exec::prepared::ExecutionSlice;
    let container = v3_container();
    let full = larql_server::routes::single_model_router(v3_state(container.path()));
    let response = full
        .oneshot(Request::builder().uri(PATH).body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::NOT_IMPLEMENTED);
    for (range, backend) in [
        ((2, 3), larql_server::vindex3::V3Backend::Cpu),
        ((1, 0), larql_server::vindex3::V3Backend::Cpu),
        ((0, usize::MAX), larql_server::vindex3::V3Backend::Cpu),
        ((0, 0), larql_server::vindex3::V3Backend::Metal),
    ] {
        assert!(load_artifact(
            container.path().to_str().unwrap(),
            LoadVindexOptions {
                layer_range: Some(range),
                v3_backend: backend,
                ..Default::default()
            }
        )
        .is_err());
    }
    let mut urls = Vec::new();
    let mut servers = Vec::new();
    for layer in 0..2 {
        let state = v3_state(container.path());
        let LoadedArtifact::V3(model) = load_artifact(
            container.path().to_str().unwrap(),
            LoadVindexOptions {
                layer_range: Some(
                    larql_server::bootstrap::parse_layer_range(&format!("{layer}-{layer}"))
                        .unwrap(),
                ),
                ..Default::default()
            },
        )
        .unwrap() else {
            panic!("V3")
        };
        assert_eq!(model.runtime.operands().layer_count(), 1);
        assert!(!model.runtime.operands().has_output());
        state.model_set.write().unwrap().v3_models = vec![Arc::new(*model)];
        let app = larql_server::routes::single_model_router(state);
        // A worker never serves a partial stack as a complete language model.
        let denied = common::post_json(
            app.clone(),
            "/v1/completions",
            serde_json::json!({"prompt":PROMPT,"max_tokens":1}),
        )
        .await;
        assert!(!denied.status().is_success());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        urls.push(format!("http://{}", listener.local_addr().unwrap()));
        servers.push(tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap()
        }));
    }
    let client = reqwest::Client::new();
    let binding: Binding = client
        .get(format!("{}{PATH}", urls[0]))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    let mut wrong = binding.clone();
    wrong.end = 2;
    for body in [
        serde_json::json!({"binding":wrong,"rows":[vec![0.0;binding.hidden]]}),
        serde_json::json!({"binding":binding,"rows":[[0.0]]}),
    ] {
        assert_eq!(
            client
                .post(format!("{}{PATH}", urls[0]))
                .json(&body)
                .send()
                .await
                .unwrap()
                .status(),
            StatusCode::BAD_REQUEST
        );
    }
    let path = container.path().to_path_buf();
    tokio::task::spawn_blocking(move || {
        let runtime = Vindex3Runtime::open(&path, COMPONENT, ProductionBackend::new()).unwrap();
        let identity = artifact_identity(&path, runtime.plan()).unwrap();
        let endpoints = runtime.prepare_slice(ExecutionSlice::Endpoints).unwrap();
        let transport = HttpLayerShards::connect(&urls, None).unwrap();
        let mut remote = DistributedSession::new(
            endpoints.plan(),
            endpoints.operands(),
            endpoints.backend(),
            &identity,
            endpoints.store().into(),
            transport,
        )
        .unwrap();
        let local = Vindex3Runtime::open(&path, COMPONENT, ProductionBackend::new())
            .unwrap()
            .prepare()
            .unwrap();
        let mut kv = CanonicalKvState::new();
        let mut reference = local.session_with_kv(&mut kv).unwrap();
        // Sliding window is three: this fixture actually crosses its boundary.
        for id in [3, 17, 28, 0, 11, 3, 17, 28, 0, 11] {
            let a = reference.step(id).unwrap();
            let b = remote.step(id).unwrap();
            let delta = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(delta < 1e-5, "HTTP logit delta {delta}");
        }
    })
    .await
    .unwrap();
    for server in servers {
        server.abort();
    }
}

/// CPU serving binds the stored bytes: its size policy would otherwise
/// requantise large BF16 matrices to Q8 at load, a lossy change of
/// arithmetic that depends on the host's L2 (PREC-DIAG-1). Metal keeps its
/// declared representation.
#[test]
fn cpu_serving_binds_the_stored_bytes_and_manufactures_nothing() {
    use larql_server::vindex3::{load_v3_model_with_backend, V3Backend};
    use larql_vindex::format::vindex3::opplan::exec::operands::RepresentationSource;
    assert_eq!(
        V3Backend::Cpu.representation_source(),
        RepresentationSource::Stored
    );
    assert_eq!(
        V3Backend::Metal.representation_source(),
        RepresentationSource::Auto
    );
    let container = v3_container();
    let model = load_v3_model_with_backend(container.path(), V3Backend::Cpu).unwrap();
    let store = model.runtime.store();
    assert_eq!(store.representation_source(), RepresentationSource::Stored);
    assert_eq!(store.runtime_quantised(), 0);
}
