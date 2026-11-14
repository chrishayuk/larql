//! V3 routed-expert HTTP grid: order, ownership and failed-step history.

use super::*;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn v3_routed_expert_http_grid_preserves_order_ownership_and_failed_step_history() {
    use larql_inference::vindex3::{
        routed_experts::{self, ExpertOutput, ExpertTransport, RoutedExpertSession},
        LogitsSession,
    };
    use larql_router::vindex3_experts::HttpExpertShards;
    use larql_router_protocol::vindex3_experts as wire;
    use larql_vindex::format::vindex3::fixtures_routed::{miniature_routed, VOCAB};
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Transport {
        inner: Arc<HttpExpertShards>,
        bindings: Vec<wire::Binding>,
        fault: Arc<AtomicUsize>,
    }
    impl ExpertTransport for Transport {
        fn bindings(&self) -> Vec<wire::Binding> {
            self.bindings.clone()
        }
        fn forward(
            &self,
            shard: usize,
            layer: usize,
            ids: &[usize],
            row: &[f32],
        ) -> Result<Vec<ExpertOutput>, String> {
            if layer == 1 && self.fault.load(Ordering::SeqCst) == 1 {
                return Err("injected worker timeout".into());
            }
            let mut rows = self.inner.forward(shard, layer, ids, row)?;
            if layer == 1 {
                match self.fault.load(Ordering::SeqCst) {
                    2 => {
                        rows.pop();
                    }
                    3 => rows[0].expert = 99,
                    4 => rows[0].row[0] = f32::NAN,
                    5 => {
                        rows[0].row.pop();
                    }
                    _ => {}
                }
            }
            rows.reverse();
            Ok(rows)
        }
        fn forward_rows(
            &self,
            shard: usize,
            layer: usize,
            rows: &[larql_router_protocol::vindex3_expert_rows::Row],
        ) -> Result<Vec<larql_router_protocol::vindex3_expert_rows::Output>, String> {
            self.inner.forward_rows(shard, layer, rows)
        }
    }
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_routed,
        checkpoint.path(),
        container.path(),
        "routed-serve-fixture",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        synthetic_tokenizer_json(VOCAB),
    )
    .unwrap();
    for topology in [
        vec![(0, 2, 0, 4)],
        vec![(0, 2, 0, 2), (0, 2, 2, 4)],
        vec![(0, 1, 0, 4), (1, 2, 0, 1), (1, 2, 1, 4)],
    ] {
        let mut servers = Vec::new();
        let mut urls = Vec::new();
        for (start, end, expert_start, expert_end) in topology {
            let LoadedArtifact::V3(model) = load_artifact(
                container.path().to_str().unwrap(),
                LoadVindexOptions {
                    ffn_only: true,
                    layer_range: Some((start, end)),
                    expert_filter: Some((expert_start, expert_end)),
                    ..Default::default()
                },
            )
            .unwrap() else {
                panic!("V3 worker")
            };
            assert!(model.runtime.operands().routed_experts().is_some());
            assert!(!model.runtime.operands().has_output());
            let state = v3_state(container.path());
            state.model_set.write().unwrap().v3_models = vec![Arc::new(*model)];
            let app = larql_server::routes::single_model_router(state);
            assert!(!common::post_json(
                app.clone(),
                "/v1/completions",
                serde_json::json!({"prompt":PROMPT,"max_tokens":1})
            )
            .await
            .status()
            .is_success());
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            urls.push(format!("http://{}", listener.local_addr().unwrap()));
            servers.push(tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap()
            }));
        }
        let path = container.path().to_path_buf();
        tokio::task::spawn_blocking(move || {
            let runtime = Vindex3Runtime::open(&path, COMPONENT, ProductionBackend::new()).unwrap();
            let inner = Arc::new(HttpExpertShards::connect(&urls, None).unwrap());
            let bindings = inner.bindings();
            let fault = Arc::new(AtomicUsize::new(0));
            let make = |bindings| Transport {
                inner: inner.clone(),
                bindings,
                fault: fault.clone(),
            };
            // Each mutated field must independently trip admission before local payload reads.
            for mutation in 0..7 {
                let mut bad = bindings.clone();
                let want = match mutation {
                    0 => {
                        bad.clear();
                        "incomplete"
                    }
                    1 => {
                        bad.push(bad[0].clone());
                        "overlapping"
                    }
                    2 => {
                        bad[0].program.artifact = "0".repeat(64);
                        "artifact"
                    }
                    3 => {
                        bad[0].program.lowering = "cpu-production/v999".into();
                        "provider"
                    }
                    4 => {
                        bad[0].regions.push(' ');
                        "region"
                    }
                    5 => {
                        bad[0].program.hidden += 1;
                        "dimensions"
                    }
                    _ => {
                        bad[0].operands[0].representation = "different".into();
                        "representation"
                    }
                };
                let before = runtime.operands().store().bytes_read();
                let error = match routed_experts::prepare_coordinator(
                    &path,
                    runtime.plan(),
                    runtime.operands(),
                    runtime.backend(),
                    make(bad),
                ) {
                    Ok(_) => panic!("accepted mutation {mutation}"),
                    Err(e) => e.to_string(),
                };
                assert!(error.contains(want), "{mutation}: {error}");
                assert_eq!(runtime.operands().store().bytes_read(), before);
            }
            let ops = routed_experts::prepare_coordinator(
                &path,
                runtime.plan(),
                runtime.operands(),
                runtime.backend(),
                make(bindings.clone()),
            )
            .unwrap();
            assert_eq!(ops.residency_census().ffn.total(), 0);
            let mut remote_a = RoutedExpertSession::new(
                runtime.plan(),
                &ops,
                runtime.backend(),
                &row_continuation(runtime.plan()),
            )
            .unwrap();
            let mut remote_b = RoutedExpertSession::new(
                runtime.plan(),
                &ops,
                runtime.backend(),
                &row_continuation(runtime.plan()),
            )
            .unwrap();
            let mut local_a = runtime.session(&row_continuation(runtime.plan())).unwrap();
            let mut local_b = runtime.session(&row_continuation(runtime.plan())).unwrap();
            let bits = |row: Vec<f32>| row.into_iter().map(f32::to_bits).collect::<Vec<_>>();
            for id in [3, 17, 28, 0, 11, 3, 17, 28, 0, 11] {
                let capture =
                    larql_inference::vindex3::dense_ffn::profile::Capture::start().unwrap();
                let profiled = remote_a.step(id).unwrap();
                let trace = capture.finish();
                assert_eq!(bits(profiled), bits(local_a.step(id).unwrap()));
                assert_eq!(trace.len(), 1);
                assert!(trace[0].complete);
                let calls = &trace[0].provider_calls;
                assert_eq!(
                    calls.iter().filter(|c| c["kind"] == "routed_ffn").count(),
                    2
                );
                assert_eq!(
                    calls
                        .iter()
                        .filter(|c| c["kind"] == "expert_fanout")
                        .count(),
                    2
                );
                for shard in calls.iter().filter(|c| c["kind"] == "expert_shard") {
                    assert_eq!(shard["complete"], true);
                    let transport = &shard["transport"][0];
                    assert_eq!(transport["worker_profile_complete"], true);
                    assert_eq!(transport["complete"], true);
                    let count = shard["selected_count"].as_u64().unwrap();
                    assert_eq!(transport["request_bytes"], 40 + count * 4 + 32 * 4);
                    assert_eq!(transport["response_bytes"], 40 + count * (4 + 32 * 4));
                    assert!(
                        transport["worker"]["experts_ns"].as_u64().unwrap()
                            <= transport["worker"]["execute_ns"].as_u64().unwrap()
                    );
                    assert!(
                        shard["dispatch_finish_ns"].as_u64().unwrap()
                            >= shard["dispatch_start_ns"].as_u64().unwrap()
                    );
                }
                assert_eq!(
                    bits(remote_b.step((id + 1) % VOCAB as u32).unwrap()),
                    bits(local_b.step((id + 1) % VOCAB as u32).unwrap())
                );
            }
            for mode in 1..=5 {
                let mut failed = RoutedExpertSession::new(
                    runtime.plan(),
                    &ops,
                    runtime.backend(),
                    &row_continuation(runtime.plan()),
                )
                .unwrap();
                assert_eq!(
                    bits(failed.step(3).unwrap()),
                    bits(
                        runtime
                            .session(&row_continuation(runtime.plan()))
                            .unwrap()
                            .step(3)
                            .unwrap()
                    )
                );
                fault.store(mode, Ordering::SeqCst);
                assert!(failed.step(17).is_err());
                assert_eq!(failed.position(), 1);
                fault.store(0, Ordering::SeqCst);
                assert!(failed.step(17).unwrap_err().to_string().contains("invalid"));
                let mut recovered = RoutedExpertSession::new(
                    runtime.plan(),
                    &ops,
                    runtime.backend(),
                    &row_continuation(runtime.plan()),
                )
                .unwrap();
                let mut local = runtime.session(&row_continuation(runtime.plan())).unwrap();
                for id in [3, 17] {
                    assert_eq!(
                        bits(recovered.step(id).unwrap()),
                        bits(local.step(id).unwrap())
                    );
                }
            }
            let client = reqwest::blocking::Client::new();
            let b = &bindings[0];
            let open = format!("{}{}", urls[0], wire::OPEN_PATH);
            let opened: wire::Opened = client.post(&open).json(b).send().unwrap().json().unwrap();
            let mut bad = b.clone();
            bad.regions.push(' ');
            assert_eq!(
                client.post(&open).json(&bad).send().unwrap().status(),
                StatusCode::BAD_REQUEST
            );
            let good = wire::encode_request(
                opened.handle,
                1,
                b.program.start,
                &[b.expert_start],
                &vec![0.1; b.program.hidden],
            )
            .unwrap();
            let post = |body: Vec<u8>| {
                client
                    .post(format!("{}{}", urls[0], wire::BINARY_PATH))
                    .header(reqwest::header::CONTENT_TYPE, wire::CONTENT_TYPE)
                    .body(body)
                    .send()
                    .unwrap()
            };
            let first = post(good.clone())
                .error_for_status()
                .unwrap()
                .bytes()
                .unwrap();
            let profiled = client
                .post(format!("{}{}", urls[0], wire::BINARY_PATH))
                .header(reqwest::header::CONTENT_TYPE, wire::CONTENT_TYPE)
                .header(wire::PROFILE_HEADER, "1")
                .body(good.clone())
                .send()
                .unwrap()
                .error_for_status()
                .unwrap();
            assert!(profiled.headers().contains_key(wire::PROFILE_HEADER));
            assert_eq!(first, profiled.bytes().unwrap());
            assert_eq!(
                first,
                post(good.clone())
                    .error_for_status()
                    .unwrap()
                    .bytes()
                    .unwrap()
            );
            for mutation in 0..6 {
                let mut bad = good.clone();
                match mutation {
                    0 => bad[4] ^= 1,
                    1 => bad[28..32].copy_from_slice(&(b.program.end as u32).to_le_bytes()),
                    2 => bad[40..44].copy_from_slice(&(b.expert_end as u32).to_le_bytes()),
                    3 => {
                        bad.pop();
                    }
                    4 => bad.push(0),
                    _ => bad[44..48].copy_from_slice(&f32::NAN.to_bits().to_le_bytes()),
                }
                assert_eq!(
                    post(bad).status(),
                    StatusCode::BAD_REQUEST,
                    "worker mutation {mutation}"
                );
            }
        })
        .await
        .unwrap();
        for server in servers {
            server.abort();
        }
    }
}
