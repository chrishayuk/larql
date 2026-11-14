//! V3 multi-row expert frames against real workers over HTTP: bitwise the
//! single-row operation, one request per contacted worker, whole-call
//! failure, and endpoints that refuse each other's frames.

use super::*;

use larql_inference::vindex3::routed_experts::{self, ExpertOutput, ExpertRow, ExpertTransport};
use larql_router::vindex3_experts::HttpExpertShards;
use larql_router_protocol::{vindex3_expert_rows as rows_wire, vindex3_experts as wire};
use larql_server::routes::vindex3_experts::ROWS_ACCOUNTING_HEADER;
use larql_vindex::format::vindex3::fixtures_routed::{
    miniature_routed, EXPERTS, HIDDEN, LAYERS, TOP_K, VOCAB,
};
use std::sync::atomic::{AtomicUsize, Ordering};

const LAYER: usize = 1;
/// Three rows over all four experts; expert 1 is selected by every row, and
/// under the split topology every row spans both workers.
const SELECTIONS: [[usize; TOP_K]; 3] = [[0, 3], [1, 2], [3, 1]];
const ROW_IDS: [u32; 3] = [40, 7, 19];
const PAIRS: usize = SELECTIONS.len() * TOP_K;
/// Transport faults injected on the multi-row path of the last worker.
const FAULT_ERROR: usize = 1;
const FAULT_DROP: usize = 2;
const FAULT_SUBSTITUTE: usize = 3;
const FAULT_FOREIGN_EXPERT: usize = 4;
const FAULTS: [usize; 4] = [
    FAULT_ERROR,
    FAULT_DROP,
    FAULT_SUBSTITUTE,
    FAULT_FOREIGN_EXPERT,
];

fn inputs() -> Vec<Vec<f32>> {
    (0..SELECTIONS.len())
        .map(|row| {
            (0..HIDDEN)
                .map(|i| ((row * HIDDEN + i) % 11) as f32 / 16.0 - 0.3)
                .collect()
        })
        .collect()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Counts network requests per path and can corrupt the last worker's
/// multi-row answer.
struct Transport {
    inner: Arc<HttpExpertShards>,
    single: Arc<AtomicUsize>,
    rows: Arc<AtomicUsize>,
    fault: Arc<AtomicUsize>,
}

impl ExpertTransport for Transport {
    fn bindings(&self) -> Vec<wire::Binding> {
        self.inner.bindings()
    }
    fn forward(
        &self,
        shard: usize,
        layer: usize,
        experts: &[usize],
        row: &[f32],
    ) -> Result<Vec<ExpertOutput>, String> {
        self.single.fetch_add(1, Ordering::SeqCst);
        self.inner.forward(shard, layer, experts, row)
    }
    fn forward_rows(
        &self,
        shard: usize,
        layer: usize,
        rows: &[rows_wire::Row],
    ) -> Result<Vec<rows_wire::Output>, String> {
        self.rows.fetch_add(1, Ordering::SeqCst);
        let mut outputs = self.inner.forward_rows(shard, layer, rows)?;
        if shard + 1 == self.inner.bindings().len() {
            match self.fault.load(Ordering::SeqCst) {
                FAULT_ERROR => return Err("injected worker timeout".into()),
                FAULT_DROP => {
                    outputs.pop();
                }
                FAULT_SUBSTITUTE => outputs[0].row = u32::MAX,
                // An expert the row did ask for elsewhere, answered by a
                // worker that does not own it.
                FAULT_FOREIGN_EXPERT => outputs[0].expert = EXPERTS + 1,
                _ => {}
            }
        }
        Ok(outputs)
    }
}

async fn post(
    app: axum::Router,
    path: &str,
    content_type: &str,
    accounted: bool,
    body: Vec<u8>,
) -> axum::response::Response {
    let mut request = Request::builder()
        .method("POST")
        .uri(path)
        .header(axum::http::header::CONTENT_TYPE, content_type);
    if accounted {
        request = request.header(wire::PROFILE_HEADER, "1");
    }
    app.oneshot(request.body(Body::from(body)).unwrap())
        .await
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn v3_multi_row_expert_frames_match_single_row_execution_and_fail_whole() {
    let checkpoint = tempfile::tempdir().unwrap();
    let container = tempfile::tempdir().unwrap();
    encode_fixture_container(
        miniature_routed,
        checkpoint.path(),
        container.path(),
        "routed-rows-fixture",
    );
    std::fs::write(
        container.path().join("tokenizer.json"),
        synthetic_tokenizer_json(VOCAB),
    )
    .unwrap();
    for topology in [
        vec![(0, EXPERTS)],
        vec![(0, EXPERTS / 2), (EXPERTS / 2, EXPERTS)],
    ] {
        let workers = topology.len();
        let mut servers = Vec::new();
        let mut urls = Vec::new();
        let mut apps = Vec::new();
        for (expert_start, expert_end) in topology {
            let LoadedArtifact::V3(model) = load_artifact(
                container.path().to_str().unwrap(),
                LoadVindexOptions {
                    ffn_only: true,
                    layer_range: Some((0, LAYERS)),
                    expert_filter: Some((expert_start, expert_end)),
                    ..Default::default()
                },
            )
            .unwrap() else {
                panic!("V3 worker")
            };
            let state = v3_state(container.path());
            state.model_set.write().unwrap().v3_models = vec![Arc::new(*model)];
            let app = larql_server::routes::single_model_router(state);
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            urls.push(format!("http://{}", listener.local_addr().unwrap()));
            apps.push(app.clone());
            servers.push(tokio::spawn(async move {
                axum::serve(listener, app).await.unwrap()
            }));
        }

        let path = container.path().to_path_buf();
        let client_urls = urls.clone();
        tokio::task::spawn_blocking(move || {
            let runtime = Vindex3Runtime::open(&path, COMPONENT, ProductionBackend::new()).unwrap();
            let (single, rows, fault) = (
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
                Arc::new(AtomicUsize::new(0)),
            );
            let provider = routed_experts::admit_expert_provider(
                &path,
                runtime.plan(),
                runtime.operands(),
                runtime.backend(),
                Transport {
                    inner: Arc::new(HttpExpertShards::connect(&client_urls, None).unwrap()),
                    single: single.clone(),
                    rows: rows.clone(),
                    fault: fault.clone(),
                },
            )
            .unwrap();
            let inputs = inputs();
            let batch: Vec<ExpertRow<'_>> = (0..SELECTIONS.len())
                .map(|i| ExpertRow {
                    id: ROW_IDS[i],
                    input: &inputs[i],
                    experts: &SELECTIONS[i],
                })
                .collect();

            // Each row alone, through the existing single-row frames.
            let mut alone = std::collections::BTreeMap::new();
            for row in &batch {
                for output in provider.apply(LAYER, row.input, row.experts).unwrap() {
                    alone.insert((row.id, output.expert), bits(&output.row));
                }
            }
            assert_eq!(alone.len(), PAIRS);
            let single_requests = single.load(Ordering::SeqCst);
            assert!(
                single_requests >= batch.len(),
                "at least one request per row"
            );
            assert_eq!(rows.load(Ordering::SeqCst), 0);

            for count in [1, batch.len()] {
                rows.store(0, Ordering::SeqCst);
                let outputs = provider.apply_rows(LAYER, &batch[..count]).unwrap();
                assert_eq!(outputs.len(), count * TOP_K);
                for output in &outputs {
                    assert_eq!(
                        Some(&bits(&output.output)),
                        alone.get(&(output.row, output.expert)),
                        "{workers} workers, {count} rows: row {} expert {}",
                        output.row,
                        output.expert
                    );
                }
                // One request per contacted worker, whatever the row count.
                assert!(rows.load(Ordering::SeqCst) <= workers);
                if count == batch.len() {
                    assert_eq!(rows.load(Ordering::SeqCst), workers);
                }
            }
            assert_eq!(
                single.load(Ordering::SeqCst),
                single_requests,
                "no single-row fallback"
            );

            // One worker answering wrongly fails the call, even though every
            // other worker answered correctly.
            for mode in FAULTS {
                fault.store(mode, Ordering::SeqCst);
                assert!(provider.apply_rows(LAYER, &batch).is_err(), "fault {mode}");
            }
            fault.store(0, Ordering::SeqCst);
            assert!(provider.apply_rows(LAYER, &batch).is_ok(), "healthy again");
            assert!(
                provider.apply_rows(LAYERS, &batch).is_err(),
                "unowned layer"
            );
            let foreign = [EXPERTS];
            assert!(provider
                .apply_rows(
                    LAYER,
                    &[ExpertRow {
                        id: 1,
                        input: &inputs[0],
                        experts: &foreign
                    }]
                )
                .is_err());
        })
        .await
        .unwrap();

        // Raw frames against the first worker's own router.
        let app = apps[0].clone();
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(wire::PATH)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        let binding: wire::Binding = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        let response = post(
            app.clone(),
            wire::OPEN_PATH,
            "application/json",
            false,
            serde_json::to_vec(&binding).unwrap(),
        )
        .await;
        let opened: wire::Opened = serde_json::from_slice(
            &axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap(),
        )
        .unwrap();
        let owned: Vec<usize> = (binding.expert_start..binding.expert_end)
            .take(TOP_K)
            .collect();
        let inputs = inputs();
        let frame = |handle, layer, experts: &[usize]| {
            let rows: Vec<rows_wire::Row> = (0..SELECTIONS.len())
                .map(|i| rows_wire::Row {
                    id: ROW_IDS[i],
                    experts: experts.to_vec(),
                    input: inputs[i].clone(),
                })
                .collect();
            rows_wire::encode_request(handle, 1, layer, HIDDEN, &rows).unwrap()
        };
        let good = frame(opened.handle, LAYER, &owned);

        let response = post(
            app.clone(),
            rows_wire::ROWS_PATH,
            rows_wire::CONTENT_TYPE,
            true,
            good.clone(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        let accounting: serde_json::Value = serde_json::from_slice(
            response
                .headers()
                .get(ROWS_ACCOUNTING_HEADER)
                .unwrap()
                .as_bytes(),
        )
        .unwrap();
        // Three rows sharing the same experts: grouped, but still one
        // transform per pair. This header never claims a saved read.
        assert_eq!(accounting["rows"], SELECTIONS.len());
        assert_eq!(accounting["pairs"], SELECTIONS.len() * owned.len());
        assert_eq!(accounting["distinct_experts"], owned.len());
        assert_eq!(accounting["transform_calls"], accounting["pairs"]);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let decoded =
            rows_wire::decode_response(&body, HIDDEN, rows_wire::MAX_ROWS, TOP_K).unwrap();
        assert_eq!(decoded.outputs.len(), SELECTIONS.len() * owned.len());

        let response = post(
            app.clone(),
            rows_wire::ROWS_PATH,
            rows_wire::CONTENT_TYPE,
            false,
            good.clone(),
        )
        .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers().get(ROWS_ACCOUNTING_HEADER).is_none());

        let single_frame =
            wire::encode_request(opened.handle, 1, LAYER, &owned, &inputs[0]).unwrap();
        let mut stale = opened.handle;
        stale[0] ^= 1;
        let unowned = [binding.expert_end % EXPERTS + if workers == 1 { EXPERTS } else { 0 }];
        for (name, path, content_type, body) in [
            (
                "single-row frame on the multi-row endpoint",
                rows_wire::ROWS_PATH,
                rows_wire::CONTENT_TYPE,
                single_frame.clone(),
            ),
            (
                "multi-row frame on the single-row endpoint",
                wire::BINARY_PATH,
                wire::CONTENT_TYPE,
                good.clone(),
            ),
            (
                "single-row content type",
                rows_wire::ROWS_PATH,
                wire::CONTENT_TYPE,
                good.clone(),
            ),
            (
                "stale handle",
                rows_wire::ROWS_PATH,
                rows_wire::CONTENT_TYPE,
                frame(stale, LAYER, &owned),
            ),
            (
                "layer outside the worker",
                rows_wire::ROWS_PATH,
                rows_wire::CONTENT_TYPE,
                frame(opened.handle, LAYERS, &owned),
            ),
            (
                "expert outside the worker",
                rows_wire::ROWS_PATH,
                rows_wire::CONTENT_TYPE,
                frame(opened.handle, LAYER, &unowned),
            ),
            (
                "truncated frame",
                rows_wire::ROWS_PATH,
                rows_wire::CONTENT_TYPE,
                good[..good.len() - 1].to_vec(),
            ),
        ] {
            let response = post(app.clone(), path, content_type, false, body).await;
            assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{name}");
        }
        for server in servers {
            server.abort();
        }
    }
}
