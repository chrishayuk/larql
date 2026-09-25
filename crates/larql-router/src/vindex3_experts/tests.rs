use super::*;
use axum::{
    body::Bytes,
    extract::State,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use std::sync::{atomic::AtomicUsize, Arc};
fn binding() -> wire::Binding {
    use larql_router_protocol::{vindex3, vindex3_ffn::OperandIdentity};
    wire::Binding {
        program: vindex3::Binding {
            schema: 1,
            artifact: "a".repeat(64),
            backend: "cpu".into(),
            lowering: "cpu-production/v1".into(),
            start: 0,
            end: 1,
            layers: 1,
            hidden: 4,
        },
        expert_start: 0,
        expert_end: 4,
        regions: "test".into(),
        operands: vec![OperandIdentity {
            layer: 0,
            operand: "test".into(),
            representation: "MXFP4".into(),
            codec: "test".into(),
            realization: "test".into(),
            extent: "test".into(),
            dependencies: "test".into(),
        }],
    }
}
async fn open(
    State(mode): State<Arc<AtomicUsize>>,
    Json(mut binding): Json<wire::Binding>,
) -> Json<wire::Opened> {
    let fault = mode.load(Ordering::SeqCst);
    if fault == 20 {
        binding.program.artifact = "b".repeat(64);
    }
    Json(wire::Opened {
        version: if fault == 21 { 99 } else { 1 },
        handle: [7; 16],
        binding,
    })
}
async fn forward(State(mode): State<Arc<AtomicUsize>>, bytes: Bytes) -> axum::response::Response {
    let request = wire::decode_request(&bytes, 4, 2).unwrap();
    let rows: Vec<_> = request
        .experts
        .iter()
        .rev()
        .map(|id| (*id, request.row.clone()))
        .collect();
    let mut response =
        wire::encode_response(request.handle, request.sequence, request.layer, 4, &rows).unwrap();
    let mut content_type = wire::CONTENT_TYPE;
    match mode.load(Ordering::SeqCst) {
        1 => response[4] ^= 1,
        2 => response[20] ^= 1,
        3 => response[28] ^= 1,
        4 => {
            response.pop();
        }
        5 => response.push(0),
        6 => response[40..44].copy_from_slice(&99u32.to_le_bytes()),
        7 => response[60..64].copy_from_slice(&(rows[0].0 as u32).to_le_bytes()),
        8 => response[44..48].copy_from_slice(&f32::NAN.to_bits().to_le_bytes()),
        9 => content_type = "application/json",
        10 => return axum::http::StatusCode::SERVICE_UNAVAILABLE.into_response(),
        11 => {
            response.truncate(60);
            response[36..40].copy_from_slice(&1u32.to_le_bytes());
        }
        _ => {}
    }
    ([(axum::http::header::CONTENT_TYPE, content_type)], response).into_response()
}
#[tokio::test]
async fn http_rejects_changed_authority_and_corrupt_expert_replies() {
    let mode = Arc::new(AtomicUsize::new(0));
    let app = Router::new()
        .route(wire::PATH, get(|| async { Json(binding()) }))
        .route(wire::OPEN_PATH, post(open))
        .route(wire::BINARY_PATH, post(forward))
        .with_state(mode.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::task::spawn_blocking(move || {
        let urls = [url];
        let row = [0.0, -0.0, f32::from_bits(1), 1.25];
        let client = HttpExpertShards::connect(&urls, None).unwrap();
        let actual = client.forward(0, 0, &[3, 1], &row).unwrap();
        assert_eq!(actual[0].expert, 1);
        assert_eq!(
            actual[0]
                .row
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            row.map(f32::to_bits)
        );
        // Missing optional telemetry must not change numerical execution or
        // become a fabricated zero worker time in the diagnostic.
        let capture = profile::Capture::start().unwrap();
        let profiled = client.forward(0, 0, &[3, 1], &row).unwrap();
        let trace = capture.finish_provider_calls();
        assert_eq!(
            profiled[0]
                .row
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            row.map(f32::to_bits)
        );
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0]["complete"], true);
        assert_eq!(trace[0]["worker_profile_complete"], false);
        assert!(trace[0].get("worker").is_none());
        for fault in 1..=11 {
            mode.store(fault, Ordering::SeqCst);
            assert!(
                client.forward(0, 0, &[3, 1], &row).is_err(),
                "fault {fault}"
            );
        }
        for fault in [20, 21] {
            mode.store(fault, Ordering::SeqCst);
            assert!(HttpExpertShards::connect(&urls, None).is_err());
        }
        client.sequences[0].store(u64::MAX, Ordering::SeqCst);
        assert!(client
            .forward(0, 0, &[3, 1], &row)
            .err()
            .unwrap()
            .contains("sequence exhausted"));
    })
    .await
    .unwrap();
    server.abort();
}
