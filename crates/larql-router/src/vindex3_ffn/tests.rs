use super::*;
use axum::{
    body::Bytes,
    extract::State,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use std::sync::{atomic::AtomicUsize, Arc};

fn binding() -> Binding {
    use larql_router_protocol::{vindex3, vindex3_ffn::OperandIdentity};
    Binding {
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
        operands: vec![OperandIdentity {
            layer: 0,
            operand: "test".into(),
            representation: "F32".into(),
            codec: "test".into(),
            realization: "test".into(),
            extent: "test".into(),
            dependencies: "test".into(),
        }],
    }
}
async fn open(
    State(mode): State<Arc<AtomicUsize>>,
    Json(request): Json<Binding>,
) -> Json<binary::Opened> {
    let mut binding = request;
    let fault = mode.load(Ordering::SeqCst);
    if fault == 10 {
        binding.program.artifact = "b".repeat(64);
    }
    Json(binary::Opened {
        version: if fault == 11 { 99 } else { binary::VERSION },
        handle: [7; 16],
        binding,
    })
}
async fn forward(State(mode): State<Arc<AtomicUsize>>, bytes: Bytes) -> axum::response::Response {
    let frame = binary::decode(&bytes, Direction::Request, 4).unwrap();
    let mut response = binary::encode(
        Direction::Response,
        frame.handle,
        frame.sequence,
        frame.layer as usize,
        &frame.row,
    )
    .unwrap();
    let fault = mode.load(Ordering::SeqCst);
    match fault {
        1 => response[4] ^= 1,
        2 => response[20] ^= 1,
        3 => response[28] ^= 1,
        4 => {
            response.pop();
        }
        5 => response[36..40].copy_from_slice(&f32::NAN.to_bits().to_le_bytes()),
        7 => return axum::http::StatusCode::SERVICE_UNAVAILABLE.into_response(),
        8 => response[0] = 0,
        9 => response.push(0),
        _ => {}
    }
    (
        [(
            axum::http::header::CONTENT_TYPE,
            if fault == 6 {
                "application/json"
            } else {
                binary::CONTENT_TYPE
            },
        )],
        response,
    )
        .into_response()
}
#[tokio::test]
async fn binary_client_refuses_mismatched_open_and_corrupt_or_uncorrelated_replies() {
    let mode = Arc::new(AtomicUsize::new(0));
    let app = Router::new()
        .route(PATH, get(|| async { Json(binding()) }))
        .route(binary::OPEN_PATH, post(open))
        .route(binary::PATH, post(forward))
        .with_state(mode.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    tokio::task::spawn_blocking(move || {
        let urls = vec![url];
        let client = HttpFfnShards::connect_binary(&urls, None).unwrap();
        let row = [0.0, -0.0, f32::from_bits(1), 1.25];
        let actual = client.forward_bound(0, 0, &row, &binding()).unwrap();
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            row.map(f32::to_bits)
        );
        for fault in 1..=9 {
            mode.store(fault, Ordering::SeqCst);
            assert!(
                client.forward_bound(0, 0, &row, &binding()).is_err(),
                "fault {fault}"
            );
        }
        for fault in 10..=11 {
            mode.store(fault, Ordering::SeqCst);
            assert!(HttpFfnShards::connect_binary(&urls, None).is_err());
        }
        mode.store(0, Ordering::SeqCst);
        client.sequences[0].store(u64::MAX, Ordering::Relaxed);
        assert!(client.forward_bound(0, 0, &row, &binding()).is_err());
    })
    .await
    .unwrap();
    server.abort();
}
