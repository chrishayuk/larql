//! The multi-row client: one exchange per call, every `(row, expert)`
//! answered exactly once or the call fails, and a worker without the route
//! is reported as such rather than retried row by row.
use super::tests::{binding, open};
use super::*;
use axum::{
    body::Bytes,
    extract::State,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use std::sync::{atomic::AtomicUsize, Arc};

const HIDDEN: usize = 4;
const LAYER: usize = 0;
const MAX_SELECTED: usize = 2;
/// Byte offsets of response header fields the stub corrupts.
const HANDLE_AT: usize = 4;
const SEQUENCE_AT: usize = 20;
const LAYER_AT: usize = 28;
const ROWS_AT: usize = 36;
/// First pair's row id and expert id, just after the 44-byte header.
const FIRST_ROW_AT: usize = rows_wire::HEADER_BYTES;
const FIRST_EXPERT_AT: usize = rows_wire::HEADER_BYTES + 4;
const FIRST_VALUE_AT: usize = rows_wire::HEADER_BYTES + 8;
/// Stub modes that answer something other than what was asked.
const FAULTS: std::ops::RangeInclusive<usize> = 1..=13;

fn requested() -> Vec<rows_wire::Row> {
    vec![
        rows_wire::Row {
            id: 8,
            experts: vec![3, 1],
            input: vec![0.0, -0.0, f32::from_bits(1), 1.25],
        },
        rows_wire::Row {
            id: 2,
            experts: vec![1],
            input: vec![4.0, 3.0, 2.0, 1.0],
        },
    ]
}

/// Echo each row's input as every one of its expert outputs, in reverse.
async fn rows(State(mode): State<Arc<AtomicUsize>>, bytes: Bytes) -> axum::response::Response {
    let request =
        rows_wire::decode_request(&bytes, HIDDEN, rows_wire::MAX_ROWS, MAX_SELECTED).unwrap();
    let mut outputs: Vec<rows_wire::Output> = request
        .rows
        .iter()
        .flat_map(|row| {
            row.experts.iter().map(|&expert| rows_wire::Output {
                row: row.id,
                expert,
                values: row.input.clone(),
            })
        })
        .collect();
    outputs.reverse();
    let fault = mode.load(Ordering::SeqCst);
    match fault {
        1 => {
            outputs.pop();
        }
        2 => outputs[1] = outputs[0].clone(),
        _ => {}
    }
    let mut response = rows_wire::encode_response(
        request.handle,
        request.sequence,
        request.layer,
        HIDDEN,
        request.rows.len(),
        &outputs,
    )
    .unwrap();
    let mut content_type = rows_wire::CONTENT_TYPE;
    match fault {
        3 => response[FIRST_EXPERT_AT..FIRST_EXPERT_AT + 4].copy_from_slice(&99u32.to_le_bytes()),
        4 => response[FIRST_ROW_AT..FIRST_ROW_AT + 4].copy_from_slice(&77u32.to_le_bytes()),
        5 => response[FIRST_VALUE_AT..FIRST_VALUE_AT + 4]
            .copy_from_slice(&f32::NAN.to_bits().to_le_bytes()),
        6 => response[HANDLE_AT] ^= 1,
        7 => response[SEQUENCE_AT] ^= 1,
        8 => response[LAYER_AT] ^= 1,
        9 => response.push(0),
        10 => content_type = "application/json",
        11 => return axum::http::StatusCode::SERVICE_UNAVAILABLE.into_response(),
        12 => response[ROWS_AT..ROWS_AT + 4].copy_from_slice(&1u32.to_le_bytes()),
        13 => response[..4].copy_from_slice(b"VEY1"),
        _ => {}
    }
    ([(axum::http::header::CONTENT_TYPE, content_type)], response).into_response()
}

async fn serve(app: Router) -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (url, server)
}

#[tokio::test]
async fn rows_cross_once_keep_their_bits_and_refuse_every_wrong_answer() {
    let mode = Arc::new(AtomicUsize::new(0));
    let app = Router::new()
        .route(wire::PATH, get(|| async { Json(binding()) }))
        .route(wire::OPEN_PATH, post(open))
        .route(rows_wire::ROWS_PATH, post(rows))
        .with_state(mode.clone());
    let (url, server) = serve(app).await;
    tokio::task::spawn_blocking(move || {
        let client = HttpExpertShards::connect(&[url], None).unwrap();
        let asked = requested();

        let capture = crate::profile_testing::Capture::start();
        let outputs = client.forward_rows(0, LAYER, &asked).unwrap();
        let trace = capture.finish_provider_calls();
        // Arrival order is the stub's (reversed); identity and bits are exact.
        assert_eq!(outputs.len(), 3);
        for output in &outputs {
            let row = asked.iter().find(|row| row.id == output.row).unwrap();
            assert!(row.experts.contains(&output.expert));
            assert_eq!(
                output
                    .values
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                row.input.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        // One record for one network request, however many pairs it carried.
        assert_eq!(trace.len(), 1);
        assert_eq!(trace[0]["wire"], "expert-rows-f32-v1");
        assert_eq!(trace[0]["rows"], 2);
        assert_eq!(trace[0]["pairs"], 3);
        assert_eq!(trace[0]["complete"], true);

        for fault in FAULTS {
            mode.store(fault, Ordering::SeqCst);
            assert!(
                client.forward_rows(0, LAYER, &asked).is_err(),
                "fault {fault}"
            );
        }
        mode.store(0, Ordering::SeqCst);

        // Caller errors never reach the wire as a malformed frame.
        let mut narrow = asked.clone();
        narrow[0].input.pop();
        assert!(client.forward_rows(0, LAYER, &narrow).is_err(), "width");
        let mut repeated = asked.clone();
        repeated[1].id = repeated[0].id;
        assert!(client.forward_rows(0, LAYER, &repeated).is_err(), "row id");
        assert!(client.forward_rows(0, LAYER, &[]).is_err(), "no rows");
        assert!(
            client.forward_rows(1, LAYER, &asked).is_err(),
            "unknown worker"
        );

        client.sequences[0].store(u64::MAX, Ordering::SeqCst);
        assert!(client
            .forward_rows(0, LAYER, &asked)
            .err()
            .unwrap()
            .contains("sequence exhausted"));
    })
    .await
    .unwrap();
    server.abort();
}

#[tokio::test]
async fn a_worker_without_the_multi_row_route_is_refused_not_retried_row_by_row() {
    let single_row_calls = Arc::new(AtomicUsize::new(0));
    let counter = single_row_calls.clone();
    // A worker that predates the frame: it serves the single-row endpoint only.
    let app = Router::new()
        .route(wire::PATH, get(|| async { Json(binding()) }))
        .route(wire::OPEN_PATH, post(open))
        .route(
            wire::BINARY_PATH,
            post(move || {
                counter.fetch_add(1, Ordering::SeqCst);
                async { axum::http::StatusCode::OK }
            }),
        )
        .with_state(Arc::new(AtomicUsize::new(0)));
    let (url, server) = serve(app).await;
    tokio::task::spawn_blocking(move || {
        let client = HttpExpertShards::connect(&[url], None).unwrap();
        let error = client.forward_rows(0, LAYER, &requested()).err().unwrap();
        assert!(error.contains("refused the multi-row frame"), "{error}");
        assert!(error.contains("404"), "{error}");
        assert_eq!(
            single_row_calls.load(Ordering::SeqCst),
            0,
            "no silent fallback"
        );
    })
    .await
    .unwrap();
    server.abort();
}
