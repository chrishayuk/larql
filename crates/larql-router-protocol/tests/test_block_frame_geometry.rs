//! The block-frame sizes the link qualification uses are the encoder's.
//!
//! `bench/v3-routed-experts/block_frames.json` gives the request and
//! response bytes of a one-worker multi-row exchange per block width. The
//! link addendum measures the relay at exactly those sizes, so they must be
//! what `VXR1` and `VXS1` frames of that shape really occupy, not a formula
//! copied beside them.
use larql_router_protocol::vindex3_expert_rows as rows_wire;

const GEOMETRY: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../bench/v3-routed-experts/block_frames.json"
);
const HANDLE: [u8; 16] = [7; 16];
const SEQUENCE: u64 = 1;
const LAYER: usize = 0;

#[test]
fn recorded_block_frame_sizes_are_what_the_encoder_writes() {
    let geometry: serde_json::Value =
        serde_json::from_slice(&std::fs::read(GEOMETRY).unwrap()).unwrap();
    let hidden = geometry["hidden"].as_u64().unwrap() as usize;
    let per_row = geometry["experts_per_row"].as_u64().unwrap() as usize;
    let widths = geometry["widths"].as_object().unwrap();
    assert!(!widths.is_empty());
    for (width, sizes) in widths {
        let width: usize = width.parse().unwrap();
        let rows: Vec<rows_wire::Row> = (0..width)
            .map(|row| rows_wire::Row {
                id: row as u32,
                experts: (0..per_row).collect(),
                input: vec![0.5; hidden],
            })
            .collect();
        let outputs: Vec<rows_wire::Output> = rows
            .iter()
            .flat_map(|row| {
                row.experts.iter().map(|&expert| rows_wire::Output {
                    row: row.id,
                    expert,
                    values: vec![0.25; hidden],
                })
            })
            .collect();
        let pairs = width * per_row;
        let request = rows_wire::encode_request(HANDLE, SEQUENCE, LAYER, hidden, &rows).unwrap();
        let response =
            rows_wire::encode_response(HANDLE, SEQUENCE, LAYER, hidden, width, &outputs).unwrap();
        assert_eq!(
            request.len() as u64,
            sizes["request_bytes"].as_u64().unwrap(),
            "width {width} request"
        );
        assert_eq!(
            response.len() as u64,
            sizes["response_bytes"].as_u64().unwrap(),
            "width {width} response"
        );
        assert_eq!(
            request.len(),
            rows_wire::request_len(hidden, width, pairs).unwrap()
        );
        assert_eq!(
            response.len(),
            rows_wire::response_len(hidden, pairs).unwrap()
        );
    }
}
