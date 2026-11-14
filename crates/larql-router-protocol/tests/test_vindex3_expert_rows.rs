//! The multi-row expert frame pair: exact carriers, strict bounds, and no
//! way to read a single-row frame as a multi-row one or the reverse.
use larql_router_protocol::vindex3_expert_rows::{
    check_correspondence, decode_request, decode_response, encode_request, encode_response,
    request_len, response_len, Output, Row, HEADER_BYTES, MAX_ROWS,
};
use larql_router_protocol::vindex3_experts as single;

const HIDDEN: usize = 4;
const MAX_SELECTED: usize = 2;
const HANDLE: [u8; 16] = [7; 16];
const SEQUENCE: u64 = 91;
const LAYER: usize = 3;
/// Carrier values whose bit patterns a lossy path would not preserve.
const EXACT: [f32; HIDDEN] = [0.0, -0.0, f32::from_bits(1), f32::MAX];
/// Header offsets a corrupted field is flipped at: magic, hidden, rows, pairs.
const HEADER_FIELDS: [usize; 4] = [0, 32, 36, 40];

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

fn rows() -> Vec<Row> {
    vec![
        Row {
            id: 5,
            experts: vec![9, 2],
            input: EXACT.to_vec(),
        },
        Row {
            id: 0,
            experts: vec![2],
            input: vec![1.0, 2.0, 3.0, 4.0],
        },
    ]
}

fn outputs() -> Vec<Output> {
    [(0, 2), (5, 2), (5, 9)]
        .into_iter()
        .map(|(row, expert)| Output {
            row,
            expert,
            values: EXACT.to_vec(),
        })
        .collect()
}

fn request() -> Vec<u8> {
    encode_request(HANDLE, SEQUENCE, LAYER, HIDDEN, &rows()).unwrap()
}

fn response() -> Vec<u8> {
    encode_response(HANDLE, SEQUENCE, LAYER, HIDDEN, 2, &outputs()).unwrap()
}

#[test]
fn frames_round_trip_every_field_and_every_carrier_bit() {
    let bytes = request();
    assert_eq!(bytes.len(), request_len(HIDDEN, 2, 3).unwrap());
    let decoded = decode_request(&bytes, HIDDEN, MAX_ROWS, MAX_SELECTED).unwrap();
    assert_eq!(
        (decoded.handle, decoded.sequence, decoded.layer),
        (HANDLE, SEQUENCE, LAYER)
    );
    assert_eq!(decoded.rows.len(), 2);
    for (got, want) in decoded.rows.iter().zip(rows()) {
        assert_eq!((got.id, &got.experts), (want.id, &want.experts));
        assert_eq!(bits(&got.input), bits(&want.input));
    }

    let bytes = response();
    assert_eq!(bytes.len(), response_len(HIDDEN, 3).unwrap());
    let decoded = decode_response(&bytes, HIDDEN, MAX_ROWS, MAX_SELECTED).unwrap();
    assert_eq!(
        (
            decoded.handle,
            decoded.sequence,
            decoded.layer,
            decoded.rows
        ),
        (HANDLE, SEQUENCE, LAYER, 2)
    );
    for (got, want) in decoded.outputs.iter().zip(outputs()) {
        assert_eq!((got.row, got.expert), (want.row, want.expert));
        assert_eq!(bits(&got.values), bits(&want.values));
    }
}

#[test]
fn every_truncation_extension_and_header_corruption_refuses() {
    type Decode = fn(&[u8]) -> bool;
    let request_ok: Decode = |b| decode_request(b, HIDDEN, MAX_ROWS, MAX_SELECTED).is_ok();
    let response_ok: Decode = |b| decode_response(b, HIDDEN, MAX_ROWS, MAX_SELECTED).is_ok();
    for (frame, ok) in [(request(), request_ok), (response(), response_ok)] {
        assert!(ok(&frame));
        for length in 0..frame.len() {
            assert!(!ok(&frame[..length]), "accepted {length} bytes");
        }
        let mut longer = frame.clone();
        longer.push(0);
        assert!(!ok(&longer));
        for offset in HEADER_FIELDS {
            let mut bad = frame.clone();
            bad[offset] ^= 0x80;
            assert!(!ok(&bad), "accepted corruption at {offset}");
        }
        let mut nonfinite = frame.clone();
        let end = nonfinite.len();
        nonfinite[end - 4..].copy_from_slice(&f32::NAN.to_bits().to_le_bytes());
        assert!(!ok(&nonfinite));
    }
}

#[test]
fn request_shape_rules_hold_on_both_sides_of_the_wire() {
    let row = |id, experts: &[usize]| Row {
        id,
        experts: experts.to_vec(),
        input: EXACT.to_vec(),
    };
    let encode = |rows: &[Row]| encode_request(HANDLE, SEQUENCE, LAYER, HIDDEN, rows);
    assert!(encode(&[]).is_err(), "no rows");
    assert!(
        encode(&[row(1, &[2]), row(1, &[3])]).is_err(),
        "duplicate row id"
    );
    assert!(
        encode(&[row(1, &[2, 2])]).is_err(),
        "duplicate expert in a row"
    );
    assert!(encode(&[row(1, &[])]).is_err(), "row selecting nothing");
    let short = Row {
        id: 1,
        experts: vec![2],
        input: vec![0.0; HIDDEN - 1],
    };
    assert!(encode(&[short]).is_err(), "input width");
    let many: Vec<Row> = (0..=MAX_ROWS as u32).map(|id| row(id, &[2])).collect();
    assert!(encode(&many).is_err(), "more rows than a frame may carry");

    // A decoder enforces the worker's own bounds, not the sender's claims.
    let two_rows = request();
    assert!(
        decode_request(&two_rows, HIDDEN, 1, MAX_SELECTED).is_err(),
        "row bound"
    );
    assert!(
        decode_request(&two_rows, HIDDEN, MAX_ROWS, 1).is_err(),
        "selection bound"
    );
    assert!(decode_request(&two_rows, HIDDEN + 1, MAX_ROWS, MAX_SELECTED).is_err());

    // Hand-built frames a well-behaved encoder would never produce.
    let duplicate_id = {
        let mut bytes = request();
        let second_row = HEADER_BYTES + 4 + 4 + 2 * 4 + HIDDEN * 4;
        bytes[second_row..second_row + 4].copy_from_slice(&5u32.to_le_bytes());
        bytes
    };
    assert!(decode_request(&duplicate_id, HIDDEN, MAX_ROWS, MAX_SELECTED).is_err());
    let duplicate_expert = {
        let mut bytes = request();
        let second_expert = HEADER_BYTES + 4 + 4 + 4;
        bytes[second_expert..second_expert + 4].copy_from_slice(&9u32.to_le_bytes());
        bytes
    };
    assert!(decode_request(&duplicate_expert, HIDDEN, MAX_ROWS, MAX_SELECTED).is_err());
    // A row count that would move every later row boundary.
    let shifted = {
        let mut bytes = request();
        let first_count = HEADER_BYTES + 4;
        bytes[first_count..first_count + 4].copy_from_slice(&1u32.to_le_bytes());
        bytes
    };
    assert!(decode_request(&shifted, HIDDEN, MAX_ROWS, MAX_SELECTED).is_err());

    assert!(encode_response(HANDLE, SEQUENCE, LAYER, HIDDEN, 2, &[]).is_err());
    assert!(encode_response(HANDLE, SEQUENCE, LAYER, HIDDEN, 4, &outputs()).is_err());
    assert!(request_len(usize::MAX, 2, 2).is_err());
    assert!(response_len(usize::MAX, 2).is_err());
}

#[test]
fn single_row_and_multi_row_frames_are_never_read_as_each_other() {
    let single_request = single::encode_request(HANDLE, SEQUENCE, LAYER, &[9, 2], &EXACT).unwrap();
    let single_response = single::encode_response(
        HANDLE,
        SEQUENCE,
        LAYER,
        HIDDEN,
        &[(2, EXACT.to_vec()), (9, EXACT.to_vec())],
    )
    .unwrap();
    for frame in [&single_request, &single_response] {
        assert!(decode_request(frame, HIDDEN, MAX_ROWS, MAX_SELECTED).is_err());
        assert!(decode_response(frame, HIDDEN, MAX_ROWS, MAX_SELECTED).is_err());
    }
    for frame in [request(), response()] {
        assert!(single::decode_request(&frame, HIDDEN, MAX_SELECTED).is_err());
        assert!(single::decode_response(&frame, HIDDEN, MAX_SELECTED).is_err());
    }
    // A request is not a response and a response is not a request.
    assert!(decode_response(&request(), HIDDEN, MAX_ROWS, MAX_SELECTED).is_err());
    assert!(decode_request(&response(), HIDDEN, MAX_ROWS, MAX_SELECTED).is_err());
}

#[test]
fn a_response_is_accepted_only_if_it_answers_exactly_what_was_asked() {
    let asked = rows();
    let complete = outputs();
    assert!(check_correspondence(&asked, &complete, HIDDEN).is_ok());
    let mut shuffled = complete.clone();
    shuffled.reverse();
    assert!(
        check_correspondence(&asked, &shuffled, HIDDEN).is_ok(),
        "order is free"
    );

    let mut missing = complete.clone();
    missing.pop();
    let mut duplicated = complete.clone();
    duplicated[1] = duplicated[0].clone();
    let mut substituted = complete.clone();
    substituted[0].expert = 9; // row 0 never asked for expert 9
    let mut wrong_row = complete.clone();
    wrong_row[0].row = 77;
    let mut extra = complete.clone();
    extra.push(Output {
        row: 5,
        expert: 3,
        values: EXACT.to_vec(),
    });
    let mut narrow = complete.clone();
    narrow[2].values.pop();
    let mut nonfinite = complete.clone();
    nonfinite[2].values[0] = f32::INFINITY;
    for (name, bad) in [
        ("missing", missing),
        ("duplicated", duplicated),
        ("substituted", substituted),
        ("wrong row", wrong_row),
        ("extra", extra),
        ("narrow", narrow),
        ("nonfinite", nonfinite),
    ] {
        assert!(
            check_correspondence(&asked, &bad, HIDDEN).is_err(),
            "{name}"
        );
    }
}
