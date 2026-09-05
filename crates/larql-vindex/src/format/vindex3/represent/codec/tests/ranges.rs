//! Range-aware decode, and the refusals around it — over every codec.

use super::*;

fn whole(fixture: &Fixture) -> Vec<f32> {
    fixture
        .codec
        .decode_all(
            &fixture.operands(),
            &fixture.shape,
            RepresentationExtent::TERMINAL,
            TENSOR,
        )
        .unwrap()
}

#[test]
fn a_row_range_decodes_to_that_slice_of_the_whole() {
    for fixture in fixtures() {
        let label = fixture.label();
        let all = whole(&fixture);
        for (start, end) in [(0, 1), (1, 3), (2, 3), (0, 3), (1, 1)] {
            let mut dst = vec![f32::NAN; (end - start) * K];
            fixture
                .codec
                .decode_rows(
                    &fixture.operands(),
                    &fixture.shape,
                    start..end,
                    RepresentationExtent::TERMINAL,
                    &mut dst,
                    TENSOR,
                )
                .unwrap_or_else(|e| panic!("{label} rows {start}..{end}: {e}"));
            assert_eq!(dst, all[start * K..end * K], "{label} rows {start}..{end}");
        }
    }
}

#[test]
fn a_row_range_past_the_end_is_refused_naming_the_rows_held() {
    for fixture in fixtures() {
        let mut dst = vec![0.0; K];
        let err = fixture
            .codec
            .decode_rows(
                &fixture.operands(),
                &fixture.shape,
                ROWS..ROWS + 1,
                RepresentationExtent::TERMINAL,
                &mut dst,
                TENSOR,
            )
            .unwrap_err();
        assert!(
            matches!(&err, CodecError::RowRange { start, end, rows, .. } if *start == ROWS && *end == ROWS + 1 && *rows == ROWS),
            "{}: {err}",
            fixture.label()
        );
    }
}

#[test]
fn a_destination_of_the_wrong_size_is_refused_before_any_byte_is_read() {
    for fixture in fixtures() {
        let mut dst = vec![0.0; K + 1];
        let err = fixture
            .codec
            .decode_rows(
                &fixture.operands(),
                &fixture.shape,
                0..1,
                RepresentationExtent::TERMINAL,
                &mut dst,
                TENSOR,
            )
            .unwrap_err();
        assert!(
            matches!(&err, CodecError::Destination { need, have, .. } if *need == K && *have == K + 1),
            "{}: {err}",
            fixture.label()
        );
    }
}

#[test]
fn a_stream_shorter_than_the_shape_implies_is_refused_by_stream_name() {
    for fixture in fixtures() {
        let label = fixture.label();
        let mut streams = NamedStreams::new();
        for (spec, bytes) in fixture.codec.streams().iter().zip(&fixture.buffers) {
            streams = streams.with(*spec, &bytes[..bytes.len() - 1]);
        }
        // A packed codec that derives its split refuses at the bind.
        if fixture.packed && fixture.codec.streams().len() > 1 {
            let short = &fixture.buffers[0][..fixture.buffers[0].len() - 1];
            let err = fixture
                .codec
                .bind_packed(short, &fixture.shape, TENSOR)
                .unwrap_err();
            assert!(
                matches!(err, CodecError::StreamLength { .. }),
                "{label}: {err}"
            );
            continue;
        }
        let operands = CodecOperands::from_streams(streams);
        let err = fixture
            .codec
            .validate(
                &operands,
                &fixture.shape,
                RepresentationExtent::TERMINAL,
                TENSOR,
            )
            .unwrap_err();
        assert!(
            matches!(&err, CodecError::StreamLength { stream, .. } if stream == "values"),
            "{label}: {err}"
        );
    }
}

#[test]
fn a_row_the_codec_cannot_block_is_refused_by_group() {
    // 100 is not a whole number of any group these codecs use; the
    // floats accept anything.
    for codec in builtin() {
        let label = codec.encoding_label();
        let result = codec.stored_bytes(&[2, 100], RepresentationExtent::TERMINAL, TENSOR);
        let group = codec.capabilities().row_align_elems;
        if group == 1 {
            assert_eq!(
                result.unwrap(),
                200 * codec.capabilities().physical_align_bytes as u64,
                "{label}"
            );
            continue;
        }
        let err = result.unwrap_err();
        // Worded by whichever derivation owns the geometry — the shared
        // row helper or NVFP4's pack layout — but always naming the group.
        assert!(
            matches!(&err, CodecError::Geometry { why, .. } if why.contains(&format!("{group}-element group"))),
            "{label}: {err}"
        );
    }
}

#[test]
fn a_missing_stream_is_named_together_with_what_was_bound() {
    let fixture = fixtures()
        .into_iter()
        .find(|f| f.label() == DTYPE_MXFP4)
        .unwrap();
    let only_values =
        CodecOperands::from_streams(NamedStreams::single(VALUES, &fixture.buffers[0]));
    let err = MXFP4
        .validate(
            &only_values,
            &fixture.shape,
            RepresentationExtent::TERMINAL,
            TENSOR,
        )
        .unwrap_err();
    assert_eq!(
        err,
        CodecError::MissingStream {
            tensor: TENSOR.into(),
            label: DTYPE_MXFP4.into(),
            stream: "group_scales".into(),
            bound: vec!["values".into()],
        }
    );
}
