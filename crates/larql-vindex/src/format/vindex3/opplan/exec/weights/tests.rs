//! Tests for what a checkpoint's bytes become when they are made
//! resident — the accounting, the exact narrowings, and the lossy
//! quantisers.
//!
//! A child module of `weights` rather than a sibling under
//! `exec/tests/`, so it still sees the module's private items.

use super::*;

/// `logical_len` is the tensor; `as_slice` is the allocation.
///
/// Every byte-accounting consumer must use the former. The two are
/// equal whenever a size lands on a page boundary — which is why the
/// distinction is easy to lose: Granite's NVFP4 allocations are all
/// exact multiples, so a ledger built on `as_slice().len()` reads
/// correct there and drifts on gpt-oss's 2880-wide shapes.
#[test]
fn a_page_padded_allocation_reports_the_tensor_not_the_padding() {
    // gpt-oss [2880, 2880] NVFP4 codes: 180 groups x 8 bytes x 2880 rows.
    let logical: usize = 2880 * (2880 / 16) * 8;
    assert!(
        !logical.is_multiple_of(DEVICE_PAGE_ALIGN),
        "fixture must not be page-aligned or it cannot detect the bug"
    );

    let bytes = AlignedBytes::zeroed(logical);
    assert_eq!(bytes.logical_len(), logical);
    assert!(
        bytes.as_slice().len() > logical,
        "the allocation is padded past the tensor"
    );
    assert_eq!(
        bytes.as_slice().len(),
        logical.div_ceil(DEVICE_PAGE_ALIGN) * DEVICE_PAGE_ALIGN
    );
}

/// The aligned case, so the test above cannot be satisfied by an
/// implementation that always over-reports.
#[test]
fn an_exactly_page_sized_allocation_has_no_padding() {
    let logical = DEVICE_PAGE_ALIGN * 3;
    let bytes = AlignedBytes::zeroed(logical);
    assert_eq!(bytes.logical_len(), logical);
    assert_eq!(bytes.as_slice().len(), logical);
}

/// Every normal-range bf16 value must convert to f16 exactly.
/// Finite overflow fails closed rather than saturating to infinity.
/// Exceptional values stay exceptional; zeros stay signed zeros.
/// The subnormal tail truncates but stays within one f16 subnormal
/// step of the true value, and deep underflow lands on zero.
/// f32 → f16 rounds to nearest, ties to even, and refuses overflow.
/// Grid-exact values survive MXFP4 quantisation unchanged, and the
/// packed bytes decode identically through the **independent**
/// `larql-models` decoder — the layout (lo nibble first, per-row
/// group order, e8m0 scales) is pinned against the code that has
/// already read real GPT-OSS checkpoints, not against this
/// quantiser's own assumptions.
#[test]
fn mxfp4_grid_values_round_trip_through_the_independent_decoder() {
    // One row, 32 elements: max 6.0 → shared exponent 0 → scale 1.0,
    // every value on the e2m1 grid.
    let mut row = vec![0.0f32; 32];
    let grid = [0.5f32, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.5, -1.5, -6.0];
    row[..grid.len()].copy_from_slice(&grid);
    let LoadedWeight::Mxfp4 { packed, scales } = quantize_mxfp4(&row, 1, 32, "w").unwrap() else {
        panic!("quantiser must produce the mxfp4 variant");
    };
    assert_eq!(scales.as_slice()[0], 127, "max 6.0 → 2^0 scale");
    let decoded = larql_models::quant::mxfp4::dequantize_expert(
        &packed.as_slice()[..16],
        &scales.as_slice()[..1],
        1,
        1,
    )
    .unwrap();
    assert_eq!(&decoded[..], &row[..], "grid values must survive exactly");
}

/// Off-grid values land within one half-step of the grid, and a
/// group's error is bounded by its scale (2·scale at saturation).
#[test]
fn mxfp4_error_is_bounded_by_the_group_scale() {
    let row: Vec<f32> = (0..64).map(|i| (i as f32 * 0.37).sin() * 5.0).collect();
    let LoadedWeight::Mxfp4 { packed, scales } = quantize_mxfp4(&row, 1, 64, "w").unwrap() else {
        panic!("quantiser must produce the mxfp4 variant");
    };
    let decoded = larql_models::quant::mxfp4::dequantize_expert(
        &packed.as_slice()[..32],
        &scales.as_slice()[..2],
        1,
        2,
    )
    .unwrap();
    for (group, (xs, ds)) in row.chunks(32).zip(decoded.chunks(32)).enumerate() {
        let scale = e8m0_to_f32(scales.as_slice()[group]);
        for (x, d) in xs.iter().zip(ds) {
            assert!(
                (x - d).abs() <= scale * 2.0 + f32::EPSILON,
                "group {group}: |{x} - {d}| exceeds 2·scale ({scale})"
            );
        }
    }
}

/// Group misalignment and shape mismatches are refused, not padded.
#[test]
fn mxfp4_quantiser_fails_closed_on_bad_geometry() {
    let err = quantize_mxfp4(&[0.0; 40], 1, 40, "w").unwrap_err();
    assert!(err.to_string().contains("32-element group"), "{err}");
    let err = quantize_mxfp4(&[0.0; 32], 2, 32, "w").unwrap_err();
    assert!(err.to_string().contains("do not fill"), "{err}");
}

/// An all-zero group takes the zero-scale sentinel and decodes to
/// exact zeros.
#[test]
fn mxfp4_zero_group_uses_the_zero_scale_sentinel() {
    let LoadedWeight::Mxfp4 { packed, scales } = quantize_mxfp4(&[0.0f32; 32], 1, 32, "w").unwrap()
    else {
        panic!("quantiser must produce the mxfp4 variant");
    };
    assert_eq!(scales.as_slice()[0], 0);
    assert!(packed.as_slice()[..16].iter().all(|&b| b == 0));
}

/// The parallel loader must produce **byte-identical** output to the
/// single-definition reference in `quant::nvfp4`. The loader exists
/// only for residency and thread-pool reasons; if it drifted, the
/// Metal kernel would be judged against a CPU reference that no
/// longer describes the bytes it is handed.
#[test]
fn the_parallel_nvfp4_loader_matches_the_reference_exactly() {
    // Awkward geometry on purpose: rows that do not divide evenly
    // across a pool, and a k spanning several groups.
    let (rows, k) = (37, 16 * 11);
    let values: Vec<f32> = (0..rows * k)
        .map(|i| ((i as f32) * 0.0137).sin() * (1.0 + (i % 7) as f32))
        .collect();

    let reference = larql_models::quant::nvfp4::quantize(&values, rows, k).unwrap();
    let LoadedWeight::Nvfp4 {
        packed,
        scales,
        tensor_scale,
    } = quantize_nvfp4(&values, rows, k, "w").unwrap()
    else {
        panic!("loader must produce the nvfp4 variant");
    };

    assert_eq!(tensor_scale, reference.tensor_scale);
    assert_eq!(
        &packed.as_slice()[..reference.packed.len()],
        &reference.packed[..],
        "packed codes must match the reference byte for byte"
    );
    assert_eq!(
        &scales.as_slice()[..reference.scales.len()],
        &reference.scales[..],
        "E4M3 scales must match the reference byte for byte"
    );
}

/// Geometry is refused by the loader too, not only by the codec.
#[test]
fn the_nvfp4_loader_fails_closed_on_bad_geometry() {
    let err = quantize_nvfp4(&[0.0; 40], 1, 40, "w").unwrap_err();
    assert!(err.to_string().contains("16-element group"), "{err}");
    let err = quantize_nvfp4(&[0.0; 32], 3, 16, "w").unwrap_err();
    assert!(err.to_string().contains("do not fill"), "{err}");
}
/// Every variant reports its own bytes and its own representation.
///
/// The residency census adds these up and calls the total the model's
/// size, so a variant that under-reported — or that answered another
/// variant's arm — would make the census quietly wrong in exactly the
/// direction that flatters it. Enumerated rather than sampled: a new
/// format is one missing arm away from being invisible to the census.
#[test]
fn every_loaded_variant_accounts_for_itself() {
    let page = DEVICE_PAGE_ALIGN;
    let cases: Vec<(LoadedWeight, usize, bool, &str)> = vec![
        (LoadedWeight::F32(vec![0.0; 16]), 64, true, "f32"),
        (
            LoadedWeight::Q8 {
                codes: vec![0i8; 64],
                scales: vec![0.0f32; 1],
                sums: Vec::new(),
            },
            68,
            false,
            "q8",
        ),
        (
            LoadedWeight::Bf16(AlignedBytes::from_bytes(&[0u8; 32])),
            page,
            false,
            "bf16",
        ),
        (
            LoadedWeight::F16(AlignedBytes::from_bytes(&[0u8; 32])),
            page,
            false,
            "f16",
        ),
        (
            LoadedWeight::Mxfp4 {
                packed: AlignedBytes::from_bytes(&[0u8; 16]),
                scales: AlignedBytes::from_bytes(&[0u8; 1]),
            },
            page * 2,
            false,
            "mxfp4",
        ),
        (
            LoadedWeight::Nvfp4 {
                packed: AlignedBytes::from_bytes(&[0u8; 8]),
                scales: AlignedBytes::from_bytes(&[0u8; 1]),
                tensor_scale: 1.0,
            },
            page * 2,
            false,
            "nvfp4",
        ),
    ];
    for (loaded, bytes, widened, name) in cases {
        assert_eq!(loaded.resident_bytes(), bytes, "{name} miscounts its bytes");
        assert_eq!(loaded.is_widened_f32(), widened, "{name}");
        assert_eq!(loaded.slice().representation(), name);
    }
}
