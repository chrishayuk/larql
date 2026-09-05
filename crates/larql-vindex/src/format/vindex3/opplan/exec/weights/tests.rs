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
        (LoadedWeight::F32(vec![0.0; 16].into()), 64, true, "f32"),
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
        // Plain owned bytes, so the census charges exactly the stream —
        // no page padding, and the codec's name is the representation.
        (
            LoadedWeight::KQuant {
                blocks: vec![0u8; 2 * kquant::Q6_K.bytes_per_block],
                codec: kquant::Q6_K,
            },
            2 * kquant::Q6_K.bytes_per_block,
            false,
            "Q6_K",
        ),
    ];
    for (loaded, bytes, widened, name) in cases {
        assert_eq!(loaded.resident_bytes(), bytes, "{name} miscounts its bytes");
        assert_eq!(loaded.is_widened_f32(), widened, "{name}");
        assert_eq!(loaded.slice().representation(), name);
    }
}

/// The residency census must report the ALLOCATION, never the geometry.
///
/// A census computed from the matrix's logical size would agree with
/// itself no matter how much memory was really in use — which is the one
/// thing a residency claim must not do. Enumerated across every variant
/// because the compact formats are the ones whose accounting is least
/// obvious: Q8 holds three buffers, bf16 one.
#[test]
fn resident_bytes_counts_every_buffer_a_variant_holds() {
    let f32w = LoadedWeight::F32(vec![0.0f32; 100].into());
    assert_eq!(f32w.resident_bytes(), 400, "f32 is four bytes an element");

    // Q8: codes + scales + the optional sums index.
    let q8 = LoadedWeight::Q8 {
        codes: vec![0i8; 256],
        scales: vec![0.0f32; 4],
        sums: vec![0i16; 16],
    };
    assert_eq!(q8.resident_bytes(), 256 + 4 * 4 + 16 * 2);

    // The index is derivable and may legitimately be absent; the census
    // must then not charge for it.
    let q8_bare = LoadedWeight::Q8 {
        codes: vec![0i8; 256],
        scales: vec![0.0f32; 4],
        sums: Vec::new(),
    };
    assert_eq!(q8_bare.resident_bytes(), 256 + 16);
    assert!(
        q8_bare.resident_bytes() < q8.resident_bytes(),
        "an absent index must cost less, not the same"
    );

    // Q4 packs two codes to the byte — half Q8's codes at equal count.
    let q4 = LoadedWeight::Q4 {
        packed: vec![0u8; 128],
        scales: vec![0.0f32; 4],
    };
    assert_eq!(q4.resident_bytes(), 128 + 16);

    // The AlignedBytes variants report the PADDED allocation, because
    // that is what the process holds.
    let bf16 = LoadedWeight::Bf16(AlignedBytes::zeroed(100));
    assert_eq!(bf16.resident_bytes(), DEVICE_PAGE_ALIGN);
    assert!(
        bf16.resident_bytes() > 100,
        "page padding is real memory and must be charged"
    );

    let mx = LoadedWeight::Mxfp4 {
        packed: AlignedBytes::zeroed(10),
        scales: AlignedBytes::zeroed(10),
    };
    assert_eq!(mx.resident_bytes(), 2 * DEVICE_PAGE_ALIGN);

    let nv = LoadedWeight::Nvfp4 {
        packed: AlignedBytes::zeroed(10),
        scales: AlignedBytes::zeroed(10),
        tensor_scale: 1.0,
    };
    assert_eq!(nv.resident_bytes(), 2 * DEVICE_PAGE_ALIGN);
}

/// Plural because the compact formats are not one buffer, and where
/// those allocations land is invisible to a kernel benchmark that
/// allocates one matrix and reuses it.
#[test]
fn allocations_enumerate_each_backing_buffer_separately() {
    let f32w = LoadedWeight::F32(vec![0.0f32; 100].into());
    assert_eq!(f32w.allocations().len(), 1);
    assert_eq!(f32w.allocations()[0].1, 400);

    // Three buffers with the index, two without — a model resident as Q8
    // holds roughly twice as many as the same model as bf16.
    let q8 = LoadedWeight::Q8 {
        codes: vec![0i8; 256],
        scales: vec![0.0f32; 4],
        sums: vec![0i16; 16],
    };
    assert_eq!(q8.allocations().len(), 3);
    let q8_bare = LoadedWeight::Q8 {
        codes: vec![0i8; 256],
        scales: vec![0.0f32; 4],
        sums: Vec::new(),
    };
    assert_eq!(
        q8_bare.allocations().len(),
        2,
        "an empty index must not be reported as a buffer"
    );

    assert_eq!(
        LoadedWeight::Q4 {
            packed: vec![0u8; 128],
            scales: vec![0.0f32; 4],
        }
        .allocations()
        .len(),
        2
    );

    let bf16 = LoadedWeight::Bf16(AlignedBytes::zeroed(100));
    assert_eq!(bf16.allocations().len(), 1);
    assert_eq!(bf16.allocations()[0].1, DEVICE_PAGE_ALIGN);
    assert_eq!(
        LoadedWeight::F16(AlignedBytes::zeroed(100))
            .allocations()
            .len(),
        1
    );

    assert_eq!(
        LoadedWeight::Mxfp4 {
            packed: AlignedBytes::zeroed(10),
            scales: AlignedBytes::zeroed(10),
        }
        .allocations()
        .len(),
        2
    );
    assert_eq!(
        LoadedWeight::Nvfp4 {
            packed: AlignedBytes::zeroed(10),
            scales: AlignedBytes::zeroed(10),
            tensor_scale: 1.0,
        }
        .allocations()
        .len(),
        2
    );

    // The sum of the parts is the census: two instruments that disagreed
    // would make a residency claim unfalsifiable.
    for w in [&q8, &q8_bare, &f32w, &bf16] {
        let summed: usize = w.allocations().iter().map(|(_, n)| n).sum();
        assert_eq!(
            summed,
            w.resident_bytes(),
            "the allocation list and the census must agree"
        );
    }
}

/// `F32` over a bf16 checkpoint means the loader DOUBLED the model, and
/// no total alone can say where that happened — so the flag is read off
/// the variant, never inferred from a size.
#[test]
fn only_the_widened_variant_admits_to_being_widened() {
    assert!(LoadedWeight::F32(vec![0.0; 4].into()).is_widened_f32());
    assert!(!LoadedWeight::Bf16(AlignedBytes::zeroed(8)).is_widened_f32());
    assert!(!LoadedWeight::Q8 {
        codes: vec![0i8; 8],
        scales: vec![0.0f32; 1],
        sums: Vec::new(),
    }
    .is_widened_f32());
    assert!(!LoadedWeight::Q4 {
        packed: vec![0u8; 4],
        scales: vec![0.0f32; 1],
    }
    .is_widened_f32());
}

/// `from_bytes` copies the checkpoint's bytes and changes only the
/// alignment — the numeric content must survive exactly, and the tail of
/// the page must stay zero so a device wrapping the whole allocation
/// reads no garbage.
#[test]
fn from_bytes_preserves_the_content_and_zeroes_the_padding() {
    let src: Vec<u8> = (0..300u32).map(|i| (i % 251) as u8).collect();
    let a = AlignedBytes::from_bytes(&src);

    assert_eq!(a.logical_len(), 300, "the tensor is what was handed in");
    assert_eq!(
        a.as_slice().len(),
        DEVICE_PAGE_ALIGN,
        "the allocation is a page multiple"
    );
    assert_eq!(
        &a.as_slice()[..300],
        &src[..],
        "the bytes are the checkpoint's"
    );
    assert!(
        a.as_slice()[300..].iter().all(|b| *b == 0),
        "the padding must be zero, or a device reads garbage past the tensor"
    );
    assert_eq!(
        a.as_slice().as_ptr() as usize % DEVICE_PAGE_ALIGN,
        0,
        "the allocation must be page-aligned for zero-copy wrapping"
    );
}

/// An empty tensor still gets one page: the layout must have non-zero
/// size, and a zero-length allocation would fail it.
#[test]
fn a_zero_length_tensor_still_holds_one_page() {
    let a = AlignedBytes::zeroed(0);
    assert_eq!(a.logical_len(), 0);
    assert_eq!(a.as_slice().len(), DEVICE_PAGE_ALIGN);
}

/// Every resident variant hands the kernel a slice in its OWN
/// representation. The block sizes travel with it: a Q4 slice that
/// reported Q8's block would have the kernel read scales at the wrong
/// stride and still return finite numbers.
#[test]
fn each_resident_variant_slices_as_its_own_representation() {
    let q8 = LoadedWeight::Q8 {
        codes: vec![1i8; 128],
        scales: vec![0.5f32; 128 / Q8_BLOCK],
        sums: Vec::new(),
    };
    match q8.slice() {
        WeightSlice::Q8 { codes, block, .. } => {
            assert_eq!(codes.len(), 128);
            assert_eq!(block, Q8_BLOCK, "the Q8 block must travel with the slice");
        }
        other => panic!("q8 must slice as q8, got {}", other.representation()),
    }

    let q4 = LoadedWeight::Q4 {
        packed: vec![0x21u8; 64],
        scales: vec![0.25f32; 128 / Q4_BLOCK],
    };
    match q4.slice() {
        WeightSlice::Q4 {
            packed,
            scales,
            block,
        } => {
            assert_eq!(packed.len(), 64, "Q4 stays packed two codes to the byte");
            assert_eq!(scales.len(), 128 / Q4_BLOCK);
            assert_eq!(block, Q4_BLOCK, "the Q4 block must travel with the slice");
        }
        other => panic!("q4 must slice as q4, got {}", other.representation()),
    }

    assert_eq!(
        LoadedWeight::F32(vec![0.0; 8].into())
            .slice()
            .representation(),
        "f32"
    );
}
