//! A rank-one `add_low_rank_columns` over a whole real `down_proj`, through the compact `weight-patches/v4` path.
//!
//! The SG-EDIT-1 S1 shape: every column of one Gemma 3 4B `down_proj` (`[2560, 10240]`, 26,214,400 elements),
//! far beyond `MAX_PATCH_ELEMENTS`, which an element ledger could not hold. Ignored by default; run with a
//! BF16-only canonical container:
//!
//! ```text
//! LARQL_V3_CONTAINER=~/chris-models/gemma3-4b-it.vindex3 \
//!   cargo test -p larql-vindex --test weight_patch real_scale -- --ignored --nocapture
//! ```
//!
//! The expected value of every element is computed here in closed form, by a route that does not use the
//! interpreter: `a·b = 2^-6` exactly, so `widen(w) + 2^-6` is one f64 addition, rounded once to BF16.

use larql_vindex::format::vindex3::weight_patch::structured::OperandBits;
use larql_vindex::format::vindex3::weight_patch::{
    apply_declared_patches, check_ledger, declare_patches, read_bf16_tensor, tensor_shape,
    LowRankColumnsOp, WeightOpRef, MAX_PATCH_ELEMENTS, WEIGHT_PATCHES_SCHEMA_V4,
};

use crate::fixture::index_of;

const CONTAINER_VAR: &str = "LARQL_V3_CONTAINER";
const LAYER_TENSOR: &str = "12.mlp.down_proj.weight";
const PROFILE: &str = "f64-seq-rne1/v1";
/// `a[r] = 2^-3` and `b[p] = 2^-3`, so every term is exactly `2^-6`.
const F64_EIGHTH: u64 = 0x3fc0_0000_0000_0000;
const UPDATE: f64 = 1.0 / 64.0;
/// The compact ledger of this edit is its two operands (12,800 f64 hex strings) and a few digests.
const LEDGER_BYTES_CEILING: u64 = 1 << 20;
const BF16_MANTISSA_SHIFT: u32 = 16;
const F64_MANTISSA_BITS: u32 = 52;
const F64_EXPONENT_BIAS: i32 = 1023;

/// `x` rounded to the nearest BF16, ties to even, for finite `x` that rounds to a finite BF16. Through f32 is NOT
/// used: two roundings can differ from one. This rounds the f64 directly on its bits.
fn f64_to_bf16_rne(x: f64) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 63) as u16) << 15;
    let mag = x.abs();
    if mag == 0.0 {
        return sign;
    }
    // BF16 has 8 significant bits: the spacing at magnitude `mag` is 2^(e - 7) for normal values, and at least the
    // subnormal spacing 2^-133.
    // The binary exponent from the bits (the f64 inputs here are normal), never from a float log.
    let e = ((mag.to_bits() >> F64_MANTISSA_BITS) as i32) - F64_EXPONENT_BIAS;
    let spacing = 2f64.powi((e - 7).max(-133));
    let q = mag / spacing; // exact: a power-of-two scale
    let lower = q.floor();
    let rest = q - lower;
    let n = if rest > 0.5 || (rest == 0.5 && lower % 2.0 == 1.0) {
        lower + 1.0
    } else {
        lower
    };
    let rounded = n * spacing;
    sign | ((rounded as f32).to_bits() >> BF16_MANTISSA_SHIFT) as u16
}

#[test]
#[ignore = "needs a real BF16-only VINDEX3 container in LARQL_V3_CONTAINER"]
fn real_scale_a_rank_one_edit_of_a_whole_down_proj_compiles_through_the_compact_ledger() {
    let Ok(path) = std::env::var(CONTAINER_VAR) else {
        panic!("set {CONTAINER_VAR} to a container directory");
    };
    let container = std::path::PathBuf::from(path);
    let [rows, cols] = tensor_shape(&container, LAYER_TENSOR).unwrap();
    assert!(
        rows * cols > MAX_PATCH_ELEMENTS,
        "the case must exceed the element bound"
    );

    let started = std::time::Instant::now();
    let op = LowRankColumnsOp::bind(
        &container,
        LAYER_TENSOR,
        PROFILE,
        1,
        &OperandBits::F64(vec![F64_EIGHTH; rows]),
        &OperandBits::F64(vec![F64_EIGHTH; cols]),
        (0..cols).collect(),
    )
    .unwrap();
    let (declared, origins) =
        declare_patches(&container, &[WeightOpRef::LowRankColumns(&op)]).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("s1-scale.vindex3");
    let report = apply_declared_patches(&container, &declared, &origins, &out).unwrap();
    let compiled_in = started.elapsed();

    assert_eq!(report.ledger.schema, WEIGHT_PATCHES_SCHEMA_V4);
    assert!(check_ledger(&out, &index_of(&out)).is_empty());
    let ledger_bytes = std::fs::metadata(out.join("weight_patches.json"))
        .unwrap()
        .len();
    assert!(ledger_bytes < LEDGER_BYTES_CEILING, "{ledger_bytes} bytes");

    // Every element against the closed form, by a route that does not use the interpreter.
    let (_, base) = read_bf16_tensor(&container, LAYER_TENSOR).unwrap();
    let (_, got) = read_bf16_tensor(&out, LAYER_TENSOR).unwrap();
    let mut changed = 0usize;
    for (i, (&old, &new)) in base.iter().zip(&got).enumerate() {
        let w = f32::from_bits((old as u32) << BF16_MANTISSA_SHIFT) as f64;
        let want = f64_to_bf16_rne(w + UPDATE);
        assert_eq!(new, want, "element ({}, {})", i / cols, i % cols);
        changed += usize::from(old != new);
    }
    assert_eq!(report.ledger.targets[0].changed_elements, Some(changed));
    assert!(changed > MAX_PATCH_ELEMENTS);

    // Every other tensor of the edited segment is byte-identical.
    let src_index = index_of(&container);
    let segment = &src_index.representations["target.decoder_stack@BF16"].segment;
    let (header, _) = larql_vindex::format::vindex3::encode::segment::read_segment_header(
        &container.join(segment),
    )
    .unwrap();
    for t in header
        .tensors
        .iter()
        .filter(|t| t.name != LAYER_TENSOR && t.dtype == "BF16")
    {
        if t.shape.len() != 2 {
            continue;
        }
        assert_eq!(
            read_bf16_tensor(&container, &t.name).unwrap(),
            read_bf16_tensor(&out, &t.name).unwrap(),
            "{}",
            t.name
        );
    }
    println!(
        "real_scale: {rows}x{cols} = {} elements, {changed} changed, ledger {ledger_bytes} bytes, compiled in {:.1}s",
        rows * cols,
        compiled_in.as_secs_f64()
    );
}

#[test]
fn the_closed_form_rounder_agrees_with_the_bf16_ties() {
    // 1 + 2^-8 is a tie between 1.0 (even) and 1 + 2^-7; 1 + 3·2^-8 is a tie that goes up to the even 1 + 2^-6.
    assert_eq!(f64_to_bf16_rne(1.0 + 2f64.powi(-8)), 0x3f80);
    assert_eq!(f64_to_bf16_rne(1.0 + 3.0 * 2f64.powi(-8)), 0x3f82);
    assert_eq!(f64_to_bf16_rne(-(1.0 + 2f64.powi(-9))), 0xbf80);
    assert_eq!(f64_to_bf16_rne(0.0), 0x0000);
    assert_eq!(f64_to_bf16_rne(-0.0), 0x8000);
}
