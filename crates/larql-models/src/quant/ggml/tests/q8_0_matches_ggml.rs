//! `quantize_q8_0` against bytes produced by pinned ggml's `quantize_row_q8_0_ref`.

use super::super::legacy::dequantize_q8_0;
use super::super::quantize::quantize_q8_0;
use super::q8_0_ggml_golden::{Q8_0_GGML_GOLDEN, Q8_0_RANDOM_BLOCKS, Q8_0_RANDOM_FNV1A64};

fn input(bits: &[u32; 32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Every case is reported before anything fails, so a regression shows its width.
#[test]
fn every_block_matches_ggml_byte_for_byte() {
    let differing: Vec<_> = Q8_0_GGML_GOLDEN
        .iter()
        .filter_map(|c| {
            let got = hex(&quantize_q8_0(&input(&c.input_bits)));
            (got != c.block_hex).then(|| format!("{}: got {got}, ggml {}", c.name, c.block_hex))
        })
        .collect();
    assert!(differing.is_empty(), "{}", differing.join("\n"));
}

#[test]
fn all_blocks_together_encode_as_the_concatenation() {
    let all: Vec<f32> = Q8_0_GGML_GOLDEN
        .iter()
        .flat_map(|c| input(&c.input_bits))
        .collect();
    let want: String = Q8_0_GGML_GOLDEN.iter().map(|c| c.block_hex).collect();
    assert_eq!(hex(&quantize_q8_0(&all)), want);
}

/// The defect this guards against: a block whose scale is an f16 subnormal
/// stored scale 0, so a non-zero block decoded to all zeros.
#[test]
fn a_subnormal_scale_does_not_decode_a_non_zero_block_to_zeros() {
    let block: Vec<f32> = (0..32)
        .map(|i| if i == 0 { 0.0077 } else { 0.001 })
        .collect();
    let decoded = dequantize_q8_0(&quantize_q8_0(&block), 32).unwrap();
    assert!(
        (decoded[0] - 0.0077).abs() < 0.0077 * 0.01,
        "{}",
        decoded[0]
    );
    assert!(
        (decoded[1] - 0.001).abs() < 0.0077 / 127.0,
        "{}",
        decoded[1]
    );
}

/// The same integer arithmetic as the golden file's generator: finite values
/// from `2^-38` up to just under `2^5`.
fn seeded_blocks(n: usize) -> Vec<f32> {
    let mut state: u32 = 0x02F6_E2B1;
    let mut step = move || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        state
    };
    let mut out = Vec::with_capacity(n * 32);
    for _ in 0..n {
        let top = 96 + (step() >> 16) % 36;
        for _ in 0..32 {
            let s = step();
            let exp = top - (s & 0xFF) % 8;
            out.push(f32::from_bits(
                (s >> 31 << 31) | (exp << 23) | ((s >> 8) & 0x7F_FFFF),
            ));
        }
    }
    out
}

/// The chosen blocks above can't see an encoder that differs from ggml in the
/// last bit of f32 arithmetic: `amax * (1.0 / 127.0)` for `amax / 127.0`, or
/// `127.0 / amax` for `1.0 / scale`, change a few blocks in a million.
#[test]
fn a_million_seeded_blocks_match_ggml() {
    let encoded = quantize_q8_0(&seeded_blocks(Q8_0_RANDOM_BLOCKS));
    let digest = encoded.iter().fold(0xCBF2_9CE4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3)
    });
    assert_eq!(digest, Q8_0_RANDOM_FNV1A64);
}
