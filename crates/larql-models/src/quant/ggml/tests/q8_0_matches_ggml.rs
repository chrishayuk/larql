//! `quantize_q8_0` against bytes produced by pinned ggml's `quantize_row_q8_0_ref`.

use super::super::legacy::dequantize_q8_0;
use super::super::quantize::quantize_q8_0;
use super::q8_0_ggml_golden::Q8_0_GGML_GOLDEN;

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
