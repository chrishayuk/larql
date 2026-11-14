//! Non-finite f16 scales are outside the kernel's contract, but the NEON and scalar paths must still agree bit for bit
//! (they use a hardware and a software f16 conversion), including NaN payload bits.
use larql_compute::cpu::kquant_gemv::q8_0_gemv;
use larql_models::quant::half::f16_to_f32;

#[test]
fn nan_and_infinity_scales_agree_with_the_scalar_program_bit_for_bit() {
    let k = 32;
    for h in [0x7C00u16, 0xFC00, 0x7C01, 0x7E00, 0xFE01, 0x7FFF] {
        for rows in [3usize, 16, 33] {
            let mut blocks = vec![];
            for r in 0..rows {
                blocks.extend_from_slice(&h.to_le_bytes());
                blocks.extend((0..32).map(|j| (r * 7 + j * 3) as u8));
            }
            let x: Vec<f32> = (0..k).map(|i| i as f32 / 5.0 - 1.0).collect();
            let got = q8_0_gemv(&blocks, &x, rows, k).unwrap();
            for (r, g) in got.iter().enumerate() {
                let d = f16_to_f32(h);
                let mut acc = 0.0f32;
                for j in 0..32 {
                    acc += ((blocks[r * 34 + 2 + j] as i8 as f32) * d) * x[j];
                }
                assert_eq!(
                    g.to_bits(),
                    acc.to_bits(),
                    "scale {h:#06x}, {rows} rows, row {r}"
                );
            }
        }
    }
}
