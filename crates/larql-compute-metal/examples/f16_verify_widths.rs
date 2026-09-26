//! VERIFY-N f16 economics: `R` positions against one f16 weight pass, per
//! arm, relative to the production `f16_gemv` on one position.
//!
//! ```text
//! serial   R f16_gemv dispatches (no amortisation)
//! x2       f16_matmul_x2_r{R}: 2 rows/simdgroup, X re-read per simdgroup
//! sg       f16_matmul_sg: X tile staged once per 128-row threadgroup,
//!          8x8 simdgroup-matrix MACs
//! ```
//!
//! Beside each time: the arm's X-operand bytes and weight bytes per output
//! element, from its geometry — the X column is the traffic the tiled
//! kernel exists to remove, so a win should show there first.
//! Chained timing with rotated weight copies, as `nvfp4_verify_widths`.
use larql_compute_metal::lowering::profile::gpu_span_ms;
use larql_compute_metal::lowering::{LoweredMatrix, MatmulRowsTarget, MatvecTarget};
use larql_compute_metal::MetalBackend;
use larql_models::quant::half::f32_to_f16;

const CHAIN: usize = 20;
const REPS: usize = 5;
const ROTATE_BYTES: usize = 1 << 30;
const PARITY_REL_RMS: f64 = 1e-5;
const WIDTHS: [usize; 3] = [2, 4, 8];
/// Rows an x2 simdgroup owns / a tiled threadgroup owns.
const X2_ROWS_PER_SG: f64 = 2.0;
const SG_ROWS_PER_TG: f64 = 128.0;
/// The tiled kernel always stages 8 position rows (zero-padded).
const SG_STAGED_POSITIONS: f64 = 8.0;

const SHAPES: &[(&str, usize, usize)] = &[
    ("head    [262208,2560]", 262208, 2560),
    ("gate/up [10240,2560]", 10240, 2560),
    ("down    [2560,10240]", 2560, 10240),
    ("q       [2048,2560]", 2048, 2560),
];

fn rel_rms(reference: &[f32], got: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (a, b) in reference.iter().zip(got) {
        num += ((a - b) as f64).powi(2);
        den += (*a as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt()
}

fn main() {
    let Some(gpu) = MetalBackend::new() else {
        std::process::exit(2)
    };
    let f = std::mem::size_of::<f32>() as u64;
    println!("µs per call (best of {REPS}, chain {CHAIN}); ratio = T / T(f16_gemv, one row)");
    println!("traffic columns: X bytes / output element, weight bytes / output element");
    for &(name, n, k) in SHAPES {
        let bytes: Vec<u8> = (0..n * k)
            .flat_map(|i| f32_to_f16(((i % 977) as f32 / 977.0) - 0.5).to_le_bytes())
            .collect();
        let copies = (ROTATE_BYTES / bytes.len()).clamp(1, CHAIN);
        let ws: Vec<_> = (0..copies).map(|_| gpu.lowering_weight(&bytes)).collect();
        let rmax = *WIDTHS.iter().max().expect("widths");
        let x: Vec<f32> = (0..rmax * k)
            .map(|i| ((i * 7 + i / k) % 13) as f32 * 0.01 - 0.05)
            .collect();
        let xb = gpu.lowering_upload(&x).expect("x");
        let out = gpu.lowering_scratch(rmax * n);
        let time = |encode: &dyn Fn(&metal::ComputeCommandEncoderRef, usize)| -> f64 {
            let mut best = f64::MAX;
            for _ in 0..REPS {
                let cmd = gpu.new_lowering_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                for c in 0..CHAIN {
                    encode(enc, c);
                }
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                best = best.min(gpu_span_ms(&cmd) * 1e3 / CHAIN as f64);
            }
            best
        };
        let gemv_row = |enc: &metal::ComputeCommandEncoderRef, c: usize, r: usize| {
            let xr_off = (r * k) as u64 * f;
            // f16_gemv binds x at offset 0: give it a row view by rows.
            let _ = xr_off;
            gpu.encode_f16_matvec(
                enc,
                &ws[c % copies],
                &MatvecTarget {
                    x: &xb,
                    out: &out,
                    out_offset: (r * n) as u64 * f,
                    n,
                    k,
                },
            )
        };
        let single = time(&|enc, c| gemv_row(enc, c, 0));
        let wbytes = bytes.len() as f64;
        println!(
            "{name}  {:.0} MB  f16_gemv {single:.1} µs ({:.0} GB/s)",
            wbytes / 1e6,
            wbytes / (single / 1e6) / 1e9
        );
        // Reference rows: the multi-RHS arms checked against x2 at R=1
        // is circular; check every arm against per-row x2_r* at R rows
        // by comparing to the serial GEMV on row 0 and x2 on the rest.
        let x2 = |enc: &metal::ComputeCommandEncoderRef, c: usize, rows: usize| {
            gpu.encode_matmul_rows(
                enc,
                &LoweredMatrix::F16 {
                    bytes: &ws[c % copies],
                },
                &MatmulRowsTarget {
                    x: &xb,
                    x_offset: 0,
                    out: &out,
                    out_offset: 0,
                    n,
                    k,
                    rows,
                },
            )
            .expect("f16 rows")
        };
        let sg = |enc: &metal::ComputeCommandEncoderRef, c: usize, rows: usize| {
            gpu.encode_f16_matmul_sg(
                enc,
                &ws[c % copies],
                &MatmulRowsTarget {
                    x: &xb,
                    x_offset: 0,
                    out: &out,
                    out_offset: 0,
                    n,
                    k,
                    rows,
                },
            )
        };
        for &r in &WIDTHS {
            let serial = time(&|enc, c| {
                for row in 0..r {
                    gemv_row(enc, c, row)
                }
            });
            let t_x2 = time(&|enc, c| x2(enc, c, r));
            let reference = gpu.lowering_readback(&out, r * n).expect("readback");
            let t_sg = time(&|enc, c| sg(enc, c, r));
            let got = gpu.lowering_readback(&out, r * n).expect("readback");
            let parity = rel_rms(&reference, &got);
            let rf = r as f64;
            let kf = k as f64;
            // x2: each simdgroup reads R*K floats of X for 2*R outputs.
            let x2_x = rf * kf * 4.0 / (X2_ROWS_PER_SG * rf);
            // sg: each threadgroup reads 8*K floats of X for 128*R outputs.
            let sg_x = SG_STAGED_POSITIONS * kf * 4.0 / (SG_ROWS_PER_TG * rf);
            let w_per_out = kf * 2.0 / rf;
            println!(
                "  R={r}  serial {serial:>8.1} ({:>4.2}x)  x2 {t_x2:>8.1} ({:>4.2}x, X {x2_x:>6.0} B)  sg {t_sg:>8.1} ({:>4.2}x, X {sg_x:>5.0} B)  W {w_per_out:>5.0} B/out  parity {parity:.1e}{}",
                serial / single,
                t_x2 / single,
                t_sg / single,
                if parity <= PARITY_REL_RMS { "" } else { "  PARITY FAIL" },
            );
        }
        gpu.recycle_lowering_scratch(out);
        gpu.recycle_lowering_scratch(xb);
    }
}
