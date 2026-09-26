//! VERIFY-N kernel economics: what does evaluating `R` positions against
//! one NVFP4 weight pass cost, relative to one position?
//!
//! Per Gemma 3 4B projection shape, two arms at each width `R`:
//!
//! ```text
//! serial   R dependent-free x2 GEMVs (what the lowered step pays today)
//! matmul   one nvfp4_matmul_x2_r{R} dispatch (weight read once)
//! ```
//!
//! reported as `T(R) / T(1)` where `T(1)` is the production x2 GEMV. A
//! ratio near 1 means the extra positions ride the weight stream for
//! free; near `R` means nothing was amortised. Timing is the chained
//! form of `nvfp4_gemv_shapes`: `CHAIN` dispatches in one command buffer
//! rotating over `ROTATE_BYTES` of weight copies so each reads DRAM.
//! Every matmul column is checked against x2 on the same row.
//!
//! Run on AC power.
use larql_compute_metal::lowering::profile::gpu_span_ms;
use larql_compute_metal::lowering::{MatmulRowsTarget, MatvecOperands, Nvfp4Kernel};
use larql_compute_metal::shaders::nvfp4_matvec::{MATMUL_ARMS, SGF_SWEEP};
use larql_compute_metal::MetalBackend;
use larql_models::quant::nvfp4;

const CHAIN: usize = 40;
const REPS: usize = 5;
const ROTATE_BYTES: usize = 256 << 20;
/// Parity bound per column against x2: same element order and fold, so
/// only fast-math contraction differs.
const PARITY_REL_RMS: f64 = 1e-5;

/// (name, n rows, k cols) — Gemma 3 4B (hidden 2560, 8 q / 4 kv heads of
/// 256, intermediate 10240, vocab 262144).
const SHAPES: &[(&str, usize, usize)] = &[
    ("q      [2048,2560]", 2048, 2560),
    ("k/v    [1024,2560]", 1024, 2560),
    ("o      [2560,2048]", 2560, 2048),
    ("gate/up[10240,2560]", 10240, 2560),
    ("down   [2560,10240]", 2560, 10240),
    ("head   [262144,2560]", 262144, 2560),
];

/// Per-layer projection multiplicity (q, k, v, o, gate, up, down), for
/// the whole-layer weighted ratio; the head is reported on its own.
const LAYER_COUNTS: &[usize] = &[1, 2, 1, 2, 1];

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
    let widths: Vec<usize> = MATMUL_ARMS.iter().map(|a| a.1).collect();
    let rmax = *widths.iter().max().expect("widths");
    println!("µs per call (best of {REPS}, chain {CHAIN}); ratio = T / T(x2, one row)");
    // layer_us[w][arm]: summed µs across one layer's projections.
    let mut layer_single = 0.0f64;
    let mut layer_serial = vec![0.0f64; widths.len()];
    let mut layer_matmul = vec![0.0f64; widths.len()];
    let mut layer_sgk = vec![0.0f64; widths.len()];
    let mut layer_sgf = vec![vec![0.0f64; widths.len()]; SGF_SWEEP.len()];
    for (si, &(name, n, k)) in SHAPES.iter().enumerate() {
        let values: Vec<f32> = (0..n * k)
            .map(|i| ((i % 977) as f32 / 977.0) - 0.5)
            .collect();
        let x: Vec<f32> = (0..rmax * k)
            .map(|i| ((i * 7 + i / k) % 13) as f32 * 0.01 - 0.05)
            .collect();
        let xb = gpu.lowering_upload(&x).expect("x");
        let out = gpu.lowering_scratch(rmax * n);
        let nv = nvfp4::quantize(&values, n, k).expect("nvfp4");
        let bytes = nv.packed.len() + nv.scales.len();
        let copies = (ROTATE_BYTES / bytes).clamp(1, CHAIN);
        let packed: Vec<_> = (0..copies)
            .map(|_| gpu.lowering_weight(&nv.packed))
            .collect();
        let scales: Vec<_> = (0..copies)
            .map(|_| gpu.lowering_weight(&nv.scales))
            .collect();
        let out_row_bytes = (n * std::mem::size_of::<f32>()) as u64;

        // One timed chain; `encode` gets the chain index.
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

        // x2 on row r of X into row r of out — via a row-offset view.
        let x_rows: Vec<metal::Buffer> = (0..rmax)
            .map(|r| gpu.lowering_upload(&x[r * k..(r + 1) * k]).expect("x row"))
            .collect();
        let x2_row = |enc: &metal::ComputeCommandEncoderRef, c: usize, r: usize| {
            gpu.encode_nvfp4_kernel(
                Nvfp4Kernel::X2,
                enc,
                &MatvecOperands {
                    packed: &packed[c % copies],
                    scales: &scales[c % copies],
                    x: &x_rows[r],
                    out: &out,
                    out_offset: r as u64 * out_row_bytes,
                    n,
                    k,
                },
                nv.tensor_scale,
            )
        };

        let single = time(&|enc, c| x2_row(enc, c, 0));
        // Reference columns: x2 per row.
        {
            let cmd = gpu.new_lowering_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            for r in 0..rmax {
                x2_row(enc, 0, r);
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let reference = gpu.lowering_readback(&out, rmax * n).expect("readback");

        println!(
            "{name}  {:.1} MB  x2 {single:.1} µs ({:.0} GB/s)",
            bytes as f64 / 1e6,
            bytes as f64 / (single / 1e6) / 1e9
        );
        let weight = if si < LAYER_COUNTS.len() {
            LAYER_COUNTS[si] as f64
        } else {
            0.0
        };
        layer_single += weight * single;
        for (wi, &r) in widths.iter().enumerate() {
            let serial = time(&|enc, c| {
                for row in 0..r {
                    x2_row(enc, c, row)
                }
            });
            let matmul = time(&|enc, c| {
                gpu.encode_nvfp4_matmul(
                    enc,
                    &MatvecOperands {
                        packed: &packed[c % copies],
                        scales: &scales[c % copies],
                        x: &xb,
                        out: &out,
                        out_offset: 0,
                        n,
                        k,
                    },
                    nv.tensor_scale,
                    wi,
                )
            });
            let got = gpu.lowering_readback(&out, r * n).expect("readback");
            let worst = (0..r)
                .map(|row| {
                    rel_rms(
                        &reference[row * n..(row + 1) * n],
                        &got[row * n..(row + 1) * n],
                    )
                })
                .fold(0.0f64, f64::max);
            let flag = if worst <= PARITY_REL_RMS {
                ""
            } else {
                "  PARITY FAIL"
            };
            println!(
                "  {:<20} R={r}  serial {serial:>8.1} µs ({:>4.2}x)   matmul {matmul:>8.1} µs ({:>4.2}x)   per-row {:>4.2}x   parity {worst:.1e}{flag}",
                MATMUL_ARMS[wi].0,
                serial / single,
                matmul / single,
                matmul / single / r as f64,
            );
            layer_serial[wi] += weight * serial;
            layer_matmul[wi] += weight * matmul;
        }
        // Split-K simdgroup-matrix tiles, with each arm's X-operand bytes
        // per output element (x2: a simdgroup re-reads R*K floats for 2R
        // outputs; sgk: a 32-row threadgroup reads 8*K floats for 32R).
        let w_out = |r: usize| bytes as f64 / (n * r) as f64;
        for &r in &[2usize, 4, 8] {
            let t = time(&|enc, c| {
                gpu.encode_nvfp4_matmul_sgk(
                    enc,
                    &packed[c % copies],
                    0,
                    &scales[c % copies],
                    0,
                    nv.tensor_scale,
                    &MatmulRowsTarget {
                        x: &xb,
                        x_offset: 0,
                        out: &out,
                        out_offset: 0,
                        n,
                        k,
                        rows: r,
                    },
                    None,
                )
            });
            let got = gpu.lowering_readback(&out, r * n).expect("readback");
            let worst = (0..r)
                .map(|row| {
                    rel_rms(
                        &reference[row * n..(row + 1) * n],
                        &got[row * n..(row + 1) * n],
                    )
                })
                .fold(0.0f64, f64::max);
            let flag = if worst <= PARITY_REL_RMS {
                ""
            } else {
                "  PARITY FAIL"
            };
            println!(
                "  {:<20} R={r}  sgk {t:>8.1} µs ({:>4.2}x)   X {:>5.0} B/out (x2 {:>5.0})  W {:>5.0} B/out   parity {worst:.1e}{flag}",
                "nvfp4_matmul_sgk",
                t / single,
                (8 * k * 4) as f64 / (32 * r) as f64,
                (2 * k) as f64,
                w_out(r),
            );
            if let Some(wi) = widths.iter().position(|&w| w == r) {
                layer_sgk[wi] += weight * t;
            }
        }
        for ai in 0..SGF_SWEEP.len() {
            for &r in &[2usize, 4, 8] {
                let t = time(&|enc, c| {
                    gpu.encode_nvfp4_matmul_sgk(
                        enc,
                        &packed[c % copies],
                        0,
                        &scales[c % copies],
                        0,
                        nv.tensor_scale,
                        &MatmulRowsTarget {
                            x: &xb,
                            x_offset: 0,
                            out: &out,
                            out_offset: 0,
                            n,
                            k,
                            rows: r,
                        },
                        Some(ai),
                    )
                });
                let got = gpu.lowering_readback(&out, r * n).expect("readback");
                let worst = (0..r)
                    .map(|row| {
                        rel_rms(
                            &reference[row * n..(row + 1) * n],
                            &got[row * n..(row + 1) * n],
                        )
                    })
                    .fold(0.0f64, f64::max);
                let flag = if worst <= PARITY_REL_RMS {
                    ""
                } else {
                    "  PARITY FAIL"
                };
                println!(
                "  {:<20} R={r}  sgf {t:>8.1} µs ({:>4.2}x)   X {:>5.0} B/out (x2 {:>5.0})  W {:>5.0} B/out   parity {worst:.1e}{flag}",
                SGF_SWEEP[ai].0,
                t / single,
                (8 * k * 4) as f64 / (32 * r) as f64,
                (2 * k) as f64,
                w_out(r),
            );
                if let Some(wi) = widths.iter().position(|&w| w == r) {
                    layer_sgf[ai][wi] += weight * t;
                }
            }
        }
        for b in x_rows {
            gpu.recycle_lowering_scratch(b);
        }
        gpu.recycle_lowering_scratch(out);
        gpu.recycle_lowering_scratch(xb);
    }
    println!();
    println!(
        "one layer's projections (q,k,v,o,gate,up,down), head excluded: x2 {layer_single:.1} µs"
    );
    for (wi, &r) in widths.iter().enumerate() {
        println!(
            "  {:<20} R={r}  serial {:>4.2}x   matmul {:>4.2}x   (per verified position {:>4.2}x)",
            MATMUL_ARMS[wi].0,
            layer_serial[wi] / layer_single,
            layer_matmul[wi] / layer_single,
            layer_matmul[wi] / layer_single / r as f64,
        );
    }
    for (wi, &r) in widths.iter().enumerate() {
        if layer_sgk[wi] > 0.0 && MATMUL_ARMS[wi].0.starts_with("nvfp4_matmul_x2") {
            println!(
                "  {:<20} R={r}  sgk {:>4.2}x   (per verified position {:>4.2}x)",
                "nvfp4_matmul_sgk",
                layer_sgk[wi] / layer_single,
                layer_sgk[wi] / layer_single / r as f64,
            );
        }
    }
    for (ai, arm) in SGF_SWEEP.iter().enumerate() {
        for (wi, &r) in widths.iter().enumerate() {
            if layer_sgf[ai][wi] > 0.0 && MATMUL_ARMS[wi].0.starts_with("nvfp4_matmul_x2") {
                println!(
                    "  {:<24} R={r}  {:>4.2}x   (per verified position {:>4.2}x)",
                    arm.0,
                    layer_sgf[ai][wi] / layer_single,
                    layer_sgf[ai][wi] / layer_single / r as f64,
                );
            }
        }
    }
}
