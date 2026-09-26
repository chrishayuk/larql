//! STEP-PROFILE-1: each projection class of a Gemma-3-4B token, alone.
//!
//! The lowered token trails MLX by ~1.5 ms fixed + ~0.28 ms per 1K of
//! context. `--profile` prices each class *in the step*; this prices the
//! same dispatch *standalone*, so the difference is what the step costs
//! the kernel — scheduling, cache state, the norm ahead of it — and not
//! the kernel itself.
//!
//! Every class is encoded the way the lowering encodes it: Q/K/V and
//! gate/up as ONE segmented dispatch, O and down as the plain NVFP4
//! GEMV, the head as one GEMV over the vocabulary. Weights are Gemma-3-4B
//! shapes, rotated past the system cache so each call streams from DRAM
//! as the token does. Arms, per class:
//!
//!   A  GEMV chain            — the isolated number;
//!   B  RMS norm -> GEMV      — the token's dependency shape: each GEMV
//!                              reads the norm of the previous result;
//!   N  RMS norm chain alone  — the norm's own cost.
//!
//! Per-boundary bubble = B − A − N: what a dependent norm costs the GEMV
//! beyond the norm's own time.
//!
//! Run on AC with the machine quiet; on battery the GEMV alone drifts
//! ~25% between runs.
use larql_compute_metal::lowering::profile::gpu_span_ms;
use larql_compute_metal::lowering::{MatvecOperands, Nvfp4Segment};
use larql_compute_metal::MetalBackend;
use larql_models::quant::nvfp4;

/// Gemma-3-4B: hidden 2560, 8 query heads and 4 KV heads of 256, FFN
/// 10240, vocabulary 262208, 34 layers.
const HIDDEN: usize = 2560;
const Q_ROWS: usize = 8 * 256;
const KV_ROWS: usize = 4 * 256;
const INTERMEDIATE: usize = 10240;
const VOCAB: usize = 262_208;
const LAYERS: usize = 34;
/// One chain = one token's worth of calls of a per-layer class.
const CHAIN: usize = LAYERS;
const REPS: usize = 7;
/// Distinct weight bytes a chain cycles through — past the M3 Max system
/// cache, so the steady state reads from DRAM.
const ROTATE_BYTES: usize = 256 << 20;
/// Probe-measured GPU read ceiling on this class of machine (membw_probe).
const GPU_READ_CEILING_GB_S: f64 = 367.0;
const NORM_EPS: f32 = 1e-6;

/// A projection class: its segments' row counts (one entry = a plain
/// GEMV, several = one segmented dispatch) over `k` input columns, and
/// how many calls a token makes.
struct Class {
    name: &'static str,
    rows: &'static [usize],
    k: usize,
    calls: usize,
}

const CLASSES: &[Class] = &[
    Class {
        name: "attn.proj (qkv seg3)",
        rows: &[Q_ROWS, KV_ROWS, KV_ROWS],
        k: HIDDEN,
        calls: LAYERS,
    },
    Class {
        name: "attn.o_proj",
        rows: &[HIDDEN],
        k: Q_ROWS,
        calls: LAYERS,
    },
    Class {
        name: "ffn.gate_up (seg2)",
        rows: &[INTERMEDIATE, INTERMEDIATE],
        k: HIDDEN,
        calls: LAYERS,
    },
    Class {
        name: "ffn.down",
        rows: &[HIDDEN],
        k: INTERMEDIATE,
        calls: LAYERS,
    },
    Class {
        name: "head",
        rows: &[VOCAB],
        k: HIDDEN,
        calls: 1,
    },
];

/// One quantised matrix resident on the device.
struct Resident {
    packed: metal::Buffer,
    scales: metal::Buffer,
    tensor_scale: f32,
    bytes: usize,
}

fn quantised(n: usize, k: usize, seed: usize) -> nvfp4::Nvfp4Matrix {
    let values: Vec<f32> = (0..n * k)
        .map(|i| (((i + seed) % 977) as f32 / 977.0) - 0.5)
        .collect();
    nvfp4::quantize(&values, n, k).expect("quantise")
}

#[derive(Clone, Copy)]
enum Arm {
    Gemv,
    NormGemv,
    Norm,
}

fn main() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device");
        std::process::exit(2);
    };
    println!(
        "Gemma-3-4B projection classes, standalone: µs per call (best of {REPS}); \
         floor = bytes / {GPU_READ_CEILING_GB_S:.0} GB/s"
    );
    println!(
        "{:<22} {:>8} {:>9} {:>7} {:>9} {:>7} {:>8} {:>8} {:>10}",
        "class", "MB/call", "A gemv", "GB/s", "B nrm+g", "N norm", "bubble", "floor", "A ms/tok"
    );
    for class in CLASSES {
        let seg_bytes: usize = class
            .rows
            .iter()
            .map(|&n| {
                let m = quantised(n, class.k, 0);
                m.packed.len() + m.scales.len()
            })
            .sum();
        let copies = (ROTATE_BYTES / seg_bytes).clamp(1, CHAIN);
        // copies × segments. Every host matrix stays alive until the class
        // is done: `get_bytes` keys its cache on (ptr, len), so a matrix
        // dropped after upload lets the next one land at the same address
        // and come back as the SAME device buffer — the rotation collapses
        // to one cache-resident matrix and reads above the DRAM ceiling.
        let host: Vec<Vec<nvfp4::Nvfp4Matrix>> = (0..copies)
            .map(|c| {
                class
                    .rows
                    .iter()
                    .enumerate()
                    .map(|(si, &n)| quantised(n, class.k, c * 7 + si))
                    .collect()
            })
            .collect();
        let banks: Vec<Vec<Resident>> = host
            .iter()
            .map(|copy| {
                copy.iter()
                    .map(|m| Resident {
                        bytes: m.packed.len() + m.scales.len(),
                        packed: gpu.lowering_weight(&m.packed),
                        scales: gpu.lowering_weight(&m.scales),
                        tensor_scale: m.tensor_scale,
                    })
                    .collect()
            })
            .collect();
        let call_bytes: usize = banks[0].iter().map(|r| r.bytes).sum();
        // Outputs sized so the norm can read `k` floats of segment 0's
        // result whatever the shape.
        let outs: Vec<metal::Buffer> = class
            .rows
            .iter()
            .map(|&n| gpu.lowering_scratch(n.max(class.k)))
            .collect();
        let x: Vec<f32> = (0..class.k)
            .map(|i| (i % 13) as f32 * 0.01 - 0.05)
            .collect();
        let xb = gpu.lowering_upload(&x).expect("x");
        let norm_weight = gpu.lowering_upload(&vec![1.0f32; class.k]).expect("w");
        let normed = gpu.lowering_scratch(class.k);

        // Reps outer, arms inner: GPU clock/power state drifts over
        // seconds, and running all of one arm before the next puts that
        // drift entirely on the later arms.
        let mut best = [f64::MAX; 3];
        for _ in 0..REPS {
            for (ai, arm) in [Arm::Gemv, Arm::NormGemv, Arm::Norm]
                .into_iter()
                .enumerate()
            {
                let cmd = gpu.new_lowering_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                for c in 0..CHAIN {
                    let x_in = match arm {
                        Arm::Gemv => &xb,
                        Arm::NormGemv | Arm::Norm => {
                            larql_compute_metal::stages::input_norm::encode_f32(
                                enc,
                                &gpu.norms.rms_norm_pipeline,
                                if c == 0 { &xb } else { &outs[0] },
                                0,
                                &norm_weight,
                                &normed,
                                0,
                                class.k,
                                NORM_EPS,
                                0.0,
                            );
                            &normed
                        }
                    };
                    if matches!(arm, Arm::Norm) {
                        continue;
                    }
                    let bank = &banks[c % banks.len()];
                    if bank.len() == 1 {
                        let r = &bank[0];
                        gpu.encode_nvfp4_matvec(
                            enc,
                            &MatvecOperands {
                                packed: &r.packed,
                                scales: &r.scales,
                                x: x_in,
                                out: &outs[0],
                                out_offset: 0,
                                n: class.rows[0],
                                k: class.k,
                            },
                            r.tensor_scale,
                        );
                    } else {
                        let segments: Vec<Nvfp4Segment<'_>> = bank
                            .iter()
                            .zip(&outs)
                            .zip(class.rows)
                            .map(|((r, out), &n)| Nvfp4Segment {
                                packed: &r.packed,
                                packed_offset: 0,
                                scales: &r.scales,
                                scales_offset: 0,
                                tensor_scale: r.tensor_scale,
                                out,
                                out_offset: 0,
                                n,
                            })
                            .collect();
                        gpu.encode_nvfp4_matvec_segments(enc, x_in, class.k, &segments);
                    }
                }
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
                best[ai] = best[ai].min(gpu_span_ms(&cmd) * 1e3 / CHAIN as f64);
            }
        }
        let [a, b, n] = best;
        let floor_us = call_bytes as f64 / 1e9 / GPU_READ_CEILING_GB_S * 1e6;
        // Faster than DRAM allows means the chain read from cache: the
        // number is not a DRAM number and must not be compared as one.
        let cache_fed = if a < floor_us {
            "  CACHE-FED (a < floor)"
        } else {
            ""
        };
        println!(
            "{:<22} {:>8.1} {:>9.1} {:>7.0} {:>9.1} {:>7.1} {:>8.1} {:>8.1} {:>10.3}",
            class.name,
            call_bytes as f64 / 1e6,
            a,
            call_bytes as f64 / (a / 1e6) / 1e9,
            b,
            n,
            b - a - n,
            floor_us,
            a * class.calls as f64 / 1e3,
        );
        if !cache_fed.is_empty() {
            println!("{:<22}{cache_fed}", "");
        }
        drop(host);
    }
    println!(
        "\nA ms/tok = the class's standalone cost for one token ({LAYERS} calls per layer class, 1 head). \
         Compare with the in-step `--profile` row for the same class."
    );
}
