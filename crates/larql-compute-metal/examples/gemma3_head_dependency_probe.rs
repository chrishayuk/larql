//! Why does an RMS norm ahead of each Gemma-3-4B head GEMV double it?
//!
//! `gemma3_step_classes` measured the head at ~1.35 ms per call in a
//! plain GEMV chain and ~2.7 ms when each call follows an RMS norm of the
//! previous result — a 1.4 ms difference no µs-scale dependency bubble
//! explains. Four arms, same GEMV and bytes, divide the explanations:
//!
//!   A   GEMV chain, x = uploaded vector            (the chain number)
//!   A2  GEMV chain, x = the norm's output buffer, no norm dispatched
//!       — isolates "which buffer x lives in"
//!   C   norm(uploaded x) -> GEMV: a norm in the chain the GEMV reads,
//!       but not dependent on the previous GEMV
//!   B   norm(previous GEMV result) -> GEMV (the token's shape)
//!
//! A2 ≈ B      → x's buffer, not the dependency;
//! C ≈ B ≫ A   → any dispatch between GEMVs breaks overlap A enjoyed,
//!               so the plain chain overstates standalone speed;
//! B ≫ C ≈ A   → the true data dependency on the previous result.
use larql_compute_metal::lowering::profile::gpu_span_ms;
use larql_compute_metal::lowering::MatvecOperands;
use larql_compute_metal::MetalBackend;
use larql_models::quant::nvfp4;

const HIDDEN: usize = 2560;
const VOCAB: usize = 262_208;
const CHAIN: usize = 8;
const REPS: usize = 7;
const NORM_EPS: f32 = 1e-6;

#[derive(Clone, Copy, Debug)]
enum Arm {
    A,
    A2,
    C,
    B,
}

fn main() {
    let Some(gpu) = MetalBackend::new() else {
        eprintln!("no Metal device");
        std::process::exit(2);
    };
    let values: Vec<f32> = (0..VOCAB * HIDDEN)
        .map(|i| ((i % 977) as f32 / 977.0) - 0.5)
        .collect();
    let m = nvfp4::quantize(&values, VOCAB, HIDDEN).expect("quantise");
    drop(values);
    let packed = gpu.lowering_weight(&m.packed);
    let scales = gpu.lowering_weight(&m.scales);
    let bytes = (m.packed.len() + m.scales.len()) as f64;
    let x: Vec<f32> = (0..HIDDEN).map(|i| (i % 13) as f32 * 0.01 - 0.05).collect();
    let xb = gpu.lowering_upload(&x).expect("x");
    let norm_weight = gpu.lowering_upload(&vec![1.0f32; HIDDEN]).expect("w");
    let normed = gpu.lowering_upload(&x).expect("normed");
    let out = gpu.lowering_scratch(VOCAB);

    println!(
        "head [{VOCAB}, {HIDDEN}] NVFP4, {:.1} MB; µs per call, best of {REPS}, chain {CHAIN}",
        bytes / 1e6
    );
    // Reps outer, arms inner, so clock/power drift over the run lands on
    // every arm alike rather than on whichever ran last.
    let arms = [Arm::A, Arm::A2, Arm::C, Arm::B];
    let mut best = [f64::MAX; 4];
    for _ in 0..REPS {
        for (ai, &arm) in arms.iter().enumerate() {
            let cmd = gpu.new_lowering_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            for _ in 0..CHAIN {
                let norm_src = match arm {
                    Arm::C => Some(&xb),
                    Arm::B => Some(&out),
                    Arm::A | Arm::A2 => None,
                };
                if let Some(src) = norm_src {
                    larql_compute_metal::stages::input_norm::encode_f32(
                        enc,
                        &gpu.norms.rms_norm_pipeline,
                        src,
                        0,
                        &norm_weight,
                        &normed,
                        0,
                        HIDDEN,
                        NORM_EPS,
                        0.0,
                    );
                }
                let x_in = match arm {
                    Arm::A => &xb,
                    Arm::A2 | Arm::C | Arm::B => &normed,
                };
                gpu.encode_nvfp4_matvec(
                    enc,
                    &MatvecOperands {
                        packed: &packed,
                        scales: &scales,
                        x: x_in,
                        out: &out,
                        out_offset: 0,
                        n: VOCAB,
                        k: HIDDEN,
                    },
                    m.tensor_scale,
                );
            }
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
            best[ai] = best[ai].min(gpu_span_ms(&cmd) * 1e3 / CHAIN as f64);
        }
    }
    for (arm, best) in arms.iter().zip(best) {
        println!(
            "  {:<3} {:>8.1} µs  {:>4.0} GB/s",
            format!("{arm:?}"),
            best,
            bytes / (best / 1e6) / 1e9
        );
    }
}
