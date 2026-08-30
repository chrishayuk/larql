//! ANE-0b — the frozen GPU-alone baseline for ANE-3's concurrency arm.
//!
//! This is a CONTROL experiment, not a performance rung. Nothing here may
//! be optimised: the number it produces is only useful if it is the
//! number the GPU actually sustains today on the exact shape, dtype and
//! kernel family that ANE-3 will run concurrently with the Neural
//! Engine. Once ANE-3 begins, this result is immutable.
//!
//! Why a same-shape control at all: the roofline programme's headline
//! ~367 GB/s came from a different access pattern. Scoring a later
//! concurrency result against it would make a 5-10% "gain" trivially
//! easy to manufacture.
//!
//! Subject shape is pinned from the real container — Qwen3.8-27B,
//! `hidden_size` 5120, `intermediate_size` 17408. The FFN gate/up
//! projection is the most repeated large op in the model (128 instances
//! per token) and FFN is ~63% of the bytes read per token.
//!
//! Three lines are measured, and only the first is the control:
//!
//! ```text
//! ffn_gate_up      5120 -> 17408   THE CONTROL
//! ffn_down        17408 ->  5120   context: same bytes, 3.4x fewer
//!                                  threadgroups, guards against a
//!                                  ROWS_PER_TG geometry surprise
//! dispatch_floor   5120 ->     8   per-call submission cost with almost
//!                                  no weight traffic, so the control's
//!                                  ms can be read as work rather than
//!                                  overhead
//! ```
//!
//! Usage:
//!   cargo run --release -p larql-compute-metal --example ane0b_gpu_baseline \
//!       -- <session-label> [out.json]

use larql_compute::backend::matmul::MatMul;
use larql_models::quant::half::{f16_to_f32, f32_to_f16};
use std::time::Instant;

/// Qwen3.8-27B text config, snapshot 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
const HIDDEN_SIZE: usize = 5120;
const INTERMEDIATE_SIZE: usize = 17408;

/// Long enough that first-touch upload, wiring and pipeline warm-up are
/// out of the measured window.
const WARMUP_ITERS: usize = 64;
/// Every iteration is recorded — the distribution is part of the result,
/// not just the minimum.
const MEASURED_ITERS: usize = 1024;

const BYTES_PER_F16: usize = 2;
const BYTES_PER_F32: usize = 4;

/// `f16_gemv` geometry, mirrored from the shader so the report can state
/// the dispatch width without reaching into the backend.
const ROWS_PER_TG: usize = 8;

/// Above the ~400 GB/s physical fabric ceiling of this machine, an
/// implied rate is a broken measurement rather than a discovery. Refuse
/// to bank it.
const IMPLAUSIBLE_GBS: f64 = 600.0;

/// The GPU reduces with `simd_sum`; the CPU check sums in natural order.
/// The orders differ, so exact equality is not the contract — agreement
/// to well inside f16 input precision is.
const MAX_REL_ERR: f64 = 2e-3;

/// One measured line. `is_control` marks the single row ANE-3 may score
/// against; the others exist to make that row interpretable.
struct Shape {
    name: &'static str,
    role: &'static str,
    is_control: bool,
    n: usize,
    k: usize,
}

const SHAPES: &[Shape] = &[
    Shape {
        name: "ffn_gate_up",
        role: "control",
        is_control: true,
        n: INTERMEDIATE_SIZE,
        k: HIDDEN_SIZE,
    },
    Shape {
        name: "ffn_down",
        role: "context",
        is_control: false,
        n: HIDDEN_SIZE,
        k: INTERMEDIATE_SIZE,
    },
    Shape {
        name: "dispatch_floor",
        role: "overhead",
        is_control: false,
        n: ROWS_PER_TG,
        k: HIDDEN_SIZE,
    },
];

/// Deterministic weights in a benign f16 range. The values do not matter
/// for bandwidth; reproducibility does, so the correctness check below
/// means the same thing on every run.
fn make_weights(n: usize, k: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(n * k * BYTES_PER_F16);
    for i in 0..n * k {
        let v = ((i % 977) as f32 / 977.0) - 0.5;
        bytes.extend_from_slice(&f32_to_f16(v).to_le_bytes());
    }
    bytes
}

fn make_activation(k: usize) -> Vec<f32> {
    (0..k).map(|i| ((i % 13) as f32) * 0.01 - 0.06).collect()
}

/// One row of `out = W · x`, computed on the CPU from the same f16 bytes.
/// The control that proves the kernel is doing the projection: a timing
/// harness that is fast because it computed nothing is the failure mode
/// this rules out.
fn cpu_row(w_f16: &[u8], x: &[f32], row: usize, k: usize) -> f64 {
    let base = row * k;
    (0..k)
        .map(|j| {
            let idx = (base + j) * BYTES_PER_F16;
            let bits = u16::from_le_bytes([w_f16[idx], w_f16[idx + 1]]);
            f16_to_f32(bits) as f64 * x[j] as f64
        })
        .sum()
}

struct Stats {
    min: f64,
    p50: f64,
    p90: f64,
    max: f64,
    mean: f64,
    stdev: f64,
}

fn stats(samples: &[f64]) -> Stats {
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).expect("no NaN in timings"));
    let n = sorted.len();
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let var = sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    Stats {
        min: sorted[0],
        p50: sorted[n / 2],
        p90: sorted[(n * 9) / 10],
        max: sorted[n - 1],
        mean,
        stdev: var.sqrt(),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let session = args.next().unwrap_or_else(|| "unlabelled".to_string());
    let json_path = args.next();

    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("ANE-0b: no Metal device — refusing to bank a baseline");
        std::process::exit(2);
    };

    println!("ANE-0b GPU-alone baseline — session '{session}'");
    println!("kernel family: f16_gemv (f16 weights, f32 activation, f32 out)");
    println!("warmup {WARMUP_ITERS}, measured {MEASURED_ITERS}\n");
    println!(
        "{:<16}{:>9}{:>10}{:>10}{:>10}{:>10}{:>10}{:>11}",
        "shape", "MB", "min ms", "p50 ms", "p90 ms", "max ms", "sd ms", "GB/s min"
    );

    let mut rows_json: Vec<String> = Vec::new();
    let mut failed = false;

    for shape in SHAPES {
        let (n, k) = (shape.n, shape.k);
        let w = make_weights(n, k);
        let x = make_activation(k);
        let weight_bytes = n * k * BYTES_PER_F16;
        let total_bytes = weight_bytes + k * BYTES_PER_F32 + n * BYTES_PER_F32;
        let threadgroups = n.div_ceil(ROWS_PER_TG);

        // First touch pays buffer creation and wiring, which steady-state
        // decode does not.
        for _ in 0..WARMUP_ITERS {
            gpu.f16_gemv_force(&w, &x, n, k).expect("f16_gemv_force");
        }

        let mut samples = Vec::with_capacity(MEASURED_ITERS);
        let mut last = Vec::new();
        for _ in 0..MEASURED_ITERS {
            let t = Instant::now();
            let out = gpu.f16_gemv_force(&w, &x, n, k).expect("f16_gemv_force");
            samples.push(t.elapsed().as_secs_f64() * 1e3);
            last = out;
        }

        let s = stats(&samples);
        let gbs_min = weight_bytes as f64 / (s.min / 1e3) / 1e9;
        let gbs_p50 = weight_bytes as f64 / (s.p50 / 1e3) / 1e9;

        // Control 1: the kernel computed the projection.
        let probe_row = n / 2;
        let cpu = cpu_row(&w, &x, probe_row, k);
        let gpu_val = last[probe_row] as f64;
        let denom = cpu.abs().max(1e-6);
        let rel_err = (gpu_val - cpu).abs() / denom;
        if rel_err > MAX_REL_ERR {
            eprintln!(
                "ANE-0b: {} row {probe_row} gpu {gpu_val:.6} vs cpu {cpu:.6} \
                 rel_err {rel_err:.3e} > {MAX_REL_ERR:.0e} — the kernel is not \
                 computing this projection; the timing is void",
                shape.name
            );
            failed = true;
        }
        // Control 2: an implied rate above the fabric is a broken clock,
        // not a result. Only meaningful where weight traffic dominates.
        if shape.is_control && gbs_min > IMPLAUSIBLE_GBS {
            eprintln!(
                "ANE-0b: {} implies {gbs_min:.1} GB/s, above the {IMPLAUSIBLE_GBS:.0} \
                 GB/s plausibility ceiling — measurement is broken",
                shape.name
            );
            failed = true;
        }

        println!(
            "{:<16}{:>9.1}{:>10.3}{:>10.3}{:>10.3}{:>10.3}{:>10.3}{:>11.1}",
            shape.name,
            weight_bytes as f64 / 1e6,
            s.min,
            s.p50,
            s.p90,
            s.max,
            s.stdev,
            gbs_min
        );

        rows_json.push(format!(
            r#"    {{"name":"{}","role":"{}","is_control":{},"n":{},"k":{},
     "weight_bytes":{},"total_bytes":{},"threadgroups":{},
     "ms":{{"min":{:.6},"p50":{:.6},"p90":{:.6},"max":{:.6},"mean":{:.6},"stdev":{:.6}}},
     "gbs_weight_min":{:.3},"gbs_weight_p50":{:.3},
     "check":{{"row":{},"gpu":{:.6},"cpu":{:.6},"rel_err":{:.3e}}}}}"#,
            shape.name,
            shape.role,
            shape.is_control,
            n,
            k,
            weight_bytes,
            total_bytes,
            threadgroups,
            s.min,
            s.p50,
            s.p90,
            s.max,
            s.mean,
            s.stdev,
            gbs_min,
            gbs_p50,
            probe_row,
            gpu_val,
            cpu,
            rel_err
        ));
    }

    let doc = format!(
        r#"{{
  "experiment": "ANE-0b",
  "purpose": "frozen GPU-alone control for ANE-3; immutable once ANE-3 begins",
  "session": "{session}",
  "model": {{"name":"Qwen3.8-27B","hidden_size":{HIDDEN_SIZE},"intermediate_size":{INTERMEDIATE_SIZE}}},
  "kernel": {{"family":"f16_gemv","weights":"f16","activation":"f32","out":"f32","rows_per_tg":{ROWS_PER_TG}}},
  "iters": {{"warmup":{WARMUP_ITERS},"measured":{MEASURED_ITERS}}},
  "shapes": [
{}
  ]
}}
"#,
        rows_json.join(",\n")
    );

    if let Some(path) = json_path {
        std::fs::write(&path, &doc).expect("write json");
        println!("\nwrote {path}");
    } else {
        println!("\n{doc}");
    }

    if failed {
        eprintln!("\nANE-0b: FAILED its own controls — do not bank this run");
        std::process::exit(1);
    }
}
