//! ANE-3 GPU worker — one long-lived engine driver for the concurrency rung.
//!
//! Runs the same `5120 -> 17408` f16 projection as ANE-0b and ANE-1, in a
//! steady-state loop, for a fixed wall-clock window. It is a *worker*, not
//! an experiment: the coordinator decides when it starts and the analyser
//! decides which of its samples count.
//!
//! ## The protocol, and why it is shaped this way
//!
//! Process launch is not the thing to synchronise — steady state is. Each
//! worker compiles/allocates, warms up, announces readiness, and only then
//! blocks. The coordinator releases every worker at once:
//!
//! ```text
//!   build + warm            (not measured)
//!   touch <name>.ready
//!   spin until `go` exists  <- barrier
//!   loop for DURATION       (measured, every sample epoch-stamped)
//!   write <name>.json
//! ```
//!
//! Individual operations here are ~0.6 ms, so a sloppy launch could
//! manufacture or destroy apparent overlap. Two defences: the barrier
//! above, and **every sample carries an absolute UNIX-epoch start time**
//! so the analyser can keep only the samples that fall inside the window
//! where both engines were genuinely running.
//!
//! Usage:
//!   ane3_gpu_worker <run_dir> <duration_ms> [measure|ramp]

use larql_compute::backend::matmul::MatMul;
use larql_models::quant::half::f32_to_f16;
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const HIDDEN_SIZE: usize = 5120;
const INTERMEDIATE_SIZE: usize = 17408;
const BYTES_PER_F16: usize = 2;
const WARMUP_ITERS: usize = 64;
fn now_epoch() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_secs_f64()
}

fn weights(n: usize, k: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(n * k * BYTES_PER_F16);
    for i in 0..n * k {
        let v = ((i % 977) as f32 / 977.0) - 0.5;
        bytes.extend_from_slice(&f32_to_f16(v).to_le_bytes());
    }
    bytes
}

fn activation(k: usize) -> Vec<f32> {
    (0..k).map(|i| ((i % 13) as f32) * 0.01 - 0.06).collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let run_dir = args.next().expect("run_dir");
    let duration_ms: f64 = args
        .next()
        .expect("duration_ms")
        .parse()
        .expect("duration_ms number");
    let role = args.next().unwrap_or_else(|| "measure".to_string());

    let Some(gpu) = larql_compute_metal::MetalBackend::new() else {
        eprintln!("ane3_gpu_worker: no Metal device");
        std::process::exit(2);
    };

    let (n, k) = (INTERMEDIATE_SIZE, HIDDEN_SIZE);
    let w = weights(n, k);
    let x = activation(k);
    let weight_bytes = n * k * BYTES_PER_F16;

    for _ in 0..WARMUP_ITERS {
        gpu.f16_gemv_force(&w, &x, n, k).expect("gemv");
    }

    let dir = Path::new(&run_dir);
    std::fs::write(dir.join("gpu.ready"), b"1").expect("ready");
    let go = dir.join("go");

    // ANE-3b: ramp, don't idle, while waiting for the barrier.
    //
    // ANE-3's defect was that the GPU-alone arm read ~10% higher when it
    // ran last, i.e. after sustained load, than when it ran first on an
    // idle SoC — a drift the same size as the contention being measured.
    // The repair is to enter every condition from the SAME power state:
    // BOTH engines run flat out from readiness until the barrier opens,
    // in every condition, including the ones where an engine does not
    // then take part. A `ramp` worker exits at `go`; a `measure` worker
    // starts its window there.
    let mut ramp_iters: u64 = 0;
    while !go.exists() {
        gpu.f16_gemv_force(&w, &x, n, k).expect("gemv");
        ramp_iters += 1;
    }

    if role == "ramp" {
        // One op may be in flight when `go` lands — ~0.6 ms against a
        // multi-second window, and recorded rather than hidden.
        let doc = serde_json::json!({"engine": "gpu", "role": "ramp", "ramp_iters": ramp_iters});
        std::fs::write(
            dir.join("gpu.ramp.json"),
            serde_json::to_string(&doc).expect("json"),
        )
        .expect("write");
        eprintln!("gpu worker: ramp only, {ramp_iters} iters");
        return;
    }

    let window_start = now_epoch();
    let clock = Instant::now();
    let mut starts: Vec<f64> = Vec::with_capacity(8192);
    let mut ms: Vec<f64> = Vec::with_capacity(8192);
    while clock.elapsed().as_secs_f64() * 1e3 < duration_ms {
        let t_epoch = now_epoch();
        let t = Instant::now();
        gpu.f16_gemv_force(&w, &x, n, k).expect("gemv");
        ms.push(t.elapsed().as_secs_f64() * 1e3);
        starts.push(t_epoch);
    }
    let window_end = now_epoch();

    let doc = serde_json::json!({
        "engine": "gpu",
        "kernel": "f16_gemv",
        "n": n,
        "k": k,
        "weight_bytes": weight_bytes,
        "window_start": window_start,
        "window_end": window_end,
        "iters": ms.len(),
        "sample_start_epoch": starts,
        "sample_ms": ms,
    });
    std::fs::write(
        dir.join("gpu.json"),
        serde_json::to_string(&doc).expect("json"),
    )
    .expect("write");
    eprintln!(
        "gpu worker: {} iters over {:.3} s",
        ms.len(),
        window_end - window_start
    );
}
